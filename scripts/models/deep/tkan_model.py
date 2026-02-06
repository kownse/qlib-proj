"""
TKAN (Temporal Kolmogorov-Arnold Network) Stock Return Prediction Model

Pure PyTorch reimplementation based on the TKAN paper (arXiv: 2405.07344).
Combines LSTM-style gating with KAN (learnable B-spline) sub-layers.

Architecture:
- LSTM gates (input, forget, candidate) for temporal memory
- KAN sub-layers replace the output gate with learnable spline activations
- Multi-scale sub-layer outputs aggregated to produce the output gate

Input: (batch, total_features) flat features, reshaped to (batch, seq_len, d_feat)
Output: (batch,) predicted returns

Suitable handlers: alpha158-talib-macro (flat), alpha360 (temporal), alpha360-macro, etc.
"""

import os
import sys
import copy
import math
import time
import numpy as np
import pandas as pd
from typing import Optional, List, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


# ============================================================================
# KANLinear (from efficient-kan, self-contained copy)
# ============================================================================

class KANLinear(nn.Module):
    """
    KAN linear layer using B-spline basis functions.

    Replaces a standard Linear layer with two branches:
      1. base_activation(x) @ base_weight  (standard non-linearity)
      2. b_splines(x) @ spline_weight      (learnable B-spline transform)
    """

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))


# ============================================================================
# Activation helper
# ============================================================================

def _get_activation(name: str) -> nn.Module:
    """Get activation module by name string."""
    return {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'silu': nn.SiLU(),
        'gelu': nn.GELU(),
    }.get(name, nn.ReLU())


# ============================================================================
# TKAN Cell (Pure PyTorch)
# ============================================================================

class TKANCellPyTorch(nn.Module):
    """
    TKAN cell processing one timestep.

    Modified LSTM cell where the output gate is produced by aggregating
    multi-scale KAN sub-layer outputs instead of a standard linear gate.

    Gate equations (following TKAN paper):
        gates = sigmoid(x @ kernel + h @ recurrent_kernel + bias)
        i, f, c_gate = split(gates, 3)
        c = f * c_old + i * tanh(c_gate)

    Sub-layer processing:
        For each sub-layer k:
            agg_input = x @ sub_kernel_x[k] + sub_state[k] @ sub_kernel_h[k]
            sub_out[k] = KANLinear(agg_input)  or  Linear(agg_input)
            new_sub_state[k] = rk_h[k] * sub_out[k] + rk_x[k] * sub_state[k]

    Output gate:
        o = sigmoid(concat(sub_out_0, ..., sub_out_N) @ agg_weight + agg_bias)
        h = o * tanh(c)

    Args:
        input_dim: Input feature dimension
        units: Hidden state dimension (output dimension)
        sub_kan_configs: Config for each sub-layer. Each entry can be:
            - None: default KANLinear
            - int/float: KANLinear with that spline_order
            - dict: KANLinear with those kwargs
            - str: Linear + named activation (e.g., 'relu', 'tanh')
        sub_kan_output_dim: Output dim of each sub-layer (default: min(input_dim, units))
        sub_kan_input_dim: Input dim to each sub-layer (default: sub_kan_output_dim)
        dropout: Input dropout rate
        recurrent_dropout: Recurrent state dropout rate
        grid_size: B-spline grid segments for KAN sub-layers
        spline_order: B-spline order for KAN sub-layers
        grid_range: Initial grid range for KAN sub-layers
    """

    def __init__(
        self,
        input_dim: int,
        units: int,
        sub_kan_configs: list = None,
        sub_kan_output_dim: int = None,
        sub_kan_input_dim: int = None,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: list = [-3, 3],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.units = units
        self.sub_kan_configs = sub_kan_configs or [None]
        self.num_sub_layers = len(self.sub_kan_configs)

        # Default sub dims: keep manageable regardless of input_dim
        if sub_kan_output_dim is None:
            self.sub_kan_output_dim = min(input_dim, units)
        else:
            self.sub_kan_output_dim = sub_kan_output_dim
        if sub_kan_input_dim is None:
            self.sub_kan_input_dim = self.sub_kan_output_dim
        else:
            self.sub_kan_input_dim = sub_kan_input_dim

        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = list(grid_range)

        # --- LSTM gates: input, forget, candidate (3 gates, no output gate) ---
        self.kernel = nn.Parameter(torch.empty(input_dim, units * 3))
        self.recurrent_kernel = nn.Parameter(torch.empty(units, units * 3))
        self.bias = nn.Parameter(torch.zeros(units * 3))

        nn.init.xavier_uniform_(self.kernel)
        nn.init.orthogonal_(self.recurrent_kernel)
        # Unit forget bias (forget gate bias initialized to 1)
        with torch.no_grad():
            self.bias[units:2 * units].fill_(1.0)

        # --- Sub-KAN layers ---
        self.sub_layers = nn.ModuleList()
        for config in self.sub_kan_configs:
            if config is None or isinstance(config, (int, float, dict)):
                kan_kwargs = {}
                if isinstance(config, (int, float)):
                    kan_kwargs['spline_order'] = int(config)
                elif isinstance(config, dict):
                    kan_kwargs = config.copy()
                kan_kwargs.setdefault('grid_size', grid_size)
                kan_kwargs.setdefault('spline_order', spline_order)
                kan_kwargs.setdefault('grid_range', grid_range)
                self.sub_layers.append(
                    KANLinear(self.sub_kan_input_dim, self.sub_kan_output_dim, **kan_kwargs)
                )
            elif isinstance(config, str):
                layer = nn.Sequential(
                    nn.Linear(self.sub_kan_input_dim, self.sub_kan_output_dim),
                    _get_activation(config),
                )
                self.sub_layers.append(layer)

        # --- Sub-layer projection weights ---
        # Project input to sub_kan_input_dim
        self.sub_kernel_inputs = nn.Parameter(
            torch.empty(self.num_sub_layers, input_dim, self.sub_kan_input_dim)
        )
        # Project previous sub-state to sub_kan_input_dim
        self.sub_kernel_states = nn.Parameter(
            torch.empty(self.num_sub_layers, self.sub_kan_output_dim, self.sub_kan_input_dim)
        )
        # Recurrent kernel for sub-state update (h_coeff, x_coeff per sub-layer)
        self.sub_recurrent_kernel = nn.Parameter(
            torch.empty(self.num_sub_layers, self.sub_kan_output_dim * 2)
        )

        # Initialize sub-layer projection weights
        for i in range(self.num_sub_layers):
            nn.init.orthogonal_(self.sub_kernel_inputs[i])
            nn.init.orthogonal_(self.sub_kernel_states[i])
        nn.init.uniform_(self.sub_recurrent_kernel, -0.1, 0.1)

        # --- Aggregation: sub-layer outputs -> output gate ---
        agg_dim = self.num_sub_layers * self.sub_kan_output_dim
        self.aggregated_weight = nn.Parameter(torch.empty(agg_dim, units))
        self.aggregated_bias = nn.Parameter(torch.zeros(units))
        nn.init.xavier_uniform_(self.aggregated_weight)

        # --- Dropout ---
        self.input_dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.recurrent_dropout_mod = nn.Dropout(recurrent_dropout) if recurrent_dropout > 0 else None

    def get_initial_state(self, batch_size: int, device: torch.device):
        """Return list of zero-initialized states: [h, c, sub_0, ..., sub_N]."""
        return [
            torch.zeros(batch_size, self.units, device=device),
            torch.zeros(batch_size, self.units, device=device),
        ] + [
            torch.zeros(batch_size, self.sub_kan_output_dim, device=device)
            for _ in range(self.num_sub_layers)
        ]

    def forward(self, inputs, states, training=False):
        """
        Process one timestep.

        Args:
            inputs: (batch, input_dim)
            states: [h, c, sub_state_0, ..., sub_state_N]
            training: whether in training mode

        Returns:
            h: (batch, units)
            new_states: updated [h, c, sub_0, ..., sub_N]
        """
        h_tm1 = states[0]
        c_tm1 = states[1]
        sub_states = states[2:]

        # Dropout
        if training and self.input_dropout is not None:
            inputs = self.input_dropout(inputs)
        if training and self.recurrent_dropout_mod is not None:
            h_tm1 = self.recurrent_dropout_mod(h_tm1)

        # LSTM gates: all 3 through sigmoid (following TKAN paper)
        gates = (torch.matmul(inputs, self.kernel)
                 + torch.matmul(h_tm1, self.recurrent_kernel)
                 + self.bias)
        gates = torch.sigmoid(gates)
        i, f, c_gate = gates.chunk(3, dim=-1)

        # Cell state update
        c = f * c_tm1 + i * torch.tanh(c_gate)

        # Process sub-KAN layers
        sub_outputs = []
        new_sub_states = []

        for idx in range(self.num_sub_layers):
            sub_layer = self.sub_layers[idx]
            sub_state = sub_states[idx]

            # Project input and previous sub-state
            agg_input = (torch.matmul(inputs, self.sub_kernel_inputs[idx])
                         + torch.matmul(sub_state, self.sub_kernel_states[idx]))

            # KAN or Linear sub-layer
            sub_output = sub_layer(agg_input)

            # Update sub-state via learnable recurrent kernel
            sub_rk_h, sub_rk_x = self.sub_recurrent_kernel[idx].chunk(2, dim=0)
            new_sub_state = sub_rk_h * sub_output + sub_rk_x * sub_state

            sub_outputs.append(sub_output)
            new_sub_states.append(new_sub_state)

        # Aggregate sub-layer outputs -> output gate
        aggregated = torch.cat(sub_outputs, dim=-1)
        aggregated_input = (torch.matmul(aggregated, self.aggregated_weight)
                            + self.aggregated_bias)
        o = torch.sigmoid(aggregated_input)

        # Final output
        h = o * torch.tanh(c)

        return h, [h, c] + new_sub_states


# ============================================================================
# TKAN Network
# ============================================================================

class TKANNetwork(nn.Module):
    """
    Multi-layer TKAN network for stock prediction.

    Processes (batch, total_features) by reshaping to (batch, seq_len, d_feat)
    and running through stacked TKANCellPyTorch layers.

    Args:
        d_feat: Features per timestep
        seq_len: Number of timesteps (total_features = d_feat * seq_len)
        hidden_size: TKAN hidden units
        num_layers: Number of stacked TKAN layers
        sub_kan_configs: KAN sub-layer configs (see TKANCellPyTorch)
        sub_kan_output_dim: Sub-layer output dim (default: auto)
        sub_kan_input_dim: Sub-layer input dim (default: auto)
        dropout: Dropout rate between layers
        recurrent_dropout: Recurrent dropout rate
        grid_size: KAN B-spline grid segments
        spline_order: KAN B-spline order
        grid_range: KAN initial grid range
    """

    def __init__(
        self,
        d_feat: int,
        seq_len: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        sub_kan_configs: list = None,
        sub_kan_output_dim: int = None,
        sub_kan_input_dim: int = None,
        dropout: float = 0.1,
        recurrent_dropout: float = 0.0,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: list = [-3, 3],
    ):
        super().__init__()
        self.d_feat = d_feat
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input normalization
        self.input_norm = nn.LayerNorm(d_feat)

        # Stacked TKAN cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = d_feat if i == 0 else hidden_size
            # Per-layer sub dims: proportional to input_dim of that layer
            layer_sub_out = sub_kan_output_dim or min(input_dim, hidden_size)
            layer_sub_in = sub_kan_input_dim or layer_sub_out

            cell = TKANCellPyTorch(
                input_dim=input_dim,
                units=hidden_size,
                sub_kan_configs=sub_kan_configs,
                sub_kan_output_dim=layer_sub_out,
                sub_kan_input_dim=layer_sub_in,
                dropout=dropout if i < num_layers - 1 else 0.0,
                recurrent_dropout=recurrent_dropout,
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range,
            )
            self.cells.append(cell)

        # Layer norm between TKAN layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers - 1)
        ])

        # Output head
        self.output_dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, total_features) flat features

        Returns:
            (batch,) predictions
        """
        batch_size = x.size(0)

        # Reshape flat features to (batch, seq_len, d_feat)
        if x.dim() == 2:
            x = x.view(batch_size, self.seq_len, self.d_feat)

        # Normalize each timestep
        x = self.input_norm(x)

        # Process through stacked TKAN layers
        for layer_idx, cell in enumerate(self.cells):
            states = cell.get_initial_state(batch_size, x.device)

            outputs = []
            for t in range(x.size(1)):
                h, states = cell(x[:, t, :], states, training=self.training)
                outputs.append(h)

            x = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_size)

            # Layer norm between TKAN layers
            if layer_idx < self.num_layers - 1:
                x = self.layer_norms[layer_idx](x)

        # Use last hidden state
        last_h = x[:, -1, :]

        # Output
        out = self.output_dropout(last_h)
        out = self.output_linear(out)
        return out.squeeze(-1)

    def get_kan_sublayers(self):
        """Collect all KANLinear sub-layers for grid updates."""
        kan_layers = []
        for cell in self.cells:
            for sub_layer in cell.sub_layers:
                if isinstance(sub_layer, KANLinear):
                    kan_layers.append(sub_layer)
        return kan_layers

    def regularization_loss(self, reg_activation=1.0, reg_entropy=1.0):
        """KAN regularization (L1 + entropy) across all KAN sub-layers."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.get_kan_sublayers():
            l1_fake = layer.spline_weight.abs().mean(-1)
            l1_sum = l1_fake.sum()
            if l1_sum > 0:
                p = l1_fake / l1_sum
                p = p.clamp(min=1e-12)
                entropy = -torch.sum(p * p.log())
            else:
                entropy = torch.tensor(0.0, device=total.device)
            total = total + reg_activation * l1_sum + reg_entropy * entropy
        return total


# ============================================================================
# Early Stopping
# ============================================================================

class EarlyStopping:
    """Early stopping on validation IC (higher is better)."""

    def __init__(self, patience: int = 15):
        self.patience = patience
        self.counter = 0
        self.best_score = -float('inf')
        self.best_state = None

    def __call__(self, score: float, model: nn.Module) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ============================================================================
# Qlib Model Interface
# ============================================================================

class TKANStock:
    """
    TKAN stock prediction model with Qlib interface.

    Trains a TKANNetwork on features to predict N-day forward returns.
    Supports both flat handlers (alpha158-talib-macro, seq_len=1) and
    sequential handlers (alpha360, seq_len=60).

    Parameters
    ----------
    d_feat : int
        Features per timestep (auto-detected if 0)
    seq_len : int
        Number of timesteps (1 for flat handlers)
    hidden_size : int
        TKAN hidden units
    num_layers : int
        Number of stacked TKAN layers
    sub_kan_configs : list
        KAN sub-layer configs (None=default KAN, int=spline_order,
        dict=KAN kwargs, str=activation name)
    sub_kan_output_dim : int
        Sub-layer output dim (default: auto)
    sub_kan_input_dim : int
        Sub-layer input dim (default: auto)
    dropout : float
        Dropout rate
    recurrent_dropout : float
        Recurrent state dropout rate
    learning_rate : float
        AdamW learning rate
    weight_decay : float
        AdamW weight decay
    reg_lambda : float
        KAN regularization weight
    batch_size : int
        Training batch size
    n_epochs : int
        Maximum training epochs
    early_stop : int
        Early stopping patience
    grid_update_freq : int
        Update KAN B-spline grids every N epochs
    grid_size : int
        B-spline grid segments
    spline_order : int
        B-spline order
    grid_range : list
        Initial grid range
    loss_type : str
        'mse' or 'ic'
    GPU : int
        GPU device ID (-1 for CPU)
    seed : int
        Random seed
    """

    def __init__(
        self,
        d_feat: int = 158,
        seq_len: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        sub_kan_configs: list = None,
        sub_kan_output_dim: int = None,
        sub_kan_input_dim: int = None,
        dropout: float = 0.1,
        recurrent_dropout: float = 0.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        reg_lambda: float = 1e-5,
        batch_size: int = 2048,
        n_epochs: int = 100,
        early_stop: int = 15,
        grid_update_freq: int = 20,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: list = [-3, 3],
        loss_type: str = 'mse',
        GPU: int = 0,
        seed: int = 42,
    ):
        self.d_feat = d_feat
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sub_kan_configs = sub_kan_configs
        self.sub_kan_output_dim = sub_kan_output_dim
        self.sub_kan_input_dim = sub_kan_input_dim
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.reg_lambda = reg_lambda
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stop_patience = early_stop
        self.grid_update_freq = grid_update_freq
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = list(grid_range)
        self.loss_type = loss_type
        self.GPU = GPU
        self.seed = seed

        self.model: Optional[TKANNetwork] = None
        self.fitted = False

        self._set_seed()
        self._setup_device()

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _setup_device(self):
        if self.GPU >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.GPU}')
            print(f"    Using GPU: cuda:{self.GPU}")
        else:
            self.device = torch.device('cpu')
            print("    Using CPU")

    def _build_model(self) -> TKANNetwork:
        model = TKANNetwork(
            d_feat=self.d_feat,
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            sub_kan_configs=self.sub_kan_configs,
            sub_kan_output_dim=self.sub_kan_output_dim,
            sub_kan_input_dim=self.sub_kan_input_dim,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
            grid_size=self.grid_size,
            spline_order=self.spline_order,
            grid_range=self.grid_range,
        )
        return model.to(self.device)

    def _prepare_data(self, dataset: DatasetH, segment: str):
        """
        Prepare data from Qlib dataset.

        Returns:
            X: (N, total_features) tensor
            y: (N,) tensor
            index: MultiIndex (datetime, instrument)
        """
        features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
        labels = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_R)

        if isinstance(labels, pd.DataFrame):
            labels = labels.iloc[:, 0]

        # Align indices
        common_idx = features.index.intersection(labels.index)
        features = features.loc[common_idx]
        labels = labels.loc[common_idx]

        # Remove NaN labels
        valid = ~labels.isna()
        features = features[valid]
        labels = labels[valid]

        # Fill NaN features with 0 (z-scored, so 0 = mean)
        features = features.fillna(0)

        # Convert to tensors
        X = torch.tensor(features.values, dtype=torch.float32)
        y = torch.tensor(labels.values, dtype=torch.float32)

        # Clip extreme values
        X = torch.clamp(X, -10, 10)

        return X, y, features.index

    def _compute_ic(self, pred: np.ndarray, label: np.ndarray, index) -> float:
        """Compute average cross-sectional IC (Spearman rank correlation)."""
        from scipy.stats import spearmanr

        df = pd.DataFrame({'pred': pred, 'label': label}, index=index)

        ics = []
        for _, group in df.groupby(level='datetime'):
            if len(group) >= 5:
                ic, _ = spearmanr(group['pred'], group['label'])
                if not np.isnan(ic):
                    ics.append(ic)

        return np.mean(ics) if ics else 0.0

    def _ic_loss(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Differentiable IC loss (negative Pearson correlation)."""
        pred_centered = pred - pred.mean()
        label_centered = label - label.mean()

        pred_std = pred_centered.std()
        label_std = label_centered.std()

        if pred_std < 1e-8 or label_std < 1e-8:
            return torch.tensor(0.0, device=pred.device)

        ic = (pred_centered * label_centered).mean() / (pred_std * label_std)
        return -ic

    def _update_grid(self, X_train: torch.Tensor):
        """Update B-spline grids in KAN sub-layers using a data sample."""
        saved_state = copy.deepcopy(self.model.state_dict())

        self.model.eval()
        sample_size = min(4096, len(X_train))
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        sample = X_train[indices].to(self.device)

        # Reshape to sequence format
        batch_size = sample.size(0)
        sample_seq = sample.view(batch_size, self.seq_len, self.d_feat)
        sample_seq = self.model.input_norm(sample_seq)

        # Update grids in each cell's KAN sub-layers
        with torch.no_grad():
            for cell in self.model.cells:
                for sub_layer in cell.sub_layers:
                    if isinstance(sub_layer, KANLinear):
                        # Project a representative input through the sub-layer projection
                        # Use the first timestep as representative
                        step_input = sample_seq[:, 0, :]
                        sub_input = torch.matmul(step_input, cell.sub_kernel_inputs[0])
                        sub_layer.update_grid(sub_input)

        # Check for NaN
        has_nan = any(torch.isnan(p).any().item() for p in self.model.parameters())
        if has_nan:
            print("    WARNING: NaN weights after grid update, reverting")
            self.model.load_state_dict(saved_state)

        self.model.train()

    def fit(self, dataset: DatasetH):
        """Train TKAN model."""
        print("\n    Preparing data...")
        t0 = time.time()
        X_train, y_train, idx_train = self._prepare_data(dataset, "train")
        X_valid, y_valid, idx_valid = self._prepare_data(dataset, "valid")
        print(f"    Data prepared in {time.time() - t0:.1f}s")

        # Auto-detect dimensions
        total_features = X_train.shape[1]
        if total_features != self.d_feat * self.seq_len:
            if total_features % self.d_feat == 0:
                self.seq_len = total_features // self.d_feat
                print(f"    Auto-detected seq_len={self.seq_len} from "
                      f"total_features={total_features}, d_feat={self.d_feat}")
            else:
                print(f"    Total features ({total_features}) not divisible by "
                      f"d_feat ({self.d_feat}), using d_feat=total_features, seq_len=1")
                self.d_feat = total_features
                self.seq_len = 1

        print(f"    Train: {X_train.shape[0]:,} samples, {total_features} features")
        print(f"    Valid: {X_valid.shape[0]:,} samples")
        print(f"    Sequence: {self.seq_len} timesteps x {self.d_feat} features")

        # Data diagnostics
        print(f"    Feature range: [{X_train.min():.2f}, {X_train.max():.2f}]")
        print(f"    Label  range: [{y_train.min():.4f}, {y_train.max():.4f}], "
              f"mean={y_train.mean():.4f}, std={y_train.std():.4f}")

        # Build model
        self.model = self._build_model()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"    Parameters: {total_params:,} total, {trainable_params:,} trainable")

        # DataLoader
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size, shuffle=True, drop_last=False,
        )

        # Optimizer & scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_epochs, eta_min=1e-6)

        # Early stopping
        early_stopping = EarlyStopping(patience=self.early_stop_patience)

        # Training loop
        print(f"\n    Training ({self.n_epochs} epochs, loss={self.loss_type}, "
              f"grid_update every {self.grid_update_freq} epochs)...")

        for epoch in range(self.n_epochs):
            # Grid update (skip epoch 0)
            do_grid_update = (epoch > 0 and epoch % self.grid_update_freq == 0)
            if do_grid_update:
                self._update_grid(X_train)

            # --- Train ---
            self.model.train()
            total_loss = 0.0
            total_mse = 0.0
            total_reg = 0.0
            num_batches = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                pred = self.model(batch_X)

                # Task loss
                if self.loss_type == 'ic':
                    task_loss = self._ic_loss(pred, batch_y)
                else:
                    task_loss = F.mse_loss(pred, batch_y)

                # KAN regularization
                reg_loss = self.model.regularization_loss()
                loss = task_loss + self.reg_lambda * reg_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_mse += F.mse_loss(pred, batch_y).item()
                total_reg += reg_loss.item()
                num_batches += 1

            scheduler.step()

            avg_loss = total_loss / num_batches
            avg_mse = total_mse / num_batches
            avg_reg = total_reg / num_batches

            # --- Validate (batched) ---
            self.model.eval()
            val_preds = []
            with torch.no_grad():
                for i in range(0, len(X_valid), self.batch_size):
                    batch = X_valid[i:i + self.batch_size].to(self.device)
                    val_preds.append(self.model(batch).cpu())

            val_pred = torch.cat(val_preds)
            val_mse = F.mse_loss(val_pred, y_valid).item()
            val_ic = self._compute_ic(val_pred.numpy(), y_valid.numpy(), idx_valid)

            # Print progress
            lr_now = optimizer.param_groups[0]['lr']
            grid_tag = " [grid]" if do_grid_update else ""
            print(f"    Epoch {epoch + 1:3d}/{self.n_epochs}: "
                  f"loss={avg_loss:.4e} mse={avg_mse:.4e} reg={avg_reg:.2e} | "
                  f"val_mse={val_mse:.4e} IC={val_ic:.4f} lr={lr_now:.1e}{grid_tag}")

            # Early stopping on validation IC
            if early_stopping(val_ic, self.model):
                print(f"    Early stopping at epoch {epoch + 1} "
                      f"(best IC: {early_stopping.best_score:.4f})")
                break

        # Restore best model
        early_stopping.restore(self.model)
        self.fitted = True
        print(f"\n    Training completed (best IC: {early_stopping.best_score:.4f})")

    def predict(self, dataset: DatasetH, segment: str = "test") -> pd.Series:
        """Generate predictions as pd.Series with (datetime, instrument) index."""
        if not self.fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X, _, index = self._prepare_data(dataset, segment)

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X[i:i + self.batch_size].to(self.device)
                pred = self.model(batch)
                predictions.append(pred.cpu().numpy())

        pred_np = np.concatenate(predictions)
        return pd.Series(pred_np, index=index, name='score')

    def save(self, path: str):
        """Save model checkpoint."""
        if self.model is None:
            raise ValueError("No model to save")

        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'd_feat': self.d_feat,
            'seq_len': self.seq_len,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'sub_kan_configs': self.sub_kan_configs,
            'sub_kan_output_dim': self.sub_kan_output_dim,
            'sub_kan_input_dim': self.sub_kan_input_dim,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'grid_size': self.grid_size,
            'spline_order': self.spline_order,
            'grid_range': self.grid_range,
        }
        torch.save(save_dict, path)
        print(f"    Model saved to: {path}")

    @classmethod
    def load(cls, path: str, GPU: int = 0) -> 'TKANStock':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        instance = cls(
            d_feat=checkpoint['d_feat'],
            seq_len=checkpoint['seq_len'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            sub_kan_configs=checkpoint.get('sub_kan_configs'),
            sub_kan_output_dim=checkpoint.get('sub_kan_output_dim'),
            sub_kan_input_dim=checkpoint.get('sub_kan_input_dim'),
            dropout=checkpoint['dropout'],
            recurrent_dropout=checkpoint.get('recurrent_dropout', 0.0),
            grid_size=checkpoint.get('grid_size', 5),
            spline_order=checkpoint.get('spline_order', 3),
            grid_range=checkpoint.get('grid_range', [-3, 3]),
            GPU=GPU,
        )

        instance.model = instance._build_model()
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.model.eval()
        instance.fitted = True

        print(f"    Model loaded from: {path}")
        return instance


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Testing TKAN stock prediction model...")

    # Test 1: Flat features (alpha158-talib-macro style)
    print("\n--- Test 1: Flat features (275 features, seq_len=1) ---")
    d_feat = 275
    batch_size = 32

    model = TKANNetwork(
        d_feat=d_feat,
        seq_len=1,
        hidden_size=64,
        num_layers=2,
        sub_kan_configs=[None, 3],  # 2 sub-layers: default KAN, spline_order=3
        grid_size=5,
        spline_order=3,
        grid_range=[-3, 3],
        dropout=0.1,
    )

    x = torch.randn(batch_size, d_feat)
    y = model(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:   {total_params:,}")

    # Test regularization
    reg = model.regularization_loss()
    print(f"  Regularization loss: {reg.item():.4f}")

    # Test 2: Sequential features (alpha360 style)
    print("\n--- Test 2: Sequential features (6 features x 60 timesteps) ---")
    d_feat2 = 6
    seq_len2 = 60

    model2 = TKANNetwork(
        d_feat=d_feat2,
        seq_len=seq_len2,
        hidden_size=64,
        num_layers=2,
        sub_kan_configs=[None],
        grid_size=5,
        spline_order=3,
        grid_range=[-3, 3],
    )

    x2 = torch.randn(batch_size, d_feat2 * seq_len2)  # 360 flat features
    y2 = model2(x2)
    print(f"  Input shape:  {x2.shape}")
    print(f"  Output shape: {y2.shape}")

    total_params2 = sum(p.numel() for p in model2.parameters())
    print(f"  Parameters:   {total_params2:,}")

    # Test 3: Sub-layer with string activation
    print("\n--- Test 3: Mixed sub-layers (KAN + relu) ---")
    model3 = TKANNetwork(
        d_feat=158,
        seq_len=1,
        hidden_size=64,
        num_layers=1,
        sub_kan_configs=[None, 'relu'],
    )

    x3 = torch.randn(batch_size, 158)
    y3 = model3(x3)
    print(f"  Input shape:  {x3.shape}")
    print(f"  Output shape: {y3.shape}")

    total_params3 = sum(p.numel() for p in model3.parameters())
    print(f"  Parameters:   {total_params3:,}")

    # Test backward pass
    print("\n--- Test 4: Backward pass ---")
    target = torch.randn(batch_size)
    loss = F.mse_loss(y, target)
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients computed successfully")

    print("\nAll tests passed!")
