"""
Neural network components for DoubleAdapt + FiLM.

Components:
  - FeatureAdapter: multi-head cosine-gated affine transform on stock features
  - LabelAdapter + LabelAdaptHeads: gated scale+bias on labels
  - GRUFiLM: GRU backbone with FiLM macro conditioning
  - ForecastModelFiLM: wrapper with optimizer (extends for macro input)
  - DoubleAdaptFiLM: combines adapters + FiLM backbone

Based on DoubleAdapt (KDD'23) with FiLM macro modulation extension.
"""

import math
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F, init


def cosine(x1, x2, eps=1e-8):
    x1 = x1 / (torch.norm(x1, p=2, dim=-1, keepdim=True) + eps)
    x2 = x2 / (torch.norm(x2, p=2, dim=-1, keepdim=True) + eps)
    return x1 @ x2.transpose(0, 1)


# ============================================================
# Adapters (from original DoubleAdapt)
# ============================================================

class LabelAdaptHeads(nn.Module):
    def __init__(self, num_head):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, num_head))
        self.bias = nn.Parameter(torch.ones(1, num_head) / 8)
        init.uniform_(self.weight, 0.75, 1.25)

    def forward(self, y, inverse=False):
        if inverse:
            return (y.view(-1, 1) - self.bias) / (self.weight + 1e-9)
        else:
            return (self.weight + 1e-9) * y.view(-1, 1) + self.bias


class LabelAdapter(nn.Module):
    def __init__(self, x_dim, num_head=4, temperature=4, hid_dim=32):
        super().__init__()
        self.num_head = num_head
        self.linear = nn.Linear(x_dim, hid_dim, bias=False)
        self.P = nn.Parameter(torch.empty(num_head, hid_dim))
        init.kaiming_uniform_(self.P, a=math.sqrt(5))
        self.heads = LabelAdaptHeads(num_head)
        self.temperature = temperature

    def forward(self, x, y, inverse=False):
        v = self.linear(x.reshape(len(x), -1))
        gate = cosine(v, self.P)
        gate = torch.softmax(gate / self.temperature, -1)
        return (gate * self.heads(y, inverse=inverse)).sum(-1)


class FeatureAdapter(nn.Module):
    def __init__(self, in_dim, num_head=4, temperature=4):
        super().__init__()
        self.num_head = num_head
        self.P = nn.Parameter(torch.empty(num_head, in_dim))
        init.kaiming_uniform_(self.P, a=math.sqrt(5))
        self.heads = nn.ModuleList([
            nn.Linear(in_dim, in_dim, bias=True) for _ in range(num_head)
        ])
        self.temperature = temperature

    def forward(self, x):
        """x: [batch, seq_len, in_dim] -> adapted [batch, seq_len, in_dim]"""
        s_hat = torch.cat(
            [torch.cosine_similarity(x, self.P[i], dim=-1).unsqueeze(-1)
             for i in range(self.num_head)],
            -1,
        )
        s = torch.softmax(s_hat / self.temperature, -1).unsqueeze(-1)
        return x + sum(s[..., i, :] * self.heads[i](x) for i in range(self.num_head))


# ============================================================
# GRU + FiLM backbone
# ============================================================

class GRUFiLM(nn.Module):
    """GRU backbone with FiLM macro conditioning.

    Stock features go through GRU, then the hidden state is modulated
    by macro-derived gamma/beta before the final prediction head.
    """

    def __init__(
        self,
        d_feat=5,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_macro=6,
        film_hidden=32,
    ):
        super().__init__()
        self.d_feat = d_feat
        self.hidden_size = hidden_size

        self.fc_in = nn.Linear(d_feat, hidden_size)
        self.rnn = nn.GRU(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )

        # FiLM generator: macro -> (gamma, beta)
        self.film_generator = nn.Sequential(
            nn.Linear(n_macro, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, hidden_size * 2),
        )
        # Initialize gamma=1, beta=0 (identity transform)
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)
        self.film_generator[-1].bias.data[:hidden_size] = 1.0

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x, x_macro=None):
        """
        Args:
            x: [batch, 300] flat OR [batch, 60, 5] 3D (after FeatureAdapter)
            x_macro: [batch, n_macro] or None
        Returns:
            predictions: [batch]
        """
        # Handle flat input from raw alpha300
        if x.dim() == 2:
            batch = x.size(0)
            x = x.view(batch, self.d_feat, -1).permute(0, 2, 1)  # [batch, 60, d_feat]

        x = torch.tanh(self.fc_in(x))   # [batch, seq_len, hidden_size]
        out, _ = self.rnn(x)             # [batch, seq_len, hidden_size]
        hidden = out[:, -1, :]           # [batch, hidden_size]

        # FiLM modulation
        if x_macro is not None:
            film = self.film_generator(x_macro)
            gamma, beta = film.chunk(2, dim=1)
            hidden = gamma * hidden + beta

        return self.fc_out(self.dropout(hidden)).squeeze(-1)


# ============================================================
# Forecast model wrappers
# ============================================================

class ForecastModelFiLM(nn.Module):
    """Wrapper around GRUFiLM with optimizer, extending ForecastModel for macro input."""

    def __init__(self, model, x_dim=None, lr=0.001, weight_decay=0, need_permute=False):
        super().__init__()
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.need_permute = need_permute
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.to(self.device)

    def forward(self, X, macro=None, model=None):
        """
        Args:
            X: [batch, x_dim] flat or [batch, seq_len, d_feat] 3D
            macro: [batch, n_macro] or None
            model: optional functional model from higher
        Returns:
            predictions [batch]
        """
        if model is None:
            model = self.model

        # If 3D with need_permute, reshape to flat then let GRUFiLM handle it
        if X.dim() == 3 and self.need_permute:
            X = X.permute(0, 2, 1).reshape(len(X), -1)
        elif X.dim() == 3:
            # Already [batch, seq_len, d_feat], pass through directly
            pass

        if macro is not None:
            y_hat = model(X, macro)
        else:
            y_hat = model(X)
        return y_hat.view(-1)


class DoubleAdaptFiLM(ForecastModelFiLM):
    """DoubleAdapt with FiLM: FeatureAdapter + LabelAdapter + GRUFiLM backbone."""

    def __init__(
        self, model, factor_num=5, x_dim=None, lr=0.001, weight_decay=0,
        need_permute=False, num_head=8, temperature=10,
    ):
        super().__init__(
            model, x_dim=x_dim, lr=lr,
            need_permute=need_permute, weight_decay=weight_decay,
        )
        self.teacher_x = FeatureAdapter(factor_num, num_head, temperature)
        self.teacher_y = LabelAdapter(
            factor_num if x_dim is None else x_dim, num_head, temperature
        )
        self.meta_params = list(self.teacher_x.parameters()) + list(self.teacher_y.parameters())
        self.to(self.device)

    def forward(self, X, macro=None, model=None, transform=False):
        """
        Args:
            X: [batch, x_dim] or [batch, seq_len, factor_num]
            macro: [batch, n_macro] or None
            model: functional model from higher (for inner loop)
            transform: whether to apply FeatureAdapter
        Returns:
            (predictions, X) - X may be adapted if transform=True
        """
        if transform:
            # FeatureAdapter expects [batch, seq_len, factor_num]
            if X.dim() == 2:
                batch = X.size(0)
                d_feat = self.teacher_x.P.size(1)  # factor_num
                X = X.view(batch, d_feat, -1).permute(0, 2, 1)
            X = self.teacher_x(X)
        return super().forward(X, macro, model), X


# ============================================================
# Utilities
# ============================================================

def override_state(groups, new_opt):
    """Copy optimizer state from higher's differentiable optimizer back to original."""
    saved_groups = new_opt.param_groups
    id_map = {
        old_id: p
        for old_id, p in zip(range(len(saved_groups[0]["params"])), groups[0]["params"])
    }
    state = defaultdict(dict)
    for k, v in new_opt.state[0].items():
        if k in id_map:
            param = id_map[k]
            for _k, _v in v.items():
                state[param][_k] = _v.detach() if isinstance(_v, torch.Tensor) else _v
        else:
            state[k] = v
    return state


def has_rnn(module):
    """Check if module contains any RNN submodule."""
    for m in module.modules():
        if isinstance(m, nn.RNNBase):
            return True
    return False
