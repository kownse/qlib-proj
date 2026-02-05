"""
StockMixer: AAAI 2024
Complete port of the official implementation, adapted for Qlib.

Paper: StockMixer: A Simple yet Strong MLP-based Architecture for Stock Price Forecasting

Key insight: StockMixer processes ALL stocks simultaneously per date, enabling:
1. Multi-scale time mixing with causal (TriU) constraints
2. Stock-dimension mixing to capture cross-stock relationships
3. Pairwise ranking loss for better relative predictions

Data format requirements:
- Input: (num_stocks, time_steps, channels) per date
- Output: (num_stocks, 1) predictions per date
- Training iterates over dates, each batch contains all stocks for one date

Original implementation: https://github.com/SJTU-Quant/StockMixer
"""

import os
import copy
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


# ============================================================================
# Loss Functions
# ============================================================================

def get_stockmixer_loss(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    base_price: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    StockMixer loss function: MSE + Ranking Loss

    The ranking loss encourages correct pairwise ordering of predictions.

    Args:
        prediction: Model output (num_stocks, 1)
        ground_truth: Target returns (num_stocks, 1)
        base_price: Base price for return calculation (num_stocks, 1)
        mask: Valid stock mask (num_stocks, 1)
        alpha: Weight for ranking loss

    Returns:
        total_loss, reg_loss, rank_loss, return_ratio
    """
    device = prediction.device
    batch_size = prediction.shape[0]
    all_one = torch.ones(batch_size, 1, dtype=torch.float32, device=device)

    # Return ratio (predicted return)
    return_ratio = (prediction - base_price) / (base_price + 1e-8)

    # Regression loss (MSE)
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)

    # Pairwise ranking loss
    # pre_pw_dif[i,j] = return_ratio[i] - return_ratio[j]
    pre_pw_dif = return_ratio @ all_one.t() - all_one @ return_ratio.t()
    # gt_pw_dif[i,j] = ground_truth[j] - ground_truth[i]
    gt_pw_dif = all_one @ ground_truth.t() - ground_truth @ all_one.t()
    # Mask for valid pairs
    mask_pw = mask @ mask.t()

    # If gt[i] > gt[j], we want pred[i] > pred[j]
    # Loss when: pred[i] < pred[j] but gt[i] > gt[j]
    rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw))

    total_loss = reg_loss + alpha * rank_loss
    return total_loss, reg_loss, rank_loss, return_ratio


def get_stockmixer_loss_simple(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simplified StockMixer loss (directly using predictions as returns).

    Args:
        prediction: Model output as return predictions (num_stocks, 1)
        ground_truth: Target returns (num_stocks, 1)
        mask: Valid stock mask (num_stocks, 1)
        alpha: Weight for ranking loss

    Returns:
        total_loss, reg_loss, rank_loss
    """
    device = prediction.device
    batch_size = prediction.shape[0]
    all_one = torch.ones(batch_size, 1, dtype=torch.float32, device=device)

    # Regression loss (MSE) - only over valid stocks
    valid_count = mask.sum().clamp(min=1.0)
    diff = (prediction - ground_truth) * mask
    reg_loss = (diff ** 2).sum() / valid_count

    # Pairwise ranking loss
    pre_pw_dif = prediction @ all_one.t() - all_one @ prediction.t()
    gt_pw_dif = all_one @ ground_truth.t() - ground_truth @ all_one.t()
    mask_pw = mask @ mask.t()

    rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw))

    total_loss = reg_loss + alpha * rank_loss
    return total_loss, reg_loss, rank_loss


# ============================================================================
# Network Components (from official implementation)
# ============================================================================

class MixerBlock(nn.Module):
    """
    Standard MLP Mixer block.

    Linear -> GELU -> Dropout -> Linear -> Dropout
    """

    def __init__(self, mlp_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)
        self.act = nn.GELU()
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_1(x)
        x = self.act(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.dense_2(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TriU(nn.Module):
    """
    Triangular Linear Layer - Causal time modeling.

    Each time step i can only see information from [0:i+1].
    Prevents future information leakage.

    Implementation: Create different-sized linear layers for each time step.
    triU[i]: Linear(i+1, 1) for time step i
    """

    def __init__(self, time_step: int):
        super().__init__()
        self.time_step = time_step
        # Use nn.ModuleList (original uses ParameterList which is incorrect)
        self.triU = nn.ModuleList([
            nn.Linear(i + 1, 1) for i in range(time_step)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channel, time) or (batch, time, channel) depending on usage

        Returns:
            Same shape as input
        """
        outputs = [self.triU[0](x[:, :, 0:1])]
        for i in range(1, self.time_step):
            outputs.append(self.triU[i](x[:, :, :i+1]))
        return torch.cat(outputs, dim=-1)


class MultiScaleTimeMixer(nn.Module):
    """
    Multi-scale time mixing with TriU causal constraints.

    Scale 0: Original resolution (LayerNorm + TriU + Hardswish + TriU)
    Scale i: Downsampled by 2^i (Conv1d + TriU + Hardswish + TriU)

    Captures patterns at different time scales.
    """

    def __init__(self, time_step: int, channel: int, scale_count: int = 1):
        super().__init__()
        self.time_step = time_step
        self.scale_count = scale_count

        # Create layers for each scale
        self.mix_layers = nn.ModuleList()

        for i in range(scale_count):
            if i == 0:
                # Scale 0: Original resolution with LayerNorm
                layer = nn.Sequential(
                    nn.LayerNorm([time_step, channel]),
                    TriU(time_step),
                    nn.Hardswish(),
                    TriU(time_step)
                )
            else:
                # Scale i: Downsampled resolution
                downsampled_time = time_step // (2 ** i)
                if downsampled_time < 1:
                    downsampled_time = 1
                layer = nn.Sequential(
                    nn.Conv1d(channel, channel, kernel_size=2**i, stride=2**i),
                    TriU(downsampled_time),
                    nn.Hardswish(),
                    TriU(downsampled_time)
                )
            self.mix_layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, channel)

        Returns:
            Concatenated multi-scale features
        """
        # Permute for Conv1d: (batch, channel, time)
        x_perm = x.permute(0, 2, 1)

        # Scale 0 uses original format
        y = self.mix_layers[0](x)  # (batch, channel, time)

        # Permute back for consistency
        y = y.permute(0, 2, 1)  # (batch, time, channel)

        # Other scales
        for i in range(1, self.scale_count):
            yi = self.mix_layers[i](x_perm)  # (batch, channel, downsampled_time)
            yi = yi.permute(0, 2, 1)  # (batch, downsampled_time, channel)
            # Pad to match time dimension or concatenate
            y = torch.cat([y, yi], dim=1)

        return y


class Mixer2dTriU(nn.Module):
    """
    2D Mixer with TriU for time mixing and MixerBlock for channel mixing.

    Structure:
    1. LayerNorm -> TriU (time mixing) with residual
    2. LayerNorm -> MixerBlock (channel mixing) with residual
    """

    def __init__(self, time_steps: int, channels: int):
        super().__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = TriU(time_steps)
        self.channelMixer = MixerBlock(channels, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch, time_steps, channels)
        """
        # Time mixing with TriU
        x = self.LN_1(inputs)
        x = x.permute(0, 2, 1)  # (batch, channels, time_steps)
        x = self.timeMixer(x)
        x = x.permute(0, 2, 1)  # (batch, time_steps, channels)

        # Residual connection
        x = self.LN_2(x + inputs)

        # Channel mixing
        y = self.channelMixer(x)
        return x + y


class MultTime2dMixer(nn.Module):
    """
    Multi-scale time + 2D mixing.

    Combines original scale mixing with downsampled scale mixing.
    """

    def __init__(self, time_step: int, channel: int, scale_dim: int = 8):
        super().__init__()
        self.mix_layer = Mixer2dTriU(time_step, channel)
        self.scale_mix_layer = Mixer2dTriU(scale_dim, channel)

    def forward(self, inputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Original scale input (batch, time_step, channel)
            y: Downsampled input (batch, scale_dim, channel)

        Returns:
            Concatenated features (batch, time_step*2 + scale_dim, channel)
        """
        y = self.scale_mix_layer(y)
        x = self.mix_layer(inputs)
        return torch.cat([inputs, x, y], dim=1)


class NoGraphMixer(nn.Module):
    """
    Stock-dimension mixing (no graph structure).

    Mixes information across stocks using MLPs.
    This is the key component for cross-stock relationship learning.

    Structure:
    1. Permute to (features, stocks)
    2. LayerNorm
    3. Linear(stocks -> hidden) -> Hardswish -> Linear(hidden -> stocks)
    4. Permute back to (stocks, features)
    """

    def __init__(self, stocks: int, hidden_dim: int = 20):
        super().__init__()
        self.dense1 = nn.Linear(stocks, hidden_dim)
        self.activation = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, stocks)
        self.layer_norm_stock = nn.LayerNorm(stocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (stocks, features)

        Returns:
            (stocks, features) with cross-stock information mixed
        """
        x = inputs.permute(1, 0)  # (features, stocks)
        x = self.layer_norm_stock(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = x.permute(1, 0)  # (stocks, features)
        return x


class StockMixerNetwork(nn.Module):
    """
    Complete StockMixer network.

    Architecture:
    1. Conv1d downsampling
    2. MultTime2dMixer (multi-scale time + 2D mixing)
    3. channel_fc (reduce channel dimension)
    4. NoGraphMixer (stock mixing)
    5. time_fc (reduce to final prediction)

    Args:
        stocks: Number of stocks in the universe
        time_steps: Lookback window size
        channels: Number of features per time step
        market: Hidden dimension for NoGraphMixer
        scale: Number of scales (unused in current impl, kept for compatibility)
    """

    def __init__(
        self,
        stocks: int,
        time_steps: int,
        channels: int,
        market: int = 20,
        scale: int = 3
    ):
        super().__init__()

        self.stocks = stocks
        self.time_steps = time_steps
        self.channels = channels

        # Downsampling conv (halves time dimension)
        scale_dim = time_steps // 2
        if scale_dim < 1:
            scale_dim = 1

        self.conv = nn.Conv1d(channels, channels, kernel_size=2, stride=2)
        self.mixer = MultTime2dMixer(time_steps, channels, scale_dim=scale_dim)

        # Channel reduction
        self.channel_fc = nn.Linear(channels, 1)

        # Stock mixing
        self.stock_mixer = NoGraphMixer(stocks, market)

        # Time reduction to single prediction
        # After MultTime2dMixer: time_steps * 2 + scale_dim
        combined_time = time_steps * 2 + scale_dim
        self.time_fc = nn.Linear(combined_time, 1)
        self.time_fc_ = nn.Linear(combined_time, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (stocks, time_steps, channels)

        Returns:
            predictions: (stocks, 1)
        """
        # Downsample for multi-scale
        x = inputs.permute(0, 2, 1)  # (stocks, channels, time_steps)
        x = self.conv(x)  # (stocks, channels, time_steps/2)
        x = x.permute(0, 2, 1)  # (stocks, time_steps/2, channels)

        # Multi-scale time mixing
        y = self.mixer(inputs, x)  # (stocks, time_steps*2 + scale_dim, channels)

        # Reduce channel dimension
        y = self.channel_fc(y).squeeze(-1)  # (stocks, time_steps*2 + scale_dim)

        # Stock mixing branch
        z = self.stock_mixer(y)  # (stocks, time_steps*2 + scale_dim)

        # Time reduction
        y = self.time_fc(y)  # (stocks, 1)
        z = self.time_fc_(z)  # (stocks, 1)

        return y + z


# ============================================================================
# Qlib Model Interface
# ============================================================================

class EarlyStopping:
    """Early stopping mechanism."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = None
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best:
                self.best_state = copy.deepcopy(model.state_dict())
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False

    def restore(self, model: nn.Module):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)


class StockMixer:
    """
    StockMixer model adapted for Qlib interface.

    Key differences from standard Qlib models:
    1. Training iterates by DATE, not by sample
    2. Each batch contains ALL stocks for a single date
    3. Model requires fixed stock universe

    Parameters
    ----------
    num_stocks : int
        Number of stocks in the universe
    time_steps : int
        Lookback window (default: 60 for Alpha300)
    channels : int
        Features per time step (default: 5 for OHLCV)
    market_num : int
        Hidden dimension for stock mixing (default: 20)
    scale_factor : int
        Number of scales for multi-scale mixing (default: 3)
    learning_rate : float
        Learning rate (default: 0.001)
    alpha : float
        Ranking loss weight (default: 0.1)
    n_epochs : int
        Training epochs (default: 100)
    early_stop : int
        Early stopping patience (default: 10)
    GPU : int
        GPU device ID (-1 for CPU)
    seed : int
        Random seed
    """

    def __init__(
        self,
        num_stocks: int,
        time_steps: int = 60,
        channels: int = 5,
        market_num: int = 20,
        scale_factor: int = 3,
        learning_rate: float = 0.001,
        alpha: float = 0.1,
        n_epochs: int = 100,
        early_stop: int = 10,
        GPU: int = 0,
        seed: int = 42,
    ):
        self.num_stocks = num_stocks
        self.time_steps = time_steps
        self.channels = channels
        self.market_num = market_num
        self.scale_factor = scale_factor
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.early_stop_patience = early_stop
        self.GPU = GPU
        self.seed = seed

        self.model: Optional[StockMixerNetwork] = None
        self.fitted = False
        self.device = None

        # Stock index mapping (filled during training)
        self.stock_to_idx: Dict[str, int] = {}
        self.idx_to_stock: Dict[int, str] = {}

        self._set_seed()
        self._setup_device()

    def _set_seed(self):
        """Set random seeds."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _setup_device(self):
        """Configure GPU/CPU."""
        if self.GPU >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.GPU}')
            print(f"    Using GPU: cuda:{self.GPU}")
        else:
            self.device = torch.device('cpu')
            print("    Using CPU")

    def _build_model(self) -> StockMixerNetwork:
        """Build StockMixer network."""
        model = StockMixerNetwork(
            stocks=self.num_stocks,
            time_steps=self.time_steps,
            channels=self.channels,
            market=self.market_num,
            scale=self.scale_factor,
        )
        return model.to(self.device)

    def _prepare_data_by_date(
        self,
        dataset: DatasetH,
        segment: str
    ) -> Tuple[Dict, Dict, Dict, List[str], List[str]]:
        """
        Prepare data grouped by date for StockMixer (optimized version).

        StockMixer requires all stocks to be processed together per date.
        This method reshapes Qlib data from (datetime, instrument) to
        per-date tensors of shape (num_stocks, time_steps, channels).

        Returns:
            data_by_date: dict[date] -> (num_stocks, time_steps, channels)
            labels_by_date: dict[date] -> (num_stocks,)
            masks_by_date: dict[date] -> (num_stocks,) valid stock indicators
            dates: sorted list of dates
            instruments: sorted list of instrument names
        """
        import time
        start_time = time.time()

        # Get data: features are z-scored (DK_L), labels are raw returns (DK_R)
        # Using raw labels matches the original StockMixer which predicts returns directly
        features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
        labels = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_R)

        if isinstance(labels, pd.DataFrame):
            labels = labels.iloc[:, 0]

        # Align indices: DK_L may have fewer rows than DK_R (DropnaLabel in learn_processors)
        common_idx = features.index.intersection(labels.index)
        if len(common_idx) < len(features):
            features = features.loc[common_idx]
        if len(common_idx) < len(labels):
            labels = labels.loc[common_idx]

        # Get unique dates and instruments
        dates = sorted(features.index.get_level_values('datetime').unique())
        segment_instruments = sorted(features.index.get_level_values('instrument').unique())

        # Build instrument index mapping (only on first call, typically for training)
        # StockMixer requires FIXED stock universe - use training stocks for all segments
        if not self.stock_to_idx:
            self.stock_to_idx = {inst: i for i, inst in enumerate(segment_instruments)}
            self.idx_to_stock = {i: inst for inst, i in self.stock_to_idx.items()}
            self.num_stocks = len(segment_instruments)
            print(f"    Stock universe initialized: {self.num_stocks} stocks")

        # Use the fixed stock set from initialization
        instruments = list(self.stock_to_idx.keys())
        num_stocks = len(instruments)
        num_features = features.shape[1]

        # Infer time_steps and channels
        if num_features != self.time_steps * self.channels:
            possible_channels = [5, 6]
            for ch in possible_channels:
                if num_features % ch == 0:
                    inferred_time_steps = num_features // ch
                    if inferred_time_steps > 0:
                        self.time_steps = inferred_time_steps
                        self.channels = ch
                        print(f"    Inferred: time_steps={self.time_steps}, channels={self.channels}")
                        break

        # Minimum valid stocks threshold
        min_valid_stocks = max(2, num_stocks // 10)

        # Convert to numpy for faster processing
        features_np = features.values.astype(np.float32)
        labels_np = labels.values.astype(np.float32)

        # Get index arrays
        date_idx = features.index.get_level_values('datetime')
        inst_idx = features.index.get_level_values('instrument')

        # Create mapping arrays
        inst_to_pos = np.array([self.stock_to_idx.get(inst, -1) for inst in inst_idx])

        data_by_date = {}
        labels_by_date = {}
        masks_by_date = {}

        print(f"    Processing {len(dates)} dates...", end=" ", flush=True)

        for i, date in enumerate(dates):
            # Find rows for this date
            date_mask = date_idx == date
            date_features = features_np[date_mask]
            date_labels_vals = labels_np[date_mask]
            date_inst_pos = inst_to_pos[date_mask]

            # Initialize tensors
            data_tensor = torch.zeros(num_stocks, self.time_steps, self.channels)
            label_tensor = torch.zeros(num_stocks)
            mask_tensor = torch.zeros(num_stocks)

            # Filter valid instruments (in our stock universe)
            valid_inst_mask = date_inst_pos >= 0
            if not valid_inst_mask.any():
                continue

            valid_features = date_features[valid_inst_mask]
            valid_labels = date_labels_vals[valid_inst_mask]
            valid_positions = date_inst_pos[valid_inst_mask]

            # Check for NaN in features (row-wise)
            nan_mask = ~np.isnan(valid_features).any(axis=1)
            nan_label_mask = ~np.isnan(valid_labels)
            final_mask = nan_mask & nan_label_mask

            if not final_mask.any():
                continue

            valid_features = valid_features[final_mask]
            valid_labels = valid_labels[final_mask]
            valid_positions = valid_positions[final_mask]

            # Reshape and fill tensors
            # Alpha300 features are stored channels-first:
            #   [CLOSE59..CLOSE0, OPEN59..OPEN0, HIGH59..HIGH0, LOW59..LOW0, VOL59..VOL0]
            # So reshape to (n, channels, time_steps) then transpose to (n, time_steps, channels)
            try:
                reshaped_features = valid_features.reshape(
                    -1, self.channels, self.time_steps
                ).transpose(0, 2, 1)  # -> (n, time_steps, channels)
                for j, pos in enumerate(valid_positions):
                    data_tensor[pos] = torch.from_numpy(reshaped_features[j].copy())
                    label_tensor[pos] = float(valid_labels[j])
                    mask_tensor[pos] = 1.0
            except ValueError:
                # Shape mismatch, skip this date
                continue

            # Only keep dates with enough valid stocks
            valid_count = int(mask_tensor.sum().item())
            if valid_count >= min_valid_stocks:
                data_by_date[date] = data_tensor
                labels_by_date[date] = label_tensor
                masks_by_date[date] = mask_tensor

            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"{i+1}", end=" ", flush=True)

        elapsed = time.time() - start_time
        print(f"Done ({elapsed:.1f}s)")

        return data_by_date, labels_by_date, masks_by_date, dates, instruments

    def fit(self, dataset: DatasetH):
        """
        Train StockMixer model.

        Training process:
        1. Group data by date
        2. For each epoch, iterate through dates (shuffled)
        3. Each batch is all stocks for one date
        4. Use MSE + Ranking loss
        """
        print("\n    Preparing training data (grouped by date)...")
        train_data, train_labels, train_masks, train_dates, instruments = \
            self._prepare_data_by_date(dataset, "train")
        valid_data, valid_labels, valid_masks, valid_dates, _ = \
            self._prepare_data_by_date(dataset, "valid")

        # Update num_stocks based on actual data
        actual_num_stocks = len(instruments)
        if actual_num_stocks != self.num_stocks:
            print(f"    Updating num_stocks: {self.num_stocks} -> {actual_num_stocks}")
            self.num_stocks = actual_num_stocks

        train_dates_list = [d for d in train_dates if d in train_data]
        valid_dates_list = [d for d in valid_dates if d in valid_data]

        print(f"    Train dates: {len(train_dates_list)}, Valid dates: {len(valid_dates_list)}")
        print(f"    Stock universe: {self.num_stocks}")
        print(f"    Data shape per date: ({self.num_stocks}, {self.time_steps}, {self.channels})")

        # === Diagnostic logging ===
        if train_dates_list:
            sample_date = train_dates_list[0]
            sample_data = train_data[sample_date]
            sample_labels = train_labels[sample_date]
            sample_masks = train_masks[sample_date]
            valid_count = int(sample_masks.sum().item())
            valid_idx = torch.where(sample_masks > 0)[0]

            print(f"\n    --- Data Diagnostics (date={sample_date}) ---")
            print(f"    Valid stocks: {valid_count}/{self.num_stocks}")
            if valid_count > 0:
                first_valid = valid_idx[0].item()
                sample_stock = sample_data[first_valid]  # (time_steps, channels)
                print(f"    Sample stock[{first_valid}] shape: {sample_stock.shape}")
                print(f"    Sample stock[{first_valid}] channel means: {sample_stock.mean(dim=0).tolist()}")
                print(f"    Sample stock[{first_valid}] channel stds:  {sample_stock.std(dim=0).tolist()}")
                print(f"    Sample stock[{first_valid}] [t=0]: {sample_stock[0].tolist()}")
                print(f"    Sample stock[{first_valid}] [t=-1]: {sample_stock[-1].tolist()}")

                valid_labels_vals = sample_labels[valid_idx]
                print(f"    Labels - mean: {valid_labels_vals.mean():.6f}, std: {valid_labels_vals.std():.6f}, "
                      f"min: {valid_labels_vals.min():.6f}, max: {valid_labels_vals.max():.6f}")
        print()

        # Build model
        print("\n    Building StockMixer network...")
        self.model = self._build_model()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"    Total parameters: {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")

        # Optimizer and scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

        # Early stopping
        early_stopping = EarlyStopping(patience=self.early_stop_patience, restore_best=True)

        # Training loop with gradient accumulation for speed
        accum_steps = 8  # Accumulate gradients over 8 dates before updating
        print(f"\n    Training (gradient accumulation over {accum_steps} dates)...")

        for epoch in range(self.n_epochs):
            # Training phase
            self.model.train()
            train_losses = {'total': 0, 'reg': 0, 'rank': 0}
            np.random.shuffle(train_dates_list)
            optimizer.zero_grad()

            for di, date in enumerate(train_dates_list):
                data_batch = train_data[date].to(self.device)
                label_batch = train_labels[date].unsqueeze(1).to(self.device)
                mask_batch = train_masks[date].unsqueeze(1).to(self.device)

                # Forward pass
                prediction = self.model(data_batch)

                # Diagnostic: log model output range on first batch of first epoch
                if epoch == 0 and di == 0:
                    with torch.no_grad():
                        valid_mask = mask_batch.squeeze() > 0
                        valid_pred = prediction.squeeze()[valid_mask]
                        valid_gt = label_batch.squeeze()[valid_mask]
                        print(f"\n    --- Epoch 0, Batch 0 Diagnostics ---")
                        print(f"    Prediction range: [{valid_pred.min():.6f}, {valid_pred.max():.6f}], "
                              f"mean={valid_pred.mean():.6f}, std={valid_pred.std():.6f}")
                        print(f"    Label range:      [{valid_gt.min():.6f}, {valid_gt.max():.6f}], "
                              f"mean={valid_gt.mean():.6f}, std={valid_gt.std():.6f}")
                        print(f"    Valid stocks: {valid_mask.sum().item()}")

                # Compute loss (scale by accumulation steps)
                total_loss, reg_loss, rank_loss = get_stockmixer_loss_simple(
                    prediction, label_batch, mask_batch, self.alpha
                )
                scaled_loss = total_loss / accum_steps

                # Backward pass (accumulate gradients)
                scaled_loss.backward()

                # Update every accum_steps dates
                if (di + 1) % accum_steps == 0 or (di + 1) == len(train_dates_list):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                train_losses['total'] += total_loss.item()
                train_losses['reg'] += reg_loss.item()
                train_losses['rank'] += rank_loss.item()

            # Average losses
            num_train = len(train_dates_list)
            for key in train_losses:
                train_losses[key] /= num_train

            # Validation phase
            self.model.eval()
            val_losses = {'total': 0, 'reg': 0, 'rank': 0}

            with torch.no_grad():
                for date in valid_dates_list:
                    data_batch = valid_data[date].to(self.device)
                    label_batch = valid_labels[date].unsqueeze(1).to(self.device)
                    mask_batch = valid_masks[date].unsqueeze(1).to(self.device)

                    prediction = self.model(data_batch)
                    total_loss, reg_loss, rank_loss = get_stockmixer_loss_simple(
                        prediction, label_batch, mask_batch, self.alpha
                    )

                    val_losses['total'] += total_loss.item()
                    val_losses['reg'] += reg_loss.item()
                    val_losses['rank'] += rank_loss.item()

            num_valid = len(valid_dates_list)
            for key in val_losses:
                val_losses[key] /= num_valid

            # Learning rate scheduling
            scheduler.step(val_losses['total'])

            # Print progress
            print(f"    Epoch {epoch+1:3d}/{self.n_epochs}: "
                  f"train_loss={train_losses['total']:.4e} (reg={train_losses['reg']:.4e}, rank={train_losses['rank']:.4e}), "
                  f"val_loss={val_losses['total']:.4e}")

            # Early stopping
            if early_stopping(val_losses['total'], self.model):
                print(f"    Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        early_stopping.restore(self.model)
        self.fitted = True
        print("\n    Training completed")

    def predict(self, dataset: DatasetH, segment: str = "test") -> pd.Series:
        """
        Generate predictions.

        Returns predictions with (datetime, instrument) MultiIndex.
        """
        if not self.fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Prepare data
        data_by_date, labels_by_date, masks_by_date, dates, _ = \
            self._prepare_data_by_date(dataset, segment)

        dates_list = [d for d in dates if d in data_by_date]

        # Predict
        self.model.eval()
        all_predictions = []
        all_indices = []

        with torch.no_grad():
            for date in dates_list:
                data_batch = data_by_date[date].to(self.device)
                mask_batch = masks_by_date[date]

                prediction = self.model(data_batch)
                pred_np = prediction.cpu().numpy().flatten()

                # Collect predictions for valid stocks
                for idx in range(self.num_stocks):
                    if mask_batch[idx] > 0:
                        inst = self.idx_to_stock.get(idx)
                        if inst is not None:
                            all_predictions.append(pred_np[idx])
                            all_indices.append((date, inst))

        # Create Series with MultiIndex
        index = pd.MultiIndex.from_tuples(all_indices, names=['datetime', 'instrument'])
        pred_series = pd.Series(all_predictions, index=index, name='score')

        return pred_series

    def save(self, path: str):
        """Save model."""
        if self.model is None:
            raise ValueError("No model to save")

        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'num_stocks': self.num_stocks,
            'time_steps': self.time_steps,
            'channels': self.channels,
            'market_num': self.market_num,
            'scale_factor': self.scale_factor,
            'alpha': self.alpha,
            'stock_to_idx': self.stock_to_idx,
            'idx_to_stock': self.idx_to_stock,
        }
        torch.save(save_dict, path)
        print(f"    Model saved to: {path}")

    @classmethod
    def load(cls, path: str, GPU: int = 0) -> 'StockMixer':
        """Load model."""
        checkpoint = torch.load(path, map_location='cpu')

        instance = cls(
            num_stocks=checkpoint['num_stocks'],
            time_steps=checkpoint['time_steps'],
            channels=checkpoint['channels'],
            market_num=checkpoint['market_num'],
            scale_factor=checkpoint['scale_factor'],
            alpha=checkpoint['alpha'],
            GPU=GPU,
        )

        instance.stock_to_idx = checkpoint['stock_to_idx']
        instance.idx_to_stock = checkpoint['idx_to_stock']
        instance.model = instance._build_model()
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.model.eval()
        instance.fitted = True

        print(f"    Model loaded from: {path}")
        return instance


# ============================================================================
# Factory Function
# ============================================================================

def create_stockmixer(
    handler_type: str,
    num_stocks: int,
    **kwargs
) -> StockMixer:
    """
    Create StockMixer model based on handler type.

    Args:
        handler_type: Handler name (e.g., 'alpha300')
        num_stocks: Number of stocks in the universe
        **kwargs: Additional parameters

    Returns:
        StockMixer instance
    """
    # Handler-specific configurations
    HANDLER_CONFIGS = {
        'alpha300': {'time_steps': 60, 'channels': 5},
        'alpha300-ts': {'time_steps': 60, 'channels': 5},
        'alpha360': {'time_steps': 60, 'channels': 6},
        'alpha180': {'time_steps': 30, 'channels': 6},
    }

    config = HANDLER_CONFIGS.get(handler_type, {'time_steps': 60, 'channels': 5})

    return StockMixer(
        num_stocks=num_stocks,
        time_steps=config['time_steps'],
        channels=config['channels'],
        **kwargs
    )


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Testing StockMixer implementation...")

    # Test parameters
    num_stocks = 100
    time_steps = 60
    channels = 5

    # Create network
    model = StockMixerNetwork(
        stocks=num_stocks,
        time_steps=time_steps,
        channels=channels,
        market=20,
        scale=3,
    )

    # Test forward pass
    x = torch.randn(num_stocks, time_steps, channels)
    y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test loss
    gt = torch.randn(num_stocks, 1)
    mask = torch.ones(num_stocks, 1)
    loss, reg, rank = get_stockmixer_loss_simple(y, gt, mask, alpha=0.1)
    print(f"Loss: {loss.item():.4f} (reg={reg.item():.4f}, rank={rank.item():.4f})")

    print("\nTest passed!")
