"""
TCN with Macro Conditioning - Cross-Validation Training Script

Architecture:
    Input: Stock features (batch, 5, 60)  +  Macro features (batch, N)
                        ↓                              ↓
                  TemporalConvNet                      │
                        ↓                              │
             output[:, :, -1] (batch, 32)              │
                        ↓                              │
                     Concat ←──────────────────────────┘
                        ↓
             (batch, 32 + N)
                        ↓
                  Linear(32+N, 16) + ReLU
                        ↓
                  Linear(16, 1)
                        ↓
                   prediction

Key Design:
    - Stock features only through TCN: (5, 60) = CLOSE, OPEN, HIGH, LOW, VOLUME × 60 days
    - Macro features via MLP: N current-day values
        - minimal (default): 6 features (MINIMAL_MACRO_FEATURES)
        - core: 23 features (CORE_MACRO_FEATURES) - VIX, macro assets, benchmark, credit, treasury, cross-asset
    - Concat after TCN: Combine temporal embedding with macro conditioning
    - Small MLP: Learn non-linear stock×macro interactions

Usage:
    # Basic CV training with minimal macro features (6)
    python scripts/models/deep/run_tcn_macro_cv.py --stock-pool sp500

    # With CORE macro features (23 features)
    python scripts/models/deep/run_tcn_macro_cv.py --stock-pool sp500 --macro-set core

    # With backtest
    python scripts/models/deep/run_tcn_macro_cv.py --stock-pool sp500 --backtest

    # Ablation tests
    python scripts/models/deep/run_tcn_macro_cv.py --stock-pool sp500 --n-macro 1  # VIX only
    python scripts/models/deep/run_tcn_macro_cv.py --stock-pool sp500 --n-macro 2  # VIX + credit
    python scripts/models/deep/run_tcn_macro_cv.py --stock-pool sp500 --n-macro 6  # all minimal
    python scripts/models/deep/run_tcn_macro_cv.py --stock-pool sp500 --n-macro 23  # all core
"""

import os
import sys
from pathlib import Path

# Set environment variables
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))
project_root = script_dir.parent

import argparse
import copy
from datetime import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from data.stock_pools import STOCK_POOLS

from models.common import (
    PROJECT_ROOT, MODEL_SAVE_PATH,
    HANDLER_CONFIG,
    init_qlib,
    run_backtest,
    CV_FOLDS,
    FINAL_TEST,
)
from models.common.handlers import get_handler_class

# Import TCN backbone from qlib
from qlib.contrib.model.tcn import TemporalConvNet


# ============================================================================
# Macro Feature Configuration
# ============================================================================

# Default macro features path
DEFAULT_MACRO_PATH = PROJECT_ROOT / "my_data" / "macro_processed" / "macro_features.parquet"

# Minimal macro features (6) - based on CatBoost forward selection
# All features should have similar scale (std ≈ 1.0) for MLP
MINIMAL_MACRO_FEATURES = [
    "macro_vix_zscore20",      # VIX normalized level
    "macro_hy_spread_zscore",  # High-yield credit spread
    "macro_credit_stress",     # Credit market stress
    "macro_tlt_pct_20d",       # Bond momentum
    "macro_uso_pct_5d",        # Oil momentum
    "macro_risk_on_off",       # Risk regime indicator
]

# Feature subsets for ablation
SINGLE_MACRO_FEATURES = ["macro_vix_zscore20"]
DUO_MACRO_FEATURES = ["macro_vix_zscore20", "macro_credit_stress"]

# CORE macro features (23) - comprehensive set from datahandler_macro.py
CORE_MACRO_FEATURES = [
    # VIX (5)
    "macro_vix_level", "macro_vix_zscore20", "macro_vix_pct_5d",
    "macro_vix_regime", "macro_vix_term_structure",
    # Macro Assets (5)
    "macro_gld_pct_5d", "macro_tlt_pct_5d", "macro_yield_curve",
    "macro_uup_pct_5d", "macro_uso_pct_5d",
    # Benchmark (2)
    "macro_spy_pct_5d", "macro_spy_vol20",
    # Credit (3)
    "macro_hyg_vs_lqd", "macro_credit_stress", "macro_hy_spread_zscore",
    # Global (2)
    "macro_eem_vs_spy", "macro_global_risk",
    # Treasury (3)
    "macro_yield_10y", "macro_yield_2s10s", "macro_yield_inversion",
    # Cross-asset (3)
    "macro_risk_on_off", "macro_market_stress", "macro_hy_spread",
]

# Features that need z-score normalization (small std)
FEATURES_NEED_ZSCORE = [
    "macro_tlt_pct_20d",
    "macro_tlt_pct_5d",
    "macro_uso_pct_5d",
    "macro_gld_pct_5d",
    "macro_uup_pct_5d",
    "macro_spy_pct_5d",
]


# ============================================================================
# TCN with Macro Conditioning Model
# ============================================================================

class TCNWithMacro(nn.Module):
    """
    TCN model with macro conditioning via MLP.

    Architecture:
        - TCN processes stock features (d_feat, step_len) -> (n_chans,)
        - Macro features (n_macro,) are concatenated with TCN output
        - MLP combines both for final prediction
    """

    def __init__(
        self,
        num_input: int,
        num_channels: list,
        kernel_size: int,
        dropout: float,
        n_macro: int = 6,
        hidden_size: int = 16,
    ):
        """
        Args:
            num_input: Number of input features per timestep (d_feat)
            num_channels: List of channel sizes for TCN layers
            kernel_size: Kernel size for TCN convolutions
            dropout: Dropout rate
            n_macro: Number of macro features
            hidden_size: Hidden layer size for MLP
        """
        super().__init__()

        self.num_input = num_input
        self.n_macro = n_macro

        # TCN for stock features
        self.tcn = TemporalConvNet(num_input, num_channels, kernel_size, dropout=dropout)

        # MLP for combining TCN output with macro features
        tcn_out_size = num_channels[-1]
        combined_size = tcn_out_size + n_macro

        self.fc1 = nn.Linear(combined_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_stock, x_macro):
        """
        Forward pass.

        Args:
            x_stock: Stock features (batch, d_feat, step_len)
            x_macro: Macro features (batch, n_macro)

        Returns:
            predictions: (batch,)
        """
        # TCN: (batch, d_feat, step_len) -> (batch, n_chans, step_len)
        tcn_out = self.tcn(x_stock)
        # Take last timestep: (batch, n_chans)
        tcn_out = tcn_out[:, :, -1]

        # Concat with macro features: (batch, n_chans + n_macro)
        combined = torch.cat([tcn_out, x_macro], dim=1)

        # MLP: (batch, combined_size) -> (batch, 1)
        out = self.dropout(self.relu(self.fc1(combined)))
        out = self.fc2(out)

        return out.squeeze(-1)


# ============================================================================
# Dataset for TCN with Macro
# ============================================================================

class TCNMacroDataset(Dataset):
    """
    Dataset for TCN with macro conditioning.

    Returns (stock_features, macro_features, label) tuples.
    """

    def __init__(
        self,
        stock_features: np.ndarray,
        macro_features: np.ndarray,
        labels: np.ndarray,
        d_feat: int,
        step_len: int,
    ):
        """
        Args:
            stock_features: Flattened stock features (N, d_feat * step_len)
            macro_features: Macro features (N, n_macro)
            labels: Target labels (N,)
            d_feat: Features per timestep
            step_len: Time series length
        """
        # Validate dimensions
        expected_features = d_feat * step_len
        if stock_features.shape[1] != expected_features:
            raise ValueError(
                f"Stock feature dimension mismatch: got {stock_features.shape[1]}, "
                f"expected {expected_features} (d_feat={d_feat} × step_len={step_len})"
            )

        # Reshape stock features: (N, d_feat * step_len) -> (N, d_feat, step_len)
        self.stock = stock_features.reshape(-1, d_feat, step_len)
        self.macro = macro_features
        self.labels = labels

    def __len__(self):
        return len(self.stock)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.stock[idx], dtype=torch.float32),
            torch.tensor(self.macro[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# ============================================================================
# Macro Feature Loading
# ============================================================================

def load_macro_df(macro_path: Path = None) -> pd.DataFrame:
    """
    Load macro features from parquet file.

    Returns:
        DataFrame with datetime index and macro feature columns
    """
    if macro_path is None:
        macro_path = DEFAULT_MACRO_PATH

    if not macro_path.exists():
        raise FileNotFoundError(f"Macro features file not found: {macro_path}")

    df = pd.read_parquet(macro_path)
    print(f"Loaded macro features: {df.shape}, date range: {df.index.min()} to {df.index.max()}")
    return df


def prepare_macro_features(
    index: pd.MultiIndex,
    macro_df: pd.DataFrame,
    macro_cols: list,
    macro_lag: int = 1,
) -> np.ndarray:
    """
    Prepare and align macro features to sample index.

    Args:
        index: MultiIndex from dataset (datetime, instrument)
        macro_df: DataFrame with macro features
        macro_cols: List of macro column names to use
        macro_lag: Days to lag macro features to avoid look-ahead bias

    Returns:
        Aligned macro features array (N, n_macro)
    """
    # Get dates from index
    dates = index.get_level_values('datetime')

    # Filter available columns
    available_cols = [c for c in macro_cols if c in macro_df.columns]
    if len(available_cols) < len(macro_cols):
        missing = set(macro_cols) - set(available_cols)
        print(f"Warning: Missing macro columns: {missing}")

    if not available_cols:
        raise ValueError("No macro features available!")

    # Extract and lag macro features
    macro_subset = macro_df[available_cols].copy()

    # Apply z-score normalization for features with small std
    for col in available_cols:
        if col in FEATURES_NEED_ZSCORE:
            rolling_mean = macro_subset[col].rolling(window=60, min_periods=20).mean()
            rolling_std = macro_subset[col].rolling(window=60, min_periods=20).std()
            macro_subset[col] = (macro_subset[col] - rolling_mean) / (rolling_std + 1e-8)
            macro_subset[col] = macro_subset[col].clip(-5, 5)

    # Lag to avoid look-ahead bias
    if macro_lag > 0:
        macro_subset = macro_subset.shift(macro_lag)

    # Align to sample dates
    macro_aligned = macro_subset.reindex(dates)

    # Fill NaN with 0 (conservative approach)
    macro_aligned = macro_aligned.fillna(0)

    return macro_aligned.values


def get_macro_feature_list(n_macro: int = None, macro_set: str = "minimal") -> list:
    """
    Get macro feature list based on macro_set or n_macro count.

    Args:
        n_macro: Number of macro features to use (1, 2, 6, or up to 23).
                 If specified, overrides macro_set for counts > 6.
        macro_set: Macro feature set: 'minimal' (6), 'core' (23)

    Returns:
        List of macro feature column names
    """
    # If n_macro is explicitly specified, use it to determine features
    if n_macro is not None:
        if n_macro == 1:
            return SINGLE_MACRO_FEATURES
        elif n_macro == 2:
            return DUO_MACRO_FEATURES
        elif n_macro <= 6:
            return MINIMAL_MACRO_FEATURES[:n_macro]
        else:
            # For n_macro > 6, use CORE features
            return CORE_MACRO_FEATURES[:n_macro]

    # Otherwise, use macro_set
    if macro_set == "core":
        return CORE_MACRO_FEATURES  # 23 features
    else:
        return MINIMAL_MACRO_FEATURES  # 6 features (default)


# ============================================================================
# TCN Macro Trainer
# ============================================================================

class TCNMacroTrainer:
    """Trainer for TCN with Macro conditioning."""

    def __init__(
        self,
        d_feat: int = 5,
        n_macro: int = 6,
        n_chans: int = 32,
        kernel_size: int = 7,
        num_layers: int = 5,
        dropout: float = 0.5,
        hidden_size: int = 16,
        n_epochs: int = 200,
        lr: float = 1e-4,
        batch_size: int = 2000,
        early_stop: int = 20,
        gpu: int = 0,
        seed: int = None,
    ):
        self.d_feat = d_feat
        self.n_macro = n_macro
        self.n_chans = n_chans
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.seed = seed

        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        self.model = None
        self.fitted = False

    def _init_model(self):
        """Initialize the model."""
        self.model = TCNWithMacro(
            num_input=self.d_feat,
            num_channels=[self.n_chans] * self.num_layers,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            n_macro=self.n_macro,
            hidden_size=self.hidden_size,
        )
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _mse_loss(self, pred, label):
        """MSE loss with NaN handling."""
        mask = ~torch.isnan(label)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        return ((pred[mask] - label[mask]) ** 2).mean()

    def _train_epoch(self, data_loader):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0

        for stock_feat, macro_feat, label in data_loader:
            stock_feat = stock_feat.to(self.device)
            macro_feat = macro_feat.to(self.device)
            label = label.to(self.device)

            pred = self.model(stock_feat, macro_feat)
            loss = self._mse_loss(pred, label)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _eval_epoch(self, data_loader):
        """Evaluate one epoch."""
        self.model.eval()
        preds = []
        labels = []

        with torch.no_grad():
            for stock_feat, macro_feat, label in data_loader:
                stock_feat = stock_feat.to(self.device)
                macro_feat = macro_feat.to(self.device)

                pred = self.model(stock_feat, macro_feat)
                preds.append(pred.cpu().numpy())
                labels.append(label.numpy())

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

        # Compute loss
        mask = ~np.isnan(labels)
        if mask.sum() == 0:
            return float('inf'), preds

        loss = np.mean((preds[mask] - labels[mask]) ** 2)
        return loss, preds

    def fit(self, train_loader, valid_loader, verbose=True):
        """Train the model."""
        self._init_model()

        best_loss = float('inf')
        best_epoch = 0
        best_params = None
        stop_steps = 0

        for epoch in range(self.n_epochs):
            train_loss = self._train_epoch(train_loader)
            valid_loss, _ = self._eval_epoch(valid_loader)

            if verbose:
                print(f"    Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}")

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_params = copy.deepcopy(self.model.state_dict())
                stop_steps = 0
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    if verbose:
                        print(f"    Early stop at epoch {epoch+1}")
                    break

        if best_params is not None:
            self.model.load_state_dict(best_params)

        self.fitted = True
        return best_epoch + 1, best_loss

    def predict(self, data_loader):
        """Generate predictions."""
        if not self.fitted:
            raise ValueError("Model not fitted yet")

        self.model.eval()
        preds = []

        with torch.no_grad():
            for stock_feat, macro_feat, _ in data_loader:
                stock_feat = stock_feat.to(self.device)
                macro_feat = macro_feat.to(self.device)

                pred = self.model(stock_feat, macro_feat)
                preds.append(pred.cpu().numpy())

        return np.concatenate(preds)

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'd_feat': self.d_feat,
                'n_macro': self.n_macro,
                'n_chans': self.n_chans,
                'kernel_size': self.kernel_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'hidden_size': self.hidden_size,
            }
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        config = checkpoint['config']

        self.d_feat = config['d_feat']
        self.n_macro = config['n_macro']
        self.n_chans = config['n_chans']
        self.kernel_size = config['kernel_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.hidden_size = config.get('hidden_size', 16)

        self._init_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.fitted = True


# ============================================================================
# Data Preparation
# ============================================================================

def create_handler_for_fold(args, fold_config):
    """Create Handler for a specific fold."""
    end_time = fold_config.get('test_end', fold_config['valid_end'])

    # Use Alpha300 handler (no VWAP, for US stocks)
    HandlerClass = get_handler_class('alpha300')
    handler = HandlerClass(
        volatility_window=args.nday,
        instruments=STOCK_POOLS[args.stock_pool],
        start_time=fold_config['train_start'],
        end_time=end_time,
        fit_start_time=fold_config['train_start'],
        fit_end_time=fold_config['train_end'],
    )

    return handler


def create_dataset_for_fold(handler, fold_config):
    """Create Dataset for a specific fold."""
    segments = {
        "train": (fold_config['train_start'], fold_config['train_end']),
        "valid": (fold_config['valid_start'], fold_config['valid_end']),
    }

    if 'test_start' in fold_config:
        segments["test"] = (fold_config['test_start'], fold_config['test_end'])

    return DatasetH(handler=handler, segments=segments)


def prepare_data_for_tcn_macro(
    dataset,
    segment: str,
    macro_df: pd.DataFrame,
    macro_cols: list,
    d_feat: int = 5,
    step_len: int = 60,
    macro_lag: int = 1,
):
    """
    Prepare data for TCN with macro conditioning.

    Returns:
        TCNMacroDataset, index, labels
    """
    # Get stock features and labels
    features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
    labels = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)

    # Process features
    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]
    labels = labels.fillna(0)

    index = features.index

    # Prepare macro features aligned to sample index
    macro_features = prepare_macro_features(
        index, macro_df, macro_cols, macro_lag=macro_lag
    )

    # Create dataset
    tcn_dataset = TCNMacroDataset(
        stock_features=features.values,
        macro_features=macro_features,
        labels=labels.values,
        d_feat=d_feat,
        step_len=step_len,
    )

    return tcn_dataset, index, labels.values


def compute_ic_from_arrays(pred, labels, index):
    """Compute IC from prediction arrays."""
    df = pd.DataFrame({'pred': pred, 'label': labels}, index=index)

    # IC by date
    ic_by_date = df.groupby(level='datetime').apply(
        lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
    )
    ic_by_date = ic_by_date.dropna()

    if len(ic_by_date) == 0:
        return 0.0, 0.0, 0.0

    mean_ic = ic_by_date.mean()
    ic_std = ic_by_date.std()
    icir = mean_ic / ic_std if ic_std > 0 else 0

    return mean_ic, ic_std, icir


def evaluate_model(test_pred, test_labels, test_index):
    """Evaluate model performance."""
    df = pd.DataFrame({'pred': test_pred, 'label': test_labels}, index=test_index)

    # Remove NaN
    valid_idx = ~(np.isnan(df['pred']) | np.isnan(df['label']))
    df_clean = df[valid_idx]

    print(f"    Valid test samples: {len(df_clean)}")

    # IC by date
    ic_by_date = df_clean.groupby(level='datetime').apply(
        lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
    )
    ic_by_date = ic_by_date.dropna()

    # Error metrics
    mse = ((df_clean['pred'] - df_clean['label']) ** 2).mean()
    mae = (df_clean['pred'] - df_clean['label']).abs().mean()
    rmse = np.sqrt(mse)

    print(f"\n    ╔════════════════════════════════════════╗")
    print(f"    ║  Information Coefficient (IC)          ║")
    print(f"    ╠════════════════════════════════════════╣")
    print(f"    ║  Mean IC:   {ic_by_date.mean():>8.4f}                  ║")
    print(f"    ║  IC Std:    {ic_by_date.std():>8.4f}                  ║")
    print(f"    ║  ICIR:      {ic_by_date.mean() / ic_by_date.std() if ic_by_date.std() > 0 else 0:>8.4f}                  ║")
    print(f"    ╚════════════════════════════════════════╝")

    print(f"\n    ╔════════════════════════════════════════╗")
    print(f"    ║  Prediction Error Metrics              ║")
    print(f"    ╠════════════════════════════════════════╣")
    print(f"    ║  MSE:   {mse:>8.6f}                       ║")
    print(f"    ║  MAE:   {mae:>8.6f}                       ║")
    print(f"    ║  RMSE:  {rmse:>8.6f}                       ║")
    print(f"    ╚════════════════════════════════════════╝")

    return ic_by_date.mean(), ic_by_date.std()


# ============================================================================
# CV Training
# ============================================================================

def run_cv_training(args, macro_df, macro_cols):
    """Run cross-validation training."""
    print("\n" + "=" * 70)
    print("TCN WITH MACRO CONDITIONING - CROSS-VALIDATION TRAINING")
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool} ({len(STOCK_POOLS[args.stock_pool])} stocks)")
    print(f"N-day: {args.nday}")
    print(f"Stock d_feat: {args.d_feat}, step_len: {args.step_len}")
    print(f"Macro features: {args.n_macro} ({macro_cols})")
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: train {fold['train_start']}~{fold['train_end']}, "
              f"valid {fold['valid_start']}~{fold['valid_end']}")
    print(f"Test: {FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']}")
    print("=" * 70)

    # Prepare 2025 test set
    print("\n[*] Preparing 2025 test data...")
    test_handler = create_handler_for_fold(args, FINAL_TEST)
    test_dataset = create_dataset_for_fold(test_handler, FINAL_TEST)

    test_data, test_index, test_labels = prepare_data_for_tcn_macro(
        test_dataset, "test", macro_df, macro_cols,
        d_feat=args.d_feat, step_len=args.step_len, macro_lag=args.macro_lag
    )
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"    Test samples: {len(test_data)}")

    fold_results = []
    fold_ics = []
    fold_test_ics = []

    for fold_idx, fold in enumerate(CV_FOLDS):
        print(f"\n[*] Training {fold['name']}...")

        # Set seed
        if args.seed is not None:
            seed = args.seed + fold_idx
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Prepare data
        handler = create_handler_for_fold(args, fold)
        dataset = create_dataset_for_fold(handler, fold)

        train_data, train_index, train_labels = prepare_data_for_tcn_macro(
            dataset, "train", macro_df, macro_cols,
            d_feat=args.d_feat, step_len=args.step_len, macro_lag=args.macro_lag
        )
        valid_data, valid_index, valid_labels = prepare_data_for_tcn_macro(
            dataset, "valid", macro_df, macro_cols,
            d_feat=args.d_feat, step_len=args.step_len, macro_lag=args.macro_lag
        )

        print(f"    Train: {len(train_data)}, Valid: {len(valid_data)}")

        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True
        )
        valid_loader = DataLoader(
            valid_data, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        # Create and train model
        trainer = TCNMacroTrainer(
            d_feat=args.d_feat,
            n_macro=args.n_macro,
            n_chans=args.n_chans,
            kernel_size=args.kernel_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            hidden_size=args.hidden_size,
            n_epochs=args.n_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            early_stop=args.early_stop,
            gpu=args.gpu,
            seed=args.seed + fold_idx if args.seed else None,
        )

        best_epoch, best_loss = trainer.fit(
            train_loader, valid_loader, verbose=args.verbose
        )

        # Validation prediction
        valid_pred = trainer.predict(valid_loader)
        valid_ic, valid_ic_std, valid_icir = compute_ic_from_arrays(valid_pred, valid_labels, valid_index)

        # Test prediction
        test_pred = trainer.predict(test_loader)
        test_ic, test_ic_std, test_icir = compute_ic_from_arrays(test_pred, test_labels, test_index)

        fold_ics.append(valid_ic)
        fold_test_ics.append(test_ic)
        fold_results.append({
            'name': fold['name'],
            'valid_ic': valid_ic,
            'valid_icir': valid_icir,
            'test_ic': test_ic,
            'test_icir': test_icir,
            'best_epoch': best_epoch,
        })

        print(f"    {fold['name']}: Valid IC={valid_ic:.4f}, Test IC (2025)={test_ic:.4f}, epoch={best_epoch}")

    # Summary
    mean_valid_ic = np.mean(fold_ics)
    std_valid_ic = np.std(fold_ics)
    mean_test_ic = np.mean(fold_test_ics)
    std_test_ic = np.std(fold_test_ics)

    print("\n" + "=" * 70)
    print("CV TRAINING COMPLETE")
    print("=" * 70)
    print(f"Valid Mean IC: {mean_valid_ic:.4f} (±{std_valid_ic:.4f})")
    print(f"Test Mean IC (2025): {mean_test_ic:.4f} (±{std_test_ic:.4f})")
    print("\nIC by fold:")
    print(f"  {'Fold':<25s} {'Valid IC':>10s} {'Test IC':>10s} {'Epoch':>8s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")
    for r in fold_results:
        print(f"  {r['name']:<25s} {r['valid_ic']:>10.4f} {r['test_ic']:>10.4f} {r['best_epoch']:>8d}")
    print("=" * 70)

    return fold_results, mean_valid_ic, std_valid_ic, test_dataset


def train_final_model(args, macro_df, macro_cols, test_dataset):
    """Train final model on full data."""
    print("\n[*] Training final model on full data...")

    # Prepare data
    handler = create_handler_for_fold(args, FINAL_TEST)
    dataset = create_dataset_for_fold(handler, FINAL_TEST)

    train_data, train_index, train_labels = prepare_data_for_tcn_macro(
        dataset, "train", macro_df, macro_cols,
        d_feat=args.d_feat, step_len=args.step_len, macro_lag=args.macro_lag
    )
    valid_data, valid_index, valid_labels = prepare_data_for_tcn_macro(
        dataset, "valid", macro_df, macro_cols,
        d_feat=args.d_feat, step_len=args.step_len, macro_lag=args.macro_lag
    )
    test_data, test_index, test_labels = prepare_data_for_tcn_macro(
        dataset, "test", macro_df, macro_cols,
        d_feat=args.d_feat, step_len=args.step_len, macro_lag=args.macro_lag
    )

    print(f"    Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Train
    trainer = TCNMacroTrainer(
        d_feat=args.d_feat,
        n_macro=args.n_macro,
        n_chans=args.n_chans,
        kernel_size=args.kernel_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        hidden_size=args.hidden_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        early_stop=args.early_stop,
        gpu=args.gpu,
        seed=args.seed,
    )

    best_epoch, best_loss = trainer.fit(train_loader, valid_loader, verbose=True)
    print(f"    Best epoch: {best_epoch}, Best loss: {best_loss:.6f}")

    # Predict
    test_pred = trainer.predict(test_loader)
    test_pred_series = pd.Series(test_pred, index=test_index, name='score')

    # Save model
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_SAVE_PATH / f"tcn_macro_cv_{args.stock_pool}_{args.nday}d_m{args.n_macro}_{timestamp}.pt"
    trainer.save(model_path)
    print(f"    Model saved to: {model_path}")

    return trainer, test_pred_series, dataset, model_path, test_labels


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TCN with Macro Conditioning - Cross-Validation Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Basic parameters
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--nday', type=int, default=5,
                        help='Label prediction horizon (default: 5)')

    # Stock feature parameters (Alpha300)
    parser.add_argument('--d-feat', type=int, default=5,
                        help='Stock features per timestep (default: 5 for Alpha300)')
    parser.add_argument('--step-len', type=int, default=60,
                        help='Time series length (default: 60)')

    # Macro feature parameters
    parser.add_argument('--macro-set', type=str, default='minimal',
                        choices=['minimal', 'core'],
                        help='Macro feature set: minimal (6) or core (23) (default: minimal)')
    parser.add_argument('--n-macro', type=int, default=None,
                        help='Number of macro features (overrides --macro-set if specified). '
                             '1-6 uses minimal features, 7-23 uses core features.')
    parser.add_argument('--macro-lag', type=int, default=1,
                        help='Days to lag macro features (default: 1)')
    parser.add_argument('--macro-path', type=str, default=None,
                        help='Path to macro features parquet file')

    # TCN parameters
    parser.add_argument('--n-chans', type=int, default=32,
                        help='Number of TCN channels (default: 32)')
    parser.add_argument('--kernel-size', type=int, default=7,
                        help='TCN kernel size (default: 7)')
    parser.add_argument('--num-layers', type=int, default=5,
                        help='Number of TCN layers (default: 5)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
    parser.add_argument('--hidden-size', type=int, default=16,
                        help='MLP hidden layer size (default: 16)')

    # Training parameters
    parser.add_argument('--n-epochs', type=int, default=200,
                        help='Max epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=2000,
                        help='Batch size (default: 2000)')
    parser.add_argument('--early-stop', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')

    # Mode selection
    parser.add_argument('--cv-only', action='store_true',
                        help='Only run CV training, skip final model')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate pre-trained model')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to pre-trained model for evaluation')
    parser.add_argument('--verbose', action='store_true',
                        help='Show training progress for each epoch')

    # Backtest parameters
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=1)
    parser.add_argument('--account', type=float, default=10000)
    parser.add_argument('--rebalance-freq', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss'])

    args = parser.parse_args()

    # Validate parameters
    if args.eval_only and not args.model_path:
        parser.error("--eval-only requires --model-path")

    # Initialize Qlib
    init_qlib(use_talib=False)

    # Load macro features
    macro_path = Path(args.macro_path) if args.macro_path else DEFAULT_MACRO_PATH
    macro_df = load_macro_df(macro_path)

    # Get macro feature list
    macro_cols = get_macro_feature_list(n_macro=args.n_macro, macro_set=args.macro_set)

    # Verify macro features are available
    available_cols = [c for c in macro_cols if c in macro_df.columns]
    if len(available_cols) < len(macro_cols):
        missing = set(macro_cols) - set(available_cols)
        print(f"Warning: Missing macro features: {missing}")
        macro_cols = available_cols

    # Set n_macro to actual count for model configuration
    args.n_macro = len(macro_cols)

    print("\n" + "=" * 70)
    print("TCN with Macro Conditioning - US Data")
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool}")
    print(f"N-day: {args.nday}")
    print(f"Stock d_feat: {args.d_feat}, step_len: {args.step_len}")
    print(f"Macro features ({args.n_macro}): {macro_cols}")
    print(f"TCN: {args.n_chans} channels × {args.num_layers} layers, kernel={args.kernel_size}")
    print(f"MLP hidden: {args.hidden_size}")
    print(f"GPU: {args.gpu}")
    print("=" * 70)

    # ========== Evaluation mode ==========
    if args.eval_only:
        print("\n[*] Evaluation mode - Loading pre-trained model...")
        print(f"    Model path: {args.model_path}")

        trainer = TCNMacroTrainer(gpu=args.gpu)
        trainer.load(args.model_path)
        print(f"    Model loaded: d_feat={trainer.d_feat}, n_macro={trainer.n_macro}")

        # Prepare test data
        print("\n[*] Preparing test data...")
        test_handler = create_handler_for_fold(args, FINAL_TEST)
        test_dataset = create_dataset_for_fold(test_handler, FINAL_TEST)

        test_data, test_index, test_labels = prepare_data_for_tcn_macro(
            test_dataset, "test", macro_df, macro_cols,
            d_feat=trainer.d_feat, step_len=args.step_len, macro_lag=args.macro_lag
        )
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(f"    Test samples: {len(test_data)}")

        # Predict
        print("\n[*] Running predictions...")
        test_pred = trainer.predict(test_loader)
        test_pred_series = pd.Series(test_pred, index=test_index, name='score')

        # Evaluate
        print("\n[*] Evaluation Results on Test Set:")
        mean_ic, ic_std = evaluate_model(test_pred, test_labels, test_index)

        icir = mean_ic / ic_std if ic_std > 0 else 0
        print("\n" + "=" * 70)
        print("TEST SET EVALUATION COMPLETE")
        print("=" * 70)
        print(f"Test IC:   {mean_ic:.4f}")
        print(f"IC Std:    {ic_std:.4f}")
        print(f"ICIR:      {icir:.4f}")
        print("=" * 70)

        # Backtest
        if args.backtest:
            pred_df = test_pred_series.to_frame("score")
            time_splits = {
                'train_start': FINAL_TEST['train_start'],
                'train_end': FINAL_TEST['train_end'],
                'valid_start': FINAL_TEST['valid_start'],
                'valid_end': FINAL_TEST['valid_end'],
                'test_start': FINAL_TEST['test_start'],
                'test_end': FINAL_TEST['test_end'],
            }

            def load_model(path):
                t = TCNMacroTrainer(gpu=args.gpu)
                t.load(path)
                return t

            def get_feature_count(m):
                return args.d_feat * args.step_len

            run_backtest(
                args.model_path, test_dataset, pred_df, args, time_splits,
                model_name="TCN+Macro (Eval)",
                load_model_func=load_model,
                get_feature_count_func=get_feature_count
            )

        return

    # ========== CV Training ==========
    fold_results, mean_ic, std_ic, test_dataset = run_cv_training(args, macro_df, macro_cols)

    if args.cv_only:
        print("\n[*] CV-only mode, skipping final model training.")
        return

    # ========== Train final model ==========
    trainer, test_pred, dataset, model_path, test_labels = train_final_model(
        args, macro_df, macro_cols, test_dataset
    )

    # ========== Evaluate ==========
    print("\n[*] Final Evaluation on Test Set (2025)...")
    evaluate_model(test_pred.values, test_labels, test_pred.index)

    # ========== Backtest ==========
    if args.backtest:
        pred_df = test_pred.to_frame("score")
        time_splits = {
            'train_start': FINAL_TEST['train_start'],
            'train_end': FINAL_TEST['train_end'],
            'valid_start': FINAL_TEST['valid_start'],
            'valid_end': FINAL_TEST['valid_end'],
            'test_start': FINAL_TEST['test_start'],
            'test_end': FINAL_TEST['test_end'],
        }

        def load_model(path):
            t = TCNMacroTrainer(gpu=args.gpu)
            t.load(path)
            return t

        def get_feature_count(m):
            return args.d_feat * args.step_len

        run_backtest(
            model_path, dataset, pred_df, args, time_splits,
            model_name="TCN+Macro (CV)",
            load_model_func=load_model,
            get_feature_count_func=get_feature_count
        )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"CV Valid Mean IC: {mean_ic:.4f} (±{std_ic:.4f})")
    print(f"Model saved to: {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
