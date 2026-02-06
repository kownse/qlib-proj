"""
KAN (Kolmogorov-Arnold Network) Stock Return Prediction Model

Based on efficient-kan implementation. Uses learnable B-spline activation
functions instead of fixed activations, providing superior function
approximation for stock return prediction.

Architecture:
- Input: (batch, num_features) flat features from any handler
- KANLinear layers with B-spline basis functions + base activation
- Output: (batch,) predicted returns

Key advantages for stock prediction:
1. Learnable non-linear activations capture complex feature-return relationships
2. Adaptive grid updates ensure B-splines cover actual data distribution
3. L1 + entropy regularization promotes sparse, interpretable representations

Suitable handlers: alpha158, alpha158-talib, alpha158-macro, alpha360 (flat), etc.
"""

import os
import sys
import copy
import time
import numpy as np
import pandas as pd
from typing import Optional, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

# Import KANLinear from efficient-kan
_kan_src = str(Path(__file__).resolve().parent.parent.parent.parent / 'efficient-kan' / 'src')
if _kan_src not in sys.path:
    sys.path.insert(0, _kan_src)
from efficient_kan import KANLinear


# ============================================================================
# Network
# ============================================================================

class KANStockNetwork(nn.Module):
    """
    KAN-based stock prediction network.

    Uses KANLinear layers with B-spline basis functions for learnable
    non-linear transformations. Each KANLinear replaces a standard Linear
    layer with two branches:
      1. base_activation(x) @ base_weight  (standard non-linearity)
      2. b_splines(x) @ spline_weight      (learnable B-spline transform)

    Args:
        d_feat: Number of input features
        hidden_sizes: Hidden layer dimensions (default: [256, 128])
        grid_size: B-spline grid segments (default: 8)
        spline_order: B-spline order, 3 = cubic (default: 3)
        dropout: Dropout rate (default: 0.1)
        scale_noise: Spline weight initialization noise (default: 0.1)
        grid_range: Initial grid range (default: [-3, 3] for z-scored data)
        base_activation: Base activation class (default: SiLU)
    """

    def __init__(
        self,
        d_feat: int,
        hidden_sizes: List[int] = [256, 128],
        grid_size: int = 8,
        spline_order: int = 3,
        dropout: float = 0.1,
        scale_noise: float = 0.1,
        grid_range: list = [-3, 3],
        base_activation: type = nn.SiLU,
    ):
        super().__init__()
        self.d_feat = d_feat

        # Input normalization
        self.input_norm = nn.BatchNorm1d(d_feat)

        # Build KAN layers
        sizes = [d_feat] + hidden_sizes + [1]
        self.kan_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drops = nn.ModuleList()

        for i in range(len(sizes) - 1):
            self.kan_layers.append(
                KANLinear(
                    sizes[i], sizes[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    base_activation=base_activation,
                    grid_range=grid_range,
                )
            )
            # No norm/dropout after the last layer
            if i < len(sizes) - 2:
                self.norms.append(nn.BatchNorm1d(sizes[i + 1]))
                self.drops.append(nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch, d_feat)
            update_grid: Whether to adapt B-spline grids to data distribution

        Returns:
            (batch,) predictions
        """
        x = self.input_norm(x)
        for i, layer in enumerate(self.kan_layers):
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
            if i < len(self.norms):
                x = self.norms[i](x)
                x = self.drops[i](x)
        return x.squeeze(-1)

    def regularization_loss(self, reg_activation=1.0, reg_entropy=1.0):
        """
        Safe KAN regularization (L1 + entropy) across all layers.

        Adds epsilon to avoid log(0) NaN in entropy computation, which
        occurs when spline weights are exactly 0 (e.g., constant features
        after grid update).
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.kan_layers:
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

class KANStock:
    """
    KAN stock prediction model with Qlib interface.

    Trains a KANStockNetwork on tabular features to predict N-day forward
    returns. Uses MSE (or IC) loss with KAN B-spline regularization.
    Validates using cross-sectional IC (rank correlation) for early stopping.

    Parameters
    ----------
    d_feat : int
        Number of input features (auto-detected if 0)
    hidden_sizes : list
        Hidden layer dimensions
    grid_size : int
        B-spline grid segments
    spline_order : int
        B-spline order (3 = cubic)
    dropout : float
        Dropout rate
    learning_rate : float
        AdamW learning rate
    weight_decay : float
        AdamW weight decay
    reg_lambda : float
        KAN regularization weight (L1 + entropy)
    batch_size : int
        Training batch size
    n_epochs : int
        Maximum training epochs
    early_stop : int
        Early stopping patience (on validation IC)
    grid_update_freq : int
        Update B-spline grids every N epochs
    grid_range : list
        Initial grid range ([-3, 3] for z-scored features)
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
        hidden_sizes: list = [256, 128],
        grid_size: int = 8,
        spline_order: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        reg_lambda: float = 1e-5,
        batch_size: int = 2048,
        n_epochs: int = 100,
        early_stop: int = 15,
        grid_update_freq: int = 20,
        grid_range: list = [-3, 3],
        loss_type: str = 'mse',
        GPU: int = 0,
        seed: int = 42,
    ):
        self.d_feat = d_feat
        self.hidden_sizes = list(hidden_sizes)
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.reg_lambda = reg_lambda
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stop_patience = early_stop
        self.grid_update_freq = grid_update_freq
        self.grid_range = list(grid_range)
        self.loss_type = loss_type
        self.GPU = GPU
        self.seed = seed

        self.model: Optional[KANStockNetwork] = None
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

    def _build_model(self) -> KANStockNetwork:
        model = KANStockNetwork(
            d_feat=self.d_feat,
            hidden_sizes=self.hidden_sizes,
            grid_size=self.grid_size,
            spline_order=self.spline_order,
            dropout=self.dropout,
            grid_range=self.grid_range,
        )
        return model.to(self.device)

    def _prepare_data(self, dataset: DatasetH, segment: str):
        """
        Prepare data from Qlib dataset.

        Returns:
            X: (N, d_feat) tensor
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
        """Update B-spline grids using a representative data sample."""
        # Save state in case grid update produces NaN weights
        saved_state = copy.deepcopy(self.model.state_dict())

        self.model.eval()
        sample_size = min(8192, len(X_train))
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        sample = X_train[indices].to(self.device)
        with torch.no_grad():
            self.model(sample, update_grid=True)

        # Check for NaN in weights after grid update
        has_nan = any(torch.isnan(p).any().item() for p in self.model.parameters())
        if has_nan:
            print("    WARNING: NaN weights after grid update, reverting")
            self.model.load_state_dict(saved_state)

        self.model.train()

    def fit(self, dataset: DatasetH):
        """Train KAN model."""
        print("\n    Preparing data...")
        t0 = time.time()
        X_train, y_train, idx_train = self._prepare_data(dataset, "train")
        X_valid, y_valid, idx_valid = self._prepare_data(dataset, "valid")
        print(f"    Data prepared in {time.time() - t0:.1f}s")

        # Auto-detect d_feat
        actual_d_feat = X_train.shape[1]
        if actual_d_feat != self.d_feat:
            print(f"    Updating d_feat: {self.d_feat} -> {actual_d_feat}")
            self.d_feat = actual_d_feat

        print(f"    Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        print(f"    Valid: {X_valid.shape[0]:,} samples")

        # Data diagnostics
        print(f"    Feature range: [{X_train.min():.2f}, {X_train.max():.2f}]")
        print(f"    Label  range: [{y_train.min():.4f}, {y_train.max():.4f}], "
              f"mean={y_train.mean():.4f}, std={y_train.std():.4f}")

        nan_count = torch.isnan(X_train).sum().item() + torch.isnan(y_train).sum().item()
        if nan_count > 0:
            print(f"    WARNING: {nan_count} NaN values detected after preparation")

        # Build model
        self.model = self._build_model()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"    Parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"    Model device: {next(self.model.parameters()).device}")

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
            # --- Grid update (skip epoch 0: initial grid [-3,3] covers z-scored data) ---
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
                print(f"    Early stopping at epoch {epoch + 1} (best IC: {early_stopping.best_score:.4f})")
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
            'hidden_sizes': self.hidden_sizes,
            'grid_size': self.grid_size,
            'spline_order': self.spline_order,
            'dropout': self.dropout,
            'grid_range': self.grid_range,
        }
        torch.save(save_dict, path)
        print(f"    Model saved to: {path}")

    @classmethod
    def load(cls, path: str, GPU: int = 0) -> 'KANStock':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        instance = cls(
            d_feat=checkpoint['d_feat'],
            hidden_sizes=checkpoint['hidden_sizes'],
            grid_size=checkpoint['grid_size'],
            spline_order=checkpoint['spline_order'],
            dropout=checkpoint['dropout'],
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
    print("Testing KAN stock prediction model...")

    d_feat = 158
    batch_size = 64

    # Test network
    model = KANStockNetwork(
        d_feat=d_feat,
        hidden_sizes=[256, 128],
        grid_size=8,
        spline_order=3,
        dropout=0.1,
        grid_range=[-3, 3],
    )

    x = torch.randn(batch_size, d_feat)
    y = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:   {total_params:,}")

    # Test grid update
    y_updated = model(x, update_grid=True)
    print(f"After grid update: {y_updated.shape}")

    # Test regularization
    reg = model.regularization_loss()
    print(f"Regularization loss: {reg.item():.4f}")

    # Test loss
    target = torch.randn(batch_size)
    mse = F.mse_loss(y, target)
    print(f"MSE loss: {mse.item():.4f}")

    print("\nTest passed!")
