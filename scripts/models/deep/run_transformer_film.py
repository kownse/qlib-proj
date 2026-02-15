"""
Transformer-FiLM with Macro Conditioning - Training Script

Usage:
    python scripts/models/deep/run_transformer_film.py --stock-pool sp500 --macro-set core --skip-cv
    python scripts/models/deep/run_transformer_film.py --stock-pool sp500 --macro-set minimal --skip-cv
    python scripts/models/deep/run_transformer_film.py --stock-pool sp500 --backtest
"""

import os
import sys
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import argparse
import copy
from datetime import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from data.stock_pools import STOCK_POOLS
from models.common import (
    PROJECT_ROOT, MODEL_SAVE_PATH,
    init_qlib, run_backtest, CV_FOLDS, FINAL_TEST,
    compute_ic,
)
from models.common.handlers import get_handler_class
from models.common.macro_features import load_macro_df, get_macro_cols, prepare_macro
from models.deep.transformer_film import TransformerFiLM


# ============================================================================
# Dataset
# ============================================================================

class TransformerMacroDataset(Dataset):
    """Dataset for Transformer-FiLM with macro features."""

    def __init__(self, stock_features, macro_features, labels, d_feat, seq_len):
        # Reshape: (N, seq_len * d_feat) -> (N, seq_len, d_feat)
        self.stock = stock_features.reshape(-1, seq_len, d_feat)
        self.macro = macro_features
        self.labels = labels

    def __len__(self):
        return len(self.stock)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.stock[idx], dtype=torch.float32),  # (seq_len, d_feat)
            torch.tensor(self.macro[idx], dtype=torch.float32),  # (n_macro,)
            torch.tensor(self.labels[idx], dtype=torch.float32),  # scalar
        )


# ============================================================================
# Trainer
# ============================================================================

class TransformerFiLMTrainer:
    """Trainer for Transformer-FiLM model."""

    def __init__(
        self,
        d_feat: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        seq_len: int = 60,
        n_macro: int = 23,
        film_hidden: int = 32,
        n_epochs: int = 200,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        batch_size: int = 2048,
        early_stop: int = 20,
        gpu: int = 0,
        seed: int = None,
    ):
        self.config = {
            'd_feat': d_feat,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'seq_len': seq_len,
            'n_macro': n_macro,
            'film_hidden': film_hidden,
        }
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.early_stop = early_stop

        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        self.model = None
        self.fitted = False

    def _init_model(self):
        self.model = TransformerFiLM(**self.config).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

    def fit(self, train_loader, valid_loader, verbose=True):
        """Train the model."""
        self._init_model()
        best_loss, best_epoch, best_params = float('inf'), 0, None
        stop_steps = 0

        for epoch in range(self.n_epochs):
            # Train
            self.model.train()
            train_loss = 0
            train_preds, train_labels = [], []

            for stock, macro, label in train_loader:
                stock = stock.to(self.device)
                macro = macro.to(self.device)
                label = label.to(self.device)

                pred = self.model(stock, macro)
                mask = ~torch.isnan(label)

                if mask.sum() > 0:
                    loss = ((pred[mask] - label[mask]) ** 2).mean()
                else:
                    loss = torch.tensor(0.0, device=self.device)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item()
                train_preds.extend(pred.detach().cpu().numpy())
                train_labels.extend(label.detach().cpu().numpy())

            # Validate
            self.model.eval()
            valid_loss = 0
            valid_preds, valid_labels = [], []

            with torch.no_grad():
                for stock, macro, label in valid_loader:
                    stock = stock.to(self.device)
                    macro = macro.to(self.device)
                    label = label.to(self.device)

                    pred = self.model(stock, macro)
                    mask = ~torch.isnan(label)

                    if mask.sum() > 0:
                        valid_loss += ((pred[mask] - label[mask]) ** 2).mean().item()

                    valid_preds.extend(pred.cpu().numpy())
                    valid_labels.extend(label.cpu().numpy())

            train_loss /= len(train_loader)
            valid_loss /= len(valid_loader)

            # Compute IC
            train_ic = self._compute_ic(np.array(train_preds), np.array(train_labels))
            valid_ic = self._compute_ic(np.array(valid_preds), np.array(valid_labels))

            # Learning rate scheduling
            self.scheduler.step(valid_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            if verbose:
                print(f"    Epoch {epoch+1:3d}: train={train_loss:.6f}, valid={valid_loss:.6f}, "
                      f"train_ic={train_ic:.4f}, valid_ic={valid_ic:.4f}, lr={current_lr:.2e}")

            # NaN protection
            if np.isnan(train_loss) or np.isnan(valid_loss):
                if verbose:
                    print(f"    NaN detected, restoring checkpoint")
                if best_params:
                    self.model.load_state_dict(best_params)
                break

            if valid_loss < best_loss:
                best_loss, best_epoch = valid_loss, epoch
                best_params = copy.deepcopy(self.model.state_dict())
                stop_steps = 0
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    if verbose:
                        print(f"    Early stop at epoch {epoch+1}")
                    break

        if best_params:
            self.model.load_state_dict(best_params)
        self.fitted = True
        return best_epoch + 1, best_loss

    def _compute_ic(self, pred, label):
        """Compute Information Coefficient."""
        mask = ~np.isnan(label) & ~np.isnan(pred)
        if mask.sum() < 2:
            return 0.0
        pred_centered = pred[mask] - pred[mask].mean()
        label_centered = label[mask] - label[mask].mean()
        cov = (pred_centered * label_centered).mean()
        pred_std = pred_centered.std() + 1e-8
        label_std = label_centered.std() + 1e-8
        return cov / (pred_std * label_std)

    def predict(self, data_loader):
        """Generate predictions."""
        self.model.eval()
        preds = []
        with torch.no_grad():
            for stock, macro, _ in data_loader:
                pred = self.model(stock.to(self.device), macro.to(self.device))
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds)

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'state_dict': self.model.state_dict(),
            'config': self.config
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.config = ckpt['config']
        self._init_model()
        self.model.load_state_dict(ckpt['state_dict'])
        self.fitted = True


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_data(dataset, segment, macro_df, macro_cols, d_feat=6, seq_len=60, lag=1):
    """Prepare dataset for training/validation/test."""
    features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
    labels = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)

    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    labels = labels.iloc[:, 0].fillna(0) if isinstance(labels, pd.DataFrame) else labels.fillna(0)

    macro = prepare_macro(features.index, macro_df, macro_cols, lag)
    ds = TransformerMacroDataset(features.values, macro, labels.values, d_feat, seq_len)
    return ds, features.index, labels.values


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Transformer-FiLM Training')
    parser.add_argument('--stock-pool', default='sp500', choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--handler', default='alpha300', choices=['alpha300', 'alpha360'])
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--macro-set', default='core', choices=['minimal', 'core'])
    parser.add_argument('--macro-lag', type=int, default=1)

    # Model hyperparameters
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dim-feedforward', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--film-hidden', type=int, default=32)

    # Training hyperparameters
    parser.add_argument('--n-epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--early-stop', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None)

    # Options
    parser.add_argument('--skip-cv', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=1)

    args = parser.parse_args()

    init_qlib(use_talib=False)

    macro_df = load_macro_df()
    macro_cols = get_macro_cols(args.macro_set)
    available = [c for c in macro_cols if c in macro_df.columns]
    n_macro = len(available)

    # Determine d_feat and seq_len based on handler
    if args.handler == 'alpha360':
        d_feat, seq_len = 6, 60
    else:  # alpha300
        d_feat, seq_len = 5, 60

    print(f"\n{'='*70}")
    print(f"Transformer-FiLM | {args.stock_pool} | {args.handler} | macro={args.macro_set}({n_macro})")
    print(f"{'='*70}")

    # Create handler and dataset
    HandlerClass = get_handler_class(args.handler)
    handler = HandlerClass(
        volatility_window=args.nday,
        instruments=STOCK_POOLS[args.stock_pool],
        start_time=FINAL_TEST['train_start'],
        end_time=FINAL_TEST['test_end'],
        fit_start_time=FINAL_TEST['train_start'],
        fit_end_time=FINAL_TEST['train_end'],
    )
    dataset = DatasetH(handler=handler, segments={
        "train": (FINAL_TEST['train_start'], FINAL_TEST['train_end']),
        "valid": (FINAL_TEST['valid_start'], FINAL_TEST['valid_end']),
        "test": (FINAL_TEST['test_start'], FINAL_TEST['test_end']),
    })

    train_ds, _, _ = prepare_data(dataset, "train", macro_df, available, d_feat, seq_len, args.macro_lag)
    valid_ds, _, _ = prepare_data(dataset, "valid", macro_df, available, d_feat, seq_len, args.macro_lag)
    test_ds, test_idx, test_labels = prepare_data(dataset, "test", macro_df, available, d_feat, seq_len, args.macro_lag)

    print(f"Train: {len(train_ds):,}, Valid: {len(valid_ds):,}, Test: {len(test_ds):,}")
    print(f"Input shape: ({seq_len}, {d_feat}), Macro features: {n_macro}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Train
    trainer = TransformerFiLMTrainer(
        d_feat=d_feat,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        seq_len=seq_len,
        n_macro=n_macro,
        film_hidden=args.film_hidden,
        n_epochs=args.n_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        early_stop=args.early_stop,
        gpu=args.gpu,
        seed=args.seed,
    )

    print(f"\nModel config: d_model={args.d_model}, nhead={args.nhead}, "
          f"layers={args.num_layers}, ff={args.dim_feedforward}")
    print(f"FiLM config: n_macro={n_macro}, film_hidden={args.film_hidden}")

    best_epoch, best_loss = trainer.fit(train_loader, valid_loader, verbose=args.verbose)
    print(f"\nBest epoch: {best_epoch}, loss: {best_loss:.6f}")

    # Count parameters
    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Evaluate
    test_pred = trainer.predict(test_loader)
    ic, ic_std, icir = compute_ic(test_pred, test_labels, test_idx)

    print(f"\n{'='*40}")
    print(f"  Test IC:   {ic:.4f}")
    print(f"  IC Std:    {ic_std:.4f}")
    print(f"  ICIR:      {icir:.4f}")
    print(f"{'='*40}")

    # Save
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_SAVE_PATH / f"transformer_film_{args.stock_pool}_{args.macro_set}_{ts}.pt"
    trainer.save(model_path)
    print(f"Model saved: {model_path}")

    # Backtest
    if args.backtest:
        pred_df = pd.Series(test_pred, index=test_idx, name='score').to_frame("score")
        run_backtest(
            model_path, dataset, pred_df, args,
            {k: FINAL_TEST[k] for k in ['train_start', 'train_end', 'valid_start', 'valid_end', 'test_start', 'test_end']},
            model_name="Transformer-FiLM",
            load_model_func=lambda p: TransformerFiLMTrainer(gpu=args.gpu).load(p) or trainer,
            get_feature_count_func=lambda m: d_feat * seq_len
        )


if __name__ == "__main__":
    main()
