"""
TCN-FiLM with Macro Conditioning - Training Script

Usage:
    python scripts/models/deep/run_tcn_macro_cv.py --stock-pool sp500 --macro-set core --skip-cv
    python scripts/models/deep/run_tcn_macro_cv.py --stock-pool sp500 --macro-set minimal --skip-cv
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
from models.deep.tcn_film import TCNFiLM


# ============================================================================
# Dataset
# ============================================================================

class TCNMacroDataset(Dataset):
    def __init__(self, stock_features, macro_features, labels, d_feat, step_len):
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
# Trainer
# ============================================================================

class TCNFiLMTrainer:
    def __init__(self, d_feat=5, n_macro=6, n_chans=32, num_layers=5,
                 kernel_size=7, dropout=0.5, film_hidden=32,
                 n_epochs=200, lr=1e-4, batch_size=2000, early_stop=20,
                 gpu=0, seed=None):
        self.config = {
            'd_feat': d_feat, 'n_macro': n_macro, 'n_chans': n_chans,
            'num_layers': num_layers, 'kernel_size': kernel_size,
            'dropout': dropout, 'film_hidden': film_hidden,
        }
        self.n_epochs = n_epochs
        self.lr = lr
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
        self.model = TCNFiLM(**self.config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, train_loader, valid_loader, verbose=True):
        self._init_model()
        best_loss, best_epoch, best_params = float('inf'), 0, None
        stop_steps = 0

        for epoch in range(self.n_epochs):
            # Train
            self.model.train()
            train_loss = 0
            for stock, macro, label in train_loader:
                stock, macro, label = stock.to(self.device), macro.to(self.device), label.to(self.device)
                pred = self.model(stock, macro)
                mask = ~torch.isnan(label)
                loss = ((pred[mask] - label[mask]) ** 2).mean() if mask.sum() > 0 else torch.tensor(0.0)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
                self.optimizer.step()
                train_loss += loss.item()

            # Validate
            self.model.eval()
            valid_loss = 0
            with torch.no_grad():
                for stock, macro, label in valid_loader:
                    stock, macro, label = stock.to(self.device), macro.to(self.device), label.to(self.device)
                    pred = self.model(stock, macro)
                    mask = ~torch.isnan(label)
                    if mask.sum() > 0:
                        valid_loss += ((pred[mask] - label[mask]) ** 2).mean().item()

            train_loss /= len(train_loader)
            valid_loss /= len(valid_loader)

            if verbose:
                print(f"    Epoch {epoch+1:3d}: train={train_loss:.6f}, valid={valid_loss:.6f}")

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

    def predict(self, data_loader):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for stock, macro, _ in data_loader:
                pred = self.model(stock.to(self.device), macro.to(self.device))
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds)

    def save(self, path):
        torch.save({'state_dict': self.model.state_dict(), 'config': self.config}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.config = ckpt['config']
        self._init_model()
        self.model.load_state_dict(ckpt['state_dict'])
        self.fitted = True


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_data(dataset, segment, macro_df, macro_cols, d_feat=5, step_len=60, lag=1):
    features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
    labels = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)

    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    labels = labels.iloc[:, 0].fillna(0) if isinstance(labels, pd.DataFrame) else labels.fillna(0)

    macro = prepare_macro(features.index, macro_df, macro_cols, lag)
    ds = TCNMacroDataset(features.values, macro, labels.values, d_feat, step_len)
    return ds, features.index, labels.values


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='TCN-FiLM Training')
    parser.add_argument('--stock-pool', default='sp500', choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--macro-set', default='core', choices=['minimal', 'core'])
    parser.add_argument('--macro-lag', type=int, default=1)
    parser.add_argument('--n-chans', type=int, default=32)
    parser.add_argument('--num-layers', type=int, default=5)
    parser.add_argument('--kernel-size', type=int, default=7)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--film-hidden', type=int, default=32)
    parser.add_argument('--n-epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=2000)
    parser.add_argument('--early-stop', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None)
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

    print(f"\n{'='*60}")
    print(f"TCN-FiLM | {args.stock_pool} | macro={args.macro_set}({n_macro})")
    print(f"{'='*60}")

    # Create handler and dataset
    HandlerClass = get_handler_class('alpha300')
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

    d_feat, step_len = 5, 60
    train_ds, _, _ = prepare_data(dataset, "train", macro_df, available, d_feat, step_len, args.macro_lag)
    valid_ds, _, _ = prepare_data(dataset, "valid", macro_df, available, d_feat, step_len, args.macro_lag)
    test_ds, test_idx, test_labels = prepare_data(dataset, "test", macro_df, available, d_feat, step_len, args.macro_lag)

    print(f"Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Train
    trainer = TCNFiLMTrainer(
        d_feat=d_feat, n_macro=n_macro, n_chans=args.n_chans,
        num_layers=args.num_layers, kernel_size=args.kernel_size,
        dropout=args.dropout, film_hidden=args.film_hidden,
        n_epochs=args.n_epochs, lr=args.lr, batch_size=args.batch_size,
        early_stop=args.early_stop, gpu=args.gpu, seed=args.seed,
    )

    best_epoch, best_loss = trainer.fit(train_loader, valid_loader, verbose=args.verbose)
    print(f"Best epoch: {best_epoch}, loss: {best_loss:.6f}")

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
    model_path = MODEL_SAVE_PATH / f"tcn_film_{args.stock_pool}_{args.macro_set}_{ts}.pt"
    trainer.save(model_path)
    print(f"Model saved: {model_path}")

    # Backtest
    if args.backtest:
        pred_df = pd.Series(test_pred, index=test_idx, name='score').to_frame("score")
        run_backtest(
            model_path, dataset, pred_df, args,
            {k: FINAL_TEST[k] for k in ['train_start', 'train_end', 'valid_start', 'valid_end', 'test_start', 'test_end']},
            model_name="TCN-FiLM",
            load_model_func=lambda p: TCNFiLMTrainer(gpu=args.gpu).load(p) or trainer,
            get_feature_count_func=lambda m: d_feat * step_len
        )


if __name__ == "__main__":
    main()
