"""
TCN-FiLM 超参数搜索 - Optuna 版本 (带 Pruning + Macro Conditioning)

融合 run_tcn_optuna_cv.py 的 Optuna 搜索框架和 run_tcn_macro_cv.py 的 FiLM + macro 数据管道。

搜索空间 (相比 plain TCN 扩展):
  - n_chans: [32, 64, 128, 256, 512]
  - film_hidden: [16, 32, 64, 128]
  - kernel_size: [3, 5, 7]
  - num_layers: [2, 6]
  - dropout: [0.05, 0.6]
  - lr: [1e-5, 1e-2]
  - batch_size: [512, 1024, 2048, 4096, 8192]

时间窗口 (4-fold CV):
  Fold 1: train 2000-2020, valid 2021
  Fold 2: train 2000-2021, valid 2022
  Fold 3: train 2000-2022, valid 2023
  Fold 4: train 2000-2023, valid 2024
  Test:   2025 (完全独立)

使用方法:
    python scripts/models/deep/run_tcn_film_optuna_cv.py
    python scripts/models/deep/run_tcn_film_optuna_cv.py --n-trials 80 --macro-set core
    python scripts/models/deep/run_tcn_film_optuna_cv.py --resume  # 断点续传

    # 用已有最优参数直接训练 + 回测
    python scripts/models/deep/run_tcn_film_optuna_cv.py --skip-search --params-file outputs/hyperopt_cv/tcn_film_optuna_best.json --backtest
"""

import os
import sys
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import qlib
from qlib.constant import REG_US
qlib_data_path = script_dir.parent / "my_data" / "qlib_us"
qlib.init(provider_uri=str(qlib_data_path), region=REG_US)

import argparse
import copy
import json
from datetime import datetime
import numpy as np
import pandas as pd

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import torch
from torch.utils.data import DataLoader, Dataset

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from data.stock_pools import STOCK_POOLS
from models.common import (
    PROJECT_ROOT, MODEL_SAVE_PATH,
    run_backtest, CV_FOLDS, FINAL_TEST,
    compute_ic,
)
from models.common.handlers import get_handler_class
from models.common.macro_features import (
    load_macro_df, get_macro_cols, prepare_macro,
)
from models.deep.tcn_film import TCNFiLM


# ============================================================================
# Dataset
# ============================================================================

class TCNMacroDataset(Dataset):
    def __init__(self, stock_features, macro_features, labels):
        """stock_features already reshaped to (N, d_feat, step_len)"""
        self.stock = stock_features
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
# Data Preparation
# ============================================================================

def prepare_fold_data(dataset, segment, macro_df, macro_cols, d_feat=5, step_len=60, lag=1):
    """Prepare stock (reshaped) + macro + labels for a dataset segment."""
    features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
    labels = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)

    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    labels = labels.iloc[:, 0].fillna(0) if isinstance(labels, pd.DataFrame) else labels.fillna(0)

    macro = prepare_macro(features.index, macro_df, macro_cols, lag)

    # Reshape stock features: (N, d_feat*step_len) -> (N, d_feat, step_len)
    stock = features.values.reshape(-1, d_feat, step_len)

    return stock, macro, labels.values, features.index


# ============================================================================
# Optuna Objective
# ============================================================================

class TCNFiLMObjective:
    """Optuna 目标函数：TCN-FiLM with macro conditioning + pruning"""

    def __init__(self, macro_df, macro_cols, n_macro, args,
                 n_epochs=50, early_stop=10, gpu=0):
        self.macro_df = macro_df
        self.macro_cols = macro_cols
        self.n_macro = n_macro
        self.args = args
        self.n_epochs = n_epochs
        self.early_stop = early_stop

        self.device = torch.device(
            f'cuda:{gpu}' if gpu >= 0 and torch.cuda.is_available() else 'cpu'
        )

        self.d_feat = 5
        self.step_len = 60

        # Prepare all fold data
        print("\n[*] Preparing data for all CV folds...")
        self.fold_data = []

        for fold in CV_FOLDS:
            print(f"    Preparing {fold['name']}...")
            HandlerClass = get_handler_class('alpha300')
            handler = HandlerClass(
                volatility_window=args.nday,
                instruments=STOCK_POOLS[args.stock_pool],
                start_time=fold['train_start'],
                end_time=fold['valid_end'],
                fit_start_time=fold['train_start'],
                fit_end_time=fold['train_end'],
            )
            dataset = DatasetH(handler=handler, segments={
                "train": (fold['train_start'], fold['train_end']),
                "valid": (fold['valid_start'], fold['valid_end']),
            })

            stock_train, macro_train, y_train, _ = prepare_fold_data(
                dataset, "train", macro_df, macro_cols,
                self.d_feat, self.step_len, args.macro_lag,
            )
            stock_valid, macro_valid, y_valid, valid_index = prepare_fold_data(
                dataset, "valid", macro_df, macro_cols,
                self.d_feat, self.step_len, args.macro_lag,
            )

            self.fold_data.append({
                'name': fold['name'],
                'stock_train': stock_train, 'macro_train': macro_train, 'y_train': y_train,
                'stock_valid': stock_valid, 'macro_valid': macro_valid, 'y_valid': y_valid,
                'valid_index': valid_index,
            })

            print(f"      Train: {stock_train.shape[0]} samples, Valid: {stock_valid.shape[0]} samples")

        print(f"    All {len(CV_FOLDS)} folds prepared")
        print(f"    Stock: d_feat={self.d_feat}, step_len={self.step_len}")
        print(f"    Macro: n_macro={self.n_macro}")

    def __call__(self, trial: optuna.Trial):
        """目标函数：搜索超参数 + 4-fold CV"""

        # Hyperparameter search space
        n_chans = trial.suggest_categorical('n_chans', [32, 64, 128, 256, 512])
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
        num_layers = trial.suggest_int('num_layers', 2, 6)
        dropout = trial.suggest_float('dropout', 0.05, 0.6)
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048, 4096, 8192])
        film_hidden = trial.suggest_categorical('film_hidden', [16, 32, 64, 128])

        print(f"\n{'='*70}")
        print(f"Trial {trial.number}: chans={n_chans}, layers={num_layers}, "
              f"kernel={kernel_size}, dropout={dropout:.3f}, lr={lr:.6f}, "
              f"batch={batch_size}, film_hidden={film_hidden}")
        print(f"{'='*70}")

        fold_ics = []

        for fold_idx, fold in enumerate(self.fold_data):
            print(f"\n  [{fold['name']}]")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Create model
            model = TCNFiLM(
                d_feat=self.d_feat,
                n_macro=self.n_macro,
                n_chans=n_chans,
                num_layers=num_layers,
                kernel_size=kernel_size,
                dropout=dropout,
                film_hidden=film_hidden,
            ).to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Create DataLoaders
            train_ds = TCNMacroDataset(fold['stock_train'], fold['macro_train'], fold['y_train'])
            valid_ds = TCNMacroDataset(fold['stock_valid'], fold['macro_valid'], fold['y_valid'])
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
            valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

            # Training loop
            best_loss = float('inf')
            best_state = None
            best_val_ic = 0.0
            stop_steps = 0

            for epoch in range(self.n_epochs):
                # Train
                model.train()
                train_losses = []
                for stock, macro, label in train_loader:
                    stock = stock.to(self.device)
                    macro = macro.to(self.device)
                    label = label.to(self.device)

                    pred = model(stock, macro)
                    mask = ~torch.isnan(label)
                    if mask.sum() == 0:
                        continue
                    loss = ((pred[mask] - label[mask]) ** 2).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
                    optimizer.step()
                    train_losses.append(loss.item())

                if not train_losses:
                    break
                train_loss = np.mean(train_losses)

                # NaN protection
                if np.isnan(train_loss):
                    print(f"    NaN detected at epoch {epoch+1}, stopping fold")
                    if best_state:
                        model.load_state_dict(best_state)
                    break

                # Validate
                model.eval()
                val_preds, val_labels_list = [], []
                with torch.no_grad():
                    for stock, macro, label in valid_loader:
                        stock = stock.to(self.device)
                        macro = macro.to(self.device)
                        pred = model(stock, macro)
                        val_preds.append(pred.cpu().numpy())
                        val_labels_list.append(label.numpy())

                val_pred = np.concatenate(val_preds)
                val_label = np.concatenate(val_labels_list)
                val_loss = np.mean((val_pred - val_label) ** 2)

                # IC on validation set
                val_ic, _, _ = compute_ic(val_pred, fold['y_valid'], fold['valid_index'])

                # Track best
                is_best = ""
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_val_ic = val_ic
                    stop_steps = 0
                    is_best = " *"
                else:
                    stop_steps += 1

                print(f"    Epoch {epoch+1:2d}/{self.n_epochs}: "
                      f"train_loss={train_loss:.6f} | "
                      f"val_loss={val_loss:.6f} val_IC={val_ic:.4f}{is_best}")

                if stop_steps >= self.early_stop:
                    print(f"    Early stopping at epoch {epoch+1} (best val_IC={best_val_ic:.4f})")
                    break

                # Pruning: report to Optuna every 5 epochs
                if epoch % 5 == 0 and epoch > 0:
                    intermediate = float(best_val_ic if fold_idx == 0 else np.mean(fold_ics + [best_val_ic]))
                    trial.report(intermediate, epoch + fold_idx * self.n_epochs)
                    if trial.should_prune():
                        print(f"    >>> PRUNED (intermediate IC={intermediate:.4f})")
                        raise optuna.TrialPruned()

            # Final IC for this fold using best model
            if best_state is not None:
                model.load_state_dict(best_state)

            model.eval()
            val_preds = []
            with torch.no_grad():
                for stock, macro, _ in valid_loader:
                    pred = model(stock.to(self.device), macro.to(self.device))
                    val_preds.append(pred.cpu().numpy())
            val_pred = np.concatenate(val_preds)

            mean_ic, _, icir = compute_ic(val_pred, fold['y_valid'], fold['valid_index'])
            fold_ics.append(mean_ic)

            print(f"  [{fold['name']}] Final IC={mean_ic:.4f}, ICIR={icir:.4f}")

            del model, optimizer, train_loader, valid_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Return mean IC across folds
        mean_ic_all = float(np.mean(fold_ics))
        std_ic_all = float(np.std(fold_ics))

        print(f"\n  Trial {trial.number} Result: Mean IC={mean_ic_all:.4f} (+-{std_ic_all:.4f})")

        trial.set_user_attr('fold_ics', [float(ic) for ic in fold_ics])
        trial.set_user_attr('std_ic', std_ic_all)

        return mean_ic_all


# ============================================================================
# Search + Train
# ============================================================================

def run_optuna_search(args, macro_df, macro_cols, n_macro):
    """Run Optuna hyperparameter search."""
    print("\n" + "=" * 70)
    print("OPTUNA SEARCH: TCN-FiLM with Macro Conditioning")
    print("=" * 70)
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: train {fold['train_start']}~{fold['train_end']}, "
              f"valid {fold['valid_start']}~{fold['valid_end']}")
    print(f"Macro set: {args.macro_set} ({n_macro} features)")
    print(f"Max trials: {args.n_trials}")
    print(f"Epochs per fold: {args.cv_epochs}")
    print("=" * 70)

    objective = TCNFiLMObjective(
        macro_df, macro_cols, n_macro, args,
        n_epochs=args.cv_epochs,
        early_stop=args.cv_early_stop,
        gpu=args.gpu,
    )

    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=5,
    )
    sampler = TPESampler(seed=42)
    storage = f"sqlite:///{PROJECT_ROOT}/outputs/optuna_tcn_film.db"

    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    def print_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            fold_ics = trial.user_attrs.get('fold_ics', [])
            fold_str = ", ".join([f"{ic:.4f}" for ic in fold_ics])
            print(f"  Trial {trial.number:3d}: IC={trial.value:.4f} [{fold_str}] "
                  f"chans={trial.params['n_chans']} layers={trial.params['num_layers']} "
                  f"lr={trial.params['lr']:.5f} film={trial.params['film_hidden']}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"  Trial {trial.number:3d}: PRUNED")

    print("\n[*] Running optimization...")
    study.optimize(
        objective,
        n_trials=args.n_trials,
        callbacks=[print_callback],
        show_progress_bar=True,
    )

    best_trial = study.best_trial
    best_params = dict(best_trial.params)

    print("\n" + "=" * 70)
    print("SEARCH COMPLETE")
    print("=" * 70)
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Total trials: {len(study.trials)} (completed: {n_complete}, pruned: {n_pruned})")
    print(f"\nBest CV IC: {best_trial.value:.4f}")
    if 'fold_ics' in best_trial.user_attrs:
        fold_ics = best_trial.user_attrs['fold_ics']
        print(f"Fold ICs: {[f'{ic:.4f}' for ic in fold_ics]}")
    print(f"\nBest parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    print("=" * 70)

    # Save best params
    output_dir = PROJECT_ROOT / "outputs" / "hyperopt_cv"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    params_file = output_dir / f"tcn_film_optuna_best_{timestamp}.json"
    save_data = {
        'best_cv_ic': float(best_trial.value),
        'fold_ics': best_trial.user_attrs.get('fold_ics', []),
        'std_ic': float(best_trial.user_attrs.get('std_ic', 0)),
        'params': {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in best_params.items()},
        'macro_set': args.macro_set,
        'n_macro': n_macro,
        'stock_pool': args.stock_pool,
        'nday': args.nday,
        'n_trials': len(study.trials),
        'n_pruned': n_pruned,
        'timestamp': timestamp,
    }
    with open(params_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nBest parameters saved to: {params_file}")

    # Also save a stable-name copy for easy reference
    stable_file = output_dir / "tcn_film_optuna_best.json"
    with open(stable_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Also saved to: {stable_file}")

    return best_params, study


def train_final_model(args, macro_df, macro_cols, n_macro, best_params):
    """Train final model using best params on FINAL_TEST split."""
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL")
    print("=" * 70)

    d_feat, step_len = 5, 60

    # Create dataset
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

    stock_train, macro_train, y_train, _ = prepare_fold_data(
        dataset, "train", macro_df, macro_cols, d_feat, step_len, args.macro_lag,
    )
    stock_valid, macro_valid, y_valid, valid_index = prepare_fold_data(
        dataset, "valid", macro_df, macro_cols, d_feat, step_len, args.macro_lag,
    )
    stock_test, macro_test, y_test, test_index = prepare_fold_data(
        dataset, "test", macro_df, macro_cols, d_feat, step_len, args.macro_lag,
    )

    print(f"Train: {stock_train.shape[0]}, Valid: {stock_valid.shape[0]}, Test: {stock_test.shape[0]}")

    batch_size = best_params['batch_size']
    train_ds = TCNMacroDataset(stock_train, macro_train, y_train)
    valid_ds = TCNMacroDataset(stock_valid, macro_valid, y_valid)
    test_ds = TCNMacroDataset(stock_test, macro_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device(
        f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'
    )

    model = TCNFiLM(
        d_feat=d_feat,
        n_macro=n_macro,
        n_chans=best_params['n_chans'],
        num_layers=best_params['num_layers'],
        kernel_size=best_params['kernel_size'],
        dropout=best_params['dropout'],
        film_hidden=best_params['film_hidden'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])

    # Training loop
    best_loss = float('inf')
    best_state = None
    stop_steps = 0

    print("Training...")
    for epoch in range(args.n_epochs):
        model.train()
        train_loss = 0
        n_batches = 0
        for stock, macro, label in train_loader:
            stock, macro, label = stock.to(device), macro.to(device), label.to(device)
            pred = model(stock, macro)
            mask = ~torch.isnan(label)
            if mask.sum() == 0:
                continue
            loss = ((pred[mask] - label[mask]) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            break
        train_loss /= n_batches

        if np.isnan(train_loss):
            print(f"  NaN detected at epoch {epoch+1}, restoring checkpoint")
            if best_state:
                model.load_state_dict(best_state)
            break

        # Validate
        model.eval()
        valid_loss = 0
        n_val = 0
        with torch.no_grad():
            for stock, macro, label in valid_loader:
                stock, macro, label = stock.to(device), macro.to(device), label.to(device)
                pred = model(stock, macro)
                mask = ~torch.isnan(label)
                if mask.sum() > 0:
                    valid_loss += ((pred[mask] - label[mask]) ** 2).mean().item()
                    n_val += 1
        valid_loss = valid_loss / n_val if n_val > 0 else float('inf')

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            stop_steps = 0
            if epoch % 10 == 0:
                print(f"  Epoch {epoch+1:3d}: train={train_loss:.6f}, valid={valid_loss:.6f} (best)")
        else:
            stop_steps += 1
            if stop_steps >= args.early_stop:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # Evaluate on validation
    model.eval()
    val_preds = []
    with torch.no_grad():
        for stock, macro, _ in valid_loader:
            pred = model(stock.to(device), macro.to(device))
            val_preds.append(pred.cpu().numpy())
    val_pred = np.concatenate(val_preds)
    valid_ic, valid_std, valid_icir = compute_ic(val_pred, y_valid, valid_index)
    print(f"\nValid IC: {valid_ic:.4f} (+-{valid_std:.4f}), ICIR: {valid_icir:.4f}")

    # Evaluate on test
    test_preds = []
    with torch.no_grad():
        for stock, macro, _ in test_loader:
            pred = model(stock.to(device), macro.to(device))
            test_preds.append(pred.cpu().numpy())
    test_pred_values = np.concatenate(test_preds)
    test_ic, test_std, test_icir = compute_ic(test_pred_values, y_test, test_index)

    print(f"Test  IC: {test_ic:.4f} (+-{test_std:.4f}), ICIR: {test_icir:.4f}")

    # Save model
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_SAVE_PATH / f"tcn_film_optuna_{args.stock_pool}_{args.macro_set}_{args.nday}d_{ts}.pt"
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'd_feat': d_feat, 'n_macro': n_macro,
            'n_chans': best_params['n_chans'],
            'num_layers': best_params['num_layers'],
            'kernel_size': best_params['kernel_size'],
            'dropout': best_params['dropout'],
            'film_hidden': best_params['film_hidden'],
        },
        'best_params': best_params,
        'valid_ic': valid_ic,
        'test_ic': test_ic,
    }, model_path)
    print(f"Model saved: {model_path}")

    # Prepare prediction series for backtest
    test_pred = pd.Series(test_pred_values, index=test_index, name='score')

    return model, test_pred, dataset, model_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='TCN-FiLM Optuna CV with Macro Conditioning')

    # Basic
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--stock-pool', default='sp500', choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--macro-set', default='core', choices=['minimal', 'core'])
    parser.add_argument('--macro-lag', type=int, default=1)

    # Optuna
    parser.add_argument('--n-trials', type=int, default=60)
    parser.add_argument('--cv-epochs', type=int, default=40,
                        help='Max epochs per CV fold during search')
    parser.add_argument('--cv-early-stop', type=int, default=10)
    parser.add_argument('--study-name', default='tcn_film_cv_search')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-search', action='store_true',
                        help='Skip search, use existing params file')
    parser.add_argument('--params-file', type=str, default=None,
                        help='Path to params JSON (with --skip-search)')

    # Final training
    parser.add_argument('--n-epochs', type=int, default=200)
    parser.add_argument('--early-stop', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=0)

    # Backtest
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=1)
    parser.add_argument('--account', type=float, default=10000)

    args = parser.parse_args()

    # Load macro data
    macro_df = load_macro_df()
    macro_cols = get_macro_cols(args.macro_set)
    available = [c for c in macro_cols if c in macro_df.columns]
    n_macro = len(available)

    print("=" * 70)
    print("TCN-FiLM Hyperparameter Search (Optuna + Pruning)")
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool} ({len(STOCK_POOLS[args.stock_pool])} stocks)")
    print(f"Macro: {args.macro_set} ({n_macro} features)")
    print(f"N-day: {args.nday}")
    print(f"Device: cuda:{args.gpu}" if torch.cuda.is_available() else "Device: cpu")
    print("=" * 70)

    # Search or load params
    if args.skip_search:
        params_file = args.params_file or str(
            PROJECT_ROOT / "outputs" / "hyperopt_cv" / "tcn_film_optuna_best.json"
        )
        print(f"\n[*] Loading params from: {params_file}")
        with open(params_file, 'r') as f:
            saved = json.load(f)
        best_params = saved['params']
        print(f"  CV IC: {saved.get('best_cv_ic', 'N/A')}")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    else:
        best_params, study = run_optuna_search(args, macro_df, macro_cols, n_macro)

    # Train final model
    model, test_pred, dataset, model_path = train_final_model(
        args, macro_df, macro_cols, n_macro, best_params,
    )

    # Backtest
    if args.backtest:
        pred_df = test_pred.to_frame("score")
        time_splits = {k: FINAL_TEST[k] for k in
                       ['train_start', 'train_end', 'valid_start', 'valid_end',
                        'test_start', 'test_end']}

        run_backtest(
            model_path, dataset, pred_df, args, time_splits,
            model_name="TCN-FiLM (Optuna)",
            load_model_func=lambda p: None,
            get_feature_count_func=lambda m: 5 * 60,
        )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
