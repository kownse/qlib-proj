"""
TCN-FiLM 单次训练 / 评估脚本

支持两种模式:
  1. 加载已有 checkpoint 进行评估 (--model-path)
  2. 用指定超参数从头训练 (默认或 --params-file)

使用方法:
    # 验证已训练模型
    python scripts/models/deep/run_tcn_film.py --model-path my_models/tcn_film_optuna_sp500_core_5d_best.pt

    # 验证 + 回测
    python scripts/models/deep/run_tcn_film.py --model-path my_models/tcn_film_optuna_sp500_core_5d_best.pt --backtest

    # 从头训练
    python scripts/models/deep/run_tcn_film.py --stock-pool sp500 --macro-set core

    # 用 JSON 参数文件训练
    python scripts/models/deep/run_tcn_film.py --params-file outputs/hyperopt_cv/tcn_film_optuna_best.json
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

import torch
from torch.utils.data import DataLoader, Dataset

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from data.stock_pools import STOCK_POOLS
from models.common import (
    PROJECT_ROOT, MODEL_SAVE_PATH,
    run_backtest, FINAL_TEST,
    compute_ic,
)
from models.common.handlers import get_handler_class
from models.common.macro_features import (
    load_macro_df, get_macro_cols, prepare_macro,
)
from models.deep.tcn_film import TCNFiLM


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


def evaluate_segment(model, dataset, segment, macro_df, macro_cols, d_feat, step_len,
                     macro_lag, batch_size, device):
    """Evaluate model on a dataset segment, return predictions and IC metrics."""
    stock, macro, y, idx = prepare_fold_data(
        dataset, segment, macro_df, macro_cols, d_feat, step_len, macro_lag,
    )
    ds = TCNMacroDataset(stock, macro, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for s, m, _ in loader:
            preds.append(model(s.to(device), m.to(device)).cpu().numpy())
    pred = np.concatenate(preds)
    ic, std, icir = compute_ic(pred, y, idx)
    pred_series = pd.Series(pred, index=idx, name='score')
    return pred_series, ic, std, icir


def train_model(args, macro_df, macro_cols, n_macro, params, device):
    """Train TCN-FiLM model from scratch using given params."""
    d_feat, step_len = 5, 60

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

    print(f"Train: {stock_train.shape[0]}, Valid: {stock_valid.shape[0]}")

    batch_size = params['batch_size']
    train_ds = TCNMacroDataset(stock_train, macro_train, y_train)
    valid_ds = TCNMacroDataset(stock_valid, macro_valid, y_valid)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    model = TCNFiLM(
        d_feat=d_feat,
        n_macro=n_macro,
        n_chans=params['n_chans'],
        num_layers=params['num_layers'],
        kernel_size=params['kernel_size'],
        dropout=params['dropout'],
        film_hidden=params['film_hidden'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

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

    return model, dataset


def main():
    parser = argparse.ArgumentParser(description='TCN-FiLM Training / Evaluation')

    # Basic
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--stock-pool', default='sp500', choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--macro-set', default='core', choices=['minimal', 'core'])
    parser.add_argument('--macro-lag', type=int, default=1)

    # Model loading
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to pre-trained checkpoint (eval-only mode)')
    parser.add_argument('--params-file', type=str, default=None,
                        help='Path to params JSON (for training from scratch)')

    # Training (ignored if --model-path is given)
    parser.add_argument('--n-epochs', type=int, default=200)
    parser.add_argument('--early-stop', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=0)

    # Backtest
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=2)
    parser.add_argument('--account', type=float, default=1000000)
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss',
                                 'mvo', 'rp', 'gmv', 'inv'])
    parser.add_argument('--rebalance-freq', type=int, default=1)

    # Dynamic risk / vol_stoploss params
    parser.add_argument('--risk-lookback', type=int, default=20)
    parser.add_argument('--drawdown-threshold', type=float, default=-0.10)
    parser.add_argument('--momentum-threshold', type=float, default=0.03)
    parser.add_argument('--risk-high', type=float, default=0.50)
    parser.add_argument('--risk-medium', type=float, default=0.75)
    parser.add_argument('--risk-normal', type=float, default=0.95)
    parser.add_argument('--market-proxy', type=str, default='AAPL')
    parser.add_argument('--vol-high', type=float, default=0.35)
    parser.add_argument('--vol-medium', type=float, default=0.25)
    parser.add_argument('--stop-loss', type=float, default=-0.15)
    parser.add_argument('--no-sell-after-drop', type=float, default=-0.20)

    # Portfolio optimization params
    parser.add_argument('--opt-lamb', type=float, default=2.0)
    parser.add_argument('--opt-delta', type=float, default=0.2)
    parser.add_argument('--opt-alpha', type=float, default=0.01)
    parser.add_argument('--cov-lookback', type=int, default=60)
    parser.add_argument('--max-weight', type=float, default=0.0)

    args = parser.parse_args()

    # Load macro data
    macro_df = load_macro_df()
    macro_cols = get_macro_cols(args.macro_set)
    available = [c for c in macro_cols if c in macro_df.columns]
    n_macro = len(available)

    device = torch.device(
        f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'
    )

    print("=" * 70)
    print("TCN-FiLM" + (" Evaluation" if args.model_path else " Training"))
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool} ({len(STOCK_POOLS[args.stock_pool])} stocks)")
    print(f"Macro: {args.macro_set} ({n_macro} features)")
    print(f"N-day: {args.nday}")
    print(f"Device: {device}")
    print("=" * 70)

    d_feat, step_len = 5, 60

    if args.model_path:
        # ========== Eval-only mode ==========
        model_path = Path(args.model_path)
        print(f"\n[1] Loading checkpoint: {model_path}")
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

        cfg = ckpt['config']
        print(f"    d_feat={cfg['d_feat']}, n_macro={cfg['n_macro']}, "
              f"n_chans={cfg['n_chans']}, layers={cfg['num_layers']}, "
              f"kernel={cfg['kernel_size']}, dropout={cfg['dropout']:.4f}, "
              f"film_hidden={cfg['film_hidden']}")
        if 'best_params' in ckpt:
            bp = ckpt['best_params']
            print(f"    lr={bp.get('lr', 'N/A')}, batch_size={bp.get('batch_size', 'N/A')}")
        if 'valid_ic' in ckpt:
            print(f"    Saved Valid IC: {ckpt['valid_ic']:.4f}")
        if 'test_ic' in ckpt:
            print(f"    Saved Test  IC: {ckpt['test_ic']:.4f}")

        model = TCNFiLM(**cfg).to(device)
        model.load_state_dict(ckpt['state_dict'])
        batch_size = ckpt.get('best_params', {}).get('batch_size', 2048)

        # Build dataset
        print(f"\n[2] Preparing data (alpha300 handler, FINAL_TEST split)...")
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
    else:
        # ========== Training mode ==========
        # Load params
        if args.params_file:
            print(f"\n[1] Loading params from: {args.params_file}")
            with open(args.params_file, 'r') as f:
                saved = json.load(f)
            params = saved['params']
            print(f"    CV IC: {saved.get('best_cv_ic', 'N/A')}")
        else:
            # Default params (from Optuna best)
            params = {
                'n_chans': 512,
                'kernel_size': 3,
                'num_layers': 5,
                'dropout': 0.29,
                'lr': 0.0005,
                'batch_size': 2048,
                'film_hidden': 128,
            }
            print(f"\n[1] Using default params")

        for k, v in params.items():
            print(f"    {k}: {v}")

        batch_size = params['batch_size']

        print(f"\n[2] Training...")
        model, dataset = train_model(args, macro_df, macro_cols, n_macro, params, device)
        model_path = None

    # ========== Evaluation ==========
    print(f"\n[3] Evaluation")
    print("-" * 70)

    valid_pred, valid_ic, valid_std, valid_icir = evaluate_segment(
        model, dataset, 'valid', macro_df, macro_cols,
        d_feat, step_len, args.macro_lag, batch_size, device,
    )
    print(f"  valid: IC={valid_ic:.4f} (+-{valid_std:.4f}), ICIR={valid_icir:.4f}, N={len(valid_pred)}")

    test_pred, test_ic, test_std, test_icir = evaluate_segment(
        model, dataset, 'test', macro_df, macro_cols,
        d_feat, step_len, args.macro_lag, batch_size, device,
    )
    print(f"  test : IC={test_ic:.4f} (+-{test_std:.4f}), ICIR={test_icir:.4f}, N={len(test_pred)}")

    # Print detailed test IC by date
    print(f"\n[4] Test IC by month")
    print("-" * 70)
    _, _, y_test, test_index = prepare_fold_data(
        dataset, "test", macro_df, macro_cols, d_feat, step_len, args.macro_lag,
    )
    df_ic = pd.DataFrame({'pred': test_pred.values, 'label': y_test}, index=test_index)
    ic_by_date = df_ic.groupby(level='datetime').apply(
        lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
    ).dropna()
    ic_monthly = ic_by_date.groupby(pd.Grouper(freq='M')).agg(['mean', 'std', 'count'])
    ic_monthly.columns = ['IC_mean', 'IC_std', 'days']
    ic_monthly['ICIR'] = ic_monthly['IC_mean'] / ic_monthly['IC_std']
    print(ic_monthly.to_string())

    # Save model (training mode only)
    if not args.model_path:
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = MODEL_SAVE_PATH / f"tcn_film_{args.stock_pool}_{args.macro_set}_{args.nday}d_{ts}.pt"
        torch.save({
            'state_dict': model.state_dict(),
            'config': {
                'd_feat': d_feat, 'n_macro': n_macro,
                'n_chans': params['n_chans'],  # type: ignore[possibly-undefined]
                'num_layers': params['num_layers'],
                'kernel_size': params['kernel_size'],
                'dropout': params['dropout'],
                'film_hidden': params['film_hidden'],
            },
            'best_params': params,
            'valid_ic': float(valid_ic),
            'test_ic': float(test_ic),
        }, save_path)
        print(f"\n[*] Model saved: {save_path}")
        model_path = save_path

    # Backtest
    if args.backtest:
        print(f"\n[5] Backtest")
        print("-" * 70)
        pred_df = test_pred.to_frame("score")
        time_splits = {k: FINAL_TEST[k] for k in
                       ['train_start', 'train_end', 'valid_start', 'valid_end',
                        'test_start', 'test_end']}

        run_backtest(
            model_path, dataset, pred_df, args, time_splits,
            model_name="TCN-FiLM",
            load_model_func=lambda p: None,
            get_feature_count_func=lambda m: 5 * 60,
        )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
