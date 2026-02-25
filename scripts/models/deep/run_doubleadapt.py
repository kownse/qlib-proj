"""
DoubleAdapt + FiLM Macro Conditioning Training Script

Meta-learning based incremental learning with FiLM macro modulation.
Stock features (alpha300) go through FeatureAdapter + GRU backbone,
macro features modulate GRU hidden state via FiLM.

Usage:
    # Smoke test
    python scripts/models/deep/run_doubleadapt.py \
        --stock-pool test --macro-set minimal \
        --step 20 --hidden-size 32 --num-layers 1

    # SP500 full training
    python scripts/models/deep/run_doubleadapt.py \
        --stock-pool sp500 --macro-set minimal \
        --step 10 --hidden-size 64 --num-layers 2 \
        --num-head 8 --tau 10 --first-order \
        --lr 0.001 --lr-da 0.01 --lr-ma 0.001

    # Naive baseline (no adapters)
    python scripts/models/deep/run_doubleadapt.py \
        --stock-pool sp500 --naive
"""

import os
import sys
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from data.stock_pools import STOCK_POOLS
from models.common import (
    MODEL_SAVE_PATH, FINAL_TEST,
    compute_ic, run_backtest,
)
from models.common.config import QLIB_DATA_PATH
from models.common.handlers import get_handler_class
from models.common.macro_features import load_macro_df, get_macro_cols

from models.deep.doubleadapt.net import GRUFiLM
from models.deep.doubleadapt.model import IncrementalManager, DoubleAdaptFiLMManager
from models.deep.doubleadapt.utils import (
    TimeAdjuster, organize_all_tasks, get_rolling_data_with_macro,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='DoubleAdapt + FiLM Macro Conditioning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument('--stock-pool', default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--nday', type=int, default=5, help='Prediction horizon')
    parser.add_argument('--macro-set', default='minimal',
                        choices=['minimal', 'core'])
    parser.add_argument('--macro-lag', type=int, default=1)

    # Rolling window
    parser.add_argument('--step', type=int, default=10,
                        help='Rolling step in trading days (~2 weeks)')
    parser.add_argument('--trunc-days', type=int, default=2,
                        help='Gap between train end and test start')

    # GRU backbone
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)

    # FiLM
    parser.add_argument('--film-hidden', type=int, default=32)

    # Adapters
    parser.add_argument('--num-head', type=int, default=8)
    parser.add_argument('--tau', type=float, default=10.0, help='Adapter temperature')
    parser.add_argument('--adapt-x', action='store_true', default=True,
                        help='Enable FeatureAdapter')
    parser.add_argument('--no-adapt-x', action='store_true',
                        help='Disable FeatureAdapter')
    parser.add_argument('--adapt-y', action='store_true', default=True,
                        help='Enable LabelAdapter')
    parser.add_argument('--no-adapt-y', action='store_true',
                        help='Disable LabelAdapter')

    # Meta-learning
    parser.add_argument('--lr', type=float, default=0.001, help='Backbone LR')
    parser.add_argument('--lr-da', type=float, default=0.01,
                        help='Data adapter LR (FeatureAdapter + LabelAdapter)')
    parser.add_argument('--lr-ma', type=float, default=0.001,
                        help='Model adapter LR (backbone outer loop)')
    parser.add_argument('--reg', type=float, default=0.5,
                        help='LabelAdapter regularization')
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--first-order', action='store_true', default=True,
                        help='First-order MAML (saves memory)')
    parser.add_argument('--no-first-order', action='store_true',
                        help='Use second-order MAML')
    parser.add_argument('--begin-valid-epoch', type=int, default=0)
    parser.add_argument('--patience', type=int, default=8,
                        help='Early stopping patience')

    # Naive baseline
    parser.add_argument('--naive', action='store_true',
                        help='Use naive incremental learning (no adapters)')

    # Hardware
    parser.add_argument('--gpu', type=int, default=0)

    # Backtest
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=2)
    parser.add_argument('--account', type=float, default=1000000)
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss'])

    args = parser.parse_args()

    # Handle negation flags
    if args.no_adapt_x:
        args.adapt_x = False
    if args.no_adapt_y:
        args.adapt_y = False
    if args.no_first_order:
        args.first_order = False

    return args


def get_trade_calendar():
    """Get US trade calendar from qlib."""
    from qlib.data import D
    cal = D.calendar(start_time='2000-01-01', end_time='2026-12-31', freq='day')
    return list(cal)


def extract_full_dataframe(dataset):
    """Extract full stock data as a single DataFrame from DatasetH."""
    features = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
    labels = dataset.prepare("train", col_set="label", data_key=DataHandlerLP.DK_L)

    features_v = dataset.prepare("valid", col_set="feature", data_key=DataHandlerLP.DK_L)
    labels_v = dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)

    features_t = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_L)
    labels_t = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)

    all_features = pd.concat([features, features_v, features_t])
    all_labels = pd.concat([labels, labels_v, labels_t])

    all_features = all_features.fillna(0).replace([np.inf, -np.inf], 0)
    all_labels = all_labels.fillna(0)

    # Combine into single DataFrame with 'feature' and 'label' column groups
    df = pd.concat(
        {"feature": all_features, "label": all_labels},
        axis=1,
    )
    print(f"  Full data: {df.shape}, {df.index.get_level_values(0).min()} ~ "
          f"{df.index.get_level_values(0).max()}")
    return df


def main():
    args = parse_args()

    # Device
    device = torch.device(
        f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'
    )

    # Init qlib
    qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US)

    # Constants
    d_feat = 5      # OHLCV per timestep
    seq_len = 60    # 60 trading days
    x_dim = 300     # d_feat * seq_len
    factor_num = d_feat  # FeatureAdapter operates on d_feat dimension

    symbols = STOCK_POOLS[args.stock_pool]

    print("=" * 70)
    print("DoubleAdapt + FiLM Macro Conditioning")
    print("=" * 70)
    print(f"  Stock pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"  Macro: {args.macro_set}")
    print(f"  N-day: {args.nday}")
    print(f"  Step: {args.step} trading days")
    print(f"  Device: {device}")
    print(f"  Mode: {'Naive' if args.naive else 'DoubleAdapt'}")
    if not args.naive:
        print(f"  Adapt X: {args.adapt_x}, Adapt Y: {args.adapt_y}")
        print(f"  First order: {args.first_order}")
        print(f"  Heads: {args.num_head}, Tau: {args.tau}")
        print(f"  LR: backbone={args.lr}, adapter={args.lr_da}, model={args.lr_ma}")
    print(f"  GRU: hidden={args.hidden_size}, layers={args.num_layers}")
    print(f"  FiLM hidden: {args.film_hidden}")
    print("=" * 70)

    # [1] Load macro data
    print(f"\n[1] Loading macro data...")
    macro_df = load_macro_df()
    macro_cols = get_macro_cols(args.macro_set)
    available_macro = [c for c in macro_cols if c in macro_df.columns]
    n_macro = len(available_macro)
    print(f"  Macro features: {n_macro} ({args.macro_set})")

    # [2] Create dataset with alpha300 handler
    print(f"\n[2] Creating dataset (alpha300 handler)...")
    HandlerClass = get_handler_class('alpha300')
    handler = HandlerClass(
        volatility_window=args.nday,
        instruments=symbols,
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

    # [3] Extract full dataframe
    print(f"\n[3] Extracting full dataframe...")
    stock_data = extract_full_dataframe(dataset)

    # [4] Build trade calendar and rolling tasks
    print(f"\n[4] Building rolling tasks...")
    calendar = get_trade_calendar()
    ta = TimeAdjuster(calendar)
    segments = {
        'train': (FINAL_TEST['train_start'], FINAL_TEST['train_end']),
        'valid': (FINAL_TEST['valid_start'], FINAL_TEST['valid_end']),
        'test': (FINAL_TEST['test_start'], FINAL_TEST['test_end']),
    }
    rolling_tasks = organize_all_tasks(
        segments, ta, step=args.step, trunc_days=args.trunc_days,
    )
    print(f"  Train tasks: {len(rolling_tasks['train'])}")
    print(f"  Valid tasks: {len(rolling_tasks['valid'])}")
    print(f"  Test tasks: {len(rolling_tasks['test'])}")

    # [5] Prepare task data with macro
    print(f"\n[5] Preparing task data with macro features...")
    meta_tasks = {}
    for phase in ['train', 'valid', 'test']:
        print(f"  Processing {phase}...")
        meta_tasks[phase] = get_rolling_data_with_macro(
            rolling_tasks[phase],
            stock_data,
            macro_df=macro_df,
            macro_cols=available_macro,
            factor_num=factor_num,
            horizon=args.nday,
            sequence_last_dim=True,
            macro_lag=args.macro_lag,
        )
        print(f"    {phase}: {len(meta_tasks[phase])} tasks")
        if len(meta_tasks[phase]) > 0:
            t0 = meta_tasks[phase][0]
            print(f"    X_train: {t0['X_train'].shape}, macro_train: {t0.get('macro_train', torch.tensor([])).shape}")

    # [6] Create model
    print(f"\n[6] Creating model...")
    gru_film = GRUFiLM(
        d_feat=d_feat,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        n_macro=n_macro,
        film_hidden=args.film_hidden,
    )
    print(f"  GRUFiLM params: {sum(p.numel() for p in gru_film.parameters()):,}")

    if args.naive:
        manager = IncrementalManager(
            model=gru_film,
            lr_model=args.lr,
            x_dim=x_dim,
            need_permute=False,
            over_patience=args.patience,
            begin_valid_epoch=args.begin_valid_epoch,
        )
    else:
        manager = DoubleAdaptFiLMManager(
            model=gru_film,
            lr_model=args.lr,
            lr_da=args.lr_da,
            lr_ma=args.lr_ma,
            reg=args.reg,
            adapt_x=args.adapt_x,
            adapt_y=args.adapt_y,
            first_order=args.first_order,
            factor_num=factor_num,
            x_dim=x_dim,
            need_permute=False,
            num_head=args.num_head,
            temperature=args.tau,
            begin_valid_epoch=args.begin_valid_epoch,
            weight_decay=args.weight_decay,
        )
    total_params = sum(p.numel() for p in manager.framework.parameters())
    print(f"  Total framework params: {total_params:,}")

    # [7] Training
    print(f"\n[7] Training...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    mode_str = "naive" if args.naive else "da_film"
    ckpt_path = str(MODEL_SAVE_PATH / f"doubleadapt_{mode_str}_{args.stock_pool}_{args.macro_set}_{args.nday}d_{timestamp}.pt")

    manager.fit(
        meta_tasks['train'],
        meta_tasks['valid'],
        checkpoint_path=ckpt_path,
    )

    # [8] Inference on test set
    print(f"\n[8] Test inference...")
    pred_y = manager.inference(meta_tasks['test'])

    # [9] Evaluation
    print(f"\n[9] Evaluation")
    print("-" * 70)

    # Overall test IC
    if isinstance(pred_y, pd.DataFrame):
        pred_vals = pred_y["pred"].values
        label_vals = pred_y["label"].values
        test_index = pred_y.index

        ic, std, icir = compute_ic(pred_vals, label_vals, test_index)
        print(f"  Test IC: {ic:.4f} (std: {std:.4f}), ICIR: {icir:.4f}")

        # Monthly IC breakdown
        print(f"\n[10] Monthly IC breakdown")
        print("-" * 70)
        ic_by_date = pred_y.groupby("datetime").apply(
            lambda df: df["pred"].corr(df["label"]) if len(df) > 1 else np.nan
        ).dropna()

        if len(ic_by_date) > 0:
            ic_by_date.index = pd.to_datetime(ic_by_date.index)
            monthly = ic_by_date.groupby(ic_by_date.index.to_period('M')).agg(['mean', 'std', 'count'])
            monthly.columns = ['IC_mean', 'IC_std', 'days']
            monthly['ICIR'] = monthly['IC_mean'] / monthly['IC_std']
            print(monthly.to_string())
    else:
        print("  WARNING: unexpected prediction format")
        ic = 0.0

    # Save info
    info = {
        'model': 'DoubleAdapt-FiLM' if not args.naive else 'Naive-Incremental',
        'stock_pool': args.stock_pool,
        'macro_set': args.macro_set,
        'n_macro': n_macro,
        'nday': args.nday,
        'step': args.step,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'film_hidden': args.film_hidden,
        'num_head': args.num_head,
        'tau': args.tau,
        'adapt_x': args.adapt_x,
        'adapt_y': args.adapt_y,
        'first_order': args.first_order,
        'lr': args.lr,
        'lr_da': args.lr_da,
        'lr_ma': args.lr_ma,
        'reg': args.reg,
        'test_ic': float(ic),
        'checkpoint': ckpt_path,
    }
    info_path = ckpt_path.replace('.pt', '_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"\n  Checkpoint: {ckpt_path}")
    print(f"  Info: {info_path}")

    # Backtest
    if args.backtest and isinstance(pred_y, pd.DataFrame):
        print(f"\n[11] Backtest")
        print("-" * 70)
        pred_df = pred_y["pred"].to_frame("score")
        time_splits = {k: FINAL_TEST[k] for k in
                       ['train_start', 'train_end', 'valid_start', 'valid_end',
                        'test_start', 'test_end']}
        run_backtest(
            ckpt_path, dataset, pred_df, args, time_splits,
            model_name="DoubleAdapt-FiLM",
            load_model_func=lambda p: None,
            get_feature_count_func=lambda m: x_dim,
        )

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
