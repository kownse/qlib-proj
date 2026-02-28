"""
AE-MLP + AE-MLP (mkt-neutral) + AE-MLP-V9R (rolling window) + CatBoost Ensemble (V5)

Four-model ensemble:
  1. AE-MLP (alpha158-enhanced-v7, train 2000-2023)
  2. AE-MLP (v9-mkt-neutral, train 2000-2023)
  3. AE-MLP V9R (alpha158-enhanced-v9, train 2015-2023)
  4. CatBoost (catboost-v1)

Based on run_ae_cb_ensemble_v4.py with V9 rolling window model added.

Usage:
    python scripts/models/ensemble/run_ae_cb_ensemble_v5.py
    python scripts/models/ensemble/run_ae_cb_ensemble_v5.py --ensemble-method rank_mean
    python scripts/models/ensemble/run_ae_cb_ensemble_v5.py --stacking
    python scripts/models/ensemble/run_ae_cb_ensemble_v5.py --backtest --topk 10 --n-drop 2
"""

import os

# Set thread limits before any other imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
from pathlib import Path
import multiprocessing

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Add scripts directory to path
script_dir = Path(__file__).parent.parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

import argparse
import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_US
from qlib.data.dataset.handler import DataHandlerLP

from utils.talib_ops import TALIB_OPS
from utils.ai_filter import apply_ai_affinity_filter
from data.stock_pools import STOCK_POOLS

from models.common import (
    PROJECT_ROOT,
    QLIB_DATA_PATH,
    MODEL_SAVE_PATH,
    CV_FOLDS,
    FINAL_TEST,
)
from models.common.ensemble import (
    load_ae_mlp_model,
    load_model_meta,
    create_ensemble_data_handler,
    create_ensemble_dataset,
    predict_with_ae_mlp,
    predict_with_catboost,
    zscore_by_day,
    calculate_pairwise_correlations,
    compute_ic,
    ensemble_predictions,
    learn_optimal_weights,
    run_ensemble_backtest,
)
from models.common.training import load_catboost_model


# ── V5-specific helpers ──────────────────────────────────────────────

def align_features_for_catboost(X, model):
    """Align data features to match CatBoost model's expected feature names"""
    model_features = model.feature_names_
    if not model_features:
        return X

    if isinstance(X.columns, pd.MultiIndex):
        X = X.copy()
        X.columns = [col[1] if isinstance(col, tuple) else col for col in X.columns]

    data_cols = set(X.columns)
    missing = [c for c in model_features if c not in data_cols]
    if missing:
        for c in missing:
            X[c] = 0.0

    return X[list(model_features)]


def predict_catboost_raw(model, dataset, segment):
    """Predict with CatBoost using raw data, with feature alignment"""
    data = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
    data = data.fillna(0).replace([np.inf, -np.inf], 0)
    data = align_features_for_catboost(data, model)
    pred_values = model.predict(data)
    return pd.Series(pred_values, index=data.index, name='score')


# ── CV evaluation ─────────────────────────────────────────────────────

def run_cv_evaluation(models: dict, handlers: dict, args, symbols, MODEL_CONFIG):
    """Evaluate ensemble IC on CV folds."""
    model_names = list(models.keys())
    display_names = {k: cfg['display'] for k, cfg in MODEL_CONFIG.items()}

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION EVALUATION (4-Model Ensemble V5)")
    print("=" * 80)
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: valid {fold['valid_start']}~{fold['valid_end']}")
    print(f"Test Set: {FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']} (2025)")
    print(f"Ensemble Method: {args.ensemble_method}")
    print("=" * 80)

    # Prepare 2025 test set
    print("\n[CV] Preparing 2025 test data...")
    test_time = {
        'train_start': FINAL_TEST['train_start'],
        'train_end': FINAL_TEST['train_end'],
        'valid_start': FINAL_TEST['valid_start'],
        'valid_end': FINAL_TEST['valid_end'],
        'test_start': FINAL_TEST['test_start'],
        'test_end': FINAL_TEST['test_end'],
    }

    # Create test datasets for each handler
    test_datasets = {}
    for key in model_names:
        h = create_ensemble_data_handler(handlers[key], symbols, test_time, args.nday,
                                         include_valid=True)
        test_datasets[key] = create_ensemble_dataset(h, test_time, include_valid=True)

    # Test predictions
    test_preds = {}
    for key in model_names:
        if key == 'cb':
            test_preds[key] = predict_catboost_raw(models[key], test_datasets[key], "test")
        else:
            test_preds[key] = predict_with_ae_mlp(models[key], test_datasets[key], segment="test")

    weights = None
    if args.ensemble_method in ['weighted', 'zscore_weighted']:
        weights = {n: getattr(args, f'{n}_weight') for n in model_names}

    test_pred_ens = ensemble_predictions(test_preds, args.ensemble_method, weights)

    # Labels from first AE-MLP dataset
    test_label_df = test_datasets['ae'].prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)
    test_label = test_label_df.iloc[:, 0] if isinstance(test_label_df, pd.DataFrame) else test_label_df

    test_ics = {}
    for key in model_names:
        ic, _, _, _ = compute_ic(test_preds[key], test_label)
        test_ics[key] = ic
    test_ens_ic, _, _, _ = compute_ic(test_pred_ens, test_label)

    parts = ", ".join(f"{display_names[k]} IC={test_ics[k]:.4f}" for k in model_names)
    print(f"    Test (2025): {parts}, Ensemble IC={test_ens_ic:.4f}")

    fold_results = []

    for fold in CV_FOLDS:
        print(f"\n[CV] Evaluating on {fold['name']}...")

        fold_time = {
            'train_start': fold['train_start'],
            'train_end': fold['train_end'],
            'valid_start': fold['valid_start'],
            'valid_end': fold['valid_end'],
            'test_start': fold['valid_start'],
            'test_end': fold['valid_end'],
        }

        # Create fold datasets
        fold_datasets = {}
        for key in model_names:
            h = create_ensemble_data_handler(handlers[key], symbols, fold_time, args.nday,
                                             include_valid=True)
            fold_datasets[key] = create_ensemble_dataset(h, fold_time, include_valid=True)

        # Validation predictions
        val_preds = {}
        for key in model_names:
            if key == 'cb':
                val_preds[key] = predict_catboost_raw(models[key], fold_datasets[key], "valid")
            else:
                val_preds[key] = predict_with_ae_mlp(models[key], fold_datasets[key], segment="valid")

        val_pred_ens = ensemble_predictions(val_preds, args.ensemble_method, weights)

        # Labels
        val_label_df = fold_datasets['ae'].prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
        val_label = val_label_df.iloc[:, 0] if isinstance(val_label_df, pd.DataFrame) else val_label_df

        # IC
        fold_ic = {}
        for key in model_names:
            ic, _, _, _ = compute_ic(val_preds[key], val_label)
            fold_ic[key] = ic
        ens_ic, _, ens_icir, _ = compute_ic(val_pred_ens, val_label)

        result = {'name': fold['name'], 'ens_ic': ens_ic, 'ens_icir': ens_icir}
        for key in model_names:
            result[f'{key}_ic'] = fold_ic[key]
        fold_results.append(result)

        parts = ", ".join(f"{display_names[k]}={fold_ic[k]:.4f}" for k in model_names)
        print(f"    {fold['name']}: {parts}, Ensemble={ens_ic:.4f}")

    # Summary table
    header_parts = "".join(f" {display_names[k]:>12s}" for k in model_names)
    print("\n" + "=" * 80)
    print("CV EVALUATION COMPLETE (4-Model Ensemble V5)")
    print("=" * 80)
    print(f"{'':25s}{header_parts} {'Ensemble':>12s}")
    print(f"{'-'*25}" + " ".join([f"{'-'*12}"] * (len(model_names) + 1)))
    for r in fold_results:
        vals = "".join(f" {r[f'{k}_ic']:>12.4f}" for k in model_names)
        print(f"{r['name']:<25s}{vals} {r['ens_ic']:>12.4f}")
    print(f"{'-'*25}" + " ".join([f"{'-'*12}"] * (len(model_names) + 1)))

    mean_vals = "".join(f" {np.mean([r[f'{k}_ic'] for r in fold_results]):>12.4f}" for k in model_names)
    std_vals = "".join(f" {np.std([r[f'{k}_ic'] for r in fold_results]):>12.4f}" for k in model_names)
    test_vals = "".join(f" {test_ics[k]:>12.4f}" for k in model_names)

    ens_ics = [r['ens_ic'] for r in fold_results]
    print(f"{'Valid Mean IC':<25s}{mean_vals} {np.mean(ens_ics):>12.4f}")
    print(f"{'Valid IC Std':<25s}{std_vals} {np.std(ens_ics):>12.4f}")
    print(f"{'Test IC (2025)':<25s}{test_vals} {test_ens_ic:>12.4f}")
    print("=" * 80)

    return fold_results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='AE-MLP + AE-MLP(mkt-neutral) + AE-MLP-V9R + CatBoost Ensemble (V5)',
    )

    # Model paths
    parser.add_argument('--ae-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'ae_mlp_cv_alpha158-enhanced-v7_sp500_5d_best.keras'),
                        help='AE-MLP V7 model path (.keras)')
    parser.add_argument('--ae-mn-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'ae_mlp_cv_v9-mkt-neutral_sp500_5d.keras'),
                        help='AE-MLP market-neutral model path (.keras)')
    parser.add_argument('--ae-v9r-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'ae_mlp_cv_alpha158-enhanced-v9_sp500_5d_20_from_2015.keras'),
                        help='AE-MLP V9 rolling window model path (.keras)')
    parser.add_argument('--cb-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'catboost_cv_catboost-v1_sp500_5d_20260129_141353_best.cbm'),
                        help='CatBoost model path (.cbm)')

    # Handler configuration (override metadata)
    parser.add_argument('--ae-handler', type=str, default='alpha158-enhanced-v7',
                        help='Handler for AE-MLP V7 model')
    parser.add_argument('--ae-mn-handler', type=str, default='v9-mkt-neutral',
                        help='Handler for AE-MLP market-neutral model')
    parser.add_argument('--ae-v9r-handler', type=str, default='alpha158-enhanced-v9',
                        help='Handler for AE-MLP V9 rolling model')
    parser.add_argument('--cb-handler', type=str, default='catboost-v1',
                        help='Handler for CatBoost model')

    # Ensemble parameters
    parser.add_argument('--ensemble-method', type=str, default='zscore_mean',
                        choices=['mean', 'weighted', 'rank_mean', 'zscore_mean', 'zscore_weighted'],
                        help='Ensemble method (default: zscore_mean)')
    parser.add_argument('--ae-weight', type=float, default=0.25,
                        help='AE-MLP V7 weight (default: 0.25)')
    parser.add_argument('--ae-mn-weight', type=float, default=0.25,
                        help='AE-MLP mkt-neutral weight (default: 0.25)')
    parser.add_argument('--ae-v9r-weight', type=float, default=0.25,
                        help='AE-MLP V9 rolling weight (default: 0.25)')
    parser.add_argument('--cb-weight', type=float, default=0.25,
                        help='CatBoost weight (default: 0.25)')

    # Stacking parameters
    parser.add_argument('--stacking', action='store_true',
                        help='Learn optimal weights from validation set (Stacking)')
    parser.add_argument('--stacking-method', type=str, default='grid_search',
                        choices=['grid_search', 'grid_search_icir'],
                        help='Stacking weight learning method (default: grid_search)')
    parser.add_argument('--min-weight', type=float, default=0.05,
                        help='Minimum weight for each model (default: 0.05)')
    parser.add_argument('--diversity-bonus', type=float, default=0.1,
                        help='Bonus for balanced weights (default: 0.1)')

    # Data parameters
    parser.add_argument('--nday', type=int, default=5,
                        help='Prediction horizon (default: 5)')
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'],
                        help='Stock pool (default: sp500)')

    # Backtest parameters
    parser.add_argument('--backtest', action='store_true',
                        help='Run backtest after ensemble')
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of stocks to hold (default: 10)')
    parser.add_argument('--n-drop', type=int, default=2,
                        help='Number of stocks to drop/replace each day (default: 2)')
    parser.add_argument('--account', type=float, default=100000,
                        help='Initial account value (default: 100000)')
    parser.add_argument('--rebalance-freq', type=int, default=5,
                        help='Rebalance frequency in days (default: 5)')
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss',
                                 'mvo', 'rp', 'gmv', 'inv'],
                        help='Trading strategy (default: topk)')

    # Skip CV evaluation
    parser.add_argument('--skip-cv', action='store_true',
                        help='Skip cross-validation evaluation, go directly to ensemble + backtest')

    # Strategy parameters
    parser.add_argument('--risk-lookback', type=int, default=20)
    parser.add_argument('--drawdown-threshold', type=float, default=-0.10)
    parser.add_argument('--momentum-threshold', type=float, default=0.03)
    parser.add_argument('--risk-high', type=float, default=0.50)
    parser.add_argument('--risk-medium', type=float, default=0.75)
    parser.add_argument('--risk-normal', type=float, default=0.95)
    parser.add_argument('--market-proxy', type=str, default='SPY')
    parser.add_argument('--vol-high', type=float, default=0.35)
    parser.add_argument('--vol-medium', type=float, default=0.25)
    parser.add_argument('--stop-loss', type=float, default=-0.15)
    parser.add_argument('--no-sell-after-drop', type=float, default=-0.20)

    # Portfolio optimization parameters (for mvo/rp/gmv/inv strategies)
    parser.add_argument('--opt-lamb', type=float, default=2.0,
                        help='[mvo] Risk aversion (higher=safer, default: 2.0)')
    parser.add_argument('--opt-delta', type=float, default=0.2,
                        help='[mvo/rp/gmv] Max turnover per rebalance (default: 0.2)')
    parser.add_argument('--opt-alpha', type=float, default=0.01,
                        help='[mvo/rp/gmv] L2 regularization (default: 0.01)')
    parser.add_argument('--cov-lookback', type=int, default=60,
                        help='[mvo/rp/gmv/inv] Covariance lookback days (default: 60)')
    parser.add_argument('--max-weight', type=float, default=0.0,
                        help='[mvo/rp/gmv/inv] Max weight per stock, 0=no limit (default: 0, try 0.15)')

    # AI affinity filter
    parser.add_argument('--ai-filter', type=str, default='none',
                        choices=['none', 'penalty', 'exclude'],
                        help='AI affinity filter mode (default: none)')
    parser.add_argument('--ai-penalty-weight', type=float, default=0.5,
                        help='Penalty multiplier for negative-affinity stocks (default: 0.5)')
    parser.add_argument('--ai-bonus-weight', type=float, default=0.0,
                        help='Bonus multiplier for positive-affinity stocks (default: 0.0)')
    parser.add_argument('--ai-exclude-threshold', type=int, default=-1,
                        help='Affinity threshold for exclude mode, drop if <= this (default: -1)')
    parser.add_argument('--no-ai-time-scale', action='store_true',
                        help='Disable AI affinity time scaling (ramp 2020-2024)')

    args = parser.parse_args()

    # Use FINAL_TEST time splits
    time_splits = {
        'train_start': FINAL_TEST['train_start'],
        'train_end': FINAL_TEST['train_end'],
        'valid_start': FINAL_TEST['valid_start'],
        'valid_end': FINAL_TEST['valid_end'],
        'test_start': FINAL_TEST['test_start'],
        'test_end': FINAL_TEST['test_end'],
    }

    # Model config
    MODEL_CONFIG = {
        'ae': {
            'path': Path(args.ae_model),
            'handler_arg': 'ae_handler',
            'handler': args.ae_handler,
            'display': 'AE-MLP-V7',
            'type': 'ae_mlp',
        },
        'ae_mn': {
            'path': Path(args.ae_mn_model),
            'handler_arg': 'ae_mn_handler',
            'handler': args.ae_mn_handler,
            'display': 'AE-MLP-MN',
            'type': 'ae_mlp',
        },
        'ae_v9r': {
            'path': Path(args.ae_v9r_model),
            'handler_arg': 'ae_v9r_handler',
            'handler': args.ae_v9r_handler,
            'display': 'AE-MLP-V9R',
            'type': 'ae_mlp',
        },
        'cb': {
            'path': Path(args.cb_model),
            'handler_arg': 'cb_handler',
            'handler': args.cb_handler,
            'display': 'CatBoost',
            'type': 'catboost',
        },
    }

    print("=" * 80)
    print("AE-MLP-V7 + AE-MLP-MN + AE-MLP-V9R + CatBoost Ensemble (V5)")
    print("=" * 80)
    for key, cfg in MODEL_CONFIG.items():
        print(f"{cfg['display']} Model:   {cfg['path']}")
        print(f"{cfg['display']} Handler: {cfg['handler']}")
    print(f"Stock Pool: {args.stock_pool}")
    print(f"Prediction Horizon: {args.nday} days")
    print(f"Ensemble Method: {args.ensemble_method}")
    if args.ensemble_method in ['weighted', 'zscore_weighted']:
        parts = ", ".join(f"{cfg['display']}={getattr(args, f'{key}_weight')}"
                          for key, cfg in MODEL_CONFIG.items())
        print(f"Weights: {parts}")
    print(f"Test Period: {time_splits['test_start']} to {time_splits['test_end']}")
    print("=" * 80)

    # Check model files exist
    for key, cfg in MODEL_CONFIG.items():
        if not cfg['path'].exists():
            print(f"Error: {cfg['display']} model not found: {cfg['path']}")
            sys.exit(1)

    # [1] Load metadata
    print("\n[1] Loading model metadata...")
    for key, cfg in MODEL_CONFIG.items():
        meta = load_model_meta(cfg['path'])
        if meta:
            print(f"    {cfg['display']} metadata: handler={meta.get('handler', 'N/A')}, nday={meta.get('nday', 'N/A')}")
            if 'handler' in meta:
                cfg['handler'] = meta['handler']
                setattr(args, cfg['handler_arg'], meta['handler'])
        else:
            print(f"    {cfg['display']} metadata not found, using default: {cfg['handler']}")

    # [2] Initialize Qlib
    print("\n[2] Initializing Qlib...")
    qlib.init(
        provider_uri=str(QLIB_DATA_PATH),
        region=REG_US,
        custom_ops=TALIB_OPS,
        kernels=1,
        joblib_backend=None,
    )
    print("    Qlib initialized with TA-Lib support")

    # [3] Stock pool
    symbols = STOCK_POOLS[args.stock_pool]
    print(f"\n[3] Using stock pool: {args.stock_pool} ({len(symbols)} stocks)")

    # [4] Create datasets
    print("\n[4] Creating datasets...")
    datasets = {}
    handlers_dict = {}
    for key, cfg in MODEL_CONFIG.items():
        print(f"    Creating {cfg['handler']} dataset for {cfg['display']}...")
        h = create_ensemble_data_handler(cfg['handler'], symbols, time_splits, args.nday,
                                         include_valid=True)
        datasets[key] = create_ensemble_dataset(h, time_splits, include_valid=True)
        handlers_dict[key] = cfg['handler']

    # [5] Load models
    print("\n[5] Loading models...")
    models = {}
    for key, cfg in MODEL_CONFIG.items():
        if cfg['type'] == 'ae_mlp':
            models[key] = load_ae_mlp_model(cfg['path'])
        elif cfg['type'] == 'catboost':
            models[key] = load_catboost_model(cfg['path'])

    # [6] CV evaluation
    if args.skip_cv:
        print("\n[6] Skipping cross-validation evaluation (--skip-cv)")
    else:
        print("\n[6] Cross-validation evaluation...")
        run_cv_evaluation(models, handlers_dict, args, symbols, MODEL_CONFIG)

    # [7] Generate predictions
    print("\n[7] Generating predictions...")
    pred_dict = {}
    for key, cfg in MODEL_CONFIG.items():
        print(f"    {cfg['display']} predictions...")
        if cfg['type'] == 'ae_mlp':
            pred = predict_with_ae_mlp(models[key], datasets[key])
        elif cfg['type'] == 'catboost':
            pred = predict_with_catboost(models[key], datasets[key])
        pred_dict[key] = pred
        print(f"      Shape: {len(pred)}, Range: [{pred.min():.4f}, {pred.max():.4f}]")

    # Prediction statistics
    print("\n    Prediction Statistics Comparison:")
    header = f"    {'Metric':<15s}"
    for cfg in MODEL_CONFIG.values():
        header += f" | {cfg['display']:>12s}"
    print("    " + "=" * (15 + 17 * len(MODEL_CONFIG)))
    print(header)
    print("    " + "-" * (15 + 17 * len(MODEL_CONFIG)))

    for metric_name, metric_fn in [('Mean', lambda s: s.mean()), ('Std', lambda s: s.std()),
                                    ('Median', lambda s: s.median()), ('Abs Mean', lambda s: s.abs().mean()),
                                    ('Min', lambda s: s.min()), ('Max', lambda s: s.max())]:
        row = f"    {metric_name:<15s}"
        for key in MODEL_CONFIG:
            row += f" | {metric_fn(pred_dict[key]):>12.6f}"
        print(row)
    print("    " + "=" * (15 + 17 * len(MODEL_CONFIG)))

    # [8] Correlation
    print("\n[8] Calculating pairwise correlations...")
    overall_corr = calculate_pairwise_correlations(pred_dict)

    print("\n    Overall Correlation Matrix:")
    print(overall_corr.to_string())

    # Compute daily correlations
    names = list(pred_dict.keys())
    common_idx = pred_dict[names[0]].index
    for name in names[1:]:
        common_idx = common_idx.intersection(pred_dict[name].index)
    corr_df = pd.DataFrame({name: pred_dict[name].loc[common_idx] for name in names})

    daily_corrs = {}
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j <= i:
                continue
            pair_key = f"{n1} vs {n2}"
            dc = corr_df.groupby(level='datetime').apply(
                lambda x, _n1=n1, _n2=n2: x[_n1].corr(x[_n2]) if len(x) > 1 else np.nan
            ).dropna()
            daily_corrs[pair_key] = (dc.mean(), dc.std())

    print("\n    Daily Correlation (mean +/- std):")
    print("    " + "-" * 50)
    for pair, (mean_c, std_c) in daily_corrs.items():
        print(f"    {pair:<30s}: {mean_c:.4f} +/- {std_c:.4f}")
    print("    " + "-" * 50)

    # [9] Stacking (optional)
    learned_weights = None
    if args.stacking:
        print(f"\n[9] Stacking: Learning optimal weights from validation set...")
        print(f"    Method: {args.stacking_method}")

        val_preds = {}
        for key, cfg in MODEL_CONFIG.items():
            if cfg['type'] == 'ae_mlp':
                val_preds[key] = predict_with_ae_mlp(models[key], datasets[key], segment="valid")
            elif cfg['type'] == 'catboost':
                val_preds[key] = predict_with_catboost(models[key], datasets[key], segment="valid")
            print(f"      {cfg['display']} valid: {len(val_preds[key])} samples")

        val_label = datasets['ae'].prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(val_label, pd.DataFrame):
            val_label = val_label.iloc[:, 0]

        learned_weights, learn_info = learn_optimal_weights(
            val_preds, val_label,
            method=args.stacking_method,
            min_weight=args.min_weight,
            diversity_bonus=args.diversity_bonus,
        )

        print(f"\n    Learned Weights:")
        print(f"    " + "-" * 50)
        for key, cfg in MODEL_CONFIG.items():
            print(f"    {cfg['display']:15s}: {learned_weights[key]:>10.4f}")
        print(f"    " + "-" * 50)

        for k, v in learn_info.items():
            if k != 'method':
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

        args.ensemble_method = 'zscore_weighted'
        for key in MODEL_CONFIG:
            setattr(args, f'{key}_weight', learned_weights[key])
        print(f"\n    Using zscore_weighted ensemble with learned weights")

    # Ensemble
    step_num = 10 if args.stacking else 9
    print(f"\n[{step_num}] Ensembling predictions ({args.ensemble_method})...")

    weights = None
    if args.ensemble_method in ['weighted', 'zscore_weighted']:
        if learned_weights:
            weights = learned_weights
        else:
            weights = {key: getattr(args, f'{key}_weight') for key in MODEL_CONFIG}

    pred_ensemble = ensemble_predictions(pred_dict, args.ensemble_method, weights)
    print(f"    Ensemble shape: {len(pred_ensemble)}, Range: [{pred_ensemble.min():.4f}, {pred_ensemble.max():.4f}]")

    # AI affinity filter
    pred_raw = None
    if args.ai_filter != 'none':
        step_num += 1
        print(f"\n[{step_num}] Applying AI affinity filter ({args.ai_filter})...")
        pred_raw = pred_ensemble.copy()
        pred_ensemble = apply_ai_affinity_filter(
            pred_ensemble,
            mode=args.ai_filter,
            penalty_weight=args.ai_penalty_weight,
            bonus_weight=args.ai_bonus_weight,
            exclude_threshold=args.ai_exclude_threshold,
            time_scale=not args.no_ai_time_scale,
        )
        print(f"    Filtered shape: {len(pred_ensemble)}, Range: [{pred_ensemble.min():.4f}, {pred_ensemble.max():.4f}]")

    # IC metrics
    step_num += 1
    print(f"\n[{step_num}] Calculating IC metrics...")
    test_label = datasets['ae'].prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(test_label, pd.DataFrame):
        label = test_label.iloc[:, 0]
    else:
        label = test_label

    ic_results = {}
    for key, cfg in MODEL_CONFIG.items():
        ic, std, icir, ic_series = compute_ic(pred_dict[key], label)
        ic_results[key] = {'ic': ic, 'std': std, 'icir': icir}

    ens_ic, ens_std, ens_icir, ens_ic_series = compute_ic(pred_ensemble, label)

    print("\n    +" + "=" * 70 + "+")
    print("    |" + " " * 10 + "Information Coefficient (IC) Comparison (V5)" + " " * 14 + "|")
    print("    +" + "=" * 70 + "+")
    print(f"    |  {'Model':<20s} | {'Mean IC':>10s} | {'IC Std':>10s} | {'ICIR':>10s} |")
    print("    +" + "-" * 70 + "+")
    for key, cfg in MODEL_CONFIG.items():
        r = ic_results[key]
        print(f"    |  {cfg['display']:<20s} | {r['ic']:>10.4f} | {r['std']:>10.4f} | {r['icir']:>10.4f} |")
    print("    +" + "-" * 70 + "+")
    print(f"    |  {'Ensemble (V5)':<20s} | {ens_ic:>10.4f} | {ens_std:>10.4f} | {ens_icir:>10.4f} |")
    print("    +" + "=" * 70 + "+")

    # Improvement
    best_single_ic = max(r['ic'] for r in ic_results.values())
    best_single_icir = max(r['icir'] for r in ic_results.values())

    ic_improvement = (ens_ic - best_single_ic) / abs(best_single_ic) * 100 if best_single_ic != 0 else 0
    icir_improvement = (ens_icir - best_single_icir) / abs(best_single_icir) * 100 if best_single_icir != 0 else 0

    print(f"\n    Ensemble Performance vs Best Single Model:")
    print(f"    IC improvement:   {ic_improvement:>+.2f}%")
    print(f"    ICIR improvement: {icir_improvement:>+.2f}%")

    # Dual IC comparison when AI filter is active
    if pred_raw is not None:
        raw_ic, raw_std, raw_icir, _ = compute_ic(pred_raw, label)
        print(f"\n    AI Filter Impact on IC:")
        print(f"      Before filter: IC={raw_ic:.4f}, ICIR={raw_icir:.4f}")
        print(f"      After filter:  IC={ens_ic:.4f}, ICIR={ens_icir:.4f}")
        ic_delta = ens_ic - raw_ic
        icir_delta = ens_icir - raw_icir
        print(f"      Delta:         IC={ic_delta:+.4f}, ICIR={icir_delta:+.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("ENSEMBLE V5 ANALYSIS COMPLETE")
    print("=" * 80)
    print("Pairwise Correlations (daily mean):")
    for pair, (mean_c, _) in daily_corrs.items():
        print(f"  {pair}: {mean_c:.4f}")
    if learned_weights:
        print("Stacking Weights:")
        for key, cfg in MODEL_CONFIG.items():
            print(f"  {cfg['display']}: {learned_weights[key]:.3f}")
    for key, cfg in MODEL_CONFIG.items():
        r = ic_results[key]
        print(f"{cfg['display']:15s} IC: {r['ic']:.4f} (ICIR: {r['icir']:.4f})")
    print(f"{'Ensemble V5':15s} IC: {ens_ic:.4f} (ICIR: {ens_icir:.4f})")
    print("=" * 80)

    # Backtest
    if args.backtest:
        run_ensemble_backtest(pred_ensemble, args, time_splits, model_name="Ensemble_V5")

        print("\n" + "=" * 80)
        print("ENSEMBLE V5 BACKTEST COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    main()
