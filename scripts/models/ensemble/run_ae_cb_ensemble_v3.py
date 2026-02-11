"""
AE-MLP + CatBoost Ensemble V3: 4x AE-MLP + 1 CatBoost

Extends V2 ensemble by adding AE-MLP with V12 handler
(V9 mkt-neutral + CBOE skew features).

Models:
  - AE-MLP v7: alpha158-enhanced-v7 handler, raw return target
  - AE-MLP v9: alpha158-enhanced-v9 handler, raw return target
  - AE-MLP mkt-neutral: v9-mkt-neutral handler, market-neutral return target
  - AE-MLP v12: alpha158-enhanced-v12 handler, mkt-neutral + CBOE skew
  - CatBoost: catboost-v1 handler, raw return target

Usage:
    # Basic ensemble with auto-learned weights
    python scripts/models/ensemble/run_ae_cb_ensemble_v3.py

    # With backtest
    python scripts/models/ensemble/run_ae_cb_ensemble_v3.py --backtest --topk 10

    # Compare with V2 ensemble (4 models, without v12)
    python scripts/models/ensemble/run_ae_cb_ensemble_v3.py --exclude-v12
"""

import os

# Set thread limits before any other imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['JOBLIB_START_METHOD'] = 'fork'
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

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
from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG,
    PROJECT_ROOT,
    QLIB_DATA_PATH,
    MODEL_SAVE_PATH,
    FINAL_TEST,
)

# Import ensemble utilities from V1
from models.ensemble.run_ae_cb_ensemble import (
    load_ae_mlp_model,
    load_catboost_model,
    create_data_handler,
    create_dataset,
    predict_with_ae_mlp,
    predict_with_catboost,
    calculate_pairwise_correlations,
    learn_optimal_weights_multi,
    ensemble_predictions_multi,
    compute_ic,
    run_ensemble_backtest,
)


# ============================================================================
# Model Configuration
# ============================================================================

MODEL_CONFIGS = {
    'AE-MLP-v7': {
        'model_path': MODEL_SAVE_PATH / 'ae_mlp_cv_alpha158-enhanced-v7_sp500_5d_best.keras',
        'handler': 'alpha158-enhanced-v7',
        'type': 'ae_mlp',
    },
    'AE-MLP-v9': {
        'model_path': MODEL_SAVE_PATH / 'ae_mlp_cv_alpha158-enhanced-v9_test_5d_best.keras',
        'handler': 'alpha158-enhanced-v9',
        'type': 'ae_mlp',
    },
    'AE-MLP-mkt-neutral': {
        'model_path': MODEL_SAVE_PATH / 'ae_mlp_cv_v9-mkt-neutral_test_5d_20260210_151953.keras',
        'handler': 'v9-mkt-neutral',
        'type': 'ae_mlp',
    },
    'AE-MLP-v12': {
        'model_path': MODEL_SAVE_PATH / 'ae_mlp_cv_alpha158-enhanced-v12_test_5d.keras',
        'handler': 'alpha158-enhanced-v12',
        'type': 'ae_mlp',
    },
    'CatBoost': {
        'model_path': MODEL_SAVE_PATH / 'catboost_cv_catboost-v1_test_5d_20260129_105915_best.cbm',
        'handler': 'catboost-v1',
        'type': 'catboost',
    },
}


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='AE-MLP + CatBoost Ensemble V3: 4x AE-MLP + 1 CatBoost',
    )

    # Model paths (override defaults)
    parser.add_argument('--ae-v7-model', type=str, default=None,
                        help='Override AE-MLP v7 model path')
    parser.add_argument('--ae-v9-model', type=str, default=None,
                        help='Override AE-MLP v9 model path')
    parser.add_argument('--ae-mkt-model', type=str, default=None,
                        help='Override AE-MLP mkt-neutral model path')
    parser.add_argument('--ae-v12-model', type=str, default=None,
                        help='Override AE-MLP v12 model path')
    parser.add_argument('--cb-model', type=str, default=None,
                        help='Override CatBoost model path')

    # Model exclusion
    parser.add_argument('--exclude-v7', action='store_true',
                        help='Exclude AE-MLP v7')
    parser.add_argument('--exclude-v9', action='store_true',
                        help='Exclude AE-MLP v9')
    parser.add_argument('--exclude-mkt-neutral', action='store_true',
                        help='Exclude AE-MLP mkt-neutral')
    parser.add_argument('--exclude-v12', action='store_true',
                        help='Exclude AE-MLP v12 (run V2 ensemble for comparison)')
    parser.add_argument('--exclude-catboost', action='store_true',
                        help='Exclude CatBoost')

    # Ensemble parameters
    parser.add_argument('--ensemble-method', type=str, default='zscore_weighted',
                        choices=['mean', 'weighted', 'rank_mean', 'zscore_mean', 'zscore_weighted'],
                        help='Ensemble method (default: zscore_weighted)')

    # Weight learning parameters
    parser.add_argument('--learn-weights', action='store_true', default=True,
                        help='Learn optimal weights from validation set (default: True)')
    parser.add_argument('--no-learn-weights', dest='learn_weights', action='store_false',
                        help='Use equal weights instead of learning')
    parser.add_argument('--weight-method', type=str, default='grid_search',
                        choices=['grid_search', 'grid_search_icir', 'regression', 'ridge', 'equal'],
                        help='Weight learning method (default: grid_search)')
    parser.add_argument('--min-weight', type=float, default=0.05,
                        help='Minimum weight for each model (default: 0.05)')
    parser.add_argument('--diversity-bonus', type=float, default=0.05,
                        help='Bonus for balanced weights (default: 0.05)')

    # Data parameters
    parser.add_argument('--nday', type=int, default=5,
                        help='Prediction horizon (default: 5)')
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'],
                        help='Stock pool (default: sp500)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device (-1 for CPU)')

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
                        choices=['topk', 'dynamic_risk', 'vol_stoploss'],
                        help='Trading strategy (default: topk)')

    args = parser.parse_args()

    # Build active model list
    active_models = {}
    exclude_map = {
        'AE-MLP-v7': args.exclude_v7,
        'AE-MLP-v9': args.exclude_v9,
        'AE-MLP-mkt-neutral': args.exclude_mkt_neutral,
        'AE-MLP-v12': args.exclude_v12,
        'CatBoost': args.exclude_catboost,
    }
    for name, config in MODEL_CONFIGS.items():
        if not exclude_map.get(name, False):
            active_models[name] = config.copy()

    # Apply path overrides
    path_overrides = {
        'AE-MLP-v7': args.ae_v7_model,
        'AE-MLP-v9': args.ae_v9_model,
        'AE-MLP-mkt-neutral': args.ae_mkt_model,
        'AE-MLP-v12': args.ae_v12_model,
        'CatBoost': args.cb_model,
    }
    for name, path in path_overrides.items():
        if path and name in active_models:
            active_models[name]['model_path'] = Path(path)

    if len(active_models) < 2:
        print("Error: Need at least 2 models for ensemble")
        sys.exit(1)

    # Use FINAL_TEST time splits
    time_splits = {
        'train_start': FINAL_TEST['train_start'],
        'train_end': FINAL_TEST['train_end'],
        'valid_start': FINAL_TEST['valid_start'],
        'valid_end': FINAL_TEST['valid_end'],
        'test_start': FINAL_TEST['test_start'],
        'test_end': FINAL_TEST['test_end'],
    }

    n_models = len(active_models)
    model_names = list(active_models.keys())

    print("=" * 70)
    print(f"AE-MLP + CatBoost Ensemble V3: {n_models} Models")
    print("=" * 70)
    for name, config in active_models.items():
        print(f"  {name:<22s}: {config['model_path'].name} (handler: {config['handler']})")
    print(f"Stock Pool:      {args.stock_pool}")
    print(f"Prediction Horizon: {args.nday} days")
    print(f"Ensemble Method: {args.ensemble_method}")
    print(f"Learn Weights:   {args.learn_weights} (method: {args.weight_method})")
    print(f"Test Period:     {time_splits['test_start']} to {time_splits['test_end']}")
    print("=" * 70)

    # Check model files exist
    for name, config in active_models.items():
        path = config['model_path']
        if not path.exists():
            print(f"Error: {name} model not found: {path}")
            sys.exit(1)

    # Initialize Qlib
    print("\n[1] Initializing Qlib...")
    qlib.init(
        provider_uri=str(QLIB_DATA_PATH),
        region=REG_US,
        custom_ops=TALIB_OPS,
        kernels=1,
        joblib_backend=None,
    )
    print("    Qlib initialized with TA-Lib support")

    # Get symbols
    symbols = STOCK_POOLS[args.stock_pool]
    print(f"\n[2] Using stock pool: {args.stock_pool} ({len(symbols)} stocks)")

    # Create datasets for each model
    include_valid = args.learn_weights
    data_start = time_splits['valid_start'] if include_valid else time_splits['test_start']
    print(f"\n[3] Creating datasets (from {data_start})...")

    datasets = {}
    for name, config in active_models.items():
        print(f"    Creating {config['handler']} dataset for {name}...")
        handler = create_data_handler(config['handler'], symbols, time_splits, args.nday, include_valid)
        datasets[name] = create_dataset(handler, time_splits, include_valid)

    # Load models
    print("\n[4] Loading models...")
    models = {}
    for name, config in active_models.items():
        if config['type'] == 'ae_mlp':
            models[name] = load_ae_mlp_model(config['model_path'])
        elif config['type'] == 'catboost':
            models[name] = load_catboost_model(config['model_path'])

    # Generate test predictions
    print("\n[5] Generating test predictions...")
    preds = {}
    for name, config in active_models.items():
        print(f"    {name} predictions...")
        if config['type'] == 'ae_mlp':
            pred = predict_with_ae_mlp(models[name], datasets[name])
        elif config['type'] == 'catboost':
            pred = predict_with_catboost(models[name], datasets[name])
        preds[name] = pred
        print(f"      Shape: {len(pred)}, Range: [{pred.min():.4f}, {pred.max():.4f}]")

    # Calculate pairwise correlations
    print("\n[6] Calculating pairwise correlations...")
    corr_matrix = calculate_pairwise_correlations(preds)

    print("\n    Prediction Correlation Matrix:")
    print("    " + "=" * (15 * n_models + 5))
    print(corr_matrix.to_string())
    print("    " + "=" * (15 * n_models + 5))

    # Learn optimal weights
    weights = None
    if args.learn_weights:
        print(f"\n[7] Learning optimal weights from validation set...")
        print(f"    Method: {args.weight_method}")
        print(f"    Min weight: {args.min_weight}")

        # Generate validation predictions
        print("    Generating validation predictions...")
        val_preds = {}
        for name, config in active_models.items():
            if config['type'] == 'ae_mlp':
                val_pred = predict_with_ae_mlp(models[name], datasets[name], segment="valid")
            elif config['type'] == 'catboost':
                val_pred = predict_with_catboost(models[name], datasets[name], segment="valid")
            val_preds[name] = val_pred
            print(f"      {name} valid: {len(val_pred)} samples")

        # Get validation labels from a raw-return handler (not mkt-neutral)
        label_source = None
        for name in model_names:
            handler_name = active_models[name]['handler']
            if 'mkt-neutral' not in handler_name and 'v12' not in handler_name:
                label_source = name
                break
        if label_source is None:
            label_source = model_names[0]

        val_label = datasets[label_source].prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(val_label, pd.DataFrame):
            val_label = val_label.iloc[:, 0]

        # Learn weights
        weights, learn_info = learn_optimal_weights_multi(
            val_preds, val_label,
            method=args.weight_method,
            min_weight=args.min_weight,
            diversity_bonus=args.diversity_bonus
        )

        print(f"\n    Learned Weights:")
        print(f"    " + "-" * 50)
        for name, w in weights.items():
            print(f"    {name:<22s}: {w:>10.4f}")
        print(f"    " + "-" * 50)

        for k, v in learn_info.items():
            if k != 'method':
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    else:
        # Equal weights
        weights = {name: 1.0 / n_models for name in model_names}
        print(f"\n[7] Using equal weights: {1.0/n_models:.3f} each")

    # Ensemble predictions
    print(f"\n[8] Ensembling predictions ({args.ensemble_method})...")
    pred_ensemble = ensemble_predictions_multi(preds, args.ensemble_method, weights)
    print(f"    Ensemble shape: {len(pred_ensemble)}, Range: [{pred_ensemble.min():.4f}, {pred_ensemble.max():.4f}]")

    # Get labels from raw-return handler for evaluation
    print(f"\n[9] Calculating IC metrics...")
    label_source = None
    for name in model_names:
        handler_name = active_models[name]['handler']
        if 'mkt-neutral' not in handler_name and 'v12' not in handler_name:
            label_source = name
            break
    if label_source is None:
        label_source = model_names[0]

    test_label = datasets[label_source].prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(test_label, pd.DataFrame):
        label = test_label.iloc[:, 0]
    else:
        label = test_label

    # Calculate IC for each model and ensemble
    ic_results = {}
    for name in model_names:
        ic, std, icir, _ = compute_ic(preds[name], label)
        ic_results[name] = {'ic': ic, 'std': std, 'icir': icir}

    ens_ic, ens_std, ens_icir, _ = compute_ic(pred_ensemble, label)

    # Print results table
    col_width = max(len(name) for name in model_names) + 2
    print(f"\n    +{'=' * (col_width + 48)}+")
    print(f"    |{' ' * 8}Information Coefficient (IC) Comparison{' ' * (col_width + 1)}|")
    print(f"    +{'=' * (col_width + 48)}+")
    print(f"    |  {'Model':<{col_width}s} | {'Weight':>8s} | {'Mean IC':>10s} | {'IC Std':>10s} | {'ICIR':>10s} |")
    print(f"    +{'-' * (col_width + 48)}+")
    for name in model_names:
        r = ic_results[name]
        w = weights[name]
        print(f"    |  {name:<{col_width}s} | {w:>8.3f} | {r['ic']:>10.4f} | {r['std']:>10.4f} | {r['icir']:>10.4f} |")
    print(f"    +{'-' * (col_width + 48)}+")
    print(f"    |  {'ENSEMBLE':<{col_width}s} | {'1.000':>8s} | {ens_ic:>10.4f} | {ens_std:>10.4f} | {ens_icir:>10.4f} |")
    print(f"    +{'=' * (col_width + 48)}+")

    # Calculate improvement
    best_single_ic = max(r['ic'] for r in ic_results.values())
    best_single_icir = max(r['icir'] for r in ic_results.values())
    best_model = max(ic_results, key=lambda k: ic_results[k]['ic'])

    if best_single_ic != 0:
        ic_improvement = (ens_ic - best_single_ic) / abs(best_single_ic) * 100
    else:
        ic_improvement = 0

    if best_single_icir != 0:
        icir_improvement = (ens_icir - best_single_icir) / abs(best_single_icir) * 100
    else:
        icir_improvement = 0

    print(f"\n    Ensemble Performance vs Best Single Model ({best_model}):")
    print(f"    IC improvement:   {ic_improvement:>+.2f}%")
    print(f"    ICIR improvement: {icir_improvement:>+.2f}%")

    # Summary
    print("\n" + "=" * 70)
    print(f"ENSEMBLE V3 COMPLETE ({n_models} models)")
    print("=" * 70)
    print("Weights:")
    for name in model_names:
        r = ic_results[name]
        w = weights[name]
        print(f"  {name:<22s}: weight={w:.3f}, IC={r['ic']:.4f}, ICIR={r['icir']:.4f}")
    print(f"  {'ENSEMBLE':<22s}: {'':>12s} IC={ens_ic:.4f}, ICIR={ens_icir:.4f}")
    print("=" * 70)

    # Run backtest if requested
    if args.backtest:
        if not hasattr(args, 'handler'):
            args.handler = "ensemble_v3"
        run_ensemble_backtest(pred_ensemble, args, time_splits)

        print("\n" + "=" * 70)
        print("ENSEMBLE V3 BACKTEST COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    main()
