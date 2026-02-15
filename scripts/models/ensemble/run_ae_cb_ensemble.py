"""
AE-MLP + CatBoost Ensemble: 2x AE-MLP + 1 CatBoost

Load pre-trained AE-MLP (v7, v9) and CatBoost models, generate predictions on test set,
learn optimal ensemble weights, and compute IC.

Usage:
    # Basic ensemble with auto-learned weights
    python scripts/models/ensemble/run_ae_cb_ensemble.py

    # With specific ensemble method
    python scripts/models/ensemble/run_ae_cb_ensemble.py --ensemble-method zscore_weighted

    # With backtest
    python scripts/models/ensemble/run_ae_cb_ensemble.py --backtest --topk 10

    # Custom model paths
    python scripts/models/ensemble/run_ae_cb_ensemble.py \
        --ae-model my_models/ae_mlp_v7.keras \
        --ae2-model my_models/ae_mlp_v9.keras \
        --cb-model my_models/catboost.cbm
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
import pandas as pd

import qlib
from qlib.constant import REG_US
from qlib.data.dataset.handler import DataHandlerLP

from utils.talib_ops import TALIB_OPS
from data.stock_pools import STOCK_POOLS

from models.common import (
    QLIB_DATA_PATH,
    MODEL_SAVE_PATH,
    FINAL_TEST,
)
from models.common.ensemble import (
    load_ae_mlp_model,
    load_model_meta,
    create_ensemble_data_handler,
    create_ensemble_dataset,
    predict_with_ae_mlp,
    predict_with_catboost,
    calculate_pairwise_correlations,
    learn_optimal_weights,
    ensemble_predictions,
    compute_ic,
    run_ensemble_backtest,
)
# Re-export for backward compatibility (V2/V3 import from here)
from models.common.training import load_catboost_model


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='AE-MLP + CatBoost Ensemble: 2x AE-MLP + 1 CatBoost',
    )

    # Model paths
    parser.add_argument('--ae-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'ae_mlp_cv_alpha158-enhanced-v7_sp500_5d_best.keras'),
                        help='AE-MLP v7 model path (.keras)')
    parser.add_argument('--ae2-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'ae_mlp_cv_alpha158-enhanced-v9_test_5d_best.keras'),
                        help='AE-MLP v9 model path (.keras)')
    parser.add_argument('--cb-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'catboost_cv_catboost-v1_test_5d_20260129_105915_best.cbm'),
                        help='CatBoost model path (.cbm)')

    # Handler configuration
    parser.add_argument('--ae-handler', type=str, default='alpha158-enhanced-v7',
                        help='Handler for AE-MLP v7 model')
    parser.add_argument('--ae2-handler', type=str, default='alpha158-enhanced-v9',
                        help='Handler for AE-MLP v9 model')
    parser.add_argument('--cb-handler', type=str, default='catboost-v1',
                        help='Handler for CatBoost model')

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
    parser.add_argument('--min-weight', type=float, default=0.1,
                        help='Minimum weight for each model (default: 0.1)')
    parser.add_argument('--diversity-bonus', type=float, default=0.05,
                        help='Bonus for balanced weights (default: 0.05)')

    # Manual weights (if not learning)
    parser.add_argument('--ae-weight', type=float, default=None,
                        help='Manual AE-MLP v7 weight')
    parser.add_argument('--ae2-weight', type=float, default=None,
                        help='Manual AE-MLP v9 weight')
    parser.add_argument('--cb-weight', type=float, default=None,
                        help='Manual CatBoost weight')

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

    # Use FINAL_TEST time splits
    time_splits = {
        'train_start': FINAL_TEST['train_start'],
        'train_end': FINAL_TEST['train_end'],
        'valid_start': FINAL_TEST['valid_start'],
        'valid_end': FINAL_TEST['valid_end'],
        'test_start': FINAL_TEST['test_start'],
        'test_end': FINAL_TEST['test_end'],
    }

    print("=" * 70)
    print("AE-MLP + CatBoost Ensemble: 2x AE-MLP + 1 CatBoost")
    print("=" * 70)
    print(f"AE-MLP v7 Model: {args.ae_model}")
    print(f"AE-MLP v9 Model: {args.ae2_model}")
    print(f"CatBoost Model:  {args.cb_model}")
    print(f"Handlers:        AE-MLP-v7={args.ae_handler}, AE-MLP-v9={args.ae2_handler}, CatBoost={args.cb_handler}")
    print(f"Stock Pool:      {args.stock_pool}")
    print(f"Prediction Horizon: {args.nday} days")
    print(f"Ensemble Method: {args.ensemble_method}")
    print(f"Learn Weights:   {args.learn_weights} (method: {args.weight_method})")
    print(f"Test Period:     {time_splits['test_start']} to {time_splits['test_end']}")
    print("=" * 70)

    # Check model files exist
    ae_path = Path(args.ae_model)
    ae2_path = Path(args.ae2_model)
    cb_path = Path(args.cb_model)

    for path, name in [(ae_path, 'AE-MLP-v7'), (ae2_path, 'AE-MLP-v9'), (cb_path, 'CatBoost')]:
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
    # Only load validation data if learning weights, otherwise just test data
    include_valid = args.learn_weights
    data_start = time_splits['valid_start'] if include_valid else time_splits['test_start']
    print(f"\n[3] Creating datasets (from {data_start})...")

    print(f"    Creating {args.ae_handler} dataset for AE-MLP v7...")
    ae_handler = create_ensemble_data_handler(args.ae_handler, symbols, time_splits, args.nday, include_valid)
    ae_dataset = create_ensemble_dataset(ae_handler, time_splits, include_valid)

    print(f"    Creating {args.ae2_handler} dataset for AE-MLP v9...")
    ae2_handler = create_ensemble_data_handler(args.ae2_handler, symbols, time_splits, args.nday, include_valid)
    ae2_dataset = create_ensemble_dataset(ae2_handler, time_splits, include_valid)

    print(f"    Creating {args.cb_handler} dataset for CatBoost...")
    cb_handler = create_ensemble_data_handler(args.cb_handler, symbols, time_splits, args.nday, include_valid)
    cb_dataset = create_ensemble_dataset(cb_handler, time_splits, include_valid)

    # Load models
    print("\n[4] Loading models...")
    ae_model = load_ae_mlp_model(ae_path)
    ae2_model = load_ae_mlp_model(ae2_path)
    cb_model = load_catboost_model(cb_path)

    # Generate test predictions
    print("\n[5] Generating test predictions...")

    print("    AE-MLP v7 predictions...")
    pred_ae = predict_with_ae_mlp(ae_model, ae_dataset)
    print(f"      Shape: {len(pred_ae)}, Range: [{pred_ae.min():.4f}, {pred_ae.max():.4f}]")

    print("    AE-MLP v9 predictions...")
    pred_ae2 = predict_with_ae_mlp(ae2_model, ae2_dataset)
    print(f"      Shape: {len(pred_ae2)}, Range: [{pred_ae2.min():.4f}, {pred_ae2.max():.4f}]")

    print("    CatBoost predictions...")
    pred_cb = predict_with_catboost(cb_model, cb_dataset)
    print(f"      Shape: {len(pred_cb)}, Range: [{pred_cb.min():.4f}, {pred_cb.max():.4f}]")

    # Store predictions in dict
    preds = {
        'AE-MLP-v7': pred_ae,
        'AE-MLP-v9': pred_ae2,
        'CatBoost': pred_cb,
    }

    # Calculate pairwise correlations
    print("\n[6] Calculating pairwise correlations...")
    corr_matrix = calculate_pairwise_correlations(preds)

    print("\n    Prediction Correlation Matrix:")
    print("    " + "=" * 50)
    print(corr_matrix.to_string())
    print("    " + "=" * 50)

    # Learn optimal weights
    weights = None
    if args.learn_weights:
        print(f"\n[7] Learning optimal weights from validation set...")
        print(f"    Method: {args.weight_method}")
        print(f"    Min weight: {args.min_weight}")

        # Generate validation predictions
        print("    Generating validation predictions...")
        val_pred_ae = predict_with_ae_mlp(ae_model, ae_dataset, segment="valid")
        val_pred_ae2 = predict_with_ae_mlp(ae2_model, ae2_dataset, segment="valid")
        val_pred_cb = predict_with_catboost(cb_model, cb_dataset, segment="valid")

        val_preds = {
            'AE-MLP-v7': val_pred_ae,
            'AE-MLP-v9': val_pred_ae2,
            'CatBoost': val_pred_cb,
        }

        print(f"      AE-MLP v7 valid: {len(val_pred_ae)} samples")
        print(f"      AE-MLP v9 valid: {len(val_pred_ae2)} samples")
        print(f"      CatBoost valid: {len(val_pred_cb)} samples")

        # Get validation labels
        val_label = ae_dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(val_label, pd.DataFrame):
            val_label = val_label.iloc[:, 0]

        # Learn weights
        weights, learn_info = learn_optimal_weights(
            val_preds, val_label,
            method=args.weight_method,
            min_weight=args.min_weight,
            diversity_bonus=args.diversity_bonus
        )

        print(f"\n    Learned Weights:")
        print(f"    " + "-" * 50)
        for name, w in weights.items():
            print(f"    {name:<15s}: {w:>10.4f}")
        print(f"    " + "-" * 50)

        for k, v in learn_info.items():
            if k != 'method':
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    elif args.ae_weight is not None and args.ae2_weight is not None and args.cb_weight is not None:
        # Use manual weights
        total = args.ae_weight + args.ae2_weight + args.cb_weight
        weights = {
            'AE-MLP-v7': args.ae_weight / total,
            'AE-MLP-v9': args.ae2_weight / total,
            'CatBoost': args.cb_weight / total,
        }
        print(f"\n[7] Using manual weights: AE-MLP-v7={weights['AE-MLP-v7']:.3f}, "
              f"AE-MLP-v9={weights['AE-MLP-v9']:.3f}, CatBoost={weights['CatBoost']:.3f}")
    else:
        # Equal weights
        weights = {'AE-MLP-v7': 1/3, 'AE-MLP-v9': 1/3, 'CatBoost': 1/3}
        print(f"\n[7] Using equal weights: 1/3 each")

    # Ensemble predictions
    step_num = 8
    print(f"\n[{step_num}] Ensembling predictions ({args.ensemble_method})...")
    pred_ensemble = ensemble_predictions(preds, args.ensemble_method, weights)
    print(f"    Ensemble shape: {len(pred_ensemble)}, Range: [{pred_ensemble.min():.4f}, {pred_ensemble.max():.4f}]")

    # Get labels
    step_num += 1
    print(f"\n[{step_num}] Calculating IC metrics...")
    test_label = ae_dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(test_label, pd.DataFrame):
        label = test_label.iloc[:, 0]
    else:
        label = test_label

    # Calculate IC for each model and ensemble
    ae_ic, ae_std, ae_icir, _ = compute_ic(pred_ae, label)
    ae2_ic, ae2_std, ae2_icir, _ = compute_ic(pred_ae2, label)
    cb_ic, cb_std, cb_icir, _ = compute_ic(pred_cb, label)
    ens_ic, ens_std, ens_icir, _ = compute_ic(pred_ensemble, label)

    print("\n    +" + "=" * 68 + "+")
    print("    |" + " " * 12 + "Information Coefficient (IC) Comparison" + " " * 16 + "|")
    print("    +" + "=" * 68 + "+")
    print(f"    |  {'Model':<15s} | {'Weight':>8s} | {'Mean IC':>10s} | {'IC Std':>10s} | {'ICIR':>10s} |")
    print("    +" + "-" * 68 + "+")
    print(f"    |  {'AE-MLP-v7':<15s} | {weights['AE-MLP-v7']:>8.3f} | {ae_ic:>10.4f} | {ae_std:>10.4f} | {ae_icir:>10.4f} |")
    print(f"    |  {'AE-MLP-v9':<15s} | {weights['AE-MLP-v9']:>8.3f} | {ae2_ic:>10.4f} | {ae2_std:>10.4f} | {ae2_icir:>10.4f} |")
    print(f"    |  {'CatBoost':<15s} | {weights['CatBoost']:>8.3f} | {cb_ic:>10.4f} | {cb_std:>10.4f} | {cb_icir:>10.4f} |")
    print("    +" + "-" * 68 + "+")
    print(f"    |  {'ENSEMBLE':<15s} | {'1.000':>8s} | {ens_ic:>10.4f} | {ens_std:>10.4f} | {ens_icir:>10.4f} |")
    print("    +" + "=" * 68 + "+")

    # Calculate improvement
    best_single_ic = max(ae_ic, ae2_ic, cb_ic)
    best_single_icir = max(ae_icir, ae2_icir, cb_icir)
    ic_values = {'AE-MLP-v7': ae_ic, 'AE-MLP-v9': ae2_ic, 'CatBoost': cb_ic}
    best_model = max(ic_values, key=ic_values.get)

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
    print("AE-MLP + CATBOOST ENSEMBLE COMPLETE")
    print("=" * 70)
    print(f"Weights: AE-MLP-v7={weights['AE-MLP-v7']:.3f}, AE-MLP-v9={weights['AE-MLP-v9']:.3f}, CatBoost={weights['CatBoost']:.3f}")
    print(f"AE-MLP v7 IC: {ae_ic:.4f} (ICIR: {ae_icir:.4f})")
    print(f"AE-MLP v9 IC: {ae2_ic:.4f} (ICIR: {ae2_icir:.4f})")
    print(f"CatBoost IC:  {cb_ic:.4f} (ICIR: {cb_icir:.4f})")
    print(f"Ensemble IC:  {ens_ic:.4f} (ICIR: {ens_icir:.4f})")
    print("=" * 70)

    # Run backtest if requested
    if args.backtest:
        run_ensemble_backtest(pred_ensemble, args, time_splits,
                              model_name="AE_CB_Ensemble")

        print("\n" + "=" * 70)
        print("ENSEMBLE BACKTEST COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    main()
