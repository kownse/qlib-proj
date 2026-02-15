"""
AE-MLP + CatBoost Ensemble

Load pre-trained AE-MLP and CatBoost models, generate predictions on test set,
calculate correlation between outputs, ensemble them, and compute IC.

Usage:
    python scripts/models/ensemble/run_ae_catboost_ensemble.py
    python scripts/models/ensemble/run_ae_catboost_ensemble.py --ensemble-method rank_mean
    python scripts/models/ensemble/run_ae_catboost_ensemble.py --ae-weight 0.6 --cb-weight 0.4
    python scripts/models/ensemble/run_ae_catboost_ensemble.py --backtest --topk 10 --n-drop 2
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
from data.stock_pools import STOCK_POOLS

from models.common import (
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
    calculate_pairwise_correlations,
    compute_ic,
    ensemble_predictions,
    learn_optimal_weights,
    run_ensemble_backtest,
)
from models.common.training import load_catboost_model


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


def run_cv_evaluation(ae_model, cb_model, args, symbols):
    """
    在 CV folds 上评估 ensemble IC。

    对每个 fold:
    1. 创建两个 handler 的数据集 (AE-MLP handler + CatBoost handler)
    2. 分别生成验证集预测
    3. Ensemble 并计算 IC
    4. 同时在 2025 测试集上计算 IC
    """
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION EVALUATION (Ensemble)")
    print("=" * 70)
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: valid {fold['valid_start']}~{fold['valid_end']}")
    print(f"Test Set: {FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']} (2025)")
    print(f"Ensemble Method: {args.ensemble_method}")
    print("=" * 70)

    # 准备 2025 测试集
    print("\n[CV] Preparing 2025 test data...")
    test_time = {
        'train_start': FINAL_TEST['train_start'],
        'train_end': FINAL_TEST['train_end'],
        'valid_start': FINAL_TEST['valid_start'],
        'valid_end': FINAL_TEST['valid_end'],
        'test_start': FINAL_TEST['test_start'],
        'test_end': FINAL_TEST['test_end'],
    }
    ae_test_handler = create_ensemble_data_handler(args.ae_handler, symbols, test_time, args.nday)
    ae_test_dataset = create_ensemble_dataset(ae_test_handler, test_time)
    cb_test_handler = create_ensemble_data_handler(args.cb_handler, symbols, test_time, args.nday)
    cb_test_dataset = create_ensemble_dataset(cb_test_handler, test_time)

    # 测试集预测 (只需做一次)
    test_pred_ae = predict_with_ae_mlp(ae_model, ae_test_dataset, segment="test")
    test_pred_cb = predict_catboost_raw(cb_model, cb_test_dataset, "test")
    weights_dict = {'AE-MLP': args.ae_weight, 'CatBoost': args.cb_weight} \
        if args.ensemble_method in ['weighted', 'zscore_weighted'] else None
    test_pred_ens = ensemble_predictions(
        {'AE-MLP': test_pred_ae, 'CatBoost': test_pred_cb},
        args.ensemble_method, weights_dict
    )

    test_label_df = ae_test_dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)
    test_label = test_label_df.iloc[:, 0] if isinstance(test_label_df, pd.DataFrame) else test_label_df

    test_ens_ic, _, test_ens_icir, _ = compute_ic(test_pred_ens, test_label)
    test_ae_ic, _, _, _ = compute_ic(test_pred_ae, test_label)
    test_cb_ic, _, _, _ = compute_ic(test_pred_cb, test_label)

    print(f"    Test (2025): AE-MLP IC={test_ae_ic:.4f}, CatBoost IC={test_cb_ic:.4f}, Ensemble IC={test_ens_ic:.4f}")

    fold_results = []

    for fold in CV_FOLDS:
        print(f"\n[CV] Evaluating on {fold['name']}...")

        fold_time = {
            'train_start': fold['train_start'],
            'train_end': fold['train_end'],
            'valid_start': fold['valid_start'],
            'valid_end': fold['valid_end'],
            'test_start': fold['valid_start'],  # no test segment needed
            'test_end': fold['valid_end'],
        }

        # AE-MLP dataset (include_valid=True to get valid segment)
        ae_handler = create_ensemble_data_handler(args.ae_handler, symbols, fold_time, args.nday, include_valid=True)
        ae_dataset = create_ensemble_dataset(ae_handler, fold_time, include_valid=True)

        # CatBoost dataset (include_valid=True to get valid segment)
        cb_handler = create_ensemble_data_handler(args.cb_handler, symbols, fold_time, args.nday, include_valid=True)
        cb_dataset = create_ensemble_dataset(cb_handler, fold_time, include_valid=True)

        # 验证集预测
        val_pred_ae = predict_with_ae_mlp(ae_model, ae_dataset, segment="valid")
        val_pred_cb = predict_catboost_raw(cb_model, cb_dataset, "valid")
        val_pred_ens = ensemble_predictions(
            {'AE-MLP': val_pred_ae, 'CatBoost': val_pred_cb},
            args.ensemble_method, weights_dict
        )

        # 验证集标签
        val_label_df = ae_dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
        val_label = val_label_df.iloc[:, 0] if isinstance(val_label_df, pd.DataFrame) else val_label_df

        # IC
        ae_ic, _, _, _ = compute_ic(val_pred_ae, val_label)
        cb_ic, _, _, _ = compute_ic(val_pred_cb, val_label)
        ens_ic, _, ens_icir, _ = compute_ic(val_pred_ens, val_label)

        fold_results.append({
            'name': fold['name'],
            'ae_ic': ae_ic,
            'cb_ic': cb_ic,
            'ens_ic': ens_ic,
            'ens_icir': ens_icir,
            'test_ens_ic': test_ens_ic,
        })

        print(f"    {fold['name']}: AE-MLP={ae_ic:.4f}, CatBoost={cb_ic:.4f}, Ensemble={ens_ic:.4f}")

    # 汇总
    ens_ics = [r['ens_ic'] for r in fold_results]
    ae_ics = [r['ae_ic'] for r in fold_results]
    cb_ics = [r['cb_ic'] for r in fold_results]

    print("\n" + "=" * 70)
    print("CV EVALUATION COMPLETE (Ensemble)")
    print("=" * 70)
    print(f"{'':25s} {'AE-MLP':>10s} {'CatBoost':>10s} {'Ensemble':>10s}")
    print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for r in fold_results:
        print(f"{r['name']:<25s} {r['ae_ic']:>10.4f} {r['cb_ic']:>10.4f} {r['ens_ic']:>10.4f}")
    print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    print(f"{'Valid Mean IC':<25s} {np.mean(ae_ics):>10.4f} {np.mean(cb_ics):>10.4f} {np.mean(ens_ics):>10.4f}")
    print(f"{'Valid IC Std':<25s} {np.std(ae_ics):>10.4f} {np.std(cb_ics):>10.4f} {np.std(ens_ics):>10.4f}")
    print(f"{'Test IC (2025)':<25s} {test_ae_ic:>10.4f} {test_cb_ic:>10.4f} {test_ens_ic:>10.4f}")
    print("=" * 70)

    return fold_results


def main():
    parser = argparse.ArgumentParser(
        description='AE-MLP + CatBoost Ensemble',
    )

    # Model paths
    parser.add_argument('--ae-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'ae_mlp_cv_alpha158-enhanced-v7_sp500_5d_best.keras'),
                        help='AE-MLP model path (.keras)')
    parser.add_argument('--cb-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'catboost_cv_catboost-v1_sp500_5d_20260129_141353_best.cbm'),
                        help='CatBoost model path (.cbm)')

    # Handler configuration (override metadata)
    parser.add_argument('--ae-handler', type=str, default='alpha158-enhanced-v7',
                        help='Handler for AE-MLP model')
    parser.add_argument('--cb-handler', type=str, default='catboost-v1',
                        help='Handler for CatBoost model')

    # Ensemble parameters
    parser.add_argument('--ensemble-method', type=str, default='zscore_mean',
                        choices=['mean', 'weighted', 'rank_mean', 'zscore_mean', 'zscore_weighted'],
                        help='Ensemble method (default: zscore_mean)')
    parser.add_argument('--ae-weight', type=float, default=0.5,
                        help='AE-MLP weight for weighted ensemble (default: 0.5)')
    parser.add_argument('--cb-weight', type=float, default=0.5,
                        help='CatBoost weight for weighted ensemble (default: 0.5)')

    # Stacking parameters
    parser.add_argument('--stacking', action='store_true',
                        help='Learn optimal weights from validation set (Stacking)')
    parser.add_argument('--stacking-method', type=str, default='grid_search',
                        choices=['grid_search', 'grid_search_icir', 'regression', 'ridge', 'equal'],
                        help='Stacking weight learning method (default: grid_search)')
    parser.add_argument('--min-weight', type=float, default=0.1,
                        help='Minimum weight for each model (default: 0.1, prevents extreme weights)')
    parser.add_argument('--diversity-bonus', type=float, default=0.1,
                        help='Bonus for balanced weights (default: 0.1, set to 0 to disable)')

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
                        choices=['topk', 'dynamic_risk', 'vol_stoploss'],
                        help='Trading strategy (default: topk)')

    # Strategy parameters (for dynamic_risk and vol_stoploss)
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
    print("AE-MLP + CatBoost Ensemble")
    print("=" * 70)
    print(f"AE-MLP Model: {args.ae_model}")
    print(f"CatBoost Model: {args.cb_model}")
    print(f"AE-MLP Handler: {args.ae_handler}")
    print(f"CatBoost Handler: {args.cb_handler}")
    print(f"Stock Pool: {args.stock_pool}")
    print(f"Prediction Horizon: {args.nday} days")
    print(f"Ensemble Method: {args.ensemble_method}")
    if args.ensemble_method == 'weighted':
        print(f"Weights: AE-MLP={args.ae_weight}, CatBoost={args.cb_weight}")
    print(f"Test Period: {time_splits['test_start']} to {time_splits['test_end']}")
    print("=" * 70)

    # Check model files exist
    ae_path = Path(args.ae_model)
    cb_path = Path(args.cb_model)

    if not ae_path.exists():
        print(f"Error: AE-MLP model not found: {ae_path}")
        sys.exit(1)
    if not cb_path.exists():
        print(f"Error: CatBoost model not found: {cb_path}")
        sys.exit(1)

    # Load metadata
    print("\n[1] Loading model metadata...")
    ae_meta = load_model_meta(ae_path)
    cb_meta = load_model_meta(cb_path)

    if ae_meta:
        print(f"    AE-MLP metadata found: handler={ae_meta.get('handler', 'N/A')}, nday={ae_meta.get('nday', 'N/A')}")
        if 'handler' in ae_meta:
            args.ae_handler = ae_meta['handler']
    else:
        print(f"    AE-MLP metadata not found, using default handler: {args.ae_handler}")

    if cb_meta:
        print(f"    CatBoost metadata found: handler={cb_meta.get('handler', 'N/A')}, nday={cb_meta.get('nday', 'N/A')}")
        if 'handler' in cb_meta:
            args.cb_handler = cb_meta['handler']
    else:
        print(f"    CatBoost metadata not found, using default handler: {args.cb_handler}")

    # Initialize Qlib
    print("\n[2] Initializing Qlib...")
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
    print(f"\n[3] Using stock pool: {args.stock_pool} ({len(symbols)} stocks)")

    # Create datasets for each model (different handlers)
    print("\n[4] Creating datasets...")

    print(f"    Creating {args.ae_handler} dataset for AE-MLP...")
    ae_handler = create_ensemble_data_handler(args.ae_handler, symbols, time_splits, args.nday, include_valid=True)
    ae_dataset = create_ensemble_dataset(ae_handler, time_splits, include_valid=True)

    print(f"    Creating {args.cb_handler} dataset for CatBoost...")
    cb_handler = create_ensemble_data_handler(args.cb_handler, symbols, time_splits, args.nday, include_valid=True)
    cb_dataset = create_ensemble_dataset(cb_handler, time_splits, include_valid=True)

    # Load models
    print("\n[5] Loading models...")
    ae_model = load_ae_mlp_model(ae_path)
    cb_model = load_catboost_model(cb_path)

    # CV evaluation
    print("\n[6] Cross-validation evaluation...")
    run_cv_evaluation(ae_model, cb_model, args, symbols)

    # Generate predictions
    print("\n[7] Generating predictions...")

    print("    AE-MLP predictions...")
    pred_ae = predict_with_ae_mlp(ae_model, ae_dataset)
    print(f"      Shape: {len(pred_ae)}, Range: [{pred_ae.min():.4f}, {pred_ae.max():.4f}]")

    print("    CatBoost predictions...")
    pred_cb = predict_with_catboost(cb_model, cb_dataset)
    print(f"      Shape: {len(pred_cb)}, Range: [{pred_cb.min():.4f}, {pred_cb.max():.4f}]")

    # Compare prediction statistics
    print("\n    Prediction Statistics Comparison:")
    print("    " + "=" * 60)
    print(f"    {'Metric':<20s} | {'AE-MLP':>15s} | {'CatBoost':>15s} | {'Ratio':>10s}")
    print("    " + "-" * 60)

    ae_mean, cb_mean = pred_ae.mean(), pred_cb.mean()
    ae_std, cb_std = pred_ae.std(), pred_cb.std()
    ae_median, cb_median = pred_ae.median(), pred_cb.median()
    ae_abs_mean, cb_abs_mean = pred_ae.abs().mean(), pred_cb.abs().mean()

    # Calculate ratios (AE-MLP / CatBoost)
    mean_ratio = ae_mean / cb_mean if cb_mean != 0 else float('inf')
    std_ratio = ae_std / cb_std if cb_std != 0 else float('inf')
    abs_mean_ratio = ae_abs_mean / cb_abs_mean if cb_abs_mean != 0 else float('inf')

    print(f"    {'Mean':<20s} | {ae_mean:>15.6f} | {cb_mean:>15.6f} | {mean_ratio:>10.2f}x")
    print(f"    {'Std':<20s} | {ae_std:>15.6f} | {cb_std:>15.6f} | {std_ratio:>10.2f}x")
    print(f"    {'Median':<20s} | {ae_median:>15.6f} | {cb_median:>15.6f} | {'-':>10s}")
    print(f"    {'Abs Mean':<20s} | {ae_abs_mean:>15.6f} | {cb_abs_mean:>15.6f} | {abs_mean_ratio:>10.2f}x")
    print(f"    {'Min':<20s} | {pred_ae.min():>15.6f} | {pred_cb.min():>15.6f} | {'-':>10s}")
    print(f"    {'Max':<20s} | {pred_ae.max():>15.6f} | {pred_cb.max():>15.6f} | {'-':>10s}")
    print("    " + "=" * 60)

    # Warning if scales are very different
    if abs(std_ratio) > 10 or abs(std_ratio) < 0.1:
        print(f"\n    WARNING: Prediction scales differ significantly ({std_ratio:.1f}x)!")
        print(f"    Consider using 'rank_mean' ensemble method or normalizing predictions.")

    # Calculate correlation between predictions
    print("\n[8] Calculating correlation between model outputs...")
    preds_dict = {'AE-MLP': pred_ae, 'CatBoost': pred_cb}
    corr_matrix = calculate_pairwise_correlations(preds_dict)
    overall_corr = corr_matrix.loc['AE-MLP', 'CatBoost']

    print(f"\n    Prediction Correlation:")
    print(f"    " + "-" * 50)
    print(f"    Overall Correlation:     {overall_corr:>10.4f}")
    print(f"    " + "-" * 50)

    # Stacking: Learn optimal weights from validation set
    learned_weights = None
    if args.stacking:
        print(f"\n[9] Stacking: Learning optimal weights from validation set...")
        print(f"    Method: {args.stacking_method}")

        # Generate predictions on validation set
        print("    Generating validation predictions...")
        val_pred_ae = predict_with_ae_mlp(ae_model, ae_dataset, segment="valid")
        val_pred_cb = predict_with_catboost(cb_model, cb_dataset, segment="valid")
        print(f"      AE-MLP valid: {len(val_pred_ae)} samples")
        print(f"      CatBoost valid: {len(val_pred_cb)} samples")

        # Get validation labels
        val_label = ae_dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(val_label, pd.DataFrame):
            val_label = val_label.iloc[:, 0]

        # Learn optimal weights (new API: dict-based)
        val_preds_dict = {'AE-MLP': val_pred_ae, 'CatBoost': val_pred_cb}
        learned_weights, learn_info = learn_optimal_weights(
            val_preds_dict, val_label,
            method=args.stacking_method,
            min_weight=args.min_weight,
            diversity_bonus=args.diversity_bonus
        )

        print(f"\n    Learned Weights:")
        print(f"    " + "-" * 50)
        print(f"    AE-MLP weight:   {learned_weights['AE-MLP']:>10.4f}")
        print(f"    CatBoost weight: {learned_weights['CatBoost']:>10.4f}")
        print(f"    " + "-" * 50)

        # Print additional info from learning
        for k, v in learn_info.items():
            if k != 'method':
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

        # Override ensemble method to use learned weights
        args.ensemble_method = 'zscore_weighted'
        args.ae_weight = learned_weights['AE-MLP']
        args.cb_weight = learned_weights['CatBoost']
        print(f"\n    Using zscore_weighted ensemble with learned weights")

    # Ensemble predictions
    step_num = 10 if args.stacking else 9
    print(f"\n[{step_num}] Ensembling predictions ({args.ensemble_method})...")
    weights_dict = {'AE-MLP': args.ae_weight, 'CatBoost': args.cb_weight} \
        if args.ensemble_method in ['weighted', 'zscore_weighted'] else None
    pred_ensemble = ensemble_predictions(preds_dict, args.ensemble_method, weights_dict)
    print(f"    Ensemble shape: {len(pred_ensemble)}, Range: [{pred_ensemble.min():.4f}, {pred_ensemble.max():.4f}]")

    # Get labels (use AE-MLP dataset's label as reference)
    step_num += 1
    print(f"\n[{step_num}] Calculating IC metrics...")
    test_label = ae_dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(test_label, pd.DataFrame):
        label = test_label.iloc[:, 0]
    else:
        label = test_label

    # Calculate IC for each model and ensemble
    ae_ic, ae_std, ae_icir, ae_ic_series = compute_ic(pred_ae, label)
    cb_ic, cb_std, cb_icir, cb_ic_series = compute_ic(pred_cb, label)
    ens_ic, ens_std, ens_icir, ens_ic_series = compute_ic(pred_ensemble, label)

    print("\n    +" + "=" * 60 + "+")
    print("    |" + " " * 10 + "Information Coefficient (IC) Comparison" + " " * 10 + "|")
    print("    +" + "=" * 60 + "+")
    print(f"    |  {'Model':<15s} | {'Mean IC':>10s} | {'IC Std':>10s} | {'ICIR':>10s} |")
    print("    +" + "-" * 60 + "+")
    print(f"    |  {'AE-MLP':<15s} | {ae_ic:>10.4f} | {ae_std:>10.4f} | {ae_icir:>10.4f} |")
    print(f"    |  {'CatBoost':<15s} | {cb_ic:>10.4f} | {cb_std:>10.4f} | {cb_icir:>10.4f} |")
    print("    +" + "-" * 60 + "+")
    print(f"    |  {'Ensemble':<15s} | {ens_ic:>10.4f} | {ens_std:>10.4f} | {ens_icir:>10.4f} |")
    print("    +" + "=" * 60 + "+")

    # Calculate improvement
    best_single_ic = max(ae_ic, cb_ic)
    best_single_icir = max(ae_icir, cb_icir)

    if best_single_ic != 0:
        ic_improvement = (ens_ic - best_single_ic) / abs(best_single_ic) * 100
    else:
        ic_improvement = 0

    if best_single_icir != 0:
        icir_improvement = (ens_icir - best_single_icir) / abs(best_single_icir) * 100
    else:
        icir_improvement = 0

    print(f"\n    Ensemble Performance vs Best Single Model:")
    print(f"    IC improvement:   {ic_improvement:>+.2f}%")
    print(f"    ICIR improvement: {icir_improvement:>+.2f}%")

    # Summary
    print("\n" + "=" * 70)
    print("ENSEMBLE ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Prediction Correlation: {overall_corr:.4f}")
    if learned_weights:
        print(f"Stacking Weights: AE-MLP={learned_weights['AE-MLP']:.3f}, CatBoost={learned_weights['CatBoost']:.3f}")
    print(f"AE-MLP IC:     {ae_ic:.4f} (ICIR: {ae_icir:.4f})")
    print(f"CatBoost IC:   {cb_ic:.4f} (ICIR: {cb_icir:.4f})")
    print(f"Ensemble IC:   {ens_ic:.4f} (ICIR: {ens_icir:.4f})")
    print("=" * 70)

    # Run backtest if requested
    if args.backtest:
        run_ensemble_backtest(pred_ensemble, args, time_splits, model_name="AE_CatBoost_Ensemble")

        print("\n" + "=" * 70)
        print("ENSEMBLE BACKTEST COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    main()
