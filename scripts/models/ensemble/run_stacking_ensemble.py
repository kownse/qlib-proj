"""
Stacking Ensemble: Meta-Learner on Validation Predictions

用 2024 validation set 上各基础模型的 out-of-sample 预测训练 meta-learner，
然后应用到 2025 test set。对比 V2 的固定权重加权平均。

做法:
  1. 加载 4 个基础模型 (AE-MLP v7/v9/mkt-neutral + CatBoost)
  2. 各模型在 2024 validation set 上生成预测 → z-score 归一化
  3. 训练 meta-learner: X = [zscore(v7), zscore(v9), zscore(mkt), zscore(cb)], y = true_return
  4. 各模型在 2025 test set 上生成预测 → z-score 归一化
  5. Meta-learner 组合 test 预测 → 计算 IC/ICIR
  6. 对比 V2 zscore_weighted baseline

Meta-learners:
  - Ridge (默认, alpha 由 CV 选择)
  - Lasso
  - ElasticNet
  - OLS (无正则化)

Usage:
    python scripts/models/ensemble/run_stacking_ensemble.py
    python scripts/models/ensemble/run_stacking_ensemble.py --backtest
    python scripts/models/ensemble/run_stacking_ensemble.py --add-meta-features
"""

import os

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

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import argparse
import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from utils.talib_ops import TALIB_OPS
from data.stock_pools import STOCK_POOLS

from models.common import (
    PROJECT_ROOT,
    MODEL_SAVE_PATH,
    FINAL_TEST,
)

from models.ensemble.run_ae_cb_ensemble import (
    load_ae_mlp_model,
    load_catboost_model,
    create_data_handler,
    create_dataset,
    predict_with_ae_mlp,
    predict_with_catboost,
    calculate_pairwise_correlations,
    ensemble_predictions_multi,
    compute_ic,
    run_ensemble_backtest,
)


# ============================================================================
# Model Configuration (same as V2)
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
    'CatBoost': {
        'model_path': MODEL_SAVE_PATH / 'catboost_cv_catboost-v1_test_5d_20260129_105915_best.cbm',
        'handler': 'catboost-v1',
        'type': 'catboost',
    },
}


# ============================================================================
# Stacking Utils
# ============================================================================

def zscore_by_day(series):
    """Cross-sectional z-score normalization within each day."""
    mean = series.groupby(level='datetime').transform('mean')
    std = series.groupby(level='datetime').transform('std')
    return (series - mean) / (std + 1e-8)


def prepare_stacking_matrix(preds, model_names, add_meta_features=False):
    """
    Prepare feature matrix for meta-learner.

    Args:
        preds: dict of {model_name: prediction_series}
        model_names: ordered list of model names
        add_meta_features: if True, add disagreement and mean features

    Returns:
        X: DataFrame with z-scored predictions as columns
        common_idx: aligned MultiIndex
    """
    # Find common index
    common_idx = preds[model_names[0]].index
    for name in model_names[1:]:
        common_idx = common_idx.intersection(preds[name].index)

    # Z-score normalize and align
    X = pd.DataFrame(index=common_idx)
    for name in model_names:
        X[name] = zscore_by_day(preds[name].loc[common_idx])

    if add_meta_features:
        # Model disagreement: std of z-scored predictions per stock-day
        X['disagreement'] = X[model_names].std(axis=1)
        # Mean prediction
        X['mean_pred'] = X[model_names].mean(axis=1)

    # Clean
    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    return X, common_idx


def train_meta_learners(X_val, y_val):
    """
    Train multiple meta-learners and return them with CV scores.

    Returns:
        dict of {name: {'model': fitted_model, 'cv_score': float, 'coefs': array}}
    """
    from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LinearRegression

    results = {}

    # 1. Ridge with built-in CV for alpha
    ridge = RidgeCV(
        alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
        fit_intercept=True,
        cv=5,
    )
    ridge.fit(X_val, y_val)
    results['Ridge'] = {
        'model': ridge,
        'alpha': ridge.alpha_,
        'coefs': ridge.coef_,
        'intercept': ridge.intercept_,
    }

    # 2. Lasso with built-in CV
    lasso = LassoCV(
        alphas=[0.0001, 0.001, 0.01, 0.1],
        fit_intercept=True,
        cv=5,
        max_iter=10000,
    )
    lasso.fit(X_val, y_val)
    results['Lasso'] = {
        'model': lasso,
        'alpha': lasso.alpha_,
        'coefs': lasso.coef_,
        'intercept': lasso.intercept_,
    }

    # 3. ElasticNet with built-in CV
    enet = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9],
        alphas=[0.0001, 0.001, 0.01, 0.1],
        fit_intercept=True,
        cv=5,
        max_iter=10000,
    )
    enet.fit(X_val, y_val)
    results['ElasticNet'] = {
        'model': enet,
        'alpha': enet.alpha_,
        'coefs': enet.coef_,
        'intercept': enet.intercept_,
    }

    # 4. OLS (no regularization)
    ols = LinearRegression(fit_intercept=True)
    ols.fit(X_val, y_val)
    results['OLS'] = {
        'model': ols,
        'alpha': 0,
        'coefs': ols.coef_,
        'intercept': ols.intercept_,
    }

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stacking Ensemble with Meta-Learner')

    # Data
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--stock-pool', default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--gpu', type=int, default=0)

    # Stacking options
    parser.add_argument('--add-meta-features', action='store_true',
                        help='Add disagreement and mean as extra meta-features')

    # Backtest
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=2)
    parser.add_argument('--account', type=float, default=100000)
    parser.add_argument('--rebalance-freq', type=int, default=5)
    parser.add_argument('--strategy', default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss'])

    args = parser.parse_args()

    time_splits = {
        'train_start': FINAL_TEST['train_start'],
        'train_end': FINAL_TEST['train_end'],
        'valid_start': FINAL_TEST['valid_start'],
        'valid_end': FINAL_TEST['valid_end'],
        'test_start': FINAL_TEST['test_start'],
        'test_end': FINAL_TEST['test_end'],
    }

    model_names = list(MODEL_CONFIGS.keys())
    symbols = STOCK_POOLS[args.stock_pool]

    print("=" * 70)
    print("STACKING ENSEMBLE: Meta-Learner on Validation Predictions")
    print("=" * 70)
    for name, config in MODEL_CONFIGS.items():
        print(f"  {name:<22s}: {config['model_path'].name}")
    print(f"Stock Pool:       {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Meta-features:    {'Yes (disagreement + mean)' if args.add_meta_features else 'No (predictions only)'}")
    print(f"Valid (meta-train): {time_splits['valid_start']} ~ {time_splits['valid_end']}")
    print(f"Test (evaluate):    {time_splits['test_start']} ~ {time_splits['test_end']}")
    print("=" * 70)

    # Check model files
    for name, config in MODEL_CONFIGS.items():
        if not config['model_path'].exists():
            print(f"Error: {name} model not found: {config['model_path']}")
            sys.exit(1)

    # ----------------------------------------------------------------
    # [1] Initialize Qlib
    # ----------------------------------------------------------------
    print("\n[1] Initializing Qlib...")
    qlib.init(
        provider_uri=str(PROJECT_ROOT / "my_data" / "qlib_us"),
        region=REG_US,
        custom_ops=TALIB_OPS,
        kernels=1,
        joblib_backend=None,
    )

    # ----------------------------------------------------------------
    # [2] Create datasets (include validation)
    # ----------------------------------------------------------------
    print("\n[2] Creating datasets (valid + test)...")
    datasets = {}
    for name, config in MODEL_CONFIGS.items():
        print(f"    {name} ({config['handler']})...")
        handler = create_data_handler(config['handler'], symbols, time_splits, args.nday, include_valid=True)
        datasets[name] = create_dataset(handler, time_splits, include_valid=True)

    # ----------------------------------------------------------------
    # [3] Load models
    # ----------------------------------------------------------------
    print("\n[3] Loading models...")
    models = {}
    for name, config in MODEL_CONFIGS.items():
        if config['type'] == 'ae_mlp':
            models[name] = load_ae_mlp_model(config['model_path'])
        elif config['type'] == 'catboost':
            models[name] = load_catboost_model(config['model_path'])
        print(f"    {name} loaded")

    # ----------------------------------------------------------------
    # [4] Generate validation predictions (for meta-learner training)
    # ----------------------------------------------------------------
    print("\n[4] Generating validation predictions (2024)...")
    val_preds = {}
    for name, config in MODEL_CONFIGS.items():
        if config['type'] == 'ae_mlp':
            val_preds[name] = predict_with_ae_mlp(models[name], datasets[name], segment="valid")
        elif config['type'] == 'catboost':
            val_preds[name] = predict_with_catboost(models[name], datasets[name], segment="valid")
        print(f"    {name}: {len(val_preds[name])} samples, "
              f"range [{val_preds[name].min():.4f}, {val_preds[name].max():.4f}]")

    # Get validation labels (from raw-return handler)
    label_source = 'AE-MLP-v7'  # raw return, not mkt-neutral
    val_label = datasets[label_source].prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(val_label, pd.DataFrame):
        val_label = val_label.iloc[:, 0]
    print(f"    Labels: {len(val_label)} samples from {label_source}")

    # ----------------------------------------------------------------
    # [5] Generate test predictions
    # ----------------------------------------------------------------
    print("\n[5] Generating test predictions (2025)...")
    test_preds = {}
    for name, config in MODEL_CONFIGS.items():
        if config['type'] == 'ae_mlp':
            test_preds[name] = predict_with_ae_mlp(models[name], datasets[name], segment="test")
        elif config['type'] == 'catboost':
            test_preds[name] = predict_with_catboost(models[name], datasets[name], segment="test")
        print(f"    {name}: {len(test_preds[name])} samples")

    # Get test labels
    test_label = datasets[label_source].prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(test_label, pd.DataFrame):
        test_label = test_label.iloc[:, 0]

    # ----------------------------------------------------------------
    # [6] Prepare stacking matrices
    # ----------------------------------------------------------------
    print("\n[6] Preparing stacking matrices...")
    X_val, val_idx = prepare_stacking_matrix(val_preds, model_names, args.add_meta_features)
    X_test, test_idx = prepare_stacking_matrix(test_preds, model_names, args.add_meta_features)

    # Align labels with common index
    y_val = val_label.loc[val_idx].fillna(0)
    y_test = test_label.loc[test_idx].fillna(0)

    # Remove NaN rows
    valid_mask = ~(X_val.isna().any(axis=1) | y_val.isna())
    X_val_clean = X_val[valid_mask]
    y_val_clean = y_val[valid_mask]

    print(f"    Validation: {X_val_clean.shape[0]} samples, {X_val_clean.shape[1]} features")
    print(f"    Test:       {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"    Features:   {list(X_val_clean.columns)}")

    # ----------------------------------------------------------------
    # [7] Correlation analysis
    # ----------------------------------------------------------------
    print("\n[7] Prediction correlations (test set)...")
    corr = calculate_pairwise_correlations(test_preds)
    print(corr.to_string())

    # ----------------------------------------------------------------
    # [8] Train meta-learners
    # ----------------------------------------------------------------
    print("\n[8] Training meta-learners on validation data...")
    meta_results = train_meta_learners(X_val_clean.values, y_val_clean.values)

    feature_names = list(X_val_clean.columns)
    print(f"\n    {'Meta-Learner':<14s} | {'Alpha':>8s} | ", end="")
    for fn in feature_names:
        print(f"{fn[:10]:>10s} | ", end="")
    print(f"{'Intercept':>10s}")
    print("    " + "-" * (16 + 12 + 13 * len(feature_names) + 12))

    for ml_name, info in meta_results.items():
        coefs = info['coefs']
        intercept = info['intercept']
        alpha = info['alpha']
        print(f"    {ml_name:<14s} | {alpha:>8.4f} | ", end="")
        for c in coefs:
            print(f"{c:>10.6f} | ", end="")
        print(f"{intercept:>10.6f}")

    # ----------------------------------------------------------------
    # [9] Generate stacking predictions on test set & evaluate
    # ----------------------------------------------------------------
    print("\n[9] Evaluating on test set (2025)...")
    print()

    # Baseline: V2 zscore_weighted (grid_search weights from V2 results)
    v2_weights = {'AE-MLP-v7': 0.400, 'AE-MLP-v9': 0.250,
                  'AE-MLP-mkt-neutral': 0.150, 'CatBoost': 0.200}
    baseline_pred = ensemble_predictions_multi(test_preds, 'zscore_weighted', v2_weights)
    base_ic, base_std, base_icir, _ = compute_ic(baseline_pred, test_label)

    # Also compute equal-weight baseline
    equal_pred = ensemble_predictions_multi(test_preds, 'zscore_mean')
    eq_ic, eq_std, eq_icir, _ = compute_ic(equal_pred, test_label)

    # Print header
    col_w = 18
    print(f"    {'Method':<{col_w}s} | {'Mean IC':>10s} | {'IC Std':>10s} | {'ICIR':>10s} | {'vs V2 IC':>10s} | {'vs V2 ICIR':>10s}")
    print(f"    {'-' * (col_w + 62)}")

    # Individual models
    for name in model_names:
        ic, std, icir, _ = compute_ic(test_preds[name], test_label)
        print(f"    {name:<{col_w}s} | {ic:>10.4f} | {std:>10.4f} | {icir:>10.4f} |{'':>11s} |")

    print(f"    {'-' * (col_w + 62)}")

    # Baselines
    print(f"    {'V2 zscore_wt':<{col_w}s} | {base_ic:>10.4f} | {base_std:>10.4f} | {base_icir:>10.4f} | {'baseline':>10s} | {'baseline':>10s}")
    print(f"    {'Equal weight':<{col_w}s} | {eq_ic:>10.4f} | {eq_std:>10.4f} | {eq_icir:>10.4f} | {eq_ic - base_ic:>+10.4f} | {eq_icir - base_icir:>+10.4f}")
    print(f"    {'-' * (col_w + 62)}")

    # Stacking meta-learners
    best_method = None
    best_icir = base_icir

    for ml_name, info in meta_results.items():
        model = info['model']
        stacking_scores = model.predict(X_test.values)
        stacking_pred = pd.Series(stacking_scores, index=test_idx, name='score')

        ic, std, icir, _ = compute_ic(stacking_pred, test_label)
        ic_diff = ic - base_ic
        icir_diff = icir - base_icir

        marker = ""
        if icir > best_icir:
            best_icir = icir
            best_method = ml_name
            marker = " <-- best"

        print(f"    Stacking-{ml_name:<{col_w - 9}s} | {ic:>10.4f} | {std:>10.4f} | {icir:>10.4f} | {ic_diff:>+10.4f} | {icir_diff:>+10.4f}{marker}")

    print(f"    {'=' * (col_w + 62)}")

    # ----------------------------------------------------------------
    # [10] Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    if best_method:
        info = meta_results[best_method]
        print(f"BEST: Stacking-{best_method} (ICIR={best_icir:.4f}, vs V2 baseline {base_icir:.4f})")
        print(f"\nCoefficients ({best_method}, alpha={info['alpha']:.4f}):")
        for fn, c in zip(feature_names, info['coefs']):
            print(f"  {fn:<22s}: {c:>10.6f}")
        print(f"  {'Intercept':<22s}: {info['intercept']:>10.6f}")
    else:
        print(f"No stacking method beat V2 baseline (ICIR={base_icir:.4f})")
    print("=" * 70)

    # ----------------------------------------------------------------
    # [11] Backtest (using best stacking method, or V2 baseline if none better)
    # ----------------------------------------------------------------
    if args.backtest:
        if best_method:
            print(f"\n[11] Running backtest with Stacking-{best_method}...")
            model = meta_results[best_method]['model']
            bt_scores = model.predict(X_test.values)
            bt_pred = pd.Series(bt_scores, index=test_idx, name='score')
        else:
            print(f"\n[11] Running backtest with V2 baseline (no stacking improvement)...")
            bt_pred = baseline_pred

        if not hasattr(args, 'handler'):
            args.handler = "stacking_ensemble"

        run_ensemble_backtest(bt_pred, args, time_splits)


if __name__ == "__main__":
    main()
