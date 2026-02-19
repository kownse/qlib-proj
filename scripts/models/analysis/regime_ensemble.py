"""
Regime-Dependent Ensemble Weights (Research K - Step 2)

Uses 2-regime splits to assign different ensemble weights depending on
the current market environment. Optimizes weights on 2024 (validation),
evaluates out-of-sample on 2025 (test).

Regime definitions tested:
  1. VIX Level: Calm (VIX < 20) vs Stressed (VIX >= 20)
  2. Market Trend: Bull (SPY > 50d MA) vs Bear (SPY <= 50d MA)
  3. VIX Trend: VIX Falling (20d change <= 0) vs VIX Rising (20d change > 0)

Usage:
    python scripts/models/analysis/regime_ensemble.py
"""

import os

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

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import argparse
import numpy as np
import pandas as pd
from itertools import product

import qlib
from qlib.constant import REG_US
from qlib.data.dataset.handler import DataHandlerLP

from utils.talib_ops import TALIB_OPS
from data.stock_pools import STOCK_POOLS

from models.common import (
    PROJECT_ROOT,
    QLIB_DATA_PATH,
    MODEL_SAVE_PATH,
    FINAL_TEST,
)
from models.common.ensemble import (
    load_ae_mlp_model,
    create_ensemble_data_handler,
    create_ensemble_dataset,
    predict_with_ae_mlp,
    predict_with_catboost,
    compute_ic,
    zscore_by_day,
)
from models.common.training import load_catboost_model


# ── Model Config (same as V5) ──────────────────────────────────────

MODEL_CONFIG = {
    'ae': {
        'path': MODEL_SAVE_PATH / 'ae_mlp_cv_alpha158-enhanced-v7_sp500_5d_best.keras',
        'handler': 'alpha158-enhanced-v7',
        'display': 'AE-MLP-V7',
        'type': 'ae_mlp',
    },
    'ae_mn': {
        'path': MODEL_SAVE_PATH / 'ae_mlp_cv_v9-mkt-neutral_sp500_5d.keras',
        'handler': 'v9-mkt-neutral',
        'display': 'AE-MLP-MN',
        'type': 'ae_mlp',
    },
    'ae_v9r': {
        'path': MODEL_SAVE_PATH / 'ae_mlp_cv_alpha158-enhanced-v9_sp500_5d_20_from_2015.keras',
        'handler': 'alpha158-enhanced-v9',
        'display': 'AE-MLP-V9R',
        'type': 'ae_mlp',
    },
    'cb': {
        'path': MODEL_SAVE_PATH / 'catboost_cv_catboost-v1_sp500_5d_20260129_141353_best.cbm',
        'handler': 'catboost-v1',
        'display': 'CatBoost',
        'type': 'catboost',
    },
}

MODEL_NAMES = list(MODEL_CONFIG.keys())


# ── Regime definitions ──────────────────────────────────────────────

def load_regime_data():
    """Load VIX and SPY data, return regime labels DataFrame."""
    vix_path = PROJECT_ROOT / 'my_data' / 'macro_csv' / 'VIX.csv'
    spy_path = PROJECT_ROOT / 'my_data' / 'macro_csv' / 'SPY.csv'

    vix = pd.read_csv(vix_path, parse_dates=['date']).set_index('date')['close'].sort_index()
    spy = pd.read_csv(spy_path, parse_dates=['date']).set_index('date')['close'].sort_index()

    common = vix.index.intersection(spy.index)
    vix = vix.loc[common]
    spy = spy.loc[common]

    regimes = pd.DataFrame(index=common)

    # 1. VIX level: Calm vs Stressed
    regimes['vix_20'] = np.where(vix >= 20, 'Stressed', 'Calm')

    # 2. Market trend: Bull vs Bear
    spy_ma50 = spy.rolling(50).mean()
    regimes['mkt_trend'] = np.where(spy > spy_ma50, 'Bull', 'Bear')

    # 3. VIX trend: Rising vs Falling
    vix_chg = vix - vix.shift(20)
    regimes['vix_trend'] = np.where(vix_chg > 0, 'VIX_Rising', 'VIX_Falling')

    return regimes


# ── Weight grid search per regime ───────────────────────────────────

def generate_weight_combos(n_models, step=0.05, min_weight=0.05):
    """Generate all weight combinations that sum to 1.0."""
    steps = int(round(1.0 / step))
    min_steps = max(1, int(round(min_weight / step)))

    combos = []
    for combo in product(range(min_steps, steps + 1), repeat=n_models):
        if sum(combo) == steps:
            combos.append(tuple(c * step for c in combo))
    return combos


def grid_search_weights(zscores: dict, label: pd.Series, step=0.05, min_weight=0.05):
    """Find optimal weights via grid search on IC.

    Returns: (best_weights_dict, best_ic, best_icir)
    """
    names = list(zscores.keys())
    n = len(names)
    combos = generate_weight_combos(n, step, min_weight)

    # Align
    common_idx = label.index
    for name in names:
        common_idx = common_idx.intersection(zscores[name].index)
    aligned = {name: zscores[name].loc[common_idx] for name in names}
    y = label.loc[common_idx]
    valid = ~y.isna()
    for name in names:
        valid &= ~aligned[name].isna()
    aligned = {name: aligned[name][valid] for name in names}
    y = y[valid]

    best_ic = -np.inf
    best_icir = -np.inf
    best_weights = {name: 1.0 / n for name in names}

    for combo in combos:
        weights = {name: w for name, w in zip(names, combo)}
        total = sum(combo)
        ens = sum(aligned[name] * weights[name] for name in names) / total

        df = pd.DataFrame({'pred': ens, 'label': y})
        ic_by_date = df.groupby(level='datetime').apply(
            lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
        ).dropna()

        if len(ic_by_date) < 5:
            continue

        mean_ic = ic_by_date.mean()
        ic_std = ic_by_date.std()
        icir = mean_ic / ic_std if ic_std > 0 else 0

        if mean_ic > best_ic:
            best_ic = mean_ic
            best_icir = icir
            best_weights = weights.copy()

    return best_weights, best_ic, best_icir


# ── Regime-aware ensemble ───────────────────────────────────────────

def regime_ensemble_predict(zscores: dict, regime_labels: pd.Series,
                            regime_weights: dict, default_weights: dict):
    """Apply regime-dependent weights to z-scored predictions.

    Args:
        zscores: {model_name: pd.Series} z-scored predictions
        regime_labels: pd.Series (datetime index -> regime label)
        regime_weights: {regime_label: {model_name: weight}}
        default_weights: fallback weights if regime not found
    Returns:
        pd.Series of ensemble predictions
    """
    names = list(zscores.keys())

    common_idx = zscores[names[0]].index
    for name in names[1:]:
        common_idx = common_idx.intersection(zscores[name].index)
    aligned = {name: zscores[name].loc[common_idx] for name in names}

    result = pd.Series(0.0, index=common_idx, name='score')
    dates = common_idx.get_level_values('datetime')

    # Process each regime as a group for efficiency
    regime_assigned = dates.map(lambda d: regime_labels.get(d, None))

    for regime_val, weights in regime_weights.items():
        mask = (regime_assigned == regime_val)
        if mask.sum() == 0:
            continue
        total_w = sum(weights.values())
        for name in names:
            result.loc[mask] += aligned[name].loc[mask] * weights[name]
        result.loc[mask] /= total_w

    # Handle dates not assigned to any regime
    unassigned = regime_assigned.isna()
    if unassigned.sum() > 0:
        total_w = sum(default_weights.values())
        for name in names:
            result.loc[unassigned] += aligned[name].loc[unassigned] * default_weights[name]
        result.loc[unassigned] /= total_w

    return result


def fixed_ensemble_predict(zscores: dict, weights: dict):
    """Apply fixed weights (baseline)."""
    names = list(zscores.keys())
    common_idx = zscores[names[0]].index
    for name in names[1:]:
        common_idx = common_idx.intersection(zscores[name].index)
    aligned = {name: zscores[name].loc[common_idx] for name in names}

    total = sum(weights.values())
    result = sum(aligned[name] * weights[name] for name in names) / total
    result.name = 'score'
    return result


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Regime-Dependent Ensemble (Step 2)')
    parser.add_argument('--stock-pool', type=str, default='sp500')
    parser.add_argument('--nday', type=int, default=5)
    args = parser.parse_args()

    print("=" * 90)
    print("  REGIME-DEPENDENT ENSEMBLE WEIGHTS (Research K - Step 2)")
    print("=" * 90)
    print(f"  Approach: Optimize weights on 2024, evaluate OOS on 2025")
    print(f"  Models: {', '.join(cfg['display'] for cfg in MODEL_CONFIG.values())}")

    # [1] Init
    print("\n[1] Initializing Qlib...")
    qlib.init(
        provider_uri=str(QLIB_DATA_PATH),
        region=REG_US,
        custom_ops=TALIB_OPS,
        kernels=1,
        joblib_backend=None,
    )
    symbols = STOCK_POOLS[args.stock_pool]

    # [2] Load models
    print("\n[2] Loading models...")
    models = {}
    for key, cfg in MODEL_CONFIG.items():
        if cfg['type'] == 'ae_mlp':
            models[key] = load_ae_mlp_model(cfg['path'])
        elif cfg['type'] == 'catboost':
            models[key] = load_catboost_model(cfg['path'])
        print(f"    {cfg['display']}: loaded")

    # [3] Create datasets
    print("\n[3] Creating datasets...")
    time_splits = {
        'train_start': FINAL_TEST['train_start'],
        'train_end': FINAL_TEST['train_end'],
        'valid_start': FINAL_TEST['valid_start'],
        'valid_end': FINAL_TEST['valid_end'],
        'test_start': FINAL_TEST['test_start'],
        'test_end': FINAL_TEST['test_end'],
    }
    datasets = {}
    for key, cfg in MODEL_CONFIG.items():
        h = create_ensemble_data_handler(cfg['handler'], symbols, time_splits, args.nday,
                                         include_valid=True)
        datasets[key] = create_ensemble_dataset(h, time_splits, include_valid=True)

    # [4] Generate predictions
    print("\n[4] Generating predictions...")
    pred_valid = {}
    pred_test = {}
    for key, cfg in MODEL_CONFIG.items():
        if cfg['type'] == 'ae_mlp':
            pred_valid[key] = predict_with_ae_mlp(models[key], datasets[key], segment='valid')
            pred_test[key] = predict_with_catboost(models[key], datasets[key], segment='test') \
                if False else predict_with_ae_mlp(models[key], datasets[key], segment='test')
        elif cfg['type'] == 'catboost':
            pred_valid[key] = predict_with_catboost(models[key], datasets[key], segment='valid')
            pred_test[key] = predict_with_catboost(models[key], datasets[key], segment='test')
        print(f"    {cfg['display']}: valid={len(pred_valid[key])}, test={len(pred_test[key])}")

    # Get labels
    label_valid = datasets['ae'].prepare('valid', col_set='label', data_key=DataHandlerLP.DK_L)
    label_test = datasets['ae'].prepare('test', col_set='label', data_key=DataHandlerLP.DK_L)
    if isinstance(label_valid, pd.DataFrame):
        label_valid = label_valid.iloc[:, 0]
    if isinstance(label_test, pd.DataFrame):
        label_test = label_test.iloc[:, 0]

    # Z-score normalize
    zscores_valid = {name: zscore_by_day(pred_valid[name]) for name in MODEL_NAMES}
    zscores_test = {name: zscore_by_day(pred_test[name]) for name in MODEL_NAMES}

    # [5] Baseline: fixed equal weights
    print("\n[5] Baseline evaluation...")
    equal_weights = {name: 0.25 for name in MODEL_NAMES}

    # Also optimize fixed weights on 2024
    print("    Grid search for fixed weights on 2024...")
    fixed_opt_weights, fixed_opt_ic, fixed_opt_icir = grid_search_weights(
        zscores_valid, label_valid, step=0.05, min_weight=0.05)

    # Evaluate baselines on 2025
    ens_equal_test = fixed_ensemble_predict(zscores_test, equal_weights)
    ens_fixed_opt_test = fixed_ensemble_predict(zscores_test, fixed_opt_weights)

    eq_ic, eq_std, eq_icir, eq_ic_series = compute_ic(ens_equal_test, label_test)
    fo_ic, fo_std, fo_icir, fo_ic_series = compute_ic(ens_fixed_opt_test, label_test)

    print(f"\n    Equal weights:     2025 IC={eq_ic:.4f}, ICIR={eq_icir:.4f}")
    print(f"    Fixed optimized:  2025 IC={fo_ic:.4f}, ICIR={fo_icir:.4f}")
    w_str = ", ".join(f"{MODEL_CONFIG[k]['display']}={v:.2f}" for k, v in fixed_opt_weights.items())
    print(f"      Weights: {w_str}")
    print(f"      (Optimized on 2024: IC={fixed_opt_ic:.4f})")

    # [6] Load regime data
    print("\n[6] Loading regime data...")
    regimes = load_regime_data()

    # [7] Regime-dependent weight optimization
    regime_definitions = {
        'VIX Level (>=20)': 'vix_20',
        'Market Trend': 'mkt_trend',
        'VIX Trend (20d)': 'vix_trend',
    }

    print("\n" + "=" * 90)
    print("  REGIME-DEPENDENT WEIGHT OPTIMIZATION")
    print("=" * 90)

    results = []

    for regime_name, regime_col in regime_definitions.items():
        print(f"\n{'─' * 90}")
        print(f"  Regime: {regime_name}")
        print(f"{'─' * 90}")

        regime_labels = regimes[regime_col]
        regime_values = sorted(regime_labels.dropna().unique())

        # Show regime distribution in 2024 and 2025
        valid_dates = pred_valid[MODEL_NAMES[0]].index.get_level_values('datetime').unique()
        test_dates = pred_test[MODEL_NAMES[0]].index.get_level_values('datetime').unique()

        for period_name, period_dates in [('2024 (train)', valid_dates), ('2025 (test)', test_dates)]:
            print(f"\n  {period_name} distribution:")
            for rv in regime_values:
                n = (regime_labels.loc[regime_labels.index.isin(period_dates)] == rv).sum()
                total = len(period_dates)
                print(f"    {rv}: {n} days ({n/total*100:.1f}%)")

        # Optimize weights per regime on 2024
        print(f"\n  Optimizing weights per regime on 2024...")
        per_regime_weights = {}

        for rv in regime_values:
            # Filter 2024 predictions by regime days
            regime_dates = regime_labels[regime_labels == rv].index
            valid_regime_dates = valid_dates.intersection(regime_dates)

            if len(valid_regime_dates) < 20:
                print(f"    {rv}: Only {len(valid_regime_dates)} days, using equal weights")
                per_regime_weights[rv] = equal_weights.copy()
                continue

            # Filter z-scores to regime days
            zs_regime = {}
            for name in MODEL_NAMES:
                dates_in_pred = zscores_valid[name].index.get_level_values('datetime')
                mask = dates_in_pred.isin(valid_regime_dates)
                zs_regime[name] = zscores_valid[name].loc[mask]

            # Filter labels
            dates_in_label = label_valid.index.get_level_values('datetime')
            label_regime = label_valid.loc[dates_in_label.isin(valid_regime_dates)]

            weights, opt_ic, opt_icir = grid_search_weights(
                zs_regime, label_regime, step=0.05, min_weight=0.05)
            per_regime_weights[rv] = weights

            w_str = ", ".join(f"{MODEL_CONFIG[k]['display']}={v:.2f}" for k, v in weights.items())
            print(f"    {rv}: IC={opt_ic:.4f}, ICIR={opt_icir:.4f}")
            print(f"      Weights: {w_str}")

        # Evaluate on 2025
        print(f"\n  Evaluating on 2025 (OOS)...")
        ens_regime_test = regime_ensemble_predict(
            zscores_test, regime_labels, per_regime_weights, equal_weights)
        r_ic, r_std, r_icir, r_ic_series = compute_ic(ens_regime_test, label_test)

        # Compute per-regime IC on 2025
        print(f"\n  Per-regime 2025 IC:")
        for rv in regime_values:
            regime_dates = regime_labels[regime_labels == rv].index
            test_regime_dates = test_dates.intersection(regime_dates)
            if len(test_regime_dates) < 5:
                print(f"    {rv}: insufficient data")
                continue

            dates_in_pred = r_ic_series.index
            mask = dates_in_pred.isin(test_regime_dates)
            regime_ic = r_ic_series.loc[mask]
            if len(regime_ic) > 0:
                print(f"    {rv}: {len(regime_ic)} days, IC={regime_ic.mean():.4f}")

        # Compare
        ic_diff = r_ic - fo_ic
        icir_diff = r_icir - fo_icir
        ic_pct = ic_diff / abs(fo_ic) * 100 if fo_ic != 0 else 0
        icir_pct = icir_diff / abs(fo_icir) * 100 if fo_icir != 0 else 0

        print(f"\n  ┌──────────────────────────────────────────────────────────┐")
        print(f"  │  2025 OOS Results: {regime_name:<39s}│")
        print(f"  ├──────────────────────────────────────────────────────────┤")
        print(f"  │  Fixed optimized:    IC={fo_ic:.4f}  ICIR={fo_icir:.4f}         │")
        print(f"  │  Regime-dependent:   IC={r_ic:.4f}  ICIR={r_icir:.4f}         │")
        print(f"  │  Delta:              IC={ic_diff:+.4f} ({ic_pct:+.1f}%)  "
              f"ICIR={icir_diff:+.4f} ({icir_pct:+.1f}%)  │")
        print(f"  └──────────────────────────────────────────────────────────┘")

        results.append({
            'regime': regime_name,
            'regime_col': regime_col,
            'ic': r_ic,
            'icir': r_icir,
            'ic_diff': ic_diff,
            'icir_diff': icir_diff,
            'weights': per_regime_weights,
        })

    # [8] Summary
    print(f"\n{'=' * 90}")
    print(f"  FINAL COMPARISON")
    print(f"{'=' * 90}")

    # Individual model baselines on 2025
    print(f"\n  Individual Models (2025):")
    for key, cfg in MODEL_CONFIG.items():
        m_ic, m_std, m_icir, _ = compute_ic(pred_test[key], label_test)
        print(f"    {cfg['display']:<15s}: IC={m_ic:.4f}, ICIR={m_icir:.4f}")

    print(f"\n  Ensemble Methods (2025 OOS):")
    print(f"  {'Method':<40s} {'IC':>8s} {'ICIR':>8s} {'IC vs Fixed':>12s}")
    print(f"  {'-' * 70}")
    print(f"  {'Equal weights (0.25 each)':<40s} {eq_ic:>8.4f} {eq_icir:>8.4f} {'baseline':>12s}")
    print(f"  {'Fixed optimized (2024)':<40s} {fo_ic:>8.4f} {fo_icir:>8.4f} {'baseline':>12s}")

    for r in results:
        label_str = f"Regime: {r['regime']}"
        ic_str = f"{r['ic_diff']:+.4f}"
        print(f"  {label_str:<40s} {r['ic']:>8.4f} {r['icir']:>8.4f} {ic_str:>12s}")

    best = max(results, key=lambda x: x['ic'])
    print(f"\n  Best regime method: {best['regime']}")
    print(f"    IC improvement over fixed: {best['ic_diff']:+.4f} ({best['ic_diff']/abs(fo_ic)*100:+.1f}%)")
    print(f"    ICIR improvement: {best['icir_diff']:+.4f} ({best['icir_diff']/abs(fo_icir)*100:+.1f}%)")

    if best['ic_diff'] > 0.002:
        print(f"\n  -> POSITIVE RESULT: Regime conditioning improves OOS performance.")
        print(f"     Recommend integrating into daily trading ensemble.")
    elif best['ic_diff'] > 0:
        print(f"\n  -> MARGINAL RESULT: Small improvement, may not justify complexity.")
    else:
        print(f"\n  -> NEGATIVE RESULT: Regime conditioning does not help OOS.")
        print(f"     Fixed weights are more robust.")

    # Print best regime's per-regime weights for integration
    if best['ic_diff'] > 0:
        print(f"\n  Recommended weights for {best['regime']}:")
        for rv, weights in best['weights'].items():
            w_str = ", ".join(f"{MODEL_CONFIG[k]['display']}={v:.2f}" for k, v in weights.items())
            print(f"    {rv}: {w_str}")

    print(f"\n{'=' * 90}")


if __name__ == '__main__':
    main()
