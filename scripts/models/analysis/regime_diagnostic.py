"""
Regime Diagnostic Analysis (Research K - Step 1)

Analyze whether different ensemble models perform differently across market regimes.
This script loads the V5 ensemble models, generates predictions on 2024-2025,
computes daily IC for each model, and breaks down performance by VIX regime
and market trend regime.

Usage:
    python scripts/models/analysis/regime_diagnostic.py
    python scripts/models/analysis/regime_diagnostic.py --period test   # 2025 only
    python scripts/models/analysis/regime_diagnostic.py --period all    # 2024-2025 (default)
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
    PROJECT_ROOT,
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
    compute_ic,
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


def load_vix_data() -> pd.Series:
    """Load VIX closing prices as a time-indexed Series."""
    vix_path = PROJECT_ROOT / 'my_data' / 'macro_csv' / 'VIX.csv'
    df = pd.read_csv(vix_path, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    return df['close'].rename('vix')


def load_spy_data() -> pd.DataFrame:
    """Load SPY closing prices as a time-indexed Series."""
    spy_path = PROJECT_ROOT / 'my_data' / 'macro_csv' / 'SPY.csv'
    df = pd.read_csv(spy_path, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    return df['close'].rename('spy')


def define_regimes(vix: pd.Series, spy: pd.Series) -> pd.DataFrame:
    """Define regime labels from VIX and SPY data.

    Returns DataFrame with columns:
        - vix_regime: 'Low' (<15), 'Normal' (15-25), 'High' (>25)
        - vix_trend: 'Rising' (VIX 20d change > 0), 'Falling' (VIX 20d change <= 0)
        - mkt_trend: 'Bull' (SPY > 50d MA), 'Bear' (SPY <= 50d MA)
        - combined: vix_regime × mkt_trend
    """
    # Align on common dates
    common_dates = vix.index.intersection(spy.index)
    vix = vix.loc[common_dates]
    spy = spy.loc[common_dates]

    regimes = pd.DataFrame(index=common_dates)

    # VIX level regime
    regimes['vix_regime'] = pd.cut(
        vix, bins=[0, 15, 25, 100],
        labels=['Low (<15)', 'Normal (15-25)', 'High (>25)'],
    )

    # VIX trend (20-day change direction)
    vix_change_20d = vix - vix.shift(20)
    regimes['vix_trend'] = np.where(vix_change_20d > 0, 'VIX Rising', 'VIX Falling')

    # Market trend (SPY vs 50-day MA)
    spy_ma50 = spy.rolling(50).mean()
    regimes['mkt_trend'] = np.where(spy > spy_ma50, 'Bull', 'Bear')

    # Combined regime
    regimes['combined'] = regimes['vix_regime'].astype(str) + ' / ' + regimes['mkt_trend']

    return regimes


def compute_daily_ic(pred: pd.Series, label: pd.Series) -> pd.Series:
    """Compute daily IC (cross-sectional correlation) between predictions and labels."""
    common_idx = pred.index.intersection(label.index)
    pred_aligned = pred.loc[common_idx]
    label_aligned = label.loc[common_idx]

    valid_idx = ~(pred_aligned.isna() | label_aligned.isna())
    df = pd.DataFrame({
        'pred': pred_aligned[valid_idx],
        'label': label_aligned[valid_idx],
    })

    ic_by_date = df.groupby(level='datetime').apply(
        lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
    )
    return ic_by_date.dropna()


def analyze_regime(ic_by_date: pd.Series, regime_labels: pd.Series,
                   regime_name: str) -> pd.DataFrame:
    """Compute IC statistics within each regime category.

    Returns DataFrame with columns: regime, count, mean_ic, ic_std, icir
    """
    # Align dates
    common_dates = ic_by_date.index.intersection(regime_labels.index)
    ic_aligned = ic_by_date.loc[common_dates]
    regime_aligned = regime_labels.loc[common_dates]

    results = []
    for regime_val in regime_aligned.unique():
        if pd.isna(regime_val):
            continue
        mask = regime_aligned == regime_val
        ic_in_regime = ic_aligned[mask]
        n = len(ic_in_regime)
        if n < 5:
            continue
        mean_ic = ic_in_regime.mean()
        ic_std = ic_in_regime.std()
        icir = mean_ic / ic_std if ic_std > 0 else 0
        results.append({
            'regime': regime_val,
            'days': n,
            'mean_ic': mean_ic,
            'ic_std': ic_std,
            'icir': icir,
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values('regime').reset_index(drop=True)
    return df


def print_regime_table(model_results: dict, regime_name: str, regime_col: str):
    """Print a formatted table showing IC by regime for each model."""
    print(f"\n{'=' * 90}")
    print(f"  Regime Analysis: {regime_name}")
    print(f"{'=' * 90}")

    # Collect all regime values across models
    all_regimes = set()
    for model_key, df in model_results.items():
        all_regimes.update(df['regime'].tolist())
    all_regimes = sorted(all_regimes)

    if not all_regimes:
        print("  No data available for this regime definition.")
        return

    # Header
    model_names = [MODEL_CONFIG[k]['display'] for k in model_results.keys()]
    header = f"  {'Regime':<20s} {'Days':>5s}"
    for name in model_names:
        header += f" | {'IC':>7s} {'ICIR':>7s}"
    print(header)
    print("  " + "-" * (26 + 17 * len(model_names)))

    # Rows
    for regime in all_regimes:
        days_str = ""
        row = f"  {str(regime):<20s}"
        cells = []
        for model_key, df in model_results.items():
            match = df[df['regime'] == regime]
            if len(match) > 0:
                r = match.iloc[0]
                if not days_str:
                    days_str = f"{int(r['days']):>5d}"
                cells.append(f" | {r['mean_ic']:>7.4f} {r['icir']:>7.4f}")
            else:
                cells.append(f" | {'N/A':>7s} {'N/A':>7s}")
        row += days_str + "".join(cells)
        print(row)

    # Overall row
    print("  " + "-" * (26 + 17 * len(model_names)))
    row = f"  {'OVERALL':<20s}"
    days_str = ""
    cells = []
    for model_key, df in model_results.items():
        total_days = df['days'].sum()
        if not days_str:
            days_str = f"{int(total_days):>5d}"
        # Weighted average IC
        if total_days > 0:
            weighted_ic = (df['mean_ic'] * df['days']).sum() / total_days
            # Overall ICIR from all days
            cells.append(f" | {weighted_ic:>7.4f} {'':>7s}")
        else:
            cells.append(f" | {'N/A':>7s} {'':>7s}")
    row += days_str + "".join(cells)
    print(row)


def print_regime_diff_table(model_results: dict, regime_name: str):
    """Print IC difference between best and worst regime for each model."""
    print(f"\n  IC Spread (Max - Min regime) for {regime_name}:")
    print(f"  {'Model':<15s} {'Best Regime':<22s} {'IC':>7s} {'Worst Regime':<22s} {'IC':>7s} {'Spread':>8s}")
    print("  " + "-" * 85)

    for model_key, df in model_results.items():
        if len(df) < 2:
            continue
        best_idx = df['mean_ic'].idxmax()
        worst_idx = df['mean_ic'].idxmin()
        best = df.loc[best_idx]
        worst = df.loc[worst_idx]
        spread = best['mean_ic'] - worst['mean_ic']
        name = MODEL_CONFIG[model_key]['display']
        print(f"  {name:<15s} {str(best['regime']):<22s} {best['mean_ic']:>7.4f} "
              f"{str(worst['regime']):<22s} {worst['mean_ic']:>7.4f} {spread:>8.4f}")


def main():
    parser = argparse.ArgumentParser(description='Regime Diagnostic Analysis')
    parser.add_argument('--period', type=str, default='all',
                        choices=['all', 'test', 'valid'],
                        help='Analysis period: all=2024-2025, test=2025, valid=2024')
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--nday', type=int, default=5)
    args = parser.parse_args()

    print("=" * 90)
    print("  REGIME DIAGNOSTIC ANALYSIS (Research K - Step 1)")
    print("=" * 90)
    print(f"  Period: {args.period}")
    print(f"  Stock Pool: {args.stock_pool}")
    print(f"  Models: {', '.join(cfg['display'] for cfg in MODEL_CONFIG.values())}")

    # Check model files
    for key, cfg in MODEL_CONFIG.items():
        if not cfg['path'].exists():
            print(f"  ERROR: {cfg['display']} model not found: {cfg['path']}")
            sys.exit(1)

    # [1] Initialize Qlib
    print("\n[1] Initializing Qlib...")
    qlib.init(
        provider_uri=str(QLIB_DATA_PATH),
        region=REG_US,
        custom_ops=TALIB_OPS,
        kernels=1,
        joblib_backend=None,
    )

    symbols = STOCK_POOLS[args.stock_pool]
    print(f"    Stock pool: {args.stock_pool} ({len(symbols)} stocks)")

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
    print(f"    Valid: {time_splits['valid_start']} ~ {time_splits['valid_end']}")
    print(f"    Test:  {time_splits['test_start']} ~ {time_splits['test_end']}")

    datasets = {}
    for key, cfg in MODEL_CONFIG.items():
        h = create_ensemble_data_handler(cfg['handler'], symbols, time_splits, args.nday,
                                         include_valid=True)
        datasets[key] = create_ensemble_dataset(h, time_splits, include_valid=True)

    # [4] Generate predictions and compute daily IC
    print("\n[4] Generating predictions & computing daily IC...")

    # Determine which segments to use
    if args.period == 'all':
        segments = ['valid', 'test']
    elif args.period == 'test':
        segments = ['test']
    else:
        segments = ['valid']

    # Get labels
    label_parts = []
    for seg in segments:
        seg_label = datasets['ae'].prepare(seg, col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(seg_label, pd.DataFrame):
            seg_label = seg_label.iloc[:, 0]
        label_parts.append(seg_label)
    label = pd.concat(label_parts) if len(label_parts) > 1 else label_parts[0]

    # Generate predictions and compute daily IC per model
    daily_ic = {}
    for key, cfg in MODEL_CONFIG.items():
        pred_parts = []
        for seg in segments:
            if cfg['type'] == 'ae_mlp':
                pred = predict_with_ae_mlp(models[key], datasets[key], segment=seg)
            elif cfg['type'] == 'catboost':
                pred = predict_with_catboost(models[key], datasets[key], segment=seg)
            pred_parts.append(pred)
        pred_all = pd.concat(pred_parts) if len(pred_parts) > 1 else pred_parts[0]

        ic_series = compute_daily_ic(pred_all, label)
        daily_ic[key] = ic_series

        mean_ic = ic_series.mean()
        icir = mean_ic / ic_series.std() if ic_series.std() > 0 else 0
        print(f"    {cfg['display']:<15s}: {len(ic_series)} days, Mean IC={mean_ic:.4f}, ICIR={icir:.4f}")

    # [5] Load regime data
    print("\n[5] Loading regime data (VIX + SPY)...")
    vix = load_vix_data()
    spy = load_spy_data()
    regimes = define_regimes(vix, spy)

    # Show regime distribution in analysis period
    first_date = min(ic.index.min() for ic in daily_ic.values())
    last_date = max(ic.index.max() for ic in daily_ic.values())
    print(f"    Analysis period: {first_date.date()} ~ {last_date.date()}")

    regime_period = regimes.loc[first_date:last_date]
    print(f"\n    VIX Regime Distribution:")
    for regime_val in regime_period['vix_regime'].value_counts().sort_index().items():
        print(f"      {regime_val[0]}: {regime_val[1]} days ({regime_val[1]/len(regime_period)*100:.1f}%)")

    print(f"\n    Market Trend Distribution:")
    for trend_val in regime_period['mkt_trend'].value_counts().sort_index().items():
        print(f"      {trend_val[0]}: {trend_val[1]} days ({trend_val[1]/len(regime_period)*100:.1f}%)")

    print(f"\n    VIX Trend Distribution:")
    for trend_val in regime_period['vix_trend'].value_counts().sort_index().items():
        print(f"      {trend_val[0]}: {trend_val[1]} days ({trend_val[1]/len(regime_period)*100:.1f}%)")

    # [6] Regime analysis
    print("\n[6] Regime-conditional IC analysis...")

    regime_definitions = {
        'VIX Level': 'vix_regime',
        'VIX Trend (20d)': 'vix_trend',
        'Market Trend (SPY vs 50d MA)': 'mkt_trend',
        'Combined (VIX Level × Market Trend)': 'combined',
    }

    for regime_name, regime_col in regime_definitions.items():
        model_results = {}
        for key in MODEL_CONFIG:
            regime_labels = regimes[regime_col]
            result_df = analyze_regime(daily_ic[key], regime_labels, regime_name)
            model_results[key] = result_df

        print_regime_table(model_results, regime_name, regime_col)
        print_regime_diff_table(model_results, regime_name)

    # [7] Time-varying analysis: rolling 20-day IC
    print(f"\n{'=' * 90}")
    print(f"  Rolling 20-day IC Analysis")
    print(f"{'=' * 90}")

    for key, cfg in MODEL_CONFIG.items():
        ic = daily_ic[key]
        rolling_ic = ic.rolling(20, min_periods=10).mean()
        best_20d = rolling_ic.max()
        worst_20d = rolling_ic.min()
        best_date = rolling_ic.idxmax()
        worst_date = rolling_ic.idxmin()
        print(f"  {cfg['display']:<15s}: Best 20d IC={best_20d:.4f} ({best_date.date()}), "
              f"Worst 20d IC={worst_20d:.4f} ({worst_date.date()})")

    # [8] Model agreement analysis
    print(f"\n{'=' * 90}")
    print(f"  Model Agreement vs IC Analysis")
    print(f"{'=' * 90}")
    print(f"  (Do models predict better when they agree with each other?)")

    # Compute daily agreement: std of daily IC across models
    ic_df = pd.DataFrame(daily_ic)
    common_dates = ic_df.dropna().index

    if len(common_dates) > 20:
        # Agreement = avg pairwise correlation of IC
        ic_common = ic_df.loc[common_dates]
        model_agreement = ic_common.mean(axis=1)  # mean IC across models per day

        # Split into high-agreement (all models IC > 0) vs low-agreement (mixed signs)
        all_positive = (ic_common > 0).all(axis=1)
        all_negative = (ic_common < 0).all(axis=1)
        mixed = ~all_positive & ~all_negative

        print(f"\n  Days where ALL models have positive IC: {all_positive.sum()} "
              f"({all_positive.sum()/len(common_dates)*100:.1f}%)")
        print(f"  Days where ALL models have negative IC: {all_negative.sum()} "
              f"({all_negative.sum()/len(common_dates)*100:.1f}%)")
        print(f"  Days with mixed IC signs:               {mixed.sum()} "
              f"({mixed.sum()/len(common_dates)*100:.1f}%)")

        for condition, label_str in [(all_positive, 'All Positive IC'),
                                     (all_negative, 'All Negative IC'),
                                     (mixed, 'Mixed IC Signs')]:
            if condition.sum() < 5:
                continue
            print(f"\n  When {label_str} ({condition.sum()} days):")
            for key, cfg in MODEL_CONFIG.items():
                sub_ic = ic_common.loc[condition, key]
                print(f"    {cfg['display']:<15s}: Mean IC={sub_ic.mean():>7.4f}, "
                      f"Std={sub_ic.std():>7.4f}")

    # [9] Conclusion
    print(f"\n{'=' * 90}")
    print(f"  CONCLUSION")
    print(f"{'=' * 90}")

    # Find the model with largest regime spread
    max_spread = 0
    max_spread_model = None
    for key in MODEL_CONFIG:
        regime_labels = regimes['vix_regime']
        result_df = analyze_regime(daily_ic[key], regime_labels, 'VIX Level')
        if len(result_df) >= 2:
            spread = result_df['mean_ic'].max() - result_df['mean_ic'].min()
            if spread > max_spread:
                max_spread = spread
                max_spread_model = key

    if max_spread_model:
        print(f"\n  Largest VIX regime IC spread: {MODEL_CONFIG[max_spread_model]['display']} "
              f"(spread={max_spread:.4f})")

    # Threshold for "meaningful" regime difference
    if max_spread > 0.02:
        print(f"  -> Spread > 0.02: Regime conditioning has STRONG signal. Proceed to Step 2.")
    elif max_spread > 0.01:
        print(f"  -> Spread > 0.01: Regime conditioning has MODERATE signal. Consider Step 2.")
    else:
        print(f"  -> Spread < 0.01: Regime conditioning has WEAK signal. May not be worth pursuing.")

    print(f"\n{'=' * 90}")


if __name__ == '__main__':
    main()
