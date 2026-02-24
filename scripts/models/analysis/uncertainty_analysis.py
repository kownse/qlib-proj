"""
Uncertainty Diagnostic Analysis (Research J - Step 1)

Evaluates whether model disagreement is a useful uncertainty proxy:
  1. Load 4 V5 ensemble models, generate per-stock-day predictions on test period
  2. Compute uncertainty metrics:
     - disagreement: std of zscore predictions across 4 models
     - direction_agreement: count of models ranking stock in top 50%
  3. Analyze uncertainty vs ensemble IC by disagreement quantile groups
  4. Analyze temporal distribution of disagreement vs market volatility

Key validation: if high-agreement group IC >> low-agreement group IC → proceed to Step 2

Usage:
    conda run -n qlib310 python scripts/models/analysis/uncertainty_analysis.py
    conda run -n qlib310 python scripts/models/analysis/uncertainty_analysis.py --stock-pool sp100
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
    ensemble_predictions,
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


# ── Uncertainty computation ──────────────────────────────────────────

def compute_uncertainty_metrics(pred_dict):
    """Compute per-stock-day uncertainty metrics from individual model predictions.

    Returns:
        disagreement: pd.Series - std of zscored predictions (higher = more disagreement)
        direction_agreement: pd.Series - count of models ranking stock in top 50% (0-4)
        zscore_df: pd.DataFrame - zscored predictions for all models
    """
    names = list(pred_dict.keys())

    # Find common index across all models
    common_idx = pred_dict[names[0]].index
    for name in names[1:]:
        common_idx = common_idx.intersection(pred_dict[name].index)

    # Z-score each model's predictions cross-sectionally
    zscores = {name: zscore_by_day(pred_dict[name].loc[common_idx]) for name in names}
    zscore_df = pd.DataFrame(zscores)

    # Per-stock-day disagreement: std across models
    disagreement = zscore_df.std(axis=1)
    disagreement.name = 'disagreement'

    # Direction agreement: how many models rank this stock in top 50%?
    ranks = {}
    for name in names:
        ranks[name] = pred_dict[name].loc[common_idx].groupby(level='datetime').rank(pct=True)
    rank_df = pd.DataFrame(ranks)
    direction_agreement = (rank_df > 0.5).sum(axis=1)
    direction_agreement.name = 'direction_agreement'

    return disagreement, direction_agreement, zscore_df


def compute_disagreement_quantile(disagreement):
    """Compute cross-sectional quantile of disagreement per day (0=most agreement, 1=most disagreement)."""
    return disagreement.groupby(level='datetime').rank(pct=True)


# ── Analysis functions ───────────────────────────────────────────────

def analyze_ic_by_disagreement_quantile(ensemble_pred, label, disagreement, n_groups=4):
    """Analyze ensemble IC within disagreement quantile groups.

    Groups stocks by disagreement quantile each day, computes IC within each group.
    """
    common_idx = ensemble_pred.index.intersection(label.index).intersection(disagreement.index)
    pred_a = ensemble_pred.loc[common_idx]
    label_a = label.loc[common_idx]
    disagree_a = disagreement.loc[common_idx]

    # Cross-sectional quantile assignment per day
    quantile_rank = disagree_a.groupby(level='datetime').rank(pct=True)
    bins = np.linspace(0, 1, n_groups + 1)
    labels = [f'Q{i+1}' for i in range(n_groups)]
    group_labels = pd.cut(quantile_rank, bins=bins, labels=labels, include_lowest=True)

    results = {}
    for q_label in labels:
        mask = group_labels == q_label
        q_pred = pred_a[mask]
        q_label_vals = label_a[mask]

        if len(q_pred) < 100:
            results[q_label] = {'ic': np.nan, 'ic_std': np.nan, 'icir': np.nan, 'n_samples': len(q_pred)}
            continue

        ic, ic_std, icir, _ = compute_ic(q_pred, q_label_vals)
        results[q_label] = {'ic': ic, 'ic_std': ic_std, 'icir': icir, 'n_samples': len(q_pred)}

    return results


def analyze_ic_by_direction_agreement(ensemble_pred, label, direction_agreement):
    """Analyze ensemble IC grouped by direction agreement level (0-4)."""
    common_idx = ensemble_pred.index.intersection(label.index).intersection(direction_agreement.index)
    pred_a = ensemble_pred.loc[common_idx]
    label_a = label.loc[common_idx]
    da = direction_agreement.loc[common_idx]

    results = {}
    for level in sorted(da.unique()):
        mask = da == level
        q_pred = pred_a[mask]
        q_label_vals = label_a[mask]

        if len(q_pred) < 100:
            results[int(level)] = {'ic': np.nan, 'ic_std': np.nan, 'icir': np.nan, 'n_samples': len(q_pred)}
            continue

        ic, ic_std, icir, _ = compute_ic(q_pred, q_label_vals)
        results[int(level)] = {'ic': ic, 'ic_std': ic_std, 'icir': icir, 'n_samples': len(q_pred)}

    return results


def analyze_temporal_disagreement(disagreement):
    """Analyze how disagreement varies over time.

    Returns per-day summary statistics of cross-sectional disagreement.
    """
    daily_stats = disagreement.groupby(level='datetime').agg(['mean', 'median', 'std', 'max'])
    daily_stats.columns = ['mean_disagree', 'median_disagree', 'std_disagree', 'max_disagree']
    return daily_stats


def analyze_disagreement_vs_vix(daily_disagree_stats):
    """Compute correlation between daily market-level disagreement and VIX."""
    vix_path = PROJECT_ROOT / 'my_data' / 'macro_csv' / 'VIX.csv'
    if not vix_path.exists():
        print("    VIX data not found, skipping VIX correlation analysis")
        return None

    vix = pd.read_csv(vix_path, parse_dates=['date']).set_index('date')['close'].sort_index()
    vix.name = 'vix'

    common = daily_disagree_stats.index.intersection(vix.index)
    if len(common) < 10:
        print(f"    Only {len(common)} overlapping dates with VIX, skipping")
        return None

    merged = pd.DataFrame({
        'mean_disagree': daily_disagree_stats['mean_disagree'].loc[common],
        'vix': vix.loc[common],
    })

    corr = merged['mean_disagree'].corr(merged['vix'])
    return corr, merged


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Uncertainty Diagnostic Analysis (Research J - Step 1)',
    )
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--nday', type=int, default=5)
    args = parser.parse_args()

    time_splits = {
        'train_start': FINAL_TEST['train_start'],
        'train_end': FINAL_TEST['train_end'],
        'valid_start': FINAL_TEST['valid_start'],
        'valid_end': FINAL_TEST['valid_end'],
        'test_start': FINAL_TEST['test_start'],
        'test_end': FINAL_TEST['test_end'],
    }

    print("=" * 80)
    print("UNCERTAINTY DIAGNOSTIC ANALYSIS (Research J - Step 1)")
    print("=" * 80)
    print(f"Stock Pool: {args.stock_pool}")
    print(f"Test Period: {time_splits['test_start']} to {time_splits['test_end']}")
    print("=" * 80)

    # Check model files
    for key, cfg in MODEL_CONFIG.items():
        if not cfg['path'].exists():
            print(f"Error: {cfg['display']} model not found: {cfg['path']}")
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
    print(f"    {len(symbols)} stocks loaded")

    # [2] Create datasets and load models
    print("\n[2] Creating datasets and loading models...")
    datasets = {}
    models = {}
    for key, cfg in MODEL_CONFIG.items():
        print(f"    {cfg['display']}: loading dataset ({cfg['handler']}) and model...")
        h = create_ensemble_data_handler(cfg['handler'], symbols, time_splits, args.nday)
        datasets[key] = create_ensemble_dataset(h, time_splits)

        if cfg['type'] == 'ae_mlp':
            models[key] = load_ae_mlp_model(cfg['path'])
        elif cfg['type'] == 'catboost':
            models[key] = load_catboost_model(cfg['path'])

    # [3] Generate predictions
    print("\n[3] Generating predictions...")
    pred_dict = {}
    for key, cfg in MODEL_CONFIG.items():
        if cfg['type'] == 'ae_mlp':
            pred = predict_with_ae_mlp(models[key], datasets[key])
        else:
            pred = predict_with_catboost(models[key], datasets[key])
        pred_dict[key] = pred
        print(f"    {cfg['display']}: {len(pred)} predictions")

    # [4] Compute ensemble and uncertainty metrics
    print("\n[4] Computing ensemble predictions and uncertainty metrics...")
    ensemble_pred = ensemble_predictions(pred_dict, method='zscore_mean')
    disagreement, direction_agreement, zscore_df = compute_uncertainty_metrics(pred_dict)

    print(f"    Ensemble shape: {len(ensemble_pred)}")
    print(f"    Disagreement stats: mean={disagreement.mean():.4f}, "
          f"std={disagreement.std():.4f}, median={disagreement.median():.4f}")
    print(f"    Direction agreement distribution:")
    da_counts = direction_agreement.value_counts().sort_index()
    for level, count in da_counts.items():
        pct = count / len(direction_agreement) * 100
        print(f"      {int(level)} models agree: {count:>8d} ({pct:.1f}%)")

    # [5] Get labels
    print("\n[5] Loading test labels...")
    test_label = datasets['ae'].prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(test_label, pd.DataFrame):
        label = test_label.iloc[:, 0]
    else:
        label = test_label

    # Overall IC
    overall_ic, overall_std, overall_icir, _ = compute_ic(ensemble_pred, label)
    print(f"    Overall Ensemble IC: {overall_ic:.4f} (ICIR: {overall_icir:.4f})")

    # [6] IC by disagreement quantile
    print("\n[6] Analyzing IC by disagreement quantile (Q1=most agreement, Q4=most disagreement)...")
    ic_by_quantile = analyze_ic_by_disagreement_quantile(ensemble_pred, label, disagreement, n_groups=4)

    print("\n    +" + "=" * 65 + "+")
    print(f"    | {'Quantile':<12s} | {'Mean IC':>10s} | {'IC Std':>10s} | {'ICIR':>10s} | {'N Samples':>10s} |")
    print("    +" + "-" * 65 + "+")
    for q_label, metrics in ic_by_quantile.items():
        desc = ""
        if q_label == 'Q1':
            desc = " (agree)"
        elif q_label == 'Q4':
            desc = " (disagr)"
        print(f"    | {q_label + desc:<12s} | {metrics['ic']:>10.4f} | {metrics['ic_std']:>10.4f} | "
              f"{metrics['icir']:>10.4f} | {metrics['n_samples']:>10d} |")
    print("    +" + "=" * 65 + "+")

    # Key metric: Q1 vs Q4 IC ratio
    q1_ic = ic_by_quantile.get('Q1', {}).get('ic', 0)
    q4_ic = ic_by_quantile.get('Q4', {}).get('ic', 0)
    if q4_ic != 0:
        ic_ratio = q1_ic / q4_ic
        ic_diff_pct = (q1_ic - q4_ic) / abs(q4_ic) * 100
    else:
        ic_ratio = float('inf') if q1_ic > 0 else float('nan')
        ic_diff_pct = float('inf') if q1_ic > 0 else float('nan')

    print(f"\n    Q1 IC vs Q4 IC: {q1_ic:.4f} vs {q4_ic:.4f}")
    print(f"    IC ratio (Q1/Q4): {ic_ratio:.2f}")
    print(f"    IC difference: {ic_diff_pct:+.1f}%")

    # [7] IC by direction agreement
    print("\n[7] Analyzing IC by direction agreement level...")
    ic_by_direction = analyze_ic_by_direction_agreement(ensemble_pred, label, direction_agreement)

    print("\n    +" + "=" * 70 + "+")
    print(f"    | {'Models Agree':>12s} | {'Mean IC':>10s} | {'IC Std':>10s} | {'ICIR':>10s} | {'N Samples':>10s} |")
    print("    +" + "-" * 70 + "+")
    for level, metrics in sorted(ic_by_direction.items()):
        desc = f"{level}/4 agree"
        print(f"    | {desc:>12s} | {metrics['ic']:>10.4f} | {metrics['ic_std']:>10.4f} | "
              f"{metrics['icir']:>10.4f} | {metrics['n_samples']:>10d} |")
    print("    +" + "=" * 70 + "+")

    # [8] Temporal analysis
    print("\n[8] Analyzing temporal disagreement distribution...")
    daily_stats = analyze_temporal_disagreement(disagreement)

    print(f"    Trading days analyzed: {len(daily_stats)}")
    print(f"    Daily mean disagreement: mean={daily_stats['mean_disagree'].mean():.4f}, "
          f"std={daily_stats['mean_disagree'].std():.4f}")
    print(f"    Highest disagreement days:")
    top_5 = daily_stats.nlargest(5, 'mean_disagree')
    for date, row in top_5.iterrows():
        print(f"      {date.strftime('%Y-%m-%d')}: mean={row['mean_disagree']:.4f}, max={row['max_disagree']:.4f}")

    print(f"    Lowest disagreement days:")
    bottom_5 = daily_stats.nsmallest(5, 'mean_disagree')
    for date, row in bottom_5.iterrows():
        print(f"      {date.strftime('%Y-%m-%d')}: mean={row['mean_disagree']:.4f}, max={row['max_disagree']:.4f}")

    # [9] VIX correlation
    print("\n[9] Analyzing disagreement vs VIX correlation...")
    vix_result = analyze_disagreement_vs_vix(daily_stats)
    if vix_result is not None:
        corr, merged = vix_result
        print(f"    Correlation(mean_disagreement, VIX): {corr:.4f}")

        # High vs low VIX days
        vix_median = merged['vix'].median()
        high_vix = merged[merged['vix'] >= vix_median]['mean_disagree'].mean()
        low_vix = merged[merged['vix'] < vix_median]['mean_disagree'].mean()
        print(f"    Mean disagreement on high-VIX days (VIX >= {vix_median:.1f}): {high_vix:.4f}")
        print(f"    Mean disagreement on low-VIX days (VIX < {vix_median:.1f}): {low_vix:.4f}")

    # [10] Summary & recommendation
    print("\n" + "=" * 80)
    print("UNCERTAINTY DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"Overall Ensemble IC:  {overall_ic:.4f} (ICIR: {overall_icir:.4f})")
    print(f"Q1 IC (most agree):   {q1_ic:.4f}")
    print(f"Q4 IC (most disagr):  {q4_ic:.4f}")
    print(f"IC improvement Q1→Q4: {ic_diff_pct:+.1f}%")

    # Decision criterion: Q1 IC > Q4 IC by at least 30%
    threshold = 30.0
    if ic_diff_pct > threshold:
        verdict = "PASS"
        msg = f"Q1 IC exceeds Q4 IC by {ic_diff_pct:.1f}% (> {threshold}% threshold)"
    elif q1_ic > q4_ic and ic_diff_pct > 0:
        verdict = "MARGINAL"
        msg = f"Q1 IC exceeds Q4 IC by {ic_diff_pct:.1f}% (< {threshold}% threshold)"
    else:
        verdict = "FAIL"
        msg = f"Q1 IC does not exceed Q4 IC meaningfully"

    print(f"\nVerdict: {verdict}")
    print(f"  {msg}")
    if verdict == "PASS":
        print("  -> Proceed to Step 2: uncertainty-weighted backtest")
    elif verdict == "MARGINAL":
        print("  -> Consider proceeding with caution; signal is weak but present")
    else:
        print("  -> Uncertainty signal is not useful; abort Research J")
    print("=" * 80)


if __name__ == '__main__':
    main()
