"""
Uncertainty-Weighted Ensemble Backtest (Research J - Step 2)

Tests 3 uncertainty-aware strategies against baseline V5 ensemble:
  A. Confidence-Scaled Signal: shrink scores of high-disagreement stocks
  B. Dynamic Topk: reduce number of held positions on high-disagreement days
  C. Confidence-Filtered Topk: exclude stocks above disagreement threshold

All strategies work by adjusting the input signal to TopkDropoutStrategy —
no strategy class modifications needed.

Prerequisite: Run uncertainty_analysis.py first (Step 1) to confirm signal validity.

Usage:
    conda run -n qlib310 python scripts/models/analysis/uncertainty_backtest.py --backtest
    conda run -n qlib310 python scripts/models/analysis/uncertainty_backtest.py --backtest --topk 15
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
    run_ensemble_backtest,
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


# ── Signal adjustment strategies ─────────────────────────────────────

def strategy_a_confidence_scaled(ensemble_pred, disagreement, scale_power=1.0):
    """Strategy A: Confidence-Scaled Signal.

    adjusted_signal = ensemble_score * (1 - disagreement_quantile^scale_power)
    High disagreement -> score shrunk toward 0 -> less likely to be selected.
    """
    disagree_quantile = disagreement.groupby(level='datetime').rank(pct=True)
    confidence_weight = 1.0 - disagree_quantile ** scale_power
    adjusted = ensemble_pred * confidence_weight
    adjusted.name = 'score'
    return adjusted


def strategy_b_dynamic_topk(ensemble_pred, disagreement, topk_base=10, topk_min=5,
                            high_disagree_threshold=0.7):
    """Strategy B: Dynamic Topk.

    On days when market-level disagreement is high, effectively reduce topk
    by setting lower-ranked stocks' scores to very low values.

    This doesn't actually change the topk parameter — instead, on high-disagreement
    days we compress the signal so only the top-N stocks have normal scores.
    """
    # Market-level disagreement: median of cross-sectional disagreement per day
    market_disagree = disagreement.groupby(level='datetime').median()

    # Quantile of market-level disagreement across days
    market_disagree_quantile = market_disagree.rank(pct=True)

    # Map market disagreement to effective topk
    # High disagreement days -> fewer stocks (topk_min)
    # Low disagreement days -> normal (topk_base)
    adjusted = ensemble_pred.copy()

    for date in ensemble_pred.index.get_level_values('datetime').unique():
        day_mask = ensemble_pred.index.get_level_values('datetime') == date

        if date not in market_disagree_quantile.index:
            continue

        mkt_q = market_disagree_quantile.loc[date]
        if mkt_q > high_disagree_threshold:
            # Reduce effective positions: scale down factor
            reduction = (mkt_q - high_disagree_threshold) / (1.0 - high_disagree_threshold)
            effective_topk = int(topk_base - reduction * (topk_base - topk_min))

            # Get day's scores and only keep top effective_topk at normal levels
            day_scores = ensemble_pred[day_mask].sort_values(ascending=False)
            if len(day_scores) > effective_topk:
                cutoff_idx = day_scores.index[effective_topk:]
                adjusted.loc[cutoff_idx] = -999.0  # Effectively exclude these

    adjusted.name = 'score'
    return adjusted


def strategy_b_dynamic_topk_vectorized(ensemble_pred, disagreement, topk_base=10,
                                        topk_min=5, high_disagree_threshold=0.7):
    """Strategy B (vectorized): Dynamic Topk using groupby for better performance."""
    market_disagree = disagreement.groupby(level='datetime').median()
    market_disagree_quantile = market_disagree.rank(pct=True)

    adjusted = ensemble_pred.copy()

    # Identify high-disagreement dates
    high_dates = market_disagree_quantile[market_disagree_quantile > high_disagree_threshold]

    for date, mkt_q in high_dates.items():
        reduction = (mkt_q - high_disagree_threshold) / (1.0 - high_disagree_threshold)
        effective_topk = int(topk_base - reduction * (topk_base - topk_min))

        day_mask = adjusted.index.get_level_values('datetime') == date
        day_scores = adjusted[day_mask].sort_values(ascending=False)

        if len(day_scores) > effective_topk:
            cutoff_idx = day_scores.index[effective_topk:]
            adjusted.loc[cutoff_idx] = -999.0

    adjusted.name = 'score'
    return adjusted


def strategy_c_confidence_filtered(ensemble_pred, disagreement, filter_quantile=0.75):
    """Strategy C: Confidence-Filtered Topk.

    Exclude stocks whose disagreement exceeds the filter_quantile threshold
    by setting their scores to very low values.
    """
    disagree_quantile = disagreement.groupby(level='datetime').rank(pct=True)
    adjusted = ensemble_pred.copy()

    # Stocks above the disagreement threshold get excluded
    high_disagree_mask = disagree_quantile > filter_quantile
    adjusted[high_disagree_mask] = -999.0

    adjusted.name = 'score'
    return adjusted


# ── Lightweight backtest comparison ──────────────────────────────────

def run_comparison_backtest(strategy_preds, args, time_splits):
    """Run backtest for multiple strategy variants and collect results.

    Args:
        strategy_preds: dict of {strategy_name: adjusted_pred_series}
        args: parsed arguments with backtest config
        time_splits: time period configuration

    Returns:
        dict of {strategy_name: portfolio_metric_dict}
    """
    results = {}
    for name, pred in strategy_preds.items():
        print(f"\n{'=' * 70}")
        print(f"BACKTEST: {name}")
        print(f"{'=' * 70}")
        portfolio_metrics = run_ensemble_backtest(
            pred, args, time_splits, model_name=name,
        )
        results[name] = portfolio_metrics
    return results


def extract_backtest_metrics(portfolio_metric_dict):
    """Extract key metrics from backtest results for comparison."""
    if portfolio_metric_dict is None:
        return {'total_return': np.nan, 'sharpe': np.nan, 'max_drawdown': np.nan,
                'annualized_return': np.nan, 'calmar': np.nan}

    from qlib.contrib.evaluate import risk_analysis

    for freq, (report_df, positions) in portfolio_metric_dict.items():
        has_bench = "bench" in report_df.columns and not report_df["bench"].isna().all()
        if has_bench:
            ret_series = report_df["return"] - report_df["bench"]
        else:
            ret_series = report_df["return"]

        analysis = risk_analysis(ret_series, freq=freq)

        total_return = (report_df["return"] + 1).prod() - 1
        if has_bench:
            bench_return = (report_df["bench"] + 1).prod() - 1
            excess_return = total_return - bench_return
        else:
            excess_return = total_return

        metrics = {
            'total_return': total_return,
            'excess_return': excess_return,
        }

        if analysis is not None and not analysis.empty:
            for metric_name, value in analysis.items():
                if isinstance(value, (int, float)):
                    metrics[metric_name.lower().replace(' ', '_')] = value

        return metrics

    return {}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Uncertainty-Weighted Ensemble Backtest (Research J - Step 2)',
    )
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--backtest', action='store_true',
                        help='Run backtest (required for full comparison)')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=2)
    parser.add_argument('--account', type=float, default=100000)
    parser.add_argument('--rebalance-freq', type=int, default=5)
    parser.add_argument('--strategy', type=str, default='topk')

    # Strategy A params
    parser.add_argument('--scale-power', type=float, default=1.0,
                        help='Strategy A: exponent for disagreement quantile (default: 1.0)')

    # Strategy B params
    parser.add_argument('--topk-min', type=int, default=5,
                        help='Strategy B: minimum topk on high-disagreement days (default: 5)')
    parser.add_argument('--high-disagree-threshold', type=float, default=0.7,
                        help='Strategy B: market disagreement threshold (default: 0.7)')

    # Strategy C params
    parser.add_argument('--filter-quantile', type=float, default=0.75,
                        help='Strategy C: disagreement quantile threshold for filtering (default: 0.75)')

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
    print("UNCERTAINTY-WEIGHTED ENSEMBLE BACKTEST (Research J - Step 2)")
    print("=" * 80)
    print(f"Stock Pool: {args.stock_pool}")
    print(f"Test Period: {time_splits['test_start']} to {time_splits['test_end']}")
    print(f"Topk: {args.topk}, N_drop: {args.n_drop}, Rebalance: {args.rebalance_freq}d")
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
        print(f"    {cfg['display']}: {cfg['handler']}...")
        h = create_ensemble_data_handler(cfg['handler'], symbols, time_splits, args.nday)
        datasets[key] = create_ensemble_dataset(h, time_splits)

        if cfg['type'] == 'ae_mlp':
            models[key] = load_ae_mlp_model(cfg['path'])
        else:
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

    # [4] Compute baseline ensemble and uncertainty
    print("\n[4] Computing ensemble and uncertainty metrics...")
    ensemble_pred = ensemble_predictions(pred_dict, method='zscore_mean')

    # Compute disagreement
    names = list(pred_dict.keys())
    common_idx = pred_dict[names[0]].index
    for name in names[1:]:
        common_idx = common_idx.intersection(pred_dict[name].index)

    zscores = {name: zscore_by_day(pred_dict[name].loc[common_idx]) for name in names}
    zscore_df = pd.DataFrame(zscores)
    disagreement = zscore_df.std(axis=1)
    disagreement.name = 'disagreement'

    print(f"    Ensemble: {len(ensemble_pred)} predictions")
    print(f"    Disagreement: mean={disagreement.mean():.4f}, std={disagreement.std():.4f}")

    # [5] IC comparison
    print("\n[5] Loading labels and computing IC...")
    test_label = datasets['ae'].prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(test_label, pd.DataFrame):
        label = test_label.iloc[:, 0]
    else:
        label = test_label

    # [6] Generate strategy variants
    print("\n[6] Generating strategy variants...")

    strategy_preds = {}

    # Baseline
    strategy_preds['Baseline'] = ensemble_pred

    # Strategy A: Confidence-Scaled
    pred_a = strategy_a_confidence_scaled(ensemble_pred, disagreement,
                                          scale_power=args.scale_power)
    strategy_preds['A_ConfScaled'] = pred_a
    print(f"    Strategy A (Confidence-Scaled, power={args.scale_power}):")
    print(f"      Score range: [{pred_a.min():.4f}, {pred_a.max():.4f}]")

    # Strategy B: Dynamic Topk
    pred_b = strategy_b_dynamic_topk_vectorized(
        ensemble_pred, disagreement,
        topk_base=args.topk, topk_min=args.topk_min,
        high_disagree_threshold=args.high_disagree_threshold,
    )
    n_suppressed = (pred_b == -999.0).sum()
    n_total = len(pred_b)
    strategy_preds['B_DynTopk'] = pred_b
    print(f"    Strategy B (Dynamic Topk, base={args.topk}, min={args.topk_min}):")
    print(f"      Suppressed {n_suppressed}/{n_total} stock-days ({n_suppressed/n_total*100:.1f}%)")

    # Strategy C: Confidence-Filtered
    pred_c = strategy_c_confidence_filtered(ensemble_pred, disagreement,
                                            filter_quantile=args.filter_quantile)
    n_filtered = (pred_c == -999.0).sum()
    strategy_preds['C_ConfFilter'] = pred_c
    print(f"    Strategy C (Confidence-Filtered, quantile={args.filter_quantile}):")
    print(f"      Filtered {n_filtered}/{n_total} stock-days ({n_filtered/n_total*100:.1f}%)")

    # [7] IC comparison across strategies
    print("\n[7] IC comparison across strategies...")
    print("\n    +" + "=" * 65 + "+")
    print(f"    | {'Strategy':<20s} | {'Mean IC':>10s} | {'IC Std':>10s} | {'ICIR':>10s} |")
    print("    +" + "-" * 65 + "+")
    for name, pred_s in strategy_preds.items():
        # For strategies with -999 values, only compare valid scores
        if (pred_s == -999.0).any():
            valid_mask = pred_s != -999.0
            ic, ic_std, icir, _ = compute_ic(pred_s[valid_mask], label)
        else:
            ic, ic_std, icir, _ = compute_ic(pred_s, label)
        print(f"    | {name:<20s} | {ic:>10.4f} | {ic_std:>10.4f} | {icir:>10.4f} |")
    print("    +" + "=" * 65 + "+")

    # [8] Backtest
    if not args.backtest:
        print("\n[8] Skipping backtest (use --backtest to run)")
        print("    IC comparison above should give preliminary signal quality assessment.")
        return

    print("\n[8] Running backtests for all strategies...")
    all_bt_results = {}
    for name, pred_s in strategy_preds.items():
        print(f"\n{'─' * 70}")
        print(f"Running backtest: {name}")
        print(f"{'─' * 70}")
        bt_result = run_ensemble_backtest(
            pred_s, args, time_splits, model_name=f"UncJ_{name}",
        )
        all_bt_results[name] = bt_result

    # [9] Compare backtest results
    print("\n" + "=" * 80)
    print("BACKTEST COMPARISON SUMMARY")
    print("=" * 80)

    comparison = {}
    for name, bt_result in all_bt_results.items():
        metrics = extract_backtest_metrics(bt_result)
        comparison[name] = metrics

    if comparison:
        comp_df = pd.DataFrame(comparison).T
        # Show key columns if available
        key_cols = ['total_return', 'excess_return', 'annualized_return',
                    'information_ratio', 'max_drawdown']
        available_cols = [c for c in key_cols if c in comp_df.columns]
        if available_cols:
            print("\n    Key Metrics:")
            print("    " + "-" * 70)
            for col in available_cols:
                print(f"    {col:<25s}", end="")
                for name in comparison:
                    val = comparison[name].get(col, np.nan)
                    if not np.isnan(val):
                        print(f" | {name}: {val:>8.4f}", end="")
                print()
            print("    " + "-" * 70)

        # Full comparison table
        print("\n    Full Comparison:")
        print(comp_df.to_string())

    # [10] Final verdict
    print("\n" + "=" * 80)
    print("UNCERTAINTY BACKTEST VERDICT")
    print("=" * 80)

    baseline_metrics = comparison.get('Baseline', {})
    best_strategy = None
    best_improvement = -float('inf')

    # Compare using excess_return or total_return
    baseline_ret = baseline_metrics.get('excess_return',
                                        baseline_metrics.get('total_return', 0))

    for name, metrics in comparison.items():
        if name == 'Baseline':
            continue
        strat_ret = metrics.get('excess_return', metrics.get('total_return', 0))
        if not np.isnan(strat_ret) and not np.isnan(baseline_ret):
            improvement = strat_ret - baseline_ret
            print(f"  {name}: excess return delta = {improvement:+.4f}")
            if improvement > best_improvement:
                best_improvement = improvement
                best_strategy = name

    if best_strategy and best_improvement > 0:
        print(f"\n  Best strategy: {best_strategy} (improvement: {best_improvement:+.4f})")
        print("  -> Consider integrating into production ensemble")
    else:
        print(f"\n  No strategy improved over baseline")
        print("  -> Uncertainty-based position sizing does not help for this ensemble")

    print("=" * 80)


if __name__ == '__main__':
    main()
