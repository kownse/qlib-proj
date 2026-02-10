#!/usr/bin/env python
"""
Daily Trading Script (Ensemble V3) - 3x AE-MLP + CatBoost (4-model ensemble)

Extends V2 by adding AE-MLP with Market-Neutral target (v9-mkt-neutral).
Also adds data pipeline for SPY forward returns and CBOE data required
by the market-neutral handler.

Flow:
1. Download latest US stock data (download_us_data_to_date.py)
2. Incremental macro data update (download_macro_data_to_date.py)
3. Process macro data into features (process_macro_data.py)
4. Download SPY forward returns (download_spy_forward_returns.py)
5. Download CBOE data (download_cboe_data.py)
6. Process CBOE data (process_cboe_data.py)
7. Load AE-MLP v7, AE-MLP v9, AE-MLP mkt-neutral, CatBoost models
8. Generate predictions and calculate correlations
9. 4-model ensemble prediction (zscore_weighted)
10. Run backtest or live prediction, output daily trading info

Usage:
    python scripts/run_daily_trading_ensemble_v3.py
    python scripts/run_daily_trading_ensemble_v3.py --skip-download
    python scripts/run_daily_trading_ensemble_v3.py --predict-only
    python scripts/run_daily_trading_ensemble_v3.py --exclude-mkt-neutral
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pickle

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

from daily_trading_common import (
    PROJECT_ROOT,
    SCRIPTS_DIR,
    send_email,
    format_trading_details_html,
    format_trading_details_text,
    get_latest_data_date,
    get_latest_calendar_dates,
    download_data,
    init_qlib_for_talib,
    create_dataset_for_trading,
    collect_trading_details_from_positions,
    print_trading_details,
    run_command,
)


# ============================================================================
# Model Configuration (matching run_ae_cb_ensemble_v2.py)
# ============================================================================

MODEL_CONFIGS = {
    'AE-MLP-v7': {
        'model_path': 'my_models/ae_mlp_cv_alpha158-enhanced-v7_sp500_5d_best.keras',
        'handler': 'alpha158-enhanced-v7',
        'type': 'ae_mlp',
        'default_weight': 0.400,
    },
    'AE-MLP-v9': {
        'model_path': 'my_models/ae_mlp_cv_alpha158-enhanced-v9_test_5d_best.keras',
        'handler': 'alpha158-enhanced-v9',
        'type': 'ae_mlp',
        'default_weight': 0.250,
    },
    'AE-MLP-mkt-neutral': {
        'model_path': 'my_models/ae_mlp_cv_v9-mkt-neutral_test_5d_20260210_151953.keras',
        'handler': 'v9-mkt-neutral',
        'type': 'ae_mlp',
        'default_weight': 0.150,
    },
    'CatBoost': {
        'model_path': 'my_models/catboost_cv_catboost-v1_test_5d_20260129_105915_best.cbm',
        'handler': 'catboost-v1',
        'type': 'catboost',
        'default_weight': 0.200,
    },
}


# ============================================================================
# Data Pipeline
# ============================================================================

def download_data_v3(stock_pool: str = "sp500"):
    """Extended data download pipeline including SPY forward returns and CBOE data."""
    # Step 1-3: Standard data downloads (stock, macro)
    download_data(stock_pool)

    # Step 4: Download SPY forward returns (needed by mkt-neutral handler)
    if not run_command(
        [sys.executable, str(SCRIPTS_DIR / "data" / "download_spy_forward_returns.py")],
        "Downloading SPY forward returns"
    ):
        print("Warning: SPY forward returns download had issues, continuing...")

    # Step 5: Download CBOE data (SKEW, VVIX, VIX9D)
    if not run_command(
        [sys.executable, str(SCRIPTS_DIR / "data" / "download_cboe_data.py")],
        "Downloading CBOE data"
    ):
        print("Warning: CBOE data download had issues, continuing...")

    # Step 6: Process CBOE data into features
    if not run_command(
        [sys.executable, str(SCRIPTS_DIR / "data" / "process_cboe_data.py")],
        "Processing CBOE data into features"
    ):
        print("Warning: CBOE data processing had issues, continuing...")


# ============================================================================
# Model Loading & Prediction
# ============================================================================

def load_model_meta(model_path: Path) -> dict:
    """Load model metadata."""
    meta_path = model_path.with_suffix('.meta.pkl')
    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            return pickle.load(f)

    stem = model_path.stem
    if stem.endswith('_best'):
        alt_meta_path = model_path.parent / (stem[:-5] + '.meta.pkl')
        if alt_meta_path.exists():
            with open(alt_meta_path, 'rb') as f:
                return pickle.load(f)

    return {}


def load_ae_mlp_model(model_path: Path):
    """Load AE-MLP model."""
    from models.deep.ae_mlp_model import AEMLP

    print(f"    Loading AE-MLP model from: {model_path}")
    model = AEMLP.load(str(model_path))
    print(f"    AE-MLP loaded: {model.num_columns} features")
    return model


def load_catboost_model(model_path: Path):
    """Load CatBoost model."""
    print(f"    Loading CatBoost model from: {model_path}")
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    feature_count = model.feature_names_ if model.feature_names_ else "N/A"
    if isinstance(feature_count, list):
        feature_count = len(feature_count)
    print(f"    CatBoost loaded: {feature_count} features")
    return model


def predict_with_ae_mlp(model, dataset) -> pd.Series:
    """Generate predictions with AE-MLP model."""
    pred = model.predict(dataset, segment="test")
    pred.name = 'score'
    return pred


def predict_with_catboost(model, dataset) -> pd.Series:
    """Generate predictions with CatBoost model."""
    from qlib.data.dataset.handler import DataHandlerLP

    test_data = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_L)
    test_data = test_data.fillna(0).replace([np.inf, -np.inf], 0)

    pred_values = model.predict(test_data.values)
    pred = pd.Series(pred_values, index=test_data.index, name='score')
    return pred


# ============================================================================
# Ensemble Utilities
# ============================================================================

def calculate_pairwise_correlations(preds: dict) -> pd.DataFrame:
    """Calculate pairwise correlations between model predictions."""
    model_names = list(preds.keys())
    n_models = len(model_names)

    common_idx = preds[model_names[0]].index
    for name in model_names[1:]:
        common_idx = common_idx.intersection(preds[name].index)

    corr_matrix = np.zeros((n_models, n_models))
    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                p_i = preds[name_i].loc[common_idx]
                p_j = preds[name_j].loc[common_idx]
                corr_matrix[i, j] = p_i.corr(p_j)

    return pd.DataFrame(corr_matrix, index=model_names, columns=model_names)


def ensemble_predictions_multi(preds: dict, method: str = 'zscore_weighted',
                               weights: dict = None) -> pd.Series:
    """
    Ensemble multiple model predictions.

    Parameters
    ----------
    preds : dict
        Dict of {model_name: prediction_series}
    method : str
        'mean', 'weighted', 'rank_mean', 'zscore_mean', 'zscore_weighted'
    weights : dict, optional
        {model_name: weight} for weighted ensemble
    """
    model_names = list(preds.keys())

    common_idx = preds[model_names[0]].index
    for name in model_names[1:]:
        common_idx = common_idx.intersection(preds[name].index)

    aligned = {name: preds[name].loc[common_idx] for name in model_names}

    def zscore_by_day(x):
        mean = x.groupby(level='datetime').transform('mean')
        std = x.groupby(level='datetime').transform('std')
        return (x - mean) / (std + 1e-8)

    if method == 'mean':
        ensemble_pred = sum(aligned[name] for name in model_names) / len(model_names)

    elif method == 'weighted':
        if weights is None:
            weights = {name: 1.0 / len(model_names) for name in model_names}
        total = sum(weights.values())
        ensemble_pred = sum(aligned[name] * weights[name] for name in model_names) / total

    elif method == 'rank_mean':
        ranks = {name: aligned[name].groupby(level='datetime').rank(pct=True) for name in model_names}
        ensemble_pred = sum(ranks[name] for name in model_names) / len(model_names)

    elif method == 'zscore_mean':
        zscores = {name: zscore_by_day(aligned[name]) for name in model_names}
        ensemble_pred = sum(zscores[name] for name in model_names) / len(model_names)

    elif method == 'zscore_weighted':
        if weights is None:
            weights = {name: 1.0 / len(model_names) for name in model_names}
        zscores = {name: zscore_by_day(aligned[name]) for name in model_names}
        total = sum(weights.values())
        ensemble_pred = sum(zscores[name] * weights[name] for name in model_names) / total

    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    ensemble_pred.name = 'score'
    return ensemble_pred


# ============================================================================
# Trading Functions
# ============================================================================

def run_live_prediction(
    pred_ensemble: pd.Series,
    preds: dict,
    stock_pool: str,
    topk: int = 10,
    account: float = 8000,
):
    """Generate live trading predictions (no backtest)."""
    print(f"\n{'='*70}")
    print("LIVE TRADING PREDICTIONS (Ensemble V3: 3x AE-MLP + CatBoost)")
    print(f"{'='*70}")
    print(f"Account: ${account:,.2f}")
    print(f"Top-K stocks: {topk}")

    pred_df = pred_ensemble.to_frame("score")
    pred_df = pred_df.reset_index()
    pred_df['instrument'] = pred_df['instrument'].str.lower()
    pred_df = pred_df.set_index(['datetime', 'instrument'])

    pred_dates = pred_df.index.get_level_values(0).unique().sort_values()
    print(f"\nPredictions available for: {pred_dates.min().strftime('%Y-%m-%d')} to {pred_dates.max().strftime('%Y-%m-%d')}")
    print(f"Total dates: {len(pred_dates)}, Total predictions: {len(pred_df)}")

    latest_date = pred_dates[-1]
    latest_preds = pred_df.loc[latest_date].sort_values("score", ascending=False)

    print(f"\n{'='*70}")
    print(f"TRADING SIGNALS FOR: {latest_date.strftime('%Y-%m-%d')}")
    print(f"{'='*70}")

    print(f"\nTOP {topk} STOCKS TO BUY (highest predicted return):")
    print("-" * 60)
    print(f"{'Rank':<6} {'Symbol':<10} {'Score':>12} {'Est. Allocation':>18}")
    print("-" * 60)

    allocation_per_stock = account * 0.95 / topk
    top_stocks = latest_preds.head(topk)

    for i, (stock, row) in enumerate(top_stocks.iterrows(), 1):
        print(f"{i:<6} {stock.upper():<10} {row['score']:>12.4f} ${allocation_per_stock:>15,.2f}")

    print(f"\nTOP {topk} STOCKS TO AVOID (lowest predicted return):")
    print("-" * 60)
    bottom_stocks = latest_preds.tail(topk).iloc[::-1]
    for i, (stock, row) in enumerate(bottom_stocks.iterrows(), 1):
        print(f"{i:<6} {stock.upper():<10} {row['score']:>12.4f}")

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_output = latest_preds.copy()
    pred_output.index = pred_output.index.str.upper()
    pred_output = pred_output.sort_values("score", ascending=False)
    pred_path = output_dir / f"ensemble_v3_predictions_{latest_date.strftime('%Y%m%d')}.csv"
    pred_output.to_csv(pred_path)
    print(f"\nFull predictions saved to: {pred_path}")

    recommendations = []
    for i, (stock, row) in enumerate(top_stocks.iterrows(), 1):
        recommendations.append({
            'rank': i,
            'symbol': stock.upper(),
            'action': 'BUY',
            'score': row['score'],
            'suggested_allocation': allocation_per_stock,
        })

    rec_df = pd.DataFrame(recommendations)
    rec_path = output_dir / f"ensemble_v3_recommendations_{latest_date.strftime('%Y%m%d')}.csv"
    rec_df.to_csv(rec_path, index=False)
    print(f"Buy recommendations saved to: {rec_path}")

    print(f"\n{'='*70}")
    print("LIVE PREDICTION COMPLETED")
    print(f"{'='*70}")


def run_trading_backtest(
    pred_ensemble: pd.Series,
    stock_pool: str,
    test_start: str,
    test_end: str,
    account: float = 8000,
    topk: int = 5,
    n_drop: int = 1,
    rebalance_freq: int = 5,
) -> list:
    """Run trading backtest, return trading details list."""
    from qlib.backtest import backtest as qlib_backtest
    from qlib.contrib.evaluate import risk_analysis
    from data.stock_pools import STOCK_POOLS
    from utils.strategy import get_strategy_config

    print(f"\n{'='*70}")
    print("TRADING BACKTEST (Ensemble V3: 3x AE-MLP + CatBoost)")
    print(f"{'='*70}")
    print(f"Period: {test_start} to {test_end}")
    print(f"Initial Account: ${account:,.2f}")
    print(f"Rebalance Frequency: every {rebalance_freq} day(s)")
    print(f"Top-K stocks: {topk}")
    print(f"N-drop: {n_drop}")

    pred_df = pred_ensemble.to_frame("score")
    pred_df = pred_df.reset_index()
    pred_df['instrument'] = pred_df['instrument'].str.lower()
    pred_df = pred_df.set_index(['datetime', 'instrument'])

    pred_dates = pred_df.index.get_level_values(0).unique().sort_values()
    print(f"\nPredictions shape: {len(pred_df)}")
    print(f"Date range: {pred_dates.min()} to {pred_dates.max()}")
    print(f"Unique dates: {len(pred_dates)}")

    latest_pred_date = pred_dates[-1]
    print(f"\n{'='*70}")
    print(f"TODAY'S RECOMMENDATIONS (based on {latest_pred_date.strftime('%Y-%m-%d')} predictions)")
    print(f"{'='*70}")

    latest_preds = pred_df.loc[latest_pred_date].sort_values("score", ascending=False)
    print(f"\nTop {topk} stocks to BUY:")
    print("-" * 50)
    for i, (stock, row) in enumerate(latest_preds.head(topk).iterrows(), 1):
        print(f"  {i:2d}. {stock.upper():<8s}  Score: {row['score']:>8.4f}")

    if len(pred_dates) > 0:
        actual_test_end = pred_dates[-1].strftime("%Y-%m-%d")
        actual_test_start = pred_dates[0].strftime("%Y-%m-%d")

        if pd.Timestamp(test_start) > pred_dates[0]:
            actual_test_start = test_start

        if actual_test_start != test_start or actual_test_end != test_end:
            print(f"\n[NOTE] Adjusting backtest period:")
            print(f"       Original: {test_start} to {test_end}")
            print(f"       Adjusted: {actual_test_start} to {actual_test_end}")
            test_start = actual_test_start
            test_end = actual_test_end

    strategy_config = get_strategy_config(
        pred_df, topk, n_drop, rebalance_freq=rebalance_freq,
        strategy_type="topk"
    )

    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    }

    pool_symbols = STOCK_POOLS[stock_pool]
    pool_symbols_lower = [s.lower() for s in pool_symbols]

    backtest_config = {
        "start_time": test_start,
        "end_time": test_end,
        "account": account,
        "benchmark": "SPY",
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": None,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0005,
            "min_cost": 1,
            "trade_unit": None,
            "codes": pool_symbols_lower,
        },
    }

    print(f"\n[*] Running backtest...")
    portfolio_metric_dict, indicator_dict = qlib_backtest(
        executor=executor_config,
        strategy=strategy_config,
        **backtest_config
    )

    print("    Backtest completed!")

    for freq, (report_df, positions) in portfolio_metric_dict.items():
        print(f"\n{'='*70}")
        print(f"TRADING RESULTS ({freq})")
        print(f"{'='*70}")

        total_return = (report_df["return"] + 1).prod() - 1
        final_value = account * (1 + total_return)

        print(f"\nPerformance Summary:")
        print(f"  Initial Capital:  ${account:>12,.2f}")
        print(f"  Final Value:      ${final_value:>12,.2f}")
        print(f"  Total Return:     {total_return:>12.2%}")
        print(f"  Total P&L:        ${final_value - account:>12,.2f}")

        if "bench" in report_df.columns and not report_df["bench"].isna().all():
            bench_return = (report_df["bench"] + 1).prod() - 1
            excess_return = total_return - bench_return
            print(f"  Benchmark Return: {bench_return:>12.2%}")
            print(f"  Excess Return:    {excess_return:>12.2%}")

        analysis = risk_analysis(report_df["return"], freq=freq)
        if analysis is not None and not analysis.empty:
            print(f"\nRisk Metrics:")
            for metric, value in analysis.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric:<20s}: {value:>10.4f}")

        print(f"\nDaily Statistics:")
        print(f"  Trading Days:     {len(report_df):>12d}")
        print(f"  Mean Daily Return:{report_df['return'].mean():>12.4%}")
        print(f"  Std Daily Return: {report_df['return'].std():>12.4%}")
        print(f"  Best Day:         {report_df['return'].max():>12.4%}")
        print(f"  Worst Day:        {report_df['return'].min():>12.4%}")

        print(f"\n{'='*70}")
        print("DAILY TRADING DETAILS (Last 10 days)")
        print(f"{'='*70}")

        all_trading_details = collect_trading_details_from_positions(positions, report_df)
        print_trading_details(all_trading_details, show_last_n=10)

        output_dir = PROJECT_ROOT / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / f"ensemble_v3_daily_trading_report_{freq}.csv"
        report_df.to_csv(report_path)
        print(f"\n\nReport saved to: {report_path}")

        return all_trading_details

    return []


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Daily trading script (Ensemble V3) - 3x AE-MLP + CatBoost',
    )
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip data download step')
    parser.add_argument('--predict-only', action='store_true',
                        help='Only generate predictions, skip backtest')

    # Model exclusion flags
    parser.add_argument('--exclude-mkt-neutral', action='store_true',
                        help='Exclude mkt-neutral model (run 3-model ensemble)')
    parser.add_argument('--exclude-v7', action='store_true',
                        help='Exclude AE-MLP v7')
    parser.add_argument('--exclude-v9', action='store_true',
                        help='Exclude AE-MLP v9')
    parser.add_argument('--exclude-catboost', action='store_true',
                        help='Exclude CatBoost')

    # Ensemble parameters
    parser.add_argument('--ensemble-method', type=str, default='zscore_weighted',
                        choices=['mean', 'weighted', 'rank_mean', 'zscore_mean', 'zscore_weighted'],
                        help='Ensemble method (default: zscore_weighted)')

    # Trading parameters
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'],
                        help='Stock pool')
    parser.add_argument('--account', type=float, default=8000,
                        help='Initial account balance')
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of stocks to hold')
    parser.add_argument('--n-drop', type=int, default=2,
                        help='Number of stocks to drop each rebalance')
    parser.add_argument('--rebalance-freq', type=int, default=5,
                        help='Rebalance frequency in days')
    parser.add_argument('--backtest-start', type=str, default='2026-02-01',
                        help='Backtest start date')
    parser.add_argument('--test-end', type=str, default=None,
                        help='Backtest end date')
    parser.add_argument('--nday', type=int, default=5,
                        help='N-day forward prediction')
    parser.add_argument('--feature-lookback', type=int, default=5,
                        help='Days before backtest-start for feature calculation')

    # Email parameters
    parser.add_argument('--send-email', action='store_true',
                        help='Send trading report via email')
    parser.add_argument('--email-to', type=str, default='kownse@gmail.com',
                        help='Email recipient (default: kownse@gmail.com)')
    parser.add_argument('--email-days', type=int, default=5,
                        help='Number of recent days to include in email (default: 5)')

    args = parser.parse_args()

    # Build active model list with exclusion support
    active_models = {}
    exclude_map = {
        'AE-MLP-v7': args.exclude_v7,
        'AE-MLP-v9': args.exclude_v9,
        'AE-MLP-mkt-neutral': args.exclude_mkt_neutral,
        'CatBoost': args.exclude_catboost,
    }
    for name, config in MODEL_CONFIGS.items():
        if exclude_map.get(name, False):
            continue
        active_models[name] = config.copy()

    if len(active_models) < 2:
        print("Error: Need at least 2 models for ensemble")
        sys.exit(1)

    model_names = list(active_models.keys())
    n_models = len(active_models)

    # Determine if mkt-neutral is active (need extra data downloads)
    has_mkt_neutral = 'AE-MLP-mkt-neutral' in active_models

    print("=" * 70)
    mode = "LIVE PREDICTION MODE" if args.predict_only else "BACKTEST MODE"
    print(f"DAILY TRADING SCRIPT (ENSEMBLE V3: {n_models} Models) - {mode}")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nActive Models:")
    for name, config in active_models.items():
        print(f"  {name:<22s}: {config['model_path']} (handler: {config['handler']}, weight: {config['default_weight']:.3f})")
    print(f"\nEnsemble Method: {args.ensemble_method}")
    print(f"Stock Pool: {args.stock_pool}")
    print(f"Account: ${args.account:,.2f}")
    print(f"Top-K: {args.topk}")
    print(f"Backtest Start: {args.backtest_start}")
    print("=" * 70)

    # Step 1: Download data
    if not args.skip_download:
        if has_mkt_neutral:
            # Extended pipeline with SPY forward returns and CBOE data
            download_data_v3(args.stock_pool)
        else:
            # Standard pipeline (no mkt-neutral data needed)
            download_data(args.stock_pool)
    else:
        print("\n[SKIP] Data download skipped")

    # Detect date range
    print(f"\n{'='*60}")
    print("[STEP] Detecting data date range")
    print(f"{'='*60}")
    latest_data_str = get_latest_data_date()
    latest_calendar_str, usable_calendar_str = get_latest_calendar_dates()
    latest_data = datetime.strptime(latest_data_str, "%Y-%m-%d")
    usable_calendar = datetime.strptime(usable_calendar_str, "%Y-%m-%d")

    print(f"Latest available data date: {latest_data_str}")
    print(f"Usable calendar date: {usable_calendar_str}")

    max_backtest_date = min(latest_data, usable_calendar)
    max_backtest_str = max_backtest_date.strftime("%Y-%m-%d")

    backtest_start_date = datetime.strptime(args.backtest_start, "%Y-%m-%d")

    if backtest_start_date > max_backtest_date:
        print(f"\nWARNING: Adjusting backtest start to: {max_backtest_str}")
        backtest_start_date = max_backtest_date
        args.backtest_start = max_backtest_str

    data_start_date = backtest_start_date - timedelta(days=args.feature_lookback)
    args.test_start = data_start_date.strftime("%Y-%m-%d")

    if args.test_end is None:
        args.test_end = max_backtest_str

    test_end_date = datetime.strptime(args.test_end, "%Y-%m-%d")
    if test_end_date > max_backtest_date:
        args.test_end = max_backtest_str

    print(f"\nFinal date settings:")
    print(f"  Data start: {args.test_start}")
    print(f"  Backtest start: {args.backtest_start}")
    print(f"  Backtest end: {args.test_end}")

    # Step 2: Initialize Qlib
    print(f"\n{'='*60}")
    print("[STEP] Initializing Qlib")
    print(f"{'='*60}")
    init_qlib_for_talib()

    # Step 3: Check model files
    print(f"\n{'='*60}")
    print("[STEP] Checking model files")
    print(f"{'='*60}")

    for name, config in active_models.items():
        path = PROJECT_ROOT / config['model_path']
        if not path.exists():
            print(f"Error: {name} model not found: {path}")
            sys.exit(1)

    # Load metadata and override handlers if available
    for name, config in active_models.items():
        path = PROJECT_ROOT / config['model_path']
        meta = load_model_meta(path)
        if meta and 'handler' in meta:
            config['handler'] = meta['handler']
            print(f"    {name} handler from metadata: {config['handler']}")

    # Step 4: Create datasets (one per unique handler)
    print(f"\n{'='*60}")
    print("[STEP] Creating datasets")
    print(f"{'='*60}")

    datasets = {}
    for name, config in active_models.items():
        print(f"\n  Creating {config['handler']} dataset for {name}...")
        datasets[name] = create_dataset_for_trading(
            config['handler'], args.stock_pool,
            args.test_start, args.test_end, args.nday,
            verbose=False
        )

    # Step 5: Load models
    print(f"\n{'='*60}")
    print("[STEP] Loading models")
    print(f"{'='*60}")

    models = {}
    for name, config in active_models.items():
        path = PROJECT_ROOT / config['model_path']
        if config['type'] == 'ae_mlp':
            models[name] = load_ae_mlp_model(path)
        elif config['type'] == 'catboost':
            models[name] = load_catboost_model(path)

    # Step 6: Generate predictions
    print(f"\n{'='*60}")
    print("[STEP] Generating predictions")
    print(f"{'='*60}")

    preds = {}
    for name, config in active_models.items():
        print(f"\n  {name} predictions...")
        if config['type'] == 'ae_mlp':
            pred = predict_with_ae_mlp(models[name], datasets[name])
        elif config['type'] == 'catboost':
            pred = predict_with_catboost(models[name], datasets[name])
        preds[name] = pred
        print(f"    Shape: {len(pred)}, Range: [{pred.min():.4f}, {pred.max():.4f}]")

    # Step 7: Calculate correlations
    print(f"\n{'='*60}")
    print("[STEP] Calculating pairwise correlations")
    print(f"{'='*60}")

    corr_matrix = calculate_pairwise_correlations(preds)
    print("\n  Prediction Correlation Matrix:")
    print("  " + "=" * 50)
    print(corr_matrix.to_string())
    print("  " + "=" * 50)

    # Step 8: Ensemble predictions
    print(f"\n{'='*60}")
    print(f"[STEP] Ensembling predictions ({args.ensemble_method})")
    print(f"{'='*60}")

    # Use default weights from config, re-normalize for active models
    weights = {name: active_models[name]['default_weight'] for name in model_names}
    total_w = sum(weights.values())
    print(f"  Weights (normalized):")
    for name in model_names:
        print(f"    {name:<22s}: {weights[name]/total_w:.3f}")

    pred_ensemble = ensemble_predictions_multi(preds, args.ensemble_method, weights)
    print(f"  Ensemble shape: {len(pred_ensemble)}")
    print(f"  Range: [{pred_ensemble.min():.4f}, {pred_ensemble.max():.4f}]")

    # Step 9: Run prediction or backtest
    trading_details = []

    if args.predict_only:
        run_live_prediction(
            pred_ensemble=pred_ensemble,
            preds=preds,
            stock_pool=args.stock_pool,
            topk=args.topk,
            account=args.account,
        )
    else:
        trading_details = run_trading_backtest(
            pred_ensemble=pred_ensemble,
            stock_pool=args.stock_pool,
            test_start=args.backtest_start,
            test_end=args.test_end,
            account=args.account,
            topk=args.topk,
            n_drop=args.n_drop,
            rebalance_freq=args.rebalance_freq,
        )

    print(f"\n{'='*70}")
    print("ENSEMBLE V3 TRADING SCRIPT COMPLETED")
    print(f"{'='*70}")

    # Step 10: Send email report
    if args.send_email and trading_details:
        print(f"\n{'='*60}")
        print("[STEP] Sending email report")
        print(f"{'='*60}")

        recent_details = trading_details[-args.email_days:] if len(trading_details) > args.email_days else trading_details

        if recent_details:
            model_info = f"3x AE-MLP + CatBoost ({args.ensemble_method})"
            text_body = format_trading_details_text(recent_details, model_info)
            html_body = format_trading_details_html(recent_details, model_info)

            start_date = recent_details[0]['date']
            end_date = recent_details[-1]['date']
            subject = f"Ensemble V3 Trading Report: {start_date} to {end_date}"

            send_email(
                subject=subject,
                body=text_body,
                to_email=args.email_to,
                html_body=html_body
            )
        else:
            print("No trading details to send.")


if __name__ == "__main__":
    main()
