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
from datetime import datetime

from daily_trading_common import (
    PROJECT_ROOT,
    SCRIPTS_DIR,
    add_common_trading_args,
    download_data,
    init_qlib_for_talib,
    create_dataset_for_trading,
    load_model_meta,
    load_ae_mlp_model,
    load_catboost_model,
    predict_with_ae_mlp,
    predict_with_catboost,
    calculate_pairwise_correlations,
    ensemble_predictions_multi,
    detect_and_validate_dates,
    send_trading_email,
    run_ensemble_live_prediction,
    run_ensemble_backtest,
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
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Daily trading script (Ensemble V3) - 3x AE-MLP + CatBoost',
    )
    add_common_trading_args(parser)

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
    detect_and_validate_dates(args)

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
        run_ensemble_live_prediction(
            pred_ensemble=pred_ensemble,
            stock_pool=args.stock_pool,
            topk=args.topk,
            account=args.account,
            version_label=f"Ensemble V3: {n_models} Models",
            file_prefix="ensemble_v3",
        )
    else:
        trading_details = run_ensemble_backtest(
            pred_ensemble=pred_ensemble,
            stock_pool=args.stock_pool,
            test_start=args.backtest_start,
            test_end=args.test_end,
            account=args.account,
            topk=args.topk,
            n_drop=args.n_drop,
            rebalance_freq=args.rebalance_freq,
            version_label=f"Ensemble V3: {n_models} Models",
            file_prefix="ensemble_v3",
            use_signal_shift=False,
        )

    print(f"\n{'='*70}")
    print("ENSEMBLE V3 TRADING SCRIPT COMPLETED")
    print(f"{'='*70}")

    # Step 10: Send email report
    send_trading_email(
        args, trading_details,
        model_info=f"3x AE-MLP + CatBoost ({args.ensemble_method})",
    )


if __name__ == "__main__":
    main()
