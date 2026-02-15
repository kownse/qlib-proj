#!/usr/bin/env python
"""
每日交易脚本 (Ensemble V4) - AE-MLP + AE-MLP(mkt-neutral) + CatBoost 三模型集成

模型:
  1. AE-MLP (alpha158-enhanced-v7)
  2. AE-MLP mkt-neutral (v9-mkt-neutral)
  3. CatBoost (catboost-v1)

默认 Stacking Weights: AE-MLP=0.350, AE-MLP-MN=0.300, CatBoost=0.350

流程:
1. 下载最新美股数据
2. 增量更新宏观数据
3. 处理宏观数据为特征
4. 加载 3 个预训练模型
5. 生成各自的预测并计算相关性
6. 集成预测结果 (zscore_weighted)
7. 运行回测，输出每日交易信息

使用方法:
    python scripts/run_daily_trading_ensemble_v4.py
    python scripts/run_daily_trading_ensemble_v4.py --skip-download
    python scripts/run_daily_trading_ensemble_v4.py --predict-only
    python scripts/run_daily_trading_ensemble_v4.py --send-email
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

from daily_trading_common import (
    PROJECT_ROOT,
    add_common_trading_args,
    download_data,
    init_qlib_for_talib,
    create_dataset_for_trading,
    load_model_meta,
    load_ae_mlp_model,
    load_catboost_model,
    predict_with_ae_mlp,
    predict_with_catboost,
    ensemble_predictions_multi,
    detect_and_validate_dates,
    send_trading_email,
    run_ensemble_live_prediction,
    run_ensemble_backtest,
)
from utils.ai_filter import apply_ai_affinity_filter


def calculate_correlation_multi(pred_dict: dict) -> dict:
    """Calculate pairwise daily correlation between all model predictions."""
    import numpy as np
    import pandas as pd

    names = list(pred_dict.keys())

    common_idx = pred_dict[names[0]].index
    for name in names[1:]:
        common_idx = common_idx.intersection(pred_dict[name].index)

    df = pd.DataFrame({name: pred_dict[name].loc[common_idx] for name in names})

    daily_corrs = {}
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j <= i:
                continue
            pair_key = f"{n1} vs {n2}"
            dc = df.groupby(level='datetime').apply(
                lambda x, a=n1, b=n2: x[a].corr(x[b]) if len(x) > 1 else np.nan
            ).dropna()
            daily_corrs[pair_key] = (dc.mean(), dc.std())

    return daily_corrs


def main():
    parser = argparse.ArgumentParser(
        description='Daily trading script (Ensemble V4) - AE-MLP + AE-MLP(mkt-neutral) + CatBoost',
    )
    add_common_trading_args(parser)

    # Override defaults for V4
    parser.set_defaults(backtest_start='2026-01-28')

    # Model paths (same as run_ae_cb_ensemble_v4.py)
    parser.add_argument('--ae-model', type=str,
                        default='my_models/ae_mlp_cv_alpha158-enhanced-v7_sp500_5d_best.keras',
                        help='Path to AE-MLP model')
    parser.add_argument('--ae-mn-model', type=str,
                        default='my_models/ae_mlp_cv_v9-mkt-neutral_sp500_5d.keras',
                        help='Path to AE-MLP market-neutral model')
    parser.add_argument('--cb-model', type=str,
                        default='my_models/catboost_cv_catboost-v1_sp500_5d_20260129_141353_best.cbm',
                        help='Path to CatBoost model')

    # Handlers (same as run_ae_cb_ensemble_v4.py)
    parser.add_argument('--ae-handler', type=str, default='alpha158-enhanced-v7',
                        help='Handler for AE-MLP model')
    parser.add_argument('--ae-mn-handler', type=str, default='v9-mkt-neutral',
                        help='Handler for AE-MLP market-neutral model')
    parser.add_argument('--cb-handler', type=str, default='catboost-v1',
                        help='Handler for CatBoost model')

    # Ensemble parameters (default: zscore_weighted with stacking weights)
    parser.add_argument('--ensemble-method', type=str, default='zscore_weighted',
                        choices=['mean', 'weighted', 'rank_mean', 'zscore_mean', 'zscore_weighted'],
                        help='Ensemble method (default: zscore_weighted)')
    parser.add_argument('--ae-weight', type=float, default=0.350,
                        help='AE-MLP weight (default: 0.350)')
    parser.add_argument('--ae-mn-weight', type=float, default=0.300,
                        help='AE-MLP mkt-neutral weight (default: 0.300)')
    parser.add_argument('--cb-weight', type=float, default=0.350,
                        help='CatBoost weight (default: 0.350)')

    # Strategy parameters
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'mvo', 'rp', 'gmv', 'inv'],
                        help='Trading strategy (default: topk)')
    parser.add_argument('--opt-lamb', type=float, default=15.0,
                        help='[mvo] Risk aversion (default: 15.0)')
    parser.add_argument('--opt-delta', type=float, default=0.2,
                        help='[mvo/rp/gmv] Max turnover per rebalance (default: 0.2)')
    parser.add_argument('--opt-alpha', type=float, default=0.05,
                        help='[mvo/rp/gmv] L2 regularization (default: 0.05)')
    parser.add_argument('--cov-lookback', type=int, default=60,
                        help='[mvo/rp/gmv/inv] Covariance lookback days (default: 60)')
    parser.add_argument('--max-weight', type=float, default=0.30,
                        help='[mvo/rp/gmv/inv] Max weight per stock (default: 0.30)')

    # AI affinity filter
    parser.add_argument('--ai-filter', type=str, default='none',
                        choices=['none', 'penalty', 'exclude'],
                        help='AI affinity filter mode (default: none)')
    parser.add_argument('--ai-penalty-weight', type=float, default=0.5,
                        help='Penalty multiplier for negative-affinity stocks (default: 0.5)')
    parser.add_argument('--ai-bonus-weight', type=float, default=0.0,
                        help='Bonus multiplier for positive-affinity stocks (default: 0.0)')
    parser.add_argument('--ai-exclude-threshold', type=int, default=-1,
                        help='Affinity threshold for exclude mode, drop if <= this (default: -1)')
    parser.add_argument('--no-ai-time-scale', action='store_true',
                        help='Disable AI affinity time scaling (ramp 2020-2024)')

    args = parser.parse_args()

    # Model config
    MODEL_CONFIG = {
        'ae': {
            'model_arg': 'ae_model',
            'handler_arg': 'ae_handler',
            'handler': args.ae_handler,
            'display': 'AE-MLP',
            'type': 'ae_mlp',
        },
        'ae_mn': {
            'model_arg': 'ae_mn_model',
            'handler_arg': 'ae_mn_handler',
            'handler': args.ae_mn_handler,
            'display': 'AE-MLP-MN',
            'type': 'ae_mlp',
        },
        'cb': {
            'model_arg': 'cb_model',
            'handler_arg': 'cb_handler',
            'handler': args.cb_handler,
            'display': 'CatBoost',
            'type': 'catboost',
        },
    }

    display_names = {k: cfg['display'] for k, cfg in MODEL_CONFIG.items()}
    weights = {
        'ae': args.ae_weight,
        'ae_mn': args.ae_mn_weight,
        'cb': args.cb_weight,
    }

    print("=" * 70)
    mode = "LIVE PREDICTION MODE" if args.predict_only else "BACKTEST MODE"
    print(f"DAILY TRADING SCRIPT (ENSEMBLE V4) - {mode}")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for key, cfg in MODEL_CONFIG.items():
        model_path = getattr(args, cfg['model_arg'])
        print(f"{cfg['display']} Model:   {model_path}")
        print(f"{cfg['display']} Handler: {cfg['handler']}")
    print(f"Ensemble Method: {args.ensemble_method}")
    print(f"Weights: " + ", ".join(f"{cfg['display']}={weights[k]:.3f}" for k, cfg in MODEL_CONFIG.items()))
    print(f"Stock Pool: {args.stock_pool}")
    print(f"Account: ${args.account:,.2f}")
    print(f"Top-K: {args.topk}")
    print(f"Backtest Start: {args.backtest_start}")
    print("=" * 70)

    # Step 1: 下载数据
    if not args.skip_download:
        download_data(args.stock_pool)
    else:
        print("\n[SKIP] Data download skipped")

    # 检测日期范围
    detect_and_validate_dates(args)

    # Step 2: 初始化 Qlib
    print(f"\n{'='*60}")
    print("[STEP] Initializing Qlib")
    print(f"{'='*60}")
    init_qlib_for_talib()

    # Step 3: 检查模型文件并加载 metadata
    print(f"\n{'='*60}")
    print("[STEP] Checking model files")
    print(f"{'='*60}")

    model_paths = {}
    for key, cfg in MODEL_CONFIG.items():
        model_path = PROJECT_ROOT / getattr(args, cfg['model_arg'])
        if not model_path.exists():
            print(f"Error: {cfg['display']} model not found: {model_path}")
            sys.exit(1)
        model_paths[key] = model_path

        meta = load_model_meta(model_path)
        if meta and 'handler' in meta:
            cfg['handler'] = meta['handler']
            setattr(args, cfg['handler_arg'], meta['handler'])
            print(f"    {cfg['display']} handler from metadata: {cfg['handler']}")

    # Step 4: 创建数据集
    print(f"\n{'='*60}")
    print("[STEP] Creating datasets")
    print(f"{'='*60}")

    datasets = {}
    for key, cfg in MODEL_CONFIG.items():
        print(f"\n  Creating {cfg['display']} dataset ({cfg['handler']})...")
        datasets[key] = create_dataset_for_trading(
            cfg['handler'], args.stock_pool,
            args.test_start, args.test_end, args.nday,
            verbose=False
        )

    # Step 5: 加载模型
    print(f"\n{'='*60}")
    print("[STEP] Loading models")
    print(f"{'='*60}")

    models = {}
    for key, cfg in MODEL_CONFIG.items():
        if cfg['type'] == 'ae_mlp':
            models[key] = load_ae_mlp_model(model_paths[key])
        elif cfg['type'] == 'catboost':
            models[key] = load_catboost_model(model_paths[key])

    # Step 6: 生成预测
    print(f"\n{'='*60}")
    print("[STEP] Generating predictions")
    print(f"{'='*60}")

    pred_dict = {}
    for key, cfg in MODEL_CONFIG.items():
        print(f"\n  {cfg['display']} predictions...")
        if cfg['type'] == 'ae_mlp':
            pred = predict_with_ae_mlp(models[key], datasets[key])
        elif cfg['type'] == 'catboost':
            pred = predict_with_catboost(models[key], datasets[key])
        pred_dict[key] = pred
        print(f"    Shape: {len(pred)}, Range: [{pred.min():.4f}, {pred.max():.4f}]")

    # Step 7: 计算相关性
    print(f"\n{'='*60}")
    print("[STEP] Calculating pairwise correlations")
    print(f"{'='*60}")

    daily_corrs = calculate_correlation_multi(pred_dict)
    for pair, (mean_c, std_c) in daily_corrs.items():
        n1, n2 = pair.split(' vs ')
        d1 = display_names.get(n1, n1)
        d2 = display_names.get(n2, n2)
        print(f"  {d1} vs {d2}: {mean_c:.4f} +/- {std_c:.4f}")

    # Step 8: 集成预测
    print(f"\n{'='*60}")
    print(f"[STEP] Ensembling predictions ({args.ensemble_method})")
    print(f"{'='*60}")
    print(f"  Weights: " + ", ".join(f"{display_names[k]}={weights[k]:.3f}" for k in MODEL_CONFIG))

    pred_ensemble = ensemble_predictions_multi(pred_dict, args.ensemble_method, weights)
    print(f"  Ensemble shape: {len(pred_ensemble)}")
    print(f"  Range: [{pred_ensemble.min():.4f}, {pred_ensemble.max():.4f}]")

    # AI affinity filter
    if args.ai_filter != 'none':
        print(f"\n{'='*60}")
        print(f"[STEP] Applying AI affinity filter ({args.ai_filter})")
        print(f"{'='*60}")
        pred_ensemble = apply_ai_affinity_filter(
            pred_ensemble,
            mode=args.ai_filter,
            penalty_weight=args.ai_penalty_weight,
            bonus_weight=args.ai_bonus_weight,
            exclude_threshold=args.ai_exclude_threshold,
            time_scale=not args.no_ai_time_scale,
        )
        print(f"  Filtered shape: {len(pred_ensemble)}")
        print(f"  Range: [{pred_ensemble.min():.4f}, {pred_ensemble.max():.4f}]")

    # Step 9: 运行预测或回测
    trading_details = []

    if args.predict_only:
        run_ensemble_live_prediction(
            pred_ensemble=pred_ensemble,
            stock_pool=args.stock_pool,
            topk=args.topk,
            account=args.account,
            version_label="Ensemble V4",
            file_prefix="ensemble_v4",
        )
    else:
        # Build optimizer params if using portfolio optimization strategy
        optimizer_params = None
        if args.strategy in ('mvo', 'rp', 'gmv', 'inv'):
            optimizer_params = {
                "lamb": args.opt_lamb,
                "delta": args.opt_delta,
                "alpha": args.opt_alpha,
                "cov_lookback": args.cov_lookback,
                "max_weight": args.max_weight,
            }

        trading_details = run_ensemble_backtest(
            pred_ensemble=pred_ensemble,
            stock_pool=args.stock_pool,
            test_start=args.backtest_start,
            test_end=args.test_end,
            account=args.account,
            topk=args.topk,
            n_drop=args.n_drop,
            rebalance_freq=args.rebalance_freq,
            strategy_type=args.strategy,
            optimizer_params=optimizer_params,
            version_label="Ensemble V4",
            file_prefix="ensemble_v4",
            use_signal_shift=True,
        )

    print(f"\n{'='*70}")
    print("ENSEMBLE V4 TRADING SCRIPT COMPLETED")
    print(f"{'='*70}")

    # Step 10: 发送邮件报告
    send_trading_email(
        args, trading_details,
        model_info="AE-MLP + AE-MLP-MN + CatBoost (Ensemble V4, zscore_weighted)",
    )


if __name__ == "__main__":
    main()
