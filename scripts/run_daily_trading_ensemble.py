#!/usr/bin/env python
"""
每日交易脚本 (Ensemble版本) - 使用 AE-MLP + CatBoost 模型集成

流程:
1. 下载最新美股数据 (download_us_data_to_date.py)
2. 增量更新宏观数据 (download_macro_data_to_date.py)
3. 处理宏观数据为特征 (process_macro_data.py)
4. 加载 AE-MLP 和 CatBoost 预训练模型
5. 生成各自的预测并计算相关性
6. 集成预测结果
7. 运行回测，输出每日交易信息

使用方法:
    python scripts/run_daily_trading_ensemble.py
    python scripts/run_daily_trading_ensemble.py --skip-download
    python scripts/run_daily_trading_ensemble.py --ensemble-method zscore_mean
    python scripts/run_daily_trading_ensemble.py --predict-only  # 只输出预测，不回测
"""

import sys
import argparse
from datetime import datetime

import pandas as pd
import numpy as np

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
    detect_and_validate_dates,
    send_trading_email,
    run_ensemble_live_prediction,
    run_ensemble_backtest,
)


# ============================================================================
# V1-specific 2-model ensemble functions (kept here, not in common)
# ============================================================================

def calculate_correlation(pred1: pd.Series, pred2: pd.Series) -> tuple:
    """Calculate correlation between two prediction series"""
    common_idx = pred1.index.intersection(pred2.index)
    p1 = pred1.loc[common_idx]
    p2 = pred2.loc[common_idx]

    overall_corr = p1.corr(p2)

    daily_corr = pd.DataFrame({'p1': p1, 'p2': p2}).groupby(level='datetime').apply(
        lambda x: x['p1'].corr(x['p2']) if len(x) > 1 else np.nan
    )
    daily_corr = daily_corr.dropna()

    mean_daily_corr = daily_corr.mean()
    std_daily_corr = daily_corr.std()

    return overall_corr, mean_daily_corr, std_daily_corr


def ensemble_predictions(pred1: pd.Series, pred2: pd.Series,
                         method: str = 'zscore_mean', weights: tuple = None) -> pd.Series:
    """Ensemble two model predictions"""
    common_idx = pred1.index.intersection(pred2.index)
    p1 = pred1.loc[common_idx]
    p2 = pred2.loc[common_idx]

    if method == 'mean':
        ensemble_pred = (p1 + p2) / 2
    elif method == 'weighted':
        if weights is None:
            weights = (0.5, 0.5)
        w1, w2 = weights
        total = w1 + w2
        ensemble_pred = (p1 * w1 + p2 * w2) / total
    elif method == 'rank_mean':
        rank1 = p1.groupby(level='datetime').rank(pct=True)
        rank2 = p2.groupby(level='datetime').rank(pct=True)
        ensemble_pred = (rank1 + rank2) / 2
    elif method == 'zscore_mean':
        def zscore_by_day(x):
            mean = x.groupby(level='datetime').transform('mean')
            std = x.groupby(level='datetime').transform('std')
            return (x - mean) / (std + 1e-8)
        z1 = zscore_by_day(p1)
        z2 = zscore_by_day(p2)
        ensemble_pred = (z1 + z2) / 2
    elif method == 'zscore_weighted':
        if weights is None:
            weights = (0.5, 0.5)
        w1, w2 = weights
        total = w1 + w2
        def zscore_by_day(x):
            mean = x.groupby(level='datetime').transform('mean')
            std = x.groupby(level='datetime').transform('std')
            return (x - mean) / (std + 1e-8)
        z1 = zscore_by_day(p1)
        z2 = zscore_by_day(p2)
        ensemble_pred = (z1 * w1 + z2 * w2) / total
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    ensemble_pred.name = 'score'
    return ensemble_pred


def main():
    parser = argparse.ArgumentParser(
        description='Daily trading script (Ensemble) - AE-MLP + CatBoost',
    )
    add_common_trading_args(parser)

    # Model paths
    parser.add_argument('--ae-model', type=str,
                        default='my_models/ae_mlp_cv_alpha158-enhanced-v7_sp500_5d_best.keras',
                        help='Path to AE-MLP model')
    parser.add_argument('--cb-model', type=str,
                        default='my_models/catboost_cv_catboost-v1_sp500_5d_20260129_141353_best.cbm',
                        help='Path to CatBoost model')

    # Handlers
    parser.add_argument('--ae-handler', type=str, default='alpha158-enhanced-v7',
                        help='Handler for AE-MLP model')
    parser.add_argument('--cb-handler', type=str, default='catboost-v1',
                        help='Handler for CatBoost model')

    # Ensemble parameters
    parser.add_argument('--ensemble-method', type=str, default='zscore_mean',
                        choices=['mean', 'weighted', 'rank_mean', 'zscore_mean', 'zscore_weighted'],
                        help='Ensemble method (default: zscore_mean)')
    parser.add_argument('--ae-weight', type=float, default=0.5,
                        help='AE-MLP weight for weighted ensemble')
    parser.add_argument('--cb-weight', type=float, default=0.5,
                        help='CatBoost weight for weighted ensemble')

    args = parser.parse_args()

    print("="*70)
    mode = "LIVE PREDICTION MODE" if args.predict_only else "BACKTEST MODE"
    print(f"DAILY TRADING SCRIPT (ENSEMBLE) - {mode}")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"AE-MLP Model: {args.ae_model}")
    print(f"CatBoost Model: {args.cb_model}")
    print(f"AE Handler: {args.ae_handler}")
    print(f"CB Handler: {args.cb_handler}")
    print(f"Ensemble Method: {args.ensemble_method}")
    print(f"Stock Pool: {args.stock_pool}")
    print(f"Account: ${args.account:,.2f}")
    print(f"Top-K: {args.topk}")
    print(f"Backtest Start: {args.backtest_start}")
    print("="*70)

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

    # Step 3: 检查模型文件
    print(f"\n{'='*60}")
    print("[STEP] Checking model files")
    print(f"{'='*60}")

    ae_path = PROJECT_ROOT / args.ae_model
    cb_path = PROJECT_ROOT / args.cb_model

    if not ae_path.exists():
        print(f"Error: AE-MLP model not found: {ae_path}")
        sys.exit(1)
    if not cb_path.exists():
        print(f"Error: CatBoost model not found: {cb_path}")
        sys.exit(1)

    # Load metadata
    ae_meta = load_model_meta(ae_path)
    cb_meta = load_model_meta(cb_path)

    if ae_meta and 'handler' in ae_meta:
        args.ae_handler = ae_meta['handler']
        print(f"    AE-MLP handler from metadata: {args.ae_handler}")
    if cb_meta and 'handler' in cb_meta:
        args.cb_handler = cb_meta['handler']
        print(f"    CatBoost handler from metadata: {args.cb_handler}")

    # Step 4: 创建数据集
    print(f"\n{'='*60}")
    print("[STEP] Creating datasets")
    print(f"{'='*60}")

    print("\n  Creating AE-MLP dataset...")
    ae_dataset = create_dataset_for_trading(
        args.ae_handler, args.stock_pool,
        args.test_start, args.test_end, args.nday,
        verbose=False
    )

    print("\n  Creating CatBoost dataset...")
    cb_dataset = create_dataset_for_trading(
        args.cb_handler, args.stock_pool,
        args.test_start, args.test_end, args.nday,
        verbose=False
    )

    # Step 5: 加载模型
    print(f"\n{'='*60}")
    print("[STEP] Loading models")
    print(f"{'='*60}")

    ae_model = load_ae_mlp_model(ae_path)
    cb_model = load_catboost_model(cb_path)

    # Step 6: 生成预测
    print(f"\n{'='*60}")
    print("[STEP] Generating predictions")
    print(f"{'='*60}")

    print("\n  AE-MLP predictions...")
    pred_ae = predict_with_ae_mlp(ae_model, ae_dataset)
    print(f"    Shape: {len(pred_ae)}, Range: [{pred_ae.min():.4f}, {pred_ae.max():.4f}]")

    print("\n  CatBoost predictions...")
    pred_cb = predict_with_catboost(cb_model, cb_dataset)
    print(f"    Shape: {len(pred_cb)}, Range: [{pred_cb.min():.4f}, {pred_cb.max():.4f}]")

    # Step 7: 计算相关性
    print(f"\n{'='*60}")
    print("[STEP] Calculating correlation")
    print(f"{'='*60}")

    overall_corr, mean_daily_corr, std_daily_corr = calculate_correlation(pred_ae, pred_cb)
    print(f"  Overall Correlation:     {overall_corr:.4f}")
    print(f"  Mean Daily Correlation:  {mean_daily_corr:.4f}")
    print(f"  Std Daily Correlation:   {std_daily_corr:.4f}")

    # Step 8: 集成预测
    print(f"\n{'='*60}")
    print(f"[STEP] Ensembling predictions ({args.ensemble_method})")
    print(f"{'='*60}")

    weights = (args.ae_weight, args.cb_weight) if args.ensemble_method == 'weighted' else None
    pred_ensemble = ensemble_predictions(pred_ae, pred_cb, args.ensemble_method, weights)
    print(f"  Ensemble shape: {len(pred_ensemble)}")
    print(f"  Range: [{pred_ensemble.min():.4f}, {pred_ensemble.max():.4f}]")

    # Step 9: 运行预测或回测
    trading_details = []

    if args.predict_only:
        run_ensemble_live_prediction(
            pred_ensemble=pred_ensemble,
            stock_pool=args.stock_pool,
            topk=args.topk,
            account=args.account,
            version_label="Ensemble",
            file_prefix="ensemble",
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
            version_label="Ensemble",
            file_prefix="ensemble",
            use_signal_shift=False,
        )

    print(f"\n{'='*70}")
    print("ENSEMBLE TRADING SCRIPT COMPLETED")
    print(f"{'='*70}")

    # Step 10: 发送邮件报告
    send_trading_email(
        args, trading_details,
        model_info=f"AE-MLP + CatBoost ({args.ensemble_method})",
    )


if __name__ == "__main__":
    main()
