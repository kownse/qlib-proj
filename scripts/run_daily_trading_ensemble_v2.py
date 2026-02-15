#!/usr/bin/env python
"""
每日交易脚本 (Ensemble V2) - 使用 2x AE-MLP + CatBoost 三模型集成

流程:
1. 下载最新美股数据 (download_us_data_to_date.py)
2. 增量更新宏观数据 (download_macro_data_to_date.py)
3. 处理宏观数据为特征 (process_macro_data.py)
4. 加载 AE-MLP v7, AE-MLP v9, CatBoost 预训练模型
5. 生成各自的预测并计算相关性
6. 三模型集成预测结果 (zscore_weighted)
7. 运行回测，输出每日交易信息

使用方法:
    python scripts/run_daily_trading_ensemble_v2.py
    python scripts/run_daily_trading_ensemble_v2.py --skip-download
    python scripts/run_daily_trading_ensemble_v2.py --predict-only
    python scripts/run_daily_trading_ensemble_v2.py --ensemble-method zscore_mean
"""

import sys
import argparse
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
    calculate_pairwise_correlations,
    ensemble_predictions_multi,
    detect_and_validate_dates,
    send_trading_email,
    run_ensemble_live_prediction,
    run_ensemble_backtest,
)


def main():
    parser = argparse.ArgumentParser(
        description='Daily trading script (Ensemble V2) - 2x AE-MLP + CatBoost',
    )
    add_common_trading_args(parser)

    # Model paths (defaults from run_ae_cb_ensemble.py)
    parser.add_argument('--ae-model', type=str,
                        default='my_models/ae_mlp_cv_alpha158-enhanced-v7_sp500_5d_best.keras',
                        help='Path to AE-MLP v7 model')
    parser.add_argument('--ae2-model', type=str,
                        default='my_models/ae_mlp_cv_alpha158-enhanced-v9_test_5d_best.keras',
                        help='Path to AE-MLP v9 model')
    parser.add_argument('--cb-model', type=str,
                        default='my_models/catboost_cv_catboost-v1_test_5d_20260129_105915_best.cbm',
                        help='Path to CatBoost model')

    # Handlers (defaults from run_ae_cb_ensemble.py)
    parser.add_argument('--ae-handler', type=str, default='alpha158-enhanced-v7',
                        help='Handler for AE-MLP v7 model')
    parser.add_argument('--ae2-handler', type=str, default='alpha158-enhanced-v9',
                        help='Handler for AE-MLP v9 model')
    parser.add_argument('--cb-handler', type=str, default='catboost-v1',
                        help='Handler for CatBoost model')

    # Ensemble parameters
    parser.add_argument('--ensemble-method', type=str, default='zscore_weighted',
                        choices=['mean', 'weighted', 'rank_mean', 'zscore_mean', 'zscore_weighted'],
                        help='Ensemble method (default: zscore_weighted)')
    parser.add_argument('--ae-weight', type=float, default=0.4,
                        help='AE-MLP v7 weight (default: 0.4)')
    parser.add_argument('--ae2-weight', type=float, default=0.3,
                        help='AE-MLP v9 weight (default: 0.3)')
    parser.add_argument('--cb-weight', type=float, default=0.3,
                        help='CatBoost weight (default: 0.3)')

    args = parser.parse_args()

    print("="*70)
    mode = "LIVE PREDICTION MODE" if args.predict_only else "BACKTEST MODE"
    print(f"DAILY TRADING SCRIPT (ENSEMBLE V2: 2x AE-MLP + CatBoost) - {mode}")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"AE-MLP v7 Model: {args.ae_model}")
    print(f"AE-MLP v9 Model: {args.ae2_model}")
    print(f"CatBoost Model:  {args.cb_model}")
    print(f"Handlers: AE-v7={args.ae_handler}, AE-v9={args.ae2_handler}, CB={args.cb_handler}")
    print(f"Ensemble Method: {args.ensemble_method}")
    print(f"Weights: AE-v7={args.ae_weight}, AE-v9={args.ae2_weight}, CB={args.cb_weight}")
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
    ae2_path = PROJECT_ROOT / args.ae2_model
    cb_path = PROJECT_ROOT / args.cb_model

    for path, name in [(ae_path, 'AE-MLP-v7'), (ae2_path, 'AE-MLP-v9'), (cb_path, 'CatBoost')]:
        if not path.exists():
            print(f"Error: {name} model not found: {path}")
            sys.exit(1)

    # Load metadata and override handlers if available
    ae_meta = load_model_meta(ae_path)
    ae2_meta = load_model_meta(ae2_path)
    cb_meta = load_model_meta(cb_path)

    if ae_meta and 'handler' in ae_meta:
        args.ae_handler = ae_meta['handler']
        print(f"    AE-MLP v7 handler from metadata: {args.ae_handler}")
    if ae2_meta and 'handler' in ae2_meta:
        args.ae2_handler = ae2_meta['handler']
        print(f"    AE-MLP v9 handler from metadata: {args.ae2_handler}")
    if cb_meta and 'handler' in cb_meta:
        args.cb_handler = cb_meta['handler']
        print(f"    CatBoost handler from metadata: {args.cb_handler}")

    # Step 4: 创建数据集
    print(f"\n{'='*60}")
    print("[STEP] Creating datasets")
    print(f"{'='*60}")

    print(f"\n  Creating AE-MLP v7 dataset ({args.ae_handler})...")
    ae_dataset = create_dataset_for_trading(
        args.ae_handler, args.stock_pool,
        args.test_start, args.test_end, args.nday,
        verbose=False
    )

    print(f"\n  Creating AE-MLP v9 dataset ({args.ae2_handler})...")
    ae2_dataset = create_dataset_for_trading(
        args.ae2_handler, args.stock_pool,
        args.test_start, args.test_end, args.nday,
        verbose=False
    )

    print(f"\n  Creating CatBoost dataset ({args.cb_handler})...")
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
    ae2_model = load_ae_mlp_model(ae2_path)
    cb_model = load_catboost_model(cb_path)

    # Step 6: 生成预测
    print(f"\n{'='*60}")
    print("[STEP] Generating predictions")
    print(f"{'='*60}")

    print("\n  AE-MLP v7 predictions...")
    pred_ae = predict_with_ae_mlp(ae_model, ae_dataset)
    print(f"    Shape: {len(pred_ae)}, Range: [{pred_ae.min():.4f}, {pred_ae.max():.4f}]")

    print("\n  AE-MLP v9 predictions...")
    pred_ae2 = predict_with_ae_mlp(ae2_model, ae2_dataset)
    print(f"    Shape: {len(pred_ae2)}, Range: [{pred_ae2.min():.4f}, {pred_ae2.max():.4f}]")

    print("\n  CatBoost predictions...")
    pred_cb = predict_with_catboost(cb_model, cb_dataset)
    print(f"    Shape: {len(pred_cb)}, Range: [{pred_cb.min():.4f}, {pred_cb.max():.4f}]")

    preds = {
        'AE-MLP-v7': pred_ae,
        'AE-MLP-v9': pred_ae2,
        'CatBoost': pred_cb,
    }

    # Step 7: 计算相关性
    print(f"\n{'='*60}")
    print("[STEP] Calculating pairwise correlations")
    print(f"{'='*60}")

    corr_matrix = calculate_pairwise_correlations(preds)
    print("\n  Prediction Correlation Matrix:")
    print("  " + "=" * 50)
    print(corr_matrix.to_string())
    print("  " + "=" * 50)

    # Step 8: 集成预测
    print(f"\n{'='*60}")
    print(f"[STEP] Ensembling predictions ({args.ensemble_method})")
    print(f"{'='*60}")

    weights = {
        'AE-MLP-v7': args.ae_weight,
        'AE-MLP-v9': args.ae2_weight,
        'CatBoost': args.cb_weight,
    }
    total_w = sum(weights.values())
    print(f"  Weights: AE-v7={weights['AE-MLP-v7']/total_w:.3f}, "
          f"AE-v9={weights['AE-MLP-v9']/total_w:.3f}, "
          f"CB={weights['CatBoost']/total_w:.3f}")

    pred_ensemble = ensemble_predictions_multi(preds, args.ensemble_method, weights)
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
            version_label="Ensemble V2: 2x AE-MLP + CatBoost",
            file_prefix="ensemble_v2",
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
            version_label="Ensemble V2: 2x AE-MLP + CatBoost",
            file_prefix="ensemble_v2",
            use_signal_shift=False,
        )

    print(f"\n{'='*70}")
    print("ENSEMBLE V2 TRADING SCRIPT COMPLETED")
    print(f"{'='*70}")

    # Step 10: 发送邮件报告
    send_trading_email(
        args, trading_details,
        model_info=f"2x AE-MLP + CatBoost ({args.ensemble_method})",
    )


if __name__ == "__main__":
    main()
