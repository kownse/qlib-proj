#!/usr/bin/env python
"""
每日交易脚本 - 更新数据并运行模型预测

流程:
1. 下载最新美股数据 (download_us_data_to_date.py)
2. 增量更新宏观数据 (download_macro_data_to_date.py)
3. 处理宏观数据为特征 (process_macro_data.py)
4. 加载预训练模型，生成预测
5. 运行回测，输出每日交易信息

使用方法:
    python scripts/run_daily_trading.py
    python scripts/run_daily_trading.py --skip-download  # 跳过数据下载
    python scripts/run_daily_trading.py --account 10000  # 设置初始资金
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

# 导入公共模块（会自动设置环境变量和路径）
from daily_trading_common import (
    PROJECT_ROOT,
    add_common_trading_args,
    download_data,
    init_qlib_for_talib,
    create_dataset_for_trading,
    collect_trading_details_from_positions,
    print_trading_details,
    generate_detailed_trade_records,
    detect_and_validate_dates,
    send_trading_email,
)


def load_model(model_path: Path):
    """加载 AE-MLP 模型"""
    from models.deep.ae_mlp_model import AEMLP

    print(f"\nLoading model from: {model_path}")
    model = AEMLP.load(str(model_path))
    print(f"Model loaded: {model.num_columns} features")
    return model


def run_live_prediction(
    model,
    dataset,
    stock_pool: str,
    topk: int = 10,
    account: float = 8000,
):
    """
    生成实盘交易预测（不运行回测）

    这个函数用于实际交易场景：
    - 基于最新可用数据生成预测
    - 显示今天应该买入/卖出的股票
    - 不依赖未来数据，可以用于实时决策
    """
    from data.stock_pools import STOCK_POOLS

    print(f"\n{'='*70}")
    print("LIVE TRADING PREDICTIONS")
    print(f"{'='*70}")
    print(f"Account: ${account:,.2f}")
    print(f"Top-K stocks: {topk}")

    # 生成预测
    print(f"\n[1] Generating predictions...")
    pred = model.predict(dataset, segment="test")

    if pred.empty:
        print("Error: No predictions generated!")
        return

    # 转换为 DataFrame 并处理符号大小写
    pred_df = pred.to_frame("score")
    pred_df = pred_df.reset_index()
    pred_df['instrument'] = pred_df['instrument'].str.lower()
    pred_df = pred_df.set_index(['datetime', 'instrument'])

    pred_dates = pred_df.index.get_level_values(0).unique().sort_values()
    print(f"    Predictions available for: {pred_dates.min().strftime('%Y-%m-%d')} to {pred_dates.max().strftime('%Y-%m-%d')}")
    print(f"    Total dates: {len(pred_dates)}, Total predictions: {len(pred_df)}")

    # 获取最新日期的预测
    latest_date = pred_dates[-1]
    latest_preds = pred_df.loc[latest_date].sort_values("score", ascending=False)

    print(f"\n{'='*70}")
    print(f"TRADING SIGNALS FOR: {latest_date.strftime('%Y-%m-%d')}")
    print(f"(Based on data available up to this date)")
    print(f"{'='*70}")

    # 显示买入推荐
    print(f"\nTOP {topk} STOCKS TO BUY (highest predicted 5-day return):")
    print("-" * 60)
    print(f"{'Rank':<6} {'Symbol':<10} {'Score':>12} {'Est. Allocation':>18}")
    print("-" * 60)

    allocation_per_stock = account * 0.95 / topk  # 95% 仓位分配
    top_stocks = latest_preds.head(topk)

    for i, (stock, row) in enumerate(top_stocks.iterrows(), 1):
        print(f"{i:<6} {stock.upper():<10} {row['score']:>12.4f} ${allocation_per_stock:>15,.2f}")

    # 显示应避免的股票
    print(f"\nTOP {topk} STOCKS TO AVOID (lowest predicted return):")
    print("-" * 60)
    bottom_stocks = latest_preds.tail(topk).iloc[::-1]
    for i, (stock, row) in enumerate(bottom_stocks.iterrows(), 1):
        print(f"{i:<6} {stock.upper():<10} {row['score']:>12.4f}")

    # 统计信息
    print(f"\n{'='*70}")
    print("PREDICTION STATISTICS")
    print(f"{'='*70}")
    print(f"Total stocks analyzed: {len(latest_preds)}")
    print(f"Mean score: {latest_preds['score'].mean():.4f}")
    print(f"Std score: {latest_preds['score'].std():.4f}")
    print(f"Max score: {latest_preds['score'].max():.4f} ({latest_preds['score'].idxmax().upper()})")
    print(f"Min score: {latest_preds['score'].min():.4f} ({latest_preds['score'].idxmin().upper()})")

    # 保存预测结果
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存完整预测
    pred_output = latest_preds.copy()
    pred_output.index = pred_output.index.str.upper()  # 转回大写
    pred_output = pred_output.sort_values("score", ascending=False)
    pred_path = output_dir / f"predictions_{latest_date.strftime('%Y%m%d')}.csv"
    pred_output.to_csv(pred_path)
    print(f"\nFull predictions saved to: {pred_path}")

    # 保存交易建议
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
    rec_path = output_dir / f"recommendations_{latest_date.strftime('%Y%m%d')}.csv"
    rec_df.to_csv(rec_path, index=False)
    print(f"Buy recommendations saved to: {rec_path}")

    print(f"\n{'='*70}")
    print("LIVE PREDICTION COMPLETED")
    print(f"{'='*70}")
    print(f"\nNext steps for live trading:")
    print(f"   1. Review the recommendations above")
    print(f"   2. Check current market conditions")
    print(f"   3. Execute trades at market open tomorrow")
    print(f"   4. Re-run this script daily after market close")

    return pred_df


def run_trading_backtest(
    model,
    dataset,
    stock_pool: str,
    test_start: str,
    test_end: str,
    account: float = 8000,
    topk: int = 5,
    n_drop: int = 1,
    rebalance_freq: int = 5,
):
    """
    运行交易回测并输出详细信息
    """
    from qlib.backtest import backtest as qlib_backtest
    from qlib.contrib.evaluate import risk_analysis
    from utils.strategy import get_strategy_config

    print(f"\n{'='*70}")
    print("TRADING BACKTEST")
    print(f"{'='*70}")
    print(f"Period: {test_start} to {test_end}")
    print(f"Initial Account: ${account:,.2f}")
    print(f"Rebalance Frequency: every {rebalance_freq} day(s)")
    print(f"Top-K stocks: {topk}")
    print(f"N-drop: {n_drop}")

    # 生成预测
    print(f"\n[1] Generating predictions...")
    pred = model.predict(dataset, segment="test")

    if pred.empty:
        print("Error: No predictions generated!")
        return

    print(f"    Predictions shape: {pred.shape}")
    pred_dates = pred.index.get_level_values(0).unique().sort_values()
    print(f"    Date range: {pred_dates.min()} to {pred_dates.max()}")
    print(f"    Unique dates: {len(pred_dates)}")
    print(f"    Unique stocks: {pred.index.get_level_values(1).nunique()}")

    # 显示预测统计
    print(f"\n    Prediction statistics:")
    print(f"    Min: {pred.min():.6f}")
    print(f"    Max: {pred.max():.6f}")
    print(f"    Mean: {pred.mean():.6f}")
    print(f"    Std: {pred.std():.6f}")

    # 转换为 DataFrame
    pred_df = pred.to_frame("score")

    # ============ 将股票符号转为小写以匹配 Qlib 数据格式 ============
    pred_df = pred_df.reset_index()
    pred_df['instrument'] = pred_df['instrument'].str.lower()
    pred_df = pred_df.set_index(['datetime', 'instrument'])

    # ============ 显示最新日期的推荐股票 ============
    latest_pred_date = pred_dates[-1]
    print(f"\n{'='*70}")
    print(f"TODAY'S RECOMMENDATIONS (based on {latest_pred_date.strftime('%Y-%m-%d')} predictions)")
    print(f"{'='*70}")

    latest_preds = pred_df.loc[latest_pred_date].sort_values("score", ascending=False)
    print(f"\nTop {topk} stocks to BUY (highest predicted return):")
    print("-" * 50)
    for i, (stock, row) in enumerate(latest_preds.head(topk).iterrows(), 1):
        print(f"  {i:2d}. {stock.upper():<8s}  Score: {row['score']:>8.4f}")

    print(f"\nTop {topk} stocks to AVOID (lowest predicted return):")
    print("-" * 50)
    for i, (stock, row) in enumerate(latest_preds.tail(topk).iloc[::-1].iterrows(), 1):
        print(f"  {i:2d}. {stock.upper():<8s}  Score: {row['score']:>8.4f}")

    # ============ 调整回测日期 ============
    if len(pred_dates) > 0:
        actual_test_end = pred_dates[-1].strftime("%Y-%m-%d")
        actual_test_start = pred_dates[0].strftime("%Y-%m-%d")

        if pd.Timestamp(test_start) > pred_dates[0]:
            actual_test_start = test_start

        if actual_test_start != test_start or actual_test_end != test_end:
            print(f"\n[NOTE] Adjusting backtest period to match available predictions:")
            print(f"       Original: {test_start} to {test_end}")
            print(f"       Adjusted: {actual_test_start} to {actual_test_end}")
            test_start = actual_test_start
            test_end = actual_test_end

    # 配置策略
    print(f"\n[2] Configuring strategy...")
    strategy_config = get_strategy_config(
        pred_df, topk, n_drop, rebalance_freq=rebalance_freq,
        strategy_type="topk"
    )

    # 配置执行器
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    }

    # 创建空的 benchmark series（避免 Qlib 默认使用 SH000300）
    empty_benchmark = pd.Series(dtype=float)

    # 回测配置
    backtest_config = {
        "start_time": test_start,
        "end_time": test_end,
        "account": account,
        "benchmark": empty_benchmark,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": None,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0005,
            "min_cost": 1,
        },
    }

    # 运行回测
    print(f"\n[3] Running backtest...")
    portfolio_metric_dict, indicator_dict = qlib_backtest(
        executor=executor_config,
        strategy=strategy_config,
        **backtest_config
    )

    print("    Backtest completed!")

    # 分析结果
    print(f"\n[4] Analyzing results...")

    for freq, (report_df, positions) in portfolio_metric_dict.items():
        print(f"\n{'='*70}")
        print(f"TRADING RESULTS ({freq})")
        print(f"{'='*70}")

        # 基本统计
        total_return = (report_df["return"] + 1).prod() - 1
        final_value = account * (1 + total_return)

        print(f"\nPerformance Summary:")
        print(f"  Initial Capital:  ${account:>12,.2f}")
        print(f"  Final Value:      ${final_value:>12,.2f}")
        print(f"  Total Return:     {total_return:>12.2%}")
        print(f"  Total P&L:        ${final_value - account:>12,.2f}")

        # Benchmark 收益
        if "bench" in report_df.columns and not report_df["bench"].isna().all():
            bench_return = (report_df["bench"] + 1).prod() - 1
            excess_return = total_return - bench_return
            print(f"  Benchmark Return: {bench_return:>12.2%}")
            print(f"  Excess Return:    {excess_return:>12.2%}")

        # 风险分析
        analysis = risk_analysis(report_df["return"], freq=freq)
        if analysis is not None and not analysis.empty:
            print(f"\nRisk Metrics:")
            for metric, value in analysis.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric:<20s}: {value:>10.4f}")

        # 每日收益统计
        print(f"\nDaily Statistics:")
        print(f"  Trading Days:     {len(report_df):>12d}")
        print(f"  Mean Daily Return:{report_df['return'].mean():>12.4%}")
        print(f"  Std Daily Return: {report_df['return'].std():>12.4%}")
        print(f"  Best Day:         {report_df['return'].max():>12.4%}")
        print(f"  Worst Day:        {report_df['return'].min():>12.4%}")

        # 输出每日详细交易信息
        print(f"\n{'='*70}")
        print("DAILY TRADING DETAILS")
        print(f"{'='*70}")

        # 使用公共函数收集和打印交易详情
        all_trading_details = collect_trading_details_from_positions(positions, report_df)
        print_trading_details(all_trading_details)

        # 保存报告
        output_dir = PROJECT_ROOT / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存每日报告
        report_path = output_dir / f"daily_trading_report_{freq}.csv"
        report_df.to_csv(report_path)
        print(f"\n\nReport saved to: {report_path}")

        # 生成交易记录
        generate_detailed_trade_records(positions, output_dir, freq)

        # 返回交易详情供邮件使用
        return all_trading_details

    return []


def main():
    parser = argparse.ArgumentParser(
        description='Daily trading script - update data and run model predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_trading_args(parser)

    # Override defaults for base script
    parser.set_defaults(
        n_drop=1,
        backtest_start='2026-01-01',
    )

    parser.add_argument('--model-path', type=str,
                        default='my_models/ae_mlp_cv_alpha158-enhanced-v7_sp500_5d_best.keras',
                        help='Path to trained model')
    parser.add_argument('--handler', type=str, default='alpha158-enhanced-v7',
                        help='Handler name')
    args = parser.parse_args()

    print("="*70)
    mode = "LIVE PREDICTION MODE" if args.predict_only else "BACKTEST MODE"
    print(f"DAILY TRADING SCRIPT - {mode}")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model_path}")
    print(f"Handler: {args.handler}")
    print(f"Stock Pool: {args.stock_pool}")
    print(f"Account: ${args.account:,.2f}")
    print(f"Top-K: {args.topk}")
    print(f"Backtest Start: {args.backtest_start}")
    print(f"Test End: {args.test_end or 'auto (latest data - 1 day)'}")
    print("="*70)

    # Step 1: 下载数据
    if not args.skip_download:
        download_data(args.stock_pool)
    else:
        print("\n[SKIP] Data download skipped")

    # 自动检测日期范围
    detect_and_validate_dates(args)

    # Step 2: 初始化 Qlib
    print(f"\n{'='*60}")
    print("[STEP] Initializing Qlib")
    print(f"{'='*60}")
    init_qlib_for_talib()

    # Step 3: 加载模型
    print(f"\n{'='*60}")
    print("[STEP] Loading model")
    print(f"{'='*60}")
    model_path = PROJECT_ROOT / args.model_path
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    model = load_model(model_path)

    # Step 4: 创建数据集
    print(f"\n{'='*60}")
    print("[STEP] Creating dataset")
    print(f"{'='*60}")
    dataset = create_dataset_for_trading(
        args.handler,
        args.stock_pool,
        args.test_start,
        args.test_end,
        args.nday,
    )

    # Step 5: 运行预测或回测
    trading_details = []

    if args.predict_only:
        # 实盘模式：只生成预测，不运行回测
        run_live_prediction(
            model=model,
            dataset=dataset,
            stock_pool=args.stock_pool,
            topk=args.topk,
            account=args.account,
        )
    else:
        # 回测模式：运行完整回测
        trading_details = run_trading_backtest(
            model=model,
            dataset=dataset,
            stock_pool=args.stock_pool,
            test_start=args.backtest_start,
            test_end=args.test_end,
            account=args.account,
            topk=args.topk,
            n_drop=args.n_drop,
            rebalance_freq=args.rebalance_freq,
        )

        print(f"\n{'='*70}")
        print("TRADING SCRIPT COMPLETED")
        print(f"{'='*70}")

    # Step 6: 发送邮件报告
    send_trading_email(args, trading_details)


if __name__ == "__main__":
    main()
