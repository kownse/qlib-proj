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
    python scripts/run_daily_trading_ensemble_4.py
    python scripts/run_daily_trading_ensemble_4.py --skip-download
    python scripts/run_daily_trading_ensemble_4.py --predict-only
    python scripts/run_daily_trading_ensemble_4.py --send-email
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pickle

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# 导入公共模块（会自动设置环境变量和路径）
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
)


def load_model_meta(model_path: Path) -> dict:
    """Load model metadata"""
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
    """加载 AE-MLP 模型"""
    from models.deep.ae_mlp_model import AEMLP

    print(f"    Loading AE-MLP model from: {model_path}")
    model = AEMLP.load(str(model_path))
    print(f"    AE-MLP loaded: {model.num_columns} features")
    return model


def load_catboost_model(model_path: Path):
    """加载 CatBoost 模型"""
    print(f"    Loading CatBoost model from: {model_path}")
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    feature_count = model.feature_names_ if model.feature_names_ else "N/A"
    if isinstance(feature_count, list):
        feature_count = len(feature_count)
    print(f"    CatBoost loaded: {feature_count} features")
    return model


def predict_with_ae_mlp(model, dataset) -> pd.Series:
    """Generate predictions with AE-MLP model"""
    pred = model.predict(dataset, segment="test")
    pred.name = 'score'
    return pred


def predict_with_catboost(model, dataset) -> pd.Series:
    """Generate predictions with CatBoost model"""
    from qlib.data.dataset.handler import DataHandlerLP

    test_data = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_L)
    test_data = test_data.fillna(0).replace([np.inf, -np.inf], 0)

    pred_values = model.predict(test_data.values)
    pred = pd.Series(pred_values, index=test_data.index, name='score')
    return pred


def zscore_by_day(x):
    mean = x.groupby(level='datetime').transform('mean')
    std = x.groupby(level='datetime').transform('std')
    return (x - mean) / (std + 1e-8)


def ensemble_predictions_multi(pred_dict: dict, method: str = 'zscore_weighted',
                                weights: dict = None) -> pd.Series:
    """
    Ensemble multiple model predictions.

    Parameters
    ----------
    pred_dict : dict
        {model_name: pd.Series}
    method : str
        'mean', 'weighted', 'rank_mean', 'zscore_mean', 'zscore_weighted'
    weights : dict, optional
        {model_name: weight}
    """
    names = list(pred_dict.keys())

    # Find common index
    common_idx = pred_dict[names[0]].index
    for name in names[1:]:
        common_idx = common_idx.intersection(pred_dict[name].index)

    preds = {name: pred_dict[name].loc[common_idx] for name in names}

    if method == 'mean':
        ensemble_pred = sum(preds.values()) / len(preds)
    elif method == 'weighted':
        if weights is None:
            weights = {n: 1.0 / len(names) for n in names}
        total = sum(weights.values())
        ensemble_pred = sum(preds[n] * weights[n] for n in names) / total
    elif method == 'rank_mean':
        ranks = {n: preds[n].groupby(level='datetime').rank(pct=True) for n in names}
        ensemble_pred = sum(ranks.values()) / len(ranks)
    elif method == 'zscore_mean':
        zscores = {n: zscore_by_day(preds[n]) for n in names}
        ensemble_pred = sum(zscores.values()) / len(zscores)
    elif method == 'zscore_weighted':
        if weights is None:
            weights = {n: 1.0 / len(names) for n in names}
        total = sum(weights.values())
        zscores = {n: zscore_by_day(preds[n]) for n in names}
        ensemble_pred = sum(zscores[n] * weights[n] for n in names) / total
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    ensemble_pred.name = 'score'
    return ensemble_pred


def calculate_correlation_multi(pred_dict: dict) -> dict:
    """Calculate pairwise daily correlation between all model predictions."""
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


def run_live_prediction(
    pred_ensemble: pd.Series,
    pred_dict: dict,
    display_names: dict,
    stock_pool: str,
    topk: int = 10,
    account: float = 8000,
):
    """生成实盘交易预测（不运行回测）"""
    from data.stock_pools import STOCK_POOLS

    print(f"\n{'='*70}")
    print("LIVE TRADING PREDICTIONS (Ensemble V4)")
    print(f"{'='*70}")
    print(f"Account: ${account:,.2f}")
    print(f"Top-K stocks: {topk}")

    # 转换为 DataFrame 并处理符号大小写
    pred_df = pred_ensemble.to_frame("score")
    pred_df = pred_df.reset_index()
    pred_df['instrument'] = pred_df['instrument'].str.lower()
    pred_df = pred_df.set_index(['datetime', 'instrument'])

    pred_dates = pred_df.index.get_level_values(0).unique().sort_values()
    print(f"\nPredictions available for: {pred_dates.min().strftime('%Y-%m-%d')} to {pred_dates.max().strftime('%Y-%m-%d')}")
    print(f"Total dates: {len(pred_dates)}, Total predictions: {len(pred_df)}")

    # 获取最新日期的预测
    latest_date = pred_dates[-1]
    latest_preds = pred_df.loc[latest_date].sort_values("score", ascending=False)

    print(f"\n{'='*70}")
    print(f"TRADING SIGNALS FOR: {latest_date.strftime('%Y-%m-%d')}")
    print(f"{'='*70}")

    # 显示买入推荐
    print(f"\nTOP {topk} STOCKS TO BUY (highest predicted return):")
    print("-" * 60)
    print(f"{'Rank':<6} {'Symbol':<10} {'Score':>12} {'Est. Allocation':>18}")
    print("-" * 60)

    allocation_per_stock = account * 0.95 / topk
    top_stocks = latest_preds.head(topk)

    for i, (stock, row) in enumerate(top_stocks.iterrows(), 1):
        print(f"{i:<6} {stock.upper():<10} {row['score']:>12.4f} ${allocation_per_stock:>15,.2f}")

    # 显示应避免的股票
    print(f"\nTOP {topk} STOCKS TO AVOID (lowest predicted return):")
    print("-" * 60)
    bottom_stocks = latest_preds.tail(topk).iloc[::-1]
    for i, (stock, row) in enumerate(bottom_stocks.iterrows(), 1):
        print(f"{i:<6} {stock.upper():<10} {row['score']:>12.4f}")

    # 保存预测结果
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_output = latest_preds.copy()
    pred_output.index = pred_output.index.str.upper()
    pred_output = pred_output.sort_values("score", ascending=False)
    pred_path = output_dir / f"ensemble_v4_predictions_{latest_date.strftime('%Y%m%d')}.csv"
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
    rec_path = output_dir / f"ensemble_v4_recommendations_{latest_date.strftime('%Y%m%d')}.csv"
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
    """运行交易回测，返回交易详情列表"""
    from qlib.backtest import backtest as qlib_backtest
    from qlib.contrib.evaluate import risk_analysis
    from data.stock_pools import STOCK_POOLS
    from utils.strategy import get_strategy_config

    print(f"\n{'='*70}")
    print("TRADING BACKTEST (Ensemble V4)")
    print(f"{'='*70}")
    print(f"Period: {test_start} to {test_end}")
    print(f"Initial Account: ${account:,.2f}")
    print(f"Rebalance Frequency: every {rebalance_freq} day(s)")
    print(f"Top-K stocks: {topk}")
    print(f"N-drop: {n_drop}")

    # 转换为 DataFrame
    pred_df = pred_ensemble.to_frame("score")
    pred_df = pred_df.reset_index()
    pred_df['instrument'] = pred_df['instrument'].str.lower()
    pred_df = pred_df.set_index(['datetime', 'instrument'])

    pred_dates = pred_df.index.get_level_values(0).unique().sort_values()
    print(f"\nPredictions shape: {len(pred_df)}")
    print(f"Date range: {pred_dates.min()} to {pred_dates.max()}")
    print(f"Unique dates: {len(pred_dates)}")

    # 显示最新日期的推荐股票
    latest_pred_date = pred_dates[-1]
    print(f"\n{'='*70}")
    print(f"TODAY'S RECOMMENDATIONS (based on {latest_pred_date.strftime('%Y-%m-%d')} predictions)")
    print(f"{'='*70}")

    latest_preds = pred_df.loc[latest_pred_date].sort_values("score", ascending=False)
    print(f"\nTop {topk} stocks to BUY:")
    print("-" * 50)
    for i, (stock, row) in enumerate(latest_preds.head(topk).iterrows(), 1):
        print(f"  {i:2d}. {stock.upper():<8s}  Score: {row['score']:>8.4f}")

    # 调整回测日期
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

    # 配置策略
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

    # 运行回测
    print(f"\n[*] Running backtest...")
    portfolio_metric_dict, indicator_dict = qlib_backtest(
        executor=executor_config,
        strategy=strategy_config,
        **backtest_config
    )

    print("    Backtest completed!")

    # 分析结果
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

        # 输出每日详细交易信息
        print(f"\n{'='*70}")
        print("DAILY TRADING DETAILS (Last 10 days)")
        print(f"{'='*70}")

        all_trading_details = collect_trading_details_from_positions(positions, report_df)
        print_trading_details(all_trading_details, show_last_n=10)

        # 保存报告
        output_dir = PROJECT_ROOT / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / f"ensemble_v4_daily_trading_report_{freq}.csv"
        report_df.to_csv(report_path)
        print(f"\n\nReport saved to: {report_path}")

        return all_trading_details

    return []


def main():
    parser = argparse.ArgumentParser(
        description='Daily trading script (Ensemble V4) - AE-MLP + AE-MLP(mkt-neutral) + CatBoost',
    )
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip data download step')
    parser.add_argument('--predict-only', action='store_true',
                        help='Only generate predictions, skip backtest')

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

    # Step 9: 运行预测或回测
    trading_details = []

    if args.predict_only:
        run_live_prediction(
            pred_ensemble=pred_ensemble,
            pred_dict=pred_dict,
            display_names=display_names,
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
    print("ENSEMBLE V4 TRADING SCRIPT COMPLETED")
    print(f"{'='*70}")

    # Step 10: 发送邮件报告
    if args.send_email and trading_details:
        print(f"\n{'='*60}")
        print("[STEP] Sending email report")
        print(f"{'='*60}")

        recent_details = trading_details[-args.email_days:] if len(trading_details) > args.email_days else trading_details

        if recent_details:
            model_info = "AE-MLP + AE-MLP-MN + CatBoost (Ensemble V4, zscore_weighted)"
            text_body = format_trading_details_text(recent_details, model_info)
            html_body = format_trading_details_html(recent_details, model_info)

            start_date = recent_details[0]['date']
            end_date = recent_details[-1]['date']
            subject = f"Ensemble V4 Trading Report: {start_date} to {end_date}"

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
