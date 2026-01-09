"""
回测共用逻辑

包含回测配置、执行和结果分析的共用函数
"""

import pickle
from pathlib import Path
import pandas as pd

from qlib.backtest import backtest as qlib_backtest
from qlib.contrib.evaluate import risk_analysis

from data.stock_pools import STOCK_POOLS
from utils.strategy import get_strategy_config
from utils.backtest_utils import plot_backtest_curve, generate_trade_records

from .common_config import PROJECT_ROOT


def run_backtest(model_path, dataset, pred, args, time_splits: dict, model_name: str,
                 load_model_func, get_feature_count_func=None):
    """
    使用 TopkDropoutStrategy 进行回测

    Parameters
    ----------
    model_path : Path or str
        训练好的模型保存路径
    dataset : DatasetH
        数据集
    pred : pd.Series
        预测结果
    args : argparse.Namespace
        命令行参数
    time_splits : dict
        时间划分配置
    model_name : str
        模型名称（用于显示和文件名）
    load_model_func : callable
        模型加载函数，接受 model_path 参数，返回加载的模型
    get_feature_count_func : callable, optional
        获取模型特征数量的函数，接受 model 参数，返回特征数量
    """
    print("\n" + "=" * 70)
    print(f"BACKTEST with TopkDropoutStrategy ({model_name})")
    print("=" * 70)

    # 加载模型和元数据
    model_path = Path(model_path)
    meta_path = model_path.with_suffix('.meta.pkl')

    print(f"\n[BT-0] Loading model from: {model_path}")
    loaded_model = load_model_func(model_path)
    print(f"    ✓ Model loaded successfully")

    # 获取特征数量
    if get_feature_count_func:
        try:
            n_features = get_feature_count_func(loaded_model)
        except Exception:
            n_features = "N/A"
    else:
        n_features = "N/A"
    print(f"    Model features: {n_features}")

    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            meta_data = pickle.load(f)
        print(f"    ✓ Metadata loaded")
        print(f"    Handler: {meta_data.get('handler', 'N/A')}")
        print(f"    Stock pool: {meta_data.get('stock_pool', 'N/A')}")
        print(f"    N-day: {meta_data.get('nday', 'N/A')}")
        if meta_data.get('top_k', 0) > 0:
            print(f"    Top-k features: {meta_data.get('top_k')}")
    else:
        print(f"    ⚠ Metadata file not found: {meta_path}")
        meta_data = {}

    # 将预测结果转换为 DataFrame 格式
    if isinstance(pred, pd.Series):
        pred_df = pred.to_frame("score")
    else:
        pred_df = pred

    print(f"\n[BT-1] Configuring backtest...")
    print(f"    Strategy: {args.strategy}")
    print(f"    Topk: {args.topk}")
    print(f"    N_drop: {args.n_drop}")
    print(f"    Account: ${args.account:,.0f}")
    print(f"    Rebalance Freq: every {args.rebalance_freq} day(s)")
    print(f"    Period: {time_splits['test_start']} to {time_splits['test_end']}")

    # 动态风险策略参数
    dynamic_risk_params = None
    if args.strategy == "dynamic_risk":
        dynamic_risk_params = {
            "lookback": args.risk_lookback,
            "drawdown_threshold": args.drawdown_threshold,
            "momentum_threshold": args.momentum_threshold,
            "risk_degree_high": args.risk_high,
            "risk_degree_medium": args.risk_medium,
            "risk_degree_normal": args.risk_normal,
            "market_proxy": args.market_proxy,
        }
        print(f"    Dynamic Risk Params:")
        print(f"      Lookback: {args.risk_lookback} days")
        print(f"      Drawdown Threshold: {args.drawdown_threshold:.1%}")
        print(f"      Momentum Threshold: {args.momentum_threshold:.1%}")
        print(f"      Risk Degrees: high={args.risk_high:.0%}, medium={args.risk_medium:.0%}, normal={args.risk_normal:.0%}")
        print(f"      Market Proxy: {args.market_proxy}")

    # 配置策略
    strategy_config = get_strategy_config(
        pred_df, args.topk, args.n_drop, args.rebalance_freq,
        strategy_type=args.strategy,
        dynamic_risk_params=dynamic_risk_params
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

    # 配置回测参数（美股市场）
    pool_symbols = STOCK_POOLS[args.stock_pool]

    backtest_config = {
        "start_time": time_splits['test_start'],
        "end_time": time_splits['test_end'],
        "account": args.account,
        "benchmark": pool_symbols,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": None,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0005,
            "min_cost": 1,
        },
    }

    print(f"\n[BT-2] Running backtest...")
    try:
        portfolio_metric_dict, indicator_dict = qlib_backtest(
            executor=executor_config,
            strategy=strategy_config,
            **backtest_config
        )

        print("    ✓ Backtest completed")

        # 分析结果
        print(f"\n[BT-3] Analyzing results...")

        for freq, (report_df, positions) in portfolio_metric_dict.items():
            _analyze_backtest_results(report_df, positions, freq, args, time_splits,
                                      model_name, PROJECT_ROOT)

        # 输出交易指标
        for freq, (indicator_df, indicator_obj) in indicator_dict.items():
            if indicator_df is not None and not indicator_df.empty:
                print(f"\n    Trading Indicators ({freq}):")
                print(f"    " + "-" * 50)
                print(indicator_df.head(20).to_string(index=True))
                if len(indicator_df) > 20:
                    print(f"    ... ({len(indicator_df)} rows total)")

    except Exception as e:
        print(f"\n    ✗ Backtest failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETED")
    print("=" * 70)


def _analyze_backtest_results(report_df: pd.DataFrame, positions, freq: str, args,
                              time_splits: dict, model_name: str, project_root: Path):
    """
    分析回测结果

    Parameters
    ----------
    report_df : pd.DataFrame
        回测报告
    positions : object
        持仓信息
    freq : str
        频率
    args : argparse.Namespace
        命令行参数
    time_splits : dict
        时间划分配置
    model_name : str
        模型名称
    project_root : Path
        项目根目录
    """
    print(f"\n    Frequency: {freq}")
    print(f"    Report shape: {report_df.shape}")
    print(f"    Date range: {report_df.index.min()} to {report_df.index.max()}")

    # 计算关键指标
    total_return = (report_df["return"] + 1).prod() - 1

    # 检查是否有 benchmark
    has_bench = "bench" in report_df.columns and not report_df["bench"].isna().all()
    if has_bench:
        bench_return = (report_df["bench"] + 1).prod() - 1
        excess_return = total_return - bench_return
        excess_return_series = report_df["return"] - report_df["bench"]
        analysis = risk_analysis(excess_return_series, freq=freq)
    else:
        bench_return = None
        excess_return = None
        analysis = risk_analysis(report_df["return"], freq=freq)

    print(f"\n    Performance Summary:")
    print(f"    " + "-" * 50)
    print(f"    Total Return:      {total_return:>10.2%}")
    if has_bench:
        print(f"    Benchmark Return:  {bench_return:>10.2%}")
        print(f"    Excess Return:     {excess_return:>10.2%}")
    else:
        print(f"    Benchmark Return:  N/A (no benchmark)")
    print(f"    " + "-" * 50)

    if analysis is not None and not analysis.empty:
        analysis_title = "Risk Analysis (Excess Return)" if has_bench else "Risk Analysis (Strategy Return)"
        print(f"\n    {analysis_title}:")
        print(f"    " + "-" * 50)
        for metric, value in analysis.items():
            if isinstance(value, (int, float)):
                print(f"    {metric:<25s}: {value:>10.4f}")
        print(f"    " + "-" * 50)

    # 输出详细报告
    print(f"\n    Daily Returns Statistics:")
    print(f"    " + "-" * 50)
    print(f"    Mean Daily Return:   {report_df['return'].mean():>10.4%}")
    print(f"    Std Daily Return:    {report_df['return'].std():>10.4%}")
    print(f"    Max Daily Return:    {report_df['return'].max():>10.4%}")
    print(f"    Min Daily Return:    {report_df['return'].min():>10.4%}")
    print(f"    Total Trading Days:  {len(report_df):>10d}")
    print(f"    " + "-" * 50)

    # 保存报告
    model_name_lower = model_name.lower()
    output_path = project_root / "outputs" / f"{model_name_lower}_backtest_report_{freq}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path)
    print(f"\n    ✓ Report saved to: {output_path}")

    # [BT-4] 绘制净值曲线图
    print(f"\n[BT-4] Generating performance chart...")
    plot_backtest_curve(report_df, args, freq, project_root, model_name=model_name)

    # [BT-5] 生成交易记录 CSV
    print(f"\n[BT-5] Generating trade records...")
    generate_trade_records(positions, args, freq, project_root, model_name=model_name_lower)
