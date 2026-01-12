"""
参数网格搜索：寻找最优的 topk, n_drop, nday, rebalance_freq 组合

使用方法:
    python scripts/models/param_search.py --model-path ./my_models/your_model.txt --stock-pool sp500 --handler alpha360
"""

import sys
from pathlib import Path
import itertools
import pandas as pd

script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

import pickle
import lightgbm as lgb

from models.common_config import HANDLER_CONFIG, PROJECT_ROOT, QLIB_DATA_PATH
from models.training_utils import (
    init_qlib,
    create_data_handler,
    create_dataset,
    get_time_splits,
    prepare_test_data_for_prediction,
)
from data.stock_pools import STOCK_POOLS
from utils.strategy import get_strategy_config

from qlib.backtest import backtest as qlib_backtest


def run_single_backtest(pred_df, args, time_splits, topk, n_drop, rebalance_freq):
    """运行单次回测，返回结果指标"""

    pool_symbols = STOCK_POOLS[args.stock_pool]

    strategy_config = get_strategy_config(
        pred_df, topk, n_drop, rebalance_freq,
        strategy_type="topk",  # 使用简单策略
    )

    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    }

    backtest_config = {
        "start_time": time_splits['test_start'],
        "end_time": time_splits['test_end'],
        "account": 1000000,
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

    try:
        portfolio_metric_dict, indicator_dict = qlib_backtest(
            executor=executor_config,
            strategy=strategy_config,
            **backtest_config
        )

        # 提取结果
        for freq, (report_df, positions) in portfolio_metric_dict.items():
            total_return = (report_df["return"] + 1).prod() - 1

            has_bench = "bench" in report_df.columns and not report_df["bench"].isna().all()
            if has_bench:
                bench_return = (report_df["bench"] + 1).prod() - 1
                excess_return = total_return - bench_return
            else:
                bench_return = 0
                excess_return = total_return

            # 计算夏普比率
            daily_returns = report_df["return"]
            sharpe = daily_returns.mean() / daily_returns.std() * (252 ** 0.5) if daily_returns.std() > 0 else 0

            # 计算最大回撤
            cumulative = (1 + daily_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            return {
                "total_return": total_return,
                "bench_return": bench_return,
                "excess_return": excess_return,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
            }

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Parameter Grid Search')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--stock-pool', type=str, default='sp500', choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--handler', type=str, default='alpha360', choices=list(HANDLER_CONFIG.keys()))
    parser.add_argument('--nday', type=int, default=5, help='Prediction window (for data handler)')
    args = parser.parse_args()

    # 参数网格
    TOPK_VALUES = [5, 10, 15, 20, 30]
    N_DROP_VALUES = [0, 1, 2, 3]
    REBALANCE_FREQ_VALUES = [1, 3, 5, 10]

    print("=" * 70)
    print("PARAMETER GRID SEARCH")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Stock Pool: {args.stock_pool}")
    print(f"Handler: {args.handler}")
    print(f"\nGrid:")
    print(f"  topk: {TOPK_VALUES}")
    print(f"  n_drop: {N_DROP_VALUES}")
    print(f"  rebalance_freq: {REBALANCE_FREQ_VALUES}")
    print(f"  Total combinations: {len(TOPK_VALUES) * len(N_DROP_VALUES) * len(REBALANCE_FREQ_VALUES)}")
    print("=" * 70)

    # 加载模型和元数据
    model_path = Path(args.model_path)
    meta_path = model_path.with_suffix('.meta.pkl')

    with open(meta_path, 'rb') as f:
        meta_data = pickle.load(f)

    handler_name = meta_data.get('handler', args.handler)
    handler_config = HANDLER_CONFIG[handler_name]
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = get_time_splits(False)

    # 初始化
    init_qlib(handler_config['use_talib'])

    # 创建参数对象
    class Args:
        pass
    data_args = Args()
    data_args.handler = handler_name
    data_args.stock_pool = args.stock_pool
    data_args.nday = meta_data.get('nday', args.nday)
    data_args.news_features = 'core'
    data_args.news_rolling = False

    handler = create_data_handler(data_args, handler_config, symbols, time_splits)
    dataset = create_dataset(handler, time_splits)

    # 加载模型并生成预测
    print("\n[1] Loading model and generating predictions...")
    model = lgb.Booster(model_file=str(model_path))
    num_features = model.num_feature()

    feature_names = meta_data.get('feature_names', [])
    top_k = meta_data.get('top_k', 0)

    if top_k > 0 and feature_names:
        test_data = dataset.prepare("test", col_set="feature")
        available_features = [f for f in feature_names if f in test_data.columns]
        test_data_filtered = test_data[available_features]
    else:
        test_data_filtered = prepare_test_data_for_prediction(dataset, num_features)

    pred_values = model.predict(test_data_filtered.values)
    pred_df = pd.Series(pred_values, index=test_data_filtered.index, name='score').to_frame("score")

    print(f"    Predictions shape: {pred_df.shape}")

    # 运行网格搜索
    print("\n[2] Running grid search...")
    results = []
    total = len(TOPK_VALUES) * len(N_DROP_VALUES) * len(REBALANCE_FREQ_VALUES)
    count = 0

    for topk, n_drop, rebalance_freq in itertools.product(TOPK_VALUES, N_DROP_VALUES, REBALANCE_FREQ_VALUES):
        count += 1
        print(f"\n  [{count}/{total}] topk={topk}, n_drop={n_drop}, rebalance_freq={rebalance_freq}")

        # n_drop 不能大于 topk
        if n_drop >= topk:
            print(f"    Skip: n_drop >= topk")
            continue

        result = run_single_backtest(pred_df, args, time_splits, topk, n_drop, rebalance_freq)

        if result:
            result['topk'] = topk
            result['n_drop'] = n_drop
            result['rebalance_freq'] = rebalance_freq
            results.append(result)

            print(f"    Return: {result['total_return']:.2%}, Excess: {result['excess_return']:.2%}, "
                  f"Sharpe: {result['sharpe']:.2f}, MaxDD: {result['max_drawdown']:.2%}")

    # 整理结果
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(results)

    # 按超额收益排序
    df_sorted = df.sort_values('excess_return', ascending=False)

    print("\nTop 10 by Excess Return:")
    print("-" * 90)
    print(f"{'topk':>6} {'n_drop':>7} {'rebal':>6} {'Return':>10} {'Excess':>10} {'Sharpe':>8} {'MaxDD':>10}")
    print("-" * 90)

    for _, row in df_sorted.head(10).iterrows():
        print(f"{row['topk']:>6} {row['n_drop']:>7} {row['rebalance_freq']:>6} "
              f"{row['total_return']:>10.2%} {row['excess_return']:>10.2%} "
              f"{row['sharpe']:>8.2f} {row['max_drawdown']:>10.2%}")

    print("-" * 90)

    # 按夏普比率排序
    df_sharpe = df.sort_values('sharpe', ascending=False)

    print("\nTop 10 by Sharpe Ratio:")
    print("-" * 90)
    for _, row in df_sharpe.head(10).iterrows():
        print(f"{row['topk']:>6} {row['n_drop']:>7} {row['rebalance_freq']:>6} "
              f"{row['total_return']:>10.2%} {row['excess_return']:>10.2%} "
              f"{row['sharpe']:>8.2f} {row['max_drawdown']:>10.2%}")

    print("-" * 90)

    # 保存结果
    output_path = PROJECT_ROOT / "outputs" / "param_search_results.csv"
    df_sorted.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # 最优参数
    best = df_sorted.iloc[0]
    print(f"\n{'='*70}")
    print(f"BEST PARAMETERS (by Excess Return):")
    print(f"  topk={int(best['topk'])}, n_drop={int(best['n_drop'])}, rebalance_freq={int(best['rebalance_freq'])}")
    print(f"  Excess Return: {best['excess_return']:.2%}")
    print(f"  Sharpe Ratio: {best['sharpe']:.2f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
