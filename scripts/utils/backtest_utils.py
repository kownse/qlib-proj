"""
Backtest utility functions for visualization and reporting.

Shared utilities for plotting backtest results and generating trade records.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_backtest_curve(report_df, args, freq, project_root, model_name=""):
    """
    绘制回测净值曲线图

    Parameters
    ----------
    report_df : pd.DataFrame
        回测报告，包含 return, bench, account 等列
    args : argparse.Namespace
        命令行参数，需要包含 topk, stock_pool, handler 属性
    freq : str
        频率标识
    project_root : Path
        项目根目录
    model_name : str
        模型名称，用于图表标题和文件名前缀（如 "CatBoost", "LightGBM"）
    """
    try:
        # 计算累计净值
        strategy_nav = (1 + report_df["return"]).cumprod()

        has_bench = "bench" in report_df.columns and not report_df["bench"].isna().all()
        if has_bench:
            bench_nav = (1 + report_df["bench"]).cumprod()

        # 创建图表
        fig, ax = plt.subplots(figsize=(14, 7))

        # 绘制策略净值曲线
        ax.plot(strategy_nav.index, strategy_nav.values,
                label=f'Strategy (topk={args.topk})',
                color='#2E86AB', linewidth=2)

        # 绘制基准净值曲线
        if has_bench:
            ax.plot(bench_nav.index, bench_nav.values,
                    label='Benchmark (Pool Avg)',
                    color='#A23B72', linewidth=2, linestyle='--')

        # 设置图表样式
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Net Asset Value (NAV)', fontsize=12)

        # 标题包含模型名称
        title_prefix = f'{model_name} ' if model_name else ''
        ax.set_title(f'{title_prefix}Backtest Performance: {args.stock_pool.upper()} + {args.handler}\n'
                     f'Period: {report_df.index.min().strftime("%Y-%m-%d")} to {report_df.index.max().strftime("%Y-%m-%d")}',
                     fontsize=14)

        # 格式化 x 轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        # 添加网格
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=11)

        # 添加最终收益标注
        final_strategy = strategy_nav.iloc[-1]
        ax.annotate(f'{(final_strategy-1)*100:.1f}%',
                    xy=(strategy_nav.index[-1], final_strategy),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=10, color='#2E86AB', fontweight='bold')

        if has_bench:
            final_bench = bench_nav.iloc[-1]
            ax.annotate(f'{(final_bench-1)*100:.1f}%',
                        xy=(bench_nav.index[-1], final_bench),
                        xytext=(10, 0), textcoords='offset points',
                        fontsize=10, color='#A23B72', fontweight='bold')

        # 添加统计信息文本框
        stats_text = f'Strategy Return: {(final_strategy-1)*100:.2f}%'
        if has_bench:
            stats_text += f'\nBenchmark Return: {(final_bench-1)*100:.2f}%'
            stats_text += f'\nExcess Return: {(final_strategy-final_bench)*100:.2f}%'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        plt.tight_layout()

        # 保存图表 - 文件名包含模型名称前缀
        file_prefix = f"{model_name.lower()}_" if model_name else ""
        output_path = project_root / "outputs" / f"{file_prefix}backtest_nav_curve_{freq}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Chart saved to: {output_path}")

    except Exception as e:
        print(f"    ✗ Chart generation failed: {e}")


def generate_trade_records(positions, args, freq, project_root, model_name=""):
    """
    生成每日交易记录 CSV

    Parameters
    ----------
    positions : dict
        每日持仓信息，格式为 {date: Position}
    args : argparse.Namespace
        命令行参数
    freq : str
        频率标识
    project_root : Path
        项目根目录
    model_name : str
        模型名称，用于文件名前缀（如 "catboost", "lgb"）
    """
    try:
        trade_records = []

        # 将 positions 转换为有序的日期列表
        sorted_dates = sorted(positions.keys())

        prev_holdings = {}  # 前一天的持仓

        for date in sorted_dates:
            pos = positions[date]

            # 获取当前持仓
            current_holdings = {}
            if hasattr(pos, 'get_stock_list'):
                for stock in pos.get_stock_list():
                    amount = pos.get_stock_amount(stock)
                    if amount > 0:
                        current_holdings[stock] = amount

            # 计算买入（新增或加仓）
            for stock, amount in current_holdings.items():
                prev_amount = prev_holdings.get(stock, 0)
                if amount > prev_amount:
                    trade_records.append({
                        'date': date,
                        'stock': stock,
                        'action': 'BUY',
                        'amount': amount - prev_amount,
                        'position_after': amount
                    })

            # 计算卖出（减仓或清仓）
            for stock, prev_amount in prev_holdings.items():
                current_amount = current_holdings.get(stock, 0)
                if current_amount < prev_amount:
                    trade_records.append({
                        'date': date,
                        'stock': stock,
                        'action': 'SELL',
                        'amount': prev_amount - current_amount,
                        'position_after': current_amount
                    })

            prev_holdings = current_holdings.copy()

        # 创建 DataFrame 并保存
        if trade_records:
            trade_df = pd.DataFrame(trade_records)
            trade_df = trade_df.sort_values(['date', 'action', 'stock'])

            # 文件名包含模型名称前缀
            file_prefix = f"{model_name.lower()}_" if model_name else ""
            output_path = project_root / "outputs" / f"{file_prefix}backtest_trades_{freq}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            trade_df.to_csv(output_path, index=False)

            # 统计信息
            buy_count = len(trade_df[trade_df['action'] == 'BUY'])
            sell_count = len(trade_df[trade_df['action'] == 'SELL'])
            unique_stocks = trade_df['stock'].nunique()

            print(f"    ✓ Trade records saved to: {output_path}")
            print(f"    Total trades: {len(trade_df)} (Buy: {buy_count}, Sell: {sell_count})")
            print(f"    Unique stocks traded: {unique_stocks}")

            # 显示前几条记录
            print(f"\n    Sample trades (first 10):")
            print(trade_df.head(10).to_string(index=False))
        else:
            print("    ⚠ No trades recorded")

    except Exception as e:
        print(f"    ✗ Trade record generation failed: {e}")
        import traceback
        traceback.print_exc()
