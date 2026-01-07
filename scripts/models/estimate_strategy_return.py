"""
基于 IC 估算 TopkDropoutStrategy 的预期收益

这个脚本帮助理解：给定 IC，策略能否盈利？
"""

import numpy as np
import pandas as pd


def estimate_return_from_ic(
    ic_mean: float,
    ic_std: float,
    label_std: float = 0.02,  # 日收益率标准差，约2%
    topk: int = 10,
    n_stocks: int = 100,
    trading_days: int = 252,
    turnover_rate: float = 0.2,  # 每日换手率
    transaction_cost: float = 0.002,  # 单边交易成本 0.2%
):
    """
    估算基于 IC 的策略预期收益

    理论背景：
    - IC 衡量预测分数与实际收益的相关性
    - 如果我们选 top-k 股票，预期收益与 IC 成正比
    """

    print("=" * 60)
    print("TopkDropoutStrategy 收益估算")
    print("=" * 60)

    # 输入参数
    print(f"\n输入参数:")
    print(f"  IC Mean:        {ic_mean:.4f}")
    print(f"  IC Std:         {ic_std:.4f}")
    print(f"  ICIR:           {ic_mean/ic_std:.4f}")
    print(f"  Label Std:      {label_std:.2%} (日收益波动)")
    print(f"  Top-k:          {topk}")
    print(f"  股票池:         {n_stocks}")
    print(f"  交易日:         {trading_days}")
    print(f"  日换手率:       {turnover_rate:.1%}")
    print(f"  单边交易成本:   {transaction_cost:.2%}")

    # 理论计算
    # 当 IC > 0 时，top-k 股票的预期超额收益
    # 参考: Grinold & Kahn, "Active Portfolio Management"
    # E[r_active] ≈ IC × σ_label × z_score
    # 其中 z_score 是 top-k 在正态分布中的期望值

    # Top-k 的期望 z-score（假设正态分布）
    from scipy import stats

    # top-k 占比
    top_ratio = topk / n_stocks
    # Top-k 的期望 z-score
    z_top = stats.norm.ppf(1 - top_ratio / 2)  # 双边

    # 预期日超额收益
    daily_alpha = ic_mean * label_std * z_top

    # 换手带来的交易成本
    daily_cost = turnover_rate * 2 * transaction_cost  # 双边成本

    # 净日收益
    daily_net = daily_alpha - daily_cost

    # 年化
    annual_alpha = daily_alpha * trading_days
    annual_cost = daily_cost * trading_days
    annual_net = daily_net * trading_days

    # 夏普比率估算
    # 日收益标准差 ≈ IC_std × label_std × z_top
    daily_std = ic_std * label_std * z_top
    annual_std = daily_std * np.sqrt(trading_days)
    sharpe = annual_net / annual_std if annual_std > 0 else 0

    print(f"\n理论估算:")
    print(f"  Top-k z-score:  {z_top:.2f}")
    print(f"  日超额收益:     {daily_alpha:.4%}")
    print(f"  日交易成本:     {daily_cost:.4%}")
    print(f"  日净收益:       {daily_net:.4%}")

    print(f"\n年化估算:")
    print(f"  年化超额收益:   {annual_alpha:.2%}")
    print(f"  年化交易成本:   {annual_cost:.2%}")
    print(f"  年化净收益:     {annual_net:.2%}")
    print(f"  年化波动率:     {annual_std:.2%}")
    print(f"  夏普比率:       {sharpe:.2f}")

    print(f"\n结论:")
    if annual_net > 0.05:
        print(f"  ✓ 预期年化净收益 {annual_net:.2%}，策略可能盈利")
    elif annual_net > 0:
        print(f"  △ 预期年化净收益 {annual_net:.2%}，勉强盈利，风险较大")
    else:
        print(f"  ✗ 预期年化净收益 {annual_net:.2%}，策略可能亏损")

    # 敏感性分析
    print(f"\n敏感性分析 - 不同 IC 下的预期收益:")
    print("-" * 50)
    print(f"{'IC':<10} {'年化Alpha':<12} {'年化成本':<12} {'年化净收益':<12}")
    print("-" * 50)

    for test_ic in [0.01, 0.02, 0.03, 0.04, 0.05]:
        test_daily_alpha = test_ic * label_std * z_top
        test_annual_alpha = test_daily_alpha * trading_days
        test_annual_net = test_annual_alpha - annual_cost
        print(f"{test_ic:<10.2f} {test_annual_alpha:<12.2%} {annual_cost:<12.2%} {test_annual_net:<12.2%}")

    return {
        'annual_alpha': annual_alpha,
        'annual_cost': annual_cost,
        'annual_net': annual_net,
        'sharpe': sharpe,
    }


if __name__ == "__main__":
    # 使用你的实际 IC 结果
    results = estimate_return_from_ic(
        ic_mean=0.0183,
        ic_std=0.1166,
        topk=10,       # TopkDropoutStrategy 默认 topk
        n_stocks=500,  # SP100 股票池
    )
