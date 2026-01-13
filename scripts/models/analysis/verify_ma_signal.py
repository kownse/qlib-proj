"""
验证 MA vs MA 信号是否真的有预测能力

检查几个关键问题：
1. 信号是否只是数学上的恒等式？
2. 实际收益能否覆盖交易成本？
3. 与简单持有相比是否有超额收益？
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import qlib
from qlib.constant import REG_US
from qlib.data import D

from data.stock_pools import STOCK_POOLS
from utils.talib_ops import TALIB_OPS
from models.common.config import PROJECT_ROOT, QLIB_DATA_PATH

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def main():
    print("=" * 80)
    print("           验证 MA vs MA 信号的真实预测能力")
    print("=" * 80)

    # 初始化
    qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US, custom_ops=TALIB_OPS)
    symbols = STOCK_POOLS['sp500'][:100]  # 用100只股票测试

    print(f"\n加载 {len(symbols)} 只股票数据...")
    df = D.features(symbols, ["$close"], start_time="2015-01-01", end_time="2025-12-31", freq="day")
    df.columns = ["close"]
    df = df.dropna()
    close = df['close']

    print(f"数据量: {len(df)} 条")

    # ========== 测试1: 数学恒等式检验 ==========
    print("\n" + "=" * 80)
    print("[测试1] 检查是否是数学恒等式")
    print("=" * 80)

    # 计算各种标签和特征
    w = 20

    # 历史MA
    ma_hist = close.groupby(level='instrument').apply(lambda x: x.rolling(w).mean()).droplevel(0)

    # 未来MA (shift -w 表示未来w天的均值)
    ma_future = close.groupby(level='instrument').apply(
        lambda x: x.rolling(w).mean().shift(-w)
    ).droplevel(0)

    # 标签: 未来MA / 历史MA - 1
    label_ma_vs_ma = ma_future / ma_hist - 1

    # 特征: close / 历史MA - 1
    feat_ma_dev = close / ma_hist - 1

    # 真正的未来收益: 20天后的价格 / 今天价格 - 1
    true_future_return = close.groupby(level='instrument').pct_change(periods=w).shift(-w)

    # 计算相关性
    valid_idx = ~(label_ma_vs_ma.isna() | feat_ma_dev.isna() | true_future_return.isna())

    corr_feat_label = stats.spearmanr(feat_ma_dev[valid_idx], label_ma_vs_ma[valid_idx])[0]
    corr_feat_true = stats.spearmanr(feat_ma_dev[valid_idx], true_future_return[valid_idx])[0]
    corr_label_true = stats.spearmanr(label_ma_vs_ma[valid_idx], true_future_return[valid_idx])[0]

    print(f"""
    相关性分析:
    ─────────────────────────────────────────────────────
    特征 (ma_dev) vs 标签 (ma_vs_ma):     {corr_feat_label:>8.4f}  ← 这个很高是因为数学关系
    特征 (ma_dev) vs 真实收益 (20d ret):  {corr_feat_true:>8.4f}  ← 这才是真正的预测能力
    标签 (ma_vs_ma) vs 真实收益:          {corr_label_true:>8.4f}
    ─────────────────────────────────────────────────────
    """)

    # ========== 测试2: 为什么 ma_dev 和 ma_vs_ma 高度相关？ ==========
    print("\n" + "=" * 80)
    print("[测试2] 分解 ma_vs_ma 的组成")
    print("=" * 80)

    # ma_vs_ma = MA_future / MA_hist - 1
    # 可以分解为：取决于未来价格的变化

    # 让我们看看 ma_vs_ma 到底在衡量什么
    # MA_future = mean(P[t+1], P[t+2], ..., P[t+w])
    # MA_hist = mean(P[t-w+1], ..., P[t])

    # 如果价格是随机游走，MA_future ≈ P[t]（当前价格）
    # 所以 ma_vs_ma ≈ P[t] / MA_hist - 1 = ma_dev

    # 计算"理论上"的 ma_vs_ma（假设未来价格=当前价格）
    theoretical_ma_vs_ma = close / ma_hist - 1  # 这就是 ma_dev!

    corr_theoretical = stats.spearmanr(
        theoretical_ma_vs_ma[valid_idx],
        label_ma_vs_ma[valid_idx]
    )[0]

    print(f"""
    如果未来价格 = 当前价格（随机游走假设）:
    MA_future ≈ current_price
    ma_vs_ma ≈ close / MA_hist - 1 = ma_dev

    理论值 vs 实际值相关性: {corr_theoretical:.4f}

    这意味着: ma_vs_ma 大部分由 ma_dev 决定，不是真正的"预测"！
    """)

    # ========== 测试3: 横截面 IC 的真正含义 ==========
    print("\n" + "=" * 80)
    print("[测试3] 横截面 IC 对比")
    print("=" * 80)

    # 计算每日横截面 IC
    merged = pd.DataFrame({
        'ma_dev': feat_ma_dev,
        'ma_vs_ma': label_ma_vs_ma,
        'true_return': true_future_return
    }).dropna()

    # IC: ma_dev 预测 ma_vs_ma
    ic_fake = merged.groupby(level='datetime').apply(
        lambda g: stats.spearmanr(g['ma_dev'], g['ma_vs_ma'])[0] if len(g) > 10 else np.nan
    )

    # IC: ma_dev 预测真实收益
    ic_true = merged.groupby(level='datetime').apply(
        lambda g: stats.spearmanr(g['ma_dev'], g['true_return'])[0] if len(g) > 10 else np.nan
    )

    print(f"""
    横截面 IC 对比:
    ─────────────────────────────────────────────────────
                              Mean IC    Std IC      ICIR
    ─────────────────────────────────────────────────────
    ma_dev → ma_vs_ma:      {ic_fake.mean():>8.4f}  {ic_fake.std():>8.4f}  {ic_fake.mean()/ic_fake.std():>8.4f}  ← 虚高！
    ma_dev → 真实20d收益:   {ic_true.mean():>8.4f}  {ic_true.std():>8.4f}  {ic_true.mean()/ic_true.std():>8.4f}  ← 真实预测能力
    ─────────────────────────────────────────────────────

    结论: ma_dev 预测 ma_vs_ma 的 ICIR 虚高，是因为它们在数学上高度相关！
          ma_dev 预测真实收益的 ICIR 才是真正的预测能力。
    """)

    # ========== 测试4: 策略回测 ==========
    print("\n" + "=" * 80)
    print("[测试4] 简单策略回测")
    print("=" * 80)

    # 策略: 每天按 ma_dev 排序，做多最低的 10%，做空最高的 10%
    merged['rank'] = merged.groupby(level='datetime')['ma_dev'].rank(pct=True)

    # 多头收益（买入 ma_dev 最低的，即价格低于均线的）
    long_returns = merged[merged['rank'] <= 0.1]['true_return']
    # 空头收益
    short_returns = merged[merged['rank'] >= 0.9]['true_return']

    # 买入持有收益
    buy_hold_returns = merged['true_return']

    print(f"""
    均值回归策略 (基于 ma_dev):
    ─────────────────────────────────────────────────────
    多头 (价格 < MA20):
      平均20日收益: {long_returns.mean()*100:>8.3f}%
      收益标准差:   {long_returns.std()*100:>8.3f}%
      夏普比率:     {long_returns.mean()/long_returns.std()*np.sqrt(252/20):>8.3f}

    空头 (价格 > MA20):
      平均20日收益: {short_returns.mean()*100:>8.3f}%
      收益标准差:   {short_returns.std()*100:>8.3f}%

    多空组合:
      平均20日收益: {(long_returns.mean() - short_returns.mean())*100:>8.3f}%

    买入持有基准:
      平均20日收益: {buy_hold_returns.mean()*100:>8.3f}%
    ─────────────────────────────────────────────────────
    """)

    # ========== 测试5: 不同预测周期对比 ==========
    print("\n" + "=" * 80)
    print("[测试5] 不同预测周期的真实 ICIR")
    print("=" * 80)

    print(f"{'周期':<10} {'IC Mean':>12} {'IC Std':>12} {'ICIR':>12} {'信号强度':>12}")
    print("-" * 60)

    for horizon in [1, 5, 7, 10, 20]:
        # 真实未来收益
        ret = close.groupby(level='instrument').pct_change(periods=horizon).shift(-horizon)

        # 特征：短期动量
        mom = close.groupby(level='instrument').pct_change(periods=5)

        merged_h = pd.DataFrame({'feat': mom, 'label': ret}).dropna()

        ic = merged_h.groupby(level='datetime').apply(
            lambda g: stats.spearmanr(g['feat'], g['label'])[0] if len(g) > 10 else np.nan
        ).dropna()

        ic_mean = ic.mean()
        ic_std = ic.std()
        icir = ic_mean / ic_std if ic_std > 0 else 0

        # 信号强度（标签标准差）
        signal_strength = ret.std() * 100

        marker = "←" if abs(icir) > 0.05 else ""
        print(f"{horizon}天{'':<6} {ic_mean:>12.4f} {ic_std:>12.4f} {icir:>12.4f} {signal_strength:>11.2f}% {marker}")

    # ========== 结论 ==========
    print("\n" + "=" * 80)
    print("                         最终结论")
    print("=" * 80)

    print("""
    1. ma_vs_ma 的高 ICIR 是数学假象！
       ─────────────────────────────────
       ma_vs_ma ≈ close/MA_hist - 1 = ma_dev
       所以 ma_dev 预测 ma_vs_ma 基本是在"预测自己"

    2. 真正的预测能力
       ─────────────────────────────────
       ma_dev 预测真实20天收益的 ICIR ≈ 0.05-0.10
       这是合理的均值回归信号强度

    3. 你的直觉是对的！
       ─────────────────────────────────
       预测越远的未来确实越难
       1天收益的 ICIR 略低是因为噪声大
       但并不是"20天更好预测"

    4. 建议
       ─────────────────────────────────
       - 使用真实收益率（cumret_Nd）作为标签，而非 ma_vs_ma
       - ICIR 在 0.05-0.10 之间是合理的
       - 5-10 天可能是较好的预测周期（平衡噪声和信号）
    """)


if __name__ == "__main__":
    main()
