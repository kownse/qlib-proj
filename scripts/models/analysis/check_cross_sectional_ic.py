"""
检查横截面 IC (Cross-Sectional Information Coefficient)

这是 Qlib 评估模型的核心指标，与简单的特征-标签相关性不同。

关键区别：
1. 特征-标签相关：所有样本混在一起计算
2. 横截面 IC：每天单独计算，评估的是同一天内股票排名的准确性
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from utils.talib_ops import TALIB_OPS
from data.datahandler_ext import Alpha158_Volatility_TALib

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
QLIB_DATA_PATH = PROJECT_ROOT / "my_data" / "qlib_us"

TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
                "WMT", "PG", "UNH", "MA", "HD", "DIS", "NFLX", "ADBE", "CRM", "PYPL"]

TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
VALID_START = "2024-01-01"
VALID_END = "2024-12-31"

VOLATILITY_WINDOW = 2


def compute_cross_sectional_ic(features_df, label_series, method='spearman'):
    """
    计算每日横截面 IC

    Args:
        features_df: 特征 DataFrame，index 为 (datetime, instrument)
        label_series: 标签 Series
        method: 'pearson' 或 'spearman'

    Returns:
        ic_df: 每天每个特征的 IC
    """
    # 合并特征和标签
    df = features_df.copy()
    df['label'] = label_series

    # 按日期分组计算 IC
    ic_results = {}

    for feat in features_df.columns[:50]:  # 只检查前50个特征
        def daily_ic(group):
            feat_vals = group[feat].dropna()
            label_vals = group['label'].loc[feat_vals.index]
            if len(feat_vals) < 5:
                return np.nan
            if method == 'spearman':
                return stats.spearmanr(feat_vals, label_vals)[0]
            else:
                return stats.pearsonr(feat_vals, label_vals)[0]

        ic_series = df.groupby(level='datetime').apply(daily_ic)
        ic_results[feat] = ic_series

    return pd.DataFrame(ic_results)


def main():
    print("=" * 70)
    print("检查横截面 IC (Cross-Sectional Information Coefficient)")
    print("=" * 70)
    print("\n这是 Qlib 评估模型的核心指标！")
    print("与你之前检查的简单相关性不同，这里是每天单独计算 IC")

    # 初始化 Qlib
    print("\n[1] 初始化 Qlib...")
    qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US, custom_ops=TALIB_OPS)

    # 创建 instruments 文件
    print("\n[2] 创建数据集...")
    instruments_path = PROJECT_ROOT / "my_data" / "instruments" / "test_symbols_cs"
    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(instruments_path) + ".txt", "w") as f:
        for symbol in TEST_SYMBOLS:
            f.write(f"{symbol}\t2010-01-01\t2025-12-31\n")

    handler = Alpha158_Volatility_TALib(
        instruments=str(instruments_path),
        start_time=TRAIN_START,
        end_time=VALID_END,
        volatility_window=VOLATILITY_WINDOW,
    )

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (TRAIN_START, TRAIN_END),
            "valid": (VALID_START, VALID_END),
        },
    )

    # 获取数据
    print("\n[3] 加载数据...")
    df_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_R)

    train_features = df_train["feature"]
    train_label = df_train["label"]["LABEL0"]

    # 基本统计
    print(f"\n[4] 数据统计:")
    print(f"    样本数: {len(train_label)}")
    print(f"    股票数: {len(TEST_SYMBOLS)}")
    print(f"    日期范围: {TRAIN_START} to {TRAIN_END}")

    # 计算横截面 IC
    print("\n[5] 计算横截面 IC (每天分别计算)...")
    print("    这可能需要一些时间...")

    ic_df = compute_cross_sectional_ic(train_features, train_label, method='spearman')

    # IC 统计
    print("\n[6] 横截面 IC 统计 (Spearman Rank IC):")
    ic_mean = ic_df.mean()
    ic_std = ic_df.std()
    icir = ic_mean / ic_std  # IC Information Ratio

    # 按 ICIR 排序
    sorted_features = icir.abs().sort_values(ascending=False)

    print("\n    Top 20 特征 (按 |ICIR| 排序):")
    print("-" * 60)
    print(f"    {'Feature':<30} {'IC_mean':>10} {'IC_std':>10} {'ICIR':>10}")
    print("-" * 60)
    for feat in sorted_features.head(20).index:
        print(f"    {feat:<30} {ic_mean[feat]:>10.4f} {ic_std[feat]:>10.4f} {icir[feat]:>10.4f}")

    # 对比之前的简单相关
    print("\n[7] 对比：简单相关 vs 横截面 IC")
    print("-" * 70)

    simple_corr = {}
    for col in train_features.columns[:50]:
        corr = train_features[col].corr(train_label)
        if not np.isnan(corr):
            simple_corr[col] = corr

    print(f"    {'Feature':<30} {'Simple Corr':>12} {'CS IC Mean':>12} {'ICIR':>10}")
    print("-" * 70)
    for feat in sorted_features.head(15).index:
        sc = simple_corr.get(feat, np.nan)
        print(f"    {feat:<30} {sc:>12.4f} {ic_mean[feat]:>12.4f} {icir[feat]:>10.4f}")

    # 可视化
    print("\n[8] 生成可视化...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. IC 时序
    ax1 = axes[0, 0]
    best_feat = sorted_features.head(1).index[0]
    ic_df[best_feat].plot(ax=ax1, alpha=0.7)
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(ic_mean[best_feat], color='red', linestyle='--', label=f'Mean IC: {ic_mean[best_feat]:.4f}')
    ax1.set_title(f"Daily IC Time Series: {best_feat}")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("IC")
    ax1.legend()

    # 2. IC 分布
    ax2 = axes[0, 1]
    ic_df[best_feat].hist(bins=50, ax=ax2, alpha=0.7)
    ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(ic_mean[best_feat], color='red', linestyle='--', label=f'Mean: {ic_mean[best_feat]:.4f}')
    ax2.set_title(f"IC Distribution: {best_feat}")
    ax2.set_xlabel("IC Value")
    ax2.legend()

    # 3. Top 特征 IC
    ax3 = axes[1, 0]
    top_feats = sorted_features.head(15).index
    ic_means = [ic_mean[f] for f in top_feats]
    colors = ['green' if x > 0 else 'red' for x in ic_means]
    ax3.barh(range(len(top_feats)), ic_means, color=colors, alpha=0.7)
    ax3.set_yticks(range(len(top_feats)))
    ax3.set_yticklabels(top_feats)
    ax3.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title("Top 15 Features by |ICIR|")
    ax3.set_xlabel("Mean IC")

    # 4. 简单相关 vs 横截面 IC
    ax4 = axes[1, 1]
    common_feats = list(set(simple_corr.keys()) & set(ic_mean.index))[:30]
    x = [simple_corr[f] for f in common_feats]
    y = [ic_mean[f] for f in common_feats]
    ax4.scatter(x, y, alpha=0.5)
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel("Simple Correlation")
    ax4.set_ylabel("Cross-Sectional IC Mean")
    ax4.set_title("Simple Corr vs CS IC")

    # 添加趋势线
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax4.plot(sorted(x), p(sorted(x)), "r--", alpha=0.5)

    plt.tight_layout()

    output_path = PROJECT_ROOT / "outputs" / "cross_sectional_ic_analysis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"    Saved to: {output_path}")

    # 结论
    print("\n" + "=" * 70)
    print("关键发现：")
    print("=" * 70)

    best_icir = icir.abs().max()
    best_ic = ic_mean[icir.abs().idxmax()]

    if best_icir > 0.1:
        print(f"✓ 存在有效的预测信号！")
        print(f"  最佳特征: {icir.abs().idxmax()}")
        print(f"  ICIR: {best_icir:.4f}, Mean IC: {best_ic:.4f}")
    elif best_icir > 0.05:
        print(f"△ 存在弱预测信号")
        print(f"  最佳特征: {icir.abs().idxmax()}")
        print(f"  ICIR: {best_icir:.4f}, Mean IC: {best_ic:.4f}")
    else:
        print(f"✗ 预测信号很弱")

    print("\n" + "=" * 70)
    print("Qlib 模型工作原理说明：")
    print("=" * 70)
    print("""
    1. Qlib 使用 TopkDropoutStrategy 策略
       - 每天选择预测收益最高的 top-k 只股票
       - 不需要预测准确的收益率数值
       - 只需要股票的相对排名大致正确

    2. 即使 IC 只有 0.03-0.05，也能产生超额收益
       - 因为策略依赖的是相对排名，不是绝对预测
       - 每天的小优势累积起来就是可观的收益

    3. 你之前的分析检查的是：
       - 单特征与标签的简单相关（所有样本混在一起）
       - 这不是 Qlib 评估模型的方式

    4. Qlib 评估的是：
       - 每天单独计算 IC（横截面相关）
       - ICIR = IC均值 / IC标准差（信息比率）
       - 稳定的弱信号比不稳定的强信号更有价值
    """)


if __name__ == "__main__":
    main()
