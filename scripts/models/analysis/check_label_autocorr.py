"""
检查波动率标签的自相关性和可预测性
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from utils.talib_ops import TALIB_OPS
from data.datahandler_ext import Alpha158_Volatility_TALib

# 配置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
QLIB_DATA_PATH = PROJECT_ROOT / "my_data" / "qlib_us"

TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]

TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
VALID_START = "2024-01-01"
VALID_END = "2024-12-31"

VOLATILITY_WINDOW = 2


def main():
    print("=" * 70)
    print("检查波动率标签的自相关性和可预测性")
    print("=" * 70)

    # 初始化 Qlib
    print("\n[1] 初始化 Qlib...")
    qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US, custom_ops=TALIB_OPS)

    # 创建 instruments 文件
    print("\n[2] 创建数据集...")
    instruments_path = PROJECT_ROOT / "my_data" / "instruments" / "test_symbols"
    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(instruments_path) + ".txt", "w") as f:
        for symbol in TEST_SYMBOLS:
            f.write(f"{symbol}\t2010-01-01\t2025-12-31\n")

    # 创建 DataHandler (Qlib auto-appends .txt)
    handler = Alpha158_Volatility_TALib(
        instruments=str(instruments_path),
        start_time=TRAIN_START,
        end_time=VALID_END,
        volatility_window=VOLATILITY_WINDOW,
    )

    # 创建 Dataset
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
    df_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_R)

    train_label = df_train["label"]["LABEL0"]
    valid_label = df_valid["label"]["LABEL0"]

    # 基本统计
    print("\n[4] 标签基本统计:")
    print(f"    Train samples: {len(train_label)}")
    print(f"    Valid samples: {len(valid_label)}")
    print(f"\n    Train label stats:")
    print(f"      Mean:   {train_label.mean():.6f}")
    print(f"      Std:    {train_label.std():.6f}")
    print(f"      Median: {train_label.median():.6f}")
    print(f"      Min:    {train_label.min():.6f}")
    print(f"      Max:    {train_label.max():.6f}")

    # 计算自相关
    print("\n[5] 计算自相关系数...")

    # 按股票分组计算自相关
    train_df = train_label.reset_index()
    train_df.columns = ["datetime", "instrument", "label"]

    autocorrs = {}
    for lag in [1, 2, 3, 5, 10, 20]:
        ac = train_df.groupby("instrument")["label"].apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
        )
        autocorrs[lag] = ac.mean()
        print(f"    Autocorr(lag={lag}): {autocorrs[lag]:.4f}")

    # 检查特征和标签的相关性
    print("\n[6] 检查特征和标签的相关性 (Top 10)...")
    train_features = df_train["feature"]

    correlations = {}
    for col in train_features.columns[:50]:  # 只检查前50个特征
        corr = train_features[col].corr(train_label)
        if not np.isnan(corr):
            correlations[col] = corr

    # 按绝对值排序
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\n    特征与标签相关性 (按绝对值排序):")
    for feat, corr in sorted_corr[:10]:
        print(f"      {feat}: {corr:.4f}")

    # 简单的可预测性测试：用前一天的标签预测今天
    print("\n[7] 简单预测测试...")

    # Naive 预测：用前一天的值
    naive_pred = train_df.groupby("instrument")["label"].shift(1)
    naive_mse = ((train_df["label"] - naive_pred) ** 2).mean()
    baseline_mse = train_label.var()  # 用均值预测的 MSE

    print(f"    Baseline MSE (predict mean): {baseline_mse:.8f}")
    print(f"    Naive MSE (predict t-1):     {naive_mse:.8f}")
    print(f"    Improvement: {(baseline_mse - naive_mse) / baseline_mse * 100:.2f}%")

    # 可视化
    print("\n[8] 生成可视化...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 标签分布
    ax1 = axes[0, 0]
    train_label.hist(bins=100, ax=ax1, alpha=0.7)
    ax1.axvline(train_label.mean(), color='r', linestyle='--', label=f'Mean: {train_label.mean():.4f}')
    ax1.set_title(f"Label Distribution (Volatility Window={VOLATILITY_WINDOW})")
    ax1.set_xlabel("Label Value")
    ax1.legend()

    # 2. 自相关图
    ax2 = axes[0, 1]
    lags = list(autocorrs.keys())
    values = list(autocorrs.values())
    ax2.bar(lags, values, color='steelblue')
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(0.1, color='red', linestyle='--', alpha=0.5, label='Threshold 0.1')
    ax2.axhline(-0.1, color='red', linestyle='--', alpha=0.5)
    ax2.set_title("Label Autocorrelation by Lag")
    ax2.set_xlabel("Lag (days)")
    ax2.set_ylabel("Autocorrelation")
    ax2.legend()

    # 3. 单个股票的标签时序
    ax3 = axes[1, 0]
    sample_stock = TEST_SYMBOLS[0]
    stock_data = train_df[train_df["instrument"] == sample_stock].set_index("datetime")["label"]
    stock_data[-200:].plot(ax=ax3, alpha=0.7)
    ax3.set_title(f"{sample_stock} Label Time Series (Last 200 days)")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Label Value")

    # 4. 特征相关性
    ax4 = axes[1, 1]
    top_features = [f for f, _ in sorted_corr[:15]]
    top_corr_values = [c for _, c in sorted_corr[:15]]
    colors = ['green' if c > 0 else 'red' for c in top_corr_values]
    ax4.barh(top_features, top_corr_values, color=colors, alpha=0.7)
    ax4.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_title("Top 15 Feature-Label Correlations")
    ax4.set_xlabel("Correlation")

    plt.tight_layout()

    output_path = PROJECT_ROOT / "outputs" / "label_autocorr_analysis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"    Saved to: {output_path}")

    # 结论
    print("\n" + "=" * 70)
    print("分析结论:")
    print("=" * 70)

    if autocorrs[1] > 0.3:
        print("✓ 自相关性较强 (>0.3)，标签有一定的可预测性")
    elif autocorrs[1] > 0.1:
        print("△ 自相关性中等 (0.1-0.3)，标签有弱可预测性")
    else:
        print("✗ 自相关性很弱 (<0.1)，标签难以从历史值预测")

    if abs(sorted_corr[0][1]) > 0.1:
        print(f"✓ 最强特征相关性: {sorted_corr[0][0]} = {sorted_corr[0][1]:.4f}")
    else:
        print("✗ 所有特征与标签相关性都很弱 (<0.1)")

    if train_label.std() < 0.01:
        print(f"⚠ 标签标准差很小 ({train_label.std():.6f})，模型可能直接预测均值")

    print("\n建议:")
    if autocorrs[1] < 0.1 and abs(sorted_corr[0][1]) < 0.1:
        print("  - 当前特征集可能不足以预测波动率")
        print("  - 建议尝试: VIX数据、期权隐含波动率、新闻情绪等外部数据")
        print("  - 或者简化任务: 预测波动率方向（分类）而非精确值（回归）")


if __name__ == "__main__":
    main()
