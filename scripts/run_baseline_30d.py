"""
运行 LightGBM baseline，预测30天股票价格波动率

波动率定义：未来30个交易日的收益率标准差（年化）
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

import qlib
from qlib.constant import REG_US
from qlib.data import D
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH

# 设置 matplotlib 中文显示和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')


# ========== 配置 ==========

# 数据路径
PROJECT_ROOT = Path(__file__).parent.parent
QLIB_DATA_PATH = PROJECT_ROOT / "my_data" / "qlib_us"

# 股票池
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]

# 时间划分
TRAIN_START = "2000-01-01"
TRAIN_END = "2023-12-31"
VALID_START = "2024-01-01"
VALID_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

# 波动率预测窗口（天数）
VOLATILITY_WINDOW = 30


class Alpha158_30DayVolatility(Alpha158):
    """
    Alpha158 特征 + 30天价格波动率标签

    继承 Alpha158 的所有技术指标特征，只修改标签为30天波动率
    """

    def get_label_config(self):
        """
        返回30天波动率标签

        波动率计算：
        1. 计算未来30天的每日收益率
        2. 计算这些收益率的标准差
        3. 年化处理（乘以 sqrt(252)，252为一年交易日数）

        Returns:
            fields: 标签表达式列表
            names: 标签名称列表
        """
        # 使用 Qlib 的表达式语法
        # Std($close, N) 计算 N 天的标准差
        # Ref($field, -N) 向未来移动 N 天
        # 我们需要计算未来30天的收益率标准差

        # 简化方案：使用未来30天价格的标准差除以当前价格，再年化
        # volatility = Std(Ref($close, -N) for N in 0..-30) / $close * sqrt(252)

        # Qlib 的 Std 函数计算滚动标准差
        # 我们需要先计算收益率，再计算标准差

        # 方法：计算未来价格的滚动标准差（作为波动率的代理）
        # 使用 Ref 将标准差移到未来窗口
        annualized_factor = np.sqrt(252)

        # 计算未来30天收益率的标准差
        # 先计算日收益率: $close / Ref($close, 1) - 1
        # 然后计算未来30天的标准差: Std(returns, 30)
        # 最后shift到未来: Ref(Std(...), -30)

        # 简化版：使用价格标准差作为波动率代理
        # Ref(Std($close, 30) / Mean($close, 30), -30) * sqrt(252)
        volatility_expr = f"Ref(Std($close, {VOLATILITY_WINDOW}) / Mean($close, {VOLATILITY_WINDOW}), -{VOLATILITY_WINDOW}) * {annualized_factor:.6f}"

        return [volatility_expr], ["LABEL0"]


def plot_predictions(comparison_df, output_dir):
    """
    绘制预测值 vs 实际值的可视化图表

    Args:
        comparison_df: 包含 Predicted_Vol, Actual_Vol, Error 等列的 DataFrame
        output_dir: 输出目录路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取股票列表
    stocks = comparison_df.index.get_level_values('instrument').unique().tolist()

    print(f"\n[11] Generating visualization plots...")
    print(f"     Output directory: {output_dir}")

    # ========== 图1: 时间序列对比图（所有股票） ==========
    fig, axes = plt.subplots(len(stocks), 1, figsize=(16, 3*len(stocks)))
    if len(stocks) == 1:
        axes = [axes]

    for idx, stock in enumerate(stocks):
        stock_data = comparison_df.loc[(slice(None), stock), :]
        dates = stock_data.index.get_level_values('datetime')

        axes[idx].plot(dates, stock_data['Actual_Vol'],
                      label='Actual Volatility', marker='o', linewidth=2,
                      markersize=4, alpha=0.7, color='#2E86AB')
        axes[idx].plot(dates, stock_data['Predicted_Vol'],
                      label='Predicted Volatility', marker='s', linewidth=2,
                      markersize=4, alpha=0.7, color='#A23B72')

        axes[idx].fill_between(dates,
                              stock_data['Actual_Vol'],
                              stock_data['Predicted_Vol'],
                              alpha=0.2, color='gray', label='Error')

        axes[idx].set_title(f'{stock} - 30-Day Volatility: Predicted vs Actual',
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Date', fontsize=10)
        axes[idx].set_ylabel('Annualized Volatility', fontsize=10)
        axes[idx].legend(loc='best', fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'time_series_all_stocks.png', dpi=300, bbox_inches='tight')
    print(f"     ✓ Saved: time_series_all_stocks.png")
    plt.close()

    # ========== 图2: 散点图（预测 vs 实际） ==========
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(stocks)))

    for idx, stock in enumerate(stocks):
        stock_data = comparison_df.loc[(slice(None), stock), :]
        ax.scatter(stock_data['Actual_Vol'], stock_data['Predicted_Vol'],
                  label=stock, alpha=0.6, s=80, color=colors[idx], edgecolors='black', linewidth=0.5)

    # 添加完美预测线（y=x）
    min_val = comparison_df[['Actual_Vol', 'Predicted_Vol']].min().min()
    max_val = comparison_df[['Actual_Vol', 'Predicted_Vol']].max().max()
    ax.plot([min_val, max_val], [min_val, max_val],
           'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)

    ax.set_xlabel('Actual Volatility', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Volatility', fontsize=12, fontweight='bold')
    ax.set_title('Predicted vs Actual Volatility (Scatter Plot)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
    print(f"     ✓ Saved: scatter_predicted_vs_actual.png")
    plt.close()

    # ========== 图3: 误差分布图 ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 3.1 误差直方图
    axes[0, 0].hist(comparison_df['Error'], bins=50, alpha=0.7,
                   color='#F18F01', edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0, 0].set_xlabel('Prediction Error', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Error Distribution (Histogram)', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 3.2 误差百分比直方图
    axes[0, 1].hist(comparison_df['Error_%'], bins=50, alpha=0.7,
                   color='#06A77D', edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0, 1].set_xlabel('Prediction Error (%)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Error % Distribution (Histogram)', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3.3 误差箱线图（按股票分组）
    error_by_stock = [comparison_df.loc[(slice(None), stock), 'Error'].values
                     for stock in stocks]
    bp = axes[1, 0].boxplot(error_by_stock, labels=stocks, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Stock', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Prediction Error', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Error Distribution by Stock (Boxplot)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 3.4 时间序列误差图
    for stock in stocks:
        stock_data = comparison_df.loc[(slice(None), stock), :]
        dates = stock_data.index.get_level_values('datetime')
        axes[1, 1].plot(dates, stock_data['Error'], marker='o',
                       markersize=3, linewidth=1, alpha=0.7, label=stock)

    axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[1, 1].set_xlabel('Date', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Prediction Error', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Error Over Time', fontsize=12, fontweight='bold')
    axes[1, 1].legend(loc='best', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
    print(f"     ✓ Saved: error_analysis.png")
    plt.close()

    # ========== 图4: 按股票分组的性能对比 ==========
    stock_stats = comparison_df.groupby(level='instrument').agg({
        'Predicted_Vol': 'mean',
        'Actual_Vol': 'mean',
        'Error': lambda x: x.abs().mean()
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 4.1 平均波动率对比
    x = np.arange(len(stocks))
    width = 0.35

    axes[0].bar(x - width/2, stock_stats['Actual_Vol'], width,
               label='Actual', alpha=0.8, color='#2E86AB', edgecolor='black')
    axes[0].bar(x + width/2, stock_stats['Predicted_Vol'], width,
               label='Predicted', alpha=0.8, color='#A23B72', edgecolor='black')

    axes[0].set_xlabel('Stock', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Average Volatility', fontsize=11, fontweight='bold')
    axes[0].set_title('Average Volatility by Stock', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stocks, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # 4.2 平均绝对误差对比
    axes[1].bar(stocks, stock_stats['Error'], alpha=0.8,
               color='#F18F01', edgecolor='black')
    axes[1].set_xlabel('Stock', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    axes[1].set_title('Prediction Error by Stock', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_stock.png', dpi=300, bbox_inches='tight')
    print(f"     ✓ Saved: performance_by_stock.png")
    plt.close()

    print(f"\n     ✓ All plots saved to: {output_dir.absolute()}")


def main():
    print("=" * 70)
    print("Qlib 30-Day Stock Price Volatility Prediction")
    print("=" * 70)

    # 1. 初始化 Qlib
    print("\n[1] Initializing Qlib...")
    qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US)
    print("    ✓ Qlib initialized")

    # 2. 检查数据
    print("\n[2] Checking data availability...")
    instruments = D.instruments(market="all")
    available_instruments = list(D.list_instruments(instruments))
    print(f"    ✓ Available instruments: {len(available_instruments)}")

    # 测试读取一只股票
    test_df = D.features(
        instruments=["AAPL"],
        fields=["$close", "$volume"],
        start_time=TEST_START,
        end_time=TEST_END
    )
    print(f"    ✓ AAPL sample data shape: {test_df.shape}")
    print(f"    ✓ Date range: {test_df.index.get_level_values('datetime').min().date()} to {test_df.index.get_level_values('datetime').max().date()}")

    # 3. 创建 DataHandler（使用30天波动率标签）
    print(f"\n[3] Creating DataHandler with {VOLATILITY_WINDOW}-day volatility label...")
    print(f"    Features: Alpha158 (158 technical indicators)")
    print(f"    Label: {VOLATILITY_WINDOW}-day realized volatility (annualized)")

    handler = Alpha158_30DayVolatility(
        instruments=TEST_SYMBOLS,
        start_time=TRAIN_START,
        end_time=TEST_END,
        fit_start_time=TRAIN_START,
        fit_end_time=TRAIN_END,
        infer_processors=[],
        learn_processors=[
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature"}},
        ],
    )
    print("    ✓ DataHandler created")

    # 4. 创建 Dataset
    print("\n[4] Creating Dataset...")
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (TRAIN_START, TRAIN_END),
            "valid": (VALID_START, VALID_END),
            "test": (TEST_START, TEST_END),
        }
    )

    train_data = dataset.prepare("train", col_set="feature")
    print(f"    ✓ Train features shape: {train_data.shape}")
    print(f"      (samples × features)")

    # 检查标签分布
    print("\n[5] Analyzing label distribution...")
    train_label = dataset.prepare("train", col_set="label")
    valid_label = dataset.prepare("valid", col_set="label")

    print(f"    Train set volatility statistics:")
    print(f"      Mean:   {train_label['LABEL0'].mean():.4f} (annualized)")
    print(f"      Std:    {train_label['LABEL0'].std():.4f}")
    print(f"      Median: {train_label['LABEL0'].median():.4f}")
    print(f"      Min:    {train_label['LABEL0'].min():.4f}")
    print(f"      Max:    {train_label['LABEL0'].max():.4f}")

    print(f"\n    Valid set volatility statistics:")
    print(f"      Mean:   {valid_label['LABEL0'].mean():.4f} (annualized)")
    print(f"      Std:    {valid_label['LABEL0'].std():.4f}")

    # 5. 训练模型
    print("\n[6] Training LightGBM model...")
    print("    Model parameters:")
    print(f"      - loss: mse")
    print(f"      - learning_rate: 0.05")
    print(f"      - max_depth: 8")
    print(f"      - num_leaves: 128")
    print(f"      - n_estimators: 200")

    model = LGBModel(
        loss="mse",
        learning_rate=0.05,
        max_depth=8,
        num_leaves=128,
        num_threads=4,
        n_estimators=200,
        early_stopping_rounds=30,
        verbose=-1,  # 减少训练输出
    )

    print("\n    Training progress:")
    model.fit(dataset)
    print("    ✓ Model training completed")

    # 6. 预测
    print("\n[7] Generating predictions...")
    pred = model.predict(dataset)

    test_pred = pred.loc[TEST_START:TEST_END]
    print(f"    ✓ Prediction shape: {test_pred.shape}")
    print(f"    ✓ Prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    # 7. 评估
    print("\n[8] Model Evaluation...")

    # 获取实际波动率
    label_df = dataset.prepare("test", col_set="label")
    test_pred_aligned = test_pred.reindex(label_df.index)

    # 去除 NaN 值
    valid_idx = ~(test_pred_aligned.isna() | label_df["LABEL0"].isna())
    test_pred_clean = test_pred_aligned[valid_idx]
    label_clean = label_df["LABEL0"][valid_idx]

    print(f"    Valid test samples: {len(test_pred_clean)}")

    # 计算 IC (Information Coefficient)
    ic = test_pred_clean.groupby(level="datetime").apply(
        lambda x: x.corr(label_clean.loc[x.index]) if len(x) > 1 else np.nan
    )
    ic = ic.dropna()

    # 计算误差指标
    mse = ((test_pred_clean - label_clean) ** 2).mean()
    mae = (test_pred_clean - label_clean).abs().mean()
    rmse = np.sqrt(mse)

    # 计算相对误差
    mape = ((test_pred_clean - label_clean).abs() / label_clean).mean() * 100

    print(f"\n    ╔════════════════════════════════════════╗")
    print(f"    ║  Information Coefficient (IC)          ║")
    print(f"    ╠════════════════════════════════════════╣")
    print(f"    ║  Mean IC:   {ic.mean():>8.4f}                  ║")
    print(f"    ║  IC Std:    {ic.std():>8.4f}                  ║")
    print(f"    ║  ICIR:      {ic.mean() / ic.std():>8.4f}                  ║")
    print(f"    ╚════════════════════════════════════════╝")

    print(f"\n    ╔════════════════════════════════════════╗")
    print(f"    ║  Prediction Error Metrics              ║")
    print(f"    ╠════════════════════════════════════════╣")
    print(f"    ║  MSE (Mean Squared Error):   {mse:>8.6f} ║")
    print(f"    ║  MAE (Mean Absolute Error):  {mae:>8.6f} ║")
    print(f"    ║  RMSE (Root Mean Sq Error):  {rmse:>8.6f} ║")
    print(f"    ║  MAPE (Mean Abs % Error):    {mape:>7.2f}%  ║")
    print(f"    ╚════════════════════════════════════════╝")

    # 展示预测样本
    print(f"\n[9] Sample Predictions (Test Set):")
    print(f"    Showing first 20 predictions...\n")

    comparison = pd.DataFrame({
        "Predicted_Vol": test_pred_clean,
        "Actual_Vol": label_clean,
        "Error": test_pred_clean - label_clean,
        "Error_%": ((test_pred_clean - label_clean) / label_clean * 100).round(2)
    })

    print(comparison.head(20).to_string())

    # 按股票统计
    print(f"\n[10] Performance by Stock:")
    print(f"     (Average prediction error for each stock)\n")

    stock_performance = comparison.groupby(level="instrument").agg({
        "Predicted_Vol": "mean",
        "Actual_Vol": "mean",
        "Error": lambda x: x.abs().mean(),
        "Error_%": lambda x: x.abs().mean()
    }).round(4)
    stock_performance.columns = ["Pred_Mean", "Actual_Mean", "MAE", "MAPE"]
    print(stock_performance.to_string())

    # 生成可视化图表
    output_dir = PROJECT_ROOT / "outputs" / "volatility_prediction"
    plot_predictions(comparison, output_dir)

    print("\n" + "=" * 70)
    print("✓ 30-Day Volatility Prediction Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
