from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import argparse

# 设置 matplotlib 中文显示和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

def plot_predictions(comparison_df, output_dir, nday):
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

        axes[idx].set_title(f'{stock} - {nday}-Day Volatility: Predicted vs Actual',
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

    # # ========== 图3: 误差分布图 ==========
    # fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # # 3.1 误差直方图
    # axes[0, 0].hist(comparison_df['Error'], bins=50, alpha=0.7,
    #                color='#F18F01', edgecolor='black')
    # axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    # axes[0, 0].set_xlabel('Prediction Error', fontsize=11, fontweight='bold')
    # axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    # axes[0, 0].set_title('Error Distribution (Histogram)', fontsize=12, fontweight='bold')
    # axes[0, 0].legend()
    # axes[0, 0].grid(True, alpha=0.3)

    # # 3.2 误差百分比直方图
    # axes[0, 1].hist(comparison_df['Error_%'], bins=50, alpha=0.7,
    #                color='#06A77D', edgecolor='black')
    # axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    # axes[0, 1].set_xlabel('Prediction Error (%)', fontsize=11, fontweight='bold')
    # axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    # axes[0, 1].set_title('Error % Distribution (Histogram)', fontsize=12, fontweight='bold')
    # axes[0, 1].legend()
    # axes[0, 1].grid(True, alpha=0.3)

    # # 3.3 误差箱线图（按股票分组）
    # error_by_stock = [comparison_df.loc[(slice(None), stock), 'Error'].values
    #                  for stock in stocks]
    # bp = axes[1, 0].boxplot(error_by_stock, labels=stocks, patch_artist=True)
    # for patch, color in zip(bp['boxes'], colors):
    #     patch.set_facecolor(color)
    #     patch.set_alpha(0.7)
    # axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    # axes[1, 0].set_xlabel('Stock', fontsize=11, fontweight='bold')
    # axes[1, 0].set_ylabel('Prediction Error', fontsize=11, fontweight='bold')
    # axes[1, 0].set_title('Error Distribution by Stock (Boxplot)', fontsize=12, fontweight='bold')
    # axes[1, 0].grid(True, alpha=0.3, axis='y')
    # axes[1, 0].tick_params(axis='x', rotation=45)

    # # 3.4 时间序列误差图
    # for stock in stocks:
    #     stock_data = comparison_df.loc[(slice(None), stock), :]
    #     dates = stock_data.index.get_level_values('datetime')
    #     axes[1, 1].plot(dates, stock_data['Error'], marker='o',
    #                    markersize=3, linewidth=1, alpha=0.7, label=stock)

    # axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    # axes[1, 1].set_xlabel('Date', fontsize=11, fontweight='bold')
    # axes[1, 1].set_ylabel('Prediction Error', fontsize=11, fontweight='bold')
    # axes[1, 1].set_title('Error Over Time', fontsize=12, fontweight='bold')
    # axes[1, 1].legend(loc='best', fontsize=8)
    # axes[1, 1].grid(True, alpha=0.3)
    # axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # axes[1, 1].tick_params(axis='x', rotation=45)

    # plt.tight_layout()
    # plt.savefig(output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
    # print(f"     ✓ Saved: error_analysis.png")
    # plt.close()

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


def evaluate_model(dataset, test_pred, project_root, nday):
    """
    评估模型性能并生成可视化图表

    Args:
        dataset: Qlib DatasetH 对象
        test_pred: 测试集预测结果
        project_root: 项目根目录路径
    """
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
    output_dir = project_root / "outputs" / "volatility_prediction"
    plot_predictions(comparison, output_dir, nday)

    print("\n" + "=" * 70)
    print("✓ Volatility Prediction Completed Successfully!")
    print("=" * 70)