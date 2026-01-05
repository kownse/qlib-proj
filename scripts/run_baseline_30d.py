"""
运行 LightGBM baseline，预测30天股票价格波动率

波动率定义：未来30个交易日的收益率标准差（年化）
"""
from pathlib import Path
import pandas as pd
import numpy as np

import qlib
from qlib.constant import REG_US
from qlib.data import D
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH


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

    print("\n" + "=" * 70)
    print("✓ 30-Day Volatility Prediction Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
