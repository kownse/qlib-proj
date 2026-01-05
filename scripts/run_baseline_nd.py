"""
运行 LightGBM baseline，预测N天股票价格波动率

波动率定义：未来N个交易日波动变化"""

from pathlib import Path
import argparse
from utils import evaluate_model

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
VOLATILITY_WINDOW = 2


class Alpha158_Volatility(Alpha158):
    """
    Alpha158 特征 + N天价格波动率标签

    继承 Alpha158 的所有技术指标特征，只修改标签为N天波动率
    """

    def get_label_config(self):
        """
        返回N天波动率标签

        Returns:
            fields: 标签表达式列表
            names: 标签名称列表
        """
        # 使用 Qlib 的表达式语法
        # Std($close, N) 计算 N 天的标准差
        # Ref($field, -N) 向未来移动 N 天
        volatility_expr = f"Ref($close, -{VOLATILITY_WINDOW})/Ref($close, -1) - 1"

        return [volatility_expr], ["LABEL0"]

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Qlib Stock Price Volatility Prediction')
    parser.add_argument('--nday', type=int, default=2, help='Volatility prediction window in days (default: 2)')
    args = parser.parse_args()
    
    # 更新全局变量
    global VOLATILITY_WINDOW
    VOLATILITY_WINDOW = args.nday
    
    print("=" * 70)
    print(f"Qlib {VOLATILITY_WINDOW}-Day Stock Price Volatility Prediction")
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

    # 3. 创建 DataHandler
    print(f"\n[3] Creating DataHandler with {VOLATILITY_WINDOW}-day volatility label...")
    print(f"    Features: Alpha158 (158 technical indicators)")
    print(f"    Label: {VOLATILITY_WINDOW}-day realized volatility")

    handler = Alpha158_Volatility(
        instruments=TEST_SYMBOLS,
        start_time=TRAIN_START,
        end_time=TEST_END,
        fit_start_time=TRAIN_START,
        fit_end_time=TRAIN_END,
        infer_processors=[],
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
    evaluate_model(dataset, test_pred, PROJECT_ROOT, VOLATILITY_WINDOW)


if __name__ == "__main__":
    main()
