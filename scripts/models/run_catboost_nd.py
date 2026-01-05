"""
运行 CatBoost baseline，预测N天股票价格波动率

波动率定义：未来N个交易日波动变化

扩展特征：包含 Alpha158 默认指标 + TA-Lib 技术指标
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
script_dir = Path(__file__).parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

from pathlib import Path
import argparse
from utils.utils import evaluate_model

import qlib
from qlib.constant import REG_US
from qlib.data import D
from qlib.contrib.model.catboost_model import CatBoostModel
from qlib.data.dataset import DatasetH

# Import TA-Lib custom operators
from utils.talib_ops import TALIB_OPS

# Import extended data handlers
from data.datahandler_ext import Alpha158_Volatility, Alpha158_Volatility_TALib


# ========== 配置 ==========

# 数据路径
PROJECT_ROOT = Path(__file__).parent.parent.parent  # 项目根目录
QLIB_DATA_PATH = PROJECT_ROOT / "my_data" / "qlib_us"

# 股票池
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]

# 时间划分
TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
VALID_START = "2024-01-01"
VALID_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

# 波动率预测窗口（天数）
VOLATILITY_WINDOW = 2

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Qlib Stock Price Volatility Prediction (CatBoost)')
    parser.add_argument('--nday', type=int, default=2, help='Volatility prediction window in days (default: 2)')
    parser.add_argument('--use-talib', action='store_true', help='Use extended TA-Lib features (default: False)')
    args = parser.parse_args()

    # 更新全局变量
    global VOLATILITY_WINDOW
    VOLATILITY_WINDOW = args.nday

    print("=" * 70)
    print(f"CatBoost {VOLATILITY_WINDOW}-Day Stock Price Volatility Prediction")
    if args.use_talib:
        print("Features: Alpha158 + TA-Lib Technical Indicators")
    else:
        print("Features: Alpha158 (default)")
    print("=" * 70)

    # 1. 初始化 Qlib (包含 TA-Lib 自定义算子)
    print("\n[1] Initializing Qlib...")
    if args.use_talib:
        qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US, custom_ops=TALIB_OPS)
        print("    ✓ Qlib initialized with TA-Lib custom operators")
    else:
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
    if args.use_talib:
        print(f"    Features: Alpha158 + TA-Lib (~300+ technical indicators)")
    else:
        print(f"    Features: Alpha158 (158 technical indicators)")
    print(f"    Label: {VOLATILITY_WINDOW}-day realized volatility")

    if args.use_talib:
        handler = Alpha158_Volatility_TALib(
            volatility_window=VOLATILITY_WINDOW,
            instruments=TEST_SYMBOLS,
            start_time=TRAIN_START,
            end_time=TEST_END,
            fit_start_time=TRAIN_START,
            fit_end_time=TRAIN_END,
            infer_processors=[],
        )
    else:
        handler = Alpha158_Volatility(
            volatility_window=VOLATILITY_WINDOW,
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
    print("\n[6] Training CatBoost model...")
    print("    Model parameters:")
    print(f"      - loss: RMSE")
    print(f"      - learning_rate: 0.05")
    print(f"      - max_depth: 6")
    print(f"      - l2_leaf_reg: 3")
    print(f"      - random_strength: 1")

    model = CatBoostModel(
        loss="RMSE",
        learning_rate=0.05,
        max_depth=6,
        l2_leaf_reg=3,
        random_strength=1,
        thread_count=4,
    )

    print("\n    Training progress:")
    model.fit(
        dataset,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=100,
    )
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
