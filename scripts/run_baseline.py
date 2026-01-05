"""
运行 LightGBM baseline，验证环境正常
"""
from pathlib import Path
from utils import evaluate_model

import qlib
from qlib.constant import REG_US
from qlib.data import D
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


# ========== 配置 ==========

# 数据路径
PROJECT_ROOT = Path(__file__).parent.parent
QLIB_DATA_PATH = PROJECT_ROOT / "my_data" / "qlib_us"

# 股票池（先用少量测试）
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]

# 时间划分
TRAIN_START = "2000-01-01"
TRAIN_END = "2023-12-31"
VALID_START = "2024-01-01"
VALID_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"


def main():
    print("=" * 60)
    print("Qlib Baseline Test")
    print("=" * 60)
    
    # 1. 初始化 Qlib
    print("\n[1] Initializing Qlib...")
    qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US)
    print("    OK")
    
    # 2. 检查数据
    print("\n[2] Checking data...")
    instruments = D.instruments(market="all")
    print(f"    Available instruments: {len(list(D.list_instruments(instruments)))}")
    
    # 测试读取一只股票
    test_df = D.features(
        instruments=["AAPL"],
        fields=["$close", "$volume"],
        start_time=TEST_START,
        end_time=TEST_END
    )
    print(f"    AAPL sample data shape: {test_df.shape}")
    print(f"    Date range: {test_df.index.get_level_values('datetime').min()} to {test_df.index.get_level_values('datetime').max()}")
    
    # 3. 创建 DataHandler
    print("\n[3] Creating Alpha158 DataHandler...")
    handler = Alpha158(
        instruments=TEST_SYMBOLS,
        start_time=TRAIN_START,
        end_time=TEST_END,
        fit_start_time=TRAIN_START,
        fit_end_time=TRAIN_END,
        infer_processors=[],  # 简化处理
    )
    print("    OK")
    
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
    print(f"    Train features shape: {train_data.shape}")
    print(f"    Feature columns: {list(train_data.columns[:10])}...")  # 前10个特征
    
    # 5. 训练模型
    print("\n[5] Training LightGBM model...")
    model = LGBModel(
        loss="mse",
        learning_rate=0.05,
        max_depth=8,
        num_leaves=128,
        num_threads=4,
        n_estimators=200,
        early_stopping_rounds=30,
    )
    model.fit(dataset)
    print("    OK")
    
    # 6. 预测
    print("\n[6] Generating predictions...")
    pred = model.predict(dataset)
    
    print(f"    Prediction shape: {pred.shape}")
    print(f"\n    Sample predictions (test set):")
    
    # 展示测试集的预测
    test_pred = pred.loc[TEST_START:TEST_END]
    print(test_pred.head(20).to_string())
    
    # 7. 简单评估
    print("\n[7] Simple evaluation...")
    evaluate_model(dataset, test_pred, PROJECT_ROOT, 2)


if __name__ == "__main__":
    main()
