"""
使用 AutoGluon Temporal Fusion Transformer (TFT) 预测股票价格波动率

数据集：Alpha158_Volatility_TALib
预测目标：LABEL0（N天价格波动率）
模型：Temporal Fusion Transformer
"""

import os
import sys
from pathlib import Path

# 修复 macOS 上 PyTorch 多线程锁问题
# 必须在导入 torch 之前设置
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 设置 multiprocessing start method (macOS)
import multiprocessing
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # 已经设置过了

# Add scripts directory to path for imports
script_dir = Path(__file__).parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

import argparse
import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH

# AutoGluon imports
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# Import TA-Lib custom operators
from utils.talib_ops import TALIB_OPS

# Import extended data handlers
from data.datahandler_ext import Alpha158_Volatility_TALib


# ========== 配置 ==========

# 数据路径
PROJECT_ROOT = Path(__file__).parent.parent.parent  # 项目根目录
QLIB_DATA_PATH = PROJECT_ROOT / "my_data" / "qlib_us"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "autogluon_tft"

# 股票池
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]

# 时间划分
TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
VALID_START = "2024-01-01"
VALID_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

# 预测步数
PREDICTION_LENGTH = 24

# 波动率预测窗口（天数）
VOLATILITY_WINDOW = 2


def prepare_all_data(dataset):
    """
    准备完整的时序数据（包含 train + valid + test）
    用于 TimeSeriesPredictor 的训练和预测

    Args:
        dataset: Qlib DatasetH 对象

    Returns:
        tuple: (train_ts, full_ts, test_start_date)
    """
    # 获取所有数据段
    train_features = dataset.prepare("train", col_set="feature")
    train_labels = dataset.prepare("train", col_set="label")
    valid_features = dataset.prepare("valid", col_set="feature")
    valid_labels = dataset.prepare("valid", col_set="label")
    test_features = dataset.prepare("test", col_set="feature")
    test_labels = dataset.prepare("test", col_set="label")

    # 合并所有数据
    all_features = pd.concat([train_features, valid_features, test_features])
    all_labels = pd.concat([train_labels, valid_labels, test_labels])

    # 只使用 train + valid 作为训练数据
    train_valid_features = pd.concat([train_features, valid_features])
    train_valid_labels = pd.concat([train_labels, valid_labels])

    # 转换为 DataFrame 格式
    def to_ts_format(features, labels):
        data = pd.concat([features, labels], axis=1)
        data = data.reset_index()
        data = data.rename(columns={
            "instrument": "item_id",
            "datetime": "timestamp",
            "LABEL0": "target"
        })
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values(["item_id", "timestamp"])

        # 创建 TimeSeriesDataFrame
        ts_data = TimeSeriesDataFrame.from_data_frame(
            data,
            id_column="item_id",
            timestamp_column="timestamp"
        )
        return ts_data

    train_ts = to_ts_format(train_valid_features, train_valid_labels)
    full_ts = to_ts_format(all_features, all_labels)

    # 获取 test 开始日期
    test_start_date = test_labels.index.get_level_values("datetime").min()

    return train_ts, full_ts, test_start_date


def evaluate_predictions(predictions, actual_data, test_start_date):
    """
    评估预测结果

    Args:
        predictions: TimeSeriesPredictor 的预测结果
        actual_data: 完整的时序数据
        test_start_date: 测试集开始日期

    Returns:
        dict: 评估指标
    """
    # 获取预测的 mean 值
    pred_df = predictions.reset_index()

    # 获取实际值
    actual_df = actual_data.reset_index()
    actual_df = actual_df[actual_df["timestamp"] >= test_start_date]

    # 合并预测和实际值
    merged = pd.merge(
        pred_df[["item_id", "timestamp", "mean"]],
        actual_df[["item_id", "timestamp", "target"]],
        on=["item_id", "timestamp"],
        how="inner"
    )

    if len(merged) == 0:
        print("Warning: No overlapping data between predictions and actual values")
        return {}

    # 计算评估指标
    y_true = merged["target"].values
    y_pred = merged["mean"].values

    # MSE
    mse = np.mean((y_true - y_pred) ** 2)

    # RMSE
    rmse = np.sqrt(mse)

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # MAPE (避免除以零)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan

    # IC (Information Coefficient - 皮尔逊相关系数)
    ic = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "IC": ic,
        "N_samples": len(merged)
    }

    return metrics


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='AutoGluon TFT for Stock Volatility Prediction')
    parser.add_argument('--nday', type=int, default=2, help='Volatility prediction window in days (default: 2)')
    parser.add_argument('--prediction-length', type=int, default=24, help='Number of steps to predict (default: 24)')
    parser.add_argument('--time-limit', type=int, default=1800, help='Time limit for training in seconds (default: 1800)')
    parser.add_argument('--eval-metric', type=str, default='MASE',
                        choices=['MASE', 'MAPE', 'sMAPE', 'RMSE', 'MAE', 'WAPE'],
                        help='Evaluation metric (default: MASE)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--hidden-size', type=int, default=64, help='Hidden size for TFT (default: 64)')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (default: 0.1)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--num-gpus', type=int, default=0, help='Number of GPUs to use (default: 0)')
    args = parser.parse_args()

    # 更新全局变量
    global VOLATILITY_WINDOW, PREDICTION_LENGTH
    VOLATILITY_WINDOW = args.nday
    PREDICTION_LENGTH = args.prediction_length

    print("=" * 70)
    print(f"AutoGluon Temporal Fusion Transformer (TFT)")
    print(f"Task: {VOLATILITY_WINDOW}-Day Volatility Prediction")
    print(f"Prediction Length: {PREDICTION_LENGTH} steps")
    print("=" * 70)
    print(f"\nTFT Hyperparameters:")
    print(f"    Hidden size: {args.hidden_size}")
    print(f"    Attention heads: {args.num_heads}")
    print(f"    Dropout: {args.dropout}")
    print(f"    Learning rate: {args.lr}")
    print(f"    Batch size: {args.batch_size}")
    print(f"    Epochs: {args.epochs}")
    print(f"    Time limit: {args.time_limit} seconds")
    print(f"    GPUs: {args.num_gpus}")

    # 1. 初始化 Qlib
    print("\n[1] Initializing Qlib...")
    qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US, custom_ops=TALIB_OPS)
    print("    ✓ Qlib initialized with TA-Lib custom operators")

    # 2. 创建 DataHandler
    print(f"\n[2] Creating DataHandler with {VOLATILITY_WINDOW}-day volatility label...")
    print(f"    Features: Alpha158 + TA-Lib (~300+ technical indicators)")
    print(f"    Label: {VOLATILITY_WINDOW}-day realized volatility")

    handler = Alpha158_Volatility_TALib(
        volatility_window=VOLATILITY_WINDOW,
        instruments=TEST_SYMBOLS,
        start_time=TRAIN_START,
        end_time=TEST_END,
        fit_start_time=TRAIN_START,
        fit_end_time=TRAIN_END,
        infer_processors=[],
    )
    print("    ✓ DataHandler created")

    # 3. 创建 Dataset
    print("\n[3] Creating Dataset...")
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

    # 4. 准备 TimeSeriesDataFrame
    print("\n[4] Preparing TimeSeriesDataFrame...")
    train_ts, full_ts, test_start_date = prepare_all_data(dataset)
    print(f"    ✓ Train+Valid TimeSeriesDataFrame: {len(train_ts)} rows")
    print(f"    ✓ Full TimeSeriesDataFrame: {len(full_ts)} rows")
    print(f"    ✓ Test start date: {test_start_date}")
    print(f"    ✓ Number of time series (stocks): {train_ts.num_items}")

    # 显示数据信息
    feature_cols = [col for col in train_ts.columns if col != "target"]
    print(f"    ✓ Number of covariates: {len(feature_cols)}")

    # 5. 配置 TFT 超参数
    print(f"\n[5] Configuring Temporal Fusion Transformer...")

    # TFT 超参数配置
    # 参考: autogluon/timeseries/models/gluonts/models.py
    tft_config = {
        "hidden_dim": args.hidden_size,      # LSTM & transformer hidden states size
        "num_heads": args.num_heads,         # Number of attention heads
        "dropout_rate": args.dropout,        # Dropout regularization
        "lr": args.lr,                       # Learning rate (注意不是 learning_rate)
        "batch_size": args.batch_size,       # Training batch size
        "max_epochs": args.epochs,           # Number of training epochs
        "context_length": 64,                # Number of past values for prediction
        "num_batches_per_epoch": 50,         # Batches per epoch
        "early_stopping_patience": 20,       # Early stopping patience
        # macOS 兼容性设置
        "trainer_kwargs": {
            "accelerator": "cpu",
            "devices": 1,
        },
    }

    # 添加 GPU 配置 (如果指定)
    if args.num_gpus > 0:
        tft_config["trainer_kwargs"] = {"accelerator": "gpu", "devices": args.num_gpus}

    tft_hyperparameters = {
        "TemporalFusionTransformer": tft_config
    }

    print(f"    TFT config: {tft_hyperparameters['TemporalFusionTransformer']}")

    # 6. 创建输出目录
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_PATH / f"tft_nday{VOLATILITY_WINDOW}_pred{PREDICTION_LENGTH}"

    # 7. 创建并训练 TimeSeriesPredictor (仅使用 TFT)
    print(f"\n[6] Training Temporal Fusion Transformer...")

    predictor = TimeSeriesPredictor(
        prediction_length=PREDICTION_LENGTH,
        target="target",
        eval_metric=args.eval_metric,
        path=str(model_path),
        freq="B",  # Business day frequency for stock data
    )

    # 训练模型 - 只使用 TFT
    predictor.fit(
        train_data=train_ts,
        hyperparameters=tft_hyperparameters,
        time_limit=args.time_limit,
    )
    print("    ✓ TFT Training completed")

    # 8. 显示模型排行榜
    print("\n[7] Model Leaderboard:")
    leaderboard = predictor.leaderboard()
    print(leaderboard.to_string())

    # 9. 生成预测
    print(f"\n[8] Generating {PREDICTION_LENGTH}-step predictions...")

    predictions = predictor.predict(train_ts)
    print(f"    ✓ Predictions shape: {predictions.shape}")
    print(f"    ✓ Prediction columns: {list(predictions.columns)}")

    # 10. 评估预测结果
    print("\n[9] Evaluating predictions...")
    metrics = evaluate_predictions(predictions, full_ts, test_start_date)

    if metrics:
        print("\n    Evaluation Metrics:")
        print("    " + "-" * 40)
        for metric_name, value in metrics.items():
            if metric_name == "N_samples":
                print(f"    {metric_name:<15s}: {value}")
            else:
                print(f"    {metric_name:<15s}: {value:.6f}")
        print("    " + "-" * 40)

    # 11. 保存预测结果
    print("\n[10] Saving results...")
    predictions_path = OUTPUT_PATH / f"tft_predictions_nday{VOLATILITY_WINDOW}_pred{PREDICTION_LENGTH}.parquet"
    predictions.reset_index().to_parquet(predictions_path)
    print(f"    ✓ Predictions saved to: {predictions_path}")

    # 显示预测样本
    print("\n[11] Sample predictions:")
    print(predictions.head(20).to_string())

    print("\n" + "=" * 70)
    print("Temporal Fusion Transformer training and prediction completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
