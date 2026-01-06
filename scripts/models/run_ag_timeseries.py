"""
使用 AutoGluon TimeSeriesPredictor 预测股票价格波动率

数据集：Alpha158_Volatility_TALib
预测目标：LABEL0（N天价格波动率）
预测步数：24步
"""

import sys
from pathlib import Path

# IMPORTANT: Fix autogluon import path conflict.
# The autogluon submodule at ./autogluon/ conflicts with the editable install.
# The nspkg.pth files pre-import autogluon during Python startup with wrong paths
# because the submodule directory structure shadows the proper src/ paths.
#
# Solution: Add the correct autogluon src/ directories to sys.path before importing,
# clear any pre-imported autogluon modules, and remove conflicting paths.

# Get project root (where autogluon submodule lives)
project_root = Path(__file__).parent.parent.parent

# Remove cwd and project root from path to prevent finding submodule
if '' in sys.path:
    sys.path.remove('')
project_root_str = str(project_root)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)

# Clear any pre-imported autogluon modules
modules_to_remove = [k for k in list(sys.modules.keys()) if k.startswith('autogluon')]
for m in modules_to_remove:
    del sys.modules[m]

# Add the correct autogluon src/ directories to sys.path
# These contain the actual autogluon namespace package components
autogluon_submodule = project_root / "autogluon"
if autogluon_submodule.exists():
    autogluon_components = ['autogluon', 'common', 'core', 'eda', 'features', 'tabular', 'timeseries']
    for component in reversed(autogluon_components):
        src_path = autogluon_submodule / component / "src"
        if src_path.exists():
            src_path_str = str(src_path)
            if src_path_str not in sys.path:
                sys.path.insert(0, src_path_str)

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

# Import evaluation utilities
from utils.utils import evaluate_model


# ========== 配置 ==========

# 数据路径
PROJECT_ROOT = Path(__file__).parent.parent.parent  # 项目根目录
QLIB_DATA_PATH = PROJECT_ROOT / "my_data" / "qlib_us"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "autogluon_ts"

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


def prepare_timeseries_data(dataset, segment):
    """
    将 Qlib 数据集转换为 AutoGluon TimeSeriesDataFrame 格式

    Args:
        dataset: Qlib DatasetH 对象
        segment: 数据段名称 ("train", "valid", "test")

    Returns:
        TimeSeriesDataFrame: AutoGluon 时序数据格式
    """
    # 获取特征和标签
    features = dataset.prepare(segment, col_set="feature")
    labels = dataset.prepare(segment, col_set="label")

    # 合并特征和标签
    data = pd.concat([features, labels], axis=1)

    # 重置索引，将 (instrument, datetime) 转为列
    data = data.reset_index()

    # 重命名列以符合 TimeSeriesDataFrame 格式
    data = data.rename(columns={
        "instrument": "item_id",
        "datetime": "timestamp",
        "LABEL0": "target"
    })

    # 确保时间戳为 datetime 类型
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # 按 item_id 和 timestamp 排序
    data = data.sort_values(["item_id", "timestamp"])

    # 创建 TimeSeriesDataFrame
    ts_data = TimeSeriesDataFrame.from_data_frame(
        data,
        id_column="item_id",
        timestamp_column="timestamp"
    )

    return ts_data


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


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='AutoGluon TimeSeriesPredictor for Stock Volatility Prediction')
    parser.add_argument('--nday', type=int, default=2, help='Volatility prediction window in days (default: 2)')
    parser.add_argument('--prediction-length', type=int, default=24, help='Number of steps to predict (default: 24)')
    parser.add_argument('--presets', type=str, default='medium_quality',
                        choices=['fast_training', 'medium_quality', 'high_quality', 'best_quality'],
                        help='AutoGluon presets (default: medium_quality)')
    parser.add_argument('--time-limit', type=int, default=600, help='Time limit for training in seconds (default: 600)')
    parser.add_argument('--eval-metric', type=str, default='MASE',
                        choices=['MASE', 'MAPE', 'sMAPE', 'RMSE', 'MAE', 'WAPE'],
                        help='Evaluation metric (default: MASE)')
    args = parser.parse_args()

    # 更新全局变量
    global VOLATILITY_WINDOW, PREDICTION_LENGTH
    VOLATILITY_WINDOW = args.nday
    PREDICTION_LENGTH = args.prediction_length

    print("=" * 70)
    print(f"AutoGluon TimeSeriesPredictor - {VOLATILITY_WINDOW}-Day Volatility Prediction")
    print(f"Prediction Length: {PREDICTION_LENGTH} steps")
    print(f"Presets: {args.presets}")
    print(f"Time Limit: {args.time_limit} seconds")
    print(f"Eval Metric: {args.eval_metric}")
    print("=" * 70)

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
    print(f"\n    Data columns: {list(train_ts.columns)[:10]}...")
    print(f"    Target column: target")

    # 5. 获取协变量列名（排除 target）
    feature_cols = [col for col in train_ts.columns if col != "target"]
    print(f"    ✓ Number of covariates: {len(feature_cols)}")

    # 6. 创建并训练 TimeSeriesPredictor
    print(f"\n[5] Training TimeSeriesPredictor...")
    print(f"    Prediction length: {PREDICTION_LENGTH}")
    print(f"    Presets: {args.presets}")
    print(f"    Time limit: {args.time_limit} seconds")

    # 创建输出目录
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_PATH / f"ts_predictor_nday{VOLATILITY_WINDOW}_pred{PREDICTION_LENGTH}"

    predictor = TimeSeriesPredictor(
        prediction_length=PREDICTION_LENGTH,
        target="target",
        eval_metric=args.eval_metric,
        path=str(model_path),
        freq="B",  # Business day frequency for stock data
    )

    # 训练模型
    predictor.fit(
        train_data=train_ts,
        presets=args.presets,
        time_limit=args.time_limit,
    )
    print("    ✓ Training completed")

    # 7. 显示模型排行榜
    print("\n[6] Model Leaderboard:")
    leaderboard = predictor.leaderboard()
    print(leaderboard.to_string())

    # 8. 生成预测
    print(f"\n[7] Generating {PREDICTION_LENGTH}-step predictions...")

    # 使用 train+valid 数据进行预测（预测未来 PREDICTION_LENGTH 步）
    predictions = predictor.predict(train_ts)
    print(f"    ✓ Predictions shape: {predictions.shape}")
    print(f"    ✓ Prediction columns: {list(predictions.columns)}")

    # 9. 转换预测结果为 evaluate_model 期望的格式
    print("\n[8] Preparing predictions for evaluation...")

    # 将 AutoGluon 预测结果转换为 (datetime, instrument) 索引的 Series
    pred_df = predictions.reset_index()
    pred_df = pred_df.rename(columns={
        "item_id": "instrument",
        "timestamp": "datetime",
        "mean": "prediction"
    })

    # 过滤出测试集日期范围内的预测
    pred_df = pred_df[pred_df["datetime"] >= test_start_date]

    # 设置 MultiIndex 并转为 Series
    pred_df = pred_df.set_index(["datetime", "instrument"])
    test_pred = pred_df["prediction"]

    print(f"    ✓ Converted predictions: {len(test_pred)} samples")

    # 10. 使用 evaluate_model 进行评估和可视化
    evaluate_model(dataset, test_pred, PROJECT_ROOT, VOLATILITY_WINDOW)

    # 11. 保存预测结果
    print("\n[11] Saving results...")
    predictions_path = OUTPUT_PATH / f"predictions_nday{VOLATILITY_WINDOW}_pred{PREDICTION_LENGTH}.parquet"
    predictions.reset_index().to_parquet(predictions_path)
    print(f"    ✓ Predictions saved to: {predictions_path}")

    print("\n" + "=" * 70)
    print("AutoGluon TimeSeriesPredictor training and prediction completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
