"""
从已保存的模型直接运行 backtest

Usage:
    python scripts/models/deep/run_backtest_from_model.py \
        --model-path my_models/ae_mlp_cv_alpha158-enhanced-v6_sp500_5d.keras \
        --handler alpha158-enhanced-v6 \
        --stock-pool sp500
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

# Setup paths
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))
project_root = script_dir.parent

# Initialize qlib
import qlib
from qlib.constant import REG_US
from utils.talib_ops import TALIB_OPS

qlib_data_path = project_root / "my_data" / "qlib_us"
qlib.init(
    provider_uri=str(qlib_data_path),
    region=REG_US,
    custom_ops=TALIB_OPS,
)

import tensorflow as tf
from tensorflow import keras
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from models.common.handlers import get_handler_class, HANDLER_CONFIG
from models.common import run_backtest, PROJECT_ROOT
from data.stock_pools import STOCK_POOLS
from utils.utils import evaluate_model


# 默认时间配置
DEFAULT_TEST_CONFIG = {
    'train_start': '2000-01-01',
    'train_end': '2024-09-30',
    'valid_start': '2024-10-01',
    'valid_end': '2024-12-31',
    'test_start': '2025-01-01',
    'test_end': '2025-12-31',
}


def create_data_handler(args, symbols, time_config):
    """创建 DataHandler"""
    HandlerClass = get_handler_class(args.handler)

    handler = HandlerClass(
        volatility_window=args.nday,
        instruments=symbols,
        start_time=time_config['train_start'],
        end_time=time_config['test_end'],
        fit_start_time=time_config['train_start'],
        fit_end_time=time_config['train_end'],
        infer_processors=[],
    )
    return handler


def create_dataset(handler, time_config):
    """创建 Dataset"""
    segments = {
        "train": (time_config['train_start'], time_config['train_end']),
        "valid": (time_config['valid_start'], time_config['valid_end']),
        "test": (time_config['test_start'], time_config['test_end']),
    }
    return DatasetH(handler=handler, segments=segments)


def prepare_data(dataset, segment):
    """准备数据"""
    features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
    features = features.fillna(0).replace([np.inf, -np.inf], 0).clip(-10, 10)

    try:
        labels = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(labels, pd.DataFrame):
            labels = labels.iloc[:, 0]
        labels = labels.fillna(0).values
        return features.values, labels, features.index
    except Exception:
        return features.values, None, features.index


def main():
    parser = argparse.ArgumentParser(
        description='Run backtest from saved model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 模型参数
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to saved .keras model')
    parser.add_argument('--handler', type=str, required=True,
                        choices=list(HANDLER_CONFIG.keys()),
                        help='Handler type (must match the model)')
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--nday', type=int, default=5,
                        help='Prediction horizon (must match the model)')

    # 时间配置
    parser.add_argument('--test-start', type=str, default='2025-01-01')
    parser.add_argument('--test-end', type=str, default='2025-12-31')

    # Backtest 参数
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=1)
    parser.add_argument('--account', type=float, default=10000)
    parser.add_argument('--rebalance-freq', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss'])

    # 策略高级参数
    parser.add_argument('--risk-lookback', type=int, default=20)
    parser.add_argument('--drawdown-threshold', type=float, default=-0.10)
    parser.add_argument('--momentum-threshold', type=float, default=0.03)
    parser.add_argument('--risk-high', type=float, default=0.50)
    parser.add_argument('--risk-medium', type=float, default=0.75)
    parser.add_argument('--risk-normal', type=float, default=0.95)
    parser.add_argument('--market-proxy', type=str, default='AAPL')
    parser.add_argument('--vol-high', type=float, default=0.35)
    parser.add_argument('--vol-medium', type=float, default=0.25)
    parser.add_argument('--stop-loss', type=float, default=-0.15)
    parser.add_argument('--no-sell-after-drop', type=float, default=-0.20)

    # 可选：只评估不回测
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate IC, skip backtest')

    args = parser.parse_args()

    # 检查模型文件
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = project_root / model_path

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    # 配置
    symbols = STOCK_POOLS[args.stock_pool]
    time_config = DEFAULT_TEST_CONFIG.copy()
    time_config['test_start'] = args.test_start
    time_config['test_end'] = args.test_end

    print("=" * 70)
    print("Backtest from Saved Model")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Handler: {args.handler}")
    print(f"Stock pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Test period: {time_config['test_start']} ~ {time_config['test_end']}")
    print("=" * 70)

    # 加载模型
    print("\n[1] Loading model...")
    model = keras.models.load_model(str(model_path))
    print(f"    Model loaded: {model.name}")
    print(f"    Input shape: {model.input_shape}")
    print(f"    Output names: {[o.name for o in model.outputs]}")

    # 创建数据
    print("\n[2] Preparing data...")
    handler = create_data_handler(args, symbols, time_config)
    dataset = create_dataset(handler, time_config)

    X_test, y_test, test_index = prepare_data(dataset, "test")
    print(f"    Test data shape: {X_test.shape}")
    print(f"    Test samples: {len(X_test)}")

    # 检查特征数是否匹配
    expected_features = model.input_shape[1]
    if X_test.shape[1] != expected_features:
        print(f"\n    ⚠️  Warning: Feature count mismatch!")
        print(f"    Model expects: {expected_features}")
        print(f"    Data has: {X_test.shape[1]}")
        print(f"    Please check if handler matches the model.")
        sys.exit(1)

    # 预测
    print("\n[3] Making predictions...")
    predictions = model.predict(X_test, verbose=0)

    # AE-MLP 有多个输出，取最后一个 (action)
    if isinstance(predictions, list):
        test_pred = predictions[-1].flatten()
        print(f"    Using output: action (index -1)")
    else:
        test_pred = predictions.flatten()

    print(f"    Predictions shape: {test_pred.shape}")
    print(f"    Pred range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    # 评估
    print("\n[4] Evaluating model...")
    # evaluate_model 需要带 index 的 pandas Series
    test_pred_series = pd.Series(test_pred, index=test_index)
    evaluate_model(dataset, test_pred_series, PROJECT_ROOT, args.nday)

    # Backtest
    if not args.eval_only:
        print("\n[5] Running backtest...")

        def load_model_func(path):
            return keras.models.load_model(str(path))

        def get_feature_count_func(m):
            return m.input_shape[1]

        run_backtest(
            model_path=model_path,
            dataset=dataset,
            pred=test_pred_series,
            args=args,
            time_splits=time_config,
            model_name="AE-MLP",
            load_model_func=load_model_func,
            get_feature_count_func=get_feature_count_func,
        )

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
