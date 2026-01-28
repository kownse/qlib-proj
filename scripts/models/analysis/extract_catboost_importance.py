"""
提取 CatBoost 模型的 Feature Importance

运行一次 CatBoost 训练，提取 top N 特征重要性，保存到 JSON 文件
"""

# ============================================================================
# 重要: 以下代码必须在任何其他导入之前执行
# ============================================================================

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

import sys
from pathlib import Path
import multiprocessing
import argparse
import json
from datetime import datetime

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))
project_root = script_dir.parent

import qlib
from qlib.constant import REG_US
from utils.talib_ops import TALIB_OPS

qlib_data_path = project_root / "my_data" / "qlib_us"
qlib.init(
    provider_uri=str(qlib_data_path),
    region=REG_US,
    custom_ops=TALIB_OPS,
    kernels=1,
    joblib_backend=None,
)

# ============================================================================
# 现在可以安全地导入其他模块
# ============================================================================

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from qlib.contrib.model.catboost_model import CatBoostModel
from qlib.data.dataset.handler import DataHandlerLP

from data.stock_pools import STOCK_POOLS
from models.common import (
    HANDLER_CONFIG, PROJECT_ROOT, MODEL_SAVE_PATH,
    get_time_splits,
    init_qlib,
    check_data_availability,
    create_data_handler,
    create_dataset,
    analyze_features,
    print_feature_importance,
)

DEFAULT_CATBOOST_PARAMS = {
    'loss_function': 'RMSE',
    'learning_rate': 0.05,
    'max_depth': 6,
    'l2_leaf_reg': 3,
    'random_strength': 1,
    'thread_count': 16,
    'verbose': False,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CatBoost Feature Importance",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--stock-pool', type=str, default='test',
                        choices=list(STOCK_POOLS.keys()),
                        help='Stock pool to use (default: test)')
    parser.add_argument('--handler', type=str, default='alpha158',
                        choices=list(HANDLER_CONFIG.keys()),
                        help='Data handler to use (default: alpha158)')
    parser.add_argument('--nday', type=int, default=5,
                        help='Prediction horizon in days (default: 5)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top features to extract (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path (default: auto-generated)')
    parser.add_argument('--max-train', action='store_true',
                        help='Use maximum training data')
    return parser.parse_args()


def train_and_extract_importance(dataset, valid_cols):
    """
    训练 CatBoost 并提取 feature importance

    Returns
    -------
    pd.DataFrame
        feature importance DataFrame, sorted by importance descending
    """
    print("\n[*] Training CatBoost model...")

    model = CatBoostModel(
        loss='RMSE',
        learning_rate=DEFAULT_CATBOOST_PARAMS['learning_rate'],
        max_depth=DEFAULT_CATBOOST_PARAMS['max_depth'],
        l2_leaf_reg=DEFAULT_CATBOOST_PARAMS['l2_leaf_reg'],
        random_strength=DEFAULT_CATBOOST_PARAMS['random_strength'],
        thread_count=DEFAULT_CATBOOST_PARAMS['thread_count'],
    )

    model.fit(
        dataset,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=100,
    )
    print("    Model training completed")

    # 提取 feature importance
    print("\n[*] Extracting feature importance...")
    importance = model.model.get_feature_importance()
    feature_names_from_model = model.model.feature_names_

    if feature_names_from_model:
        feature_names = feature_names_from_model
    else:
        feature_names = valid_cols[:len(importance)]

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    return importance_df


def save_top_features(importance_df, top_n, output_path, args):
    """
    保存 top N 特征到 JSON 文件
    """
    top_features = importance_df.head(top_n)

    result = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'stock_pool': args.stock_pool,
            'handler': args.handler,
            'nday': args.nday,
            'top_n': top_n,
            'total_features': len(importance_df),
        },
        'top_features': [
            {
                'rank': i + 1,
                'feature': row['feature'],
                'importance': float(row['importance']),
            }
            for i, row in top_features.iterrows()
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[*] Saved top {top_n} features to: {output_path}")


def main():
    args = parse_args()

    # 配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = get_time_splits(args.max_train)

    print("=" * 60)
    print("CatBoost Feature Importance Extraction")
    print("=" * 60)
    print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Handler: {args.handler}")
    print(f"N-day: {args.nday}")
    print(f"Top-N: {args.top_n}")
    print("=" * 60)

    # 初始化
    init_qlib(handler_config['use_talib'])
    check_data_availability(time_splits)

    # 创建 handler 和 dataset
    # 需要创建一个临时的 args 对象
    class TempArgs:
        pass
    temp_args = TempArgs()
    temp_args.nday = args.nday
    temp_args.handler = args.handler

    handler = create_data_handler(temp_args, handler_config, symbols, time_splits)
    dataset = create_dataset(handler, time_splits)

    # 分析特征
    train_data, valid_cols, dropped_cols = analyze_features(dataset)

    # 训练并提取 importance
    importance_df = train_and_extract_importance(dataset, valid_cols)

    # 打印 top features
    print(f"\n[*] Top {args.top_n} Features:")
    print("-" * 60)
    for i, row in importance_df.head(args.top_n).iterrows():
        print(f"    {i+1:3d}. {row['feature']:<40s} {row['importance']:>10.2f}")
    print("-" * 60)

    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "my_models" / "feature_importance"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"catboost_importance_{args.handler}_{args.stock_pool}_{timestamp}.json"

    # 保存结果
    save_top_features(importance_df, args.top_n, output_path, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
