"""
测试 DynamicAlpha300Handler 特征排列顺序

验证：
1. Alpha300 基线特征按 CLOSE, OPEN, HIGH, LOW, VOLUME 顺序排列，每个60天
2. 添加的 TALib 特征是否按60天顺序排列
3. 添加的 Macro 特征是否按60天顺序排列
"""

import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))
project_root = script_dir.parent

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from utils.talib_ops import TALIB_OPS
from data.stock_pools import STOCK_POOLS

# 初始化 Qlib
qlib_data_path = project_root / "my_data" / "qlib_us"
qlib.init(
    provider_uri=str(qlib_data_path),
    region=REG_US,
    custom_ops=TALIB_OPS,
)

# 导入 DynamicAlpha300Handler
from models.feature_engineering.nested_cv_feature_selection_transformer import (
    DynamicAlpha300Handler,
    ALPHA300_SEQ_LEN,
    ALPHA300_BASE_FEATURES,
)


def test_feature_order():
    """测试特征排列顺序"""
    symbols = STOCK_POOLS['test'][:3]  # 只用3个股票测试

    # 测试的 TALib 特征
    test_talib = {
        "TALIB_CMO14": "TALIB_CMO($close, 14)",
        "TALIB_MACD_SIGNAL": "TALIB_MACD_SIGNAL($close, 12, 26, 9)/$close",
    }

    # 测试的 Macro 特征
    test_macro = ["macro_vix_zscore20", "macro_spy_pct_5d"]

    print("=" * 70)
    print("Testing DynamicAlpha300Handler Feature Order")
    print("=" * 70)

    # 创建 handler
    handler = DynamicAlpha300Handler(
        talib_features=test_talib,
        macro_features=test_macro,
        volatility_window=5,
        instruments=symbols,
        start_time='2024-01-01',
        end_time='2024-01-31',
        fit_start_time='2024-01-01',
        fit_end_time='2024-01-31',
        infer_processors=[],
    )

    dataset = DatasetH(
        handler=handler,
        segments={"train": ('2024-01-01', '2024-01-31')}
    )

    features = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)

    print(f"\nTotal features: {features.shape[1]}")
    print(f"Expected: {ALPHA300_BASE_FEATURES} base × 60 + {len(test_talib)} talib × 60 + {len(test_macro)} macro × 60")
    expected = (ALPHA300_BASE_FEATURES + len(test_talib) + len(test_macro)) * ALPHA300_SEQ_LEN
    print(f"Expected total: {expected}")

    # 获取列名
    if hasattr(features.columns, 'get_level_values'):
        col_names = features.columns.get_level_values(-1).tolist()
    else:
        col_names = features.columns.tolist()

    print(f"\n{'='*70}")
    print("Feature Column Names (showing pattern)")
    print("=" * 70)

    # 分析特征顺序
    print("\n[Alpha300 Base Features]")
    base_features = ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME']
    for i, feat in enumerate(base_features):
        start_idx = i * ALPHA300_SEQ_LEN
        end_idx = start_idx + ALPHA300_SEQ_LEN
        cols = col_names[start_idx:end_idx]
        print(f"  {feat}: indices {start_idx}-{end_idx-1}")
        print(f"    First 5: {cols[:5]}")
        print(f"    Last 5:  {cols[-5:]}")

        # 检查是否按 59, 58, ..., 1, 0 顺序
        expected_names = [f"{feat}{j}" for j in range(ALPHA300_SEQ_LEN-1, -1, -1)]
        if cols == expected_names:
            print(f"    Order: CORRECT (59 -> 0)")
        else:
            print(f"    Order: MISMATCH!")
            print(f"    Expected: {expected_names[:5]} ... {expected_names[-5:]}")

    # TALib 特征
    talib_start = ALPHA300_BASE_FEATURES * ALPHA300_SEQ_LEN
    print(f"\n[TALib Features] starting at index {talib_start}")
    for i, name in enumerate(test_talib.keys()):
        start_idx = talib_start + i * ALPHA300_SEQ_LEN
        end_idx = start_idx + ALPHA300_SEQ_LEN
        cols = col_names[start_idx:end_idx]
        print(f"  {name}: indices {start_idx}-{end_idx-1}")
        print(f"    First 5: {cols[:5]}")
        print(f"    Last 5:  {cols[-5:]}")

        # 检查顺序：应该是 name_59, name_58, ..., name_1, name_0
        expected_names = [f"{name}_{j}" for j in range(ALPHA300_SEQ_LEN-1, -1, -1)]
        if cols == expected_names:
            print(f"    Order: CORRECT (59 -> 0)")
        else:
            print(f"    Order: MISMATCH!")
            print(f"    Expected first 5: {expected_names[:5]}")
            print(f"    Expected last 5:  {expected_names[-5:]}")

    # Macro 特征
    macro_start = talib_start + len(test_talib) * ALPHA300_SEQ_LEN
    print(f"\n[Macro Features] starting at index {macro_start}")
    for i, name in enumerate(test_macro):
        start_idx = macro_start + i * ALPHA300_SEQ_LEN
        end_idx = start_idx + ALPHA300_SEQ_LEN
        if end_idx <= len(col_names):
            cols = col_names[start_idx:end_idx]
            print(f"  {name}: indices {start_idx}-{end_idx-1}")
            print(f"    First 5: {cols[:5]}")
            print(f"    Last 5:  {cols[-5:]}")

            # 检查顺序：应该是 name_59, name_58, ..., name_1, name_0
            expected_names = [f"{name}_{j}" for j in range(ALPHA300_SEQ_LEN-1, -1, -1)]
            if cols == expected_names:
                print(f"    Order: CORRECT (59 -> 0)")
            else:
                print(f"    Order: MISMATCH!")
                print(f"    Expected first 5: {expected_names[:5]}")
                print(f"    Expected last 5:  {expected_names[-5:]}")
        else:
            print(f"  {name}: NOT FOUND (indices {start_idx}-{end_idx-1} out of range)")

    # 打印完整列名列表用于调试
    print(f"\n{'='*70}")
    print("All Column Names (for debugging)")
    print("=" * 70)
    for i, col in enumerate(col_names):
        if i < 10 or i >= len(col_names) - 10 or (i % 60 < 3) or (i % 60 >= 57):
            print(f"  [{i:4d}] {col}")
        elif i == 10:
            print("  ...")

    # 检查数据形状
    print(f"\n{'='*70}")
    print("Data Shape Analysis")
    print("=" * 70)
    print(f"Features shape: {features.shape}")
    print(f"Samples: {features.shape[0]}")
    print(f"Features: {features.shape[1]}")

    d_feat = features.shape[1] // ALPHA300_SEQ_LEN
    remainder = features.shape[1] % ALPHA300_SEQ_LEN
    print(f"\nIf reshape to (samples, {ALPHA300_SEQ_LEN}, d_feat):")
    print(f"  d_feat = {d_feat}")
    print(f"  remainder = {remainder} (should be 0)")

    if remainder != 0:
        print(f"  WARNING: Features not divisible by {ALPHA300_SEQ_LEN}!")


if __name__ == "__main__":
    test_feature_order()
