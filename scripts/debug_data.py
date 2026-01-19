"""
诊断 Alpha360 数据质量问题
"""
import sys
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
project_root = script_dir.parent

import qlib
from qlib.constant import REG_US
from utils.talib_ops import TALIB_OPS

qlib.init(
    provider_uri=str(project_root / "my_data" / "qlib_us"),
    region=REG_US,
    custom_ops=TALIB_OPS,
)

import numpy as np
import pandas as pd
from data.stock_pools import STOCK_POOLS
from qlib.contrib.data.handler import Alpha360

print("=" * 60)
print("Alpha360 Data Quality Diagnostic")
print("=" * 60)

# 创建 handler
symbols = STOCK_POOLS['sp500']
print(f"Stock pool: sp500 ({len(symbols)} stocks)")

handler = Alpha360(
    instruments=symbols,
    start_time='2020-01-01',
    end_time='2024-12-31',
    fit_start_time='2020-01-01',
    fit_end_time='2022-12-31',
)

# 获取数据
df = handler.fetch()
print(f'\nData shape: {df.shape}')

# 检查特征
features = df['feature']
print('\n' + '=' * 60)
print('FEATURE STATISTICS')
print('=' * 60)
print(f'Shape: {features.shape}')
total_nan = features.isna().sum().sum()
print(f'NaN count: {total_nan} ({total_nan / features.size * 100:.2f}%)')
print(f'Inf count: {np.isinf(features.values).sum()}')

# 检查每列的 NaN 比例
nan_pct = features.isna().mean()
high_nan_cols = nan_pct[nan_pct > 0.1].sort_values(ascending=False)
if len(high_nan_cols) > 0:
    print(f'\nColumns with >10% NaN ({len(high_nan_cols)} columns):')
    for col, pct in high_nan_cols.head(10).items():
        print(f'  {col}: {pct*100:.1f}%')

# 检查极端值
valid_values = features.values[~np.isnan(features.values)]
print('\n--- Value Range ---')
print(f'Min: {valid_values.min():.4f}')
print(f'Max: {valid_values.max():.4f}')
print(f'Mean: {np.mean(valid_values):.4f}')
print(f'Std: {np.std(valid_values):.4f}')

# 检查哪些列有极端值
col_max = features.abs().max()
extreme_cols = col_max[col_max > 100].sort_values(ascending=False)
if len(extreme_cols) > 0:
    print(f'\nColumns with extreme values (abs > 100): {len(extreme_cols)}')
    for col, val in extreme_cols.head(10).items():
        print(f'  {col}: max abs = {val:.2f}')

# 检查标签
print('\n' + '=' * 60)
print('LABEL STATISTICS')
print('=' * 60)
labels = df['label']
print(f'Shape: {labels.shape}')
label_nan = labels.isna().sum().sum()
print(f'NaN count: {label_nan} ({label_nan / labels.size * 100:.2f}%)')
label_values = labels.values.flatten()
label_values = label_values[~np.isnan(label_values)]
print(f'Min: {label_values.min():.4f}')
print(f'Max: {label_values.max():.4f}')
print(f'Mean: {np.mean(label_values):.6f}')
print(f'Std: {np.std(label_values):.4f}')

# 检查特定问题列
print('\n' + '=' * 60)
print('VWAP COLUMN CHECK (known issue in US data)')
print('=' * 60)
vwap_cols = [c for c in features.columns if 'VWAP' in c]
if vwap_cols:
    vwap_nan_pct = features[vwap_cols].isna().mean()
    print(f'VWAP columns: {len(vwap_cols)}')
    print(f'VWAP NaN percentage: {vwap_nan_pct.mean()*100:.1f}%')
    if vwap_nan_pct.mean() > 0.5:
        print('>>> WARNING: VWAP data is mostly missing!')

print('\n' + '=' * 60)
print('DIAGNOSTIC COMPLETE')
print('=' * 60)
