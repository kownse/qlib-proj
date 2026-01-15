"""
检查 Alpha158_Volatility_TALib_Macro handler 数据结构
查看 macro 特征是否和其他特征在同一维度上
"""

import sys
from pathlib import Path
import pandas as pd

# 添加路径
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

import qlib
from qlib.constant import REG_US
from utils.talib_ops import TALIB_OPS
from data.datahandler_macro import Alpha158_Volatility_TALib_Macro

# 初始化 qlib
qlib.init(
    provider_uri="./my_data/qlib_us",
    region=REG_US,
    custom_ops=TALIB_OPS,
)

# 创建 handler
handler = Alpha158_Volatility_TALib_Macro(
    volatility_window=2,
    instruments=["AAPL", "MSFT"],  # 只用两只股票测试
    start_time="2024-12-01",
    end_time="2024-12-31",
    macro_features="core",  # 使用 core 特征集 (~23个)
)

# 获取数据
df = handler.fetch()

print("=" * 80)
print("数据整体结构")
print("=" * 80)
print(f"Shape: {df.shape}")
print(f"Index names: {df.index.names}")
print(f"Index levels: {df.index.nlevels}")
print(f"Columns type: {type(df.columns)}")

print("\n" + "=" * 80)
print("列信息")
print("=" * 80)
if isinstance(df.columns, tuple) or (hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1):
    print("MultiIndex columns:")
    for i, col in enumerate(df.columns[:20]):
        print(f"  {i}: {col}")
    print(f"  ... (共 {len(df.columns)} 列)")
else:
    print("Flat columns:")
    print(df.columns.tolist()[:20])
    print(f"... (共 {len(df.columns)} 列)")

# 分离特征和标签
if hasattr(df.columns, 'get_level_values'):
    top_level = df.columns.get_level_values(0).unique()
    print(f"\n顶级列: {top_level.tolist()}")

    if 'feature' in top_level:
        feature_cols = [c for c in df.columns if c[0] == 'feature']
        label_cols = [c for c in df.columns if c[0] == 'label']
        print(f"Feature columns: {len(feature_cols)}")
        print(f"Label columns: {len(label_cols)}")

print("\n" + "=" * 80)
print("某一天的数据 (2024-12-02)")
print("=" * 80)

# 选择某一天的数据
target_date = "2024-12-02"
try:
    day_data = df.xs(target_date, level='datetime')
    print(f"日期 {target_date} 的数据 shape: {day_data.shape}")
    print(f"股票列表: {day_data.index.tolist()}")

    # 打印每只股票的特征值
    for symbol in day_data.index[:1]:  # 只看第一只
        print(f"\n--- {symbol} 的特征 ---")
        row = day_data.loc[symbol]

        # 分类打印
        macro_features = []
        alpha_features = []
        talib_features = []
        other_features = []
        label_features = []

        for col in row.index:
            col_name = col[1] if isinstance(col, tuple) else col
            val = row[col]

            if isinstance(col, tuple) and col[0] == 'label':
                label_features.append((col_name, val))
            elif 'macro_' in str(col_name).lower():
                macro_features.append((col_name, val))
            elif 'talib_' in str(col_name).lower():
                talib_features.append((col_name, val))
            elif col_name.startswith(('KBAR', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME',
                                       'ROC', 'MA', 'STD', 'BETA', 'RSQR', 'RESI',
                                       'MAX', 'MIN', 'QTLU', 'QTLD', 'RANK', 'CORR',
                                       'CORD', 'CNTP', 'CNTN', 'CNTD', 'SUMP', 'SUMN',
                                       'SUMD', 'WVMA', 'VSUMP', 'VSUMN', 'VSUMD')):
                alpha_features.append((col_name, val))
            else:
                other_features.append((col_name, val))

        print(f"\nAlpha158 特征 ({len(alpha_features)} 个):")
        for name, val in alpha_features[:10]:
            print(f"  {name}: {val:.6f}" if not pd.isna(val) else f"  {name}: NaN")
        if len(alpha_features) > 10:
            print(f"  ... 还有 {len(alpha_features) - 10} 个")

        print(f"\nTA-Lib 特征 ({len(talib_features)} 个):")
        for name, val in talib_features[:10]:
            print(f"  {name}: {val:.6f}" if not pd.isna(val) else f"  {name}: NaN")
        if len(talib_features) > 10:
            print(f"  ... 还有 {len(talib_features) - 10} 个")

        print(f"\nMacro 特征 ({len(macro_features)} 个):")
        for name, val in macro_features:
            print(f"  {name}: {val:.6f}" if not pd.isna(val) else f"  {name}: NaN")

        print(f"\n其他特征 ({len(other_features)} 个):")
        for name, val in other_features[:10]:
            print(f"  {name}: {val:.6f}" if not pd.isna(val) else f"  {name}: NaN")

        print(f"\n标签 ({len(label_features)} 个):")
        for name, val in label_features:
            print(f"  {name}: {val:.6f}" if not pd.isna(val) else f"  {name}: NaN")

except KeyError as e:
    print(f"无法找到日期 {target_date}: {e}")
    print(f"\n可用日期:")
    dates = df.index.get_level_values('datetime').unique()
    print(dates[:10])

print("\n" + "=" * 80)
print("检查不同股票在同一天的 macro 特征是否相同")
print("=" * 80)

import pandas as pd

try:
    day_data = df.xs("2024-12-02", level='datetime')
    if len(day_data) >= 2:
        symbols = day_data.index.tolist()[:2]

        # 获取 macro 列
        if isinstance(df.columns, pd.MultiIndex):
            macro_cols = [c for c in df.columns if 'macro_' in str(c[1]).lower()]
        else:
            macro_cols = [c for c in df.columns if 'macro_' in str(c).lower()]

        if macro_cols:
            print(f"\n比较 {symbols[0]} vs {symbols[1]} 的 macro 特征:")
            for col in macro_cols[:5]:
                val1 = day_data.loc[symbols[0], col]
                val2 = day_data.loc[symbols[1], col]
                col_name = col[1] if isinstance(col, tuple) else col
                match = "✓ 相同" if val1 == val2 or (pd.isna(val1) and pd.isna(val2)) else "✗ 不同"
                print(f"  {col_name}: {val1:.4f} vs {val2:.4f} {match}"
                      if not pd.isna(val1) else f"  {col_name}: NaN vs NaN {match}")
        else:
            print("未找到 macro 特征列")
except Exception as e:
    print(f"比较失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("数据维度总结")
print("=" * 80)
print(f"数据总维度: {df.shape}")
print(f"每行代表: 一个股票在一天的所有特征")
print(f"Index: (datetime, instrument) - 时间 x 股票")
print(f"Columns: (feature/label, feature_name) - 特征类型 x 特征名")
