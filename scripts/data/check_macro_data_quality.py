#!/usr/bin/env python
"""
检查 Alpha158_Volatility_TALib_Macro handler 数据质量
输出 NaN、Inf 等非法值的统计报告
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import qlib
from qlib.constant import REG_US
from scripts.utils.talib_ops import TALIB_OPS
from scripts.data.datahandler_macro import Alpha158_Volatility_TALib_Macro


def check_data_quality(df: pd.DataFrame, name: str) -> dict:
    """检查数据质量"""
    stats = {
        'name': name,
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'total_cells': df.size,
    }

    # NaN 统计
    nan_counts = df.isna().sum()
    stats['nan_total'] = int(nan_counts.sum())
    stats['nan_pct'] = stats['nan_total'] / stats['total_cells'] * 100
    stats['cols_with_nan'] = int((nan_counts > 0).sum())

    # Inf 统计
    numeric_df = df.select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric_df).sum()
    stats['inf_total'] = int(inf_counts.sum())
    stats['inf_pct'] = stats['inf_total'] / stats['total_cells'] * 100
    stats['cols_with_inf'] = int((inf_counts > 0).sum())

    # 每列详细统计
    col_stats = []
    for col in df.columns:
        col_data = df[col]
        col_info = {
            'column': col,
            'nan_count': int(col_data.isna().sum()),
            'nan_pct': col_data.isna().sum() / len(col_data) * 100,
        }
        if np.issubdtype(col_data.dtype, np.number):
            col_info['inf_count'] = int(np.isinf(col_data).sum())
            col_info['min'] = float(col_data.min()) if not col_data.isna().all() else np.nan
            col_info['max'] = float(col_data.max()) if not col_data.isna().all() else np.nan
            col_info['mean'] = float(col_data.mean()) if not col_data.isna().all() else np.nan
            col_info['std'] = float(col_data.std()) if not col_data.isna().all() else np.nan
        else:
            col_info['inf_count'] = 0
        col_stats.append(col_info)

    stats['col_stats'] = col_stats
    return stats


def print_report(stats: dict):
    """打印报告"""
    print("=" * 80)
    print(f"数据质量报告: {stats['name']}")
    print("=" * 80)

    print(f"\n总体统计:")
    print(f"  总行数: {stats['total_rows']:,}")
    print(f"  总列数: {stats['total_cols']:,}")
    print(f"  总单元格数: {stats['total_cells']:,}")

    print(f"\nNaN 统计:")
    print(f"  NaN 总数: {stats['nan_total']:,} ({stats['nan_pct']:.2f}%)")
    print(f"  含 NaN 的列数: {stats['cols_with_nan']}")

    print(f"\nInf 统计:")
    print(f"  Inf 总数: {stats['inf_total']:,} ({stats['inf_pct']:.2f}%)")
    print(f"  含 Inf 的列数: {stats['cols_with_inf']}")

    # 问题列详情
    problem_cols = [c for c in stats['col_stats']
                    if c['nan_count'] > 0 or c.get('inf_count', 0) > 0]

    if problem_cols:
        print(f"\n问题列详情 (共 {len(problem_cols)} 列):")
        print("-" * 80)
        print(f"{'列名':<45} {'NaN数':>10} {'NaN%':>8} {'Inf数':>8}")
        print("-" * 80)

        # 按 NaN 数量排序
        problem_cols.sort(key=lambda x: x['nan_count'], reverse=True)
        for col in problem_cols:
            print(f"{col['column']:<45} {col['nan_count']:>10,} {col['nan_pct']:>7.2f}% {col.get('inf_count', 0):>8,}")
    else:
        print("\n✓ 所有列均无 NaN 或 Inf 值")

    # 数值范围异常检测
    print(f"\n数值范围检查:")
    print("-" * 80)
    extreme_cols = []
    for col in stats['col_stats']:
        if 'min' in col and not np.isnan(col['min']):
            if abs(col['min']) > 1e10 or abs(col['max']) > 1e10:
                extreme_cols.append(col)

    if extreme_cols:
        print(f"发现 {len(extreme_cols)} 列有极端值:")
        print(f"{'列名':<45} {'最小值':>15} {'最大值':>15}")
        for col in extreme_cols:
            print(f"{col['column']:<45} {col['min']:>15.2e} {col['max']:>15.2e}")
    else:
        print("✓ 所有列数值范围正常")


def main():
    print("初始化 Qlib...")
    qlib.init(
        provider_uri="./my_data/qlib_us",
        region=REG_US,
        custom_ops=TALIB_OPS,
    )

    # 使用 test stock pool 快速测试
    from scripts.data.stock_pools import TEST_SYMBOLS

    # 创建 instruments 文件
    instruments_path = Path("./my_data/qlib_us/instruments/test_check.txt")
    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    with open(instruments_path, 'w') as f:
        for symbol in TEST_SYMBOLS:
            f.write(f"{symbol}\t2020-01-01\t2025-12-31\n")

    print("\n加载 Alpha158_Volatility_TALib_Macro handler...")
    print("Stock pool: test (10 stocks)")
    print("Date range: 2020-01-01 ~ 2025-12-31")

    handler = Alpha158_Volatility_TALib_Macro(
        instruments="test_check",
        start_time="2020-01-01",
        end_time="2025-12-31",
        fit_start_time="2020-01-01",
        fit_end_time="2024-12-31",
        infer_processors=[],
        learn_processors=[],
        label=["Ref($close, -5) / Ref($close, -1) - 1"],
    )

    # 获取数据
    print("\n获取数据...")
    df_learn = handler.fetch(col_set="feature")
    df_label = handler.fetch(col_set="label")

    print(f"\nFeature shape: {df_learn.shape}")
    print(f"Label shape: {df_label.shape}")

    # 重置 index 以便分析
    df_learn_reset = df_learn.reset_index()
    df_label_reset = df_label.reset_index()

    # 检查特征数据
    print("\n" + "=" * 80)
    feature_stats = check_data_quality(df_learn, "Features")
    print_report(feature_stats)

    # 检查标签数据
    print("\n" + "=" * 80)
    label_stats = check_data_quality(df_label, "Labels")
    print_report(label_stats)

    # 按时间段分析 NaN 分布
    print("\n" + "=" * 80)
    print("按时间段分析 NaN 分布")
    print("=" * 80)

    df_with_time = df_learn.reset_index()
    df_with_time['year'] = pd.to_datetime(df_with_time['datetime']).dt.year

    feature_cols = [c for c in df_with_time.columns if c not in ['datetime', 'instrument', 'year']]

    print(f"\n{'年份':<10} {'总行数':>12} {'NaN行数':>12} {'NaN%':>10}")
    print("-" * 50)
    for year in sorted(df_with_time['year'].unique()):
        year_data = df_with_time[df_with_time['year'] == year][feature_cols]
        total_rows = len(year_data)
        nan_rows = year_data.isna().any(axis=1).sum()
        nan_pct = nan_rows / total_rows * 100 if total_rows > 0 else 0
        print(f"{year:<10} {total_rows:>12,} {nan_rows:>12,} {nan_pct:>9.2f}%")

    # 分类统计宏观特征
    print("\n" + "=" * 80)
    print("宏观特征分类统计")
    print("=" * 80)

    macro_cols = [c for c in df_learn.columns if 'macro_' in str(c)]
    non_macro_cols = [c for c in df_learn.columns if 'macro_' not in str(c)]

    if macro_cols:
        macro_df = df_learn[macro_cols]
        macro_nan = macro_df.isna().sum().sum()
        macro_total = macro_df.size
        print(f"\n宏观特征 ({len(macro_cols)} 列):")
        print(f"  NaN 总数: {macro_nan:,} ({macro_nan/macro_total*100:.2f}%)")

        # 各类宏观特征
        categories = {
            'VIX': [c for c in macro_cols if 'vix' in str(c).lower() or 'uvxy' in str(c).lower() or 'svxy' in str(c).lower()],
            'GLD': [c for c in macro_cols if 'gld' in str(c).lower()],
            'Bond': [c for c in macro_cols if any(x in str(c).lower() for x in ['tlt', 'ief', 'shy', 'bond'])],
            'Dollar': [c for c in macro_cols if any(x in str(c).lower() for x in ['uup', 'dollar'])],
            'Oil': [c for c in macro_cols if 'uso' in str(c).lower()],
            'Sector': [c for c in macro_cols if 'xl' in str(c).lower()],
            'Credit': [c for c in macro_cols if any(x in str(c).lower() for x in ['hyg', 'lqd', 'jnk', 'credit', 'spread'])],
            'Global': [c for c in macro_cols if any(x in str(c).lower() for x in ['eem', 'efa', 'fxi', 'ewj', 'global'])],
            'SPY/QQQ': [c for c in macro_cols if any(x in str(c).lower() for x in ['spy', 'qqq'])],
            'Yield': [c for c in macro_cols if any(x in str(c).lower() for x in ['yield', 'curve'])],
        }

        print(f"\n  {'类别':<15} {'特征数':>8} {'NaN数':>12} {'NaN%':>10}")
        print("  " + "-" * 50)
        for cat, cols in categories.items():
            if cols:
                cat_df = df_learn[cols]
                cat_nan = cat_df.isna().sum().sum()
                cat_total = cat_df.size
                print(f"  {cat:<15} {len(cols):>8} {cat_nan:>12,} {cat_nan/cat_total*100:>9.2f}%")

    if non_macro_cols:
        non_macro_df = df_learn[non_macro_cols]
        non_macro_nan = non_macro_df.isna().sum().sum()
        non_macro_total = non_macro_df.size
        print(f"\n非宏观特征 ({len(non_macro_cols)} 列):")
        print(f"  NaN 总数: {non_macro_nan:,} ({non_macro_nan/non_macro_total*100:.2f}%)")

    # 清理临时文件
    instruments_path.unlink(missing_ok=True)

    print("\n" + "=" * 80)
    print("检查完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
