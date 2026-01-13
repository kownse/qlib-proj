"""
检查 alpha158-talib-lite handler 在 SP500 数据中的异常值

诊断项目:
1. NaN 值统计
2. 无穷大值检测
3. 极端异常值检测（超过均值±10倍标准差）
4. 常量列检测（方差为0）
5. 数据范围统计
6. 每只股票的数据完整性
"""

import os

# 设置环境变量，避免 TA-Lib 内存冲突
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
from pathlib import Path
import multiprocessing

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# 设置路径
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))
project_root = script_dir.parent

import numpy as np
import pandas as pd
from datetime import datetime

# 初始化 qlib
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

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from data.datahandler_ext import Alpha158_Volatility_TALib_Lite
from data.stock_pools import SP500_SYMBOLS


def check_anomalies():
    """检查数据异常值"""

    print("=" * 70)
    print("Alpha158-TALib-Lite 数据异常值检查")
    print(f"股票池: SP500 ({len(SP500_SYMBOLS)} 只股票)")
    print(f"时间范围: 2000-01-01 到 2025-12-31")
    print("=" * 70)

    # 创建 Handler
    print("\n[1] 创建 DataHandler...")
    try:
        handler = Alpha158_Volatility_TALib_Lite(
            volatility_window=5,
            instruments=SP500_SYMBOLS,
            start_time="2000-01-01",
            end_time="2025-12-31",
            fit_start_time="2000-01-01",
            fit_end_time="2022-12-31",
            infer_processors=[],
        )
        print("    ✓ DataHandler 创建成功")
    except Exception as e:
        print(f"    ✗ DataHandler 创建失败: {e}")
        return

    # 创建 Dataset
    print("\n[2] 创建 Dataset...")
    try:
        dataset = DatasetH(
            handler=handler,
            segments={
                "train": ("2000-01-01", "2022-12-31"),
                "valid": ("2023-01-01", "2023-12-31"),
                "test": ("2024-01-01", "2025-12-31"),
            }
        )
        print("    ✓ Dataset 创建成功")
    except Exception as e:
        print(f"    ✗ Dataset 创建失败: {e}")
        return

    # 获取训练数据
    print("\n[3] 加载数据...")
    try:
        # 使用 DK_L (learn) 获取经过处理的数据
        train_features = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
        train_labels = dataset.prepare("train", col_set="label")
        print(f"    ✓ 训练特征形状: {train_features.shape}")
        print(f"    ✓ 训练标签形状: {train_labels.shape}")
    except Exception as e:
        print(f"    ✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==================== 特征异常值检查 ====================
    print("\n" + "=" * 70)
    print("特征异常值分析")
    print("=" * 70)

    df = train_features

    # 1. NaN 值统计
    print("\n[4] NaN 值统计...")
    nan_counts = df.isna().sum()
    nan_pct = (nan_counts / len(df) * 100).round(2)

    # 高 NaN 比例的列
    high_nan_cols = nan_pct[nan_pct > 10].sort_values(ascending=False)
    if len(high_nan_cols) > 0:
        print(f"    ⚠ {len(high_nan_cols)} 个特征 NaN 比例 > 10%:")
        for col, pct in high_nan_cols.head(20).items():
            print(f"      - {col}: {pct:.1f}%")
        if len(high_nan_cols) > 20:
            print(f"      ... 还有 {len(high_nan_cols) - 20} 个")
    else:
        print("    ✓ 没有特征 NaN 比例 > 10%")

    total_nan_pct = (df.isna().sum().sum() / df.size * 100)
    print(f"    总体 NaN 比例: {total_nan_pct:.2f}%")

    # 2. 无穷大值检测
    print("\n[5] 无穷大值检测...")
    inf_counts = {}
    for col in df.columns:
        col_data = df[col]
        if col_data.dtype in [np.float64, np.float32]:
            n_inf = np.isinf(col_data).sum()
            if n_inf > 0:
                inf_counts[col] = n_inf

    if inf_counts:
        print(f"    ⚠ {len(inf_counts)} 个特征包含无穷大值:")
        for col, count in sorted(inf_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"      - {col}: {count} 个无穷大值")
    else:
        print("    ✓ 没有检测到无穷大值")

    # 3. 极端异常值检测（超过均值±10倍标准差）
    print("\n[6] 极端异常值检测 (|x - mean| > 10 * std)...")
    extreme_outliers = {}
    for col in df.columns:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            mean = col_data.mean()
            std = col_data.std()
            if std > 0:
                n_extreme = ((col_data - mean).abs() > 10 * std).sum()
                if n_extreme > 0:
                    extreme_outliers[col] = {
                        'count': n_extreme,
                        'pct': n_extreme / len(col_data) * 100,
                        'mean': mean,
                        'std': std,
                        'min': col_data.min(),
                        'max': col_data.max()
                    }

    if extreme_outliers:
        # 按异常值数量排序
        sorted_outliers = sorted(extreme_outliers.items(), key=lambda x: -x[1]['count'])
        print(f"    ⚠ {len(extreme_outliers)} 个特征包含极端异常值:")
        for col, info in sorted_outliers[:20]:
            print(f"      - {col}: {info['count']} 个 ({info['pct']:.2f}%)")
            print(f"        范围: [{info['min']:.4f}, {info['max']:.4f}], mean={info['mean']:.4f}, std={info['std']:.4f}")
    else:
        print("    ✓ 没有检测到极端异常值")

    # 4. 常量列检测（方差接近0）
    print("\n[7] 常量列检测 (std < 1e-10)...")
    constant_cols = []
    for col in df.columns:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            std = col_data.std()
            if std < 1e-10:
                constant_cols.append((col, col_data.iloc[0] if len(col_data) > 0 else None))

    if constant_cols:
        print(f"    ⚠ {len(constant_cols)} 个常量列:")
        for col, val in constant_cols[:20]:
            print(f"      - {col}: 值 = {val}")
    else:
        print("    ✓ 没有检测到常量列")

    # 5. 数据范围统计
    print("\n[8] 数据范围统计...")
    stats = df.describe().T
    stats['range'] = stats['max'] - stats['min']

    # 范围极大的特征
    large_range = stats[stats['range'] > 1000].sort_values('range', ascending=False)
    if len(large_range) > 0:
        print(f"    ⚠ {len(large_range)} 个特征范围 > 1000:")
        for col in large_range.head(10).index:
            row = large_range.loc[col]
            print(f"      - {col}: [{row['min']:.2f}, {row['max']:.2f}], range={row['range']:.2f}")
    else:
        print("    ✓ 所有特征范围 <= 1000")

    # 6. 按股票统计数据完整性
    print("\n[9] 按股票统计数据完整性...")
    if isinstance(df.index, pd.MultiIndex):
        # 获取每只股票的样本数
        instrument_counts = df.groupby(level='instrument').size()

        # 获取每只股票的 NaN 比例
        instrument_nan_pct = df.groupby(level='instrument').apply(
            lambda x: x.isna().sum().sum() / x.size * 100
        )

        print(f"    股票数量: {len(instrument_counts)}")
        print(f"    每只股票样本数: min={instrument_counts.min()}, max={instrument_counts.max()}, median={instrument_counts.median():.0f}")

        # 样本数少的股票
        low_sample_stocks = instrument_counts[instrument_counts < 100]
        if len(low_sample_stocks) > 0:
            print(f"    ⚠ {len(low_sample_stocks)} 只股票样本数 < 100:")
            for stock, count in low_sample_stocks.head(10).items():
                print(f"      - {stock}: {count} 个样本")

        # NaN 比例高的股票
        high_nan_stocks = instrument_nan_pct[instrument_nan_pct > 50]
        if len(high_nan_stocks) > 0:
            print(f"    ⚠ {len(high_nan_stocks)} 只股票 NaN 比例 > 50%:")
            for stock, pct in high_nan_stocks.sort_values(ascending=False).head(10).items():
                print(f"      - {stock}: {pct:.1f}%")

    # ==================== 标签异常值检查 ====================
    print("\n" + "=" * 70)
    print("标签异常值分析")
    print("=" * 70)

    label_col = train_labels.columns[0] if len(train_labels.columns) > 0 else None
    if label_col:
        label_data = train_labels[label_col].dropna()

        print(f"\n[10] 标签统计 ({label_col})...")
        print(f"    样本数: {len(label_data)}")
        print(f"    NaN 数: {train_labels[label_col].isna().sum()} ({train_labels[label_col].isna().mean()*100:.2f}%)")
        print(f"    均值: {label_data.mean():.6f}")
        print(f"    标准差: {label_data.std():.6f}")
        print(f"    中位数: {label_data.median():.6f}")
        print(f"    范围: [{label_data.min():.6f}, {label_data.max():.6f}]")

        # 标签极端值
        mean = label_data.mean()
        std = label_data.std()
        n_extreme = ((label_data - mean).abs() > 5 * std).sum()
        print(f"    极端值 (|x - mean| > 5*std): {n_extreme} ({n_extreme/len(label_data)*100:.2f}%)")

        # 标签分位数
        print(f"\n    分位数:")
        for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
            print(f"      {q*100:.0f}%: {label_data.quantile(q):.6f}")

    # ==================== 特征列表 ====================
    print("\n" + "=" * 70)
    print("特征列表")
    print("=" * 70)

    print(f"\n总共 {len(df.columns)} 个特征:")

    # 分类显示
    alpha158_features = [c for c in df.columns if not c.startswith('TALIB_')]
    talib_features = [c for c in df.columns if c.startswith('TALIB_')]

    print(f"\n  Alpha158 特征 ({len(alpha158_features)} 个):")
    for i, col in enumerate(alpha158_features[:20]):
        print(f"    {i+1}. {col}")
    if len(alpha158_features) > 20:
        print(f"    ... 还有 {len(alpha158_features) - 20} 个")

    print(f"\n  TA-Lib 特征 ({len(talib_features)} 个):")
    for i, col in enumerate(talib_features):
        print(f"    {i+1}. {col}")

    # ==================== 总结 ====================
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)

    issues = []
    if len(high_nan_cols) > 0:
        issues.append(f"{len(high_nan_cols)} 个特征 NaN > 10%")
    if inf_counts:
        issues.append(f"{len(inf_counts)} 个特征有无穷大值")
    if extreme_outliers:
        issues.append(f"{len(extreme_outliers)} 个特征有极端异常值")
    if constant_cols:
        issues.append(f"{len(constant_cols)} 个常量列")

    if issues:
        print("\n⚠ 发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ 数据质量良好，未发现严重问题")

    print("\n" + "=" * 70)
    print("检查完成")
    print("=" * 70)


if __name__ == "__main__":
    check_anomalies()
