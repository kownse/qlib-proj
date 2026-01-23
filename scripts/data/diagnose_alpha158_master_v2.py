"""
深入诊断 Alpha158-master 训练数据的极端值问题
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH

from data.stock_pools import STOCK_POOLS
from data.datahandler_master import Alpha158_Master


def main():
    print("\n深入诊断 Alpha158-master 极端值问题")
    print("="*70)

    # 初始化
    qlib.init(provider_uri="./my_data/qlib_us", region=REG_US)

    time_splits = {
        'train_start': "2000-01-01",
        'train_end': "2022-12-31",
        'valid_start': "2023-01-01",
        'valid_end': "2023-12-31",
        'test_start': "2024-01-01",
        'test_end': "2025-12-31",
    }

    symbols = STOCK_POOLS['sp500']

    # 创建 handler
    handler = Alpha158_Master(
        volatility_window=2,
        instruments=symbols,
        start_time=time_splits['train_start'],
        end_time=time_splits['test_end'],
        fit_start_time=time_splits['train_start'],
        fit_end_time=time_splits['train_end'],
    )

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (time_splits['train_start'], time_splits['train_end']),
            "valid": (time_splits['valid_start'], time_splits['valid_end']),
            "test": (time_splits['test_start'], time_splits['test_end']),
        }
    )

    # 获取训练数据
    print("\n[1] 检查训练数据的特征分布")
    train_data = dataset.prepare("train", col_set="feature")

    # 获取每个特征的统计信息
    feature_stats = []
    for col in train_data.columns:
        col_name = col[1] if isinstance(col, tuple) else col
        values = train_data[col].values
        valid_values = values[~np.isnan(values)]

        if len(valid_values) > 0:
            stats = {
                'name': col_name,
                'mean': np.mean(valid_values),
                'std': np.std(valid_values),
                'min': np.min(valid_values),
                'max': np.max(valid_values),
                'abs_max': np.max(np.abs(valid_values)),
                'nan_pct': np.isnan(values).mean() * 100,
            }
            feature_stats.append(stats)

    # 按绝对最大值排序
    feature_stats.sort(key=lambda x: x['abs_max'], reverse=True)

    print("\n最极端的10个特征:")
    print(f"{'特征名':<30} {'均值':>15} {'标准差':>15} {'最大值':>15} {'NaN%':>8}")
    print("-" * 90)
    for stat in feature_stats[:10]:
        print(f"{stat['name']:<30} {stat['mean']:>15.2e} {stat['std']:>15.2e} "
              f"{stat['max']:>15.2e} {stat['nan_pct']:>8.1f}")

    # 找出有极端值的样本
    print("\n[2] 检查极端值出现的位置")
    extreme_col = feature_stats[0]['name']
    print(f"\n分析最极端的特征: {extreme_col}")

    if isinstance(train_data.columns[0], tuple):
        col_idx = [c for c in train_data.columns if c[1] == extreme_col][0]
    else:
        col_idx = extreme_col

    extreme_values = train_data[col_idx]
    threshold = 1e6  # 100万以上视为极端值

    extreme_mask = np.abs(extreme_values.values) > threshold
    extreme_samples = train_data.index[extreme_mask]

    print(f"极端值样本数: {len(extreme_samples)} / {len(train_data)} "
          f"({len(extreme_samples)/len(train_data)*100:.4f}%)")

    if len(extreme_samples) > 0:
        # 分析极端值的日期和股票
        extreme_dates = extreme_samples.get_level_values(0)
        extreme_stocks = extreme_samples.get_level_values(1)

        print(f"\n极端值日期分布:")
        date_counts = pd.Series(extreme_dates).value_counts()
        print(f"  涉及 {len(date_counts)} 个不同日期")
        print(f"  日期范围: {extreme_dates.min().date()} 到 {extreme_dates.max().date()}")
        print(f"  Top 5 日期: {date_counts.head().to_dict()}")

        print(f"\n极端值股票分布:")
        stock_counts = pd.Series(extreme_stocks).value_counts()
        print(f"  涉及 {len(stock_counts)} 只不同股票")
        print(f"  Top 10 股票: {stock_counts.head(10).to_dict()}")

        # 检查是否是 CSZScoreNorm 导致的
        print("\n[3] 检查是否是标准化导致的问题")
        sample_date = extreme_dates[0]
        day_data = train_data.xs(sample_date, level=0)
        print(f"\n{sample_date.date()} 当天的 {extreme_col} 分布:")
        col_data = day_data[col_idx]
        print(f"  样本数: {len(col_data)}")
        print(f"  均值: {col_data.mean():.4f}")
        print(f"  标准差: {col_data.std():.4f}")
        print(f"  最大值: {col_data.max():.2e}")
        print(f"  最小值: {col_data.min():.2e}")

    # 检查原始数据（不经过处理器）
    print("\n[4] 检查原始数据（跳过处理器）")
    handler_raw = Alpha158_Master(
        volatility_window=2,
        instruments=symbols[:10],  # 只用10只股票加速
        start_time="2020-01-01",  # 较短时间
        end_time="2020-12-31",
        learn_processors=[],  # 不用处理器
        infer_processors=[],
    )

    raw_data = handler_raw.fetch(col_set="feature")
    print(f"\n原始数据形状: {raw_data.shape}")

    # 检查原始特征分布
    raw_stats = []
    for col in raw_data.columns:
        col_name = col[1] if isinstance(col, tuple) else col
        values = raw_data[col].values
        valid_values = values[~np.isnan(values)]

        if len(valid_values) > 0:
            raw_stats.append({
                'name': col_name,
                'mean': np.mean(valid_values),
                'std': np.std(valid_values),
                'max': np.max(valid_values),
                'abs_max': np.max(np.abs(valid_values)),
            })

    raw_stats.sort(key=lambda x: x['abs_max'], reverse=True)

    print("\n原始数据最极端的10个特征:")
    print(f"{'特征名':<30} {'均值':>15} {'标准差':>15} {'最大值':>15}")
    print("-" * 80)
    for stat in raw_stats[:10]:
        print(f"{stat['name']:<30} {stat['mean']:>15.4f} {stat['std']:>15.4f} {stat['max']:>15.4f}")

    print("\n" + "="*70)
    print("诊断结论:")
    print("="*70)
    if feature_stats[0]['abs_max'] > 1e10:
        print("""
  ❌ 发现严重的极端值问题!

  问题原因:
  - CSZScoreNorm (Cross-Sectional Z-Score Normalization) 在某些日期出现问题
  - 当某个特征在某天只有很少几只股票有值时，标准差接近0
  - 导致 z-score = (x - mean) / (std + eps) 产生极大的值

  解决方案:
  1. 使用 RobustZScoreNorm 代替 CSZScoreNorm (使用 MAD 而非标准差)
  2. 在 CSZScoreNorm 后添加 clip 处理
  3. 过滤掉极端值样本
  4. 增加 fillna 处理，确保每天有足够的样本

  建议修改 datahandler_master.py:
  - 添加 Fillna 处理器
  - 使用 CSRankNorm 代替 CSZScoreNorm
  - 或者添加 clip 后处理
""")
    else:
        print("  ✓ 原始数据正常，问题可能在处理器")


if __name__ == "__main__":
    main()
