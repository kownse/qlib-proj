"""
诊断 Alpha158-master 数据集质量问题

检查内容：
1. 市场信息特征文件质量 (master_market_info.parquet)
2. Alpha158 特征质量
3. 合并后数据的对齐和完整性
4. 特征与标签的相关性（IC）
5. 训练/验证/测试集的分布差异
6. 潜在的数据泄露问题

Usage:
    python scripts/data/diagnose_alpha158_master.py
    python scripts/data/diagnose_alpha158_master.py --stock-pool sp500
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

# Qlib imports
import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH


PROJECT_ROOT = script_dir.parent
MARKET_INFO_PATH = PROJECT_ROOT / "my_data" / "macro_processed" / "master_market_info.parquet"


def print_section(title: str):
    """打印分隔符"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def check_market_info_file():
    """检查市场信息文件质量"""
    print_section("1. 检查市场信息文件 (master_market_info.parquet)")

    if not MARKET_INFO_PATH.exists():
        print(f"  ❌ 文件不存在: {MARKET_INFO_PATH}")
        print("     请先运行: python scripts/data/process_master_market_info.py")
        return None

    df = pd.read_parquet(MARKET_INFO_PATH)
    print(f"  ✓ 文件加载成功")
    print(f"    形状: {df.shape}")
    print(f"    日期范围: {df.index.min().date()} 到 {df.index.max().date()}")
    print(f"    总天数: {len(df)}")

    # 检查 NaN
    nan_counts = df.isna().sum()
    nan_pct = df.isna().mean() * 100

    print(f"\n  NaN 检查:")
    if nan_counts.sum() > 0:
        print(f"    ⚠ 存在 NaN 值:")
        for col in nan_counts[nan_counts > 0].index:
            print(f"      {col}: {nan_counts[col]} ({nan_pct[col]:.2f}%)")
    else:
        print(f"    ✓ 无 NaN 值")

    # 检查零值比例
    zero_pct = (df == 0).mean() * 100
    print(f"\n  零值检查:")
    for col in zero_pct[zero_pct > 50].index[:5]:  # 只显示前5个
        print(f"    ⚠ {col}: {zero_pct[col]:.1f}% 为零")

    # 检查统计分布
    print(f"\n  特征分布摘要 (前10个特征):")
    print(f"    {'特征名':<30} {'均值':>10} {'标准差':>10} {'最小':>10} {'最大':>10}")
    print(f"    {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for col in df.columns[:10]:
        stats_row = df[col]
        print(f"    {col:<30} {stats_row.mean():>10.4f} {stats_row.std():>10.4f} "
              f"{stats_row.min():>10.4f} {stats_row.max():>10.4f}")

    # 检查日期连续性
    expected_dates = pd.bdate_range(df.index.min(), df.index.max())
    missing_dates = set(expected_dates) - set(df.index)
    print(f"\n  日期连续性:")
    print(f"    预期交易日: {len(expected_dates)}")
    print(f"    实际交易日: {len(df)}")
    print(f"    缺失交易日: {len(missing_dates)}")
    if len(missing_dates) > 0 and len(missing_dates) < 20:
        print(f"    缺失日期示例: {sorted(list(missing_dates))[:5]}")

    return df


def check_handler_data(stock_pool: str, time_splits: dict):
    """检查 DataHandler 加载的数据"""
    print_section("2. 检查 Alpha158_Master DataHandler 数据")

    from data.stock_pools import STOCK_POOLS
    from data.datahandler_master import Alpha158_Master

    symbols = STOCK_POOLS.get(stock_pool, STOCK_POOLS['test'])
    print(f"  Stock pool: {stock_pool} ({len(symbols)} symbols)")

    # 创建 handler
    handler = Alpha158_Master(
        volatility_window=2,
        instruments=symbols,
        start_time=time_splits['train_start'],
        end_time=time_splits['test_end'],
        fit_start_time=time_splits['train_start'],
        fit_end_time=time_splits['train_end'],
    )

    # 获取数据
    learn_df = handler.fetch(col_set="feature")
    label_df = handler.fetch(col_set="label")

    print(f"\n  特征数据:")
    print(f"    形状: {learn_df.shape}")
    print(f"    日期范围: {learn_df.index.get_level_values(0).min().date()} 到 "
          f"{learn_df.index.get_level_values(0).max().date()}")

    # 分离 Alpha158 和市场信息特征
    feature_cols = learn_df.columns.tolist()
    if isinstance(feature_cols[0], tuple):
        feature_cols = [c[1] if isinstance(c, tuple) else c for c in feature_cols]

    market_cols = [c for c in feature_cols if c.startswith('mkt_')]
    stock_cols = [c for c in feature_cols if not c.startswith('mkt_')]

    print(f"\n  特征分解:")
    print(f"    Alpha158 股票特征: {len(stock_cols)}")
    print(f"    市场信息特征: {len(market_cols)}")
    print(f"    总特征数: {len(feature_cols)}")

    # NaN 检查
    nan_pct_by_col = learn_df.isna().mean() * 100
    total_nan_pct = learn_df.isna().mean().mean() * 100

    print(f"\n  NaN 检查:")
    print(f"    总体 NaN 比例: {total_nan_pct:.2f}%")

    # 分别统计
    if isinstance(learn_df.columns, pd.MultiIndex):
        stock_nan = learn_df[[c for c in learn_df.columns if not str(c[1]).startswith('mkt_')]].isna().mean().mean() * 100
        market_nan = learn_df[[c for c in learn_df.columns if str(c[1]).startswith('mkt_')]].isna().mean().mean() * 100
    else:
        stock_nan = learn_df[stock_cols].isna().mean().mean() * 100
        market_nan = learn_df[market_cols].isna().mean().mean() * 100 if market_cols else 0

    print(f"    股票特征 NaN: {stock_nan:.2f}%")
    print(f"    市场特征 NaN: {market_nan:.2f}%")

    # 高 NaN 特征
    high_nan_cols = nan_pct_by_col[nan_pct_by_col > 10]
    if len(high_nan_cols) > 0:
        print(f"\n    ⚠ 高 NaN 比例特征 (>10%):")
        for col, pct in high_nan_cols.head(10).items():
            col_name = col[1] if isinstance(col, tuple) else col
            print(f"      {col_name}: {pct:.1f}%")

    return handler, learn_df, label_df


def check_feature_label_correlation(learn_df: pd.DataFrame, label_df: pd.DataFrame):
    """检查特征与标签的相关性（IC）"""
    print_section("3. 检查特征与标签的相关性 (IC)")

    # 将数据对齐
    common_idx = learn_df.index.intersection(label_df.index)
    X = learn_df.loc[common_idx].values
    y = label_df.loc[common_idx].values.flatten()

    # 移除 NaN
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"  有效样本数: {len(y)}")
    print(f"  标签分布: mean={y.mean():.4f}, std={y.std():.4f}, "
          f"min={y.min():.4f}, max={y.max():.4f}")

    # 计算每个特征与标签的 IC
    feature_cols = learn_df.columns.tolist()
    ics = []

    for i, col in enumerate(feature_cols):
        col_name = col[1] if isinstance(col, tuple) else col
        x = X[:, i]
        valid = ~np.isnan(x)
        if valid.sum() > 100:
            ic, _ = stats.spearmanr(x[valid], y[valid])
            ics.append((col_name, ic))

    # 排序
    ics.sort(key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)

    # 分离市场和股票特征的 IC
    market_ics = [(n, ic) for n, ic in ics if n.startswith('mkt_')]
    stock_ics = [(n, ic) for n, ic in ics if not n.startswith('mkt_')]

    print(f"\n  股票特征 IC (Top 10):")
    print(f"    {'特征名':<40} {'IC':>10}")
    print(f"    {'-'*40} {'-'*10}")
    for name, ic in stock_ics[:10]:
        print(f"    {name:<40} {ic:>10.4f}")

    print(f"\n  市场信息特征 IC (Top 10):")
    print(f"    {'特征名':<40} {'IC':>10}")
    print(f"    {'-'*40} {'-'*10}")
    for name, ic in market_ics[:10]:
        print(f"    {name:<40} {ic:>10.4f}")

    # 统计
    stock_ic_values = [ic for _, ic in stock_ics if not np.isnan(ic)]
    market_ic_values = [ic for _, ic in market_ics if not np.isnan(ic)]

    print(f"\n  IC 统计:")
    print(f"    股票特征平均 |IC|: {np.mean(np.abs(stock_ic_values)):.4f}")
    print(f"    市场特征平均 |IC|: {np.mean(np.abs(market_ic_values)):.4f}")
    print(f"    股票特征 IC 显著 (|IC|>0.02): {sum(1 for ic in stock_ic_values if abs(ic)>0.02)}/{len(stock_ic_values)}")
    print(f"    市场特征 IC 显著 (|IC|>0.02): {sum(1 for ic in market_ic_values if abs(ic)>0.02)}/{len(market_ic_values)}")

    return ics


def check_train_valid_test_distribution(handler, time_splits: dict):
    """检查训练/验证/测试集的分布差异"""
    print_section("4. 检查数据集分布差异")

    from qlib.data.dataset import DatasetH

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (time_splits['train_start'], time_splits['train_end']),
            "valid": (time_splits['valid_start'], time_splits['valid_end']),
            "test": (time_splits['test_start'], time_splits['test_end']),
        }
    )

    for segment in ["train", "valid", "test"]:
        X = dataset.prepare(segment, col_set="feature")
        y = dataset.prepare(segment, col_set="label")

        y_values = y.values.flatten()
        valid_y = y_values[~np.isnan(y_values)]

        print(f"\n  {segment.upper()}:")
        print(f"    样本数: {len(X)}")
        print(f"    日期范围: {X.index.get_level_values(0).min().date()} 到 "
              f"{X.index.get_level_values(0).max().date()}")
        print(f"    NaN 比例: {X.isna().mean().mean()*100:.2f}%")
        print(f"    标签分布: mean={valid_y.mean():.4f}, std={valid_y.std():.4f}")

        # 检查特征统计
        feature_means = X.mean()
        feature_stds = X.std()
        print(f"    特征均值范围: [{feature_means.min():.4f}, {feature_means.max():.4f}]")
        print(f"    特征标准差范围: [{feature_stds.min():.4f}, {feature_stds.max():.4f}]")

    return dataset


def check_market_info_alignment(learn_df: pd.DataFrame, market_info_df: pd.DataFrame):
    """检查市场信息与股票数据的对齐"""
    print_section("5. 检查市场信息对齐")

    # 获取股票数据的日期
    stock_dates = learn_df.index.get_level_values(0).unique()
    market_dates = market_info_df.index

    # 检查覆盖
    covered = stock_dates.isin(market_dates)
    coverage_pct = covered.mean() * 100

    print(f"  股票数据日期数: {len(stock_dates)}")
    print(f"  市场信息日期数: {len(market_dates)}")
    print(f"  覆盖率: {coverage_pct:.1f}%")

    missing_dates = stock_dates[~covered]
    if len(missing_dates) > 0:
        print(f"  ⚠ 缺失市场信息的日期: {len(missing_dates)} 天")
        if len(missing_dates) < 10:
            print(f"    日期: {missing_dates.tolist()}")
        else:
            print(f"    示例: {missing_dates[:5].tolist()}")

    # 检查市场信息特征是否正确添加
    feature_cols = learn_df.columns.tolist()
    if isinstance(feature_cols[0], tuple):
        feature_cols = [c[1] for c in feature_cols]

    market_cols = [c for c in feature_cols if c.startswith('mkt_')]

    if len(market_cols) == 0:
        print(f"  ❌ 数据中没有市场信息特征！")
        return

    print(f"  市场信息特征数: {len(market_cols)}")

    # 检查同一天内不同股票的市场信息是否一致
    sample_date = stock_dates[len(stock_dates)//2]  # 中间日期
    day_data = learn_df.xs(sample_date, level=0)

    if isinstance(learn_df.columns, pd.MultiIndex):
        market_data = day_data[[c for c in day_data.columns if str(c[1]).startswith('mkt_')]]
    else:
        market_data = day_data[market_cols]

    # 检查是否所有股票的市场信息相同
    unique_values = market_data.drop_duplicates()
    if len(unique_values) == 1:
        print(f"  ✓ 同一天内所有股票的市场信息一致")
    else:
        print(f"  ⚠ 同一天内市场信息不一致，有 {len(unique_values)} 种不同值")


def check_potential_data_leakage(learn_df: pd.DataFrame, label_df: pd.DataFrame):
    """检查潜在的数据泄露问题"""
    print_section("6. 检查潜在数据泄露")

    # 获取时间索引
    dates = learn_df.index.get_level_values(0).unique().sort_values()

    print(f"  检查方法: 使用未来数据计算当前特征的 IC")
    print(f"  如果使用 t+1 标签计算的 IC 远高于 t 标签，可能存在数据泄露")

    # 获取一个中间时间段
    mid_start = dates[len(dates)//3]
    mid_end = dates[2*len(dates)//3]

    X = learn_df.loc[mid_start:mid_end]
    y = label_df.loc[mid_start:mid_end]

    # 计算同期 IC
    common_idx = X.index.intersection(y.index)
    X_aligned = X.loc[common_idx].values
    y_aligned = y.loc[common_idx].values.flatten()

    valid = ~np.isnan(y_aligned)

    # 计算几个特征的 IC
    feature_cols = learn_df.columns.tolist()
    sample_features = feature_cols[:10]  # 前10个特征

    print(f"\n  抽样检查前10个特征:")
    print(f"    {'特征':<40} {'同期IC':>10} {'滞后1天IC':>10}")
    print(f"    {'-'*40} {'-'*10} {'-'*10}")

    for i, col in enumerate(sample_features):
        col_name = col[1] if isinstance(col, tuple) else col
        x = X_aligned[:, i]
        x_valid = x[valid]
        y_valid = y_aligned[valid]

        # 同期 IC
        ic_same, _ = stats.spearmanr(x_valid, y_valid)

        # 滞后 IC (使用前一天的特征预测当天的标签)
        if len(x_valid) > 1:
            ic_lag, _ = stats.spearmanr(x_valid[:-1], y_valid[1:])
        else:
            ic_lag = np.nan

        print(f"    {col_name:<40} {ic_same:>10.4f} {ic_lag:>10.4f}")


def check_feature_variance(learn_df: pd.DataFrame):
    """检查特征方差"""
    print_section("7. 检查特征方差和常数特征")

    variances = learn_df.var()

    # 低方差特征
    low_var_threshold = 1e-6
    low_var_cols = variances[variances < low_var_threshold]

    if len(low_var_cols) > 0:
        print(f"  ⚠ 发现 {len(low_var_cols)} 个低方差特征 (var < {low_var_threshold}):")
        for col, var in low_var_cols.head(10).items():
            col_name = col[1] if isinstance(col, tuple) else col
            print(f"    {col_name}: var={var:.2e}")
    else:
        print(f"  ✓ 所有特征方差正常")

    # 检查常数特征（标准差接近0）
    stds = learn_df.std()
    const_cols = stds[stds < 1e-8]
    if len(const_cols) > 0:
        print(f"\n  ⚠ 发现 {len(const_cols)} 个常数特征:")
        for col in const_cols.index[:5]:
            col_name = col[1] if isinstance(col, tuple) else col
            print(f"    {col_name}")


def check_feature_correlation(learn_df: pd.DataFrame):
    """检查特征间相关性"""
    print_section("8. 检查特征间相关性")

    # 采样以加速计算
    sample_size = min(50000, len(learn_df))
    sample_idx = np.random.choice(len(learn_df), sample_size, replace=False)
    X_sample = learn_df.iloc[sample_idx].values

    # 移除 NaN
    valid_rows = ~np.isnan(X_sample).any(axis=1)
    X_clean = X_sample[valid_rows]

    if len(X_clean) < 1000:
        print(f"  ⚠ 有效样本太少 ({len(X_clean)}), 跳过相关性检查")
        return

    # 计算相关矩阵
    corr_matrix = np.corrcoef(X_clean.T)

    # 找高相关特征对
    feature_cols = learn_df.columns.tolist()
    n_features = len(feature_cols)

    high_corr_pairs = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            if not np.isnan(corr_matrix[i, j]) and abs(corr_matrix[i, j]) > 0.95:
                col_i = feature_cols[i][1] if isinstance(feature_cols[i], tuple) else feature_cols[i]
                col_j = feature_cols[j][1] if isinstance(feature_cols[j], tuple) else feature_cols[j]
                high_corr_pairs.append((col_i, col_j, corr_matrix[i, j]))

    if len(high_corr_pairs) > 0:
        print(f"  发现 {len(high_corr_pairs)} 对高度相关特征 (|r| > 0.95):")
        for i, (col1, col2, corr) in enumerate(high_corr_pairs[:10]):
            print(f"    {col1} <-> {col2}: {corr:.4f}")
        if len(high_corr_pairs) > 10:
            print(f"    ... 还有 {len(high_corr_pairs) - 10} 对")
    else:
        print(f"  ✓ 无高度冗余特征")


def main():
    parser = argparse.ArgumentParser(description="诊断 Alpha158-master 数据集")
    parser.add_argument('--stock-pool', type=str, default='test',
                        choices=['test', 'tech', 'sp100', 'sp500'],
                        help='Stock pool for testing (default: test)')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  Alpha158-master 数据集诊断工具")
    print("="*70)

    # 时间配置
    time_splits = {
        'train_start': "2000-01-01",
        'train_end': "2022-12-31",
        'valid_start': "2023-01-01",
        'valid_end': "2023-12-31",
        'test_start': "2024-01-01",
        'test_end': "2025-12-31",
    }

    # 初始化 Qlib
    print("\n初始化 Qlib...")
    qlib.init(
        provider_uri="./my_data/qlib_us",
        region=REG_US,
    )

    # 1. 检查市场信息文件
    market_info_df = check_market_info_file()

    # 2. 检查 handler 数据
    handler, learn_df, label_df = check_handler_data(args.stock_pool, time_splits)

    # 3. 检查特征与标签相关性
    ics = check_feature_label_correlation(learn_df, label_df)

    # 4. 检查数据集分布
    dataset = check_train_valid_test_distribution(handler, time_splits)

    # 5. 检查市场信息对齐
    if market_info_df is not None:
        check_market_info_alignment(learn_df, market_info_df)

    # 6. 检查数据泄露
    check_potential_data_leakage(learn_df, label_df)

    # 7. 检查特征方差
    check_feature_variance(learn_df)

    # 8. 检查特征相关性
    check_feature_correlation(learn_df)

    # 总结
    print_section("诊断总结")

    print("""
  基于以上分析，可能的问题包括:

  1. IC 过低问题:
     - 如果市场特征 IC 接近 0，说明市场信息对个股预测帮助不大
     - 可能需要检查 MASTER 模型的门控机制是否正确使用了市场信息

  2. NaN 问题:
     - 高 NaN 比例会影响模型训练
     - 考虑使用更好的缺失值处理策略

  3. 分布偏移:
     - 训练/验证/测试集标签分布差异大会影响泛化
     - 2024-2025 市场可能与 2000-2022 有显著差异

  4. 特征质量:
     - 低方差或常数特征不提供信息
     - 高度相关特征存在冗余

  建议:
  - 对比单独使用 Alpha158 handler 的效果
  - 检查 MASTER 模型的市场门控机制实现
  - 尝试不同的市场信息特征组合
""")


if __name__ == "__main__":
    main()
