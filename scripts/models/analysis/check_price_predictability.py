"""
检查 SP500 股票价格的可预测性分析

分析不同预测周期 (1天、5天、7天) 的价格可预测性：
1. 收益率自相关分析 - 历史收益能否预测未来
2. 横截面IC分析 - 特征与未来收益的横截面相关性
3. 随机游走检验 - ADF检验价格是否遵循随机游走
4. 收益分布统计 - 均值、标准差、偏度、峰度
5. 方差比检验 - 检验价格效率

使用方法:
    python scripts/models/analysis/check_price_predictability.py
    python scripts/models/analysis/check_price_predictability.py --stock-pool sp100
    python scripts/models/analysis/check_price_predictability.py --horizons 1 3 5 10 20
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import qlib
from qlib.constant import REG_US
from qlib.data import D

from data.stock_pools import STOCK_POOLS
from utils.talib_ops import TALIB_OPS
from models.common.config import PROJECT_ROOT, QLIB_DATA_PATH

# 设置 matplotlib
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='检查 SP500 股票价格的可预测性',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python check_price_predictability.py                          # 默认 SP500, 1/5/7天
    python check_price_predictability.py --stock-pool sp100       # SP100 股池
    python check_price_predictability.py --horizons 1 3 5 10 20   # 自定义预测周期
    python check_price_predictability.py --start 2020-01-01       # 自定义时间范围
"""
    )
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'],
                        help='股票池 (default: sp500)')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 5, 7],
                        help='预测周期列表 (default: 1 5 7)')
    parser.add_argument('--start', type=str, default='2015-01-01',
                        help='开始日期 (default: 2015-01-01)')
    parser.add_argument('--end', type=str, default='2025-12-31',
                        help='结束日期 (default: 2025-12-31)')
    parser.add_argument('--max-stocks', type=int, default=None,
                        help='最大股票数量 (用于测试)')
    return parser.parse_args()


def load_price_data(symbols: List[str], start_time: str, end_time: str) -> pd.DataFrame:
    """
    加载价格数据

    Args:
        symbols: 股票代码列表
        start_time: 开始时间
        end_time: 结束时间

    Returns:
        DataFrame with columns: close, high, low, volume, indexed by (datetime, instrument)
    """
    print(f"    加载 {len(symbols)} 只股票的数据...")

    # 使用 Qlib 的 D 模块加载数据
    fields = ["$close", "$high", "$low", "$volume"]
    field_names = ["close", "high", "low", "volume"]

    df = D.features(
        symbols,
        fields,
        start_time=start_time,
        end_time=end_time,
        freq="day"
    )
    df.columns = field_names

    # 清理数据
    df = df.dropna()

    print(f"    加载完成: {len(df)} 条记录, {df.index.get_level_values('instrument').nunique()} 只股票")
    return df


def compute_returns(df: pd.DataFrame, horizons: List[int]) -> Dict[int, pd.Series]:
    """
    计算不同周期的收益率

    Args:
        df: 价格数据 DataFrame
        horizons: 预测周期列表

    Returns:
        字典 {horizon: returns_series}
    """
    returns = {}
    close = df['close']

    for h in horizons:
        # 计算 h 天后的收益率: P(t+h) / P(t) - 1
        # 使用 groupby + shift 来计算每只股票的收益率
        ret = close.groupby(level='instrument').pct_change(periods=h).shift(-h)
        returns[h] = ret.dropna()

    return returns


def analyze_autocorrelation(returns: Dict[int, pd.Series], max_lag: int = 20) -> pd.DataFrame:
    """
    分析收益率自相关性

    Args:
        returns: 收益率字典
        max_lag: 最大滞后期

    Returns:
        DataFrame with autocorrelations
    """
    results = []

    for horizon, ret in returns.items():
        # 按股票分组计算自相关
        autocorrs = {}
        for lag in [1, 2, 3, 5, 10, 20]:
            if lag <= max_lag:
                ac = ret.groupby(level='instrument').apply(
                    lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
                ).mean()
                autocorrs[lag] = ac

        results.append({
            'horizon': horizon,
            **{f'autocorr_lag{lag}': ac for lag, ac in autocorrs.items()}
        })

    return pd.DataFrame(results)


def analyze_return_distribution(returns: Dict[int, pd.Series]) -> pd.DataFrame:
    """
    分析收益率分布特征

    Args:
        returns: 收益率字典

    Returns:
        DataFrame with distribution statistics
    """
    results = []

    for horizon, ret in returns.items():
        # 计算分布统计量
        results.append({
            'horizon': horizon,
            'mean': ret.mean(),
            'std': ret.std(),
            'skewness': stats.skew(ret.dropna()),
            'kurtosis': stats.kurtosis(ret.dropna()),
            'min': ret.min(),
            'max': ret.max(),
            'median': ret.median(),
            'q25': ret.quantile(0.25),
            'q75': ret.quantile(0.75),
            'positive_ratio': (ret > 0).mean(),
            'count': len(ret)
        })

    return pd.DataFrame(results)


def adf_test(series: pd.Series, max_samples: int = 10000) -> Tuple[float, float, bool]:
    """
    Augmented Dickey-Fuller 检验

    Args:
        series: 时间序列数据
        max_samples: 最大样本数 (避免内存问题)

    Returns:
        (adf_statistic, p_value, is_stationary)
    """
    try:
        from statsmodels.tsa.stattools import adfuller

        # 采样避免内存问题
        if len(series) > max_samples:
            series = series.sample(n=max_samples, random_state=42)

        result = adfuller(series.dropna(), autolag='AIC')
        return result[0], result[1], result[1] < 0.05
    except Exception as e:
        return np.nan, np.nan, False


def variance_ratio_test(returns: pd.Series, q: int = 5) -> Tuple[float, float]:
    """
    方差比检验 - 检验随机游走假设

    如果价格遵循随机游走，VR(q) 应该接近 1

    Args:
        returns: 收益率序列
        q: 滞后期

    Returns:
        (variance_ratio, z_statistic)
    """
    try:
        ret = returns.dropna().values
        n = len(ret)

        if n < q * 2:
            return np.nan, np.nan

        # 计算方差
        var_1 = np.var(ret, ddof=1)

        # 计算 q 期累积收益的方差
        ret_q = pd.Series(ret).rolling(q).sum().dropna().values
        var_q = np.var(ret_q, ddof=1)

        # 方差比
        vr = var_q / (q * var_1)

        # z-统计量
        theta = 2 * (2 * q - 1) * (q - 1) / (3 * q * n)
        z = (vr - 1) / np.sqrt(theta)

        return vr, z
    except Exception:
        return np.nan, np.nan


def analyze_random_walk(df: pd.DataFrame, returns: Dict[int, pd.Series]) -> pd.DataFrame:
    """
    随机游走检验

    Args:
        df: 价格数据
        returns: 收益率字典

    Returns:
        DataFrame with test results
    """
    results = []

    # 对整体价格做 ADF 检验
    close = df['close']

    for horizon, ret in returns.items():
        # ADF 检验收益率 (应该是平稳的)
        adf_stat, adf_pval, is_stationary = adf_test(ret)

        # 方差比检验
        vr, z_stat = variance_ratio_test(ret, q=5)

        results.append({
            'horizon': horizon,
            'adf_statistic': adf_stat,
            'adf_pvalue': adf_pval,
            'is_stationary': is_stationary,
            'variance_ratio': vr,
            'vr_z_stat': z_stat,
            'rejects_random_walk': abs(z_stat) > 1.96 if not np.isnan(z_stat) else False
        })

    return pd.DataFrame(results)


def compute_cross_sectional_ic(df: pd.DataFrame, returns: Dict[int, pd.Series],
                                top_n_features: int = 20) -> Dict[int, pd.DataFrame]:
    """
    计算特征与未来收益的横截面IC

    Args:
        df: 价格数据 (用于计算简单特征)
        returns: 收益率字典
        top_n_features: 保留的特征数量

    Returns:
        字典 {horizon: ic_df}
    """
    # 计算简单特征
    features = pd.DataFrame(index=df.index)
    close = df['close']

    # 动量特征
    for lag in [1, 3, 5, 10, 20]:
        features[f'mom_{lag}d'] = close.groupby(level='instrument').pct_change(periods=lag)

    # 波动率特征
    for window in [5, 10, 20]:
        features[f'vol_{window}d'] = close.groupby(level='instrument').apply(
            lambda x: x.pct_change().rolling(window).std()
        ).droplevel(0)

    # 价格位置特征
    for window in [10, 20]:
        features[f'price_loc_{window}d'] = close.groupby(level='instrument').apply(
            lambda x: (x - x.rolling(window).min()) / (x.rolling(window).max() - x.rolling(window).min() + 1e-10)
        ).droplevel(0)

    # 成交量特征
    volume = df['volume']
    for window in [5, 10]:
        features[f'volume_ratio_{window}d'] = volume.groupby(level='instrument').apply(
            lambda x: x / x.rolling(window).mean()
        ).droplevel(0)

    results = {}

    for horizon, ret in returns.items():
        ic_results = {}

        # 合并特征和收益
        merged = features.copy()
        merged['return'] = ret
        merged = merged.dropna()

        for feat in features.columns:
            def daily_ic(group, feature_name=feat):
                feat_vals = group[feature_name].dropna()
                ret_vals = group['return'].loc[feat_vals.index]
                # 至少需要5只股票才能计算有意义的IC
                if len(feat_vals) < 5:
                    return np.nan
                return stats.spearmanr(feat_vals, ret_vals)[0]

            ic_series = merged.groupby(level='datetime').apply(daily_ic)
            ic_results[feat] = {
                'ic_mean': ic_series.mean(),
                'ic_std': ic_series.std(),
                'icir': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
                'ic_positive_ratio': (ic_series > 0).mean()
            }

        results[horizon] = pd.DataFrame(ic_results).T
        results[horizon] = results[horizon].sort_values('icir', key=abs, ascending=False)

    return results


def analyze_naive_predictability(returns: Dict[int, pd.Series]) -> pd.DataFrame:
    """
    分析简单预测策略的有效性

    Args:
        returns: 收益率字典

    Returns:
        DataFrame with predictability metrics
    """
    results = []

    for horizon, ret in returns.items():
        ret_df = ret.reset_index()
        ret_df.columns = ['datetime', 'instrument', 'return']

        # Naive 预测：用前一天的收益预测
        naive_pred = ret_df.groupby('instrument')['return'].shift(1)

        # 计算预测 MSE
        naive_mse = ((ret_df['return'] - naive_pred) ** 2).mean()
        baseline_mse = ret.var()  # 用均值预测的 MSE

        # 计算方向准确率 (前一天收益方向 vs 今天收益方向)
        prev_sign = np.sign(naive_pred)
        curr_sign = np.sign(ret_df['return'])
        direction_acc = (prev_sign == curr_sign).mean()

        # 计算 R²
        ss_res = ((ret_df['return'] - naive_pred) ** 2).sum()
        ss_tot = ((ret_df['return'] - ret_df['return'].mean()) ** 2).sum()
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        results.append({
            'horizon': horizon,
            'baseline_mse': baseline_mse,
            'naive_mse': naive_mse,
            'mse_improvement': (baseline_mse - naive_mse) / baseline_mse * 100 if baseline_mse > 0 else 0,
            'direction_accuracy': direction_acc,
            'r_squared': r_squared
        })

    return pd.DataFrame(results)


def plot_analysis(returns: Dict[int, pd.Series], autocorr_df: pd.DataFrame,
                  dist_df: pd.DataFrame, random_walk_df: pd.DataFrame,
                  ic_results: Dict[int, pd.DataFrame], output_dir: Path):
    """
    生成综合分析图表
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== 图1: 收益率分布 =====
    fig, axes = plt.subplots(1, len(returns), figsize=(5 * len(returns), 4))
    if len(returns) == 1:
        axes = [axes]

    for idx, (horizon, ret) in enumerate(returns.items()):
        ax = axes[idx]
        ret.hist(bins=100, ax=ax, alpha=0.7, density=True)

        # 添加正态分布拟合
        x = np.linspace(ret.min(), ret.max(), 100)
        mu, std = ret.mean(), ret.std()
        ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')

        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.axvline(mu, color='red', linestyle='--', linewidth=1, label=f'Mean: {mu:.4f}')
        ax.set_title(f'{horizon}-Day Return Distribution\n(Skew: {stats.skew(ret):.2f}, Kurt: {stats.kurtosis(ret):.2f})')
        ax.set_xlabel('Return')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'return_distribution.png', dpi=150)
    print(f"    ✓ Saved: return_distribution.png")
    plt.close()

    # ===== 图2: 自相关分析 =====
    fig, ax = plt.subplots(figsize=(10, 5))

    lags = [col.replace('autocorr_lag', '') for col in autocorr_df.columns if col.startswith('autocorr_lag')]
    x = np.arange(len(lags))
    width = 0.25

    for i, row in autocorr_df.iterrows():
        horizon = row['horizon']
        values = [row[f'autocorr_lag{lag}'] for lag in lags]
        ax.bar(x + i * width, values, width, label=f'{horizon}-Day Returns', alpha=0.8)

    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(0.05, color='red', linestyle='--', alpha=0.5, label='Significance threshold')
    ax.axhline(-0.05, color='red', linestyle='--', alpha=0.5)

    ax.set_xlabel('Lag (days)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Return Autocorrelation by Horizon')
    ax.set_xticks(x + width)
    ax.set_xticklabels(lags)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'autocorrelation.png', dpi=150)
    print(f"    ✓ Saved: autocorrelation.png")
    plt.close()

    # ===== 图3: 横截面IC分析 =====
    fig, axes = plt.subplots(1, len(ic_results), figsize=(6 * len(ic_results), 6))
    if len(ic_results) == 1:
        axes = [axes]

    for idx, (horizon, ic_df) in enumerate(ic_results.items()):
        ax = axes[idx]
        top_features = ic_df.head(10)

        colors = ['green' if x > 0 else 'red' for x in top_features['icir']]
        ax.barh(range(len(top_features)), top_features['icir'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.index)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title(f'{horizon}-Day Return Predictability\nTop 10 Features by |ICIR|')
        ax.set_xlabel('ICIR')

    plt.tight_layout()
    plt.savefig(output_dir / 'cross_sectional_ic.png', dpi=150)
    print(f"    ✓ Saved: cross_sectional_ic.png")
    plt.close()

    # ===== 图4: 综合对比 =====
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 4.1 收益率统计
    ax = axes[0, 0]
    horizons = dist_df['horizon'].values
    x = np.arange(len(horizons))
    width = 0.35
    ax.bar(x - width/2, dist_df['mean'] * 100, width, label='Mean (%)', alpha=0.8)
    ax.bar(x + width/2, dist_df['std'] * 100, width, label='Std (%)', alpha=0.8)
    ax.set_xlabel('Forecast Horizon (days)')
    ax.set_ylabel('Return (%)')
    ax.set_title('Return Statistics by Horizon')
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend()
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)

    # 4.2 方差比检验
    ax = axes[0, 1]
    ax.bar(horizons, random_walk_df['variance_ratio'], alpha=0.8, color='steelblue')
    ax.axhline(1, color='red', linestyle='--', label='Random Walk (VR=1)')
    ax.set_xlabel('Forecast Horizon (days)')
    ax.set_ylabel('Variance Ratio')
    ax.set_title('Variance Ratio Test\n(VR close to 1 = Random Walk)')
    ax.legend()

    # 4.3 方向预测准确率
    ax = axes[1, 0]
    naive_df = analyze_naive_predictability(returns)
    ax.bar(horizons, naive_df['direction_accuracy'] * 100, alpha=0.8, color='orange')
    ax.axhline(50, color='red', linestyle='--', label='Random (50%)')
    ax.set_xlabel('Forecast Horizon (days)')
    ax.set_ylabel('Direction Accuracy (%)')
    ax.set_title('Naive Prediction Direction Accuracy\n(Using previous return to predict)')
    ax.legend()
    ax.set_ylim(40, 60)

    # 4.4 最佳特征IC
    ax = axes[1, 1]
    best_icirs = []
    for h in horizons:
        if h in ic_results:
            best_icirs.append(ic_results[h]['icir'].abs().max())
        else:
            best_icirs.append(0)
    colors = ['green' if x > 0.05 else 'orange' if x > 0.02 else 'red' for x in best_icirs]
    ax.bar(horizons, best_icirs, alpha=0.8, color=colors)
    ax.axhline(0.05, color='green', linestyle='--', alpha=0.5, label='Strong signal (>0.05)')
    ax.axhline(0.02, color='orange', linestyle='--', alpha=0.5, label='Weak signal (>0.02)')
    ax.set_xlabel('Forecast Horizon (days)')
    ax.set_ylabel('Best |ICIR|')
    ax.set_title('Best Feature ICIR by Horizon')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'summary_analysis.png', dpi=150)
    print(f"    ✓ Saved: summary_analysis.png")
    plt.close()

    print(f"\n    All plots saved to: {output_dir.absolute()}")


def print_summary(horizons: List[int], autocorr_df: pd.DataFrame, dist_df: pd.DataFrame,
                  random_walk_df: pd.DataFrame, ic_results: Dict[int, pd.DataFrame],
                  naive_df: pd.DataFrame):
    """打印分析总结"""

    print("\n" + "=" * 80)
    print("                          价格可预测性分析报告")
    print("=" * 80)

    # 1. 收益率分布统计
    print("\n[1] 收益率分布统计")
    print("-" * 80)
    print(f"{'Horizon':<10} {'Mean':>10} {'Std':>10} {'Skew':>10} {'Kurt':>10} {'Up%':>10}")
    print("-" * 80)
    for _, row in dist_df.iterrows():
        print(f"{row['horizon']}-day{'':<4} {row['mean']*100:>9.3f}% {row['std']*100:>9.3f}% "
              f"{row['skewness']:>10.3f} {row['kurtosis']:>10.3f} {row['positive_ratio']*100:>9.1f}%")

    # 2. 自相关分析
    print("\n[2] 收益率自相关系数")
    print("-" * 80)
    lag_cols = [col for col in autocorr_df.columns if col.startswith('autocorr_lag')]
    header = f"{'Horizon':<10}" + "".join([f" Lag{col.replace('autocorr_lag', ''):>5}" for col in lag_cols])
    print(header)
    print("-" * 80)
    for _, row in autocorr_df.iterrows():
        line = f"{row['horizon']}-day{'':<4}"
        for col in lag_cols:
            val = row[col]
            # 标记显著的自相关
            marker = "*" if abs(val) > 0.05 else " "
            line += f" {val:>5.3f}{marker}"
        print(line)
    print("    (* 表示自相关系数绝对值 > 0.05)")

    # 3. 随机游走检验
    print("\n[3] 随机游走检验")
    print("-" * 80)
    print(f"{'Horizon':<10} {'ADF Stat':>12} {'ADF p-val':>12} {'Stationary':>12} {'VR':>10} {'VR z':>10}")
    print("-" * 80)
    for _, row in random_walk_df.iterrows():
        stationary = "Yes" if row['is_stationary'] else "No"
        rejects_rw = "Yes*" if row['rejects_random_walk'] else "No"
        print(f"{row['horizon']}-day{'':<4} {row['adf_statistic']:>12.4f} {row['adf_pvalue']:>12.4f} "
              f"{stationary:>12} {row['variance_ratio']:>10.4f} {row['vr_z_stat']:>10.4f}")
    print("    (ADF p < 0.05 = 平稳; VR 接近 1 = 随机游走; |z| > 1.96 拒绝随机游走)")

    # 4. 横截面IC分析
    print("\n[4] 横截面IC分析 (Top 5 特征)")
    print("-" * 80)
    for horizon, ic_df in ic_results.items():
        print(f"\n  {horizon}-Day Horizon:")
        print(f"  {'Feature':<25} {'IC Mean':>12} {'IC Std':>12} {'ICIR':>12} {'Up%':>10}")
        print("  " + "-" * 73)
        for feat, row in ic_df.head(5).iterrows():
            print(f"  {feat:<25} {row['ic_mean']:>12.4f} {row['ic_std']:>12.4f} "
                  f"{row['icir']:>12.4f} {row['ic_positive_ratio']*100:>9.1f}%")

    # 5. 简单预测策略
    print("\n[5] 简单预测策略效果")
    print("-" * 80)
    print(f"{'Horizon':<10} {'Baseline MSE':>14} {'Naive MSE':>14} {'Improvement':>14} {'Direction%':>12}")
    print("-" * 80)
    for _, row in naive_df.iterrows():
        print(f"{row['horizon']}-day{'':<4} {row['baseline_mse']:>14.6f} {row['naive_mse']:>14.6f} "
              f"{row['mse_improvement']:>13.2f}% {row['direction_accuracy']*100:>11.1f}%")

    # 6. 结论
    print("\n" + "=" * 80)
    print("                              分析结论")
    print("=" * 80)

    for horizon in horizons:
        print(f"\n  [{horizon}-Day Horizon]")

        # 自相关结论
        autocorr_row = autocorr_df[autocorr_df['horizon'] == horizon].iloc[0]
        lag1_ac = autocorr_row.get('autocorr_lag1', 0)
        if abs(lag1_ac) > 0.05:
            print(f"    - 自相关: 存在显著的 1 阶自相关 ({lag1_ac:.4f})，历史收益有一定预测能力")
        else:
            print(f"    - 自相关: 1 阶自相关很弱 ({lag1_ac:.4f})，历史收益难以直接预测未来")

        # 随机游走结论
        rw_row = random_walk_df[random_walk_df['horizon'] == horizon].iloc[0]
        if rw_row['rejects_random_walk']:
            print(f"    - 随机游走: 方差比检验拒绝随机游走假设 (VR={rw_row['variance_ratio']:.4f})，存在可预测性")
        else:
            print(f"    - 随机游走: 无法拒绝随机游走假设 (VR={rw_row['variance_ratio']:.4f})，市场较为有效")

        # IC 结论
        if horizon in ic_results:
            best_icir = ic_results[horizon]['icir'].abs().max()
            best_feat = ic_results[horizon]['icir'].abs().idxmax()
            if best_icir > 0.05:
                print(f"    - 横截面IC: 存在有效预测信号 (最佳 ICIR={best_icir:.4f}, 特征={best_feat})")
            elif best_icir > 0.02:
                print(f"    - 横截面IC: 存在弱预测信号 (最佳 ICIR={best_icir:.4f}, 特征={best_feat})")
            else:
                print(f"    - 横截面IC: 预测信号很弱 (最佳 ICIR={best_icir:.4f})")

    print("\n" + "=" * 80)
    print("                              建议")
    print("=" * 80)

    # 综合建议
    best_horizons = []
    for horizon in horizons:
        if horizon in ic_results:
            best_icir = ic_results[horizon]['icir'].abs().max()
            if best_icir > 0.02:
                best_horizons.append((horizon, best_icir))

    if best_horizons:
        best_horizons.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  1. 推荐预测周期: {best_horizons[0][0]} 天 (ICIR={best_horizons[0][1]:.4f})")
    else:
        print("\n  1. 所有周期的可预测性都较弱，建议:")
        print("     - 增加更多特征 (新闻情绪、宏观数据、期权隐含波动率等)")
        print("     - 尝试非线性模型 (LightGBM, Neural Networks)")

    print("\n  2. 特征工程建议:")
    print("     - 动量特征 (历史收益) 通常有一定预测能力")
    print("     - 波动率特征可以捕捉市场状态")
    print("     - 考虑加入成交量异常、价格位置等特征")

    print("\n  3. 模型选择建议:")
    print("     - 弱信号环境下，集成模型 (LightGBM, CatBoost) 通常表现更好")
    print("     - Qlib 的 TopkDropout 策略可以放大微弱的预测信号")
    print("     - 建议关注 ICIR 而非单纯的 IC，稳定性更重要")


def main():
    args = parse_args()

    print("=" * 80)
    print("              SP500 股票价格可预测性分析")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  股票池: {args.stock_pool}")
    print(f"  预测周期: {args.horizons}")
    print(f"  时间范围: {args.start} to {args.end}")

    # 初始化 Qlib
    print("\n[1] 初始化 Qlib...")
    qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US, custom_ops=TALIB_OPS)

    # 获取股票池
    symbols = STOCK_POOLS[args.stock_pool]
    if args.max_stocks:
        symbols = symbols[:args.max_stocks]
    print(f"    股票数量: {len(symbols)}")

    # 加载数据
    print("\n[2] 加载价格数据...")
    df = load_price_data(symbols, args.start, args.end)

    # 计算收益率
    print("\n[3] 计算各周期收益率...")
    returns = compute_returns(df, args.horizons)
    for h, ret in returns.items():
        print(f"    {h}-day returns: {len(ret)} samples")

    # 分析自相关
    print("\n[4] 分析收益率自相关...")
    autocorr_df = analyze_autocorrelation(returns)

    # 分析收益分布
    print("\n[5] 分析收益率分布...")
    dist_df = analyze_return_distribution(returns)

    # 随机游走检验
    print("\n[6] 随机游走检验...")
    random_walk_df = analyze_random_walk(df, returns)

    # 横截面IC分析
    print("\n[7] 计算横截面IC...")
    ic_results = compute_cross_sectional_ic(df, returns)

    # 简单预测策略
    print("\n[8] 分析简单预测策略...")
    naive_df = analyze_naive_predictability(returns)

    # 生成可视化
    print("\n[9] 生成可视化...")
    output_dir = PROJECT_ROOT / "outputs" / "price_predictability"
    plot_analysis(returns, autocorr_df, dist_df, random_walk_df, ic_results, output_dir)

    # 打印总结
    print_summary(args.horizons, autocorr_df, dist_df, random_walk_df, ic_results, naive_df)

    print("\n" + "=" * 80)
    print("✓ 分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
