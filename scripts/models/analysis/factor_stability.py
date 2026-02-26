"""
因子稳定性分析 (Factor Stability Analysis)

对指定 handler 的所有因子计算:
1. 滚动窗口 IC (Rolling IC) — 60 个交易日窗口
2. Regime 条件 IC — 按 VIX 高/中/低分组
3. 训练期 vs 测试期 IC 衰减
4. 综合稳定性评分

输出: 终端报告 + CSV + 可视化

Usage:
    python scripts/models/analysis/factor_stability.py --stock-pool sp500 --handler alpha158-talib-macro
    python scripts/models/analysis/factor_stability.py --stock-pool sp500 --handler alpha158-talib-macro --rolling-window 120
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from utils.talib_ops import TALIB_OPS
from models.common.handlers import get_handler_config
from models.common.config import DEFAULT_TIME_SPLITS
from data.stock_pools import STOCK_POOLS

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
QLIB_DATA_PATH = PROJECT_ROOT / "my_data" / "qlib_us"
OUTPUT_DIR = PROJECT_ROOT / "results" / "factor_stability"


def parse_args():
    parser = argparse.ArgumentParser(description="Factor Stability Analysis")
    parser.add_argument("--stock-pool", type=str, default="sp500",
                        choices=list(STOCK_POOLS.keys()))
    parser.add_argument("--handler", type=str, default="alpha158-talib-macro")
    parser.add_argument("--nday", type=int, default=5, help="Label horizon (days)")
    parser.add_argument("--rolling-window", type=int, default=60,
                        help="Rolling IC window size (trading days)")
    parser.add_argument("--min-stocks", type=int, default=30,
                        help="Minimum stocks per day for IC calculation")
    parser.add_argument("--top-n", type=int, default=30,
                        help="Show top N factors in report")
    return parser.parse_args()


def compute_daily_ic(features_df, label_series, min_stocks=30):
    """
    计算每日横截面 Rank IC (Spearman) — 向量化实现

    按特征分批，每个特征用 groupby.corr 一次算完所有日期
    Returns: DataFrame (dates × features)
    """
    feature_cols = features_df.columns.tolist()
    n_features = len(feature_cols)
    dates = features_df.index.get_level_values('datetime').unique()
    print(f"    Computing daily IC for {n_features} features across {len(dates)} days...")

    # 过滤掉股票数不足的日期
    stocks_per_day = features_df.groupby(level='datetime').size()
    valid_dates = stocks_per_day[stocks_per_day >= min_stocks].index

    # 对 label 做 groupby rank (一次完成)
    label_rank = label_series.loc[valid_dates].groupby(level='datetime').rank()

    ic_results = {}
    batch_size = 20
    for batch_start in range(0, n_features, batch_size):
        batch_cols = feature_cols[batch_start:batch_start + batch_size]
        print(f"      Features {batch_start+1}-{min(batch_start+batch_size, n_features)}/{n_features}...")

        for col in batch_cols:
            col_data = features_df[col].loc[valid_dates]
            # groupby rank
            feat_rank = col_data.groupby(level='datetime').rank()
            # 合并 rank
            combined = pd.DataFrame({'feat': feat_rank, 'label': label_rank}).dropna()
            if len(combined) == 0:
                continue
            # groupby corr: Pearson(rank, rank) = Spearman
            ic_series = combined.groupby(level='datetime').apply(
                lambda g: g['feat'].corr(g['label']) if len(g) >= min_stocks else np.nan
            ).dropna()
            if len(ic_series) > 0:
                ic_results[col] = ic_series

    result = pd.DataFrame(ic_results).sort_index()
    print(f"    Done: {result.shape[0]} days × {result.shape[1]} features with valid IC")
    return result


def compute_macro_timeseries_ic(features_df, label_series, macro_cols, window=60):
    """
    对 macro 特征计算时间序列 IC (而非横截面 IC)

    Macro 特征对同一天所有股票相同 → 没有横截面变异。
    改用: 每天的 macro 值 vs 当天市场平均 return 的时间序列相关性。
    """
    if not macro_cols:
        return None

    print(f"    Computing time-series IC for {len(macro_cols)} macro features...")

    # 每天的市场平均 return
    daily_avg_return = label_series.groupby(level='datetime').mean()

    # 每天的 macro 值 (取第一只股票，所有股票相同)
    macro_daily = features_df[macro_cols].groupby(level='datetime').first()

    # 对齐
    common_dates = daily_avg_return.index.intersection(macro_daily.index)
    avg_ret = daily_avg_return.loc[common_dates]
    macro_vals = macro_daily.loc[common_dates]

    # 滚动时间序列相关
    results = {}
    for col in macro_cols:
        series = macro_vals[col].dropna()
        ret_aligned = avg_ret.loc[series.index]
        if len(series) < window:
            continue

        # 全局 Spearman
        overall_ic = series.rank().corr(ret_aligned.rank())

        # 滚动 IC
        combined = pd.DataFrame({'feat': series, 'ret': ret_aligned})
        rolling_corr = combined['feat'].rolling(window).corr(combined['ret'])

        results[col] = {
            'ts_ic': overall_ic,
            'ts_ic_std': rolling_corr.std(),
            'ts_icir': overall_ic / (rolling_corr.std() + 1e-12),
            'ts_sign_ratio': (rolling_corr > 0).mean(),
            'ts_n_days': len(series),
        }

    if results:
        return pd.DataFrame(results).T
    return None


def compute_rolling_ic_stats(daily_ic_df, window=60):
    """
    从每日 IC 计算滚动均值和滚动 ICIR

    Returns: (rolling_mean_df, rolling_icir_df)
    """
    rolling_mean = daily_ic_df.rolling(window=window, min_periods=window // 2).mean()
    rolling_std = daily_ic_df.rolling(window=window, min_periods=window // 2).std()
    rolling_icir = rolling_mean / (rolling_std + 1e-12)
    return rolling_mean, rolling_icir


def compute_regime_ic(daily_ic_df, vix_series):
    """
    按 VIX regime 分组计算条件 IC

    vix_series: raw VIX level per date (不需要 instrument 维度，macro 对所有股票相同)
    Returns: DataFrame with columns [low_ic, mid_ic, high_ic, low_icir, mid_icir, high_icir]
    """
    # VIX 分位数阈值
    vix_q33 = vix_series.quantile(0.33)
    vix_q67 = vix_series.quantile(0.67)

    regime_map = pd.Series('mid', index=vix_series.index)
    regime_map[vix_series <= vix_q33] = 'low'
    regime_map[vix_series >= vix_q67] = 'high'

    common_dates = daily_ic_df.index.intersection(regime_map.index)
    ic_aligned = daily_ic_df.loc[common_dates]
    regime_aligned = regime_map.loc[common_dates]

    results = {}
    for regime in ['low', 'mid', 'high']:
        mask = regime_aligned == regime
        regime_ic = ic_aligned[mask]
        n_days = mask.sum()
        mean_ic = regime_ic.mean()
        std_ic = regime_ic.std()
        icir = mean_ic / (std_ic + 1e-12)
        results[f'{regime}_ic'] = mean_ic
        results[f'{regime}_icir'] = icir
        results[f'{regime}_n'] = pd.Series(n_days, index=daily_ic_df.columns)

    return pd.DataFrame(results)


def compute_stability_score(daily_ic_df, train_end, test_start):
    """
    综合稳定性评分:
    - ic_mean: 全局平均 IC
    - icir: IC / std(IC)
    - sign_ratio: IC > 0 的天数比例
    - ic_decay: train IC - test IC (越小越稳定)
    - rolling_ic_std: 滚动 IC 均值的标准差 (越小越稳定)
    """
    train_mask = daily_ic_df.index <= pd.Timestamp(train_end)
    test_mask = daily_ic_df.index >= pd.Timestamp(test_start)

    train_ic = daily_ic_df[train_mask]
    test_ic = daily_ic_df[test_mask]

    result = pd.DataFrame({
        'ic_mean': daily_ic_df.mean(),
        'ic_std': daily_ic_df.std(),
        'icir': daily_ic_df.mean() / (daily_ic_df.std() + 1e-12),
        'sign_ratio': (daily_ic_df > 0).mean(),
        'train_ic': train_ic.mean(),
        'test_ic': test_ic.mean(),
        'ic_decay': train_ic.mean() - test_ic.mean(),
        'ic_decay_pct': ((train_ic.mean() - test_ic.mean()) / (train_ic.mean().abs() + 1e-12)) * 100,
    })

    # 综合稳定性评分: 高 ICIR + 低衰减 + 高 sign_ratio → 好
    # 归一化后加权
    icir_norm = result['icir'].rank(pct=True)
    sign_norm = result['sign_ratio'].rank(pct=True)
    decay_norm = 1 - result['ic_decay'].abs().rank(pct=True)  # 衰减小 → 分数高
    result['stability_score'] = (0.4 * icir_norm + 0.3 * sign_norm + 0.3 * decay_norm)

    return result


def extract_vix_level(features_df):
    """
    从特征中提取每日 VIX level (macro 特征对同一天所有股票相同)
    """
    vix_col = None
    for col in features_df.columns:
        # Qlib columns can be tuples like ('feature', 'macro_vix_level') or plain strings
        col_str = str(col).lower()
        if 'vix_level' in col_str:
            vix_col = col
            break

    if vix_col is None:
        # 尝试 vix_zscore20 作为替代
        for col in features_df.columns:
            col_str = str(col).lower()
            if 'vix_zscore' in col_str:
                vix_col = col
                break

    if vix_col is None:
        print("    ⚠ No VIX feature found, skipping regime analysis")
        print(f"    Available columns (first 10): {list(features_df.columns[:10])}")
        return None

    # 取每天第一只股票的 VIX (所有股票相同)
    vix_daily = features_df[vix_col].groupby(level='datetime').first()
    vix_daily = vix_daily.dropna()
    print(f"    VIX feature: {vix_col}, range: [{vix_daily.min():.2f}, {vix_daily.max():.2f}], "
          f"days: {len(vix_daily)}")
    return vix_daily


def plot_rolling_ic_heatmap(rolling_ic_mean, top_factors, output_path):
    """滚动 IC 热力图: x=时间, y=因子"""
    data = rolling_ic_mean[top_factors].dropna(how='all')
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(18, max(8, len(top_factors) * 0.35)))
    im = ax.pcolormesh(data.index, range(len(top_factors)), data.T.values,
                       cmap='RdYlGn', vmin=-0.05, vmax=0.05, shading='auto')
    ax.set_yticks(range(len(top_factors)))
    ax.set_yticklabels(top_factors, fontsize=7)
    ax.set_xlabel('Date')
    ax.set_title('Rolling IC Heatmap (60-day window)')
    plt.colorbar(im, ax=ax, label='Rolling IC')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"    Saved: {output_path}")


def plot_regime_comparison(regime_df, top_factors, output_path):
    """Regime 条件 IC 对比柱状图"""
    data = regime_df.loc[top_factors]
    fig, ax = plt.subplots(figsize=(14, max(6, len(top_factors) * 0.3)))

    y = np.arange(len(top_factors))
    h = 0.25
    ax.barh(y - h, data['low_ic'], h, label=f"Low VIX (n={int(data['low_n'].iloc[0])})", color='#2ecc71', alpha=0.8)
    ax.barh(y, data['mid_ic'], h, label=f"Mid VIX (n={int(data['mid_n'].iloc[0])})", color='#3498db', alpha=0.8)
    ax.barh(y + h, data['high_ic'], h, label=f"High VIX (n={int(data['high_n'].iloc[0])})", color='#e74c3c', alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(top_factors, fontsize=7)
    ax.set_xlabel('Mean IC')
    ax.set_title('Factor IC by VIX Regime')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"    Saved: {output_path}")


def plot_ic_decay(stability_df, top_factors, output_path):
    """训练期 vs 测试期 IC 对比"""
    data = stability_df.loc[top_factors]
    fig, ax = plt.subplots(figsize=(14, max(6, len(top_factors) * 0.3)))

    y = np.arange(len(top_factors))
    h = 0.35
    ax.barh(y - h / 2, data['train_ic'], h, label='Train IC (2000-2022)', color='#3498db', alpha=0.8)
    ax.barh(y + h / 2, data['test_ic'], h, label='Test IC (2024-2025)', color='#e74c3c', alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(top_factors, fontsize=7)
    ax.set_xlabel('Mean IC')
    ax.set_title('Factor IC: Train vs Test Period')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"    Saved: {output_path}")


def categorize_factor(name):
    """将因子分类"""
    name_str = str(name).lower()
    if name_str.startswith('macro_'):
        if 'vix' in name_str or 'uvxy' in name_str or 'svxy' in name_str:
            return 'Macro-VIX'
        elif any(s in name_str for s in ['xlk', 'xlf', 'xle', 'xlv', 'xli', 'xlp', 'xly', 'xlu', 'xlre', 'xlb', 'xlc']):
            return 'Macro-Sector'
        elif any(s in name_str for s in ['yield', 'spread', 'credit', 'hy_', 'ig_']):
            return 'Macro-Rates'
        else:
            return 'Macro-Other'
    elif 'talib' in name_str:
        return 'TA-Lib'
    else:
        return 'Alpha158'


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("Factor Stability Analysis")
    print("=" * 70)
    print(f"Handler:        {args.handler}")
    print(f"Stock Pool:     {args.stock_pool}")
    print(f"Label Horizon:  {args.nday} days")
    print(f"Rolling Window: {args.rolling_window} days")

    # ---- 1. Init Qlib ----
    print("\n[1] Initializing Qlib...")
    qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US,
              custom_ops=TALIB_OPS, kernels=1)

    # ---- 2. Load data ----
    print("\n[2] Loading data...")
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = DEFAULT_TIME_SPLITS.copy()
    handler_config = get_handler_config(args.handler)
    HandlerClass = handler_config['class']

    handler_kwargs = {
        'volatility_window': args.nday,
        'instruments': symbols,
        'start_time': time_splits['train_start'],
        'end_time': time_splits['test_end'],
        'fit_start_time': time_splits['train_start'],
        'fit_end_time': time_splits['train_end'],
        'infer_processors': [],
    }
    if 'default_kwargs' in handler_config:
        for k, v in handler_config['default_kwargs'].items():
            if k not in handler_kwargs:
                handler_kwargs[k] = v

    handler = HandlerClass(**handler_kwargs)
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (time_splits['train_start'], time_splits['train_end']),
            "valid": (time_splits['valid_start'], time_splits['valid_end']),
            "test": (time_splits['test_start'], time_splits['test_end']),
        }
    )

    # 加载所有数据 (train+valid+test)
    feat_list = dataset.prepare(
        ["train", "valid", "test"], col_set="feature", data_key=DataHandlerLP.DK_L
    )
    label_list = dataset.prepare(
        ["train", "valid", "test"], col_set="label", data_key=DataHandlerLP.DK_L
    )
    all_features = pd.concat(feat_list) if isinstance(feat_list, list) else feat_list
    all_labels = pd.concat(label_list) if isinstance(label_list, list) else label_list
    # 去重 (valid/test 可能有重叠)
    all_features = all_features[~all_features.index.duplicated(keep='first')]
    all_labels = all_labels[~all_labels.index.duplicated(keep='first')]
    label_col = all_labels.columns[0]
    all_label_series = all_labels[label_col]

    # 过滤无效列
    valid_cols = []
    for col in all_features.columns:
        col_data = all_features[col]
        if col_data.isna().all() or col_data.nunique(dropna=True) <= 1:
            continue
        valid_cols.append(col)
    all_features = all_features[valid_cols]

    print(f"    Features: {len(valid_cols)}, Samples: {len(all_features)}")
    print(f"    Date range: {all_features.index.get_level_values('datetime').min().date()} "
          f"to {all_features.index.get_level_values('datetime').max().date()}")
    # Debug: print column format
    macro_cols = [c for c in valid_cols if 'macro' in str(c).lower() or 'vix' in str(c).lower()]
    print(f"    Macro/VIX columns ({len(macro_cols)}): {macro_cols[:5]}...")

    # 分离 stock-level 特征和 macro 特征
    stock_cols = [c for c in valid_cols if 'macro' not in str(c).lower()]
    print(f"    Stock-level features: {len(stock_cols)}, Macro features: {len(macro_cols)}")

    # ---- 3. Daily IC (stock features only — macro 无横截面变异) ----
    print("\n[3] Computing daily cross-sectional IC (Spearman) for stock features...")
    daily_ic = compute_daily_ic(all_features[stock_cols], all_label_series, min_stocks=args.min_stocks)
    print(f"    Done: {daily_ic.shape[0]} days × {daily_ic.shape[1]} features")

    # ---- 3b. Macro 时间序列 IC ----
    macro_ts_ic = None
    if macro_cols:
        print("\n[3b] Computing time-series IC for macro features...")
        macro_ts_ic = compute_macro_timeseries_ic(
            all_features, all_label_series, macro_cols, window=args.rolling_window
        )

    # ---- 4. Rolling IC ----
    print(f"\n[4] Computing rolling IC (window={args.rolling_window})...")
    rolling_mean, rolling_icir = compute_rolling_ic_stats(daily_ic, window=args.rolling_window)

    # ---- 5. Regime IC ----
    print("\n[5] Computing VIX regime-conditional IC...")
    # 提取 VIX: 先尝试从 z-scored 特征找, 如果没有则直接读 macro CSV
    vix_daily = extract_vix_level(all_features)
    if vix_daily is None:
        # 直接从 macro parquet 文件读
        macro_path = PROJECT_ROOT / "my_data" / "macro_processed" / "macro_features.parquet"
        if macro_path.exists():
            print(f"    Loading VIX from {macro_path}...")
            macro_df = pd.read_parquet(macro_path)
            # 确保 index 是日期
            if 'date' in macro_df.columns:
                macro_df = macro_df.set_index('date')
            macro_df.index = pd.to_datetime(macro_df.index)
            if 'macro_vix_level' in macro_df.columns:
                vix_daily = macro_df['macro_vix_level'].dropna()
                print(f"    VIX loaded: range [{vix_daily.min():.2f}, {vix_daily.max():.2f}], "
                      f"days: {len(vix_daily)}")
    regime_df = None
    if vix_daily is not None:
        regime_df = compute_regime_ic(daily_ic, vix_daily)
        print(f"    VIX thresholds: q33={vix_daily.quantile(0.33):.2f}, q67={vix_daily.quantile(0.67):.2f}")

    # ---- 6. Stability Score ----
    print("\n[6] Computing stability scores...")
    stability = compute_stability_score(
        daily_ic,
        train_end=time_splits['train_end'],
        test_start=time_splits['test_start']
    )
    stability['category'] = [categorize_factor(f) for f in stability.index]

    # 合并 regime 信息
    if regime_df is not None:
        stability = stability.join(regime_df[['low_ic', 'mid_ic', 'high_ic']])
        # regime 一致性: 三个 regime IC 符号相同
        stability['regime_consistent'] = (
            (stability['low_ic'] > 0) == (stability['mid_ic'] > 0)
        ) & (
            (stability['mid_ic'] > 0) == (stability['high_ic'] > 0)
        )

    # ---- 7. Output ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 排序
    stability = stability.sort_values('stability_score', ascending=False)
    top_factors = stability.index[:args.top_n].tolist()

    # CSV
    csv_path = OUTPUT_DIR / f"factor_stability_{args.handler}_{args.stock_pool}_{timestamp}.csv"
    stability.to_csv(csv_path, float_format='%.4f')
    print(f"\n    Full results saved: {csv_path}")

    # 终端报告
    print("\n" + "=" * 70)
    print(f"FACTOR STABILITY REPORT — {args.handler} / {args.stock_pool}")
    print("=" * 70)

    # 概览
    print(f"\nTotal factors analyzed: {len(stability)}")
    print(f"Date range: {daily_ic.index.min().date()} to {daily_ic.index.max().date()}")
    print(f"Train: {time_splits['train_start']} ~ {time_splits['train_end']}")
    print(f"Test:  {time_splits['test_start']} ~ {time_splits['test_end']}")

    # 按类别汇总
    print(f"\n--- IC by Category ---")
    cat_summary = stability.groupby('category').agg({
        'ic_mean': 'mean',
        'icir': 'mean',
        'sign_ratio': 'mean',
        'ic_decay_pct': 'mean',
        'stability_score': 'mean',
    }).sort_values('stability_score', ascending=False)
    cat_summary.columns = ['Mean IC', 'Mean ICIR', 'Sign%', 'Decay%', 'Stability']
    cat_count = stability.groupby('category').size()
    cat_summary.insert(0, 'Count', cat_count)
    print(cat_summary.to_string(float_format=lambda x: f"{x:.4f}"))

    # Top 稳定因子
    print(f"\n--- Top {args.top_n} Most Stable Factors ---")
    display_cols = ['category', 'ic_mean', 'icir', 'sign_ratio', 'train_ic', 'test_ic', 'ic_decay_pct', 'stability_score']
    if regime_df is not None:
        display_cols.extend(['low_ic', 'mid_ic', 'high_ic', 'regime_consistent'])
    top_df = stability[display_cols].head(args.top_n)
    pd.set_option('display.max_colwidth', 30)
    pd.set_option('display.width', 200)
    print(top_df.to_string(float_format=lambda x: f"{x:.4f}"))

    # 最不稳定因子 (衰减最大)
    print(f"\n--- Top 10 Most Decayed Factors (Train→Test) ---")
    decayed = stability.sort_values('ic_decay_pct', ascending=False).head(10)
    print(decayed[['category', 'train_ic', 'test_ic', 'ic_decay_pct']].to_string(
        float_format=lambda x: f"{x:.4f}"))

    # Regime 翻转因子 (不一致)
    if regime_df is not None:
        inconsistent = stability[~stability['regime_consistent']].sort_values('ic_mean', key=abs, ascending=False)
        print(f"\n--- Regime-Inconsistent Factors ({len(inconsistent)} total, top 10) ---")
        print("(IC sign flips between VIX regimes — unreliable)")
        print(inconsistent[['category', 'low_ic', 'mid_ic', 'high_ic', 'ic_mean']].head(10).to_string(
            float_format=lambda x: f"{x:.4f}"))

    # Macro 时间序列 IC
    if macro_ts_ic is not None:
        print(f"\n--- Macro Features Time-Series IC ({len(macro_ts_ic)} features) ---")
        print("(Macro features have no cross-sectional variation; using time-series correlation instead)")
        macro_ts_ic['category'] = [categorize_factor(f) for f in macro_ts_ic.index]
        macro_ts_sorted = macro_ts_ic.sort_values('ts_icir', key=abs, ascending=False)
        print(macro_ts_sorted[['category', 'ts_ic', 'ts_icir', 'ts_sign_ratio', 'ts_n_days']].head(30).to_string(
            float_format=lambda x: f"{x:.4f}"))

        # Save macro IC too
        macro_csv = OUTPUT_DIR / f"macro_ts_ic_{args.handler}_{args.stock_pool}_{timestamp}.csv"
        macro_ts_sorted.to_csv(macro_csv, float_format='%.4f')
        print(f"\n    Macro IC saved: {macro_csv}")

    # 可视化
    print(f"\n[7] Generating plots...")
    plot_rolling_ic_heatmap(
        rolling_mean, top_factors,
        OUTPUT_DIR / f"rolling_ic_heatmap_{args.stock_pool}_{timestamp}.png"
    )
    if regime_df is not None:
        plot_regime_comparison(
            regime_df, top_factors,
            OUTPUT_DIR / f"regime_ic_{args.stock_pool}_{timestamp}.png"
        )
    plot_ic_decay(
        stability, top_factors,
        OUTPUT_DIR / f"ic_decay_{args.stock_pool}_{timestamp}.png"
    )

    # 推荐
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    robust_factors = stability[
        (stability['stability_score'] >= 0.7) &
        (stability['ic_mean'].abs() > 0.005) &
        (stability.get('regime_consistent', True) == True)
    ]
    print(f"\nRobust factors (score >= 0.7, |IC| > 0.005, regime-consistent): {len(robust_factors)}")
    if len(robust_factors) > 0:
        for f in robust_factors.index[:20]:
            row = robust_factors.loc[f]
            print(f"  {str(f):40s}  IC={row['ic_mean']:+.4f}  ICIR={row['icir']:+.3f}  "
                  f"Decay={row['ic_decay_pct']:+.1f}%  Score={row['stability_score']:.3f}")

    unstable_factors = stability[
        (stability['ic_decay_pct'].abs() > 100) |
        (stability['sign_ratio'] < 0.45) |
        (stability['sign_ratio'] > 0.55) & (stability['ic_mean'].abs() < 0.002)
    ]
    print(f"\nUnstable/weak factors to consider removing: {len(unstable_factors)}")

    print(f"\nDone. All outputs in: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
