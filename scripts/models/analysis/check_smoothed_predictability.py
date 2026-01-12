"""
检查平滑后的收益率可预测性

比较不同平滑方法对价格可预测性的影响：
1. 原始单日收益率
2. N日累积收益率（当前方法）
3. 移动平均价格变化率
4. 指数平滑收益率

使用方法:
    python scripts/models/analysis/check_smoothed_predictability.py
    python scripts/models/analysis/check_smoothed_predictability.py --stock-pool sp100
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import qlib
from qlib.constant import REG_US
from qlib.data import D

from data.stock_pools import STOCK_POOLS
from utils.talib_ops import TALIB_OPS
from models.common.config import PROJECT_ROOT, QLIB_DATA_PATH

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')


def parse_args():
    parser = argparse.ArgumentParser(description='检查平滑后的收益率可预测性')
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--start', type=str, default='2015-01-01')
    parser.add_argument('--end', type=str, default='2025-12-31')
    parser.add_argument('--windows', type=int, nargs='+', default=[3, 5, 7, 10, 20],
                        help='平滑窗口列表')
    return parser.parse_args()


def load_price_data(symbols, start_time, end_time):
    """加载价格数据"""
    print(f"    加载 {len(symbols)} 只股票的数据...")

    df = D.features(
        symbols,
        ["$close", "$volume"],
        start_time=start_time,
        end_time=end_time,
        freq="day"
    )
    df.columns = ["close", "volume"]
    df = df.dropna()

    print(f"    加载完成: {len(df)} 条记录")
    return df


def compute_smoothed_labels(df, windows):
    """
    计算各种平滑方法的标签

    返回字典: {label_name: series}
    """
    close = df['close']
    labels = {}

    # 1. 原始单日收益率
    labels['raw_1d'] = close.groupby(level='instrument').pct_change().shift(-1)

    for w in windows:
        # 2. N日累积收益率（当前方法）
        # 从今天到N天后的累积收益
        labels[f'cumret_{w}d'] = close.groupby(level='instrument').pct_change(periods=w).shift(-w)

        # 3. 移动平均价格变化率
        # 未来N日均价 vs 当前价格
        ma_future = close.groupby(level='instrument').apply(
            lambda x: x.rolling(w).mean().shift(-w)
        ).droplevel(0)
        labels[f'ma{w}_change'] = ma_future / close - 1

        # 4. 指数移动平均变化率 (EMA)
        # 使用 span=w 的 EMA
        ema_future = close.groupby(level='instrument').apply(
            lambda x: x.ewm(span=w, adjust=False).mean().shift(-w)
        ).droplevel(0)
        labels[f'ema{w}_change'] = ema_future / close - 1

        # 5. 未来N日均价 vs 历史N日均价
        ma_hist = close.groupby(level='instrument').apply(
            lambda x: x.rolling(w).mean()
        ).droplevel(0)
        labels[f'ma{w}_vs_ma{w}'] = ma_future / ma_hist - 1

    return labels


def compute_features(df, windows):
    """计算用于预测的特征"""
    close = df['close']
    volume = df['volume']
    features = pd.DataFrame(index=df.index)

    # 动量特征
    for lag in [1, 3, 5, 10, 20]:
        features[f'mom_{lag}d'] = close.groupby(level='instrument').pct_change(periods=lag)

    # 波动率特征
    for w in [5, 10, 20]:
        features[f'vol_{w}d'] = close.groupby(level='instrument').apply(
            lambda x: x.pct_change().rolling(w).std()
        ).droplevel(0)

    # 价格位置
    for w in [10, 20]:
        features[f'price_loc_{w}d'] = close.groupby(level='instrument').apply(
            lambda x: (x - x.rolling(w).min()) / (x.rolling(w).max() - x.rolling(w).min() + 1e-10)
        ).droplevel(0)

    # 成交量比率
    for w in [5, 10]:
        features[f'vol_ratio_{w}d'] = volume.groupby(level='instrument').apply(
            lambda x: x / x.rolling(w).mean()
        ).droplevel(0)

    # MA 偏离度
    for w in [5, 10, 20]:
        ma = close.groupby(level='instrument').apply(
            lambda x: x.rolling(w).mean()
        ).droplevel(0)
        features[f'ma{w}_dev'] = close / ma - 1

    return features


def compute_cross_sectional_ic(features, label, min_stocks=10):
    """计算横截面IC"""
    merged = features.copy()
    merged['label'] = label
    merged = merged.dropna()

    ic_results = {}

    for feat in features.columns:
        def daily_ic(group):
            feat_vals = group[feat].dropna()
            label_vals = group['label'].loc[feat_vals.index]
            if len(feat_vals) < min_stocks:
                return np.nan
            return stats.spearmanr(feat_vals, label_vals)[0]

        ic_series = merged.groupby(level='datetime').apply(daily_ic)
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()

        ic_results[feat] = {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': ic_mean / ic_std if ic_std > 0 else 0,
        }

    return pd.DataFrame(ic_results).T


def analyze_label_properties(labels):
    """分析各种标签的统计特性"""
    results = []

    for name, label in labels.items():
        label_clean = label.dropna()

        # 自相关
        label_df = label_clean.reset_index()
        label_df.columns = ['datetime', 'instrument', 'label']
        autocorr_1 = label_df.groupby('instrument')['label'].apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan
        ).mean()

        results.append({
            'label': name,
            'mean': label_clean.mean(),
            'std': label_clean.std(),
            'skew': stats.skew(label_clean),
            'kurtosis': stats.kurtosis(label_clean),
            'autocorr_1': autocorr_1,
            'count': len(label_clean),
        })

    return pd.DataFrame(results)


def analyze_predictability(features, labels, min_stocks=10):
    """分析各种标签的可预测性"""
    results = []

    for label_name, label in labels.items():
        ic_df = compute_cross_sectional_ic(features, label, min_stocks)

        # 找最佳特征
        best_feat = ic_df['icir'].abs().idxmax()
        best_icir = ic_df.loc[best_feat, 'icir']
        best_ic = ic_df.loc[best_feat, 'ic_mean']

        # 计算平均 ICIR
        avg_icir = ic_df['icir'].abs().mean()

        results.append({
            'label': label_name,
            'best_feature': best_feat,
            'best_icir': best_icir,
            'best_ic_mean': best_ic,
            'avg_icir': avg_icir,
            'features_with_icir_gt_005': (ic_df['icir'].abs() > 0.05).sum(),
        })

    return pd.DataFrame(results)


def plot_comparison(label_props, pred_results, output_dir):
    """绘制对比图"""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 自相关对比
    ax = axes[0, 0]
    labels = label_props['label'].values
    autocorrs = label_props['autocorr_1'].values
    colors = ['green' if abs(a) < 0.1 else 'orange' if abs(a) < 0.5 else 'red' for a in autocorrs]
    bars = ax.barh(range(len(labels)), autocorrs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(0.1, color='green', linestyle='--', alpha=0.5)
    ax.axvline(-0.1, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('1-Lag Autocorrelation')
    ax.set_title('Label Autocorrelation\n(closer to 0 = less look-ahead bias concern)')

    # 2. 最佳 ICIR 对比
    ax = axes[0, 1]
    merged = pred_results.merge(label_props[['label', 'autocorr_1']], on='label')
    labels = merged['label'].values
    icirs = merged['best_icir'].abs().values
    colors = ['green' if i > 0.1 else 'orange' if i > 0.05 else 'red' for i in icirs]
    ax.barh(range(len(labels)), icirs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0.05, color='orange', linestyle='--', alpha=0.5, label='Weak signal')
    ax.axvline(0.1, color='green', linestyle='--', alpha=0.5, label='Strong signal')
    ax.set_xlabel('Best |ICIR|')
    ax.set_title('Predictability by Label Type\n(higher = more predictable)')
    ax.legend(fontsize=8)

    # 3. 标准差对比
    ax = axes[1, 0]
    stds = label_props['std'].values * 100  # 转为百分比
    ax.barh(range(len(labels)), stds, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(label_props['label'].values, fontsize=8)
    ax.set_xlabel('Standard Deviation (%)')
    ax.set_title('Label Volatility\n(signal magnitude)')

    # 4. ICIR vs Autocorrelation 散点图
    ax = axes[1, 1]
    x = merged['autocorr_1'].abs().values
    y = merged['best_icir'].abs().values
    ax.scatter(x, y, s=100, alpha=0.7, c='steelblue', edgecolors='black')

    # 标注点
    for i, label in enumerate(merged['label'].values):
        ax.annotate(label, (x[i], y[i]), fontsize=7, ha='left', va='bottom')

    ax.set_xlabel('|Autocorrelation|')
    ax.set_ylabel('Best |ICIR|')
    ax.set_title('Predictability vs Autocorrelation\n(ideal: low autocorr, high ICIR)')
    ax.axhline(0.05, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(0.1, color='red', linestyle='--', alpha=0.5)

    # 高亮理想区域
    ax.fill_between([0, 0.1], [0.05, 0.05], [0.3, 0.3], alpha=0.1, color='green', label='Ideal zone')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'smoothed_comparison.png', dpi=150)
    print(f"    ✓ Saved: smoothed_comparison.png")
    plt.close()


def print_summary(label_props, pred_results):
    """打印分析总结"""

    print("\n" + "=" * 90)
    print("                         平滑方法可预测性对比分析")
    print("=" * 90)

    # 合并结果
    merged = pred_results.merge(label_props[['label', 'autocorr_1', 'std']], on='label')
    merged = merged.sort_values('best_icir', key=abs, ascending=False)

    print("\n[1] 各标签类型的可预测性排名")
    print("-" * 90)
    print(f"{'Label':<20} {'Best ICIR':>12} {'Best Feature':<20} {'Autocorr':>12} {'Std%':>10}")
    print("-" * 90)

    for _, row in merged.iterrows():
        # 标记理想的标签（低自相关 + 高ICIR）
        marker = ""
        if abs(row['autocorr_1']) < 0.2 and abs(row['best_icir']) > 0.05:
            marker = " ★"
        elif abs(row['best_icir']) > 0.1:
            marker = " ◆"

        print(f"{row['label']:<20} {row['best_icir']:>12.4f} {row['best_feature']:<20} "
              f"{row['autocorr_1']:>12.4f} {row['std']*100:>9.2f}%{marker}")

    print("-" * 90)
    print("  ★ = 推荐 (低自相关 + 可预测)    ◆ = 高可预测性但需注意自相关")

    # 分析结论
    print("\n" + "=" * 90)
    print("                              分析结论")
    print("=" * 90)

    # 找出最佳标签
    best_row = merged.iloc[0]

    print(f"""
  1. 可预测性最强的标签: {best_row['label']}
     - Best ICIR: {best_row['best_icir']:.4f}
     - Best Feature: {best_row['best_feature']}
     - Autocorrelation: {best_row['autocorr_1']:.4f}
    """)

    # 分析平滑效果
    raw_icir = merged[merged['label'] == 'raw_1d']['best_icir'].abs().values[0]

    print("  2. 平滑效果分析:")
    for _, row in merged.iterrows():
        if row['label'] != 'raw_1d':
            improvement = (abs(row['best_icir']) - raw_icir) / raw_icir * 100
            if improvement > 10:
                print(f"     ✓ {row['label']}: ICIR 提升 {improvement:.1f}%")
            elif improvement < -10:
                print(f"     ✗ {row['label']}: ICIR 下降 {-improvement:.1f}%")

    # 推荐
    print("\n  3. 推荐:")

    # 找低自相关 + 高ICIR的标签
    good_labels = merged[(merged['autocorr_1'].abs() < 0.3) & (merged['best_icir'].abs() > 0.05)]
    if len(good_labels) > 0:
        best_good = good_labels.iloc[0]
        print(f"     推荐使用: {best_good['label']}")
        print(f"     - 自相关较低 ({best_good['autocorr_1']:.4f})，前视偏差风险小")
        print(f"     - 可预测性较好 (ICIR={best_good['best_icir']:.4f})")
    else:
        print("     没有找到同时满足低自相关和高ICIR的标签")
        print("     建议使用累积收益率 (cumret_Nd) 作为标签")

    print("""
  4. 关于前视偏差的说明:
     - 自相关高不一定意味着前视偏差，但需要警惕
     - MA/EMA 变化率的高自相关是因为平滑窗口重叠
     - 累积收益率 (cumret) 的高自相关也是窗口重叠导致
     - 关键是：只要标签计算只用未来数据，特征只用历史数据，就没有前视偏差
    """)


def main():
    args = parse_args()

    print("=" * 90)
    print("              平滑方法对价格可预测性的影响分析")
    print("=" * 90)
    print(f"\n配置:")
    print(f"  股票池: {args.stock_pool}")
    print(f"  平滑窗口: {args.windows}")

    # 初始化 Qlib
    print("\n[1] 初始化 Qlib...")
    qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US, custom_ops=TALIB_OPS)

    # 获取股票池
    symbols = STOCK_POOLS[args.stock_pool]
    print(f"    股票数量: {len(symbols)}")

    # 加载数据
    print("\n[2] 加载价格数据...")
    df = load_price_data(symbols, args.start, args.end)

    # 计算各种平滑标签
    print("\n[3] 计算各种平滑标签...")
    labels = compute_smoothed_labels(df, args.windows)
    print(f"    生成 {len(labels)} 种标签")

    # 计算特征
    print("\n[4] 计算预测特征...")
    features = compute_features(df, args.windows)
    print(f"    生成 {len(features.columns)} 个特征")

    # 分析标签特性
    print("\n[5] 分析标签统计特性...")
    label_props = analyze_label_properties(labels)

    # 分析可预测性
    print("\n[6] 分析各标签的可预测性...")
    pred_results = analyze_predictability(features, labels)

    # 生成可视化
    print("\n[7] 生成可视化...")
    output_dir = PROJECT_ROOT / "outputs" / "smoothed_predictability"
    plot_comparison(label_props, pred_results, output_dir)

    # 打印总结
    print_summary(label_props, pred_results)

    print("\n" + "=" * 90)
    print("✓ 分析完成!")
    print("=" * 90)


if __name__ == "__main__":
    main()
