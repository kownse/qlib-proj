"""
打印 CatBoost 模型的特征重要性

Usage:
    python scripts/models/analysis/print_catboost_feature_importance.py \
        --model-path my_models/catboost_cv_alpha158-talib-macro_sp500_5d_20260114_232511.cbm
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor


def load_model_and_meta(model_path: Path):
    """加载模型和元数据"""
    # 加载模型
    model = CatBoostRegressor()
    model.load_model(str(model_path))

    # 加载元数据
    meta_path = model_path.with_suffix('.meta.pkl')
    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
    else:
        meta = None

    return model, meta


def print_feature_importance(model_path: Path, top_n: int = 0, output_csv: str = None):
    """打印特征重要性"""
    model, meta = load_model_and_meta(model_path)

    # 获取特征重要性
    importance = model.get_feature_importance()

    # 获取特征名
    if meta and 'feature_names' in meta:
        feature_names = meta['feature_names']
    elif model.feature_names_:
        feature_names = model.feature_names_
    else:
        feature_names = [f"feature_{i}" for i in range(len(importance))]

    # 创建 DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    # 添加排名
    df.index = df.index + 1
    df.index.name = 'rank'

    # 打印模型信息
    print("=" * 80)
    print(f"Model: {model_path.name}")
    print("=" * 80)

    if meta:
        print(f"Handler:    {meta.get('handler', 'N/A')}")
        print(f"Stock Pool: {meta.get('stock_pool', 'N/A')}")
        print(f"N-day:      {meta.get('nday', 'N/A')}")
        print(f"Train:      {meta.get('train_start', 'N/A')} ~ {meta.get('train_end', 'N/A')}")
        if 'cv_results' in meta:
            cv = meta['cv_results']
            print(f"CV Mean IC: {cv.get('mean_ic', 'N/A'):.4f} (±{cv.get('std_ic', 'N/A'):.4f})")

    print(f"\nTotal features: {len(df)}")
    print("-" * 80)

    # 打印特征重要性
    display_df = df.head(top_n) if top_n > 0 else df

    print(f"\n{'Rank':<6} {'Feature':<50} {'Importance':>12}")
    print("-" * 70)

    for rank, row in display_df.iterrows():
        print(f"{rank:<6} {row['feature']:<50} {row['importance']:>12.4f}")

    print("-" * 70)

    # 统计信息
    print(f"\nImportance Statistics:")
    print(f"  Total:  {df['importance'].sum():.2f}")
    print(f"  Mean:   {df['importance'].mean():.4f}")
    print(f"  Median: {df['importance'].median():.4f}")
    print(f"  Max:    {df['importance'].max():.4f}")
    print(f"  Min:    {df['importance'].min():.4f}")

    # Top 10 累计占比
    top10_pct = df.head(10)['importance'].sum() / df['importance'].sum() * 100
    top20_pct = df.head(20)['importance'].sum() / df['importance'].sum() * 100
    top50_pct = df.head(50)['importance'].sum() / df['importance'].sum() * 100
    print(f"\nCumulative importance:")
    print(f"  Top 10:  {top10_pct:.1f}%")
    print(f"  Top 20:  {top20_pct:.1f}%")
    print(f"  Top 50:  {top50_pct:.1f}%")

    # 保存到 CSV
    if output_csv:
        df.to_csv(output_csv)
        print(f"\nSaved to: {output_csv}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Print CatBoost feature importance')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the CatBoost model (.cbm file)')
    parser.add_argument('--top-n', type=int, default=0,
                        help='Only show top N features (0 = show all)')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Save results to CSV file')

    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return

    print_feature_importance(model_path, args.top_n, args.output_csv)


if __name__ == "__main__":
    main()
