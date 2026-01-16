"""
Permutation Importance for AE-MLP models.

计算每个特征对 AE-MLP 模型预测的真实重要性。
方法：打乱每个特征，测量 IC 的下降幅度。

Usage:
    python scripts/models/analysis/permutation_importance_ae_mlp.py \
        --model-path my_models/ae_mlp_cv_best_model.keras \
        --handler alpha158-enhanced-v7 \
        --stock-pool sp500

    # 使用验证集评估（更快）
    python scripts/models/analysis/permutation_importance_ae_mlp.py \
        --model-path my_models/ae_mlp_cv_best_model.keras \
        --handler alpha158-enhanced-v7 \
        --segment valid

    # 指定重复次数（更稳定）
    python scripts/models/analysis/permutation_importance_ae_mlp.py \
        --model-path my_models/ae_mlp_cv_best_model.keras \
        --handler alpha158-enhanced-v7 \
        --n-repeats 10
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

# Setup paths
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))
project_root = script_dir.parent

# Initialize qlib
import qlib
from qlib.constant import REG_US
from utils.talib_ops import TALIB_OPS

qlib_data_path = project_root / "my_data" / "qlib_us"
qlib.init(
    provider_uri=str(qlib_data_path),
    region=REG_US,
    custom_ops=TALIB_OPS,
)

import tensorflow as tf
from tensorflow import keras
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from models.common.handlers import get_handler_class, HANDLER_CONFIG
from data.stock_pools import STOCK_POOLS


# 默认时间配置
DEFAULT_TIME_CONFIG = {
    'train_start': '2000-01-01',
    'train_end': '2024-09-30',
    'valid_start': '2024-10-01',
    'valid_end': '2024-12-31',
    'test_start': '2025-01-01',
    'test_end': '2025-12-31',
}


def calculate_ic(pred: np.ndarray, label: np.ndarray) -> float:
    """Calculate Information Coefficient (Spearman correlation)."""
    mask = ~(np.isnan(pred) | np.isnan(label))
    if mask.sum() < 10:
        return 0.0
    corr, _ = stats.spearmanr(pred[mask], label[mask])
    return corr if not np.isnan(corr) else 0.0


def calculate_daily_ic(pred: np.ndarray, label: np.ndarray, index: pd.MultiIndex) -> float:
    """Calculate mean daily IC."""
    df = pd.DataFrame({
        'pred': pred,
        'label': label,
    }, index=index)

    daily_ic = df.groupby(level=0).apply(
        lambda x: x['pred'].corr(x['label'], method='spearman')
    )
    return daily_ic.mean()


def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    index: pd.MultiIndex,
    feature_names: list,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calculate permutation importance for each feature.

    Args:
        model: Trained Keras model
        X: Feature matrix (n_samples, n_features)
        y: Labels
        index: MultiIndex for daily IC calculation
        feature_names: List of feature names
        n_repeats: Number of times to shuffle each feature
        random_state: Random seed

    Returns:
        DataFrame with feature importance scores
    """
    np.random.seed(random_state)
    n_features = X.shape[1]

    # Baseline prediction
    print("Calculating baseline IC...")
    baseline_pred = model.predict(X, verbose=0)
    if isinstance(baseline_pred, list):
        baseline_pred = baseline_pred[-1].flatten()
    else:
        baseline_pred = baseline_pred.flatten()

    baseline_ic = calculate_daily_ic(baseline_pred, y, index)
    print(f"Baseline IC: {baseline_ic:.4f}")

    # Calculate importance for each feature
    importance_scores = []

    for i, name in enumerate(feature_names):
        ic_drops = []

        for r in range(n_repeats):
            # Shuffle feature i
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])

            # Predict with shuffled feature
            pred_permuted = model.predict(X_permuted, verbose=0)
            if isinstance(pred_permuted, list):
                pred_permuted = pred_permuted[-1].flatten()
            else:
                pred_permuted = pred_permuted.flatten()

            # Calculate IC drop
            permuted_ic = calculate_daily_ic(pred_permuted, y, index)
            ic_drop = baseline_ic - permuted_ic
            ic_drops.append(ic_drop)

        mean_drop = np.mean(ic_drops)
        std_drop = np.std(ic_drops)

        importance_scores.append({
            'feature': name,
            'importance': mean_drop,
            'importance_std': std_drop,
            'baseline_ic': baseline_ic,
        })

        # Progress
        print(f"[{i+1:3d}/{n_features}] {name:40s} IC drop: {mean_drop:+.4f} (±{std_drop:.4f})")

    df = pd.DataFrame(importance_scores)
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    return df


def create_data_handler(args, symbols, time_config):
    """创建 DataHandler"""
    HandlerClass = get_handler_class(args.handler)

    handler = HandlerClass(
        volatility_window=args.nday,
        instruments=symbols,
        start_time=time_config['train_start'],
        end_time=time_config['test_end'],
        fit_start_time=time_config['train_start'],
        fit_end_time=time_config['train_end'],
        infer_processors=[],
    )
    return handler


def create_dataset(handler, time_config):
    """创建 Dataset"""
    segments = {
        "train": (time_config['train_start'], time_config['train_end']),
        "valid": (time_config['valid_start'], time_config['valid_end']),
        "test": (time_config['test_start'], time_config['test_end']),
    }
    return DatasetH(handler=handler, segments=segments)


def prepare_data(dataset, segment):
    """准备数据"""
    features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
    feature_names = list(features.columns.get_level_values(-1))
    features = features.fillna(0).replace([np.inf, -np.inf], 0).clip(-10, 10)

    try:
        labels = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(labels, pd.DataFrame):
            labels = labels.iloc[:, 0]
        labels = labels.fillna(0).values
        return features.values, labels, features.index, feature_names
    except Exception:
        return features.values, None, features.index, feature_names


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Permutation Importance for AE-MLP',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Model parameters
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to saved .keras model')
    parser.add_argument('--handler', type=str, required=True,
                        choices=list(HANDLER_CONFIG.keys()),
                        help='Handler type (must match the model)')
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--nday', type=int, default=5,
                        help='Prediction horizon (must match the model)')

    # Evaluation parameters
    parser.add_argument('--segment', type=str, default='valid',
                        choices=['train', 'valid', 'test'],
                        help='Data segment to use for importance calculation')
    parser.add_argument('--n-repeats', type=int, default=5,
                        help='Number of times to shuffle each feature')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: auto-generated)')

    args = parser.parse_args()

    # Check model file
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = project_root / model_path

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    # Config
    symbols = STOCK_POOLS[args.stock_pool]
    time_config = DEFAULT_TIME_CONFIG.copy()

    print("=" * 70)
    print("Permutation Importance for AE-MLP")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Handler: {args.handler}")
    print(f"Stock pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Segment: {args.segment}")
    print(f"N repeats: {args.n_repeats}")
    print("=" * 70)

    # Load model
    print("\n[1] Loading model...")
    model = keras.models.load_model(str(model_path))
    print(f"    Model loaded: {model.name}")
    print(f"    Input shape: {model.input_shape}")

    # Prepare data
    print("\n[2] Preparing data...")
    handler = create_data_handler(args, symbols, time_config)
    dataset = create_dataset(handler, time_config)

    X, y, index, feature_names = prepare_data(dataset, args.segment)
    print(f"    Data shape: {X.shape}")
    print(f"    Features: {len(feature_names)}")
    print(f"    Samples: {len(X)}")

    # Check feature count
    expected_features = model.input_shape[1]
    if X.shape[1] != expected_features:
        print(f"\n    Warning: Feature count mismatch!")
        print(f"    Model expects: {expected_features}")
        print(f"    Data has: {X.shape[1]}")
        sys.exit(1)

    # Calculate permutation importance
    print("\n[3] Calculating permutation importance...")
    print("-" * 70)

    importance_df = permutation_importance(
        model=model,
        X=X,
        y=y,
        index=index,
        feature_names=feature_names,
        n_repeats=args.n_repeats,
        random_state=args.seed,
    )

    # Display results
    print("\n" + "=" * 70)
    print("PERMUTATION IMPORTANCE RESULTS")
    print("=" * 70)
    print(f"\nBaseline IC: {importance_df['baseline_ic'].iloc[0]:.4f}")
    print(f"Total features: {len(importance_df)}")
    print("-" * 70)
    print(f"\n{'Rank':<6} {'Feature':<45} {'Importance':>12} {'Std':>10}")
    print("-" * 70)

    for _, row in importance_df.iterrows():
        imp = row['importance']
        sign = '+' if imp >= 0 else ''
        print(f"{row['rank']:<6} {row['feature']:<45} {sign}{imp:>11.4f} {row['importance_std']:>10.4f}")

    print("-" * 70)

    # Statistics
    positive_imp = importance_df[importance_df['importance'] > 0]
    negative_imp = importance_df[importance_df['importance'] < 0]
    zero_imp = importance_df[importance_df['importance'].abs() < 0.001]

    print(f"\nImportance Statistics:")
    print(f"  Positive (IC drops when shuffled): {len(positive_imp)} features")
    print(f"  Negative (IC improves when shuffled): {len(negative_imp)} features")
    print(f"  Near zero (|importance| < 0.001): {len(zero_imp)} features")

    if len(positive_imp) > 0:
        print(f"\n  Top 5 most important:")
        for _, row in positive_imp.head(5).iterrows():
            print(f"    {row['feature']}: +{row['importance']:.4f}")

    if len(negative_imp) > 0:
        print(f"\n  Features that HURT the model (consider removing):")
        for _, row in negative_imp.iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")

    # Save results
    output_dir = project_root / "outputs" / "feature_importance"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"permutation_importance_{args.handler}_{timestamp}.csv"

    importance_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Also save as JSON for easy parsing
    json_path = output_path.with_suffix('.json')
    result = {
        'model_path': str(model_path),
        'handler': args.handler,
        'segment': args.segment,
        'n_repeats': args.n_repeats,
        'baseline_ic': float(importance_df['baseline_ic'].iloc[0]),
        'features': importance_df[['rank', 'feature', 'importance', 'importance_std']].to_dict('records'),
    }
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"JSON saved to: {json_path}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
