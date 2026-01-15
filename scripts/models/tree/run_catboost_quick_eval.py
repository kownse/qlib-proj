"""
CatBoost 快速评估脚本 - 使用固定超参数

用于快速迭代特征工程:
1. 使用已知的好超参数，跳过超参数搜索
2. 运行 CV 训练，输出 CV Mean IC
3. 输出特征重要性报告

使用方法:
    # 使用默认超参数
    python scripts/models/tree/run_catboost_quick_eval.py --handler alpha158-enhanced-v3

    # 指定超参数
    python scripts/models/tree/run_catboost_quick_eval.py --handler alpha158-enhanced-v3 \
        --lr 0.05 --depth 6 --l2 3.0

    # 从 JSON 文件加载超参数
    python scripts/models/tree/run_catboost_quick_eval.py --handler alpha158-enhanced-v3 \
        --params-file outputs/hyperopt_cv/catboost_cv_best_params_xxx.json
"""

# ============================================================================
# 重要: 以下代码必须在任何其他导入之前执行
# ============================================================================

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

import sys
from pathlib import Path
import multiprocessing

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))
project_root = script_dir.parent

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

# ============================================================================
# 现在可以安全地导入其他模块
# ============================================================================

import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG, PROJECT_ROOT, MODEL_SAVE_PATH,
    init_qlib,
)


# ============================================================================
# 时间序列交叉验证的 Fold 配置
# ============================================================================

CV_FOLDS = [
    {
        'name': 'Fold 1 (valid 2022)',
        'train_start': '2000-01-01',
        'train_end': '2021-12-31',
        'valid_start': '2022-01-01',
        'valid_end': '2022-12-31',
    },
    {
        'name': 'Fold 2 (valid 2023)',
        'train_start': '2000-01-01',
        'train_end': '2022-12-31',
        'valid_start': '2023-01-01',
        'valid_end': '2023-12-31',
    },
    {
        'name': 'Fold 3 (valid 2024)',
        'train_start': '2000-01-01',
        'train_end': '2023-12-31',
        'valid_start': '2024-01-01',
        'valid_end': '2024-12-31',
    },
]


# ============================================================================
# 默认超参数 (从之前的 hyperopt 搜索中获得)
# ============================================================================

DEFAULT_PARAMS = {
    'loss_function': 'RMSE',
    'iterations': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'l2_leaf_reg': 3.0,
    'random_strength': 1.0,
    'bagging_temperature': 0.5,
    'subsample': 0.8,
    'colsample_bylevel': 0.8,
    'min_data_in_leaf': 50,
    'thread_count': 16,
    'verbose': False,
    'random_seed': 42,
}


def create_data_handler_for_fold(args, handler_config, symbols, fold_config):
    """为特定 fold 创建 DataHandler"""
    from models.common.handlers import get_handler_class

    HandlerClass = get_handler_class(args.handler)

    end_time = fold_config['valid_end']

    handler = HandlerClass(
        volatility_window=args.nday,
        instruments=symbols,
        start_time=fold_config['train_start'],
        end_time=end_time,
        fit_start_time=fold_config['train_start'],
        fit_end_time=fold_config['train_end'],
        infer_processors=[],
    )

    return handler


def create_dataset_for_fold(handler, fold_config):
    """为特定 fold 创建 Dataset"""
    segments = {
        "train": (fold_config['train_start'], fold_config['train_end']),
        "valid": (fold_config['valid_start'], fold_config['valid_end']),
    }
    return DatasetH(handler=handler, segments=segments)


def compute_ic(pred, label, index):
    """计算 IC (按日期分组的相关系数平均值)"""
    df = pd.DataFrame({'pred': pred, 'label': label}, index=index)
    ic_by_date = df.groupby(level='datetime').apply(
        lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
    )
    ic_by_date = ic_by_date.dropna()
    if len(ic_by_date) == 0:
        return 0.0, 0.0, 0.0
    mean_ic = ic_by_date.mean()
    ic_std = ic_by_date.std()
    icir = mean_ic / ic_std if ic_std > 0 else 0
    return mean_ic, ic_std, icir


def run_cv_evaluation(args, handler_config, symbols, cb_params, shuffle=False, shuffle_seed=None):
    """运行 CV 评估，返回平均 IC 和特征重要性

    Parameters
    ----------
    shuffle : bool
        If True, randomly shuffle the labels (for shuffle test)
    shuffle_seed : int
        Random seed for shuffling (for reproducibility)
    """
    print("\n" + "=" * 70)
    if shuffle:
        print(f"CV EVALUATION - SHUFFLE TEST (seed={shuffle_seed})")
    else:
        print("CV EVALUATION WITH FIXED PARAMETERS")
    print("=" * 70)
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: train {fold['train_start']}~{fold['train_end']}, "
              f"valid {fold['valid_start']}~{fold['valid_end']}")
    if shuffle:
        print("\n  ⚠️  SHUFFLE MODE: Labels are RANDOMIZED!")
        print("     If IC ≈ 0, the original signal is real.")
        print("     If IC > 0.01, the original signal may be noise.")
    print("=" * 70)

    # 准备所有 fold 的数据
    print("\n[*] Preparing data for all CV folds...")
    fold_data = []

    for fold in CV_FOLDS:
        print(f"    Preparing {fold['name']}...")
        handler = create_data_handler_for_fold(args, handler_config, symbols, fold)
        dataset = create_dataset_for_fold(handler, fold)

        train_data = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
        valid_data = dataset.prepare("valid", col_set="feature", data_key=DataHandlerLP.DK_L)
        train_label = dataset.prepare("train", col_set="label").values.ravel()
        valid_label = dataset.prepare("valid", col_set="label").values.ravel()

        # Shuffle labels if requested
        if shuffle:
            rng = np.random.RandomState(shuffle_seed)
            train_label = rng.permutation(train_label)
            valid_label = rng.permutation(valid_label)

        # 第一个 fold 打印特征信息
        if len(fold_data) == 0:
            print(f"\n      Total features: {len(train_data.columns)}")
            print(f"      Train NaN%: {train_data.isna().mean().mean()*100:.2f}%")

        fold_data.append({
            'name': fold['name'],
            'train_data': train_data,
            'valid_data': valid_data,
            'train_label': train_label,
            'valid_label': valid_label,
            'train_pool': Pool(train_data, label=train_label),
            'valid_pool': Pool(valid_data, label=valid_label),
        })

        print(f"      Train: {train_data.shape}, Valid: {valid_data.shape}")

    print(f"\n    ✓ All {len(CV_FOLDS)} folds prepared")

    # 训练和评估
    print("\n[*] Training on all folds...")
    fold_ics = []
    fold_models = []
    all_importance = None

    for fold_idx, fold in enumerate(fold_data):
        print(f"\n  [{fold_idx+1}/{len(fold_data)}] {fold['name']}...")

        model = CatBoostRegressor(**cb_params)
        model.fit(
            fold['train_pool'],
            eval_set=fold['valid_pool'],
            early_stopping_rounds=50,
            verbose=100,
        )

        # 训练集 IC
        train_pred = model.predict(fold['train_data'])
        train_ic, _, train_icir = compute_ic(train_pred, fold['train_label'], fold['train_data'].index)

        # 验证集 IC
        valid_pred = model.predict(fold['valid_data'])
        mean_ic, ic_std, icir = compute_ic(valid_pred, fold['valid_label'], fold['valid_data'].index)

        fold_ics.append(mean_ic)
        fold_models.append(model)

        print(f"      Best iter: {model.best_iteration_}")
        print(f"      Train IC: {train_ic:.4f} (ICIR: {train_icir:.4f})")
        print(f"      Valid IC: {mean_ic:.4f} (ICIR: {icir:.4f})")

        # 累计特征重要性
        importance = model.get_feature_importance()
        if all_importance is None:
            all_importance = importance
        else:
            all_importance += importance

    # 计算平均
    mean_ic_all = np.mean(fold_ics)
    std_ic_all = np.std(fold_ics)
    avg_importance = all_importance / len(CV_FOLDS)

    # 输出汇总
    print("\n" + "=" * 70)
    print("CV EVALUATION COMPLETE")
    print("=" * 70)
    fold_ic_str = ", ".join([f"{ic:.4f}" for ic in fold_ics])
    print(f"CV Mean IC: {mean_ic_all:.4f} (±{std_ic_all:.4f})")
    print(f"Folds:      [{fold_ic_str}]")
    print("=" * 70)

    # 创建特征重要性 DataFrame
    feature_names = fold_data[0]['train_data'].columns.tolist()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': avg_importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    return mean_ic_all, std_ic_all, importance_df


def print_feature_importance(importance_df, top_n=50):
    """打印特征重要性报告"""
    print("\n" + "=" * 70)
    print(f"FEATURE IMPORTANCE (Top {top_n})")
    print("=" * 70)
    print(f"{'Rank':<6} {'Feature':<45} {'Importance':>12}")
    print("-" * 70)

    for i, row in importance_df.head(top_n).iterrows():
        print(f"{i+1:<6} {row['feature']:<45} {row['importance']:>12.2f}")

    print("-" * 70)

    # 统计低重要性特征
    low_importance = importance_df[importance_df['importance'] < 0.1]
    zero_importance = importance_df[importance_df['importance'] == 0]

    print(f"\nSummary:")
    print(f"  Total features: {len(importance_df)}")
    print(f"  Features with importance < 0.1: {len(low_importance)}")
    print(f"  Features with importance = 0: {len(zero_importance)}")

    if len(zero_importance) > 0:
        print(f"\n  Zero importance features:")
        for _, row in zero_importance.iterrows():
            print(f"    - {row['feature']}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='CatBoost Quick Evaluation with Fixed Parameters',
    )

    # 基础参数
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--handler', type=str, default='alpha158-enhanced-v4',
                        choices=list(HANDLER_CONFIG.keys()))
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])

    # 超参数 (可选覆盖)
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--depth', type=int, default=None, help='Max depth')
    parser.add_argument('--l2', type=float, default=None, help='L2 regularization')
    parser.add_argument('--subsample', type=float, default=None, help='Subsample ratio')
    parser.add_argument('--colsample', type=float, default=None, help='Column sample ratio')
    parser.add_argument('--min-leaf', type=int, default=None, help='Min data in leaf')

    # 从文件加载参数
    parser.add_argument('--params-file', type=str, default=None,
                        help='Load parameters from JSON file')

    # 输出选项
    parser.add_argument('--top-n', type=int, default=50,
                        help='Number of top features to display')
    parser.add_argument('--save-importance', action='store_true',
                        help='Save feature importance to CSV')

    # Shuffle Test (验证信号是否真实)
    parser.add_argument('--shuffle', action='store_true',
                        help='Run shuffle test: randomize labels to check if IC is noise')
    parser.add_argument('--shuffle-runs', type=int, default=5,
                        help='Number of shuffle test iterations (default: 5)')

    args = parser.parse_args()

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    # 构建超参数
    cb_params = DEFAULT_PARAMS.copy()

    # 从文件加载
    if args.params_file:
        print(f"\n[*] Loading parameters from: {args.params_file}")
        with open(args.params_file, 'r') as f:
            saved = json.load(f)
        if 'params' in saved:
            for k, v in saved['params'].items():
                if k in cb_params:
                    cb_params[k] = v
        print("    ✓ Parameters loaded")

    # CLI 参数覆盖
    if args.lr is not None:
        cb_params['learning_rate'] = args.lr
    if args.depth is not None:
        cb_params['max_depth'] = args.depth
    if args.l2 is not None:
        cb_params['l2_leaf_reg'] = args.l2
    if args.subsample is not None:
        cb_params['subsample'] = args.subsample
    if args.colsample is not None:
        cb_params['colsample_bylevel'] = args.colsample
    if args.min_leaf is not None:
        cb_params['min_data_in_leaf'] = args.min_leaf

    # 打印头部
    print("=" * 70)
    if args.shuffle:
        print("CatBoost SHUFFLE TEST")
    else:
        print("CatBoost Quick Evaluation")
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Handler: {args.handler}")
    print(f"N-day: {args.nday}")
    if args.shuffle:
        print(f"Shuffle runs: {args.shuffle_runs}")
    print(f"\nParameters:")
    for key in ['learning_rate', 'max_depth', 'l2_leaf_reg', 'subsample', 'colsample_bylevel', 'min_data_in_leaf']:
        print(f"  {key}: {cb_params[key]}")
    print("=" * 70)

    # 初始化
    init_qlib(handler_config['use_talib'])

    if args.shuffle:
        # ============================================================
        # SHUFFLE TEST MODE
        # ============================================================
        print("\n" + "#" * 70)
        print("# SHUFFLE TEST: Validating if IC is real signal or noise")
        print("#" * 70)
        print("\nRunning multiple iterations with randomized labels...")
        print("If the original IC is real, shuffled IC should be ≈ 0")
        print("If shuffled IC is similar to original, the signal may be noise\n")

        shuffle_ics = []
        for i in range(args.shuffle_runs):
            print(f"\n{'='*70}")
            print(f"SHUFFLE RUN {i+1}/{args.shuffle_runs}")
            print(f"{'='*70}")

            mean_ic, std_ic, _ = run_cv_evaluation(
                args, handler_config, symbols, cb_params,
                shuffle=True, shuffle_seed=42 + i
            )
            shuffle_ics.append(mean_ic)
            print(f"\n  Shuffle Run {i+1} IC: {mean_ic:.4f}")

        # Shuffle test summary
        shuffle_mean = np.mean(shuffle_ics)
        shuffle_std = np.std(shuffle_ics)
        shuffle_max = np.max(shuffle_ics)
        shuffle_min = np.min(shuffle_ics)

        print("\n" + "=" * 70)
        print("SHUFFLE TEST RESULTS")
        print("=" * 70)
        print(f"\nShuffle ICs: {[f'{ic:.4f}' for ic in shuffle_ics]}")
        print(f"\nStatistics:")
        print(f"  Mean IC:  {shuffle_mean:.4f}")
        print(f"  Std IC:   {shuffle_std:.4f}")
        print(f"  Max IC:   {shuffle_max:.4f}")
        print(f"  Min IC:   {shuffle_min:.4f}")

        print("\n" + "-" * 70)
        print("INTERPRETATION:")
        print("-" * 70)
        if abs(shuffle_mean) < 0.005:
            print("  ✓ Shuffle Mean IC ≈ 0")
            print("  → The original IC signal is LIKELY REAL")
            print("  → Features have genuine predictive power")
        elif abs(shuffle_mean) < 0.01:
            print("  ⚠ Shuffle Mean IC is small but non-zero")
            print("  → Some signal may be real, but be cautious")
            print("  → Consider using more regularization")
        else:
            print("  ✗ Shuffle Mean IC is significant (> 0.01)")
            print("  → The original IC may be OVERFITTING TO NOISE")
            print("  → Model might be memorizing patterns")
            print("  → Consider simpler models or fewer features")

        print("\nCompare with your original IC to draw conclusions.")
        print("=" * 70)

    else:
        # ============================================================
        # NORMAL EVALUATION MODE
        # ============================================================
        mean_ic, std_ic, importance_df = run_cv_evaluation(
            args, handler_config, symbols, cb_params
        )

        # 打印特征重要性
        print_feature_importance(importance_df, args.top_n)

        # 保存特征重要性
        if args.save_importance:
            output_dir = PROJECT_ROOT / "outputs" / "feature_importance"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{args.handler}_importance_{timestamp}.csv"
            filepath = output_dir / filename
            importance_df.to_csv(filepath, index=False)
            print(f"\n[*] Feature importance saved to: {filepath}")

        # 最终汇总
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Handler: {args.handler}")
        print(f"Total features: {len(importance_df)}")
        print(f"CV Mean IC: {mean_ic:.4f} (±{std_ic:.4f})")
        print("=" * 70)


if __name__ == "__main__":
    main()
