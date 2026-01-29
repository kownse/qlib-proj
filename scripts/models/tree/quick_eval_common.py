"""
Quick Evaluation Common Utilities

Shared functions for LightGBM and CatBoost quick evaluation scripts.
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

# 设置路径
SCRIPT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
PROJECT_ROOT = SCRIPT_DIR.parent

# ============================================================================
# Qlib 初始化
# ============================================================================

import qlib
from qlib.constant import REG_US
from utils.talib_ops import TALIB_OPS

QLIB_DATA_PATH = PROJECT_ROOT / "my_data" / "qlib_us"


def init_qlib_for_quick_eval():
    """Initialize Qlib with TA-Lib support for quick evaluation"""
    qlib.init(
        provider_uri=str(QLIB_DATA_PATH),
        region=REG_US,
        custom_ops=TALIB_OPS,
        kernels=1,
        joblib_backend=None,
    )


# ============================================================================
# 现在可以安全地导入其他模块
# ============================================================================

import json
from datetime import datetime
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from qlib.data.dataset.handler import DataHandlerLP

# 从 cv_utils 导入公共定义，避免重复
from models.common.cv_utils import (
    CV_FOLDS,
    FINAL_TEST,
    create_data_handler_for_fold,
    create_dataset_for_fold,
    compute_ic,
)


# ============================================================================
# 数据准备函数
# ============================================================================

def prepare_cv_fold_data(
    args,
    handler_config,
    symbols,
    shuffle: bool = False,
    shuffle_seed: int = None,
    fill_na: bool = True,
) -> List[Dict]:
    """准备所有 CV fold 的数据

    Parameters
    ----------
    args : argparse.Namespace
        命令行参数
    handler_config : dict
        Handler 配置
    symbols : list
        股票列表
    shuffle : bool
        是否随机打乱标签
    shuffle_seed : int
        随机种子
    fill_na : bool
        是否填充 NaN（LightGBM 需要，CatBoost 不需要）

    Returns
    -------
    list
        包含每个 fold 数据的列表
    """
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

        # 处理 NaN 和 Inf（LightGBM 需要）
        if fill_na:
            train_data = train_data.fillna(0).replace([np.inf, -np.inf], 0)
            valid_data = valid_data.fillna(0).replace([np.inf, -np.inf], 0)

        fold_data.append({
            'name': fold['name'],
            'train_data': train_data,
            'valid_data': valid_data,
            'train_label': train_label,
            'valid_label': valid_label,
        })

        print(f"      Train: {train_data.shape}, Valid: {valid_data.shape}")

    print(f"\n    All {len(CV_FOLDS)} folds prepared")
    return fold_data


# ============================================================================
# 打印函数
# ============================================================================

def print_cv_header(shuffle: bool = False, shuffle_seed: int = None):
    """打印 CV 评估头部"""
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
        print("\n  SHUFFLE MODE: Labels are RANDOMIZED!")
        print("     If IC ≈ 0, the original signal is real.")
        print("     If IC > 0.01, the original signal may be noise.")
    print("=" * 70)


def print_cv_summary(fold_ics: List[float]):
    """打印 CV 评估汇总"""
    mean_ic_all = np.mean(fold_ics)
    std_ic_all = np.std(fold_ics)

    print("\n" + "=" * 70)
    print("CV EVALUATION COMPLETE")
    print("=" * 70)
    fold_ic_str = ", ".join([f"{ic:.4f}" for ic in fold_ics])
    print(f"CV Mean IC: {mean_ic_all:.4f} (±{std_ic_all:.4f})")
    print(f"Folds:      [{fold_ic_str}]")
    print("=" * 70)

    return mean_ic_all, std_ic_all


def print_feature_importance(importance_df: pd.DataFrame, top_n: int = 50):
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


def print_shuffle_test_header():
    """打印 Shuffle Test 头部"""
    print("\n" + "#" * 70)
    print("# SHUFFLE TEST: Validating if IC is real signal or noise")
    print("#" * 70)
    print("\nRunning multiple iterations with randomized labels...")
    print("If the original IC is real, shuffled IC should be ≈ 0")
    print("If shuffled IC is similar to original, the signal may be noise\n")


def print_shuffle_test_results(shuffle_ics: List[float]):
    """打印 Shuffle Test 结果"""
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
        print("  Shuffle Mean IC ≈ 0")
        print("  → The original IC signal is LIKELY REAL")
        print("  → Features have genuine predictive power")
    elif abs(shuffle_mean) < 0.01:
        print("  Shuffle Mean IC is small but non-zero")
        print("  → Some signal may be real, but be cautious")
        print("  → Consider using more regularization")
    else:
        print("  Shuffle Mean IC is significant (> 0.01)")
        print("  → The original IC may be OVERFITTING TO NOISE")
        print("  → Model might be memorizing patterns")
        print("  → Consider simpler models or fewer features")

    print("\nCompare with your original IC to draw conclusions.")
    print("=" * 70)


def print_final_summary(handler: str, importance_df: pd.DataFrame, mean_ic: float, std_ic: float):
    """打印最终汇总"""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Handler: {handler}")
    print(f"Total features: {len(importance_df)}")
    print(f"CV Mean IC: {mean_ic:.4f} (±{std_ic:.4f})")
    print("=" * 70)


# ============================================================================
# 命令行参数工具
# ============================================================================

def add_common_args(parser):
    """添加通用命令行参数"""
    from models.common import HANDLER_CONFIG

    # 基础参数
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--handler', type=str, default='alpha158-enhanced-v4',
                        choices=list(HANDLER_CONFIG.keys()))
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])

    # 从文件加载参数
    parser.add_argument('--params-file', type=str, default=None,
                        help='Load parameters from JSON file')

    # 输出选项
    parser.add_argument('--top-n', type=int, default=50,
                        help='Number of top features to display')
    parser.add_argument('--save-importance', action='store_true',
                        help='Save feature importance to CSV')

    # Shuffle Test
    parser.add_argument('--shuffle', action='store_true',
                        help='Run shuffle test: randomize labels to check if IC is noise')
    parser.add_argument('--shuffle-runs', type=int, default=5,
                        help='Number of shuffle test iterations (default: 5)')

    return parser


def load_params_from_file(params_file: str, default_params: dict) -> dict:
    """从 JSON 文件加载参数"""
    params = default_params.copy()
    print(f"\n[*] Loading parameters from: {params_file}")
    with open(params_file, 'r') as f:
        saved = json.load(f)
    if 'params' in saved:
        for k, v in saved['params'].items():
            if k in params:
                params[k] = v
    print("    Parameters loaded")
    return params


def save_importance_to_csv(
    importance_df: pd.DataFrame,
    model_prefix: str,
    handler: str,
):
    """保存特征重要性到 CSV"""
    from models.common import PROJECT_ROOT

    output_dir = PROJECT_ROOT / "outputs" / "feature_importance"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_prefix}_{handler}_importance_{timestamp}.csv"
    filepath = output_dir / filename
    importance_df.to_csv(filepath, index=False)
    print(f"\n[*] Feature importance saved to: {filepath}")
    return filepath


# ============================================================================
# CV 评估通用框架
# ============================================================================

def run_cv_evaluation_generic(
    args,
    handler_config,
    symbols,
    model_params: dict,
    train_and_predict_func: Callable,
    get_importance_func: Callable,
    fill_na: bool = True,
    shuffle: bool = False,
    shuffle_seed: int = None,
) -> Tuple[float, float, pd.DataFrame]:
    """通用的 CV 评估函数

    Parameters
    ----------
    args : argparse.Namespace
        命令行参数
    handler_config : dict
        Handler 配置
    symbols : list
        股票列表
    model_params : dict
        模型超参数
    train_and_predict_func : Callable
        训练和预测函数，签名: (fold_data, model_params) -> (model, train_pred, valid_pred)
    get_importance_func : Callable
        获取特征重要性函数，签名: (model) -> np.ndarray
    fill_na : bool
        是否填充 NaN
    shuffle : bool
        是否随机打乱标签
    shuffle_seed : int
        随机种子

    Returns
    -------
    tuple
        (mean_ic, std_ic, importance_df)
    """
    print_cv_header(shuffle, shuffle_seed)

    # 准备数据
    fold_data = prepare_cv_fold_data(
        args, handler_config, symbols,
        shuffle=shuffle, shuffle_seed=shuffle_seed, fill_na=fill_na
    )

    # 训练和评估
    print("\n[*] Training on all folds...")
    fold_ics = []
    all_importance = None

    for fold_idx, fold in enumerate(fold_data):
        print(f"\n  [{fold_idx+1}/{len(fold_data)}] {fold['name']}...")

        # 训练和预测
        model, train_pred, valid_pred = train_and_predict_func(fold, model_params)

        # 计算 IC
        train_ic, _, train_icir = compute_ic(train_pred, fold['train_label'], fold['train_data'].index)
        mean_ic, ic_std, icir = compute_ic(valid_pred, fold['valid_label'], fold['valid_data'].index)

        fold_ics.append(mean_ic)

        print(f"      Train IC: {train_ic:.4f} (ICIR: {train_icir:.4f})")
        print(f"      Valid IC: {mean_ic:.4f} (ICIR: {icir:.4f})")

        # 累计特征重要性
        importance = get_importance_func(model)
        if all_importance is None:
            all_importance = importance.astype(float)
        else:
            all_importance += importance

    # 计算平均
    mean_ic_all, std_ic_all = print_cv_summary(fold_ics)
    avg_importance = all_importance / len(CV_FOLDS)

    # 创建特征重要性 DataFrame
    feature_names = fold_data[0]['train_data'].columns.tolist()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': avg_importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    return mean_ic_all, std_ic_all, importance_df


def run_shuffle_test(
    args,
    handler_config,
    symbols,
    model_params: dict,
    train_and_predict_func: Callable,
    get_importance_func: Callable,
    fill_na: bool = True,
    shuffle_runs: int = 5,
) -> List[float]:
    """运行 Shuffle Test

    Returns
    -------
    list
        每次 shuffle 的 IC 列表
    """
    print_shuffle_test_header()

    shuffle_ics = []
    for i in range(shuffle_runs):
        print(f"\n{'='*70}")
        print(f"SHUFFLE RUN {i+1}/{shuffle_runs}")
        print(f"{'='*70}")

        mean_ic, _, _ = run_cv_evaluation_generic(
            args, handler_config, symbols, model_params,
            train_and_predict_func, get_importance_func,
            fill_na=fill_na, shuffle=True, shuffle_seed=42 + i
        )
        shuffle_ics.append(mean_ic)
        print(f"\n  Shuffle Run {i+1} IC: {mean_ic:.4f}")

    print_shuffle_test_results(shuffle_ics)
    return shuffle_ics
