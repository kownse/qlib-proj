"""
CatBoost 交叉验证训练和评估脚本

支持两种模式:
1. 评估模式: 加载已训练模型，在 CV folds 上计算 Valid IC 和 2025 Test IC
2. 训练模式: 使用预先搜索好的超参数进行 CV 训练

时间窗口设计:
  Fold 1: train 2000-2020, valid 2021
  Fold 2: train 2000-2021, valid 2022
  Fold 3: train 2000-2022, valid 2023
  Fold 4: train 2000-2023, valid 2024
  Test:   2025 (完全独立)

使用方法:
    # ===== 评估模式 (加载已训练模型) =====
    python scripts/models/tree/run_catboost_cv.py \
        --eval-only \
        --model-path my_models/catboost_cv_xxx.cbm

    # ===== 训练模式 =====
    # 使用参数文件训练 (自动读取 sample_weight_halflife)
    python scripts/models/tree/run_catboost_cv.py \
        --params-file outputs/hyperopt_cv/catboost_cv_best_params__alpha158-talib-macro-sector.json

    # 只运行 CV 训练，不训练最终模型
    python scripts/models/tree/run_catboost_cv.py \
        --params-file outputs/hyperopt_cv/catboost_cv_best_params_xxx.json \
        --cv-only

    # 多种子训练
    python scripts/models/tree/run_catboost_cv.py \
        --params-file outputs/hyperopt_cv/catboost_cv_best_params_xxx.json \
        --cv-only --num-seeds 10

    # 训练最终模型并回测
    python scripts/models/tree/run_catboost_cv.py \
        --params-file outputs/hyperopt_cv/catboost_cv_best_params_xxx.json \
        --backtest
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

from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG, MODEL_SAVE_PATH,
    init_qlib,
    run_backtest,
    load_catboost_model,
    get_catboost_feature_count,
    CV_FOLDS,
    FINAL_TEST,
    create_data_handler_for_fold,
    create_dataset_for_fold,
    compute_time_decay_weights,
    prepare_features_and_labels,
    compute_ic,
)


# ============================================================================
# 默认 CatBoost 参数
# ============================================================================

DEFAULT_CATBOOST_PARAMS = {
    'loss_function': 'RMSE',
    'iterations': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'l2_leaf_reg': 3,
    'random_strength': 1,
    'thread_count': 16,
    'verbose': False,
    'random_seed': 42,
}


def load_params_from_json(params_file: str):
    """从 JSON 文件加载超参数"""
    with open(params_file, 'r') as f:
        data = json.load(f)

    sample_weight_halflife = 0
    if 'params' in data:
        params = data['params']
        cv_results = data.get('cv_results', {})
        if 'sample_weight_halflife' in data:
            sample_weight_halflife = data['sample_weight_halflife']
    else:
        params = data
        cv_results = {}
        if 'sample_weight_halflife' in params:
            sample_weight_halflife = params.pop('sample_weight_halflife')

    # 合并到默认参数
    final_params = DEFAULT_CATBOOST_PARAMS.copy()
    for key in ['learning_rate', 'max_depth', 'l2_leaf_reg', 'random_strength',
                'bagging_temperature', 'subsample', 'colsample_bylevel',
                'min_data_in_leaf', 'iterations', 'bootstrap_type', 'thread_count']:
        if key in params:
            final_params[key] = params[key]

    # 确保整数类型
    if 'max_depth' in final_params:
        final_params['max_depth'] = int(final_params['max_depth'])
    if 'min_data_in_leaf' in final_params:
        final_params['min_data_in_leaf'] = int(final_params['min_data_in_leaf'])

    print(f"\n[*] Loaded parameters from: {params_file}")
    for key in ['learning_rate', 'max_depth', 'l2_leaf_reg', 'random_strength',
                'bagging_temperature', 'subsample', 'colsample_bylevel', 'min_data_in_leaf']:
        if key in final_params and final_params[key] is not None:
            print(f"    {key}: {final_params[key]}")
    if sample_weight_halflife > 0:
        print(f"    sample_weight_halflife: {sample_weight_halflife} years")

    if cv_results:
        print(f"\n    Original CV results:")
        print(f"      Mean IC: {cv_results.get('mean_ic', 'N/A'):.4f} (±{cv_results.get('std_ic', 'N/A'):.4f})")
        if 'fold_results' in cv_results:
            for fold in cv_results['fold_results']:
                print(f"      {fold['name']}: IC={fold['ic']:.4f}, ICIR={fold['icir']:.4f}")

    return final_params, cv_results, sample_weight_halflife


# ============================================================================
# 特征对齐
# ============================================================================

def align_features(X, model_features):
    """
    将数据的特征列对齐到模型期望的特征名列表。

    处理两种情况:
    1. 数据列名是 MultiIndex (如 ('feature', 'KMID')) → 取第二级
    2. 数据列名与模型特征名不完全匹配 → 按名字选择并重排序
    """
    if model_features is None:
        return X

    # 处理 MultiIndex 列名
    if isinstance(X.columns, pd.MultiIndex):
        X = X.copy()
        X.columns = [col[1] if isinstance(col, tuple) else col for col in X.columns]

    data_cols = set(X.columns)
    model_cols = list(model_features)

    missing = [c for c in model_cols if c not in data_cols]
    if missing:
        print(f"    Warning: {len(missing)} model features not found in data, filling with 0:")
        for c in missing[:10]:
            print(f"      - {c}")
        if len(missing) > 10:
            print(f"      ... and {len(missing) - 10} more")
        for c in missing:
            X[c] = 0.0

    extra = data_cols - set(model_cols)
    if extra:
        print(f"    Note: {len(extra)} extra features in data (ignored by model)")

    return X[model_cols]


# ============================================================================
# 评估模式
# ============================================================================

def run_cv_evaluation(args, handler_config, symbols, model_path):
    """加载预训练模型并在 CV folds 和 2025 测试集上评估 IC"""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION EVALUATION (CatBoost)")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: valid {fold['valid_start']}~{fold['valid_end']}")
    print(f"Test Set: {FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']} (2025)")
    print("=" * 70)

    # 加载模型
    print(f"\n[*] Loading model from: {model_path}")
    model = load_catboost_model(model_path)
    model_features = model.feature_names_
    num_features = len(model_features) if model_features else "unknown"
    print(f"    Model features: {num_features}")
    print(f"    Model loaded successfully")

    # 准备 2025 测试集数据
    print("\n[*] Preparing 2025 test data...")
    test_handler = create_data_handler_for_fold(args, handler_config, symbols, FINAL_TEST)
    test_dataset = create_dataset_for_fold(test_handler, FINAL_TEST)
    X_test, y_test = prepare_features_and_labels(test_dataset, "test")
    X_test = align_features(X_test, model_features)
    print(f"    Test (2025): {X_test.shape}")

    fold_results = []
    fold_ics = []
    fold_test_ics = []

    for fold in CV_FOLDS:
        print(f"\n[*] Evaluating on {fold['name']}...")

        handler = create_data_handler_for_fold(args, handler_config, symbols, fold)
        dataset = create_dataset_for_fold(handler, fold)
        X_valid, y_valid = prepare_features_and_labels(dataset, "valid")
        X_valid = align_features(X_valid, model_features)

        print(f"    Valid: {X_valid.shape}")

        # 验证集预测
        valid_pred = model.predict(X_valid)
        mean_ic, ic_std, icir = compute_ic(valid_pred, y_valid, X_valid.index)

        # 2025 测试集预测
        test_pred = model.predict(X_test)
        test_ic, test_ic_std, test_icir = compute_ic(test_pred, y_test, X_test.index)

        fold_ics.append(mean_ic)
        fold_test_ics.append(test_ic)
        fold_results.append({
            'name': fold['name'],
            'ic': mean_ic,
            'icir': icir,
            'test_ic': test_ic,
            'test_icir': test_icir,
        })

        print(f"    {fold['name']}: Valid IC={mean_ic:.4f}, Test IC (2025)={test_ic:.4f}")

    # 汇总
    mean_ic_all = np.mean(fold_ics)
    std_ic_all = np.std(fold_ics)
    mean_test_ic_all = np.mean(fold_test_ics)
    std_test_ic_all = np.std(fold_test_ics)

    print("\n" + "=" * 70)
    print("CV EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Valid Mean IC: {mean_ic_all:.4f} (±{std_ic_all:.4f})")
    print(f"Test Mean IC (2025): {mean_test_ic_all:.4f} (±{std_test_ic_all:.4f})")
    print(f"\nIC by fold:")
    print(f"  {'Fold':<25s} {'Valid IC':>10s} {'Test IC':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    for r in fold_results:
        print(f"  {r['name']:<25s} {r['ic']:>10.4f} {r['test_ic']:>10.4f}")
    print("=" * 70)

    # 返回测试集预测
    test_pred_series = pd.Series(test_pred, index=X_test.index, name='score')
    return fold_results, mean_ic_all, std_ic_all, test_pred_series, test_dataset


# ============================================================================
# 训练模式
# ============================================================================

def run_cv_training(args, handler_config, symbols, params, halflife=0):
    """运行 CV 训练, 同时在 2025 测试集上评估"""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION TRAINING (CatBoost)")
    print("=" * 70)
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: train {fold['train_start']}~{fold['train_end']}, "
              f"valid {fold['valid_start']}~{fold['valid_end']}")
    print(f"Test Set: {FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']} (2025)")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    if halflife > 0:
        print(f"Sample weight halflife: {halflife} years")
    print(f"\nCatBoost params:")
    for key, value in params.items():
        if key not in ('verbose',):
            print(f"  {key}: {value}")
    print("=" * 70)

    # 准备 2025 测试集
    print("\n[*] Preparing 2025 test data...")
    test_handler = create_data_handler_for_fold(args, handler_config, symbols, FINAL_TEST)
    test_dataset = create_dataset_for_fold(test_handler, FINAL_TEST)
    X_test, y_test = prepare_features_and_labels(test_dataset, "test")
    print(f"    Test (2025): {X_test.shape}")

    fold_results = []
    fold_ics = []
    fold_test_ics = []

    for fold_idx, fold in enumerate(CV_FOLDS):
        print(f"\n[*] Training {fold['name']}...")

        handler = create_data_handler_for_fold(args, handler_config, symbols, fold)
        dataset = create_dataset_for_fold(handler, fold)

        X_train, y_train = prepare_features_and_labels(dataset, "train")
        X_valid, y_valid = prepare_features_and_labels(dataset, "valid")

        print(f"    Train: {X_train.shape}, Valid: {X_valid.shape}")

        # 样本权重
        train_weight = None
        valid_weight = None
        if halflife > 0:
            train_weight = compute_time_decay_weights(X_train.index, halflife)
            valid_weight = compute_time_decay_weights(X_valid.index, halflife)

        train_pool = Pool(X_train, label=y_train, weight=train_weight)
        valid_pool = Pool(X_valid, label=y_valid, weight=valid_weight)

        # 设置种子 (与 hyperopt CV 一致, 所有 fold 使用相同种子)
        fold_params = params.copy()
        if args.seed is not None:
            fold_params['random_seed'] = args.seed

        model = CatBoostRegressor(**fold_params)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=50,
            verbose=100 if args.verbose else False,
        )

        # 验证集 IC
        valid_pred = model.predict(X_valid)
        mean_ic, ic_std, icir = compute_ic(valid_pred, y_valid, X_valid.index)

        # 2025 测试集 IC
        test_pred = model.predict(X_test)
        test_ic, test_ic_std, test_icir = compute_ic(test_pred, y_test, X_test.index)

        fold_ics.append(mean_ic)
        fold_test_ics.append(test_ic)
        fold_results.append({
            'name': fold['name'],
            'ic': mean_ic,
            'icir': icir,
            'test_ic': test_ic,
            'test_icir': test_icir,
            'best_iter': model.best_iteration_,
        })

        print(f"    {fold['name']}: Valid IC={mean_ic:.4f}, Test IC (2025)={test_ic:.4f}, iter={model.best_iteration_}")

    # 汇总
    mean_ic_all = np.mean(fold_ics)
    std_ic_all = np.std(fold_ics)
    mean_test_ic_all = np.mean(fold_test_ics)
    std_test_ic_all = np.std(fold_test_ics)

    print("\n" + "=" * 70)
    print("CV TRAINING COMPLETE")
    print("=" * 70)
    print(f"Valid Mean IC: {mean_ic_all:.4f} (±{std_ic_all:.4f})")
    print(f"Test Mean IC (2025): {mean_test_ic_all:.4f} (±{std_test_ic_all:.4f})")
    print(f"\nIC by fold:")
    print(f"  {'Fold':<25s} {'Valid IC':>10s} {'Test IC':>10s} {'Iter':>8s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")
    for r in fold_results:
        print(f"  {r['name']:<25s} {r['ic']:>10.4f} {r['test_ic']:>10.4f} {r['best_iter']:>8d}")
    print("=" * 70)

    return fold_results, mean_ic_all, std_ic_all


def train_final_model(args, handler_config, symbols, params, halflife=0):
    """使用最优参数在完整数据上训练最终模型"""
    print("\n[*] Training final model on full data...")
    print("    Parameters:")
    for key in ['learning_rate', 'max_depth', 'l2_leaf_reg', 'random_strength',
                'bagging_temperature', 'subsample', 'colsample_bylevel', 'min_data_in_leaf']:
        if key in params and params[key] is not None:
            print(f"      {key}: {params[key]}")
    if halflife > 0:
        print(f"      sample_weight_halflife: {halflife}")

    handler = create_data_handler_for_fold(args, handler_config, symbols, FINAL_TEST)
    dataset = create_dataset_for_fold(handler, FINAL_TEST)

    X_train, y_train = prepare_features_and_labels(dataset, "train")
    X_valid, y_valid = prepare_features_and_labels(dataset, "valid")
    X_test, y_test = prepare_features_and_labels(dataset, "test")

    print(f"\n    Final training data:")
    print(f"      Train: {X_train.shape} ({FINAL_TEST['train_start']} ~ {FINAL_TEST['train_end']})")
    print(f"      Valid: {X_valid.shape} ({FINAL_TEST['valid_start']} ~ {FINAL_TEST['valid_end']})")
    print(f"      Test:  {X_test.shape} ({FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']})")

    # 样本权重
    train_weight = None
    valid_weight = None
    if halflife > 0:
        train_weight = compute_time_decay_weights(X_train.index, halflife)
        valid_weight = compute_time_decay_weights(X_valid.index, halflife)
        print(f"\n    Using sample weights: halflife={halflife} years")

    train_pool = Pool(X_train, label=y_train, weight=train_weight)
    valid_pool = Pool(X_valid, label=y_valid, weight=valid_weight)

    final_params = params.copy()
    if args.seed is not None:
        final_params['random_seed'] = args.seed

    print("\n    Training progress:")
    model = CatBoostRegressor(**final_params)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    print(f"\n    Best iteration: {model.best_iteration_}")

    # 验证集 IC
    valid_pred = model.predict(X_valid)
    valid_ic, _, valid_icir = compute_ic(valid_pred, y_valid, X_valid.index)
    print(f"\n    [Validation Set - for reference]")
    print(f"    Valid IC:   {valid_ic:.4f}")
    print(f"    Valid ICIR: {valid_icir:.4f}")

    # 测试集预测
    test_pred_values = model.predict(X_test)
    test_pred = pd.Series(test_pred_values, index=X_test.index, name='score')

    # 测试集 IC
    test_ic, _, test_icir = compute_ic(test_pred_values, y_test, X_test.index)
    print(f"\n    [Test Set (2025)]")
    print(f"    Test IC:   {test_ic:.4f}")
    print(f"    Test ICIR: {test_icir:.4f}")
    print(f"    Prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    return model, test_pred, dataset


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CatBoost Cross-Validation Training and Evaluation',
    )

    # 参数文件
    parser.add_argument('--params-file', type=str, default=None,
                        help='Path to JSON file with CatBoost hyperparameters')

    # 模型评估模式
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to pre-trained model for evaluation')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate pre-trained model on CV folds, no training')

    # 基础参数
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--handler', type=str, default='alpha158-talib-macro-sector',
                        choices=list(HANDLER_CONFIG.keys()))
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])

    # CV 训练参数
    parser.add_argument('--cv-only', action='store_true',
                        help='Only run CV training, skip final model')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num-seeds', type=int, default=1,
                        help='Run CV with multiple seeds and report best result')
    parser.add_argument('--verbose', action='store_true',
                        help='Show training progress for each fold')

    # 样本权重参数
    parser.add_argument('--sample-weight-halflife', type=float, default=0,
                        help='Fixed time-decay half-life in years (0=disabled). '
                             'Overrides value from params file if set.')

    # 训练参数
    parser.add_argument('--thread-count', type=int, default=16,
                        help='Number of threads for CatBoost training (default: 16)')

    # 最终训练参数
    parser.add_argument('--n-epochs', type=int, default=1000,
                        help='Max iterations for final model (default: 1000)')
    parser.add_argument('--early-stop', type=int, default=50,
                        help='Early stopping rounds for final model')

    # 回测参数
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=1)
    parser.add_argument('--account', type=float, default=10000)
    parser.add_argument('--rebalance-freq', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss'])

    args = parser.parse_args()

    # 验证参数
    if args.eval_only and not args.model_path:
        parser.error("--eval-only requires --model-path")
    if not args.eval_only and not args.params_file:
        parser.error("--params-file is required for training mode (or use --eval-only)")

    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    init_qlib(handler_config['use_talib'])

    # ========== 评估模式 ==========
    if args.eval_only:
        print("\n" + "=" * 70)
        print("CatBoost Cross-Validation EVALUATION Mode")
        print("=" * 70)
        print(f"Model: {args.model_path}")
        print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
        print(f"Handler: {args.handler}")
        print(f"N-day: {args.nday}")
        print("=" * 70)

        fold_results, mean_ic, std_ic, test_pred, test_dataset = run_cv_evaluation(
            args, handler_config, symbols, args.model_path
        )

        if args.backtest:
            print("\n[*] Running backtest on 2025 test set...")
            pred_df = test_pred.to_frame("score")

            time_splits = {
                'train_start': FINAL_TEST['train_start'],
                'train_end': FINAL_TEST['train_end'],
                'valid_start': FINAL_TEST['valid_start'],
                'valid_end': FINAL_TEST['valid_end'],
                'test_start': FINAL_TEST['test_start'],
                'test_end': FINAL_TEST['test_end'],
            }

            run_backtest(
                args.model_path, test_dataset, pred_df, args, time_splits,
                model_name="CatBoost (CV Eval)",
                load_model_func=load_catboost_model,
                get_feature_count_func=get_catboost_feature_count,
            )

        return

    # ========== 训练模式 ==========
    params, original_cv_results, file_halflife = load_params_from_json(args.params_file)

    # 确定 halflife: CLI 显式设置优先，否则用文件中的值
    halflife = args.sample_weight_halflife if args.sample_weight_halflife > 0 else file_halflife

    # thread_count: CLI --thread-count 覆盖文件中的值
    params['thread_count'] = args.thread_count

    print("\n" + "=" * 70)
    print("CatBoost Cross-Validation Training")
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Handler: {args.handler}")
    print(f"N-day: {args.nday}")
    print(f"CV Folds: {len(CV_FOLDS)}")
    if halflife > 0:
        print(f"Sample weight halflife: {halflife} years")
    if args.num_seeds > 1:
        print(f"Num seeds: {args.num_seeds}")
    print("=" * 70)

    # 运行 CV 训练
    if args.num_seeds > 1:
        base_seed = args.seed if args.seed is not None else 42
        all_results = []

        print(f"\n[*] Running CV with {args.num_seeds} different seeds...")
        for seed_idx in range(args.num_seeds):
            seed = base_seed + seed_idx * 1000
            args.seed = seed
            print(f"\n{'='*70}")
            print(f"SEED {seed_idx + 1}/{args.num_seeds}: seed={seed}")
            print(f"{'='*70}")

            fold_results, mean_ic, std_ic = run_cv_training(
                args, handler_config, symbols, params, halflife=halflife
            )
            mean_test_ic = np.mean([r['test_ic'] for r in fold_results])
            all_results.append({
                'seed': seed,
                'mean_ic': mean_ic,
                'std_ic': std_ic,
                'mean_test_ic': mean_test_ic,
                'fold_results': fold_results,
            })

        best_result = max(all_results, key=lambda x: x['mean_ic'])
        best_test_result = max(all_results, key=lambda x: x['mean_test_ic'])
        args.seed = best_result['seed']
        fold_results = best_result['fold_results']
        mean_ic = best_result['mean_ic']
        std_ic = best_result['std_ic']

        print("\n" + "=" * 70)
        print("MULTI-SEED SUMMARY")
        print("=" * 70)
        print(f"  {'Seed':<10s} {'Valid IC':>12s} {'Test IC (2025)':>16s}")
        print(f"  {'-'*10} {'-'*12} {'-'*16}")
        for r in all_results:
            valid_marker = " ★" if r['seed'] == best_result['seed'] else ""
            test_marker = " ◆" if r['seed'] == best_test_result['seed'] else ""
            print(f"  {r['seed']:<10d} {r['mean_ic']:>10.4f}{valid_marker:<2s} {r['mean_test_ic']:>14.4f}{test_marker:<2s}")
        print(f"\n★ Best valid seed: {best_result['seed']} (Valid IC={best_result['mean_ic']:.4f}, Test IC={best_result['mean_test_ic']:.4f})")
        print(f"◆ Best test seed:  {best_test_result['seed']} (Valid IC={best_test_result['mean_ic']:.4f}, Test IC={best_test_result['mean_test_ic']:.4f})")
        print("=" * 70)
    else:
        fold_results, mean_ic, std_ic = run_cv_training(
            args, handler_config, symbols, params, halflife=halflife
        )

    # 比较
    if original_cv_results:
        print("\n[*] Comparison with original results:")
        print(f"    Original Mean IC: {original_cv_results['mean_ic']:.4f}")
        print(f"    Current Mean IC:  {mean_ic:.4f}")
        diff = mean_ic - original_cv_results['mean_ic']
        print(f"    Difference: {diff:+.4f}")

    if args.cv_only:
        print("\n[*] CV-only mode, skipping final model training.")
        return

    # 训练最终模型
    model, test_pred, dataset = train_final_model(
        args, handler_config, symbols, params, halflife=halflife
    )

    # 保存模型
    print("\n[*] Saving model...")
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_SAVE_PATH / f"catboost_cv_{args.handler}_{args.stock_pool}_{args.nday}d_{timestamp}.cbm"
    model.save_model(str(model_path))
    print(f"    Model saved to: {model_path}")

    # 回测
    if args.backtest:
        pred_df = test_pred.to_frame("score")

        time_splits = {
            'train_start': FINAL_TEST['train_start'],
            'train_end': FINAL_TEST['train_end'],
            'valid_start': FINAL_TEST['valid_start'],
            'valid_end': FINAL_TEST['valid_end'],
            'test_start': FINAL_TEST['test_start'],
            'test_end': FINAL_TEST['test_end'],
        }

        run_backtest(
            model_path, dataset, pred_df, args, time_splits,
            model_name="CatBoost (CV)",
            load_model_func=load_catboost_model,
            get_feature_count_func=get_catboost_feature_count,
        )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"CV Mean IC: {mean_ic:.4f} (±{std_ic:.4f})")
    print(f"Model saved to: {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
