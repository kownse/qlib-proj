"""
AE-MLP 交叉验证训练和评估脚本

支持两种模式:
1. 训练模式: 使用预先搜索好的超参数进行训练
2. 评估模式: 加载已训练模型，在 CV folds 上计算 IC

时间窗口设计:
  Fold 1: train 2000-2021, valid 2022
  Fold 2: train 2000-2022, valid 2023
  Fold 3: train 2000-2023, valid 2024
  Test:   2025 (完全独立)

使用方法:
    # ===== 评估模式 (加载已训练模型) =====
    python scripts/models/deep/run_ae_mlp_cv.py \
        --eval-only \
        --model-path my_models/ae_mlp_cv_alpha158-enhanced-v7_sp500_5d_best.keras \
        --handler alpha158-enhanced-v7

    # ===== 训练模式 =====
    # 使用参数文件训练
    python scripts/models/deep/run_ae_mlp_cv.py \
        --params-file outputs/hyperopt_cv/ae_mlp_cv_best_params_20260117_151024.json

    # 只运行 CV 训练，不训练最终模型
    python scripts/models/deep/run_ae_mlp_cv.py \
        --params-file outputs/hyperopt_cv/ae_mlp_cv_best_params_20260117_151024.json \
        --cv-only

    # 多种子训练，寻找最佳结果
    python scripts/models/deep/run_ae_mlp_cv.py \
        --params-file outputs/hyperopt_cv/ae_mlp_cv_best_params_20260117_151024.json \
        --cv-only --num-seeds 10

    # 训练最终模型并回测
    python scripts/models/deep/run_ae_mlp_cv.py \
        --params-file outputs/hyperopt_cv/ae_mlp_cv_best_params_20260117_151024.json \
        --backtest
"""

# ============================================================================
# 重要: 以下代码必须在任何其他导入之前执行
# ============================================================================

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 抑制 TensorFlow 日志
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # 优化 GPU 线程

# 设置 GPU 内存增长（必须在导入 TensorFlow 之前或刚导入后立即设置）
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # 已经初始化

import random
import sys
from pathlib import Path

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
)

# ============================================================================
# 现在可以安全地导入其他模块
# ============================================================================

import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd

# tensorflow 已在文件开头导入
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras import mixed_precision

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from utils.utils import evaluate_model
from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG, PROJECT_ROOT, MODEL_SAVE_PATH,
    print_training_header,
    init_qlib,
    check_data_availability,
    save_model_with_meta,
    create_meta_data,
    generate_model_filename,
    run_backtest,
    # CV utilities
    CV_FOLDS,
    FINAL_TEST,
    create_data_handler_for_fold,
    create_dataset_for_fold,
    prepare_data_from_dataset,
    compute_ic,
)
from models.deep.ae_mlp_shared import set_random_seed, create_tf_dataset, build_ae_mlp_model, setup_gpu


def load_params_from_json(params_file: str) -> dict:
    """从 JSON 文件加载超参数"""
    with open(params_file, 'r') as f:
        data = json.load(f)

    params = data['params']
    cv_results = data.get('cv_results', {})

    print(f"\n[*] Loaded parameters from: {params_file}")
    print(f"    hidden_units: {params['hidden_units']}")
    print(f"    learning_rate: {params['lr']:.6f}")
    print(f"    batch_size: {params['batch_size']}")
    print(f"    dropout: {params['dropout_rates'][1]:.4f}")
    print(f"    noise_std: {params['dropout_rates'][0]:.4f}")
    print(f"    loss_weights: {params['loss_weights']}")

    if cv_results:
        print(f"\n    Original CV results:")
        print(f"      Mean IC: {cv_results['mean_ic']:.4f} (±{cv_results['std_ic']:.4f})")
        if 'fold_results' in cv_results:
            for fold in cv_results['fold_results']:
                print(f"      {fold['name']}: IC={fold['ic']:.4f}, ICIR={fold['icir']:.4f}")

    return params, cv_results


def run_cv_evaluation(args, handler_config, symbols, model_path, use_mixed_precision=False):
    """加载预训练模型并在 CV folds 和2025测试集上评估 IC"""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION EVALUATION (AE-MLP)")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: valid {fold['valid_start']}~{fold['valid_end']}")
    print(f"Test Set: {FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']} (2025)")
    print("=" * 70)

    # 设置 GPU
    setup_gpu(args.gpu, use_mixed_precision)

    # 加载模型
    print(f"\n[*] Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print(f"    Model input shape: {model.input_shape}")
    print(f"    Model loaded successfully")

    # 预先准备2025测试集数据
    print("\n[*] Preparing 2025 test data for evaluation...")
    test_handler = create_data_handler_for_fold(args, handler_config, symbols, FINAL_TEST)
    test_dataset = create_dataset_for_fold(test_handler, FINAL_TEST)
    X_test, y_test, test_index = prepare_data_from_dataset(test_dataset, "test")
    print(f"    Test (2025): {X_test.shape}")

    fold_results = []
    fold_ics = []
    fold_test_ics = []
    batch_size = 4096  # 使用较大的 batch size 加速预测

    for fold_idx, fold in enumerate(CV_FOLDS):
        print(f"\n[*] Evaluating on {fold['name']}...")

        # 准备数据
        handler = create_data_handler_for_fold(args, handler_config, symbols, fold)
        dataset = create_dataset_for_fold(handler, fold)

        X_valid, y_valid, valid_index = prepare_data_from_dataset(dataset, "valid")

        print(f"    Valid: {X_valid.shape}")

        # 预测验证集
        outputs = model.predict(X_valid, batch_size=batch_size, verbose=0)

        # AE-MLP 有 3 个输出: [decoder, ae_action, action]
        if isinstance(outputs, list) and len(outputs) == 3:
            valid_pred = outputs[2].flatten()  # action 输出
        else:
            valid_pred = outputs.flatten()

        # 计算验证集 IC
        mean_ic, ic_std, icir = compute_ic(valid_pred, y_valid, valid_index)

        # ========== 2025 测试集评估 ==========
        test_outputs = model.predict(X_test, batch_size=batch_size, verbose=0)
        if isinstance(test_outputs, list) and len(test_outputs) == 3:
            test_pred = test_outputs[2].flatten()
        else:
            test_pred = test_outputs.flatten()
        test_ic, test_ic_std, test_icir = compute_ic(test_pred, y_test, test_index)

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

    # 汇总结果
    mean_ic_all = np.mean(fold_ics)
    std_ic_all = np.std(fold_ics)
    mean_test_ic_all = np.mean(fold_test_ics)
    std_test_ic_all = np.std(fold_test_ics)

    print("\n" + "=" * 70)
    print("CV EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Valid Mean IC: {mean_ic_all:.4f} (±{std_ic_all:.4f})")
    print(f"Test Mean IC (2025): {mean_test_ic_all:.4f} (±{std_test_ic_all:.4f})")
    print("\nIC by fold:")
    print(f"  {'Fold':<25s} {'Valid IC':>10s} {'Test IC':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    for r in fold_results:
        print(f"  {r['name']:<25s} {r['ic']:>10.4f} {r['test_ic']:>10.4f}")
    print("=" * 70)

    # 返回测试集预测（用于backtest）
    test_pred_series = pd.Series(test_pred, index=test_index, name='score')

    return fold_results, mean_ic_all, std_ic_all, test_pred_series, test_dataset


def run_cv_training(args, handler_config, symbols, params, use_mixed_precision=False):
    """运行 CV 训练以复现 IC，同时在2025测试集上评估"""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION TRAINING (AE-MLP)")
    print("=" * 70)
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: train {fold['train_start']}~{fold['train_end']}, "
              f"valid {fold['valid_start']}~{fold['valid_end']}")
    print(f"Test Set: {FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']} (2025)")
    print(f"Epochs: {args.cv_epochs}")
    print(f"Early stop patience: {args.cv_early_stop}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print("=" * 70)

    # 设置 GPU
    setup_gpu(args.gpu, use_mixed_precision)

    # 预先准备2025测试集数据（只需准备一次）
    print("\n[*] Preparing 2025 test data for evaluation...")
    test_handler = create_data_handler_for_fold(args, handler_config, symbols, FINAL_TEST)
    test_dataset = create_dataset_for_fold(test_handler, FINAL_TEST)
    X_test, y_test, test_index = prepare_data_from_dataset(test_dataset, "test")
    print(f"    Test (2025): {X_test.shape}")

    fold_results = []
    fold_ics = []
    fold_test_ics = []

    for fold_idx, fold in enumerate(CV_FOLDS):
        print(f"\n[*] Training {fold['name']}...")

        # 为每个 fold 设置随机种子（使用 base_seed + fold_idx 确保每个 fold 有不同但确定的种子）
        if args.seed is not None:
            set_random_seed(args.seed + fold_idx)

        # 准备数据
        handler = create_data_handler_for_fold(args, handler_config, symbols, fold)
        dataset = create_dataset_for_fold(handler, fold)

        X_train, y_train, _ = prepare_data_from_dataset(dataset, "train")
        X_valid, y_valid, valid_index = prepare_data_from_dataset(dataset, "valid")

        print(f"    Train: {X_train.shape}, Valid: {X_valid.shape}")

        # 更新 num_columns
        fold_params = params.copy()
        fold_params['num_columns'] = X_train.shape[1]

        # 清理并构建模型
        tf.keras.backend.clear_session()

        if use_mixed_precision:
            mixed_precision.set_global_policy('mixed_float16')

        model = build_ae_mlp_model(fold_params)

        # 创建数据集
        batch_size = params['batch_size']
        train_dataset = create_tf_dataset(X_train, y_train, batch_size, shuffle=True, prefetch=True)
        valid_dataset = create_tf_dataset(X_valid, y_valid, batch_size, shuffle=False, prefetch=True)

        # 回调
        cb_list = [
            callbacks.EarlyStopping(
                monitor='val_action_loss',
                patience=args.cv_early_stop,
                min_delta=1e-5,
                restore_best_weights=True,
                verbose=0,
                mode='min'
            ),
        ]

        # 训练
        history = model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=args.cv_epochs,
            callbacks=cb_list,
            verbose=1 if args.verbose else 0,
        )

        # 验证集预测
        _, _, valid_pred = model.predict(X_valid, batch_size=batch_size, verbose=0)
        valid_pred = valid_pred.flatten()

        # 计算验证集 IC
        mean_ic, ic_std, icir = compute_ic(valid_pred, y_valid, valid_index)

        # ========== 2025 测试集评估 ==========
        _, _, test_pred = model.predict(X_test, batch_size=batch_size, verbose=0)
        test_pred = test_pred.flatten()
        test_ic, test_ic_std, test_icir = compute_ic(test_pred, y_test, test_index)

        best_epoch = len(history.history['loss']) - args.cv_early_stop
        if best_epoch < 1:
            best_epoch = len(history.history['loss'])

        fold_ics.append(mean_ic)
        fold_test_ics.append(test_ic)
        fold_results.append({
            'name': fold['name'],
            'ic': mean_ic,
            'icir': icir,
            'test_ic': test_ic,
            'test_icir': test_icir,
            'best_epoch': best_epoch,
        })

        print(f"    {fold['name']}: Valid IC={mean_ic:.4f}, Test IC (2025)={test_ic:.4f}, epoch={best_epoch}")

    # 汇总结果
    mean_ic_all = np.mean(fold_ics)
    std_ic_all = np.std(fold_ics)
    mean_test_ic_all = np.mean(fold_test_ics)
    std_test_ic_all = np.std(fold_test_ics)

    print("\n" + "=" * 70)
    print("CV TRAINING COMPLETE")
    print("=" * 70)
    print(f"Valid Mean IC: {mean_ic_all:.4f} (±{std_ic_all:.4f})")
    print(f"Test Mean IC (2025): {mean_test_ic_all:.4f} (±{std_test_ic_all:.4f})")
    print("\nIC by fold:")
    print(f"  {'Fold':<25s} {'Valid IC':>10s} {'Test IC':>10s} {'Epoch':>8s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")
    for r in fold_results:
        print(f"  {r['name']:<25s} {r['ic']:>10.4f} {r['test_ic']:>10.4f} {r['best_epoch']:>8d}")
    print("=" * 70)

    return fold_results, mean_ic_all, std_ic_all


def train_final_model(args, handler_config, symbols, params, use_mixed_precision=False):
    """使用参数在完整数据上训练最终模型"""
    print("\n[*] Training final model on full data...")
    print("    Parameters:")
    print(f"      hidden_units: {params['hidden_units']}")
    print(f"      learning_rate: {params['lr']:.6f}")
    print(f"      batch_size: {params['batch_size']}")

    # 创建最终数据集
    handler = create_data_handler_for_fold(args, handler_config, symbols, FINAL_TEST)
    dataset = create_dataset_for_fold(handler, FINAL_TEST)

    X_train, y_train, _ = prepare_data_from_dataset(dataset, "train")
    X_valid, y_valid, _ = prepare_data_from_dataset(dataset, "valid")
    X_test, _, test_index = prepare_data_from_dataset(dataset, "test")

    print(f"\n    Final training data:")
    print(f"      Train: {X_train.shape} ({FINAL_TEST['train_start']} ~ {FINAL_TEST['train_end']})")
    print(f"      Valid: {X_valid.shape} ({FINAL_TEST['valid_start']} ~ {FINAL_TEST['valid_end']})")
    print(f"      Test:  {X_test.shape} ({FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']})")

    # 更新特征数
    final_params = params.copy()
    final_params['num_columns'] = X_train.shape[1]

    # 清理并构建模型
    tf.keras.backend.clear_session()

    # 设置随机种子
    if args.seed is not None:
        set_random_seed(args.seed + 100)  # 使用不同于 CV 的种子

    if use_mixed_precision:
        mixed_precision.set_global_policy('mixed_float16')

    model = build_ae_mlp_model(final_params)

    # 回调
    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_action_loss',
            patience=args.early_stop,
            min_delta=1e-5,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_action_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
            mode='min'
        ),
    ]

    # 创建数据集
    batch_size = params['batch_size']
    train_dataset = create_tf_dataset(X_train, y_train, batch_size, shuffle=True, prefetch=True)
    valid_dataset = create_tf_dataset(X_valid, y_valid, batch_size, shuffle=False, prefetch=True)

    # 训练
    print("\n    Training progress:")
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=args.n_epochs,
        callbacks=cb_list,
        verbose=1,
    )

    # 验证集 IC
    _, _, valid_pred = model.predict(X_valid, batch_size=batch_size, verbose=0)
    valid_pred = valid_pred.flatten()
    valid_ic, valid_ic_std, valid_icir = compute_ic(
        valid_pred, y_valid,
        dataset.prepare("valid", col_set="feature", data_key=DataHandlerLP.DK_L).index
    )
    print(f"\n    [Validation Set - for reference]")
    print(f"    Valid IC:   {valid_ic:.4f}")
    print(f"    Valid ICIR: {valid_icir:.4f}")

    # 测试集预测
    _, _, test_pred_values = model.predict(X_test, batch_size=batch_size, verbose=0)
    test_pred = pd.Series(test_pred_values.flatten(), index=test_index, name='score')

    print(f"\n    Test prediction shape: {test_pred.shape}")
    print(f"    Test prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    return model, test_pred, dataset


def main():
    parser = argparse.ArgumentParser(
        description='AE-MLP Cross-Validation Training with Loaded Parameters',
    )

    # 参数文件
    parser.add_argument('--params-file', type=str, default=None,
                        help='Path to JSON file with hyperparameters')

    # 模型评估模式
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to pre-trained model for evaluation (e.g., my_models/ae_mlp_cv_xxx.keras)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate pre-trained model on CV folds, no training')

    # 基础参数
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--handler', type=str, default='alpha158-talib-macro',
                        choices=list(HANDLER_CONFIG.keys()))
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])

    # CV 训练参数
    parser.add_argument('--cv-epochs', type=int, default=50,
                        help='Epochs per CV fold')
    parser.add_argument('--cv-early-stop', type=int, default=10,
                        help='Early stopping patience for CV folds')
    parser.add_argument('--cv-only', action='store_true',
                        help='Only run CV training, skip final model')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (try different seeds like 42, 123, 2024)')
    parser.add_argument('--num-seeds', type=int, default=1,
                        help='Run CV with multiple seeds and report best result (e.g., --num-seeds 5)')

    # 最终训练参数
    parser.add_argument('--n-epochs', type=int, default=100,
                        help='Epochs for final model training')
    parser.add_argument('--early-stop', type=int, default=10,
                        help='Early stopping patience for final model')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')

    # GPU 优化参数
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision (FP16) training')
    parser.add_argument('--verbose', action='store_true',
                        help='Show training progress for each fold')

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
        parser.error("--params-file is required for training mode")

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    # GPU 优化设置
    use_mixed_precision = args.mixed_precision and args.gpu >= 0

    # 初始化
    init_qlib(handler_config['use_talib'])

    # ========== 评估模式 ==========
    if args.eval_only:
        print("\n" + "=" * 70)
        print("AE-MLP Cross-Validation EVALUATION Mode")
        print("=" * 70)
        print(f"Model: {args.model_path}")
        print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
        print(f"Handler: {args.handler}")
        print(f"N-day: {args.nday}")
        print(f"GPU: {args.gpu}")
        print(f"Backtest: {'ON' if args.backtest else 'OFF'}")
        print("=" * 70)

        fold_results, mean_ic, std_ic, test_pred, test_dataset = run_cv_evaluation(
            args, handler_config, symbols, args.model_path,
            use_mixed_precision=use_mixed_precision
        )

        # 回测（评估模式）
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

            def load_model_func(path):
                return keras.models.load_model(str(path))

            def get_feature_count_func(m):
                return m.input_shape[1]

            run_backtest(
                args.model_path, test_dataset, pred_df, args, time_splits,
                model_name="AE-MLP (CV Eval)",
                load_model_func=load_model_func,
                get_feature_count_func=get_feature_count_func
            )

        return

    # ========== 训练模式 ==========
    # 加载超参数
    params, original_cv_results = load_params_from_json(args.params_file)

    # 打印头部
    print("\n" + "=" * 70)
    print("AE-MLP Cross-Validation Training")
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Handler: {args.handler}")
    print(f"N-day: {args.nday}")
    print(f"CV epochs: {args.cv_epochs}")
    print(f"CV Folds: {len(CV_FOLDS)}")
    print(f"GPU: {args.gpu}")
    print(f"Mixed precision: {'ON' if use_mixed_precision else 'OFF'}")
    if args.num_seeds > 1:
        print(f"Num seeds: {args.num_seeds}")
    print("=" * 70)

    # 运行 CV 训练 (支持多种子)
    if args.num_seeds > 1:
        # 多种子模式：尝试多个种子，找到最好的结果
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
                args, handler_config, symbols, params,
                use_mixed_precision=use_mixed_precision
            )
            # 计算测试集平均IC
            mean_test_ic = np.mean([r['test_ic'] for r in fold_results])
            all_results.append({
                'seed': seed,
                'mean_ic': mean_ic,
                'std_ic': std_ic,
                'mean_test_ic': mean_test_ic,
                'fold_results': fold_results,
            })

        # 找到最佳种子（基于验证集IC）
        best_result = max(all_results, key=lambda x: x['mean_ic'])
        # 也找出测试集最佳种子
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
            args, handler_config, symbols, params,
            use_mixed_precision=use_mixed_precision
        )

    # 比较结果
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
        args, handler_config, symbols, params,
        use_mixed_precision=use_mixed_precision
    )

    # 评估
    print("\n[*] Final Evaluation on Test Set (2025)...")
    evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

    # 保存模型
    print("\n[*] Saving model...")
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_SAVE_PATH / f"ae_mlp_cv_{args.handler}_{args.stock_pool}_{args.nday}d_{timestamp}.keras"
    model.save(str(model_path))
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

        def load_model(path):
            return keras.models.load_model(str(path))

        def get_feature_count(m):
            return m.input_shape[1]

        run_backtest(
            model_path, dataset, pred_df, args, time_splits,
            model_name="AE-MLP (CV)",
            load_model_func=load_model,
            get_feature_count_func=get_feature_count
        )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"CV Mean IC: {mean_ic:.4f} (±{std_ic:.4f})")
    print(f"Model saved to: {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
