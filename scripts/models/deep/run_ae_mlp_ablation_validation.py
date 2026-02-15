"""
AE-MLP 消融研究结论验证脚本

基于 ablation_study_improvements.txt 中的结论，验证以下改进是否有效:
1. Bottleneck 维度从 48 改为 8
2. 辅助任务权重从 0.15/0.115 降低为 0.01/0.01
3. 保留多任务结构
4. 保持特征拼接 (concat=True)

使用 CV 方式进行验证，对比 baseline 和 optimized 配置。

使用方法:
    # 完整验证 (baseline + optimized + 2025测试)
    python scripts/models/deep/run_ae_mlp_ablation_validation.py \
        --stock-pool sp500 --handler alpha158-enhanced-v7

    # 快速测试
    python scripts/models/deep/run_ae_mlp_ablation_validation.py \
        --stock-pool sp100 --handler alpha158 --cv-epochs 30

    # 指定随机种子
    python scripts/models/deep/run_ae_mlp_ablation_validation.py \
        --stock-pool sp500 --handler alpha158-enhanced-v7 --seed 42

    # 多种子验证 (更可靠)
    python scripts/models/deep/run_ae_mlp_ablation_validation.py \
        --stock-pool sp500 --handler alpha158-enhanced-v7 --num-seeds 3
"""

# ============================================================================
# 环境设置 (必须在其他导入之前)
# ============================================================================

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

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
# 其他导入
# ============================================================================

import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras import mixed_precision

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG, PROJECT_ROOT,
    init_qlib,
    CV_FOLDS,
    FINAL_TEST,
    create_data_handler_for_fold,
    create_dataset_for_fold,
    prepare_data_from_dataset,
    compute_ic,
)
from models.deep.ae_mlp_shared import build_ae_mlp_model, set_random_seed, create_tf_dataset


# ============================================================================
# 配置定义
# ============================================================================

# Baseline 配置 (来自 hyperopt CV 搜索)
BASELINE_CONFIG = {
    'name': 'Baseline (Original)',
    'description': 'Original hyperopt params: dim=48, weights=0.15/0.115',
    'encoder_dim': 48,
    'use_decoder_loss': True,
    'use_auxiliary_output': True,
    'use_concat': True,
    'decoder_weight': 0.151,
    'aux_weight': 0.115,
    'hidden_units': [48, 112, 256, 224, 48],
    'dropout_rates': [0.091, 0.148, 0.148, 0.148, 0.148, 0.148, 0.148],
    'lr': 0.00853,
    'batch_size': 4096,
}

# 优化配置 A: dim=8 + 低权重 (消融研究最佳配置)
OPTIMIZED_CONFIG_A = {
    'name': 'Optimized A (dim=8, w=0.01)',
    'description': 'Ablation best: dim=8, weights=0.01/0.01',
    'encoder_dim': 8,
    'use_decoder_loss': True,
    'use_auxiliary_output': True,
    'use_concat': True,
    'decoder_weight': 0.01,
    'aux_weight': 0.01,
    'hidden_units': [8, 112, 256, 224, 48],  # encoder_dim 调整为 8
    'dropout_rates': [0.091, 0.148, 0.148, 0.148, 0.148, 0.148, 0.148],
    'lr': 0.00853,
    'batch_size': 4096,
}

# 优化配置 B: dim=8 + 只保留 aux output (简化模型)
OPTIMIZED_CONFIG_B = {
    'name': 'Optimized B (dim=8, aux-only)',
    'description': 'Simplified: dim=8, aux only, w=0.01',
    'encoder_dim': 8,
    'use_decoder_loss': False,
    'use_auxiliary_output': True,
    'use_concat': True,
    'decoder_weight': 0.0,
    'aux_weight': 0.01,
    'hidden_units': [8, 112, 256, 224, 48],
    'dropout_rates': [0.091, 0.148, 0.148, 0.148, 0.148, 0.148, 0.148],
    'lr': 0.00853,
    'batch_size': 4096,
}

# 优化配置 C: dim=8 + 原始权重 (仅调整 bottleneck)
OPTIMIZED_CONFIG_C = {
    'name': 'Optimized C (dim=8, original weights)',
    'description': 'Only bottleneck change: dim=8, weights=0.15/0.115',
    'encoder_dim': 8,
    'use_decoder_loss': True,
    'use_auxiliary_output': True,
    'use_concat': True,
    'decoder_weight': 0.151,
    'aux_weight': 0.115,
    'hidden_units': [8, 112, 256, 224, 48],
    'dropout_rates': [0.091, 0.148, 0.148, 0.148, 0.148, 0.148, 0.148],
    'lr': 0.00853,
    'batch_size': 4096,
}

ALL_CONFIGS = [BASELINE_CONFIG, OPTIMIZED_CONFIG_A, OPTIMIZED_CONFIG_B, OPTIMIZED_CONFIG_C]


# ============================================================================
# 输出目录
# ============================================================================
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ablation_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CV 训练
# ============================================================================

def run_cv_for_config(args, handler_config, symbols, config, X_test, y_test, test_index, seed=None):
    """对单个配置运行 CV 训练"""
    fold_results = []
    fold_ics = []
    fold_test_ics = []

    for fold_idx, fold in enumerate(CV_FOLDS):
        # 设置种子
        if seed is not None:
            set_random_seed(seed + fold_idx)

        # 准备数据
        handler = create_data_handler_for_fold(args, handler_config, symbols, fold)
        dataset = create_dataset_for_fold(handler, fold)

        X_train, y_train, _ = prepare_data_from_dataset(dataset, "train")
        X_valid, y_valid, valid_index = prepare_data_from_dataset(dataset, "valid")

        num_columns = X_train.shape[1]

        # 清理并构建模型
        tf.keras.backend.clear_session()
        model, output_names = build_ae_mlp_model(config, num_columns)

        # 创建数据集
        batch_size = config['batch_size']
        train_dataset = create_tf_dataset(X_train, y_train, output_names, batch_size, shuffle=True)
        valid_dataset = create_tf_dataset(X_valid, y_valid, output_names, batch_size, shuffle=False)

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
        model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=args.cv_epochs,
            callbacks=cb_list,
            verbose=0,
        )

        # 验证集预测
        valid_outputs = model.predict(X_valid, batch_size=batch_size, verbose=0)
        if isinstance(valid_outputs, list):
            valid_pred = valid_outputs[-1].flatten()  # action 输出
        else:
            valid_pred = valid_outputs.flatten()

        # 计算验证集 IC
        mean_ic, ic_std, icir = compute_ic(valid_pred, y_valid, valid_index)

        # 2025 测试集评估
        test_outputs = model.predict(X_test, batch_size=batch_size, verbose=0)
        if isinstance(test_outputs, list):
            test_pred = test_outputs[-1].flatten()
        else:
            test_pred = test_outputs.flatten()
        test_ic, test_ic_std, test_icir = compute_ic(test_pred, y_test, test_index)

        fold_ics.append(mean_ic)
        fold_test_ics.append(test_ic)
        fold_results.append({
            'name': fold['name'],
            'valid_ic': mean_ic,
            'valid_icir': icir,
            'test_ic': test_ic,
            'test_icir': test_icir,
        })

        # 清理
        del model
        tf.keras.backend.clear_session()

    # 汇总
    mean_valid_ic = np.mean(fold_ics)
    std_valid_ic = np.std(fold_ics)
    mean_test_ic = np.mean(fold_test_ics)
    std_test_ic = np.std(fold_test_ics)

    return {
        'config_name': config['name'],
        'fold_results': fold_results,
        'mean_valid_ic': mean_valid_ic,
        'std_valid_ic': std_valid_ic,
        'mean_test_ic': mean_test_ic,
        'std_test_ic': std_test_ic,
    }


# ============================================================================
# 主函数
# ============================================================================

def run_validation(args):
    """运行消融结论验证"""
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"ablation_validation_{args.handler}_{args.stock_pool}_{timestamp}.txt"

    # 打印头部
    print("\n" + "=" * 80)
    print("AE-MLP Ablation Conclusion Validation")
    print("=" * 80)
    print(f"Handler: {args.handler}")
    print(f"Stock pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"CV Folds: {len(CV_FOLDS)}")
    print(f"CV Epochs: {args.cv_epochs}")
    print(f"Early stop: {args.cv_early_stop}")
    print(f"Test period: {FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    if args.num_seeds > 1:
        print(f"Num seeds: {args.num_seeds}")
    print("=" * 80)

    print("\nConfigurations to validate:")
    for i, config in enumerate(ALL_CONFIGS):
        print(f"  {i+1}. {config['name']}")
        print(f"     {config['description']}")
    print()

    # 设置 GPU
    if args.gpu >= 0 and gpus:
        try:
            tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
            print(f"Using GPU: {gpus[args.gpu]}")
        except RuntimeError:
            pass
    else:
        tf.config.set_visible_devices([], 'GPU')
        print("Using CPU")

    # 预先准备 2025 测试集数据
    print("\n[*] Preparing 2025 test data...")
    test_handler = create_data_handler_for_fold(args, handler_config, symbols, FINAL_TEST)
    test_dataset = create_dataset_for_fold(test_handler, FINAL_TEST)
    X_test, y_test, test_index = prepare_data_from_dataset(test_dataset, "test")
    print(f"    Test (2025): {X_test.shape}")

    # 运行验证
    all_results = []

    if args.num_seeds > 1:
        # 多种子模式
        base_seed = args.seed if args.seed is not None else 42

        for config_idx, config in enumerate(ALL_CONFIGS):
            print(f"\n{'='*70}")
            print(f"Config {config_idx+1}/{len(ALL_CONFIGS)}: {config['name']}")
            print(f"{'='*70}")

            seed_results = []
            for seed_idx in range(args.num_seeds):
                seed = base_seed + seed_idx * 1000
                print(f"\n  Seed {seed_idx+1}/{args.num_seeds} (seed={seed})...", end=" ", flush=True)

                result = run_cv_for_config(
                    args, handler_config, symbols, config,
                    X_test, y_test, test_index, seed=seed
                )
                seed_results.append(result)
                print(f"Valid IC={result['mean_valid_ic']:.4f}, Test IC={result['mean_test_ic']:.4f}")

            # 取平均
            avg_valid_ic = np.mean([r['mean_valid_ic'] for r in seed_results])
            std_valid_ic = np.std([r['mean_valid_ic'] for r in seed_results])
            avg_test_ic = np.mean([r['mean_test_ic'] for r in seed_results])
            std_test_ic = np.std([r['mean_test_ic'] for r in seed_results])

            all_results.append({
                'config': config,
                'avg_valid_ic': avg_valid_ic,
                'std_valid_ic': std_valid_ic,
                'avg_test_ic': avg_test_ic,
                'std_test_ic': std_test_ic,
                'seed_results': seed_results,
            })

            print(f"\n  Average: Valid IC={avg_valid_ic:.4f} (+-{std_valid_ic:.4f}), "
                  f"Test IC={avg_test_ic:.4f} (+-{std_test_ic:.4f})")
    else:
        # 单种子模式
        seed = args.seed if args.seed is not None else 42

        for config_idx, config in enumerate(ALL_CONFIGS):
            print(f"\n{'='*70}")
            print(f"Config {config_idx+1}/{len(ALL_CONFIGS)}: {config['name']}")
            print(f"{'='*70}")
            print(f"  {config['description']}")

            result = run_cv_for_config(
                args, handler_config, symbols, config,
                X_test, y_test, test_index, seed=seed
            )

            all_results.append({
                'config': config,
                'avg_valid_ic': result['mean_valid_ic'],
                'std_valid_ic': result['std_valid_ic'],
                'avg_test_ic': result['mean_test_ic'],
                'std_test_ic': result['std_test_ic'],
                'fold_results': result['fold_results'],
            })

            print(f"\n  Results:")
            for fold_r in result['fold_results']:
                print(f"    {fold_r['name']}: Valid IC={fold_r['valid_ic']:.4f}, Test IC={fold_r['test_ic']:.4f}")
            print(f"\n  Mean: Valid IC={result['mean_valid_ic']:.4f} (+-{result['std_valid_ic']:.4f}), "
                  f"Test IC={result['mean_test_ic']:.4f} (+-{result['std_test_ic']:.4f})")

    # ========================================================================
    # 写入结果
    # ========================================================================
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("AE-MLP Ablation Conclusion Validation Results\n")
        f.write("=" * 100 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Handler: {args.handler}\n")
        f.write(f"Stock pool: {args.stock_pool} ({len(symbols)} stocks)\n")
        f.write(f"CV Folds: {len(CV_FOLDS)}\n")
        f.write(f"CV Epochs: {args.cv_epochs}\n")
        f.write(f"Test period: {FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']}\n")
        if args.num_seeds > 1:
            f.write(f"Num seeds: {args.num_seeds}\n")
        f.write("=" * 100 + "\n\n")

        # 配置详情
        f.write("CONFIGURATIONS TESTED:\n")
        f.write("-" * 80 + "\n\n")
        for i, config in enumerate(ALL_CONFIGS):
            f.write(f"{i+1}. {config['name']}\n")
            f.write(f"   Description: {config['description']}\n")
            f.write(f"   encoder_dim: {config['encoder_dim']}\n")
            f.write(f"   use_decoder_loss: {config['use_decoder_loss']}\n")
            f.write(f"   use_auxiliary_output: {config['use_auxiliary_output']}\n")
            f.write(f"   use_concat: {config['use_concat']}\n")
            f.write(f"   decoder_weight: {config['decoder_weight']}\n")
            f.write(f"   aux_weight: {config['aux_weight']}\n\n")

        # 结果摘要
        f.write("\n" + "=" * 100 + "\n")
        f.write("RESULTS SUMMARY\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"{'Config':<40s} {'Valid IC':>12s} {'Test IC':>12s} {'Valid vs Baseline':>18s} {'Test vs Baseline':>18s}\n")
        f.write("-" * 100 + "\n")

        baseline_valid = all_results[0]['avg_valid_ic']
        baseline_test = all_results[0]['avg_test_ic']

        for r in all_results:
            config = r['config']
            valid_diff = r['avg_valid_ic'] - baseline_valid
            test_diff = r['avg_test_ic'] - baseline_test

            valid_pct = (valid_diff / abs(baseline_valid) * 100) if baseline_valid != 0 else 0
            test_pct = (test_diff / abs(baseline_test) * 100) if baseline_test != 0 else 0

            valid_str = f"{r['avg_valid_ic']:.4f} (+-{r['std_valid_ic']:.4f})"
            test_str = f"{r['avg_test_ic']:.4f} (+-{r['std_test_ic']:.4f})"

            if config['name'] == 'Baseline (Original)':
                valid_change = "-"
                test_change = "-"
            else:
                valid_change = f"{valid_diff:+.4f} ({valid_pct:+.1f}%)"
                test_change = f"{test_diff:+.4f} ({test_pct:+.1f}%)"

            f.write(f"{config['name']:<40s} {valid_str:>12s} {test_str:>12s} {valid_change:>18s} {test_change:>18s}\n")

        # 结论
        f.write("\n" + "=" * 100 + "\n")
        f.write("VALIDATION CONCLUSIONS\n")
        f.write("=" * 100 + "\n\n")

        # 找最佳配置
        best_valid = max(all_results, key=lambda x: x['avg_valid_ic'])
        best_test = max(all_results, key=lambda x: x['avg_test_ic'])

        f.write(f"Best Valid IC: {best_valid['config']['name']}\n")
        f.write(f"  Valid IC: {best_valid['avg_valid_ic']:.4f}\n")
        f.write(f"  Test IC:  {best_valid['avg_test_ic']:.4f}\n\n")

        f.write(f"Best Test IC: {best_test['config']['name']}\n")
        f.write(f"  Valid IC: {best_test['avg_valid_ic']:.4f}\n")
        f.write(f"  Test IC:  {best_test['avg_test_ic']:.4f}\n\n")

        # 验证各项结论
        f.write("Ablation conclusions validation:\n")
        f.write("-" * 80 + "\n\n")

        # 结论1: dim=8 vs dim=48
        baseline_r = all_results[0]  # Baseline
        dim8_r = all_results[2]  # Optimized C (only dim change)

        f.write("1. Bottleneck dimension (8 vs 48):\n")
        f.write(f"   Baseline (dim=48): Valid IC={baseline_r['avg_valid_ic']:.4f}, Test IC={baseline_r['avg_test_ic']:.4f}\n")
        f.write(f"   dim=8:             Valid IC={dim8_r['avg_valid_ic']:.4f}, Test IC={dim8_r['avg_test_ic']:.4f}\n")
        valid_improvement = dim8_r['avg_valid_ic'] - baseline_r['avg_valid_ic']
        test_improvement = dim8_r['avg_test_ic'] - baseline_r['avg_test_ic']
        f.write(f"   Change: Valid {valid_improvement:+.4f}, Test {test_improvement:+.4f}\n")
        f.write(f"   Conclusion: {'CONFIRMED' if valid_improvement > 0 or test_improvement > 0 else 'NOT CONFIRMED'}\n\n")

        # 结论2: 低权重
        optimized_a = all_results[1]  # dim=8, w=0.01
        f.write("2. Lower auxiliary weights (0.01 vs 0.15):\n")
        f.write(f"   dim=8, w=0.15: Valid IC={dim8_r['avg_valid_ic']:.4f}, Test IC={dim8_r['avg_test_ic']:.4f}\n")
        f.write(f"   dim=8, w=0.01: Valid IC={optimized_a['avg_valid_ic']:.4f}, Test IC={optimized_a['avg_test_ic']:.4f}\n")
        valid_improvement = optimized_a['avg_valid_ic'] - dim8_r['avg_valid_ic']
        test_improvement = optimized_a['avg_test_ic'] - dim8_r['avg_test_ic']
        f.write(f"   Change: Valid {valid_improvement:+.4f}, Test {test_improvement:+.4f}\n")
        f.write(f"   Conclusion: {'CONFIRMED' if valid_improvement > 0 or test_improvement > 0 else 'NOT CONFIRMED'}\n\n")

        # 结论3: 简化模型
        optimized_b = all_results[2]  # aux-only
        f.write("3. Simplified model (aux-only vs full):\n")
        f.write(f"   Full (dim=8, w=0.01): Valid IC={optimized_a['avg_valid_ic']:.4f}, Test IC={optimized_a['avg_test_ic']:.4f}\n")
        f.write(f"   Aux-only:            Valid IC={optimized_b['avg_valid_ic']:.4f}, Test IC={optimized_b['avg_test_ic']:.4f}\n")
        f.write(f"   Note: Simplified model may be comparable while reducing complexity\n\n")

        # 最终推荐
        f.write("\n" + "=" * 100 + "\n")
        f.write("RECOMMENDED CONFIGURATION\n")
        f.write("=" * 100 + "\n\n")

        # 选择 test IC 最高的配置
        f.write(f"Based on Test IC (2025), recommended: {best_test['config']['name']}\n")
        f.write(f"  encoder_dim: {best_test['config']['encoder_dim']}\n")
        f.write(f"  decoder_weight: {best_test['config']['decoder_weight']}\n")
        f.write(f"  aux_weight: {best_test['config']['aux_weight']}\n")
        f.write(f"  Test IC: {best_test['avg_test_ic']:.4f}\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write(f"Results saved to: {output_file}\n")
        f.write("=" * 100 + "\n")

    # ========================================================================
    # 打印最终结果
    # ========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    print(f"\n{'Config':<40s} {'Valid IC':>14s} {'Test IC':>14s}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['config']['name']:<40s} {r['avg_valid_ic']:>10.4f} (+-{r['std_valid_ic']:.3f}) "
              f"{r['avg_test_ic']:>10.4f} (+-{r['std_test_ic']:.3f})")

    print(f"\nBest Valid IC: {best_valid['config']['name']} ({best_valid['avg_valid_ic']:.4f})")
    print(f"Best Test IC:  {best_test['config']['name']} ({best_test['avg_test_ic']:.4f})")
    print(f"\nResults saved to: {output_file}")
    print("=" * 80)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Validate AE-MLP ablation study conclusions'
    )

    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--handler', type=str, default='alpha158-enhanced-v7',
                        choices=list(HANDLER_CONFIG.keys()))
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])

    parser.add_argument('--cv-epochs', type=int, default=50,
                        help='Epochs per CV fold')
    parser.add_argument('--cv-early-stop', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--num-seeds', type=int, default=1,
                        help='Number of seeds to average over')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')

    args = parser.parse_args()

    run_validation(args)


if __name__ == "__main__":
    main()
