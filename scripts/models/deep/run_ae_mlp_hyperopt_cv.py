"""
AE-MLP 超参数搜索 - 时间序列交叉验证版本

使用多个时间窗口进行交叉验证，选出更稳健的超参数。
避免在单一验证集上过拟合。

时间窗口设计:
  Fold 1: train 2000-2021, valid 2022
  Fold 2: train 2000-2022, valid 2023
  Fold 3: train 2000-2023, valid 2024
  Test:   2025 (完全独立，不参与超参数选择)

使用方法:
    python scripts/models/deep/run_ae_mlp_hyperopt_cv.py
    python scripts/models/deep/run_ae_mlp_hyperopt_cv.py --max-evals 50
    python scripts/models/deep/run_ae_mlp_hyperopt_cv.py --stock-pool sp100 --backtest
"""

# ============================================================================
# 重要: 以下代码必须在任何其他导入之前执行
# ============================================================================

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 抑制 TensorFlow 日志
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # 优化 GPU 线程
# 注意: XLA 和混合精度可能导致数值不稳定，默认禁用
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'  # 启用 XLA JIT 编译

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
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.pyll import scope

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras import mixed_precision

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
    CV_FOLDS,
    FINAL_TEST,
    create_data_handler_for_fold,
    create_dataset_for_fold,
    prepare_data_from_dataset,
    compute_ic,
)

from models.deep.ae_mlp_model import AEMLP


# ============================================================================
# 超参数搜索空间
# ============================================================================

SEARCH_SPACE = {
    # 网络结构
    'encoder_dim': scope.int(hp.quniform('encoder_dim', 32, 128, 16)),
    'decoder_hidden': scope.int(hp.quniform('decoder_hidden', 32, 128, 16)),
    'main_layer1': scope.int(hp.quniform('main_layer1', 128, 512, 64)),
    'main_layer2': scope.int(hp.quniform('main_layer2', 64, 256, 32)),
    'main_layer3': scope.int(hp.quniform('main_layer3', 32, 128, 16)),

    # Dropout
    'noise_std': hp.uniform('noise_std', 0.01, 0.1),
    'dropout': hp.uniform('dropout', 0.01, 0.15),

    # 学习率
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),

    # Batch size - 默认保守值，可通过 --max-batch-size 调整
    'batch_size': hp.choice('batch_size', [1024, 2048, 4096]),

    # 损失权重
    'loss_decoder': hp.uniform('loss_decoder', 0.01, 0.3),
    'loss_ae': hp.uniform('loss_ae', 0.01, 0.3),
}


def create_ae_mlp_params(hyperparams: dict, num_columns: int) -> dict:
    """将 hyperopt 参数转换为 AE-MLP 参数"""
    hidden_units = [
        int(hyperparams['encoder_dim']),
        int(hyperparams['decoder_hidden']),
        int(hyperparams['main_layer1']),
        int(hyperparams['main_layer2']),
        int(hyperparams['main_layer3']),
    ]

    dropout_rates = [
        hyperparams['noise_std'],       # GaussianNoise
        hyperparams['dropout'],         # encoder dropout
        hyperparams['dropout'],         # decoder dropout
        hyperparams['dropout'],         # concat dropout
        hyperparams['dropout'],         # main layer dropouts
        hyperparams['dropout'],
        hyperparams['dropout'],
    ]

    loss_weights = {
        'decoder': hyperparams['loss_decoder'],
        'ae_action': hyperparams['loss_ae'],
        'action': 1.0,
    }

    return {
        'num_columns': num_columns,
        'hidden_units': hidden_units,
        'dropout_rates': dropout_rates,
        'lr': hyperparams['learning_rate'],
        'batch_size': hyperparams['batch_size'],
        'loss_weights': loss_weights,
    }


def create_tf_dataset(X, y, batch_size, shuffle=True, prefetch=True):
    """
    创建优化的 tf.data.Dataset。

    使用 prefetch 提升数据加载效率，减少 CPU-GPU 数据传输瓶颈。
    注意: 不使用 cache() 以避免大数据集时的内存问题。
    """
    # 创建多输出的目标字典
    outputs = {
        'decoder': X,
        'ae_action': y,
        'action': y,
    }

    dataset = tf.data.Dataset.from_tensor_slices((X, outputs))

    if shuffle:
        # 使用较小的 buffer_size 以减少内存占用
        dataset = dataset.shuffle(buffer_size=min(len(X), 50000))

    dataset = dataset.batch(batch_size)

    if prefetch:
        # prefetch(tf.data.AUTOTUNE) 让 TensorFlow 自动决定预取数量
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def build_ae_mlp_model(params: dict) -> Model:
    """构建 AE-MLP 模型"""
    num_columns = params['num_columns']
    hidden_units = params['hidden_units']
    dropout_rates = params['dropout_rates']
    lr = params['lr']
    loss_weights = params['loss_weights']

    inp = layers.Input(shape=(num_columns,), name='input')

    # 输入标准化
    x0 = layers.BatchNormalization(name='input_bn')(inp)

    # Encoder
    encoder = layers.GaussianNoise(dropout_rates[0], name='noise')(x0)
    encoder = layers.Dense(hidden_units[0], name='encoder_dense')(encoder)
    encoder = layers.BatchNormalization(name='encoder_bn')(encoder)
    encoder = layers.Activation('swish', name='encoder_act')(encoder)

    # Decoder (重建原始输入)
    decoder = layers.Dropout(dropout_rates[1], name='decoder_dropout')(encoder)
    decoder = layers.Dense(num_columns, dtype='float32', name='decoder')(decoder)

    # 辅助预测分支 (基于 decoder 输出)
    x_ae = layers.Dense(hidden_units[1], name='ae_dense1')(decoder)
    x_ae = layers.BatchNormalization(name='ae_bn1')(x_ae)
    x_ae = layers.Activation('swish', name='ae_act1')(x_ae)
    x_ae = layers.Dropout(dropout_rates[2], name='ae_dropout1')(x_ae)
    out_ae = layers.Dense(1, dtype='float32', name='ae_action')(x_ae)

    # 主分支: 原始特征 + encoder 特征
    x = layers.Concatenate(name='concat')([x0, encoder])
    x = layers.BatchNormalization(name='main_bn0')(x)
    x = layers.Dropout(dropout_rates[3], name='main_dropout0')(x)

    # MLP 主体
    for i in range(2, len(hidden_units)):
        dropout_idx = min(i + 2, len(dropout_rates) - 1)
        x = layers.Dense(hidden_units[i], name=f'main_dense{i-1}')(x)
        x = layers.BatchNormalization(name=f'main_bn{i-1}')(x)
        x = layers.Activation('swish', name=f'main_act{i-1}')(x)
        x = layers.Dropout(dropout_rates[dropout_idx], name=f'main_dropout{i-1}')(x)

    # 主输出 (使用 float32 确保数值稳定性)
    out = layers.Dense(1, dtype='float32', name='action')(x)

    model = Model(inputs=inp, outputs=[decoder, out_ae, out], name='AE_MLP')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss={
            'decoder': 'mse',
            'ae_action': 'mse',
            'action': 'mse',
        },
        loss_weights=loss_weights,
    )

    return model


class CVHyperoptObjective:
    """时间序列交叉验证的 Hyperopt 目标函数"""

    def __init__(self, args, handler_config, symbols, n_epochs=30, early_stop=5, gpu=0,
                 use_mixed_precision=False, prefetch_to_gpu=True, max_batch_size=4096):
        self.args = args
        self.handler_config = handler_config
        self.symbols = symbols
        self.n_epochs = n_epochs
        self.early_stop = early_stop
        self.gpu = gpu
        self.use_mixed_precision = use_mixed_precision
        self.prefetch_to_gpu = prefetch_to_gpu
        self.trial_count = 0
        self.best_mean_ic = -float('inf')

        # 根据 max_batch_size 生成 batch size 选项
        all_batch_sizes = [512, 1024, 2048, 4096, 8192, 16384]
        self.batch_size_choices = [b for b in all_batch_sizes if b <= max_batch_size]
        if not self.batch_size_choices:
            self.batch_size_choices = [512]  # 最小值
        print(f"    Batch size choices: {self.batch_size_choices}")

        # 设置 GPU (在数据准备之前，以便正确配置混合精度)
        self._setup_gpu()

        # 预先准备所有 fold 的数据
        print("\n[*] Preparing data for all CV folds...")
        self.fold_data = []
        self.num_columns = None

        for fold in CV_FOLDS:
            print(f"    Preparing {fold['name']}...")
            handler = create_data_handler_for_fold(args, handler_config, symbols, fold)
            dataset = create_dataset_for_fold(handler, fold)

            X_train, y_train, _ = prepare_data_from_dataset(dataset, "train")
            X_valid, y_valid, valid_index = prepare_data_from_dataset(dataset, "valid")

            if self.num_columns is None:
                self.num_columns = X_train.shape[1]

            self.fold_data.append({
                'name': fold['name'],
                'X_train': X_train,
                'y_train': y_train,
                'X_valid': X_valid,
                'y_valid': y_valid,
                'valid_index': valid_index,
            })

            print(f"      Train: {X_train.shape}, Valid: {X_valid.shape}")

        print(f"    ✓ All {len(CV_FOLDS)} folds prepared")
        print(f"    Feature count: {self.num_columns}")

    def _setup_gpu(self):
        """配置 GPU，支持混合精度"""
        gpus = tf.config.list_physical_devices('GPU')
        if self.gpu >= 0 and gpus:
            try:
                tf.config.set_visible_devices(gpus[self.gpu], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[self.gpu], True)
                print(f"    Using GPU: {gpus[self.gpu]}")

                # 启用混合精度训练 (FP16) - 大幅提升 GPU 利用率
                if self.use_mixed_precision:
                    mixed_precision.set_global_policy('mixed_float16')
                    print("    Mixed precision (FP16) enabled")
            except RuntimeError as e:
                print(f"    GPU setup error: {e}")
        else:
            tf.config.set_visible_devices([], 'GPU')
            print("    Using CPU")

    def __call__(self, hyperparams):
        """目标函数: 在所有 fold 上训练并返回平均验证集 IC"""
        self.trial_count += 1

        # 转换参数
        model_params = create_ae_mlp_params(hyperparams, self.num_columns)
        # 使用动态 batch_size_choices 替换静态值
        batch_size_idx = hyperparams['batch_size']
        batch_size = self.batch_size_choices[batch_size_idx % len(self.batch_size_choices)]
        model_params['batch_size'] = batch_size

        fold_ics = []
        fold_results = []

        try:
            for fold in self.fold_data:
                # 清理之前的模型
                tf.keras.backend.clear_session()

                # 重新设置混合精度 (clear_session 会重置)
                if self.use_mixed_precision:
                    mixed_precision.set_global_policy('mixed_float16')

                # 构建模型
                model = build_ae_mlp_model(model_params)

                # 使用 tf.data.Dataset 优化数据加载
                train_dataset = create_tf_dataset(
                    fold['X_train'], fold['y_train'],
                    batch_size=batch_size, shuffle=True, prefetch=True
                )
                valid_dataset = create_tf_dataset(
                    fold['X_valid'], fold['y_valid'],
                    batch_size=batch_size, shuffle=False, prefetch=True
                )

                # 回调
                cb_list = [
                    callbacks.EarlyStopping(
                        monitor='val_action_loss',
                        patience=self.early_stop,
                        min_delta=1e-5,
                        restore_best_weights=True,
                        verbose=0,
                        mode='min'
                    ),
                ]

                # 训练 (使用 Dataset API)
                history = model.fit(
                    train_dataset,
                    validation_data=valid_dataset,
                    epochs=self.n_epochs,
                    callbacks=cb_list,
                    verbose=0,
                )

                # 验证集预测
                _, _, valid_pred = model.predict(fold['X_valid'], batch_size=batch_size, verbose=0)
                valid_pred = valid_pred.flatten()

                # 计算 IC
                mean_ic, ic_std, icir = compute_ic(
                    valid_pred, fold['y_valid'], fold['valid_index']
                )

                best_epoch = len(history.history['loss']) - self.early_stop
                if best_epoch < 1:
                    best_epoch = len(history.history['loss'])

                fold_ics.append(mean_ic)
                fold_results.append({
                    'name': fold['name'],
                    'ic': mean_ic,
                    'icir': icir,
                    'best_epoch': best_epoch,
                })

            # 计算平均 IC
            mean_ic_all = np.mean(fold_ics)
            std_ic_all = np.std(fold_ics)

            # 更新最佳
            if mean_ic_all > self.best_mean_ic:
                self.best_mean_ic = mean_ic_all
                is_best = " ★ NEW BEST"
            else:
                is_best = ""

            # 打印进度
            fold_ic_str = ", ".join([f"{r['ic']:.4f}" for r in fold_results])
            print(f"  Trial {self.trial_count:3d}: Mean IC={mean_ic_all:.4f} (±{std_ic_all:.4f}) "
                  f"[{fold_ic_str}] lr={hyperparams['learning_rate']:.5f}{is_best}")

            return {
                'loss': -mean_ic_all,
                'status': STATUS_OK,
                'mean_ic': mean_ic_all,
                'std_ic': std_ic_all,
                'fold_results': fold_results,
                'params': model_params,
            }

        except Exception as e:
            print(f"  Trial {self.trial_count:3d}: FAILED - {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'loss': float('inf'),
                'status': STATUS_FAIL,
                'error': str(e),
            }


def run_hyperopt_cv_search(args, handler_config, symbols, use_mixed_precision=False,
                           use_prefetch=True, max_batch_size=4096):
    """运行时间序列交叉验证的超参数搜索"""
    print("\n" + "=" * 70)
    print("HYPEROPT SEARCH WITH TIME-SERIES CROSS-VALIDATION (AE-MLP)")
    print("=" * 70)
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: train {fold['train_start']}~{fold['train_end']}, "
              f"valid {fold['valid_start']}~{fold['valid_end']}")
    print(f"Max evaluations: {args.max_evals}")
    print(f"Epochs per trial: {args.cv_epochs}")
    print(f"Max batch size: {max_batch_size}")
    print("=" * 70)

    # 创建目标函数
    objective = CVHyperoptObjective(
        args, handler_config, symbols,
        n_epochs=args.cv_epochs,
        early_stop=args.cv_early_stop,
        gpu=args.gpu,
        use_mixed_precision=use_mixed_precision,
        prefetch_to_gpu=use_prefetch,
        max_batch_size=max_batch_size
    )

    # 运行搜索
    trials = Trials()
    print("\n[*] Running hyperparameter search...")

    best = fmin(
        fn=objective,
        space=SEARCH_SPACE,
        algo=tpe.suggest,
        max_evals=args.max_evals,
        trials=trials,
        show_progressbar=False,
    )

    # 获取最佳结果
    best_params = create_ae_mlp_params(best, objective.num_columns)
    # 需要处理 hp.choice 的 batch_size
    batch_size_choices = objective.batch_size_choices
    best_params['batch_size'] = batch_size_choices[best['batch_size']]

    best_trial_idx = np.argmin([t['result']['loss'] for t in trials.trials])
    best_trial = trials.trials[best_trial_idx]['result']

    print("\n" + "=" * 70)
    print("CV HYPEROPT SEARCH COMPLETE")
    print("=" * 70)
    print(f"Best Mean IC: {best_trial['mean_ic']:.4f} (±{best_trial['std_ic']:.4f})")
    print("\nIC by fold:")
    for r in best_trial['fold_results']:
        print(f"  {r['name']}: IC={r['ic']:.4f}, ICIR={r['icir']:.4f}, epoch={r['best_epoch']}")
    print("\nBest parameters:")
    print(f"  hidden_units: {best_params['hidden_units']}")
    print(f"  learning_rate: {best_params['lr']:.6f}")
    print(f"  batch_size: {best_params['batch_size']}")
    print(f"  dropout: {best_params['dropout_rates'][1]:.4f}")
    print(f"  noise_std: {best_params['dropout_rates'][0]:.4f}")
    print(f"  loss_weights: {best_params['loss_weights']}")
    print("=" * 70)

    return best_params, trials, best_trial, objective.num_columns


def train_final_model(args, handler_config, symbols, best_params, use_mixed_precision=False):
    """使用最优参数在完整数据上训练最终模型"""
    print("\n[*] Training final model on full data...")
    print("    Parameters:")
    print(f"      hidden_units: {best_params['hidden_units']}")
    print(f"      learning_rate: {best_params['lr']:.6f}")
    print(f"      batch_size: {best_params['batch_size']}")

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
    best_params['num_columns'] = X_train.shape[1]

    # 清理并构建模型
    tf.keras.backend.clear_session()

    # 重新设置混合精度 (clear_session 会重置)
    if use_mixed_precision:
        mixed_precision.set_global_policy('mixed_float16')

    model = build_ae_mlp_model(best_params)

    # 回调
    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_action_loss',
            patience=args.early_stop,
            min_delta=1e-5,  # 最小改善阈值，避免微小改善导致训练过长
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

    # 使用 tf.data.Dataset 优化数据加载
    train_dataset = create_tf_dataset(
        X_train, y_train,
        batch_size=best_params['batch_size'], shuffle=True, prefetch=True
    )
    valid_dataset = create_tf_dataset(
        X_valid, y_valid,
        batch_size=best_params['batch_size'], shuffle=False, prefetch=True
    )

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
    _, _, valid_pred = model.predict(X_valid, batch_size=best_params['batch_size'], verbose=0)
    valid_pred = valid_pred.flatten()
    valid_ic, valid_ic_std, valid_icir = compute_ic(
        valid_pred, y_valid,
        dataset.prepare("valid", col_set="feature", data_key=DataHandlerLP.DK_L).index
    )
    print(f"\n    [Validation Set - for reference]")
    print(f"    Valid IC:   {valid_ic:.4f}")
    print(f"    Valid ICIR: {valid_icir:.4f}")

    # 测试集预测
    _, _, test_pred_values = model.predict(X_test, batch_size=best_params['batch_size'], verbose=0)
    test_pred = pd.Series(test_pred_values.flatten(), index=test_index, name='score')

    print(f"\n    Test prediction shape: {test_pred.shape}")
    print(f"    Test prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    return model, test_pred, dataset


def main():
    parser = argparse.ArgumentParser(
        description='AE-MLP Hyperopt with Time-Series Cross-Validation',
    )

    # 基础参数
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--handler', type=str, default='alpha158-talib-macro',
                        choices=list(HANDLER_CONFIG.keys()))
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])

    # Hyperopt 参数
    parser.add_argument('--max-evals', type=int, default=50,
                        help='Maximum number of hyperopt evaluations')
    parser.add_argument('--cv-epochs', type=int, default=50,
                        help='Epochs per CV trial (smaller for faster search)')
    parser.add_argument('--cv-early-stop', type=int, default=10,
                        help='Early stopping patience for CV trials')

    # 最终训练参数
    parser.add_argument('--n-epochs', type=int, default=100,
                        help='Epochs for final model training')
    parser.add_argument('--early-stop', type=int, default=10,
                        help='Early stopping patience for final model')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')

    # GPU 优化参数
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision (FP16) training (may cause numerical instability)')
    parser.add_argument('--no-prefetch', action='store_true',
                        help='Disable tf.data prefetching')
    parser.add_argument('--max-batch-size', type=int, default=4096,
                        help='Maximum batch size (reduce if OOM, default: 4096)')

    # 回测参数
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=1)
    parser.add_argument('--account', type=float, default=10000)
    parser.add_argument('--rebalance-freq', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss'])

    args = parser.parse_args()

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    # GPU 优化设置
    use_mixed_precision = args.mixed_precision and args.gpu >= 0
    use_prefetch = not args.no_prefetch

    # 打印头部
    print("=" * 70)
    print("AE-MLP Hyperopt with Time-Series Cross-Validation")
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Handler: {args.handler}")
    print(f"N-day: {args.nday}")
    print(f"Max evaluations: {args.max_evals}")
    print(f"CV epochs: {args.cv_epochs}")
    print(f"CV Folds: {len(CV_FOLDS)}")
    print(f"GPU: {args.gpu}")
    print(f"GPU Optimizations:")
    print(f"  - Mixed precision (FP16): {'ON' if use_mixed_precision else 'OFF'}")
    print(f"  - tf.data prefetch: {'ON' if use_prefetch else 'OFF'}")
    print(f"  - XLA JIT compilation: OFF (disabled for numerical stability)")
    print(f"  - Max batch size: {args.max_batch_size} (use --max-batch-size to adjust)")
    print("=" * 70)

    # 初始化
    init_qlib(handler_config['use_talib'])

    # 运行 CV 超参数搜索
    best_params, trials, best_trial, num_columns = run_hyperopt_cv_search(
        args, handler_config, symbols,
        use_mixed_precision=use_mixed_precision,
        use_prefetch=use_prefetch,
        max_batch_size=args.max_batch_size
    )

    # 保存搜索结果
    output_dir = PROJECT_ROOT / "outputs" / "hyperopt_cv"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存最佳参数
    params_file = output_dir / f"ae_mlp_cv_best_params_{timestamp}.json"
    params_to_save = {
        'params': {
            'hidden_units': best_params['hidden_units'],
            'dropout_rates': best_params['dropout_rates'],
            'lr': best_params['lr'],
            'batch_size': best_params['batch_size'],
            'loss_weights': {k: float(v) for k, v in best_params['loss_weights'].items()},
        },
        'cv_results': {
            'mean_ic': float(best_trial['mean_ic']),
            'std_ic': float(best_trial['std_ic']),
            'fold_results': best_trial['fold_results'],
        }
    }
    with open(params_file, 'w') as f:
        json.dump(params_to_save, f, indent=2)
    print(f"\nBest parameters saved to: {params_file}")

    # 保存搜索历史
    history = []
    for t in trials.trials:
        if t['result']['status'] == STATUS_OK:
            history.append({
                'mean_ic': t['result']['mean_ic'],
                'std_ic': t['result']['std_ic'],
                'hidden_units': str(t['result']['params']['hidden_units']),
                'lr': t['result']['params']['lr'],
                'batch_size': t['result']['params']['batch_size'],
            })

    history_df = pd.DataFrame(history)
    history_file = output_dir / f"ae_mlp_cv_history_{timestamp}.csv"
    history_df.to_csv(history_file, index=False)
    print(f"Search history saved to: {history_file}")

    # 训练最终模型
    model, test_pred, dataset = train_final_model(
        args, handler_config, symbols, best_params,
        use_mixed_precision=use_mixed_precision
    )

    # 评估
    print("\n[*] Final Evaluation on Test Set (2025)...")
    evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

    # 保存模型
    print("\n[*] Saving model...")
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_SAVE_PATH / f"ae_mlp_cv_{args.handler}_{args.stock_pool}_{args.nday}d.keras"
    model.save(str(model_path))
    print(f"    Model saved to: {model_path}")

    # 回测
    if args.backtest:
        pred_df = test_pred.to_frame("score")

        # 构造 time_splits 用于回测
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
            model_name="AE-MLP (CV Hyperopt)",
            load_model_func=load_model,
            get_feature_count_func=get_feature_count
        )

    print("\n" + "=" * 70)
    print("CV HYPEROPT COMPLETE")
    print("=" * 70)
    print(f"CV Mean IC: {best_trial['mean_ic']:.4f} (±{best_trial['std_ic']:.4f})")
    print(f"Model saved to: {model_path}")
    print(f"Best parameters: {params_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
