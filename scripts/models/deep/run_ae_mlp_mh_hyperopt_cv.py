"""
Multi-Horizon AE-MLP 超参数搜索 - 时间序列交叉验证版本

在 multi-horizon AE-MLP 基础上使用 Hyperopt + CV 搜索最优超参数。
搜索空间包括网络结构、dropout、学习率、loss weights 和 multi-horizon 特有参数。

时间窗口设计:
  Fold 1: train 2000-2020, valid 2021
  Fold 2: train 2000-2021, valid 2022
  Fold 3: train 2000-2022, valid 2023
  Fold 4: train 2000-2023, valid 2024
  Test:   2025 (完全独立，不参与超参数选择)

使用方法:
    # 基本用法
    python scripts/models/deep/run_ae_mlp_mh_hyperopt_cv.py --handler alpha158-mh --stock-pool sp500

    # 更多搜索次数
    python scripts/models/deep/run_ae_mlp_mh_hyperopt_cv.py --handler alpha158-talib-lite-mh --max-evals 80

    # 带 macro 特征
    python scripts/models/deep/run_ae_mlp_mh_hyperopt_cv.py --handler alpha158-talib-lite-macro-mh --macro-features core

    # 自定义 horizons
    python scripts/models/deep/run_ae_mlp_mh_hyperopt_cv.py --handler alpha158-mh --horizons 2,5,10,20 --primary-horizon 5
"""

# ============================================================================
# 重要: 以下代码必须在任何其他导入之前执行
# ============================================================================

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

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

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG, PROJECT_ROOT, MODEL_SAVE_PATH,
    init_qlib,
    run_backtest,
)
from models.common.cv_utils import (
    CV_FOLDS,
    FINAL_TEST,
    create_dataset_for_fold,
    compute_ic,
)
from models.deep.ic_loss_utils import make_mixed_ic_mse_loss, ICEarlyStoppingCallback


# ============================================================================
# Multi-Horizon 数据准备
# ============================================================================

def create_mh_data_handler_for_fold(args, handler_config, symbols, fold_config, horizons, primary_horizon):
    """为特定 fold 创建 Multi-Horizon DataHandler"""
    from models.common.handlers import get_handler_class

    HandlerClass = get_handler_class(args.handler)
    end_time = fold_config.get('test_end', fold_config['valid_end'])

    handler_kwargs = {
        'horizons': horizons,
        'primary_horizon': primary_horizon,
        'instruments': symbols,
        'start_time': fold_config['train_start'],
        'end_time': end_time,
        'fit_start_time': fold_config['train_start'],
        'fit_end_time': fold_config['train_end'],
        'infer_processors': [],
    }

    if 'macro' in args.handler:
        handler_kwargs['macro_features'] = args.macro_features

    return HandlerClass(**handler_kwargs)


def prepare_mh_data_from_dataset(dataset, segment, horizons):
    """
    从 Dataset 准备 multi-horizon 数据。

    Returns:
        (features_array, labels_dict, index)
        labels_dict: {horizon: labels_array}
    """
    features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
    features = features.fillna(0).replace([np.inf, -np.inf], 0).clip(-10, 10)

    try:
        labels_df = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(labels_df, pd.Series):
            labels_df = labels_df.to_frame()

        # 处理 MultiIndex 列名
        if isinstance(labels_df.columns, pd.MultiIndex):
            labels_df.columns = [col[1] if isinstance(col, tuple) else col for col in labels_df.columns]

        labels_dict = {}
        for h in horizons:
            col_name = f'LABEL_{h}d'
            if col_name in labels_df.columns:
                labels_dict[h] = labels_df[col_name].fillna(0).values
            elif labels_df.shape[1] == 1:
                labels_dict[h] = labels_df.iloc[:, 0].fillna(0).values
            else:
                idx = horizons.index(h)
                if idx < labels_df.shape[1]:
                    labels_dict[h] = labels_df.iloc[:, idx].fillna(0).values
                else:
                    labels_dict[h] = labels_df.iloc[:, 0].fillna(0).values

        return features.values, labels_dict, features.index
    except Exception:
        return features.values, None, features.index


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

    # Head 结构
    'head_dim': scope.int(hp.quniform('head_dim', 16, 64, 8)),

    # Dropout (扩大上界，允许更强正则化)
    'noise_std': hp.uniform('noise_std', 0.01, 0.3),
    'dropout': hp.uniform('dropout', 0.02, 0.4),

    # L2 weight decay
    'l2_reg': hp.loguniform('l2_reg', np.log(1e-6), np.log(1e-2)),

    # 学习率
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),

    # Batch size
    'batch_size': hp.choice('batch_size', [1024, 2048, 4096, 8192]),

    # 损失权重
    'loss_decoder': hp.uniform('loss_decoder', 0.01, 0.3),
    'loss_ae': hp.uniform('loss_ae', 0.01, 0.3),

    # Multi-horizon 辅助权重
    'aux_weight': hp.uniform('aux_weight', 0.05, 0.8),

    # IC loss weight (0.0 = pure MSE baseline)
    'ic_loss_weight': hp.choice('ic_loss_weight', [0.0, 0.1, 0.2, 0.5, 1.0]),
}

BATCH_SIZE_CHOICES = [1024, 2048, 4096, 8192]
IC_LOSS_WEIGHT_CHOICES = [0.0, 0.1, 0.2, 0.5, 1.0]


def create_mh_model_params(hyperparams, num_columns, horizons, primary_horizon):
    """将 hyperopt 参数转换为 Multi-Horizon AE-MLP 参数"""
    hidden_units = [
        int(hyperparams['encoder_dim']),
        int(hyperparams['decoder_hidden']),
        int(hyperparams['main_layer1']),
        int(hyperparams['main_layer2']),
        int(hyperparams['main_layer3']),
    ]

    dropout_rates = [
        hyperparams['noise_std'],
        hyperparams['dropout'],
        hyperparams['dropout'],
        hyperparams['dropout'],
        hyperparams['dropout'],
        hyperparams['dropout'],
        hyperparams['dropout'],
    ]

    loss_weights = {
        'decoder': hyperparams['loss_decoder'],
        'ae_action': hyperparams['loss_ae'],
    }
    for h in horizons:
        key = f'action_{h}d'
        if h == primary_horizon:
            loss_weights[key] = 1.0
        else:
            loss_weights[key] = hyperparams['aux_weight']

    return {
        'num_columns': num_columns,
        'hidden_units': hidden_units,
        'dropout_rates': dropout_rates,
        'lr': hyperparams['learning_rate'],
        'batch_size': hyperparams['batch_size'],
        'loss_weights': loss_weights,
        'head_dim': int(hyperparams['head_dim']),
        'horizons': horizons,
        'primary_horizon': primary_horizon,
        'aux_weight': hyperparams['aux_weight'],
        'ic_loss_weight': hyperparams.get('ic_loss_weight', 0.0),
        'l2_reg': hyperparams.get('l2_reg', 0.0),
    }


def build_mh_model(params):
    """构建 Multi-Horizon AE-MLP 模型"""
    num_columns = params['num_columns']
    hidden_units = params['hidden_units']
    dropout_rates = params['dropout_rates']
    lr = params['lr']
    loss_weights = params['loss_weights']
    head_dim = params['head_dim']
    horizons = params['horizons']
    ic_loss_weight = params.get('ic_loss_weight', 0.0)
    l2_reg = params.get('l2_reg', 0.0)
    reg = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    inp = layers.Input(shape=(num_columns,), name='input')

    # 输入标准化
    x0 = layers.BatchNormalization(name='input_bn')(inp)

    # Encoder
    encoder = layers.GaussianNoise(dropout_rates[0], name='noise')(x0)
    encoder = layers.Dense(hidden_units[0], kernel_regularizer=reg, name='encoder_dense')(encoder)
    encoder = layers.BatchNormalization(name='encoder_bn')(encoder)
    encoder = layers.Activation('swish', name='encoder_act')(encoder)

    # Decoder (重建原始输入)
    decoder = layers.Dropout(dropout_rates[1], name='decoder_dropout')(encoder)
    decoder = layers.Dense(num_columns, name='decoder')(decoder)

    # 辅助预测分支
    x_ae = layers.Dense(hidden_units[1], kernel_regularizer=reg, name='ae_dense1')(decoder)
    x_ae = layers.BatchNormalization(name='ae_bn1')(x_ae)
    x_ae = layers.Activation('swish', name='ae_act1')(x_ae)
    x_ae = layers.Dropout(dropout_rates[2], name='ae_dropout1')(x_ae)
    out_ae = layers.Dense(1, name='ae_action')(x_ae)

    # 主分支
    x = layers.Concatenate(name='concat')([x0, encoder])
    x = layers.BatchNormalization(name='main_bn0')(x)
    x = layers.Dropout(dropout_rates[3], name='main_dropout0')(x)

    for i in range(2, len(hidden_units)):
        dropout_idx = min(i + 2, len(dropout_rates) - 1)
        x = layers.Dense(hidden_units[i], kernel_regularizer=reg, name=f'main_dense{i-1}')(x)
        x = layers.BatchNormalization(name=f'main_bn{i-1}')(x)
        x = layers.Activation('swish', name=f'main_act{i-1}')(x)
        x = layers.Dropout(dropout_rates[dropout_idx], name=f'main_dropout{i-1}')(x)

    # Multi-horizon prediction heads
    outputs = [decoder, out_ae]
    for h in horizons:
        head_name = f'action_{h}d'
        head = layers.Dense(head_dim, kernel_regularizer=reg, name=f'head_{h}d_dense')(x)
        head = layers.Activation('swish', name=f'head_{h}d_act')(head)
        head = layers.Dense(1, name=head_name)(head)
        outputs.append(head)

    model = Model(inputs=inp, outputs=outputs, name='AE_MLP_MultiHorizon')

    # Decoder and ae_action always use MSE (reconstruction tasks)
    # Action heads use mixed IC+MSE loss when ic_loss_weight > 0
    action_loss = make_mixed_ic_mse_loss(ic_loss_weight)
    losses = {'decoder': 'mse', 'ae_action': 'mse'}
    for h in horizons:
        losses[f'action_{h}d'] = action_loss

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=losses,
        loss_weights=loss_weights,
    )

    return model


# ============================================================================
# CV Hyperopt 目标函数
# ============================================================================

class MHCVHyperoptObjective:
    """Multi-Horizon 时间序列交叉验证的 Hyperopt 目标函数"""

    def __init__(self, args, handler_config, symbols, horizons, primary_horizon,
                 n_epochs=30, early_stop=5, gpu=0):
        self.args = args
        self.handler_config = handler_config
        self.symbols = symbols
        self.horizons = horizons
        self.primary_horizon = primary_horizon
        self.n_epochs = n_epochs
        self.early_stop = early_stop
        self.gpu = gpu
        self.trial_count = 0
        self.best_mean_ic = -float('inf')

        self._setup_gpu()

        # 预先准备所有 fold 的数据
        print("\n[*] Preparing multi-horizon data for all CV folds...")
        self.fold_data = []
        self.num_columns = None

        for fold in CV_FOLDS:
            print(f"    Preparing {fold['name']}...")
            handler = create_mh_data_handler_for_fold(
                args, handler_config, symbols, fold, horizons, primary_horizon
            )
            dataset = create_dataset_for_fold(handler, fold)

            X_train, y_train_dict, _ = prepare_mh_data_from_dataset(dataset, "train", horizons)
            X_valid, y_valid_dict, valid_index = prepare_mh_data_from_dataset(dataset, "valid", horizons)

            if self.num_columns is None:
                self.num_columns = X_train.shape[1]

            self.fold_data.append({
                'name': fold['name'],
                'X_train': X_train,
                'y_train_dict': y_train_dict,
                'X_valid': X_valid,
                'y_valid_dict': y_valid_dict,
                'valid_index': valid_index,
            })

            print(f"      Train: {X_train.shape}, Valid: {X_valid.shape}")

        print(f"    ✓ All {len(CV_FOLDS)} folds prepared")
        print(f"    Feature count: {self.num_columns}")
        print(f"    Horizons: {horizons}, Primary: {primary_horizon}d")

    def _setup_gpu(self):
        gpus = tf.config.list_physical_devices('GPU')
        if self.gpu >= 0 and gpus:
            try:
                tf.config.set_visible_devices(gpus[self.gpu], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[self.gpu], True)
                print(f"    Using GPU: {gpus[self.gpu]}")
            except RuntimeError as e:
                print(f"    GPU setup error: {e}")
        else:
            tf.config.set_visible_devices([], 'GPU')
            print("    Using CPU")

    def __call__(self, hyperparams):
        """目标函数: 在所有 fold 上训练并返回 primary horizon 的平均验证集 IC"""
        self.trial_count += 1

        model_params = create_mh_model_params(
            hyperparams, self.num_columns, self.horizons, self.primary_horizon
        )
        batch_size = model_params['batch_size']
        primary_idx = 2 + self.horizons.index(self.primary_horizon)

        fold_ics = []
        fold_results = []

        try:
            for fold in self.fold_data:
                tf.keras.backend.clear_session()

                model = build_mh_model(model_params)

                # 构建多输出训练数据
                train_outputs = {
                    'decoder': fold['X_train'],
                    'ae_action': fold['y_train_dict'][self.primary_horizon],
                }
                valid_outputs = {
                    'decoder': fold['X_valid'],
                    'ae_action': fold['y_valid_dict'][self.primary_horizon],
                }
                for h in self.horizons:
                    key = f'action_{h}d'
                    train_outputs[key] = fold['y_train_dict'][h]
                    valid_outputs[key] = fold['y_valid_dict'][h]

                # 回调: IC-based early stopping on primary horizon
                ic_cb = ICEarlyStoppingCallback(
                    X_valid=fold['X_valid'],
                    y_valid=fold['y_valid_dict'][self.primary_horizon],
                    valid_index=fold['valid_index'],
                    primary_output_idx=primary_idx,
                    patience=self.early_stop,
                    batch_size=batch_size,
                    verbose=0,
                )

                history = model.fit(
                    fold['X_train'],
                    train_outputs,
                    validation_data=(fold['X_valid'], valid_outputs),
                    epochs=self.n_epochs,
                    batch_size=batch_size,
                    callbacks=[ic_cb],
                    verbose=0,
                )

                # 取 primary horizon 的预测
                all_preds = model.predict(fold['X_valid'], batch_size=batch_size, verbose=0)
                valid_pred = all_preds[primary_idx].flatten()

                # 计算 primary horizon IC
                mean_ic, ic_std, icir = compute_ic(
                    valid_pred, fold['y_valid_dict'][self.primary_horizon], fold['valid_index']
                )

                best_epoch = ic_cb.best_epoch if ic_cb.best_epoch > 0 else len(history.history['loss'])
                total_epochs = len(history.history['loss'])

                fold_ics.append(mean_ic)
                fold_results.append({
                    'name': fold['name'],
                    'ic': mean_ic,
                    'icir': icir,
                    'best_epoch': best_epoch,
                    'total_epochs': total_epochs,
                })

            mean_ic_all = np.mean(fold_ics)
            std_ic_all = np.std(fold_ics)

            if mean_ic_all > self.best_mean_ic:
                self.best_mean_ic = mean_ic_all
                is_best = " ★ NEW BEST"
            else:
                is_best = ""

            fold_ic_str = ", ".join([f"{r['ic']:.4f}" for r in fold_results])
            fold_ep_str = ", ".join([f"{r['best_epoch']}/{r['total_epochs']}" for r in fold_results])
            ic_w = model_params.get('ic_loss_weight', 0.0)
            l2 = model_params.get('l2_reg', 0.0)
            print(f"  Trial {self.trial_count:3d}: Mean IC={mean_ic_all:.4f} (±{std_ic_all:.4f}) "
                  f"IC[{fold_ic_str}] ep[{fold_ep_str}] "
                  f"lr={hyperparams['learning_rate']:.5f} drop={hyperparams['dropout']:.3f} "
                  f"l2={l2:.1e} ic_w={ic_w:.1f}{is_best}")

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


# ============================================================================
# 主流程
# ============================================================================

def run_hyperopt_cv_search(args, handler_config, symbols, horizons, primary_horizon):
    """运行时间序列交叉验证的超参数搜索"""
    print("\n" + "=" * 70)
    print("HYPEROPT SEARCH WITH TIME-SERIES CV (Multi-Horizon AE-MLP)")
    print("=" * 70)
    print(f"Horizons: {horizons}, Primary: {primary_horizon}d")
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: train {fold['train_start']}~{fold['train_end']}, "
              f"valid {fold['valid_start']}~{fold['valid_end']}")
    print(f"Max evaluations: {args.max_evals}")
    print(f"Epochs per trial: {args.cv_epochs}")
    print("=" * 70)

    objective = MHCVHyperoptObjective(
        args, handler_config, symbols, horizons, primary_horizon,
        n_epochs=args.cv_epochs,
        early_stop=args.cv_early_stop,
        gpu=args.gpu,
    )

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
    best_params = create_mh_model_params(best, objective.num_columns, horizons, primary_horizon)
    best_params['batch_size'] = BATCH_SIZE_CHOICES[best['batch_size']]
    best_params['ic_loss_weight'] = IC_LOSS_WEIGHT_CHOICES[best['ic_loss_weight']]

    best_trial_idx = np.argmin([t['result']['loss'] for t in trials.trials])
    best_trial = trials.trials[best_trial_idx]['result']

    print("\n" + "=" * 70)
    print("CV HYPEROPT SEARCH COMPLETE")
    print("=" * 70)
    print(f"Best Mean IC ({primary_horizon}d): {best_trial['mean_ic']:.4f} (±{best_trial['std_ic']:.4f})")
    print("\nIC by fold:")
    for r in best_trial['fold_results']:
        print(f"  {r['name']}: IC={r['ic']:.4f}, ICIR={r['icir']:.4f}, epoch={r['best_epoch']}")
    print("\nBest parameters:")
    print(f"  hidden_units: {best_params['hidden_units']}")
    print(f"  head_dim: {best_params['head_dim']}")
    print(f"  learning_rate: {best_params['lr']:.6f}")
    print(f"  batch_size: {best_params['batch_size']}")
    print(f"  dropout: {best_params['dropout_rates'][1]:.4f}")
    print(f"  noise_std: {best_params['dropout_rates'][0]:.4f}")
    print(f"  aux_weight: {best_params['aux_weight']:.4f}")
    print(f"  ic_loss_weight: {best_params.get('ic_loss_weight', 0.0)}")
    print(f"  l2_reg: {best_params.get('l2_reg', 0.0):.6f}")
    print(f"  loss_weights: {best_params['loss_weights']}")
    print("=" * 70)

    return best_params, trials, best_trial, objective.num_columns


def train_final_model(args, handler_config, symbols, best_params, horizons, primary_horizon):
    """使用最优参数在完整数据上训练最终模型"""
    print("\n[*] Training final model on full data...")
    print(f"    Horizons: {horizons}, Primary: {primary_horizon}d")
    print(f"    hidden_units: {best_params['hidden_units']}")
    print(f"    head_dim: {best_params['head_dim']}")
    print(f"    learning_rate: {best_params['lr']:.6f}")
    print(f"    batch_size: {best_params['batch_size']}")
    print(f"    aux_weight: {best_params['aux_weight']:.4f}")
    print(f"    ic_loss_weight: {best_params.get('ic_loss_weight', 0.0)}")
    print(f"    l2_reg: {best_params.get('l2_reg', 0.0):.6f}")

    # 创建最终数据集
    handler = create_mh_data_handler_for_fold(
        args, handler_config, symbols, FINAL_TEST, horizons, primary_horizon
    )
    dataset = create_dataset_for_fold(handler, FINAL_TEST)

    X_train, y_train_dict, _ = prepare_mh_data_from_dataset(dataset, "train", horizons)
    X_valid, y_valid_dict, valid_index = prepare_mh_data_from_dataset(dataset, "valid", horizons)
    X_test, _, test_index = prepare_mh_data_from_dataset(dataset, "test", horizons)

    print(f"\n    Final training data:")
    print(f"      Train: {X_train.shape} ({FINAL_TEST['train_start']} ~ {FINAL_TEST['train_end']})")
    print(f"      Valid: {X_valid.shape} ({FINAL_TEST['valid_start']} ~ {FINAL_TEST['valid_end']})")
    print(f"      Test:  {X_test.shape} ({FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']})")

    best_params['num_columns'] = X_train.shape[1]

    tf.keras.backend.clear_session()
    model = build_mh_model(best_params)

    # 回调: IC-based early stopping + LR reduction
    primary_idx = 2 + horizons.index(primary_horizon)
    ic_cb = ICEarlyStoppingCallback(
        X_valid=X_valid,
        y_valid=y_valid_dict[primary_horizon],
        valid_index=valid_index,
        primary_output_idx=primary_idx,
        patience=args.early_stop,
        batch_size=best_params['batch_size'],
        verbose=1,
    )
    cb_list = [
        ic_cb,
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
            mode='min',
        ),
    ]

    # 构建多输出训练数据
    train_outputs = {
        'decoder': X_train,
        'ae_action': y_train_dict[primary_horizon],
    }
    valid_outputs = {
        'decoder': X_valid,
        'ae_action': y_valid_dict[primary_horizon],
    }
    for h in horizons:
        key = f'action_{h}d'
        train_outputs[key] = y_train_dict[h]
        valid_outputs[key] = y_valid_dict[h]

    print("\n    Training progress:")
    model.fit(
        X_train,
        train_outputs,
        validation_data=(X_valid, valid_outputs),
        epochs=args.n_epochs,
        batch_size=best_params['batch_size'],
        callbacks=cb_list,
        verbose=1,
    )

    # 验证集 IC (from IC callback's best)
    print(f"\n    [Validation Set - for reference]")
    print(f"    Valid IC ({primary_horizon}d):   {ic_cb.best_ic:.4f} (from IC early stopping)")
    print(f"    Best epoch: {ic_cb.best_epoch}")

    # Also compute IC on final model for confirmation
    valid_preds = model.predict(X_valid, batch_size=best_params['batch_size'], verbose=0)
    valid_pred = valid_preds[primary_idx].flatten()
    valid_ic, _, valid_icir = compute_ic(
        valid_pred, y_valid_dict[primary_horizon], valid_index
    )
    print(f"    Valid IC ({primary_horizon}d):   {valid_ic:.4f} (final model)")
    print(f"    Valid ICIR ({primary_horizon}d): {valid_icir:.4f}")

    # 测试集预测 (所有 horizons)
    test_preds = model.predict(X_test, batch_size=best_params['batch_size'], verbose=0)

    all_horizon_preds = {}
    for i, h in enumerate(horizons):
        pred = test_preds[2 + i].flatten()
        all_horizon_preds[h] = pd.Series(pred, index=test_index, name=f'score_{h}d')

    test_pred = all_horizon_preds[primary_horizon].rename('score')

    print(f"\n    Test predictions:")
    for h, pred in all_horizon_preds.items():
        marker = " (PRIMARY)" if h == primary_horizon else ""
        print(f"      {h}d{marker}: mean={pred.mean():.6f}, std={pred.std():.6f}, "
              f"range=[{pred.min():.4f}, {pred.max():.4f}]")

    return model, test_pred, all_horizon_preds, dataset


def train_cv_ensemble(args, handler_config, symbols, best_params, horizons, primary_horizon):
    """用 4 个 CV fold 模型做 ensemble，在 test set 上取平均预测"""
    print("\n[*] CV Ensemble: training fold models and averaging test predictions...")
    print(f"    Horizons: {horizons}, Primary: {primary_horizon}d")
    print(f"    Test period: {FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']}")
    print(f"    Folds: {len(CV_FOLDS)}")

    all_fold_test_preds = {}  # {fold_name: {horizon: pd.Series}}
    fold_valid_ics = {}
    last_dataset = None

    # 保存目录
    horizons_str = '_'.join(str(h) for h in horizons)
    ensemble_dir = MODEL_SAVE_PATH / f"cv_ensemble_{args.handler}_{args.stock_pool}_{horizons_str}"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    print(f"    Save dir: {ensemble_dir}")

    for fold_idx, fold in enumerate(CV_FOLDS):
        fold_name = fold['name']
        print(f"\n    --- Fold {fold_idx+1}/{len(CV_FOLDS)}: {fold_name} ---")
        print(f"    Train: {fold['train_start']} ~ {fold['train_end']}")
        print(f"    Valid: {fold['valid_start']} ~ {fold['valid_end']}")

        # 扩展 fold config 加入 test segment
        fold_with_test = {
            **fold,
            'test_start': FINAL_TEST['test_start'],
            'test_end': FINAL_TEST['test_end'],
        }

        handler = create_mh_data_handler_for_fold(
            args, handler_config, symbols, fold_with_test, horizons, primary_horizon
        )
        dataset = create_dataset_for_fold(handler, fold_with_test)
        last_dataset = dataset

        X_train, y_train_dict, _ = prepare_mh_data_from_dataset(dataset, "train", horizons)
        X_valid, y_valid_dict, valid_index = prepare_mh_data_from_dataset(dataset, "valid", horizons)
        X_test, _, test_index = prepare_mh_data_from_dataset(dataset, "test", horizons)

        print(f"    Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

        best_params['num_columns'] = X_train.shape[1]

        tf.keras.backend.clear_session()
        model = build_mh_model(best_params)

        # IC-based early stopping
        primary_idx = 2 + horizons.index(primary_horizon)
        ic_cb = ICEarlyStoppingCallback(
            X_valid=X_valid,
            y_valid=y_valid_dict[primary_horizon],
            valid_index=valid_index,
            primary_output_idx=primary_idx,
            patience=args.early_stop,
            batch_size=best_params['batch_size'],
            verbose=0,
        )
        cb_list = [
            ic_cb,
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0, mode='min',
            ),
        ]

        # 构建多输出训练数据
        train_outputs = {'decoder': X_train, 'ae_action': y_train_dict[primary_horizon]}
        valid_outputs = {'decoder': X_valid, 'ae_action': y_valid_dict[primary_horizon]}
        for h in horizons:
            train_outputs[f'action_{h}d'] = y_train_dict[h]
            valid_outputs[f'action_{h}d'] = y_valid_dict[h]

        model.fit(
            X_train, train_outputs,
            validation_data=(X_valid, valid_outputs),
            epochs=args.n_epochs,
            batch_size=best_params['batch_size'],
            callbacks=cb_list,
            verbose=0,
        )

        fold_valid_ics[fold_name] = ic_cb.best_ic

        # Test 预测 (所有 horizons)
        test_preds = model.predict(X_test, batch_size=best_params['batch_size'], verbose=0)

        fold_preds = {}
        for i, h in enumerate(horizons):
            pred = test_preds[2 + i].flatten()
            fold_preds[h] = pd.Series(pred, index=test_index, name=f'score_{h}d')

        all_fold_test_preds[fold_name] = fold_preds
        print(f"    Valid IC ({primary_horizon}d): {ic_cb.best_ic:.4f}, Best epoch: {ic_cb.best_epoch}")

        # 保存 fold 模型
        fold_model_path = ensemble_dir / f"fold_{fold_idx+1}.keras"
        model.save(str(fold_model_path))

        # 保存 fold 预测
        fold_pred_df = pd.DataFrame({f'score_{h}d': fold_preds[h] for h in horizons})
        fold_pred_path = ensemble_dir / f"fold_{fold_idx+1}_test_preds.pkl"
        fold_pred_df.to_pickle(str(fold_pred_path))

        print(f"    Saved: {fold_model_path.name}, {fold_pred_path.name}")

        del model
        tf.keras.backend.clear_session()

    # 保存 ensemble 元数据
    ensemble_meta = {
        'horizons': horizons,
        'primary_horizon': primary_horizon,
        'handler': args.handler,
        'stock_pool': args.stock_pool,
        'fold_valid_ics': {k: float(v) for k, v in fold_valid_ics.items()},
        'n_folds': len(CV_FOLDS),
        'folds': [{'name': f['name'], 'train_end': f['train_end'], 'valid_end': f['valid_end']} for f in CV_FOLDS],
    }
    with open(ensemble_dir / "ensemble_meta.json", 'w') as f:
        json.dump(ensemble_meta, f, indent=2)
    print(f"\n    Ensemble models and predictions saved to: {ensemble_dir}")

    # 平均 4 个 fold 的预测
    ensemble_preds = {}
    for h in horizons:
        fold_series = [all_fold_test_preds[f['name']][h] for f in CV_FOLDS]
        ensemble_preds[h] = pd.concat(fold_series, axis=1).mean(axis=1)
        ensemble_preds[h].name = f'score_{h}d'

    test_pred = ensemble_preds[primary_horizon].rename('score')

    # Fold 间一致性分析
    print(f"\n{'='*60}")
    print("CV Ensemble Summary")
    print(f"{'='*60}")

    # 每个 fold 在 test set 上的 IC + 与 ensemble 的相关性
    label_data = last_dataset.prepare("test", col_set="label")
    if isinstance(label_data, pd.DataFrame) and isinstance(label_data.columns, pd.MultiIndex):
        label_data.columns = [col[1] if isinstance(col, tuple) else col for col in label_data.columns]

    ph_col = f'LABEL_{primary_horizon}d'
    if isinstance(label_data, pd.DataFrame) and ph_col in label_data.columns:
        label_series = label_data[ph_col]

        for fold_name, fold_preds in all_fold_test_preds.items():
            pred = fold_preds[primary_horizon]
            pred_aligned = pred.reindex(label_data.index)
            valid_idx = ~(pred_aligned.isna() | label_series.isna())
            pred_clean = pred_aligned[valid_idx]
            label_clean = label_series[valid_idx]

            daily_ic = pred_clean.groupby(level="datetime").apply(
                lambda x: x.corr(label_clean.loc[x.index]) if len(x) > 1 else np.nan
            ).dropna()

            fold_ic = daily_ic.mean() if len(daily_ic) > 0 else float('nan')
            corr_with_ens = pred.corr(ensemble_preds[primary_horizon])
            print(f"  {fold_name}: Valid IC={fold_valid_ics[fold_name]:.4f}, "
                  f"Test IC({primary_horizon}d)={fold_ic:.4f}, "
                  f"corr_with_ensemble={corr_with_ens:.3f}")

        # Ensemble IC
        ens_pred = ensemble_preds[primary_horizon].reindex(label_data.index)
        valid_idx = ~(ens_pred.isna() | label_series.isna())
        ens_clean = ens_pred[valid_idx]
        label_clean = label_series[valid_idx]
        daily_ic = ens_clean.groupby(level="datetime").apply(
            lambda x: x.corr(label_clean.loc[x.index]) if len(x) > 1 else np.nan
        ).dropna()
        ens_ic = daily_ic.mean() if len(daily_ic) > 0 else float('nan')
        ens_icir = daily_ic.mean() / daily_ic.std() if len(daily_ic) > 1 else float('nan')
        print(f"\n  Ensemble: Test IC({primary_horizon}d)={ens_ic:.4f}, ICIR={ens_icir:.4f}")

    # Inter-fold prediction correlation matrix
    print(f"\n  Inter-fold prediction correlation ({primary_horizon}d):")
    fold_names = [f['name'] for f in CV_FOLDS]
    corr_data = pd.DataFrame({
        name: all_fold_test_preds[name][primary_horizon]
        for name in fold_names
    })
    corr_matrix = corr_data.corr()
    # Print compact correlation matrix
    header = "          " + "  ".join(f"{n[:8]:>8}" for n in fold_names)
    print(f"  {header}")
    for i, name in enumerate(fold_names):
        row = f"  {name[:8]:>8}  " + "  ".join(
            f"{corr_matrix.iloc[i, j]:8.3f}" for j in range(len(fold_names))
        )
        print(row)

    print(f"{'='*60}")

    return None, test_pred, ensemble_preds, last_dataset


def evaluate_all_horizons(all_horizon_preds, dataset, horizons, primary_horizon):
    """评估所有 horizons 的 IC/ICIR"""
    print("\n[*] Evaluation (all horizons on test set)...")

    label_data = dataset.prepare("test", col_set="label")
    if isinstance(label_data, pd.DataFrame) and isinstance(label_data.columns, pd.MultiIndex):
        label_data.columns = [col[1] if isinstance(col, tuple) else col for col in label_data.columns]

    if not isinstance(label_data, pd.DataFrame):
        print("    Warning: cannot evaluate, label data is not a DataFrame")
        return

    for h, pred in all_horizon_preds.items():
        col_name = f'LABEL_{h}d'
        if col_name not in label_data.columns:
            continue

        label_series = label_data[col_name]
        pred_aligned = pred.reindex(label_data.index)
        valid_idx = ~(pred_aligned.isna() | label_series.isna())
        pred_clean = pred_aligned[valid_idx]
        label_clean = label_series[valid_idx]

        if len(pred_clean) == 0:
            continue

        daily_ic = pred_clean.groupby(level="datetime").apply(
            lambda x: x.corr(label_clean.loc[x.index]) if len(x) > 1 else np.nan
        ).dropna()

        mse = ((pred_clean - label_clean) ** 2).mean()
        mae = (pred_clean - label_clean).abs().mean()
        rmse = np.sqrt(mse)

        marker = " <<< PRIMARY" if h == primary_horizon else ""
        print(f"\n    --- {h}d{marker} ---")
        print(f"    Valid samples: {len(pred_clean)}")
        if len(daily_ic) > 0:
            print(f"    IC:   {daily_ic.mean():.4f}")
            print(f"    ICIR: {daily_ic.mean()/daily_ic.std():.4f}")
        print(f"    MSE:  {mse:.6f}")
        print(f"    MAE:  {mae:.6f}")
        print(f"    RMSE: {rmse:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Horizon AE-MLP Hyperopt with Time-Series Cross-Validation',
    )

    # 基础参数
    parser.add_argument('--handler', type=str, default='alpha158-mh',
                        choices=list(HANDLER_CONFIG.keys()))
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])

    # Multi-horizon 参数
    parser.add_argument('--horizons', type=str, default='2,5,10',
                        help='Prediction horizons, comma-separated (default: "2,5,10")')
    parser.add_argument('--primary-horizon', type=int, default=5,
                        help='Primary prediction horizon in days (default: 5)')

    # Macro 参数
    parser.add_argument('--macro-features', type=str, default='core',
                        choices=['all', 'core', 'vix_only', 'none'])

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

    # CV Ensemble
    parser.add_argument('--cv-ensemble', action='store_true',
                        help='Use CV fold models ensemble instead of single final model')

    # 回测参数
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=1)
    parser.add_argument('--account', type=float, default=10000)
    parser.add_argument('--rebalance-freq', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss'])

    args = parser.parse_args()

    # 解析 horizons
    horizons = [int(x.strip()) for x in args.horizons.split(',')]
    primary_horizon = args.primary_horizon

    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    # 打印头部
    print("=" * 70)
    print("Multi-Horizon AE-MLP Hyperopt with Time-Series Cross-Validation")
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Handler: {args.handler}")
    print(f"Horizons: {horizons}, Primary: {primary_horizon}d")
    print(f"Max evaluations: {args.max_evals}")
    print(f"CV epochs: {args.cv_epochs}")
    print(f"CV Folds: {len(CV_FOLDS)}")
    print(f"GPU: {args.gpu}")
    if args.cv_ensemble:
        print(f"Mode: CV Ensemble (average {len(CV_FOLDS)} fold models)")
    print("=" * 70)

    init_qlib(handler_config['use_talib'])

    # CV 超参数搜索
    best_params, trials, best_trial, num_columns = run_hyperopt_cv_search(
        args, handler_config, symbols, horizons, primary_horizon
    )

    # 保存搜索结果
    output_dir = PROJECT_ROOT / "outputs" / "hyperopt_cv"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    horizons_str = '_'.join(str(h) for h in horizons)

    params_file = output_dir / f"ae_mlp_mh_cv_best_params_{horizons_str}_{timestamp}.json"
    params_to_save = {
        'horizons': horizons,
        'primary_horizon': primary_horizon,
        'params': {
            'hidden_units': best_params['hidden_units'],
            'head_dim': best_params['head_dim'],
            'dropout_rates': best_params['dropout_rates'],
            'lr': best_params['lr'],
            'batch_size': best_params['batch_size'],
            'aux_weight': best_params['aux_weight'],
            'ic_loss_weight': best_params.get('ic_loss_weight', 0.0),
            'l2_reg': best_params.get('l2_reg', 0.0),
            'loss_weights': {k: float(v) for k, v in best_params['loss_weights'].items()},
        },
        'cv_results': {
            'mean_ic': float(best_trial['mean_ic']),
            'std_ic': float(best_trial['std_ic']),
            'fold_results': best_trial['fold_results'],
        },
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
                'head_dim': t['result']['params']['head_dim'],
                'lr': t['result']['params']['lr'],
                'batch_size': t['result']['params']['batch_size'],
                'aux_weight': t['result']['params']['aux_weight'],
                'ic_loss_weight': t['result']['params'].get('ic_loss_weight', 0.0),
                'l2_reg': t['result']['params'].get('l2_reg', 0.0),
            })

    history_df = pd.DataFrame(history)
    history_file = output_dir / f"ae_mlp_mh_cv_history_{horizons_str}_{timestamp}.csv"
    history_df.to_csv(history_file, index=False)
    print(f"Search history saved to: {history_file}")

    # 训练最终模型 / CV Ensemble
    if args.cv_ensemble:
        _, test_pred, all_horizon_preds, dataset = train_cv_ensemble(
            args, handler_config, symbols, best_params, horizons, primary_horizon
        )
    else:
        model, test_pred, all_horizon_preds, dataset = train_final_model(
            args, handler_config, symbols, best_params, horizons, primary_horizon
        )

    # 评估
    evaluate_all_horizons(all_horizon_preds, dataset, horizons, primary_horizon)

    # 保存模型 (only for single model, not ensemble)
    model_path = None
    if not args.cv_ensemble:
        print("\n[*] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_SAVE_PATH / f"ae_mlp_mh_cv_{args.handler}_{args.stock_pool}_{horizons_str}.keras"
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

        if args.cv_ensemble:
            ensemble_model_path = MODEL_SAVE_PATH / f"ae_mlp_mh_cv_ensemble_{args.handler}_{args.stock_pool}_{horizons_str}"
            run_backtest(
                ensemble_model_path, dataset, pred_df, args, time_splits,
                model_name="MH-AE-MLP (CV Ensemble)",
                load_model_func=lambda p: None,
                get_feature_count_func=lambda m: "N/A (ensemble)",
            )
        else:
            def load_model(path):
                return keras.models.load_model(str(path))

            def get_feature_count(m):
                return m.input_shape[1]

            run_backtest(
                model_path, dataset, pred_df, args, time_splits,
                model_name="MH-AE-MLP (CV Hyperopt)",
                load_model_func=load_model,
                get_feature_count_func=get_feature_count,
            )

    print("\n" + "=" * 70)
    print(f"CV HYPEROPT COMPLETE (Multi-Horizon AE-MLP{' - CV Ensemble' if args.cv_ensemble else ''})")
    print("=" * 70)
    print(f"Horizons: {horizons}, Primary: {primary_horizon}d")
    print(f"CV Mean IC: {best_trial['mean_ic']:.4f} (±{best_trial['std_ic']:.4f})")
    if model_path:
        print(f"Model saved to: {model_path}")
    print(f"Best parameters: {params_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
