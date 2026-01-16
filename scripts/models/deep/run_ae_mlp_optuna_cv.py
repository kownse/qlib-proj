"""
AE-MLP 超参数搜索 - Optuna 版本 (带 Pruning)

使用 Optuna 进行超参数搜索，支持：
- 早期剪枝 (Pruning): 表现差的 trial 会被提前终止
- 时间序列交叉验证
- 断点续传

时间窗口设计:
  Fold 1: train 2000-2022, valid 2023
  Fold 2: train 2000-2023, valid 2024
  Test:   2025 (完全独立)

使用方法:
    python scripts/models/deep/run_ae_mlp_optuna_cv.py
    python scripts/models/deep/run_ae_mlp_optuna_cv.py --n-trials 50
    python scripts/models/deep/run_ae_mlp_optuna_cv.py --stock-pool sp100 --backtest

    # 断点续传 (使用相同的 study-name)
    python scripts/models/deep/run_ae_mlp_optuna_cv.py --study-name ae_mlp_search --resume
"""

# ============================================================================
# 重要: 以下代码必须在任何其他导入之前执行
# ============================================================================

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 抑制 TensorFlow 日志
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

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import tensorflow as tf

# GPU 配置必须在任何 TensorFlow 操作之前执行
def setup_gpu_early(gpu_id=0, memory_limit_mb=None):
    """在脚本启动时配置 GPU 内存增长"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            if gpu_id >= 0 and gpu_id < len(gpus):
                target_gpu = gpus[gpu_id]
                tf.config.set_visible_devices(target_gpu, 'GPU')

                if memory_limit_mb:
                    # 限制显存使用量
                    tf.config.set_logical_device_configuration(
                        target_gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                    )
                    print(f"[GPU] GPU {gpu_id} with {memory_limit_mb}MB memory limit")
                else:
                    # 启用内存增长模式
                    tf.config.experimental.set_memory_growth(target_gpu, True)
                    print(f"[GPU] GPU {gpu_id} with memory growth enabled")
            else:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"[GPU] Configured {len(gpus)} GPU(s) with memory growth")
        except RuntimeError as e:
            print(f"[GPU] Setup error (may be already configured): {e}")
    else:
        print("[GPU] No GPU found, using CPU")

# 解析命令行参数获取 GPU ID 和内存限制 (需要提前解析)
_gpu_id = 0
_gpu_memory = None
for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        try:
            _gpu_id = int(sys.argv[i + 1])
        except ValueError:
            pass
    if arg == '--gpu-memory' and i + 1 < len(sys.argv):
        try:
            _gpu_memory = int(sys.argv[i + 1])
        except ValueError:
            pass

setup_gpu_early(_gpu_id, _gpu_memory)

# 启用混合精度训练，减少显存占用
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("[GPU] Mixed precision (float16) enabled for memory optimization")

from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from utils.utils import evaluate_model
from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG, PROJECT_ROOT, MODEL_SAVE_PATH,
    init_qlib,
    run_backtest,
)


# ============================================================================
# 时间序列交叉验证的 Fold 配置
# ============================================================================

CV_FOLDS = [
    {
        'name': 'Fold 1 (valid 2023)',
        'train_start': '2000-01-01',
        'train_end': '2022-12-31',
        'valid_start': '2023-01-01',
        'valid_end': '2023-12-31',
    },
    {
        'name': 'Fold 2 (valid 2024)',
        'train_start': '2000-01-01',
        'train_end': '2023-12-31',
        'valid_start': '2024-01-01',
        'valid_end': '2024-12-31',
    },
]

# 最终测试集 (完全独立)
FINAL_TEST = {
    'train_start': '2000-01-01',
    'train_end': '2024-09-30',    # 修复：避免与验证集重叠
    'valid_start': '2024-10-01',  # 验证集（无重叠）
    'valid_end': '2024-12-31',
    'test_start': '2025-01-01',
    'test_end': '2025-12-31',
}


def create_data_handler_for_fold(args, handler_config, symbols, fold_config):
    """为特定 fold 创建 DataHandler"""
    from models.common.handlers import get_handler_class

    HandlerClass = get_handler_class(args.handler)

    end_time = fold_config.get('test_end', fold_config['valid_end'])

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

    if 'test_start' in fold_config:
        segments["test"] = (fold_config['test_start'], fold_config['test_end'])

    return DatasetH(handler=handler, segments=segments)


def prepare_data_from_dataset(dataset: DatasetH, segment: str):
    """从 Dataset 准备数据"""
    features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
    features = features.fillna(0).replace([np.inf, -np.inf], 0).clip(-10, 10)

    try:
        labels = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(labels, pd.DataFrame):
            labels = labels.iloc[:, 0]
        labels = labels.fillna(0).values
        return features.values, labels, features.index
    except Exception:
        return features.values, None, features.index


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

    # Decoder (重建原始输入, float32 for mixed precision)
    decoder = layers.Dropout(dropout_rates[1], name='decoder_dropout')(encoder)
    decoder = layers.Dense(num_columns, name='decoder', dtype='float32')(decoder)

    # 辅助预测分支 (基于 decoder 输出)
    x_ae = layers.Dense(hidden_units[1], name='ae_dense1')(decoder)
    x_ae = layers.BatchNormalization(name='ae_bn1')(x_ae)
    x_ae = layers.Activation('swish', name='ae_act1')(x_ae)
    x_ae = layers.Dropout(dropout_rates[2], name='ae_dropout1')(x_ae)
    out_ae = layers.Dense(1, name='ae_action', dtype='float32')(x_ae)

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

    # 主输出 (使用 float32 确保数值稳定性，混合精度要求)
    out = layers.Dense(1, name='action', dtype='float32')(x)

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


def clear_gpu_memory():
    """彻底清理 GPU 内存"""
    import gc
    tf.keras.backend.clear_session()
    gc.collect()
    # 强制释放 GPU 内存
    try:
        from numba import cuda
        cuda.select_device(0)
        cuda.close()
    except Exception:
        pass


class OptunaObjective:
    """Optuna 目标函数，支持 Pruning - 内存优化版本"""

    def __init__(self, args, handler_config, symbols, n_epochs=50, early_stop=10, gpu=0, max_train_samples=None):
        self.args = args
        self.handler_config = handler_config
        self.symbols = symbols
        self.n_epochs = n_epochs
        self.early_stop = early_stop
        self.gpu = gpu
        self.max_train_samples = max_train_samples

        # 不预加载数据，只获取特征数量
        print("\n[*] Checking data dimensions (not preloading to save memory)...")
        fold = CV_FOLDS[0]
        handler = create_data_handler_for_fold(args, handler_config, symbols, fold)
        dataset = create_dataset_for_fold(handler, fold)
        X_sample, _, _ = prepare_data_from_dataset(dataset, "train")
        self.num_columns = X_sample.shape[1]
        print(f"    Feature count: {self.num_columns}")
        print(f"    CV Folds: {len(CV_FOLDS)} (data will be loaded on-demand)")

        # 清理
        del X_sample, handler, dataset
        clear_gpu_memory()

    def _load_fold_data(self, fold_config, max_train_samples=None):
        """按需加载单个 fold 的数据，支持采样"""
        handler = create_data_handler_for_fold(
            self.args, self.handler_config, self.symbols, fold_config
        )
        dataset = create_dataset_for_fold(handler, fold_config)

        X_train, y_train, _ = prepare_data_from_dataset(dataset, "train")
        X_valid, y_valid, valid_index = prepare_data_from_dataset(dataset, "valid")

        # 如果训练数据太大，进行随机采样
        if max_train_samples and len(X_train) > max_train_samples:
            np.random.seed(42)
            idx = np.random.choice(len(X_train), max_train_samples, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]
            print(f"    Sampled training data: {len(X_train)} samples")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_valid': X_valid,
            'y_valid': y_valid,
            'valid_index': valid_index,
        }

    def __call__(self, trial: optuna.Trial):
        """目标函数：支持中间汇报和剪枝 - 内存优化版本"""
        import gc

        # 动态采样超参数 (减小搜索空间以适应 12GB 显存)
        encoder_dim = trial.suggest_int('encoder_dim', 32, 96, step=16)
        decoder_hidden = trial.suggest_int('decoder_hidden', 32, 96, step=16)
        main_layer1 = trial.suggest_int('main_layer1', 64, 256, step=64)
        main_layer2 = trial.suggest_int('main_layer2', 32, 128, step=32)
        main_layer3 = trial.suggest_int('main_layer3', 16, 64, step=16)

        noise_std = trial.suggest_float('noise_std', 0.01, 0.1)
        dropout = trial.suggest_float('dropout', 0.01, 0.15)

        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        # 减小 batch_size 避免 OOM（需要 --new-study 重新开始）
        batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048])

        loss_decoder = trial.suggest_float('loss_decoder', 0.01, 0.3)
        loss_ae = trial.suggest_float('loss_ae', 0.01, 0.3)

        # 构建模型参数
        hidden_units = [encoder_dim, decoder_hidden, main_layer1, main_layer2, main_layer3]
        dropout_rates = [noise_std, dropout, dropout, dropout, dropout, dropout, dropout]
        loss_weights = {'decoder': loss_decoder, 'ae_action': loss_ae, 'action': 1.0}

        model_params = {
            'num_columns': self.num_columns,
            'hidden_units': hidden_units,
            'dropout_rates': dropout_rates,
            'lr': lr,
            'batch_size': batch_size,
            'loss_weights': loss_weights,
        }

        # 打印当前 trial 的参数
        print(f"\n{'='*70}")
        print(f"Trial {trial.number}: units={hidden_units}, "
              f"dropout={dropout:.3f}, lr={lr:.6f}, batch={batch_size}")
        print(f"{'='*70}")

        fold_ics = []

        for fold_idx, fold_config in enumerate(CV_FOLDS):
            print(f"\n  [{fold_config['name']}] Loading data...")

            # 彻底清理 GPU 内存
            clear_gpu_memory()

            # 按需加载当前 fold 的数据
            fold = self._load_fold_data(fold_config, self.max_train_samples)
            print(f"    Train: {fold['X_train'].shape}, Valid: {fold['X_valid'].shape}")

            # 构建模型
            model = build_ae_mlp_model(model_params)

            # 训练数据
            train_outputs = {
                'decoder': fold['X_train'],
                'ae_action': fold['y_train'],
                'action': fold['y_train'],
            }
            valid_outputs = {
                'decoder': fold['X_valid'],
                'ae_action': fold['y_valid'],
                'action': fold['y_valid'],
            }

            # 自定义训练循环以支持 pruning
            best_val_loss = float('inf')
            best_val_ic = 0.0
            best_weights = None
            stop_steps = 0

            for epoch in range(self.n_epochs):
                # 训练一个 epoch
                history = model.fit(
                    fold['X_train'],
                    train_outputs,
                    validation_data=(fold['X_valid'], valid_outputs),
                    epochs=1,
                    batch_size=batch_size,
                    verbose=0,
                )

                val_loss = history.history['val_action_loss'][0]

                # 计算验证集 IC
                _, _, valid_pred = model.predict(fold['X_valid'], batch_size=batch_size, verbose=0)
                valid_pred = valid_pred.flatten()
                val_ic, _, _ = compute_ic(valid_pred, fold['y_valid'], fold['valid_index'])

                # 检查是否最优
                is_best = ""
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_ic = val_ic
                    best_weights = model.get_weights()
                    stop_steps = 0
                    is_best = " *"
                else:
                    stop_steps += 1

                # 每 5 个 epoch 打印一次
                if epoch % 5 == 0 or is_best:
                    print(f"    Epoch {epoch+1:2d}/{self.n_epochs}: "
                          f"val_loss={val_loss:.6f} val_IC={val_ic:.4f}{is_best}")

                # 早停
                if stop_steps >= self.early_stop:
                    print(f"    Early stopping at epoch {epoch+1} (best val_IC={best_val_ic:.4f})")
                    break

                # ========== 关键：中间汇报给 Optuna ==========
                # 每 5 个 epoch 汇报一次，让 Optuna 决定是否剪枝
                if epoch % 5 == 0 and epoch > 0:
                    intermediate_value = best_val_ic if fold_idx == 0 else np.mean(fold_ics + [best_val_ic])
                    trial.report(intermediate_value, epoch + fold_idx * self.n_epochs)

                    # 检查是否应该剪枝
                    if trial.should_prune():
                        print(f"    >>> PRUNED by Optuna (intermediate IC={intermediate_value:.4f})")
                        # 清理当前 fold 数据再抛出异常
                        del model, fold, train_outputs, valid_outputs
                        clear_gpu_memory()
                        raise optuna.TrialPruned()

            # 加载最佳权重计算最终 IC
            if best_weights is not None:
                model.set_weights(best_weights)

            _, _, valid_pred = model.predict(fold['X_valid'], batch_size=batch_size, verbose=0)
            valid_pred = valid_pred.flatten()
            mean_ic, _, icir = compute_ic(valid_pred, fold['y_valid'], fold['valid_index'])
            fold_ics.append(mean_ic)

            print(f"  [{fold_config['name']}] Final IC={mean_ic:.4f}, ICIR={icir:.4f}")

            # 清理当前 fold 的数据和模型，释放内存
            del model, fold, train_outputs, valid_outputs, best_weights
            clear_gpu_memory()

        # 返回平均 IC
        mean_ic_all = np.mean(fold_ics)
        std_ic_all = np.std(fold_ics)

        print(f"\n  Trial {trial.number} Result: Mean IC={mean_ic_all:.4f} (std={std_ic_all:.4f})")

        # 保存额外信息
        trial.set_user_attr('fold_ics', fold_ics)
        trial.set_user_attr('std_ic', std_ic_all)
        trial.set_user_attr('hidden_units', hidden_units)
        trial.set_user_attr('loss_weights', loss_weights)

        return mean_ic_all


def run_optuna_search(args, handler_config, symbols):
    """运行 Optuna 超参数搜索"""
    print("\n" + "=" * 70)
    print("OPTUNA SEARCH WITH PRUNING (AE-MLP)")
    print("=" * 70)
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: train {fold['train_start']}~{fold['train_end']}, "
              f"valid {fold['valid_start']}~{fold['valid_end']}")
    print(f"Max trials: {args.n_trials}")
    print(f"Epochs per fold: {args.cv_epochs}")
    print(f"Pruner: MedianPruner")
    print("=" * 70)

    # 创建目标函数
    objective = OptunaObjective(
        args, handler_config, symbols,
        n_epochs=args.cv_epochs,
        early_stop=args.cv_early_stop,
        gpu=args.gpu,
        max_train_samples=args.max_samples
    )

    # 创建 Optuna study
    pruner = MedianPruner(
        n_startup_trials=5,      # 前5个 trial 不剪枝，收集基准
        n_warmup_steps=10,       # 每个 trial 前10步不剪枝
        interval_steps=5,        # 每5步检查一次
    )

    sampler = TPESampler(seed=42)

    # 支持断点续传
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{output_dir}/optuna_ae_mlp.db"

    # 如果指定 --new-study，先删除旧的 study
    if args.new_study:
        try:
            optuna.delete_study(study_name=args.study_name, storage=storage)
            print(f"\n[*] Deleted existing study '{args.study_name}'")
        except KeyError:
            pass  # study 不存在，无需删除

    # 默认 load_if_exists=True，这样可以继续之前的 study
    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',  # 最大化 IC
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    # 显示已有进度
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if completed_trials > 0:
        print(f"\n[*] Resuming study with {completed_trials} completed trials")
        print(f"    Current best IC: {study.best_value:.4f}")

    # 回调函数：打印进度
    def print_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            fold_ics = trial.user_attrs.get('fold_ics', [])
            fold_ic_str = ", ".join([f"{ic:.4f}" for ic in fold_ics])
            print(f"\n  Summary Trial {trial.number:3d}: IC={trial.value:.4f} [{fold_ic_str}] "
                  f"lr={trial.params['lr']:.5f}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"\n  Summary Trial {trial.number:3d}: PRUNED")

    # 运行优化
    print("\n[*] Running optimization...")
    study.optimize(
        objective,
        n_trials=args.n_trials,
        callbacks=[print_callback],
        show_progress_bar=True,
    )

    # 获取最佳结果
    best_trial = study.best_trial
    best_params = best_trial.params

    # 重建完整参数
    hidden_units = best_trial.user_attrs.get('hidden_units', [
        best_params['encoder_dim'],
        best_params['decoder_hidden'],
        best_params['main_layer1'],
        best_params['main_layer2'],
        best_params['main_layer3'],
    ])
    dropout_rates = [
        best_params['noise_std'],
        best_params['dropout'],
        best_params['dropout'],
        best_params['dropout'],
        best_params['dropout'],
        best_params['dropout'],
        best_params['dropout'],
    ]
    loss_weights = best_trial.user_attrs.get('loss_weights', {
        'decoder': best_params['loss_decoder'],
        'ae_action': best_params['loss_ae'],
        'action': 1.0,
    })

    full_params = {
        'num_columns': objective.num_columns,
        'hidden_units': hidden_units,
        'dropout_rates': dropout_rates,
        'lr': best_params['lr'],
        'batch_size': best_params['batch_size'],
        'loss_weights': loss_weights,
    }

    print("\n" + "=" * 70)
    print("OPTUNA SEARCH COMPLETE")
    print("=" * 70)
    print(f"Total trials: {len(study.trials)}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"\nBest IC: {best_trial.value:.4f}")
    if 'fold_ics' in best_trial.user_attrs:
        fold_ics = best_trial.user_attrs['fold_ics']
        print(f"Fold ICs: {[f'{ic:.4f}' for ic in fold_ics]}")
    print(f"\nBest parameters:")
    print(f"  hidden_units: {hidden_units}")
    print(f"  dropout: {best_params['dropout']:.4f}")
    print(f"  noise_std: {best_params['noise_std']:.4f}")
    print(f"  lr: {best_params['lr']:.6f}")
    print(f"  batch_size: {best_params['batch_size']}")
    print(f"  loss_weights: {loss_weights}")
    print("=" * 70)

    return full_params, study


def train_final_model(args, handler_config, symbols, best_params):
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
    model = build_ae_mlp_model(best_params)

    # 回调
    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_action_loss',
            patience=args.early_stop,
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

    # 训练数据
    train_outputs = {
        'decoder': X_train,
        'ae_action': y_train,
        'action': y_train,
    }
    valid_outputs = {
        'decoder': X_valid,
        'ae_action': y_valid,
        'action': y_valid,
    }

    # 训练
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
        description='AE-MLP Optuna with Pruning and Time-Series Cross-Validation',
    )

    # 基础参数
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--handler', type=str, default='alpha158-talib-macro',
                        choices=list(HANDLER_CONFIG.keys()))
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])

    # Optuna 参数
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials')
    parser.add_argument('--cv-epochs', type=int, default=50,
                        help='Max epochs per CV fold')
    parser.add_argument('--cv-early-stop', type=int, default=10,
                        help='Early stopping patience for CV trials')
    parser.add_argument('--max-samples', type=int, default=500000,
                        help='Max training samples per fold (for memory optimization)')
    parser.add_argument('--study-name', type=str, default='ae_mlp_cv_search',
                        help='Optuna study name')
    parser.add_argument('--new-study', action='store_true',
                        help='Force create a new study (delete existing one with same name)')

    # 最终训练参数
    parser.add_argument('--n-epochs', type=int, default=100,
                        help='Epochs for final model training')
    parser.add_argument('--early-stop', type=int, default=10,
                        help='Early stopping patience for final model')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--gpu-memory', type=int, default=None,
                        help='GPU memory limit in MB (e.g., 10000 for 10GB)')

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

    # 打印头部
    print("=" * 70)
    print("AE-MLP Optuna with Pruning and Time-Series Cross-Validation")
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Handler: {args.handler}")
    print(f"N-day: {args.nday}")
    print(f"Max trials: {args.n_trials}")
    print(f"CV epochs: {args.cv_epochs}")
    print(f"CV Folds: {len(CV_FOLDS)}")
    print(f"Study name: {args.study_name}")
    print(f"New study: {args.new_study}")
    gpu_info = f"GPU: {args.gpu}"
    if args.gpu_memory:
        gpu_info += f" (memory limit: {args.gpu_memory}MB)"
    print(gpu_info)
    print(f"Max samples per fold: {args.max_samples:,}")
    print("Memory optimization: ON (mixed_float16 + on-demand loading)")
    print("=" * 70)

    # 初始化
    init_qlib(handler_config['use_talib'])

    # 运行 Optuna 超参数搜索
    best_params, study = run_optuna_search(args, handler_config, symbols)

    # 保存搜索结果
    output_dir = PROJECT_ROOT / "outputs" / "optuna_cv"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存最佳参数
    params_file = output_dir / f"ae_mlp_optuna_best_{timestamp}.json"
    best_trial = study.best_trial
    params_to_save = {
        'best_ic': float(best_trial.value),
        'std_ic': float(best_trial.user_attrs.get('std_ic', 0)),
        'fold_ics': best_trial.user_attrs.get('fold_ics', []),
        'params': {
            'hidden_units': best_params['hidden_units'],
            'dropout_rates': best_params['dropout_rates'],
            'lr': float(best_params['lr']),
            'batch_size': best_params['batch_size'],
            'loss_weights': {k: float(v) for k, v in best_params['loss_weights'].items()},
        },
        'n_trials': len(study.trials),
        'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'n_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
    }
    with open(params_file, 'w') as f:
        json.dump(params_to_save, f, indent=2)
    print(f"\nBest parameters saved to: {params_file}")

    # 保存搜索历史
    history = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            history.append({
                'trial': t.number,
                'mean_ic': t.value,
                'std_ic': t.user_attrs.get('std_ic', 0),
                'hidden_units': str(t.user_attrs.get('hidden_units', [])),
                'lr': t.params.get('lr', 0),
                'batch_size': t.params.get('batch_size', 0),
                'dropout': t.params.get('dropout', 0),
            })

    if history:
        history_df = pd.DataFrame(history)
        history_file = output_dir / f"ae_mlp_optuna_history_{timestamp}.csv"
        history_df.to_csv(history_file, index=False)
        print(f"Search history saved to: {history_file}")

    # 训练最终模型
    model, test_pred, dataset = train_final_model(
        args, handler_config, symbols, best_params
    )

    # 评估
    print("\n[*] Final Evaluation on Test Set (2025)...")
    evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

    # 保存模型
    print("\n[*] Saving model...")
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_SAVE_PATH / f"ae_mlp_optuna_{args.handler}_{args.stock_pool}_{args.nday}d.keras"
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
            model_name="AE-MLP (Optuna)",
            load_model_func=load_model,
            get_feature_count_func=get_feature_count
        )

    print("\n" + "=" * 70)
    print("OPTUNA CV COMPLETE")
    print("=" * 70)
    print(f"Best CV Mean IC: {best_trial.value:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Best parameters: {params_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
