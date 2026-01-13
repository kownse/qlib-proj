"""
CNN-AE-MLP (1D-CNN + Autoencoder-enhanced MLP) Model for Qlib

在 AE-MLP 基础上增加 1D-CNN 前置特征提取器，用于捕获时序依赖关系。

核心架构:
    Input(360) → Reshape(60, 6) → Conv1D blocks → Flatten → AE-MLP

详细架构:
    1. 时序特征提取 (CNN):
       Input(360) → Reshape(60, 6) → Conv1D(filters, kernel) → BatchNorm → ReLU → Dropout
                                   → Conv1D → BatchNorm → ReLU → Dropout
                                   → GlobalAveragePooling1D / Flatten

    2. AE-MLP 部分:
       CNN_output → BatchNorm → GaussianNoise → Encoder → Decoder (重建原始输入)
                       ↓                           ↓
                       └──────── Concat ──────────┘
                                   ↓
                           MLP layers → Main Output
                                   ↓
                           Auxiliary Output (from decoder)

特点:
- 利用 Conv1D 捕获局部时序模式 (如短期趋势、波动特征)
- AE-MLP 提供正则化和特征增强
- Decoder 重建原始输入，无需预计算 CNN 特征
- 支持多种 pooling 策略
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple

# 抑制 TensorFlow 日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class CNNAEMLP:
    """
    CNN-AE-MLP 模型，在 AE-MLP 基础上增加 1D-CNN 时序特征提取

    Parameters
    ----------
    num_columns : int
        输入特征数量 (应为 time_steps * features_per_step 的乘积)
    time_steps : int
        时间步数，默认 60
    features_per_step : int
        每个时间步的特征数，默认 6
    cnn_filters : list
        Conv1D 各层的 filter 数量，例如 [32, 64]
    cnn_kernels : list
        Conv1D 各层的 kernel size，例如 [3, 3]
    cnn_pooling : str
        pooling 方式: 'flatten', 'global_avg', 'global_max'
    hidden_units : list
        AE-MLP 各层隐藏单元数，例如 [96, 96, 512, 256, 128]
    dropout_rates : list
        各层 dropout 比例
    lr : float
        学习率
    n_epochs : int
        训练轮数
    batch_size : int
        批次大小
    early_stop : int
        早停耐心值
    loss_weights : dict
        各输出的损失权重
    GPU : int
        GPU 设备 ID，-1 表示使用 CPU
    seed : int
        随机种子
    """

    def __init__(
        self,
        num_columns: int,
        time_steps: int = 60,
        features_per_step: int = 6,
        cnn_filters: List[int] = None,
        cnn_kernels: List[int] = None,
        cnn_pooling: str = 'global_avg',
        hidden_units: List[int] = None,
        dropout_rates: List[float] = None,
        lr: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 2048,
        early_stop: int = 10,
        loss_weights: Dict[str, float] = None,
        GPU: int = 0,
        seed: int = 42,
    ):
        self.num_columns = num_columns
        self.time_steps = time_steps
        self.features_per_step = features_per_step
        self.cnn_filters = cnn_filters or [32, 64]
        self.cnn_kernels = cnn_kernels or [3, 3]
        self.cnn_pooling = cnn_pooling
        self.hidden_units = hidden_units or [96, 96, 512, 256, 128]
        self.dropout_rates = dropout_rates or [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.loss_weights = loss_weights or {'decoder': 0.1, 'ae_action': 0.1, 'action': 1.0}
        self.GPU = GPU
        self.seed = seed

        self.model = None
        self.fitted = False

        # 设置设备
        self._setup_device()

        # 设置随机种子
        tf.random.set_seed(seed)
        np.random.seed(seed)

    def _setup_device(self):
        """配置 GPU/CPU"""
        gpus = tf.config.list_physical_devices('GPU')
        if self.GPU >= 0 and gpus:
            try:
                tf.config.set_visible_devices(gpus[self.GPU], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[self.GPU], True)
                print(f"    Using GPU: {gpus[self.GPU]}")
            except RuntimeError as e:
                print(f"    GPU setup error: {e}")
        else:
            tf.config.set_visible_devices([], 'GPU')
            print("    Using CPU")

    def _build_cnn_block(self, x, filters, kernel_size, name_prefix):
        """构建单个 CNN block: Conv1D → BatchNorm → ReLU → Dropout"""
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            name=f'{name_prefix}_conv'
        )(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu')(x)
        x = layers.Dropout(self.dropout_rates[0], name=f'{name_prefix}_dropout')(x)
        return x

    def _build_model(self) -> Model:
        """
        构建 CNN-AE-MLP 模型

        架构:
        1. Input(num_columns) → Reshape(time_steps, features_per_step)
        2. Conv1D blocks → Pooling → CNN features
        3. AE-MLP: Encoder → Decoder (重建原始输入) + Main branch
        """
        # ========== Part 1: 1D-CNN 时序特征提取 ==========
        inp = layers.Input(shape=(self.num_columns,), name='input')

        # Reshape: (batch, 360) → (batch, 60, 6)
        x = layers.Reshape((self.time_steps, self.features_per_step), name='reshape')(inp)

        # Conv1D blocks
        for i, (filters, kernel) in enumerate(zip(self.cnn_filters, self.cnn_kernels)):
            x = self._build_cnn_block(x, filters, kernel, f'cnn{i+1}')

        # Pooling (使用 global pooling 减少参数量和内存)
        if self.cnn_pooling == 'global_avg':
            cnn_out = layers.GlobalAveragePooling1D(name='cnn_pool')(x)
        elif self.cnn_pooling == 'global_max':
            cnn_out = layers.GlobalMaxPooling1D(name='cnn_pool')(x)
        else:  # flatten
            cnn_out = layers.Flatten(name='cnn_flatten')(x)

        # ========== Part 2: AE-MLP ==========
        # 输入标准化 (对 CNN 输出)
        x0 = layers.BatchNormalization(name='ae_input_bn')(cnn_out)

        # Encoder
        encoder = layers.GaussianNoise(self.dropout_rates[0], name='noise')(x0)
        encoder = layers.Dense(self.hidden_units[0], name='encoder_dense')(encoder)
        encoder = layers.BatchNormalization(name='encoder_bn')(encoder)
        encoder = layers.Activation('swish', name='encoder_act')(encoder)

        # Decoder (重建原始输入，而非 CNN 输出)
        decoder = layers.Dropout(self.dropout_rates[1], name='decoder_dropout')(encoder)
        decoder = layers.Dense(self.num_columns, name='decoder')(decoder)

        # 辅助预测分支 (基于 decoder 输出)
        x_ae = layers.Dense(self.hidden_units[1], name='ae_dense1')(decoder)
        x_ae = layers.BatchNormalization(name='ae_bn1')(x_ae)
        x_ae = layers.Activation('swish', name='ae_act1')(x_ae)
        x_ae = layers.Dropout(self.dropout_rates[2], name='ae_dropout1')(x_ae)

        # 辅助输出 (回归)
        out_ae = layers.Dense(1, name='ae_action')(x_ae)

        # 主分支: CNN特征 + encoder 特征
        x = layers.Concatenate(name='concat')([x0, encoder])
        x = layers.BatchNormalization(name='main_bn0')(x)
        x = layers.Dropout(self.dropout_rates[3], name='main_dropout0')(x)

        # MLP 主体
        for i in range(2, len(self.hidden_units)):
            dropout_idx = i + 2
            if dropout_idx >= len(self.dropout_rates):
                dropout_idx = len(self.dropout_rates) - 1

            x = layers.Dense(self.hidden_units[i], name=f'main_dense{i-1}')(x)
            x = layers.BatchNormalization(name=f'main_bn{i-1}')(x)
            x = layers.Activation('swish', name=f'main_act{i-1}')(x)
            x = layers.Dropout(self.dropout_rates[dropout_idx], name=f'main_dropout{i-1}')(x)

        # 主输出 (回归)
        out = layers.Dense(1, name='action')(x)

        # 构建模型
        model = Model(inputs=inp, outputs=[decoder, out_ae, out], name='CNN_AE_MLP')

        # 编译
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss={
                'decoder': 'mse',       # 重建损失 (重建原始输入)
                'ae_action': 'mse',     # 辅助预测损失
                'action': 'mse',        # 主预测损失
            },
            loss_weights=self.loss_weights,
            metrics={
                'decoder': keras.metrics.MeanAbsoluteError(name='MAE'),
                'ae_action': keras.metrics.MeanAbsoluteError(name='MAE'),
                'action': keras.metrics.MeanAbsoluteError(name='MAE'),
            },
        )

        return model

    def _prepare_data(self, dataset: DatasetH, segment: str):
        """
        从 Qlib Dataset 准备数据

        Parameters
        ----------
        dataset : DatasetH
            Qlib 数据集
        segment : str
            数据段: 'train', 'valid', 'test'

        Returns
        -------
        tuple
            (features, labels) 或 (features,) 如果没有 labels
        """
        features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)

        # 处理 NaN 和异常值
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)

        # Clip 极端值
        features = features.clip(-10, 10)

        try:
            labels = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)
            if isinstance(labels, pd.DataFrame):
                labels = labels.iloc[:, 0]
            labels = labels.fillna(0).values
            return features.values, labels
        except Exception:
            return features.values, None

    def fit(self, dataset: DatasetH):
        """
        训练模型

        Parameters
        ----------
        dataset : DatasetH
            Qlib 数据集
        """
        print("\n    Preparing training data...")
        X_train, y_train = self._prepare_data(dataset, "train")
        X_valid, y_valid = self._prepare_data(dataset, "valid")

        print(f"    Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")

        # 检查并更新输入维度
        actual_features = X_train.shape[1]
        expected_features = self.time_steps * self.features_per_step
        if actual_features != expected_features:
            print(f"    WARNING: Feature count mismatch!")
            print(f"    Expected: {expected_features} ({self.time_steps} × {self.features_per_step})")
            print(f"    Actual: {actual_features}")

            # 尝试自动调整
            if actual_features % self.features_per_step == 0:
                self.time_steps = actual_features // self.features_per_step
                print(f"    Auto-adjusted time_steps to: {self.time_steps}")
            elif actual_features % self.time_steps == 0:
                self.features_per_step = actual_features // self.time_steps
                print(f"    Auto-adjusted features_per_step to: {self.features_per_step}")
            else:
                # 使用最接近的因子分解
                for ts in [60, 30, 20, 15, 12, 10, 6, 5, 4, 3, 2]:
                    if actual_features % ts == 0:
                        self.time_steps = ts
                        self.features_per_step = actual_features // ts
                        print(f"    Auto-adjusted to: time_steps={self.time_steps}, features_per_step={self.features_per_step}")
                        break

            self.num_columns = actual_features

        # 构建模型
        print("\n    Building CNN-AE-MLP model...")
        self.model = self._build_model()
        self.model.summary(print_fn=lambda x: print(f"    {x}"))

        # 回调函数
        cb_list = [
            callbacks.EarlyStopping(
                monitor='val_action_loss',
                patience=self.early_stop,
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

        # 训练数据: 多输出格式
        # decoder 输出 = 原始输入 (自编码器重建)
        # ae_action 和 action 输出 = 标签
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
        print("\n    Training...")
        history = self.model.fit(
            X_train,
            train_outputs,
            validation_data=(X_valid, valid_outputs),
            epochs=self.n_epochs,
            batch_size=self.batch_size,
            callbacks=cb_list,
            verbose=1,
        )

        self.fitted = True
        print("\n    ✓ Training completed")

        return history

    def predict(self, dataset: DatasetH, segment: str = "test") -> pd.Series:
        """
        预测

        Parameters
        ----------
        dataset : DatasetH
            Qlib 数据集
        segment : str
            数据段

        Returns
        -------
        pd.Series
            预测结果，index 为 (datetime, instrument)
        """
        if not self.fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # 获取原始数据以保留 index
        features_df = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
        index = features_df.index

        # 准备数据
        X, _ = self._prepare_data(dataset, segment)

        # 预测 (返回三个输出)
        _, _, pred = self.model.predict(X, batch_size=self.batch_size, verbose=0)

        # 转换为 Series
        pred_series = pd.Series(pred.flatten(), index=index, name='score')

        return pred_series

    def save(self, path: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)

        # 保存额外配置
        config_path = path.replace('.keras', '_config.npz')
        np.savez(
            config_path,
            time_steps=self.time_steps,
            features_per_step=self.features_per_step,
        )
        print(f"    ✓ Model saved to: {path}")

    @classmethod
    def load(cls, path: str, **kwargs) -> 'CNNAEMLP':
        """加载模型"""
        instance = cls(num_columns=1, **kwargs)

        # 加载配置
        config_path = path.replace('.keras', '_config.npz')
        if os.path.exists(config_path):
            config = np.load(config_path)
            instance.time_steps = int(config['time_steps'])
            instance.features_per_step = int(config['features_per_step'])

        instance.model = keras.models.load_model(path)
        instance.num_columns = instance.model.input_shape[1]
        instance.fitted = True
        print(f"    ✓ Model loaded from: {path}")
        return instance


def create_cnn_ae_mlp_for_handler(handler_type: str, **kwargs) -> CNNAEMLP:
    """
    根据 handler 类型创建 CNN-AE-MLP 模型

    Parameters
    ----------
    handler_type : str
        Handler 类型: alpha158, alpha360, alpha158_vol_talib 等
    **kwargs
        其他参数传递给 CNNAEMLP

    Returns
    -------
    CNNAEMLP
    """
    # 不同 handler 的默认配置
    HANDLER_CONFIG = {
        'alpha158': {
            'num_columns': 158,
            'time_steps': 79,  # 158 = 79 × 2 或自动调整
            'features_per_step': 2,
            'cnn_filters': [32, 64],
            'cnn_kernels': [3, 3],
            'cnn_pooling': 'global_avg',
            'hidden_units': [64, 64, 256, 128, 64],
            'dropout_rates': [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
        },
        'alpha360': {
            'num_columns': 360,
            'time_steps': 60,
            'features_per_step': 6,
            'cnn_filters': [32, 64],
            'cnn_kernels': [3, 3],
            'cnn_pooling': 'global_avg',
            'hidden_units': [96, 96, 512, 256, 128],
            'dropout_rates': [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        },
        'alpha158_vol': {
            'num_columns': 158,
            'time_steps': 79,
            'features_per_step': 2,
            'cnn_filters': [32, 64],
            'cnn_kernels': [3, 3],
            'cnn_pooling': 'global_avg',
            'hidden_units': [64, 64, 256, 128, 64],
            'dropout_rates': [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
        },
        'alpha158_vol_talib': {
            'num_columns': 180,
            'time_steps': 60,
            'features_per_step': 3,
            'cnn_filters': [32, 64],
            'cnn_kernels': [3, 3],
            'cnn_pooling': 'global_avg',
            'hidden_units': [96, 96, 384, 192, 96],
            'dropout_rates': [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
        },
        'alpha158_news': {
            'num_columns': 170,
            'time_steps': 85,
            'features_per_step': 2,
            'cnn_filters': [32, 64],
            'cnn_kernels': [3, 3],
            'cnn_pooling': 'global_avg',
            'hidden_units': [96, 96, 384, 192, 96],
            'dropout_rates': [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
        },
    }

    config = HANDLER_CONFIG.get(handler_type, HANDLER_CONFIG['alpha360'])

    # 合并用户提供的参数
    for key, value in config.items():
        if key not in kwargs:
            kwargs[key] = value

    return CNNAEMLP(**kwargs)
