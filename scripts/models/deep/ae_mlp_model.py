"""
AE-MLP (Autoencoder-enhanced MLP) Model for Qlib

基于 Kaggle 竞赛中常用的 AE-MLP 架构，适配 Qlib 回归任务。

核心思想：
1. Encoder: 将原始特征压缩到低维空间，学习有效表示
2. Decoder: 重建原始输入，作为正则化手段
3. 特征拼接: 原始特征 + encoder输出，形成增强表示
4. 多任务输出: decoder重建 + 辅助预测 + 主预测

架构图:
    Input → BatchNorm → GaussianNoise → Encoder → Decoder (重建输出)
                ↓                           ↓
                └──────── Concat ──────────┘
                            ↓
                    MLP layers → Main Output
                            ↓
                    Auxiliary Output (from decoder)

# 为了支持时序信息，可以：
#   # 方案1: 先用1D-CNN提取时序特征，再接AE-MLP
#   Input(360) → Reshape(60, 6) → Conv1D → Flatten → AE-MLP

#   # 方案2: 用时序自编码器
#   Input(60, 6) → LSTM Encoder → LSTM Decoder → MLP
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

# 抑制 TensorFlow 日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class AEMLP:
    """
    AE-MLP 模型，适配 Qlib 接口

    Parameters
    ----------
    num_columns : int
        输入特征数量
    hidden_units : list
        各层隐藏单元数，例如 [96, 96, 512, 256, 128]
        - hidden_units[0]: encoder 输出维度
        - hidden_units[1]: decoder 后的 MLP 第一层
        - hidden_units[2:]: 主分支 MLP 各层
    dropout_rates : list
        各层 dropout 比例
        - dropout_rates[0]: GaussianNoise 标准差
        - dropout_rates[1]: encoder 后 dropout
        - dropout_rates[2]: decoder 后 dropout
        - dropout_rates[3]: concat 后 dropout
        - dropout_rates[4:]: 主分支各层 dropout
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
        hidden_units: List[int] = None,
        dropout_rates: List[float] = None,
        lr: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 4096,
        early_stop: int = 10,
        loss_weights: Dict[str, float] = None,
        GPU: int = 0,
        seed: int = 42,
        verbose: int = 1,
    ):
        self.num_columns = num_columns
        self.hidden_units = hidden_units or [96, 96, 512, 256, 128]
        self.dropout_rates = dropout_rates or [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.loss_weights = loss_weights or {'decoder': 0.1, 'ae_action': 0.1, 'action': 1.0}
        self.GPU = GPU
        self.seed = seed
        self.verbose = verbose

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
                # 只使用指定的 GPU
                tf.config.set_visible_devices(gpus[self.GPU], 'GPU')
                # 允许显存按需增长
                tf.config.experimental.set_memory_growth(gpus[self.GPU], True)
                if self.verbose > 0:
                    print(f"    Using GPU: {gpus[self.GPU]}")
            except RuntimeError as e:
                if self.verbose > 0:
                    print(f"    GPU setup error: {e}")
        else:
            tf.config.set_visible_devices([], 'GPU')
            if self.verbose > 0:
                print("    Using CPU")

    def _build_model(self) -> Model:
        """
        构建 AE-MLP 模型

        架构:
        1. Input → BatchNorm → GaussianNoise
        2. Encoder: Dense → BatchNorm → Swish
        3. Decoder: Dropout → Dense (重建输入)
        4. 辅助分支: Dense → BatchNorm → Swish → Dropout → Dense (辅助预测)
        5. 主分支: Concat(原始, encoder) → BatchNorm → Dropout → MLP layers → Dense (主预测)
        """
        inp = layers.Input(shape=(self.num_columns,), name='input')

        # 输入标准化
        x0 = layers.BatchNormalization(name='input_bn')(inp)

        # Encoder
        encoder = layers.GaussianNoise(self.dropout_rates[0], name='noise')(x0)
        encoder = layers.Dense(self.hidden_units[0], name='encoder_dense')(encoder)
        encoder = layers.BatchNormalization(name='encoder_bn')(encoder)
        encoder = layers.Activation('swish', name='encoder_act')(encoder)

        # Decoder (重建原始输入)
        decoder = layers.Dropout(self.dropout_rates[1], name='decoder_dropout')(encoder)
        decoder = layers.Dense(self.num_columns, name='decoder')(decoder)

        # 辅助预测分支 (基于 decoder 输出)
        x_ae = layers.Dense(self.hidden_units[1], name='ae_dense1')(decoder)
        x_ae = layers.BatchNormalization(name='ae_bn1')(x_ae)
        x_ae = layers.Activation('swish', name='ae_act1')(x_ae)
        x_ae = layers.Dropout(self.dropout_rates[2], name='ae_dropout1')(x_ae)

        # 辅助输出 (回归)
        out_ae = layers.Dense(1, name='ae_action')(x_ae)

        # 主分支: 原始特征 + encoder 特征
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
        model = Model(inputs=inp, outputs=[decoder, out_ae, out], name='AE_MLP')

        # 编译
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss={
                'decoder': 'mse',       # 重建损失
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
        if self.verbose > 0:
            print("\n    Preparing training data...")
        X_train, y_train = self._prepare_data(dataset, "train")
        X_valid, y_valid = self._prepare_data(dataset, "valid")

        if self.verbose > 0:
            print(f"    Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")

        # 更新输入维度
        actual_features = X_train.shape[1]
        if actual_features != self.num_columns:
            if self.verbose > 0:
                print(f"    Updating num_columns: {self.num_columns} -> {actual_features}")
            self.num_columns = actual_features

        # 构建模型
        if self.verbose > 0:
            print("\n    Building AE-MLP model...")
        self.model = self._build_model()
        if self.verbose > 1:
            self.model.summary(print_fn=lambda x: print(f"    {x}"))

        # 回调函数
        cb_list = [
            callbacks.EarlyStopping(
                monitor='val_action_loss',
                patience=self.early_stop,
                restore_best_weights=True,
                verbose=self.verbose,
                mode='min'  # 监控损失，越小越好
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_action_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=self.verbose,
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
        if self.verbose > 0:
            print("\n    Training...")
        history = self.model.fit(
            X_train,
            train_outputs,
            validation_data=(X_valid, valid_outputs),
            epochs=self.n_epochs,
            batch_size=self.batch_size,
            callbacks=cb_list,
            verbose=self.verbose,
        )

        self.fitted = True
        if self.verbose > 0:
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

    def save(self, path: str, verbose: bool = True):
        """保存模型"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)
        if verbose:
            print(f"    ✓ Model saved to: {path}")

    @classmethod
    def load(cls, path: str, **kwargs) -> 'AEMLP':
        """加载模型"""
        instance = cls(num_columns=1, **kwargs)  # num_columns 会从加载的模型中推断
        instance.model = keras.models.load_model(path)
        instance.num_columns = instance.model.input_shape[1]
        instance.fitted = True
        print(f"    ✓ Model loaded from: {path}")
        return instance


def create_ae_mlp_for_handler(handler_type: str, **kwargs) -> AEMLP:
    """
    根据 handler 类型创建 AE-MLP 模型

    Parameters
    ----------
    handler_type : str
        Handler 类型: alpha158, alpha360, alpha158_vol_talib 等
    **kwargs
        其他参数传递给 AEMLP

    Returns
    -------
    AEMLP
    """
    # 不同 handler 的默认特征数
    HANDLER_FEATURES = {
        'alpha158': 158,
        'alpha360': 360,
        'alpha158_vol': 158,
        'alpha158_vol_talib': 180,  # 158 + TA-Lib 指标
        'alpha158_news': 170,  # 158 + news 特征
    }

    num_columns = HANDLER_FEATURES.get(handler_type, 158)

    # 根据特征数调整网络结构
    if num_columns <= 160:
        # Alpha158 级别
        hidden_units = [64, 64, 256, 128, 64]
        dropout_rates = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
    elif num_columns <= 200:
        # Alpha158 + TALib 级别
        hidden_units = [96, 96, 384, 192, 96]
        dropout_rates = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
    else:
        # Alpha360 级别
        hidden_units = [128, 128, 512, 256, 128]
        dropout_rates = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]

    return AEMLP(
        num_columns=num_columns,
        hidden_units=hidden_units,
        dropout_rates=dropout_rates,
        **kwargs
    )
