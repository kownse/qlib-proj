"""
Multi-Horizon AE-MLP Model

在原始 AE-MLP 基础上增加 multi-horizon prediction heads。
共享编码器学习通用特征表示，多个 prediction heads 分别预测不同时间窗口的 forward return。
辅助 horizon 作为 auxiliary tasks 提供额外的梯度信号，起到隐式正则化的作用。

架构:
    Input → BatchNorm → GaussianNoise
    → Encoder: Dense → BatchNorm → Swish
    → Decoder: Dropout → Dense (重建输入)
    → 辅助AE分支: Dense → BN → Swish → Dropout → Dense (AE辅助预测)
    → 主分支: Concat(原始, encoder) → BN → Dropout → MLP layers
        ├── Head_2d  (辅助, 权重小)
        ├── Head_5d  (主目标, 权重大) ← 推理时只用这个
        └── Head_10d (辅助, 权重小)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from typing import List, Dict, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, Model

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class AEMLPMultiHorizon:
    """
    Multi-Horizon AE-MLP 模型。

    在 AE-MLP 的基础上，主分支分出多个 prediction heads，
    每个 head 负责预测一个时间窗口的 forward return。
    训练时所有 heads 同时优化，推理时只使用 primary_horizon 对应的 head。
    """

    def __init__(
        self,
        num_columns: int,
        horizons: List[int] = None,
        primary_horizon: int = 5,
        hidden_units: List[int] = None,
        dropout_rates: List[float] = None,
        lr: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 4096,
        early_stop: int = 10,
        loss_weights: Dict[str, float] = None,
        aux_horizon_weight: float = 0.3,
        GPU: int = 0,
        seed: int = 42,
        verbose: int = 1,
    ):
        """
        Args:
            num_columns: 输入特征数量
            horizons: 预测的时间窗口列表，如 [2, 5, 10]
            primary_horizon: 主要预测目标的天数（用于 early stopping 和推理）
            hidden_units: MLP 各层的隐藏单元数
            dropout_rates: 各层的 dropout 率
            lr: 学习率
            n_epochs: 最大训练轮数
            batch_size: batch 大小
            early_stop: early stopping patience
            loss_weights: 各输出头的损失权重（decoder, ae_action, horizon heads）
            aux_horizon_weight: 辅助 horizon heads 的默认损失权重
            GPU: GPU 设备 ID（-1 表示 CPU）
            seed: 随机种子
            verbose: 日志级别
        """
        self.num_columns = num_columns
        self.horizons = horizons or [2, 5, 10]
        self.primary_horizon = primary_horizon
        self.hidden_units = hidden_units or [96, 96, 512, 256, 128]
        self.dropout_rates = dropout_rates or [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.aux_horizon_weight = aux_horizon_weight
        self.GPU = GPU
        self.seed = seed
        self.verbose = verbose

        # 构建 loss_weights
        if loss_weights is not None:
            self.loss_weights = loss_weights
        else:
            self.loss_weights = {'decoder': 0.1, 'ae_action': 0.1}
            for h in self.horizons:
                key = f'action_{h}d'
                if h == self.primary_horizon:
                    self.loss_weights[key] = 1.0
                else:
                    self.loss_weights[key] = self.aux_horizon_weight

        # horizon 列名映射
        self.horizon_label_map = {h: f'LABEL_{h}d' for h in self.horizons}
        self.primary_label = f'LABEL_{self.primary_horizon}d'
        self.primary_output_name = f'action_{self.primary_horizon}d'

        self.model = None
        self.fitted = False

        self._setup_device()

        tf.random.set_seed(seed)
        np.random.seed(seed)

    def _setup_device(self):
        """设置计算设备"""
        if self.GPU >= 0:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    tf.config.set_visible_devices(gpus[self.GPU], 'GPU')
                    tf.config.experimental.set_memory_growth(gpus[self.GPU], True)
                    if self.verbose > 0:
                        print(f"    Using GPU: {gpus[self.GPU].name}")
                except (RuntimeError, IndexError) as e:
                    print(f"    GPU setup failed: {e}, using CPU")
            else:
                if self.verbose > 0:
                    print("    No GPU found, using CPU")
        else:
            tf.config.set_visible_devices([], 'GPU')
            if self.verbose > 0:
                print("    Using CPU")

    def _build_model(self) -> Model:
        """
        构建 Multi-Horizon AE-MLP 模型。

        与原始 AE-MLP 相同的编码器和主分支，
        但在主分支末端分出多个 prediction heads。
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
        out_ae = layers.Dense(1, name='ae_action')(x_ae)

        # 主分支: 原始特征 + encoder 特征
        x = layers.Concatenate(name='concat')([x0, encoder])
        x = layers.BatchNormalization(name='main_bn0')(x)
        x = layers.Dropout(self.dropout_rates[3], name='main_dropout0')(x)

        # MLP 主体 (共享层)
        for i in range(2, len(self.hidden_units)):
            dropout_idx = min(i + 2, len(self.dropout_rates) - 1)
            x = layers.Dense(self.hidden_units[i], name=f'main_dense{i-1}')(x)
            x = layers.BatchNormalization(name=f'main_bn{i-1}')(x)
            x = layers.Activation('swish', name=f'main_act{i-1}')(x)
            x = layers.Dropout(self.dropout_rates[dropout_idx], name=f'main_dropout{i-1}')(x)

        # Multi-horizon prediction heads
        outputs = [decoder, out_ae]  # decoder + ae_action 放前面
        output_names = ['decoder', 'ae_action']

        for h in self.horizons:
            head_name = f'action_{h}d'
            # 每个 head 有自己的小 MLP (一层 dense)
            head = layers.Dense(32, name=f'head_{h}d_dense')(x)
            head = layers.Activation('swish', name=f'head_{h}d_act')(head)
            head = layers.Dense(1, name=head_name)(head)
            outputs.append(head)
            output_names.append(head_name)

        # 构建模型
        model = Model(inputs=inp, outputs=outputs, name='AE_MLP_MultiHorizon')

        # 编译
        losses = {
            'decoder': 'mse',
            'ae_action': 'mse',
        }
        metrics_dict = {
            'decoder': keras.metrics.MeanAbsoluteError(name='MAE'),
            'ae_action': keras.metrics.MeanAbsoluteError(name='MAE'),
        }
        for h in self.horizons:
            head_name = f'action_{h}d'
            losses[head_name] = 'mse'
            metrics_dict[head_name] = keras.metrics.MeanAbsoluteError(name='MAE')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss=losses,
            loss_weights=self.loss_weights,
            metrics=metrics_dict,
        )

        return model

    def _prepare_data(self, dataset: DatasetH, segment: str):
        """
        从 Qlib Dataset 准备多标签数据。

        Returns:
            (features, labels_dict) 其中 labels_dict = {horizon: values}
        """
        features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)

        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        features = features.clip(-10, 10)

        try:
            labels_df = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)
            if isinstance(labels_df, pd.Series):
                labels_df = labels_df.to_frame()

            # 处理 MultiIndex 列名: ('label', 'LABEL_5d') → 'LABEL_5d'
            if isinstance(labels_df.columns, pd.MultiIndex):
                # 取第二级作为列名
                labels_df.columns = [col[1] if isinstance(col, tuple) else col for col in labels_df.columns]

            labels_dict = {}
            for h in self.horizons:
                col_name = f'LABEL_{h}d'
                if col_name in labels_df.columns:
                    labels_dict[h] = labels_df[col_name].fillna(0).values
                elif labels_df.shape[1] == 1:
                    # 单标签回退: 所有 heads 用同一标签
                    labels_dict[h] = labels_df.iloc[:, 0].fillna(0).values
                else:
                    # 按位置匹配
                    idx = self.horizons.index(h)
                    if idx < labels_df.shape[1]:
                        labels_dict[h] = labels_df.iloc[:, idx].fillna(0).values
                    else:
                        labels_dict[h] = labels_df.iloc[:, 0].fillna(0).values

            return features.values, labels_dict
        except Exception:
            return features.values, None

    def fit(self, dataset: DatasetH):
        """训练模型"""
        if self.verbose > 0:
            print("\n    Preparing training data...")
        X_train, y_train_dict = self._prepare_data(dataset, "train")
        X_valid, y_valid_dict = self._prepare_data(dataset, "valid")

        if self.verbose > 0:
            print(f"    Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")
            print(f"    Horizons: {self.horizons}, Primary: {self.primary_horizon}d")
            for h in self.horizons:
                if h in y_train_dict:
                    y = y_train_dict[h]
                    print(f"    Label {h}d: mean={y.mean():.6f}, std={y.std():.6f}")

        # 更新输入维度
        actual_features = X_train.shape[1]
        if actual_features != self.num_columns:
            if self.verbose > 0:
                print(f"    Updating num_columns: {self.num_columns} -> {actual_features}")
            self.num_columns = actual_features

        # 构建模型
        if self.verbose > 0:
            print("\n    Building Multi-Horizon AE-MLP model...")
        self.model = self._build_model()
        if self.verbose > 1:
            self.model.summary(print_fn=lambda x: print(f"    {x}"))

        # 回调函数: 监控主目标的 validation loss
        primary_loss_name = f'val_action_{self.primary_horizon}d_loss'
        cb_list = [
            callbacks.EarlyStopping(
                monitor=primary_loss_name,
                patience=self.early_stop,
                restore_best_weights=True,
                verbose=self.verbose,
                mode='min',
            ),
            callbacks.ReduceLROnPlateau(
                monitor=primary_loss_name,
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=self.verbose,
                mode='min',
            ),
        ]

        # 构建多输出训练数据
        train_outputs = {
            'decoder': X_train,
            'ae_action': y_train_dict[self.primary_horizon],
        }
        valid_outputs = {
            'decoder': X_valid,
            'ae_action': y_valid_dict[self.primary_horizon],
        }
        for h in self.horizons:
            head_name = f'action_{h}d'
            train_outputs[head_name] = y_train_dict[h]
            valid_outputs[head_name] = y_valid_dict[h]

        # 训练
        if self.verbose > 0:
            print("\n    Training...")
            print(f"    Loss weights: {self.loss_weights}")
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
            print("\n    Training completed")

        return history

    def predict(self, dataset: DatasetH, segment: str = "test") -> pd.Series:
        """
        预测。只返回 primary_horizon 对应 head 的输出。

        Returns:
            pd.Series: 预测结果，index 为 (datetime, instrument)
        """
        if not self.fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        features_df = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
        index = features_df.index

        X, _ = self._prepare_data(dataset, segment)

        # 模型返回多个输出: [decoder, ae_action, action_2d, action_5d, action_10d, ...]
        all_preds = self.model.predict(X, batch_size=self.batch_size, verbose=0)

        # 找到 primary horizon 对应的输出索引
        # outputs 顺序: decoder(0), ae_action(1), action_Xd(2+)
        primary_idx = 2 + self.horizons.index(self.primary_horizon)
        pred = all_preds[primary_idx]

        pred_series = pd.Series(pred.flatten(), index=index, name='score')
        return pred_series

    def predict_all_horizons(self, dataset: DatasetH, segment: str = "test") -> Dict[int, pd.Series]:
        """
        返回所有 horizon 的预测结果。

        Returns:
            Dict[int, pd.Series]: {horizon_days: prediction_series}
        """
        if not self.fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        features_df = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
        index = features_df.index

        X, _ = self._prepare_data(dataset, segment)
        all_preds = self.model.predict(X, batch_size=self.batch_size, verbose=0)

        results = {}
        for i, h in enumerate(self.horizons):
            pred_idx = 2 + i  # skip decoder and ae_action
            pred = all_preds[pred_idx]
            results[h] = pd.Series(pred.flatten(), index=index, name=f'score_{h}d')

        return results

    def save(self, path: str, verbose: bool = True):
        """保存模型"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)
        if verbose:
            print(f"    Model saved to: {path}")

    @classmethod
    def load(cls, path: str, horizons: List[int] = None, primary_horizon: int = 5, **kwargs) -> 'AEMLPMultiHorizon':
        """加载模型"""
        instance = cls(
            num_columns=1,
            horizons=horizons or [2, 5, 10],
            primary_horizon=primary_horizon,
            **kwargs,
        )
        instance.model = keras.models.load_model(path)
        instance.num_columns = instance.model.input_shape[1]
        instance.fitted = True
        print(f"    Model loaded from: {path}")
        return instance
