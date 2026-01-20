"""
CNN-AE-MLP V2 - 改进版 CNN-Autoencoder-MLP 模型

针对 Alpha300 (60天 × 5特征) 时序数据优化的深度学习模型。

主要改进:
1. 多尺度卷积 (Multi-scale Conv1D) - 捕获不同时间尺度的模式
2. 残差连接 (Residual Connections) - 防止梯度消失，支持更深网络
3. 时序注意力 (Temporal Attention) - 聚焦重要时间点
4. 特征交互层 (Feature Interaction) - 建模 OHLCV 之间的关系
5. 改进的 Decoder - 重建 CNN 特征而非原始输入
6. 可选的 IC 损失 - 直接优化排序能力

架构:
    Input(300) → Reshape(60, 5)
        ↓
    Feature Interaction (OHLCV 交互)
        ↓
    Multi-scale Conv1D (kernel 3, 5, 10)
        ↓
    Residual Block × N
        ↓
    Temporal Attention (可选)
        ↓
    GlobalAvgPool → CNN_features
        ↓
    AE-MLP (Encoder → Decoder 重建 CNN_features)
        ↓
    Main Output

使用方法:
    from models.deep.cnn_ae_mlp_v2_model import CNNAEMLPV2

    model = CNNAEMLPV2(
        num_columns=300,
        time_steps=60,
        features_per_step=5,
        use_attention=True,
        use_residual=True,
    )
    model.fit(dataset)
    predictions = model.predict(dataset, "test")
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks, regularizers

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


def ic_loss(y_true, y_pred):
    """
    IC 损失函数：最大化预测与真实值的 Pearson 相关系数

    IC = corr(y_pred, y_true)
    Loss = -IC (最小化负IC = 最大化IC)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_centered = y_true - tf.reduce_mean(y_true)
    y_pred_centered = y_pred - tf.reduce_mean(y_pred)

    cov = tf.reduce_mean(y_true_centered * y_pred_centered)
    std_true = tf.math.reduce_std(y_true) + 1e-8
    std_pred = tf.math.reduce_std(y_pred) + 1e-8

    ic = cov / (std_true * std_pred)
    return -ic


def combined_loss(y_true, y_pred, mse_weight=0.5, ic_weight=0.5):
    """组合损失：MSE + IC"""
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ic = -ic_loss(y_true, y_pred)  # ic_loss 返回负IC
    return mse_weight * mse - ic_weight * ic


def ae_action_loss(y_true, y_pred):
    """AE 分支损失：MSE 0.7 + IC 0.3"""
    return combined_loss(y_true, y_pred, mse_weight=0.7, ic_weight=0.3)


def action_loss(y_true, y_pred):
    """主输出损失：MSE 0.5 + IC 0.5"""
    return combined_loss(y_true, y_pred, mse_weight=0.5, ic_weight=0.5)


class CNNAEMLPV2:
    """
    CNN-AE-MLP V2 模型

    Parameters
    ----------
    num_columns : int
        输入特征总数 (time_steps × features_per_step)
    time_steps : int
        时间步数，默认 60
    features_per_step : int
        每个时间步的特征数，默认 5 (OHLCV)
    cnn_filters : list
        多尺度 Conv1D 的 filter 数量
    multiscale_kernels : list
        多尺度卷积的 kernel sizes，默认 [3, 5, 10]
    num_residual_blocks : int
        残差块数量
    use_attention : bool
        是否使用时序注意力
    attention_heads : int
        注意力头数
    use_feature_interaction : bool
        是否使用特征交互层
    hidden_units : list
        AE-MLP 隐藏层单元数
    dropout_rates : list
        各层 dropout 比例
    l2_reg : float
        L2 正则化系数
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
    use_ic_loss : bool
        是否使用 IC 损失
    GPU : int
        GPU 设备 ID
    seed : int
        随机种子
    """

    def __init__(
        self,
        num_columns: int = 300,
        time_steps: int = 60,
        features_per_step: int = 5,
        cnn_filters: List[int] = None,
        multiscale_kernels: List[int] = None,
        num_residual_blocks: int = 2,
        use_attention: bool = True,
        attention_heads: int = 4,
        use_feature_interaction: bool = True,
        hidden_units: List[int] = None,
        dropout_rates: List[float] = None,
        l2_reg: float = 1e-5,
        lr: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 2048,
        early_stop: int = 10,
        loss_weights: Dict[str, float] = None,
        use_ic_loss: bool = False,
        gradient_accumulation_steps: int = 1,
        GPU: int = 0,
        seed: int = 42,
    ):
        self.num_columns = num_columns
        self.time_steps = time_steps
        self.features_per_step = features_per_step
        self.cnn_filters = cnn_filters or [64, 128]
        self.multiscale_kernels = multiscale_kernels or [3, 5, 10]
        self.num_residual_blocks = num_residual_blocks
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.use_feature_interaction = use_feature_interaction
        self.hidden_units = hidden_units or [128, 128, 512, 256, 128]
        self.dropout_rates = dropout_rates or [0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.l2_reg = l2_reg
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.loss_weights = loss_weights or {'decoder': 0.1, 'ae_action': 0.1, 'action': 1.0}
        self.use_ic_loss = use_ic_loss
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.GPU = GPU
        self.seed = seed

        self.model = None
        self.fitted = False
        self.cnn_output_dim = None  # 将在构建时确定

        self._setup_device()
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

    def _get_regularizer(self):
        """获取 L2 正则化器"""
        if self.l2_reg > 0:
            return regularizers.l2(self.l2_reg)
        return None

    def _build_feature_interaction(self, x, name_prefix='feat_int'):
        """
        特征交互层：建模 OHLCV 特征之间的关系

        Input shape: (batch, time_steps, features_per_step)
        Output shape: (batch, time_steps, features_per_step)
        """
        # 转置：(batch, time, feat) → (batch, feat, time)
        x_t = layers.Permute((2, 1), name=f'{name_prefix}_permute1')(x)

        # 在特征维度做 self-attention
        att = layers.MultiHeadAttention(
            num_heads=1,
            key_dim=min(16, self.time_steps // 4),
            name=f'{name_prefix}_mha'
        )(x_t, x_t)

        # 残差连接
        x_t = layers.Add(name=f'{name_prefix}_add')([x_t, att])
        x_t = layers.LayerNormalization(name=f'{name_prefix}_ln')(x_t)

        # 转置回来：(batch, feat, time) → (batch, time, feat)
        return layers.Permute((2, 1), name=f'{name_prefix}_permute2')(x_t)

    def _build_multiscale_conv(self, x, filters, name_prefix='ms_conv'):
        """
        多尺度卷积：并行使用不同 kernel size 捕获不同时间尺度的模式

        Input shape: (batch, time_steps, channels)
        Output shape: (batch, time_steps, filters * len(kernels))
        """
        conv_outputs = []

        for i, kernel_size in enumerate(self.multiscale_kernels):
            conv = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                kernel_regularizer=self._get_regularizer(),
                name=f'{name_prefix}_k{kernel_size}'
            )(x)
            conv = layers.BatchNormalization(name=f'{name_prefix}_bn_k{kernel_size}')(conv)
            conv = layers.Activation('relu', name=f'{name_prefix}_relu_k{kernel_size}')(conv)
            conv_outputs.append(conv)

        # 拼接所有尺度的输出
        if len(conv_outputs) > 1:
            return layers.Concatenate(name=f'{name_prefix}_concat')(conv_outputs)
        return conv_outputs[0]

    def _build_residual_block(self, x, filters, name_prefix='res'):
        """
        残差块：Conv1D → BN → ReLU → Conv1D → BN → Add → ReLU

        Input shape: (batch, time_steps, channels)
        Output shape: (batch, time_steps, filters)
        """
        shortcut = x

        # 第一个卷积
        x = layers.Conv1D(
            filters, 3, padding='same',
            kernel_regularizer=self._get_regularizer(),
            name=f'{name_prefix}_conv1'
        )(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu1')(x)
        x = layers.Dropout(self.dropout_rates[0], name=f'{name_prefix}_dropout1')(x)

        # 第二个卷积
        x = layers.Conv1D(
            filters, 3, padding='same',
            kernel_regularizer=self._get_regularizer(),
            name=f'{name_prefix}_conv2'
        )(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)

        # 调整 shortcut 维度（如果需要）
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(
                filters, 1, padding='same',
                name=f'{name_prefix}_shortcut'
            )(shortcut)
            shortcut = layers.BatchNormalization(name=f'{name_prefix}_shortcut_bn')(shortcut)

        # 残差连接
        x = layers.Add(name=f'{name_prefix}_add')([x, shortcut])
        x = layers.Activation('relu', name=f'{name_prefix}_relu2')(x)

        return x

    def _build_temporal_attention(self, x, name_prefix='temp_att'):
        """
        时序注意力：Self-attention over time dimension

        Input shape: (batch, time_steps, channels)
        Output shape: (batch, time_steps, channels)
        """
        # Multi-head self-attention
        att = layers.MultiHeadAttention(
            num_heads=self.attention_heads,
            key_dim=x.shape[-1] // self.attention_heads,
            name=f'{name_prefix}_mha'
        )(x, x)

        # 残差连接 + LayerNorm
        x = layers.Add(name=f'{name_prefix}_add')([x, att])
        x = layers.LayerNormalization(name=f'{name_prefix}_ln')(x)

        # FFN
        ffn = layers.Dense(
            x.shape[-1] * 2,
            activation='relu',
            kernel_regularizer=self._get_regularizer(),
            name=f'{name_prefix}_ffn1'
        )(x)
        ffn = layers.Dropout(self.dropout_rates[0], name=f'{name_prefix}_ffn_dropout')(ffn)
        ffn = layers.Dense(
            x.shape[-1],
            kernel_regularizer=self._get_regularizer(),
            name=f'{name_prefix}_ffn2'
        )(ffn)

        # 第二个残差连接
        x = layers.Add(name=f'{name_prefix}_add2')([x, ffn])
        x = layers.LayerNormalization(name=f'{name_prefix}_ln2')(x)

        return x

    def _build_model(self) -> Model:
        """构建 CNN-AE-MLP V2 模型"""

        # ========== Part 1: 输入处理 ==========
        inp = layers.Input(shape=(self.num_columns,), name='input')

        # Reshape: (batch, 300) → (batch, 60, 5)
        x = layers.Reshape(
            (self.time_steps, self.features_per_step),
            name='reshape'
        )(inp)

        # ========== Part 2: 特征交互 (可选) ==========
        if self.use_feature_interaction:
            x = self._build_feature_interaction(x, 'feat_int')

        # ========== Part 3: 多尺度卷积 ==========
        x = self._build_multiscale_conv(x, self.cnn_filters[0], 'ms_conv1')

        # ========== Part 4: 残差块 ==========
        current_filters = self.cnn_filters[0] * len(self.multiscale_kernels)
        for i in range(self.num_residual_blocks):
            target_filters = self.cnn_filters[min(i + 1, len(self.cnn_filters) - 1)]
            x = self._build_residual_block(x, target_filters, f'res{i+1}')
            current_filters = target_filters

        # ========== Part 5: 时序注意力 (可选) ==========
        if self.use_attention:
            x = self._build_temporal_attention(x, 'temp_att')

        # ========== Part 6: Pooling ==========
        # 结合 GlobalAvgPool 和 GlobalMaxPool 获取更丰富的特征
        avg_pool = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        max_pool = layers.GlobalMaxPooling1D(name='global_max_pool')(x)
        cnn_out = layers.Concatenate(name='pool_concat')([avg_pool, max_pool])

        self.cnn_output_dim = cnn_out.shape[-1]

        # ========== Part 7: AE-MLP ==========
        # 输入标准化
        x0 = layers.BatchNormalization(name='ae_input_bn')(cnn_out)

        # Encoder
        encoder = layers.GaussianNoise(self.dropout_rates[0], name='noise')(x0)
        encoder = layers.Dense(
            self.hidden_units[0],
            kernel_regularizer=self._get_regularizer(),
            name='encoder_dense'
        )(encoder)
        encoder = layers.BatchNormalization(name='encoder_bn')(encoder)
        encoder = layers.Activation('swish', name='encoder_act')(encoder)

        # Decoder (重建原始输入，提供正则化)
        decoder = layers.Dropout(self.dropout_rates[1], name='decoder_dropout')(encoder)
        decoder = layers.Dense(
            self.num_columns,  # 重建原始输入维度
            name='decoder'
        )(decoder)

        # 辅助预测分支
        x_ae = layers.Dense(
            self.hidden_units[1],
            kernel_regularizer=self._get_regularizer(),
            name='ae_dense1'
        )(decoder)
        x_ae = layers.BatchNormalization(name='ae_bn1')(x_ae)
        x_ae = layers.Activation('swish', name='ae_act1')(x_ae)
        x_ae = layers.Dropout(self.dropout_rates[2], name='ae_dropout1')(x_ae)
        out_ae = layers.Dense(1, name='ae_action')(x_ae)

        # 主分支: CNN特征 + Encoder特征
        x = layers.Concatenate(name='main_concat')([x0, encoder])
        x = layers.BatchNormalization(name='main_bn0')(x)
        x = layers.Dropout(self.dropout_rates[3], name='main_dropout0')(x)

        # MLP 主体
        for i in range(2, len(self.hidden_units)):
            dropout_idx = min(i + 2, len(self.dropout_rates) - 1)
            x = layers.Dense(
                self.hidden_units[i],
                kernel_regularizer=self._get_regularizer(),
                name=f'main_dense{i-1}'
            )(x)
            x = layers.BatchNormalization(name=f'main_bn{i-1}')(x)
            x = layers.Activation('swish', name=f'main_act{i-1}')(x)
            x = layers.Dropout(self.dropout_rates[dropout_idx], name=f'main_dropout{i-1}')(x)

        # 主输出
        out = layers.Dense(1, name='action')(x)

        # ========== 构建模型 ==========
        model = Model(inputs=inp, outputs=[decoder, out_ae, out], name='CNN_AE_MLP_V2')

        # 编译
        if self.use_ic_loss:
            # 使用自定义 IC 损失
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                loss={
                    'decoder': 'mse',
                    'ae_action': ae_action_loss,
                    'action': action_loss,
                },
                loss_weights=self.loss_weights,
            )
        else:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                loss={
                    'decoder': 'mse',
                    'ae_action': 'mse',
                    'action': 'mse',
                },
                loss_weights=self.loss_weights,
            )

        return model

    def _prepare_data(self, dataset: DatasetH, segment: str):
        """从 Qlib Dataset 准备数据"""
        features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)

        # 处理 NaN 和异常值
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        features = features.clip(-10, 10)

        try:
            labels = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)
            if isinstance(labels, pd.DataFrame):
                labels = labels.iloc[:, 0]
            labels = labels.fillna(0).values
            return features.values, labels, features.index
        except Exception:
            return features.values, None, features.index

    def _auto_adjust_dimensions(self, actual_features: int):
        """自动调整时序维度"""
        expected = self.time_steps * self.features_per_step

        if actual_features == expected:
            return

        print(f"    WARNING: Feature count mismatch!")
        print(f"    Expected: {expected} ({self.time_steps} × {self.features_per_step})")
        print(f"    Actual: {actual_features}")

        # 尝试常见的因子分解
        common_time_steps = [60, 50, 40, 30, 20, 15, 12, 10]
        common_features = [5, 6, 4, 3, 2]

        for ts in common_time_steps:
            for fs in common_features:
                if ts * fs == actual_features:
                    self.time_steps = ts
                    self.features_per_step = fs
                    print(f"    Auto-adjusted: time_steps={ts}, features_per_step={fs}")
                    return

        # 使用最接近的因子
        for ts in range(60, 1, -1):
            if actual_features % ts == 0:
                self.time_steps = ts
                self.features_per_step = actual_features // ts
                print(f"    Auto-adjusted: time_steps={ts}, features_per_step={self.features_per_step}")
                return

        self.num_columns = actual_features

    def fit(self, dataset: DatasetH, verbose: int = 1):
        """训练模型"""
        print("\n" + "=" * 60)
        print("CNN-AE-MLP V2 Training")
        print("=" * 60)

        print("\n[*] Preparing data...")
        X_train, y_train, _ = self._prepare_data(dataset, "train")
        X_valid, y_valid, _ = self._prepare_data(dataset, "valid")

        print(f"    Train: {X_train.shape}, Valid: {X_valid.shape}")

        # 检查并调整维度
        self._auto_adjust_dimensions(X_train.shape[1])
        self.num_columns = X_train.shape[1]

        # 构建模型
        print("\n[*] Building model...")
        print(f"    Config: time_steps={self.time_steps}, features_per_step={self.features_per_step}")
        print(f"    Multi-scale kernels: {self.multiscale_kernels}")
        print(f"    Residual blocks: {self.num_residual_blocks}")
        print(f"    Attention: {self.use_attention}")
        print(f"    Feature interaction: {self.use_feature_interaction}")
        print(f"    IC loss: {self.use_ic_loss}")

        self.model = self._build_model()

        if verbose:
            self.model.summary(print_fn=lambda x: print(f"    {x}"))

        # 回调
        # 监控 val_loss（综合损失）而不是 val_action_loss
        # 因为 val_loss 综合了 decoder/ae_action/action，更稳定
        cb_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stop,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
                mode='min'
            ),
        ]

        # 使用 tf.data.Dataset 流式加载数据，减少显存占用
        # Decoder 目标改为重建原始输入（而不是 CNN 特征），进一步减少内存
        def create_tf_dataset(X, y, batch_size, shuffle=True, repeat=False):
            """创建 tf.data.Dataset，数据保持在 CPU 上直到需要时才传输到 GPU"""
            with tf.device('/CPU:0'):
                # 转换为 float32
                X_tensor = X.astype(np.float32)
                y_tensor = y.astype(np.float32)

                # 创建输出字典
                outputs = {
                    'decoder': X_tensor,  # 重建原始输入
                    'ae_action': y_tensor,
                    'action': y_tensor,
                }

                dataset = tf.data.Dataset.from_tensor_slices((X_tensor, outputs))

                if shuffle:
                    # 使用较小的 buffer 避免内存问题
                    buffer_size = min(len(X), 50000)
                    dataset = dataset.shuffle(buffer_size=buffer_size)

                dataset = dataset.batch(batch_size)

                if repeat:
                    # 训练集需要 repeat，否则会在 epoch 结束时耗尽
                    dataset = dataset.repeat()

                dataset = dataset.prefetch(tf.data.AUTOTUNE)

            return dataset

        print("\n[*] Creating tf.data.Dataset...")
        train_dataset = create_tf_dataset(X_train, y_train, self.batch_size, shuffle=True, repeat=True)
        valid_dataset = create_tf_dataset(X_valid, y_valid, self.batch_size, shuffle=False, repeat=True)

        # 计算 steps
        steps_per_epoch = len(X_train) // self.batch_size
        validation_steps = len(X_valid) // self.batch_size

        # 释放原始 numpy 数组以节省内存
        del X_train, X_valid, y_train, y_valid
        import gc
        gc.collect()

        # 训练
        print("\n[*] Training...")
        history = self.model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=self.n_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=cb_list,
            verbose=verbose,
        )

        self.fitted = True
        print("\n    Training completed")

        return history

    def predict(self, dataset: DatasetH, segment: str = "test") -> pd.Series:
        """预测"""
        if not self.fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X, _, index = self._prepare_data(dataset, segment)

        # 预测
        _, _, pred = self.model.predict(X, batch_size=self.batch_size, verbose=0)

        return pd.Series(pred.flatten(), index=index, name='score')

    def save(self, path: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("No model to save")

        self.model.save(path)

        # 保存配置
        config_path = path.replace('.keras', '_config.npz')
        np.savez(
            config_path,
            time_steps=self.time_steps,
            features_per_step=self.features_per_step,
            cnn_output_dim=self.cnn_output_dim,
            use_attention=self.use_attention,
            use_feature_interaction=self.use_feature_interaction,
        )
        print(f"    Model saved to: {path}")

    @classmethod
    def load(cls, path: str, **kwargs) -> 'CNNAEMLPV2':
        """加载模型"""
        instance = cls(num_columns=1, **kwargs)

        config_path = path.replace('.keras', '_config.npz')
        if os.path.exists(config_path):
            config = np.load(config_path)
            instance.time_steps = int(config['time_steps'])
            instance.features_per_step = int(config['features_per_step'])
            instance.cnn_output_dim = int(config['cnn_output_dim'])

        instance.model = keras.models.load_model(path, compile=False)
        instance.num_columns = instance.model.input_shape[1]
        instance.fitted = True
        print(f"    Model loaded from: {path}")
        return instance


# ============================================================================
# 预设配置
# ============================================================================

PRESET_CONFIGS = {
    'alpha300': {
        'num_columns': 300,
        'time_steps': 60,
        'features_per_step': 5,
        'cnn_filters': [64, 128],
        'multiscale_kernels': [3, 5, 10],
        'num_residual_blocks': 2,
        'use_attention': True,
        'attention_heads': 4,
        'use_feature_interaction': True,
        'hidden_units': [128, 128, 512, 256, 128],
        'dropout_rates': [0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1],
    },
    'alpha300_lite': {
        # 轻量版：关闭 attention 和 feature interaction，减少 filters
        'num_columns': 300,
        'time_steps': 60,
        'features_per_step': 5,
        'cnn_filters': [32, 64],
        'multiscale_kernels': [3, 5],
        'num_residual_blocks': 1,
        'use_attention': False,
        'use_feature_interaction': False,
        'hidden_units': [64, 64, 256, 128],
        'dropout_rates': [0.05, 0.05, 0.1, 0.1, 0.1],
    },
    'alpha300_minimal': {
        # 超轻量版：最小化所有组件
        'num_columns': 300,
        'time_steps': 60,
        'features_per_step': 5,
        'cnn_filters': [32],
        'multiscale_kernels': [5],
        'num_residual_blocks': 0,
        'use_attention': False,
        'use_feature_interaction': False,
        'hidden_units': [64, 64, 128, 64],
        'dropout_rates': [0.05, 0.05, 0.1, 0.1],
    },
    'alpha300_simple': {
        # 简单版：简化架构但保持学习能力
        # 修正：降低正则化强度，让模型能够学习
        'num_columns': 300,
        'time_steps': 60,
        'features_per_step': 5,
        'cnn_filters': [48, 64],  # 两层 CNN，保持一定表达能力
        'multiscale_kernels': [5],  # 单一 kernel
        'num_residual_blocks': 0,
        'use_attention': False,
        'use_feature_interaction': False,
        'hidden_units': [64, 64, 128, 64],
        'dropout_rates': [0.05, 0.1, 0.1, 0.1, 0.1],  # 适中 dropout
        'l2_reg': 1e-5,  # 标准 L2
        'loss_weights': {'decoder': 0.1, 'ae_action': 0.1, 'action': 1.0},  # 标准权重
    },
    'alpha300_regularized': {
        # 正则化版：在 simple 基础上增加适度正则化
        'num_columns': 300,
        'time_steps': 60,
        'features_per_step': 5,
        'cnn_filters': [48, 64],
        'multiscale_kernels': [3, 7],  # 短期 + 中期模式
        'num_residual_blocks': 1,
        'use_attention': False,
        'use_feature_interaction': False,
        'hidden_units': [64, 64, 128, 64],
        'dropout_rates': [0.05, 0.15, 0.15, 0.15, 0.1],  # 适中 dropout
        'l2_reg': 2e-5,
        'loss_weights': {'decoder': 0.15, 'ae_action': 0.1, 'action': 1.0},
    },
    'alpha360': {
        'num_columns': 360,
        'time_steps': 60,
        'features_per_step': 6,
        'cnn_filters': [64, 128],
        'multiscale_kernels': [3, 5, 10],
        'num_residual_blocks': 2,
        'use_attention': True,
        'attention_heads': 4,
        'use_feature_interaction': True,
        'hidden_units': [128, 128, 512, 256, 128],
        'dropout_rates': [0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1],
    },
}


def create_model(preset: str = 'alpha300', **kwargs) -> CNNAEMLPV2:
    """
    根据预设配置创建模型

    Parameters
    ----------
    preset : str
        预设名称: 'alpha300', 'alpha300_lite', 'alpha360'
    **kwargs
        覆盖预设的参数

    Returns
    -------
    CNNAEMLPV2
    """
    config = PRESET_CONFIGS.get(preset, PRESET_CONFIGS['alpha300']).copy()
    config.update(kwargs)
    return CNNAEMLPV2(**config)
