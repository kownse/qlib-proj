"""
AE-MLP (Autoencoder-enhanced MLP) Model for Qlib - PyTorch Implementation

基于 Kaggle 竞赛中常用的 AE-MLP 架构，适配 Qlib 回归任务。
使用 PyTorch 实现，与 TensorFlow 版本保持相同的网络结构。

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
"""

import os
import copy
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class GaussianNoise(nn.Module):
    """
    高斯噪声层，仅在训练时添加噪声

    与 Keras GaussianNoise 层行为一致
    """

    def __init__(self, std: float = 0.03):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.std > 0:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


class Swish(nn.Module):
    """Swish 激活函数: x * sigmoid(x)"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class AEMLPNetwork(nn.Module):
    """
    AE-MLP 网络模块

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
    """

    def __init__(
        self,
        num_columns: int,
        hidden_units: List[int] = None,
        dropout_rates: List[float] = None,
    ):
        super().__init__()

        self.num_columns = num_columns
        self.hidden_units = hidden_units or [96, 96, 512, 256, 128]
        self.dropout_rates = dropout_rates or [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]

        # 输入层 BatchNorm
        self.input_bn = nn.BatchNorm1d(num_columns)

        # Encoder
        self.noise = GaussianNoise(self.dropout_rates[0])
        self.encoder_dense = nn.Linear(num_columns, self.hidden_units[0])
        self.encoder_bn = nn.BatchNorm1d(self.hidden_units[0])
        self.encoder_act = Swish()

        # Decoder (重建原始输入)
        self.decoder_dropout = nn.Dropout(self.dropout_rates[1])
        self.decoder = nn.Linear(self.hidden_units[0], num_columns)

        # 辅助预测分支 (基于 decoder 输出)
        self.ae_dense1 = nn.Linear(num_columns, self.hidden_units[1])
        self.ae_bn1 = nn.BatchNorm1d(self.hidden_units[1])
        self.ae_act1 = Swish()
        self.ae_dropout1 = nn.Dropout(self.dropout_rates[2])
        self.ae_action = nn.Linear(self.hidden_units[1], 1)

        # 主分支: 原始特征 + encoder 特征
        concat_dim = num_columns + self.hidden_units[0]
        self.main_bn0 = nn.BatchNorm1d(concat_dim)
        self.main_dropout0 = nn.Dropout(self.dropout_rates[3])

        # MLP 主体
        self.main_layers = nn.ModuleList()
        in_features = concat_dim

        for i in range(2, len(self.hidden_units)):
            dropout_idx = min(i + 2, len(self.dropout_rates) - 1)

            layer_block = nn.Sequential(
                nn.Linear(in_features, self.hidden_units[i]),
                nn.BatchNorm1d(self.hidden_units[i]),
                Swish(),
                nn.Dropout(self.dropout_rates[dropout_idx]),
            )
            self.main_layers.append(layer_block)
            in_features = self.hidden_units[i]

        # 主输出
        self.action = nn.Linear(in_features, 1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Returns
        -------
        tuple
            (decoder_output, ae_action_output, main_action_output)
        """
        # 输入标准化
        x0 = self.input_bn(x)

        # Encoder
        encoder = self.noise(x0)
        encoder = self.encoder_dense(encoder)
        encoder = self.encoder_bn(encoder)
        encoder = self.encoder_act(encoder)

        # Decoder (重建)
        decoder = self.decoder_dropout(encoder)
        decoder_out = self.decoder(decoder)

        # 辅助预测分支
        x_ae = self.ae_dense1(decoder_out)
        x_ae = self.ae_bn1(x_ae)
        x_ae = self.ae_act1(x_ae)
        x_ae = self.ae_dropout1(x_ae)
        ae_action_out = self.ae_action(x_ae)

        # 主分支: concat 原始特征和 encoder 输出
        main = torch.cat([x0, encoder], dim=1)
        main = self.main_bn0(main)
        main = self.main_dropout0(main)

        # MLP 主体
        for layer in self.main_layers:
            main = layer(main)

        # 主输出
        action_out = self.action(main)

        return decoder_out, ae_action_out, action_out


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = None
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best:
                self.best_state = copy.deepcopy(model.state_dict())
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False

    def restore(self, model: nn.Module):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)


class AEMLP:
    """
    AE-MLP 模型，适配 Qlib 接口 (PyTorch 实现)

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
    ):
        self.num_columns = num_columns
        self.hidden_units = hidden_units or [96, 96, 512, 256, 128]
        self.dropout_rates = dropout_rates or [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stop_patience = early_stop
        self.loss_weights = loss_weights or {'decoder': 0.1, 'ae_action': 0.1, 'action': 1.0}
        self.GPU = GPU
        self.seed = seed

        self.model: Optional[AEMLPNetwork] = None
        self.fitted = False
        self.device = None

        # 设置随机种子
        self._set_seed()

        # 设置设备
        self._setup_device()

    def _set_seed(self):
        """设置随机种子"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _setup_device(self):
        """配置 GPU/CPU"""
        if self.GPU >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.GPU}')
            print(f"    Using GPU: cuda:{self.GPU}")
        else:
            self.device = torch.device('cpu')
            print("    Using CPU")

    def _build_model(self) -> AEMLPNetwork:
        """构建 AE-MLP 模型"""
        model = AEMLPNetwork(
            num_columns=self.num_columns,
            hidden_units=self.hidden_units,
            dropout_rates=self.dropout_rates,
        )
        return model.to(self.device)

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
            (features, labels) 或 (features, None) 如果没有 labels
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

    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        shuffle: bool = True,
    ) -> DataLoader:
        """创建 DataLoader"""
        X_tensor = torch.FloatTensor(X)

        if y is not None:
            y_tensor = torch.FloatTensor(y).unsqueeze(1)
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False,
        )

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

        # 更新输入维度
        actual_features = X_train.shape[1]
        if actual_features != self.num_columns:
            print(f"    Updating num_columns: {self.num_columns} -> {actual_features}")
            self.num_columns = actual_features

        # 构建模型
        print("\n    Building AE-MLP model (PyTorch)...")
        self.model = self._build_model()

        # 打印模型结构
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"    Total parameters: {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")

        # 创建 DataLoader
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        valid_loader = self._create_dataloader(X_valid, y_valid, shuffle=False)

        # 优化器和调度器
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True,
        )

        # 损失函数
        mse_loss = nn.MSELoss()

        # 早停
        early_stopping = EarlyStopping(
            patience=self.early_stop_patience,
            restore_best=True,
        )

        # 训练循环
        print("\n    Training...")
        for epoch in range(self.n_epochs):
            # 训练阶段
            self.model.train()
            train_losses = {'decoder': 0, 'ae_action': 0, 'action': 0, 'total': 0}
            num_batches = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                # 前向传播
                decoder_out, ae_action_out, action_out = self.model(batch_X)

                # 计算损失
                loss_decoder = mse_loss(decoder_out, batch_X)
                loss_ae_action = mse_loss(ae_action_out, batch_y)
                loss_action = mse_loss(action_out, batch_y)

                # 加权总损失
                total_loss = (
                    self.loss_weights['decoder'] * loss_decoder +
                    self.loss_weights['ae_action'] * loss_ae_action +
                    self.loss_weights['action'] * loss_action
                )

                # 反向传播
                total_loss.backward()
                optimizer.step()

                train_losses['decoder'] += loss_decoder.item()
                train_losses['ae_action'] += loss_ae_action.item()
                train_losses['action'] += loss_action.item()
                train_losses['total'] += total_loss.item()
                num_batches += 1

            # 计算平均训练损失
            for key in train_losses:
                train_losses[key] /= num_batches

            # 验证阶段
            self.model.eval()
            val_losses = {'decoder': 0, 'ae_action': 0, 'action': 0, 'total': 0}
            num_val_batches = 0

            with torch.no_grad():
                for batch_X, batch_y in valid_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    decoder_out, ae_action_out, action_out = self.model(batch_X)

                    loss_decoder = mse_loss(decoder_out, batch_X)
                    loss_ae_action = mse_loss(ae_action_out, batch_y)
                    loss_action = mse_loss(action_out, batch_y)

                    total_loss = (
                        self.loss_weights['decoder'] * loss_decoder +
                        self.loss_weights['ae_action'] * loss_ae_action +
                        self.loss_weights['action'] * loss_action
                    )

                    val_losses['decoder'] += loss_decoder.item()
                    val_losses['ae_action'] += loss_ae_action.item()
                    val_losses['action'] += loss_action.item()
                    val_losses['total'] += total_loss.item()
                    num_val_batches += 1

            # 计算平均验证损失
            for key in val_losses:
                val_losses[key] /= num_val_batches

            # 学习率调度
            scheduler.step(val_losses['action'])

            # 打印进度
            print(f"    Epoch {epoch+1:3d}/{self.n_epochs}: "
                  f"train_loss={train_losses['total']:.4f}, "
                  f"val_loss={val_losses['total']:.4f}, "
                  f"val_action_loss={val_losses['action']:.4f}")

            # 早停检查
            if early_stopping(val_losses['action'], self.model):
                print(f"    Early stopping at epoch {epoch+1}")
                break

        # 恢复最佳模型
        early_stopping.restore(self.model)

        self.fitted = True
        print("\n    ✓ Training completed")

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

        # 转换为 tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # 预测
        self.model.eval()
        with torch.no_grad():
            # 分批预测以避免内存问题
            predictions = []
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i:i+self.batch_size]
                _, _, pred = self.model(batch)
                predictions.append(pred.cpu().numpy())

            pred = np.concatenate(predictions, axis=0)

        # 转换为 Series
        pred_series = pd.Series(pred.flatten(), index=index, name='score')

        return pred_series

    def save(self, path: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("No model to save")

        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'num_columns': self.num_columns,
            'hidden_units': self.hidden_units,
            'dropout_rates': self.dropout_rates,
            'loss_weights': self.loss_weights,
        }
        torch.save(save_dict, path)
        print(f"    ✓ Model saved to: {path}")

    @classmethod
    def load(cls, path: str, GPU: int = 0, **kwargs) -> 'AEMLP':
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu')

        instance = cls(
            num_columns=checkpoint['num_columns'],
            hidden_units=checkpoint['hidden_units'],
            dropout_rates=checkpoint['dropout_rates'],
            loss_weights=checkpoint['loss_weights'],
            GPU=GPU,
            **kwargs
        )

        instance.model = instance._build_model()
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.model.eval()
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


if __name__ == "__main__":
    # 简单测试
    print("Testing AE-MLP PyTorch implementation...")

    # 创建随机数据
    batch_size = 32
    num_features = 158

    X = torch.randn(batch_size, num_features)

    # 创建模型
    model = AEMLPNetwork(
        num_columns=num_features,
        hidden_units=[64, 64, 256, 128, 64],
        dropout_rates=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
    )

    # 前向传播
    model.train()
    decoder_out, ae_action_out, action_out = model(X)

    print(f"Input shape: {X.shape}")
    print(f"Decoder output shape: {decoder_out.shape}")
    print(f"AE Action output shape: {ae_action_out.shape}")
    print(f"Action output shape: {action_out.shape}")

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("\n✓ Test passed!")
