"""
Transformer Model for Stock Prediction (PyTorch Implementation)

使用自注意力机制捕获时序数据中的长程依赖关系。

架构:
    Input(seq_len, d_feat) → Linear(d_model) → PositionalEncoding
        → TransformerEncoder × num_layers
        → GlobalAvgPool → MLP → Output(1)

特点:
- 自注意力机制捕获任意时间点之间的关系
- 位置编码保留时序信息
- 支持 IC loss 直接优化排序能力
- 友好的训练日志
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class PositionalEncoding(nn.Module):
    """位置编码，为 Transformer 提供时序位置信息"""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerNet(nn.Module):
    """Transformer 网络结构"""

    def __init__(
        self,
        d_feat: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        seq_len: int = 60,
    ):
        super().__init__()

        self.d_feat = d_feat
        self.d_model = d_model
        self.seq_len = seq_len

        # 输入投影
        self.input_proj = nn.Linear(d_feat, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (batch, seq, feature)
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出 MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len * d_feat) 或 (batch, seq_len, d_feat)
        Returns:
            (batch, 1)
        """
        batch_size = x.size(0)

        # 如果输入是展平的，reshape 成 (batch, seq_len, d_feat)
        if x.dim() == 2:
            x = x.view(batch_size, self.seq_len, self.d_feat)

        # 输入投影: (batch, seq_len, d_feat) → (batch, seq_len, d_model)
        x = self.input_proj(x)

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Global Average Pooling: (batch, seq_len, d_model) → (batch, d_model)
        x = x.mean(dim=1)

        # 输出
        out = self.output_mlp(x)

        return out.squeeze(-1)


def ic_loss(pred, label):
    """IC Loss: 最大化预测与标签的相关系数"""
    pred = pred - pred.mean()
    label = label - label.mean()

    cov = (pred * label).mean()
    pred_std = pred.std() + 1e-8
    label_std = label.std() + 1e-8

    ic = cov / (pred_std * label_std)
    return -ic  # 最小化负 IC = 最大化 IC


class TransformerModel:
    """
    Transformer 模型封装类

    Parameters
    ----------
    d_feat : int
        每个时间步的特征数
    seq_len : int
        序列长度（时间步数）
    d_model : int
        Transformer 模型维度
    nhead : int
        注意力头数
    num_layers : int
        Transformer 层数
    dim_feedforward : int
        FFN 隐藏层维度
    dropout : float
        Dropout 率
    lr : float
        学习率
    weight_decay : float
        L2 正则化
    n_epochs : int
        训练轮数
    batch_size : int
        批次大小
    early_stop : int
        早停耐心值
    use_ic_loss : bool
        是否使用 IC 损失
    GPU : int
        GPU 设备 ID，-1 表示 CPU
    seed : int
        随机种子
    """

    def __init__(
        self,
        d_feat: int = 5,
        seq_len: int = 60,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 2048,
        early_stop: int = 10,
        use_ic_loss: bool = False,
        GPU: int = 0,
        seed: int = 42,
    ):
        self.d_feat = d_feat
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.use_ic_loss = use_ic_loss
        self.GPU = GPU
        self.seed = seed

        self.model = None
        self.fitted = False
        self.device = None

        self._setup_device()
        self._set_seed()

    def _setup_device(self):
        """配置设备"""
        if self.GPU >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.GPU}')
            print(f"    Using GPU: cuda:{self.GPU}")
        else:
            self.device = torch.device('cpu')
            print("    Using CPU")

    def _set_seed(self):
        """设置随机种子"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

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

    def _compute_ic(self, pred, label):
        """计算 IC"""
        if len(pred) == 0:
            return 0.0
        pred_centered = pred - pred.mean()
        label_centered = label - label.mean()
        cov = (pred_centered * label_centered).mean()
        pred_std = pred.std() + 1e-8
        label_std = label.std() + 1e-8
        return cov / (pred_std * label_std)

    def fit(self, dataset: DatasetH, verbose: bool = True):
        """训练模型"""
        print("\n[*] Preparing data...")
        X_train, y_train, _ = self._prepare_data(dataset, "train")
        X_valid, y_valid, _ = self._prepare_data(dataset, "valid")

        print(f"    Train samples: {len(X_train):,}")
        print(f"    Valid samples: {len(X_valid):,}")

        # 自动推断 seq_len 和 d_feat
        num_features = X_train.shape[1]
        if num_features != self.seq_len * self.d_feat:
            # 尝试自动调整
            if num_features % self.seq_len == 0:
                self.d_feat = num_features // self.seq_len
                print(f"    Auto-adjusted d_feat to {self.d_feat}")
            elif num_features % self.d_feat == 0:
                self.seq_len = num_features // self.d_feat
                print(f"    Auto-adjusted seq_len to {self.seq_len}")
            else:
                raise ValueError(
                    f"Feature count {num_features} != seq_len({self.seq_len}) * d_feat({self.d_feat})"
                )

        print(f"    Input shape: ({self.seq_len}, {self.d_feat})")

        # 创建模型
        print("\n[*] Creating model...")
        self.model = TransformerNet(
            d_feat=self.d_feat,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            seq_len=self.seq_len,
        ).to(self.device)

        # 打印模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"    Total parameters: {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")

        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        valid_dataset = TensorDataset(
            torch.FloatTensor(X_valid),
            torch.FloatTensor(y_valid)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.GPU >= 0 else False
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.GPU >= 0 else False
        )

        # 优化器和损失函数
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        mse_loss = nn.MSELoss()

        # 训练循环
        print("\n[*] Training...")
        print("-" * 90)
        print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>11} | {'Train IC':>9} | {'Val IC':>9} | {'LR':>10}")
        print("-" * 90)

        best_val_loss = float('inf')
        best_val_ic = -float('inf')
        best_model_state = None
        patience_counter = 0

        history = {'train_loss': [], 'val_loss': [], 'train_ic': [], 'val_ic': []}

        for epoch in range(1, self.n_epochs + 1):
            # 训练
            self.model.train()
            train_losses = []
            train_preds = []
            train_labels = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                pred = self.model(batch_x)

                if self.use_ic_loss:
                    loss = mse_loss(pred, batch_y) + 0.5 * ic_loss(pred, batch_y)
                else:
                    loss = mse_loss(pred, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())
                train_preds.extend(pred.detach().cpu().numpy())
                train_labels.extend(batch_y.detach().cpu().numpy())

            train_loss = np.mean(train_losses)
            train_ic = self._compute_ic(np.array(train_preds), np.array(train_labels))

            # 验证
            self.model.eval()
            val_losses = []
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for batch_x, batch_y in valid_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    pred = self.model(batch_x)

                    if self.use_ic_loss:
                        loss = mse_loss(pred, batch_y) + 0.5 * ic_loss(pred, batch_y)
                    else:
                        loss = mse_loss(pred, batch_y)

                    val_losses.append(loss.item())
                    val_preds.extend(pred.cpu().numpy())
                    val_labels.extend(batch_y.cpu().numpy())

            val_loss = np.mean(val_losses)
            val_ic = self._compute_ic(np.array(val_preds), np.array(val_labels))

            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_ic'].append(train_ic)
            history['val_ic'].append(val_ic)

            # 学习率调度
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # 打印日志
            print(f"{epoch:>6} | {train_loss:>11.6f} | {val_loss:>11.6f} | {train_ic:>9.4f} | {val_ic:>9.4f} | {current_lr:>10.2e}")

            # Early stopping (基于 val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_ic = val_ic
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        print("-" * 90)
        print(f"Best validation - Loss: {best_val_loss:.6f}, IC: {best_val_ic:.4f}")

        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)

        self.fitted = True
        return history

    def predict(self, dataset: DatasetH, segment: str = "test") -> pd.Series:
        """预测"""
        if not self.fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X, _, index = self._prepare_data(dataset, segment)

        self.model.eval()
        preds = []

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_x = torch.FloatTensor(X[i:i+self.batch_size]).to(self.device)
                pred = self.model(batch_x)
                preds.extend(pred.cpu().numpy())

        return pd.Series(np.array(preds), index=index, name='score')

    def save(self, path: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("No model to save")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'd_feat': self.d_feat,
                'seq_len': self.seq_len,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'dim_feedforward': self.dim_feedforward,
                'dropout': self.dropout,
            }
        }, path)
        print(f"    Model saved to: {path}")

    @classmethod
    def load(cls, path: str, GPU: int = 0) -> 'TransformerModel':
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']

        instance = cls(
            d_feat=config['d_feat'],
            seq_len=config['seq_len'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            GPU=GPU,
        )

        instance.model = TransformerNet(**config).to(instance.device)
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.fitted = True

        print(f"    Model loaded from: {path}")
        return instance


# 预设配置
PRESET_CONFIGS = {
    'alpha300': {
        'd_feat': 5,
        'seq_len': 60,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 256,
        'dropout': 0.1,
    },
    'alpha300_lite': {
        'd_feat': 5,
        'seq_len': 60,
        'd_model': 32,
        'nhead': 2,
        'num_layers': 1,
        'dim_feedforward': 128,
        'dropout': 0.1,
    },
    'alpha360': {
        'd_feat': 6,
        'seq_len': 60,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 256,
        'dropout': 0.1,
    },
}


def create_model(preset: str = 'alpha300', **kwargs) -> TransformerModel:
    """根据预设创建模型"""
    config = PRESET_CONFIGS.get(preset, PRESET_CONFIGS['alpha300']).copy()
    config.update(kwargs)
    return TransformerModel(**config)
