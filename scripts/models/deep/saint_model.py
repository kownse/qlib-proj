"""
SAINT (Self-Attention and Intersample Attention Transformer) Model for Qlib

SAINT 是一种专为表格数据设计的深度学习模型，源自论文：
"SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training"

核心特点:
1. 行间注意力 (Row Attention): 在同一样本的不同特征之间计算注意力
2. 列间注意力 (Column/Intersample Attention): 在不同样本之间计算注意力，捕捉股票间关系
3. 双重注意力交替应用，更好地学习表格数据的结构

内存优化策略:
1. 混合精度训练 (AMP) - 使用 float16 减少约一半内存
2. 梯度累积 - 用小batch模拟大batch
3. 列注意力采样 - 只对部分特征位置计算样本间注意力
4. 数据不预加载到GPU - 按batch加载

架构图:
    Input (batch, features)
        ↓
    Feature Embedding (batch, features, d_model)
        ↓
    ┌─────────────────────────────────────┐
    │   SAINT Block (重复 N 次)            │
    │   ┌───────────────────────────────┐ │
    │   │ Row Attention (特征间注意力)   │ │
    │   │ (batch, features, d_model)    │ │
    │   └───────────────────────────────┘ │
    │              ↓                      │
    │   ┌───────────────────────────────┐ │
    │   │ Column Attention (样本间注意力) │ │
    │   │ (features, batch, d_model)    │ │
    │   └───────────────────────────────┘ │
    └─────────────────────────────────────┘
        ↓
    MLP Head → Output (batch, 1)

在股票预测中的优势:
- 行注意力：学习特征之间的复杂关系（如 RSI 与 MACD 的组合效应）
- 列注意力：捕捉股票之间的相关性（如行业联动、市场情绪传染）
"""

import os
import copy
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.amp import autocast, GradScaler

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class RowAttention(nn.Module):
    """
    行注意力层 - 在特征维度上计算自注意力

    对于每个样本，计算不同特征之间的注意力关系
    Input shape: (batch, num_features, d_model)
    Output shape: (batch, num_features, d_model)
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, num_features, d_model)
        residual = x
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x


class ColumnAttention(nn.Module):
    """
    列注意力层 (Intersample Attention) - 在样本维度上计算注意力

    对于每个特征位置，计算不同样本（股票）之间的注意力关系
    这是 SAINT 的关键创新：捕捉股票间的关系

    内存优化: 支持只对部分特征位置计算样本间注意力 (col_sample_ratio)

    Input shape: (batch, num_features, d_model)
    Output shape: (batch, num_features, d_model)
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, col_sample_ratio: float = 1.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.col_sample_ratio = col_sample_ratio

    def forward(self, x):
        # x: (batch, num_features, d_model)
        batch_size, num_features, d_model = x.shape

        # 如果采样比例 < 1，只对部分特征位置计算样本间注意力
        if self.col_sample_ratio < 1.0 and self.training:
            num_sampled = max(1, int(num_features * self.col_sample_ratio))
            indices = torch.randperm(num_features, device=x.device)[:num_sampled]
            indices, _ = torch.sort(indices)

            # 只对采样的特征计算注意力
            x_sampled = x[:, indices, :]  # (batch, num_sampled, d_model)

            # 转置: (batch, num_sampled, d_model) -> (num_sampled, batch, d_model)
            x_t = x_sampled.transpose(0, 1)

            residual = x_t
            x_t, _ = self.attention(x_t, x_t, x_t)
            x_t = self.dropout(x_t)
            x_t = self.norm(x_t + residual)

            # 转置回来并放回原位置
            x_out = x.clone()
            x_out[:, indices, :] = x_t.transpose(0, 1)
            return x_out
        else:
            # 完整计算
            # 转置: (batch, num_features, d_model) -> (num_features, batch, d_model)
            x_t = x.transpose(0, 1)

            residual = x_t
            x_t, _ = self.attention(x_t, x_t, x_t)
            x_t = self.dropout(x_t)
            x_t = self.norm(x_t + residual)

            # 转置回来
            x = x_t.transpose(0, 1)
            return x


class SAINTBlock(nn.Module):
    """
    SAINT Block = Row Attention + Column Attention + FFN
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        col_sample_ratio: float = 1.0,
    ):
        super().__init__()

        # 行注意力
        self.row_attention = RowAttention(d_model, nhead, dropout)

        # 列注意力 (Intersample Attention)
        self.col_attention = ColumnAttention(d_model, nhead, dropout, col_sample_ratio)

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Row Attention: 特征间注意力
        x = self.row_attention(x)

        # Column Attention: 样本间注意力
        x = self.col_attention(x)

        # FFN
        residual = x
        x = self.ffn(x)
        x = self.norm(x + residual)

        return x


class SAINTModel(nn.Module):
    """
    SAINT 模型

    Parameters
    ----------
    num_features : int
        输入特征数量
    d_model : int
        模型隐藏维度
    nhead : int
        注意力头数
    num_layers : int
        SAINT Block 数量
    dim_feedforward : int
        FFN 中间维度
    dropout : float
        Dropout 比例
    col_sample_ratio : float
        列注意力采样比例 (0-1)，用于减少内存
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        col_sample_ratio: float = 1.0,
    ):
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model

        # 特征嵌入：将每个标量特征映射到 d_model 维向量
        self.feature_embedding = nn.Linear(1, d_model)

        # 位置编码（可选，帮助区分不同特征）
        self.pos_encoding = nn.Parameter(torch.randn(1, num_features, d_model) * 0.02)

        # SAINT Blocks
        self.blocks = nn.ModuleList([
            SAINTBlock(d_model, nhead, dim_feedforward, dropout, col_sample_ratio)
            for _ in range(num_layers)
        ])

        # 输出头 - 使用全局平均池化减少参数量
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape: (batch, num_features)

        Returns
        -------
        torch.Tensor
            Shape: (batch,)
        """
        batch_size = x.shape[0]

        # 将每个特征从标量扩展为向量: (batch, features) -> (batch, features, 1) -> (batch, features, d_model)
        x = x.unsqueeze(-1)  # (batch, features, 1)
        x = self.feature_embedding(x)  # (batch, features, d_model)

        # 添加位置编码
        x = x + self.pos_encoding

        # 通过 SAINT Blocks
        for block in self.blocks:
            x = block(x)

        # 全局平均池化: (batch, features, d_model) -> (batch, d_model)
        x = x.transpose(1, 2)  # (batch, d_model, features)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)

        # 输出
        x = self.output_head(x)  # (batch, 1)

        return x.squeeze(-1)


class SAINT:
    """
    SAINT 模型，适配 Qlib 接口

    Parameters
    ----------
    num_features : int
        输入特征数量
    d_model : int
        模型隐藏维度 (默认32，省内存)
    nhead : int
        注意力头数
    num_layers : int
        SAINT Block 数量 (默认2，省内存)
    dim_feedforward : int
        FFN 中间维度 (默认128，省内存)
    dropout : float
        Dropout 比例
    col_sample_ratio : float
        列注意力采样比例 (0-1)，训练时只对部分特征计算样本间注意力
    lr : float
        学习率
    n_epochs : int
        训练轮数
    batch_size : int
        批次大小 (默认256，省内存)
    grad_accum_steps : int
        梯度累积步数，用于模拟更大的batch
    early_stop : int
        早停耐心值
    reg : float
        L2 正则化权重
    use_amp : bool
        是否使用混合精度训练
    GPU : int
        GPU 设备 ID，-1 表示使用 CPU
    seed : int
        随机种子
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        col_sample_ratio: float = 1.0,
        lr: float = 1e-4,
        n_epochs: int = 100,
        batch_size: int = 512,
        grad_accum_steps: int = 2,
        early_stop: int = 20,
        reg: float = 1e-3,
        use_amp: bool = True,
        GPU: int = 0,
        seed: int = 42,
    ):
        self.num_features = num_features
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.col_sample_ratio = col_sample_ratio
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.early_stop = early_stop
        self.reg = reg
        self.use_amp = use_amp
        self.GPU = GPU
        self.seed = seed

        self.model = None
        self.fitted = False
        self.device = None

        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 设置设备
        self._setup_device()

    def _setup_device(self):
        """配置 GPU/CPU"""
        if self.GPU >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.GPU}')
            print(f"    Using GPU: cuda:{self.GPU}")
        else:
            self.device = torch.device('cpu')
            self.use_amp = False  # CPU 不支持 AMP
            print("    Using CPU")

    def _prepare_data(self, dataset: DatasetH, segment: str):
        """
        从 Qlib Dataset 准备数据
        """
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
            return features.values.astype(np.float32), labels.astype(np.float32)
        except Exception:
            return features.values.astype(np.float32), None

    def fit(self, dataset: DatasetH):
        """训练模型"""
        print("\n    Preparing training data...")
        X_train, y_train = self._prepare_data(dataset, "train")
        X_valid, y_valid = self._prepare_data(dataset, "valid")

        print(f"    Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")

        # 更新输入维度
        actual_features = X_train.shape[1]
        if actual_features != self.num_features:
            print(f"    Updating num_features: {self.num_features} -> {actual_features}")
            self.num_features = actual_features

        # 构建模型
        print("\n    Building SAINT model...")
        self.model = SAINTModel(
            num_features=self.num_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            col_sample_ratio=self.col_sample_ratio,
        ).to(self.device)

        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"    Total parameters: {total_params:,}")
        print(f"    Memory optimization: AMP={self.use_amp}, col_sample_ratio={self.col_sample_ratio}")
        print(f"    Effective batch size: {self.batch_size * self.grad_accum_steps}")

        # 创建 DataLoader (数据保持在 CPU，按需加载到 GPU)
        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True
        )

        valid_dataset = TensorDataset(
            torch.from_numpy(X_valid),
            torch.from_numpy(y_valid)
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True
        )

        # 优化器和损失函数
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.reg
        )

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        criterion = nn.MSELoss()

        # 混合精度训练
        scaler = GradScaler('cuda') if self.use_amp else None

        # 训练循环
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        print("\n    Training...")
        for epoch in range(self.n_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            num_batches = 0
            optimizer.zero_grad()

            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                # 混合精度前向传播
                if self.use_amp:
                    with autocast('cuda'):
                        pred = self.model(batch_x)
                        loss = criterion(pred, batch_y) / self.grad_accum_steps
                    scaler.scale(loss).backward()
                else:
                    pred = self.model(batch_x)
                    loss = criterion(pred, batch_y) / self.grad_accum_steps
                    loss.backward()

                train_loss += loss.item() * self.grad_accum_steps

                # 梯度累积
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                    optimizer.zero_grad()

                num_batches += 1

            train_loss /= max(num_batches, 1)

            # 验证阶段
            self.model.eval()
            val_loss = 0
            num_val_batches = 0

            with torch.no_grad():
                for batch_x, batch_y in valid_loader:
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)

                    if self.use_amp:
                        with autocast('cuda'):
                            pred = self.model(batch_x)
                            loss = criterion(pred, batch_y)
                    else:
                        pred = self.model(batch_x)
                        loss = criterion(pred, batch_y)

                    val_loss += loss.item()
                    num_val_batches += 1

            val_loss /= max(num_val_batches, 1)

            # 学习率调度
            scheduler.step(val_loss)

            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"    Epoch {epoch+1}/{self.n_epochs}: "
                      f"Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {current_lr:.2e}")

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop:
                    print(f"\n    Early stopping at epoch {epoch+1}")
                    break

        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"    Restored best model with val_loss = {best_val_loss:.6f}")

        self.fitted = True
        print("\n    Training completed")

        # 清理 GPU 内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: str = "test") -> pd.Series:
        """预测"""
        if not self.fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # 获取原始数据以保留 index
        features_df = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
        index = features_df.index

        # 准备数据
        X, _ = self._prepare_data(dataset, segment)

        # 预测
        self.model.eval()
        preds = []

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X[i:i+self.batch_size]
                batch_tensor = torch.from_numpy(batch).to(self.device)

                if self.use_amp:
                    with autocast('cuda'):
                        batch_pred = self.model(batch_tensor)
                else:
                    batch_pred = self.model(batch_tensor)

                preds.append(batch_pred.cpu().numpy())

        pred_values = np.concatenate(preds)
        pred_series = pd.Series(pred_values.flatten(), index=index, name='score')

        return pred_series

    def save(self, path: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("No model to save")

        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': {
                'num_features': self.num_features,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'dim_feedforward': self.dim_feedforward,
                'dropout': self.dropout,
                'col_sample_ratio': self.col_sample_ratio,
            }
        }
        torch.save(save_dict, path)
        print(f"    Model saved to: {path}")

    @classmethod
    def load(cls, path: str, GPU: int = 0, **kwargs) -> 'SAINT':
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint['config']

        instance = cls(
            num_features=config['num_features'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            col_sample_ratio=config.get('col_sample_ratio', 0.5),
            GPU=GPU,
            **kwargs
        )

        instance.model = SAINTModel(
            num_features=config['num_features'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            col_sample_ratio=config.get('col_sample_ratio', 0.5),
        ).to(instance.device)

        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.fitted = True
        print(f"    Model loaded from: {path}")

        return instance


def create_saint_for_handler(handler_type: str, **kwargs) -> SAINT:
    """
    根据 handler 类型创建 SAINT 模型

    Parameters
    ----------
    handler_type : str
        Handler 类型
    **kwargs
        其他参数传递给 SAINT

    Returns
    -------
    SAINT
    """
    HANDLER_FEATURES = {
        'alpha158': 158,
        'alpha360': 360,
        'alpha158_vol': 158,
        'alpha158_vol_talib': 180,
        'alpha158_talib_lite': 60,
        'alpha158_news': 170,
    }

    num_features = HANDLER_FEATURES.get(handler_type, 158)

    # 保守配置，适合 12GB GPU
    return SAINT(
        num_features=num_features,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        col_sample_ratio=1.0,
        batch_size=512,
        grad_accum_steps=2,
        use_amp=True,
        **kwargs
    )
