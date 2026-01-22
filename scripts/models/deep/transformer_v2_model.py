"""
Transformer V2 Model for Stock Prediction - 针对60天时序数据优化

改进点:
1. Time2Vec 可学习位置编码 - 自动学习周期性模式（日、周等）
2. 局部时序卷积 - 在全局注意力前捕获局部模式
3. CLS Token 聚合 - 用可学习的聚合token替代平均池化
4. 相对位置偏置 - 让注意力感知相对时间距离
5. 特征交互层 - 显式建模不同特征间的关系
6. 残差门控 - 控制信息流动

架构:
    Input(seq_len, d_feat)
        → FeatureEmbedding(d_model)
        → Time2Vec + LocalConv
        → TransformerEncoder × num_layers (with relative position bias)
        → CLS Token Aggregation
        → GatedMLP → Output(1)
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class Time2Vec(nn.Module):
    """
    Time2Vec: 可学习的时间表示

    将时间位置映射到周期性和非周期性表示，
    比固定的正弦位置编码更灵活，可以学习数据中的周期模式。

    参考: "Time2Vec: Learning a Universal Time Representation" (Kazemi et al., 2019)
    """

    def __init__(self, d_model: int, seq_len: int = 60):
        super().__init__()
        self.d_model = d_model

        # 线性部分 (非周期性趋势)
        self.w_linear = nn.Parameter(torch.randn(1, 1, 1))
        self.b_linear = nn.Parameter(torch.randn(1, 1, 1))

        # 周期性部分
        self.w_periodic = nn.Parameter(torch.randn(1, 1, d_model - 1))
        self.b_periodic = nn.Parameter(torch.randn(1, 1, d_model - 1))

        # 时间位置 (0, 1, 2, ..., seq_len-1)
        positions = torch.arange(seq_len).float().view(1, -1, 1)
        self.register_buffer('positions', positions)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with time encoding added
        """
        batch_size, seq_len, _ = x.shape

        # 扩展positions到batch大小
        positions = self.positions[:, :seq_len, :].expand(batch_size, -1, -1)

        # 线性部分: w * t + b
        linear = self.w_linear * positions + self.b_linear  # (batch, seq_len, 1)

        # 周期性部分: sin(w * t + b)
        periodic = torch.sin(self.w_periodic * positions + self.b_periodic)  # (batch, seq_len, d_model-1)

        # 拼接
        time_encoding = torch.cat([linear, periodic], dim=-1)  # (batch, seq_len, d_model)

        return x + time_encoding


class LocalTemporalConv(nn.Module):
    """
    局部时序卷积层

    在全局注意力之前捕获局部时序模式（如短期趋势、动量）。
    使用深度可分离卷积减少参数量。
    """

    def __init__(self, d_model: int, kernel_sizes: List[int] = [3, 5, 7], dropout: float = 0.1):
        super().__init__()

        n_kernels = len(kernel_sizes)
        # 确保输出维度之和等于 d_model
        base_dim = d_model // n_kernels
        remainder = d_model % n_kernels

        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            # 最后一个卷积吸收余数
            out_dim = base_dim + (remainder if i == n_kernels - 1 else 0)
            self.convs.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2, groups=d_model),
                nn.Conv1d(d_model, out_dim, kernel_size=1),
                nn.GELU(),
            ))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        residual = x

        # (batch, seq_len, d_model) -> (batch, d_model, seq_len)
        x = x.transpose(1, 2)

        # 多尺度卷积
        conv_outputs = [conv(x) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)  # (batch, d_model, seq_len)

        # (batch, d_model, seq_len) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2)

        x = self.dropout(x)

        return self.norm(residual + x)


class RelativePositionBias(nn.Module):
    """
    相对位置偏置

    让注意力感知token之间的相对距离，
    适合时序数据中的时间关系建模。
    """

    def __init__(self, num_heads: int, max_len: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len

        # 相对位置范围: [-max_len+1, max_len-1]
        num_positions = 2 * max_len - 1
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, num_positions))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, seq_len: int):
        """
        Returns:
            (num_heads, seq_len, seq_len) bias matrix
        """
        # 计算相对位置索引
        positions = torch.arange(seq_len, device=self.relative_bias.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq_len, seq_len)
        relative_positions = relative_positions + self.max_len - 1  # 移到非负范围

        # 截断到有效范围
        relative_positions = relative_positions.clamp(0, 2 * self.max_len - 2)

        # 获取偏置
        bias = self.relative_bias[:, relative_positions]  # (num_heads, seq_len, seq_len)

        return bias


class TransformerEncoderLayerWithBias(nn.Module):
    """
    带相对位置偏置的 Transformer Encoder Layer
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        max_len: int = 128,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.relative_bias = RelativePositionBias(nhead, max_len)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)

        # 计算相对位置偏置
        bias = self.relative_bias(seq_len)  # (nhead, seq_len, seq_len)

        # Self-attention with bias
        # 将bias转换为attn_mask格式
        attn_bias = bias.unsqueeze(0)  # (1, nhead, seq_len, seq_len)

        # 由于 PyTorch MultiheadAttention 不直接支持加性偏置，
        # 我们通过修改 attention scores 来实现
        residual = x
        x = self.norm1(x)

        # 手动计算带偏置的注意力
        attn_output, _ = self._attention_with_bias(x, attn_bias)
        x = residual + self.dropout(attn_output)

        # FFN
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x

    def _attention_with_bias(self, x, bias):
        """手动实现带偏置的注意力"""
        batch_size, seq_len, d_model = x.shape
        nhead = bias.size(1)
        head_dim = d_model // nhead

        # 投影 Q, K, V
        qkv = nn.functional.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, nhead, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, nhead, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, nhead, head_dim).transpose(1, 2)

        # 计算注意力分数
        scale = math.sqrt(head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # 添加相对位置偏置
        attn_scores = attn_scores + bias

        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.self_attn.dropout, training=self.training)

        # 应用注意力
        attn_output = torch.matmul(attn_probs, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # 输出投影
        attn_output = self.self_attn.out_proj(attn_output)

        return attn_output, attn_probs


class FeatureInteraction(nn.Module):
    """
    特征交互层

    显式建模不同特征之间的关系（如价格与成交量的关系）。
    """

    def __init__(self, d_feat: int, d_model: int, dropout: float = 0.1):
        super().__init__()

        # 特征级别的注意力
        self.feature_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # 在特征维度上做注意力
        residual = x
        x = self.norm(x)
        x, _ = self.feature_attn(x, x, x)
        return residual + self.dropout(x)


class GatedMLP(nn.Module):
    """
    门控 MLP

    使用门控机制控制信息流，比普通MLP更灵活。
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(d_model, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Split into value and gate
        h = self.fc1(x)
        value, gate = h.chunk(2, dim=-1)

        # Gated activation
        h = value * torch.sigmoid(gate)
        h = self.dropout(h)

        return self.fc2(h)


class TransformerNetV2(nn.Module):
    """Transformer V2 网络结构 - 针对60天时序数据优化"""

    def __init__(
        self,
        d_feat: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        seq_len: int = 60,
        use_local_conv: bool = True,
        use_relative_bias: bool = True,
    ):
        super().__init__()

        self.d_feat = d_feat
        self.d_model = d_model
        self.seq_len = seq_len
        self.use_local_conv = use_local_conv
        self.use_relative_bias = use_relative_bias

        # 特征嵌入
        self.feature_embedding = nn.Sequential(
            nn.Linear(d_feat, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Time2Vec 位置编码
        self.time2vec = Time2Vec(d_model, seq_len + 10)

        # 局部时序卷积 (可选)
        if use_local_conv:
            self.local_conv = LocalTemporalConv(d_model, kernel_sizes=[3, 5, 7], dropout=dropout)

        # CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer Encoder Layers
        if use_relative_bias:
            self.encoder_layers = nn.ModuleList([
                TransformerEncoderLayerWithBias(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    max_len=seq_len + 10,
                )
                for _ in range(num_layers)
            ])
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation='gelu',
            )
            self.encoder_layers = nn.ModuleList([
                encoder_layer for _ in range(num_layers)
            ])
            # 共享encoder
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 特征交互层
        self.feature_interaction = FeatureInteraction(d_feat, d_model, dropout)

        # 输出层
        self.output_norm = nn.LayerNorm(d_model)
        self.output_mlp = GatedMLP(d_model, d_model * 2, dropout)
        self.output_proj = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len * d_feat) 或 (batch, seq_len, d_feat)
        Returns:
            (batch,) predictions
        """
        batch_size = x.size(0)

        # Reshape if needed
        if x.dim() == 2:
            x = x.view(batch_size, self.seq_len, self.d_feat)

        # 特征嵌入
        x = self.feature_embedding(x)  # (batch, seq_len, d_model)

        # Time2Vec 位置编码
        x = self.time2vec(x)

        # 局部时序卷积
        if self.use_local_conv:
            x = self.local_conv(x)

        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1+seq_len, d_model)

        # Transformer Encoder
        if self.use_relative_bias:
            for layer in self.encoder_layers:
                x = layer(x)
        else:
            x = self.encoder(x)

        # 特征交互
        x = self.feature_interaction(x)

        # 取 CLS token 的输出
        cls_output = x[:, 0, :]  # (batch, d_model)

        # 输出层
        out = self.output_norm(cls_output)
        out = out + self.output_mlp(out)  # 残差连接
        out = self.output_proj(out)

        return out.squeeze(-1)


def ic_loss(pred, label):
    """IC Loss: 最大化预测与标签的相关系数"""
    pred = pred - pred.mean()
    label = label - label.mean()

    cov = (pred * label).mean()
    pred_std = pred.std() + 1e-8
    label_std = label.std() + 1e-8

    ic = cov / (pred_std * label_std)
    return -ic


class TransformerModelV2:
    """
    Transformer V2 模型封装类

    改进点:
    - Time2Vec 可学习位置编码
    - 局部时序卷积
    - CLS Token 聚合
    - 相对位置偏置
    - 门控 MLP 输出
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
        use_local_conv: bool = True,
        use_relative_bias: bool = True,
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
        self.use_local_conv = use_local_conv
        self.use_relative_bias = use_relative_bias
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
        print("\n[*] Creating Transformer V2 model...")
        print(f"    Features: Time2Vec, LocalConv={self.use_local_conv}, RelativeBias={self.use_relative_bias}")

        self.model = TransformerNetV2(
            d_feat=self.d_feat,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            seq_len=self.seq_len,
            use_local_conv=self.use_local_conv,
            use_relative_bias=self.use_relative_bias,
        ).to(self.device)

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

        # 优化器
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
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

            # 记录
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_ic'].append(train_ic)
            history['val_ic'].append(val_ic)

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            print(f"{epoch:>6} | {train_loss:>11.6f} | {val_loss:>11.6f} | {train_ic:>9.4f} | {val_ic:>9.4f} | {current_lr:>10.2e}")

            # Early stopping
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
                'use_local_conv': self.use_local_conv,
                'use_relative_bias': self.use_relative_bias,
            }
        }, path)
        print(f"    Model saved to: {path}")

    @classmethod
    def load(cls, path: str, GPU: int = 0) -> 'TransformerModelV2':
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
            use_local_conv=config.get('use_local_conv', True),
            use_relative_bias=config.get('use_relative_bias', True),
            GPU=GPU,
        )

        instance.model = TransformerNetV2(
            d_feat=config['d_feat'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            seq_len=config['seq_len'],
            use_local_conv=config.get('use_local_conv', True),
            use_relative_bias=config.get('use_relative_bias', True),
        ).to(instance.device)

        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.fitted = True

        print(f"    Model loaded from: {path}")
        return instance


# 预设配置
PRESET_CONFIGS_V2 = {
    'alpha300': {
        'd_feat': 5,
        'seq_len': 60,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'use_local_conv': True,
        'use_relative_bias': True,
    },
    'alpha300_large': {
        'd_feat': 5,
        'seq_len': 60,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 3,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'use_local_conv': True,
        'use_relative_bias': True,
    },
    'alpha300_lite': {
        'd_feat': 5,
        'seq_len': 60,
        'd_model': 32,
        'nhead': 2,
        'num_layers': 1,
        'dim_feedforward': 128,
        'dropout': 0.1,
        'use_local_conv': False,
        'use_relative_bias': False,
    },
}


def create_model_v2(preset: str = 'alpha300', **kwargs) -> TransformerModelV2:
    """根据预设创建 V2 模型"""
    config = PRESET_CONFIGS_V2.get(preset, PRESET_CONFIGS_V2['alpha300']).copy()
    config.update(kwargs)
    return TransformerModelV2(**config)
