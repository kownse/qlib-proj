"""
MASTER Model: Market-Guided Stock Transformer for Stock Price Forecasting

Reference: https://arxiv.org/abs/2312.15235

MASTER 使用市场信息来引导个股预测，通过门控机制融合市场状态和个股特征。

架构:
    Stock Features (158) → Stock Encoder → Stock Embedding
    Market Features (63) → Market Encoder → Market Embedding
    [Stock Embedding, Market Embedding] → Gating Fusion → Output

特点:
1. 双编码器架构：分别处理个股特征和市场信息
2. 市场引导门控：用市场状态调制个股预测
3. 跨股票市场信息共享：同一天所有股票共享相同的市场特征
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class StockEncoder(nn.Module):
    """
    个股特征编码器

    将个股特征编码为固定维度的嵌入向量。
    使用多层 MLP + LayerNorm + Dropout。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, stock_features)
        Returns:
            (batch, output_dim)
        """
        return self.encoder(x)


class MarketEncoder(nn.Module):
    """
    市场信息编码器

    将市场信息特征编码为固定维度的嵌入向量。
    使用多层 MLP + LayerNorm + Dropout。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, market_features)
        Returns:
            (batch, output_dim)
        """
        return self.encoder(x)


class MarketGuidedGating(nn.Module):
    """
    市场引导门控机制

    使用市场信息生成门控信号，调制个股嵌入。
    gate = sigmoid(W_gate @ market_emb + b_gate)
    output = stock_emb * gate + stock_emb
    """

    def __init__(
        self,
        stock_dim: int,
        market_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # 门控网络：从市场嵌入生成门控权重
        self.gate_net = nn.Sequential(
            nn.Linear(market_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, stock_dim),
            nn.Sigmoid(),
        )

        # 可选：交叉注意力风格的融合
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=stock_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(stock_dim)

    def forward(self, stock_emb, market_emb):
        """
        Args:
            stock_emb: (batch, stock_dim)
            market_emb: (batch, market_dim)
        Returns:
            (batch, stock_dim)
        """
        # 门控
        gate = self.gate_net(market_emb)  # (batch, stock_dim)
        gated_stock = stock_emb * gate

        # 残差连接
        output = self.norm(stock_emb + gated_stock)

        return output


class MarketGuidedAttention(nn.Module):
    """
    市场引导注意力机制

    使用市场嵌入作为 Query，个股嵌入作为 Key/Value，
    实现市场对个股的注意力加权。
    """

    def __init__(
        self,
        stock_dim: int,
        market_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 将 market_dim 投影到 stock_dim
        self.market_proj = nn.Linear(market_dim, stock_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=stock_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(stock_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stock_emb, market_emb):
        """
        Args:
            stock_emb: (batch, stock_dim)
            market_emb: (batch, market_dim)
        Returns:
            (batch, stock_dim)
        """
        # 投影市场嵌入
        market_proj = self.market_proj(market_emb)  # (batch, stock_dim)

        # 添加序列维度
        stock_emb_seq = stock_emb.unsqueeze(1)  # (batch, 1, stock_dim)
        market_proj_seq = market_proj.unsqueeze(1)  # (batch, 1, stock_dim)

        # 交叉注意力: 用市场信息查询个股信息
        attn_out, _ = self.cross_attn(
            query=market_proj_seq,
            key=stock_emb_seq,
            value=stock_emb_seq,
        )
        attn_out = attn_out.squeeze(1)  # (batch, stock_dim)

        # 残差连接
        output = self.norm(stock_emb + self.dropout(attn_out))

        return output


class MASTERNet(nn.Module):
    """
    MASTER 网络结构

    Market-Guided Stock Transformer for Stock Price Forecasting
    """

    def __init__(
        self,
        d_feat: int = 221,
        stock_feat_dim: int = 158,
        market_feat_dim: int = 63,
        stock_hidden_dim: int = 256,
        stock_emb_dim: int = 128,
        market_hidden_dim: int = 128,
        market_emb_dim: int = 64,
        fusion_hidden_dim: int = 128,
        num_stock_layers: int = 2,
        num_market_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
    ):
        super().__init__()

        self.d_feat = d_feat
        self.stock_feat_dim = stock_feat_dim
        self.market_feat_dim = market_feat_dim
        self.use_attention = use_attention

        # 个股编码器
        self.stock_encoder = StockEncoder(
            input_dim=stock_feat_dim,
            hidden_dim=stock_hidden_dim,
            output_dim=stock_emb_dim,
            num_layers=num_stock_layers,
            dropout=dropout,
        )

        # 市场编码器
        self.market_encoder = MarketEncoder(
            input_dim=market_feat_dim,
            hidden_dim=market_hidden_dim,
            output_dim=market_emb_dim,
            num_layers=num_market_layers,
            dropout=dropout,
        )

        # 门控融合
        self.gating = MarketGuidedGating(
            stock_dim=stock_emb_dim,
            market_dim=market_emb_dim,
            hidden_dim=fusion_hidden_dim,
        )

        # 可选：注意力融合
        if use_attention:
            self.attention = MarketGuidedAttention(
                stock_dim=stock_emb_dim,
                market_dim=market_emb_dim,
                num_heads=4,
                dropout=dropout,
            )

        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(stock_emb_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

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
            x: (batch, d_feat) - 前 stock_feat_dim 是个股特征，后 market_feat_dim 是市场特征
        Returns:
            (batch,) predictions
        """
        # 分离个股特征和市场特征
        stock_features = x[:, :self.stock_feat_dim]
        market_features = x[:, self.stock_feat_dim:]

        # 编码
        stock_emb = self.stock_encoder(stock_features)  # (batch, stock_emb_dim)
        market_emb = self.market_encoder(market_features)  # (batch, market_emb_dim)

        # 门控融合
        fused_emb = self.gating(stock_emb, market_emb)

        # 可选：注意力融合
        if self.use_attention:
            fused_emb = self.attention(fused_emb, market_emb)

        # 输出
        output = self.output_head(fused_emb)

        return output.squeeze(-1)


def ic_loss(pred, label):
    """IC Loss: 最大化预测与标签的相关系数"""
    pred = pred - pred.mean()
    label = label - label.mean()

    cov = (pred * label).mean()
    pred_std = pred.std() + 1e-8
    label_std = label.std() + 1e-8

    ic = cov / (pred_std * label_std)
    return -ic


class MASTERModel:
    """
    MASTER 模型封装类

    Market-Guided Stock Transformer for Stock Price Forecasting

    参数说明:
    - d_feat: 总特征数 (stock_feat_dim + market_feat_dim)
    - gate_input_start_index: 市场特征开始索引 (即 stock_feat_dim)
    - gate_input_end_index: 市场特征结束索引 (即 d_feat)
    """

    def __init__(
        self,
        d_feat: int = 221,
        gate_input_start_index: int = 158,
        gate_input_end_index: int = 221,
        stock_hidden_dim: int = 256,
        stock_emb_dim: int = 128,
        market_hidden_dim: int = 128,
        market_emb_dim: int = 64,
        fusion_hidden_dim: int = 128,
        num_stock_layers: int = 2,
        num_market_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
        use_ic_loss: bool = False,
        ic_loss_weight: float = 0.5,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 4096,
        early_stop: int = 10,
        GPU: int = 0,
        seed: int = 42,
    ):
        self.d_feat = d_feat
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.stock_feat_dim = gate_input_start_index
        self.market_feat_dim = gate_input_end_index - gate_input_start_index

        self.stock_hidden_dim = stock_hidden_dim
        self.stock_emb_dim = stock_emb_dim
        self.market_hidden_dim = market_hidden_dim
        self.market_emb_dim = market_emb_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        self.num_stock_layers = num_stock_layers
        self.num_market_layers = num_market_layers
        self.dropout = dropout
        self.use_attention = use_attention
        self.use_ic_loss = use_ic_loss
        self.ic_loss_weight = ic_loss_weight

        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stop = early_stop
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

        # 检查特征维度
        num_features = X_train.shape[1]
        if num_features != self.d_feat:
            print(f"    Warning: Adjusting d_feat from {self.d_feat} to {num_features}")
            # 重新计算维度
            self.d_feat = num_features
            # 假设市场特征数量不变
            self.stock_feat_dim = num_features - self.market_feat_dim
            self.gate_input_start_index = self.stock_feat_dim

        print(f"    Stock features: {self.stock_feat_dim}")
        print(f"    Market features: {self.market_feat_dim}")
        print(f"    Total features: {self.d_feat}")

        # 创建模型
        print("\n[*] Creating MASTER model...")
        self.model = MASTERNet(
            d_feat=self.d_feat,
            stock_feat_dim=self.stock_feat_dim,
            market_feat_dim=self.market_feat_dim,
            stock_hidden_dim=self.stock_hidden_dim,
            stock_emb_dim=self.stock_emb_dim,
            market_hidden_dim=self.market_hidden_dim,
            market_emb_dim=self.market_emb_dim,
            fusion_hidden_dim=self.fusion_hidden_dim,
            num_stock_layers=self.num_stock_layers,
            num_market_layers=self.num_market_layers,
            dropout=self.dropout,
            use_attention=self.use_attention,
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

                loss = mse_loss(pred, batch_y)
                if self.use_ic_loss:
                    loss = loss + self.ic_loss_weight * ic_loss(pred, batch_y)

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

                    loss = mse_loss(pred, batch_y)
                    if self.use_ic_loss:
                        loss = loss + self.ic_loss_weight * ic_loss(pred, batch_y)

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

            # Early stopping (基于 val_ic)
            if val_ic > best_val_ic:
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
                'gate_input_start_index': self.gate_input_start_index,
                'gate_input_end_index': self.gate_input_end_index,
                'stock_feat_dim': self.stock_feat_dim,
                'market_feat_dim': self.market_feat_dim,
                'stock_hidden_dim': self.stock_hidden_dim,
                'stock_emb_dim': self.stock_emb_dim,
                'market_hidden_dim': self.market_hidden_dim,
                'market_emb_dim': self.market_emb_dim,
                'fusion_hidden_dim': self.fusion_hidden_dim,
                'num_stock_layers': self.num_stock_layers,
                'num_market_layers': self.num_market_layers,
                'dropout': self.dropout,
                'use_attention': self.use_attention,
            }
        }, path)
        print(f"    Model saved to: {path}")

    @classmethod
    def load(cls, path: str, GPU: int = 0) -> 'MASTERModel':
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint['config']

        instance = cls(
            d_feat=config['d_feat'],
            gate_input_start_index=config['gate_input_start_index'],
            gate_input_end_index=config['gate_input_end_index'],
            stock_hidden_dim=config['stock_hidden_dim'],
            stock_emb_dim=config['stock_emb_dim'],
            market_hidden_dim=config['market_hidden_dim'],
            market_emb_dim=config['market_emb_dim'],
            fusion_hidden_dim=config['fusion_hidden_dim'],
            num_stock_layers=config['num_stock_layers'],
            num_market_layers=config['num_market_layers'],
            dropout=config['dropout'],
            use_attention=config['use_attention'],
            GPU=GPU,
        )

        instance.model = MASTERNet(
            d_feat=config['d_feat'],
            stock_feat_dim=config['stock_feat_dim'],
            market_feat_dim=config['market_feat_dim'],
            stock_hidden_dim=config['stock_hidden_dim'],
            stock_emb_dim=config['stock_emb_dim'],
            market_hidden_dim=config['market_hidden_dim'],
            market_emb_dim=config['market_emb_dim'],
            fusion_hidden_dim=config['fusion_hidden_dim'],
            num_stock_layers=config['num_stock_layers'],
            num_market_layers=config['num_market_layers'],
            dropout=config['dropout'],
            use_attention=config['use_attention'],
        ).to(instance.device)

        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.fitted = True

        print(f"    Model loaded from: {path}")
        return instance


# 预设配置
PRESET_CONFIGS = {
    'alpha158_master': {
        'd_feat': 221,
        'gate_input_start_index': 158,
        'gate_input_end_index': 221,
        'stock_hidden_dim': 256,
        'stock_emb_dim': 128,
        'market_hidden_dim': 128,
        'market_emb_dim': 64,
        'fusion_hidden_dim': 128,
        'num_stock_layers': 2,
        'num_market_layers': 2,
        'dropout': 0.1,
        'use_attention': True,
    },
    'alpha158_master_large': {
        'd_feat': 221,
        'gate_input_start_index': 158,
        'gate_input_end_index': 221,
        'stock_hidden_dim': 512,
        'stock_emb_dim': 256,
        'market_hidden_dim': 256,
        'market_emb_dim': 128,
        'fusion_hidden_dim': 256,
        'num_stock_layers': 3,
        'num_market_layers': 2,
        'dropout': 0.1,
        'use_attention': True,
    },
    'alpha158_master_lite': {
        'd_feat': 221,
        'gate_input_start_index': 158,
        'gate_input_end_index': 221,
        'stock_hidden_dim': 128,
        'stock_emb_dim': 64,
        'market_hidden_dim': 64,
        'market_emb_dim': 32,
        'fusion_hidden_dim': 64,
        'num_stock_layers': 1,
        'num_market_layers': 1,
        'dropout': 0.1,
        'use_attention': False,
    },
}


def create_master_model(preset: str = 'alpha158_master', **kwargs) -> MASTERModel:
    """根据预设创建 MASTER 模型"""
    config = PRESET_CONFIGS.get(preset, PRESET_CONFIGS['alpha158_master']).copy()
    config.update(kwargs)
    return MASTERModel(**config)
