"""
MASTER Model: Market-Guided Stock Transformer for Stock Price Forecasting

Official implementation based on: https://github.com/SJTU-DMTai/MASTER
Paper: https://arxiv.org/abs/2312.15235

MASTER 使用市场信息来引导个股预测，通过门控机制融合市场状态和个股特征。

架构:
    输入 (N, T, F) → Gate(market_features) → stock × gate_weight
    → Linear → PositionalEncoding → TAttention → SAttention
    → TemporalAttention → Linear → 输出

特点:
1. 双注意力架构：TAttention (时间维度) + SAttention (股票维度)
2. 市场引导门控：用市场状态的 softmax 权重调制股票特征
3. 按天采样：每个 batch 是同一天的所有股票，支持跨股票注意力

数据格式:
    - 输入: (N, T, F) = (股票数, 时间步=8, 特征数)
    - F = d_feat (股票特征) + market_features (市场特征)
    - 标签: (N,) 最后一个时间步的收益率
"""

import math
import copy
import numpy as np
import pandas as pd
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from torch.utils.data import DataLoader, Sampler

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


# =============================================================================
# Utility Functions
# =============================================================================

def calc_ic(pred, label):
    """计算 IC 和 RankIC"""
    df = pd.DataFrame({'pred': pred, 'label': label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric


def zscore(x):
    """Cross-sectional z-score normalization"""
    return (x - x.mean()).div(x.std() + 1e-8)


def drop_extreme(x):
    """Drop top and bottom 2.5% extreme values"""
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025 * N)
    if percent_2_5 == 0:
        return torch.ones_like(x, dtype=torch.bool), x
    filtered_indices = indices[percent_2_5:-percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]


def drop_na(x):
    """Drop NaN values"""
    mask = ~torch.isnan(x)
    return mask, x[mask]


# =============================================================================
# Network Components (from official MASTER)
# =============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for time series"""

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (N, T, D)
        return x + self.pe[:x.shape[1], :]


class TAttention(nn.Module):
    """
    Temporal Attention (intra-stock)

    Self-attention along the time dimension for each stock.
    """

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        # x: (N, T, D)
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        # FFN with residual
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class SAttention(nn.Module):
    """
    Stock Attention (inter-stock)

    Self-attention across stocks for each time step.
    This is the key component that enables cross-stock modeling.
    """

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        # x: (N, T, D)
        x = self.norm1(x)
        # Transpose to (T, N, D) for cross-stock attention
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            # Attention across stocks (N dimension)
            atten_ave_matrixh = torch.softmax(
                torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1
            )
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        # FFN with residual
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class Gate(nn.Module):
    """
    Feature Gate with softmax

    Uses market information to generate soft feature selection weights.
    """

    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta  # temperature

    def forward(self, gate_input):
        # gate_input: (N, d_gate_input)
        output = self.trans(gate_input)
        output = torch.softmax(output / self.t, dim=-1)
        return self.d_output * output  # scale by d_output


class TemporalAttention(nn.Module):
    """
    Temporal Attention for aggregation

    Uses the last time step as query to aggregate all time steps.
    """

    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        # z: (N, T, D)
        h = self.trans(z)
        query = h[:, -1, :].unsqueeze(-1)  # (N, D, 1)
        lam = torch.matmul(h, query).squeeze(-1)  # (N, T)
        lam = torch.softmax(lam, dim=1).unsqueeze(1)  # (N, 1, T)
        output = torch.matmul(lam, z).squeeze(1)  # (N, D)
        return output


class MASTER(nn.Module):
    """
    MASTER Network

    Market-Guided Stock Transformer for Stock Price Forecasting
    """

    def __init__(
        self,
        d_feat: int = 158,
        d_model: int = 256,
        t_nhead: int = 4,
        s_nhead: int = 2,
        T_dropout_rate: float = 0.5,
        S_dropout_rate: float = 0.5,
        gate_input_start_index: int = 158,
        gate_input_end_index: int = 221,
        beta: float = 5.0,
        use_market_norm: bool = True,  # Whether to use LayerNorm on market features
        use_gate: bool = True,  # Whether to use market-guided gate (set False to test baseline)
    ):
        super(MASTER, self).__init__()

        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = gate_input_end_index - gate_input_start_index
        self.d_feat = d_feat
        self.use_market_norm = use_market_norm
        self.use_gate = use_gate

        # Layer normalization for market features (optional, not in official code)
        # This helps when market features have small magnitude (e.g., US market data)
        if use_gate:
            if use_market_norm:
                self.market_norm = nn.LayerNorm(self.d_gate_input)
            else:
                self.market_norm = None
            # Feature gate using market information
            self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)
        else:
            self.market_norm = None
            self.feature_gate = None

        # Main network
        self.x2y = nn.Linear(d_feat, d_model)
        self.pe = PositionalEncoding(d_model)
        self.tatten = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate)
        self.satten = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
        self.temporalatten = TemporalAttention(d_model=d_model)
        self.decoder = nn.Linear(d_model, 1)

        # Initialize decoder bias to 0 to prevent output shift
        # This is important because labels are centered at 0
        nn.init.zeros_(self.decoder.bias)
        # Use default xavier initialization (gain=1.0) to maintain output variance
        # Input std ≈ 1.1, we want output std ≈ 1.0
        nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, x, debug=False):
        """
        Args:
            x: (N, T, F) where F = d_feat + market_features
            debug: If True, print detailed debug information

        Returns:
            (N,) predictions
        """
        # Split stock features and market features
        src = x[:, :, :self.gate_input_start_index]  # (N, T, d_feat)

        if debug:
            print("\n" + "="*80)
            print("[DEBUG] MASTER Forward Pass")
            print("="*80)
            print(f"  Input x shape: {x.shape}")
            print(f"  Stock features (src) shape: {src.shape}")
            print(f"  Stock features stats: mean={src.mean().item():.6f}, std={src.std().item():.6f}, "
                  f"min={src.min().item():.6f}, max={src.max().item():.6f}")
            print(f"  Stock features NaN count: {torch.isnan(src).sum().item()}")

        # Apply feature gate if enabled
        if self.use_gate:
            gate_input_raw = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]  # (N, d_gate)

            if debug:
                print(f"\n  Gate input (raw) shape: {gate_input_raw.shape}")
                print(f"  Gate input (raw) stats: mean={gate_input_raw.mean().item():.6f}, std={gate_input_raw.std().item():.6f}, "
                      f"min={gate_input_raw.min().item():.6f}, max={gate_input_raw.max().item():.6f}")
                print(f"  Gate input (raw) NaN count: {torch.isnan(gate_input_raw).sum().item()}")

            # Optionally normalize market features for better gate behavior
            if self.market_norm is not None:
                gate_input = self.market_norm(gate_input_raw)
                if debug:
                    print(f"\n  Gate input (after LayerNorm) stats: mean={gate_input.mean().item():.6f}, std={gate_input.std().item():.6f}, "
                          f"min={gate_input.min().item():.6f}, max={gate_input.max().item():.6f}")
            else:
                gate_input = gate_input_raw
                if debug:
                    print(f"\n  Gate input (no normalization, using raw): same as above")

            # Apply feature gate (broadcast to all time steps)
            gate_weight = self.feature_gate(gate_input)  # (N, d_feat)

            if debug:
                print(f"\n  Gate weight shape: {gate_weight.shape}")
                print(f"  Gate weight stats: mean={gate_weight.mean().item():.6f}, std={gate_weight.std().item():.6f}, "
                      f"min={gate_weight.min().item():.6f}, max={gate_weight.max().item():.6f}")
                # Check gate weight distribution (should not be uniform if gate is working)
                gate_entropy = -(gate_weight * torch.log(gate_weight / self.d_feat + 1e-10)).sum(dim=-1).mean()
                print(f"  Gate weight entropy (lower=more selective): {gate_entropy.item():.4f}")
                # Show top-5 and bottom-5 feature weights (averaged across samples)
                avg_weights = gate_weight.mean(dim=0)
                top5_idx = avg_weights.argsort(descending=True)[:5]
                bot5_idx = avg_weights.argsort()[:5]
                top5_str = ", ".join([f"({i.item()}: {avg_weights[i].item():.4f})" for i in top5_idx])
                bot5_str = ", ".join([f"({i.item()}: {avg_weights[i].item():.4f})" for i in bot5_idx])
                print(f"  Top-5 feature weights: [{top5_str}]")
                print(f"  Bottom-5 feature weights: [{bot5_str}]")

            src = src * torch.unsqueeze(gate_weight, dim=1)  # (N, T, d_feat)

            if debug:
                print(f"\n  Gated src stats: mean={src.mean().item():.6f}, std={src.std().item():.6f}, "
                      f"min={src.min().item():.6f}, max={src.max().item():.6f}")
        else:
            if debug:
                print(f"\n  Gate DISABLED - using stock features directly")

        # Main forward pass
        x = self.x2y(src)  # (N, T, d_model)

        if debug:
            print(f"\n  After x2y (Linear) stats: mean={x.mean().item():.6f}, std={x.std().item():.6f}")

        x = self.pe(x)
        x = self.tatten(x)

        if debug:
            print(f"  After TAttention stats: mean={x.mean().item():.6f}, std={x.std().item():.6f}")

        x = self.satten(x)

        if debug:
            print(f"  After SAttention stats: mean={x.mean().item():.6f}, std={x.std().item():.6f}")

        x = self.temporalatten(x)  # (N, d_model)

        if debug:
            print(f"  After TemporalAttention stats: mean={x.mean().item():.6f}, std={x.std().item():.6f}")

        output = self.decoder(x).squeeze(-1)  # (N,)

        if debug:
            print(f"\n  Output shape: {output.shape}")
            print(f"  Output stats: mean={output.mean().item():.6f}, std={output.std().item():.6f}, "
                  f"min={output.min().item():.6f}, max={output.max().item():.6f}")
            print("="*80 + "\n")

        return output


# =============================================================================
# Daily Batch Sampler
# =============================================================================

class DailyBatchSamplerRandom(Sampler):
    """
    Sampler that yields indices for one day at a time.

    This enables cross-stock attention by ensuring all stocks
    in a batch are from the same trading day.
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle

        # Calculate number of samples in each day
        self.daily_count = pd.Series(
            index=self.data_source.get_index()
        ).groupby("datetime").size().values

        # Calculate begin index of each day
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.daily_count)


# =============================================================================
# MASTER Model Wrapper
# =============================================================================

class MASTERModel:
    """
    MASTER Model wrapper compatible with Qlib

    Market-Guided Stock Transformer for Stock Price Forecasting
    """

    def __init__(
        self,
        d_feat: int = 158,
        d_model: int = 256,
        t_nhead: int = 4,
        s_nhead: int = 2,
        gate_input_start_index: int = 158,
        gate_input_end_index: int = 221,
        T_dropout_rate: float = 0.5,
        S_dropout_rate: float = 0.5,
        beta: float = 5.0,
        seq_len: int = 8,
        n_epochs: int = 40,
        lr: float = 8e-6,
        early_stop: int = 10,
        train_stop_loss_thred: float = None,
        GPU: int = 0,
        seed: int = 42,
        save_path: str = 'model/',
        save_prefix: str = '',
        use_market_norm: bool = True,  # Whether to use LayerNorm on market features
        use_gate: bool = True,  # Whether to use market-guided gate
    ):
        self.d_feat = d_feat
        self.d_model = d_model
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.beta = beta
        self.seq_len = seq_len
        self.use_market_norm = use_market_norm
        self.use_gate = use_gate

        self.n_epochs = n_epochs
        self.lr = lr
        self.early_stop = early_stop
        self.train_stop_loss_thred = train_stop_loss_thred
        self.GPU = GPU
        self.seed = seed
        self.save_path = save_path
        self.save_prefix = save_prefix

        self.fitted = False
        self.model = None
        self.device = None

        self._setup_device()
        self._set_seed()

    def _setup_device(self):
        """Setup compute device"""
        if self.GPU >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.GPU}')
            print(f"    Using GPU: cuda:{self.GPU}")
        else:
            self.device = torch.device('cpu')
            print("    Using CPU")

    def _set_seed(self):
        """Set random seed for reproducibility"""
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.deterministic = True

    def _init_model(self):
        """Initialize MASTER network"""
        self.model = MASTER(
            d_feat=self.d_feat,
            d_model=self.d_model,
            t_nhead=self.t_nhead,
            s_nhead=self.s_nhead,
            T_dropout_rate=self.T_dropout_rate,
            S_dropout_rate=self.S_dropout_rate,
            gate_input_start_index=self.gate_input_start_index,
            gate_input_end_index=self.gate_input_end_index,
            beta=self.beta,
            use_market_norm=self.use_market_norm,
            use_gate=self.use_gate,
        )
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)

    def loss_fn(self, pred, label):
        """MSE loss ignoring NaN"""
        mask = ~torch.isnan(label)
        loss = (pred[mask] - label[mask]) ** 2
        return torch.mean(loss)

    def _train_epoch(self, dl_train, debug_first_iter=False):
        """Train one epoch"""
        self.model.train()
        losses = []
        ics = []
        is_first_iter = True

        # MTSDatasetH yields dict with 'data', 'label', etc.
        for batch in dl_train:
            if isinstance(batch, dict):
                # MTSDatasetH format: data is (N, T, F) features only, label is separate
                feature = batch['data'].to(self.device)  # (N, T, F)
                label = batch['label'].to(self.device)  # (N,)
            else:
                # Legacy format: data includes label in last column
                data = torch.squeeze(batch, dim=0)
                feature = data[:, :, 0:-1].to(self.device)
                label = data[:, -1, -1].to(self.device)

            # Debug: Print input data statistics for first iteration
            if debug_first_iter and is_first_iter:
                print("\n" + "#"*80)
                print("[DEBUG] First Training Iteration - Input Data")
                print("#"*80)
                print(f"  Raw feature shape: {feature.shape}")
                print(f"  Raw feature stats: mean={feature.mean().item():.6f}, std={feature.std().item():.6f}, "
                      f"min={feature.min().item():.6f}, max={feature.max().item():.6f}")
                print(f"  Raw feature NaN count: {torch.isnan(feature).sum().item()}")
                print(f"\n  Raw label shape: {label.shape}")
                print(f"  Raw label stats: mean={label.mean().item():.6f}, std={label.std().item():.6f}, "
                      f"min={label.min().item():.6f}, max={label.max().item():.6f}")
                print(f"  Raw label NaN count: {torch.isnan(label).sum().item()}")

                # Check feature ranges by region
                n_stock_feat = self.gate_input_start_index
                n_market_feat = self.gate_input_end_index - self.gate_input_start_index
                stock_feat = feature[:, :, :n_stock_feat]
                market_feat = feature[:, :, n_stock_feat:self.gate_input_end_index]
                print(f"\n  Stock features [{0}:{n_stock_feat}] stats:")
                print(f"    mean={stock_feat.mean().item():.6f}, std={stock_feat.std().item():.6f}")
                print(f"  Market features [{n_stock_feat}:{self.gate_input_end_index}] stats:")
                print(f"    mean={market_feat.mean().item():.6f}, std={market_feat.std().item():.6f}")

            # Drop extreme labels (top/bottom 2.5%) and apply zscore
            # Note: zscore is applied AFTER filtering to ensure proper normalization of filtered subset
            mask, label_clean = drop_extreme(label)
            if mask.sum() < 10:  # Skip if too few samples
                if debug_first_iter and is_first_iter:
                    print(f"  [WARN] Skipping batch: only {mask.sum().item()} samples after drop_extreme")
                continue
            feature = feature[mask, :, :]

            if debug_first_iter and is_first_iter:
                print(f"\n  After drop_extreme: {mask.sum().item()}/{len(mask)} samples kept")
                print(f"  Label before zscore: mean={label_clean.mean().item():.6f}, std={label_clean.std().item():.6f}")

            label_clean = zscore(label_clean)  # Normalize filtered labels

            if debug_first_iter and is_first_iter:
                print(f"  Label after zscore: mean={label_clean.mean().item():.6f}, std={label_clean.std().item():.6f}, "
                      f"min={label_clean.min().item():.6f}, max={label_clean.max().item():.6f}")

            # Forward pass (with debug for first iteration)
            pred = self.model(feature.float(), debug=(debug_first_iter and is_first_iter))

            if debug_first_iter and is_first_iter:
                print(f"\n[DEBUG] Prediction vs Label comparison:")
                print(f"  Pred stats: mean={pred.mean().item():.6f}, std={pred.std().item():.6f}")
                print(f"  Label stats: mean={label_clean.mean().item():.6f}, std={label_clean.std().item():.6f}")
                # Compute correlation
                pred_np = pred.detach().cpu().numpy()
                label_np = label_clean.detach().cpu().numpy()
                corr = np.corrcoef(pred_np, label_np)[0, 1]
                print(f"  Pearson correlation: {corr:.6f}")

            loss = self.loss_fn(pred, label_clean)
            losses.append(loss.item())

            # Compute IC
            with torch.no_grad():
                ic, ric = calc_ic(pred.cpu().numpy(), label_clean.cpu().numpy())
                if not np.isnan(ic):
                    ics.append(ic)

            if debug_first_iter and is_first_iter:
                print(f"\n[DEBUG] Loss and IC:")
                print(f"  Loss: {loss.item():.6f}")
                print(f"  IC: {ic:.6f}, RankIC: {ric:.6f}")

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            if debug_first_iter and is_first_iter:
                # Check gradient statistics
                total_grad_norm = 0.0
                max_grad = 0.0
                min_grad = float('inf')
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_grad_norm += grad_norm ** 2
                        max_grad = max(max_grad, param.grad.abs().max().item())
                        min_grad = min(min_grad, param.grad.abs().min().item())
                total_grad_norm = total_grad_norm ** 0.5
                print(f"\n[DEBUG] Gradient statistics:")
                print(f"  Total gradient norm: {total_grad_norm:.6f}")
                print(f"  Max gradient value: {max_grad:.6f}")
                print(f"  Min gradient value: {min_grad:.6f}")

            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.optimizer.step()

            if debug_first_iter and is_first_iter:
                print("#"*80 + "\n")

            is_first_iter = False

        return float(np.mean(losses)), float(np.mean(ics)) if ics else 0.0

    def _valid_epoch(self, dl_valid):
        """Validate one epoch"""
        self.model.eval()
        losses = []
        ics = []

        with torch.no_grad():
            for batch in dl_valid:
                if isinstance(batch, dict):
                    # MTSDatasetH format
                    feature = batch['data'].to(self.device)
                    label = batch['label'].to(self.device)
                else:
                    data = torch.squeeze(batch, dim=0)
                    feature = data[:, :, 0:-1].to(self.device)
                    label = data[:, -1, -1].to(self.device)

                # Drop NaN labels and apply zscore (match official implementation)
                # Note: zscore is applied AFTER filtering to ensure proper normalization
                mask, label_clean = drop_na(label)
                if mask.sum() < 10:
                    continue
                label_clean = zscore(label_clean)

                # Forward pass (use all features for cross-stock attention)
                pred = self.model(feature.float())
                loss = self.loss_fn(pred[mask], label_clean)
                losses.append(loss.item())

                # Compute IC
                ic, _ = calc_ic(pred[mask].cpu().numpy(), label_clean.cpu().numpy())
                if not np.isnan(ic):
                    ics.append(ic)

        return float(np.mean(losses)), float(np.mean(ics)) if ics else 0.0

    def fit(self, dl_train, dl_valid=None, verbose=True, debug=True):
        """
        Train the model

        Args:
            dl_train: Training data (TSDataSampler or similar)
            dl_valid: Validation data (optional)
            verbose: Print training progress
            debug: Enable debug output for first epoch (default: True)
        """
        self._debug = debug
        print("\n[*] Initializing MASTER model...")
        self._init_model()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"    Total parameters: {total_params:,}")
        print(f"    d_feat: {self.d_feat}, d_model: {self.d_model}")
        print(f"    gate_input: [{self.gate_input_start_index}, {self.gate_input_end_index})")
        print(f"    beta: {self.beta}, seq_len: {self.seq_len}")

        # Enable training mode for MTSDatasetH (if applicable)
        if hasattr(dl_train, 'train'):
            dl_train.train()
        if dl_valid and hasattr(dl_valid, 'eval'):
            dl_valid.eval()

        print("\n[*] Training...")
        print("-" * 80)
        print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Train IC':>9} | {'Val Loss':>11} | {'Val IC':>9}")
        print("-" * 80)

        best_val_ic = -float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(1, self.n_epochs + 1):
            # Enable debug output for first epoch only (if debug mode is on)
            debug_first = self._debug and (epoch == 1)
            train_loss, train_ic = self._train_epoch(dl_train, debug_first_iter=debug_first)

            if dl_valid:
                val_loss, val_ic = self._valid_epoch(dl_valid)
                print(f"{epoch:>6} | {train_loss:>11.6f} | {train_ic:>9.4f} | {val_loss:>11.6f} | {val_ic:>9.4f}")

                # Early stopping based on validation IC
                if val_ic > best_val_ic:
                    best_val_ic = val_ic
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stop:
                        print(f"\nEarly stopping at epoch {epoch}")
                        break
            else:
                print(f"{epoch:>6} | {train_loss:>11.6f} | {train_ic:>9.4f} |      -      |      -")

            # Stop if training loss threshold reached
            if self.train_stop_loss_thred and train_loss <= self.train_stop_loss_thred:
                print(f"\nTraining loss threshold reached at epoch {epoch}")
                best_model_state = copy.deepcopy(self.model.state_dict())
                break

        print("-" * 80)

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Best validation IC: {best_val_ic:.4f}")

        self.fitted = True

    def predict(self, dl_test) -> pd.Series:
        """
        Generate predictions

        Args:
            dl_test: Test data (MTSDatasetH or similar)

        Returns:
            pd.Series with predictions indexed by (datetime, instrument)
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Enable eval mode for MTSDatasetH
        if hasattr(dl_test, 'eval'):
            dl_test.eval()

        preds = []
        indices = []
        ics = []

        self.model.eval()
        with torch.no_grad():
            for batch in dl_test:
                if isinstance(batch, dict):
                    # MTSDatasetH format
                    feature = batch['data'].to(self.device)
                    label = batch['label']
                    batch_indices = batch['index']  # indices into original data
                else:
                    data = torch.squeeze(batch, dim=0)
                    feature = data[:, :, 0:-1].to(self.device)
                    label = data[:, -1, -1]
                    batch_indices = None

                pred = self.model(feature.float()).detach().cpu().numpy()
                preds.append(pred.ravel())

                if batch_indices is not None:
                    indices.extend(batch_indices.tolist())

                # Compute daily IC
                if isinstance(label, torch.Tensor):
                    label_np = label.cpu().numpy()
                else:
                    label_np = label
                daily_ic, _ = calc_ic(pred, label_np)
                if not np.isnan(daily_ic):
                    ics.append(daily_ic)

        # Restore original index using MTSDatasetH's restore_index method
        all_preds = np.concatenate(preds)

        if hasattr(dl_test, 'restore_index') and indices:
            original_index = dl_test.restore_index(indices)
            predictions = pd.Series(all_preds, index=original_index)
        else:
            # Fallback: create simple integer index
            predictions = pd.Series(all_preds)

        # Print metrics
        if ics:
            mean_ic = np.mean(ics)
            std_ic = np.std(ics) if len(ics) > 1 else 1.0
            print(f"    Test IC: {mean_ic:.4f}, ICIR: {mean_ic/std_ic:.4f}")

        return predictions

    def save(self, path: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'd_feat': self.d_feat,
                'd_model': self.d_model,
                't_nhead': self.t_nhead,
                's_nhead': self.s_nhead,
                'gate_input_start_index': self.gate_input_start_index,
                'gate_input_end_index': self.gate_input_end_index,
                'T_dropout_rate': self.T_dropout_rate,
                'S_dropout_rate': self.S_dropout_rate,
                'beta': self.beta,
                'seq_len': self.seq_len,
                'use_market_norm': self.use_market_norm,
                'use_gate': self.use_gate,
            }
        }, path)
        print(f"    Model saved to: {path}")

    @classmethod
    def load(cls, path: str, GPU: int = 0) -> 'MASTERModel':
        """Load model from file"""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint['config']

        instance = cls(
            d_feat=config['d_feat'],
            d_model=config['d_model'],
            t_nhead=config['t_nhead'],
            s_nhead=config['s_nhead'],
            gate_input_start_index=config['gate_input_start_index'],
            gate_input_end_index=config['gate_input_end_index'],
            T_dropout_rate=config['T_dropout_rate'],
            S_dropout_rate=config['S_dropout_rate'],
            beta=config['beta'],
            seq_len=config.get('seq_len', 8),
            use_market_norm=config.get('use_market_norm', True),
            use_gate=config.get('use_gate', True),
            GPU=GPU,
        )

        instance._init_model()
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.fitted = True

        print(f"    Model loaded from: {path}")
        return instance


# =============================================================================
# Preset Configurations
# =============================================================================

PRESET_CONFIGS = {
    'default': {
        'd_feat': 158,
        'd_model': 256,
        't_nhead': 4,
        's_nhead': 2,
        'gate_input_start_index': 158,
        'gate_input_end_index': 221,
        'T_dropout_rate': 0.5,
        'S_dropout_rate': 0.5,
        'beta': 5.0,
        'seq_len': 8,
        'lr': 8e-6,
        'n_epochs': 40,
    },
    'us_market': {
        'd_feat': 142,  # Alpha158 without VMA/VSTD/WVMA
        'd_model': 256,
        't_nhead': 4,
        's_nhead': 2,
        'gate_input_start_index': 142,
        'gate_input_end_index': 205,  # 142 + 63 market features
        'T_dropout_rate': 0.5,
        'S_dropout_rate': 0.5,
        'beta': 5.0,
        'seq_len': 8,
        'lr': 8e-6,
        'n_epochs': 40,
    },
}


def create_master_model(preset: str = 'default', **kwargs) -> MASTERModel:
    """Create MASTER model from preset configuration"""
    config = PRESET_CONFIGS.get(preset, PRESET_CONFIGS['default']).copy()
    config.update(kwargs)
    return MASTERModel(**config)
