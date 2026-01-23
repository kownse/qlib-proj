"""
ALSTM Model: Attention-based LSTM for Stock Prediction

基于 Qlib 官方实现，增加了更好的调试和 NaN 处理。

架构:
    Input (batch, seq_len * d_feat)
    -> Reshape (batch, seq_len, d_feat)
    -> Linear + Tanh
    -> GRU/LSTM
    -> Attention
    -> Linear -> Output

使用方法:
    python scripts/models/deep/run_alstm_v2.py --stock-pool sp500 --handler alpha360
"""

import copy
import numpy as np
import pandas as pd
from typing import Optional, Union, Text

import torch
import torch.nn as nn
import torch.optim as optim

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


def calc_ic(pred, label):
    """计算 IC 和 RankIC"""
    df = pd.DataFrame({'pred': pred, 'label': label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric


class ALSTMNet(nn.Module):
    """ALSTM Network with Attention mechanism"""

    def __init__(
        self,
        d_feat: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        rnn_type: str = "GRU",
    ):
        super().__init__()
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_type = rnn_type

        # Input projection
        self.fc_in = nn.Linear(d_feat, hidden_size)
        self.act = nn.Tanh()

        # RNN layer
        rnn_class = getattr(nn, rnn_type.upper())
        self.rnn = rnn_class(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention
        self.att_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.att_dropout = nn.Dropout(dropout)
        self.att_act = nn.Tanh()
        self.att_fc2 = nn.Linear(hidden_size // 2, 1, bias=False)

        # Output
        self.fc_out = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, debug=False):
        """
        Args:
            x: (batch, seq_len * d_feat) flattened time series

        Returns:
            (batch,) predictions
        """
        batch_size = x.shape[0]

        if debug:
            print(f"\n[ALSTM Forward] Input shape: {x.shape}")
            print(f"  Input stats: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
            print(f"  Input NaN: {torch.isnan(x).sum().item()}, Inf: {torch.isinf(x).sum().item()}")

        # Reshape: (batch, seq_len * d_feat) -> (batch, seq_len, d_feat)
        seq_len = x.shape[1] // self.d_feat
        x = x.view(batch_size, self.d_feat, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, d_feat)

        if debug:
            print(f"  After reshape: {x.shape}")

        # Input projection
        x = self.act(self.fc_in(x))  # (batch, seq_len, hidden_size)

        if debug:
            print(f"  After fc_in: mean={x.mean().item():.6f}, std={x.std().item():.6f}")

        # RNN
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden_size)

        if debug:
            print(f"  After RNN: mean={rnn_out.mean().item():.6f}, std={rnn_out.std().item():.6f}")
            print(f"  RNN NaN: {torch.isnan(rnn_out).sum().item()}")

        # Attention
        att_score = self.att_fc1(rnn_out)
        att_score = self.att_dropout(att_score)
        att_score = self.att_act(att_score)
        att_score = self.att_fc2(att_score)  # (batch, seq_len, 1)
        att_weight = torch.softmax(att_score, dim=1)

        if debug:
            print(f"  Attention weights: mean={att_weight.mean().item():.6f}, std={att_weight.std().item():.6f}")

        # Weighted sum
        out_att = torch.sum(rnn_out * att_weight, dim=1)  # (batch, hidden_size)

        # Concatenate with last hidden state
        out = torch.cat([rnn_out[:, -1, :], out_att], dim=1)  # (batch, hidden_size * 2)

        # Output
        out = self.fc_out(out).squeeze(-1)  # (batch,)

        if debug:
            print(f"  Output: mean={out.mean().item():.6f}, std={out.std().item():.6f}")
            print(f"  Output NaN: {torch.isnan(out).sum().item()}")

        return out


class ALSTMModel:
    """ALSTM Model wrapper compatible with our training framework"""

    def __init__(
        self,
        d_feat: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        rnn_type: str = "GRU",
        n_epochs: int = 200,
        lr: float = 0.001,
        batch_size: int = 2000,
        early_stop: int = 20,
        GPU: int = 0,
        seed: int = None,
    ):
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.seed = seed

        # Device
        if GPU >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{GPU}')
            print(f"Using GPU: cuda:{GPU}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        # Seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Model
        self.model = ALSTMNet(
            d_feat=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
        )
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.fitted = False

        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")

    def loss_fn(self, pred, label):
        """MSE loss, ignoring NaN"""
        mask = ~torch.isnan(label) & ~torch.isinf(label) & ~torch.isnan(pred) & ~torch.isinf(pred)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        loss = (pred[mask] - label[mask]) ** 2
        return torch.mean(loss)

    def _check_data(self, x, y, name="data"):
        """检查数据是否有 NaN 或 Inf"""
        x_nan = np.isnan(x).sum()
        x_inf = np.isinf(x).sum()
        y_nan = np.isnan(y).sum()
        y_inf = np.isinf(y).sum()

        if x_nan > 0 or x_inf > 0 or y_nan > 0 or y_inf > 0:
            print(f"WARNING [{name}]: x_nan={x_nan}, x_inf={x_inf}, y_nan={y_nan}, y_inf={y_inf}")
            return False
        return True

    def train_epoch(self, x_train, y_train, debug_first=False):
        """Train one epoch"""
        x_values = x_train.values
        y_values = np.squeeze(y_train.values)

        # 检查数据
        self._check_data(x_values, y_values, "train")

        # 替换 NaN 和 Inf
        x_values = np.nan_to_num(x_values, nan=0.0, posinf=0.0, neginf=0.0)
        y_values = np.nan_to_num(y_values, nan=0.0, posinf=0.0, neginf=0.0)

        self.model.train()
        losses = []
        ics = []

        indices = np.arange(len(x_values))
        np.random.shuffle(indices)

        is_first = True
        for i in range(0, len(indices), self.batch_size):
            if len(indices) - i < self.batch_size:
                break

            batch_idx = indices[i:i + self.batch_size]
            feature = torch.from_numpy(x_values[batch_idx]).float().to(self.device)
            label = torch.from_numpy(y_values[batch_idx]).float().to(self.device)

            # Forward
            pred = self.model(feature, debug=(debug_first and is_first))
            loss = self.loss_fn(pred, label)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss detected, skipping batch")
                is_first = False
                continue

            losses.append(loss.item())

            # IC
            with torch.no_grad():
                pred_np = pred.cpu().numpy()
                label_np = label.cpu().numpy()
                mask = ~np.isnan(pred_np) & ~np.isnan(label_np)
                if mask.sum() > 10:
                    ic, _ = calc_ic(pred_np[mask], label_np[mask])
                    if not np.isnan(ic):
                        ics.append(ic)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.optimizer.step()

            is_first = False

        return np.mean(losses) if losses else float('nan'), np.mean(ics) if ics else 0.0

    def valid_epoch(self, x_valid, y_valid):
        """Validate one epoch"""
        x_values = x_valid.values
        y_values = np.squeeze(y_valid.values)

        # 替换 NaN 和 Inf
        x_values = np.nan_to_num(x_values, nan=0.0, posinf=0.0, neginf=0.0)
        y_values = np.nan_to_num(y_values, nan=0.0, posinf=0.0, neginf=0.0)

        self.model.eval()
        losses = []
        ics = []

        indices = np.arange(len(x_values))

        with torch.no_grad():
            for i in range(0, len(indices), self.batch_size):
                if len(indices) - i < self.batch_size:
                    break

                batch_idx = indices[i:i + self.batch_size]
                feature = torch.from_numpy(x_values[batch_idx]).float().to(self.device)
                label = torch.from_numpy(y_values[batch_idx]).float().to(self.device)

                pred = self.model(feature)
                loss = self.loss_fn(pred, label)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    losses.append(loss.item())

                # IC
                pred_np = pred.cpu().numpy()
                label_np = label.cpu().numpy()
                mask = ~np.isnan(pred_np) & ~np.isnan(label_np)
                if mask.sum() > 10:
                    ic, _ = calc_ic(pred_np[mask], label_np[mask])
                    if not np.isnan(ic):
                        ics.append(ic)

        return np.mean(losses) if losses else float('nan'), np.mean(ics) if ics else 0.0

    def fit(self, dataset: DatasetH, debug=True):
        """Train the model"""
        # Prepare data
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )

        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        print(f"\nTraining data: {x_train.shape}")
        print(f"Validation data: {x_valid.shape}")

        # 检查数据统计
        print(f"\n[Data Statistics]")
        print(f"  Train X: mean={x_train.values.mean():.6f}, std={x_train.values.std():.6f}")
        print(f"  Train X NaN: {np.isnan(x_train.values).sum()}, Inf: {np.isinf(x_train.values).sum()}")
        print(f"  Train Y: mean={y_train.values.mean():.6f}, std={y_train.values.std():.6f}")
        print(f"  Train Y NaN: {np.isnan(y_train.values).sum()}, Inf: {np.isinf(y_train.values).sum()}")

        print("\n[Training]")
        print("-" * 70)
        print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Train IC':>9} | {'Val Loss':>11} | {'Val IC':>9}")
        print("-" * 70)

        best_val_ic = -float('inf')
        best_model_state = None
        patience = 0

        for epoch in range(1, self.n_epochs + 1):
            debug_first = debug and (epoch == 1)
            train_loss, train_ic = self.train_epoch(x_train, y_train, debug_first=debug_first)
            val_loss, val_ic = self.valid_epoch(x_valid, y_valid)

            print(f"{epoch:>6} | {train_loss:>11.6f} | {train_ic:>9.4f} | {val_loss:>11.6f} | {val_ic:>9.4f}")

            if val_ic > best_val_ic:
                best_val_ic = val_ic
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stop:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        print("-" * 70)

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Best validation IC: {best_val_ic:.4f}")

        self.fitted = True

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test") -> pd.Series:
        """Generate predictions"""
        if not self.fitted:
            raise ValueError("Model not fitted")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        x_values = x_test.values

        # 替换 NaN
        x_values = np.nan_to_num(x_values, nan=0.0, posinf=0.0, neginf=0.0)

        self.model.eval()
        preds = []

        with torch.no_grad():
            for i in range(0, len(x_values), self.batch_size):
                end = min(i + self.batch_size, len(x_values))
                feature = torch.from_numpy(x_values[i:end]).float().to(self.device)
                pred = self.model(feature).cpu().numpy()
                preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)

    def save(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'd_feat': self.d_feat,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'rnn_type': self.rnn_type,
            }
        }, path)
        print(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: str, GPU: int = 0):
        """Load model"""
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']

        instance = cls(
            d_feat=config['d_feat'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            rnn_type=config['rnn_type'],
            GPU=GPU,
        )
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.fitted = True
        print(f"Model loaded from: {path}")
        return instance
