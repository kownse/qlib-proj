"""
TCN (Temporal Convolutional Network) Model

基于 qlib 的 TCN 实现，优化了日志输出并增加了灵活性。

主要改进:
- 添加 verbose 参数控制日志输出级别
- 支持直接传入 numpy 数组训练
- 更简洁的进度显示
- 保留原版所有特性

Usage:
    from models.deep.tcn_model import TCN

    model = TCN(d_feat=6, n_epochs=50, verbose=0)
    model.fit(dataset)
    pred = model.predict(dataset)

    # 或者直接使用 numpy 数组
    model.fit_numpy(X_train, y_train, X_valid, y_valid)
    pred = model.predict_numpy(X_test)
"""

from __future__ import division
from __future__ import print_function

import copy
from pathlib import Path
from typing import Text, Union, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm


# ============================================================================
# TCN Network Components
# ============================================================================

class Chomp1d(nn.Module):
    """Remove trailing padding to maintain causal convolution"""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Single temporal block with residual connection"""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network"""

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    """TCN Model wrapper with linear output layer"""

    def __init__(self, num_input, output_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.num_input = num_input
        self.tcn = TemporalConvNet(num_input, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # Reshape: (batch, features) -> (batch, d_feat, seq_len)
        x = x.reshape(x.shape[0], self.num_input, -1)
        output = self.tcn(x)
        output = self.linear(output[:, :, -1])
        return output.squeeze()


# ============================================================================
# TCN Training Wrapper
# ============================================================================

class TCN:
    """
    TCN Model for time series prediction

    Parameters
    ----------
    d_feat : int
        Number of features per timestep
    n_chans : int
        Number of channels in TCN layers
    kernel_size : int
        Kernel size for convolutions
    num_layers : int
        Number of TCN layers
    dropout : float
        Dropout rate
    n_epochs : int
        Maximum training epochs
    lr : float
        Learning rate
    batch_size : int
        Training batch size
    early_stop : int
        Early stopping patience
    metric : str
        Evaluation metric ('loss' or '')
    loss : str
        Loss function ('mse')
    optimizer : str
        Optimizer ('adam' or 'gd')
    GPU : int
        GPU device ID (-1 for CPU)
    seed : int
        Random seed
    verbose : int
        Verbosity level (0=silent, 1=progress, 2=detailed)
    """

    def __init__(
        self,
        d_feat: int = 6,
        n_chans: int = 128,
        kernel_size: int = 5,
        num_layers: int = 5,
        dropout: float = 0.5,
        n_epochs: int = 200,
        lr: float = 0.0001,
        metric: str = "",
        batch_size: int = 2000,
        early_stop: int = 20,
        loss: str = "mse",
        optimizer: str = "adam",
        GPU: int = 0,
        seed: Optional[int] = None,
        verbose: int = 1,
        **kwargs,
    ):
        self.d_feat = d_feat
        self.n_chans = n_chans
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer_name = optimizer.lower()
        self.loss = loss
        self.seed = seed
        self.verbose = verbose

        # Device setup
        self.device = torch.device(
            f"cuda:{GPU}" if torch.cuda.is_available() and GPU >= 0 else "cpu"
        )

        if self.verbose >= 2:
            print(f"    TCN: d_feat={d_feat}, n_chans={n_chans}, layers={num_layers}, "
                  f"kernel={kernel_size}, dropout={dropout}")
            print(f"    Device: {self.device}")

        # Set random seed
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)

        # Build model
        self.tcn_model = TCNModel(
            num_input=self.d_feat,
            output_size=1,
            num_channels=[self.n_chans] * self.num_layers,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )

        if self.verbose >= 2:
            n_params = sum(p.numel() for p in self.tcn_model.parameters())
            print(f"    Parameters: {n_params / 1e6:.2f}M")

        # Optimizer
        if self.optimizer_name == "adam":
            self.train_optimizer = optim.Adam(self.tcn_model.parameters(), lr=self.lr)
        elif self.optimizer_name == "gd":
            self.train_optimizer = optim.SGD(self.tcn_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(f"optimizer {optimizer} is not supported!")

        self.fitted = False
        self.tcn_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device.type == "cuda"

    def _mse_loss(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def _loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        if self.loss == "mse":
            return self._mse_loss(pred[mask], label[mask])
        raise ValueError(f"unknown loss `{self.loss}`")

    def _metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        if self.metric in ("", "loss"):
            return -self._loss_fn(pred[mask], label[mask])
        raise ValueError(f"unknown metric `{self.metric}`")

    def _train_epoch(self, x_values: np.ndarray, y_values: np.ndarray) -> float:
        """Train one epoch"""
        self.tcn_model.train()
        indices = np.arange(len(x_values))
        np.random.shuffle(indices)

        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(indices), self.batch_size):
            if len(indices) - i < self.batch_size:
                break

            batch_idx = indices[i:i + self.batch_size]
            feature = torch.from_numpy(x_values[batch_idx]).float().to(self.device)
            label = torch.from_numpy(y_values[batch_idx]).float().to(self.device)

            pred = self.tcn_model(feature)
            loss = self._loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.tcn_model.parameters(), 3.0)
            self.train_optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _eval_epoch(self, x_values: np.ndarray, y_values: np.ndarray) -> Tuple[float, float]:
        """Evaluate one epoch"""
        self.tcn_model.eval()
        indices = np.arange(len(x_values))

        losses = []
        scores = []

        with torch.no_grad():
            for i in range(0, len(indices), self.batch_size):
                if len(indices) - i < self.batch_size:
                    break

                batch_idx = indices[i:i + self.batch_size]
                feature = torch.from_numpy(x_values[batch_idx]).float().to(self.device)
                label = torch.from_numpy(y_values[batch_idx]).float().to(self.device)

                pred = self.tcn_model(feature)
                loss = self._loss_fn(pred, label)
                losses.append(loss.item())

                score = self._metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(self, dataset, evals_result: dict = None, save_path: str = None):
        """
        Fit model using Qlib DatasetH

        Parameters
        ----------
        dataset : DatasetH
            Qlib dataset
        evals_result : dict
            Dictionary to store evaluation results
        save_path : str
            Path to save best model
        """
        from qlib.data.dataset.handler import DataHandlerLP

        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )[:2]

        x_train = df_train["feature"].values
        y_train = np.squeeze(df_train["label"].values)
        x_valid = df_valid["feature"].values
        y_valid = np.squeeze(df_valid["label"].values)

        return self.fit_numpy(x_train, y_train, x_valid, y_valid, evals_result, save_path)

    def fit_numpy(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        evals_result: dict = None,
        save_path: str = None,
    ):
        """
        Fit model using numpy arrays

        Parameters
        ----------
        x_train, y_train : np.ndarray
            Training data
        x_valid, y_valid : np.ndarray
            Validation data
        evals_result : dict
            Dictionary to store evaluation results
        save_path : str
            Path to save best model
        """
        if evals_result is None:
            evals_result = {}

        evals_result["train"] = []
        evals_result["valid"] = []

        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        best_param = None

        self.fitted = True

        for epoch in range(self.n_epochs):
            # Train
            train_loss = self._train_epoch(x_train, y_train)

            # Evaluate
            _, train_score = self._eval_epoch(x_train, y_train)
            val_loss, val_score = self._eval_epoch(x_valid, y_valid)

            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if self.verbose >= 1:
                print(f"    Epoch {epoch+1:3d}: train={train_score:.4f}, valid={val_score:.4f}", end="")

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = epoch
                best_param = copy.deepcopy(self.tcn_model.state_dict())
                if self.verbose >= 1:
                    print(" *")
            else:
                stop_steps += 1
                if self.verbose >= 1:
                    print()
                if stop_steps >= self.early_stop:
                    if self.verbose >= 1:
                        print(f"    Early stop at epoch {epoch+1}")
                    break

        if self.verbose >= 1:
            print(f"    Best score: {best_score:.4f} @ epoch {best_epoch+1}")

        # Load best model
        if best_param is not None:
            self.tcn_model.load_state_dict(best_param)

        # Save model
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

        return evals_result

    def predict(self, dataset, segment: Union[Text, slice] = "test") -> pd.Series:
        """
        Predict using Qlib DatasetH

        Parameters
        ----------
        dataset : DatasetH
            Qlib dataset
        segment : str or slice
            Data segment to predict

        Returns
        -------
        pd.Series
            Predictions with index
        """
        from qlib.data.dataset.handler import DataHandlerLP

        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        preds = self.predict_numpy(x_test.values)

        return pd.Series(preds, index=index)

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        """
        Predict using numpy array

        Parameters
        ----------
        x : np.ndarray
            Input features

        Returns
        -------
        np.ndarray
            Predictions
        """
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        self.tcn_model.eval()
        preds = []

        with torch.no_grad():
            for begin in range(0, len(x), self.batch_size):
                end = min(begin + self.batch_size, len(x))
                x_batch = torch.from_numpy(x[begin:end]).float().to(self.device)
                pred = self.tcn_model(x_batch).detach().cpu().numpy()
                preds.append(pred)

        return np.concatenate(preds)

    def save(self, path: str):
        """Save model to file"""
        if self.tcn_model is None:
            raise ValueError("No model to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.tcn_model.state_dict(),
            'd_feat': self.d_feat,
            'n_chans': self.n_chans,
            'kernel_size': self.kernel_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
        }, path)

    @classmethod
    def load(cls, path: str, GPU: int = 0, verbose: int = 0) -> "TCN":
        """Load model from file"""
        checkpoint = torch.load(path, map_location='cpu')

        model = cls(
            d_feat=checkpoint['d_feat'],
            n_chans=checkpoint['n_chans'],
            kernel_size=checkpoint['kernel_size'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout'],
            GPU=GPU,
            verbose=verbose,
        )
        model.tcn_model.load_state_dict(checkpoint['model_state_dict'])
        model.fitted = True

        return model
