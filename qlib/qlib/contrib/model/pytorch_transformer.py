# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from typing import Text, Union
import copy
import math
from ...utils import get_or_create_path
from ...log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP

# qrun examples/benchmarks/Transformer/workflow_config_transformer_Alpha360.yaml ”


class TransformerModel(Model):
    def __init__(
        self,
        d_feat: int = 20,
        d_model: int = 64,
        batch_size: int = 2048,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0,
        n_epochs=100,
        lr=0.0001,
        metric="",
        early_stop=5,
        loss="mse",
        optimizer="adam",
        reg=1e-3,
        n_jobs=10,
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # set hyper-parameters.
        self.d_model = d_model
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.n_jobs = n_jobs
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.logger = get_module_logger("TransformerModel")
        self.logger.info("Naive Transformer:" "\nbatch_size : {}" "\ndevice : {}".format(self.batch_size, self.device))

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.model = Transformer(d_feat, d_model, nhead, num_layers, dropout, self.device)
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred.float() - label.float()) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, x_train, y_train, debug_first_batch=False):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        self.model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        batch_count = 0
        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)

            # Debug logging for first batch
            if debug_first_batch and batch_count == 0:
                self.logger.info("=== DEBUG: First batch diagnostics ===")
                self.logger.info(f"Feature shape: {feature.shape}")
                self.logger.info(f"Label shape: {label.shape}")
                self.logger.info(f"Feature NaN count: {torch.isnan(feature).sum().item()}")
                self.logger.info(f"Feature Inf count: {torch.isinf(feature).sum().item()}")
                self.logger.info(f"Label NaN count: {torch.isnan(label).sum().item()}")
                self.logger.info(f"Feature min/max: {feature.min().item():.4f} / {feature.max().item():.4f}")
                self.logger.info(f"Label min/max: {label.min().item():.4f} / {label.max().item():.4f}")

            pred = self.model(feature, debug=(debug_first_batch and batch_count == 0))

            # Debug logging for first batch
            if debug_first_batch and batch_count == 0:
                self.logger.info(f"Pred shape: {pred.shape}")
                self.logger.info(f"Pred NaN count: {torch.isnan(pred).sum().item()}")
                self.logger.info(f"Pred Inf count: {torch.isinf(pred).sum().item()}")
                if not torch.isnan(pred).any():
                    self.logger.info(f"Pred min/max: {pred.min().item():.4f} / {pred.max().item():.4f}")

            loss = self.loss_fn(pred, label)

            # Debug logging for first batch
            if debug_first_batch and batch_count == 0:
                self.logger.info(f"Loss value: {loss.item()}")
                if torch.isnan(loss):
                    self.logger.warning("!!! Loss is NaN !!!")

            self.train_optimizer.zero_grad()
            loss.backward()

            # Debug: check gradients before clipping
            if debug_first_batch and batch_count == 0:
                total_grad_norm = 0
                nan_grad_count = 0
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        total_grad_norm += grad_norm ** 2
                        if torch.isnan(param.grad).any():
                            nan_grad_count += 1
                            self.logger.warning(f"NaN gradient in {name}")
                total_grad_norm = total_grad_norm ** 0.5
                self.logger.info(f"Total gradient norm (before clip): {total_grad_norm:.4f}")
                if nan_grad_count > 0:
                    self.logger.warning(f"!!! {nan_grad_count} parameters have NaN gradients !!!")
                self.logger.info("=== END DEBUG ===")

            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()
            batch_count += 1

    def test_epoch(self, data_x, data_y):
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            with torch.no_grad():
                pred = self.model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        # Debug: Check raw data statistics
        self.logger.info("=== DEBUG: Raw data diagnostics ===")
        self.logger.info(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        self.logger.info(f"x_valid shape: {x_valid.shape}, y_valid shape: {y_valid.shape}")

        x_train_nan_count = x_train.isna().sum().sum()
        y_train_nan_count = y_train.isna().sum().sum()
        x_valid_nan_count = x_valid.isna().sum().sum()
        y_valid_nan_count = y_valid.isna().sum().sum()

        self.logger.info(f"x_train NaN count: {x_train_nan_count} ({x_train_nan_count / x_train.size * 100:.2f}%)")
        self.logger.info(f"y_train NaN count: {y_train_nan_count}")
        self.logger.info(f"x_valid NaN count: {x_valid_nan_count} ({x_valid_nan_count / x_valid.size * 100:.2f}%)")
        self.logger.info(f"y_valid NaN count: {y_valid_nan_count}")

        # Handle NaN values by filling with 0 (or column mean)
        if x_train_nan_count > 0 or x_valid_nan_count > 0:
            self.logger.warning(f"Filling NaN values in features with 0...")
            x_train = x_train.fillna(0)
            x_valid = x_valid.fillna(0)

        if y_train_nan_count > 0 or y_valid_nan_count > 0:
            self.logger.warning(f"Dropping samples with NaN labels...")
            # For labels, we need to drop NaN samples
            train_valid_mask = ~y_train.isna().any(axis=1)
            x_train = x_train[train_valid_mask]
            y_train = y_train[train_valid_mask]

            valid_valid_mask = ~y_valid.isna().any(axis=1)
            x_valid = x_valid[valid_valid_mask]
            y_valid = y_valid[valid_valid_mask]

            self.logger.info(f"After dropping NaN labels: x_train {x_train.shape}, x_valid {x_valid.shape}")

        # Check for extreme values and normalize if needed
        x_train_max = np.abs(x_train.values).max()
        x_valid_max = np.abs(x_valid.values).max()
        self.logger.info(f"x_train abs max: {x_train_max:.4e}, x_valid abs max: {x_valid_max:.4e}")

        if x_train_max > 1e6:
            self.logger.warning(f"Extreme values detected! Applying robust normalization...")
            # Use robust scaling: clip outliers and normalize
            # Clip to 3 standard deviations
            for col in x_train.columns:
                col_mean = x_train[col].mean()
                col_std = x_train[col].std()
                if col_std > 0:
                    # Clip to [-3*std, 3*std] around mean
                    lower = col_mean - 3 * col_std
                    upper = col_mean + 3 * col_std
                    x_train[col] = x_train[col].clip(lower, upper)
                    x_valid[col] = x_valid[col].clip(lower, upper)
                    # Z-score normalization
                    x_train[col] = (x_train[col] - col_mean) / col_std
                    x_valid[col] = (x_valid[col] - col_mean) / col_std

            # Replace any remaining inf/nan with 0
            x_train = x_train.replace([np.inf, -np.inf], 0).fillna(0)
            x_valid = x_valid.replace([np.inf, -np.inf], 0).fillna(0)

            self.logger.info(f"After normalization - x_train min/max: {x_train.values.min():.4f} / {x_train.values.max():.4f}")
            self.logger.info(f"After normalization - x_train mean/std: {x_train.values.mean():.4f} / {x_train.values.std():.4f}")

        self.logger.info(f"x_train Inf count: {np.isinf(x_train.values).sum()}")
        self.logger.info(f"x_train min/max: {x_train.values.min():.4f} / {x_train.values.max():.4f}")
        self.logger.info(f"y_train min/max: {y_train.values.min():.4f} / {y_train.values.max():.4f}")
        self.logger.info(f"x_train mean/std: {x_train.values.mean():.4f} / {x_train.values.std():.4f}")
        self.logger.info("=== END Raw data diagnostics ===")

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        best_param = copy.deepcopy(self.model.state_dict())  # Initialize to avoid UnboundLocalError
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            # Enable debug logging for first epoch to diagnose NaN issues
            self.train_epoch(x_train, y_train, debug_first_batch=(step == 0))
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:
            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size

            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)

            with torch.no_grad():
                pred = self.model(x_batch).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src, debug=False):
        # src [N, F*T] --> [N, T, F]
        if debug:
            print(f"[DEBUG] Input src shape: {src.shape}, d_feat: {self.d_feat}")
            print(f"[DEBUG] Input NaN: {torch.isnan(src).sum().item()}, Inf: {torch.isinf(src).sum().item()}")

        src = src.reshape(len(src), self.d_feat, -1).permute(0, 2, 1)
        if debug:
            print(f"[DEBUG] After reshape: {src.shape}")

        src = self.feature_layer(src)
        if debug:
            print(f"[DEBUG] After feature_layer NaN: {torch.isnan(src).sum().item()}")

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        if debug:
            print(f"[DEBUG] After pos_encoder NaN: {torch.isnan(src).sum().item()}")

        output = self.transformer_encoder(src, mask)  # [60, 512, 8]
        if debug:
            print(f"[DEBUG] After transformer_encoder NaN: {torch.isnan(output).sum().item()}")

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]
        if debug:
            print(f"[DEBUG] After decoder_layer NaN: {torch.isnan(output).sum().item()}")

        return output.squeeze()
