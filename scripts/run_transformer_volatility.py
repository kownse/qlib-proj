"""
运行 PyTorch Transformer 模型，预测N天股票价格波动率

波动率定义：未来N个交易日波动变化

扩展特征：包含 Alpha158 默认指标 + TA-Lib 技术指标

使用 Qlib 内置的 PyTorch Transformer 实现，兼容 Apple Silicon
"""

from pathlib import Path
import argparse
import copy
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import qlib
from qlib.constant import REG_US
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

# Import TA-Lib custom operators
from talib_ops import TALIB_OPS

# Import extended data handlers
from datahandler_ext import Alpha158_Volatility, Alpha158_Volatility_TALib
from utils import evaluate_model


# ========== 配置 ==========

# 数据路径
PROJECT_ROOT = Path(__file__).parent.parent
QLIB_DATA_PATH = PROJECT_ROOT / "my_data" / "qlib_us"

# 股票池
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]

# 时间划分
TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
VALID_START = "2024-01-01"
VALID_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

# 波动率预测窗口（天数）
VOLATILITY_WINDOW = 2


# ========== 特征选择 ==========
SELECTED_FEATURES = [
    "RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10",
    "CORR5", "CORD5", "CORR10", "ROC60", "RESI10",
    "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5",
    "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"
]

TALIB_SELECTED_FEATURES = [
    "TALIB_RSI14",
    "TALIB_ATR14",
    "TALIB_ADX14",
    "TALIB_MACD",
    "TALIB_BB_WIDTH20",
    "TALIB_MOM10",
    "TALIB_CCI14",
    "TALIB_WILLR14",
    "TALIB_NATR14",
    "TALIB_EMA20",
]


# ========== Simple Transformer Model ==========

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SimpleTransformer(nn.Module):
    """Simple Transformer for feature-based prediction (not time-series)."""

    def __init__(self, d_feat, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        self.d_feat = d_feat
        self.d_model = d_model

        # Project features to model dimension
        self.input_proj = nn.Linear(d_feat, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x: [batch_size, d_feat]
        # Reshape to [batch_size, 1, d_feat] for transformer
        x = x.unsqueeze(1)

        # Project to model dimension
        x = self.input_proj(x)  # [batch_size, 1, d_model]

        # Transformer
        x = self.transformer(x)  # [batch_size, 1, d_model]

        # Output
        x = x.squeeze(1)  # [batch_size, d_model]
        x = self.output_proj(x)  # [batch_size, 1]

        return x.squeeze(-1)


class TransformerVolatilityModel:
    """Transformer Model for Volatility Prediction."""

    def __init__(
        self,
        d_feat,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        lr=0.0001,
        n_epochs=100,
        batch_size=2048,
        early_stop=10,
        device="cpu",
        seed=42,
    ):
        self.d_feat = d_feat
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.device = torch.device(device)
        self.seed = seed

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize model
        self.model = SimpleTransformer(d_feat, d_model, nhead, num_layers, dropout)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.fitted = False

    def fit(self, dataset, save_path=None):
        """Train the model."""
        # Prepare data
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L
        )

        x_train = df_train["feature"].values.astype(np.float32)
        y_train = df_train["label"].values.astype(np.float32).squeeze()
        x_valid = df_valid["feature"].values.astype(np.float32)
        y_valid = df_valid["label"].values.astype(np.float32).squeeze()

        # Remove NaN
        train_mask = ~(np.isnan(x_train).any(axis=1) | np.isnan(y_train))
        valid_mask = ~(np.isnan(x_valid).any(axis=1) | np.isnan(y_valid))
        x_train, y_train = x_train[train_mask], y_train[train_mask]
        x_valid, y_valid = x_valid[valid_mask], y_valid[valid_mask]

        print(f"    Training samples: {len(x_train)}, Validation samples: {len(x_valid)}")

        best_val_loss = float('inf')
        best_epoch = 0
        best_state = None
        patience_counter = 0

        for epoch in range(self.n_epochs):
            # Training
            self.model.train()
            indices = np.random.permutation(len(x_train))
            train_losses = []

            for i in range(0, len(indices), self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                if len(batch_idx) < 8:  # Skip very small batches
                    continue

                x_batch = torch.from_numpy(x_train[batch_idx]).to(self.device)
                y_batch = torch.from_numpy(y_train[batch_idx]).to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(x_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                x_val_tensor = torch.from_numpy(x_valid).to(self.device)
                y_val_tensor = torch.from_numpy(y_valid).to(self.device)
                val_pred = self.model(x_val_tensor)
                val_loss = self.criterion(val_pred, y_val_tensor).item()

            train_loss = np.mean(train_losses)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{self.n_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        print(f"    Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")

        # Save model
        if save_path:
            torch.save(self.model.state_dict(), save_path)
            print(f"    Model saved to: {save_path}")

        self.fitted = True

    def predict(self, dataset, segment="test"):
        """Generate predictions."""
        if not self.fitted:
            raise ValueError("Model not fitted yet!")

        df_test = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        x_test = df_test["feature"].values.astype(np.float32)
        index = df_test["feature"].index

        # Fill NaN with 0 for prediction
        x_test = np.nan_to_num(x_test, nan=0.0)

        self.model.eval()
        preds = []

        with torch.no_grad():
            for i in range(0, len(x_test), self.batch_size):
                x_batch = torch.from_numpy(x_test[i:i + self.batch_size]).to(self.device)
                pred = self.model(x_batch).cpu().numpy()
                preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


def main():
    parser = argparse.ArgumentParser(description='Transformer Stock Price Volatility Prediction')
    parser.add_argument('--nday', type=int, default=2,
                        help='Volatility prediction window in days (default: 2)')
    parser.add_argument('--use-talib', action='store_true',
                        help='Use extended TA-Lib features (default: False)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Batch size (default: 2048)')
    parser.add_argument('--d-model', type=int, default=64,
                        help='Transformer model dimension (default: 64)')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads (default: 4)')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of transformer layers (default: 2)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--early-stop', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    args = parser.parse_args()

    global VOLATILITY_WINDOW
    VOLATILITY_WINDOW = args.nday

    print("=" * 70)
    print(f"PyTorch Transformer {VOLATILITY_WINDOW}-Day Stock Price Volatility Prediction")
    if args.use_talib:
        print("Features: Alpha158 + TA-Lib Technical Indicators")
    else:
        print("Features: Alpha158 (default)")
    print("=" * 70)

    # 1. Initialize Qlib
    print("\n[1] Initializing Qlib...")
    if args.use_talib:
        qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US, custom_ops=TALIB_OPS)
        print("    Qlib initialized with TA-Lib custom operators")
    else:
        qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US)
        print("    Qlib initialized")

    # 2. Check data availability
    print("\n[2] Checking data availability...")
    instruments = D.instruments(market="all")
    available_instruments = list(D.list_instruments(instruments))
    print(f"    Available instruments: {len(available_instruments)}")

    test_df = D.features(
        instruments=["AAPL"],
        fields=["$close", "$volume"],
        start_time=TEST_START,
        end_time=TEST_END
    )
    print(f"    AAPL sample data shape: {test_df.shape}")
    print(f"    Date range: {test_df.index.get_level_values('datetime').min().date()} to "
          f"{test_df.index.get_level_values('datetime').max().date()}")

    # 3. Select features
    if args.use_talib:
        feature_cols = SELECTED_FEATURES + TALIB_SELECTED_FEATURES
    else:
        feature_cols = SELECTED_FEATURES

    d_feat = len(feature_cols)
    print(f"\n[3] Feature configuration:")
    print(f"    Number of features (d_feat): {d_feat}")

    # 4. Create DataHandler with feature filtering
    print(f"\n[4] Creating DataHandler with {VOLATILITY_WINDOW}-day volatility label...")

    infer_processors = [
        {"class": "FilterCol", "kwargs": {"fields_group": "feature", "col_list": feature_cols}},
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ]

    learn_processors = [
        {"class": "DropnaLabel"},
    ]

    if args.use_talib:
        print(f"    Features: Alpha158 + TA-Lib ({len(feature_cols)} selected features)")
        handler = Alpha158_Volatility_TALib(
            volatility_window=VOLATILITY_WINDOW,
            instruments=TEST_SYMBOLS,
            start_time=TRAIN_START,
            end_time=TEST_END,
            fit_start_time=TRAIN_START,
            fit_end_time=TRAIN_END,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
        )
    else:
        print(f"    Features: Alpha158 ({len(feature_cols)} selected features)")
        handler = Alpha158_Volatility(
            volatility_window=VOLATILITY_WINDOW,
            instruments=TEST_SYMBOLS,
            start_time=TRAIN_START,
            end_time=TEST_END,
            fit_start_time=TRAIN_START,
            fit_end_time=TRAIN_END,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
        )

    print(f"    Label: {VOLATILITY_WINDOW}-day realized volatility")
    print("    DataHandler created")

    # 5. Create Dataset
    print("\n[5] Creating Dataset...")
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (TRAIN_START, TRAIN_END),
            "valid": (VALID_START, VALID_END),
            "test": (TEST_START, TEST_END),
        }
    )

    train_data = dataset.prepare("train", col_set="feature")
    print(f"    Train features shape: {train_data.shape}")

    # 6. Verify feature count matches
    actual_d_feat = train_data.shape[1]
    if actual_d_feat != d_feat:
        print(f"    Warning: Expected {d_feat} features, got {actual_d_feat}")
        d_feat = actual_d_feat

    # 7. Analyze label distribution
    print("\n[6] Analyzing label distribution...")
    train_label = dataset.prepare("train", col_set="label")
    valid_label = dataset.prepare("valid", col_set="label")

    print(f"    Train set volatility statistics:")
    print(f"      Mean:   {train_label['LABEL0'].mean():.4f}")
    print(f"      Std:    {train_label['LABEL0'].std():.4f}")
    print(f"      Median: {train_label['LABEL0'].median():.4f}")

    print(f"\n    Valid set volatility statistics:")
    print(f"      Mean:   {valid_label['LABEL0'].mean():.4f}")
    print(f"      Std:    {valid_label['LABEL0'].std():.4f}")

    # 8. Train Transformer model
    print("\n[7] Training PyTorch Transformer model...")
    print(f"    Model parameters:")
    print(f"      - d_feat: {d_feat}")
    print(f"      - d_model: {args.d_model}")
    print(f"      - nhead: {args.nhead}")
    print(f"      - num_layers: {args.num_layers}")
    print(f"      - batch_size: {args.batch_size}")
    print(f"      - n_epochs: {args.epochs}")
    print(f"      - learning_rate: {args.lr}")
    print(f"      - early_stop: {args.early_stop}")

    # Determine device
    if torch.cuda.is_available():
        device = "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
    else:
        device = "cpu"
    print(f"      - device: {device}")

    model = TransformerVolatilityModel(
        d_feat=d_feat,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=0.1,
        lr=args.lr,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        early_stop=args.early_stop,
        device=device,
        seed=42,
    )

    model_save_path = PROJECT_ROOT / "my_models" / f"transformer_volatility_{VOLATILITY_WINDOW}d.pt"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n    Training progress:")
    model.fit(dataset, save_path=str(model_save_path))

    # 9. Predict
    print("\n[8] Generating predictions...")
    pred = model.predict(dataset, segment="test")

    test_pred = pred.loc[TEST_START:TEST_END]
    print(f"    Prediction shape: {test_pred.shape}")
    print(f"    Prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    # 10. Evaluate
    evaluate_model(dataset, test_pred, PROJECT_ROOT, VOLATILITY_WINDOW)


if __name__ == "__main__":
    main()
