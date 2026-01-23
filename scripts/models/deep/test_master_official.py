"""
测试官方 MASTER 数据格式

使用 TSDataSampler 创建 (N, T, F) 格式数据，其中 F 包含标签在最后一列
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import TSDataSampler, DatasetH
from qlib.data.dataset.handler import DataHandlerLP

# Import our handler
from data.datahandler_master import Alpha158_Master


# ========== Utility functions (from official MASTER) ==========
def zscore(x):
    return (x - x.mean()).div(x.std() + 1e-8)

def drop_extreme(x):
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025 * N)
    if percent_2_5 == 0:
        return torch.ones_like(x, dtype=torch.bool), x
    filtered_indices = indices[percent_2_5:-percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]

def calc_ic(pred, label):
    df = pd.DataFrame({'pred': pred, 'label': label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric


class DailyBatchSamplerRandom(Sampler):
    """Daily batch sampler from official MASTER"""
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
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


def main():
    # Init qlib
    qlib.init(provider_uri="./my_data/qlib_us", region=REG_US)

    # Stock pool - use smaller set for testing
    from data.stock_pools import STOCK_POOLS
    symbols = STOCK_POOLS['sp500']

    # Time splits
    train_start, train_end = "2015-01-01", "2022-12-31"
    valid_start, valid_end = "2023-01-01", "2023-06-30"
    test_start, test_end = "2023-07-01", "2024-12-31"

    print("Creating DataHandler...")
    handler = Alpha158_Master(
        volatility_window=2,
        instruments=symbols,
        start_time=train_start,
        end_time=test_end,
        fit_start_time=train_start,
        fit_end_time=train_end,
        infer_processors=[],
    )

    # Get model config
    config = Alpha158_Master.get_model_config()
    d_feat = config['n_stock_features']  # 142
    gate_start = config['gate_input_start_index']  # 142
    gate_end = config['gate_input_end_index']  # 205

    print(f"d_feat: {d_feat}, gate_input: [{gate_start}, {gate_end})")

    # Create TSDataSampler - this is what official MASTER uses
    print("\nCreating TSDataSampler (official MASTER format)...")

    step_len = 8

    # Fetch data from handler and combine features + labels
    # TSDataSampler expects a DataFrame with MultiIndex (datetime, instrument)
    features = handler.fetch(col_set="feature")
    labels = handler.fetch(col_set="label")

    # Combine features and labels (label should be last column)
    data = pd.concat([features, labels], axis=1)
    print(f"Combined data shape: {data.shape}")
    print(f"Columns: features={features.shape[1]}, labels={labels.shape[1]}")

    # Filter by time periods
    train_data = data.loc[train_start:train_end]
    valid_data = data.loc[valid_start:valid_end]

    print(f"Train data: {train_data.shape}")
    print(f"Valid data: {valid_data.shape}")

    # Create TSDataSampler
    dl_train = TSDataSampler(
        data=train_data,
        start=train_start,
        end=train_end,
        step_len=step_len,
        fillna_type="ffill+bfill",
    )

    dl_valid = TSDataSampler(
        data=valid_data,
        start=valid_start,
        end=valid_end,
        step_len=step_len,
        fillna_type="ffill+bfill",
    )

    print(f"Train samples: {len(dl_train)}")
    print(f"Valid samples: {len(dl_valid)}")

    # Check data format
    print("\nChecking data format...")
    sampler = DailyBatchSamplerRandom(dl_train, shuffle=False)
    loader = DataLoader(dl_train, sampler=sampler, drop_last=True)

    for batch in loader:
        data = torch.squeeze(batch, dim=0)
        print(f"Batch shape: {data.shape}")  # Should be (N, T, F) where F includes label
        print(f"Expected: (N_stocks, T={step_len}, F={gate_end + 1})")  # +1 for label

        # Check feature and label extraction (official format)
        feature = data[:, :, 0:-1]
        label = data[:, -1, -1]

        print(f"Feature shape: {feature.shape}")
        print(f"Label shape: {label.shape}")
        print(f"Feature stats: mean={feature.mean():.4f}, std={feature.std():.4f}")
        print(f"Label stats: mean={label.mean():.4f}, std={label.std():.4f}")
        print(f"Label NaN: {torch.isnan(label).sum().item()}")
        break

    # Now test training with official format
    print("\n" + "="*60)
    print("Training MASTER with official data format")
    print("="*60)

    # Import official MASTER model components
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "MASTER"))
    from master import MASTER

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    model = MASTER(
        d_feat=d_feat,
        d_model=256,
        t_nhead=4,
        s_nhead=2,
        T_dropout_rate=0.5,
        S_dropout_rate=0.5,
        gate_input_start_index=gate_start,
        gate_input_end_index=gate_end,
        beta=5.0,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training
    train_loader = DataLoader(dl_train, sampler=DailyBatchSamplerRandom(dl_train, shuffle=True), drop_last=True)
    valid_loader = DataLoader(dl_valid, sampler=DailyBatchSamplerRandom(dl_valid, shuffle=False), drop_last=False)

    print("\nTraining...")
    print("-" * 60)
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Train IC':>9} | {'Val IC':>9}")
    print("-" * 60)

    for epoch in range(20):
        # Train
        model.train()
        train_losses = []
        train_ics = []

        for batch in train_loader:
            data = torch.squeeze(batch, dim=0)
            feature = data[:, :, 0:-1].to(device)
            label = data[:, -1, -1].to(device)

            # Drop extreme labels
            mask, label_clean = drop_extreme(label)
            if mask.sum() < 10:
                continue
            feature = feature[mask]
            label_clean = zscore(label_clean)

            pred = model(feature.float())
            loss = ((pred - label_clean) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
            optimizer.step()

            train_losses.append(loss.item())

            with torch.no_grad():
                ic, _ = calc_ic(pred.cpu().numpy(), label_clean.cpu().numpy())
                if not np.isnan(ic):
                    train_ics.append(ic)

        # Validate
        model.eval()
        valid_ics = []

        with torch.no_grad():
            for batch in valid_loader:
                data = torch.squeeze(batch, dim=0)
                feature = data[:, :, 0:-1].to(device)
                label = data[:, -1, -1].to(device)

                mask = ~torch.isnan(label)
                if mask.sum() < 10:
                    continue
                label_clean = zscore(label[mask])

                pred = model(feature.float())
                ic, _ = calc_ic(pred[mask].cpu().numpy(), label_clean.cpu().numpy())
                if not np.isnan(ic):
                    valid_ics.append(ic)

        train_loss = np.mean(train_losses) if train_losses else float('nan')
        train_ic = np.mean(train_ics) if train_ics else 0
        val_ic = np.mean(valid_ics) if valid_ics else 0

        print(f"{epoch+1:>6} | {train_loss:>11.6f} | {train_ic:>9.4f} | {val_ic:>9.4f}")


if __name__ == "__main__":
    main()
