"""
直接使用官方 MASTER 训练代码

使用 TSDataSampler + DailyBatchSamplerRandom 训练 MASTER
"""

import sys
import copy
from pathlib import Path

# Add script directory to path
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import TSDataSampler

# Import official MASTER components
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "MASTER"))
from master import MASTER, PositionalEncoding, TAttention, SAttention, Gate, TemporalAttention

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

def drop_na(x):
    mask = ~torch.isnan(x)
    return mask, x[mask]

def calc_ic(pred, label):
    df = pd.DataFrame({'pred': pred, 'label': label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric


class DailyBatchSamplerRandom:
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


def prepare_data(stock_pool='test'):
    """准备数据 - 使用标准 Alpha158 handler"""
    from qlib.contrib.data.handler import Alpha158

    # Stock pool
    from data.stock_pools import STOCK_POOLS
    symbols = STOCK_POOLS[stock_pool]

    # Time splits - 确保没有重叠
    train_start, train_end = "2015-01-01", "2022-12-31"
    valid_start, valid_end = "2023-01-01", "2023-12-31"
    test_start, test_end = "2024-01-01", "2024-12-31"

    print(f"Time splits (no overlap):")
    print(f"  Train: {train_start} to {train_end}")
    print(f"  Valid: {valid_start} to {valid_end}")
    print(f"  Test:  {test_start} to {test_end}")

    print("\nCreating Alpha158 handler...")
    handler = Alpha158(
        instruments=symbols,
        start_time=train_start,
        end_time=test_end,
        fit_start_time=train_start,
        fit_end_time=train_end,
        infer_processors=[
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        learn_processors=[
            {"class": "DropnaLabel"},
        ],
        label=["Ref($close, -2) / Ref($close, -1) - 1"],
    )

    # Fetch features and labels
    features = handler.fetch(col_set="feature")
    labels = handler.fetch(col_set="label")

    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    # Combine features and labels (label should be last column)
    # This is the key: official MASTER expects label in last column of data tensor
    data = pd.concat([features, labels], axis=1)
    print(f"Combined data shape: {data.shape}")

    # Check for NaN
    nan_count = data.isna().sum().sum()
    print(f"NaN count in combined data: {nan_count}")
    if nan_count > 0:
        data = data.fillna(0)
        print(f"Filled NaN with 0")

    # Get feature dimensions
    n_features = features.shape[1]
    print(f"Number of features: {n_features}")

    return data, n_features, {
        'train': (train_start, train_end),
        'valid': (valid_start, valid_end),
        'test': (test_start, test_end),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock-pool', type=str, default='sp500', help='Stock pool: test, sp100, sp500')
    parser.add_argument('--n-epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--early-stop', type=int, default=10, help='Early stopping patience')
    args = parser.parse_args()

    # Init qlib
    print("Initializing Qlib...")
    qlib.init(provider_uri="./my_data/qlib_us", region=REG_US)

    # Prepare data
    data, n_features, splits = prepare_data(args.stock_pool)

    # Config
    step_len = 8  # time steps (official default)
    d_feat = n_features  # feature dimension = Alpha158 features
    d_model = 256
    t_nhead = 4
    s_nhead = 2

    # For Alpha158, we use all features as stock features (no separate market features)
    # This means gate won't have separate market input - we'll use last N features as "market proxy"
    gate_start = d_feat - 20  # Use last 20 features as market proxy
    gate_end = d_feat

    print(f"\nModel config:")
    print(f"  d_feat (for linear): {gate_start}")
    print(f"  gate_input: [{gate_start}, {gate_end})")
    print(f"  step_len: {step_len}")

    # Create TSDataSampler for each split
    train_start, train_end = splits['train']
    valid_start, valid_end = splits['valid']
    test_start, test_end = splits['test']

    # Filter data by time period - 确保没有重叠
    train_data = data.loc[train_start:train_end]
    valid_data = data.loc[valid_start:valid_end]
    test_data = data.loc[test_start:test_end]

    # 验证数据集没有重叠
    train_dates = set(train_data.index.get_level_values('datetime').unique())
    valid_dates = set(valid_data.index.get_level_values('datetime').unique())
    test_dates = set(test_data.index.get_level_values('datetime').unique())

    assert len(train_dates & valid_dates) == 0, "Train and valid have overlapping dates!"
    assert len(train_dates & test_dates) == 0, "Train and test have overlapping dates!"
    assert len(valid_dates & test_dates) == 0, "Valid and test have overlapping dates!"
    print("\n✓ Verified: No overlapping dates between train/valid/test")

    print(f"\nTrain data: {train_data.shape} ({len(train_dates)} days)")
    print(f"Valid data: {valid_data.shape} ({len(valid_dates)} days)")
    print(f"Test data:  {test_data.shape} ({len(test_dates)} days)")

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

    dl_test = TSDataSampler(
        data=test_data,
        start=test_start,
        end=test_end,
        step_len=step_len,
        fillna_type="ffill+bfill",
    )

    print(f"Train samples: {len(dl_train)}")
    print(f"Valid samples: {len(dl_valid)}")
    print(f"Test samples:  {len(dl_test)}")

    # Create data loaders with daily batch sampler
    train_sampler = DailyBatchSamplerRandom(dl_train, shuffle=True)
    valid_sampler = DailyBatchSamplerRandom(dl_valid, shuffle=False)
    test_sampler = DailyBatchSamplerRandom(dl_test, shuffle=False)

    train_loader = DataLoader(dl_train, sampler=train_sampler, drop_last=True)
    valid_loader = DataLoader(dl_valid, sampler=valid_sampler, drop_last=False)
    test_loader = DataLoader(dl_test, sampler=test_sampler, drop_last=False)

    # Check data format
    print("\nChecking data format...")
    for batch in train_loader:
        batch_data = torch.squeeze(batch, dim=0)
        print(f"Batch shape: {batch_data.shape}")  # Should be (N, T, F+1) where +1 is label

        # Extract feature and label (official format)
        feature = batch_data[:, :, 0:-1]  # All except last column
        label = batch_data[:, -1, -1]  # Last column, last timestep

        print(f"Feature shape: {feature.shape}")  # (N, T, F)
        print(f"Label shape: {label.shape}")  # (N,)
        print(f"Feature stats: mean={feature.mean():.4f}, std={feature.std():.4f}")
        print(f"Label stats: mean={label.mean():.4f}, std={label.std():.4f}, nan={torch.isnan(label).sum()}")
        break

    # Create MASTER model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = MASTER(
        d_feat=gate_start,  # Stock feature dimension (up to gate_start)
        d_model=d_model,
        t_nhead=t_nhead,
        s_nhead=s_nhead,
        T_dropout_rate=0.5,
        S_dropout_rate=0.5,
        gate_input_start_index=gate_start,
        gate_input_end_index=gate_end,
        beta=5.0,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training with early stopping
    print("\n" + "="*80)
    print("Training MASTER with official data format")
    print("="*80)
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Train IC':>9} | {'Val Loss':>11} | {'Val IC':>9}")
    print("-" * 80)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(args.n_epochs):
        # Train
        model.train()
        train_losses = []
        train_ics = []

        for batch in train_loader:
            batch_data = torch.squeeze(batch, dim=0)
            feature = batch_data[:, :, 0:-1].to(device)  # (N, T, F)
            label = batch_data[:, -1, -1].to(device)  # (N,)

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
        valid_losses = []
        valid_ics = []

        with torch.no_grad():
            for batch in valid_loader:
                batch_data = torch.squeeze(batch, dim=0)
                feature = batch_data[:, :, 0:-1].to(device)
                label = batch_data[:, -1, -1].to(device)

                mask, label_clean = drop_na(label)
                if mask.sum() < 10:
                    continue
                label_clean = zscore(label_clean)

                pred = model(feature.float())
                loss = ((pred[mask] - label_clean) ** 2).mean()
                valid_losses.append(loss.item())

                ic, _ = calc_ic(pred[mask].cpu().numpy(), label_clean.cpu().numpy())
                if not np.isnan(ic):
                    valid_ics.append(ic)

        train_loss = np.mean(train_losses) if train_losses else float('nan')
        train_ic = np.mean(train_ics) if train_ics else 0
        val_loss = np.mean(valid_losses) if valid_losses else float('nan')
        val_ic = np.mean(valid_ics) if valid_ics else 0

        print(f"{epoch+1:>6} | {train_loss:>11.6f} | {train_ic:>9.4f} | {val_loss:>11.6f} | {val_ic:>9.4f}")

        # Early stopping based on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.early_stop} epochs)")
                break

    print("-" * 80)

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with val_loss: {best_val_loss:.6f}")

    # Test evaluation
    print("\n" + "="*80)
    print("Evaluating on Test Set")
    print("="*80)

    model.eval()
    test_ics = []
    test_rics = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch_data = torch.squeeze(batch, dim=0)
            feature = batch_data[:, :, 0:-1].to(device)
            label = batch_data[:, -1, -1].to(device)

            mask, label_clean = drop_na(label)
            if mask.sum() < 10:
                continue

            pred = model(feature.float())

            # Daily IC (before zscore for fair comparison)
            ic, ric = calc_ic(pred[mask].cpu().numpy(), label[mask].cpu().numpy())
            if not np.isnan(ic):
                test_ics.append(ic)
                test_rics.append(ric)

            all_preds.extend(pred[mask].cpu().numpy().tolist())
            all_labels.extend(label[mask].cpu().numpy().tolist())

    # Calculate overall metrics
    mean_ic = np.mean(test_ics) if test_ics else 0
    std_ic = np.std(test_ics) if len(test_ics) > 1 else 1
    icir = mean_ic / std_ic if std_ic > 0 else 0

    mean_ric = np.mean(test_rics) if test_rics else 0
    std_ric = np.std(test_rics) if len(test_rics) > 1 else 1
    ricir = mean_ric / std_ric if std_ric > 0 else 0

    # Overall IC
    overall_ic, overall_ric = calc_ic(np.array(all_preds), np.array(all_labels))

    print(f"\nTest Set Results:")
    print(f"  Daily IC:     {mean_ic:.4f} ± {std_ic:.4f}")
    print(f"  ICIR:         {icir:.4f}")
    print(f"  Daily RankIC: {mean_ric:.4f} ± {std_ric:.4f}")
    print(f"  RankICIR:     {ricir:.4f}")
    print(f"  Overall IC:   {overall_ic:.4f}")
    print(f"  Overall RIC:  {overall_ric:.4f}")
    print(f"  Test days:    {len(test_ics)}")

    print("="*80)
    print("Done!")


if __name__ == "__main__":
    main()
