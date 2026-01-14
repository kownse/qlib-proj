"""
Debug script to investigate IC=0 issue with macro handler
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
project_root = script_dir.parent

import qlib
from qlib.constant import REG_US
from utils.talib_ops import TALIB_OPS

qlib_data_path = project_root / "my_data" / "qlib_us"
qlib.init(
    provider_uri=str(qlib_data_path),
    region=REG_US,
    custom_ops=TALIB_OPS,
    kernels=1,
)

import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from data.datahandler_macro import Alpha158_Volatility_TALib_Macro
from data.stock_pools import STOCK_POOLS

# Test parameters
symbols = STOCK_POOLS['sp500'][:50]  # Use subset for faster test
fold_config = {
    'train_start': '2000-01-01',
    'train_end': '2021-12-31',
    'valid_start': '2022-01-01',
    'valid_end': '2022-12-31',
}

print("=" * 70)
print("DEBUG: Macro Handler Data Check")
print("=" * 70)

# Create handler
print("\n[1] Creating handler...")
handler = Alpha158_Volatility_TALib_Macro(
    volatility_window=5,
    instruments=symbols,
    start_time=fold_config['train_start'],
    end_time=fold_config['valid_end'],
    fit_start_time=fold_config['train_start'],
    fit_end_time=fold_config['train_end'],
    infer_processors=[],
)

# Create dataset
print("\n[2] Creating dataset...")
dataset = DatasetH(
    handler=handler,
    segments={
        "train": (fold_config['train_start'], fold_config['train_end']),
        "valid": (fold_config['valid_start'], fold_config['valid_end']),
    }
)

# Get data
print("\n[3] Preparing data...")
train_data = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
valid_data = dataset.prepare("valid", col_set="feature", data_key=DataHandlerLP.DK_L)
train_label = dataset.prepare("train", col_set="label")
valid_label = dataset.prepare("valid", col_set="label")

print(f"\n[4] Data shapes:")
print(f"    Train features: {train_data.shape}")
print(f"    Valid features: {valid_data.shape}")
print(f"    Train labels: {train_label.shape}")
print(f"    Valid labels: {valid_label.shape}")

print(f"\n[5] NaN analysis:")
train_nan_pct = train_data.isna().mean().mean() * 100
valid_nan_pct = valid_data.isna().mean().mean() * 100
train_label_nan = train_label.isna().mean().values[0] * 100
valid_label_nan = valid_label.isna().mean().values[0] * 100
print(f"    Train features NaN: {train_nan_pct:.2f}%")
print(f"    Valid features NaN: {valid_nan_pct:.2f}%")
print(f"    Train label NaN: {train_label_nan:.2f}%")
print(f"    Valid label NaN: {valid_label_nan:.2f}%")

print(f"\n[6] Label statistics:")
train_label_vals = train_label.values.ravel()
valid_label_vals = valid_label.values.ravel()
print(f"    Train label - mean: {np.nanmean(train_label_vals):.6f}, std: {np.nanstd(train_label_vals):.6f}")
print(f"    Valid label - mean: {np.nanmean(valid_label_vals):.6f}, std: {np.nanstd(valid_label_vals):.6f}")
print(f"    Train label - min: {np.nanmin(train_label_vals):.6f}, max: {np.nanmax(train_label_vals):.6f}")
print(f"    Valid label - min: {np.nanmin(valid_label_vals):.6f}, max: {np.nanmax(valid_label_vals):.6f}")

print(f"\n[7] Feature sample (first 5 columns):")
print(train_data.iloc[:5, :5])

print(f"\n[8] Label sample:")
print(train_label.head(10))

print(f"\n[9] Index structure:")
print(f"    Train index names: {train_data.index.names}")
print(f"    Valid index names: {valid_data.index.names}")
print(f"    Train index sample:\n{train_data.index[:5].tolist()}")

print(f"\n[10] Valid data index dates (unique):")
valid_dates = valid_data.index.get_level_values('datetime').unique()
print(f"     Number of unique dates: {len(valid_dates)}")
print(f"     Date range: {valid_dates.min()} to {valid_dates.max()}")

# Quick model test
print(f"\n[11] Quick CatBoost test...")
from catboost import CatBoostRegressor

# Check how many rows have NaN
train_rows_with_nan = train_data.isna().any(axis=1).sum()
valid_rows_with_nan = valid_data.isna().any(axis=1).sum()
print(f"     Train rows with any NaN: {train_rows_with_nan} / {len(train_data)} ({train_rows_with_nan/len(train_data)*100:.1f}%)")
print(f"     Valid rows with any NaN: {valid_rows_with_nan} / {len(valid_data)} ({valid_rows_with_nan/len(valid_data)*100:.1f}%)")

# Check which columns have NaN
nan_cols = train_data.columns[train_data.isna().any()].tolist()
print(f"     Columns with NaN: {len(nan_cols)}")
if nan_cols:
    print(f"     Sample NaN columns: {nan_cols[:10]}")

# CatBoost can handle NaN natively - don't drop
X_train = train_data
y_train = train_label_vals
X_valid = valid_data
y_valid = valid_label_vals

print(f"     Training with NaN (CatBoost handles it):")
print(f"     Train: {X_train.shape}, Valid: {X_valid.shape}")

model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    verbose=False,
)
model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose_eval=False)

valid_pred = model.predict(X_valid)

print(f"\n[12] Prediction analysis:")
print(f"     Pred mean: {valid_pred.mean():.6f}, std: {valid_pred.std():.6f}")
print(f"     Pred min: {valid_pred.min():.6f}, max: {valid_pred.max():.6f}")
print(f"     Label mean: {y_valid.mean():.6f}, std: {y_valid.std():.6f}")

# Compute IC
df = pd.DataFrame({
    'pred': valid_pred,
    'label': y_valid
}, index=X_valid.index)

ic_by_date = df.groupby(level='datetime').apply(
    lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
)
ic_by_date = ic_by_date.dropna()

print(f"\n[13] IC analysis:")
print(f"     Number of dates for IC: {len(ic_by_date)}")
print(f"     Mean IC: {ic_by_date.mean():.4f}")
print(f"     IC std: {ic_by_date.std():.4f}")
print(f"     IC sample:\n{ic_by_date.head(10)}")

# Check if predictions are constant per day
print(f"\n[14] Check prediction variance per date:")
pred_std_by_date = df.groupby(level='datetime')['pred'].std()
print(f"     Mean pred std per date: {pred_std_by_date.mean():.6f}")
print(f"     Min pred std: {pred_std_by_date.min():.6f}")
print(f"     Max pred std: {pred_std_by_date.max():.6f}")

label_std_by_date = df.groupby(level='datetime')['label'].std()
print(f"\n     Mean label std per date: {label_std_by_date.mean():.6f}")

print("\n" + "=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)
