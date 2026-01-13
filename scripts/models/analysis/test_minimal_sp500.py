"""
Minimal test for sp500 + alpha158-talib to isolate memory corruption issue
"""
import sys
import os
from pathlib import Path
import multiprocessing

# Force spawn method before any other imports
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Set up paths
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))
project_root = script_dir.parent

print("[1] Importing qlib...")
import qlib
from qlib.constant import REG_US

print("[2] Importing talib_ops...")
from utils.talib_ops import TALIB_OPS

print("[3] Initializing qlib...")
qlib_data_path = project_root / "my_data" / "qlib_us"
qlib.init(
    provider_uri=str(qlib_data_path),
    region=REG_US,
    custom_ops=TALIB_OPS,
    kernels=1,
    joblib_backend=None,
)
print("   qlib initialized successfully")

print("[4] Importing DataHandlerLP...")
from qlib.data.dataset.handler import DataHandlerLP
print("   DataHandlerLP imported")

print("[5] Importing DatasetH...")
from qlib.data.dataset import DatasetH
print("   DatasetH imported")

print("[6] Importing Alpha158_Volatility_TALib...")
from data.datahandler_ext import Alpha158_Volatility_TALib
print("   Alpha158_Volatility_TALib imported")

print("[7] Importing stock pools...")
from data.stock_pools import STOCK_POOLS
symbols = STOCK_POOLS['sp500']
print(f"   Got {len(symbols)} symbols")

print("[8] Creating handler config...")
handler_config = {
    'start_time': '2024-01-01',
    'end_time': '2025-12-31',
    'fit_start_time': '2024-01-01',
    'fit_end_time': '2025-09-30',
    'instruments': symbols,
}
print(f"   Handler config created")

print("[9] Creating Alpha158_Volatility_TALib handler...")
print("   This is where the crash typically occurs...")
handler = Alpha158_Volatility_TALib(
    instruments=handler_config['instruments'],
    start_time=handler_config['start_time'],
    end_time=handler_config['end_time'],
    fit_start_time=handler_config['fit_start_time'],
    fit_end_time=handler_config['fit_end_time'],
    volatility_window=5,
)
print("   Handler created successfully!")

print("[10] Checking data shape...")
df = handler.fetch()
print(f"   Data shape: {df.shape}")
print(f"   Columns: {len(df.columns)}")

print("\nSUCCESS: All tests passed!")
