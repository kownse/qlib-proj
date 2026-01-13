"""
Test TA-Lib with 50 symbols directly (no previous load)
"""
import sys
import os
from pathlib import Path
import multiprocessing

# Force spawn method
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Set up paths
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))
project_root = script_dir.parent

# Initialize qlib
import qlib
from qlib.constant import REG_US
from utils.talib_ops import TALIB_OPS
from data.datahandler_ext import Alpha158_Volatility_TALib
from data.stock_pools import STOCK_POOLS

qlib_data_path = project_root / "my_data" / "qlib_us"
qlib.init(
    provider_uri=str(qlib_data_path),
    region=REG_US,
    custom_ops=TALIB_OPS,
    kernels=1,
    joblib_backend=None,
)

# Get 50 symbols from sp500
all_symbols = STOCK_POOLS['sp500']
symbols = all_symbols[:50]

print(f"Testing with {len(symbols)} symbols directly...", flush=True)
print("   Creating handler...", flush=True)
handler = Alpha158_Volatility_TALib(
    instruments=symbols,
    start_time='2024-01-01',
    end_time='2025-12-31',
    fit_start_time='2024-01-01',
    fit_end_time='2025-09-30',
    volatility_window=5,
)
print("   Handler created, fetching data...", flush=True)
df = handler.fetch()
print(f"   SUCCESS: Data shape: {df.shape}", flush=True)
print("\nTest complete!", flush=True)
