"""
Test the 36th symbol specifically
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

# Get first 36 symbols and last symbol
all_symbols = STOCK_POOLS['sp500']
print(f"Symbol 35 (index 34): {all_symbols[34]}")
print(f"Symbol 36 (index 35): {all_symbols[35]}")
print(f"Symbol 37 (index 36): {all_symbols[36]}")

# Test just the 36th symbol alone
symbol_36 = [all_symbols[35]]
print(f"\nTesting symbol 36 alone: {symbol_36}", flush=True)
handler = Alpha158_Volatility_TALib(
    instruments=symbol_36,
    start_time='2024-01-01',
    end_time='2025-12-31',
    fit_start_time='2024-01-01',
    fit_end_time='2025-09-30',
    volatility_window=5,
)
df = handler.fetch()
print(f"   SUCCESS: Data shape: {df.shape}", flush=True)

# Test symbols 31-36 (skipping first 30)
skip_first_30 = all_symbols[30:36]
print(f"\nTesting symbols 31-36: {skip_first_30}", flush=True)
handler2 = Alpha158_Volatility_TALib(
    instruments=skip_first_30,
    start_time='2024-01-01',
    end_time='2025-12-31',
    fit_start_time='2024-01-01',
    fit_end_time='2025-09-30',
    volatility_window=5,
)
df2 = handler2.fetch()
print(f"   SUCCESS: Data shape: {df2.shape}", flush=True)

print("\nTest complete!")
