"""
Test Alpha158 without TA-Lib to isolate the issue
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

# Get count from command line
count = int(sys.argv[1]) if len(sys.argv) > 1 else 100

# Initialize qlib WITHOUT TA-Lib
import qlib
from qlib.constant import REG_US
from qlib.contrib.data.handler import Alpha158

qlib_data_path = project_root / "my_data" / "qlib_us"
qlib.init(
    provider_uri=str(qlib_data_path),
    region=REG_US,
    # No custom_ops - don't use TA-Lib
    kernels=1,
    joblib_backend=None,
)

# Get symbols
from data.stock_pools import STOCK_POOLS
all_symbols = STOCK_POOLS['sp500']
symbols = all_symbols[:count]

print(f"Testing Alpha158 (no TA-Lib) with {len(symbols)} symbols...", flush=True)
print("   Creating handler...", flush=True)
handler = Alpha158(
    instruments=symbols,
    start_time='2024-01-01',
    end_time='2025-12-31',
    fit_start_time='2024-01-01',
    fit_end_time='2025-09-30',
)
print("   Handler created, fetching data...", flush=True)
df = handler.fetch()
print(f"   SUCCESS: Data shape: {df.shape}", flush=True)
