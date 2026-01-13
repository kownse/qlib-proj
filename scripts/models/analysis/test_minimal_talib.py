"""
Test with minimal TA-Lib features to isolate the issue
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
count = int(sys.argv[1]) if len(sys.argv) > 1 else 50

# Initialize qlib with TA-Lib
import qlib
from qlib.constant import REG_US
from utils.talib_ops import TALIB_OPS
from qlib.data.dataset.handler import DataHandlerLP

qlib_data_path = project_root / "my_data" / "qlib_us"
qlib.init(
    provider_uri=str(qlib_data_path),
    region=REG_US,
    custom_ops=TALIB_OPS,  # Enable TA-Lib operators
    kernels=1,
    joblib_backend=None,
)

# Get symbols
from data.stock_pools import STOCK_POOLS
all_symbols = STOCK_POOLS['sp500']
symbols = all_symbols[:count]

print(f"Testing minimal TA-Lib handler with {len(symbols)} symbols...", flush=True)

# Use only RSI - the simplest TA-Lib indicator
data_loader = {
    "class": "QlibDataLoader",
    "kwargs": {
        "config": {
            "feature": (["$close", "TALIB_RSI($close, 14)"], ["close", "rsi"]),
            "label": (["Ref($close, -5)/Ref($close, -1) - 1"], ["LABEL0"]),
        },
    },
}

from qlib.contrib.data.handler import check_transform_proc, _DEFAULT_LEARN_PROCESSORS

infer_processors = check_transform_proc([], '2024-01-01', '2025-09-30')
learn_processors = check_transform_proc(_DEFAULT_LEARN_PROCESSORS, '2024-01-01', '2025-09-30')

print("   Creating handler...", flush=True)
handler = DataHandlerLP(
    instruments=symbols,
    start_time='2024-01-01',
    end_time='2025-12-31',
    data_loader=data_loader,
    infer_processors=infer_processors,
    learn_processors=learn_processors,
)
print("   Handler created, fetching data...", flush=True)
df = handler.fetch()
print(f"   SUCCESS: Data shape: {df.shape}", flush=True)
