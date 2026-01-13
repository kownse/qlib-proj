"""
Test with varying numbers of TA-Lib features to find the limit
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

# Get feature count from command line
feature_count = int(sys.argv[1]) if len(sys.argv) > 1 else 5
symbols_count = int(sys.argv[2]) if len(sys.argv) > 2 else 100

# Initialize qlib with TA-Lib
import qlib
from qlib.constant import REG_US
from utils.talib_ops import TALIB_OPS
from qlib.data.dataset.handler import DataHandlerLP

qlib_data_path = project_root / "my_data" / "qlib_us"
qlib.init(
    provider_uri=str(qlib_data_path),
    region=REG_US,
    custom_ops=TALIB_OPS,
    kernels=1,
    joblib_backend=None,
)

# Get symbols
from data.stock_pools import STOCK_POOLS
symbols = STOCK_POOLS['sp500'][:symbols_count]

# Build feature list with increasing TA-Lib features
all_talib_features = [
    ("TALIB_RSI($close, 14)", "rsi14"),
    ("TALIB_RSI($close, 7)", "rsi7"),
    ("TALIB_MOM($close, 10)/$close", "mom10"),
    ("TALIB_MOM($close, 5)/$close", "mom5"),
    ("TALIB_ROC($close, 10)", "roc10"),
    ("TALIB_ROC($close, 5)", "roc5"),
    ("TALIB_CMO($close, 14)", "cmo14"),
    ("TALIB_EMA($close, 20)/$close", "ema20"),
    ("TALIB_EMA($close, 10)/$close", "ema10"),
    ("TALIB_SMA($close, 20)/$close", "sma20"),
    ("TALIB_BBANDS_UPPER($close, 20, 2)/$close", "bb_upper"),
    ("TALIB_BBANDS_LOWER($close, 20, 2)/$close", "bb_lower"),
    ("TALIB_ATR($high, $low, $close, 14)/$close", "atr14"),
    ("TALIB_ADX($high, $low, $close, 14)", "adx14"),
    ("TALIB_WILLR($high, $low, $close, 14)", "willr14"),
    ("TALIB_CCI($high, $low, $close, 14)", "cci14"),
    ("TALIB_MACD_MACD($close, 12, 26, 9)/$close", "macd"),
    ("TALIB_MACD_SIGNAL($close, 12, 26, 9)/$close", "macd_signal"),
    ("TALIB_MACD_HIST($close, 12, 26, 9)/$close", "macd_hist"),
    ("TALIB_STOCH_K($high, $low, $close, 5, 3, 3)", "stoch_k"),
]

# Select features
features = all_talib_features[:feature_count]
fields = ["$close"] + [f[0] for f in features]
names = ["close"] + [f[1] for f in features]

print(f"Testing {len(features)} TA-Lib features with {len(symbols)} symbols...", flush=True)
print(f"Features: {names[1:]}", flush=True)

data_loader = {
    "class": "QlibDataLoader",
    "kwargs": {
        "config": {
            "feature": (fields, names),
            "label": (["Ref($close, -5)/Ref($close, -1) - 1"], ["LABEL0"]),
        },
    },
}

from qlib.contrib.data.handler import check_transform_proc, _DEFAULT_LEARN_PROCESSORS

print("   Creating handler...", flush=True)
handler = DataHandlerLP(
    instruments=symbols,
    start_time='2024-01-01',
    end_time='2025-12-31',
    data_loader=data_loader,
    infer_processors=check_transform_proc([], '2024-01-01', '2025-09-30'),
    learn_processors=check_transform_proc(_DEFAULT_LEARN_PROCESSORS, '2024-01-01', '2025-09-30'),
)
print("   Handler created, fetching data...", flush=True)
df = handler.fetch()
print(f"   SUCCESS: Data shape: {df.shape}", flush=True)
