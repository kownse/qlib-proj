"""
Enhanced DataHandler V7 + Sector-Relative Features.

Extends V7 (40 features) with 4 sector-relative features (total: 43).
SR features compute stock - sector_median for key cross-sectional features,
providing stock-level differentiation that market-level macro features lack.

Methodology:
- SR_feature = stock_feature - sector_median(stock_feature) per day per sector
- Uses median (not mean) to be robust against large-cap outliers (AAPL, MSFT)
- Computed on raw _data BEFORE ZScoreNorm, so SR features get normalized together
- No look-ahead bias: same-day cross-sectional comparison only

Feature selection: top 4 by |ICIR| from diagnostic on 2024 H1 data.

Sector data: my_data/sector_data/sector_info.json (518 stocks, 11 GICS sectors)
"""

import json
import sys
from pathlib import Path
from typing import Union

import pandas as pd

script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from data.datahandler_enhanced_v7 import Alpha158_Enhanced_V7

PROJECT_ROOT = script_dir.parent
DEFAULT_SECTOR_PATH = PROJECT_ROOT / "my_data" / "sector_data" / "sector_info.json"

# Top 4 SR features by |ICIR| (diagnostic on 2024 H1 SP500 data)
SR_BASE_FEATURES = [
    "PCT_FROM_52W_HIGH",   # ICIR=0.30 — distance from 52-week high
    "SHARPE_60D",          # ICIR=0.24 — 60-day Sharpe ratio
    "ROC5",                # ICIR=0.20 — 5-day momentum
    "ROC60",               # ICIR=-0.20 — 60-day momentum
]


class Alpha158_Enhanced_V7_SectorRelative(Alpha158_Enhanced_V7):
    """
    V7 + 4 sector-relative features (43 total).

    Stock-specific: 28 features (from V7)
    Macro regime (lagged): 11 features (from V7)
    Sector-relative: 4 features (top |ICIR|, normalized by ZScoreNorm)
    Total: 43 features
    """

    def __init__(
        self,
        sector_info_path: Union[str, Path] = None,
        **kwargs,
    ):
        # Load sector mapping before super().__init__ triggers data loading
        sector_path = DEFAULT_SECTOR_PATH if sector_info_path is None else Path(sector_info_path)
        self._sector_map = self._load_sector_map(sector_path)

        super().__init__(**kwargs)

    @staticmethod
    def _load_sector_map(path: Path) -> dict:
        """Load sector_info.json → {lowercase_symbol: sector_name}."""
        with open(path, "r") as f:
            raw = json.load(f)
        # Qlib US uses lowercase instrument names
        return {sym.lower(): info["sector"] for sym, info in raw.items()}

    def process_data(self, with_fit: bool = False):
        """
        Override to add SR features on raw _data BEFORE ZScoreNorm.

        Flow:
        1. Compute SR features on self._data (raw features from QlibDataLoader)
        2. V7's process_data() → DataHandlerLP.process_data() applies ZScoreNorm
           to ALL features including SR → then adds macro features post-norm
        """
        # Add SR features to raw data so they go through ZScoreNorm
        if hasattr(self, "_data") and self._data is not None:
            self._data = self._compute_sector_relative(self._data)

        # V7's process_data: ZScoreNorm (now includes SR) → macro features
        super().process_data(with_fit=with_fit)

    def _compute_sector_relative(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute SR features: stock_value - sector_median per day per sector.

        Args:
            df: DataFrame with (datetime, instrument) MultiIndex

        Returns:
            DataFrame with SR features appended
        """
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        # Map instruments to sectors (lowercase lookup — Qlib _data uses uppercase)
        instruments = df.index.get_level_values(1)
        sector_series = instruments.map(
            lambda x: self._sector_map.get(x.lower(), "Unknown")
        )

        # Build grouping keys: (datetime, sector)
        datetime_level = df.index.get_level_values(0)
        group_keys = [datetime_level, sector_series]

        sr_data = {}
        added = 0

        for base_name in SR_BASE_FEATURES:
            # Locate the base feature column
            if has_multi_columns:
                col_key = ("feature", base_name)
            else:
                col_key = base_name

            if col_key not in df.columns:
                continue

            feature_values = df[col_key]

            # Compute sector median per day
            sector_median = feature_values.groupby(group_keys).transform("median")

            # SR feature = stock - sector_median
            sr_values = feature_values - sector_median

            sr_name = f"SR_{base_name}"
            if has_multi_columns:
                sr_data[("feature", sr_name)] = sr_values.values
            else:
                sr_data[sr_name] = sr_values.values

            added += 1

        if sr_data:
            sr_df = pd.DataFrame(sr_data, index=df.index)
            result = pd.concat([df, sr_df], axis=1, copy=False)
            print(f"Added {added} sector-relative features (pre-normalization)")
            return result.copy()

        print("Warning: No sector-relative features could be computed")
        return df
