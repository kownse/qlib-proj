"""
Enhanced DataHandler V12 - V9 Market-Neutral + Selected CBOE Features.

Based on CBOE forward selection on v9-mkt-neutral baseline
(cboe_fwd_sel_mkt_neutral_20260210_214832.json).

Forward Selection Results:
- Baseline (V9 mkt-neutral, 11 features): CV IC = 0.0128
- +cboe_skew_regime:  CV IC = 0.0258 (+0.0130)
- +cboe_skew_level:   CV IC = 0.0271 (+0.0013)

8 CBOE features excluded (hurt or didn't help IC):
  cboe_vvix_regime, cboe_skew_pct_5d, cboe_vvix_zscore20,
  cboe_vix9d_vs_vix, cboe_skew_zscore20, cboe_vvix_level,
  cboe_vvix_vs_vix, cboe_vix9d_spike

Features (13 total):
- Stock-specific: 7 features (from V9)
- Macro regime (lagged): 4 features (from V9, including sector_dispersion)
- CBOE options (lagged): 2 features (skew_regime, skew_level)
- Label: market-neutral (stock return - SPY return)
"""

import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd

script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from data.datahandler_target_variants import Alpha158_V9_MarketNeutral

PROJECT_ROOT = script_dir.parent
DEFAULT_CBOE_PATH = PROJECT_ROOT / "my_data" / "cboe_processed" / "cboe_features.parquet"


class Alpha158_Enhanced_V12(Alpha158_V9_MarketNeutral):
    """
    V9 market-neutral + selected CBOE features.

    Inherits from Alpha158_V9_MarketNeutral which provides:
    - 7 stock features (V9 forward selection optimal)
    - 4 macro features (vix_zscore, hy_spread, credit_stress, sector_dispersion)
    - Market-neutral label (stock return - SPY return)

    Adds 2 CBOE features selected by forward selection:
    - cboe_skew_regime: High skew regime indicator (tail risk)
    - cboe_skew_level: SKEW index level

    Total: 13 features + market-neutral label
    """

    CBOE_FEATURES = [
        "cboe_skew_regime",
        "cboe_skew_level",
    ]

    def __init__(
        self,
        cboe_data_path: Union[str, Path] = None,
        cboe_lag: int = 1,
        **kwargs,
    ):
        self.cboe_data_path = Path(cboe_data_path) if cboe_data_path else DEFAULT_CBOE_PATH
        self.cboe_lag = cboe_lag
        self._cboe_df = self._load_cboe_features()
        super().__init__(**kwargs)

    def process_data(self, with_fit: bool = False):
        """Add CBOE features after parent processing (V9 + macro + mkt-neutral)."""
        super().process_data(with_fit=with_fit)

        if self._cboe_df is not None:
            self._add_cboe_features()

    def _load_cboe_features(self) -> Optional[pd.DataFrame]:
        """Load CBOE features from parquet."""
        if not self.cboe_data_path.exists():
            print(f"Warning: CBOE features not found: {self.cboe_data_path}")
            return None

        try:
            df = pd.read_parquet(self.cboe_data_path)
            print(f"Loaded CBOE features: {df.shape}, "
                  f"range: {df.index.min()} to {df.index.max()}")
            return df
        except Exception as e:
            print(f"Warning: Failed to load CBOE features: {e}")
            return None

    def _add_cboe_features(self):
        """Add selected CBOE features to _learn and _infer."""
        available = [c for c in self.CBOE_FEATURES if c in self._cboe_df.columns]
        if not available:
            print("Warning: No selected CBOE features found in data")
            return

        for df_attr in ["_learn", "_infer"]:
            df = getattr(self, df_attr, None)
            if df is None:
                continue
            self._merge_cboe(df, available)

        print(f"Added {len(available)} CBOE features (lag={self.cboe_lag}d): {available}")

    def _merge_cboe(self, df: pd.DataFrame, cols: list):
        """Merge lagged CBOE features into a DataFrame."""
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        for col in cols:
            cboe_series = self._cboe_df[col].shift(self.cboe_lag)
            aligned_values = cboe_series.reindex(main_datetimes).values

            col_name = f"{col}_lag{self.cboe_lag}"
            if has_multi_columns:
                df[("feature", col_name)] = aligned_values
            else:
                df[col_name] = aligned_values
