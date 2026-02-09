"""
DataHandler: V9 features + CBOE options-derived features.

Extends Alpha158_Enhanced_V9 with CBOE features (SKEW, VVIX, VIX9D)
that capture additional dimensions of options market sentiment:
- Tail risk (SKEW)
- Volatility of volatility (VVIX)
- Short-term vs medium-term fear (VIX9D/VIX ratio)

These features are market-level (same for all stocks on a given day)
and are lagged by 1 day to avoid look-ahead bias, consistent with
existing macro feature handling.

Total features: 11 (V9 stock + macro) + ~9 (CBOE) = ~20

Usage:
    python scripts/models/deep/run_ae_mlp_cv.py \
        --params-file outputs/hyperopt_cv/ae_mlp_cv_best_params_v9_best.json \
        --handler v9-cboe --cv-only --seed 42
"""

import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd

script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from data.datahandler_enhanced_v9 import Alpha158_Enhanced_V9

PROJECT_ROOT = script_dir.parent
DEFAULT_CBOE_PATH = PROJECT_ROOT / "my_data" / "cboe_processed" / "cboe_features.parquet"


class Alpha158_V9_CBOE(Alpha158_Enhanced_V9):
    """
    V9 features + CBOE options-derived features.

    Adds ~9 CBOE features on top of V9's 11 features:
    - cboe_skew_level: SKEW index level (tail risk)
    - cboe_skew_zscore20: SKEW 20-day z-score
    - cboe_skew_pct_5d: SKEW 5-day change
    - cboe_skew_regime: High skew indicator
    - cboe_vvix_level: VVIX level
    - cboe_vvix_zscore20: VVIX 20-day z-score
    - cboe_vvix_regime: High VVIX indicator
    - cboe_vvix_vs_vix: VVIX/VIX ratio z-scored
    - cboe_vix9d_vs_vix: VIX9D/VIX ratio z-scored
    - cboe_vix9d_spike: Short-term fear spike indicator
    """

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
        """Override to add CBOE features after parent processing."""
        super().process_data(with_fit=with_fit)

        if self._cboe_df is not None:
            self._add_cboe_features()

    def _load_cboe_features(self) -> Optional[pd.DataFrame]:
        """Load CBOE features from parquet."""
        if not self.cboe_data_path.exists():
            print(f"Warning: CBOE features not found: {self.cboe_data_path}")
            print("Run: python scripts/data/download_cboe_data.py && "
                  "python scripts/data/process_cboe_data.py")
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
        """Add lagged CBOE features to _learn and _infer."""
        try:
            cboe_cols = [c for c in self._cboe_df.columns]
            if not cboe_cols:
                return

            if hasattr(self, "_learn") and self._learn is not None:
                self._learn = self._merge_cboe(self._learn, cboe_cols)
                print(f"Added {len(cboe_cols)} CBOE features (lag={self.cboe_lag}d)")

            if hasattr(self, "_infer") and self._infer is not None:
                self._infer = self._merge_cboe(self._infer, cboe_cols)

        except Exception as e:
            print(f"Warning: Error adding CBOE features: {e}")

    def _merge_cboe(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Merge lagged CBOE features into a DataFrame."""
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        cboe_data = {}
        for col in cols:
            # Apply lag to avoid look-ahead bias
            cboe_series = self._cboe_df[col].shift(self.cboe_lag)
            aligned_values = cboe_series.reindex(main_datetimes).values

            col_name = f"{col}_lag{self.cboe_lag}"
            if has_multi_columns:
                cboe_data[('feature', col_name)] = aligned_values
            else:
                cboe_data[col_name] = aligned_values

        cboe_df = pd.DataFrame(cboe_data, index=df.index)
        merged = pd.concat([df, cboe_df], axis=1, copy=False)
        return merged.copy()
