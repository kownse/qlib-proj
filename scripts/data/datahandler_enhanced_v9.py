"""
Enhanced DataHandler V9 - Optimal Feature Set from AE-MLP Forward Selection.

Design Philosophy:
- Features selected by AE-MLP forward selection (NOT simple MLP)
- 11 features outperform V7's 40 features (Test IC: 0.0453 vs 0.0446)
- Focus on credit/macro features which provide largest IC gains

Forward Selection Results:
- Baseline (8 features): Test IC = -0.0009
- +macro_hy_spread_zscore: Test IC = 0.0316 (+0.0324)
- +macro_credit_stress: Test IC = 0.0400 (+0.0085)
- +ROC60: Test IC = 0.0453 (+0.0052)

Key Insight: Credit spread features (hy_spread_zscore, credit_stress) are
the most important predictors for the AE-MLP model.

Features (11 total):
- Stock-specific: 7 features
- Macro regime (lagged): 4 features
"""

import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from qlib.data.dataset.handler import DataHandlerLP

PROJECT_ROOT = script_dir.parent
DEFAULT_MACRO_PATH = PROJECT_ROOT / "my_data" / "macro_processed" / "macro_features.parquet"


class Alpha158_Enhanced_V9(DataHandlerLP):
    """
    Enhanced Alpha158 V9 - Optimal AE-MLP Feature Set.

    Stock-specific features (7):
    - MOMENTUM_QUALITY: Risk-adjusted momentum (from V8 top)
    - PCT_FROM_52W_HIGH: Distance from 52-week high (from V8 top)
    - MAX60: 60-day max high ratio (from V8 top)
    - TALIB_NATR14: Normalized ATR (discovered by forward selection)
    - MAX_DRAWDOWN_60: 60-day max drawdown (new theory feature)
    - RESI20: 20-day residual (discovered by forward selection)
    - ROC60: 60-day rate of change (added in round 3)

    Macro regime features (4, all 1-day lagged):
    - macro_sector_dispersion: Sector return dispersion (computed)
    - macro_vix_zscore20: VIX z-score
    - macro_hy_spread_zscore: High-yield spread z-score (biggest IC gain!)
    - macro_credit_stress: Credit stress indicator

    Total: 11 features

    Performance:
    - Test IC: 0.0453 (vs V7's 0.0446 with 40 features)
    - 73% fewer features, slightly better IC
    """

    # Macro features with highest IC contribution
    MACRO_REGIME_FEATURES = [
        "macro_vix_zscore20",         # From V8 base
        "macro_hy_spread_zscore",     # +0.0324 IC gain (biggest!)
        "macro_credit_stress",        # +0.0085 IC gain
        # sector_dispersion computed separately
    ]

    def __init__(
        self,
        volatility_window: int = 5,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq: str = "day",
        infer_processors=[],
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        macro_lag: int = 1,
        macro_data_path: Union[str, Path] = None,
        **kwargs,
    ):
        self.volatility_window = volatility_window
        self.macro_lag = macro_lag
        self.macro_data_path = Path(macro_data_path) if macro_data_path else DEFAULT_MACRO_PATH

        self._macro_df = self._load_macro_features()

        from qlib.contrib.data.handler import check_transform_proc, _DEFAULT_LEARN_PROCESSORS

        if learn_processors is None:
            learn_processors = _DEFAULT_LEARN_PROCESSORS

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        """Get feature config - 7 stock features from forward selection."""
        fields = []
        names = []

        # 1. MOMENTUM_QUALITY - Risk-adjusted momentum
        fields.append("($close/Ref($close, 20) - 1) / (Std($close/Ref($close,1)-1, 20) + 1e-12)")
        names.append("MOMENTUM_QUALITY")

        # 2. PCT_FROM_52W_HIGH - Distance from 52-week high
        fields.append("($close - Max($high, 252)) / (Max($high, 252) + 1e-12)")
        names.append("PCT_FROM_52W_HIGH")

        # 3. MAX60 - 60-day max high ratio
        fields.append("Max($high, 60)/$close")
        names.append("MAX60")

        # 4. TALIB_NATR14 - Normalized Average True Range
        fields.append("TALIB_NATR($high, $low, $close, 14)")
        names.append("TALIB_NATR14")

        # 5. MAX_DRAWDOWN_60 - 60-day maximum drawdown
        fields.append("(Min($close, 60) - Max($high, 60)) / (Max($high, 60) + 1e-12)")
        names.append("MAX_DRAWDOWN_60")

        # 6. RESI20 - 20-day residual
        fields.append("Resi($close, 20)/$close")
        names.append("RESI20")

        # 7. ROC60 - 60-day rate of change (added in round 3)
        fields.append("Ref($close, 60)/$close")
        names.append("ROC60")

        return fields, names

    # ========== Macro Features ==========

    def process_data(self, with_fit: bool = False):
        """Override to add lagged macro regime features."""
        super().process_data(with_fit=with_fit)

        if self._macro_df is not None:
            self._add_lagged_macro_features()
            self._add_computed_macro_features()

    def _add_lagged_macro_features(self):
        """Add lagged macro regime indicators."""
        try:
            available_cols = [c for c in self.MACRO_REGIME_FEATURES if c in self._macro_df.columns]

            if not available_cols:
                print("Warning: No macro regime features available")
                return

            if hasattr(self, "_learn") and self._learn is not None:
                self._learn = self._merge_lagged_macro(self._learn, available_cols)
                print(f"Added {len(available_cols)} lagged macro features (lag={self.macro_lag}d)")

            if hasattr(self, "_infer") and self._infer is not None:
                self._infer = self._merge_lagged_macro(self._infer, available_cols)

        except Exception as e:
            print(f"Warning: Error adding macro features: {e}")

    def _add_computed_macro_features(self):
        """Add sector dispersion - computed from sector ETF returns."""
        try:
            sector_pct_cols = [c for c in self._macro_df.columns
                             if c.endswith('_pct_20d') and c.startswith('macro_xl')]

            if len(sector_pct_cols) >= 5:
                sector_df = self._macro_df[sector_pct_cols]
                sector_dispersion = sector_df.std(axis=1).shift(self.macro_lag)

                if hasattr(self, "_learn") and self._learn is not None:
                    self._learn = self._add_single_macro(self._learn, sector_dispersion, "macro_sector_dispersion")

                if hasattr(self, "_infer") and self._infer is not None:
                    self._infer = self._add_single_macro(self._infer, sector_dispersion, "macro_sector_dispersion")

                print(f"Added computed macro feature: sector_dispersion (lagged)")

        except Exception as e:
            print(f"Warning: Error adding computed macro features: {e}")

    def _add_single_macro(self, df: pd.DataFrame, macro_series: pd.Series, name: str) -> pd.DataFrame:
        """Add a single macro feature to DataFrame."""
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        aligned_values = macro_series.reindex(main_datetimes).values
        col_name = f"{name}_lag{self.macro_lag}"

        if has_multi_columns:
            df[('feature', col_name)] = aligned_values
        else:
            df[col_name] = aligned_values

        return df

    def _merge_lagged_macro(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Merge lagged macro features into DataFrame."""
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        macro_data = {}
        for col in cols:
            macro_series = self._macro_df[col].shift(self.macro_lag)
            aligned_values = macro_series.reindex(main_datetimes).values

            new_name = f"{col}_lag{self.macro_lag}"
            if has_multi_columns:
                macro_data[('feature', new_name)] = aligned_values
            else:
                macro_data[new_name] = aligned_values

        macro_df = pd.DataFrame(macro_data, index=df.index)
        merged = pd.concat([df, macro_df], axis=1, copy=False)
        return merged.copy()

    def _load_macro_features(self) -> Optional[pd.DataFrame]:
        """Load macro features from parquet file."""
        if not self.macro_data_path.exists():
            print(f"Warning: Macro features file not found: {self.macro_data_path}")
            print("Will use stock-specific features only")
            return None

        try:
            df = pd.read_parquet(self.macro_data_path)
            print(f"Loaded macro data: {df.shape}, range: {df.index.min()} to {df.index.max()}")
            return df
        except Exception as e:
            print(f"Warning: Failed to load macro features: {e}")
            return None

    def get_label_config(self):
        """Return N-day return label."""
        label_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [label_expr], ["LABEL0"]
