"""
LightGBM Feature-Selected DataHandler V1

Design Philosophy:
- Features selected via nested CV forward selection with LightGBM
- Only features that improve CV IC are included

Forward Selection Results (20260129_172823):
- Baseline IC: 0.0059
- Final IC: 0.0346
- Method: nested_cv_lightgbm_forward_selection

Selected Features (12 total):
- Stock-specific: 2 features (RSV5, TALIB_ATR14)
- Macro regime (lagged): 10 features

Selection History:
1. Round 0 (BASELINE): IC = 0.0059
2. Round 1 (ADD RSV5): IC = 0.0288 (+0.0230)
3. Round 2 (ADD macro_uso_pct_5d): IC = 0.0346 (+0.0058)
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


class Alpha158_LightGBM_V1(DataHandlerLP):
    """
    LightGBM Feature-Selected DataHandler V1

    Stock-specific features (2):
    - RSV5: Stochastic oscillator variant (5-day)
    - TALIB_ATR14: Average True Range normalized by close

    Macro regime features (10, all 1-day lagged):
    - macro_vix_term_structure: VIX term structure
    - macro_gld_vol20: Gold 20-day volatility
    - macro_vix_level: VIX level
    - macro_xly_pct_20d: Consumer discretionary 20-day return
    - macro_uso_pct_20d: Oil 20-day return
    - macro_hy_spread: High yield spread
    - macro_gld_pct_20d: Gold 20-day return
    - macro_vix_term_zscore: VIX term structure z-score
    - macro_uup_ma20_ratio: Dollar index MA ratio
    - macro_uso_pct_5d: Oil 5-day return

    Total: 12 features
    """

    # Macro features selected by forward selection
    MACRO_REGIME_FEATURES = [
        "macro_vix_term_structure",
        "macro_gld_vol20",
        "macro_vix_level",
        "macro_xly_pct_20d",
        "macro_uso_pct_20d",
        "macro_hy_spread",
        "macro_gld_pct_20d",
        "macro_vix_term_zscore",
        "macro_uup_ma20_ratio",
        "macro_uso_pct_5d",
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
        """Get feature config based on LightGBM forward selection."""
        fields = []
        names = []

        # Stock-specific features selected by forward selection
        f, n = self._get_stock_features()
        fields.extend(f)
        names.extend(n)

        return fields, names

    def _get_stock_features(self):
        """Stock features selected by forward selection."""
        fields = []
        names = []

        # RSV5 - Selected in Round 1, IC improvement: +0.0230
        # Stochastic oscillator variant: position within 5-day range
        fields.append("($close-Min($low, 5))/(Max($high, 5)-Min($low, 5)+1e-12)")
        names.append("RSV5")

        # TALIB_ATR14 - Part of baseline features
        # Average True Range normalized by close price
        fields.append("TALIB_ATR($high, $low, $close, 14)/$close")
        names.append("TALIB_ATR14")

        return fields, names

    # ========== Macro Features ==========

    def process_data(self, with_fit: bool = False):
        """Override to add lagged macro regime features."""
        super().process_data(with_fit=with_fit)

        if self._macro_df is not None:
            self._add_lagged_macro_features()

    def _add_lagged_macro_features(self):
        """Add lagged macro regime indicators selected by forward selection."""
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

    def _merge_lagged_macro(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Merge lagged macro features into DataFrame.

        Note: 为与 DynamicTabularHandler 保持一致，特征名不添加 _lag 后缀。
        """
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        macro_data = {}
        for col in cols:
            macro_series = self._macro_df[col].shift(self.macro_lag)
            aligned_values = macro_series.reindex(main_datetimes).values

            # 不添加 _lag 后缀，与 DynamicTabularHandler 保持一致
            if has_multi_columns:
                macro_data[('feature', col)] = aligned_values
            else:
                macro_data[col] = aligned_values

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
