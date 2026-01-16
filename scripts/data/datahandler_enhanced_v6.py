"""
Enhanced DataHandler V6 - Balanced version with selective lagged macro features.

Design Philosophy:
- Keep all 19 stock-specific features from V5 (proven to work)
- Add only 6 carefully selected macro REGIME indicators (not momentum)
- All macro features are 1-day LAGGED to avoid look-ahead bias
- Focus on regime indicators that should generalize across time periods

V5 Results: Valid IC 0.0603, Test IC 0.0215 (19 features, no macro)
V6 Goal: Improve IC while maintaining generalization

Feature Selection Criteria for Macro:
- Regime indicators (VIX level, credit stress) > Momentum indicators (pct_5d, pct_20d)
- Lagged features to ensure no data leakage
- Cross-sectionally useful (helps differentiate stocks in different regimes)

Features (25 total):
- Stock-specific: 19 features (from V5)
- Macro regime (lagged): 6 features
"""

import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd

script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from qlib.data.dataset.handler import DataHandlerLP

PROJECT_ROOT = script_dir.parent
DEFAULT_MACRO_PATH = PROJECT_ROOT / "my_data" / "macro_processed" / "macro_features.parquet"


class Alpha158_Enhanced_V6(DataHandlerLP):
    """
    Enhanced Alpha158 V6 - Balanced stock + selective lagged macro features.

    Stock-specific features (19):
    - Alpha158: MAX60, MAX20, MA60, MA5, STD60, ROC60, QTLU60, RESI20, IMAX60 (9)
    - Volatility: VOL_10D, GK_VOL_20, PARKINSON_VOL_20 (3)
    - TA-Lib: TALIB_ATR14, TALIB_NATR14, TALIB_STOCH_K (3)
    - 52-week: PCT_FROM_52W_HIGH, PCT_FROM_52W_LOW (2)
    - Other: SHARPE_60D, ILLIQUIDITY_20D (2)

    Macro regime features (6, all 1-day lagged):
    - VIX regime: vix_level, vix_zscore20 (market fear)
    - Credit regime: credit_stress, hy_spread_zscore (credit risk)
    - Economic regime: yield_curve_slope, risk_on_off (macro environment)

    Total: 25 features
    """

    # Selected macro regime indicators (NOT momentum features)
    # These should be more stable and generalizable
    MACRO_REGIME_FEATURES = [
        # VIX regime (2)
        "macro_vix_level",        # Absolute fear level
        "macro_vix_zscore20",     # Normalized fear (helps compare across periods)

        # Credit regime (2)
        "macro_credit_stress",    # Overall credit stress indicator
        "macro_hy_spread_zscore", # Normalized high-yield spread

        # Economic regime (2)
        "macro_yield_curve_slope", # Yield curve (recession indicator)
        "macro_risk_on_off",       # Risk appetite indicator
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
        macro_lag: int = 1,  # Days to lag macro features
        macro_data_path: Union[str, Path] = None,
        **kwargs,
    ):
        """
        Initialize Enhanced V6 DataHandler.

        Args:
            macro_lag: Number of days to lag macro features (default: 1)
        """
        self.volatility_window = volatility_window
        self.macro_lag = macro_lag
        self.macro_data_path = Path(macro_data_path) if macro_data_path else DEFAULT_MACRO_PATH

        # Load macro data
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
        """Get feature config - same as V5 stock-specific features."""
        fields = []
        names = []

        # 1. Alpha158 features (9)
        alpha_fields, alpha_names = self._get_alpha158_features()
        fields.extend(alpha_fields)
        names.extend(alpha_names)

        # 2. Volatility features (3)
        vol_fields, vol_names = self._get_volatility_features()
        fields.extend(vol_fields)
        names.extend(vol_names)

        # 3. TA-Lib features (3)
        talib_fields, talib_names = self._get_talib_features()
        fields.extend(talib_fields)
        names.extend(talib_names)

        # 4. 52-week features (2)
        week52_fields, week52_names = self._get_52w_features()
        fields.extend(week52_fields)
        names.extend(week52_names)

        # 5. Other features (2)
        other_fields, other_names = self._get_other_features()
        fields.extend(other_fields)
        names.extend(other_names)

        return fields, names

    def _get_alpha158_features(self):
        """Alpha158 features (9 total)."""
        fields = []
        names = []

        fields.append("Max($high, 20)/$close")
        names.append("MAX20")
        fields.append("Max($high, 60)/$close")
        names.append("MAX60")
        fields.append("Quantile($close, 60, 0.8)/$close")
        names.append("QTLU60")
        fields.append("Mean($close, 5)/$close")
        names.append("MA5")
        fields.append("Mean($close, 60)/$close")
        names.append("MA60")
        fields.append("Std($close, 60)/$close")
        names.append("STD60")
        fields.append("Ref($close, 60)/$close")
        names.append("ROC60")
        fields.append("Resi($close, 20)/$close")
        names.append("RESI20")
        fields.append("IdxMax($high, 60)/60")
        names.append("IMAX60")

        return fields, names

    def _get_volatility_features(self):
        """Volatility features (3 total)."""
        fields = []
        names = []

        fields.append("Power(Mean(0.5*Power(Log($high/$low+1e-12), 2) - 0.386*Power(Log($close/$open+1e-12), 2), 20), 0.5)")
        names.append("GK_VOL_20")
        fields.append("Power(Mean(Power(Log($high/$low+1e-12), 2), 20) / 0.6931, 0.5)")
        names.append("PARKINSON_VOL_20")
        fields.append("Std($close/Ref($close,1)-1, 10)")
        names.append("VOL_10D")

        return fields, names

    def _get_talib_features(self):
        """TA-Lib features (3 total)."""
        fields = []
        names = []

        fields.append("TALIB_NATR($high, $low, $close, 14)")
        names.append("TALIB_NATR14")
        fields.append("TALIB_ATR($high, $low, $close, 14)/$close")
        names.append("TALIB_ATR14")
        fields.append("TALIB_STOCH_K($high, $low, $close, 5, 3, 3)")
        names.append("TALIB_STOCH_K")

        return fields, names

    def _get_52w_features(self):
        """52-week features (2 total)."""
        fields = []
        names = []

        fields.append("($close - Max($high, 252)) / (Max($high, 252) + 1e-12)")
        names.append("PCT_FROM_52W_HIGH")
        fields.append("($close - Min($low, 252)) / (Min($low, 252) + 1e-12)")
        names.append("PCT_FROM_52W_LOW")

        return fields, names

    def _get_other_features(self):
        """Other features (2 total)."""
        fields = []
        names = []

        fields.append("Mean($close/Ref($close,1)-1, 60) / (Std($close/Ref($close,1)-1, 60) + 1e-12)")
        names.append("SHARPE_60D")
        fields.append("Mean(Abs($close/Ref($close,1)-1) / (Log($volume+1)+1e-12), 20)")
        names.append("ILLIQUIDITY_20D")

        return fields, names

    def process_data(self, with_fit: bool = False):
        """Override to add lagged macro regime features."""
        super().process_data(with_fit=with_fit)

        if self._macro_df is not None:
            self._add_lagged_macro_features()

    def _add_lagged_macro_features(self):
        """Add lagged macro regime indicators."""
        try:
            available_cols = [c for c in self.MACRO_REGIME_FEATURES if c in self._macro_df.columns]

            if not available_cols:
                print("Warning: No macro regime features available")
                return

            # Add to _learn
            if hasattr(self, "_learn") and self._learn is not None:
                self._learn = self._merge_lagged_macro(self._learn, available_cols)
                print(f"Added {len(available_cols)} lagged macro regime features (lag={self.macro_lag}d)")

            # Add to _infer
            if hasattr(self, "_infer") and self._infer is not None:
                self._infer = self._merge_lagged_macro(self._infer, available_cols)

        except Exception as e:
            print(f"Warning: Error adding macro features: {e}")

    def _merge_lagged_macro(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Merge lagged macro features into DataFrame."""
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        macro_data = {}
        for col in cols:
            # Apply lag to avoid look-ahead bias
            macro_series = self._macro_df[col].shift(self.macro_lag)
            aligned_values = macro_series.reindex(main_datetimes).values

            # Use descriptive name indicating the lag
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
