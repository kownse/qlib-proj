"""
Enhanced DataHandler V11 - Optimal Features from Nested CV Forward Selection.

Design Philosophy:
- Started from V10's protected features as baseline
- Added features via forward selection that improved IC
- 38 total features: 26 stock + 12 macro

Selection Method:
- Nested CV forward selection (inner 4-fold CV 2021-2024)
- Baseline IC: 0.0244 -> Final IC: 0.0339
- Added macro_risk_on_off (+0.0083 IC) and macro_credit_stress (+0.0013 IC)

Features Summary:
- Stock features: 26 (same as V10)
- Macro features: 12 (V10's 10 + risk_on_off + credit_stress)

Key Additions vs V10:
- macro_risk_on_off: Risk-on/off regime indicator (+0.0083 IC gain)
- macro_credit_stress: Credit stress indicator (+0.0013 IC gain)
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


class Alpha158_Enhanced_V11(DataHandlerLP):
    """
    Enhanced Alpha158 V11 - Forward Selection Optimal Features.

    Stock Features (26, same as V10):
    === Momentum (4) ===
    - MOMENTUM_QUALITY, ROC5, ROC20, TALIB_RSI14

    === Volatility (4) ===
    - TALIB_NATR14, STD20, STD60, TALIB_ATR14

    === Price Position (4) ===
    - PCT_FROM_52W_HIGH, PCT_FROM_52W_LOW, MAX60, MIN60

    === Mean Reversion (4) ===
    - RESI5, RESI10, RESI20, MA_RATIO_5_20

    === Volume (2) ===
    - VOLUME_RATIO_5_20, VWAP_BIAS

    === Trend (2) ===
    - SLOPE20, SLOPE60

    === Drawdown (2) ===
    - MAX_DRAWDOWN_20, MAX_DRAWDOWN_60

    === Technical (4) ===
    - TALIB_ADX14, TALIB_WILLR14, TALIB_CCI14, TALIB_MFI14

    Macro Features (12, all 1-day lagged):
    === VIX (3) ===
    - macro_vix_zscore20, macro_vix_regime, macro_vix_pct_5d

    === Credit (3) ===
    - macro_hy_spread_zscore, macro_hyg_pct_5d, macro_credit_stress (NEW)

    === Yield/Bonds (1) ===
    - macro_yield_curve

    === Commodities (2) ===
    - macro_gld_pct_20d, macro_uso_pct_20d

    === Dollar (1) ===
    - macro_uup_pct_5d

    === Market (2) ===
    - macro_spy_pct_5d, macro_risk_on_off (NEW)

    Total: 38 features
    """

    MACRO_FEATURES = [
        # VIX (3)
        "macro_vix_zscore20",
        "macro_vix_regime",
        "macro_vix_pct_5d",
        # Credit (3)
        "macro_hy_spread_zscore",
        "macro_hyg_pct_5d",
        "macro_credit_stress",  # NEW: +0.0013 IC
        # Yield/Bonds (1)
        "macro_yield_curve",
        # Commodities (2)
        "macro_gld_pct_20d",
        "macro_uso_pct_20d",
        # Dollar (1)
        "macro_uup_pct_5d",
        # Market (2)
        "macro_spy_pct_5d",
        "macro_risk_on_off",  # NEW: +0.0083 IC
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
        """Get feature config - 26 stock features."""
        fields = []
        names = []

        # === Momentum (4) ===
        fields.append("($close/Ref($close, 20) - 1) / (Std($close/Ref($close,1)-1, 20) + 1e-12)")
        names.append("MOMENTUM_QUALITY")

        fields.append("Ref($close, 5)/$close")
        names.append("ROC5")

        fields.append("Ref($close, 20)/$close")
        names.append("ROC20")

        fields.append("TALIB_RSI($close, 14)")
        names.append("TALIB_RSI14")

        # === Volatility (4) ===
        fields.append("TALIB_NATR($high, $low, $close, 14)")
        names.append("TALIB_NATR14")

        fields.append("Std($close, 20)/$close")
        names.append("STD20")

        fields.append("Std($close, 60)/$close")
        names.append("STD60")

        fields.append("TALIB_ATR($high, $low, $close, 14)/$close")
        names.append("TALIB_ATR14")

        # === Price Position (4) ===
        fields.append("($close - Max($high, 252)) / (Max($high, 252) + 1e-12)")
        names.append("PCT_FROM_52W_HIGH")

        fields.append("($close - Min($low, 252)) / (Min($low, 252) + 1e-12)")
        names.append("PCT_FROM_52W_LOW")

        fields.append("Max($high, 60)/$close")
        names.append("MAX60")

        fields.append("Min($low, 60)/$close")
        names.append("MIN60")

        # === Mean Reversion (4) ===
        fields.append("Resi($close, 5)/$close")
        names.append("RESI5")

        fields.append("Resi($close, 10)/$close")
        names.append("RESI10")

        fields.append("Resi($close, 20)/$close")
        names.append("RESI20")

        fields.append("Mean($close, 5)/Mean($close, 20)")
        names.append("MA_RATIO_5_20")

        # === Volume (2) ===
        fields.append("Mean($volume, 5)/(Mean($volume, 20)+1e-12)")
        names.append("VOLUME_RATIO_5_20")

        fields.append("($close*$volume)/Sum($volume, 20) - Mean($close, 20)")
        names.append("VWAP_BIAS")

        # === Trend (2) ===
        fields.append("Slope($close, 20)/$close")
        names.append("SLOPE20")

        fields.append("Slope($close, 60)/$close")
        names.append("SLOPE60")

        # === Drawdown (2) ===
        fields.append("(Min($close, 20) - Max($high, 20)) / (Max($high, 20) + 1e-12)")
        names.append("MAX_DRAWDOWN_20")

        fields.append("(Min($close, 60) - Max($high, 60)) / (Max($high, 60) + 1e-12)")
        names.append("MAX_DRAWDOWN_60")

        # === Technical (4) ===
        fields.append("TALIB_ADX($high, $low, $close, 14)")
        names.append("TALIB_ADX14")

        fields.append("TALIB_WILLR($high, $low, $close, 14)")
        names.append("TALIB_WILLR14")

        fields.append("TALIB_CCI($high, $low, $close, 14)")
        names.append("TALIB_CCI14")

        fields.append("TALIB_MFI($high, $low, $close, $volume, 14)")
        names.append("TALIB_MFI14")

        return fields, names

    def process_data(self, with_fit: bool = False):
        """Override to add lagged macro features."""
        super().process_data(with_fit=with_fit)

        if self._macro_df is not None:
            self._add_lagged_macro_features()

    def _add_lagged_macro_features(self):
        """Add lagged macro features."""
        try:
            available_cols = [c for c in self.MACRO_FEATURES if c in self._macro_df.columns]

            if not available_cols:
                print("Warning: No macro features available")
                return

            if hasattr(self, "_learn") and self._learn is not None:
                self._learn = self._merge_lagged_macro(self._learn, available_cols)
                print(f"Added {len(available_cols)} lagged macro features (lag={self.macro_lag}d)")

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
