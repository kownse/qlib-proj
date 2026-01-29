"""
CatBoost DataHandler V1 - Optimal Features from Nested CV Forward Selection.

Design Philosophy:
- Features selected via CatBoost nested CV forward selection
- Started from importance-based baseline (10 features)
- Added features that improved IC via forward selection
- 14 total features: 8 stock + 6 macro

Selection Method:
- Nested CV forward selection (inner 4-fold CV 2021-2024)
- Baseline IC: 0.0206 -> Final IC: 0.0392
- Total IC improvement: +0.0186

Features Summary:
- Stock features: 8
- Macro features: 6
- Total: 14 features

Selection History:
- Round 0: Baseline (6 stock + 4 macro) IC=0.0206
- Round 1: +macro_uso_pct_5d (+0.0068 IC)
- Round 2: +PCT_FROM_52W_LOW (+0.0067 IC)
- Round 3: +macro_vix_zscore20 (+0.0033 IC)
- Round 4: +RSQR30 (+0.0018 IC)
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


class Alpha158_CatBoost_V1(DataHandlerLP):
    """
    CatBoost V1 - Forward Selection Optimal Features.

    Stock Features (8):
    === Volatility (4) ===
    - TALIB_ATR14: Average True Range (14-day)
    - TALIB_NATR14: Normalized ATR (14-day)
    - STD60: 60-day price standard deviation
    - BETA60: 60-day price slope (trend strength)

    === Momentum (1) ===
    - ROC60: 60-day rate of change

    === Price Position (2) ===
    - MAX5: 5-day high relative to close
    - PCT_FROM_52W_LOW: Distance from 52-week low

    === Statistical (1) ===
    - RSQR30: 30-day R-squared (trend consistency)

    Macro Features (6, all 1-day lagged):
    === Credit/Risk (3) ===
    - macro_hy_spread_zscore: High-yield spread z-score
    - macro_hy_spread: High-yield spread level
    - macro_credit_risk: Credit risk indicator

    === Bonds (1) ===
    - macro_tlt_pct_20d: 20-day TLT return

    === Commodities (1) ===
    - macro_uso_pct_5d: 5-day oil return

    === VIX (1) ===
    - macro_vix_zscore20: VIX 20-day z-score

    Total: 14 features
    Final CV IC: 0.0392
    """

    MACRO_FEATURES = [
        # Credit/Risk (3)
        "macro_hy_spread_zscore",
        "macro_hy_spread",
        "macro_credit_risk",
        # Bonds (1)
        "macro_tlt_pct_20d",
        # Commodities (1)
        "macro_uso_pct_5d",
        # VIX (1)
        "macro_vix_zscore20",
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
        """Get feature config - 8 stock features."""
        fields = []
        names = []

        # === Volatility (4) ===
        fields.append("TALIB_ATR($high, $low, $close, 14)/$close")
        names.append("TALIB_ATR14")

        fields.append("TALIB_NATR($high, $low, $close, 14)")
        names.append("TALIB_NATR14")

        fields.append("Std($close, 60)/$close")
        names.append("STD60")

        fields.append("Slope($close, 60)/Std(Mean($close, 60), 60)")
        names.append("BETA60")

        # === Momentum (1) ===
        fields.append("Ref($close, 60)/$close")
        names.append("ROC60")

        # === Price Position (2) ===
        fields.append("Max($high, 5)/$close")
        names.append("MAX5")

        fields.append("($close - Min($low, 252)) / (Min($low, 252) + 1e-12)")
        names.append("PCT_FROM_52W_LOW")

        # === Statistical (1) ===
        fields.append("Rsquare($close, 30)")
        names.append("RSQR30")

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
                print(f"Added {len(available_cols)} macro features (lag={self.macro_lag}d)")

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

            # 直接使用原始列名（与特征选择脚本保持一致）
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
