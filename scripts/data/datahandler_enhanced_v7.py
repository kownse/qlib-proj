"""
Enhanced DataHandler V7 - Expanded features for higher IC.

Design Philosophy:
- Expand V6's feature set while maintaining generalization
- Add volume-based and cross-sectional features
- Expand macro regime indicators (all lagged)
- Target: ~40 features (vs V6's 25)

V6 Results: Test IC 0.0310 (25 features)
V7 Goal: Test IC > 0.04 with better feature coverage

Features:
- Stock-specific: 28 features (expanded from V6's 19)
- Macro regime (lagged): 12 features (expanded from V6's 6)
- Total: 40 features
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


class Alpha158_Enhanced_V7(DataHandlerLP):
    """
    Enhanced Alpha158 V7 - Expanded features for higher IC.

    Stock-specific features (28):
    ===== From V6 (19) =====
    - Alpha158: MAX60, MAX20, MA60, MA5, STD60, ROC60, QTLU60, RESI20, IMAX60 (9)
    - Volatility: VOL_10D, GK_VOL_20, PARKINSON_VOL_20 (3)
    - TA-Lib: TALIB_ATR14, TALIB_NATR14, TALIB_STOCH_K (3)
    - 52-week: PCT_FROM_52W_HIGH, PCT_FROM_52W_LOW (2)
    - Other: SHARPE_60D, ILLIQUIDITY_20D (2)

    ===== New in V7 (9) =====
    - Volume: VOLUME_RATIO_20, VOLUME_TREND_10, VOLUME_ZSCORE_20 (3)
    - Momentum: ROC20, ROC5, MOMENTUM_QUALITY (3)
    - Mean Reversion: MEAN_REV_20, PRICE_ZSCORE_60 (2)
    - Trend: ADX_TREND (1)

    Macro regime features (12, all 1-day lagged):
    ===== From V6 (6) =====
    - VIX: vix_level, vix_zscore20 (2)
    - Credit: credit_stress, hy_spread_zscore (2)
    - Economy: yield_curve_slope, risk_on_off (2)

    ===== New in V7 (6) =====
    - Sector: sector_dispersion (computed) (1)
    - Volatility regime: spy_vol20, vix_term_structure (2)
    - Cross-asset: stock_bond_corr, gld_momentum (2)
    - Dollar: uup_trend (1)

    Total: 40 features
    """

    # Expanded macro regime features
    MACRO_REGIME_FEATURES = [
        # === From V6 (6) ===
        # VIX regime
        "macro_vix_level",
        "macro_vix_zscore20",
        # Credit regime
        "macro_credit_stress",
        "macro_hy_spread_zscore",
        # Economic regime
        "macro_yield_curve_slope",
        "macro_risk_on_off",

        # === New in V7 (5) ===
        # Volatility regime
        "macro_spy_vol20",
        "macro_vix_term_structure",
        # Cross-asset
        "macro_stock_bond_corr",
        "macro_gld_pct_20d",  # Gold momentum as safe-haven indicator
        # Dollar strength
        "macro_uup_pct_20d",  # Dollar trend
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
        """Get feature config with V7 expanded features."""
        fields = []
        names = []

        # 1. Alpha158 features (9) - from V6
        f, n = self._get_alpha158_features()
        fields.extend(f)
        names.extend(n)

        # 2. Volatility features (3) - from V6
        f, n = self._get_volatility_features()
        fields.extend(f)
        names.extend(n)

        # 3. TA-Lib features (3) - from V6
        f, n = self._get_talib_features()
        fields.extend(f)
        names.extend(n)

        # 4. 52-week features (2) - from V6
        f, n = self._get_52w_features()
        fields.extend(f)
        names.extend(n)

        # 5. Other features (2) - from V6
        f, n = self._get_other_features()
        fields.extend(f)
        names.extend(n)

        # 6. NEW: Volume features (3)
        f, n = self._get_volume_features()
        fields.extend(f)
        names.extend(n)

        # 7. NEW: Momentum features (3)
        f, n = self._get_momentum_features()
        fields.extend(f)
        names.extend(n)

        # 8. NEW: Mean reversion features (2)
        f, n = self._get_mean_reversion_features()
        fields.extend(f)
        names.extend(n)

        # 9. NEW: Trend features (1)
        f, n = self._get_trend_features()
        fields.extend(f)
        names.extend(n)

        return fields, names

    # ========== V6 Features (19) ==========

    def _get_alpha158_features(self):
        """Alpha158 features (9)."""
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
        """Volatility features (3)."""
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
        """TA-Lib features (3)."""
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
        """52-week features (2)."""
        fields = []
        names = []

        fields.append("($close - Max($high, 252)) / (Max($high, 252) + 1e-12)")
        names.append("PCT_FROM_52W_HIGH")
        fields.append("($close - Min($low, 252)) / (Min($low, 252) + 1e-12)")
        names.append("PCT_FROM_52W_LOW")

        return fields, names

    def _get_other_features(self):
        """Other features (2)."""
        fields = []
        names = []

        fields.append("Mean($close/Ref($close,1)-1, 60) / (Std($close/Ref($close,1)-1, 60) + 1e-12)")
        names.append("SHARPE_60D")
        fields.append("Mean(Abs($close/Ref($close,1)-1) / (Log($volume+1)+1e-12), 20)")
        names.append("ILLIQUIDITY_20D")

        return fields, names

    # ========== V7 New Features (9) ==========

    def _get_volume_features(self):
        """Volume-based features (3) - NEW in V7."""
        fields = []
        names = []

        # Volume ratio: current vs 20-day average
        fields.append("$volume / (Mean($volume, 20) + 1e-12)")
        names.append("VOLUME_RATIO_20")

        # Volume trend: 10-day volume momentum
        fields.append("Mean($volume, 5) / (Mean($volume, 20) + 1e-12)")
        names.append("VOLUME_TREND_10")

        # Volume z-score: normalized volume
        fields.append("($volume - Mean($volume, 20)) / (Std($volume, 20) + 1e-12)")
        names.append("VOLUME_ZSCORE_20")

        return fields, names

    def _get_momentum_features(self):
        """Momentum features (3) - NEW in V7."""
        fields = []
        names = []

        # 20-day momentum
        fields.append("Ref($close, 20)/$close")
        names.append("ROC20")

        # 5-day momentum
        fields.append("Ref($close, 5)/$close")
        names.append("ROC5")

        # Momentum quality: momentum adjusted for volatility
        fields.append("($close/Ref($close, 20) - 1) / (Std($close/Ref($close,1)-1, 20) + 1e-12)")
        names.append("MOMENTUM_QUALITY")

        return fields, names

    def _get_mean_reversion_features(self):
        """Mean reversion features (2) - NEW in V7."""
        fields = []
        names = []

        # Mean reversion signal: distance from 20-day MA
        fields.append("($close - Mean($close, 20)) / (Mean($close, 20) + 1e-12)")
        names.append("MEAN_REV_20")

        # Price z-score: standardized price position
        fields.append("($close - Mean($close, 60)) / (Std($close, 60) + 1e-12)")
        names.append("PRICE_ZSCORE_60")

        return fields, names

    def _get_trend_features(self):
        """Trend features (1) - NEW in V7."""
        fields = []
        names = []

        # ADX-like trend strength (simplified)
        fields.append("TALIB_ADX($high, $low, $close, 14) / 100")
        names.append("ADX_TREND")

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
        """Add computed macro features like sector dispersion."""
        try:
            # Sector dispersion - very important feature from V4
            sector_pct_cols = [c for c in self._macro_df.columns
                             if c.endswith('_pct_20d') and c.startswith('macro_xl')]

            if len(sector_pct_cols) >= 5:
                sector_df = self._macro_df[sector_pct_cols]
                sector_dispersion = sector_df.std(axis=1).shift(self.macro_lag)  # Lagged!

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
