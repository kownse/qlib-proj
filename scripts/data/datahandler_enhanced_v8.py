"""
Enhanced DataHandler V8 - Data-driven by AE-MLP Permutation Importance.

Design Philosophy:
- Features selected based on AE-MLP permutation importance (NOT CatBoost!)
- Keep only features with positive importance (help the model)
- Remove all features with negative importance (hurt the model)

AE-MLP Permutation Importance Analysis (V7 model, test set):
- Baseline IC: 0.0396
- 22 features with positive importance (KEEP)
- 17 features with negative importance (REMOVE)

Key Insights (AE-MLP vs CatBoost differences):
- PCT_FROM_52W_LOW: AE-MLP +0.0097 vs CatBoost 0 (CatBoost missed this!)
- MOMENTUM_QUALITY: AE-MLP +0.0093 vs CatBoost 0.07
- macro_spy_vol20: AE-MLP -0.0056 vs CatBoost 9.36 (CatBoost wrong!)
- macro_gld_pct_20d: AE-MLP -0.0037 vs CatBoost 11.54 (CatBoost wrong!)
- MA5: AE-MLP -0.0013 vs CatBoost 2.08 (CatBoost wrong!)

Features (22 total):
- Stock-specific: 16 features (only positive importance)
- Macro regime (lagged): 6 features (only positive importance)
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


class Alpha158_Enhanced_V8(DataHandlerLP):
    """
    Enhanced Alpha158 V8 - Data-driven by AE-MLP Permutation Importance.

    Stock-specific features (16) - ordered by AE-MLP importance:
    ===== High Importance (>0.005) =====
    - PCT_FROM_52W_LOW (+0.0097) - #1 most important!
    - MOMENTUM_QUALITY (+0.0093) - #2
    - PCT_FROM_52W_HIGH (+0.0074) - #3

    ===== Medium Importance (0.001-0.005) =====
    - ROC60 (+0.0023)
    - PRICE_ZSCORE_60 (+0.0022)
    - GK_VOL_20 (+0.0017)
    - STD60 (+0.0014)
    - ADX_TREND (+0.0013)

    ===== Low Importance (0-0.001) =====
    - PARKINSON_VOL_20 (+0.0006)
    - VOLUME_RATIO_20 (+0.0006)
    - ROC20 (+0.0005)
    - MAX60 (+0.0005)
    - IMAX60 (+0.0003)
    - MA60 (+0.0003)
    - SHARPE_60D (+0.00005)
    - ILLIQUIDITY_20D (+0.00001)

    Macro regime features (6, all 1-day lagged) - ordered by importance:
    - macro_credit_stress (+0.0065)
    - macro_hy_spread_zscore (+0.0063)
    - macro_sector_dispersion (+0.0053) - computed
    - macro_vix_zscore20 (+0.0049)
    - macro_vix_term_structure (+0.0009)
    - macro_stock_bond_corr (+0.0001)

    REMOVED (17 features with NEGATIVE importance):
    - macro_spy_vol20 (-0.0056), macro_gld_pct_20d (-0.0037)
    - macro_vix_level (-0.0015), MA5 (-0.0013), ROC5 (-0.0011)
    - VOL_10D (-0.0008), MAX20 (-0.0008), macro_risk_on_off (-0.0007)
    - MEAN_REV_20 (-0.0006), VOLUME_ZSCORE_20 (-0.0006)
    - TALIB_STOCH_K (-0.0006), TALIB_ATR14 (-0.0006)
    - TALIB_NATR14 (-0.0005), QTLU60 (-0.0004)
    - macro_yield_curve_slope (-0.0003), RESI20 (-0.0001)
    - VOLUME_TREND_10 (-0.00008)

    Total: 22 features
    """

    # Macro features with POSITIVE importance only
    MACRO_REGIME_FEATURES = [
        # === Positive Importance ===
        "macro_credit_stress",        # +0.0065
        "macro_hy_spread_zscore",     # +0.0063
        # sector_dispersion computed separately (+0.0053)
        "macro_vix_zscore20",         # +0.0049
        "macro_vix_term_structure",   # +0.0009
        "macro_stock_bond_corr",      # +0.0001
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
        """Get feature config based on AE-MLP permutation importance."""
        fields = []
        names = []

        # 1. 52-week features (TOP importance: +0.0097, +0.0074)
        f, n = self._get_52w_features()
        fields.extend(f)
        names.extend(n)

        # 2. Momentum features (+0.0093, +0.0023, +0.0005)
        f, n = self._get_momentum_features()
        fields.extend(f)
        names.extend(n)

        # 3. Volatility/Std features (+0.0017, +0.0014, +0.0006)
        f, n = self._get_volatility_features()
        fields.extend(f)
        names.extend(n)

        # 4. Price structure features (+0.0022, +0.0013)
        f, n = self._get_price_structure_features()
        fields.extend(f)
        names.extend(n)

        # 5. Alpha158 core features (+0.0005, +0.0003, +0.0003)
        f, n = self._get_alpha158_features()
        fields.extend(f)
        names.extend(n)

        # 6. Volume feature (+0.0006)
        f, n = self._get_volume_features()
        fields.extend(f)
        names.extend(n)

        # 7. Other features (+0.00005, +0.00001)
        f, n = self._get_other_features()
        fields.extend(f)
        names.extend(n)

        return fields, names

    # ========== 52-week Features (2) - TOP IMPORTANCE ==========

    def _get_52w_features(self):
        """52-week features - highest importance in AE-MLP!"""
        fields = []
        names = []

        # PCT_FROM_52W_LOW - importance: +0.0097 (#1!)
        fields.append("($close - Min($low, 252)) / (Min($low, 252) + 1e-12)")
        names.append("PCT_FROM_52W_LOW")

        # PCT_FROM_52W_HIGH - importance: +0.0074 (#3)
        fields.append("($close - Max($high, 252)) / (Max($high, 252) + 1e-12)")
        names.append("PCT_FROM_52W_HIGH")

        return fields, names

    # ========== Momentum Features (3) ==========

    def _get_momentum_features(self):
        """Momentum features with positive importance."""
        fields = []
        names = []

        # MOMENTUM_QUALITY - importance: +0.0093 (#2!)
        fields.append("($close/Ref($close, 20) - 1) / (Std($close/Ref($close,1)-1, 20) + 1e-12)")
        names.append("MOMENTUM_QUALITY")

        # ROC60 - importance: +0.0023
        fields.append("Ref($close, 60)/$close")
        names.append("ROC60")

        # ROC20 - importance: +0.0005
        fields.append("Ref($close, 20)/$close")
        names.append("ROC20")

        # REMOVED: ROC5 (-0.0011)

        return fields, names

    # ========== Volatility Features (3) ==========

    def _get_volatility_features(self):
        """Volatility features with positive importance."""
        fields = []
        names = []

        # GK_VOL_20 - importance: +0.0017
        fields.append("Power(Mean(0.5*Power(Log($high/$low+1e-12), 2) - 0.386*Power(Log($close/$open+1e-12), 2), 20), 0.5)")
        names.append("GK_VOL_20")

        # STD60 - importance: +0.0014
        fields.append("Std($close, 60)/$close")
        names.append("STD60")

        # PARKINSON_VOL_20 - importance: +0.0006
        fields.append("Power(Mean(Power(Log($high/$low+1e-12), 2), 20) / 0.6931, 0.5)")
        names.append("PARKINSON_VOL_20")

        # REMOVED: VOL_10D (-0.0008)

        return fields, names

    # ========== Price Structure Features (2) ==========

    def _get_price_structure_features(self):
        """Price structure features with positive importance."""
        fields = []
        names = []

        # PRICE_ZSCORE_60 - importance: +0.0022
        fields.append("($close - Mean($close, 60)) / (Std($close, 60) + 1e-12)")
        names.append("PRICE_ZSCORE_60")

        # ADX_TREND - importance: +0.0013
        fields.append("TALIB_ADX($high, $low, $close, 14) / 100")
        names.append("ADX_TREND")

        # REMOVED: MEAN_REV_20 (-0.0006)

        return fields, names

    # ========== Alpha158 Core Features (3) ==========

    def _get_alpha158_features(self):
        """Alpha158 core features with positive importance."""
        fields = []
        names = []

        # MAX60 - importance: +0.0005
        fields.append("Max($high, 60)/$close")
        names.append("MAX60")

        # IMAX60 - importance: +0.0003
        fields.append("IdxMax($high, 60)/60")
        names.append("IMAX60")

        # MA60 - importance: +0.0003
        fields.append("Mean($close, 60)/$close")
        names.append("MA60")

        # REMOVED: MAX20 (-0.0008), QTLU60 (-0.0004), RESI20 (-0.0001), MA5 (-0.0013)

        return fields, names

    # ========== Volume Features (1) ==========

    def _get_volume_features(self):
        """Volume features with positive importance."""
        fields = []
        names = []

        # VOLUME_RATIO_20 - importance: +0.0006
        fields.append("$volume / (Mean($volume, 20) + 1e-12)")
        names.append("VOLUME_RATIO_20")

        # REMOVED: VOLUME_ZSCORE_20 (-0.0006), VOLUME_TREND_10 (-0.00008)

        return fields, names

    # ========== Other Features (2) ==========

    def _get_other_features(self):
        """Other features with positive importance."""
        fields = []
        names = []

        # SHARPE_60D - importance: +0.00005
        fields.append("Mean($close/Ref($close,1)-1, 60) / (Std($close/Ref($close,1)-1, 60) + 1e-12)")
        names.append("SHARPE_60D")

        # ILLIQUIDITY_20D - importance: +0.00001
        fields.append("Mean(Abs($close/Ref($close,1)-1) / (Log($volume+1)+1e-12), 20)")
        names.append("ILLIQUIDITY_20D")

        # REMOVED: TALIB_NATR14 (-0.0005), TALIB_ATR14 (-0.0006), TALIB_STOCH_K (-0.0006)

        return fields, names

    # ========== Macro Features ==========

    def process_data(self, with_fit: bool = False):
        """Override to add lagged macro regime features."""
        super().process_data(with_fit=with_fit)

        if self._macro_df is not None:
            self._add_lagged_macro_features()
            self._add_computed_macro_features()

    def _add_lagged_macro_features(self):
        """Add lagged macro regime indicators with positive importance only."""
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
        """Add sector dispersion - importance: +0.0053 (#6)."""
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
