"""
DataHandler with macro features for time series prediction

Extends Alpha158_Volatility_TALib_Lite with market regime indicators.
Designed for time series prediction of individual stock returns/volatility.
"""

import sys
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

# Add scripts directory to path for imports
script_dir = Path(__file__).parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset.handler import DataHandlerLP

# Import TA-Lib custom operators
from utils.talib_ops import TALIB_OPS

# Project root
PROJECT_ROOT = script_dir.parent

# Default macro features path
DEFAULT_MACRO_PATH = PROJECT_ROOT / "my_data" / "macro_processed" / "macro_features.parquet"


class Alpha158_Volatility_TALib_Macro(DataHandlerLP):
    """
    Alpha158 features + TA-Lib indicators + Macro features + N-day volatility label

    Extends Alpha158_Volatility_TALib_Lite with macro market regime features:
    - VIX features: level, zscore, changes, regime, term structure
    - Gold features: changes, volatility
    - Bond features: changes, yield curve proxy
    - Dollar features: changes, strength
    - Oil features: changes, volatility
    - Sector features: momentum, relative strength vs SPY
    - Credit features: spreads, stress indicators
    - Global features: EM momentum, global risk
    - Treasury yield features: levels, curve slope, changes
    - Cross-asset features: risk indicators, correlations

    Total features: ~170 (Alpha158+TA-Lib) + ~105 (Macro) = ~275

    Usage:
        handler = Alpha158_Volatility_TALib_Macro(
            volatility_window=2,
            instruments=["AAPL", "MSFT", "NVDA"],
            start_time="2020-01-01",
            end_time="2024-12-31",
            macro_features="all",  # or "core", "vix_only"
        )
    """

    # Sector symbols for dynamic feature generation
    SECTOR_SYMBOLS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLRE", "XLB", "XLC"]

    # Macro feature column names (~105 total)
    ALL_MACRO_FEATURES = [
        # VIX (8)
        "macro_vix_level", "macro_vix_zscore20",
        "macro_vix_pct_1d", "macro_vix_pct_5d", "macro_vix_pct_10d",
        "macro_vix_ma5_ratio", "macro_vix_ma20_ratio", "macro_vix_regime",
        # VIX Term Structure (5)
        "macro_vix_term_structure", "macro_vix_contango", "macro_vix_term_zscore",
        "macro_uvxy_pct_5d", "macro_svxy_pct_5d",
        # Gold (5)
        "macro_gld_pct_1d", "macro_gld_pct_5d", "macro_gld_pct_20d",
        "macro_gld_ma20_ratio", "macro_gld_vol20",
        # Bond (8)
        "macro_tlt_pct_1d", "macro_tlt_pct_5d", "macro_tlt_pct_20d",
        "macro_tlt_ma20_ratio", "macro_yield_curve", "macro_yield_curve_chg5",
        "macro_ief_pct_5d", "macro_bond_vol20",
        # Dollar (4)
        "macro_uup_pct_1d", "macro_uup_pct_5d",
        "macro_uup_ma20_ratio", "macro_uup_strength",
        # Oil (4)
        "macro_uso_pct_1d", "macro_uso_pct_5d", "macro_uso_pct_20d",
        "macro_uso_vol20",
        # Sector ETFs (33 = 11 sectors x 3 features)
        "macro_xlk_pct_5d", "macro_xlk_pct_20d", "macro_xlk_vs_spy",
        "macro_xlf_pct_5d", "macro_xlf_pct_20d", "macro_xlf_vs_spy",
        "macro_xle_pct_5d", "macro_xle_pct_20d", "macro_xle_vs_spy",
        "macro_xlv_pct_5d", "macro_xlv_pct_20d", "macro_xlv_vs_spy",
        "macro_xli_pct_5d", "macro_xli_pct_20d", "macro_xli_vs_spy",
        "macro_xlp_pct_5d", "macro_xlp_pct_20d", "macro_xlp_vs_spy",
        "macro_xly_pct_5d", "macro_xly_pct_20d", "macro_xly_vs_spy",
        "macro_xlu_pct_5d", "macro_xlu_pct_20d", "macro_xlu_vs_spy",
        "macro_xlre_pct_5d", "macro_xlre_pct_20d", "macro_xlre_vs_spy",
        "macro_xlb_pct_5d", "macro_xlb_pct_20d", "macro_xlb_vs_spy",
        "macro_xlc_pct_5d", "macro_xlc_pct_20d", "macro_xlc_vs_spy",
        # Credit/Risk (8)
        "macro_hyg_pct_5d", "macro_hyg_pct_20d", "macro_hyg_vs_lqd",
        "macro_hyg_lqd_chg5", "macro_jnk_vol20", "macro_credit_stress",
        "macro_hyg_tlt_ratio", "macro_hyg_tlt_chg5",
        # Global (8)
        "macro_eem_pct_5d", "macro_eem_pct_20d", "macro_eem_vs_spy",
        "macro_efa_pct_5d", "macro_efa_vs_spy",
        "macro_fxi_pct_5d", "macro_ewj_pct_5d", "macro_global_risk",
        # Benchmark (6)
        "macro_spy_pct_1d", "macro_spy_pct_5d", "macro_spy_pct_20d",
        "macro_spy_vol20", "macro_qqq_vs_spy", "macro_spy_ma20_ratio",
        # Treasury Yields (10)
        "macro_yield_2y", "macro_yield_10y", "macro_yield_30y",
        "macro_yield_2s10s", "macro_yield_3m10y",
        "macro_yield_10y_chg5", "macro_yield_10y_chg20",
        "macro_yield_curve_slope", "macro_yield_curve_zscore", "macro_yield_inversion",
        # FRED Credit Spreads (5)
        "macro_hy_spread", "macro_hy_spread_zscore", "macro_hy_spread_chg5",
        "macro_ig_spread", "macro_credit_risk",
        # Cross-asset (5)
        "macro_risk_on_off", "macro_gold_oil_ratio", "macro_gold_oil_ratio_chg",
        "macro_stock_bond_corr", "macro_market_stress",
    ]

    # Core features subset for lightweight usage (~25)
    CORE_MACRO_FEATURES = [
        # VIX
        "macro_vix_level", "macro_vix_zscore20", "macro_vix_pct_5d", "macro_vix_regime",
        "macro_vix_term_structure",
        # Macro
        "macro_gld_pct_5d", "macro_tlt_pct_5d", "macro_yield_curve",
        "macro_uup_pct_5d", "macro_uso_pct_5d",
        # Benchmark
        "macro_spy_pct_5d", "macro_spy_vol20",
        # Credit
        "macro_hyg_vs_lqd", "macro_credit_stress",
        # Global
        "macro_eem_vs_spy", "macro_global_risk",
        # Treasury
        "macro_yield_10y", "macro_yield_2s10s", "macro_yield_inversion",
        # FRED Credit
        "macro_hy_spread", "macro_hy_spread_zscore",
        # Cross-asset
        "macro_risk_on_off", "macro_market_stress",
    ]

    # VIX only features (~13)
    VIX_ONLY_FEATURES = [
        "macro_vix_level", "macro_vix_zscore20",
        "macro_vix_pct_1d", "macro_vix_pct_5d", "macro_vix_pct_10d",
        "macro_vix_ma5_ratio", "macro_vix_ma20_ratio", "macro_vix_regime",
        "macro_vix_term_structure", "macro_vix_contango", "macro_vix_term_zscore",
        "macro_uvxy_pct_5d", "macro_svxy_pct_5d",
    ]

    def __init__(
        self,
        volatility_window: int = 2,
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
        # Macro feature parameters
        macro_data_path: Union[str, Path] = None,
        macro_features: str = "all",  # "all", "core", "vix_only", "none"
        **kwargs,
    ):
        """
        Initialize DataHandler with macro features.

        Args:
            volatility_window: Prediction window (days)
            macro_data_path: Path to macro features parquet file
            macro_features: Macro feature set to use
                - "all": All macro features (~105)
                - "core": Core features (~25)
                - "vix_only": VIX features only (~13)
                - "none": No macro features (same as TALib_Lite)
            **kwargs: Additional arguments for parent class
        """
        self.volatility_window = volatility_window
        self.macro_data_path = Path(macro_data_path) if macro_data_path else DEFAULT_MACRO_PATH
        self.macro_features = macro_features

        # Load macro features
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

    def process_data(self, with_fit: bool = False):
        """
        Override process_data to add macro features AFTER processors run.

        This ensures macro features are included in _learn and _infer.
        """
        # First, call parent's process_data
        super().process_data(with_fit=with_fit)

        # Then add macro features to _learn and _infer
        if self._macro_df is not None and self.macro_features != "none":
            self._add_macro_to_processed_data()

    def _add_macro_to_processed_data(self):
        """Add macro features to _learn and _infer after processors run."""
        try:
            macro_cols = self._get_macro_feature_columns()
            available_cols = [c for c in macro_cols if c in self._macro_df.columns]

            if not available_cols:
                return

            # Add to _learn
            if hasattr(self, "_learn") and self._learn is not None:
                self._learn = self._merge_macro_to_df(self._learn, available_cols)
                print(f"Added {len(available_cols)} macro features to learn data")

            # Add to _infer
            if hasattr(self, "_infer") and self._infer is not None:
                self._infer = self._merge_macro_to_df(self._infer, available_cols)

        except Exception as e:
            print(f"Warning: Error adding macro features: {e}")

    def _merge_macro_to_df(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Merge macro features into a DataFrame using pd.concat to avoid fragmentation."""
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        # Build all macro columns at once to avoid DataFrame fragmentation
        macro_data = {}
        for col in cols:
            macro_series = self._macro_df[col]
            aligned_values = macro_series.reindex(main_datetimes).values

            if has_multi_columns:
                macro_data[('feature', col)] = aligned_values
            else:
                macro_data[col] = aligned_values

        # Create DataFrame with all macro columns
        macro_df = pd.DataFrame(macro_data, index=df.index)

        # Use pd.concat to merge all columns at once (avoids fragmentation warning)
        merged = pd.concat([df, macro_df], axis=1, copy=False)

        # Return a copy to ensure defragmentation
        return merged.copy()

    def _load_macro_features(self) -> Optional[pd.DataFrame]:
        """Load macro features from parquet file."""
        if self.macro_features == "none":
            return None

        if not self.macro_data_path.exists():
            print(f"Warning: Macro features file not found: {self.macro_data_path}")
            print("Will use market features only, without macro data")
            return None

        try:
            df = pd.read_parquet(self.macro_data_path)
            print(f"Loaded macro features: {df.shape}, "
                  f"date range: {df.index.min()} to {df.index.max()}")
            return df
        except Exception as e:
            print(f"Warning: Failed to load macro features: {e}")
            return None

    def _get_macro_feature_columns(self) -> List[str]:
        """Get macro feature columns based on configuration."""
        if self.macro_features == "core":
            return self.CORE_MACRO_FEATURES
        elif self.macro_features == "vix_only":
            return self.VIX_ONLY_FEATURES
        elif self.macro_features == "none":
            return []
        else:  # "all"
            return self.ALL_MACRO_FEATURES

    def get_feature_config(self):
        """
        Get feature config: Alpha158 (excluding problematic features) + TA-Lib indicators.

        Excluded features:
        - VWAP0: US stock VWAP data often missing
        - VMA*: Volume MA divided by current volume produces extreme values
        - VSTD*: Volume std divided by current volume produces extreme values
        """
        # Custom Alpha158 config excluding VWAP and problematic rolling features
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                # Exclude VWAP as it's often missing in US stock data
                "feature": ["OPEN", "HIGH", "LOW"],
            },
            "rolling": {
                # Exclude VMA and VSTD as they produce extreme outliers
                "exclude": ["VMA", "VSTD"],
            },
        }
        fields, names = Alpha158.get_feature_config(conf)

        # Add TA-Lib indicators
        talib_fields, talib_names = self._get_talib_features()
        fields.extend(talib_fields)
        names.extend(talib_names)

        return fields, names

    def _get_talib_features(self):
        """Get selected TA-Lib technical indicators (Lite version)."""
        fields = []
        names = []

        # Momentum indicators (4)
        fields.append("TALIB_RSI($close, 14)")
        names.append("TALIB_RSI14")
        fields.append("TALIB_MOM($close, 10)/$close")
        names.append("TALIB_MOM10")
        fields.append("TALIB_ROC($close, 10)")
        names.append("TALIB_ROC10")
        fields.append("TALIB_CMO($close, 14)")
        names.append("TALIB_CMO14")

        # MACD (3)
        fields.append("TALIB_MACD_MACD($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD")
        fields.append("TALIB_MACD_SIGNAL($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD_SIGNAL")
        fields.append("TALIB_MACD_HIST($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD_HIST")

        # Moving averages (2)
        fields.append("TALIB_EMA($close, 20)/$close")
        names.append("TALIB_EMA20")
        fields.append("TALIB_SMA($close, 20)/$close")
        names.append("TALIB_SMA20")

        # Bollinger Bands (3)
        fields.append("(TALIB_BBANDS_UPPER($close, 20, 2) - $close)/$close")
        names.append("TALIB_BB_UPPER_DIST")
        fields.append("($close - TALIB_BBANDS_LOWER($close, 20, 2))/$close")
        names.append("TALIB_BB_LOWER_DIST")
        fields.append("(TALIB_BBANDS_UPPER($close, 20, 2) - TALIB_BBANDS_LOWER($close, 20, 2))/$close")
        names.append("TALIB_BB_WIDTH")

        # Volatility (2)
        fields.append("TALIB_ATR($high, $low, $close, 14)/$close")
        names.append("TALIB_ATR14")
        fields.append("TALIB_NATR($high, $low, $close, 14)")
        names.append("TALIB_NATR14")

        # Trend indicators (3)
        fields.append("TALIB_ADX($high, $low, $close, 14)")
        names.append("TALIB_ADX14")
        fields.append("TALIB_PLUS_DI($high, $low, $close, 14)")
        names.append("TALIB_PLUS_DI14")
        fields.append("TALIB_MINUS_DI($high, $low, $close, 14)")
        names.append("TALIB_MINUS_DI14")

        # Stochastic (2)
        fields.append("TALIB_STOCH_K($high, $low, $close, 5, 3, 3)")
        names.append("TALIB_STOCH_K")
        fields.append("TALIB_STOCH_D($high, $low, $close, 5, 3, 3)")
        names.append("TALIB_STOCH_D")

        # Statistics (1)
        fields.append("TALIB_STDDEV($close, 20, 1)/$close")
        names.append("TALIB_STDDEV20")

        return fields, names

    def get_label_config(self):
        """Return N-day volatility label."""
        volatility_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [volatility_expr], ["LABEL0"]


class Alpha158_Macro(DataHandlerLP):
    """
    Alpha158 features + Macro features (without TA-Lib).

    Lighter version for larger stock pools where TA-Lib may cause memory issues.

    Total features: ~158 (Alpha158) + ~105 (Macro) = ~263
    """

    # Reuse macro feature definitions from parent
    ALL_MACRO_FEATURES = Alpha158_Volatility_TALib_Macro.ALL_MACRO_FEATURES
    CORE_MACRO_FEATURES = Alpha158_Volatility_TALib_Macro.CORE_MACRO_FEATURES
    VIX_ONLY_FEATURES = Alpha158_Volatility_TALib_Macro.VIX_ONLY_FEATURES

    def __init__(
        self,
        volatility_window: int = 2,
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
        # Macro feature parameters
        macro_data_path: Union[str, Path] = None,
        macro_features: str = "all",
        **kwargs,
    ):
        """
        Initialize DataHandler with macro features (no TA-Lib).

        Args:
            volatility_window: Prediction window (days)
            macro_data_path: Path to macro features parquet file
            macro_features: Macro feature set ("all", "core", "vix_only", "none")
        """
        self.volatility_window = volatility_window
        self.macro_data_path = Path(macro_data_path) if macro_data_path else DEFAULT_MACRO_PATH
        self.macro_features = macro_features

        # Load macro features
        self._macro_df = self._load_macro_features()

        from qlib.contrib.data.handler import check_transform_proc, _DEFAULT_LEARN_PROCESSORS

        if learn_processors is None:
            learn_processors = _DEFAULT_LEARN_PROCESSORS

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        # Use standard Alpha158 config (excluding problematic features)
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW"],  # Exclude VWAP
            },
            "rolling": {
                "exclude": ["VMA", "VSTD"],
            },
        }
        fields, names = Alpha158.get_feature_config(conf)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": (fields, names),
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

    # Reuse methods from Alpha158_Volatility_TALib_Macro
    _load_macro_features = Alpha158_Volatility_TALib_Macro._load_macro_features
    _get_macro_feature_columns = Alpha158_Volatility_TALib_Macro._get_macro_feature_columns
    _add_macro_to_processed_data = Alpha158_Volatility_TALib_Macro._add_macro_to_processed_data
    _merge_macro_to_df = Alpha158_Volatility_TALib_Macro._merge_macro_to_df

    def process_data(self, with_fit: bool = False):
        """
        Override process_data to add macro features AFTER processors run.

        This ensures macro features are included in _learn and _infer.
        """
        # First, call parent's process_data (DataHandlerLP)
        DataHandlerLP.process_data(self, with_fit=with_fit)

        # Then add macro features to _learn and _infer
        if self._macro_df is not None and self.macro_features != "none":
            self._add_macro_to_processed_data()

    def get_label_config(self):
        """Return N-day volatility label."""
        volatility_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [volatility_expr], ["LABEL0"]


class Alpha360_Macro(DataHandlerLP):
    """
    Alpha360 (360 features) + Time-aligned Macro features

    Combines Alpha360's 60-day OHLCV history with macro market features,
    where each timestep includes both stock and macro data from that day.

    Structure: (60, 6+M) where M = number of macro features
    - Stock features: CLOSE, OPEN, HIGH, LOW, VWAP, VOLUME per timestep
    - Macro features: VIX, bonds, yields, etc. per timestep (same for all stocks on that day)

    Feature counts:
    - macro_features="none": 360 features (6 × 60)
    - macro_features="vix_only": 1140 features ((6+13) × 60)
    - macro_features="core": 1860 features ((6+25) × 60)
    - macro_features="all": 6660 features ((6+105) × 60)

    Usage:
        handler = Alpha360_Macro(
            volatility_window=2,
            instruments=["AAPL", "MSFT", "NVDA"],
            start_time="2020-01-01",
            end_time="2024-12-31",
            macro_features="core",  # or "all", "vix_only", "none"
        )

        # For ALSTM/TCN/Transformer:
        # d_feat = 6 + num_macro_features (e.g., 31 for core)
        # seq_len = 60
    """

    # Reuse macro feature definitions from Alpha158_Volatility_TALib_Macro
    ALL_MACRO_FEATURES = Alpha158_Volatility_TALib_Macro.ALL_MACRO_FEATURES
    CORE_MACRO_FEATURES = Alpha158_Volatility_TALib_Macro.CORE_MACRO_FEATURES
    VIX_ONLY_FEATURES = Alpha158_Volatility_TALib_Macro.VIX_ONLY_FEATURES

    def __init__(
        self,
        volatility_window: int = 2,
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
        # Macro feature parameters
        macro_data_path: Union[str, Path] = None,
        macro_features: str = "core",  # "all", "core", "vix_only", "none"
        **kwargs,
    ):
        """
        Initialize Alpha360 + Macro DataHandler.

        Args:
            volatility_window: Prediction window (days) for label
            macro_data_path: Path to macro features parquet file
            macro_features: Macro feature set to use
                - "all": All macro features (~105)
                - "core": Core features (~25) [default]
                - "vix_only": VIX features only (~13)
                - "none": No macro features (pure Alpha360)
            **kwargs: Additional arguments for parent class
        """
        self.volatility_window = volatility_window
        self.macro_data_path = Path(macro_data_path) if macro_data_path else DEFAULT_MACRO_PATH
        self.macro_features = macro_features

        # Load macro features
        self._macro_df = self._load_macro_features()

        from qlib.contrib.data.handler import check_transform_proc, _DEFAULT_LEARN_PROCESSORS
        from qlib.contrib.data.loader import Alpha360DL

        if learn_processors is None:
            learn_processors = _DEFAULT_LEARN_PROCESSORS

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        # Use Alpha360's feature config
        fields, names = Alpha360DL.get_feature_config()

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": (fields, names),
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

    def process_data(self, with_fit: bool = False):
        """
        Override process_data to add time-aligned macro features AFTER processors run.
        """
        # First, call parent's process_data
        super().process_data(with_fit=with_fit)

        # Then add macro features with temporal expansion
        if self._macro_df is not None and self.macro_features != "none":
            self._add_macro_to_processed_data()

    def _add_macro_to_processed_data(self):
        """Add time-aligned macro features to _learn and _infer."""
        try:
            macro_cols = self._get_macro_feature_columns()
            available_cols = [c for c in macro_cols if c in self._macro_df.columns]

            if not available_cols:
                print("Warning: No macro features available")
                return

            # Add to _learn with temporal expansion
            if hasattr(self, "_learn") and self._learn is not None:
                self._learn = self._expand_macro_temporally(self._learn, available_cols)
                num_macro_expanded = len(available_cols) * 60
                print(f"Added {num_macro_expanded} macro features to learn data "
                      f"({len(available_cols)} features × 60 timesteps)")

            # Add to _infer with temporal expansion
            if hasattr(self, "_infer") and self._infer is not None:
                self._infer = self._expand_macro_temporally(self._infer, available_cols)

        except Exception as e:
            print(f"Warning: Error adding macro features: {e}")
            import traceback
            traceback.print_exc()

    def _expand_macro_temporally(self, df: pd.DataFrame, macro_cols: list) -> pd.DataFrame:
        """
        Expand macro features temporally to align with Alpha360's 60-day structure.

        For each macro feature col, creates 60 columns: col_59, col_58, ..., col_0
        where col_i contains the macro value from i days ago.

        Args:
            df: DataFrame with Alpha360 features (index: datetime, instrument)
            macro_cols: List of macro feature column names

        Returns:
            DataFrame with additional macro columns for each timestep
        """
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        # Build all expanded macro columns at once to avoid fragmentation
        expanded_data = {}
        for col in macro_cols:
            base_series = self._macro_df[col]
            for i in range(59, -1, -1):
                col_name = f"{col}_{i}"
                # Shift macro data by i days (shift(i) means value from i days ago)
                shifted = base_series.shift(i)
                aligned_values = shifted.reindex(main_datetimes).values

                if has_multi_columns:
                    expanded_data[('feature', col_name)] = aligned_values
                else:
                    expanded_data[col_name] = aligned_values

        # Create DataFrame with all expanded macro columns
        expanded_df = pd.DataFrame(expanded_data, index=df.index)

        # Use pd.concat to merge all columns at once (avoids fragmentation warning)
        merged = pd.concat([df, expanded_df], axis=1, copy=False)

        # Return a copy to ensure defragmentation
        return merged.copy()

    # Reuse methods from Alpha158_Volatility_TALib_Macro
    _load_macro_features = Alpha158_Volatility_TALib_Macro._load_macro_features
    _get_macro_feature_columns = Alpha158_Volatility_TALib_Macro._get_macro_feature_columns

    def get_label_config(self):
        """Return N-day volatility label."""
        volatility_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [volatility_expr], ["LABEL0"]


class Alpha180_Macro(DataHandlerLP):
    """
    Alpha180 (180 features) + Time-aligned Macro features

    Combines Alpha180's 30-day OHLCV history with macro market features,
    where each timestep includes both stock and macro data from that day.

    Structure: (30, 6+M) where M = number of macro features
    - Stock features: CLOSE, OPEN, HIGH, LOW, VWAP, VOLUME per timestep
    - Macro features: VIX, bonds, yields, etc. per timestep (same for all stocks on that day)

    Feature counts:
    - macro_features="none": 180 features (6 × 30)
    - macro_features="vix_only": 570 features ((6+13) × 30)
    - macro_features="core": 930 features ((6+25) × 30)
    - macro_features="all": 3330 features ((6+105) × 30)

    Usage:
        handler = Alpha180_Macro(
            volatility_window=2,
            instruments=["AAPL", "MSFT", "NVDA"],
            start_time="2020-01-01",
            end_time="2024-12-31",
            macro_features="core",  # or "all", "vix_only", "none"
        )

        # For ALSTM/TCN/Transformer:
        # d_feat = 6 + num_macro_features (e.g., 31 for core)
        # seq_len = 30
    """

    # Reuse macro feature definitions from Alpha158_Volatility_TALib_Macro
    ALL_MACRO_FEATURES = Alpha158_Volatility_TALib_Macro.ALL_MACRO_FEATURES
    CORE_MACRO_FEATURES = Alpha158_Volatility_TALib_Macro.CORE_MACRO_FEATURES
    VIX_ONLY_FEATURES = Alpha158_Volatility_TALib_Macro.VIX_ONLY_FEATURES

    def __init__(
        self,
        volatility_window: int = 2,
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
        # Macro feature parameters
        macro_data_path: Union[str, Path] = None,
        macro_features: str = "core",  # "all", "core", "vix_only", "none"
        **kwargs,
    ):
        """
        Initialize Alpha180 + Macro DataHandler.

        Args:
            volatility_window: Prediction window (days) for label
            macro_data_path: Path to macro features parquet file
            macro_features: Macro feature set to use
                - "all": All macro features (~105)
                - "core": Core features (~25) [default]
                - "vix_only": VIX features only (~13)
                - "none": No macro features (pure Alpha180)
            **kwargs: Additional arguments for parent class
        """
        self.volatility_window = volatility_window
        self.macro_data_path = Path(macro_data_path) if macro_data_path else DEFAULT_MACRO_PATH
        self.macro_features = macro_features

        # Load macro features
        self._macro_df = self._load_macro_features()

        from qlib.contrib.data.handler import check_transform_proc, _DEFAULT_LEARN_PROCESSORS
        from data.datahandler_ext import Alpha180DL

        if learn_processors is None:
            learn_processors = _DEFAULT_LEARN_PROCESSORS

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        # Use Alpha180's feature config
        fields, names = Alpha180DL.get_feature_config()

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": (fields, names),
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

    def process_data(self, with_fit: bool = False):
        """
        Override process_data to add time-aligned macro features AFTER processors run.
        """
        # First, call parent's process_data
        super().process_data(with_fit=with_fit)

        # Then add macro features with temporal expansion
        if self._macro_df is not None and self.macro_features != "none":
            self._add_macro_to_processed_data()

    def _add_macro_to_processed_data(self):
        """Add time-aligned macro features to _learn and _infer."""
        try:
            macro_cols = self._get_macro_feature_columns()
            available_cols = [c for c in macro_cols if c in self._macro_df.columns]

            if not available_cols:
                print("Warning: No macro features available")
                return

            # Add to _learn with temporal expansion
            if hasattr(self, "_learn") and self._learn is not None:
                self._learn = self._expand_macro_temporally(self._learn, available_cols)
                num_macro_expanded = len(available_cols) * 30
                print(f"Added {num_macro_expanded} macro features to learn data "
                      f"({len(available_cols)} features × 30 timesteps)")

            # Add to _infer with temporal expansion
            if hasattr(self, "_infer") and self._infer is not None:
                self._infer = self._expand_macro_temporally(self._infer, available_cols)

        except Exception as e:
            print(f"Warning: Error adding macro features: {e}")
            import traceback
            traceback.print_exc()

    def _expand_macro_temporally(self, df: pd.DataFrame, macro_cols: list) -> pd.DataFrame:
        """
        Expand macro features temporally to align with Alpha180's 30-day structure.

        For each macro feature col, creates 30 columns: col_29, col_28, ..., col_0
        where col_i contains the macro value from i days ago.

        Args:
            df: DataFrame with Alpha180 features (index: datetime, instrument)
            macro_cols: List of macro feature column names

        Returns:
            DataFrame with additional macro columns for each timestep
        """
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        # Build all expanded macro columns at once to avoid fragmentation
        expanded_data = {}
        for col in macro_cols:
            base_series = self._macro_df[col]
            for i in range(29, -1, -1):  # 30 days: 29, 28, ..., 0
                col_name = f"{col}_{i}"
                # Shift macro data by i days (shift(i) means value from i days ago)
                shifted = base_series.shift(i)
                aligned_values = shifted.reindex(main_datetimes).values

                if has_multi_columns:
                    expanded_data[('feature', col_name)] = aligned_values
                else:
                    expanded_data[col_name] = aligned_values

        # Create DataFrame with all expanded macro columns
        expanded_df = pd.DataFrame(expanded_data, index=df.index)

        # Use pd.concat to merge all columns at once (avoids fragmentation warning)
        merged = pd.concat([df, expanded_df], axis=1, copy=False)

        # Return a copy to ensure defragmentation
        return merged.copy()

    # Reuse methods from Alpha158_Volatility_TALib_Macro
    _load_macro_features = Alpha158_Volatility_TALib_Macro._load_macro_features
    _get_macro_feature_columns = Alpha158_Volatility_TALib_Macro._get_macro_feature_columns

    def get_label_config(self):
        """Return N-day volatility label."""
        volatility_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [volatility_expr], ["LABEL0"]


class Alpha300_Macro(DataHandlerLP):
    """
    Alpha300 (300 features, no VWAP) + Time-aligned Macro features

    Combines Alpha300's 60-day OHLCV history (without VWAP) with macro market features,
    where each timestep includes both stock and macro data from that day.

    Designed for US stocks where VWAP data is typically 100% missing.

    Structure: (60, 5+M) where M = number of macro features
    - Stock features: CLOSE, OPEN, HIGH, LOW, VOLUME per timestep (no VWAP)
    - Macro features: VIX, bonds, yields, etc. per timestep (same for all stocks on that day)

    Feature counts (recommended: minimal for memory efficiency):
    - macro_features="minimal": ~600 features ((5+5) x 60), d_feat~10 [DEFAULT, memory efficient]
    - macro_features="none": 300 features (5 x 60), d_feat=5
    - macro_features="vix_only": varies based on available data
    - macro_features="core": varies based on available data [HIGH MEMORY]
    - macro_features="all": varies based on available data [VERY HIGH MEMORY]

    Note: Actual d_feat depends on which macro features are available in your data.
    The handler will print the correct d_feat value during initialization.

    Minimal macro features (6, selected via CatBoost forward selection):
    - macro_vix_zscore20: VIX normalized level
    - macro_hy_spread_zscore: High-yield credit spread
    - macro_credit_stress: Credit market stress
    - macro_tlt_pct_20d: Bond momentum
    - macro_uso_pct_5d: Oil momentum
    - macro_risk_on_off: Risk regime indicator

    Usage:
        handler = Alpha300_Macro(
            volatility_window=5,
            instruments=["AAPL", "MSFT", "NVDA"],
            start_time="2020-01-01",
            end_time="2024-12-31",
            macro_features="minimal",  # recommended for TCN
        )

        # For TCN:
        # d_feat = 11 (minimal), step_len = 60
    """

    # Reuse macro feature definitions from Alpha158_Volatility_TALib_Macro
    ALL_MACRO_FEATURES = Alpha158_Volatility_TALib_Macro.ALL_MACRO_FEATURES
    CORE_MACRO_FEATURES = Alpha158_Volatility_TALib_Macro.CORE_MACRO_FEATURES
    VIX_ONLY_FEATURES = Alpha158_Volatility_TALib_Macro.VIX_ONLY_FEATURES

    # Minimal macro features (6) - based on CatBoost forward selection
    # These 6 features provided the best IC improvement in nested CV
    # IMPORTANT: All features should have similar scale (std ≈ 1.0) for TCN
    MINIMAL_MACRO_FEATURES = [
        # VIX (1) - most important volatility indicator (already z-scored, std≈1.2)
        "macro_vix_zscore20",
        # Credit/Risk (2) - credit spreads are strong predictors
        "macro_hy_spread_zscore",  # Use z-scored version (std≈1.0), not raw (std≈0.03)
        "macro_credit_stress",     # Already z-scored (std≈1.0)
        # Bonds (1) - interest rate sensitivity (will be z-scored in handler)
        "macro_tlt_pct_20d",
        # Commodities (1) - oil as economic indicator (will be z-scored in handler)
        "macro_uso_pct_5d",
        # Cross-asset (1) - risk regime (already z-scored, std≈1.1)
        "macro_risk_on_off",
    ]

    def __init__(
        self,
        volatility_window: int = 2,
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
        # Macro feature parameters
        macro_data_path: Union[str, Path] = None,
        macro_features: str = "minimal",  # "minimal", "core", "vix_only", "all", "none"
        **kwargs,
    ):
        """
        Initialize Alpha300 + Macro DataHandler.

        Args:
            volatility_window: Prediction window (days) for label
            macro_data_path: Path to macro features parquet file
            macro_features: Macro feature set to use
                - "minimal": 6 key features (recommended, d_feat=11) [default]
                - "core": Core features (~23, d_feat=28)
                - "vix_only": VIX features only (~13, d_feat=18)
                - "all": All macro features (~105)
                - "none": No macro features (pure Alpha300, d_feat=5)
            **kwargs: Additional arguments for parent class
        """
        self.volatility_window = volatility_window
        self.macro_data_path = Path(macro_data_path) if macro_data_path else DEFAULT_MACRO_PATH
        self.macro_features = macro_features

        # Load macro features
        self._macro_df = self._load_macro_features()

        from qlib.contrib.data.handler import check_transform_proc, _DEFAULT_LEARN_PROCESSORS
        from data.datahandler_ext import Alpha300DL

        if learn_processors is None:
            learn_processors = _DEFAULT_LEARN_PROCESSORS

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        # Use Alpha300's feature config (without VWAP)
        fields, names = Alpha300DL.get_feature_config()

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": (fields, names),
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

    def process_data(self, with_fit: bool = False):
        """
        Override process_data to add time-aligned macro features AFTER processors run.
        """
        # First, call parent's process_data
        super().process_data(with_fit=with_fit)

        # Then add macro features with temporal expansion
        if self._macro_df is not None and self.macro_features != "none":
            self._add_macro_to_processed_data()

    def _add_macro_to_processed_data(self):
        """Add time-aligned macro features to _learn and _infer."""
        try:
            macro_cols = self._get_macro_feature_columns()
            available_cols = [c for c in macro_cols if c in self._macro_df.columns]

            if not available_cols:
                print("Warning: No macro features available")
                return

            # Show detailed feature info
            missing_cols = [c for c in macro_cols if c not in self._macro_df.columns]
            print(f"Alpha300_Macro: Requested {len(macro_cols)} macro features, found {len(available_cols)}")
            print(f"Alpha300_Macro: Available: {available_cols}")
            if missing_cols:
                print(f"Alpha300_Macro: Missing: {missing_cols}")

            # Add to _learn with temporal expansion
            if hasattr(self, "_learn") and self._learn is not None:
                # First normalize stock features to match macro feature scale
                self._learn = self._normalize_stock_features(self._learn)
                # Then add macro features
                self._learn = self._expand_macro_temporally(self._learn, available_cols)
                num_macro_expanded = len(available_cols) * 60
                d_feat = 5 + len(available_cols)  # 5 base OHLCV + macro
                total_features = 300 + num_macro_expanded
                print(f"Alpha300_Macro: Total features = {total_features}, d_feat = {d_feat}, step_len = 60")
                print(f"Alpha300_Macro: Use --d-feat {d_feat} when running TCN")

            # Add to _infer with temporal expansion
            if hasattr(self, "_infer") and self._infer is not None:
                # First normalize stock features to match macro feature scale
                self._infer = self._normalize_stock_features(self._infer)
                # Then add macro features
                self._infer = self._expand_macro_temporally(self._infer, available_cols)

        except Exception as e:
            print(f"Warning: Error adding macro features: {e}")
            import traceback
            traceback.print_exc()

    def _normalize_stock_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize Alpha300 stock features (CLOSE, OPEN, HIGH, LOW, VOLUME) to have std ≈ 1.0.

        Problem: Stock price features have std ~0.08, VOLUME has std ~0.45,
        while macro features have std ~1.0. This scale difference causes TCN
        to focus on macro features and ignore stock features.

        Solution: Z-score normalize all stock features so all features have similar scale.
        """
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        # Identify all stock feature columns (CLOSE, OPEN, HIGH, LOW, VOLUME)
        stock_prefixes = ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME']
        stock_cols = []
        for col in df.columns:
            col_name = col[1] if has_multi_columns else col
            if any(col_name.startswith(prefix) for prefix in stock_prefixes):
                stock_cols.append(col)

        if not stock_cols:
            return df

        # Apply z-score normalization to all stock features
        df = df.copy()
        for col in stock_cols:
            col_data = df[col]
            mean = col_data.mean()
            std = col_data.std()
            if std > 1e-8:
                # Z-score normalization
                normalized = (col_data - mean) / std
                # Clip extreme values to prevent outliers from dominating
                df[col] = normalized.clip(-5, 5)

        print(f"Alpha300_Macro: Normalized {len(stock_cols)} stock features to std≈1.0")
        return df

    # Features that need z-score normalization (small std, not already z-scored)
    FEATURES_NEED_ZSCORE = [
        "macro_tlt_pct_20d",  # std ≈ 0.035
        "macro_uso_pct_5d",   # std ≈ 0.043
        "macro_tlt_pct_5d",   # std ≈ 0.018
        "macro_tlt_pct_1d",   # std ≈ 0.009
        "macro_uso_pct_1d",   # std ≈ 0.020
        "macro_uso_pct_20d",  # std ≈ 0.086
        "macro_gld_pct_20d",  # pct features have small std
        "macro_gld_pct_5d",
        "macro_gld_pct_1d",
        "macro_uup_pct_5d",
        "macro_uup_pct_1d",
        "macro_spy_pct_20d",
        "macro_spy_pct_5d",
        "macro_spy_pct_1d",
    ]

    def _expand_macro_temporally(self, df: pd.DataFrame, macro_cols: list) -> pd.DataFrame:
        """
        Expand macro features temporally to align with Alpha300's 60-day structure.

        For each macro feature col, creates 60 columns: col_59, col_58, ..., col_0
        where col_i contains the macro value from i days ago.

        Features with small std (pct changes) are z-score normalized using
        rolling 60-day statistics to maintain consistent scale with other features.

        Args:
            df: DataFrame with Alpha300 features (index: datetime, instrument)
            macro_cols: List of macro feature column names

        Returns:
            DataFrame with additional macro columns for each timestep
        """
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        # Build all expanded macro columns at once to avoid fragmentation
        expanded_data = {}
        for col in macro_cols:
            base_series = self._macro_df[col].copy()

            # Apply rolling z-score normalization for features with small std
            if col in self.FEATURES_NEED_ZSCORE:
                rolling_mean = base_series.rolling(window=60, min_periods=20).mean()
                rolling_std = base_series.rolling(window=60, min_periods=20).std()
                # Z-score with protection against division by zero
                base_series = (base_series - rolling_mean) / (rolling_std + 1e-8)
                # Clip extreme values
                base_series = base_series.clip(-5, 5)

            for i in range(59, -1, -1):  # 60 days: 59, 58, ..., 0
                col_name = f"{col}_{i}"
                # Shift macro data by i days (shift(i) means value from i days ago)
                shifted = base_series.shift(i)
                aligned_values = shifted.reindex(main_datetimes).values

                if has_multi_columns:
                    expanded_data[('feature', col_name)] = aligned_values
                else:
                    expanded_data[col_name] = aligned_values

        # Create DataFrame with all expanded macro columns
        expanded_df = pd.DataFrame(expanded_data, index=df.index)

        # Use pd.concat to merge all columns at once (avoids fragmentation warning)
        merged = pd.concat([df, expanded_df], axis=1, copy=False)

        # Return a copy to ensure defragmentation
        return merged.copy()

    # Reuse load method from Alpha158_Volatility_TALib_Macro
    _load_macro_features = Alpha158_Volatility_TALib_Macro._load_macro_features

    def _get_macro_feature_columns(self):
        """Get macro feature columns based on configuration."""
        if self.macro_features == "minimal":
            return self.MINIMAL_MACRO_FEATURES
        elif self.macro_features == "core":
            return self.CORE_MACRO_FEATURES
        elif self.macro_features == "vix_only":
            return self.VIX_ONLY_FEATURES
        elif self.macro_features == "none":
            return []
        else:  # "all"
            return self.ALL_MACRO_FEATURES

    def get_label_config(self):
        """Return N-day volatility label."""
        volatility_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [volatility_expr], ["LABEL0"]
