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
                self._merge_macro_to_df(self._learn, available_cols)
                print(f"Added {len(available_cols)} macro features to learn data")

            # Add to _infer
            if hasattr(self, "_infer") and self._infer is not None:
                self._merge_macro_to_df(self._infer, available_cols)

        except Exception as e:
            print(f"Warning: Error adding macro features: {e}")

    def _merge_macro_to_df(self, df: pd.DataFrame, cols: list):
        """Merge macro features into a DataFrame."""
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        for col in cols:
            macro_series = self._macro_df[col]
            aligned_values = macro_series.reindex(main_datetimes).values

            if has_multi_columns:
                df[('feature', col)] = aligned_values
            else:
                df[col] = aligned_values

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
    process_data = Alpha158_Volatility_TALib_Macro.process_data
    _add_macro_to_processed_data = Alpha158_Volatility_TALib_Macro._add_macro_to_processed_data
    _merge_macro_to_df = Alpha158_Volatility_TALib_Macro._merge_macro_to_df

    def get_label_config(self):
        """Return N-day volatility label."""
        volatility_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [volatility_expr], ["LABEL0"]
