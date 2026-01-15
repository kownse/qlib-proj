"""
Enhanced DataHandler with refined feature set based on feature importance analysis.

Combines:
- Refined Alpha158 features (removing low-importance features like RANK, VSUMP, etc.)
- Enhanced volatility features (multi-period, Parkinson, up/down vol)
- High-importance TA-Lib features only
- Refined macro features (keeping high-importance only)
- New relative strength and liquidity features

Target: ~130-150 features (down from 287 in alpha158-talib-macro)

Based on feature importance analysis from CatBoost model:
- Top features: ATR14, macro_gld_vol20, macro_vix_term_structure, macro_spy_vol20
- Removed: RANK*, VSUMP*, VSUMD*, VSUMN*, short-period CNTP/CNTN, zero-importance macros
"""

import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

# Add scripts directory to path for imports
script_dir = Path(__file__).parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

from qlib.data.dataset.handler import DataHandlerLP

# Project root
PROJECT_ROOT = script_dir.parent

# Default macro features path
DEFAULT_MACRO_PATH = PROJECT_ROOT / "my_data" / "macro_processed" / "macro_features.parquet"


class Alpha158_Enhanced(DataHandlerLP):
    """
    Enhanced Alpha158 with feature importance-based refinement.

    Feature Categories:
    1. Refined Alpha158 (kbar + price + selected rolling): ~55 features
    2. Enhanced Volatility (multi-period, Parkinson, up/down): ~10 features
    3. TA-Lib high-importance only: ~15 features
    4. Refined Macro features: ~35 features
    5. Momentum Quality features: ~5 features
    6. Liquidity features: ~5 features
    7. Relative Strength (computed in process_data): ~3 features

    Total: ~128-140 features

    Usage:
        handler = Alpha158_Enhanced(
            volatility_window=5,
            instruments=["AAPL", "MSFT", "NVDA"],
            start_time="2020-01-01",
            end_time="2024-12-31",
        )
    """

    # High-importance macro features based on feature importance analysis
    # Sorted by importance, keeping only those with importance > 0.3
    SELECTED_MACRO_FEATURES = [
        # VIX features (high importance: term_structure=2.85, level=1.09)
        "macro_vix_term_structure", "macro_vix_level", "macro_vix_zscore20",
        "macro_vix_pct_5d", "macro_vix_pct_10d", "macro_vix_term_zscore",
        "macro_vix_ma5_ratio", "macro_vix_ma20_ratio",
        # Asset volatility (high importance: gld_vol20=2.88, spy_vol20=2.40, bond_vol20=1.44)
        "macro_gld_vol20", "macro_spy_vol20", "macro_bond_vol20",
        "macro_uso_vol20", "macro_jnk_vol20",
        # Credit spreads (high importance: ig_spread=1.75, hy_spread_zscore=1.58, hy_spread=1.44)
        "macro_ig_spread", "macro_hy_spread_zscore", "macro_hy_spread",
        "macro_hy_spread_chg5", "macro_credit_risk", "macro_credit_stress",
        # Yield curve (high importance: yield_curve_zscore=1.49, yield_2y=1.21, yield_3m10y=1.11)
        "macro_yield_curve_zscore", "macro_yield_2y", "macro_yield_3m10y",
        "macro_yield_10y", "macro_yield_2s10s", "macro_yield_curve_slope",
        "macro_yield_10y_chg20", "macro_yield_curve_chg5",
        # Sector rotation (high importance: xly_pct_20d=1.90, xlu series)
        "macro_xly_pct_20d", "macro_xly_pct_5d", "macro_xly_vs_spy",
        "macro_xlu_pct_20d", "macro_xlu_pct_5d", "macro_xlu_vs_spy",
        "macro_xlf_pct_20d", "macro_xlf_vs_spy",
        "macro_xle_pct_20d", "macro_xle_vs_spy",
        "macro_xlk_pct_20d", "macro_xlk_vs_spy",
        "macro_xlv_pct_20d", "macro_xlv_pct_5d",
        "macro_xli_pct_20d", "macro_xlp_pct_20d",
        # Market sentiment (various importance levels)
        "macro_spy_pct_5d", "macro_spy_pct_20d", "macro_spy_ma20_ratio",
        "macro_qqq_vs_spy", "macro_risk_on_off", "macro_market_stress",
        "macro_stock_bond_corr",
        # Gold/Bond momentum
        "macro_gld_pct_5d", "macro_gld_pct_20d", "macro_gld_ma20_ratio",
        "macro_tlt_pct_5d", "macro_tlt_pct_20d",
        # Credit/Risk
        "macro_hyg_pct_20d", "macro_hyg_vs_lqd", "macro_hyg_tlt_ratio",
        # Global
        "macro_eem_vs_spy", "macro_efa_vs_spy",
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
        # Macro feature parameters
        macro_data_path: Union[str, Path] = None,
        **kwargs,
    ):
        """
        Initialize Enhanced DataHandler.

        Args:
            volatility_window: Prediction window (days) for label
            macro_data_path: Path to macro features parquet file
            **kwargs: Additional arguments for parent class
        """
        self.volatility_window = volatility_window
        self.macro_data_path = Path(macro_data_path) if macro_data_path else DEFAULT_MACRO_PATH

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

    def get_feature_config(self):
        """
        Get feature config combining refined Alpha158, enhanced volatility,
        high-importance TA-Lib, momentum quality, and liquidity features.
        """
        fields = []
        names = []

        # 1. Refined Alpha158 features (~55)
        alpha_fields, alpha_names = self._get_refined_alpha158_features()
        fields.extend(alpha_fields)
        names.extend(alpha_names)

        # 2. Enhanced volatility features (~10)
        vol_fields, vol_names = self._get_enhanced_volatility_features()
        fields.extend(vol_fields)
        names.extend(vol_names)

        # 3. High-importance TA-Lib features (~15)
        talib_fields, talib_names = self._get_talib_features()
        fields.extend(talib_fields)
        names.extend(talib_names)

        # 4. Momentum quality features (~5)
        mom_fields, mom_names = self._get_momentum_quality_features()
        fields.extend(mom_fields)
        names.extend(mom_names)

        # 5. Liquidity features (~5)
        liq_fields, liq_names = self._get_liquidity_features()
        fields.extend(liq_fields)
        names.extend(liq_names)

        return fields, names

    def _get_refined_alpha158_features(self):
        """
        Get refined Alpha158 features, removing low-importance features.

        Removed features (based on importance < 0.02):
        - RANK5/10/20/30/60 (all ~0)
        - VSUMP/VSUMD/VSUMN all periods (all < 0.02)
        - Short period features: CNTP5/10, CNTN5/10, CNTD5/10/20/30, IMAX5/10, IMIN5/10

        Kept features:
        - All Kbar features (9)
        - Price features: OPEN0, HIGH0, LOW0 (3)
        - Rolling features: only 20/60 day windows for most indicators
        """
        fields = []
        names = []

        # Kbar features (9) - all high importance
        fields.extend([
            "($close-$open)/$open",
            "($high-$low)/$open",
            "($close-$open)/($high-$low+1e-12)",
            "($high-Greater($open, $close))/$open",
            "($high-Greater($open, $close))/($high-$low+1e-12)",
            "($low-Less($open, $close))/$open",
            "($low-Less($open, $close))/($high-$low+1e-12)",
            "(2*$close-$high-$low)/$open",
            "(2*$close-$high-$low)/($high-$low+1e-12)",
        ])
        names.extend(["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"])

        # Price features (3)
        fields.extend(["$open/$close", "$high/$close", "$low/$close"])
        names.extend(["OPEN0", "HIGH0", "LOW0"])

        # Rolling features - refined selection
        # ROC (5 windows) - moderate importance
        for d in [5, 10, 20, 30, 60]:
            fields.append(f"Ref($close, {d})/$close")
            names.append(f"ROC{d}")

        # MA (5 windows) - moderate importance
        for d in [5, 10, 20, 30, 60]:
            fields.append(f"Mean($close, {d})/$close")
            names.append(f"MA{d}")

        # STD - only 20, 60 (STD60 = 0.49 importance)
        for d in [20, 60]:
            fields.append(f"Std($close, {d})/$close")
            names.append(f"STD{d}")

        # BETA - only 20, 60 (BETA60 = 0.59 importance)
        for d in [20, 60]:
            fields.append(f"Slope($close, {d})/Ref($close, {d})")
            names.append(f"BETA{d}")

        # RSQR - only 20, 60
        for d in [20, 60]:
            fields.append(f"Rsquare($close, {d})")
            names.append(f"RSQR{d}")

        # RESI - only 20, 60
        for d in [20, 60]:
            fields.append(f"Resi($close, {d})/$close")
            names.append(f"RESI{d}")

        # MAX - only 20, 60 (MAX60 = 0.57 importance)
        for d in [20, 60]:
            fields.append(f"Max($high, {d})/$close")
            names.append(f"MAX{d}")

        # MIN - only 20, 60 (MIN60 = 0.34, MIN30 = 0.17)
        for d in [20, 60]:
            fields.append(f"Min($low, {d})/$close")
            names.append(f"MIN{d}")

        # QTLU - only 20, 60
        for d in [20, 60]:
            fields.append(f"Quantile($close, {d}, 0.8)/$close")
            names.append(f"QTLU{d}")

        # QTLD - only 20, 60
        for d in [20, 60]:
            fields.append(f"Quantile($close, {d}, 0.2)/$close")
            names.append(f"QTLD{d}")

        # RSV - only 20, 60
        for d in [20, 60]:
            fields.append(f"($close-Min($low, {d}))/(Max($high, {d})-Min($low, {d})+1e-12)")
            names.append(f"RSV{d}")

        # IMAX - only 20, 60
        for d in [20, 60]:
            fields.append(f"IdxMax($high, {d})/{d}")
            names.append(f"IMAX{d}")

        # IMIN - only 20, 60
        for d in [20, 60]:
            fields.append(f"IdxMin($low, {d})/{d}")
            names.append(f"IMIN{d}")

        # IMXD - only 60 (IMXD60 = 0.22)
        fields.append("(IdxMax($high, 60)-IdxMin($low, 60))/60")
        names.append("IMXD60")

        # CORR - only 20, 60 (CORR60 = 0.23)
        for d in [20, 60]:
            fields.append(f"Corr($close, Log($volume+1), {d})")
            names.append(f"CORR{d}")

        # CORD - only 20, 60 (CORD60 = 0.20)
        for d in [20, 60]:
            fields.append(f"Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {d})")
            names.append(f"CORD{d}")

        # CNTP - only 60 (CNTP60 = 0.10)
        fields.append("Mean($close>Ref($close, 1), 60)")
        names.append("CNTP60")

        # CNTN - only 60 (CNTN60 = 0.07)
        fields.append("Mean($close<Ref($close, 1), 60)")
        names.append("CNTN60")

        # CNTD - only 60
        fields.append("Mean($close>Ref($close, 1), 60)-Mean($close<Ref($close, 1), 60)")
        names.append("CNTD60")

        # SUMP - only 20, 60
        for d in [20, 60]:
            fields.append(f"Sum(Greater($close-Ref($close, 1), 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)")
            names.append(f"SUMP{d}")

        # SUMN - only 20, 60
        for d in [20, 60]:
            fields.append(f"Sum(Greater(Ref($close, 1)-$close, 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)")
            names.append(f"SUMN{d}")

        # SUMD - only 20, 60
        for d in [20, 60]:
            fields.append(f"(Sum(Greater($close-Ref($close, 1), 0), {d})-Sum(Greater(Ref($close, 1)-$close, 0), {d}))/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)")
            names.append(f"SUMD{d}")

        # WVMA - only 20, 60 (WVMA60 = 0.20, WVMA20 = 0.11)
        for d in [20, 60]:
            fields.append(f"Std(Abs($close/Ref($close, 1)-1)*$volume, {d})/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {d})+1e-12)")
            names.append(f"WVMA{d}")

        return fields, names

    def _get_enhanced_volatility_features(self):
        """
        Get enhanced volatility features.

        New features:
        - Multi-period realized volatility
        - Volatility ratio (short/long term)
        - Parkinson volatility estimator
        - Volatility momentum
        - Up/Down volatility asymmetry
        """
        fields = []
        names = []

        # Multi-period realized volatility
        fields.append("Std($close/Ref($close,1)-1, 5)")
        names.append("VOL_5D")

        fields.append("Std($close/Ref($close,1)-1, 10)")
        names.append("VOL_10D")

        # Volatility ratio (short term / long term)
        fields.append("Std($close/Ref($close,1)-1, 5) / (Std($close/Ref($close,1)-1, 20) + 1e-12)")
        names.append("VOL_RATIO_5_20")

        # Parkinson volatility estimator (more efficient, uses high-low range)
        # Formula: sqrt(mean(ln(high/low)^2) / (4*ln(2)))
        # Use Power(x, 0.5) instead of Sqrt(x) as Sqrt is not a registered Qlib operator
        fields.append("Power(Mean(Power(Log($high/$low+1e-12), 2), 20) / 0.6931, 0.5)")
        names.append("PARKINSON_VOL_20")

        # Volatility momentum (current vol - lagged vol)
        fields.append("Std($close/Ref($close,1)-1, 5) - Ref(Std($close/Ref($close,1)-1, 5), 5)")
        names.append("VOL_MOMENTUM")

        # Upside volatility (volatility of positive returns)
        fields.append("Std(Max($close/Ref($close,1)-1, 0), 20)")
        names.append("UPSIDE_VOL_20")

        # Downside volatility (volatility of negative returns)
        fields.append("Std(Abs(Min($close/Ref($close,1)-1, 0)), 20)")
        names.append("DOWNSIDE_VOL_20")

        # Volatility skew (upside vol / downside vol)
        fields.append("Std(Max($close/Ref($close,1)-1, 0), 20) / (Std(Abs(Min($close/Ref($close,1)-1, 0)), 20) + 1e-12)")
        names.append("VOL_SKEW")

        return fields, names

    def _get_talib_features(self):
        """
        Get high-importance TA-Lib technical indicators.

        Kept (based on importance):
        - ATR14 (3.12), NATR14 (2.16) - highest importance
        - ADX14, PLUS_DI14, MINUS_DI14 - trend indicators
        - RSI14, CMO14, ROC10 - momentum
        - MACD, MACD_HIST - trend following
        - BB_WIDTH, BB_LOWER_DIST - volatility bands
        - STDDEV20 - statistics
        """
        fields = []
        names = []

        # Volatility (highest importance)
        fields.append("TALIB_ATR($high, $low, $close, 14)/$close")
        names.append("TALIB_ATR14")

        fields.append("TALIB_NATR($high, $low, $close, 14)")
        names.append("TALIB_NATR14")

        fields.append("TALIB_TRANGE($high, $low, $close)/$close")
        names.append("TALIB_TRANGE")

        # Trend indicators
        fields.append("TALIB_ADX($high, $low, $close, 14)")
        names.append("TALIB_ADX14")

        fields.append("TALIB_PLUS_DI($high, $low, $close, 14)")
        names.append("TALIB_PLUS_DI14")

        fields.append("TALIB_MINUS_DI($high, $low, $close, 14)")
        names.append("TALIB_MINUS_DI14")

        # Momentum indicators
        fields.append("TALIB_RSI($close, 14)")
        names.append("TALIB_RSI14")

        fields.append("TALIB_CMO($close, 14)")
        names.append("TALIB_CMO14")

        fields.append("TALIB_ROC($close, 10)")
        names.append("TALIB_ROC10")

        fields.append("TALIB_MOM($close, 10)/$close")
        names.append("TALIB_MOM10")

        # MACD
        fields.append("TALIB_MACD_MACD($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD")

        fields.append("TALIB_MACD_HIST($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD_HIST")

        fields.append("TALIB_MACD_SIGNAL($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD_SIGNAL")

        # Bollinger Bands
        fields.append("($close - TALIB_BBANDS_LOWER($close, 20, 2))/$close")
        names.append("TALIB_BB_LOWER_DIST")

        fields.append("(TALIB_BBANDS_UPPER($close, 20, 2) - TALIB_BBANDS_LOWER($close, 20, 2))/$close")
        names.append("TALIB_BB_WIDTH")

        # Stochastic (moderate importance)
        fields.append("TALIB_STOCH_K($high, $low, $close, 5, 3, 3)")
        names.append("TALIB_STOCH_K")

        fields.append("TALIB_STOCH_D($high, $low, $close, 5, 3, 3)")
        names.append("TALIB_STOCH_D")

        # Statistics
        fields.append("TALIB_STDDEV($close, 20, 1)/$close")
        names.append("TALIB_STDDEV20")

        return fields, names

    def _get_momentum_quality_features(self):
        """
        Get momentum quality features.

        New features:
        - Sharpe-like ratio (risk-adjusted momentum)
        - 52-week high/low position
        """
        fields = []
        names = []

        # Sharpe-like ratio (20-day)
        fields.append("Mean($close/Ref($close,1)-1, 20) / (Std($close/Ref($close,1)-1, 20) + 1e-12)")
        names.append("SHARPE_20D")

        # Sharpe-like ratio (60-day)
        fields.append("Mean($close/Ref($close,1)-1, 60) / (Std($close/Ref($close,1)-1, 60) + 1e-12)")
        names.append("SHARPE_60D")

        # Distance from 52-week high
        fields.append("($close - Max($high, 252)) / (Max($high, 252) + 1e-12)")
        names.append("PCT_FROM_52W_HIGH")

        # Distance from 52-week low
        fields.append("($close - Min($low, 252)) / (Min($low, 252) + 1e-12)")
        names.append("PCT_FROM_52W_LOW")

        return fields, names

    def _get_liquidity_features(self):
        """
        Get liquidity features.

        New features:
        - Volume z-score
        - Volume trend (short/long term ratio)
        - Dollar volume ratio
        - Simplified illiquidity measure
        """
        fields = []
        names = []

        # Volume z-score
        fields.append("($volume - Mean($volume, 20)) / (Std($volume, 20) + 1e-12)")
        names.append("VOLUME_ZSCORE")

        # Volume trend (short term / long term)
        fields.append("Mean($volume, 5) / (Mean($volume, 20) + 1e-12)")
        names.append("VOLUME_TREND")

        # Dollar volume ratio (current / average)
        fields.append("($close * $volume) / (Mean($close * $volume, 20) + 1e-12)")
        names.append("DOLLAR_VOL_RATIO")

        # Simplified illiquidity (Amihud-like)
        fields.append("Mean(Abs($close/Ref($close,1)-1) / (Log($volume+1)+1e-12), 20)")
        names.append("ILLIQUIDITY_20D")

        return fields, names

    def process_data(self, with_fit: bool = False):
        """
        Override process_data to add macro features and relative strength features
        AFTER processors run.
        """
        # First, call parent's process_data
        super().process_data(with_fit=with_fit)

        # Add macro features
        if self._macro_df is not None:
            self._add_macro_features()

        # Add relative strength features (requires SPY data from macro)
        self._add_relative_strength_features()

    def _add_macro_features(self):
        """Add selected macro features to _learn and _infer."""
        try:
            available_cols = [c for c in self.SELECTED_MACRO_FEATURES if c in self._macro_df.columns]

            if not available_cols:
                print("Warning: No macro features available")
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

    def _add_relative_strength_features(self):
        """
        Add relative strength features computed from macro data.

        Features:
        - RS_VS_SPY_20D: Relative strength vs SPY (20-day)
        - BETA_CHANGE_20D: Change in beta over 20 days (requires BETA60 from base features)
        """
        try:
            # Check if we have SPY data in macro
            if self._macro_df is None:
                return

            # Use macro_spy_pct_20d as proxy for SPY 20-day return
            if "macro_spy_pct_20d" not in self._macro_df.columns:
                return

            # Add to _learn
            if hasattr(self, "_learn") and self._learn is not None:
                self._learn = self._add_rs_features_to_df(self._learn)

            # Add to _infer
            if hasattr(self, "_infer") and self._infer is not None:
                self._infer = self._add_rs_features_to_df(self._infer)

        except Exception as e:
            print(f"Warning: Error adding relative strength features: {e}")

    def _add_rs_features_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add relative strength features to DataFrame."""
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        rs_data = {}

        # Get SPY 20-day return from macro
        spy_ret_20d = self._macro_df["macro_spy_pct_20d"].reindex(main_datetimes).values

        # Get stock 20-day return from ROC20 feature
        if has_multi_columns:
            if ('feature', 'ROC20') in df.columns:
                stock_ret_20d = df[('feature', 'ROC20')].values
                # ROC20 = Ref($close, 20)/$close, convert to return
                stock_ret_20d = 1.0 / stock_ret_20d - 1.0
                # Relative strength vs SPY
                rs_vs_spy = stock_ret_20d / (spy_ret_20d + 1e-12)
                rs_data[('feature', 'RS_VS_SPY_20D')] = rs_vs_spy
        else:
            if 'ROC20' in df.columns:
                stock_ret_20d = df['ROC20'].values
                stock_ret_20d = 1.0 / stock_ret_20d - 1.0
                rs_vs_spy = stock_ret_20d / (spy_ret_20d + 1e-12)
                rs_data['RS_VS_SPY_20D'] = rs_vs_spy

        # Beta change (if BETA60 exists)
        if has_multi_columns:
            if ('feature', 'BETA60') in df.columns:
                beta60 = df[('feature', 'BETA60')].values
                # Shift by grouping on instrument
                beta_shifted = df[('feature', 'BETA60')].groupby(level='instrument').shift(20).values
                beta_change = beta60 - beta_shifted
                rs_data[('feature', 'BETA_CHANGE_20D')] = beta_change
        else:
            if 'BETA60' in df.columns:
                beta60 = df['BETA60'].values
                beta_shifted = df['BETA60'].groupby(level='instrument').shift(20).values
                beta_change = beta60 - beta_shifted
                rs_data['BETA_CHANGE_20D'] = beta_change

        if rs_data:
            rs_df = pd.DataFrame(rs_data, index=df.index)
            merged = pd.concat([df, rs_df], axis=1, copy=False)
            return merged.copy()

        return df

    def _load_macro_features(self) -> Optional[pd.DataFrame]:
        """Load macro features from parquet file."""
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

    def get_label_config(self):
        """Return N-day return label."""
        label_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [label_expr], ["LABEL0"]
