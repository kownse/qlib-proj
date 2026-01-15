"""
Enhanced DataHandler V3 - Streamlined version based on V2 feature importance.

Changes from V2:
- Removed features with importance < 0.1 (~20 features removed)
- Removed ineffective features: BREAK_52W_HIGH/LOW (importance=0), VOL_ACCEL, BETA_STABILITY
- Streamlined 52-week features: kept only PCT_FROM_52W_HIGH, PCT_FROM_52W_LOW, DAYS_FROM_52W_HIGH
- Focus on proven high-value features

Target: ~130 features (down from 152 in V2)

Key findings from V2:
- macro_sector_dispersion (#5, 2.19) - best new feature
- macro_defensive_vs_cyclical (#22, 1.42) - strong performer
- GK_VOL_20 (#52, 0.85) - valuable volatility measure
- OVERNIGHT_VOL (#66, 0.55) - useful
- Binary features (BREAK_*) completely ineffective
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


class Alpha158_Enhanced_V3(DataHandlerLP):
    """
    Enhanced Alpha158 V3 - Streamlined version with ~130 proven features.

    Removed from V2 (importance < 0.1 or ineffective):
    - BREAK_52W_HIGH, BREAK_52W_LOW (importance = 0)
    - VOL_ACCEL, BETA_STABILITY (< 0.03)
    - PCT_52W_HIGH_CHG5, PCT_FROM_13W_LOW (< 0.11)
    - STD20, MIN20, MA30, IMIN20, QTLU20, MA20, ROC5, HIGH0, SUMN60 (< 0.1)
    - TALIB_BB_WIDTH, TALIB_MACD, TALIB_STDDEV20 (< 0.1)
    - DOLLAR_VOL_RATIO, VOL_MOMENTUM (< 0.1)
    - SUMP60, CNTD60, SUMD60 (< 0.1)

    Streamlined 52-week features:
    - Kept: PCT_FROM_52W_HIGH (#4), PCT_FROM_52W_LOW (#61), DAYS_FROM_52W_HIGH (#51)
    - Removed: PCT_FROM_26W_*, PCT_FROM_13W_*, DAYS_FROM_52W_LOW, PCT_52W_HIGH_CHG5, BREAK_*

    Total: ~130 features
    """

    # High-importance macro features (kept all from V2 as they all performed well)
    SELECTED_MACRO_FEATURES = [
        # VIX features
        "macro_vix_term_structure", "macro_vix_level", "macro_vix_zscore20",
        "macro_vix_pct_5d", "macro_vix_pct_10d", "macro_vix_term_zscore",
        "macro_vix_ma5_ratio", "macro_vix_ma20_ratio",
        # Asset volatility
        "macro_gld_vol20", "macro_spy_vol20", "macro_bond_vol20",
        "macro_uso_vol20", "macro_jnk_vol20",
        # Credit spreads
        "macro_ig_spread", "macro_hy_spread_zscore", "macro_hy_spread",
        "macro_hy_spread_chg5", "macro_credit_risk", "macro_credit_stress",
        # Yield curve
        "macro_yield_curve_zscore", "macro_yield_2y", "macro_yield_3m10y",
        "macro_yield_10y", "macro_yield_2s10s", "macro_yield_curve_slope",
        "macro_yield_10y_chg20", "macro_yield_curve_chg5",
        # Sector rotation
        "macro_xly_pct_20d", "macro_xly_pct_5d", "macro_xly_vs_spy",
        "macro_xlu_pct_20d", "macro_xlu_pct_5d", "macro_xlu_vs_spy",
        "macro_xlf_pct_20d", "macro_xlf_vs_spy",
        "macro_xle_pct_20d", "macro_xle_vs_spy",
        "macro_xlk_pct_20d", "macro_xlk_vs_spy",
        "macro_xlv_pct_20d", "macro_xlv_pct_5d",
        "macro_xli_pct_20d", "macro_xlp_pct_20d",
        # Market sentiment
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
        macro_data_path: Union[str, Path] = None,
        **kwargs,
    ):
        """Initialize Enhanced V3 DataHandler."""
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
        """Get feature config with V3 streamlined features."""
        fields = []
        names = []

        # 1. Streamlined Alpha158 features
        alpha_fields, alpha_names = self._get_streamlined_alpha158_features()
        fields.extend(alpha_fields)
        names.extend(alpha_names)

        # 2. Streamlined volatility features
        vol_fields, vol_names = self._get_streamlined_volatility_features()
        fields.extend(vol_fields)
        names.extend(vol_names)

        # 3. Streamlined TA-Lib features
        talib_fields, talib_names = self._get_streamlined_talib_features()
        fields.extend(talib_fields)
        names.extend(talib_names)

        # 4. Streamlined 52-week features (only proven ones)
        week52_fields, week52_names = self._get_streamlined_52w_features()
        fields.extend(week52_fields)
        names.extend(week52_names)

        # 5. Streamlined momentum/liquidity features
        other_fields, other_names = self._get_streamlined_other_features()
        fields.extend(other_fields)
        names.extend(other_names)

        return fields, names

    def _get_streamlined_alpha158_features(self):
        """
        Get streamlined Alpha158 features.

        Removed (importance < 0.1):
        - STD20 (0.094), MIN20 (0.093), MA30 (0.089), IMIN20 (0.087)
        - QTLU20 (0.084), MA20 (0.076), ROC5 (0.041), HIGH0 (0.053), SUMN60 (0.026)
        - SUMP60 (0.068), CNTD60 (0.067), SUMD60 (0.063)
        """
        fields = []
        names = []

        # Kbar features - only 3 with importance > 0.1
        fields.extend([
            "($close-$open)/$open",  # KMID - 0.117
            "($low-Less($open, $close))/$open",  # KLOW - 0.108
            "(2*$close-$high-$low)/$open",  # KSFT - 0.114
        ])
        names.extend(["KMID", "KLOW", "KSFT"])

        # No price features (HIGH0 = 0.053 < 0.1)

        # ROC - removed ROC5 (0.041), ROC20 already removed in V2
        for d in [10, 30, 60]:
            fields.append(f"Ref($close, {d})/$close")
            names.append(f"ROC{d}")

        # MA - removed MA20 (0.076), MA30 (0.089)
        for d in [5, 10, 60]:
            fields.append(f"Mean($close, {d})/$close")
            names.append(f"MA{d}")

        # STD - only 60 (STD20 = 0.094 < 0.1)
        fields.append("Std($close, 60)/$close")
        names.append("STD60")

        # BETA - only 60 (BETA20 = 0.153 > 0.1, keep it)
        for d in [20, 60]:
            fields.append(f"Slope($close, {d})/Ref($close, {d})")
            names.append(f"BETA{d}")

        # RSQR - both kept (RSQR60 = 0.358, RSQR20 = 0.136)
        for d in [20, 60]:
            fields.append(f"Rsquare($close, {d})")
            names.append(f"RSQR{d}")

        # RESI - both kept (RESI60 = 0.167, RESI20 = 0.226)
        for d in [20, 60]:
            fields.append(f"Resi($close, {d})/$close")
            names.append(f"RESI{d}")

        # MAX - only 60 (MAX20 = 0.115, MAX60 = 0.230)
        for d in [20, 60]:
            fields.append(f"Max($high, {d})/$close")
            names.append(f"MAX{d}")

        # MIN - only 60 (MIN20 = 0.093 < 0.1)
        fields.append("Min($low, 60)/$close")
        names.append("MIN60")

        # QTLU - only 60 (QTLU20 = 0.084 < 0.1)
        fields.append("Quantile($close, 60, 0.8)/$close")
        names.append("QTLU60")

        # QTLD - both kept (QTLD60 = 0.124, QTLD20 = 0.116)
        for d in [20, 60]:
            fields.append(f"Quantile($close, {d}, 0.2)/$close")
            names.append(f"QTLD{d}")

        # RSV - only 60
        fields.append("($close-Min($low, 60))/(Max($high, 60)-Min($low, 60)+1e-12)")
        names.append("RSV60")

        # IMAX - both kept (IMAX60 = 0.283, IMAX20 = 0.245)
        for d in [20, 60]:
            fields.append(f"IdxMax($high, {d})/{d}")
            names.append(f"IMAX{d}")

        # IMIN - only 60 (IMIN20 = 0.087 < 0.1)
        fields.append("IdxMin($low, 60)/60")
        names.append("IMIN60")

        # IMXD - only 60
        fields.append("(IdxMax($high, 60)-IdxMin($low, 60))/60")
        names.append("IMXD60")

        # CORR - both kept (CORR60 = 0.452, CORR20 = 0.104)
        for d in [20, 60]:
            fields.append(f"Corr($close, Log($volume+1), {d})")
            names.append(f"CORR{d}")

        # CORD - both kept (CORD60 = 0.323, CORD20 = 0.178)
        for d in [20, 60]:
            fields.append(f"Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {d})")
            names.append(f"CORD{d}")

        # CNTP, CNTN - only 60 (CNTP60 = 0.197, CNTN60 = 0.118)
        fields.append("Mean($close>Ref($close, 1), 60)")
        names.append("CNTP60")

        fields.append("Mean($close<Ref($close, 1), 60)")
        names.append("CNTN60")

        # Removed CNTD60, SUMP60, SUMN60, SUMD60 (all < 0.1)

        # WVMA - both kept (WVMA60 = 0.392, WVMA20 = 0.206)
        for d in [20, 60]:
            fields.append(f"Std(Abs($close/Ref($close, 1)-1)*$volume, {d})/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {d})+1e-12)")
            names.append(f"WVMA{d}")

        return fields, names

    def _get_streamlined_volatility_features(self):
        """
        Get streamlined volatility features.

        Kept (importance > 0.1):
        - VOL_10D (0.344), VOL_5D (0.109), VOL_RATIO_5_20 (0.126)
        - PARKINSON_VOL_20 (0.670), GK_VOL_20 (0.851)
        - UPSIDE_VOL_20 (0.252), DOWNSIDE_VOL_20 (0.210), VOL_SKEW (0.147)
        - OVERNIGHT_VOL (0.555), INTRADAY_VOL (0.296)

        Removed:
        - VOL_MOMENTUM (0.080 < 0.1)
        - VOL_ACCEL (0.061 < 0.1)
        """
        fields = []
        names = []

        # Multi-period realized volatility
        fields.append("Std($close/Ref($close,1)-1, 5)")
        names.append("VOL_5D")

        fields.append("Std($close/Ref($close,1)-1, 10)")
        names.append("VOL_10D")

        # Volatility ratio
        fields.append("Std($close/Ref($close,1)-1, 5) / (Std($close/Ref($close,1)-1, 20) + 1e-12)")
        names.append("VOL_RATIO_5_20")

        # Parkinson volatility
        fields.append("Power(Mean(Power(Log($high/$low+1e-12), 2), 20) / 0.6931, 0.5)")
        names.append("PARKINSON_VOL_20")

        # Garman-Klass volatility
        fields.append("Power(Mean(0.5*Power(Log($high/$low+1e-12), 2) - 0.386*Power(Log($close/$open+1e-12), 2), 20), 0.5)")
        names.append("GK_VOL_20")

        # Upside/Downside volatility
        fields.append("Std(Max($close/Ref($close,1)-1, 0), 20)")
        names.append("UPSIDE_VOL_20")

        fields.append("Std(Abs(Min($close/Ref($close,1)-1, 0)), 20)")
        names.append("DOWNSIDE_VOL_20")

        # Volatility skew
        fields.append("Std(Max($close/Ref($close,1)-1, 0), 20) / (Std(Abs(Min($close/Ref($close,1)-1, 0)), 20) + 1e-12)")
        names.append("VOL_SKEW")

        # Intraday/Overnight volatility
        fields.append("Std(($high-$low)/$open, 20)")
        names.append("INTRADAY_VOL")

        fields.append("Std($open/Ref($close,1)-1, 20)")
        names.append("OVERNIGHT_VOL")

        return fields, names

    def _get_streamlined_talib_features(self):
        """
        Get streamlined TA-Lib features.

        Kept (importance > 0.1):
        - TALIB_ATR14 (1.353), TALIB_NATR14 (1.455)
        - TALIB_ADX14 (0.375)
        - TALIB_PLUS_DI14 (0.104), TALIB_MINUS_DI14 (0.142)
        - TALIB_MACD_HIST (0.275), TALIB_MACD_SIGNAL (0.269)
        - TALIB_BB_LOWER_DIST (0.171)
        - TALIB_STOCH_K (0.142), TALIB_STOCH_D (0.116)

        Removed:
        - TALIB_BB_WIDTH (0.085 < 0.1)
        - TALIB_MACD (0.083 < 0.1)
        - TALIB_STDDEV20 (0.033 < 0.1)
        """
        fields = []
        names = []

        # Volatility (highest importance)
        fields.append("TALIB_ATR($high, $low, $close, 14)/$close")
        names.append("TALIB_ATR14")

        fields.append("TALIB_NATR($high, $low, $close, 14)")
        names.append("TALIB_NATR14")

        # Trend indicators
        fields.append("TALIB_ADX($high, $low, $close, 14)")
        names.append("TALIB_ADX14")

        fields.append("TALIB_PLUS_DI($high, $low, $close, 14)")
        names.append("TALIB_PLUS_DI14")

        fields.append("TALIB_MINUS_DI($high, $low, $close, 14)")
        names.append("TALIB_MINUS_DI14")

        # MACD (only HIST and SIGNAL, not MACD itself)
        fields.append("TALIB_MACD_HIST($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD_HIST")

        fields.append("TALIB_MACD_SIGNAL($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD_SIGNAL")

        # Bollinger Bands (only LOWER_DIST)
        fields.append("($close - TALIB_BBANDS_LOWER($close, 20, 2))/$close")
        names.append("TALIB_BB_LOWER_DIST")

        # Stochastic
        fields.append("TALIB_STOCH_K($high, $low, $close, 5, 3, 3)")
        names.append("TALIB_STOCH_K")

        fields.append("TALIB_STOCH_D($high, $low, $close, 5, 3, 3)")
        names.append("TALIB_STOCH_D")

        return fields, names

    def _get_streamlined_52w_features(self):
        """
        Get streamlined 52-week features.

        Kept (proven high-value):
        - PCT_FROM_52W_HIGH (2.207) - #4 overall
        - PCT_FROM_52W_LOW (0.677) - #61
        - DAYS_FROM_52W_HIGH (0.873) - #51

        Removed (lower value or redundant):
        - PCT_52W_HIGH_CHG5 (0.102)
        - PCT_FROM_26W_HIGH (0.693), PCT_FROM_26W_LOW (0.469)
        - PCT_FROM_13W_HIGH (0.398), PCT_FROM_13W_LOW (0.208)
        - DAYS_FROM_52W_LOW (0.505) - correlated with DAYS_FROM_52W_HIGH
        - BREAK_52W_HIGH (0.000), BREAK_52W_LOW (0.000) - completely ineffective
        """
        fields = []
        names = []

        # Core 52-week features (proven high value)
        fields.append("($close - Max($high, 252)) / (Max($high, 252) + 1e-12)")
        names.append("PCT_FROM_52W_HIGH")

        fields.append("($close - Min($low, 252)) / (Min($low, 252) + 1e-12)")
        names.append("PCT_FROM_52W_LOW")

        fields.append("IdxMax($high, 252) / 252")
        names.append("DAYS_FROM_52W_HIGH")

        return fields, names

    def _get_streamlined_other_features(self):
        """
        Get streamlined momentum/liquidity features.

        Kept:
        - SHARPE_60D (0.413)
        - VOLUME_TREND (0.219)
        - ILLIQUIDITY_20D (0.767)

        Removed:
        - DOLLAR_VOL_RATIO (0.072 < 0.1)
        """
        fields = []
        names = []

        # Sharpe ratio (60-day only)
        fields.append("Mean($close/Ref($close,1)-1, 60) / (Std($close/Ref($close,1)-1, 60) + 1e-12)")
        names.append("SHARPE_60D")

        # Volume trend
        fields.append("Mean($volume, 5) / (Mean($volume, 20) + 1e-12)")
        names.append("VOLUME_TREND")

        # Illiquidity
        fields.append("Mean(Abs($close/Ref($close,1)-1) / (Log($volume+1)+1e-12), 20)")
        names.append("ILLIQUIDITY_20D")

        return fields, names

    def process_data(self, with_fit: bool = False):
        """Override process_data to add macro features and computed features."""
        super().process_data(with_fit=with_fit)

        # Add macro features
        if self._macro_df is not None:
            self._add_macro_features()

        # Add computed features
        self._add_computed_features()

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
                self._learn = self._add_computed_macro_features(self._learn)
                print(f"Added {len(available_cols)} macro features to learn data")

            # Add to _infer
            if hasattr(self, "_infer") and self._infer is not None:
                self._infer = self._merge_macro_to_df(self._infer, available_cols)
                self._infer = self._add_computed_macro_features(self._infer)

        except Exception as e:
            print(f"Warning: Error adding macro features: {e}")

    def _add_computed_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed macro features (sector_dispersion, defensive_vs_cyclical)."""
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        computed_data = {}

        # Defensive vs Cyclical indicator (#22 in V2, importance 1.42)
        defensive_cols = ["macro_xlu_pct_20d", "macro_xlp_pct_20d", "macro_xlv_pct_20d"]
        cyclical_cols = ["macro_xly_pct_20d", "macro_xli_pct_20d", "macro_xlf_pct_20d"]

        if all(c in self._macro_df.columns for c in defensive_cols + cyclical_cols):
            defensive = sum(self._macro_df[c] for c in defensive_cols) / 3
            cyclical = sum(self._macro_df[c] for c in cyclical_cols) / 3
            defensive_vs_cyclical = defensive - cyclical

            aligned = defensive_vs_cyclical.reindex(main_datetimes).values
            if has_multi_columns:
                computed_data[('feature', 'macro_defensive_vs_cyclical')] = aligned
            else:
                computed_data['macro_defensive_vs_cyclical'] = aligned

        # Sector dispersion (#5 in V2, importance 2.19) - BEST NEW FEATURE!
        sector_pct_cols = [c for c in self._macro_df.columns if c.endswith('_pct_20d') and c.startswith('macro_xl')]
        if len(sector_pct_cols) >= 5:
            sector_df = self._macro_df[sector_pct_cols]
            sector_dispersion = sector_df.std(axis=1)

            aligned = sector_dispersion.reindex(main_datetimes).values
            if has_multi_columns:
                computed_data[('feature', 'macro_sector_dispersion')] = aligned
            else:
                computed_data['macro_sector_dispersion'] = aligned

        if computed_data:
            computed_df = pd.DataFrame(computed_data, index=df.index)
            merged = pd.concat([df, computed_df], axis=1, copy=False)
            return merged.copy()

        return df

    def _merge_macro_to_df(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Merge macro features into a DataFrame using pd.concat."""
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        macro_data = {}
        for col in cols:
            macro_series = self._macro_df[col]
            aligned_values = macro_series.reindex(main_datetimes).values

            if has_multi_columns:
                macro_data[('feature', col)] = aligned_values
            else:
                macro_data[col] = aligned_values

        macro_df = pd.DataFrame(macro_data, index=df.index)
        merged = pd.concat([df, macro_df], axis=1, copy=False)
        return merged.copy()

    def _add_computed_features(self):
        """Add computed features like relative strength and beta change."""
        try:
            if self._macro_df is None:
                return

            if "macro_spy_pct_20d" not in self._macro_df.columns:
                return

            # Add to _learn
            if hasattr(self, "_learn") and self._learn is not None:
                self._learn = self._add_rs_features(self._learn)

            # Add to _infer
            if hasattr(self, "_infer") and self._infer is not None:
                self._infer = self._add_rs_features(self._infer)

        except Exception as e:
            print(f"Warning: Error adding computed features: {e}")

    def _add_rs_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add relative strength and beta change features."""
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        rs_data = {}

        # Get SPY 20-day return from macro
        spy_ret_20d = self._macro_df["macro_spy_pct_20d"].reindex(main_datetimes).values

        # Relative strength vs SPY (importance 0.191)
        roc_col = ('feature', 'ROC30') if has_multi_columns else 'ROC30'
        if roc_col in df.columns:
            stock_ret = df[roc_col].values
            stock_ret = 1.0 / stock_ret - 1.0
            rs_vs_spy = stock_ret / (spy_ret_20d + 1e-12)
            key = ('feature', 'RS_VS_SPY_20D') if has_multi_columns else 'RS_VS_SPY_20D'
            rs_data[key] = rs_vs_spy

        # Beta change (importance 0.799)
        beta_col = ('feature', 'BETA60') if has_multi_columns else 'BETA60'
        if beta_col in df.columns:
            beta60 = df[beta_col].values
            beta_shifted = df[beta_col].groupby(level='instrument').shift(20).values
            beta_change = beta60 - beta_shifted
            key = ('feature', 'BETA_CHANGE_20D') if has_multi_columns else 'BETA_CHANGE_20D'
            rs_data[key] = beta_change

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
