"""
Enhanced DataHandler V2 with refined feature set based on second-round importance analysis.

Changes from V1:
- Removed features with importance < 0.05 (21 features removed)
- Added new high-potential features based on V1 analysis:
  - Extended 52-week features (best performing in V1)
  - Garman-Klass volatility
  - Volatility acceleration
  - Intraday/Overnight volatility split
  - Beta stability
  - Macro defensive vs cyclical indicator

Target: ~140-150 features

Based on V1 results:
- CV Mean IC: 0.0242 (±0.0051) - 10.5% improvement over baseline
- Top feature: PCT_FROM_52W_HIGH (2.93)
- New features performed well: PARKINSON_VOL_20, BETA_CHANGE_20D, ILLIQUIDITY_20D
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


class Alpha158_Enhanced_V2(DataHandlerLP):
    """
    Enhanced Alpha158 V2 with second-round feature importance-based refinement.

    Removed (importance < 0.05 in V1):
    - Kbar: KUP, KUP2, KLOW2, KLEN, KMID2, KSFT2
    - Price: OPEN0, LOW0
    - Rolling: ROC20, RSV20, SUMP20, SUMP60, SUMN20, SUMD20
    - TA-Lib: TALIB_RSI14, TALIB_CMO14, TALIB_MOM10, TALIB_ROC10, TALIB_TRANGE
    - Other: SHARPE_20D, VOLUME_ZSCORE

    Added new features:
    - 52-week extensions: PCT_52W_HIGH_CHG5, PCT_FROM_26W_HIGH, PCT_FROM_13W_HIGH,
      BREAK_52W_HIGH, BREAK_52W_LOW, DAYS_FROM_52W_HIGH
    - Volatility: GK_VOL_20, VOL_ACCEL, INTRADAY_VOL, OVERNIGHT_VOL
    - Beta: BETA_STABILITY
    - Macro: defensive_vs_cyclical, sector_dispersion (computed in process_data)

    Total: ~140-150 features
    """

    # High-importance macro features (same as V1, all performed well)
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
        """
        Initialize Enhanced V2 DataHandler.

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
        """Get feature config with V2 refinements."""
        fields = []
        names = []

        # 1. Refined Alpha158 features (removed low-importance)
        alpha_fields, alpha_names = self._get_refined_alpha158_features_v2()
        fields.extend(alpha_fields)
        names.extend(alpha_names)

        # 2. Enhanced volatility features V2 (added new ones)
        vol_fields, vol_names = self._get_enhanced_volatility_features_v2()
        fields.extend(vol_fields)
        names.extend(vol_names)

        # 3. High-importance TA-Lib features (removed low-importance)
        talib_fields, talib_names = self._get_talib_features_v2()
        fields.extend(talib_fields)
        names.extend(talib_names)

        # 4. Extended 52-week features (new in V2)
        week52_fields, week52_names = self._get_extended_52w_features()
        fields.extend(week52_fields)
        names.extend(week52_names)

        # 5. Momentum quality features (removed SHARPE_20D)
        mom_fields, mom_names = self._get_momentum_quality_features_v2()
        fields.extend(mom_fields)
        names.extend(mom_names)

        # 6. Liquidity features (removed VOLUME_ZSCORE)
        liq_fields, liq_names = self._get_liquidity_features_v2()
        fields.extend(liq_fields)
        names.extend(liq_names)

        return fields, names

    def _get_refined_alpha158_features_v2(self):
        """
        Get refined Alpha158 features V2.

        Removed (importance < 0.05):
        - Kbar: KUP (0.034), KUP2 (0.007), KLOW2 (0.012), KLEN (0.036), KMID2 (0.039), KSFT2 (0.012)
        - Price: OPEN0 (0.038), LOW0 (0.035)
        - Rolling: ROC20 (0.041), RSV20 (0.047), SUMP20 (0.034), SUMP60 (0.033), SUMN20 (0.013), SUMD20 (0.026)
        """
        fields = []
        names = []

        # Kbar features - kept only high importance ones (3 instead of 9)
        fields.extend([
            "($close-$open)/$open",  # KMID - 0.065
            "($low-Less($open, $close))/$open",  # KLOW - 0.082
            "(2*$close-$high-$low)/$open",  # KSFT - 0.100
        ])
        names.extend(["KMID", "KLOW", "KSFT"])

        # Price features - only HIGH0 (importance 0.077)
        fields.append("$high/$close")
        names.append("HIGH0")

        # Rolling features - refined selection
        # ROC - removed ROC20, keep 5, 10, 30, 60
        for d in [5, 10, 30, 60]:
            fields.append(f"Ref($close, {d})/$close")
            names.append(f"ROC{d}")

        # MA (5 windows) - all kept
        for d in [5, 10, 20, 30, 60]:
            fields.append(f"Mean($close, {d})/$close")
            names.append(f"MA{d}")

        # STD - only 20, 60
        for d in [20, 60]:
            fields.append(f"Std($close, {d})/$close")
            names.append(f"STD{d}")

        # BETA - only 20, 60
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

        # MAX - only 20, 60
        for d in [20, 60]:
            fields.append(f"Max($high, {d})/$close")
            names.append(f"MAX{d}")

        # MIN - only 20, 60
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

        # RSV - only 60 (RSV20 removed, importance 0.047)
        fields.append("($close-Min($low, 60))/(Max($high, 60)-Min($low, 60)+1e-12)")
        names.append("RSV60")

        # IMAX - only 20, 60
        for d in [20, 60]:
            fields.append(f"IdxMax($high, {d})/{d}")
            names.append(f"IMAX{d}")

        # IMIN - only 20, 60
        for d in [20, 60]:
            fields.append(f"IdxMin($low, {d})/{d}")
            names.append(f"IMIN{d}")

        # IMXD - only 60
        fields.append("(IdxMax($high, 60)-IdxMin($low, 60))/60")
        names.append("IMXD60")

        # CORR - only 20, 60
        for d in [20, 60]:
            fields.append(f"Corr($close, Log($volume+1), {d})")
            names.append(f"CORR{d}")

        # CORD - only 20, 60
        for d in [20, 60]:
            fields.append(f"Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {d})")
            names.append(f"CORD{d}")

        # CNTP, CNTN, CNTD - only 60
        fields.append("Mean($close>Ref($close, 1), 60)")
        names.append("CNTP60")

        fields.append("Mean($close<Ref($close, 1), 60)")
        names.append("CNTN60")

        fields.append("Mean($close>Ref($close, 1), 60)-Mean($close<Ref($close, 1), 60)")
        names.append("CNTD60")

        # SUMP, SUMN, SUMD - only 60 (20 removed due to low importance)
        fields.append("Sum(Greater($close-Ref($close, 1), 0), 60)/(Sum(Abs($close-Ref($close, 1)), 60)+1e-12)")
        names.append("SUMP60")

        fields.append("Sum(Greater(Ref($close, 1)-$close, 0), 60)/(Sum(Abs($close-Ref($close, 1)), 60)+1e-12)")
        names.append("SUMN60")

        fields.append("(Sum(Greater($close-Ref($close, 1), 0), 60)-Sum(Greater(Ref($close, 1)-$close, 0), 60))/(Sum(Abs($close-Ref($close, 1)), 60)+1e-12)")
        names.append("SUMD60")

        # WVMA - only 20, 60
        for d in [20, 60]:
            fields.append(f"Std(Abs($close/Ref($close, 1)-1)*$volume, {d})/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {d})+1e-12)")
            names.append(f"WVMA{d}")

        return fields, names

    def _get_enhanced_volatility_features_v2(self):
        """
        Get enhanced volatility features V2.

        Kept from V1:
        - VOL_5D, VOL_10D, VOL_RATIO_5_20
        - PARKINSON_VOL_20 (importance 1.11)
        - VOL_MOMENTUM
        - UPSIDE_VOL_20, DOWNSIDE_VOL_20, VOL_SKEW

        New in V2:
        - GK_VOL_20: Garman-Klass volatility (more accurate)
        - VOL_ACCEL: Volatility acceleration
        - INTRADAY_VOL: Intraday volatility (high-low range)
        - OVERNIGHT_VOL: Overnight volatility (open vs prev close)
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

        # Parkinson volatility estimator
        fields.append("Power(Mean(Power(Log($high/$low+1e-12), 2), 20) / 0.6931, 0.5)")
        names.append("PARKINSON_VOL_20")

        # Garman-Klass volatility (NEW in V2) - more accurate than Parkinson
        # GK = 0.5 * (ln(H/L))^2 - (2*ln(2)-1) * (ln(C/O))^2
        fields.append("Power(Mean(0.5*Power(Log($high/$low+1e-12), 2) - 0.386*Power(Log($close/$open+1e-12), 2), 20), 0.5)")
        names.append("GK_VOL_20")

        # Volatility momentum
        fields.append("Std($close/Ref($close,1)-1, 5) - Ref(Std($close/Ref($close,1)-1, 5), 5)")
        names.append("VOL_MOMENTUM")

        # Volatility acceleration (NEW in V2) - second derivative of volatility
        fields.append("Std($close/Ref($close,1)-1, 5) - 2*Ref(Std($close/Ref($close,1)-1, 5), 5) + Ref(Std($close/Ref($close,1)-1, 5), 10)")
        names.append("VOL_ACCEL")

        # Upside volatility
        fields.append("Std(Max($close/Ref($close,1)-1, 0), 20)")
        names.append("UPSIDE_VOL_20")

        # Downside volatility
        fields.append("Std(Abs(Min($close/Ref($close,1)-1, 0)), 20)")
        names.append("DOWNSIDE_VOL_20")

        # Volatility skew
        fields.append("Std(Max($close/Ref($close,1)-1, 0), 20) / (Std(Abs(Min($close/Ref($close,1)-1, 0)), 20) + 1e-12)")
        names.append("VOL_SKEW")

        # Intraday volatility (NEW in V2) - based on high-low range
        fields.append("Std(($high-$low)/$open, 20)")
        names.append("INTRADAY_VOL")

        # Overnight volatility (NEW in V2) - based on open vs previous close
        fields.append("Std($open/Ref($close,1)-1, 20)")
        names.append("OVERNIGHT_VOL")

        return fields, names

    def _get_talib_features_v2(self):
        """
        Get high-importance TA-Lib features V2.

        Removed (importance < 0.05):
        - TALIB_RSI14 (0.017)
        - TALIB_CMO14 (0.015)
        - TALIB_MOM10 (0.043)
        - TALIB_ROC10 (0.035)
        - TALIB_TRANGE (0.028)
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

        # Stochastic
        fields.append("TALIB_STOCH_K($high, $low, $close, 5, 3, 3)")
        names.append("TALIB_STOCH_K")

        fields.append("TALIB_STOCH_D($high, $low, $close, 5, 3, 3)")
        names.append("TALIB_STOCH_D")

        # Statistics
        fields.append("TALIB_STDDEV($close, 20, 1)/$close")
        names.append("TALIB_STDDEV20")

        return fields, names

    def _get_extended_52w_features(self):
        """
        Get extended 52-week features (NEW in V2).

        PCT_FROM_52W_HIGH was the #1 feature in V1 (importance 2.93).
        Adding more variants to capture this signal better.
        """
        fields = []
        names = []

        # Original 52-week features (kept from V1)
        fields.append("($close - Max($high, 252)) / (Max($high, 252) + 1e-12)")
        names.append("PCT_FROM_52W_HIGH")

        fields.append("($close - Min($low, 252)) / (Min($low, 252) + 1e-12)")
        names.append("PCT_FROM_52W_LOW")

        # 52-week high position change over 5 days (NEW)
        fields.append("(($close - Max($high, 252)) / (Max($high, 252) + 1e-12)) - Ref(($close - Max($high, 252)) / (Max($high, 252) + 1e-12), 5)")
        names.append("PCT_52W_HIGH_CHG5")

        # 26-week (half year) high/low position (NEW)
        fields.append("($close - Max($high, 126)) / (Max($high, 126) + 1e-12)")
        names.append("PCT_FROM_26W_HIGH")

        fields.append("($close - Min($low, 126)) / (Min($low, 126) + 1e-12)")
        names.append("PCT_FROM_26W_LOW")

        # 13-week (quarter) high/low position (NEW)
        fields.append("($close - Max($high, 65)) / (Max($high, 65) + 1e-12)")
        names.append("PCT_FROM_13W_HIGH")

        fields.append("($close - Min($low, 65)) / (Min($low, 65) + 1e-12)")
        names.append("PCT_FROM_13W_LOW")

        # Breakout signals (NEW)
        # 1 if price breaks above 52-week high, 0 otherwise
        fields.append("If($close > Ref(Max($high, 252), 1), 1, 0)")
        names.append("BREAK_52W_HIGH")

        # 1 if price breaks below 52-week low, 0 otherwise
        fields.append("If($close < Ref(Min($low, 252), 1), 1, 0)")
        names.append("BREAK_52W_LOW")

        # Days since 52-week high (NEW) - normalized to 0-1
        fields.append("IdxMax($high, 252) / 252")
        names.append("DAYS_FROM_52W_HIGH")

        # Days since 52-week low (NEW)
        fields.append("IdxMin($low, 252) / 252")
        names.append("DAYS_FROM_52W_LOW")

        return fields, names

    def _get_momentum_quality_features_v2(self):
        """
        Get momentum quality features V2.

        Removed: SHARPE_20D (importance 0.039)
        Kept: SHARPE_60D (importance 0.291)
        """
        fields = []
        names = []

        # Sharpe-like ratio (60-day only)
        fields.append("Mean($close/Ref($close,1)-1, 60) / (Std($close/Ref($close,1)-1, 60) + 1e-12)")
        names.append("SHARPE_60D")

        return fields, names

    def _get_liquidity_features_v2(self):
        """
        Get liquidity features V2.

        Removed: VOLUME_ZSCORE (importance 0.043)
        Kept: VOLUME_TREND, DOLLAR_VOL_RATIO, ILLIQUIDITY_20D
        """
        fields = []
        names = []

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
        """Override process_data to add macro features and computed features."""
        super().process_data(with_fit=with_fit)

        # Add macro features
        if self._macro_df is not None:
            self._add_macro_features()

        # Add computed features (relative strength, beta stability, etc.)
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
                # Add computed macro features
                self._learn = self._add_computed_macro_features(self._learn)
                print(f"Added {len(available_cols)} macro features to learn data")

            # Add to _infer
            if hasattr(self, "_infer") and self._infer is not None:
                self._infer = self._merge_macro_to_df(self._infer, available_cols)
                self._infer = self._add_computed_macro_features(self._infer)

        except Exception as e:
            print(f"Warning: Error adding macro features: {e}")

    def _add_computed_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed macro features like defensive vs cyclical."""
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        computed_data = {}

        # Defensive vs Cyclical indicator (NEW in V2)
        # Defensive: XLU, XLP, XLV
        # Cyclical: XLY, XLI, XLF
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

        # Sector dispersion (NEW in V2)
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
        """Add computed features like relative strength and beta stability."""
        try:
            if self._macro_df is None:
                return

            if "macro_spy_pct_20d" not in self._macro_df.columns:
                return

            # Add to _learn
            if hasattr(self, "_learn") and self._learn is not None:
                self._learn = self._add_rs_and_beta_features(self._learn)

            # Add to _infer
            if hasattr(self, "_infer") and self._infer is not None:
                self._infer = self._add_rs_and_beta_features(self._infer)

        except Exception as e:
            print(f"Warning: Error adding computed features: {e}")

    def _add_rs_and_beta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add relative strength and beta stability features."""
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        rs_data = {}

        # Get SPY 20-day return from macro
        spy_ret_20d = self._macro_df["macro_spy_pct_20d"].reindex(main_datetimes).values

        # Relative strength vs SPY
        roc_col = ('feature', 'ROC30') if has_multi_columns else 'ROC30'
        if roc_col in df.columns:
            stock_ret = df[roc_col].values
            stock_ret = 1.0 / stock_ret - 1.0  # Convert ROC to return
            rs_vs_spy = stock_ret / (spy_ret_20d + 1e-12)
            key = ('feature', 'RS_VS_SPY_20D') if has_multi_columns else 'RS_VS_SPY_20D'
            rs_data[key] = rs_vs_spy

        # Beta change
        beta_col = ('feature', 'BETA60') if has_multi_columns else 'BETA60'
        if beta_col in df.columns:
            beta60 = df[beta_col].values
            beta_shifted = df[beta_col].groupby(level='instrument').shift(20).values
            beta_change = beta60 - beta_shifted
            key = ('feature', 'BETA_CHANGE_20D') if has_multi_columns else 'BETA_CHANGE_20D'
            rs_data[key] = beta_change

            # Beta stability (NEW in V2) - rolling std of beta
            beta_std = df[beta_col].groupby(level='instrument').rolling(60, min_periods=20).std().values
            # Flatten the multi-index result
            if hasattr(beta_std, 'values'):
                beta_std = beta_std.values
            key = ('feature', 'BETA_STABILITY') if has_multi_columns else 'BETA_STABILITY'
            rs_data[key] = beta_std

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
