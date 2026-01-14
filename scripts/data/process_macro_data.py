"""
Process macro CSV data into features for time series prediction

Usage:
    python process_macro_data.py                     # Process with defaults
    python process_macro_data.py --output ./my_path  # Custom output path
    python process_macro_data.py --feature-set core  # Generate core features only

Generated features (~105 total):
- VIX features: level, zscore, pct changes, ma ratios, regime, term structure
- Gold features: pct changes, ma ratio, volatility
- Bond features: pct changes, yield curve proxy, volatility
- Dollar features: pct changes, strength indicator
- Oil features: pct changes, volatility
- Sector features: momentum, relative strength vs SPY
- Credit features: spreads, stress indicators
- Global features: EM momentum, global risk
- Treasury yield features: levels, curve slope, changes
- Cross-asset features: risk indicators, ratios, correlations
"""

import argparse
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
MACRO_CSV_DIR = PROJECT_ROOT / "my_data" / "macro_csv"
MACRO_OUTPUT_DIR = PROJECT_ROOT / "my_data" / "macro_processed"
DEFAULT_OUTPUT_PATH = MACRO_OUTPUT_DIR / "macro_features.parquet"


class MacroFeatureProcessor:
    """Generate macro features from raw ETF/index data."""

    # ==========================================================================
    # Feature column names for each category
    # ==========================================================================

    # Original VIX features (8)
    VIX_FEATURES = [
        "macro_vix_level",
        "macro_vix_zscore20",
        "macro_vix_pct_1d",
        "macro_vix_pct_5d",
        "macro_vix_pct_10d",
        "macro_vix_ma5_ratio",
        "macro_vix_ma20_ratio",
        "macro_vix_regime",
    ]

    # VIX term structure features (5)
    VIX_TERM_FEATURES = [
        "macro_vix_term_structure",    # VIX / VIX3M ratio
        "macro_vix_contango",          # Binary: contango (1) or backwardation (0)
        "macro_vix_term_zscore",       # Z-score of term structure
        "macro_uvxy_pct_5d",           # UVXY momentum
        "macro_svxy_pct_5d",           # SVXY momentum (inverse VIX)
    ]

    # Gold features (5)
    GOLD_FEATURES = [
        "macro_gld_pct_1d",
        "macro_gld_pct_5d",
        "macro_gld_pct_20d",
        "macro_gld_ma20_ratio",
        "macro_gld_vol20",
    ]

    # Bond features (8)
    BOND_FEATURES = [
        "macro_tlt_pct_1d",
        "macro_tlt_pct_5d",
        "macro_tlt_pct_20d",
        "macro_tlt_ma20_ratio",
        "macro_yield_curve",
        "macro_yield_curve_chg5",
        "macro_ief_pct_5d",
        "macro_bond_vol20",
    ]

    # Dollar features (4)
    DOLLAR_FEATURES = [
        "macro_uup_pct_1d",
        "macro_uup_pct_5d",
        "macro_uup_ma20_ratio",
        "macro_uup_strength",
    ]

    # Oil features (4)
    OIL_FEATURES = [
        "macro_uso_pct_1d",
        "macro_uso_pct_5d",
        "macro_uso_pct_20d",
        "macro_uso_vol20",
    ]

    # Sector features (33 = 11 sectors x 3 features each)
    SECTOR_SYMBOLS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLRE", "XLB", "XLC"]
    SECTOR_FEATURES = []
    for sector in SECTOR_SYMBOLS:
        SECTOR_FEATURES.extend([
            f"macro_{sector.lower()}_pct_5d",
            f"macro_{sector.lower()}_pct_20d",
            f"macro_{sector.lower()}_vs_spy",
        ])

    # Credit/Risk features (8)
    CREDIT_FEATURES = [
        "macro_hyg_pct_5d",
        "macro_hyg_pct_20d",
        "macro_hyg_vs_lqd",           # HYG/LQD spread proxy
        "macro_hyg_lqd_chg5",         # Spread change
        "macro_jnk_vol20",            # Junk bond volatility
        "macro_credit_stress",         # Composite credit stress
        "macro_hyg_tlt_ratio",        # Risk appetite: HYG vs TLT
        "macro_hyg_tlt_chg5",         # Risk appetite change
    ]

    # Global market features (8)
    GLOBAL_FEATURES = [
        "macro_eem_pct_5d",           # Emerging markets momentum
        "macro_eem_pct_20d",
        "macro_eem_vs_spy",           # EM relative strength
        "macro_efa_pct_5d",           # Developed markets momentum
        "macro_efa_vs_spy",           # Developed relative strength
        "macro_fxi_pct_5d",           # China momentum
        "macro_ewj_pct_5d",           # Japan momentum
        "macro_global_risk",          # Composite global risk
    ]

    # Market benchmark features (6)
    BENCHMARK_FEATURES = [
        "macro_spy_pct_1d",
        "macro_spy_pct_5d",
        "macro_spy_pct_20d",
        "macro_spy_vol20",
        "macro_qqq_vs_spy",           # Tech relative strength
        "macro_spy_ma20_ratio",
    ]

    # FRED Treasury yield features (10)
    TREASURY_FEATURES = [
        "macro_yield_2y",             # 2-year yield level
        "macro_yield_10y",            # 10-year yield level
        "macro_yield_30y",            # 30-year yield level
        "macro_yield_2s10s",          # True 2s10s spread from FRED
        "macro_yield_3m10y",          # 3m10y spread from FRED
        "macro_yield_10y_chg5",       # 10y yield 5-day change
        "macro_yield_10y_chg20",      # 10y yield 20-day change
        "macro_yield_curve_slope",    # Curve slope (30y - 2y)
        "macro_yield_curve_zscore",   # Z-score of curve slope
        "macro_yield_inversion",      # Binary: inverted (1) or normal (0)
    ]

    # FRED Credit spread features (5)
    FRED_CREDIT_FEATURES = [
        "macro_hy_spread",            # High yield spread level
        "macro_hy_spread_zscore",     # HY spread z-score
        "macro_hy_spread_chg5",       # HY spread 5-day change
        "macro_ig_spread",            # Investment grade spread
        "macro_credit_risk",          # HY - IG spread difference
    ]

    # Cross-asset features (5)
    CROSS_ASSET_FEATURES = [
        "macro_risk_on_off",
        "macro_gold_oil_ratio",
        "macro_gold_oil_ratio_chg",
        "macro_stock_bond_corr",
        "macro_market_stress",
    ]

    # ==========================================================================
    # Feature sets
    # ==========================================================================

    # All features combined
    ALL_FEATURES = (
        VIX_FEATURES + VIX_TERM_FEATURES + GOLD_FEATURES + BOND_FEATURES +
        DOLLAR_FEATURES + OIL_FEATURES + SECTOR_FEATURES + CREDIT_FEATURES +
        GLOBAL_FEATURES + BENCHMARK_FEATURES + TREASURY_FEATURES +
        FRED_CREDIT_FEATURES + CROSS_ASSET_FEATURES
    )

    # Core features subset for lightweight usage (~25)
    CORE_FEATURES = [
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

    # VIX only features
    VIX_ONLY_FEATURES = VIX_FEATURES + VIX_TERM_FEATURES

    def __init__(self, macro_csv_dir: Path = MACRO_CSV_DIR):
        """
        Initialize processor with macro CSV directory.

        Args:
            macro_csv_dir: Directory containing macro CSV files
        """
        self.macro_csv_dir = Path(macro_csv_dir)
        self._data: Dict[str, pd.DataFrame] = {}
        self._fred_data: Dict[str, pd.DataFrame] = {}
        self._load_all_data()

    def _load_all_data(self):
        """Load all macro CSV files into memory."""
        # Original symbols
        yahoo_symbols = [
            # Core macro
            "VIX", "GLD", "TLT", "UUP", "USO", "SHY", "IEF",
            # VIX derivatives
            "VIX3M", "UVXY", "SVXY",
            # Sector ETFs
            "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLRE", "XLB", "XLC",
            # Risk/Credit ETFs
            "HYG", "LQD", "JNK",
            # Global ETFs
            "EEM", "EFA", "FXI", "EWJ",
            # Market benchmarks
            "SPY", "QQQ",
        ]

        print("Loading Yahoo Finance data...")
        for symbol in yahoo_symbols:
            csv_path = self.macro_csv_dir / f"{symbol}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
                self._data[symbol] = df
            else:
                pass  # Silently skip missing files

        print(f"  Loaded {len(self._data)} Yahoo Finance symbols")

        # FRED data
        fred_series = [
            "DGS2", "DGS10", "DGS30", "DGS3MO",
            "T10Y2Y", "T10Y3M",
            "BAMLH0A0HYM2", "BAMLC0A0CM",
            "DTWEXBGS",
        ]

        print("Loading FRED data...")
        for series in fred_series:
            csv_path = self.macro_csv_dir / f"FRED_{series}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
                self._fred_data[series] = df
            else:
                pass  # Silently skip missing files

        print(f"  Loaded {len(self._fred_data)} FRED series")

    def _pct_change(self, series: pd.Series, periods: int) -> pd.Series:
        """Calculate percentage change over N periods."""
        return series.pct_change(periods)

    def _ma_ratio(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate ratio of current value to moving average."""
        ma = series.rolling(window=window, min_periods=1).mean()
        return series / ma - 1

    def _zscore(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling z-score."""
        ma = series.rolling(window=window, min_periods=1).mean()
        std = series.rolling(window=window, min_periods=1).std()
        return (series - ma) / std.replace(0, np.nan)

    def _volatility(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling volatility (std of returns)."""
        returns = series.pct_change()
        return returns.rolling(window=window, min_periods=1).std()

    def _vix_regime(self, vix_series: pd.Series) -> pd.Series:
        """
        Classify VIX into regime buckets.
        - 0: Low volatility (VIX < 15)
        - 1: Normal (15 <= VIX < 25)
        - 2: High volatility (VIX >= 25)
        """
        regime = pd.Series(index=vix_series.index, dtype=float)
        regime[vix_series < 15] = 0
        regime[(vix_series >= 15) & (vix_series < 25)] = 1
        regime[vix_series >= 25] = 2
        return regime

    def _get_close(self, symbol: str, dates: pd.DatetimeIndex) -> Optional[pd.Series]:
        """Get close price for a symbol, reindexed to dates."""
        if symbol in self._data:
            return self._data[symbol]["close"].reindex(dates)
        return None

    def _get_fred(self, series: str, dates: pd.DatetimeIndex) -> Optional[pd.Series]:
        """Get FRED series value, reindexed to dates."""
        if series in self._fred_data:
            return self._fred_data[series]["value"].reindex(dates)
        return None

    def compute_features(self, feature_set: str = "all") -> pd.DataFrame:
        """
        Compute all macro features.

        Args:
            feature_set: Which features to compute
                - "all": All features (~105)
                - "core": Core features (~25)
                - "vix_only": VIX features only (~13)

        Returns:
            DataFrame with date index and feature columns
        """
        # Get common date index from all available data
        all_dates = set()
        for df in self._data.values():
            all_dates.update(df.index)
        for df in self._fred_data.values():
            all_dates.update(df.index)
        dates = pd.DatetimeIndex(sorted(all_dates))

        features = pd.DataFrame(index=dates)
        features.index.name = "date"

        # ======================================================================
        # VIX features
        # ======================================================================
        vix = self._get_close("VIX", dates)
        if vix is not None:
            features["macro_vix_level"] = vix / 100  # Scale to 0-1 range
            features["macro_vix_zscore20"] = self._zscore(vix, 20)
            features["macro_vix_pct_1d"] = self._pct_change(vix, 1)
            features["macro_vix_pct_5d"] = self._pct_change(vix, 5)
            features["macro_vix_pct_10d"] = self._pct_change(vix, 10)
            features["macro_vix_ma5_ratio"] = self._ma_ratio(vix, 5)
            features["macro_vix_ma20_ratio"] = self._ma_ratio(vix, 20)
            features["macro_vix_regime"] = self._vix_regime(vix)

        # VIX term structure
        vix3m = self._get_close("VIX3M", dates)
        if vix is not None and vix3m is not None:
            term_struct = vix / vix3m
            features["macro_vix_term_structure"] = term_struct
            features["macro_vix_contango"] = (term_struct < 1).astype(float)
            features["macro_vix_term_zscore"] = self._zscore(term_struct, 20)

        uvxy = self._get_close("UVXY", dates)
        if uvxy is not None:
            features["macro_uvxy_pct_5d"] = self._pct_change(uvxy, 5)

        svxy = self._get_close("SVXY", dates)
        if svxy is not None:
            features["macro_svxy_pct_5d"] = self._pct_change(svxy, 5)

        # ======================================================================
        # Gold features
        # ======================================================================
        gld = self._get_close("GLD", dates)
        if gld is not None:
            features["macro_gld_pct_1d"] = self._pct_change(gld, 1)
            features["macro_gld_pct_5d"] = self._pct_change(gld, 5)
            features["macro_gld_pct_20d"] = self._pct_change(gld, 20)
            features["macro_gld_ma20_ratio"] = self._ma_ratio(gld, 20)
            features["macro_gld_vol20"] = self._volatility(gld, 20)

        # ======================================================================
        # Bond features
        # ======================================================================
        tlt = self._get_close("TLT", dates)
        if tlt is not None:
            features["macro_tlt_pct_1d"] = self._pct_change(tlt, 1)
            features["macro_tlt_pct_5d"] = self._pct_change(tlt, 5)
            features["macro_tlt_pct_20d"] = self._pct_change(tlt, 20)
            features["macro_tlt_ma20_ratio"] = self._ma_ratio(tlt, 20)
            features["macro_bond_vol20"] = self._volatility(tlt, 20)

        shy = self._get_close("SHY", dates)
        if shy is not None and tlt is not None:
            yield_curve = tlt / shy
            features["macro_yield_curve"] = yield_curve
            features["macro_yield_curve_chg5"] = self._pct_change(yield_curve, 5)

        ief = self._get_close("IEF", dates)
        if ief is not None:
            features["macro_ief_pct_5d"] = self._pct_change(ief, 5)

        # ======================================================================
        # Dollar features
        # ======================================================================
        uup = self._get_close("UUP", dates)
        if uup is not None:
            features["macro_uup_pct_1d"] = self._pct_change(uup, 1)
            features["macro_uup_pct_5d"] = self._pct_change(uup, 5)
            features["macro_uup_ma20_ratio"] = self._ma_ratio(uup, 20)
            vol = self._volatility(uup, 20)
            features["macro_uup_strength"] = self._pct_change(uup, 20) / vol.replace(0, np.nan)

        # ======================================================================
        # Oil features
        # ======================================================================
        uso = self._get_close("USO", dates)
        if uso is not None:
            features["macro_uso_pct_1d"] = self._pct_change(uso, 1)
            features["macro_uso_pct_5d"] = self._pct_change(uso, 5)
            features["macro_uso_pct_20d"] = self._pct_change(uso, 20)
            features["macro_uso_vol20"] = self._volatility(uso, 20)

        # ======================================================================
        # Sector features
        # ======================================================================
        spy = self._get_close("SPY", dates)
        spy_ret_5d = self._pct_change(spy, 5) if spy is not None else None
        spy_ret_20d = self._pct_change(spy, 20) if spy is not None else None

        for sector in self.SECTOR_SYMBOLS:
            sector_close = self._get_close(sector, dates)
            if sector_close is not None:
                sector_lower = sector.lower()
                features[f"macro_{sector_lower}_pct_5d"] = self._pct_change(sector_close, 5)
                features[f"macro_{sector_lower}_pct_20d"] = self._pct_change(sector_close, 20)
                if spy_ret_5d is not None:
                    # Relative strength vs SPY
                    features[f"macro_{sector_lower}_vs_spy"] = (
                        self._pct_change(sector_close, 5) - spy_ret_5d
                    )

        # ======================================================================
        # Credit/Risk features
        # ======================================================================
        hyg = self._get_close("HYG", dates)
        lqd = self._get_close("LQD", dates)
        jnk = self._get_close("JNK", dates)

        if hyg is not None:
            features["macro_hyg_pct_5d"] = self._pct_change(hyg, 5)
            features["macro_hyg_pct_20d"] = self._pct_change(hyg, 20)

        if hyg is not None and lqd is not None:
            hyg_lqd = hyg / lqd
            features["macro_hyg_vs_lqd"] = hyg_lqd
            features["macro_hyg_lqd_chg5"] = self._pct_change(hyg_lqd, 5)
            # Credit stress: negative of HYG/LQD z-score (higher = more stress)
            features["macro_credit_stress"] = -self._zscore(hyg_lqd, 20)

        if jnk is not None:
            features["macro_jnk_vol20"] = self._volatility(jnk, 20)

        if hyg is not None and tlt is not None:
            hyg_tlt = hyg / tlt
            features["macro_hyg_tlt_ratio"] = hyg_tlt
            features["macro_hyg_tlt_chg5"] = self._pct_change(hyg_tlt, 5)

        # ======================================================================
        # Global market features
        # ======================================================================
        eem = self._get_close("EEM", dates)
        efa = self._get_close("EFA", dates)
        fxi = self._get_close("FXI", dates)
        ewj = self._get_close("EWJ", dates)

        if eem is not None:
            features["macro_eem_pct_5d"] = self._pct_change(eem, 5)
            features["macro_eem_pct_20d"] = self._pct_change(eem, 20)
            if spy_ret_5d is not None:
                features["macro_eem_vs_spy"] = self._pct_change(eem, 5) - spy_ret_5d

        if efa is not None:
            features["macro_efa_pct_5d"] = self._pct_change(efa, 5)
            if spy_ret_5d is not None:
                features["macro_efa_vs_spy"] = self._pct_change(efa, 5) - spy_ret_5d

        if fxi is not None:
            features["macro_fxi_pct_5d"] = self._pct_change(fxi, 5)

        if ewj is not None:
            features["macro_ewj_pct_5d"] = self._pct_change(ewj, 5)

        # Global risk: composite of EM and developed weakness
        if eem is not None and efa is not None:
            eem_zscore = self._zscore(self._pct_change(eem, 5), 20)
            efa_zscore = self._zscore(self._pct_change(efa, 5), 20)
            features["macro_global_risk"] = -(eem_zscore + efa_zscore) / 2

        # ======================================================================
        # Market benchmark features
        # ======================================================================
        if spy is not None:
            features["macro_spy_pct_1d"] = self._pct_change(spy, 1)
            features["macro_spy_pct_5d"] = self._pct_change(spy, 5)
            features["macro_spy_pct_20d"] = self._pct_change(spy, 20)
            features["macro_spy_vol20"] = self._volatility(spy, 20)
            features["macro_spy_ma20_ratio"] = self._ma_ratio(spy, 20)

        qqq = self._get_close("QQQ", dates)
        if qqq is not None and spy is not None:
            features["macro_qqq_vs_spy"] = self._pct_change(qqq, 5) - spy_ret_5d

        # ======================================================================
        # FRED Treasury yield features
        # ======================================================================
        yield_2y = self._get_fred("DGS2", dates)
        yield_10y = self._get_fred("DGS10", dates)
        yield_30y = self._get_fred("DGS30", dates)
        yield_3mo = self._get_fred("DGS3MO", dates)
        spread_2s10s = self._get_fred("T10Y2Y", dates)
        spread_3m10y = self._get_fred("T10Y3M", dates)

        if yield_2y is not None:
            features["macro_yield_2y"] = yield_2y / 100  # Convert to decimal

        if yield_10y is not None:
            features["macro_yield_10y"] = yield_10y / 100
            features["macro_yield_10y_chg5"] = yield_10y.diff(5) / 100
            features["macro_yield_10y_chg20"] = yield_10y.diff(20) / 100

        if yield_30y is not None:
            features["macro_yield_30y"] = yield_30y / 100

        if spread_2s10s is not None:
            features["macro_yield_2s10s"] = spread_2s10s / 100
            features["macro_yield_inversion"] = (spread_2s10s < 0).astype(float)

        if spread_3m10y is not None:
            features["macro_yield_3m10y"] = spread_3m10y / 100

        if yield_30y is not None and yield_2y is not None:
            curve_slope = yield_30y - yield_2y
            features["macro_yield_curve_slope"] = curve_slope / 100
            features["macro_yield_curve_zscore"] = self._zscore(curve_slope, 60)

        # ======================================================================
        # FRED Credit spread features
        # ======================================================================
        hy_spread = self._get_fred("BAMLH0A0HYM2", dates)
        ig_spread = self._get_fred("BAMLC0A0CM", dates)

        if hy_spread is not None:
            features["macro_hy_spread"] = hy_spread / 100  # Convert bps to decimal
            features["macro_hy_spread_zscore"] = self._zscore(hy_spread, 60)
            features["macro_hy_spread_chg5"] = hy_spread.diff(5) / 100

        if ig_spread is not None:
            features["macro_ig_spread"] = ig_spread / 100

        if hy_spread is not None and ig_spread is not None:
            features["macro_credit_risk"] = (hy_spread - ig_spread) / 100

        # ======================================================================
        # Cross-asset features
        # ======================================================================
        if gld is not None and uso is not None:
            gold_oil = gld / uso
            features["macro_gold_oil_ratio"] = gold_oil / gold_oil.rolling(60).mean() - 1
            features["macro_gold_oil_ratio_chg"] = self._pct_change(gold_oil, 5)

        if vix is not None and tlt is not None:
            vix_norm = self._zscore(vix, 20)
            tlt_ret = self._pct_change(tlt, 5)
            features["macro_risk_on_off"] = -vix_norm + tlt_ret * 10
            features["macro_market_stress"] = vix_norm.clip(-3, 3) / 3

        if vix is not None and tlt is not None:
            vix_ret = self._pct_change(vix, 1)
            tlt_ret = self._pct_change(tlt, 1)
            features["macro_stock_bond_corr"] = vix_ret.rolling(20).corr(tlt_ret)

        # ======================================================================
        # Filter to requested feature set
        # ======================================================================
        if feature_set == "core":
            available = [f for f in self.CORE_FEATURES if f in features.columns]
            features = features[available]
        elif feature_set == "vix_only":
            available = [f for f in self.VIX_ONLY_FEATURES if f in features.columns]
            features = features[available]

        # Forward fill missing values (up to 5 days)
        features = features.ffill(limit=5)

        # Fill remaining NaN with appropriate values
        for col in features.columns:
            if "pct" in col or "chg" in col or "ratio" in col or "vs_" in col:
                features[col] = features[col].fillna(0)
            elif "level" in col or "regime" in col or "yield" in col or "spread" in col:
                features[col] = features[col].fillna(features[col].median())
            else:
                features[col] = features[col].fillna(0)

        return features

    def save_features(
        self,
        output_path: Path = DEFAULT_OUTPUT_PATH,
        feature_set: str = "all"
    ):
        """
        Compute and save features to parquet file.

        Args:
            output_path: Output file path
            feature_set: Which features to compute
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        features = self.compute_features(feature_set)

        # Save to parquet
        features.to_parquet(output_path)

        print(f"\nSaved {len(features.columns)} features to {output_path}")
        print(f"Date range: {features.index.min()} to {features.index.max()}")
        print(f"Total rows: {len(features)}")
        print(f"\nFeature categories:")
        print(f"  VIX: {sum(1 for c in features.columns if 'vix' in c)}")
        print(f"  Gold: {sum(1 for c in features.columns if 'gld' in c)}")
        print(f"  Bond: {sum(1 for c in features.columns if 'tlt' in c or 'ief' in c or 'bond' in c)}")
        print(f"  Dollar: {sum(1 for c in features.columns if 'uup' in c)}")
        print(f"  Oil: {sum(1 for c in features.columns if 'uso' in c)}")
        print(f"  Sector: {sum(1 for c in features.columns if any(s.lower() in c for s in self.SECTOR_SYMBOLS))}")
        print(f"  Credit: {sum(1 for c in features.columns if 'hyg' in c or 'lqd' in c or 'jnk' in c or 'credit' in c)}")
        print(f"  Global: {sum(1 for c in features.columns if 'eem' in c or 'efa' in c or 'fxi' in c or 'ewj' in c or 'global' in c)}")
        print(f"  Benchmark: {sum(1 for c in features.columns if 'spy' in c or 'qqq' in c)}")
        print(f"  Treasury: {sum(1 for c in features.columns if 'yield' in c)}")

        return features


def main():
    parser = argparse.ArgumentParser(description="Process macro data into features")
    parser.add_argument(
        "--input",
        type=str,
        default=str(MACRO_CSV_DIR),
        help=f"Input directory with macro CSVs (default: {MACRO_CSV_DIR})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output parquet file path (default: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="all",
        choices=["all", "core", "vix_only"],
        help="Feature set to generate (default: all)"
    )

    args = parser.parse_args()

    processor = MacroFeatureProcessor(macro_csv_dir=Path(args.input))
    processor.save_features(
        output_path=Path(args.output),
        feature_set=args.feature_set
    )


if __name__ == "__main__":
    main()
