"""
Process macro CSV data into features for time series prediction

Usage:
    python process_macro_data.py                     # Process with defaults
    python process_macro_data.py --output ./my_path  # Custom output path
    python process_macro_data.py --feature-set core  # Generate core features only

Generated features (~35 total):
- VIX features: level, zscore, pct changes, ma ratios, regime
- Gold features: pct changes, ma ratio, volatility
- Bond features: pct changes, yield curve proxy, volatility
- Dollar features: pct changes, strength indicator
- Oil features: pct changes, volatility
- Cross-asset features: risk indicators, ratios, correlations
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
MACRO_CSV_DIR = PROJECT_ROOT / "my_data" / "macro_csv"
MACRO_OUTPUT_DIR = PROJECT_ROOT / "my_data" / "macro_processed"
DEFAULT_OUTPUT_PATH = MACRO_OUTPUT_DIR / "macro_features.parquet"


class MacroFeatureProcessor:
    """Generate macro features from raw ETF/index data."""

    # Feature column names for each set
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

    GOLD_FEATURES = [
        "macro_gld_pct_1d",
        "macro_gld_pct_5d",
        "macro_gld_pct_20d",
        "macro_gld_ma20_ratio",
        "macro_gld_vol20",
    ]

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

    DOLLAR_FEATURES = [
        "macro_uup_pct_1d",
        "macro_uup_pct_5d",
        "macro_uup_ma20_ratio",
        "macro_uup_strength",
    ]

    OIL_FEATURES = [
        "macro_uso_pct_1d",
        "macro_uso_pct_5d",
        "macro_uso_pct_20d",
        "macro_uso_vol20",
    ]

    CROSS_ASSET_FEATURES = [
        "macro_risk_on_off",
        "macro_gold_oil_ratio",
        "macro_gold_oil_ratio_chg",
        "macro_stock_bond_corr",
        "macro_market_stress",
    ]

    # Core features subset for lightweight usage
    CORE_FEATURES = [
        "macro_vix_level",
        "macro_vix_zscore20",
        "macro_vix_pct_5d",
        "macro_vix_regime",
        "macro_gld_pct_5d",
        "macro_tlt_pct_5d",
        "macro_yield_curve",
        "macro_uup_pct_5d",
        "macro_uso_pct_5d",
        "macro_risk_on_off",
        "macro_market_stress",
    ]

    # VIX only features
    VIX_ONLY_FEATURES = VIX_FEATURES

    def __init__(self, macro_csv_dir: Path = MACRO_CSV_DIR):
        """
        Initialize processor with macro CSV directory.

        Args:
            macro_csv_dir: Directory containing macro CSV files
        """
        self.macro_csv_dir = Path(macro_csv_dir)
        self._data = {}
        self._load_all_data()

    def _load_all_data(self):
        """Load all macro CSV files into memory."""
        symbols = ["VIX", "GLD", "TLT", "UUP", "USO", "SHY", "IEF"]

        for symbol in symbols:
            csv_path = self.macro_csv_dir / f"{symbol}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
                self._data[symbol] = df
                print(f"Loaded {symbol}: {len(df)} rows")
            else:
                print(f"Warning: {symbol}.csv not found")

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

    def compute_features(self, feature_set: str = "all") -> pd.DataFrame:
        """
        Compute all macro features.

        Args:
            feature_set: Which features to compute
                - "all": All features (~35)
                - "core": Core features (~11)
                - "vix_only": VIX features only (~8)

        Returns:
            DataFrame with date index and feature columns
        """
        # Get common date index from all available data
        all_dates = set()
        for df in self._data.values():
            all_dates.update(df.index)
        dates = pd.DatetimeIndex(sorted(all_dates))

        features = pd.DataFrame(index=dates)
        features.index.name = "date"

        # Compute VIX features
        if "VIX" in self._data:
            vix = self._data["VIX"]["close"].reindex(dates)
            features["macro_vix_level"] = vix / 100  # Scale to 0-1 range
            features["macro_vix_zscore20"] = self._zscore(vix, 20)
            features["macro_vix_pct_1d"] = self._pct_change(vix, 1)
            features["macro_vix_pct_5d"] = self._pct_change(vix, 5)
            features["macro_vix_pct_10d"] = self._pct_change(vix, 10)
            features["macro_vix_ma5_ratio"] = self._ma_ratio(vix, 5)
            features["macro_vix_ma20_ratio"] = self._ma_ratio(vix, 20)
            features["macro_vix_regime"] = self._vix_regime(vix)

        # Compute Gold features
        if "GLD" in self._data:
            gld = self._data["GLD"]["close"].reindex(dates)
            features["macro_gld_pct_1d"] = self._pct_change(gld, 1)
            features["macro_gld_pct_5d"] = self._pct_change(gld, 5)
            features["macro_gld_pct_20d"] = self._pct_change(gld, 20)
            features["macro_gld_ma20_ratio"] = self._ma_ratio(gld, 20)
            features["macro_gld_vol20"] = self._volatility(gld, 20)

        # Compute Bond features
        if "TLT" in self._data:
            tlt = self._data["TLT"]["close"].reindex(dates)
            features["macro_tlt_pct_1d"] = self._pct_change(tlt, 1)
            features["macro_tlt_pct_5d"] = self._pct_change(tlt, 5)
            features["macro_tlt_pct_20d"] = self._pct_change(tlt, 20)
            features["macro_tlt_ma20_ratio"] = self._ma_ratio(tlt, 20)
            features["macro_bond_vol20"] = self._volatility(tlt, 20)

        if "SHY" in self._data and "TLT" in self._data:
            shy = self._data["SHY"]["close"].reindex(dates)
            tlt = self._data["TLT"]["close"].reindex(dates)
            # Yield curve proxy: TLT/SHY ratio (higher = steeper curve)
            yield_curve = tlt / shy
            features["macro_yield_curve"] = yield_curve
            features["macro_yield_curve_chg5"] = self._pct_change(yield_curve, 5)

        if "IEF" in self._data:
            ief = self._data["IEF"]["close"].reindex(dates)
            features["macro_ief_pct_5d"] = self._pct_change(ief, 5)

        # Compute Dollar features
        if "UUP" in self._data:
            uup = self._data["UUP"]["close"].reindex(dates)
            features["macro_uup_pct_1d"] = self._pct_change(uup, 1)
            features["macro_uup_pct_5d"] = self._pct_change(uup, 5)
            features["macro_uup_ma20_ratio"] = self._ma_ratio(uup, 20)
            # Dollar strength: 20-day momentum normalized
            features["macro_uup_strength"] = self._pct_change(uup, 20) / self._volatility(uup, 20)

        # Compute Oil features
        if "USO" in self._data:
            uso = self._data["USO"]["close"].reindex(dates)
            features["macro_uso_pct_1d"] = self._pct_change(uso, 1)
            features["macro_uso_pct_5d"] = self._pct_change(uso, 5)
            features["macro_uso_pct_20d"] = self._pct_change(uso, 20)
            features["macro_uso_vol20"] = self._volatility(uso, 20)

        # Compute cross-asset features
        if "GLD" in self._data and "USO" in self._data:
            gld = self._data["GLD"]["close"].reindex(dates)
            uso = self._data["USO"]["close"].reindex(dates)
            gold_oil = gld / uso
            features["macro_gold_oil_ratio"] = gold_oil / gold_oil.rolling(60).mean() - 1
            features["macro_gold_oil_ratio_chg"] = self._pct_change(gold_oil, 5)

        if "VIX" in self._data and "TLT" in self._data:
            vix = self._data["VIX"]["close"].reindex(dates)
            tlt = self._data["TLT"]["close"].reindex(dates)
            # Risk-on/off indicator: VIX normalized + TLT momentum (inverted)
            vix_norm = self._zscore(vix, 20)
            tlt_ret = self._pct_change(tlt, 5)
            features["macro_risk_on_off"] = -vix_norm + tlt_ret * 10  # Higher = risk-on

            # Market stress: composite of VIX level and volatility
            features["macro_market_stress"] = vix_norm.clip(-3, 3) / 3  # Normalized -1 to 1

        # Stock-bond correlation (using TLT as bond proxy)
        # This requires external stock data, so we use VIX as proxy for stock vol
        if "VIX" in self._data and "TLT" in self._data:
            vix_ret = self._pct_change(self._data["VIX"]["close"].reindex(dates), 1)
            tlt_ret = self._pct_change(self._data["TLT"]["close"].reindex(dates), 1)
            features["macro_stock_bond_corr"] = vix_ret.rolling(20).corr(tlt_ret)

        # Filter to requested feature set
        if feature_set == "core":
            available = [f for f in self.CORE_FEATURES if f in features.columns]
            features = features[available]
        elif feature_set == "vix_only":
            available = [f for f in self.VIX_ONLY_FEATURES if f in features.columns]
            features = features[available]

        # Forward fill missing values (up to 5 days)
        features = features.ffill(limit=5)

        # Fill remaining NaN with 0 for change features, median for levels
        for col in features.columns:
            if "pct" in col or "chg" in col or "ratio" in col:
                features[col] = features[col].fillna(0)
            elif "level" in col or "regime" in col:
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
        print(f"Features: {features.columns.tolist()}")

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
