"""
Process market data into MASTER model market information features

Generates 63 market information features for MASTER model integration:
- 3 indices: SPY (S&P 500), QQQ (Nasdaq 100), IWM (Russell 2000)
- 21 features per index:
  - Daily return (1)
  - Return mean over 5, 10, 20, 30, 60 days (5)
  - Return std over 5, 10, 20, 30, 60 days (5)
  - Volume mean ratio over 5, 10, 20, 30, 60 days (5)
  - Volume std ratio over 5, 10, 20, 30, 60 days (5)

Usage:
    python process_master_market_info.py                     # Process with defaults
    python process_master_market_info.py --output ./my_path  # Custom output path

Output: my_data/macro_processed/master_market_info.parquet
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
MACRO_CSV_DIR = PROJECT_ROOT / "my_data" / "macro_csv"
MACRO_OUTPUT_DIR = PROJECT_ROOT / "my_data" / "macro_processed"
DEFAULT_OUTPUT_PATH = MACRO_OUTPUT_DIR / "master_market_info.parquet"

# Market indices for MASTER model
MARKET_INDICES = ["SPY", "QQQ", "IWM"]

# Rolling windows
WINDOWS = [5, 10, 20, 30, 60]


class MasterMarketInfoProcessor:
    """Generate MASTER model market information features."""

    def __init__(self, macro_csv_dir: Path = MACRO_CSV_DIR):
        """
        Initialize processor with macro CSV directory.

        Args:
            macro_csv_dir: Directory containing macro CSV files
        """
        self.macro_csv_dir = Path(macro_csv_dir)
        self._data: Dict[str, pd.DataFrame] = {}
        self._load_market_data()

    def _load_market_data(self):
        """Load market index data from CSV files."""
        print("Loading market index data...")
        for symbol in MARKET_INDICES:
            csv_path = self.macro_csv_dir / f"{symbol}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
                self._data[symbol] = df
                print(f"  {symbol}: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
            else:
                print(f"  Warning: {symbol}.csv not found at {csv_path}")

        if len(self._data) < len(MARKET_INDICES):
            missing = set(MARKET_INDICES) - set(self._data.keys())
            print(f"\n  Missing indices: {missing}")
            print("  Run: python scripts/data/download_macro_data.py --symbols " + " ".join(missing))

    def _compute_daily_return(self, close: pd.Series) -> pd.Series:
        """Compute daily return."""
        return close.pct_change(fill_method=None)

    def _compute_return_mean(self, returns: pd.Series, window: int) -> pd.Series:
        """Compute rolling mean of returns over window."""
        return returns.rolling(window=window, min_periods=1).mean()

    def _compute_return_std(self, returns: pd.Series, window: int) -> pd.Series:
        """Compute rolling std of returns over window."""
        return returns.rolling(window=window, min_periods=1).std()

    def _compute_volume_mean_ratio(self, volume: pd.Series, window: int) -> pd.Series:
        """Compute volume / rolling mean volume ratio."""
        vol_ma = volume.rolling(window=window, min_periods=1).mean()
        return volume / vol_ma.replace(0, np.nan) - 1

    def _compute_volume_std_ratio(self, volume: pd.Series, window: int) -> pd.Series:
        """Compute volume / rolling std volume ratio (normalized)."""
        vol_std = volume.rolling(window=window, min_periods=1).std()
        vol_ma = volume.rolling(window=window, min_periods=1).mean()
        # Normalize: (V - MA) / STD
        return (volume - vol_ma) / vol_std.replace(0, np.nan)

    def get_feature_names(self) -> List[str]:
        """Get list of all 63 feature names."""
        names = []
        for symbol in MARKET_INDICES:
            sym_lower = symbol.lower()
            # Daily return
            names.append(f"mkt_{sym_lower}_ret_1d")
            # Return mean
            for w in WINDOWS:
                names.append(f"mkt_{sym_lower}_ret_mean_{w}d")
            # Return std
            for w in WINDOWS:
                names.append(f"mkt_{sym_lower}_ret_std_{w}d")
            # Volume mean ratio
            for w in WINDOWS:
                names.append(f"mkt_{sym_lower}_vol_mean_{w}d")
            # Volume std ratio
            for w in WINDOWS:
                names.append(f"mkt_{sym_lower}_vol_std_{w}d")
        return names

    def compute_features(self) -> pd.DataFrame:
        """
        Compute all 63 MASTER market information features.

        Returns:
            DataFrame with date index and 63 feature columns
        """
        warnings.filterwarnings('ignore', message='DataFrame is highly fragmented')

        # Get common date index from all available data
        all_dates = set()
        for df in self._data.values():
            all_dates.update(df.index)
        dates = pd.DatetimeIndex(sorted(all_dates))

        features = pd.DataFrame(index=dates)
        features.index.name = "date"

        for symbol in MARKET_INDICES:
            if symbol not in self._data:
                print(f"  Skipping {symbol} (data not loaded)")
                continue

            df = self._data[symbol]
            close = df["close"].reindex(dates)
            volume = df["volume"].reindex(dates) if "volume" in df.columns else None

            sym_lower = symbol.lower()

            # Compute daily returns
            returns = self._compute_daily_return(close)

            # 1. Daily return (1 feature)
            features[f"mkt_{sym_lower}_ret_1d"] = returns

            # 2. Return mean over windows (5 features)
            for w in WINDOWS:
                features[f"mkt_{sym_lower}_ret_mean_{w}d"] = self._compute_return_mean(returns, w)

            # 3. Return std over windows (5 features)
            for w in WINDOWS:
                features[f"mkt_{sym_lower}_ret_std_{w}d"] = self._compute_return_std(returns, w)

            # Volume features (if volume data available)
            if volume is not None and not volume.isna().all():
                # 4. Volume mean ratio over windows (5 features)
                for w in WINDOWS:
                    features[f"mkt_{sym_lower}_vol_mean_{w}d"] = self._compute_volume_mean_ratio(volume, w)

                # 5. Volume std ratio over windows (5 features)
                for w in WINDOWS:
                    features[f"mkt_{sym_lower}_vol_std_{w}d"] = self._compute_volume_std_ratio(volume, w)
            else:
                print(f"  Warning: No volume data for {symbol}, filling with 0")
                for w in WINDOWS:
                    features[f"mkt_{sym_lower}_vol_mean_{w}d"] = 0.0
                    features[f"mkt_{sym_lower}_vol_std_{w}d"] = 0.0

        # Defragment DataFrame
        features = features.copy()

        # Forward fill missing values (up to 5 days)
        features = features.ffill(limit=5)

        # Fill remaining NaN with 0 (appropriate for returns and ratios)
        features = features.fillna(0)

        # Clip extreme values for stability
        for col in features.columns:
            if "ret" in col:
                features[col] = features[col].clip(-0.5, 0.5)  # ±50% max
            elif "vol" in col:
                features[col] = features[col].clip(-10, 10)  # ±10 std max

        return features

    def save_features(self, output_path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
        """
        Compute and save features to parquet file.

        Args:
            output_path: Output file path

        Returns:
            Computed features DataFrame
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        features = self.compute_features()

        # Save to parquet
        features.to_parquet(output_path)

        print(f"\n{'='*60}")
        print("MASTER Market Info Features Summary")
        print(f"{'='*60}")
        print(f"Output: {output_path}")
        print(f"Total features: {len(features.columns)}")
        print(f"Date range: {features.index.min().date()} to {features.index.max().date()}")
        print(f"Total rows: {len(features)}")

        print(f"\nFeatures per index (21 each):")
        for symbol in MARKET_INDICES:
            sym_lower = symbol.lower()
            count = sum(1 for c in features.columns if sym_lower in c)
            print(f"  {symbol}: {count} features")

        print(f"\nFeature breakdown:")
        print(f"  Daily returns: {sum(1 for c in features.columns if 'ret_1d' in c)}")
        print(f"  Return means: {sum(1 for c in features.columns if 'ret_mean' in c)}")
        print(f"  Return stds: {sum(1 for c in features.columns if 'ret_std' in c)}")
        print(f"  Volume means: {sum(1 for c in features.columns if 'vol_mean' in c)}")
        print(f"  Volume stds: {sum(1 for c in features.columns if 'vol_std' in c)}")

        print(f"\nFeature names:")
        for i, col in enumerate(features.columns):
            print(f"  {i+1:2d}. {col}")

        return features


def main():
    parser = argparse.ArgumentParser(
        description="Process market data into MASTER model features (63 features)"
    )
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

    args = parser.parse_args()

    processor = MasterMarketInfoProcessor(macro_csv_dir=Path(args.input))
    processor.save_features(output_path=Path(args.output))


if __name__ == "__main__":
    main()
