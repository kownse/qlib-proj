"""
Process CBOE CSV data into features for integration with stock prediction models.

Reads raw CBOE data (SKEW, VIX9D, VVIX) and engineers features:
- Level, z-score, percentage changes, regime indicators
- Cross-index ratios (VIX9D/VIX, VVIX/VIX)

Output: my_data/cboe_processed/cboe_features.parquet

Usage:
    python scripts/data/process_cboe_data.py
    python scripts/data/process_cboe_data.py --force
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CBOE_CSV_DIR = PROJECT_ROOT / "my_data" / "cboe_csv"
MACRO_CSV_DIR = PROJECT_ROOT / "my_data" / "macro_csv"
CBOE_OUTPUT_DIR = PROJECT_ROOT / "my_data" / "cboe_processed"
DEFAULT_OUTPUT_PATH = CBOE_OUTPUT_DIR / "cboe_features.parquet"


def load_cboe_csv(name: str) -> pd.Series:
    """Load a CBOE CSV and return the close price series."""
    path = CBOE_CSV_DIR / f"{name}.csv"
    if not path.exists():
        print(f"  Warning: {path} not found")
        return pd.Series(dtype=float)
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    if "close" in df.columns:
        return df["close"].dropna()
    return pd.Series(dtype=float)


def load_vix() -> pd.Series:
    """Load VIX from existing macro CSV data."""
    path = MACRO_CSV_DIR / "VIX.csv"
    if not path.exists():
        print("  Warning: VIX.csv not found in macro_csv")
        return pd.Series(dtype=float)
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    if "close" in df.columns:
        return df["close"].dropna()
    return pd.Series(dtype=float)


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score."""
    mean = series.rolling(window, min_periods=max(5, window // 2)).mean()
    std = series.rolling(window, min_periods=max(5, window // 2)).std()
    return (series - mean) / (std + 1e-8)


def pct_change_n(series: pd.Series, n: int) -> pd.Series:
    """N-day percentage change."""
    return series.pct_change(n)


def process_cboe_features(force: bool = False) -> pd.DataFrame:
    """
    Process CBOE data into features.

    Returns:
        DataFrame indexed by date with CBOE feature columns.
    """
    if DEFAULT_OUTPUT_PATH.exists() and not force:
        print(f"CBOE features already exist at {DEFAULT_OUTPUT_PATH}")
        df = pd.read_parquet(DEFAULT_OUTPUT_PATH)
        print(f"Loaded: {df.shape}, range: {df.index.min()} to {df.index.max()}")
        return df

    print("=" * 60)
    print("Processing CBOE Features")
    print("=" * 60)

    # Load raw data
    skew = load_cboe_csv("SKEW")
    vix9d = load_cboe_csv("VIX9D")
    vvix = load_cboe_csv("VVIX")
    vix = load_vix()

    # Align all series to a common date index
    all_dates = sorted(set(skew.index) | set(vix9d.index) | set(vvix.index) | set(vix.index))
    if not all_dates:
        raise ValueError("No CBOE data available. Run download_cboe_data.py first.")

    date_index = pd.DatetimeIndex(all_dates)

    features = pd.DataFrame(index=date_index)
    features.index.name = "datetime"

    # ========================
    # SKEW features
    # ========================
    if len(skew) > 0:
        skew_aligned = skew.reindex(date_index)
        print(f"  SKEW: {skew_aligned.notna().sum()} valid values")

        # Level (normalized to ~0-1 range: SKEW typically 100-150)
        features["cboe_skew_level"] = (skew_aligned - 100) / 50

        # Z-score (20-day rolling)
        features["cboe_skew_zscore20"] = zscore(skew_aligned, 20)

        # Percentage changes
        features["cboe_skew_pct_5d"] = pct_change_n(skew_aligned, 5)

        # Regime: high skew (>130) indicates tail risk concerns
        features["cboe_skew_regime"] = (skew_aligned > 130).astype(float)
    else:
        print("  SKEW: no data")

    # ========================
    # VVIX features
    # ========================
    if len(vvix) > 0:
        vvix_aligned = vvix.reindex(date_index)
        print(f"  VVIX: {vvix_aligned.notna().sum()} valid values")

        # Level (normalized: VVIX typically 70-150)
        features["cboe_vvix_level"] = (vvix_aligned - 80) / 40

        # Z-score
        features["cboe_vvix_zscore20"] = zscore(vvix_aligned, 20)

        # Regime: high VVIX (>120) = extreme vol of vol
        features["cboe_vvix_regime"] = (vvix_aligned > 120).astype(float)

        # VVIX/VIX ratio (when both available)
        if len(vix) > 0:
            vix_aligned = vix.reindex(date_index)
            ratio = vvix_aligned / (vix_aligned + 1e-8)
            features["cboe_vvix_vs_vix"] = zscore(ratio, 20)
    else:
        print("  VVIX: no data")

    # ========================
    # VIX9D features
    # ========================
    if len(vix9d) > 0:
        vix9d_aligned = vix9d.reindex(date_index)
        print(f"  VIX9D: {vix9d_aligned.notna().sum()} valid values")

        # VIX9D / VIX ratio (short-term vs medium-term fear)
        # >1 means short-term fear spike, <1 means term structure in contango
        if len(vix) > 0:
            vix_aligned = vix.reindex(date_index)
            vix9d_vix_ratio = vix9d_aligned / (vix_aligned + 1e-8)

            # Raw ratio z-scored
            features["cboe_vix9d_vs_vix"] = zscore(vix9d_vix_ratio, 20)

            # Binary: short-term fear spike (VIX9D > VIX)
            features["cboe_vix9d_spike"] = (vix9d_vix_ratio > 1.0).astype(float)
    else:
        print("  VIX9D: no data")

    # Forward fill short gaps (weekends/holidays already handled, this is for missing data)
    features = features.ffill(limit=3)

    # Drop rows where ALL features are NaN
    features = features.dropna(how="all")

    # Report
    print(f"\nGenerated {len(features.columns)} CBOE features:")
    for col in features.columns:
        valid = features[col].notna().sum()
        print(f"  {col}: {valid} valid values")

    # Save
    CBOE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(DEFAULT_OUTPUT_PATH)
    print(f"\nSaved to {DEFAULT_OUTPUT_PATH}")
    print(f"Shape: {features.shape}, range: {features.index.min()} to {features.index.max()}")

    return features


def main():
    parser = argparse.ArgumentParser(description="Process CBOE data into features")
    parser.add_argument("--force", action="store_true",
                        help="Force re-process even if output exists")
    args = parser.parse_args()

    process_cboe_features(args.force)


if __name__ == "__main__":
    main()
