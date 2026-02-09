"""
Download SPY prices and compute N-day forward returns for market-neutral targets.

This computes the same forward return formula as the stock labels:
    spy_forward_return_N = spy_close.shift(-N) / spy_close.shift(-1) - 1

This matches the label formula: Ref($close, -N)/Ref($close, -1) - 1
so that subtracting SPY forward return from stock label gives alpha (excess return).

NOTE: No lag is applied here because the label itself is forward-looking.
The SPY forward return needs to be computed over the same future window
as the stock label to properly subtract market beta.

Usage:
    python scripts/data/download_spy_forward_returns.py
    python scripts/data/download_spy_forward_returns.py --force
    python scripts/data/download_spy_forward_returns.py --start 2000-01-01
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
OUTPUT_PATH = PROJECT_ROOT / "my_data" / "spy_forward_returns.parquet"

DEFAULT_START_DATE = "2000-01-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")

# Forward return windows to compute (must include the label window used in training)
FORWARD_WINDOWS = [2, 5, 10]


def download_and_compute(start_date: str, end_date: str, force: bool = False) -> pd.DataFrame:
    """
    Download SPY prices and compute N-day forward returns.

    Returns:
        DataFrame indexed by date with columns like spy_fwd_return_5d
    """
    if OUTPUT_PATH.exists() and not force:
        print(f"SPY forward returns already exist at {OUTPUT_PATH}")
        print("Use --force to re-download")
        df = pd.read_parquet(OUTPUT_PATH)
        print(f"Loaded: {df.shape}, range: {df.index.min()} to {df.index.max()}")
        return df

    print(f"Downloading SPY data from {start_date} to {end_date}...")
    spy = yf.Ticker("SPY")
    hist = spy.history(start=start_date, end=end_date, auto_adjust=True)

    if hist.empty:
        raise ValueError("No SPY data returned from Yahoo Finance")

    # Clean index
    hist.index = pd.to_datetime(hist.index)
    hist.index = hist.index.tz_localize(None)
    hist.index.name = "datetime"

    spy_close = hist["Close"]
    print(f"Downloaded SPY: {len(spy_close)} trading days, "
          f"{spy_close.index.min().date()} to {spy_close.index.max().date()}")

    # Compute forward returns matching the label formula:
    # Ref($close, -N)/Ref($close, -1) - 1
    # = close[t+N] / close[t+1] - 1
    # = close.shift(-N) / close.shift(-1) - 1
    result = pd.DataFrame(index=spy_close.index)

    for n in FORWARD_WINDOWS:
        col_name = f"spy_fwd_return_{n}d"
        result[col_name] = spy_close.shift(-n) / spy_close.shift(-1) - 1
        valid = result[col_name].notna().sum()
        print(f"  {col_name}: {valid} valid values")

    # Drop rows where all forward returns are NaN (end of series)
    result = result.dropna(how="all")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Shape: {result.shape}, range: {result.index.min()} to {result.index.max()}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Download SPY prices and compute forward returns for market-neutral targets"
    )
    parser.add_argument("--start", type=str, default=DEFAULT_START_DATE,
                        help=f"Start date (default: {DEFAULT_START_DATE})")
    parser.add_argument("--end", type=str, default=DEFAULT_END_DATE,
                        help=f"End date (default: today)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if file exists")

    args = parser.parse_args()
    download_and_compute(args.start, args.end, args.force)


if __name__ == "__main__":
    main()
