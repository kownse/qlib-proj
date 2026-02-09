"""
Download CBOE options-derived indices from Yahoo Finance.

Downloads:
- ^SKEW: CBOE Skew Index (tail risk indicator)
- ^VIX9D: 9-day VIX (short-term implied volatility)
- ^VVIX: VIX of VIX (volatility of volatility)

These complement the existing VIX data with additional dimensions of
options market sentiment.

Usage:
    python scripts/data/download_cboe_data.py
    python scripts/data/download_cboe_data.py --force
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CBOE_CSV_DIR = PROJECT_ROOT / "my_data" / "cboe_csv"

DEFAULT_START_DATE = "2000-01-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")

# CBOE indices to download
CBOE_SYMBOLS = {
    "SKEW": "^SKEW",       # CBOE Skew Index (tail risk, available ~2011+)
    "VIX9D": "^VIX9D",     # 9-day VIX (short-term fear, available ~2011+)
    "VVIX": "^VVIX",       # VIX of VIX (vol of vol, available ~2007+)
}


def download_cboe_data(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    force: bool = False,
) -> tuple:
    """
    Download CBOE indices from Yahoo Finance.

    Returns:
        Tuple of (success_list, failed_list)
    """
    CBOE_CSV_DIR.mkdir(parents=True, exist_ok=True)

    success = []
    failed = []

    print("=" * 60)
    print("Downloading CBOE Options-Derived Indices")
    print("=" * 60)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output: {CBOE_CSV_DIR}")
    print()

    for name, ticker in CBOE_SYMBOLS.items():
        csv_path = CBOE_CSV_DIR / f"{name}.csv"

        if csv_path.exists() and not force:
            print(f"  Skipped {name} (already exists)")
            success.append(name)
            continue

        try:
            t = yf.Ticker(ticker)
            df = t.history(start=start_date, end=end_date, auto_adjust=False)

            if df.empty:
                print(f"  {name}: No data returned")
                failed.append(name)
                continue

            # Standardize columns
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            })

            # These indices don't have meaningful volume
            if "volume" not in df.columns or df.get("volume", pd.Series()).isna().all():
                df["volume"] = 0

            # Keep standard columns
            keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[keep_cols]

            # Clean index
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)
            df.index.name = "date"

            df.to_csv(csv_path)
            print(f"  {name}: OK ({len(df)} rows, {df.index.min().date()} to {df.index.max().date()})")
            success.append(name)

        except Exception as e:
            print(f"  {name}: Error - {e}")
            failed.append(name)

    # Summary
    print(f"\nSuccess: {len(success)}, Failed: {len(failed)}")
    if failed:
        print(f"Failed: {failed}")

    return success, failed


def main():
    parser = argparse.ArgumentParser(
        description="Download CBOE options-derived indices (SKEW, VIX9D, VVIX)"
    )
    parser.add_argument("--start", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--end", type=str, default=DEFAULT_END_DATE)
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if files exist")
    args = parser.parse_args()

    download_cboe_data(args.start, args.end, args.force)


if __name__ == "__main__":
    main()
