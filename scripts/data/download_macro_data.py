"""
Download macro/ETF data from Yahoo Finance for market regime features

Usage:
    python download_macro_data.py                    # Download all macro data
    python download_macro_data.py --symbols VIX GLD # Download specific symbols
    python download_macro_data.py --start 2010-01-01 # Custom start date
"""

import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
MACRO_CSV_DIR = PROJECT_ROOT / "my_data" / "macro_csv"

# Default date range
DEFAULT_START_DATE = "2000-01-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")

# Macro symbol mapping: internal name -> Yahoo Finance ticker
MACRO_SYMBOLS = {
    "VIX": "^VIX",    # Volatility Index (no volume data)
    "GLD": "GLD",     # Gold ETF
    "TLT": "TLT",     # Long-term Treasury Bond ETF (20+ years)
    "UUP": "UUP",     # US Dollar Index ETF
    "USO": "USO",     # Crude Oil ETF
    "SHY": "SHY",     # Short-term Treasury ETF (1-3 years)
    "IEF": "IEF",     # Intermediate-term Treasury ETF (7-10 years)
}

# Symbols that don't have volume data
NO_VOLUME_SYMBOLS = {"VIX"}


def download_single_symbol(
    name: str,
    ticker: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    force: bool = False
) -> bool:
    """
    Download data for a single symbol from Yahoo Finance.

    Args:
        name: Internal symbol name (e.g., "VIX")
        ticker: Yahoo Finance ticker (e.g., "^VIX")
        start_date: Start date string
        end_date: End date string
        output_dir: Output directory for CSV files
        force: Force re-download even if file exists

    Returns:
        True if successful, False otherwise
    """
    csv_path = output_dir / f"{name}.csv"

    if csv_path.exists() and not force:
        print(f"  Skipped {name} (already exists)")
        return True

    try:
        yf_ticker = yf.Ticker(ticker)
        df = yf_ticker.history(start=start_date, end=end_date, auto_adjust=False)

        if df.empty:
            print(f"  {name}: No data returned")
            return False

        # Standardize column names (Qlib expects lowercase)
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume"
        })

        # Handle symbols without volume data (e.g., VIX)
        if name in NO_VOLUME_SYMBOLS:
            if "volume" not in df.columns or df["volume"].isna().all():
                df["volume"] = 0

        # Keep required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        available_cols = [c for c in required_cols if c in df.columns]

        # Add adj_close if available
        if "adj_close" in df.columns:
            available_cols.append("adj_close")

        df = df[available_cols]

        # Process index (date)
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)  # Remove timezone
        df.index.name = "date"

        # Save CSV
        df.to_csv(csv_path)
        print(f"  {name}: OK ({len(df)} rows, {df.index.min().date()} to {df.index.max().date()})")
        return True

    except Exception as e:
        print(f"  {name}: Error - {e}")
        return False


def download_macro_data(
    symbols: list = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    output_dir: Path = MACRO_CSV_DIR,
    force: bool = False
) -> tuple:
    """
    Download all macro data from Yahoo Finance.

    Args:
        symbols: List of symbol names to download. If None, download all.
        start_date: Start date string
        end_date: End date string
        output_dir: Output directory for CSV files
        force: Force re-download even if files exist

    Returns:
        Tuple of (success_list, failed_list)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which symbols to download
    if symbols is None:
        symbols_to_download = MACRO_SYMBOLS
    else:
        symbols_to_download = {k: v for k, v in MACRO_SYMBOLS.items() if k in symbols}

    print(f"Downloading macro data: {list(symbols_to_download.keys())}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output directory: {output_dir}")
    print()

    success = []
    failed = []

    for name, ticker in symbols_to_download.items():
        if download_single_symbol(name, ticker, start_date, end_date, output_dir, force):
            success.append(name)
        else:
            failed.append(name)

    print()
    print(f"Summary: {len(success)} success, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")

    return success, failed


def main():
    parser = argparse.ArgumentParser(description="Download macro/ETF data from Yahoo Finance")
    parser.add_argument(
        "--symbols",
        nargs="+",
        choices=list(MACRO_SYMBOLS.keys()),
        help="Specific symbols to download (default: all)"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=DEFAULT_START_DATE,
        help=f"Start date (default: {DEFAULT_START_DATE})"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=DEFAULT_END_DATE,
        help=f"End date (default: {DEFAULT_END_DATE})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(MACRO_CSV_DIR),
        help=f"Output directory (default: {MACRO_CSV_DIR})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )

    args = parser.parse_args()

    download_macro_data(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        output_dir=Path(args.output),
        force=args.force
    )


if __name__ == "__main__":
    main()
