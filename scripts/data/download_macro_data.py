"""
Download macro/ETF data from Yahoo Finance and FRED for market regime features

Data Sources:
1. Yahoo Finance: ETFs, Indices, VIX derivatives
2. FRED: Treasury yields, credit spreads, economic indicators

Usage:
    python download_macro_data.py                    # Download all macro data
    python download_macro_data.py --symbols VIX GLD # Download specific symbols
    python download_macro_data.py --start 2010-01-01 # Custom start date
    python download_macro_data.py --no-fred          # Skip FRED data
"""

import argparse
import os
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

# =============================================================================
# Yahoo Finance Symbol Categories
# =============================================================================

# Original macro symbols (already existing)
CORE_MACRO_SYMBOLS = {
    "VIX": "^VIX",      # Volatility Index (no volume data)
    "GLD": "GLD",       # Gold ETF
    "TLT": "TLT",       # Long-term Treasury Bond ETF (20+ years)
    "UUP": "UUP",       # US Dollar Index ETF
    "USO": "USO",       # Crude Oil ETF
    "SHY": "SHY",       # Short-term Treasury ETF (1-3 years)
    "IEF": "IEF",       # Intermediate-term Treasury ETF (7-10 years)
}

# VIX derivatives (fear/greed sentiment)
VIX_DERIVATIVES = {
    "VIX3M": "^VIX3M",  # 3-month VIX (term structure)
    "UVXY": "UVXY",     # ProShares Ultra VIX Short-Term
    "SVXY": "SVXY",     # ProShares Short VIX Short-Term
}

# Sector ETFs (sector rotation signals)
SECTOR_ETFS = {
    "XLK": "XLK",   # Technology
    "XLF": "XLF",   # Financials
    "XLE": "XLE",   # Energy
    "XLV": "XLV",   # Healthcare
    "XLI": "XLI",   # Industrials
    "XLP": "XLP",   # Consumer Staples
    "XLY": "XLY",   # Consumer Discretionary
    "XLU": "XLU",   # Utilities
    "XLRE": "XLRE", # Real Estate
    "XLB": "XLB",   # Materials
    "XLC": "XLC",   # Communication Services
}

# Risk/Credit ETFs (credit risk signals)
RISK_ETFS = {
    "HYG": "HYG",   # iShares High Yield Corporate Bond
    "LQD": "LQD",   # iShares Investment Grade Corporate Bond
    "JNK": "JNK",   # SPDR Bloomberg High Yield Bond
}

# Global market ETFs (global linkage signals)
GLOBAL_ETFS = {
    "EEM": "EEM",   # iShares Emerging Markets
    "EFA": "EFA",   # iShares EAFE (Developed Markets ex-US)
    "FXI": "FXI",   # iShares China Large-Cap
    "EWJ": "EWJ",   # iShares Japan
}

# Market benchmarks (relative strength calculations)
MARKET_BENCHMARKS = {
    "SPY": "SPY",   # S&P 500 ETF (Large-cap)
    "QQQ": "QQQ",   # Nasdaq 100 ETF (Tech)
    "IWM": "IWM",   # Russell 2000 ETF (Small-cap)
}

# Combine all Yahoo Finance symbols
ALL_YAHOO_SYMBOLS = {
    **CORE_MACRO_SYMBOLS,
    **VIX_DERIVATIVES,
    **SECTOR_ETFS,
    **RISK_ETFS,
    **GLOBAL_ETFS,
    **MARKET_BENCHMARKS,
}

# Symbols that don't have volume data
NO_VOLUME_SYMBOLS = {"VIX", "VIX3M"}

# =============================================================================
# FRED Data Series
# =============================================================================

# Treasury yields (daily)
FRED_TREASURY = {
    "DGS2": "DGS2",           # 2-Year Treasury Constant Maturity Rate
    "DGS10": "DGS10",         # 10-Year Treasury Constant Maturity Rate
    "DGS30": "DGS30",         # 30-Year Treasury Constant Maturity Rate
    "DGS3MO": "DGS3MO",       # 3-Month Treasury Bill Rate
}

# Yield curve spreads (daily)
FRED_YIELD_CURVE = {
    "T10Y2Y": "T10Y2Y",       # 10-Year minus 2-Year Treasury Spread
    "T10Y3M": "T10Y3M",       # 10-Year minus 3-Month Treasury Spread
}

# Credit spreads (daily)
FRED_CREDIT = {
    "BAMLH0A0HYM2": "BAMLH0A0HYM2",   # ICE BofA US High Yield Index Option-Adjusted Spread
    "BAMLC0A0CM": "BAMLC0A0CM",       # ICE BofA US Corporate Index Option-Adjusted Spread
}

# Dollar index (daily)
FRED_DOLLAR = {
    "DTWEXBGS": "DTWEXBGS",   # Trade Weighted U.S. Dollar Index: Broad, Goods and Services
}

# Combine all FRED series
ALL_FRED_SERIES = {
    **FRED_TREASURY,
    **FRED_YIELD_CURVE,
    **FRED_CREDIT,
    **FRED_DOLLAR,
}

# For backward compatibility
MACRO_SYMBOLS = ALL_YAHOO_SYMBOLS


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


def download_fred_data(
    series_dict: dict,
    start_date: str,
    end_date: str,
    output_dir: Path,
    force: bool = False
) -> tuple:
    """
    Download data from FRED (Federal Reserve Economic Data).

    Args:
        series_dict: Dictionary of {name: series_id}
        start_date: Start date string
        end_date: End date string
        output_dir: Output directory for CSV files
        force: Force re-download even if files exist

    Returns:
        Tuple of (success_list, failed_list)
    """
    # Try to import fredapi
    try:
        from fredapi import Fred
    except ImportError:
        print("  fredapi not installed. Run: pip install fredapi")
        return [], list(series_dict.keys())

    # Get FRED API key from environment
    fred_api_key = os.environ.get("FRED_API_KEY")
    if not fred_api_key:
        # Try loading from .env file
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("FRED_API_KEY="):
                        fred_api_key = line.strip().split("=", 1)[1].strip('"\'')
                        break

    if not fred_api_key:
        print("  FRED_API_KEY not found. Set it in .env or environment variable.")
        print("  Register at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return [], list(series_dict.keys())

    fred = Fred(api_key=fred_api_key)
    success = []
    failed = []

    for name, series_id in series_dict.items():
        csv_path = output_dir / f"FRED_{name}.csv"

        if csv_path.exists() and not force:
            print(f"  Skipped FRED_{name} (already exists)")
            success.append(name)
            continue

        try:
            series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)

            if series is None or series.empty:
                print(f"  FRED_{name}: No data returned")
                failed.append(name)
                continue

            # Convert to DataFrame
            df = pd.DataFrame({"value": series})
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
            df.index.name = "date"

            # Save CSV
            df.to_csv(csv_path)
            print(f"  FRED_{name}: OK ({len(df)} rows, {df.index.min().date()} to {df.index.max().date()})")
            success.append(name)

        except Exception as e:
            print(f"  FRED_{name}: Error - {e}")
            failed.append(name)

    return success, failed


def download_macro_data(
    symbols: list = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    output_dir: Path = MACRO_CSV_DIR,
    force: bool = False,
    include_fred: bool = True
) -> tuple:
    """
    Download all macro data from Yahoo Finance and FRED.

    Args:
        symbols: List of symbol names to download. If None, download all.
        start_date: Start date string
        end_date: End date string
        output_dir: Output directory for CSV files
        force: Force re-download even if files exist
        include_fred: Whether to include FRED data

    Returns:
        Tuple of (success_list, failed_list)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_success = []
    all_failed = []

    # Determine which Yahoo symbols to download
    if symbols is None:
        yahoo_symbols = ALL_YAHOO_SYMBOLS
    else:
        yahoo_symbols = {k: v for k, v in ALL_YAHOO_SYMBOLS.items() if k in symbols}

    # Download Yahoo Finance data
    print("=" * 60)
    print("Downloading Yahoo Finance Data")
    print("=" * 60)
    print(f"Symbols: {len(yahoo_symbols)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output directory: {output_dir}")
    print()

    # Download by category for better organization
    categories = [
        ("Core Macro", CORE_MACRO_SYMBOLS),
        ("VIX Derivatives", VIX_DERIVATIVES),
        ("Sector ETFs", SECTOR_ETFS),
        ("Risk/Credit ETFs", RISK_ETFS),
        ("Global ETFs", GLOBAL_ETFS),
        ("Market Benchmarks", MARKET_BENCHMARKS),
    ]

    for cat_name, cat_symbols in categories:
        cat_to_download = {k: v for k, v in cat_symbols.items() if symbols is None or k in symbols}
        if not cat_to_download:
            continue

        print(f"\n[{cat_name}]")
        for name, ticker in cat_to_download.items():
            if download_single_symbol(name, ticker, start_date, end_date, output_dir, force):
                all_success.append(name)
            else:
                all_failed.append(name)

    # Download FRED data
    if include_fred:
        print("\n" + "=" * 60)
        print("Downloading FRED Data")
        print("=" * 60)

        fred_categories = [
            ("Treasury Yields", FRED_TREASURY),
            ("Yield Curve Spreads", FRED_YIELD_CURVE),
            ("Credit Spreads", FRED_CREDIT),
            ("Dollar Index", FRED_DOLLAR),
        ]

        for cat_name, cat_series in fred_categories:
            print(f"\n[{cat_name}]")
            success, failed = download_fred_data(cat_series, start_date, end_date, output_dir, force)
            all_success.extend([f"FRED_{s}" for s in success])
            all_failed.extend([f"FRED_{s}" for s in failed])

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Success: {len(all_success)}")
    print(f"Failed: {len(all_failed)}")
    if all_failed:
        print(f"Failed items: {all_failed}")

    return all_success, all_failed


def main():
    parser = argparse.ArgumentParser(description="Download macro/ETF data from Yahoo Finance and FRED")
    parser.add_argument(
        "--symbols",
        nargs="+",
        choices=list(ALL_YAHOO_SYMBOLS.keys()),
        help="Specific Yahoo Finance symbols to download (default: all)"
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
    parser.add_argument(
        "--no-fred",
        action="store_true",
        help="Skip FRED data download"
    )

    args = parser.parse_args()

    download_macro_data(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        output_dir=Path(args.output),
        force=args.force,
        include_fred=not args.no_fred
    )


if __name__ == "__main__":
    main()
