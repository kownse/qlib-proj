"""
Download sector/industry data from Yahoo Finance for SP500 stocks.

Fetches sector and industry classification for each stock and caches
to a JSON file. Only needs to run once (or periodically to update).

Usage:
    python scripts/data/download_sector_data.py
    python scripts/data/download_sector_data.py --pool tech
"""

import argparse
import json
import time
from pathlib import Path

import yfinance as yf

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SECTOR_DATA_DIR = PROJECT_ROOT / "my_data" / "sector_data"
DEFAULT_OUTPUT = SECTOR_DATA_DIR / "sector_info.json"


def download_sector_info(symbols: list, output_path: Path = DEFAULT_OUTPUT, delay: float = 0.1):
    """
    Download sector/industry info for each symbol from Yahoo Finance.

    Args:
        symbols: List of stock ticker symbols
        output_path: Where to save the JSON file
        delay: Seconds between API calls to avoid rate limiting
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data to avoid re-downloading
    existing = {}
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing entries from {output_path}")

    # Only download missing symbols
    missing = [s for s in symbols if s not in existing]
    print(f"Need to download {len(missing)} of {len(symbols)} symbols")

    result = dict(existing)
    errors = []

    for i, symbol in enumerate(missing):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector = info.get("sector", "Unknown")
            industry = info.get("industry", "Unknown")

            result[symbol] = {
                "sector": sector,
                "industry": industry,
            }

            if (i + 1) % 20 == 0 or (i + 1) == len(missing):
                print(f"  [{i+1}/{len(missing)}] {symbol}: {sector} / {industry}")
                # Save intermediate results
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2, sort_keys=True)

            time.sleep(delay)

        except Exception as e:
            print(f"  Error for {symbol}: {e}")
            errors.append(symbol)
            result[symbol] = {"sector": "Unknown", "industry": "Unknown"}

    # Final save
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(f"\nSaved {len(result)} entries to {output_path}")
    if errors:
        print(f"Errors for {len(errors)} symbols: {errors}")

    # Print sector distribution
    sectors = {}
    for sym, info in result.items():
        s = info["sector"]
        sectors[s] = sectors.get(s, 0) + 1
    print("\nSector distribution:")
    for s, count in sorted(sectors.items(), key=lambda x: -x[1]):
        print(f"  {s}: {count}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Download sector/industry data from Yahoo Finance")
    parser.add_argument("--pool", default="sp500", choices=["test", "tech", "sp100", "sp500"],
                        help="Stock pool to download")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    # Import stock pools
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.stock_pools import STOCK_POOLS

    symbols = STOCK_POOLS[args.pool]
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT

    print(f"Downloading sector data for {len(symbols)} stocks ({args.pool} pool)")
    download_sector_info(symbols, output_path, delay=args.delay)


if __name__ == "__main__":
    main()
