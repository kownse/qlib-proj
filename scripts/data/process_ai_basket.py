"""
Compute AI basket features for quantitative trading models.

Downloads price data for core AI stocks from Yahoo Finance, computes an
equal-weight basket index, and derives features capturing AI-specific
market dynamics.

Features are date-aligned (same value for all stocks on a given date),
lagged by 1 day to avoid look-ahead bias.

Usage:
    python scripts/data/process_ai_basket.py
    python scripts/data/process_ai_basket.py --start 2005-01-01

Output:
    my_data/ai_basket/ai_basket_features.parquet
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "my_data" / "ai_basket"
OUTPUT_PATH = OUTPUT_DIR / "ai_basket_features.parquet"

# ---------------------------------------------------------------------------
# AI basket composition
# ---------------------------------------------------------------------------
# Core stocks that are direct beneficiaries of the AI revolution.
# Equal-weight; only available stocks are included on each date.
AI_BASKET_STOCKS = [
    "NVDA",   # AI GPU hardware
    "MSFT",   # OpenAI partnership, Copilot, Azure AI
    "GOOGL",  # Gemini, TPU, search AI
    "META",   # Llama, AI-driven ads
    "AMZN",   # AWS AI, Alexa
    "AMD",    # AI chips (MI300)
    "AVGO",   # AI networking (VMware, custom ASICs)
    "CRM",    # AI CRM (Einstein)
    "ORCL",   # Cloud AI infrastructure
    "ADBE",   # Creative AI (Firefly)
]

# Market benchmarks for relative performance
BENCHMARKS = ["SPY", "QQQ"]


def download_prices(symbols: list, start_date: str) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    all_symbols = symbols + BENCHMARKS
    print(f"Downloading {len(all_symbols)} symbols from Yahoo Finance...")

    data = yf.download(all_symbols, start=start_date, auto_adjust=True, progress=False)

    # yf.download returns MultiIndex columns (Price, Ticker) for multi-symbol
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = all_symbols

    # Strip timezone if present
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    print(f"Price data: {prices.shape}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # Show availability
    for sym in all_symbols:
        if sym in prices.columns:
            first_valid = prices[sym].first_valid_index()
            print(f"  {sym}: from {first_valid.date() if first_valid else 'N/A'}")

    return prices


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute AI basket features from stock prices."""
    basket_cols = [c for c in AI_BASKET_STOCKS if c in prices.columns]
    basket_prices = prices[basket_cols]

    # Individual stock daily returns
    basket_returns = basket_prices.pct_change()

    # Equal-weight basket return (mean of available stocks each day)
    basket_ret_daily = basket_returns.mean(axis=1)

    # Cumulative basket index (100-based)
    basket_index = (1 + basket_ret_daily).cumprod() * 100

    # SPY / QQQ returns for relative performance
    spy_close = prices["SPY"] if "SPY" in prices.columns else None
    qqq_close = prices["QQQ"] if "QQQ" in prices.columns else None

    features = pd.DataFrame(index=prices.index)

    # --- Returns at different horizons ---
    features["ai_basket_ret_1d"] = basket_ret_daily
    features["ai_basket_ret_5d"] = basket_index.pct_change(5)
    features["ai_basket_ret_20d"] = basket_index.pct_change(20)

    # --- Momentum ---
    features["ai_basket_ma5_ratio"] = basket_index / basket_index.rolling(5).mean()
    features["ai_basket_ma20_ratio"] = basket_index / basket_index.rolling(20).mean()

    # --- Volatility ---
    features["ai_basket_vol20"] = basket_ret_daily.rolling(20).std() * np.sqrt(252)

    # --- Relative to broad market ---
    if spy_close is not None:
        spy_ret_5d = spy_close.pct_change(5)
        features["ai_basket_vs_spy"] = features["ai_basket_ret_5d"] - spy_ret_5d

    if qqq_close is not None:
        qqq_ret_5d = qqq_close.pct_change(5)
        features["ai_basket_vs_qqq"] = features["ai_basket_ret_5d"] - qqq_ret_5d

    # --- Breadth ---
    # Fraction of basket stocks with positive 5-day return
    ret_5d_individual = basket_prices.pct_change(5)
    features["ai_basket_breadth"] = (ret_5d_individual > 0).mean(axis=1)

    # --- Drawdown from 60-day high ---
    rolling_max = basket_index.rolling(60, min_periods=1).max()
    features["ai_basket_dd60"] = basket_index / rolling_max - 1

    # --- Cross-sectional dispersion ---
    # High dispersion = divergent AI stock performance (idiosyncratic events)
    features["ai_basket_dispersion"] = basket_returns.std(axis=1)

    # --- Lag by 1 day to avoid look-ahead bias ---
    features = features.shift(1)

    # Drop rows where all features are NaN (initial period)
    features = features.dropna(how="all")

    return features


def main():
    parser = argparse.ArgumentParser(description="Process AI basket features")
    parser.add_argument("--start", default="2000-01-01", help="Start date (default: 2000-01-01)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prices = download_prices(AI_BASKET_STOCKS, args.start)
    features = compute_features(prices)

    features.to_parquet(OUTPUT_PATH)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Shape: {features.shape}")
    print(f"Date range: {features.index[0].date()} to {features.index[-1].date()}")
    print(f"\nFeature summary:")
    for col in features.columns:
        valid = features[col].notna()
        if valid.any():
            vals = features[col][valid]
            print(f"  {col:30s}  count={len(vals):5d}  "
                  f"range=[{vals.min():+.4f}, {vals.max():+.4f}]  "
                  f"mean={vals.mean():+.4f}")


if __name__ == "__main__":
    main()
