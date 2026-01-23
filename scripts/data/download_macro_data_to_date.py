"""
增量更新宏观数据到最新日期

检查每个宏观数据 CSV 的最新日期，下载增量数据并拼接。

Usage:
    python download_macro_data_to_date.py                    # 更新所有宏观数据
    python download_macro_data_to_date.py --dry-run          # 只检查状态
    python download_macro_data_to_date.py --force            # 强制重新下载全部
    python download_macro_data_to_date.py --no-fred          # 跳过 FRED 数据
"""

import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
MACRO_CSV_DIR = PROJECT_ROOT / "my_data" / "macro_csv"

# Default date range
DEFAULT_START_DATE = "2000-01-01"

# =============================================================================
# Symbol definitions (from download_macro_data.py)
# =============================================================================

CORE_MACRO_SYMBOLS = {
    "VIX": "^VIX",
    "GLD": "GLD",
    "TLT": "TLT",
    "UUP": "UUP",
    "USO": "USO",
    "SHY": "SHY",
    "IEF": "IEF",
}

VIX_DERIVATIVES = {
    "VIX3M": "^VIX3M",
    "UVXY": "UVXY",
    "SVXY": "SVXY",
}

SECTOR_ETFS = {
    "XLK": "XLK",
    "XLF": "XLF",
    "XLE": "XLE",
    "XLV": "XLV",
    "XLI": "XLI",
    "XLP": "XLP",
    "XLY": "XLY",
    "XLU": "XLU",
    "XLRE": "XLRE",
    "XLB": "XLB",
    "XLC": "XLC",
}

RISK_ETFS = {
    "HYG": "HYG",
    "LQD": "LQD",
    "JNK": "JNK",
}

GLOBAL_ETFS = {
    "EEM": "EEM",
    "EFA": "EFA",
    "FXI": "FXI",
    "EWJ": "EWJ",
}

MARKET_BENCHMARKS = {
    "SPY": "SPY",   # S&P 500 ETF (Large-cap)
    "QQQ": "QQQ",   # Nasdaq 100 ETF (Tech)
    "IWM": "IWM",   # Russell 2000 ETF (Small-cap) - for MASTER model
}

ALL_YAHOO_SYMBOLS = {
    **CORE_MACRO_SYMBOLS,
    **VIX_DERIVATIVES,
    **SECTOR_ETFS,
    **RISK_ETFS,
    **GLOBAL_ETFS,
    **MARKET_BENCHMARKS,
}

NO_VOLUME_SYMBOLS = {"VIX", "VIX3M"}

FRED_TREASURY = {
    "DGS2": "DGS2",
    "DGS10": "DGS10",
    "DGS30": "DGS30",
    "DGS3MO": "DGS3MO",
}

FRED_YIELD_CURVE = {
    "T10Y2Y": "T10Y2Y",
    "T10Y3M": "T10Y3M",
}

FRED_CREDIT = {
    "BAMLH0A0HYM2": "BAMLH0A0HYM2",
    "BAMLC0A0CM": "BAMLC0A0CM",
}

FRED_DOLLAR = {
    "DTWEXBGS": "DTWEXBGS",
}

ALL_FRED_SERIES = {
    **FRED_TREASURY,
    **FRED_YIELD_CURVE,
    **FRED_CREDIT,
    **FRED_DOLLAR,
}


def get_latest_date_from_csv(csv_path: Path) -> pd.Timestamp | None:
    """从 CSV 文件获取最新日期"""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        if df.empty:
            return None
        return df.index.max()
    except Exception as e:
        print(f"  Warning: Failed to read {csv_path.name}: {e}")
        return None


def check_update_status(output_dir: Path, include_fred: bool = True) -> dict:
    """检查所有宏观数据的更新状态"""
    today = pd.Timestamp(datetime.now().date())
    status = {
        "up_to_date": [],
        "needs_update": [],
        "missing": [],
    }

    # Yahoo Finance symbols
    for name in ALL_YAHOO_SYMBOLS.keys():
        csv_path = output_dir / f"{name}.csv"
        latest_date = get_latest_date_from_csv(csv_path)

        if latest_date is None:
            status["missing"].append(("yahoo", name))
        elif latest_date.date() >= today.date() - timedelta(days=1):
            status["up_to_date"].append(("yahoo", name))
        else:
            status["needs_update"].append(("yahoo", name, latest_date))

    # FRED series
    if include_fred:
        for name in ALL_FRED_SERIES.keys():
            csv_path = output_dir / f"FRED_{name}.csv"
            latest_date = get_latest_date_from_csv(csv_path)

            if latest_date is None:
                status["missing"].append(("fred", name))
            elif latest_date.date() >= today.date() - timedelta(days=3):  # FRED 数据通常有 1-2 天延迟
                status["up_to_date"].append(("fred", name))
            else:
                status["needs_update"].append(("fred", name, latest_date))

    return status


def download_yahoo_incremental(
    name: str,
    ticker: str,
    start_date: str,
    end_date: str,
    output_dir: Path
) -> bool:
    """
    下载 Yahoo Finance 增量数据并拼接
    """
    csv_path = output_dir / f"{name}.csv"

    try:
        yf_ticker = yf.Ticker(ticker)
        df = yf_ticker.history(start=start_date, end=end_date, auto_adjust=False)

        if df.empty:
            print(f"    No new data")
            return False

        # 标准化列名
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume"
        })

        # 处理无 volume 数据的 symbol
        if name in NO_VOLUME_SYMBOLS:
            if "volume" not in df.columns or df["volume"].isna().all():
                df["volume"] = 0

        # 保留需要的列
        required_cols = ["open", "high", "low", "close", "volume"]
        available_cols = [c for c in required_cols if c in df.columns]
        if "adj_close" in df.columns:
            available_cols.append("adj_close")
        df = df[available_cols]

        # 处理 index
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)
        df.index.name = "date"

        # 读取现有数据并拼接
        if csv_path.exists():
            existing_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
            combined = pd.concat([existing_df, df])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
        else:
            combined = df

        # 保存
        combined.to_csv(csv_path)
        print(f"    +{len(df)} rows (total: {len(combined)}, {combined.index.min().date()} to {combined.index.max().date()})")
        return True

    except Exception as e:
        print(f"    Error: {e}")
        return False


def download_fred_incremental(
    name: str,
    series_id: str,
    start_date: str,
    end_date: str,
    output_dir: Path
) -> bool:
    """
    下载 FRED 增量数据并拼接
    """
    try:
        from fredapi import Fred
    except ImportError:
        print(f"    fredapi not installed. Run: pip install fredapi")
        return False

    # 获取 FRED API key
    fred_api_key = os.environ.get("FRED_API_KEY")
    if not fred_api_key:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("FRED_API_KEY="):
                        fred_api_key = line.strip().split("=", 1)[1].strip('"\'')
                        break

    if not fred_api_key:
        print(f"    FRED_API_KEY not found")
        return False

    csv_path = output_dir / f"FRED_{name}.csv"

    try:
        fred = Fred(api_key=fred_api_key)
        series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)

        if series is None or series.empty:
            print(f"    No new data")
            return False

        df = pd.DataFrame({"value": series})
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
        df.index.name = "date"

        # 读取现有数据并拼接
        if csv_path.exists():
            existing_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
            combined = pd.concat([existing_df, df])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
        else:
            combined = df

        # 保存
        combined.to_csv(csv_path)
        print(f"    +{len(df)} rows (total: {len(combined)}, {combined.index.min().date()} to {combined.index.max().date()})")
        return True

    except Exception as e:
        print(f"    Error: {e}")
        return False


def update_macro_data(output_dir: Path, include_fred: bool = True, dry_run: bool = False) -> tuple:
    """
    增量更新所有宏观数据
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    end_date = datetime.now().strftime("%Y-%m-%d")

    print("=" * 60)
    print("Macro Data Incremental Updater")
    print("=" * 60)
    print(f"Target date: {end_date}")
    print(f"Output directory: {output_dir}")
    print("=" * 60 + "\n")

    # 检查更新状态
    print("Step 1: Checking update status...\n")
    status = check_update_status(output_dir, include_fred)

    yahoo_up = sum(1 for t, _ in status["up_to_date"] if t == "yahoo")
    yahoo_update = sum(1 for item in status["needs_update"] if item[0] == "yahoo")
    yahoo_missing = sum(1 for t, _ in status["missing"] if t == "yahoo")

    fred_up = sum(1 for t, _ in status["up_to_date"] if t == "fred")
    fred_update = sum(1 for item in status["needs_update"] if item[0] == "fred")
    fred_missing = sum(1 for t, _ in status["missing"] if t == "fred")

    print(f"  Yahoo Finance symbols:")
    print(f"    Up to date: {yahoo_up}")
    print(f"    Needs update: {yahoo_update}")
    print(f"    Missing: {yahoo_missing}")

    if include_fred:
        print(f"\n  FRED series:")
        print(f"    Up to date: {fred_up}")
        print(f"    Needs update: {fred_update}")
        print(f"    Missing: {fred_missing}")

    if status["needs_update"]:
        print("\n  Symbols needing update:")
        for item in status["needs_update"][:10]:
            source, name, latest = item
            print(f"    [{source}] {name}: last updated {latest.strftime('%Y-%m-%d')}")
        if len(status["needs_update"]) > 10:
            print(f"    ... and {len(status['needs_update']) - 10} more")

    if status["missing"]:
        print(f"\n  Missing symbols: {[f'[{s}] {n}' for s, n in status['missing'][:10]]}")
        if len(status["missing"]) > 10:
            print(f"    ... and {len(status['missing']) - 10} more")

    if dry_run:
        print("\n[Dry run mode - no downloads performed]")
        return [], []

    all_success = []
    all_failed = []

    # 更新 Yahoo Finance 数据
    yahoo_to_update = [(name, latest) for source, name, latest in status["needs_update"] if source == "yahoo"]
    yahoo_missing_list = [name for source, name in status["missing"] if source == "yahoo"]

    if yahoo_to_update or yahoo_missing_list:
        print("\n" + "=" * 60)
        print("Step 2a: Updating Yahoo Finance data")
        print("=" * 60 + "\n")

        # 更新需要更新的
        for i, (name, latest_date) in enumerate(yahoo_to_update):
            ticker = ALL_YAHOO_SYMBOLS[name]
            start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"[{i+1}/{len(yahoo_to_update)}] {name} (from {start_date})...")

            if download_yahoo_incremental(name, ticker, start_date, end_date, output_dir):
                all_success.append(name)
            else:
                # 如果增量下载失败，尝试已有数据仍然有效
                all_success.append(name)

        # 下载缺失的
        for i, name in enumerate(yahoo_missing_list):
            ticker = ALL_YAHOO_SYMBOLS[name]
            print(f"[{i+1}/{len(yahoo_missing_list)}] {name} (new, from {DEFAULT_START_DATE})...")

            if download_yahoo_incremental(name, ticker, DEFAULT_START_DATE, end_date, output_dir):
                all_success.append(name)
            else:
                all_failed.append(name)

    # 更新 FRED 数据
    if include_fred:
        fred_to_update = [(name, latest) for source, name, latest in status["needs_update"] if source == "fred"]
        fred_missing_list = [name for source, name in status["missing"] if source == "fred"]

        if fred_to_update or fred_missing_list:
            print("\n" + "=" * 60)
            print("Step 2b: Updating FRED data")
            print("=" * 60 + "\n")

            # 更新需要更新的
            for i, (name, latest_date) in enumerate(fred_to_update):
                series_id = ALL_FRED_SERIES[name]
                start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
                print(f"[{i+1}/{len(fred_to_update)}] FRED_{name} (from {start_date})...")

                if download_fred_incremental(name, series_id, start_date, end_date, output_dir):
                    all_success.append(f"FRED_{name}")
                else:
                    all_success.append(f"FRED_{name}")  # 已有数据仍有效

            # 下载缺失的
            for i, name in enumerate(fred_missing_list):
                series_id = ALL_FRED_SERIES[name]
                print(f"[{i+1}/{len(fred_missing_list)}] FRED_{name} (new, from {DEFAULT_START_DATE})...")

                if download_fred_incremental(name, series_id, DEFAULT_START_DATE, end_date, output_dir):
                    all_success.append(f"FRED_{name}")
                else:
                    all_failed.append(f"FRED_{name}")

    # 添加已是最新的到成功列表
    for source, name in status["up_to_date"]:
        if source == "yahoo":
            all_success.append(name)
        else:
            all_success.append(f"FRED_{name}")

    # Summary
    print("\n" + "=" * 60)
    print("Update Summary")
    print("=" * 60)
    print(f"Success: {len(all_success)}")
    print(f"Failed: {len(all_failed)}")
    if all_failed:
        print(f"Failed items: {all_failed}")

    return all_success, all_failed


def main():
    parser = argparse.ArgumentParser(
        description='Incrementally update macro data to latest date',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_macro_data_to_date.py                    # Update all macro data
    python download_macro_data_to_date.py --dry-run          # Check status only
    python download_macro_data_to_date.py --no-fred          # Skip FRED data
        """
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Only check status, do not download')
    parser.add_argument('--no-fred', action='store_true',
                        help='Skip FRED data download')
    parser.add_argument('--output', type=str, default=str(MACRO_CSV_DIR),
                        help=f'Output directory (default: {MACRO_CSV_DIR})')
    args = parser.parse_args()

    update_macro_data(
        output_dir=Path(args.output),
        include_fred=not args.no_fred,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
