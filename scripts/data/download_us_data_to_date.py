"""
增量更新美股数据到最新日期

遍历 SP500 所有股票，检查每只股票的最新日期，下载增量数据并拼接到现有 CSV。
最后生成 Qlib 格式数据。

使用方法:
    python download_us_data_to_date.py                    # 更新所有 SP500 股票
    python download_us_data_to_date.py --pool sp100       # 只更新 SP100 股票
    python download_us_data_to_date.py --convert-only     # 只转换，不下载
    python download_us_data_to_date.py --dry-run          # 只检查，不下载
"""
import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# 确保项目路径正确
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "qlib"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd
import yfinance as yf

from data.stock_pools import STOCK_POOLS, SP500_SYMBOLS

# 输出路径
CSV_DIR = PROJECT_ROOT / "my_data" / "csv_us"
QLIB_DIR = PROJECT_ROOT / "my_data" / "qlib_us"

# 默认起始日期（用于新股票）
DEFAULT_START_DATE = "2000-01-01"


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
        print(f"  Warning: Failed to read {csv_path}: {e}")
        return None


def check_update_status(symbols: list, csv_dir: Path) -> dict:
    """检查所有股票的更新状态"""
    today = pd.Timestamp(datetime.now().date())
    status = {
        "up_to_date": [],
        "needs_update": [],
        "missing": [],
    }

    for symbol in symbols:
        csv_path = csv_dir / f"{symbol}.csv"
        latest_date = get_latest_date_from_csv(csv_path)

        if latest_date is None:
            status["missing"].append(symbol)
        elif latest_date.date() >= today.date() - timedelta(days=1):
            # 允许 1 天的延迟（周末/节假日）
            status["up_to_date"].append(symbol)
        else:
            status["needs_update"].append((symbol, latest_date))

    return status


def download_incremental_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """下载增量数据"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

        if df.empty:
            return None

        # 标准化列名
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume"
        })

        # 保留需要的列
        df = df[["open", "high", "low", "close", "adj_close", "volume"]]

        # 处理 index
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)
        df.index.name = "date"

        return df

    except Exception as e:
        print(f"  Error downloading {symbol}: {e}")
        return None


def update_stock_data(symbols_to_update: list, csv_dir: Path, end_date: str):
    """更新股票数据（增量下载并拼接）"""
    csv_dir.mkdir(parents=True, exist_ok=True)

    success = []
    failed = []

    for i, item in enumerate(symbols_to_update):
        if isinstance(item, tuple):
            symbol, latest_date = item
            # 从最新日期的下一天开始下载
            start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            symbol = item
            latest_date = None
            start_date = DEFAULT_START_DATE

        csv_path = csv_dir / f"{symbol}.csv"

        print(f"[{i+1}/{len(symbols_to_update)}] Updating {symbol}...", end=" ")

        if latest_date:
            print(f"(from {start_date})", end=" ")
        else:
            print(f"(new, from {start_date})", end=" ")

        # 下载增量数据
        new_data = download_incremental_data(symbol, start_date, end_date)

        if new_data is None or new_data.empty:
            print("No new data")
            if latest_date:
                success.append(symbol)  # 已有数据，只是没有新数据
            else:
                failed.append(symbol)  # 完全没有数据
            continue

        # 读取现有数据并拼接
        if csv_path.exists() and latest_date:
            existing_data = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
            # 拼接，去重
            combined = pd.concat([existing_data, new_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
        else:
            combined = new_data

        # 保存
        combined.to_csv(csv_path)
        print(f"OK (+{len(new_data)} rows, total {len(combined)} rows)")
        success.append(symbol)

    return success, failed


def download_missing_data(symbols: list, csv_dir: Path, end_date: str):
    """下载缺失的股票数据"""
    csv_dir.mkdir(parents=True, exist_ok=True)

    success = []
    failed = []

    for i, symbol in enumerate(symbols):
        csv_path = csv_dir / f"{symbol}.csv"

        print(f"[{i+1}/{len(symbols)}] Downloading {symbol}...", end=" ")

        new_data = download_incremental_data(symbol, DEFAULT_START_DATE, end_date)

        if new_data is None or new_data.empty:
            print("No data")
            failed.append(symbol)
            continue

        new_data.to_csv(csv_path)
        print(f"OK ({len(new_data)} rows)")
        success.append(symbol)

    return success, failed


def convert_to_qlib_bin(csv_dir: Path, qlib_dir: Path):
    """将 CSV 转换为 Qlib bin 格式"""
    # 尝试多个可能的路径
    possible_paths = [
        PROJECT_ROOT / "qlib-src" / "scripts",
        PROJECT_ROOT / "qlib" / "scripts",
    ]

    for scripts_path in possible_paths:
        if scripts_path.exists():
            if str(scripts_path) not in sys.path:
                sys.path.insert(0, str(scripts_path))
            break

    from dump_bin import DumpDataAll

    # 清理旧数据
    if qlib_dir.exists():
        shutil.rmtree(qlib_dir)
    qlib_dir.mkdir(parents=True)

    print(f"\nConverting CSV to Qlib format...")
    print(f"  Source: {csv_dir}")
    print(f"  Target: {qlib_dir}")

    dumper = DumpDataAll(
        data_path=str(csv_dir),
        qlib_dir=str(qlib_dir),
        freq="day",
        date_field_name="date",
        file_suffix=".csv",
        include_fields="open,high,low,close,adj_close,volume",
    )
    dumper.dump()

    print("Conversion completed!")


def create_instruments_file(symbols: list, qlib_dir: Path, pool_name: str = "sp500"):
    """创建股票池文件，根据实际 CSV 数据确定日期范围

    注意：Qlib dump_bin 会将股票代码转为小写，所以 instruments 文件也要用小写
    """
    instruments_dir = qlib_dir / "instruments"
    instruments_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = qlib_dir.parent / "csv_us"

    # 收集每只股票的实际日期范围
    stock_ranges = []
    for symbol in symbols:
        csv_path = csv_dir / f"{symbol}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
                if not df.empty:
                    start = df.index.min().strftime("%Y-%m-%d")
                    end = df.index.max().strftime("%Y-%m-%d")
                    # 使用小写符号以匹配 Qlib dump_bin 的行为
                    stock_ranges.append((symbol.lower(), start, end))
            except Exception:
                pass

    # all.txt
    with open(instruments_dir / "all.txt", "w") as f:
        for symbol, start, end in stock_ranges:
            f.write(f"{symbol}\t{start}\t{end}\n")

    # 对应股票池文件
    # 也需要将 pool_symbols 转为小写进行比较
    pool_symbols = set(s.lower() for s in STOCK_POOLS.get(pool_name, symbols))
    with open(instruments_dir / f"{pool_name}.txt", "w") as f:
        for symbol, start, end in stock_ranges:
            if symbol in pool_symbols:
                f.write(f"{symbol}\t{start}\t{end}\n")

    print(f"Created instruments files at {instruments_dir}")
    print(f"  - all.txt ({len(stock_ranges)} stocks)")
    print(f"  - {pool_name}.txt ({sum(1 for s,_,_ in stock_ranges if s in pool_symbols)} stocks)")


def main():
    parser = argparse.ArgumentParser(
        description='Incrementally update US stock data to latest date',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_us_data_to_date.py                    # Update all SP500 stocks
    python download_us_data_to_date.py --pool sp100       # Update only SP100 stocks
    python download_us_data_to_date.py --convert-only     # Only convert existing CSVs
    python download_us_data_to_date.py --dry-run          # Check status without downloading
        """
    )
    parser.add_argument('--pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'],
                        help='Stock pool to update (default: sp500)')
    parser.add_argument('--convert-only', action='store_true',
                        help='Skip download, only convert existing CSV files')
    parser.add_argument('--download-only', action='store_true',
                        help='Only download, skip conversion to Qlib format')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only check status, do not download')
    args = parser.parse_args()

    symbols = list(STOCK_POOLS[args.pool])
    # 确保 SPY 总是被包含（作为 benchmark）
    if "SPY" not in symbols:
        symbols.append("SPY")
    end_date = datetime.now().strftime("%Y-%m-%d")

    print("=" * 60)
    print("US Stock Data Incremental Updater")
    print("=" * 60)
    print(f"Stock pool: {args.pool} ({len(symbols)} stocks, including SPY as benchmark)")
    print(f"Target date: {end_date}")
    print(f"CSV directory: {CSV_DIR}")
    print(f"Qlib directory: {QLIB_DIR}")
    print("=" * 60 + "\n")

    # 检查更新状态
    print("Step 1: Checking update status...\n")
    status = check_update_status(symbols, CSV_DIR)

    print(f"  Up to date: {len(status['up_to_date'])} stocks")
    print(f"  Needs update: {len(status['needs_update'])} stocks")
    print(f"  Missing: {len(status['missing'])} stocks")

    if status['needs_update']:
        print("\n  Stocks needing update:")
        for symbol, latest in status['needs_update'][:10]:
            print(f"    {symbol}: last updated {latest.strftime('%Y-%m-%d')}")
        if len(status['needs_update']) > 10:
            print(f"    ... and {len(status['needs_update']) - 10} more")

    if status['missing']:
        print(f"\n  Missing stocks (will be skipped): {status['missing'][:10]}")
        if len(status['missing']) > 10:
            print(f"    ... and {len(status['missing']) - 10} more")

    if args.dry_run:
        print("\n[Dry run mode - no downloads performed]")
        return

    if args.convert_only:
        print("\n[Convert-only mode - skipping downloads]")
    else:
        # 下载增量数据
        all_success = list(status['up_to_date'])
        all_failed = []

        if status['needs_update']:
            print("\n" + "=" * 60)
            print("Step 2a: Downloading incremental data")
            print("=" * 60 + "\n")
            success, failed = update_stock_data(status['needs_update'], CSV_DIR, end_date)
            all_success.extend(success)
            all_failed.extend(failed)

        # 跳过缺失的股票（通常是已退市或被移除的），但 SPY 必须下载
        if status['missing']:
            # 检查 SPY 是否在缺失列表中
            if 'SPY' in status['missing']:
                print("\n" + "=" * 60)
                print("Step 2b: Downloading SPY (benchmark)")
                print("=" * 60 + "\n")
                spy_success, spy_failed = download_missing_data(['SPY'], CSV_DIR, end_date)
                all_success.extend(spy_success)
                all_failed.extend(spy_failed)
                # 从缺失列表中移除 SPY
                remaining_missing = [s for s in status['missing'] if s != 'SPY']
            else:
                remaining_missing = status['missing']

            if remaining_missing:
                print(f"\n  Skipping {len(remaining_missing)} missing stocks (likely delisted or removed from index)")

        print(f"\nDownload summary: {len(all_success)} success, {len(all_failed)} failed")
        if all_failed:
            print(f"Failed: {all_failed[:20]}")
            if len(all_failed) > 20:
                print(f"  ... and {len(all_failed) - 20} more")

    if args.download_only:
        print("\n[Download-only mode - skipping conversion]")
        return

    # 转换格式
    print("\n" + "=" * 60)
    print("Step 3: Converting to Qlib format")
    print("=" * 60)
    convert_to_qlib_bin(CSV_DIR, QLIB_DIR)

    # 创建股票池文件
    print("\n" + "=" * 60)
    print("Step 4: Creating instruments files")
    print("=" * 60)

    # 获取所有已下载的股票
    existing_csvs = list(CSV_DIR.glob("*.csv"))
    all_symbols = [f.stem for f in existing_csvs]
    create_instruments_file(all_symbols, QLIB_DIR, args.pool)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Qlib data path: {QLIB_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
