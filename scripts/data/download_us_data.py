"""
从 Yahoo Finance 下载美股数据，转换为 Qlib 格式

使用方法:
    python download_us_data.py                    # 下载 SP100（默认）
    python download_us_data.py --pool sp500       # 下载 SP500
    python download_us_data.py --pool tech        # 下载科技股
    python download_us_data.py --pool sp100 --convert-only  # 只转换，不下载
"""
import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# 确保项目路径正确
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "qlib"))  # 添加 qlib 源码路径
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))  # 添加 scripts 路径

import pandas as pd
import yfinance as yf

# 从 stock_pools.py 导入股票池定义（单一数据源）
from data.stock_pools import STOCK_POOLS, SP100_SYMBOLS, SP500_SYMBOLS, TECH_SYMBOLS

# 时间范围
START_DATE = "2000-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# 输出路径（文件在 scripts/data/ 下，需要向上三级）
# PROJECT_ROOT 已在文件开头定义
CSV_DIR = PROJECT_ROOT / "my_data" / "csv_us"
QLIB_DIR = PROJECT_ROOT / "my_data" / "qlib_us"


# ========== 下载函数 ==========

def download_stock_data(symbols: list, start_date: str, end_date: str, output_dir: Path):
    """下载股票数据并保存为 CSV"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success = []
    failed = []
    
    for i, symbol in enumerate(symbols):
        csv_path = output_dir / f"{symbol}.csv"
        if csv_path.exists():
            print(f"[{i+1}/{len(symbols)}] Skipped {symbol} (already exists)")
            success.append(symbol)
            continue
        print(f"[{i+1}/{len(symbols)}] Downloading {symbol}...", end=" ")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
            
            if df.empty:
                print("No data")
                failed.append(symbol)
                continue
            
            # 标准化列名（Qlib 期望小写）
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
            
            # 处理 index（日期）
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)  # 移除时区
            df.index.name = "date"
            
            # 保存（文件名用大写，与 symbol 一致）
            df.to_csv(csv_path)
            print(f"OK ({len(df)} rows)")
            success.append(symbol)
            
        except Exception as e:
            print(f"Error: {e}")
            failed.append(symbol)
    
    print(f"\nSummary: {len(success)} success, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")
    
    return success, failed


def convert_to_qlib_bin(csv_dir: Path, qlib_dir: Path):
    """将 CSV 转换为 Qlib bin 格式"""
    import sys

    # 把 scripts 目录加到 path
    scripts_path = str(PROJECT_ROOT / "qlib" / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    from dump_bin import DumpDataAll

    # 清理旧数据
    if qlib_dir.exists():
        shutil.rmtree(qlib_dir)
    qlib_dir.mkdir(parents=True)

    print(f"\nConverting CSV to Qlib format...")
    print(f"  Source: {csv_dir}")
    print(f"  Target: {qlib_dir}")

    dumper = DumpDataAll(
        data_path=str(csv_dir),  # Changed from csv_path to data_path
        qlib_dir=str(qlib_dir),
        freq="day",
        date_field_name="date",
        file_suffix=".csv",  # Added file_suffix parameter
        include_fields="open,high,low,close,adj_close,volume",
    )
    dumper.dump()

    print("Conversion completed!")



def create_instruments_file(symbols: list, qlib_dir: Path, start_date: str, end_date: str, pool_name: str = "sp100"):
    """创建股票池文件"""
    instruments_dir = qlib_dir / "instruments"
    instruments_dir.mkdir(parents=True, exist_ok=True)

    # all.txt - 包含所有股票及其有效日期范围
    with open(instruments_dir / "all.txt", "w") as f:
        for symbol in symbols:
            f.write(f"{symbol}\t{start_date}\t{end_date}\n")

    # 创建对应股票池文件（如 sp100.txt, sp500.txt）
    with open(instruments_dir / f"{pool_name}.txt", "w") as f:
        for symbol in symbols:
            f.write(f"{symbol}\t{start_date}\t{end_date}\n")

    print(f"Created instruments files at {instruments_dir}")
    print(f"  - all.txt ({len(symbols)} stocks)")
    print(f"  - {pool_name}.txt ({len(symbols)} stocks)")


# ========== 主函数 ==========

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Download US stock data and convert to Qlib format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_us_data.py                     # Download SP100 (default)
    python download_us_data.py --pool sp500        # Download SP500 (~500 stocks)
    python download_us_data.py --pool tech         # Download tech stocks (~30)
    python download_us_data.py --convert-only      # Only convert existing CSVs
    python download_us_data.py --download-only     # Only download, skip conversion
        """
    )
    parser.add_argument('--pool', type=str, default='sp100',
                        choices=['test', 'tech', 'sp100', 'sp500'],
                        help='Stock pool to download (default: sp100)')
    parser.add_argument('--convert-only', action='store_true',
                        help='Skip download, only convert existing CSV files')
    parser.add_argument('--download-only', action='store_true',
                        help='Only download, skip conversion to Qlib format')
    parser.add_argument('--start-date', type=str, default=START_DATE,
                        help=f'Start date (default: {START_DATE})')
    parser.add_argument('--end-date', type=str, default=END_DATE,
                        help=f'End date (default: {END_DATE})')
    args = parser.parse_args()

    # 获取选定的股票池
    symbols = STOCK_POOLS[args.pool]

    print("=" * 60)
    print("US Stock Data Downloader for Qlib")
    print("=" * 60)
    print(f"Stock pool: {args.pool} ({len(symbols)} stocks)")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"CSV output: {CSV_DIR}")
    print(f"Qlib output: {QLIB_DIR}")
    if args.convert_only:
        print("Mode: Convert only (skip download)")
    elif args.download_only:
        print("Mode: Download only (skip conversion)")
    print("=" * 60 + "\n")

    success = symbols  # 默认假设所有股票都成功

    # 1. 下载数据
    if not args.convert_only:
        print("Step 1: Downloading from Yahoo Finance\n")
        success, failed = download_stock_data(symbols, args.start_date, args.end_date, CSV_DIR)

        if not success:
            print("No data downloaded. Exiting.")
            sys.exit(1)
    else:
        print("Step 1: Skipped (convert-only mode)\n")
        # 检查已有的 CSV 文件
        existing_csvs = list(CSV_DIR.glob("*.csv"))
        success = [f.stem for f in existing_csvs if f.stem in symbols]
        print(f"Found {len(success)} existing CSV files for {args.pool} pool")

    if args.download_only:
        print("\nDownload complete (conversion skipped).")
        return

    # 2. 转换格式
    print("\n" + "=" * 60)
    print("Step 2: Converting to Qlib format")
    print("=" * 60)
    convert_to_qlib_bin(CSV_DIR, QLIB_DIR)

    # 3. 创建股票池文件
    print("\n" + "=" * 60)
    print("Step 3: Creating instruments files")
    print("=" * 60)
    create_instruments_file(success, QLIB_DIR, args.start_date, args.end_date, args.pool)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Qlib data path: {QLIB_DIR}")
    print(f"Stock pool: {args.pool} ({len(success)} stocks)")
    print("=" * 60)


if __name__ == "__main__":
    main()