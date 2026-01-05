"""
从 Yahoo Finance 下载美股数据，转换为 Qlib 格式
"""
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# 确保项目路径正确
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "qlib"))  # 添加 qlib 源码路径

import pandas as pd
import yfinance as yf

# from qlib.scripts.dump_bin import DumpDataAll


# ========== 配置 ==========

# S&P 100 成分股（流动性好）
SP100_SYMBOLS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "C", "CAT",
    "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX",
    "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX", "GD",
    "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC",
    "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD",
    "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE",
    "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM",
    "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA",
    "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM",
]

# 美股核心科技股（约30只）
TECH_SYMBOLS = [
    # 超大型科技（Magnificent 7）
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet (Google)
    "AMZN",   # Amazon
    "META",   # Meta (Facebook)
    "NVDA",   # NVIDIA
    "TSLA",   # Tesla
    
    # 半导体
    "AMD",    # AMD
    "INTC",   # Intel
    "AVGO",   # Broadcom
    "QCOM",   # Qualcomm
    "MU",     # Micron
    "AMAT",   # Applied Materials
    
    # 软件/云服务
    "CRM",    # Salesforce
    "ORCL",   # Oracle
    "ADBE",   # Adobe
    "NOW",    # ServiceNow
    "SNOW",   # Snowflake
    "PLTR",   # Palantir
    
    # 互联网/消费科技
    "NFLX",   # Netflix
    "UBER",   # Uber
    "ABNB",   # Airbnb
    "SHOP",   # Shopify
    # "SQ",     # Block (Square)
    "PYPL",   # PayPal
    "SPOT",   # Spotify
    
    # 网络/通信设备
    "CSCO",   # Cisco
    "PANW",   # Palo Alto Networks
    "CRWD",   # CrowdStrike
]

# 时间范围
START_DATE = "2000-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# 输出路径
PROJECT_ROOT = Path(__file__).parent.parent
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



def create_instruments_file(symbols: list, qlib_dir: Path, start_date: str, end_date: str):
    """创建股票池文件"""
    instruments_dir = qlib_dir / "instruments"
    instruments_dir.mkdir(parents=True, exist_ok=True)
    
    # all.txt - 包含所有股票及其有效日期范围
    with open(instruments_dir / "all.txt", "w") as f:
        for symbol in symbols:
            f.write(f"{symbol}\t{start_date}\t{end_date}\n")
    
    # sp100.txt - 作为自定义股票池
    with open(instruments_dir / "sp100.txt", "w") as f:
        for symbol in symbols:
            f.write(f"{symbol}\t{start_date}\t{end_date}\n")
    
    print(f"Created instruments files at {instruments_dir}")


# ========== 主函数 ==========

def main():
    print("=" * 60)
    print("US Stock Data Downloader for Qlib")
    print("=" * 60)
    print(f"Symbols: {len(TECH_SYMBOLS)}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"CSV output: {CSV_DIR}")
    print(f"Qlib output: {QLIB_DIR}")
    print("=" * 60 + "\n")
    
    # 1. 下载数据
    print("Step 1: Downloading from Yahoo Finance\n")
    success, failed = download_stock_data(TECH_SYMBOLS, START_DATE, END_DATE, CSV_DIR)
    
    if not success:
        print("No data downloaded. Exiting.")
        sys.exit(1)
    
    # 2. 转换格式
    print("\n" + "=" * 60)
    print("Step 2: Converting to Qlib format")
    print("=" * 60)
    convert_to_qlib_bin(CSV_DIR, QLIB_DIR)
    
    # 3. 创建股票池文件
    print("\n" + "=" * 60)
    print("Step 3: Creating instruments files")
    print("=" * 60)
    create_instruments_file(success, QLIB_DIR, START_DATE, END_DATE)
    
    print("\n" + "=" * 60)
    print("Done!")
    print(f"Qlib data path: {QLIB_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()