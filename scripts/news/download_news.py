"""
新闻数据下载主脚本

下载美股新闻数据并保存到本地
使用方法:
    python download_news.py --symbols AAPL MSFT NVDA --start 2020-01-01 --end 2024-12-31
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

# 项目路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 加载 .env 文件中的环境变量
# 优先从项目根目录加载，其次从当前目录加载
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()  # 尝试从当前目录或父目录加载

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from news.apis.finnhub_client import FinnhubNewsClient

# 数据存储路径
NEWS_CSV_DIR = PROJECT_ROOT / "my_data" / "news_csv"

# 默认股票列表 (与现有项目一致的科技股)
DEFAULT_SYMBOLS = [
    # Magnificent 7
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # 半导体
    "AMD", "INTC", "AVGO", "QCOM", "MU", "AMAT",
    # 软件/云
    "CRM", "ORCL", "ADBE", "NOW", "SNOW", "PLTR",
    # 互联网/消费科技
    "NFLX", "UBER", "ABNB", "SHOP", "PYPL", "SPOT",
]

logger = logging.getLogger(__name__)


def download_news_for_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Path = NEWS_CSV_DIR,
    api_key: str = None,
) -> pd.DataFrame:
    """
    下载单只股票的新闻数据

    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录
        api_key: Finnhub API 密钥

    Returns:
        新闻 DataFrame
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    client = FinnhubNewsClient(api_key=api_key)
    df = client.download_company_news(symbol, start_date, end_date)

    if not df.empty:
        output_path = output_dir / f"{symbol}_news.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"已保存 {symbol} 新闻到 {output_path}")

    return df


def download_all_news(
    symbols: list = None,
    start_date: str = "2015-01-01",
    end_date: str = None,
    output_dir: Path = NEWS_CSV_DIR,
    api_key: str = None,
) -> pd.DataFrame:
    """
    下载多只股票的新闻数据

    Args:
        symbols: 股票代码列表，默认使用 DEFAULT_SYMBOLS
        start_date: 开始日期
        end_date: 结束日期，默认为今天
        output_dir: 输出目录
        api_key: Finnhub API 密钥

    Returns:
        合并后的新闻 DataFrame
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    output_dir.mkdir(parents=True, exist_ok=True)

    client = FinnhubNewsClient(api_key=api_key)

    all_dfs = []
    for i, symbol in enumerate(symbols):
        logger.info(f"下载进度: {i + 1}/{len(symbols)} - {symbol}")

        df = client.download_company_news(symbol, start_date, end_date)

        if not df.empty:
            # 保存单个股票的新闻
            output_path = output_dir / f"{symbol}_news.csv"
            df.to_csv(output_path, index=False)
            all_dfs.append(df)

    if all_dfs:
        # 保存合并后的新闻
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_path = output_dir / "all_news.csv"
        combined_df.to_csv(combined_path, index=False)
        logger.info(f"已保存合并新闻到 {combined_path}，共 {len(combined_df)} 条")
        return combined_df

    return pd.DataFrame()


def load_existing_news(symbol: str = None, news_dir: Path = NEWS_CSV_DIR) -> pd.DataFrame:
    """
    加载已下载的新闻数据

    Args:
        symbol: 股票代码，如果为 None 则加载所有新闻
        news_dir: 新闻数据目录

    Returns:
        新闻 DataFrame
    """
    if symbol:
        file_path = news_dir / f"{symbol}_news.csv"
        if file_path.exists():
            return pd.read_csv(file_path, parse_dates=["datetime"])
        return pd.DataFrame()

    # 加载所有新闻
    all_news_path = news_dir / "all_news.csv"
    if all_news_path.exists():
        return pd.read_csv(all_news_path, parse_dates=["datetime"])

    # 如果没有合并文件，尝试合并单个文件
    all_dfs = []
    for csv_file in news_dir.glob("*_news.csv"):
        if csv_file.name != "all_news.csv":
            df = pd.read_csv(csv_file, parse_dates=["datetime"])
            all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)

    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="下载股票新闻数据")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="股票代码列表，默认下载科技股",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="开始日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="结束日期 (YYYY-MM-DD)，默认为今天",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Finnhub API 密钥 (也可通过 FINNHUB_API_KEY 环境变量设置)",
    )

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    output_dir = Path(args.output_dir) if args.output_dir else NEWS_CSV_DIR

    logger.info("开始下载新闻数据...")
    logger.info(f"股票列表: {args.symbols or DEFAULT_SYMBOLS}")
    logger.info(f"日期范围: {args.start} 至 {args.end or '今天'}")
    logger.info(f"输出目录: {output_dir}")

    download_all_news(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        output_dir=output_dir,
        api_key=args.api_key,
    )

    logger.info("新闻数据下载完成!")


if __name__ == "__main__":
    main()
