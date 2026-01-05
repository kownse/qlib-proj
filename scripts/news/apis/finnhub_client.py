"""
Finnhub API 客户端

提供股票新闻数据下载功能，包含速率限制和错误处理
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

try:
    import finnhub
except ImportError:
    finnhub = None

logger = logging.getLogger(__name__)


class RateLimiter:
    """速率限制器，确保不超过 API 调用限制"""

    def __init__(self, calls_per_minute: int = 60):
        """
        初始化速率限制器

        Args:
            calls_per_minute: 每分钟最大调用次数 (Finnhub 免费版为 60)
        """
        self.calls_per_minute = calls_per_minute
        self.call_times: List[datetime] = []

    def wait_if_needed(self):
        """如果达到速率限制，等待直到可以继续"""
        now = datetime.now()
        # 清理一分钟之前的调用记录
        self.call_times = [t for t in self.call_times if (now - t).seconds < 60]

        if len(self.call_times) >= self.calls_per_minute:
            # 计算需要等待的时间
            oldest_call = self.call_times[0]
            sleep_time = 60 - (now - oldest_call).seconds + 1
            logger.info(f"达到速率限制，等待 {sleep_time} 秒...")
            time.sleep(sleep_time)

        self.call_times.append(datetime.now())


class FinnhubNewsClient:
    """
    Finnhub 新闻 API 客户端

    使用方法:
        client = FinnhubNewsClient(api_key="your_api_key")
        news_df = client.download_company_news("AAPL", "2024-01-01", "2024-12-31")
    """

    def __init__(self, api_key: Optional[str] = None, calls_per_minute: int = 60):
        """
        初始化 Finnhub 客户端

        Args:
            api_key: Finnhub API 密钥，如果为 None 则从环境变量 FINNHUB_API_KEY 获取
            calls_per_minute: 每分钟最大调用次数
        """
        if finnhub is None:
            raise ImportError(
                "finnhub-python 未安装。请运行: pip install finnhub-python"
            )

        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "未提供 API 密钥。请设置环境变量 FINNHUB_API_KEY 或传入 api_key 参数。"
                "\n获取免费 API 密钥: https://finnhub.io/register"
            )

        self.client = finnhub.Client(api_key=self.api_key)
        self.rate_limiter = RateLimiter(calls_per_minute)

    def _load_existing_data(self, output_path: str, symbol: str) -> tuple:
        """
        从文件加载已有数据

        Returns:
            (existing_df, downloaded_date_ranges): DataFrame 和已下载的日期范围集合
        """
        from pathlib import Path

        output_file = Path(output_path)
        if not output_file.exists():
            return pd.DataFrame(), set()

        try:
            existing_df = pd.read_csv(output_file, parse_dates=["datetime"])
            if existing_df.empty or "datetime" not in existing_df.columns:
                return pd.DataFrame(), set()

            # 提取已有新闻的日期 (只保留日期部分)
            downloaded_dates = set(existing_df["datetime"].dt.date)
            logger.info(
                f"{symbol}: 发现已有 {len(existing_df)} 条新闻，"
                f"日期范围 {min(downloaded_dates)} 至 {max(downloaded_dates)}"
            )
            return existing_df, downloaded_dates
        except Exception as e:
            logger.warning(f"{symbol}: 读取已有文件失败: {e}")
            return pd.DataFrame(), set()

    def _is_date_range_downloaded(
        self, start: datetime, end: datetime, downloaded_dates: set
    ) -> bool:
        """
        检查指定日期范围是否已经完全下载

        Args:
            start: 开始日期
            end: 结束日期
            downloaded_dates: 已下载的日期集合

        Returns:
            如果该区间所有日期都已下载返回 True
        """
        check_date = start
        while check_date <= end:
            if check_date.date() not in downloaded_dates:
                return False
            check_date += timedelta(days=1)
        return True

    def _save_news_to_disk(
        self, news: List[dict], symbol: str, output_path: str
    ) -> set:
        """
        将新闻数据保存到硬盘，支持追加模式

        Args:
            news: 新闻数据列表
            symbol: 股票代码
            output_path: 输出文件路径

        Returns:
            新增的日期集合
        """
        from pathlib import Path

        if not news:
            return set()

        # 转换为 DataFrame
        chunk_df = pd.DataFrame(news)
        chunk_df = self._process_news_df(chunk_df, symbol)

        if chunk_df.empty:
            return set()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 获取新增的日期
        new_dates = set(chunk_df["datetime"].dt.date)

        # 如果文件存在，读取并合并
        if output_file.exists():
            try:
                existing_df = pd.read_csv(output_file, parse_dates=["datetime"])
                combined_df = pd.concat([existing_df, chunk_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=["datetime", "headline"])
                combined_df = combined_df.sort_values("datetime").reset_index(drop=True)
                combined_df.to_csv(output_file, index=False)
            except Exception as e:
                logger.warning(f"合并数据失败，覆盖保存: {e}")
                chunk_df.to_csv(output_file, index=False)
        else:
            chunk_df.to_csv(output_file, index=False)

        return new_dates

    def download_company_news(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        chunk_days: int = 30,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        下载公司新闻数据，支持增量下载和断点续传

        每次下载到新闻后立即保存到硬盘，下载前会检查已有文件中的日期，
        跳过已下载的日期范围。

        Args:
            symbol: 股票代码 (如 "AAPL")
            start_date: 开始日期 (格式 "YYYY-MM-DD")
            end_date: 结束日期 (格式 "YYYY-MM-DD")
            chunk_days: 每次请求的天数 (避免单次请求数据过大)
            output_path: 输出文件路径，如果提供则增量保存到该文件

        Returns:
            包含新闻数据的 DataFrame，列包括:
            - datetime: 发布时间
            - headline: 标题
            - summary: 摘要
            - source: 来源
            - url: 链接
            - symbol: 股票代码
        """
        from pathlib import Path

        # 将日期范围分成小块，避免单次请求过大
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # 加载已有数据，获取已下载的日期
        downloaded_dates = set()
        if output_path:
            _, downloaded_dates = self._load_existing_data(output_path, symbol)

        all_news = []
        new_news_count = 0

        current_start = start
        while current_start < end:
            current_end = min(current_start + timedelta(days=chunk_days), end)

            # 检查该日期范围是否已经完全下载
            if downloaded_dates and self._is_date_range_downloaded(
                current_start, current_end, downloaded_dates
            ):
                logger.info(
                    f"{symbol}: 跳过已下载区间 "
                    f"({current_start.strftime('%Y-%m-%d')} 至 {current_end.strftime('%Y-%m-%d')})"
                )
                current_start = current_end + timedelta(days=1)
                continue

            # 速率限制
            self.rate_limiter.wait_if_needed()

            try:
                news = self.client.company_news(
                    symbol,
                    _from=current_start.strftime("%Y-%m-%d"),
                    to=current_end.strftime("%Y-%m-%d"),
                )

                if news:
                    all_news.extend(news)
                    new_news_count += len(news)
                    logger.info(
                        f"{symbol}: 获取 {len(news)} 条新闻 "
                        f"({current_start.strftime('%Y-%m-%d')} 至 {current_end.strftime('%Y-%m-%d')})"
                    )

                    # 立即保存到硬盘
                    if output_path:
                        new_dates = self._save_news_to_disk(news, symbol, output_path)
                        downloaded_dates.update(new_dates)
                        logger.debug(f"{symbol}: 已保存 {len(news)} 条新闻到 {output_path}")
                else:
                    # 即使没有新闻，也记录该日期范围已查询
                    # 将该区间的所有日期标记为已下载
                    check_date = current_start
                    while check_date <= current_end:
                        downloaded_dates.add(check_date.date())
                        check_date += timedelta(days=1)

            except Exception as e:
                logger.warning(
                    f"{symbol}: 获取新闻失败 ({current_start.strftime('%Y-%m-%d')} "
                    f"至 {current_end.strftime('%Y-%m-%d')}): {e}"
                )

            current_start = current_end + timedelta(days=1)

        # 返回最终数据
        if output_path and Path(output_path).exists():
            df = pd.read_csv(output_path, parse_dates=["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
            df = df.drop_duplicates(subset=["datetime", "headline"])
            # 保存去重后的数据
            df.to_csv(output_path, index=False)
            logger.info(f"{symbol}: 总共 {len(df)} 条唯一新闻 (本次新增 {new_news_count} 条)")
            return df

        # 如果没有 output_path，使用内存中的数据
        if not all_news:
            logger.warning(f"{symbol}: 未找到任何新闻")
            return pd.DataFrame()

        # 转换为 DataFrame
        df = pd.DataFrame(all_news)
        df = self._process_news_df(df, symbol)

        logger.info(f"{symbol}: 总共获取 {len(df)} 条唯一新闻")
        return df

    def _process_news_df(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """处理新闻 DataFrame，统一格式"""
        if df.empty:
            return df

        # 处理时间戳
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
        elif "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            df.drop("timestamp", axis=1, inplace=True)

        # 添加股票代码列
        df["symbol"] = symbol

        # 选择需要的列
        columns_to_keep = ["datetime", "headline", "summary", "source", "url", "symbol"]
        available_columns = [col for col in columns_to_keep if col in df.columns]
        df = df[available_columns]

        # 按时间排序
        df = df.sort_values("datetime").reset_index(drop=True)

        # 去重
        df = df.drop_duplicates(subset=["datetime", "headline"])

        return df

    def download_multiple_symbols(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        chunk_days: int = 30,
    ) -> pd.DataFrame:
        """
        下载多只股票的新闻数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            chunk_days: 每次请求的天数

        Returns:
            合并后的新闻 DataFrame
        """
        all_dfs = []

        for i, symbol in enumerate(symbols):
            logger.info(f"下载进度: {i + 1}/{len(symbols)} - {symbol}")

            df = self.download_company_news(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                chunk_days=chunk_days,
            )

            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"总共下载 {len(combined_df)} 条新闻，涵盖 {len(symbols)} 只股票")

        return combined_df

    def test_connection(self) -> bool:
        """测试 API 连接是否正常"""
        try:
            # 使用一个简单的 API 调用测试连接
            self.client.company_news("AAPL", _from="2024-01-01", to="2024-01-02")
            logger.info("Finnhub API 连接测试成功")
            return True
        except Exception as e:
            logger.error(f"Finnhub API 连接测试失败: {e}")
            return False


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    client = FinnhubNewsClient()
    if client.test_connection():
        df = client.download_company_news("AAPL", "2024-01-01", "2024-01-31")
        print(df.head())
        print(f"总条数: {len(df)}")
