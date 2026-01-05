"""
新闻数据处理主脚本

将原始新闻数据处理为模型可用的特征
使用方法:
    python process_news.py --input my_data/news_csv --output my_data/news_processed
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# 项目路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from news.processors.sentiment_analyzer import (
    FinBERTSentimentAnalyzer,
    VADERSentimentAnalyzer,
    get_sentiment_analyzer,
)
from news.processors.feature_extractor import NewsStatisticalFeatures, NewsAggregator

# 数据路径
NEWS_CSV_DIR = PROJECT_ROOT / "my_data" / "news_csv"
NEWS_PROCESSED_DIR = PROJECT_ROOT / "my_data" / "news_processed"

logger = logging.getLogger(__name__)


class NewsFeatureProcessor:
    """
    新闻特征处理器

    整合情感分析和统计特征提取，生成模型可用的特征

    使用方法:
        processor = NewsFeatureProcessor(sentiment_method="finbert")
        features_df = processor.process_all(news_df, trading_days)
        processor.save_features(features_df, output_dir)
    """

    # 新闻特征列名 (添加 news_ 前缀避免与市场特征冲突)
    SENTIMENT_COLUMNS = [
        "news_positive",
        "news_negative",
        "news_neutral",
        "news_sentiment_score",
    ]

    STAT_COLUMNS = [
        "news_count",
        "news_count_log",
        "news_kw_bullish",
        "news_kw_bearish",
        "news_kw_volatility",
        "news_bull_bear_ratio",
        "news_avg_headline_len",
        "news_total_text_len",
    ]

    def __init__(
        self,
        sentiment_method: str = "finbert",
        market_close_hour: int = 16,
        device: str = None,
    ):
        """
        初始化处理器

        Args:
            sentiment_method: 情感分析方法 ("finbert" 或 "vader")
            market_close_hour: 市场收盘时间
            device: 计算设备 (用于 FinBERT)
        """
        self.sentiment_method = sentiment_method
        self.market_close_hour = market_close_hour

        # 初始化组件
        logger.info(f"初始化情感分析器: {sentiment_method}")
        self.sentiment_analyzer = get_sentiment_analyzer(sentiment_method, device=device)
        self.stat_extractor = NewsStatisticalFeatures()
        self.aggregator = NewsAggregator(market_close_hour=market_close_hour)

    def process_all(
        self,
        news_df: pd.DataFrame,
        trading_days: pd.DatetimeIndex,
        symbols: List[str] = None,
    ) -> pd.DataFrame:
        """
        处理所有新闻数据，生成特征

        Args:
            news_df: 原始新闻 DataFrame
            trading_days: 交易日列表
            symbols: 股票代码列表，None 则使用新闻中的所有股票

        Returns:
            特征 DataFrame，索引为 (datetime, instrument)
        """
        if news_df.empty:
            logger.warning("新闻数据为空")
            return pd.DataFrame()

        # 确保 datetime 列格式正确
        if not pd.api.types.is_datetime64_any_dtype(news_df["datetime"]):
            news_df = news_df.copy()
            news_df["datetime"] = pd.to_datetime(news_df["datetime"])

        # 获取股票列表
        if symbols is None:
            symbols = news_df["symbol"].unique().tolist()
        logger.info(f"处理 {len(symbols)} 只股票的新闻")

        # 1. 时间对齐
        logger.info("对齐新闻到交易日...")
        aligned_df = self.aggregator.align_news_to_trading_day(news_df, trading_days)
        logger.info(f"对齐后新闻数量: {len(aligned_df)}")

        # 2. 情感分析
        logger.info("执行情感分析...")
        sentiment_df = self._analyze_sentiment(aligned_df)

        # 3. 聚合每日情感
        logger.info("聚合每日情感...")
        daily_sentiment = self._aggregate_daily_sentiment(sentiment_df, symbols, trading_days)

        # 4. 提取统计特征
        logger.info("提取统计特征...")
        daily_stats = self._extract_statistical_features(aligned_df, symbols, trading_days)

        # 5. 合并特征
        logger.info("合并特征...")
        features_df = self._merge_features(daily_sentiment, daily_stats)

        # 6. 填充缺失值
        logger.info("填充缺失值...")
        features_df = self._fill_missing_values(features_df)

        logger.info(f"生成特征完成: {features_df.shape}")
        return features_df

    def _analyze_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """分析每条新闻的情感"""
        df = news_df.copy()

        # 合并标题和摘要作为输入文本
        texts = (
            df["headline"].fillna("") + " " + df["summary"].fillna("")
        ).tolist()

        # 批量分析
        results = self.sentiment_analyzer.analyze_batch(texts)

        # 添加情感分数列
        df["positive"] = [r["positive"] for r in results]
        df["negative"] = [r["negative"] for r in results]
        df["neutral"] = [r["neutral"] for r in results]
        df["sentiment_score"] = [r["sentiment_score"] for r in results]

        return df

    def _aggregate_daily_sentiment(
        self,
        sentiment_df: pd.DataFrame,
        symbols: List[str],
        trading_days: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """聚合每日情感分数"""
        # 按交易日和股票聚合
        sentiment_cols = ["positive", "negative", "neutral", "sentiment_score"]

        aggregated = (
            sentiment_df.groupby(["trading_date", "symbol"])[sentiment_cols]
            .mean()
            .reset_index()
        )

        # 创建完整的日期-股票网格
        full_index = pd.MultiIndex.from_product(
            [trading_days.date, symbols],
            names=["trading_date", "symbol"],
        )
        full_df = pd.DataFrame(index=full_index).reset_index()

        # 合并
        merged = full_df.merge(
            aggregated,
            on=["trading_date", "symbol"],
            how="left",
        )

        # 重命名列，添加 news_ 前缀
        merged = merged.rename(columns={
            "positive": "news_positive",
            "negative": "news_negative",
            "neutral": "news_neutral",
            "sentiment_score": "news_sentiment_score",
        })

        return merged

    def _extract_statistical_features(
        self,
        news_df: pd.DataFrame,
        symbols: List[str],
        trading_days: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """提取统计特征"""
        # 将新闻的 trading_date 转为字符串日期列表
        dates = [d.strftime("%Y-%m-%d") for d in trading_days]

        # 为统计特征提取准备数据
        # 需要将 trading_date 映射回 datetime 用于提取
        news_with_date = news_df.copy()
        news_with_date["datetime"] = pd.to_datetime(news_with_date["trading_date"])

        features_df = self.stat_extractor.extract_features_for_df(
            news_with_date, dates, symbols
        )

        return features_df.reset_index()

    def _merge_features(
        self,
        sentiment_df: pd.DataFrame,
        stats_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """合并情感和统计特征"""
        # 统一列名
        sentiment_df["datetime"] = pd.to_datetime(sentiment_df["trading_date"])
        sentiment_df["instrument"] = sentiment_df["symbol"]

        stats_df["datetime"] = pd.to_datetime(stats_df["datetime"])

        # 合并
        merged = sentiment_df.merge(
            stats_df,
            on=["datetime", "instrument"],
            how="outer",
        )

        # 选择需要的列
        feature_cols = self.SENTIMENT_COLUMNS + self.STAT_COLUMNS
        keep_cols = ["datetime", "instrument"] + [c for c in feature_cols if c in merged.columns]
        merged = merged[keep_cols]

        # 设置索引
        merged = merged.set_index(["datetime", "instrument"]).sort_index()

        return merged

    def _fill_missing_values(
        self,
        df: pd.DataFrame,
        max_ffill: int = 5,
    ) -> pd.DataFrame:
        """
        填充缺失值

        策略:
        1. 先进行 forward fill (最多 max_ffill 天)
        2. 剩余缺失值填充为中性值
        """
        df = df.copy()

        # Forward fill (按股票分组)
        df = df.groupby(level="instrument").ffill(limit=max_ffill)

        # 填充剩余缺失值
        fill_values = {
            "news_positive": 0.33,
            "news_negative": 0.33,
            "news_neutral": 0.34,
            "news_sentiment_score": 0.0,
            "news_count": 0,
            "news_count_log": 0.0,
            "news_kw_bullish": 0,
            "news_kw_bearish": 0,
            "news_kw_volatility": 0,
            "news_bull_bear_ratio": 0.0,
            "news_avg_headline_len": 0.0,
            "news_total_text_len": 0,
        }

        for col, value in fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)

        return df

    def save_features(
        self,
        features_df: pd.DataFrame,
        output_dir: Path,
        filename: str = "news_features.parquet",
    ):
        """
        保存特征到文件

        Args:
            features_df: 特征 DataFrame
            output_dir: 输出目录
            filename: 文件名
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / filename
        features_df.to_parquet(output_path)
        logger.info(f"特征已保存到 {output_path}")

        # 同时保存 CSV 版本便于查看
        csv_path = output_dir / filename.replace(".parquet", ".csv")
        features_df.to_csv(csv_path)
        logger.info(f"CSV 版本已保存到 {csv_path}")


def load_news_data(news_dir: Path) -> pd.DataFrame:
    """加载新闻数据"""
    all_news_path = news_dir / "all_news.csv"

    if all_news_path.exists():
        df = pd.read_csv(all_news_path, parse_dates=["datetime"])
        logger.info(f"从 {all_news_path} 加载 {len(df)} 条新闻")
        return df

    # 如果没有合并文件，尝试合并单个文件
    all_dfs = []
    for csv_file in news_dir.glob("*_news.csv"):
        if csv_file.name != "all_news.csv":
            df = pd.read_csv(csv_file, parse_dates=["datetime"])
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"从单个文件合并 {len(combined)} 条新闻")
        return combined

    logger.warning(f"未找到新闻数据: {news_dir}")
    return pd.DataFrame()


def get_trading_days(
    start_date: str,
    end_date: str,
    provider_uri: str = None,
) -> pd.DatetimeIndex:
    """
    获取交易日列表

    尝试从 Qlib 获取，如果失败则使用工作日近似
    """
    try:
        import qlib
        from qlib.constant import REG_US

        if provider_uri is None:
            provider_uri = str(PROJECT_ROOT / "my_data" / "qlib_us")

        qlib.init(provider_uri=provider_uri, region=REG_US)

        from qlib.data import D
        trading_days = D.calendar(start_time=start_date, end_time=end_date)
        logger.info(f"从 Qlib 获取 {len(trading_days)} 个交易日")
        return pd.DatetimeIndex(trading_days)

    except Exception as e:
        logger.warning(f"无法从 Qlib 获取交易日: {e}")
        logger.info("使用工作日作为近似")

        # 使用工作日作为近似
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        return dates


def process_all_news(
    input_dir: Path = NEWS_CSV_DIR,
    output_dir: Path = NEWS_PROCESSED_DIR,
    sentiment_method: str = "vader",
    start_date: str = None,
    end_date: str = None,
    symbols: List[str] = None,
) -> pd.DataFrame:
    """
    处理所有新闻数据的便捷函数

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        sentiment_method: 情感分析方法
        start_date: 开始日期
        end_date: 结束日期
        symbols: 股票列表

    Returns:
        处理后的特征 DataFrame
    """
    # 加载新闻数据
    news_df = load_news_data(input_dir)
    if news_df.empty:
        return pd.DataFrame()

    # 确定日期范围
    if start_date is None:
        start_date = news_df["datetime"].min().strftime("%Y-%m-%d")
    if end_date is None:
        end_date = news_df["datetime"].max().strftime("%Y-%m-%d")

    # 获取交易日
    trading_days = get_trading_days(start_date, end_date)

    # 处理新闻
    processor = NewsFeatureProcessor(sentiment_method=sentiment_method)
    features_df = processor.process_all(news_df, trading_days, symbols)

    # 保存
    processor.save_features(features_df, output_dir)

    return features_df


def main():
    parser = argparse.ArgumentParser(description="处理新闻数据生成特征")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="输入目录 (新闻 CSV 文件)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录",
    )
    parser.add_argument(
        "--sentiment",
        type=str,
        default="vader",
        choices=["finbert", "vader"],
        help="情感分析方法 (默认 vader，更快；finbert 更准确但需要 GPU)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="开始日期",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="结束日期",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="股票代码列表",
    )

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    input_dir = Path(args.input) if args.input else NEWS_CSV_DIR
    output_dir = Path(args.output) if args.output else NEWS_PROCESSED_DIR

    logger.info("开始处理新闻数据...")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"情感分析方法: {args.sentiment}")

    process_all_news(
        input_dir=input_dir,
        output_dir=output_dir,
        sentiment_method=args.sentiment,
        start_date=args.start,
        end_date=args.end,
        symbols=args.symbols,
    )

    logger.info("新闻数据处理完成!")


if __name__ == "__main__":
    main()
