"""
新闻统计特征提取器

从新闻数据中提取统计特征，如新闻数量、关键词频率等
"""

import logging
import re
from typing import Dict, List, Optional
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NewsStatisticalFeatures:
    """
    新闻统计特征提取器

    提取的特征:
    - news_count: 当日新闻数量
    - news_count_log: log(1 + count)
    - news_kw_bullish: 看涨关键词数量
    - news_kw_bearish: 看跌关键词数量
    - news_kw_volatility: 波动相关关键词数量
    - news_bull_bear_ratio: 看涨/看跌比率
    - news_avg_headline_len: 平均标题长度
    - news_total_text_len: 总文本长度

    使用方法:
        extractor = NewsStatisticalFeatures()
        features = extractor.extract_daily_features(news_df, "2024-01-15", "AAPL")
    """

    # 关键词字典
    KEYWORDS = {
        "bullish": [
            "beat", "beats", "surge", "surges", "rally", "rallies",
            "profit", "profits", "growth", "upgrade", "upgrades",
            "buy", "bullish", "gain", "gains", "rise", "rises",
            "soar", "soars", "jump", "jumps", "positive", "optimistic",
            "outperform", "outperforms", "strong", "record", "high",
        ],
        "bearish": [
            "miss", "misses", "drop", "drops", "fall", "falls",
            "loss", "losses", "decline", "declines", "downgrade", "downgrades",
            "sell", "bearish", "plunge", "plunges", "tumble", "tumbles",
            "sink", "sinks", "negative", "pessimistic", "underperform",
            "weak", "low", "crash", "crashes", "slump", "slumps",
        ],
        "volatility": [
            "volatile", "volatility", "uncertainty", "uncertain",
            "risk", "risky", "warning", "warn", "warns", "concern",
            "concerns", "fear", "fears", "anxiety", "turbulent",
            "fluctuate", "fluctuates", "swing", "swings", "unstable",
        ],
    }

    def __init__(self, custom_keywords: Dict[str, List[str]] = None):
        """
        初始化特征提取器

        Args:
            custom_keywords: 自定义关键词字典，格式同 KEYWORDS
        """
        self.keywords = self.KEYWORDS.copy()
        if custom_keywords:
            self.keywords.update(custom_keywords)

        # 预编译正则表达式
        self._compile_patterns()

    def _compile_patterns(self):
        """编译关键词匹配模式"""
        self.patterns = {}
        for category, words in self.keywords.items():
            # 创建匹配任意关键词的正则表达式
            pattern = r'\b(' + '|'.join(re.escape(w) for w in words) + r')\b'
            self.patterns[category] = re.compile(pattern, re.IGNORECASE)

    def _count_keywords(self, text: str, category: str) -> int:
        """计算文本中特定类别关键词的数量"""
        if not text:
            return 0
        matches = self.patterns[category].findall(text)
        return len(matches)

    def extract_daily_features(
        self,
        news_df: pd.DataFrame,
        date: str,
        symbol: str,
    ) -> Dict[str, float]:
        """
        提取单日单股票的新闻统计特征

        Args:
            news_df: 新闻 DataFrame，需包含 datetime, headline, summary, symbol 列
            date: 目标日期 (YYYY-MM-DD)
            symbol: 股票代码

        Returns:
            特征字典
        """
        # 过滤指定日期和股票的新闻
        target_date = pd.to_datetime(date).date()

        if news_df.empty:
            return self._empty_features()

        # 确保 datetime 列是 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(news_df["datetime"]):
            news_df = news_df.copy()
            news_df["datetime"] = pd.to_datetime(news_df["datetime"])

        mask = (news_df["datetime"].dt.date == target_date) & (
            news_df["symbol"] == symbol
        )
        day_news = news_df[mask]

        if len(day_news) == 0:
            return self._empty_features()

        # 合并所有文本
        headlines = day_news["headline"].fillna("").tolist()
        summaries = day_news["summary"].fillna("").tolist()
        all_text = " ".join(headlines + summaries).lower()

        # 计算关键词数量
        kw_bullish = self._count_keywords(all_text, "bullish")
        kw_bearish = self._count_keywords(all_text, "bearish")
        kw_volatility = self._count_keywords(all_text, "volatility")

        # 计算特征
        news_count = len(day_news)

        return {
            "news_count": news_count,
            "news_count_log": np.log1p(news_count),
            "news_kw_bullish": kw_bullish,
            "news_kw_bearish": kw_bearish,
            "news_kw_volatility": kw_volatility,
            "news_bull_bear_ratio": self._safe_ratio(kw_bullish, kw_bearish),
            "news_avg_headline_len": np.mean([len(h) for h in headlines]) if headlines else 0,
            "news_total_text_len": len(all_text),
        }

    def extract_features_for_df(
        self,
        news_df: pd.DataFrame,
        dates: List[str],
        symbols: List[str],
    ) -> pd.DataFrame:
        """
        批量提取多日多股票的新闻统计特征

        Args:
            news_df: 新闻 DataFrame
            dates: 日期列表
            symbols: 股票代码列表

        Returns:
            特征 DataFrame，索引为 (datetime, instrument)
        """
        results = []

        total = len(dates) * len(symbols)
        processed = 0

        for date in dates:
            for symbol in symbols:
                features = self.extract_daily_features(news_df, date, symbol)
                features["datetime"] = pd.to_datetime(date)
                features["instrument"] = symbol
                results.append(features)

                processed += 1
                if processed % 1000 == 0:
                    logger.info(f"统计特征提取进度: {processed}/{total}")

        df = pd.DataFrame(results)
        df = df.set_index(["datetime", "instrument"])

        return df

    def _empty_features(self) -> Dict[str, float]:
        """返回空特征 (无新闻时的默认值)"""
        return {
            "news_count": 0,
            "news_count_log": 0.0,
            "news_kw_bullish": 0,
            "news_kw_bearish": 0,
            "news_kw_volatility": 0,
            "news_bull_bear_ratio": 0.0,
            "news_avg_headline_len": 0.0,
            "news_total_text_len": 0,
        }

    @staticmethod
    def _safe_ratio(a: float, b: float) -> float:
        """安全除法，避免除零"""
        if b == 0:
            return 1.0 if a > 0 else 0.0
        return a / (b + 1e-8)


class NewsAggregator:
    """
    新闻聚合器

    将多条新闻聚合为每日特征，处理时间对齐
    """

    def __init__(self, market_close_hour: int = 16):
        """
        初始化聚合器

        Args:
            market_close_hour: 市场收盘时间 (小时)，用于新闻时间对齐
        """
        self.market_close_hour = market_close_hour

    def align_news_to_trading_day(
        self,
        news_df: pd.DataFrame,
        trading_days: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        将新闻时间对齐到交易日

        规则:
        - 收盘前的新闻 -> 当日
        - 收盘后的新闻 -> 下一个交易日
        - 周末/假日的新闻 -> 下一个交易日

        Args:
            news_df: 新闻 DataFrame，需包含 datetime 列
            trading_days: 交易日索引

        Returns:
            添加了 trading_date 列的 DataFrame
        """
        df = news_df.copy()

        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"])

        def get_trading_date(news_time):
            news_date = news_time.date()
            news_hour = news_time.hour

            # 如果在收盘后，移到下一天
            if news_hour >= self.market_close_hour:
                # 找下一个交易日
                future_days = trading_days[trading_days > pd.Timestamp(news_date)]
                if len(future_days) > 0:
                    return future_days[0].date()
                return None

            # 找当前或下一个交易日
            current_or_future = trading_days[trading_days >= pd.Timestamp(news_date)]
            if len(current_or_future) > 0:
                return current_or_future[0].date()
            return None

        df["trading_date"] = df["datetime"].apply(get_trading_date)
        df = df.dropna(subset=["trading_date"])

        return df

    def aggregate_daily_sentiment(
        self,
        sentiment_df: pd.DataFrame,
        method: str = "mean",
    ) -> pd.DataFrame:
        """
        聚合每日情感分数

        Args:
            sentiment_df: 包含情感分数的 DataFrame
            method: 聚合方法 ("mean", "median", "weighted")

        Returns:
            聚合后的 DataFrame
        """
        sentiment_cols = ["positive", "negative", "neutral", "sentiment_score"]
        available_cols = [c for c in sentiment_cols if c in sentiment_df.columns]

        if method == "mean":
            agg_func = "mean"
        elif method == "median":
            agg_func = "median"
        else:
            agg_func = "mean"  # 默认

        grouped = sentiment_df.groupby(["trading_date", "symbol"])[available_cols]
        aggregated = grouped.agg(agg_func)

        return aggregated.reset_index()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 创建测试数据
    test_news = pd.DataFrame({
        "datetime": pd.to_datetime([
            "2024-01-15 09:00:00",
            "2024-01-15 10:30:00",
            "2024-01-15 14:00:00",
            "2024-01-16 09:00:00",
        ]),
        "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
        "headline": [
            "Apple stock surges after strong earnings beat",
            "Apple announces new product launch",
            "Concerns rise over Apple supply chain",
            "Apple shares rally on positive outlook",
        ],
        "summary": [
            "Apple Inc reported quarterly earnings that beat analyst expectations",
            "The tech giant unveiled its latest product lineup",
            "Supply chain risks may impact production",
            "Analysts upgrade Apple stock rating",
        ],
    })

    extractor = NewsStatisticalFeatures()

    # 测试单日提取
    features = extractor.extract_daily_features(test_news, "2024-01-15", "AAPL")
    print("2024-01-15 AAPL 特征:")
    for k, v in features.items():
        print(f"  {k}: {v}")

    # 测试批量提取
    dates = ["2024-01-15", "2024-01-16"]
    symbols = ["AAPL"]
    features_df = extractor.extract_features_for_df(test_news, dates, symbols)
    print("\n批量特征:")
    print(features_df)
