"""
新闻特征处理器

用于 Qlib 数据处理管道的新闻特征归一化和转换处理器
"""

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from qlib.data.dataset.processor import Processor

logger = logging.getLogger(__name__)


class NewsZScoreNorm(Processor):
    """
    新闻特征 Z-Score 归一化

    使用稳健统计量 (中位数和 MAD) 处理新闻数据中的异常值

    使用方法:
        processor = NewsZScoreNorm(
            fit_start_time="2020-01-01",
            fit_end_time="2023-12-31",
            news_columns=["news_sentiment_score", "news_count_log"]
        )
    """

    def __init__(
        self,
        fit_start_time: str,
        fit_end_time: str,
        news_columns: List[str] = None,
        clip: float = 3.0,
    ):
        """
        初始化处理器

        Args:
            fit_start_time: 拟合开始时间
            fit_end_time: 拟合结束时间
            news_columns: 要归一化的新闻特征列，None 则自动检测 news_ 前缀的列
            clip: 裁剪阈值 (标准差倍数)
        """
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.news_columns = news_columns
        self.clip = clip

        # 拟合后的参数
        self.median_train = None
        self.mad_train = None
        self.cols = None

    def fit(self, df: pd.DataFrame = None):
        """拟合归一化参数"""
        from qlib.data.dataset.utils import fetch_df_by_index

        # 获取拟合期间的数据
        df_fit = fetch_df_by_index(
            df, slice(self.fit_start_time, self.fit_end_time), level="datetime"
        )

        # 确定要处理的列
        if self.news_columns is not None:
            self.cols = [c for c in self.news_columns if c in df_fit.columns]
        else:
            # 自动检测 news_ 前缀的列
            self.cols = [c for c in df_fit.columns if c.startswith("news_")]

        if not self.cols:
            logger.warning("未找到新闻特征列")
            return

        # 计算稳健统计量
        data = df_fit[self.cols].values
        self.median_train = np.nanmedian(data, axis=0)
        # MAD (Median Absolute Deviation) * 1.4826 ≈ 标准差
        self.mad_train = (
            np.nanmedian(np.abs(data - self.median_train), axis=0) * 1.4826 + 1e-8
        )

        logger.info(f"NewsZScoreNorm 拟合完成，处理 {len(self.cols)} 列")

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用归一化"""
        if self.cols is None or not self.cols:
            return df

        df = df.copy()

        # 归一化
        for i, col in enumerate(self.cols):
            if col in df.columns:
                normalized = (df[col].values - self.median_train[i]) / self.mad_train[i]

                # 裁剪异常值
                if self.clip is not None:
                    normalized = np.clip(normalized, -self.clip, self.clip)

                df[col] = normalized

        return df

    def is_for_infer(self) -> bool:
        return True

    def readonly(self) -> bool:
        return False


class NewsFillna(Processor):
    """
    新闻特征缺失值填充

    使用中性值填充新闻特征的缺失值
    """

    # 默认填充值
    DEFAULT_FILL_VALUES = {
        "news_positive": 0.33,
        "news_negative": 0.33,
        "news_neutral": 0.34,
        "news_sentiment_score": 0.0,
        "news_count": 0.0,
        "news_count_log": 0.0,
        "news_kw_bullish": 0.0,
        "news_kw_bearish": 0.0,
        "news_kw_volatility": 0.0,
        "news_bull_bear_ratio": 0.0,
        "news_avg_headline_len": 0.0,
        "news_total_text_len": 0.0,
    }

    def __init__(self, fill_values: dict = None):
        """
        初始化处理器

        Args:
            fill_values: 自定义填充值字典
        """
        self.fill_values = self.DEFAULT_FILL_VALUES.copy()
        if fill_values:
            self.fill_values.update(fill_values)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """填充缺失值"""
        df = df.copy()

        for col, value in self.fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)

        return df

    def is_for_infer(self) -> bool:
        return True

    def readonly(self) -> bool:
        return False


class NewsSentimentRolling(Processor):
    """
    新闻情感滚动特征

    生成情感分数的滚动均值和动量特征
    """

    def __init__(
        self,
        windows: List[int] = None,
        columns: List[str] = None,
    ):
        """
        初始化处理器

        Args:
            windows: 滚动窗口列表，默认 [3, 5, 10]
            columns: 要处理的列，默认 news_sentiment_score 和 news_count_log
        """
        self.windows = windows or [3, 5, 10]
        self.columns = columns or ["news_sentiment_score", "news_count_log"]

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成滚动特征"""
        df = df.copy()

        for col in self.columns:
            if col not in df.columns:
                continue

            for w in self.windows:
                # 滚动均值
                ma_col = f"{col}_ma{w}"
                df[ma_col] = (
                    df.groupby(level="instrument")[col]
                    .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
                )

                # 动量 (当前值 - 滚动均值)
                mom_col = f"{col}_mom{w}"
                df[mom_col] = df[col] - df[ma_col]

        return df

    def is_for_infer(self) -> bool:
        return True

    def readonly(self) -> bool:
        return False


class NewsCountLogTransform(Processor):
    """
    新闻计数对数转换

    对计数类特征应用 log(1 + x) 转换
    """

    def __init__(self, columns: List[str] = None):
        """
        初始化处理器

        Args:
            columns: 要转换的列
        """
        self.columns = columns or [
            "news_count",
            "news_kw_bullish",
            "news_kw_bearish",
            "news_kw_volatility",
            "news_total_text_len",
        ]

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用对数转换"""
        df = df.copy()

        for col in self.columns:
            if col in df.columns:
                df[col] = np.log1p(df[col])

        return df

    def is_for_infer(self) -> bool:
        return True

    def readonly(self) -> bool:
        return False


class NewsFeatureFilter(Processor):
    """
    新闻特征过滤器

    只保留指定的新闻特征列
    """

    # 核心新闻特征
    CORE_FEATURES = [
        "news_sentiment_score",
        "news_count_log",
        "news_bull_bear_ratio",
    ]

    # 完整新闻特征
    ALL_FEATURES = [
        "news_positive",
        "news_negative",
        "news_neutral",
        "news_sentiment_score",
        "news_count",
        "news_count_log",
        "news_kw_bullish",
        "news_kw_bearish",
        "news_kw_volatility",
        "news_bull_bear_ratio",
        "news_avg_headline_len",
        "news_total_text_len",
    ]

    def __init__(self, features: Union[str, List[str]] = "all"):
        """
        初始化处理器

        Args:
            features: "core" 使用核心特征, "all" 使用全部特征, 或自定义列表
        """
        if features == "core":
            self.features = self.CORE_FEATURES
        elif features == "all":
            self.features = self.ALL_FEATURES
        else:
            self.features = features

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤特征"""
        # 保留非新闻特征列和指定的新闻特征列
        keep_cols = []
        for col in df.columns:
            if not col.startswith("news_"):
                keep_cols.append(col)
            elif col in self.features:
                keep_cols.append(col)

        return df[keep_cols]

    def is_for_infer(self) -> bool:
        return True

    def readonly(self) -> bool:
        return False


# 便捷函数：获取推荐的新闻处理器配置
def get_news_processors(
    fit_start_time: str,
    fit_end_time: str,
    normalize: bool = True,
    add_rolling: bool = False,
    feature_set: str = "all",
) -> List[dict]:
    """
    获取推荐的新闻特征处理器配置

    Args:
        fit_start_time: 归一化拟合开始时间
        fit_end_time: 归一化拟合结束时间
        normalize: 是否归一化
        add_rolling: 是否添加滚动特征
        feature_set: 特征集 ("core" 或 "all")

    Returns:
        处理器配置列表，可直接用于 DataHandler
    """
    processors = []

    # 1. 填充缺失值
    processors.append({
        "class": "data.news_processors.NewsFillna",
        "kwargs": {},
    })

    # 2. 归一化
    if normalize:
        processors.append({
            "class": "data.news_processors.NewsZScoreNorm",
            "kwargs": {
                "fit_start_time": fit_start_time,
                "fit_end_time": fit_end_time,
            },
        })

    # 3. 滚动特征
    if add_rolling:
        processors.append({
            "class": "data.news_processors.NewsSentimentRolling",
            "kwargs": {},
        })

    # 4. 特征过滤
    if feature_set != "all":
        processors.append({
            "class": "data.news_processors.NewsFeatureFilter",
            "kwargs": {"features": feature_set},
        })

    return processors
