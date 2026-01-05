"""
新闻处理器模块

包含情感分析、特征提取等处理器
"""

from .sentiment_analyzer import FinBERTSentimentAnalyzer, VADERSentimentAnalyzer
from .feature_extractor import NewsStatisticalFeatures

__all__ = [
    "FinBERTSentimentAnalyzer",
    "VADERSentimentAnalyzer",
    "NewsStatisticalFeatures",
]
