"""
新闻情感分析器

提供 FinBERT (高精度) 和 VADER (快速) 两种情感分析方法
"""

import logging
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseSentimentAnalyzer(ABC):
    """情感分析器基类"""

    @abstractmethod
    def analyze(self, text: str) -> Dict[str, float]:
        """
        分析单条文本的情感

        Args:
            text: 输入文本

        Returns:
            包含情感分数的字典:
            - positive: 正面情感概率 [0, 1]
            - negative: 负面情感概率 [0, 1]
            - neutral: 中性情感概率 [0, 1]
            - sentiment_score: 综合情感分数 [-1, 1]
        """
        pass

    @abstractmethod
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """批量分析多条文本"""
        pass


class FinBERTSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    基于 FinBERT 的金融文本情感分析器

    使用 ProsusAI/finbert 模型，专门针对金融文本微调
    适合对准确度要求较高的场景

    使用方法:
        analyzer = FinBERTSentimentAnalyzer()
        result = analyzer.analyze("Apple stock surges after earnings beat")
        # {'positive': 0.85, 'negative': 0.05, 'neutral': 0.10, 'sentiment_score': 0.80}
    """

    def __init__(self, device: str = None, model_name: str = "ProsusAI/finbert"):
        """
        初始化 FinBERT 分析器

        Args:
            device: 设备 ("cuda", "cpu", "mps")，None 则自动选择
            model_name: 模型名称
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            raise ImportError(
                "需要安装 transformers 和 torch。"
                "运行: pip install transformers torch"
            )

        # 自动选择设备
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        logger.info(f"FinBERT 使用设备: {device}")

        # 加载模型
        logger.info(f"加载模型 {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # FinBERT 的标签顺序: positive, negative, neutral
        self.label_map = {0: "positive", 1: "negative", 2: "neutral"}

    def analyze(self, text: str) -> Dict[str, float]:
        """分析单条文本的情感"""
        import torch

        if not text or not text.strip():
            return self._empty_result()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        return {
            "positive": float(scores[0]),
            "negative": float(scores[1]),
            "neutral": float(scores[2]),
            "sentiment_score": float(scores[0] - scores[1]),  # [-1, 1]
        }

    def analyze_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[Dict[str, float]]:
        """
        批量分析文本情感

        Args:
            texts: 文本列表
            batch_size: 批次大小

        Returns:
            情感分数列表
        """
        import torch

        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # 过滤空文本
            valid_indices = []
            valid_texts = []
            for j, text in enumerate(batch):
                if text and text.strip():
                    valid_indices.append(j)
                    valid_texts.append(text)

            # 初始化结果
            batch_results = [self._empty_result() for _ in batch]

            if valid_texts:
                inputs = self.tokenizer(
                    valid_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    scores = torch.softmax(outputs.logits, dim=1).cpu().numpy()

                for idx, score in zip(valid_indices, scores):
                    batch_results[idx] = {
                        "positive": float(score[0]),
                        "negative": float(score[1]),
                        "neutral": float(score[2]),
                        "sentiment_score": float(score[0] - score[1]),
                    }

            results.extend(batch_results)

            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"FinBERT 处理进度: {min(i + batch_size, len(texts))}/{len(texts)}")

        return results

    def _empty_result(self) -> Dict[str, float]:
        """返回空/中性结果"""
        return {
            "positive": 0.33,
            "negative": 0.33,
            "neutral": 0.34,
            "sentiment_score": 0.0,
        }


class VADERSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    基于 VADER 的情感分析器

    VADER (Valence Aware Dictionary and sEntiment Reasoner) 是一个
    基于规则的情感分析工具，速度快，不需要 GPU

    适合对速度要求较高或资源有限的场景

    使用方法:
        analyzer = VADERSentimentAnalyzer()
        result = analyzer.analyze("Apple stock surges after earnings beat")
    """

    def __init__(self):
        """初始化 VADER 分析器"""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError:
            raise ImportError(
                "需要安装 vaderSentiment。运行: pip install vaderSentiment"
            )

        self.analyzer = SentimentIntensityAnalyzer()
        logger.info("VADER 情感分析器初始化完成")

    def analyze(self, text: str) -> Dict[str, float]:
        """分析单条文本的情感"""
        if not text or not text.strip():
            return self._empty_result()

        scores = self.analyzer.polarity_scores(text)

        return {
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
            "sentiment_score": scores["compound"],  # [-1, 1]
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """批量分析文本情感"""
        return [self.analyze(text) for text in texts]

    def _empty_result(self) -> Dict[str, float]:
        """返回空/中性结果"""
        return {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "sentiment_score": 0.0,
        }


def get_sentiment_analyzer(
    method: str = "finbert", **kwargs
) -> BaseSentimentAnalyzer:
    """
    获取情感分析器

    Args:
        method: 分析方法 ("finbert" 或 "vader")
        **kwargs: 传递给分析器的参数

    Returns:
        情感分析器实例
    """
    if method.lower() == "finbert":
        return FinBERTSentimentAnalyzer(**kwargs)
    elif method.lower() == "vader":
        # VADER 不需要 device 参数，过滤掉
        kwargs.pop("device", None)
        return VADERSentimentAnalyzer(**kwargs)
    else:
        raise ValueError(f"不支持的分析方法: {method}。支持: finbert, vader")


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    test_texts = [
        "Apple stock surges 5% after strong earnings beat expectations",
        "Tesla shares plunge amid concerns over declining sales",
        "Microsoft announces regular quarterly dividend",
        "",  # 空文本测试
    ]

    print("=== VADER 测试 ===")
    vader = VADERSentimentAnalyzer()
    for text in test_texts:
        result = vader.analyze(text)
        print(f"文本: {text[:50]}...")
        print(f"结果: {result}\n")

    print("=== FinBERT 测试 ===")
    try:
        finbert = FinBERTSentimentAnalyzer()
        for text in test_texts:
            result = finbert.analyze(text)
            print(f"文本: {text[:50]}...")
            print(f"结果: {result}\n")
    except ImportError as e:
        print(f"FinBERT 未安装: {e}")
