"""
包含新闻特征的数据处理器

在 Alpha158_Volatility_TALib 基础上添加新闻情感和统计特征
"""

import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

# Add scripts directory to path for imports
script_dir = Path(__file__).parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset.handler import DataHandlerLP

# Import TA-Lib custom operators
from utils.talib_ops import TALIB_OPS

# 项目根目录
PROJECT_ROOT = script_dir.parent

# 新闻特征文件默认路径
DEFAULT_NEWS_PATH = PROJECT_ROOT / "my_data" / "news_processed" / "news_features.parquet"


class Alpha158_Volatility_TALib_News(DataHandlerLP):
    """
    Alpha158 特征 + TA-Lib 技术指标 + 新闻特征 + N天价格波动率标签

    在 Alpha158_Volatility_TALib 基础上扩展了新闻特征:
    - 情感特征: positive, negative, neutral, sentiment_score
    - 统计特征: news_count, keyword counts, bull/bear ratio
    - 可选: 滚动情感特征 (MA, momentum)

    总特征数: ~250 (市场特征) + 10-20 (新闻特征)

    使用方法:
        handler = Alpha158_Volatility_TALib_News(
            volatility_window=2,
            instruments=["AAPL", "MSFT", "NVDA"],
            start_time="2020-01-01",
            end_time="2024-12-31",
            fit_start_time="2020-01-01",
            fit_end_time="2023-12-31",
            news_data_path="my_data/news_processed/news_features.parquet",
        )
    """

    # 新闻特征列名
    NEWS_SENTIMENT_FEATURES = [
        "news_positive",
        "news_negative",
        "news_neutral",
        "news_sentiment_score",
    ]

    NEWS_STAT_FEATURES = [
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
        volatility_window: int = 2,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq: str = "day",
        infer_processors=[],
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        # 新闻相关参数
        news_data_path: Union[str, Path] = None,
        news_features: str = "all",  # "all", "sentiment", "stats", "core"
        add_news_rolling: bool = False,
        news_rolling_windows: List[int] = None,
        **kwargs,
    ):
        """
        初始化包含新闻特征的波动率预测 DataHandler

        Args:
            volatility_window: 波动率预测窗口（天数）
            news_data_path: 新闻特征文件路径 (parquet 格式)
            news_features: 新闻特征集
                - "all": 所有新闻特征
                - "sentiment": 仅情感特征
                - "stats": 仅统计特征
                - "core": 核心特征 (sentiment_score, count_log, bull_bear_ratio)
            add_news_rolling: 是否添加新闻滚动特征
            news_rolling_windows: 滚动窗口列表，默认 [3, 5, 10]
            **kwargs: 传递给父类的其他参数
        """
        self.volatility_window = volatility_window
        self.news_data_path = Path(news_data_path) if news_data_path else DEFAULT_NEWS_PATH
        self.news_features = news_features
        self.add_news_rolling = add_news_rolling
        self.news_rolling_windows = news_rolling_windows or [3, 5, 10]

        # 加载新闻特征
        self._news_df = self._load_news_features()

        from qlib.contrib.data.handler import check_transform_proc, _DEFAULT_LEARN_PROCESSORS

        if learn_processors is None:
            learn_processors = _DEFAULT_LEARN_PROCESSORS

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

        # 合并新闻特征到数据中
        self._merge_news_features()

    def _load_news_features(self) -> Optional[pd.DataFrame]:
        """加载新闻特征数据"""
        if not self.news_data_path.exists():
            print(f"警告: 新闻特征文件不存在: {self.news_data_path}")
            print("将仅使用市场特征，不包含新闻数据")
            return None

        try:
            df = pd.read_parquet(self.news_data_path)
            print(f"加载新闻特征: {df.shape}, 日期范围: {df.index.get_level_values(0).min()} - {df.index.get_level_values(0).max()}")
            return df
        except Exception as e:
            print(f"警告: 无法加载新闻特征: {e}")
            return None

    def _get_news_feature_columns(self) -> List[str]:
        """根据配置获取新闻特征列"""
        if self.news_features == "sentiment":
            return self.NEWS_SENTIMENT_FEATURES
        elif self.news_features == "stats":
            return self.NEWS_STAT_FEATURES
        elif self.news_features == "core":
            return ["news_sentiment_score", "news_count_log", "news_bull_bear_ratio"]
        else:  # "all"
            return self.NEWS_SENTIMENT_FEATURES + self.NEWS_STAT_FEATURES

    def _merge_news_features(self):
        """将新闻特征合并到主数据中"""
        if self._news_df is None:
            return

        # 获取主数据
        # 注意: self._data 在父类初始化后才存在
        if not hasattr(self, "_data") or self._data is None:
            return

        try:
            # 选择需要的新闻特征列
            news_cols = self._get_news_feature_columns()
            available_cols = [c for c in news_cols if c in self._news_df.columns]

            if not available_cols:
                print("警告: 新闻数据中没有找到匹配的特征列")
                return

            news_subset = self._news_df[available_cols]

            # 合并到主数据
            # 使用 join 而不是 merge，因为两者都有 MultiIndex
            self._data = self._data.join(news_subset, how="left")

            # 添加滚动特征
            if self.add_news_rolling:
                self._add_rolling_features()

            # 填充缺失值
            self._fill_news_missing()

            print(f"新闻特征合并完成，新数据形状: {self._data.shape}")

        except Exception as e:
            print(f"警告: 合并新闻特征时出错: {e}")

    def _add_rolling_features(self):
        """添加新闻滚动特征"""
        rolling_cols = ["news_sentiment_score", "news_count_log"]

        for col in rolling_cols:
            if col not in self._data.columns:
                continue

            for w in self.news_rolling_windows:
                # 滚动均值
                ma_col = f"{col}_ma{w}"
                self._data[ma_col] = (
                    self._data.groupby(level="instrument")[col]
                    .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
                )

                # 动量
                mom_col = f"{col}_mom{w}"
                self._data[mom_col] = self._data[col] - self._data[ma_col]

    def _fill_news_missing(self):
        """填充新闻特征的缺失值"""
        fill_values = {
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

        # 先 forward fill (按股票分组)
        news_cols = [c for c in self._data.columns if c.startswith("news_")]
        if news_cols:
            self._data[news_cols] = (
                self._data.groupby(level="instrument")[news_cols]
                .ffill(limit=5)
            )

        # 再用默认值填充剩余缺失
        for col, value in fill_values.items():
            if col in self._data.columns:
                self._data[col] = self._data[col].fillna(value)

        # 填充滚动特征的缺失值
        rolling_cols = [c for c in self._data.columns if "_ma" in c or "_mom" in c]
        for col in rolling_cols:
            self._data[col] = self._data[col].fillna(0.0)

    def get_feature_config(self):
        """
        获取特征配置，包含 Alpha158 + TA-Lib 指标

        注意: 新闻特征不在这里配置，而是在初始化后合并
        """
        # 获取 Alpha158 原始特征
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        fields, names = Alpha158.get_feature_config(conf)

        # 添加 TA-Lib 指标
        talib_fields, talib_names = self._get_talib_features()
        fields.extend(talib_fields)
        names.extend(talib_names)

        return fields, names

    def _get_talib_features(self):
        """获取 TA-Lib 技术指标特征 (与 Alpha158_Volatility_TALib 相同)"""
        fields = []
        names = []

        windows = [5, 10, 14, 20, 30]

        # ==================== 动量指标 ====================
        for w in [7, 14, 21]:
            fields.append(f"TALIB_RSI($close, {w})")
            names.append(f"TALIB_RSI{w}")

        for w in windows:
            fields.append(f"TALIB_MOM($close, {w})/$close")
            names.append(f"TALIB_MOM{w}")

        for w in windows:
            fields.append(f"TALIB_ROC($close, {w})")
            names.append(f"TALIB_ROC{w}")

        for w in [7, 14, 21]:
            fields.append(f"TALIB_CMO($close, {w})")
            names.append(f"TALIB_CMO{w}")

        for w in [7, 14, 21]:
            fields.append(f"TALIB_WILLR($high, $low, $close, {w})")
            names.append(f"TALIB_WILLR{w}")

        for w in [7, 14, 20]:
            fields.append(f"TALIB_CCI($high, $low, $close, {w})")
            names.append(f"TALIB_CCI{w}")

        for w in [12, 20, 30]:
            fields.append(f"TALIB_TRIX($close, {w})")
            names.append(f"TALIB_TRIX{w}")

        fields.append("TALIB_PPO($close, 12, 26)")
        names.append("TALIB_PPO_12_26")
        fields.append("TALIB_PPO($close, 5, 10)")
        names.append("TALIB_PPO_5_10")

        # ==================== MACD 指标 ====================
        fields.append("TALIB_MACD_MACD($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD")
        fields.append("TALIB_MACD_SIGNAL($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD_SIGNAL")
        fields.append("TALIB_MACD_HIST($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD_HIST")
        fields.append("TALIB_MACD_MACD($close, 5, 10, 5)/$close")
        names.append("TALIB_MACD_FAST")
        fields.append("TALIB_MACD_HIST($close, 5, 10, 5)/$close")
        names.append("TALIB_MACD_HIST_FAST")

        # ==================== 移动平均 ====================
        for w in [5, 10, 20, 30, 60]:
            fields.append(f"TALIB_EMA($close, {w})/$close")
            names.append(f"TALIB_EMA{w}")

        for w in [10, 20, 30]:
            fields.append(f"TALIB_DEMA($close, {w})/$close")
            names.append(f"TALIB_DEMA{w}")

        for w in [10, 20, 30]:
            fields.append(f"TALIB_TEMA($close, {w})/$close")
            names.append(f"TALIB_TEMA{w}")

        for w in [10, 20, 30]:
            fields.append(f"TALIB_KAMA($close, {w})/$close")
            names.append(f"TALIB_KAMA{w}")

        for w in [10, 20, 30]:
            fields.append(f"TALIB_WMA($close, {w})/$close")
            names.append(f"TALIB_WMA{w}")

        # ==================== 布林带 ====================
        for w in [10, 20, 30]:
            fields.append(f"(TALIB_BBANDS_UPPER($close, {w}, 2) - $close)/$close")
            names.append(f"TALIB_BB_UPPER_DIST{w}")
            fields.append(f"($close - TALIB_BBANDS_LOWER($close, {w}, 2))/$close")
            names.append(f"TALIB_BB_LOWER_DIST{w}")
            fields.append(f"(TALIB_BBANDS_UPPER($close, {w}, 2) - TALIB_BBANDS_LOWER($close, {w}, 2))/$close")
            names.append(f"TALIB_BB_WIDTH{w}")

        # ==================== 波动率指标 ====================
        for w in [7, 14, 21]:
            fields.append(f"TALIB_ATR($high, $low, $close, {w})/$close")
            names.append(f"TALIB_ATR{w}")

        for w in [7, 14, 21]:
            fields.append(f"TALIB_NATR($high, $low, $close, {w})")
            names.append(f"TALIB_NATR{w}")

        fields.append("TALIB_TRANGE($high, $low, $close)/$close")
        names.append("TALIB_TRANGE")

        # ==================== 趋势指标 ====================
        for w in [7, 14, 21]:
            fields.append(f"TALIB_ADX($high, $low, $close, {w})")
            names.append(f"TALIB_ADX{w}")

        for w in [7, 14, 21]:
            fields.append(f"TALIB_ADXR($high, $low, $close, {w})")
            names.append(f"TALIB_ADXR{w}")

        for w in [7, 14, 21]:
            fields.append(f"TALIB_PLUS_DI($high, $low, $close, {w})")
            names.append(f"TALIB_PLUS_DI{w}")
            fields.append(f"TALIB_MINUS_DI($high, $low, $close, {w})")
            names.append(f"TALIB_MINUS_DI{w}")
            fields.append(f"TALIB_PLUS_DI($high, $low, $close, {w}) - TALIB_MINUS_DI($high, $low, $close, {w})")
            names.append(f"TALIB_DI_DIFF{w}")

        for w in [14, 25]:
            fields.append(f"TALIB_AROON_UP($high, $low, {w})")
            names.append(f"TALIB_AROON_UP{w}")
            fields.append(f"TALIB_AROON_DOWN($high, $low, {w})")
            names.append(f"TALIB_AROON_DOWN{w}")
            fields.append(f"TALIB_AROONOSC($high, $low, {w})")
            names.append(f"TALIB_AROONOSC{w}")

        # ==================== 成交量指标 ====================
        fields.append("TALIB_OBV($close, $volume)/($volume+1e-12)")
        names.append("TALIB_OBV")
        fields.append("TALIB_AD($high, $low, $close, $volume)/($volume+1e-12)")
        names.append("TALIB_AD")
        fields.append("TALIB_ADOSC($high, $low, $close, $volume, 3, 10)/($volume+1e-12)")
        names.append("TALIB_ADOSC")

        for w in [7, 14, 21]:
            fields.append(f"TALIB_MFI($high, $low, $close, $volume, {w})")
            names.append(f"TALIB_MFI{w}")

        # ==================== 随机指标 ====================
        for fk, sk, sd in [(5, 3, 3), (14, 3, 3), (21, 5, 5)]:
            fields.append(f"TALIB_STOCH_K($high, $low, $close, {fk}, {sk}, {sd})")
            names.append(f"TALIB_STOCH_K_{fk}_{sk}_{sd}")
            fields.append(f"TALIB_STOCH_D($high, $low, $close, {fk}, {sk}, {sd})")
            names.append(f"TALIB_STOCH_D_{fk}_{sk}_{sd}")

        for tp in [14, 21]:
            fields.append(f"TALIB_STOCHRSI_K($close, {tp}, 5, 3)")
            names.append(f"TALIB_STOCHRSI_K{tp}")
            fields.append(f"TALIB_STOCHRSI_D($close, {tp}, 5, 3)")
            names.append(f"TALIB_STOCHRSI_D{tp}")

        # ==================== 统计指标 ====================
        for w in [5, 10, 20, 30]:
            fields.append(f"TALIB_STDDEV($close, {w}, 1)/$close")
            names.append(f"TALIB_STDDEV{w}")

        for w in [5, 10, 20]:
            fields.append(f"TALIB_VAR($close, {w}, 1)/($close*$close)")
            names.append(f"TALIB_VAR{w}")

        for w in [10, 20, 30]:
            fields.append(f"TALIB_LINEARREG($close, {w})/$close")
            names.append(f"TALIB_LINEARREG{w}")
            fields.append(f"TALIB_LINEARREG_SLOPE($close, {w})")
            names.append(f"TALIB_LINEARREG_SLOPE{w}")
            fields.append(f"TALIB_LINEARREG_ANGLE($close, {w})")
            names.append(f"TALIB_LINEARREG_ANGLE{w}")

        for w in [10, 20, 30]:
            fields.append(f"TALIB_TSF($close, {w})/$close")
            names.append(f"TALIB_TSF{w}")

        return fields, names

    def get_label_config(self):
        """返回N天波动率标签"""
        volatility_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [volatility_expr], ["LABEL0"]

    def get_news_feature_names(self) -> List[str]:
        """获取实际使用的新闻特征名称列表"""
        base_features = self._get_news_feature_columns()

        if self.add_news_rolling:
            rolling_features = []
            for col in ["news_sentiment_score", "news_count_log"]:
                if col in base_features:
                    for w in self.news_rolling_windows:
                        rolling_features.append(f"{col}_ma{w}")
                        rolling_features.append(f"{col}_mom{w}")
            return base_features + rolling_features

        return base_features


# 便捷函数: 创建带新闻特征的数据处理器
def create_handler_with_news(
    volatility_window: int = 2,
    instruments: list = None,
    start_time: str = "2020-01-01",
    end_time: str = "2024-12-31",
    fit_start_time: str = "2020-01-01",
    fit_end_time: str = "2023-12-31",
    news_data_path: str = None,
    news_features: str = "all",
    add_news_rolling: bool = False,
) -> Alpha158_Volatility_TALib_News:
    """
    便捷函数: 创建带新闻特征的数据处理器

    Args:
        volatility_window: 波动率预测窗口
        instruments: 股票列表
        start_time: 数据开始时间
        end_time: 数据结束时间
        fit_start_time: 归一化拟合开始时间
        fit_end_time: 归一化拟合结束时间
        news_data_path: 新闻特征文件路径
        news_features: 新闻特征集
        add_news_rolling: 是否添加滚动特征

    Returns:
        Alpha158_Volatility_TALib_News 实例
    """
    if instruments is None:
        instruments = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
            "AMD", "INTC", "AVGO", "QCOM",
        ]

    return Alpha158_Volatility_TALib_News(
        volatility_window=volatility_window,
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
        news_data_path=news_data_path,
        news_features=news_features,
        add_news_rolling=add_news_rolling,
    )
