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

        # 强制加载数据，然后合并新闻特征
        self._force_load_and_merge_news()

    def _force_load_and_merge_news(self):
        """强制加载数据并合并新闻特征"""
        if self._news_df is None:
            return

        # 强制触发数据加载
        # 通过访问 fetch 方法来确保 _data 被初始化
        try:
            # 这会触发数据加载
            _ = self.fetch(col_set="feature")
        except Exception as e:
            print(f"警告: 强制加载数据时出错: {e}")
            return

        # 现在 _data 应该已经被初始化，进行合并
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
            print("警告: 没有新闻数据可合并")
            return

        # 获取主数据
        # 注意: self._data 在父类初始化后才存在
        if not hasattr(self, "_data") or self._data is None:
            print("警告: 主数据 _data 不存在，无法合并新闻特征")
            return

        try:
            # 选择需要的新闻特征列
            news_cols = self._get_news_feature_columns()
            available_cols = [c for c in news_cols if c in self._news_df.columns]

            if not available_cols:
                print(f"警告: 新闻数据中没有找到匹配的特征列")
                print(f"  期望的列: {news_cols}")
                print(f"  实际的列: {self._news_df.columns.tolist()}")
                return

            news_subset = self._news_df[available_cols].copy()

            # 确保索引格式匹配
            # 主数据索引: (datetime, instrument)
            # 需要确保新闻数据也是相同格式
            main_index_names = self._data.index.names
            news_index_names = news_subset.index.names

            print(f"主数据索引名: {main_index_names}, 形状: {self._data.shape}")
            print(f"新闻数据索引名: {news_index_names}, 形状: {news_subset.shape}")

            # 检查主数据是否有 MultiIndex 列
            has_multi_columns = isinstance(self._data.columns, pd.MultiIndex)
            print(f"主数据是否有 MultiIndex 列: {has_multi_columns}")

            original_cols = self._data.columns.tolist()

            if has_multi_columns:
                # 主数据有 MultiIndex 列，需要特殊处理
                # 保存原始的 MultiIndex 列结构
                original_column_index = self._data.columns

                # 对于每个新闻特征，直接添加到 _data 中
                # 使用索引对齐，不需要重置索引
                for col in available_cols:
                    if col in news_subset.columns:
                        # 创建一个与主数据索引对齐的 Series
                        news_series = news_subset[col]

                        # 使用 reindex 对齐到主数据的索引
                        aligned_series = news_series.reindex(self._data.index)

                        # 添加为新列，使用二级列名为空字符串
                        self._data[('feature', col)] = aligned_series

                print(f"直接添加新闻特征到 MultiIndex 列数据中")

            else:
                # 主数据是普通列，使用标准 merge
                # 重置索引进行合并
                main_reset = self._data.reset_index()
                news_reset = news_subset.reset_index()

                # 确保日期格式一致
                date_col = main_index_names[0]  # datetime
                inst_col = main_index_names[1]  # instrument

                # 标准化日期格式
                main_reset[date_col] = pd.to_datetime(main_reset[date_col])
                news_reset[news_index_names[0]] = pd.to_datetime(news_reset[news_index_names[0]])

                # 重命名新闻数据的索引列以匹配主数据
                rename_map = {}
                if news_index_names[0] != date_col:
                    rename_map[news_index_names[0]] = date_col
                if news_index_names[1] != inst_col:
                    rename_map[news_index_names[1]] = inst_col
                if rename_map:
                    news_reset = news_reset.rename(columns=rename_map)

                # 使用 merge 合并
                merged = pd.merge(
                    main_reset,
                    news_reset,
                    on=[date_col, inst_col],
                    how='left'
                )

                # 重新设置索引
                merged = merged.set_index([date_col, inst_col])
                merged.index.names = main_index_names

                # 更新 _data
                self._data = merged

            # 检查新增的列
            new_cols = [c for c in self._data.columns if c not in original_cols]
            print(f"成功添加 {len(new_cols)} 个新闻特征列: {new_cols}")

            # 添加滚动特征
            if self.add_news_rolling:
                self._add_rolling_features()

            # 填充缺失值
            self._fill_news_missing()

            # 最终检查 - 处理 MultiIndex 列的情况
            if has_multi_columns:
                news_cols_in_data = [c for c in self._data.columns
                                     if isinstance(c, tuple) and len(c) > 1 and str(c[1]).startswith("news_")]
            else:
                news_cols_in_data = [c for c in self._data.columns if str(c).startswith("news_")]
            print(f"新闻特征合并完成，总数据形状: {self._data.shape}")
            print(f"包含 {len(news_cols_in_data)} 个新闻特征列: {news_cols_in_data}")

        except Exception as e:
            import traceback
            print(f"警告: 合并新闻特征时出错: {e}")
            traceback.print_exc()

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

        # 检查是否是 MultiIndex 列
        has_multi_columns = isinstance(self._data.columns, pd.MultiIndex)

        # 找到新闻相关的列
        if has_multi_columns:
            news_cols = [c for c in self._data.columns
                         if isinstance(c, tuple) and len(c) > 1 and str(c[1]).startswith("news_")]
        else:
            news_cols = [c for c in self._data.columns if str(c).startswith("news_")]

        if not news_cols:
            return

        # 先 forward fill (按股票分组)
        try:
            for col in news_cols:
                # 使用 transform 进行分组 forward fill，更简洁
                self._data[col] = (
                    self._data[col]
                    .groupby(level="instrument")
                    .transform(lambda x: x.ffill(limit=5))
                )
        except Exception as e:
            # 如果分组 ffill 失败，直接对整列做 ffill
            try:
                for col in news_cols:
                    self._data[col] = self._data[col].ffill(limit=5)
            except Exception:
                pass  # 忽略，后面会用默认值填充

        # 再用默认值填充剩余缺失
        for col in news_cols:
            # 获取列的简单名称用于查找默认值
            if has_multi_columns and isinstance(col, tuple):
                simple_name = col[1]  # ('feature', 'news_xxx') -> 'news_xxx'
            else:
                simple_name = col

            default_value = fill_values.get(simple_name, 0.0)
            self._data[col] = self._data[col].fillna(default_value)

        # 填充滚动特征的缺失值
        if has_multi_columns:
            rolling_cols = [c for c in self._data.columns
                           if isinstance(c, tuple) and len(c) > 1 and ("_ma" in str(c[1]) or "_mom" in str(c[1]))]
        else:
            rolling_cols = [c for c in self._data.columns if "_ma" in str(c) or "_mom" in str(c)]

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
