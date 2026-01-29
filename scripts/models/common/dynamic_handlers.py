"""
动态特征 Handler 模块

提供可动态添加股票特征和宏观特征的 DataHandler 基类。

Handler 类型:
1. DynamicTabularHandler - 用于表格模型 (CatBoost, LightGBM, AE-MLP)
   - 每个特征是单一数值
   - 适合树模型和 MLP 模型

2. DynamicTimeSeriesHandler - 用于时序模型 (TCN, Transformer)
   - 每个特征扩展为 N 天历史数据
   - 适合 RNN/TCN/Transformer 等序列模型
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from qlib.data.dataset.handler import DataHandlerLP

from models.feature_engineering.feature_selection_utils import load_macro_data


# ============================================================================
# 表格模型动态 Handler (CatBoost, LightGBM, AE-MLP)
# ============================================================================

class DynamicTabularHandler(DataHandlerLP):
    """
    动态表格模型 Handler，支持增量添加 Stock 和 Macro 特征。

    用于 CatBoost, LightGBM, AE-MLP 等表格模型。
    每个特征是单一数值（不扩展为时序）。

    Parameters
    ----------
    stock_features : Dict[str, str]
        股票特征字典 {name: qlib_expression}
    macro_features : List[str]
        宏观特征名称列表
    volatility_window : int
        标签计算的 N 天窗口（默认 5 天）
    macro_lag : int
        宏观特征的滞后天数（防止前视偏差，默认 1 天）
    """

    def __init__(
        self,
        stock_features: Dict[str, str] = None,
        macro_features: List[str] = None,
        volatility_window: int = 5,
        macro_lag: int = 1,
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
        **kwargs,
    ):
        self.stock_features = stock_features or {}
        self.macro_features = macro_features or []
        self.volatility_window = volatility_window
        self.macro_lag = macro_lag

        self._macro_df = load_macro_data() if self.macro_features else None

        from qlib.contrib.data.handler import check_transform_proc, _DEFAULT_LEARN_PROCESSORS

        if learn_processors is None:
            learn_processors = _DEFAULT_LEARN_PROCESSORS

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self._get_feature_config(),
                    "label": kwargs.pop("label", self._get_label_config()),
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

    def _get_feature_config(self):
        """获取特征配置"""
        fields = []
        names = []

        for feat_name, expr in self.stock_features.items():
            fields.append(expr)
            names.append(feat_name)

        return fields, names

    def _get_label_config(self):
        """返回 N 天收益率标签"""
        label_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [label_expr], ["LABEL0"]

    def process_data(self, with_fit: bool = False):
        """处理数据，添加 macro 特征"""
        super().process_data(with_fit=with_fit)
        if self._macro_df is not None and self.macro_features:
            self._add_macro_features()

    def _add_macro_features(self):
        """添加时间对齐的 macro 特征"""
        available_cols = [c for c in self.macro_features if c in self._macro_df.columns]
        if not available_cols:
            return

        for attr in ['_learn', '_infer']:
            df = getattr(self, attr, None)
            if df is None:
                continue

            datetime_col = df.index.names[0]
            main_datetimes = df.index.get_level_values(datetime_col)
            has_multi_columns = isinstance(df.columns, pd.MultiIndex)

            macro_data = {}
            for col in available_cols:
                base_series = self._macro_df[col]
                # 滞后处理防止前视偏差
                shifted = base_series.shift(self.macro_lag)
                aligned_values = shifted.reindex(main_datetimes).values
                if has_multi_columns:
                    macro_data[('feature', col)] = aligned_values
                else:
                    macro_data[col] = aligned_values

            macro_df = pd.DataFrame(macro_data, index=df.index)
            merged = pd.concat([df, macro_df], axis=1, copy=False)
            setattr(self, attr, merged.copy())


# ============================================================================
# 时序模型动态 Handler (TCN, Transformer)
# ============================================================================

# 时序模型常量
DEFAULT_SEQ_LEN = 60  # 默认 60 天历史
ALPHA300_BASE_FEATURES = 5  # CLOSE, OPEN, HIGH, LOW, VOLUME (no VWAP)


class DynamicTimeSeriesHandler(DataHandlerLP):
    """
    动态时序模型 Handler，支持增量添加 TALib 和 Macro 特征。

    用于 TCN, Transformer 等时序模型。
    基线为 Alpha300 (5 OHLCV × 60 days = 300 features)。
    每个新增特征扩展为 seq_len 天的历史数据。

    Parameters
    ----------
    talib_features : Dict[str, str]
        TALib 特征字典 {name: qlib_expression}
    macro_features : List[str]
        宏观特征名称列表
    seq_len : int
        时序长度（默认 60 天）
    volatility_window : int
        标签计算的 N 天窗口（默认 5 天）
    include_alpha300 : bool
        是否包含 Alpha300 基线特征（默认 True）
    """

    def __init__(
        self,
        talib_features: Dict[str, str] = None,
        macro_features: List[str] = None,
        seq_len: int = DEFAULT_SEQ_LEN,
        volatility_window: int = 5,
        include_alpha300: bool = True,
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
        **kwargs,
    ):
        self.talib_features = talib_features or {}
        self.macro_features = macro_features or []
        self.seq_len = seq_len
        self.volatility_window = volatility_window
        self.include_alpha300 = include_alpha300

        self._macro_df = load_macro_data() if self.macro_features else None

        from qlib.contrib.data.handler import check_transform_proc, _DEFAULT_LEARN_PROCESSORS

        if learn_processors is None:
            learn_processors = _DEFAULT_LEARN_PROCESSORS

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self._get_feature_config(),
                    "label": kwargs.pop("label", self._get_label_config()),
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

    def _get_alpha300_features(self):
        """获取 Alpha300 基线特征 (5 OHLCV × seq_len days)"""
        fields = []
        names = []

        # CLOSE
        for i in range(self.seq_len - 1, 0, -1):
            fields.append(f"Ref($close, {i})/$close")
            names.append(f"CLOSE{i}")
        fields.append("$close/$close")
        names.append("CLOSE0")

        # OPEN
        for i in range(self.seq_len - 1, 0, -1):
            fields.append(f"Ref($open, {i})/$close")
            names.append(f"OPEN{i}")
        fields.append("$open/$close")
        names.append("OPEN0")

        # HIGH
        for i in range(self.seq_len - 1, 0, -1):
            fields.append(f"Ref($high, {i})/$close")
            names.append(f"HIGH{i}")
        fields.append("$high/$close")
        names.append("HIGH0")

        # LOW
        for i in range(self.seq_len - 1, 0, -1):
            fields.append(f"Ref($low, {i})/$close")
            names.append(f"LOW{i}")
        fields.append("$low/$close")
        names.append("LOW0")

        # VOLUME (无 VWAP，US 数据中全是 NaN)
        for i in range(self.seq_len - 1, 0, -1):
            fields.append(f"Ref($volume, {i})/($volume+1e-12)")
            names.append(f"VOLUME{i}")
        fields.append("$volume/($volume+1e-12)")
        names.append("VOLUME0")

        return fields, names

    def _get_talib_features_config(self):
        """获取 TALib 特征配置 (每个特征扩展为 seq_len 天)"""
        fields = []
        names = []

        for feat_name, expr in self.talib_features.items():
            # 扩展为 seq_len 天历史
            for i in range(self.seq_len - 1, 0, -1):
                ref_expr = f"Ref({expr}, {i})"
                fields.append(ref_expr)
                names.append(f"{feat_name}_{i}")
            # 当天的值
            fields.append(expr)
            names.append(f"{feat_name}_0")

        return fields, names

    def _get_feature_config(self):
        """获取完整特征配置"""
        fields = []
        names = []

        # Alpha300 基线
        if self.include_alpha300:
            alpha_fields, alpha_names = self._get_alpha300_features()
            fields.extend(alpha_fields)
            names.extend(alpha_names)

        # TALib 特征
        talib_fields, talib_names = self._get_talib_features_config()
        fields.extend(talib_fields)
        names.extend(talib_names)

        return fields, names

    def _get_label_config(self):
        """返回 N 天收益率标签"""
        label_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [label_expr], ["LABEL0"]

    def process_data(self, with_fit: bool = False):
        """处理数据，添加 macro 特征"""
        super().process_data(with_fit=with_fit)
        if self._macro_df is not None and self.macro_features:
            self._add_macro_features()

    def _add_macro_features(self):
        """添加时间对齐的 macro 特征 (每个扩展为 seq_len 天)"""
        available_cols = [c for c in self.macro_features if c in self._macro_df.columns]
        if not available_cols:
            return

        for attr in ['_learn', '_infer']:
            df = getattr(self, attr, None)
            if df is None:
                continue

            datetime_col = df.index.names[0]
            main_datetimes = df.index.get_level_values(datetime_col)
            has_multi_columns = isinstance(df.columns, pd.MultiIndex)

            macro_data = {}
            for col in available_cols:
                base_series = self._macro_df[col]
                # 扩展为 seq_len 天历史
                for i in range(self.seq_len - 1, -1, -1):
                    col_name = f"{col}_{i}"
                    shifted = base_series.shift(i + 1)  # +1 防止前视偏差
                    aligned_values = shifted.reindex(main_datetimes).values
                    if has_multi_columns:
                        macro_data[('feature', col_name)] = aligned_values
                    else:
                        macro_data[col_name] = aligned_values

            macro_df = pd.DataFrame(macro_data, index=df.index)
            merged = pd.concat([df, macro_df], axis=1, copy=False)
            setattr(self, attr, merged.copy())


# ============================================================================
# 数据归一化函数
# ============================================================================

def normalize_data(X, fit_stats=None):
    """
    对数据进行归一化处理：3σ clip + zscore

    Args:
        X: 输入数据 (DataFrame 或 ndarray)
        fit_stats: 可选的 (means, stds) 元组，用于应用已有的统计量

    Returns:
        normalized_data: 归一化后的数据 (numpy array)
        stats: (means, stds) 元组
    """
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X)

    X_df = X_df.fillna(0)
    X_df = X_df.replace([np.inf, -np.inf], 0)

    if fit_stats is None:
        means = X_df.mean().values
        stds = X_df.std().values
        stds = np.where(stds == 0, 1, stds)
    else:
        means, stds = fit_stats

    X_values = X_df.values.copy()
    for i in range(X_values.shape[1]):
        col_mean = means[i]
        col_std = stds[i]
        if col_std > 0:
            lower = col_mean - 3 * col_std
            upper = col_mean + 3 * col_std
            X_values[:, i] = np.clip(X_values[:, i], lower, upper)
            X_values[:, i] = (X_values[:, i] - col_mean) / col_std

    X_values = np.nan_to_num(X_values, nan=0.0, posinf=0.0, neginf=0.0)

    return X_values, (means, stds)
