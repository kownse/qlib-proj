"""
使用纯 Pandas/Qlib表达式 实现的 DataHandler

替代 TA-Lib 版本，避免大数据量时的内存问题
支持 Alpha158 和 Alpha360 两种特征集
"""

from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha360DL
from qlib.data.dataset.handler import DataHandlerLP


class Alpha158_Volatility_Pandas(DataHandlerLP):
    """
    Alpha158 特征 + Pandas 实现的技术指标 + N天价格波动率标签

    不依赖 TA-Lib，使用 Qlib 内置表达式实现技术指标
    """

    def __init__(
        self,
        volatility_window=2,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        self.volatility_window = volatility_window

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

    def get_feature_config(self):
        """
        获取特征配置：Alpha158 + 用 Qlib 表达式实现的技术指标
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

        # 添加用 Qlib 表达式实现的技术指标
        extra_fields, extra_names = self._get_pandas_features()
        fields.extend(extra_fields)
        names.extend(extra_names)

        return fields, names

    def _get_pandas_features(self):
        """
        使用 Qlib 内置表达式实现技术指标（不依赖 TA-Lib）

        Qlib 支持的表达式：
        - Ref($close, n): n期前的值
        - Mean($close, n): n期均值
        - Std($close, n): n期标准差
        - Max($high, n): n期最高
        - Min($low, n): n期最低
        - Sum($volume, n): n期求和
        - Corr($close, $volume, n): n期相关系数
        - Cov($close, $volume, n): n期协方差
        - Delta($close, n): 与n期前的差值
        - Slope($close, n): 线性回归斜率
        - Resi($close, n): 线性回归残差
        - Rank($close): 横截面排名
        - Sign($close): 符号函数
        - Abs($close): 绝对值
        - Log($close): 对数
        - Power($close, n): 幂函数
        """
        fields = []
        names = []

        # ==================== RSI (相对强弱指数) ====================
        # RSI = 100 - 100 / (1 + RS), RS = 平均涨幅 / 平均跌幅
        for w in [7, 14, 21]:
            # 简化版 RSI：用涨跌比例近似
            fields.append(f"Mean(Max($close - Ref($close, 1), 0), {w}) / (Mean(Abs($close - Ref($close, 1)), {w}) + 1e-12)")
            names.append(f"RSI{w}")

        # ==================== 动量指标 ====================
        # ROC (变化率)
        for w in [5, 10, 20, 30]:
            fields.append(f"$close / Ref($close, {w}) - 1")
            names.append(f"ROC{w}")

        # 动量 (价格变化)
        for w in [5, 10, 20, 30]:
            fields.append(f"($close - Ref($close, {w})) / $close")
            names.append(f"MOM{w}")

        # ==================== 移动平均 ====================
        # SMA vs 价格
        for w in [5, 10, 20, 30, 60]:
            fields.append(f"Mean($close, {w}) / $close")
            names.append(f"SMA{w}")

        # EMA 近似（用加权均值近似）
        for w in [5, 10, 20, 30]:
            # 简化：用最近的数据权重更高
            fields.append(f"(Mean($close, {w//2}) * 2 + Mean($close, {w})) / 3 / $close")
            names.append(f"EMA_APPROX{w}")

        # ==================== 布林带 ====================
        for w in [10, 20, 30]:
            # 布林带上轨距离
            fields.append(f"(Mean($close, {w}) + 2 * Std($close, {w}) - $close) / $close")
            names.append(f"BB_UPPER{w}")
            # 布林带下轨距离
            fields.append(f"($close - Mean($close, {w}) + 2 * Std($close, {w})) / $close")
            names.append(f"BB_LOWER{w}")
            # 布林带宽度
            fields.append(f"4 * Std($close, {w}) / $close")
            names.append(f"BB_WIDTH{w}")
            # 价格在布林带中的位置 (0-1)
            fields.append(f"($close - Mean($close, {w}) + 2*Std($close, {w})) / (4*Std($close, {w}) + 1e-12)")
            names.append(f"BB_POS{w}")

        # ==================== ATR (平均真实范围) ====================
        for w in [7, 14, 21]:
            # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
            # 简化：用 high-low 近似
            fields.append(f"Mean($high - $low, {w}) / $close")
            names.append(f"ATR{w}")

        # ==================== 波动率指标 ====================
        for w in [5, 10, 20, 30]:
            # 收益率标准差
            fields.append(f"Std($close / Ref($close, 1) - 1, {w})")
            names.append(f"VOLATILITY{w}")

        # ==================== 趋势指标 ====================
        # ADX 近似 (用方向运动指标近似)
        for w in [7, 14, 21]:
            # +DM: 上涨动量
            fields.append(f"Mean(Max($high - Ref($high, 1), 0), {w}) / (Mean($high - $low, {w}) + 1e-12)")
            names.append(f"PLUS_DM{w}")
            # -DM: 下跌动量
            fields.append(f"Mean(Max(Ref($low, 1) - $low, 0), {w}) / (Mean($high - $low, {w}) + 1e-12)")
            names.append(f"MINUS_DM{w}")

        # 线性回归斜率
        for w in [10, 20, 30]:
            fields.append(f"Slope($close, {w}) / $close")
            names.append(f"SLOPE{w}")

        # ==================== 成交量指标 ====================
        # 成交量变化
        for w in [5, 10, 20]:
            fields.append(f"$volume / Mean($volume, {w})")
            names.append(f"VOL_RATIO{w}")

        # 价量相关性
        for w in [10, 20, 30]:
            fields.append(f"Corr($close, $volume, {w})")
            names.append(f"PRICE_VOL_CORR{w}")

        # OBV 近似 (累积成交量方向)
        for w in [10, 20]:
            fields.append(f"Mean(Sign($close - Ref($close, 1)) * $volume, {w}) / Mean($volume, {w})")
            names.append(f"OBV_APPROX{w}")

        # ==================== 价格位置指标 ====================
        for w in [5, 10, 20, 30]:
            # 价格在区间中的位置
            fields.append(f"($close - Min($low, {w})) / (Max($high, {w}) - Min($low, {w}) + 1e-12)")
            names.append(f"PRICE_POS{w}")

        # ==================== 支撑阻力 ====================
        for w in [10, 20, 30]:
            # 距离最高点
            fields.append(f"($close - Max($high, {w})) / $close")
            names.append(f"DIST_HIGH{w}")
            # 距离最低点
            fields.append(f"($close - Min($low, {w})) / $close")
            names.append(f"DIST_LOW{w}")

        # ==================== MACD 近似 ====================
        # MACD = EMA12 - EMA26，用 SMA 近似
        fields.append(f"(Mean($close, 12) - Mean($close, 26)) / $close")
        names.append("MACD_APPROX")
        fields.append(f"(Mean($close, 5) - Mean($close, 10)) / $close")
        names.append("MACD_FAST_APPROX")

        # ==================== Williams %R ====================
        for w in [7, 14, 21]:
            fields.append(f"(Max($high, {w}) - $close) / (Max($high, {w}) - Min($low, {w}) + 1e-12)")
            names.append(f"WILLR{w}")

        # ==================== CCI (商品通道指数) 近似 ====================
        for w in [7, 14, 20]:
            # CCI = (TP - SMA) / (0.015 * Mean Deviation)
            # TP = (High + Low + Close) / 3
            fields.append(f"(($high + $low + $close)/3 - Mean(($high + $low + $close)/3, {w})) / (Std(($high + $low + $close)/3, {w}) + 1e-12)")
            names.append(f"CCI{w}")

        # ==================== 价格加速度 ====================
        for w in [5, 10, 20]:
            # 二阶差分
            fields.append(f"Delta($close, 1) - Ref(Delta($close, 1), {w})")
            names.append(f"ACCEL{w}")

        return fields, names

    def get_label_config(self):
        """
        返回N天波动率标签
        """
        volatility_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [volatility_expr], ["LABEL0"]


class Alpha360_Volatility_Pandas(DataHandlerLP):
    """
    Alpha360 特征 + N天价格波动率标签

    使用 Qlib 内置的 Alpha360DL 获取特征配置（60天历史数据）
    不依赖 TA-Lib，适合大数据量处理

    Alpha360 特征说明：
    - 使用最近60天的价格和成交量数据
    - 包含 CLOSE, OPEN, HIGH, LOW, VWAP, VOLUME 各60天历史
    - 所有价格数据都用当前 $close 标准化
    - 成交量数据用当前 $volume 标准化
    - 总共 360 个特征 (6 * 60)
    """

    def __init__(
        self,
        volatility_window=2,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        self.volatility_window = volatility_window

        from qlib.contrib.data.handler import check_transform_proc, _DEFAULT_LEARN_PROCESSORS

        if learn_processors is None:
            learn_processors = _DEFAULT_LEARN_PROCESSORS

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": Alpha360DL.get_feature_config(),
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

    def get_label_config(self):
        """
        返回N天波动率标签
        """
        volatility_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [volatility_expr], ["LABEL0"]
