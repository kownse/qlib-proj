"""
扩展的 DataHandler 类，用于波动率预测

包含 Alpha158 特征 + TA-Lib 技术指标 + N天价格波动率标签
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
script_dir = Path(__file__).parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

from qlib.contrib.data.handler import Alpha158, Alpha360
from qlib.data.dataset.handler import DataHandlerLP

# Import TA-Lib custom operators
from utils.talib_ops import TALIB_OPS


class Alpha360_Volatility(Alpha360):
    """
    Alpha360 特征 + N天价格波动率标签

    继承 Alpha360 的所有特征（60天的 OHLCV 数据），只修改标签为N天波动率

    Alpha360 特征说明：
    - 使用最近60天的价格和成交量数据
    - 包含 CLOSE, OPEN, HIGH, LOW, VWAP, VOLUME 各60天历史
    - 所有价格数据都用当前 $close 标准化
    - 成交量数据用当前 $volume 标准化
    - 总共 360 个特征 (6 * 60)
    """

    def __init__(self, volatility_window=2, **kwargs):
        """
        初始化波动率预测的 Alpha360 DataHandler

        Args:
            volatility_window: 波动率预测窗口（天数）
            **kwargs: 传递给父类的其他参数
        """
        self.volatility_window = volatility_window
        super().__init__(**kwargs)

    def get_label_config(self):
        """
        返回N天波动率标签

        Returns:
            fields: 标签表达式列表
            names: 标签名称列表
        """
        volatility_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [volatility_expr], ["LABEL0"]


class Alpha158_Volatility(Alpha158):
    """
    Alpha158 特征 + N天价格波动率标签

    继承 Alpha158 的所有技术指标特征，只修改标签为N天波动率
    """

    def __init__(self, volatility_window=2, **kwargs):
        """
        初始化波动率预测的 Alpha158 DataHandler

        Args:
            volatility_window: 波动率预测窗口（天数）
            **kwargs: 传递给父类的其他参数
        """
        self.volatility_window = volatility_window
        super().__init__(**kwargs)

    def get_label_config(self):
        """
        返回N天波动率标签

        Returns:
            fields: 标签表达式列表
            names: 标签名称列表
        """
        # 使用 Qlib 的表达式语法
        # Ref($close, -N)/Ref($close, -1) - 1 计算 N 天的收益率波动
        volatility_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"

        return [volatility_expr], ["LABEL0"]


class Alpha158_Volatility_TALib(DataHandlerLP):
    """
    Alpha158 特征 + TA-Lib 技术指标 + N天价格波动率标签

    在 Alpha158 基础上扩展了 TA-Lib 提供的技术指标:
    - 动量指标: RSI, MOM, ROC, WILLR, CCI, CMO, TRIX, PPO
    - MACD: MACD线, 信号线, 柱状图
    - 移动平均: EMA, DEMA, TEMA, KAMA, WMA
    - 布林带: 上轨, 中轨, 下轨
    - 波动率: ATR, NATR, True Range
    - 趋势: ADX, ADXR, +DI, -DI, Aroon
    - 成交量: OBV, AD, ADOSC, MFI
    - 随机指标: Stochastic K/D, StochRSI
    - 统计: STDDEV, VAR, Linear Regression
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
        """
        初始化包含 TA-Lib 指标的波动率预测 DataHandler

        Args:
            volatility_window: 波动率预测窗口（天数）
            **kwargs: 传递给父类的其他参数
        """
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
        获取特征配置，包含 Alpha158 + TA-Lib 指标

        Returns:
            fields: 特征表达式列表
            names: 特征名称列表
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
        """
        获取 TA-Lib 技术指标特征

        Returns:
            fields: TA-Lib 特征表达式列表
            names: TA-Lib 特征名称列表
        """
        fields = []
        names = []

        # 定义多个时间窗口
        windows = [5, 10, 14, 20, 30]

        # ==================== 动量指标 ====================
        # RSI - 相对强弱指数
        for w in [7, 14, 21]:
            fields.append(f"TALIB_RSI($close, {w})")
            names.append(f"TALIB_RSI{w}")

        # MOM - 动量
        for w in windows:
            fields.append(f"TALIB_MOM($close, {w})/$close")
            names.append(f"TALIB_MOM{w}")

        # ROC - 变化率
        for w in windows:
            fields.append(f"TALIB_ROC($close, {w})")
            names.append(f"TALIB_ROC{w}")

        # CMO - Chande 动量震荡
        for w in [7, 14, 21]:
            fields.append(f"TALIB_CMO($close, {w})")
            names.append(f"TALIB_CMO{w}")

        # WILLR - Williams %R
        for w in [7, 14, 21]:
            fields.append(f"TALIB_WILLR($high, $low, $close, {w})")
            names.append(f"TALIB_WILLR{w}")

        # CCI - 商品通道指数
        for w in [7, 14, 20]:
            fields.append(f"TALIB_CCI($high, $low, $close, {w})")
            names.append(f"TALIB_CCI{w}")

        # TRIX - 三重指数平滑移动平均
        for w in [12, 20, 30]:
            fields.append(f"TALIB_TRIX($close, {w})")
            names.append(f"TALIB_TRIX{w}")

        # PPO - 价格震荡百分比
        fields.append("TALIB_PPO($close, 12, 26)")
        names.append("TALIB_PPO_12_26")
        fields.append("TALIB_PPO($close, 5, 10)")
        names.append("TALIB_PPO_5_10")

        # ==================== MACD 指标 ====================
        # 标准 MACD (12, 26, 9)
        fields.append("TALIB_MACD_MACD($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD")
        fields.append("TALIB_MACD_SIGNAL($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD_SIGNAL")
        fields.append("TALIB_MACD_HIST($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD_HIST")

        # 快速 MACD (5, 10, 5)
        fields.append("TALIB_MACD_MACD($close, 5, 10, 5)/$close")
        names.append("TALIB_MACD_FAST")
        fields.append("TALIB_MACD_HIST($close, 5, 10, 5)/$close")
        names.append("TALIB_MACD_HIST_FAST")

        # ==================== 移动平均 ====================
        # EMA vs 价格
        for w in [5, 10, 20, 30, 60]:
            fields.append(f"TALIB_EMA($close, {w})/$close")
            names.append(f"TALIB_EMA{w}")

        # DEMA - 双重指数移动平均
        for w in [10, 20, 30]:
            fields.append(f"TALIB_DEMA($close, {w})/$close")
            names.append(f"TALIB_DEMA{w}")

        # TEMA - 三重指数移动平均
        for w in [10, 20, 30]:
            fields.append(f"TALIB_TEMA($close, {w})/$close")
            names.append(f"TALIB_TEMA{w}")

        # KAMA - 考夫曼自适应移动平均
        for w in [10, 20, 30]:
            fields.append(f"TALIB_KAMA($close, {w})/$close")
            names.append(f"TALIB_KAMA{w}")

        # WMA - 加权移动平均
        for w in [10, 20, 30]:
            fields.append(f"TALIB_WMA($close, {w})/$close")
            names.append(f"TALIB_WMA{w}")

        # ==================== 布林带 ====================
        for w in [10, 20, 30]:
            # 布林带位置 (价格相对于布林带的位置)
            fields.append(f"(TALIB_BBANDS_UPPER($close, {w}, 2) - $close)/$close")
            names.append(f"TALIB_BB_UPPER_DIST{w}")
            fields.append(f"($close - TALIB_BBANDS_LOWER($close, {w}, 2))/$close")
            names.append(f"TALIB_BB_LOWER_DIST{w}")
            # 布林带宽度
            fields.append(f"(TALIB_BBANDS_UPPER($close, {w}, 2) - TALIB_BBANDS_LOWER($close, {w}, 2))/$close")
            names.append(f"TALIB_BB_WIDTH{w}")

        # ==================== 波动率指标 ====================
        # ATR - 平均真实范围
        for w in [7, 14, 21]:
            fields.append(f"TALIB_ATR($high, $low, $close, {w})/$close")
            names.append(f"TALIB_ATR{w}")

        # NATR - 标准化平均真实范围
        for w in [7, 14, 21]:
            fields.append(f"TALIB_NATR($high, $low, $close, {w})")
            names.append(f"TALIB_NATR{w}")

        # True Range
        fields.append("TALIB_TRANGE($high, $low, $close)/$close")
        names.append("TALIB_TRANGE")

        # ==================== 趋势指标 ====================
        # ADX - 平均趋向指数
        for w in [7, 14, 21]:
            fields.append(f"TALIB_ADX($high, $low, $close, {w})")
            names.append(f"TALIB_ADX{w}")

        # ADXR - ADX 评级
        for w in [7, 14, 21]:
            fields.append(f"TALIB_ADXR($high, $low, $close, {w})")
            names.append(f"TALIB_ADXR{w}")

        # +DI / -DI - 方向指标
        for w in [7, 14, 21]:
            fields.append(f"TALIB_PLUS_DI($high, $low, $close, {w})")
            names.append(f"TALIB_PLUS_DI{w}")
            fields.append(f"TALIB_MINUS_DI($high, $low, $close, {w})")
            names.append(f"TALIB_MINUS_DI{w}")
            # DI 差值
            fields.append(f"TALIB_PLUS_DI($high, $low, $close, {w}) - TALIB_MINUS_DI($high, $low, $close, {w})")
            names.append(f"TALIB_DI_DIFF{w}")

        # Aroon 指标
        for w in [14, 25]:
            fields.append(f"TALIB_AROON_UP($high, $low, {w})")
            names.append(f"TALIB_AROON_UP{w}")
            fields.append(f"TALIB_AROON_DOWN($high, $low, {w})")
            names.append(f"TALIB_AROON_DOWN{w}")
            fields.append(f"TALIB_AROONOSC($high, $low, {w})")
            names.append(f"TALIB_AROONOSC{w}")

        # ==================== 成交量指标 ====================
        # OBV - 能量潮 (标准化)
        fields.append("TALIB_OBV($close, $volume)/($volume+1e-12)")
        names.append("TALIB_OBV")

        # AD - 累积/派发线 (标准化)
        fields.append("TALIB_AD($high, $low, $close, $volume)/($volume+1e-12)")
        names.append("TALIB_AD")

        # ADOSC - AD 震荡器
        fields.append("TALIB_ADOSC($high, $low, $close, $volume, 3, 10)/($volume+1e-12)")
        names.append("TALIB_ADOSC")

        # MFI - 资金流量指数
        for w in [7, 14, 21]:
            fields.append(f"TALIB_MFI($high, $low, $close, $volume, {w})")
            names.append(f"TALIB_MFI{w}")

        # ==================== 随机指标 ====================
        # Stochastic
        for fk, sk, sd in [(5, 3, 3), (14, 3, 3), (21, 5, 5)]:
            fields.append(f"TALIB_STOCH_K($high, $low, $close, {fk}, {sk}, {sd})")
            names.append(f"TALIB_STOCH_K_{fk}_{sk}_{sd}")
            fields.append(f"TALIB_STOCH_D($high, $low, $close, {fk}, {sk}, {sd})")
            names.append(f"TALIB_STOCH_D_{fk}_{sk}_{sd}")

        # StochRSI
        for tp in [14, 21]:
            fields.append(f"TALIB_STOCHRSI_K($close, {tp}, 5, 3)")
            names.append(f"TALIB_STOCHRSI_K{tp}")
            fields.append(f"TALIB_STOCHRSI_D($close, {tp}, 5, 3)")
            names.append(f"TALIB_STOCHRSI_D{tp}")

        # ==================== 统计指标 ====================
        # STDDEV - 标准差
        for w in [5, 10, 20, 30]:
            fields.append(f"TALIB_STDDEV($close, {w}, 1)/$close")
            names.append(f"TALIB_STDDEV{w}")

        # VAR - 方差
        for w in [5, 10, 20]:
            fields.append(f"TALIB_VAR($close, {w}, 1)/($close*$close)")
            names.append(f"TALIB_VAR{w}")

        # Linear Regression
        for w in [10, 20, 30]:
            fields.append(f"TALIB_LINEARREG($close, {w})/$close")
            names.append(f"TALIB_LINEARREG{w}")
            fields.append(f"TALIB_LINEARREG_SLOPE($close, {w})")
            names.append(f"TALIB_LINEARREG_SLOPE{w}")
            fields.append(f"TALIB_LINEARREG_ANGLE($close, {w})")
            names.append(f"TALIB_LINEARREG_ANGLE{w}")

        # TSF - 时间序列预测
        for w in [10, 20, 30]:
            fields.append(f"TALIB_TSF($close, {w})/$close")
            names.append(f"TALIB_TSF{w}")

        return fields, names

    def get_label_config(self):
        """
        返回N天波动率标签

        Returns:
            fields: 标签表达式列表
            names: 标签名称列表
        """
        volatility_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [volatility_expr], ["LABEL0"]


class Alpha158_Direction_TALib(DataHandlerLP):
    """
    Alpha158 特征 + TA-Lib 技术指标 + N天涨跌方向标签（二分类）

    标签定义:
    - 1: N天后价格上涨 (Ref($close, -N) > $close)
    - 0: N天后价格下跌或持平
    """

    def __init__(
        self,
        prediction_days=1,
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
        """
        初始化涨跌方向预测的 DataHandler

        Args:
            prediction_days: 预测天数（1表示预测明天涨跌）
            **kwargs: 传递给父类的其他参数
        """
        self.prediction_days = prediction_days

        from qlib.contrib.data.handler import check_transform_proc

        # 对于二分类标签，使用自定义的 learn_processors
        # 只对特征进行标准化，不对标签进行标准化
        if learn_processors is None:
            learn_processors = [
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature"}},  # 只标准化特征
            ]

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
        """复用 Alpha158_Volatility_TALib 的特征配置"""
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        fields, names = Alpha158.get_feature_config(conf)

        # 添加 TA-Lib 指标（复用现有方法）
        talib_fields, talib_names = Alpha158_Volatility_TALib._get_talib_features(self)
        fields.extend(talib_fields)
        names.extend(talib_names)

        return fields, names

    def _get_talib_features(self):
        """复用 Alpha158_Volatility_TALib 的 TA-Lib 特征"""
        return Alpha158_Volatility_TALib._get_talib_features(self)

    def get_label_config(self):
        """
        返回N天涨跌方向标签（二分类）

        Returns:
            fields: 标签表达式列表
            names: 标签名称列表
        """
        # 二分类标签: 1 = 上涨, 0 = 下跌/持平
        # Ref($close, -N) > $close 返回 True/False，转换为 1/0
        # 使用 If 表达式：If(condition, 1, 0)
        direction_expr = f"If(Ref($close, -{self.prediction_days}) > $close, 1, 0)"

        return [direction_expr], ["LABEL0"]