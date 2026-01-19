"""
扩展的 DataHandler 类，用于波动率预测

包含 Alpha158 特征 + TA-Lib 技术指标 + N天价格波动率标签
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
script_dir = Path(__file__).parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

from qlib.contrib.data.handler import Alpha158, Alpha360, check_transform_proc, _DEFAULT_LEARN_PROCESSORS, _DEFAULT_INFER_PROCESSORS
from qlib.data.dataset.handler import DataHandlerLP

# Import TA-Lib custom operators
from utils.talib_ops import TALIB_OPS


# ============================================================================
# Alpha180: 30天 OHLCV 数据 (6 × 30 = 180 特征)
# ============================================================================

class Alpha180DL:
    """
    Alpha180 数据加载器的特征配置。

    类似于 Alpha360，但使用 30 天而不是 60 天的历史数据。
    包含 CLOSE, OPEN, HIGH, LOW, VWAP, VOLUME 各 30 天历史。
    所有价格数据用当前 $close 标准化，成交量用当前 $volume 标准化。
    总共 180 个特征 (6 × 30)。
    """

    @staticmethod
    def get_feature_config():
        """
        获取 Alpha180 特征配置。

        Returns:
            tuple: (fields, names) 特征表达式和名称列表
        """
        fields = []
        names = []

        # CLOSE: 30 天历史收盘价，用当前收盘价标准化
        for i in range(29, 0, -1):
            fields += ["Ref($close, %d)/$close" % i]
            names += ["CLOSE%d" % i]
        fields += ["$close/$close"]
        names += ["CLOSE0"]

        # OPEN: 30 天历史开盘价，用当前收盘价标准化
        for i in range(29, 0, -1):
            fields += ["Ref($open, %d)/$close" % i]
            names += ["OPEN%d" % i]
        fields += ["$open/$close"]
        names += ["OPEN0"]

        # HIGH: 30 天历史最高价，用当前收盘价标准化
        for i in range(29, 0, -1):
            fields += ["Ref($high, %d)/$close" % i]
            names += ["HIGH%d" % i]
        fields += ["$high/$close"]
        names += ["HIGH0"]

        # LOW: 30 天历史最低价，用当前收盘价标准化
        for i in range(29, 0, -1):
            fields += ["Ref($low, %d)/$close" % i]
            names += ["LOW%d" % i]
        fields += ["$low/$close"]
        names += ["LOW0"]

        # VWAP: 30 天历史成交均价，用当前收盘价标准化
        for i in range(29, 0, -1):
            fields += ["Ref($vwap, %d)/$close" % i]
            names += ["VWAP%d" % i]
        fields += ["$vwap/$close"]
        names += ["VWAP0"]

        # VOLUME: 30 天历史成交量，用当前成交量标准化
        for i in range(29, 0, -1):
            fields += ["Ref($volume, %d)/($volume+1e-12)" % i]
            names += ["VOLUME%d" % i]
        fields += ["$volume/($volume+1e-12)"]
        names += ["VOLUME0"]

        return fields, names


class Alpha180(DataHandlerLP):
    """
    Alpha180 数据处理器。

    使用最近 30 天的价格和成交量数据，共 180 个特征。
    适用于需要较短历史窗口的模型（如某些 RNN/Transformer 模型）。

    特征说明：
    - CLOSE0-CLOSE29: 收盘价（用当前收盘价标准化）
    - OPEN0-OPEN29: 开盘价（用当前收盘价标准化）
    - HIGH0-HIGH29: 最高价（用当前收盘价标准化）
    - LOW0-LOW29: 最低价（用当前收盘价标准化）
    - VWAP0-VWAP29: 成交均价（用当前收盘价标准化）
    - VOLUME0-VOLUME29: 成交量（用当前成交量标准化）
    """

    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": Alpha180DL.get_feature_config(),
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
            learn_processors=learn_processors,
            infer_processors=infer_processors,
            **kwargs,
        )

    def get_label_config(self):
        """默认标签：2天收益率"""
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


class Alpha180_Volatility(Alpha180):
    """
    Alpha180 特征 + N天价格波动率标签。

    继承 Alpha180 的所有特征（30天的 OHLCV 数据），只修改标签为N天波动率。

    Alpha180 特征说明：
    - 使用最近 30 天的价格和成交量数据
    - 包含 CLOSE, OPEN, HIGH, LOW, VWAP, VOLUME 各 30 天历史
    - 所有价格数据用当前 $close 标准化
    - 成交量数据用当前 $volume 标准化
    - 总共 180 个特征 (6 × 30)
    """

    def __init__(self, volatility_window=2, **kwargs):
        """
        初始化波动率预测的 Alpha180 DataHandler。

        Args:
            volatility_window: 波动率预测窗口（天数）
            **kwargs: 传递给父类的其他参数
        """
        self.volatility_window = volatility_window
        super().__init__(**kwargs)

    def get_label_config(self):
        """
        返回N天波动率标签。

        Returns:
            fields: 标签表达式列表
            names: 标签名称列表
        """
        volatility_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [volatility_expr], ["LABEL0"]


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

        排除的问题特征：
        - VWAP0: US 股票数据中 VWAP 经常缺失 (100% NaN)
        - VMA5/10/20/30/60: 成交量移动平均，除以当前成交量会产生极端值
        - VSTD5/10/20/30/60: 成交量标准差，同样会产生极端值

        Returns:
            fields: 特征表达式列表
            names: 特征名称列表
        """
        # 获取 Alpha158 原始特征，排除问题特征
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                # 排除 VWAP，因为 US 股票数据中经常缺失
                "feature": ["OPEN", "HIGH", "LOW"],
            },
            "rolling": {
                # 排除 VMA 和 VSTD，因为它们会产生极端异常值
                "exclude": ["VMA", "VSTD"],
            },
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


class Alpha158_Volatility_TALib_Lite(DataHandlerLP):
    """
    Alpha158 特征 + 精选 TA-Lib 技术指标（Lite版本）+ N天价格波动率标签

    Lite版本使用较少的TA-Lib指标，避免大规模股票池时的内存问题。
    保留最重要的技术指标类别，每类只选择1-2个代表性指标。

    已排除的问题特征：
    - VWAP0: US 股票数据中 VWAP 经常缺失 (100% NaN)
    - VMA5/10/20/30/60: 成交量移动平均，除以当前成交量会产生极端值
    - VSTD5/10/20/30/60: 成交量标准差，同样会产生极端值

    总共约20个TA-Lib指标（vs 完整版的100+）：
    - 动量: RSI(14), MOM(10), ROC(10), CMO(14)
    - MACD: MACD, Signal, Hist
    - 移动平均: EMA(20), SMA(20)
    - 布林带: Upper, Lower, Width
    - 波动率: ATR(14), NATR(14)
    - 趋势: ADX(14), PLUS_DI(14), MINUS_DI(14)
    - 随机: STOCH_K, STOCH_D
    - 统计: STDDEV(20)
    """

    # 需要排除的 rolling 特征（会产生极端异常值）
    EXCLUDED_ROLLING_FEATURES = ['VMA', 'VSTD']

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
        初始化包含精选 TA-Lib 指标的波动率预测 DataHandler

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
        获取特征配置，包含 Alpha158（排除问题特征）+ 精选 TA-Lib 指标

        排除的特征：
        - VWAP0: US 股票 VWAP 数据缺失
        - VMA*: 成交量移动平均除以当前成交量会产生极端值
        - VSTD*: 成交量标准差除以当前成交量会产生极端值
        """
        # 自定义 Alpha158 配置，排除 VWAP 和问题 rolling 特征
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                # 排除 VWAP，因为 US 股票数据中经常缺失
                "feature": ["OPEN", "HIGH", "LOW"],
            },
            "rolling": {
                # 排除 VMA 和 VSTD，因为它们会产生极端异常值
                "exclude": ["VMA", "VSTD"],
            },
        }
        fields, names = Alpha158.get_feature_config(conf)

        # 添加精选 TA-Lib 指标
        talib_fields, talib_names = self._get_talib_features()
        fields.extend(talib_fields)
        names.extend(talib_names)

        return fields, names

    def _get_talib_features(self):
        """获取精选 TA-Lib 技术指标"""
        fields = []
        names = []

        # 动量指标 (4个)
        fields.append("TALIB_RSI($close, 14)")
        names.append("TALIB_RSI14")
        fields.append("TALIB_MOM($close, 10)/$close")
        names.append("TALIB_MOM10")
        fields.append("TALIB_ROC($close, 10)")
        names.append("TALIB_ROC10")
        fields.append("TALIB_CMO($close, 14)")
        names.append("TALIB_CMO14")

        # MACD (3个)
        fields.append("TALIB_MACD_MACD($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD")
        fields.append("TALIB_MACD_SIGNAL($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD_SIGNAL")
        fields.append("TALIB_MACD_HIST($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD_HIST")

        # 移动平均 (2个)
        fields.append("TALIB_EMA($close, 20)/$close")
        names.append("TALIB_EMA20")
        fields.append("TALIB_SMA($close, 20)/$close")
        names.append("TALIB_SMA20")

        # 布林带 (3个)
        fields.append("(TALIB_BBANDS_UPPER($close, 20, 2) - $close)/$close")
        names.append("TALIB_BB_UPPER_DIST")
        fields.append("($close - TALIB_BBANDS_LOWER($close, 20, 2))/$close")
        names.append("TALIB_BB_LOWER_DIST")
        fields.append("(TALIB_BBANDS_UPPER($close, 20, 2) - TALIB_BBANDS_LOWER($close, 20, 2))/$close")
        names.append("TALIB_BB_WIDTH")

        # 波动率 (2个)
        fields.append("TALIB_ATR($high, $low, $close, 14)/$close")
        names.append("TALIB_ATR14")
        fields.append("TALIB_NATR($high, $low, $close, 14)")
        names.append("TALIB_NATR14")

        # 趋势指标 (3个)
        fields.append("TALIB_ADX($high, $low, $close, 14)")
        names.append("TALIB_ADX14")
        fields.append("TALIB_PLUS_DI($high, $low, $close, 14)")
        names.append("TALIB_PLUS_DI14")
        fields.append("TALIB_MINUS_DI($high, $low, $close, 14)")
        names.append("TALIB_MINUS_DI14")

        # 随机指标 (2个)
        fields.append("TALIB_STOCH_K($high, $low, $close, 5, 3, 3)")
        names.append("TALIB_STOCH_K")
        fields.append("TALIB_STOCH_D($high, $low, $close, 5, 3, 3)")
        names.append("TALIB_STOCH_D")

        # 统计 (1个)
        fields.append("TALIB_STDDEV($close, 20, 1)/$close")
        names.append("TALIB_STDDEV20")

        return fields, names

    def get_label_config(self):
        """返回N天波动率标签"""
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