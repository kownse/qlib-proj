"""
TA-Lib Custom Operators for Qlib

This module provides custom Qlib expression operators that wrap TA-Lib technical indicators.
These operators can be used in Qlib's expression syntax after registration via qlib.init(custom_ops=...).

Usage:
    import qlib
    from talib_ops import TALIB_OPS
    qlib.init(provider_uri="...", custom_ops=TALIB_OPS)

Then use in expressions like:
    "TALIB_RSI($close, 14)"
    "TALIB_MACD_MACD($close, 12, 26, 9)"

Note:
    TA-Lib C library has known memory safety issues when used with multiprocessing.
    This module includes thread safety measures to mitigate these issues.
"""

import os
import threading

# ============================================================================
# 关键: 在导入 TA-Lib 之前设置环境变量，避免多线程内存冲突
# ============================================================================
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import pandas as pd

try:
    import talib
except ImportError:
    raise ImportError("TA-Lib is required. Install with: pip install TA-Lib")

# 全局锁，用于保护 TA-Lib 调用的线程安全
_talib_lock = threading.Lock()

from qlib.data.base import Expression, ExpressionOps


def _safe_talib_call(func, *args, **kwargs):
    """
    线程安全的 TA-Lib 调用包装器

    TA-Lib C 库在多线程环境下可能存在内存安全问题，
    使用全局锁确保同一时间只有一个线程调用 TA-Lib 函数。
    """
    with _talib_lock:
        return func(*args, **kwargs)


class TALibOperator(ExpressionOps):
    """Base class for TA-Lib operators"""

    def __init__(self, feature, *args):
        self.feature = feature
        self.args = args

    def __str__(self):
        args_str = ", ".join(str(a) for a in self.args)
        if args_str:
            return f"{type(self).__name__}({self.feature}, {args_str})"
        return f"{type(self).__name__}({self.feature})"

    def get_longest_back_rolling(self):
        return self.feature.get_longest_back_rolling() + self._get_window() - 1

    def get_extended_window_size(self):
        lft_etd, rght_etd = self.feature.get_extended_window_size()
        lft_etd = max(lft_etd + self._get_window() - 1, lft_etd)
        return lft_etd, rght_etd

    def _get_window(self):
        """Override in subclass to return the lookback window"""
        return 30  # Default window


class TALibMultiInputOperator(ExpressionOps):
    """Base class for TA-Lib operators requiring multiple inputs (high, low, close, etc.)"""

    def __init__(self, *features_and_args):
        # Subclasses define how many features they need
        pass

    def get_longest_back_rolling(self):
        max_br = 0
        for f in self._get_features():
            if isinstance(f, Expression):
                max_br = max(max_br, f.get_longest_back_rolling())
        return max_br + self._get_window() - 1

    def get_extended_window_size(self):
        lft_etd, rght_etd = 0, 0
        for f in self._get_features():
            if isinstance(f, Expression):
                l, r = f.get_extended_window_size()
                lft_etd = max(lft_etd, l)
                rght_etd = max(rght_etd, r)
        lft_etd = max(lft_etd + self._get_window() - 1, lft_etd)
        return lft_etd, rght_etd

    def _get_features(self):
        """Override in subclass to return list of feature inputs"""
        return []

    def _get_window(self):
        """Override in subclass to return the lookback window"""
        return 30


# ==================== Momentum Indicators ====================

class TALIB_RSI(TALibOperator):
    """Relative Strength Index

    Usage: TALIB_RSI($close, 14)
    """

    def __init__(self, feature, timeperiod=14):
        super().__init__(feature, timeperiod)
        self.timeperiod = int(timeperiod)

    def _get_window(self):
        return self.timeperiod + 1

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.RSI, series.values.astype(np.float64), timeperiod=self.timeperiod)
        return pd.Series(result, index=series.index)


class TALIB_MOM(TALibOperator):
    """Momentum

    Usage: TALIB_MOM($close, 10)
    """

    def __init__(self, feature, timeperiod=10):
        super().__init__(feature, timeperiod)
        self.timeperiod = int(timeperiod)

    def _get_window(self):
        return self.timeperiod + 1

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.MOM, series.values.astype(np.float64), timeperiod=self.timeperiod)
        return pd.Series(result, index=series.index)


class TALIB_ROC(TALibOperator):
    """Rate of Change

    Usage: TALIB_ROC($close, 10)
    """

    def __init__(self, feature, timeperiod=10):
        super().__init__(feature, timeperiod)
        self.timeperiod = int(timeperiod)

    def _get_window(self):
        return self.timeperiod + 1

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.ROC, series.values.astype(np.float64), timeperiod=self.timeperiod)
        return pd.Series(result, index=series.index)


class TALIB_ROCP(TALibOperator):
    """Rate of Change Percentage

    Usage: TALIB_ROCP($close, 10)
    """

    def __init__(self, feature, timeperiod=10):
        super().__init__(feature, timeperiod)
        self.timeperiod = int(timeperiod)

    def _get_window(self):
        return self.timeperiod + 1

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.ROCP, series.values.astype(np.float64), timeperiod=self.timeperiod)
        return pd.Series(result, index=series.index)


class TALIB_WILLR(TALibMultiInputOperator):
    """Williams %R

    Usage: TALIB_WILLR($high, $low, $close, 14)
    """

    def __init__(self, high, low, close, timeperiod=14):
        self.high = high
        self.low = low
        self.close = close
        self.timeperiod = int(timeperiod)

    def __str__(self):
        return f"TALIB_WILLR({self.high}, {self.low}, {self.close}, {self.timeperiod})"

    def _get_features(self):
        return [self.high, self.low, self.close]

    def _get_window(self):
        return self.timeperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        high = self.high.load(instrument, start_index, end_index, *args)
        low = self.low.load(instrument, start_index, end_index, *args)
        close = self.close.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.WILLR,
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            timeperiod=self.timeperiod
        )
        return pd.Series(result, index=close.index)


class TALIB_CCI(TALibMultiInputOperator):
    """Commodity Channel Index

    Usage: TALIB_CCI($high, $low, $close, 14)
    """

    def __init__(self, high, low, close, timeperiod=14):
        self.high = high
        self.low = low
        self.close = close
        self.timeperiod = int(timeperiod)

    def __str__(self):
        return f"TALIB_CCI({self.high}, {self.low}, {self.close}, {self.timeperiod})"

    def _get_features(self):
        return [self.high, self.low, self.close]

    def _get_window(self):
        return self.timeperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        high = self.high.load(instrument, start_index, end_index, *args)
        low = self.low.load(instrument, start_index, end_index, *args)
        close = self.close.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.CCI,
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            timeperiod=self.timeperiod
        )
        return pd.Series(result, index=close.index)


class TALIB_TRIX(TALibOperator):
    """Triple Exponential Average

    Usage: TALIB_TRIX($close, 30)
    """

    def __init__(self, feature, timeperiod=30):
        super().__init__(feature, timeperiod)
        self.timeperiod = int(timeperiod)

    def _get_window(self):
        return self.timeperiod * 3  # TRIX uses 3x EMA

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.TRIX, series.values.astype(np.float64), timeperiod=self.timeperiod)
        return pd.Series(result, index=series.index)


class TALIB_PPO(TALibOperator):
    """Percentage Price Oscillator

    Usage: TALIB_PPO($close, 12, 26)
    """

    def __init__(self, feature, fastperiod=12, slowperiod=26):
        super().__init__(feature, fastperiod, slowperiod)
        self.fastperiod = int(fastperiod)
        self.slowperiod = int(slowperiod)

    def _get_window(self):
        return self.slowperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.PPO,
            series.values.astype(np.float64),
            fastperiod=self.fastperiod,
            slowperiod=self.slowperiod
        )
        return pd.Series(result, index=series.index)


class TALIB_CMO(TALibOperator):
    """Chande Momentum Oscillator

    Usage: TALIB_CMO($close, 14)
    """

    def __init__(self, feature, timeperiod=14):
        super().__init__(feature, timeperiod)
        self.timeperiod = int(timeperiod)

    def _get_window(self):
        return self.timeperiod + 1

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.CMO, series.values.astype(np.float64), timeperiod=self.timeperiod)
        return pd.Series(result, index=series.index)


# ==================== MACD Indicators ====================

class TALIB_MACD_MACD(TALibOperator):
    """MACD Line (MACD - Signal)

    Usage: TALIB_MACD_MACD($close, 12, 26, 9)
    """

    def __init__(self, feature, fastperiod=12, slowperiod=26, signalperiod=9):
        super().__init__(feature, fastperiod, slowperiod, signalperiod)
        self.fastperiod = int(fastperiod)
        self.slowperiod = int(slowperiod)
        self.signalperiod = int(signalperiod)

    def _get_window(self):
        return self.slowperiod + self.signalperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        macd, signal, hist = _safe_talib_call(talib.MACD,
            series.values.astype(np.float64),
            fastperiod=self.fastperiod,
            slowperiod=self.slowperiod,
            signalperiod=self.signalperiod
        )
        return pd.Series(macd, index=series.index)


class TALIB_MACD_SIGNAL(TALibOperator):
    """MACD Signal Line

    Usage: TALIB_MACD_SIGNAL($close, 12, 26, 9)
    """

    def __init__(self, feature, fastperiod=12, slowperiod=26, signalperiod=9):
        super().__init__(feature, fastperiod, slowperiod, signalperiod)
        self.fastperiod = int(fastperiod)
        self.slowperiod = int(slowperiod)
        self.signalperiod = int(signalperiod)

    def _get_window(self):
        return self.slowperiod + self.signalperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        macd, signal, hist = _safe_talib_call(talib.MACD,
            series.values.astype(np.float64),
            fastperiod=self.fastperiod,
            slowperiod=self.slowperiod,
            signalperiod=self.signalperiod
        )
        return pd.Series(signal, index=series.index)


class TALIB_MACD_HIST(TALibOperator):
    """MACD Histogram

    Usage: TALIB_MACD_HIST($close, 12, 26, 9)
    """

    def __init__(self, feature, fastperiod=12, slowperiod=26, signalperiod=9):
        super().__init__(feature, fastperiod, slowperiod, signalperiod)
        self.fastperiod = int(fastperiod)
        self.slowperiod = int(slowperiod)
        self.signalperiod = int(signalperiod)

    def _get_window(self):
        return self.slowperiod + self.signalperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        macd, signal, hist = _safe_talib_call(talib.MACD,
            series.values.astype(np.float64),
            fastperiod=self.fastperiod,
            slowperiod=self.slowperiod,
            signalperiod=self.signalperiod
        )
        return pd.Series(hist, index=series.index)


# ==================== Moving Averages ====================

class TALIB_EMA(TALibOperator):
    """Exponential Moving Average

    Usage: TALIB_EMA($close, 20)
    """

    def __init__(self, feature, timeperiod=20):
        super().__init__(feature, timeperiod)
        self.timeperiod = int(timeperiod)

    def _get_window(self):
        return self.timeperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.EMA, series.values.astype(np.float64), timeperiod=self.timeperiod)
        return pd.Series(result, index=series.index)


class TALIB_SMA(TALibOperator):
    """Simple Moving Average

    Usage: TALIB_SMA($close, 20)
    """

    def __init__(self, feature, timeperiod=20):
        super().__init__(feature, timeperiod)
        self.timeperiod = int(timeperiod)

    def _get_window(self):
        return self.timeperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.SMA, series.values.astype(np.float64), timeperiod=self.timeperiod)
        return pd.Series(result, index=series.index)


class TALIB_WMA(TALibOperator):
    """Weighted Moving Average

    Usage: TALIB_WMA($close, 20)
    """

    def __init__(self, feature, timeperiod=20):
        super().__init__(feature, timeperiod)
        self.timeperiod = int(timeperiod)

    def _get_window(self):
        return self.timeperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.WMA, series.values.astype(np.float64), timeperiod=self.timeperiod)
        return pd.Series(result, index=series.index)


class TALIB_DEMA(TALibOperator):
    """Double Exponential Moving Average

    Usage: TALIB_DEMA($close, 20)
    """

    def __init__(self, feature, timeperiod=20):
        super().__init__(feature, timeperiod)
        self.timeperiod = int(timeperiod)

    def _get_window(self):
        return self.timeperiod * 2

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.DEMA, series.values.astype(np.float64), timeperiod=self.timeperiod)
        return pd.Series(result, index=series.index)


class TALIB_TEMA(TALibOperator):
    """Triple Exponential Moving Average

    Usage: TALIB_TEMA($close, 20)
    """

    def __init__(self, feature, timeperiod=20):
        super().__init__(feature, timeperiod)
        self.timeperiod = int(timeperiod)

    def _get_window(self):
        return self.timeperiod * 3

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.TEMA, series.values.astype(np.float64), timeperiod=self.timeperiod)
        return pd.Series(result, index=series.index)


class TALIB_KAMA(TALibOperator):
    """Kaufman Adaptive Moving Average

    Usage: TALIB_KAMA($close, 30)
    """

    def __init__(self, feature, timeperiod=30):
        super().__init__(feature, timeperiod)
        self.timeperiod = int(timeperiod)

    def _get_window(self):
        return self.timeperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.KAMA, series.values.astype(np.float64), timeperiod=self.timeperiod)
        return pd.Series(result, index=series.index)


# ==================== Bollinger Bands ====================

class TALIB_BBANDS_UPPER(TALibOperator):
    """Bollinger Bands - Upper Band

    Usage: TALIB_BBANDS_UPPER($close, 20, 2)
    """

    def __init__(self, feature, timeperiod=20, nbdevup=2):
        super().__init__(feature, timeperiod, nbdevup)
        self.timeperiod = int(timeperiod)
        self.nbdevup = float(nbdevup)

    def _get_window(self):
        return self.timeperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        upper, middle, lower = _safe_talib_call(talib.BBANDS,
            series.values.astype(np.float64),
            timeperiod=self.timeperiod,
            nbdevup=self.nbdevup,
            nbdevdn=self.nbdevup
        )
        return pd.Series(upper, index=series.index)


class TALIB_BBANDS_MIDDLE(TALibOperator):
    """Bollinger Bands - Middle Band (SMA)

    Usage: TALIB_BBANDS_MIDDLE($close, 20)
    """

    def __init__(self, feature, timeperiod=20):
        super().__init__(feature, timeperiod)
        self.timeperiod = int(timeperiod)

    def _get_window(self):
        return self.timeperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        upper, middle, lower = _safe_talib_call(talib.BBANDS,
            series.values.astype(np.float64),
            timeperiod=self.timeperiod
        )
        return pd.Series(middle, index=series.index)


class TALIB_BBANDS_LOWER(TALibOperator):
    """Bollinger Bands - Lower Band

    Usage: TALIB_BBANDS_LOWER($close, 20, 2)
    """

    def __init__(self, feature, timeperiod=20, nbdevdn=2):
        super().__init__(feature, timeperiod, nbdevdn)
        self.timeperiod = int(timeperiod)
        self.nbdevdn = float(nbdevdn)

    def _get_window(self):
        return self.timeperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        upper, middle, lower = _safe_talib_call(talib.BBANDS,
            series.values.astype(np.float64),
            timeperiod=self.timeperiod,
            nbdevup=self.nbdevdn,
            nbdevdn=self.nbdevdn
        )
        return pd.Series(lower, index=series.index)


# ==================== Volatility Indicators ====================

class TALIB_ATR(TALibMultiInputOperator):
    """Average True Range

    Usage: TALIB_ATR($high, $low, $close, 14)
    """

    def __init__(self, high, low, close, timeperiod=14):
        self.high = high
        self.low = low
        self.close = close
        self.timeperiod = int(timeperiod)

    def __str__(self):
        return f"TALIB_ATR({self.high}, {self.low}, {self.close}, {self.timeperiod})"

    def _get_features(self):
        return [self.high, self.low, self.close]

    def _get_window(self):
        return self.timeperiod + 1

    def _load_internal(self, instrument, start_index, end_index, *args):
        high = self.high.load(instrument, start_index, end_index, *args)
        low = self.low.load(instrument, start_index, end_index, *args)
        close = self.close.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.ATR,
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            timeperiod=self.timeperiod
        )
        return pd.Series(result, index=close.index)


class TALIB_NATR(TALibMultiInputOperator):
    """Normalized Average True Range

    Usage: TALIB_NATR($high, $low, $close, 14)
    """

    def __init__(self, high, low, close, timeperiod=14):
        self.high = high
        self.low = low
        self.close = close
        self.timeperiod = int(timeperiod)

    def __str__(self):
        return f"TALIB_NATR({self.high}, {self.low}, {self.close}, {self.timeperiod})"

    def _get_features(self):
        return [self.high, self.low, self.close]

    def _get_window(self):
        return self.timeperiod + 1

    def _load_internal(self, instrument, start_index, end_index, *args):
        high = self.high.load(instrument, start_index, end_index, *args)
        low = self.low.load(instrument, start_index, end_index, *args)
        close = self.close.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.NATR,
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            timeperiod=self.timeperiod
        )
        return pd.Series(result, index=close.index)


class TALIB_TRANGE(TALibMultiInputOperator):
    """True Range

    Usage: TALIB_TRANGE($high, $low, $close)
    """

    def __init__(self, high, low, close):
        self.high = high
        self.low = low
        self.close = close

    def __str__(self):
        return f"TALIB_TRANGE({self.high}, {self.low}, {self.close})"

    def _get_features(self):
        return [self.high, self.low, self.close]

    def _get_window(self):
        return 2

    def _load_internal(self, instrument, start_index, end_index, *args):
        high = self.high.load(instrument, start_index, end_index, *args)
        low = self.low.load(instrument, start_index, end_index, *args)
        close = self.close.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.TRANGE,
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64)
        )
        return pd.Series(result, index=close.index)


# ==================== Trend Indicators ====================

class TALIB_ADX(TALibMultiInputOperator):
    """Average Directional Index

    Usage: TALIB_ADX($high, $low, $close, 14)
    """

    def __init__(self, high, low, close, timeperiod=14):
        self.high = high
        self.low = low
        self.close = close
        self.timeperiod = int(timeperiod)

    def __str__(self):
        return f"TALIB_ADX({self.high}, {self.low}, {self.close}, {self.timeperiod})"

    def _get_features(self):
        return [self.high, self.low, self.close]

    def _get_window(self):
        return self.timeperiod * 2

    def _load_internal(self, instrument, start_index, end_index, *args):
        high = self.high.load(instrument, start_index, end_index, *args)
        low = self.low.load(instrument, start_index, end_index, *args)
        close = self.close.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.ADX,
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            timeperiod=self.timeperiod
        )
        return pd.Series(result, index=close.index)


class TALIB_ADXR(TALibMultiInputOperator):
    """Average Directional Movement Index Rating

    Usage: TALIB_ADXR($high, $low, $close, 14)
    """

    def __init__(self, high, low, close, timeperiod=14):
        self.high = high
        self.low = low
        self.close = close
        self.timeperiod = int(timeperiod)

    def __str__(self):
        return f"TALIB_ADXR({self.high}, {self.low}, {self.close}, {self.timeperiod})"

    def _get_features(self):
        return [self.high, self.low, self.close]

    def _get_window(self):
        return self.timeperiod * 3

    def _load_internal(self, instrument, start_index, end_index, *args):
        high = self.high.load(instrument, start_index, end_index, *args)
        low = self.low.load(instrument, start_index, end_index, *args)
        close = self.close.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.ADXR,
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            timeperiod=self.timeperiod
        )
        return pd.Series(result, index=close.index)


class TALIB_PLUS_DI(TALibMultiInputOperator):
    """Plus Directional Indicator

    Usage: TALIB_PLUS_DI($high, $low, $close, 14)
    """

    def __init__(self, high, low, close, timeperiod=14):
        self.high = high
        self.low = low
        self.close = close
        self.timeperiod = int(timeperiod)

    def __str__(self):
        return f"TALIB_PLUS_DI({self.high}, {self.low}, {self.close}, {self.timeperiod})"

    def _get_features(self):
        return [self.high, self.low, self.close]

    def _get_window(self):
        return self.timeperiod + 1

    def _load_internal(self, instrument, start_index, end_index, *args):
        high = self.high.load(instrument, start_index, end_index, *args)
        low = self.low.load(instrument, start_index, end_index, *args)
        close = self.close.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.PLUS_DI,
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            timeperiod=self.timeperiod
        )
        return pd.Series(result, index=close.index)


class TALIB_MINUS_DI(TALibMultiInputOperator):
    """Minus Directional Indicator

    Usage: TALIB_MINUS_DI($high, $low, $close, 14)
    """

    def __init__(self, high, low, close, timeperiod=14):
        self.high = high
        self.low = low
        self.close = close
        self.timeperiod = int(timeperiod)

    def __str__(self):
        return f"TALIB_MINUS_DI({self.high}, {self.low}, {self.close}, {self.timeperiod})"

    def _get_features(self):
        return [self.high, self.low, self.close]

    def _get_window(self):
        return self.timeperiod + 1

    def _load_internal(self, instrument, start_index, end_index, *args):
        high = self.high.load(instrument, start_index, end_index, *args)
        low = self.low.load(instrument, start_index, end_index, *args)
        close = self.close.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.MINUS_DI,
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            timeperiod=self.timeperiod
        )
        return pd.Series(result, index=close.index)


class TALIB_AROON_UP(TALibMultiInputOperator):
    """Aroon Up

    Usage: TALIB_AROON_UP($high, $low, 14)
    """

    def __init__(self, high, low, timeperiod=14):
        self.high = high
        self.low = low
        self.timeperiod = int(timeperiod)

    def __str__(self):
        return f"TALIB_AROON_UP({self.high}, {self.low}, {self.timeperiod})"

    def _get_features(self):
        return [self.high, self.low]

    def _get_window(self):
        return self.timeperiod + 1

    def _load_internal(self, instrument, start_index, end_index, *args):
        high = self.high.load(instrument, start_index, end_index, *args)
        low = self.low.load(instrument, start_index, end_index, *args)
        aroon_down, aroon_up = _safe_talib_call(talib.AROON,
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            timeperiod=self.timeperiod
        )
        return pd.Series(aroon_up, index=high.index)


class TALIB_AROON_DOWN(TALibMultiInputOperator):
    """Aroon Down

    Usage: TALIB_AROON_DOWN($high, $low, 14)
    """

    def __init__(self, high, low, timeperiod=14):
        self.high = high
        self.low = low
        self.timeperiod = int(timeperiod)

    def __str__(self):
        return f"TALIB_AROON_DOWN({self.high}, {self.low}, {self.timeperiod})"

    def _get_features(self):
        return [self.high, self.low]

    def _get_window(self):
        return self.timeperiod + 1

    def _load_internal(self, instrument, start_index, end_index, *args):
        high = self.high.load(instrument, start_index, end_index, *args)
        low = self.low.load(instrument, start_index, end_index, *args)
        aroon_down, aroon_up = _safe_talib_call(talib.AROON,
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            timeperiod=self.timeperiod
        )
        return pd.Series(aroon_down, index=high.index)


class TALIB_AROONOSC(TALibMultiInputOperator):
    """Aroon Oscillator

    Usage: TALIB_AROONOSC($high, $low, 14)
    """

    def __init__(self, high, low, timeperiod=14):
        self.high = high
        self.low = low
        self.timeperiod = int(timeperiod)

    def __str__(self):
        return f"TALIB_AROONOSC({self.high}, {self.low}, {self.timeperiod})"

    def _get_features(self):
        return [self.high, self.low]

    def _get_window(self):
        return self.timeperiod + 1

    def _load_internal(self, instrument, start_index, end_index, *args):
        high = self.high.load(instrument, start_index, end_index, *args)
        low = self.low.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.AROONOSC,
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            timeperiod=self.timeperiod
        )
        return pd.Series(result, index=high.index)


# ==================== Volume Indicators ====================

class TALIB_OBV(TALibMultiInputOperator):
    """On Balance Volume

    Usage: TALIB_OBV($close, $volume)
    """

    def __init__(self, close, volume):
        self.close = close
        self.volume = volume

    def __str__(self):
        return f"TALIB_OBV({self.close}, {self.volume})"

    def _get_features(self):
        return [self.close, self.volume]

    def _get_window(self):
        return 1

    def _load_internal(self, instrument, start_index, end_index, *args):
        close = self.close.load(instrument, start_index, end_index, *args)
        volume = self.volume.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.OBV,
            close.values.astype(np.float64),
            volume.values.astype(np.float64)
        )
        return pd.Series(result, index=close.index)


class TALIB_AD(TALibMultiInputOperator):
    """Chaikin A/D Line

    Usage: TALIB_AD($high, $low, $close, $volume)
    """

    def __init__(self, high, low, close, volume):
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def __str__(self):
        return f"TALIB_AD({self.high}, {self.low}, {self.close}, {self.volume})"

    def _get_features(self):
        return [self.high, self.low, self.close, self.volume]

    def _get_window(self):
        return 1

    def _load_internal(self, instrument, start_index, end_index, *args):
        high = self.high.load(instrument, start_index, end_index, *args)
        low = self.low.load(instrument, start_index, end_index, *args)
        close = self.close.load(instrument, start_index, end_index, *args)
        volume = self.volume.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.AD,
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            volume.values.astype(np.float64)
        )
        return pd.Series(result, index=close.index)


class TALIB_ADOSC(TALibMultiInputOperator):
    """Chaikin A/D Oscillator

    Usage: TALIB_ADOSC($high, $low, $close, $volume, 3, 10)
    """

    def __init__(self, high, low, close, volume, fastperiod=3, slowperiod=10):
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.fastperiod = int(fastperiod)
        self.slowperiod = int(slowperiod)

    def __str__(self):
        return f"TALIB_ADOSC({self.high}, {self.low}, {self.close}, {self.volume}, {self.fastperiod}, {self.slowperiod})"

    def _get_features(self):
        return [self.high, self.low, self.close, self.volume]

    def _get_window(self):
        return self.slowperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        high = self.high.load(instrument, start_index, end_index, *args)
        low = self.low.load(instrument, start_index, end_index, *args)
        close = self.close.load(instrument, start_index, end_index, *args)
        volume = self.volume.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.ADOSC,
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            volume.values.astype(np.float64),
            fastperiod=self.fastperiod,
            slowperiod=self.slowperiod
        )
        return pd.Series(result, index=close.index)


class TALIB_MFI(TALibMultiInputOperator):
    """Money Flow Index

    Usage: TALIB_MFI($high, $low, $close, $volume, 14)
    """

    def __init__(self, high, low, close, volume, timeperiod=14):
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.timeperiod = int(timeperiod)

    def __str__(self):
        return f"TALIB_MFI({self.high}, {self.low}, {self.close}, {self.volume}, {self.timeperiod})"

    def _get_features(self):
        return [self.high, self.low, self.close, self.volume]

    def _get_window(self):
        return self.timeperiod + 1

    def _load_internal(self, instrument, start_index, end_index, *args):
        high = self.high.load(instrument, start_index, end_index, *args)
        low = self.low.load(instrument, start_index, end_index, *args)
        close = self.close.load(instrument, start_index, end_index, *args)
        volume = self.volume.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.MFI,
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            volume.values.astype(np.float64),
            timeperiod=self.timeperiod
        )
        return pd.Series(result, index=close.index)


# ==================== Stochastic Indicators ====================

class TALIB_STOCH_K(TALibMultiInputOperator):
    """Stochastic %K

    Usage: TALIB_STOCH_K($high, $low, $close, 5, 3, 3)
    """

    def __init__(self, high, low, close, fastk_period=5, slowk_period=3, slowd_period=3):
        self.high = high
        self.low = low
        self.close = close
        self.fastk_period = int(fastk_period)
        self.slowk_period = int(slowk_period)
        self.slowd_period = int(slowd_period)

    def __str__(self):
        return f"TALIB_STOCH_K({self.high}, {self.low}, {self.close}, {self.fastk_period}, {self.slowk_period}, {self.slowd_period})"

    def _get_features(self):
        return [self.high, self.low, self.close]

    def _get_window(self):
        return self.fastk_period + self.slowk_period + self.slowd_period

    def _load_internal(self, instrument, start_index, end_index, *args):
        high = self.high.load(instrument, start_index, end_index, *args)
        low = self.low.load(instrument, start_index, end_index, *args)
        close = self.close.load(instrument, start_index, end_index, *args)
        slowk, slowd = _safe_talib_call(talib.STOCH,
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            fastk_period=self.fastk_period,
            slowk_period=self.slowk_period,
            slowd_period=self.slowd_period
        )
        return pd.Series(slowk, index=close.index)


class TALIB_STOCH_D(TALibMultiInputOperator):
    """Stochastic %D

    Usage: TALIB_STOCH_D($high, $low, $close, 5, 3, 3)
    """

    def __init__(self, high, low, close, fastk_period=5, slowk_period=3, slowd_period=3):
        self.high = high
        self.low = low
        self.close = close
        self.fastk_period = int(fastk_period)
        self.slowk_period = int(slowk_period)
        self.slowd_period = int(slowd_period)

    def __str__(self):
        return f"TALIB_STOCH_D({self.high}, {self.low}, {self.close}, {self.fastk_period}, {self.slowk_period}, {self.slowd_period})"

    def _get_features(self):
        return [self.high, self.low, self.close]

    def _get_window(self):
        return self.fastk_period + self.slowk_period + self.slowd_period

    def _load_internal(self, instrument, start_index, end_index, *args):
        high = self.high.load(instrument, start_index, end_index, *args)
        low = self.low.load(instrument, start_index, end_index, *args)
        close = self.close.load(instrument, start_index, end_index, *args)
        slowk, slowd = _safe_talib_call(talib.STOCH,
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            fastk_period=self.fastk_period,
            slowk_period=self.slowk_period,
            slowd_period=self.slowd_period
        )
        return pd.Series(slowd, index=close.index)


class TALIB_STOCHRSI_K(TALibOperator):
    """Stochastic RSI %K

    Usage: TALIB_STOCHRSI_K($close, 14, 5, 3)
    """

    def __init__(self, feature, timeperiod=14, fastk_period=5, fastd_period=3):
        super().__init__(feature, timeperiod, fastk_period, fastd_period)
        self.timeperiod = int(timeperiod)
        self.fastk_period = int(fastk_period)
        self.fastd_period = int(fastd_period)

    def _get_window(self):
        return self.timeperiod + self.fastk_period + self.fastd_period

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        fastk, fastd = _safe_talib_call(talib.STOCHRSI,
            series.values.astype(np.float64),
            timeperiod=self.timeperiod,
            fastk_period=self.fastk_period,
            fastd_period=self.fastd_period
        )
        return pd.Series(fastk, index=series.index)


class TALIB_STOCHRSI_D(TALibOperator):
    """Stochastic RSI %D

    Usage: TALIB_STOCHRSI_D($close, 14, 5, 3)
    """

    def __init__(self, feature, timeperiod=14, fastk_period=5, fastd_period=3):
        super().__init__(feature, timeperiod, fastk_period, fastd_period)
        self.timeperiod = int(timeperiod)
        self.fastk_period = int(fastk_period)
        self.fastd_period = int(fastd_period)

    def _get_window(self):
        return self.timeperiod + self.fastk_period + self.fastd_period

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        fastk, fastd = _safe_talib_call(talib.STOCHRSI,
            series.values.astype(np.float64),
            timeperiod=self.timeperiod,
            fastk_period=self.fastk_period,
            fastd_period=self.fastd_period
        )
        return pd.Series(fastd, index=series.index)


# ==================== Statistical Functions ====================

class TALIB_STDDEV(TALibOperator):
    """Standard Deviation

    Usage: TALIB_STDDEV($close, 20, 1)
    """

    def __init__(self, feature, timeperiod=20, nbdev=1):
        super().__init__(feature, timeperiod, nbdev)
        self.timeperiod = int(timeperiod)
        self.nbdev = float(nbdev)

    def _get_window(self):
        return self.timeperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.STDDEV,
            series.values.astype(np.float64),
            timeperiod=self.timeperiod,
            nbdev=self.nbdev
        )
        return pd.Series(result, index=series.index)


class TALIB_VAR(TALibOperator):
    """Variance

    Usage: TALIB_VAR($close, 20, 1)
    """

    def __init__(self, feature, timeperiod=20, nbdev=1):
        super().__init__(feature, timeperiod, nbdev)
        self.timeperiod = int(timeperiod)
        self.nbdev = float(nbdev)

    def _get_window(self):
        return self.timeperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.VAR,
            series.values.astype(np.float64),
            timeperiod=self.timeperiod,
            nbdev=self.nbdev
        )
        return pd.Series(result, index=series.index)


class TALIB_LINEARREG(TALibOperator):
    """Linear Regression

    Usage: TALIB_LINEARREG($close, 14)
    """

    def __init__(self, feature, timeperiod=14):
        super().__init__(feature, timeperiod)
        self.timeperiod = int(timeperiod)

    def _get_window(self):
        return self.timeperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.LINEARREG, series.values.astype(np.float64), timeperiod=self.timeperiod)
        return pd.Series(result, index=series.index)


class TALIB_LINEARREG_SLOPE(TALibOperator):
    """Linear Regression Slope

    Usage: TALIB_LINEARREG_SLOPE($close, 14)
    """

    def __init__(self, feature, timeperiod=14):
        super().__init__(feature, timeperiod)
        self.timeperiod = int(timeperiod)

    def _get_window(self):
        return self.timeperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.LINEARREG_SLOPE, series.values.astype(np.float64), timeperiod=self.timeperiod)
        return pd.Series(result, index=series.index)


class TALIB_LINEARREG_ANGLE(TALibOperator):
    """Linear Regression Angle

    Usage: TALIB_LINEARREG_ANGLE($close, 14)
    """

    def __init__(self, feature, timeperiod=14):
        super().__init__(feature, timeperiod)
        self.timeperiod = int(timeperiod)

    def _get_window(self):
        return self.timeperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.LINEARREG_ANGLE, series.values.astype(np.float64), timeperiod=self.timeperiod)
        return pd.Series(result, index=series.index)


class TALIB_TSF(TALibOperator):
    """Time Series Forecast

    Usage: TALIB_TSF($close, 14)
    """

    def __init__(self, feature, timeperiod=14):
        super().__init__(feature, timeperiod)
        self.timeperiod = int(timeperiod)

    def _get_window(self):
        return self.timeperiod

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = _safe_talib_call(talib.TSF, series.values.astype(np.float64), timeperiod=self.timeperiod)
        return pd.Series(result, index=series.index)


# ==================== List of all TA-Lib operators ====================

TALIB_OPS = [
    # Momentum Indicators
    TALIB_RSI,
    TALIB_MOM,
    TALIB_ROC,
    TALIB_ROCP,
    TALIB_WILLR,
    TALIB_CCI,
    TALIB_TRIX,
    TALIB_PPO,
    TALIB_CMO,
    # MACD
    TALIB_MACD_MACD,
    TALIB_MACD_SIGNAL,
    TALIB_MACD_HIST,
    # Moving Averages
    TALIB_EMA,
    TALIB_SMA,
    TALIB_WMA,
    TALIB_DEMA,
    TALIB_TEMA,
    TALIB_KAMA,
    # Bollinger Bands
    TALIB_BBANDS_UPPER,
    TALIB_BBANDS_MIDDLE,
    TALIB_BBANDS_LOWER,
    # Volatility
    TALIB_ATR,
    TALIB_NATR,
    TALIB_TRANGE,
    # Trend
    TALIB_ADX,
    TALIB_ADXR,
    TALIB_PLUS_DI,
    TALIB_MINUS_DI,
    TALIB_AROON_UP,
    TALIB_AROON_DOWN,
    TALIB_AROONOSC,
    # Volume
    TALIB_OBV,
    TALIB_AD,
    TALIB_ADOSC,
    TALIB_MFI,
    # Stochastic
    TALIB_STOCH_K,
    TALIB_STOCH_D,
    TALIB_STOCHRSI_K,
    TALIB_STOCHRSI_D,
    # Statistical
    TALIB_STDDEV,
    TALIB_VAR,
    TALIB_LINEARREG,
    TALIB_LINEARREG_SLOPE,
    TALIB_LINEARREG_ANGLE,
    TALIB_TSF,
]
