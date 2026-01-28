"""
特征选择共享工具

提供嵌套交叉验证特征选择的公共组件:
- CV fold 配置
- 候选特征池
- IC 计算
- 特征验证
- Checkpoint 管理
- Forward selection 框架
"""

import gc
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Callable, Optional

import numpy as np
import pandas as pd

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

# ============================================================================
# 项目路径
# ============================================================================

# 获取项目根目录
_current_file = Path(__file__)
SCRIPT_DIR = _current_file.parent.parent.parent  # scripts directory
PROJECT_ROOT = SCRIPT_DIR.parent


# ============================================================================
# 内层CV Folds (用于特征选择，不包含2025)
# ============================================================================

INNER_CV_FOLDS = [
    {
        'name': 'Inner Fold 1 (valid 2021)',
        'train_start': '2000-01-01',
        'train_end': '2020-12-31',
        'valid_start': '2021-01-01',
        'valid_end': '2021-12-31',
    },
    {
        'name': 'Inner Fold 2 (valid 2022)',
        'train_start': '2000-01-01',
        'train_end': '2021-12-31',
        'valid_start': '2022-01-01',
        'valid_end': '2022-12-31',
    },
    {
        'name': 'Inner Fold 3 (valid 2023)',
        'train_start': '2000-01-01',
        'train_end': '2022-12-31',
        'valid_start': '2023-01-01',
        'valid_end': '2023-12-31',
    },
    {
        'name': 'Inner Fold 4 (valid 2024)',
        'train_start': '2000-01-01',
        'train_end': '2023-12-31',
        'valid_start': '2024-01-01',
        'valid_end': '2024-12-31',
    },
]


# ============================================================================
# 候选特征池
# ============================================================================

# Alpha158 特征 (来自 Qlib)
ALPHA158_FEATURES = {
    # KBAR 特征
    "KMID": "($close-$open)/$open",
    "KLEN": "($high-$low)/$open",
    "KMID2": "($close-$open)/($high-$low+1e-12)",
    "KUP": "($high-Greater($open, $close))/$open",
    "KUP2": "($high-Greater($open, $close))/($high-$low+1e-12)",
    "KLOW": "(Less($open, $close)-$low)/$open",
    "KLOW2": "(Less($open, $close)-$low)/($high-$low+1e-12)",
    "KSFT": "(2*$close-$high-$low)/$open",
    "KSFT2": "(2*$close-$high-$low)/($high-$low+1e-12)",

    # Price 特征
    "OPEN0": "$open/$close",
    "HIGH0": "$high/$close",
    "LOW0": "$low/$close",

    # ROC 特征
    "ROC5": "Ref($close, 5)/$close",
    "ROC10": "Ref($close, 10)/$close",
    "ROC20": "Ref($close, 20)/$close",
    "ROC30": "Ref($close, 30)/$close",
    "ROC60": "Ref($close, 60)/$close",

    # MA 特征
    "MA5": "Mean($close, 5)/$close",
    "MA10": "Mean($close, 10)/$close",
    "MA20": "Mean($close, 20)/$close",
    "MA30": "Mean($close, 30)/$close",
    "MA60": "Mean($close, 60)/$close",

    # STD 特征
    "STD5": "Std($close, 5)/$close",
    "STD10": "Std($close, 10)/$close",
    "STD20": "Std($close, 20)/$close",
    "STD30": "Std($close, 30)/$close",
    "STD60": "Std($close, 60)/$close",

    # BETA 特征
    "BETA5": "Slope($close, 5)/$close",
    "BETA10": "Slope($close, 10)/$close",
    "BETA20": "Slope($close, 20)/$close",
    "BETA30": "Slope($close, 30)/$close",
    "BETA60": "Slope($close, 60)/$close",

    # RSQR 特征
    "RSQR5": "Rsquare($close, 5)",
    "RSQR10": "Rsquare($close, 10)",
    "RSQR20": "Rsquare($close, 20)",
    "RSQR30": "Rsquare($close, 30)",
    "RSQR60": "Rsquare($close, 60)",

    # RESI 特征
    "RESI5": "Resi($close, 5)/$close",
    "RESI10": "Resi($close, 10)/$close",
    "RESI20": "Resi($close, 20)/$close",
    "RESI30": "Resi($close, 30)/$close",
    "RESI60": "Resi($close, 60)/$close",

    # MAX 特征
    "MAX5": "Max($high, 5)/$close",
    "MAX10": "Max($high, 10)/$close",
    "MAX20": "Max($high, 20)/$close",
    "MAX30": "Max($high, 30)/$close",
    "MAX60": "Max($high, 60)/$close",

    # MIN 特征
    "MIN5": "Min($low, 5)/$close",
    "MIN10": "Min($low, 10)/$close",
    "MIN20": "Min($low, 20)/$close",
    "MIN30": "Min($low, 30)/$close",
    "MIN60": "Min($low, 60)/$close",

    # QTLU 特征
    "QTLU5": "Quantile($close, 5, 0.8)/$close",
    "QTLU10": "Quantile($close, 10, 0.8)/$close",
    "QTLU20": "Quantile($close, 20, 0.8)/$close",
    "QTLU30": "Quantile($close, 30, 0.8)/$close",
    "QTLU60": "Quantile($close, 60, 0.8)/$close",

    # QTLD 特征
    "QTLD5": "Quantile($close, 5, 0.2)/$close",
    "QTLD10": "Quantile($close, 10, 0.2)/$close",
    "QTLD20": "Quantile($close, 20, 0.2)/$close",
    "QTLD30": "Quantile($close, 30, 0.2)/$close",
    "QTLD60": "Quantile($close, 60, 0.2)/$close",

    # RANK 特征
    "RANK5": "Rank($close, 5)",
    "RANK10": "Rank($close, 10)",
    "RANK20": "Rank($close, 20)",
    "RANK30": "Rank($close, 30)",
    "RANK60": "Rank($close, 60)",

    # RSV 特征
    "RSV5": "($close-Min($low, 5))/(Max($high, 5)-Min($low, 5)+1e-12)",
    "RSV10": "($close-Min($low, 10))/(Max($high, 10)-Min($low, 10)+1e-12)",
    "RSV20": "($close-Min($low, 20))/(Max($high, 20)-Min($low, 20)+1e-12)",
    "RSV30": "($close-Min($low, 30))/(Max($high, 30)-Min($low, 30)+1e-12)",
    "RSV60": "($close-Min($low, 60))/(Max($high, 60)-Min($low, 60)+1e-12)",

    # IMAX 特征
    "IMAX5": "IdxMax($high, 5)/5",
    "IMAX10": "IdxMax($high, 10)/10",
    "IMAX20": "IdxMax($high, 20)/20",
    "IMAX30": "IdxMax($high, 30)/30",
    "IMAX60": "IdxMax($high, 60)/60",

    # IMIN 特征
    "IMIN5": "IdxMin($low, 5)/5",
    "IMIN10": "IdxMin($low, 10)/10",
    "IMIN20": "IdxMin($low, 20)/20",
    "IMIN30": "IdxMin($low, 30)/30",
    "IMIN60": "IdxMin($low, 60)/60",

    # IMXD 特征
    "IMXD5": "(IdxMax($high, 5)-IdxMin($low, 5))/5",
    "IMXD10": "(IdxMax($high, 10)-IdxMin($low, 10))/10",
    "IMXD20": "(IdxMax($high, 20)-IdxMin($low, 20))/20",
    "IMXD30": "(IdxMax($high, 30)-IdxMin($low, 30))/30",
    "IMXD60": "(IdxMax($high, 60)-IdxMin($low, 60))/60",

    # CORR 特征
    "CORR5": "Corr($close, Log($volume+1), 5)",
    "CORR10": "Corr($close, Log($volume+1), 10)",
    "CORR20": "Corr($close, Log($volume+1), 20)",
    "CORR30": "Corr($close, Log($volume+1), 30)",
    "CORR60": "Corr($close, Log($volume+1), 60)",

    # CORD 特征
    "CORD5": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)",
    "CORD10": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 10)",
    "CORD20": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 20)",
    "CORD30": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 30)",
    "CORD60": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 60)",

    # CNTP 特征
    "CNTP5": "Mean($close>Ref($close, 1), 5)",
    "CNTP10": "Mean($close>Ref($close, 1), 10)",
    "CNTP20": "Mean($close>Ref($close, 1), 20)",
    "CNTP30": "Mean($close>Ref($close, 1), 30)",
    "CNTP60": "Mean($close>Ref($close, 1), 60)",

    # CNTN 特征
    "CNTN5": "Mean($close<Ref($close, 1), 5)",
    "CNTN10": "Mean($close<Ref($close, 1), 10)",
    "CNTN20": "Mean($close<Ref($close, 1), 20)",
    "CNTN30": "Mean($close<Ref($close, 1), 30)",
    "CNTN60": "Mean($close<Ref($close, 1), 60)",

    # CNTD 特征
    "CNTD5": "Mean($close>Ref($close, 1), 5)-Mean($close<Ref($close, 1), 5)",
    "CNTD10": "Mean($close>Ref($close, 1), 10)-Mean($close<Ref($close, 1), 10)",
    "CNTD20": "Mean($close>Ref($close, 1), 20)-Mean($close<Ref($close, 1), 20)",
    "CNTD30": "Mean($close>Ref($close, 1), 30)-Mean($close<Ref($close, 1), 30)",
    "CNTD60": "Mean($close>Ref($close, 1), 60)-Mean($close<Ref($close, 1), 60)",

    # SUMP 特征
    "SUMP5": "Sum(Greater($close-Ref($close, 1), 0), 5)/(Sum(Abs($close-Ref($close, 1)), 5)+1e-12)",
    "SUMP10": "Sum(Greater($close-Ref($close, 1), 0), 10)/(Sum(Abs($close-Ref($close, 1)), 10)+1e-12)",
    "SUMP20": "Sum(Greater($close-Ref($close, 1), 0), 20)/(Sum(Abs($close-Ref($close, 1)), 20)+1e-12)",
    "SUMP30": "Sum(Greater($close-Ref($close, 1), 0), 30)/(Sum(Abs($close-Ref($close, 1)), 30)+1e-12)",
    "SUMP60": "Sum(Greater($close-Ref($close, 1), 0), 60)/(Sum(Abs($close-Ref($close, 1)), 60)+1e-12)",

    # SUMN 特征
    "SUMN5": "Sum(Greater(Ref($close, 1)-$close, 0), 5)/(Sum(Abs($close-Ref($close, 1)), 5)+1e-12)",
    "SUMN10": "Sum(Greater(Ref($close, 1)-$close, 0), 10)/(Sum(Abs($close-Ref($close, 1)), 10)+1e-12)",
    "SUMN20": "Sum(Greater(Ref($close, 1)-$close, 0), 20)/(Sum(Abs($close-Ref($close, 1)), 20)+1e-12)",
    "SUMN30": "Sum(Greater(Ref($close, 1)-$close, 0), 30)/(Sum(Abs($close-Ref($close, 1)), 30)+1e-12)",
    "SUMN60": "Sum(Greater(Ref($close, 1)-$close, 0), 60)/(Sum(Abs($close-Ref($close, 1)), 60)+1e-12)",

    # SUMD 特征
    "SUMD5": "(Sum(Greater($close-Ref($close, 1), 0), 5)-Sum(Greater(Ref($close, 1)-$close, 0), 5))/(Sum(Abs($close-Ref($close, 1)), 5)+1e-12)",
    "SUMD10": "(Sum(Greater($close-Ref($close, 1), 0), 10)-Sum(Greater(Ref($close, 1)-$close, 0), 10))/(Sum(Abs($close-Ref($close, 1)), 10)+1e-12)",
    "SUMD20": "(Sum(Greater($close-Ref($close, 1), 0), 20)-Sum(Greater(Ref($close, 1)-$close, 0), 20))/(Sum(Abs($close-Ref($close, 1)), 20)+1e-12)",
    "SUMD30": "(Sum(Greater($close-Ref($close, 1), 0), 30)-Sum(Greater(Ref($close, 1)-$close, 0), 30))/(Sum(Abs($close-Ref($close, 1)), 30)+1e-12)",
    "SUMD60": "(Sum(Greater($close-Ref($close, 1), 0), 60)-Sum(Greater(Ref($close, 1)-$close, 0), 60))/(Sum(Abs($close-Ref($close, 1)), 60)+1e-12)",

    # WVMA 特征
    "WVMA5": "Std(Abs($close/Ref($close, 1)-1)*$volume, 5)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 5)+1e-12)",
    "WVMA10": "Std(Abs($close/Ref($close, 1)-1)*$volume, 10)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 10)+1e-12)",
    "WVMA20": "Std(Abs($close/Ref($close, 1)-1)*$volume, 20)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 20)+1e-12)",
    "WVMA30": "Std(Abs($close/Ref($close, 1)-1)*$volume, 30)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 30)+1e-12)",
    "WVMA60": "Std(Abs($close/Ref($close, 1)-1)*$volume, 60)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 60)+1e-12)",

    # VSUMP 特征
    "VSUMP5": "Sum(Greater($volume-Ref($volume, 1), 0), 5)/(Sum(Abs($volume-Ref($volume, 1)), 5)+1e-12)",
    "VSUMP10": "Sum(Greater($volume-Ref($volume, 1), 0), 10)/(Sum(Abs($volume-Ref($volume, 1)), 10)+1e-12)",
    "VSUMP20": "Sum(Greater($volume-Ref($volume, 1), 0), 20)/(Sum(Abs($volume-Ref($volume, 1)), 20)+1e-12)",
    "VSUMP30": "Sum(Greater($volume-Ref($volume, 1), 0), 30)/(Sum(Abs($volume-Ref($volume, 1)), 30)+1e-12)",
    "VSUMP60": "Sum(Greater($volume-Ref($volume, 1), 0), 60)/(Sum(Abs($volume-Ref($volume, 1)), 60)+1e-12)",

    # VSUMN 特征
    "VSUMN5": "Sum(Greater(Ref($volume, 1)-$volume, 0), 5)/(Sum(Abs($volume-Ref($volume, 1)), 5)+1e-12)",
    "VSUMN10": "Sum(Greater(Ref($volume, 1)-$volume, 0), 10)/(Sum(Abs($volume-Ref($volume, 1)), 10)+1e-12)",
    "VSUMN20": "Sum(Greater(Ref($volume, 1)-$volume, 0), 20)/(Sum(Abs($volume-Ref($volume, 1)), 20)+1e-12)",
    "VSUMN30": "Sum(Greater(Ref($volume, 1)-$volume, 0), 30)/(Sum(Abs($volume-Ref($volume, 1)), 30)+1e-12)",
    "VSUMN60": "Sum(Greater(Ref($volume, 1)-$volume, 0), 60)/(Sum(Abs($volume-Ref($volume, 1)), 60)+1e-12)",

    # VSUMD 特征
    "VSUMD5": "(Sum(Greater($volume-Ref($volume, 1), 0), 5)-Sum(Greater(Ref($volume, 1)-$volume, 0), 5))/(Sum(Abs($volume-Ref($volume, 1)), 5)+1e-12)",
    "VSUMD10": "(Sum(Greater($volume-Ref($volume, 1), 0), 10)-Sum(Greater(Ref($volume, 1)-$volume, 0), 10))/(Sum(Abs($volume-Ref($volume, 1)), 10)+1e-12)",
    "VSUMD20": "(Sum(Greater($volume-Ref($volume, 1), 0), 20)-Sum(Greater(Ref($volume, 1)-$volume, 0), 20))/(Sum(Abs($volume-Ref($volume, 1)), 20)+1e-12)",
    "VSUMD30": "(Sum(Greater($volume-Ref($volume, 1), 0), 30)-Sum(Greater(Ref($volume, 1)-$volume, 0), 30))/(Sum(Abs($volume-Ref($volume, 1)), 30)+1e-12)",
    "VSUMD60": "(Sum(Greater($volume-Ref($volume, 1), 0), 60)-Sum(Greater(Ref($volume, 1)-$volume, 0), 60))/(Sum(Abs($volume-Ref($volume, 1)), 60)+1e-12)",
}

# 额外的自定义特征
EXTRA_STOCK_FEATURES = {
    # 动量质量
    "MOMENTUM_QUALITY": "($close/Ref($close, 20) - 1) / (Std($close/Ref($close,1)-1, 20) + 1e-12)",

    # 价格位置
    "PCT_FROM_52W_HIGH": "($close - Max($high, 252)) / (Max($high, 252) + 1e-12)",
    "PCT_FROM_52W_LOW": "($close - Min($low, 252)) / (Min($low, 252) + 1e-12)",
    "CLOSE_POSITION_60": "($close - Min($low, 60))/(Max($high, 60) - Min($low, 60) + 1e-12)",
    "CLOSE_POSITION_20": "($close - Min($low, 20))/(Max($high, 20) - Min($low, 20) + 1e-12)",

    # 均值回归
    "MA_RATIO_5_20": "Mean($close, 5)/Mean($close, 20)",
    "MA_RATIO_20_60": "Mean($close, 20)/Mean($close, 60)",

    # 成交量
    "VOLUME_RATIO_5_20": "Mean($volume, 5)/(Mean($volume, 20)+1e-12)",
    "VOLUME_STD_20": "Std($volume, 20)/(Mean($volume, 20)+1e-12)",

    # Drawdown
    "MAX_DRAWDOWN_20": "(Min($close, 20) - Max($high, 20)) / (Max($high, 20) + 1e-12)",
    "MAX_DRAWDOWN_60": "(Min($close, 60) - Max($high, 60)) / (Max($high, 60) + 1e-12)",

    # 价格比率
    "HIGH_LOW_RATIO": "$high/$low",
    "CLOSE_OPEN_RATIO": "$close/$open",
}

# TALib 技术指标特征
TALIB_STOCK_FEATURES = {
    "TALIB_RSI14": "TALIB_RSI($close, 14)",
    "TALIB_WILLR14": "TALIB_WILLR($high, $low, $close, 14)",
    "TALIB_NATR14": "TALIB_NATR($high, $low, $close, 14)",
    "TALIB_ATR14": "TALIB_ATR($high, $low, $close, 14)/$close",
    "TALIB_ADX14": "TALIB_ADX($high, $low, $close, 14)",
    "TALIB_CCI14": "TALIB_CCI($high, $low, $close, 14)",
    "TALIB_MFI14": "TALIB_MFI($high, $low, $close, $volume, 14)",
    "TALIB_MACD_HIST": "TALIB_MACD_HIST($close, 12, 26, 9)",
    "TALIB_BBANDS_UPPER": "TALIB_BBANDS_UPPER($close, 20, 2, 2)/$close",
    "TALIB_BBANDS_LOWER": "TALIB_BBANDS_LOWER($close, 20, 2, 2)/$close",
    "TALIB_MACD": "TALIB_MACD_MACD($close, 12, 26, 9)/$close",
    "TALIB_MACD_SIGNAL": "TALIB_MACD_SIGNAL($close, 12, 26, 9)/$close",
    "TALIB_EMA20": "TALIB_EMA($close, 20)/$close",
    "TALIB_SMA20": "TALIB_SMA($close, 20)/$close",
    "TALIB_PLUS_DI14": "TALIB_PLUS_DI($high, $low, $close, 14)",
    "TALIB_MINUS_DI14": "TALIB_MINUS_DI($high, $low, $close, 14)",
    "TALIB_STOCH_K": "TALIB_STOCH_K($high, $low, $close, 5, 3, 3)",
    "TALIB_STOCH_D": "TALIB_STOCH_D($high, $low, $close, 5, 3, 3)",
    "TALIB_CMO14": "TALIB_CMO($close, 14)",
    "TALIB_MOM10": "TALIB_MOM($close, 10)/$close",
    "TALIB_ROC10": "TALIB_ROC($close, 10)",
}

# 合并所有股票特征 (用于 CatBoost 等非时序模型)
ALL_STOCK_FEATURES = {
    **ALPHA158_FEATURES,
    **EXTRA_STOCK_FEATURES,
    **TALIB_STOCK_FEATURES,
}

# 所有可能的 TALib 特征 (用于 TCN 等时序模型，每个会扩展为多天历史)
ALL_TALIB_FEATURES = {
    # 动量指标
    "TALIB_RSI14": "TALIB_RSI($close, 14)",
    "TALIB_MOM10": "TALIB_MOM($close, 10)/$close",
    "TALIB_ROC10": "TALIB_ROC($close, 10)",
    "TALIB_CMO14": "TALIB_CMO($close, 14)",
    "TALIB_WILLR14": "TALIB_WILLR($high, $low, $close, 14)",

    # MACD
    "TALIB_MACD": "TALIB_MACD_MACD($close, 12, 26, 9)/$close",
    "TALIB_MACD_SIGNAL": "TALIB_MACD_SIGNAL($close, 12, 26, 9)/$close",
    "TALIB_MACD_HIST": "TALIB_MACD_HIST($close, 12, 26, 9)/$close",

    # 移动平均
    "TALIB_EMA20": "TALIB_EMA($close, 20)/$close",
    "TALIB_SMA20": "TALIB_SMA($close, 20)/$close",

    # 布林带
    "TALIB_BB_UPPER_DIST": "(TALIB_BBANDS_UPPER($close, 20, 2, 2) - $close)/$close",
    "TALIB_BB_LOWER_DIST": "($close - TALIB_BBANDS_LOWER($close, 20, 2, 2))/$close",
    "TALIB_BB_WIDTH": "(TALIB_BBANDS_UPPER($close, 20, 2, 2) - TALIB_BBANDS_LOWER($close, 20, 2, 2))/$close",

    # 波动率
    "TALIB_ATR14": "TALIB_ATR($high, $low, $close, 14)/$close",
    "TALIB_NATR14": "TALIB_NATR($high, $low, $close, 14)",

    # 趋势指标
    "TALIB_ADX14": "TALIB_ADX($high, $low, $close, 14)",
    "TALIB_PLUS_DI14": "TALIB_PLUS_DI($high, $low, $close, 14)",
    "TALIB_MINUS_DI14": "TALIB_MINUS_DI($high, $low, $close, 14)",

    # 随机指标
    "TALIB_STOCH_K": "TALIB_STOCH_K($high, $low, $close, 5, 3, 3)",
    "TALIB_STOCH_D": "TALIB_STOCH_D($high, $low, $close, 5, 3, 3)",

    # 统计
    "TALIB_STDDEV20": "TALIB_STDDEV($close, 20, 1)/$close",

    # CCI
    "TALIB_CCI14": "TALIB_CCI($high, $low, $close, 14)",

    # MFI
    "TALIB_MFI14": "TALIB_MFI($high, $low, $close, $volume, 14)",
}

# 所有可能的宏观特征
ALL_MACRO_FEATURES = [
    # VIX 相关
    "macro_vix_level",
    "macro_vix_zscore20",
    "macro_vix_regime",
    "macro_vix_pct_5d",
    "macro_vix_term_structure",
    # 信用/风险
    "macro_hy_spread_zscore",
    "macro_credit_stress",
    "macro_hyg_pct_5d",
    "macro_hyg_pct_20d",
    "macro_hyg_vs_lqd",
    # 利率/债券
    "macro_yield_curve",
    "macro_tlt_pct_5d",
    "macro_tlt_pct_20d",
    "macro_yield_10y",
    "macro_yield_2s10s",
    "macro_yield_inversion",
    # 商品
    "macro_gld_pct_5d",
    "macro_gld_pct_20d",
    "macro_uso_pct_5d",
    "macro_uso_pct_20d",
    # 美元
    "macro_uup_pct_5d",
    "macro_uup_pct_20d",
    # 市场
    "macro_spy_pct_5d",
    "macro_spy_pct_20d",
    "macro_spy_vol20",
    "macro_qqq_vs_spy",
    # 跨资产
    "macro_risk_on_off",
    "macro_market_stress",
    "macro_global_risk",
    "macro_eem_vs_spy",
]


# ============================================================================
# IC 计算
# ============================================================================

def compute_ic(pred: np.ndarray, label: np.ndarray, index: pd.MultiIndex) -> float:
    """
    计算 IC (Information Coefficient)

    Args:
        pred: 预测值数组
        label: 真实值数组
        index: 包含 datetime 级别的 MultiIndex

    Returns:
        float: 平均 IC
    """
    df = pd.DataFrame({'pred': pred, 'label': label}, index=index)
    ic_by_date = df.groupby(level='datetime').apply(
        lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
    )
    ic_by_date = ic_by_date.dropna()
    return ic_by_date.mean() if len(ic_by_date) > 0 else 0.0


# ============================================================================
# 特征验证
# ============================================================================

def validate_qlib_features(symbols: List[str], candidates: Dict[str, str]) -> Dict[str, str]:
    """
    验证 Qlib 特征表达式是否有效

    Args:
        symbols: 股票代码列表
        candidates: {name: expression} 候选特征字典

    Returns:
        有效的特征字典
    """
    from qlib.data import D

    test_start = "2024-01-01"
    test_end = "2024-01-10"
    test_symbols = symbols[:5] if len(symbols) > 5 else symbols

    valid = {}
    for name, expr in candidates.items():
        try:
            df = D.features(test_symbols, [expr], start_time=test_start, end_time=test_end)
            if df is not None and len(df) > 0 and df.notna().any().any():
                valid[name] = expr
        except Exception:
            pass

    return valid


def validate_macro_features(candidates: List[str], macro_path: Path = None) -> List[str]:
    """
    验证宏观特征是否存在于 parquet 文件中

    Args:
        candidates: 候选宏观特征名列表
        macro_path: macro 数据文件路径，默认使用项目路径

    Returns:
        存在的宏观特征列表
    """
    if macro_path is None:
        macro_path = PROJECT_ROOT / "my_data" / "macro_processed" / "macro_features.parquet"

    if not macro_path.exists():
        return []

    try:
        macro_df = pd.read_parquet(macro_path)
        available = set(macro_df.columns)
        return [m for m in candidates if m in available]
    except Exception:
        return []


def load_macro_data(macro_path: Path = None) -> Optional[pd.DataFrame]:
    """
    加载宏观数据

    Args:
        macro_path: macro 数据文件路径

    Returns:
        DataFrame 或 None
    """
    if macro_path is None:
        macro_path = PROJECT_ROOT / "my_data" / "macro_processed" / "macro_features.parquet"

    if not macro_path.exists():
        return None

    try:
        return pd.read_parquet(macro_path)
    except Exception as e:
        print(f"Warning: Failed to load macro data: {e}")
        return None


# ============================================================================
# Checkpoint 管理
# ============================================================================

def save_checkpoint(
    output_dir: Path,
    checkpoint_name: str,
    round_num: int,
    current_ic: float,
    current_features: Dict[str, Any],
    excluded_features: Set[str],
    history: List[Dict],
    extra_data: Dict = None,
):
    """
    保存 checkpoint

    Args:
        output_dir: 输出目录
        checkpoint_name: checkpoint 文件名 (不含扩展名)
        round_num: 当前轮次
        current_ic: 当前 IC
        current_features: 当前特征配置
        excluded_features: 已排除的特征
        history: 历史记录
        extra_data: 额外数据
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'round': round_num,
        'current_ic': current_ic,
        'excluded_features': list(excluded_features),
        'history': history,
        **current_features,
    }

    if extra_data:
        checkpoint.update(extra_data)

    checkpoint_file = output_dir / f"{checkpoint_name}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(checkpoint_path: Path) -> Dict:
    """
    加载 checkpoint

    Args:
        checkpoint_path: checkpoint 文件路径

    Returns:
        checkpoint 字典
    """
    with open(checkpoint_path, 'r') as f:
        return json.load(f)


def save_final_result(
    output_dir: Path,
    result_prefix: str,
    method: str,
    baseline_ic: float,
    final_ic: float,
    final_features: Dict[str, Any],
    excluded_features: Set[str],
    history: List[Dict],
    extra_data: Dict = None,
) -> Path:
    """
    保存最终结果

    Args:
        output_dir: 输出目录
        result_prefix: 结果文件前缀
        method: 方法名
        baseline_ic: 基线 IC
        final_ic: 最终 IC
        final_features: 最终特征配置
        excluded_features: 已排除的特征
        history: 历史记录
        extra_data: 额外数据

    Returns:
        结果文件路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result = {
        'timestamp': timestamp,
        'method': method,
        'baseline_ic': baseline_ic,
        'final_ic': final_ic,
        'excluded_features': list(excluded_features),
        'history': history,
        **final_features,
    }

    if extra_data:
        result.update(extra_data)

    result_file = output_dir / f"{result_prefix}_{timestamp}.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    return result_file


# ============================================================================
# 数据准备工具
# ============================================================================

def prepare_dataset_data(
    dataset: DatasetH,
    segment: str,
) -> Tuple[np.ndarray, np.ndarray, pd.MultiIndex]:
    """
    从 Dataset 准备数据

    Args:
        dataset: DatasetH 实例
        segment: 数据段名称 ("train", "valid", "test")

    Returns:
        (X, y, index) 元组
    """
    X = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
    X = X.fillna(0).replace([np.inf, -np.inf], 0).clip(-10, 10)

    y = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = y.fillna(0).values

    return X.values, y, X.index


# ============================================================================
# Forward Selection 框架
# ============================================================================

class ForwardSelectionBase(ABC):
    """
    Forward Selection 抽象基类

    子类需要实现:
    - prepare_fold_data(): 准备单个 fold 的数据
    - evaluate_feature_set(): 在内层 CV 上评估特征集
    - get_feature_counts(): 返回当前特征计数字典
    - add_feature(): 添加一个特征
    - get_current_features_dict(): 返回当前特征配置字典
    - cleanup_after_evaluation(): 评估后的清理工作
    """

    def __init__(
        self,
        symbols: List[str],
        nday: int = 5,
        max_features: int = 50,
        min_improvement: float = 0.0005,
        output_dir: Path = None,
        checkpoint_name: str = "forward_selection_checkpoint",
        result_prefix: str = "forward_selection",
        quiet: bool = False,
    ):
        self.symbols = symbols
        self.nday = nday
        self.max_features = max_features
        self.min_improvement = min_improvement
        self.output_dir = output_dir
        self.checkpoint_name = checkpoint_name
        self.result_prefix = result_prefix
        self.quiet = quiet

        self.excluded_features: Set[str] = set()
        self.history: List[Dict] = []
        self.current_ic: float = 0.0
        self.baseline_ic: float = 0.0

    @abstractmethod
    def prepare_fold_data(self, fold_config: Dict) -> Tuple:
        """准备单个 fold 的数据"""
        pass

    @abstractmethod
    def evaluate_feature_set(self) -> Tuple[float, List[float]]:
        """在内层 CV 上评估当前特征集，返回 (mean_ic, fold_ics)"""
        pass

    @abstractmethod
    def get_feature_counts(self) -> Dict[str, int]:
        """返回当前特征计数字典，如 {'stock': 10, 'macro': 5}"""
        pass

    @abstractmethod
    def add_feature(self, name: str, feature_type: str, expr: str = None):
        """添加一个特征"""
        pass

    @abstractmethod
    def get_current_features_dict(self) -> Dict[str, Any]:
        """返回当前特征配置字典，用于保存 checkpoint"""
        pass

    @abstractmethod
    def get_testable_candidates(self) -> Tuple[Dict[str, str], List[str]]:
        """
        返回可测试的候选特征

        Returns:
            (testable_stock_or_talib, testable_macro) 元组
        """
        pass

    @abstractmethod
    def test_feature(
        self, name: str, feature_type: str, expr: str = None
    ) -> Tuple[float, List[float]]:
        """
        测试添加一个特征后的 IC

        Returns:
            (ic, fold_ics) 元组
        """
        pass

    def cleanup_after_evaluation(self):
        """评估后的清理工作，子类可覆盖"""
        gc.collect()

    def run(
        self,
        excluded_features: Set[str] = None,
        method_name: str = "nested_cv_forward_selection",
    ) -> Tuple[Dict, List, Set]:
        """
        运行 forward selection

        Args:
            excluded_features: 初始排除的特征集
            method_name: 方法名（用于保存结果）

        Returns:
            (final_features_dict, history, excluded_features)
        """
        if excluded_features is not None:
            self.excluded_features = set(excluded_features)

        counts = self.get_feature_counts()
        counts_str = " + ".join([f"{v} {k}" for k, v in counts.items()])

        print("\n" + "=" * 70)
        print("NESTED CV FORWARD SELECTION")
        print("=" * 70)
        print(f"Current features: {counts_str}")
        print(f"Max features: {self.max_features}")
        print(f"Min improvement: {self.min_improvement}")
        print(f"Inner CV Folds: {len(INNER_CV_FOLDS)}")
        print("=" * 70)

        if self.excluded_features and not self.quiet:
            print(f"\nExcluded features from previous runs: {len(self.excluded_features)}")
            for f in sorted(self.excluded_features):
                print(f"  - {f}")

        # 基线评估
        print("\n[*] Evaluating baseline features...")
        self.baseline_ic, baseline_fold_ics = self.evaluate_feature_set()
        self.current_ic = self.baseline_ic

        print(f"    Baseline Inner CV IC: {self.baseline_ic:.4f}")
        if not self.quiet:
            print(f"    Fold ICs: {[f'{ic:.4f}' for ic in baseline_fold_ics]}")

        counts = self.get_feature_counts()
        self.history.append({
            'round': 0,
            'action': 'BASELINE',
            'feature': None,
            'type': None,
            'inner_cv_ic': self.baseline_ic,
            'fold_ics': baseline_fold_ics,
            'ic_change': 0,
            **{f'{k}_count': v for k, v in counts.items()},
        })

        round_num = 0

        # Forward selection loop
        while sum(self.get_feature_counts().values()) < self.max_features:
            round_num += 1

            testable_features, testable_macro = self.get_testable_candidates()

            if not testable_features and not testable_macro:
                print(f"\n[!] No more candidates to test. Stopping.")
                break

            counts = self.get_feature_counts()
            counts_str = " + ".join([f"{v} {k}" for k, v in counts.items()])
            print(f"\n[Round {round_num}] IC: {self.current_ic:.4f}, Features: {counts_str}")
            if not self.quiet:
                print(f"    Excluded: {len(self.excluded_features)}, Testing: {len(testable_features)} feature + {len(testable_macro)} macro")

            candidates = []
            newly_excluded = []

            # 测试特征类候选
            for name, expr in testable_features.items():
                for attempt in range(3):
                    try:
                        self.cleanup_after_evaluation()
                        ic, fold_ics = self.test_feature(name, 'feature', expr)
                        ic_change = ic - self.current_ic

                        candidates.append({
                            'name': name,
                            'type': 'feature',
                            'expr': expr,
                            'ic': ic,
                            'fold_ics': fold_ics,
                            'ic_change': ic_change,
                        })

                        if ic_change < 0:
                            self.excluded_features.add(name)
                            newly_excluded.append(name)
                            symbol = "X"
                            self._save_checkpoint(round_num)
                        else:
                            symbol = "+" if ic_change >= self.min_improvement else ""

                        if not self.quiet:
                            print(f"      +{name}: IC={ic:.4f} ({symbol}{ic_change:+.4f})")
                        break

                    except Exception as e:
                        if attempt < 2:
                            if not self.quiet:
                                print(f"      +{name}: Retry {attempt+1}/3 after error...")
                            self.cleanup_after_evaluation()
                        else:
                            print(f"      +{name}: ERROR - {e}")

            # 测试宏观特征候选
            for name in testable_macro:
                for attempt in range(3):
                    try:
                        self.cleanup_after_evaluation()
                        ic, fold_ics = self.test_feature(name, 'macro', None)
                        ic_change = ic - self.current_ic

                        candidates.append({
                            'name': name,
                            'type': 'macro',
                            'expr': None,
                            'ic': ic,
                            'fold_ics': fold_ics,
                            'ic_change': ic_change,
                        })

                        if ic_change < 0:
                            self.excluded_features.add(name)
                            newly_excluded.append(name)
                            symbol = "X"
                            self._save_checkpoint(round_num)
                        else:
                            symbol = "+" if ic_change >= self.min_improvement else ""

                        if not self.quiet:
                            print(f"      +{name}: IC={ic:.4f} ({symbol}{ic_change:+.4f})")
                        break

                    except Exception as e:
                        if attempt < 2:
                            if not self.quiet:
                                print(f"      +{name}: Retry {attempt+1}/3 after error...")
                            self.cleanup_after_evaluation()
                        else:
                            print(f"      +{name}: ERROR - {e}")

            # 打印本轮新排除的特征
            if newly_excluded and not self.quiet:
                print(f"\n    X Newly excluded ({len(newly_excluded)}):")
                for f in newly_excluded:
                    print(f"       - {f}")

            if not candidates:
                print("    No valid candidates, stopping.")
                break

            # 找到加入后 IC 提升最大的特征
            positive_candidates = [c for c in candidates if c['ic_change'] >= self.min_improvement]

            if not positive_candidates:
                print(f"\n[!] Stopping: No candidate improved IC by >= {self.min_improvement}")
                best_change = max(c['ic_change'] for c in candidates) if candidates else 0
                print(f"    Best improvement found: {best_change:+.4f}")
                break

            positive_candidates.sort(key=lambda x: x['ic_change'], reverse=True)
            best = positive_candidates[0]

            # 添加该特征
            self.add_feature(best['name'], best['type'], best.get('expr'))
            self.current_ic = best['ic']

            counts = self.get_feature_counts()
            self.history.append({
                'round': round_num,
                'action': 'ADD',
                'feature': best['name'],
                'type': best['type'],
                'inner_cv_ic': best['ic'],
                'fold_ics': best['fold_ics'],
                'ic_change': best['ic_change'],
                **{f'{k}_count': v for k, v in counts.items()},
            })

            counts_str = " + ".join([f"{v} {k}" for k, v in counts.items()])
            print(f"    + Added {best['name']}: IC={best['ic']:.4f} (+{best['ic_change']:.4f})")

            # 保存 checkpoint
            self._save_checkpoint(round_num)

        # 打印最终结果
        self._print_final_result()

        # 保存最终结果
        if self.output_dir:
            result_file = save_final_result(
                self.output_dir,
                self.result_prefix,
                method_name,
                self.baseline_ic,
                self.current_ic,
                self.get_current_features_dict(),
                self.excluded_features,
                self.history,
            )
            print(f"\nResults saved to: {result_file}")

        return self.get_current_features_dict(), self.history, self.excluded_features

    def _save_checkpoint(self, round_num: int):
        """保存 checkpoint"""
        if self.output_dir:
            save_checkpoint(
                self.output_dir,
                self.checkpoint_name,
                round_num,
                self.current_ic,
                self.get_current_features_dict(),
                self.excluded_features,
                self.history,
            )

    def _print_final_result(self):
        """打印最终结果"""
        print("\n" + "=" * 70)
        print("FORWARD SELECTION COMPLETE")
        print("=" * 70)

        counts = self.get_feature_counts()
        counts_str = " + ".join([f"{v} {k}" for k, v in counts.items()])
        print(f"Final features: {counts_str}")
        print(f"Baseline IC: {self.baseline_ic:.4f}")
        ic_diff = self.current_ic - self.baseline_ic
        print(f"Final IC:    {self.current_ic:.4f} ({'+' if ic_diff >= 0 else ''}{ic_diff:.4f})")

        print(f"\nExcluded Features ({len(self.excluded_features)}):")
        for name in sorted(self.excluded_features):
            print(f"  - {name}")


# ============================================================================
# 通用命令行参数
# ============================================================================

def add_common_args(parser):
    """添加通用命令行参数"""
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--max-features', type=int, default=50,
                        help='Maximum number of features')
    parser.add_argument('--min-improvement', type=float, default=0.0005,
                        help='Minimum IC improvement to add a feature')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--early-stop', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--no-countdown', action='store_true')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint file')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from a specific result file')
    return parser


def countdown(seconds: int = 3):
    """倒计时"""
    import time
    print("\nProceed with forward selection? (Press Ctrl+C to abort)")
    try:
        for i in range(seconds, 0, -1):
            print(f"  Starting in {i}...")
            time.sleep(1)
        return True
    except KeyboardInterrupt:
        print("\nAborted.")
        return False
