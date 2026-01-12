"""
共享配置和常量

包含所有模型训练脚本共用的配置、常量和 Handler 配置映射
"""

from pathlib import Path

# Import extended data handlers
from data.datahandler_ext import Alpha158_Volatility, Alpha158_Volatility_TALib, Alpha360_Volatility
from data.datahandler_news import Alpha158_Volatility_TALib_News
from data.datahandler_pandas import Alpha158_Volatility_Pandas, Alpha360_Volatility_Pandas


# ========== 路径配置 ==========
# scripts/models/common/config.py -> scripts/models/common -> scripts/models -> scripts -> qlib-proj
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # 项目根目录
QLIB_DATA_PATH = PROJECT_ROOT / "my_data" / "qlib_us"
NEWS_DATA_PATH = PROJECT_ROOT / "my_data" / "news_processed" / "news_features.parquet"
MODEL_SAVE_PATH = PROJECT_ROOT / "my_models"


# ========== 默认时间划分 ==========
DEFAULT_TIME_SPLITS = {
    'train_start': "2000-01-01",
    'train_end': "2022-12-31",
    'valid_start': "2023-01-01",
    'valid_end': "2023-12-31",
    'test_start': "2024-01-01",
    'test_end': "2025-12-31",
}

MAX_TRAIN_TIME_SPLITS = {
    'train_start': "2000-01-01",
    'train_end': "2025-09-30",
    'valid_start': "2025-10-01",
    'valid_end': "2025-12-31",
    'test_start': "2025-10-01",
    'test_end': "2025-12-31",
}


# ========== Handler 配置映射 ==========
HANDLER_CONFIG = {
    'alpha158': {
        'class': Alpha158_Volatility,
        'description': 'Alpha158 (158 technical indicators)',
        'use_talib': False,
    },
    'alpha360': {
        'class': Alpha360_Volatility,
        'description': 'Alpha360 (360 features - 60 days OHLCV)',
        'use_talib': False,
    },
    'alpha158-talib': {
        'class': Alpha158_Volatility_TALib,
        'description': 'Alpha158 + TA-Lib (~300+ technical indicators)',
        'use_talib': True,
    },
    'alpha158-pandas': {
        'class': Alpha158_Volatility_Pandas,
        'description': 'Alpha158 + Pandas indicators (no TA-Lib)',
        'use_talib': False,
    },
    'alpha360-pandas': {
        'class': Alpha360_Volatility_Pandas,
        'description': 'Alpha360 + Pandas (no TA-Lib, 360 features)',
        'use_talib': False,
    },
    'alpha158-news': {
        'class': Alpha158_Volatility_TALib_News,
        'description': 'Alpha158 + TA-Lib + News features',
        'use_talib': True,
    },
}


def get_handler_epilog():
    """返回命令行帮助的 handler 说明"""
    return """
Handler choices:
  alpha158        Alpha158 features (158 technical indicators) [default]
  alpha360        Alpha360 features (60 days of OHLCV = 360 features)
  alpha158-talib  Alpha158 + TA-Lib indicators (~300+ features, requires TA-Lib)
  alpha158-pandas Alpha158 + Pandas indicators (no TA-Lib, for large datasets)
  alpha360-pandas Alpha360 features via Pandas (no TA-Lib)
  alpha158-news   Alpha158 + TA-Lib + News sentiment features
"""
