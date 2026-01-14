"""
共享配置和常量

包含所有模型训练脚本共用的配置、常量和 Handler 配置映射

注意: datahandler 类使用延迟导入（通过字符串引用），
避免在模块加载时触发 qlib 的自动初始化，这会导致与 TA-Lib 的冲突。
"""

from pathlib import Path


# ========== 路径配置 ==========
# scripts/models/common/config.py -> scripts/models/common -> scripts/models -> scripts -> qlib-proj
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # 项目根目录
QLIB_DATA_PATH = PROJECT_ROOT / "my_data" / "qlib_us"
NEWS_DATA_PATH = PROJECT_ROOT / "my_data" / "news_processed" / "news_features.parquet"
MACRO_DATA_PATH = PROJECT_ROOT / "my_data" / "macro_processed" / "macro_features.parquet"
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


def _get_handler_class(handler_name: str):
    """
    延迟获取 Handler 类

    Parameters
    ----------
    handler_name : str
        Handler 名称

    Returns
    -------
    class
        Handler 类
    """
    if handler_name == 'alpha158':
        from data.datahandler_ext import Alpha158_Volatility
        return Alpha158_Volatility
    elif handler_name == 'alpha360':
        from data.datahandler_ext import Alpha360_Volatility
        return Alpha360_Volatility
    elif handler_name == 'alpha158-talib':
        from data.datahandler_ext import Alpha158_Volatility_TALib
        return Alpha158_Volatility_TALib
    elif handler_name == 'alpha158-talib-lite':
        from data.datahandler_ext import Alpha158_Volatility_TALib_Lite
        return Alpha158_Volatility_TALib_Lite
    elif handler_name == 'alpha158-pandas':
        from data.datahandler_pandas import Alpha158_Volatility_Pandas
        return Alpha158_Volatility_Pandas
    elif handler_name == 'alpha360-pandas':
        from data.datahandler_pandas import Alpha360_Volatility_Pandas
        return Alpha360_Volatility_Pandas
    elif handler_name == 'alpha158-news':
        from data.datahandler_news import Alpha158_Volatility_TALib_News
        return Alpha158_Volatility_TALib_News
    elif handler_name == 'alpha158-talib-macro':
        from data.datahandler_macro import Alpha158_Volatility_TALib_Macro
        return Alpha158_Volatility_TALib_Macro
    elif handler_name == 'alpha158-macro':
        from data.datahandler_macro import Alpha158_Macro
        return Alpha158_Macro
    else:
        raise ValueError(f"Unknown handler: {handler_name}")


# ========== Handler 配置映射 ==========
# 注意: 'class' 字段现在是字符串，使用 get_handler_config() 获取实际配置
_HANDLER_CONFIG_META = {
    'alpha158': {
        'description': 'Alpha158 (158 technical indicators)',
        'use_talib': False,
    },
    'alpha360': {
        'description': 'Alpha360 (360 features - 60 days OHLCV)',
        'use_talib': False,
    },
    'alpha158-talib': {
        'description': 'Alpha158 + TA-Lib (~300+ technical indicators)',
        'use_talib': True,
    },
    'alpha158-talib-lite': {
        'description': 'Alpha158 + TA-Lib Lite (~20 key indicators, works with sp500)',
        'use_talib': True,
    },
    'alpha158-pandas': {
        'description': 'Alpha158 + Pandas indicators (no TA-Lib)',
        'use_talib': False,
    },
    'alpha360-pandas': {
        'description': 'Alpha360 + Pandas (no TA-Lib, 360 features)',
        'use_talib': False,
    },
    'alpha158-news': {
        'description': 'Alpha158 + TA-Lib + News features',
        'use_talib': True,
    },
    'alpha158-talib-macro': {
        'description': 'Alpha158 + TA-Lib Lite + Macro features (~205 features)',
        'use_talib': True,
    },
    'alpha158-macro': {
        'description': 'Alpha158 + Macro features (no TA-Lib, ~193 features)',
        'use_talib': False,
    },
}


class _LazyHandlerConfig(dict):
    """延迟加载 Handler 配置的字典"""

    def __getitem__(self, key):
        if key not in _HANDLER_CONFIG_META:
            raise KeyError(f"Unknown handler: {key}")

        meta = _HANDLER_CONFIG_META[key]
        return {
            'class': _get_handler_class(key),
            'description': meta['description'],
            'use_talib': meta['use_talib'],
        }

    def keys(self):
        return _HANDLER_CONFIG_META.keys()

    def __contains__(self, key):
        return key in _HANDLER_CONFIG_META


HANDLER_CONFIG = _LazyHandlerConfig()


def get_handler_epilog():
    """返回命令行帮助的 handler 说明"""
    return """
Handler choices:
  alpha158             Alpha158 features (158 technical indicators) [default]
  alpha360             Alpha360 features (60 days of OHLCV = 360 features)
  alpha158-talib       Alpha158 + TA-Lib indicators (~300+ features, requires TA-Lib)
  alpha158-talib-lite  Alpha158 + TA-Lib Lite (~170 features, works with sp500)
  alpha158-pandas      Alpha158 + Pandas indicators (no TA-Lib, for large datasets)
  alpha360-pandas      Alpha360 features via Pandas (no TA-Lib)
  alpha158-news        Alpha158 + TA-Lib + News sentiment features
  alpha158-talib-macro Alpha158 + TA-Lib Lite + Macro features (~205 features)
  alpha158-macro       Alpha158 + Macro features (no TA-Lib, ~193 features)
"""
