"""
Centralized handler registry for all training scripts.

This module provides a unified way to get DataHandler classes by name,
avoiding code duplication across training scripts.

All handler configuration is centralized here:
- Handler class imports (lazy)
- Handler metadata (description, use_talib)
- Command line help text generation
"""

from typing import Dict, Type, List, Any


# ============================================================================
# Handler Registry - Single Source of Truth
# ============================================================================

# Each handler entry contains:
#   - module: the module path to import from
#   - class_name: the class name to import
#   - description: human-readable description
#   - use_talib: whether this handler requires TA-Lib

HANDLER_REGISTRY = {
    # Base handlers
    'alpha158': {
        'module': 'data.datahandler_ext',
        'class_name': 'Alpha158_Volatility',
        'description': 'Alpha158 (158 technical indicators)',
        'use_talib': False,
    },
    'alpha180': {
        'module': 'data.datahandler_ext',
        'class_name': 'Alpha180_Volatility',
        'description': 'Alpha180 (180 features - 30 days OHLCV)',
        'use_talib': False,
    },
    'alpha360': {
        'module': 'data.datahandler_ext',
        'class_name': 'Alpha360_Volatility',
        'description': 'Alpha360 (360 features - 60 days OHLCV, includes VWAP)',
        'use_talib': False,
    },
    'alpha300': {
        'module': 'data.datahandler_ext',
        'class_name': 'Alpha300_Volatility',
        'description': 'Alpha300 (300 features - 60 days OHLC+V, no VWAP, for US data)',
        'use_talib': False,
    },
    'alpha300-ts': {
        'module': 'data.datahandler_ext',
        'class_name': 'Alpha300_TS_Volatility',
        'description': 'Alpha300 for TCN/LSTM/Transformer (time-series normalization)',
        'use_talib': False,
    },
    'alpha158-talib': {
        'module': 'data.datahandler_ext',
        'class_name': 'Alpha158_Volatility_TALib',
        'description': 'Alpha158 + TA-Lib (~300+ technical indicators)',
        'use_talib': True,
    },
    'alpha158-talib-lite': {
        'module': 'data.datahandler_ext',
        'class_name': 'Alpha158_Volatility_TALib_Lite',
        'description': 'Alpha158 + TA-Lib Lite (~20 key indicators, works with sp500)',
        'use_talib': True,
    },

    # Pandas-based handlers
    'alpha158-pandas': {
        'module': 'data.datahandler_pandas',
        'class_name': 'Alpha158_Volatility_Pandas',
        'description': 'Alpha158 + Pandas indicators (no TA-Lib)',
        'use_talib': False,
    },
    'alpha360-pandas': {
        'module': 'data.datahandler_pandas',
        'class_name': 'Alpha360_Volatility_Pandas',
        'description': 'Alpha360 + Pandas (no TA-Lib, 360 features)',
        'use_talib': False,
    },

    # News handler
    'alpha158-news': {
        'module': 'data.datahandler_news',
        'class_name': 'Alpha158_Volatility_TALib_News',
        'description': 'Alpha158 + TA-Lib + News features',
        'use_talib': True,
    },

    # Macro handlers
    'alpha158-talib-macro': {
        'module': 'data.datahandler_macro',
        'class_name': 'Alpha158_Volatility_TALib_Macro',
        'description': 'Alpha158 + TA-Lib Lite + Macro features (~205 features)',
        'use_talib': True,
    },
    'alpha158-macro': {
        'module': 'data.datahandler_macro',
        'class_name': 'Alpha158_Macro',
        'description': 'Alpha158 + Macro features (no TA-Lib, ~193 features)',
        'use_talib': False,
    },
    'alpha180-macro': {
        'module': 'data.datahandler_macro',
        'class_name': 'Alpha180_Macro',
        'description': 'Alpha180 + Macro features (30 timesteps × (6+M) features)',
        'use_talib': False,
    },
    'alpha360-macro': {
        'module': 'data.datahandler_macro',
        'class_name': 'Alpha360_Macro',
        'description': 'Alpha360 + Macro features (60 timesteps × (6+M) features)',
        'use_talib': False,
    },

    # Enhanced handlers
    'alpha158-enhanced': {
        'module': 'data.datahandler_enhanced',
        'class_name': 'Alpha158_Enhanced',
        'description': 'Alpha158 Enhanced (~130 refined features based on importance)',
        'use_talib': True,
    },
    'alpha158-enhanced-v2': {
        'module': 'data.datahandler_enhanced_v2',
        'class_name': 'Alpha158_Enhanced_V2',
        'description': 'Alpha158 Enhanced V2 (~140 features, extended 52w features)',
        'use_talib': True,
    },
    'alpha158-enhanced-v3': {
        'module': 'data.datahandler_enhanced_v3',
        'class_name': 'Alpha158_Enhanced_V3',
        'description': 'Alpha158 Enhanced V3 (~130 streamlined features)',
        'use_talib': True,
    },
    'alpha158-enhanced-v4': {
        'module': 'data.datahandler_enhanced_v4',
        'class_name': 'Alpha158_Enhanced_V4',
        'description': 'Alpha158 Enhanced V4 (~83 minimized features with macro)',
        'use_talib': True,
    },
    'alpha158-enhanced-v5': {
        'module': 'data.datahandler_enhanced_v5',
        'class_name': 'Alpha158_Enhanced_V5',
        'description': 'Alpha158 Enhanced V5 (~19 stock-specific features, no macro)',
        'use_talib': True,
    },
    'alpha158-enhanced-v6': {
        'module': 'data.datahandler_enhanced_v6',
        'class_name': 'Alpha158_Enhanced_V6',
        'description': 'Alpha158 Enhanced V6 (~25 features: stock + lagged macro regime)',
        'use_talib': True,
    },
    'alpha158-enhanced-v7': {
        'module': 'data.datahandler_enhanced_v7',
        'class_name': 'Alpha158_Enhanced_V7',
        'description': 'Alpha158 Enhanced V7 (~40 features: expanded stock + macro)',
        'use_talib': True,
    },
    'alpha158-enhanced-v8': {
        'module': 'data.datahandler_enhanced_v8',
        'class_name': 'Alpha158_Enhanced_V8',
        'description': 'Alpha158 Enhanced V8 (~22 features: AE-MLP permutation importance)',
        'use_talib': True,
    },
    'alpha158-enhanced-v9': {
        'module': 'data.datahandler_enhanced_v9',
        'class_name': 'Alpha158_Enhanced_V9',
        'description': 'Alpha158 Enhanced V9 (~11 features: AE-MLP forward selection optimal)',
        'use_talib': True,
    },
    'alpha158-enhanced-v10': {
        'module': 'data.datahandler_enhanced_v10',
        'class_name': 'Alpha158_Enhanced_V10',
        'description': 'Alpha158 Enhanced V10 (~37 features: nested CV backward elimination protected)',
        'use_talib': True,
    },
    'alpha158-enhanced-v11': {
        'module': 'data.datahandler_enhanced_v11',
        'class_name': 'Alpha158_Enhanced_V11',
        'description': 'Alpha158 Enhanced V11 (~38 features: nested CV forward selection optimal)',
        'use_talib': True,
    },

    # CatBoost optimized handlers
    'catboost-v1': {
        'module': 'data.datahandler_catboost_v1',
        'class_name': 'Alpha158_CatBoost_V1',
        'description': 'CatBoost V1 (~14 features: nested CV forward selection for CatBoost)',
        'use_talib': True,
    },

    # LightGBM optimized handlers
    'lightgbm-v1': {
        'module': 'data.datahandler_lightgbm_v1',
        'class_name': 'Alpha158_LightGBM_V1',
        'description': 'LightGBM V1 (~12 features: nested CV forward selection for LightGBM)',
        'use_talib': True,
    },

    # MASTER model handlers
    'alpha158-master': {
        'module': 'data.datahandler_master',
        'class_name': 'Alpha158_Master',
        'description': 'Alpha158 + MASTER market info (205 features: 142 stock + 63 market)',
        'use_talib': False,
    },
    'alpha360-master': {
        'module': 'data.datahandler_master',
        'class_name': 'Alpha360_Master',
        'description': 'Alpha360 + MASTER market info (60×69 features: time-aligned)',
        'use_talib': False,
    },
}


# ============================================================================
# Public API
# ============================================================================

def get_handler_class(handler_name: str) -> Type:
    """
    Get a DataHandler class by name (lazy import).

    Args:
        handler_name: The handler name (e.g., 'alpha158', 'alpha158-talib-macro')

    Returns:
        The DataHandler class

    Raises:
        ValueError: If the handler name is unknown
    """
    if handler_name not in HANDLER_REGISTRY:
        available = ', '.join(sorted(HANDLER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown handler: {handler_name}. "
            f"Available handlers: {available}"
        )

    entry = HANDLER_REGISTRY[handler_name]
    module_name = entry['module']
    class_name = entry['class_name']

    # Lazy import
    import importlib
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Failed to import handler '{handler_name}': {e}")


def get_handler_config(handler_name: str) -> Dict[str, Any]:
    """
    Get full handler configuration including class and metadata.

    Args:
        handler_name: The handler name

    Returns:
        Dict with 'class', 'description', 'use_talib' keys
    """
    if handler_name not in HANDLER_REGISTRY:
        raise KeyError(f"Unknown handler: {handler_name}")

    entry = HANDLER_REGISTRY[handler_name]
    return {
        'class': get_handler_class(handler_name),
        'description': entry['description'],
        'use_talib': entry['use_talib'],
    }


def get_available_handlers() -> List[str]:
    """Return sorted list of available handler names."""
    return sorted(HANDLER_REGISTRY.keys())


def get_handler_metadata(handler_name: str) -> Dict[str, Any]:
    """
    Get handler metadata without importing the class.

    Args:
        handler_name: The handler name

    Returns:
        Dict with 'description', 'use_talib' keys
    """
    if handler_name not in HANDLER_REGISTRY:
        raise KeyError(f"Unknown handler: {handler_name}")

    entry = HANDLER_REGISTRY[handler_name]
    return {
        'description': entry['description'],
        'use_talib': entry['use_talib'],
    }


def handler_uses_talib(handler_name: str) -> bool:
    """Check if a handler requires TA-Lib."""
    if handler_name not in HANDLER_REGISTRY:
        return False
    return HANDLER_REGISTRY[handler_name]['use_talib']


def get_handler_epilog() -> str:
    """
    Generate command line help text for all handlers.

    Returns:
        Formatted help string for argparse epilog
    """
    lines = ["Handler choices:"]

    # Find max handler name length for alignment
    max_len = max(len(name) for name in HANDLER_REGISTRY.keys())

    for name in sorted(HANDLER_REGISTRY.keys()):
        desc = HANDLER_REGISTRY[name]['description']
        padding = ' ' * (max_len - len(name) + 2)
        lines.append(f"  {name}{padding}{desc}")

    return '\n'.join(lines)


# ============================================================================
# Lazy Config Dict (for backwards compatibility with HANDLER_CONFIG)
# ============================================================================

class _LazyHandlerConfig(dict):
    """
    Lazy-loading handler config dict for backwards compatibility.

    Usage:
        config = HANDLER_CONFIG['alpha158']
        # Returns: {'class': Alpha158_Volatility, 'description': '...', 'use_talib': False}
    """

    def __getitem__(self, key):
        return get_handler_config(key)

    def keys(self):
        return HANDLER_REGISTRY.keys()

    def __contains__(self, key):
        return key in HANDLER_REGISTRY

    def __iter__(self):
        return iter(HANDLER_REGISTRY)

    def __len__(self):
        return len(HANDLER_REGISTRY)


# Backwards-compatible config dict
HANDLER_CONFIG = _LazyHandlerConfig()
