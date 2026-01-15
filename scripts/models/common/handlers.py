"""
Centralized handler registry for all training scripts.

This module provides a unified way to get DataHandler classes by name,
avoiding code duplication across training scripts.
"""

from typing import Dict, Type, Optional
from qlib.data.dataset.handler import DataHandlerLP


def get_handler_class(handler_name: str) -> Type[DataHandlerLP]:
    """
    Get a DataHandler class by name.

    Args:
        handler_name: The handler name (e.g., 'alpha158', 'alpha158-talib-macro')

    Returns:
        The DataHandler class

    Raises:
        ValueError: If the handler name is unknown
    """
    handler_map = _build_handler_map()

    if handler_name not in handler_map:
        available = ', '.join(sorted(handler_map.keys()))
        raise ValueError(
            f"Unknown handler: {handler_name}. "
            f"Available handlers: {available}"
        )

    return handler_map[handler_name]


def get_available_handlers() -> list:
    """Return list of available handler names."""
    return sorted(_build_handler_map().keys())


def _build_handler_map() -> Dict[str, Type[DataHandlerLP]]:
    """
    Build the handler map with lazy imports.

    Uses lazy imports to avoid import errors when certain handler files
    don't exist or have missing dependencies.
    """
    handler_map = {}

    # Base handlers from datahandler_ext
    try:
        from data.datahandler_ext import (
            Alpha158_Volatility, Alpha360_Volatility,
            Alpha158_Volatility_TALib, Alpha158_Volatility_TALib_Lite
        )
        handler_map.update({
            'alpha158': Alpha158_Volatility,
            'alpha360': Alpha360_Volatility,
            'alpha158-talib': Alpha158_Volatility_TALib,
            'alpha158-talib-lite': Alpha158_Volatility_TALib_Lite,
        })
    except ImportError as e:
        print(f"Warning: Could not import base handlers: {e}")

    # Pandas-based handlers
    try:
        from data.datahandler_pandas import (
            Alpha158_Volatility_Pandas, Alpha360_Volatility_Pandas
        )
        handler_map.update({
            'alpha158-pandas': Alpha158_Volatility_Pandas,
            'alpha360-pandas': Alpha360_Volatility_Pandas,
        })
    except ImportError:
        pass  # Optional handlers

    # Macro handlers
    try:
        from data.datahandler_macro import (
            Alpha158_Volatility_TALib_Macro, Alpha158_Macro
        )
        handler_map.update({
            'alpha158-talib-macro': Alpha158_Volatility_TALib_Macro,
            'alpha158-macro': Alpha158_Macro,
        })
    except ImportError:
        pass  # Optional handlers

    # Enhanced handlers (v1-v4)
    try:
        from data.datahandler_enhanced import Alpha158_Enhanced
        handler_map['alpha158-enhanced'] = Alpha158_Enhanced
    except ImportError:
        pass

    try:
        from data.datahandler_enhanced_v2 import Alpha158_Enhanced_V2
        handler_map['alpha158-enhanced-v2'] = Alpha158_Enhanced_V2
    except ImportError:
        pass

    try:
        from data.datahandler_enhanced_v3 import Alpha158_Enhanced_V3
        handler_map['alpha158-enhanced-v3'] = Alpha158_Enhanced_V3
    except ImportError:
        pass

    try:
        from data.datahandler_enhanced_v4 import Alpha158_Enhanced_V4
        handler_map['alpha158-enhanced-v4'] = Alpha158_Enhanced_V4
    except ImportError:
        pass

    # News handler
    try:
        from data.datahandler_news import Alpha158_Volatility_TALib_News
        handler_map['alpha158-news'] = Alpha158_Volatility_TALib_News
    except ImportError:
        pass

    return handler_map
