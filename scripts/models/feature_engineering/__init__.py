"""
Feature Engineering Tools for Nested CV Feature Selection

This module contains tools for:
- Forward selection with AE-MLP (nested_cv_feature_selection.py)
- Forward selection with TCN (nested_cv_feature_selection_tcn.py)
- Backward elimination (nested_cv_backward_elimination.py)
- Shared utilities (feature_selection_utils.py)
"""

from .feature_selection_utils import (
    INNER_CV_FOLDS,
    ALL_STOCK_FEATURES,
    ALL_TALIB_FEATURES,
    ALL_MACRO_FEATURES,
    PROJECT_ROOT,
    SCRIPT_DIR,
    compute_ic,
    validate_qlib_features,
    validate_macro_features,
    load_macro_data,
    save_checkpoint,
    load_checkpoint,
    save_final_result,
    prepare_dataset_data,
    ForwardSelectionBase,
    add_common_args,
    countdown,
)

__all__ = [
    'INNER_CV_FOLDS',
    'ALL_STOCK_FEATURES',
    'ALL_TALIB_FEATURES',
    'ALL_MACRO_FEATURES',
    'PROJECT_ROOT',
    'SCRIPT_DIR',
    'compute_ic',
    'validate_qlib_features',
    'validate_macro_features',
    'load_macro_data',
    'save_checkpoint',
    'load_checkpoint',
    'save_final_result',
    'prepare_dataset_data',
    'ForwardSelectionBase',
    'add_common_args',
    'countdown',
]
