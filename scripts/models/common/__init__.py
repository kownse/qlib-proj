"""
Common utilities for model training.

Re-exports all public APIs from submodules for backward compatibility.
"""

from .config import (
    HANDLER_CONFIG,
    PROJECT_ROOT,
    QLIB_DATA_PATH,
    NEWS_DATA_PATH,
    MODEL_SAVE_PATH,
    DEFAULT_TIME_SPLITS,
    MAX_TRAIN_TIME_SPLITS,
    get_handler_epilog,
)

from .training import (
    create_argument_parser,
    get_time_splits,
    print_training_header,
    init_qlib,
    check_data_availability,
    create_data_handler,
    create_dataset,
    analyze_features,
    analyze_label_distribution,
    print_feature_importance,
    save_model_with_meta,
    create_meta_data,
    generate_model_filename,
    prepare_test_data_for_prediction,
    print_prediction_stats,
)

from .backtest import run_backtest
