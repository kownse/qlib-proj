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
    load_catboost_model,
    get_catboost_feature_count,
    load_lightgbm_model,
    get_lightgbm_feature_count,
    load_xgboost_model,
    get_xgboost_feature_count,
    print_prediction_stats,
    prepare_top_k_data,
    print_retrained_importance,
    TreeModelAdapter,
    tree_main_train_impl,
    tree_backtest_only_impl,
    tree_main,
)

from .backtest import run_backtest

from .cv_utils import (
    CV_FOLDS,
    FINAL_TEST,
    create_data_handler_for_fold,
    create_dataset_for_fold,
    prepare_data_from_dataset,
    compute_time_decay_weights,
    prepare_features_and_labels,
    compute_ic,
    print_cv_info,
    prepare_cv_fold_data,
    BaseCVHyperoptObjective,
    run_hyperopt_cv_search_generic,
    first_pass_feature_selection_generic,
    train_final_model_generic,
)

from .macro_features import (
    DEFAULT_MACRO_PATH,
    MINIMAL_MACRO_FEATURES,
    CORE_MACRO_FEATURES,
    FEATURES_NEED_ZSCORE,
    load_macro_df,
    get_macro_cols,
    prepare_macro,
)

from .ts_model_utils import (
    HANDLER_D_FEAT,
    resolve_d_feat_and_seq_len,
)

from .ensemble import (
    zscore_by_day,
    load_ae_mlp_model,
    load_model_meta,
    create_ensemble_data_handler,
    create_ensemble_dataset,
    predict_with_ae_mlp,
    predict_with_catboost,
    compute_ic as compute_ensemble_ic,
    calculate_pairwise_correlations,
    ensemble_predictions,
    learn_optimal_weights,
    run_ensemble_backtest,
)
