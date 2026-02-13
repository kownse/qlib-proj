"""
运行 CatBoost baseline，预测N天股票价格波动率

波动率定义：未来N个交易日波动变化

扩展特征：包含 Alpha158 默认指标 + TA-Lib 技术指标
"""

# ============================================================================
# 重要: 以下代码必须在任何其他导入之前执行
# 解决 TA-Lib 与 qlib 多进程的内存冲突问题
# ============================================================================

import os

# 关键: 在导入任何库之前设置环境变量，限制线程数避免内存冲突
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

import sys
from pathlib import Path
import multiprocessing

# 强制使用 spawn 方法来创建子进程，避免 fork 导致的内存冲突
# 必须在导入任何使用 multiprocessing 的模块之前设置
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 已经设置过了

# 设置 scripts 目录到 path (用于导入 utils, data 等模块)
script_dir = Path(__file__).parent.parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))
project_root = script_dir.parent

# 预初始化 qlib (kernels=1, loky backend)
# 必须在其他 qlib 相关导入之前执行
import qlib
from qlib.constant import REG_US
from utils.talib_ops import TALIB_OPS

qlib_data_path = project_root / "my_data" / "qlib_us"
qlib.init(
    provider_uri=str(qlib_data_path),
    region=REG_US,
    custom_ops=TALIB_OPS,
    kernels=1,  # 关键: 避免多进程与 TA-Lib 冲突
    joblib_backend=None,  # 使用 loky 而不是 multiprocessing，避免 fork 与 TA-Lib 冲突
)

# ============================================================================
# 现在可以安全地导入其他模块
# ============================================================================

import json
import numpy as np
import pandas as pd
print("[DEBUG] 导入 catboost...")
from catboost import CatBoostRegressor, Pool

print("[DEBUG] 导入 CatBoostModel...")
from qlib.contrib.model.catboost_model import CatBoostModel
from qlib.data.dataset.handler import DataHandlerLP


# ============================================================================
# 默认 CatBoost 参数
# ============================================================================

DEFAULT_CATBOOST_PARAMS = {
    'loss_function': 'RMSE',
    'learning_rate': 0.05,
    'max_depth': 6,
    'l2_leaf_reg': 3,
    'random_strength': 1,
    'thread_count': 16,
    'verbose': False,
}


def load_params_from_file(params_file: str) -> tuple:
    """
    从 JSON 文件加载 CatBoost 参数

    Parameters
    ----------
    params_file : str
        参数文件路径 (hyperopt CV 输出的 JSON 文件)

    Returns
    -------
    tuple
        (CatBoost 参数字典, sample_weight_halflife or None)
    """
    with open(params_file, 'r') as f:
        data = json.load(f)

    # 支持两种格式:
    # 1. hyperopt CV 格式: {"params": {...}, "cv_results": {...}}
    # 2. 直接参数格式: {"learning_rate": ..., "max_depth": ...}
    sample_weight_halflife = None
    if 'params' in data:
        params = data['params']
        print(f"    Loaded params from CV hyperopt file")
        if 'cv_results' in data:
            cv = data['cv_results']
            print(f"    CV Mean IC: {cv.get('mean_ic', 'N/A'):.4f} (±{cv.get('std_ic', 'N/A'):.4f})")
        if 'sample_weight_halflife' in data:
            sample_weight_halflife = data['sample_weight_halflife']
            print(f"    Sample weight halflife: {sample_weight_halflife} years")
    else:
        params = data
        if 'sample_weight_halflife' in params:
            sample_weight_halflife = params.pop('sample_weight_halflife')
            print(f"    Sample weight halflife: {sample_weight_halflife} years")

    # 确保必要的参数存在
    final_params = DEFAULT_CATBOOST_PARAMS.copy()
    for key in ['learning_rate', 'max_depth', 'l2_leaf_reg', 'random_strength',
                'bagging_temperature', 'subsample', 'colsample_bylevel', 'min_data_in_leaf',
                'iterations', 'random_seed', 'loss_function']:
        if key in params:
            final_params[key] = params[key]

    return final_params, sample_weight_halflife

from utils.utils import evaluate_model
from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG, PROJECT_ROOT, MODEL_SAVE_PATH,
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
    run_backtest,
    load_catboost_model,
    get_catboost_feature_count,
    # CV utilities (for --params-file aligned path)
    FINAL_TEST,
    create_data_handler_for_fold,
    create_dataset_for_fold,
    compute_time_decay_weights,
    prepare_features_and_labels,
    compute_ic,
)


def train_catboost_native(args, handler_config, symbols, cb_params, halflife=0):
    """
    使用原生 CatBoostRegressor 训练模型（与 hyperopt CV 的 train_final_model 一致）

    当 --params-file 提供时使用此函数，确保训练流程与 hyperopt CV 完全对齐:
    - 使用 FINAL_TEST 时间划分
    - 使用原生 CatBoostRegressor (非 qlib CatBoostModel wrapper)
    - 使用 compute_time_decay_weights (非 TimeDecayReweighter)
    - fillna(0).replace([np.inf, -np.inf], 0) 预处理
    - DK_L 数据键

    Returns
    -------
    tuple
        (model, feature_names, test_pred, dataset)
    """
    print("\n[*] Training with native CatBoostRegressor (aligned with hyperopt CV)...")
    print("    Parameters:")
    for key, value in cb_params.items():
        if key not in ['thread_count', 'verbose', 'loss_function', 'random_seed']:
            print(f"      {key}: {value}")
    if halflife > 0:
        print(f"      sample_weight_halflife: {halflife}")

    # 创建数据集 (使用 FINAL_TEST splits，与 hyperopt CV 一致)
    handler = create_data_handler_for_fold(args, handler_config, symbols, FINAL_TEST)
    dataset = create_dataset_for_fold(handler, FINAL_TEST)

    # 准备特征和标签 (使用 DK_L，与 hyperopt CV 一致)
    train_data, train_label = prepare_features_and_labels(dataset, "train")
    valid_data, valid_label = prepare_features_and_labels(dataset, "valid")
    test_data, _ = prepare_features_and_labels(dataset, "test")
    feature_names = train_data.columns.tolist()

    print(f"\n    Training data:")
    print(f"      Train: {train_data.shape} ({FINAL_TEST['train_start']} ~ {FINAL_TEST['train_end']})")
    print(f"      Valid: {valid_data.shape} ({FINAL_TEST['valid_start']} ~ {FINAL_TEST['valid_end']})")
    print(f"      Test:  {test_data.shape} ({FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']})")

    # 样本权重 (与 hyperopt CV 一致)
    train_weight = None
    valid_weight = None
    if halflife > 0:
        train_weight = compute_time_decay_weights(train_data.index, halflife)
        valid_weight = compute_time_decay_weights(valid_data.index, halflife)
        print(f"\n    Using sample weights: halflife={halflife} years")

    train_pool = Pool(train_data, label=train_label, weight=train_weight)
    valid_pool = Pool(valid_data, label=valid_label, weight=valid_weight)

    # 训练
    print("\n    Training progress:")
    model = CatBoostRegressor(**cb_params)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    print(f"\n    Best iteration: {model.best_iteration_}")

    # 验证集 IC
    valid_pred = model.predict(valid_data)
    valid_ic, valid_ic_std, valid_icir = compute_ic(valid_pred, valid_label, valid_data.index)
    print(f"\n    [Validation Set - for reference]")
    print(f"    Valid IC:   {valid_ic:.4f}")
    print(f"    Valid ICIR: {valid_icir:.4f}")

    # 测试集预测
    test_pred_values = model.predict(test_data)
    test_pred = pd.Series(test_pred_values, index=test_data.index, name='score')

    print(f"\n    Test prediction shape: {test_pred.shape}")
    print(f"    Test prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    # 特征重要性
    importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    print_feature_importance(importance_df, "Top 20 Features by Importance")

    # 保存特征重要性
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    importance_filename = generate_model_filename("catboost_importance", args, 0, ".csv")
    importance_path = outputs_dir / importance_filename
    importance_df.to_csv(importance_path, index=False)
    print(f"    Feature importance saved to: {importance_path}")

    return model, feature_names, test_pred, dataset


def train_catboost(dataset, valid_cols, cb_params=None, reweighter=None):
    """
    训练 CatBoost 模型

    Parameters
    ----------
    dataset : DatasetH
        数据集
    valid_cols : list
        有效特征列
    cb_params : dict, optional
        CatBoost 参数，如果为 None 则使用默认参数
    reweighter : Reweighter, optional
        样本权重器 (e.g., TimeDecayReweighter)

    Returns
    -------
    tuple
        (model, feature_names, importance_df, num_model_features)
    """
    print("\n[6] Training CatBoost model...")

    # 使用传入的参数或默认参数
    if cb_params is None:
        cb_params = DEFAULT_CATBOOST_PARAMS.copy()

    # 打印使用的参数
    print("    Parameters:")
    for key in ['learning_rate', 'max_depth', 'l2_leaf_reg', 'random_strength',
                'bagging_temperature', 'subsample', 'colsample_bylevel', 'min_data_in_leaf']:
        if key in cb_params and cb_params[key] is not None:
            print(f"      {key}: {cb_params[key]}")

    # CatBoostModel 需要特定的参数格式，只传递非 None 的参数
    model_kwargs = {
        'loss': cb_params.get('loss_function', 'RMSE'),
        'learning_rate': cb_params.get('learning_rate', 0.05),
        'max_depth': int(cb_params.get('max_depth', 6)),
        'l2_leaf_reg': cb_params.get('l2_leaf_reg', 3),
        'random_strength': cb_params.get('random_strength', 1),
        'thread_count': cb_params.get('thread_count', 16),
    }

    # 只添加非 None 的可选参数
    optional_params = ['bagging_temperature', 'subsample', 'colsample_bylevel']
    for key in optional_params:
        if key in cb_params and cb_params[key] is not None:
            model_kwargs[key] = cb_params[key]

    # subsample 需要非 Bayesian bootstrap type
    if 'subsample' in model_kwargs:
        model_kwargs['bootstrap_type'] = 'MVS'

    # colsample_bylevel (RSM) 在 GPU 上不支持回归模式，强制使用 CPU
    if 'colsample_bylevel' in model_kwargs:
        model_kwargs['task_type'] = 'CPU'

    if 'min_data_in_leaf' in cb_params and cb_params['min_data_in_leaf'] is not None:
        model_kwargs['min_data_in_leaf'] = int(cb_params['min_data_in_leaf'])

    model = CatBoostModel(**model_kwargs)

    if reweighter is not None:
        print(f"    Using sample weights: {reweighter.__class__.__name__}")

    print("\n    Training progress:")
    model.fit(
        dataset,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=100,
        reweighter=reweighter,
    )
    print("    ✓ Model training completed")

    # 特征重要性分析
    print("\n[7] Feature Importance Analysis...")
    importance = model.model.get_feature_importance()
    feature_names_from_model = model.model.feature_names_
    num_model_features = len(feature_names_from_model) if feature_names_from_model else len(importance)

    print(f"    Model has {num_model_features} features, detected {len(valid_cols)} valid columns")
    if feature_names_from_model:
        feature_names = feature_names_from_model
    elif len(valid_cols) != num_model_features:
        print(f"    ⚠ Feature count mismatch!")
        train_data = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
        all_train_cols = train_data.columns.tolist()
        feature_names = all_train_cols[:num_model_features]
        print(f"    Using first {num_model_features} columns: {feature_names[:3]}...{feature_names[-3:]}")
    else:
        feature_names = valid_cols

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print_feature_importance(importance_df, "Top 20 Features by Importance")

    return model, feature_names, importance_df, num_model_features


def retrain_with_top_features(dataset, importance_df, args, cb_params=None, reweighter=None):
    """
    使用 top-k 特征重新训练

    Parameters
    ----------
    dataset : DatasetH
        数据集
    importance_df : pd.DataFrame
        特征重要性 DataFrame
    args : argparse.Namespace
        命令行参数
    cb_params : dict, optional
        CatBoost 参数，如果为 None 则使用默认参数
    reweighter : Reweighter, optional
        样本权重器

    Returns
    -------
    tuple
        (model, top_features, test_pred)
    """
    print(f"\n[8] Feature Selection and Retraining...")
    print(f"    Selecting top {args.top_k} features...")

    top_features = importance_df.head(args.top_k)['feature'].tolist()
    print(f"    Selected features: {top_features[:10]}{'...' if len(top_features) > 10 else ''}")

    train_data_selected = dataset.prepare("train", col_set="feature")[top_features]
    valid_data_selected = dataset.prepare("valid", col_set="feature")[top_features]
    test_data_selected = dataset.prepare("test", col_set="feature")[top_features]

    train_label = dataset.prepare("train", col_set="label")
    valid_label = dataset.prepare("valid", col_set="label")

    # Compute sample weights if reweighter provided
    w_train = None
    w_valid = None
    if reweighter is not None:
        df_train = pd.concat([train_data_selected, train_label], axis=1)
        df_valid = pd.concat([valid_data_selected, valid_label], axis=1)
        w_train = reweighter.reweight(df_train).values
        w_valid = reweighter.reweight(df_valid).values
        print(f"    Sample weights: train [{w_train.min():.4f}, {w_train.max():.4f}], "
              f"valid [{w_valid.min():.4f}, {w_valid.max():.4f}]")

    print(f"    ✓ Selected train features shape: {train_data_selected.shape}")

    print("\n    Retraining with selected features...")
    train_pool = Pool(train_data_selected, label=train_label.values.ravel(), weight=w_train)
    valid_pool = Pool(valid_data_selected, label=valid_label.values.ravel(), weight=w_valid)

    # 使用传入的参数或默认参数
    if cb_params is None:
        cb_params = DEFAULT_CATBOOST_PARAMS.copy()

    # 构建 CatBoostRegressor 参数，只传递非 None 的参数
    model_kwargs = {
        'loss_function': cb_params.get('loss_function', 'RMSE'),
        'learning_rate': cb_params.get('learning_rate', 0.05),
        'max_depth': int(cb_params.get('max_depth', 6)),
        'l2_leaf_reg': cb_params.get('l2_leaf_reg', 3),
        'random_strength': cb_params.get('random_strength', 1),
        'thread_count': cb_params.get('thread_count', 16),
        'verbose': False,
    }

    # 只添加非 None 的可选参数
    optional_params = ['bagging_temperature', 'subsample', 'colsample_bylevel']
    for key in optional_params:
        if key in cb_params and cb_params[key] is not None:
            model_kwargs[key] = cb_params[key]

    # subsample 需要非 Bayesian bootstrap type
    if 'subsample' in model_kwargs:
        model_kwargs['bootstrap_type'] = 'MVS'

    # colsample_bylevel (RSM) 在 GPU 上不支持回归模式，强制使用 CPU
    if 'colsample_bylevel' in model_kwargs:
        model_kwargs['task_type'] = 'CPU'

    if 'min_data_in_leaf' in cb_params and cb_params['min_data_in_leaf'] is not None:
        model_kwargs['min_data_in_leaf'] = int(cb_params['min_data_in_leaf'])

    cb_model = CatBoostRegressor(**model_kwargs)

    cb_model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=50,
        verbose_eval=100,
    )
    print("    ✓ Retraining completed")

    # 打印重新训练后的特征重要性
    print("\n    Feature Importance after Retraining:")
    print("    " + "-" * 50)
    importance_retrained = cb_model.get_feature_importance()
    importance_retrained_df = pd.DataFrame({
        'feature': top_features,
        'importance': importance_retrained
    }).sort_values('importance', ascending=False)

    for i, row in importance_retrained_df.iterrows():
        print(f"    {importance_retrained_df.index.get_loc(i)+1:3d}. {row['feature']:<40s} {row['importance']:>10.2f}")
    print("    " + "-" * 50)

    # 预测
    print("\n[9] Generating predictions with selected features...")
    test_pred_values = cb_model.predict(test_data_selected)
    test_pred = pd.Series(test_pred_values, index=test_data_selected.index, name='score')
    print_prediction_stats(test_pred)

    return cb_model, top_features, test_pred


def main_train_impl():
    # 解析命令行参数
    parser = create_argument_parser("CatBoost", "run_catboost_nd.py")

    # 添加 CatBoost 特定参数
    parser.add_argument('--params-file', type=str, default=None,
                        help='Path to JSON file with CatBoost params (from hyperopt CV search)')

    args = parser.parse_args()

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    # 加载 CatBoost 参数
    cb_params = None
    sample_weight_halflife_from_file = None
    if args.params_file:
        print(f"\n[*] Loading CatBoost params from: {args.params_file}")
        cb_params, sample_weight_halflife_from_file = load_params_from_file(args.params_file)

    # 如果 params file 中有 sample_weight_halflife，且 CLI 未显式设置，则使用文件中的值
    if sample_weight_halflife_from_file is not None and args.sample_weight_halflife == 0:
        args.sample_weight_halflife = sample_weight_halflife_from_file

    # =========================================================================
    # --params-file 模式: 使用与 hyperopt CV train_final_model 完全对齐的流程
    # =========================================================================
    if args.params_file:
        time_splits = FINAL_TEST

        # 打印头部信息
        print_training_header("CatBoost", args, symbols, handler_config, time_splits)
        print(f"Params File: {args.params_file}")
        print(f"Mode: Aligned with hyperopt CV (FINAL_TEST splits, native CatBoostRegressor)")

        # 初始化
        init_qlib(handler_config['use_talib'])

        # 确定 halflife
        halflife = args.sample_weight_halflife

        # 训练 (使用与 hyperopt CV 完全一致的流程)
        model, feature_names, test_pred, dataset = train_catboost_native(
            args, handler_config, symbols, cb_params, halflife=halflife)

        # 评估
        print("\n[*] Evaluation on Test Set...")
        evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

        # 保存模型
        print("\n[*] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_filename = generate_model_filename("catboost", args, 0, ".cbm")
        model_path = MODEL_SAVE_PATH / model_filename
        meta_data = create_meta_data(args, handler_config, time_splits, feature_names, "catboost", 0)
        meta_data['params_file'] = args.params_file
        if halflife > 0:
            meta_data['sample_weight_halflife'] = halflife
        save_model_with_meta(model, model_path, meta_data)

        return model_path, dataset, test_pred, args, time_splits

    # =========================================================================
    # 默认模式: 使用 qlib CatBoostModel wrapper + DEFAULT_TIME_SPLITS
    # =========================================================================
    time_splits = get_time_splits(args.max_train)

    # 打印头部信息
    print_training_header("CatBoost", args, symbols, handler_config, time_splits)

    # 初始化和数据准备
    init_qlib(handler_config['use_talib'])
    check_data_availability(time_splits)
    handler = create_data_handler(args, handler_config, symbols, time_splits)
    dataset = create_dataset(handler, time_splits)
    train_data, valid_cols, dropped_cols = analyze_features(dataset)
    analyze_label_distribution(dataset)

    # 创建样本权重器
    reweighter = None
    if args.sample_weight_halflife > 0:
        from utils.reweighter import TimeDecayReweighter
        reweighter = TimeDecayReweighter(half_life_years=args.sample_weight_halflife, min_weight=0.1)
        print(f"\n[*] Sample weighting: exponential decay, half-life = {args.sample_weight_halflife} years, min_weight = 0.1")

    # 训练模型
    model, feature_names, importance_df, num_model_features = train_catboost(
        dataset, valid_cols, cb_params, reweighter=reweighter)

    # 保存特征重要性
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    importance_filename = generate_model_filename("catboost_importance", args, 0, ".csv")
    importance_path = outputs_dir / importance_filename
    importance_df.to_csv(importance_path, index=False)
    print(f"    Feature importance saved to: {importance_path}")

    # 特征选择和重新训练
    if args.top_k > 0 and args.top_k < len(feature_names):
        cb_model, top_features, test_pred = retrain_with_top_features(
            dataset, importance_df, args, cb_params, reweighter=reweighter)

        # 评估
        print("\n[10] Evaluation with selected features...")
        evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

        # 保存模型
        print("\n[11] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_filename = generate_model_filename("catboost", args, args.top_k, ".cbm")
        model_path = MODEL_SAVE_PATH / model_filename
        meta_data = create_meta_data(args, handler_config, time_splits, top_features, "catboost", args.top_k)
        save_model_with_meta(cb_model, model_path, meta_data)

        return model_path, dataset, test_pred, args, time_splits
    else:
        # 原始模型预测
        test_data_filtered = prepare_test_data_for_prediction(dataset, num_model_features)

        pred_values = model.model.predict(test_data_filtered.values)
        test_pred = pd.Series(pred_values, index=test_data_filtered.index, name='score')
        print_prediction_stats(test_pred)

        # 评估
        print("\n[9] Evaluation...")
        evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

        # 保存模型
        print("\n[10] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_filename = generate_model_filename("catboost", args, 0, ".cbm")
        model_path = MODEL_SAVE_PATH / model_filename
        meta_data = create_meta_data(args, handler_config, time_splits, valid_cols, "catboost", 0)
        save_model_with_meta(model.model, model_path, meta_data)

        return model_path, dataset, test_pred, args, time_splits


def main():
    result = main_train_impl()
    if result is not None:
        model_path, dataset, pred, args, time_splits = result
        if args.backtest:
            run_backtest(
                model_path, dataset, pred, args, time_splits,
                model_name="CatBoost",
                load_model_func=load_catboost_model,
                get_feature_count_func=get_catboost_feature_count
            )


if __name__ == "__main__":
    main()
