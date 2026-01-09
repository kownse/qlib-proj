"""
运行 LightGBM baseline，预测N天股票价格波动率

波动率定义：未来N个交易日波动变化

扩展特征：包含 Alpha158 默认指标 + TA-Lib 技术指标
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
script_dir = Path(__file__).parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

import numpy as np
import pandas as pd
import lightgbm as lgb

from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset.handler import DataHandlerLP

from utils.utils import evaluate_model
from data.stock_pools import STOCK_POOLS

from models.common_config import HANDLER_CONFIG, PROJECT_ROOT, MODEL_SAVE_PATH
from models.training_utils import (
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
from models.backtest_common import run_backtest


def train_lightgbm(dataset, valid_cols):
    """
    训练 LightGBM 模型

    Parameters
    ----------
    dataset : DatasetH
        数据集
    valid_cols : list
        有效特征列

    Returns
    -------
    tuple
        (model, feature_names, importance_df)
    """
    print("\n[6] Training LightGBM model...")
    model = LGBModel(
        loss="mse",
        learning_rate=0.01,
        max_depth=8,
        num_leaves=128,
        num_threads=16,
        n_estimators=400,
        early_stopping_rounds=100,
        verbose=-1,
    )

    print("\n    Training progress:")
    model.fit(dataset)
    print("    ✓ Model training completed")

    # 特征重要性分析
    print("\n[7] Feature Importance Analysis...")
    importance = model.model.feature_importance(importance_type='gain')
    num_model_features = model.model.num_feature()

    print(f"    Model expects {num_model_features} features, detected {len(valid_cols)} valid columns")
    if len(valid_cols) != num_model_features:
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

    print_feature_importance(importance_df, "Top 20 Features by Importance (gain)")

    return model, feature_names, importance_df


def retrain_with_top_features(dataset, importance_df, args):
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

    print(f"    ✓ Selected train features shape: {train_data_selected.shape}")

    print("\n    Retraining with selected features...")
    train_set = lgb.Dataset(train_data_selected, label=train_label.values.ravel())
    valid_set = lgb.Dataset(valid_data_selected, label=valid_label.values.ravel())

    lgb_params = {
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': 0.01,
        'max_depth': 8,
        'num_leaves': 128,
        'num_threads': 4,
        'verbose': -1,
    }

    lgb_model = lgb.train(
        lgb_params,
        train_set,
        num_boost_round=200,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    print("    ✓ Retraining completed")

    # 打印重新训练后的特征重要性
    print("\n    Feature Importance after Retraining:")
    print("    " + "-" * 50)
    importance_retrained = lgb_model.feature_importance(importance_type='gain')
    importance_retrained_df = pd.DataFrame({
        'feature': top_features,
        'importance': importance_retrained
    }).sort_values('importance', ascending=False)

    for i, row in importance_retrained_df.iterrows():
        print(f"    {importance_retrained_df.index.get_loc(i)+1:3d}. {row['feature']:<40s} {row['importance']:>10.2f}")
    print("    " + "-" * 50)

    # 预测
    print("\n[9] Generating predictions with selected features...")
    test_pred_values = lgb_model.predict(test_data_selected)
    test_pred = pd.Series(test_pred_values, index=test_data_selected.index, name='score')
    print_prediction_stats(test_pred)

    return lgb_model, top_features, test_pred


def main_train_impl():
    # 解析命令行参数
    parser = create_argument_parser("LightGBM", "run_lgb_nd.py")
    args = parser.parse_args()

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = get_time_splits(args.max_train)

    # 打印头部信息
    print_training_header("LightGBM", args, symbols, handler_config, time_splits)

    # 初始化和数据准备
    init_qlib(handler_config['use_talib'])
    check_data_availability(time_splits)
    handler = create_data_handler(args, handler_config, symbols, time_splits)
    dataset = create_dataset(handler, time_splits)
    train_data, valid_cols, dropped_cols = analyze_features(dataset)
    analyze_label_distribution(dataset)

    # 训练模型
    model, feature_names, importance_df = train_lightgbm(dataset, valid_cols)

    # 特征选择和重新训练
    if args.top_k > 0 and args.top_k < len(feature_names):
        lgb_model, top_features, test_pred = retrain_with_top_features(dataset, importance_df, args)

        # 评估
        print("\n[10] Evaluation with selected features...")
        evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

        # 保存模型
        print("\n[11] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_filename = generate_model_filename("lgb", args, args.top_k, ".txt")
        model_path = MODEL_SAVE_PATH / model_filename
        meta_data = create_meta_data(args, handler_config, time_splits, top_features, "lightgbm", args.top_k)
        save_model_with_meta(lgb_model, model_path, meta_data)

        return model_path, dataset, test_pred, args, time_splits
    else:
        # 原始模型预测
        num_model_features = model.model.num_feature()
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
        model_filename = generate_model_filename("lgb", args, 0, ".txt")
        model_path = MODEL_SAVE_PATH / model_filename
        meta_data = create_meta_data(args, handler_config, time_splits, valid_cols, "lightgbm", 0)
        save_model_with_meta(model.model, model_path, meta_data)

        return model_path, dataset, test_pred, args, time_splits


def load_lightgbm_model(model_path):
    """加载 LightGBM 模型"""
    return lgb.Booster(model_file=str(model_path))


def get_lightgbm_feature_count(model):
    """获取 LightGBM 模型特征数量"""
    return model.num_feature()


def backtest_only_impl(args):
    """
    跳过训练，直接加载模型进行回测

    Parameters
    ----------
    args : argparse.Namespace
        命令行参数，必须包含 model_path
    """
    import pickle

    model_path = Path(args.model_path)
    meta_path = model_path.with_suffix('.meta.pkl')

    print("=" * 70)
    print("BACKTEST ONLY MODE (Skip Training)")
    print("=" * 70)

    # 加载模型元数据
    if not meta_path.exists():
        print(f"Error: Metadata file not found: {meta_path}")
        print("Please provide a model with .meta.pkl file")
        return None

    with open(meta_path, 'rb') as f:
        meta_data = pickle.load(f)

    print(f"\n[1] Loading model metadata...")
    print(f"    Model path: {model_path}")
    print(f"    Handler: {meta_data.get('handler', 'N/A')}")
    print(f"    Stock pool: {meta_data.get('stock_pool', 'N/A')}")
    print(f"    N-day: {meta_data.get('nday', 'N/A')}")
    print(f"    Top-k features: {meta_data.get('top_k', 0)}")

    # 使用元数据中的配置，但允许 CLI 覆盖部分参数
    handler_name = meta_data.get('handler', args.handler)
    stock_pool = meta_data.get('stock_pool', args.stock_pool)

    handler_config = HANDLER_CONFIG[handler_name]
    symbols = STOCK_POOLS[stock_pool]
    time_splits = get_time_splits(args.max_train)

    # 初始化 Qlib
    print(f"\n[2] Initializing Qlib...")
    init_qlib(handler_config['use_talib'])

    # 创建数据集（只需要测试集用于预测）
    print(f"\n[3] Creating dataset for prediction...")
    handler = create_data_handler(args, handler_config, symbols, time_splits)
    dataset = create_dataset(handler, time_splits)

    # 加载模型
    print(f"\n[4] Loading model...")
    model = load_lightgbm_model(model_path)
    num_features = model.num_feature()
    print(f"    ✓ Model loaded, features: {num_features}")

    # 准备测试数据
    print(f"\n[5] Preparing test data...")
    feature_names = meta_data.get('feature_names', [])
    top_k = meta_data.get('top_k', 0)

    if top_k > 0 and feature_names:
        # 使用保存的特征名
        test_data = dataset.prepare("test", col_set="feature")
        available_features = [f for f in feature_names if f in test_data.columns]
        if len(available_features) < len(feature_names):
            print(f"    ⚠ Some features missing: {len(available_features)}/{len(feature_names)}")
        test_data_filtered = test_data[available_features]
    else:
        # 使用所有特征
        test_data_filtered = prepare_test_data_for_prediction(dataset, num_features)

    print(f"    Test data shape: {test_data_filtered.shape}")

    # 生成预测
    print(f"\n[6] Generating predictions...")
    pred_values = model.predict(test_data_filtered.values)
    test_pred = pd.Series(pred_values, index=test_data_filtered.index, name='score')
    print_prediction_stats(test_pred)

    # 评估
    print(f"\n[7] Evaluation...")
    evaluate_model(dataset, test_pred, PROJECT_ROOT, meta_data.get('nday', args.nday))

    return model_path, dataset, test_pred, args, time_splits


def main():
    # 解析命令行参数
    parser = create_argument_parser("LightGBM", "run_lgb_nd.py")
    args = parser.parse_args()

    # 检查是否为仅回测模式
    if args.model_path:
        # 跳过训练，直接回测
        if not args.backtest:
            print("Warning: --model-path provided but --backtest not set. Enabling backtest automatically.")
            args.backtest = True

        result = backtest_only_impl(args)
    else:
        # 正常训练流程
        result = main_train_impl()

    if result is not None:
        model_path, dataset, pred, args, time_splits = result
        if args.backtest:
            run_backtest(
                model_path, dataset, pred, args, time_splits,
                model_name="LightGBM",
                load_model_func=load_lightgbm_model,
                get_feature_count_func=get_lightgbm_feature_count
            )


if __name__ == "__main__":
    main()
