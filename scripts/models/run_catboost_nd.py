"""
运行 CatBoost baseline，预测N天股票价格波动率

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
from catboost import CatBoostRegressor, Pool

from qlib.contrib.model.catboost_model import CatBoostModel
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


def train_catboost(dataset, valid_cols):
    """
    训练 CatBoost 模型

    Parameters
    ----------
    dataset : DatasetH
        数据集
    valid_cols : list
        有效特征列

    Returns
    -------
    tuple
        (model, feature_names, importance_df, num_model_features)
    """
    print("\n[6] Training CatBoost model...")
    model = CatBoostModel(
        loss="RMSE",
        learning_rate=0.05,
        max_depth=6,
        l2_leaf_reg=3,
        random_strength=1,
        thread_count=16,
    )

    print("\n    Training progress:")
    model.fit(
        dataset,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=100,
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
    train_pool = Pool(train_data_selected, label=train_label.values.ravel())
    valid_pool = Pool(valid_data_selected, label=valid_label.values.ravel())

    cb_model = CatBoostRegressor(
        loss_function='RMSE',
        learning_rate=0.05,
        max_depth=6,
        l2_leaf_reg=3,
        random_strength=1,
        thread_count=16,
        verbose=False,
    )

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
    args = parser.parse_args()

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]
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

    # 训练模型
    model, feature_names, importance_df, num_model_features = train_catboost(dataset, valid_cols)

    # 特征选择和重新训练
    if args.top_k > 0 and args.top_k < len(feature_names):
        cb_model, top_features, test_pred = retrain_with_top_features(dataset, importance_df, args)

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


def load_catboost_model(model_path):
    """加载 CatBoost 模型"""
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return model


def get_catboost_feature_count(model):
    """获取 CatBoost 模型特征数量"""
    if hasattr(model, 'get_feature_count'):
        return model.get_feature_count()
    elif model.feature_names_:
        return len(model.feature_names_)
    return "N/A"


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
