"""
运行 XGBoost baseline，预测N天股票价格波动率

波动率定义：未来N个交易日波动变化

扩展特征：包含 Alpha158 默认指标 + TA-Lib 技术指标
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
script_dir = Path(__file__).parent.parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

import numpy as np
import pandas as pd
import xgboost as xgb

from qlib.data.dataset.handler import DataHandlerLP

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
)


def train_xgboost(dataset, valid_cols):
    """
    训练 XGBoost 模型

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
    print("\n[6] Training XGBoost model...")

    # 准备训练和验证数据
    train_data = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
    train_label = dataset.prepare("train", col_set="label")
    valid_data = dataset.prepare("valid", col_set="feature", data_key=DataHandlerLP.DK_L)
    valid_label = dataset.prepare("valid", col_set="label")

    # 创建 DMatrix
    dtrain = xgb.DMatrix(train_data, label=train_label.values.ravel())
    dvalid = xgb.DMatrix(valid_data, label=valid_label.values.ravel())

    # XGBoost 参数
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.01,
        'max_depth': 8,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'nthread': 16,
        'verbosity': 0,
        'seed': 42,
    }

    print("\n    Training progress:")
    evals = [(dtrain, 'train'), (dvalid, 'valid')]

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=400,
        evals=evals,
        early_stopping_rounds=100,
        verbose_eval=False
    )
    print(f"    ✓ Model training completed (best iteration: {model.best_iteration})")

    # 特征重要性分析
    print("\n[7] Feature Importance Analysis...")
    importance_dict = model.get_score(importance_type='gain')

    # 获取特征名称（从训练数据列名）
    feature_names = train_data.columns.tolist()
    num_model_features = len(feature_names)

    print(f"    Model trained with {num_model_features} features, detected {len(valid_cols)} valid columns")

    # 构建特征重要性 DataFrame
    # XGBoost 的 get_score 只返回有重要性的特征
    importance_list = []
    for feat in feature_names:
        # XGBoost 特征名可能是 f0, f1, ... 或原始名称
        if feat in importance_dict:
            importance_list.append({'feature': feat, 'importance': importance_dict[feat]})
        elif f'f{feature_names.index(feat)}' in importance_dict:
            importance_list.append({'feature': feat, 'importance': importance_dict[f'f{feature_names.index(feat)}']})
        else:
            importance_list.append({'feature': feat, 'importance': 0})

    importance_df = pd.DataFrame(importance_list).sort_values('importance', ascending=False)

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

    # 创建 DMatrix
    dtrain = xgb.DMatrix(train_data_selected, label=train_label.values.ravel())
    dvalid = xgb.DMatrix(valid_data_selected, label=valid_label.values.ravel())

    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.01,
        'max_depth': 8,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'nthread': 4,
        'verbosity': 0,
        'seed': 42,
    }

    evals = [(dtrain, 'train'), (dvalid, 'valid')]

    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=False
    )
    print(f"    ✓ Retraining completed (best iteration: {xgb_model.best_iteration})")

    # 打印重新训练后的特征重要性
    print("\n    Feature Importance after Retraining:")
    print("    " + "-" * 50)
    importance_dict = xgb_model.get_score(importance_type='gain')

    importance_retrained_list = []
    for feat in top_features:
        if feat in importance_dict:
            importance_retrained_list.append({'feature': feat, 'importance': importance_dict[feat]})
        else:
            importance_retrained_list.append({'feature': feat, 'importance': 0})

    importance_retrained_df = pd.DataFrame(importance_retrained_list).sort_values('importance', ascending=False)

    for idx, row in importance_retrained_df.iterrows():
        rank = importance_retrained_df.index.get_loc(idx) + 1
        print(f"    {rank:3d}. {row['feature']:<40s} {row['importance']:>10.2f}")
    print("    " + "-" * 50)

    # 预测
    print("\n[9] Generating predictions with selected features...")
    dtest = xgb.DMatrix(test_data_selected)
    test_pred_values = xgb_model.predict(dtest)
    test_pred = pd.Series(test_pred_values, index=test_data_selected.index, name='score')
    print_prediction_stats(test_pred)

    return xgb_model, top_features, test_pred


def main_train_impl():
    # 解析命令行参数
    parser = create_argument_parser("XGBoost", "run_xgboost_nd.py")
    args = parser.parse_args()

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = get_time_splits(args.max_train)

    # 打印头部信息
    print_training_header("XGBoost", args, symbols, handler_config, time_splits)

    # 初始化和数据准备
    init_qlib(handler_config['use_talib'])
    check_data_availability(time_splits)
    handler = create_data_handler(args, handler_config, symbols, time_splits)
    dataset = create_dataset(handler, time_splits)
    train_data, valid_cols, dropped_cols = analyze_features(dataset)
    analyze_label_distribution(dataset)

    # 训练模型
    model, feature_names, importance_df = train_xgboost(dataset, valid_cols)

    # 特征选择和重新训练
    if args.top_k > 0 and args.top_k < len(feature_names):
        xgb_model, top_features, test_pred = retrain_with_top_features(dataset, importance_df, args)

        # 评估
        print("\n[10] Evaluation with selected features...")
        evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

        # 保存模型
        print("\n[11] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_filename = generate_model_filename("xgb", args, args.top_k, ".json")
        model_path = MODEL_SAVE_PATH / model_filename
        meta_data = create_meta_data(args, handler_config, time_splits, top_features, "xgboost", args.top_k)
        save_model_with_meta(xgb_model, model_path, meta_data)

        return model_path, dataset, test_pred, args, time_splits
    else:
        # 原始模型预测
        num_model_features = len(feature_names)
        test_data_filtered = prepare_test_data_for_prediction(dataset, num_model_features)

        dtest = xgb.DMatrix(test_data_filtered)
        pred_values = model.predict(dtest)
        test_pred = pd.Series(pred_values, index=test_data_filtered.index, name='score')
        print_prediction_stats(test_pred)

        # 评估
        print("\n[9] Evaluation...")
        evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

        # 保存模型
        print("\n[10] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_filename = generate_model_filename("xgb", args, 0, ".json")
        model_path = MODEL_SAVE_PATH / model_filename
        meta_data = create_meta_data(args, handler_config, time_splits, valid_cols, "xgboost", 0)
        save_model_with_meta(model, model_path, meta_data)

        return model_path, dataset, test_pred, args, time_splits


def load_xgboost_model(model_path):
    """加载 XGBoost 模型"""
    model = xgb.Booster()
    model.load_model(str(model_path))
    return model


def get_xgboost_feature_count(model):
    """获取 XGBoost 模型特征数量"""
    return model.num_features()


def main():
    result = main_train_impl()
    if result is not None:
        model_path, dataset, pred, args, time_splits = result
        if args.backtest:
            run_backtest(
                model_path, dataset, pred, args, time_splits,
                model_name="XGBoost",
                load_model_func=load_xgboost_model,
                get_feature_count_func=get_xgboost_feature_count
            )


if __name__ == "__main__":
    main()
