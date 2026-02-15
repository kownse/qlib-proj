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

from models.common import (
    print_feature_importance,
    print_prediction_stats,
    prepare_top_k_data,
    print_retrained_importance,
    load_xgboost_model,
    get_xgboost_feature_count,
)
from models.common.training import TreeModelAdapter, tree_main


def train_xgboost(dataset, valid_cols, **kwargs):
    """
    训练 XGBoost 模型

    Returns
    -------
    tuple
        (model, feature_names, importance_df, num_features)
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
    importance_list = []
    for feat in feature_names:
        if feat in importance_dict:
            importance_list.append({'feature': feat, 'importance': importance_dict[feat]})
        elif f'f{feature_names.index(feat)}' in importance_dict:
            importance_list.append({'feature': feat, 'importance': importance_dict[f'f{feature_names.index(feat)}']})
        else:
            importance_list.append({'feature': feat, 'importance': 0})

    importance_df = pd.DataFrame(importance_list).sort_values('importance', ascending=False)

    print_feature_importance(importance_df, "Top 20 Features by Importance (gain)")

    return model, feature_names, importance_df, num_model_features


def retrain_predict_xgb(dataset, importance_df, args, **kwargs):
    """使用 top-k 特征重新训练 XGBoost"""
    top_features, train_data, valid_data, test_data, train_label, valid_label = \
        prepare_top_k_data(dataset, importance_df, args.top_k)

    print("\n    Retraining with selected features...")

    # 创建 DMatrix
    dtrain = xgb.DMatrix(train_data, label=train_label.values.ravel())
    dvalid = xgb.DMatrix(valid_data, label=valid_label.values.ravel())

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
    importance_dict = xgb_model.get_score(importance_type='gain')
    importance_values = [importance_dict.get(feat, 0) for feat in top_features]
    print_retrained_importance(top_features, importance_values)

    # 预测
    print("\n[9] Generating predictions with selected features...")
    dtest = xgb.DMatrix(test_data)
    test_pred_values = xgb_model.predict(dtest)
    test_pred = pd.Series(test_pred_values, index=test_data.index, name='score')
    print_prediction_stats(test_pred)

    return xgb_model, top_features, test_pred


def predict_xgb(model, test_data):
    """XGBoost 预测"""
    return model.predict(xgb.DMatrix(test_data))


adapter = TreeModelAdapter(
    model_name="XGBoost",
    model_prefix="xgb",
    model_extension=".json",
    model_type="xgboost",
    script_name="run_xgboost_nd.py",
    train_model=train_xgboost,
    predict=predict_xgb,
    retrain_predict=retrain_predict_xgb,
    load_model=load_xgboost_model,
    get_feature_count=get_xgboost_feature_count,
    # XGBoost uses raw model directly (no wrapper)
)


if __name__ == "__main__":
    tree_main(adapter)
