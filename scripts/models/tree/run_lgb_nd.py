"""
运行 LightGBM baseline，预测N天股票价格波动率

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
import lightgbm as lgb

from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset.handler import DataHandlerLP

from models.common import (
    print_feature_importance,
    print_prediction_stats,
    prepare_top_k_data,
    print_retrained_importance,
    load_lightgbm_model,
    get_lightgbm_feature_count,
)
from models.common.training import TreeModelAdapter, tree_main


def train_lightgbm(dataset, valid_cols, **kwargs):
    """
    训练 LightGBM 模型

    Returns
    -------
    tuple
        (model, feature_names, importance_df, num_features)
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

    return model, feature_names, importance_df, num_model_features


def retrain_predict_lgb(dataset, importance_df, args, **kwargs):
    """使用 top-k 特征重新训练 LightGBM"""
    top_features, train_data, valid_data, test_data, train_label, valid_label = \
        prepare_top_k_data(dataset, importance_df, args.top_k)

    print("\n    Retraining with selected features...")
    train_set = lgb.Dataset(train_data, label=train_label.values.ravel())
    valid_set = lgb.Dataset(valid_data, label=valid_label.values.ravel())

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
    importance_retrained = lgb_model.feature_importance(importance_type='gain')
    print_retrained_importance(top_features, importance_retrained)

    # 预测
    print("\n[9] Generating predictions with selected features...")
    test_pred_values = lgb_model.predict(test_data)
    test_pred = pd.Series(test_pred_values, index=test_data.index, name='score')
    print_prediction_stats(test_pred)

    return lgb_model, top_features, test_pred


def predict_lgb(model, test_data):
    """LightGBM 预测"""
    return model.predict(test_data.values)


adapter = TreeModelAdapter(
    model_name="LightGBM",
    model_prefix="lgb",
    model_extension=".txt",
    model_type="lightgbm",
    script_name="run_lgb_nd.py",
    train_model=train_lightgbm,
    predict=predict_lgb,
    retrain_predict=retrain_predict_lgb,
    load_model=load_lightgbm_model,
    get_feature_count=get_lightgbm_feature_count,
    unwrap_model=lambda wrapper: wrapper.model,
)


if __name__ == "__main__":
    tree_main(adapter)
