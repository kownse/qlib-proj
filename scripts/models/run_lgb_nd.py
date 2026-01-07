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

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from utils.utils import evaluate_model

import qlib
from qlib.constant import REG_US
from qlib.data import D
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

# Import TA-Lib custom operators
from utils.talib_ops import TALIB_OPS

# Import extended data handlers
from data.datahandler_ext import Alpha158_Volatility, Alpha158_Volatility_TALib
from data.datahandler_news import Alpha158_Volatility_TALib_News

# Import stock pools
from data.stock_pools import STOCK_POOLS


# ========== 配置 ==========

# 数据路径
PROJECT_ROOT = Path(__file__).parent.parent.parent  # 项目根目录
QLIB_DATA_PATH = PROJECT_ROOT / "my_data" / "qlib_us"
NEWS_DATA_PATH = PROJECT_ROOT / "my_data" / "news_processed" / "news_features.parquet"

# 时间划分
# TRAIN_START = "2025-01-01"
# TRAIN_END = "2025-09-30"
# VALID_START = "2025-10-01"
# VALID_END = "2025-11-30"
# TEST_START = "2025-12-01"
# TEST_END = "2025-12-31"

TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
VALID_START = "2024-01-01"
VALID_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

# 波动率预测窗口（天数）
VOLATILITY_WINDOW = 2

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Qlib Stock Price Volatility Prediction')
    parser.add_argument('--nday', type=int, default=2, help='Volatility prediction window in days (default: 2)')
    parser.add_argument('--use-talib', action='store_true', help='Use extended TA-Lib features (default: False)')
    parser.add_argument('--use-news', action='store_true', help='Use Alpha158 + TA-Lib + News features (default: False)')
    parser.add_argument('--news-features', type=str, default='core', choices=['all', 'sentiment', 'stats', 'core'],
                        help='News feature set to use (default: core)')
    parser.add_argument('--news-rolling', action='store_true', help='Add rolling news features (default: False)')
    parser.add_argument('--top-k', type=int, default=0,
                        help='Number of top features to select for retraining (0 = disable feature selection)')
    parser.add_argument('--stock-pool', type=str, default='test', choices=['test', 'tech', 'sp100', 'sp500'],
                        help='Stock pool to use: test (10), tech (~30), sp100 (100), sp500 (~500) (default: test)')
    args = parser.parse_args()

    # 选择股票池
    symbols = STOCK_POOLS[args.stock_pool]

    # 更新全局变量
    global VOLATILITY_WINDOW
    VOLATILITY_WINDOW = args.nday

    print("=" * 70)
    print(f"Qlib {VOLATILITY_WINDOW}-Day Stock Price Volatility Prediction")
    print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    if args.use_news:
        print("Features: Alpha158 + TA-Lib + News Features")
        print(f"    News feature set: {args.news_features}")
        print(f"    News rolling features: {args.news_rolling}")
    elif args.use_talib:
        print("Features: Alpha158 + TA-Lib Technical Indicators")
    else:
        print("Features: Alpha158 (default)")
    if args.top_k > 0:
        print(f"Feature Selection: Top {args.top_k} features will be selected for retraining")
    print("=" * 70)

    # 1. 初始化 Qlib (包含 TA-Lib 自定义算子)
    print("\n[1] Initializing Qlib...")
    if args.use_talib or args.use_news:
        qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US, custom_ops=TALIB_OPS)
        print("    ✓ Qlib initialized with TA-Lib custom operators")
    else:
        qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US)
        print("    ✓ Qlib initialized")

    # 2. 检查数据
    print("\n[2] Checking data availability...")
    instruments = D.instruments(market="all")
    available_instruments = list(D.list_instruments(instruments))
    print(f"    ✓ Available instruments: {len(available_instruments)}")

    # 测试读取一只股票
    test_df = D.features(
        instruments=["AAPL"],
        fields=["$close", "$volume"],
        start_time=TEST_START,
        end_time=TEST_END
    )
    print(f"    ✓ AAPL sample data shape: {test_df.shape}")
    print(f"    ✓ Date range: {test_df.index.get_level_values('datetime').min().date()} to {test_df.index.get_level_values('datetime').max().date()}")

    # 3. 创建 DataHandler
    print(f"\n[3] Creating DataHandler with {VOLATILITY_WINDOW}-day volatility label...")
    if args.use_news:
        print(f"    Features: Alpha158 + TA-Lib + News (~250+ technical indicators + news features)")
        print(f"    News data path: {NEWS_DATA_PATH}")
    elif args.use_talib:
        print(f"    Features: Alpha158 + TA-Lib (~300+ technical indicators)")
    else:
        print(f"    Features: Alpha158 (158 technical indicators)")
    print(f"    Label: {VOLATILITY_WINDOW}-day realized volatility")

    if args.use_news:
        handler = Alpha158_Volatility_TALib_News(
            volatility_window=VOLATILITY_WINDOW,
            instruments=symbols,
            start_time=TRAIN_START,
            end_time=TEST_END,
            fit_start_time=TRAIN_START,
            fit_end_time=TRAIN_END,
            infer_processors=[],
            news_data_path=str(NEWS_DATA_PATH) if NEWS_DATA_PATH.exists() else None,
            news_features=args.news_features,
            add_news_rolling=args.news_rolling,
        )
        print(f"    ✓ DataHandler created with news features: {handler.get_news_feature_names()}")
    elif args.use_talib:
        handler = Alpha158_Volatility_TALib(
            volatility_window=VOLATILITY_WINDOW,
            instruments=symbols,
            start_time=TRAIN_START,
            end_time=TEST_END,
            fit_start_time=TRAIN_START,
            fit_end_time=TRAIN_END,
            infer_processors=[],
        )
        print("    ✓ DataHandler created")
    else:
        handler = Alpha158_Volatility(
            volatility_window=VOLATILITY_WINDOW,
            instruments=symbols,
            start_time=TRAIN_START,
            end_time=TEST_END,
            fit_start_time=TRAIN_START,
            fit_end_time=TRAIN_END,
            infer_processors=[],
        )
        print("    ✓ DataHandler created")

    # 4. 创建 Dataset
    print("\n[4] Creating Dataset...")
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (TRAIN_START, TRAIN_END),
            "valid": (VALID_START, VALID_END),
            "test": (TEST_START, TEST_END),
        }
    )

    # Use same data_key as qlib's LGBModel internally uses (DK_L = learn)
    train_data = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
    print(f"    ✓ Train features shape: {train_data.shape}")
    print(f"      (samples × features)")

    # Identify columns that will be dropped by LightGBM (all NaN or constant)
    valid_cols = []
    dropped_cols = []
    for col in train_data.columns:
        col_data = train_data[col]
        if col_data.isna().all():
            dropped_cols.append(col)
        elif col_data.nunique(dropna=True) <= 1:
            dropped_cols.append(col)
        else:
            valid_cols.append(col)

    if dropped_cols:
        print(f"    ⚠ {len(dropped_cols)} features will be dropped (all NaN or constant):")
        for col in dropped_cols[:5]:
            print(f"      - {col}")
        if len(dropped_cols) > 5:
            print(f"      ... and {len(dropped_cols) - 5} more")
    print(f"    ✓ Valid features: {len(valid_cols)}")

    # 检查标签分布
    print("\n[5] Analyzing label distribution...")
    train_label = dataset.prepare("train", col_set="label")
    valid_label = dataset.prepare("valid", col_set="label")

    print(f"    Train set volatility statistics:")
    print(f"      Mean:   {train_label['LABEL0'].mean():.4f} (annualized)")
    print(f"      Std:    {train_label['LABEL0'].std():.4f}")
    print(f"      Median: {train_label['LABEL0'].median():.4f}")
    print(f"      Min:    {train_label['LABEL0'].min():.4f}")
    print(f"      Max:    {train_label['LABEL0'].max():.4f}")

    print(f"\n    Valid set volatility statistics:")
    print(f"      Mean:   {valid_label['LABEL0'].mean():.4f} (annualized)")
    print(f"      Std:    {valid_label['LABEL0'].std():.4f}")

    # 5. 训练模型
    print("\n[6] Training LightGBM model...")
    print("    Model parameters:")
    print(f"      - loss: mse")
    print(f"      - learning_rate: 0.05")
    print(f"      - max_depth: 8")
    print(f"      - num_leaves: 128")
    print(f"      - n_estimators: 200")

    model = LGBModel(
        loss="mse",
        learning_rate=0.01,
        max_depth=8,
        num_leaves=128,
        num_threads=4,
        n_estimators=200,
        early_stopping_rounds=30,
        verbose=-1,  # 减少训练输出
    )

    print("\n    Training progress:")
    model.fit(dataset)
    print("    ✓ Model training completed")

    # 6.5. 特征重要性分析
    print("\n[7] Feature Importance Analysis...")
    importance = model.model.feature_importance(importance_type='gain')
    num_model_features = model.model.num_feature()

    # Verify our valid_cols matches model's feature count
    print(f"    Model expects {num_model_features} features, detected {len(valid_cols)} valid columns")
    if len(valid_cols) != num_model_features:
        print(f"    ⚠ Feature count mismatch!")
        # Get all columns from training data - the model used these in order
        all_train_cols = train_data.columns.tolist()
        feature_names = all_train_cols[:num_model_features]
        print(f"    Using first {num_model_features} columns: {feature_names[:3]}...{feature_names[-3:]}")
    else:
        feature_names = valid_cols

    # 创建特征重要性 DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # 打印 top 20 特征
    print("\n    Top 20 Features by Importance (gain):")
    print("    " + "-" * 50)
    for i, row in importance_df.head(20).iterrows():
        print(f"    {importance_df.index.get_loc(i)+1:3d}. {row['feature']:<40s} {row['importance']:>10.2f}")
    print("    " + "-" * 50)
    print(f"    Total features: {len(feature_names)}")

    # 特征选择和重新训练
    if args.top_k > 0 and args.top_k < len(feature_names):
        print(f"\n[8] Feature Selection and Retraining...")
        print(f"    Selecting top {args.top_k} features...")

        # 选择 top-k 特征
        top_features = importance_df.head(args.top_k)['feature'].tolist()
        print(f"    Selected features: {top_features[:10]}{'...' if len(top_features) > 10 else ''}")

        # 准备筛选后的数据
        train_data_selected = dataset.prepare("train", col_set="feature")[top_features]
        valid_data_selected = dataset.prepare("valid", col_set="feature")[top_features]
        test_data_selected = dataset.prepare("test", col_set="feature")[top_features]

        train_label = dataset.prepare("train", col_set="label")
        valid_label = dataset.prepare("valid", col_set="label")
        test_label = dataset.prepare("test", col_set="label")

        print(f"    ✓ Selected train features shape: {train_data_selected.shape}")

        # 重新训练模型
        print("\n    Retraining with selected features...")
        import lightgbm as lgb
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
        # Use top_features directly (we know exactly which features were used for retraining)
        importance_retrained_df = pd.DataFrame({
            'feature': top_features,
            'importance': importance_retrained
        }).sort_values('importance', ascending=False)

        for i, row in importance_retrained_df.iterrows():
            print(f"    {importance_retrained_df.index.get_loc(i)+1:3d}. {row['feature']:<40s} {row['importance']:>10.2f}")
        print("    " + "-" * 50)

        # 预测
        print("\n[9] Generating predictions with selected features...")
        # Use test_data_selected directly (already filtered to top_features)
        test_pred_values = lgb_model.predict(test_data_selected)
        test_pred = pd.Series(test_pred_values, index=test_data_selected.index, name='score')
        print(f"    ✓ Prediction shape: {test_pred.shape}")
        print(f"    ✓ Prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

        # 评估
        print("\n[10] Evaluation with selected features...")
        evaluate_model(dataset, test_pred, PROJECT_ROOT, VOLATILITY_WINDOW)
    else:
        # 原始模型预测
        print("\n[8] Generating predictions...")
        # Get test features - use DK_I for inference (matches qlib's predict)
        test_data = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
        print(f"    Test data shape: {test_data.shape}")

        # Filter to match training features exactly
        missing_features = [f for f in feature_names if f not in test_data.columns]
        if missing_features:
            print(f"    ⚠ {len(missing_features)} training features missing in test data")
            # Add missing columns as NaN
            for f in missing_features:
                test_data[f] = np.nan

        test_data_filtered = test_data[feature_names]
        print(f"    Filtered test features: {test_data_filtered.shape[1]} (expected: {len(feature_names)})")

        pred_values = model.model.predict(test_data_filtered.values)
        test_pred = pd.Series(pred_values, index=test_data_filtered.index, name='score')
        print(f"    ✓ Prediction shape: {test_pred.shape}")
        print(f"    ✓ Prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

        # 评估
        print("\n[9] Evaluation...")
        evaluate_model(dataset, test_pred, PROJECT_ROOT, VOLATILITY_WINDOW)


if __name__ == "__main__":
    main()
