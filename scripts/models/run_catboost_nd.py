"""
运行 CatBoost baseline，预测N天股票价格波动率

波动率定义：未来N个交易日波动变化

扩展特征：包含 Alpha158 默认指标 + TA-Lib 技术指标
"""

import sys
from pathlib import Path
from datetime import datetime

# Add scripts directory to path for imports
script_dir = Path(__file__).parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
from utils.utils import evaluate_model

import qlib
from qlib.constant import REG_US
from qlib.data import D
from qlib.contrib.model.catboost_model import CatBoostModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

# Import backtest components
from qlib.backtest import backtest as qlib_backtest
from qlib.contrib.evaluate import risk_analysis

# Import TA-Lib custom operators
from utils.talib_ops import TALIB_OPS

# Import extended data handlers
from data.datahandler_ext import Alpha158_Volatility, Alpha158_Volatility_TALib, Alpha360_Volatility
from data.datahandler_news import Alpha158_Volatility_TALib_News
from data.datahandler_pandas import Alpha158_Volatility_Pandas, Alpha360_Volatility_Pandas

# Import stock pools
from data.stock_pools import STOCK_POOLS

# Import custom strategy
from utils.strategy import get_strategy_config

# Import backtest utilities
from utils.backtest_utils import plot_backtest_curve, generate_trade_records


# ========== 配置 ==========

# 数据路径
PROJECT_ROOT = Path(__file__).parent.parent.parent  # 项目根目录
QLIB_DATA_PATH = PROJECT_ROOT / "my_data" / "qlib_us"
NEWS_DATA_PATH = PROJECT_ROOT / "my_data" / "news_processed" / "news_features.parquet"
MODEL_SAVE_PATH = PROJECT_ROOT / "my_models"

# 时间划分
TRAIN_START = "2000-01-01"
TRAIN_END = "2023-12-31"
VALID_START = "2024-01-01"
VALID_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

# 波动率预测窗口（天数）
VOLATILITY_WINDOW = 2

# Handler 配置映射
def save_model(cb_model, model_path, meta_data):
    """
    保存 CatBoost 模型和元数据

    Parameters
    ----------
    cb_model : catboost.CatBoost
        训练好的 CatBoost 模型
    model_path : Path or str
        模型保存路径（.cbm 文件）
    meta_data : dict
        元数据字典，包含 handler, stock_pool, nday, top_k, feature_names 等信息
    """
    model_path = Path(model_path)

    # 保存模型
    cb_model.save_model(str(model_path))

    # 保存元数据
    meta_path = model_path.with_suffix('.meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta_data, f)

    print(f"    ✓ Model saved to: {model_path}")
    print(f"    ✓ Metadata saved to: {meta_path}")


HANDLER_CONFIG = {
    'alpha158': {
        'class': Alpha158_Volatility,
        'description': 'Alpha158 (158 technical indicators)',
        'use_talib': False,
    },
    'alpha360': {
        'class': Alpha360_Volatility,
        'description': 'Alpha360 (360 features - 60 days OHLCV)',
        'use_talib': False,
    },
    'alpha158-talib': {
        'class': Alpha158_Volatility_TALib,
        'description': 'Alpha158 + TA-Lib (~300+ technical indicators)',
        'use_talib': True,
    },
    'alpha158-pandas': {
        'class': Alpha158_Volatility_Pandas,
        'description': 'Alpha158 + Pandas indicators (no TA-Lib)',
        'use_talib': False,
    },
    'alpha360-pandas': {
        'class': Alpha360_Volatility_Pandas,
        'description': 'Alpha360 + Pandas (no TA-Lib, 360 features)',
        'use_talib': False,
    },
    'alpha158-news': {
        'class': Alpha158_Volatility_TALib_News,
        'description': 'Alpha158 + TA-Lib + News features',
        'use_talib': True,
    },
}


def main_train_impl():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Qlib Stock Price Volatility Prediction (CatBoost)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Handler choices:
  alpha158        Alpha158 features (158 technical indicators) [default]
  alpha360        Alpha360 features (60 days of OHLCV = 360 features)
  alpha158-talib  Alpha158 + TA-Lib indicators (~300+ features, requires TA-Lib)
  alpha158-pandas Alpha158 + Pandas indicators (no TA-Lib, for large datasets)
  alpha360-pandas Alpha360 features via Pandas (no TA-Lib)
  alpha158-news   Alpha158 + TA-Lib + News sentiment features

Examples:
  python run_catboost_nd.py --stock-pool sp100 --handler alpha158
  python run_catboost_nd.py --stock-pool sp500 --handler alpha360-pandas
  python run_catboost_nd.py --stock-pool sp100 --handler alpha158-news --news-features core
  python run_catboost_nd.py --stock-pool sp100 --max-train  # Max training data for deployment
  python run_catboost_nd.py --stock-pool sp100 --backtest --rebalance-freq 5  # Rebalance every 5 days
        """
    )
    parser.add_argument('--nday', type=int, default=2,
                        help='Volatility prediction window in days (default: 2)')
    parser.add_argument('--handler', type=str, default='alpha158',
                        choices=list(HANDLER_CONFIG.keys()),
                        help='Handler type (default: alpha158)')
    parser.add_argument('--stock-pool', type=str, default='test',
                        choices=['test', 'tech', 'sp100', 'sp500'],
                        help='Stock pool: test (10), tech (~30), sp100 (100), sp500 (~500) (default: test)')
    parser.add_argument('--top-k', type=int, default=0,
                        help='Top features for retraining (0 = disable feature selection)')
    # News-specific options (only used with alpha158-news handler)
    parser.add_argument('--news-features', type=str, default='core',
                        choices=['all', 'sentiment', 'stats', 'core'],
                        help='News feature set (only for alpha158-news handler)')
    parser.add_argument('--news-rolling', action='store_true',
                        help='Add rolling news features (only for alpha158-news handler)')
    # Time split options
    parser.add_argument('--max-train', action='store_true',
                        help='Use maximum training data (train to 2025-09, valid 2025-10-12, no test) for deployment')
    # Backtest options
    parser.add_argument('--backtest', action='store_true',
                        help='Run backtest after training using TopkDropoutStrategy')
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of stocks to hold in TopkDropoutStrategy (default: 10)')
    parser.add_argument('--n-drop', type=int, default=2,
                        help='Number of stocks to drop/replace each day (default: 2)')
    parser.add_argument('--account', type=float, default=1000000,
                        help='Initial account value for backtest (default: 1000000)')
    parser.add_argument('--rebalance-freq', type=int, default=1,
                        help='Rebalance frequency in days (default: 1, i.e., rebalance every day)')
    args = parser.parse_args()

    # 获取 handler 配置
    handler_config = HANDLER_CONFIG[args.handler]

    # 选择股票池
    symbols = STOCK_POOLS[args.stock_pool]

    # 更新全局变量
    global VOLATILITY_WINDOW, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END
    VOLATILITY_WINDOW = args.nday

    # 根据 --max-train 参数调整时间划分
    if args.max_train:
        TRAIN_START = "2000-01-01"
        TRAIN_END = "2025-09-30"
        VALID_START = "2025-10-01"
        VALID_END = "2025-12-31"
        TEST_START = "2025-10-01"  # 测试集与验证集相同（用于部署场景）
        TEST_END = "2025-12-31"

    print("=" * 70)
    print(f"CatBoost {VOLATILITY_WINDOW}-Day Stock Price Volatility Prediction")
    print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Handler: {args.handler}")
    if args.max_train:
        print(f"Time Split: MAX-TRAIN mode (for deployment)")
    print(f"Features: {handler_config['description']}")
    if args.handler == 'alpha158-news':
        print(f"    News feature set: {args.news_features}")
        print(f"    News rolling features: {args.news_rolling}")
    if args.top_k > 0:
        print(f"Feature Selection: Top {args.top_k} features")
    if args.backtest:
        rebalance_info = f", rebalance_freq={args.rebalance_freq}" if args.rebalance_freq > 1 else ""
        print(f"Backtest: Enabled (topk={args.topk}, n_drop={args.n_drop}{rebalance_info})")
    print("=" * 70)

    # 1. 初始化 Qlib
    print("\n[1] Initializing Qlib...")
    if handler_config['use_talib']:
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
    print(f"    Features: {handler_config['description']}")
    print(f"    Label: {VOLATILITY_WINDOW}-day realized volatility")

    # 通用参数
    handler_kwargs = {
        'volatility_window': VOLATILITY_WINDOW,
        'instruments': symbols,
        'start_time': TRAIN_START,
        'end_time': TEST_END,
        'fit_start_time': TRAIN_START,
        'fit_end_time': TRAIN_END,
        'infer_processors': [],
    }

    # News handler 需要额外参数
    if args.handler == 'alpha158-news':
        handler_kwargs['news_data_path'] = str(NEWS_DATA_PATH) if NEWS_DATA_PATH.exists() else None
        handler_kwargs['news_features'] = args.news_features
        handler_kwargs['add_news_rolling'] = args.news_rolling
        print(f"    News data path: {NEWS_DATA_PATH}")

    # 创建 handler
    HandlerClass = handler_config['class']
    handler = HandlerClass(**handler_kwargs)

    if args.handler == 'alpha158-news':
        print(f"    ✓ DataHandler created with news features: {handler.get_news_feature_names()}")
    else:
        print(f"    ✓ DataHandler created: {args.handler}")

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

    # Use same data_key as qlib's model internally uses (DK_L = learn)
    train_data = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
    print(f"    ✓ Train features shape: {train_data.shape}")
    print(f"      (samples × features)")

    # Identify columns that will be dropped (all NaN or constant)
    valid_cols = []
    dropped_cols = []
    for col in train_data.columns:
        col_data = train_data[col]
        # Handle case where col_data might be DataFrame (duplicate column names)
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
        try:
            is_all_na = col_data.isna().all()
            n_unique = col_data.nunique(dropna=True)
            if is_all_na:
                dropped_cols.append(col)
            elif n_unique <= 1:
                dropped_cols.append(col)
            else:
                valid_cols.append(col)
        except Exception:
            # If any error, keep the column
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

    # 6.5. 特征重要性分析
    print("\n[7] Feature Importance Analysis...")
    importance = model.model.get_feature_importance()
    feature_names_from_model = model.model.feature_names_
    num_model_features = len(feature_names_from_model) if feature_names_from_model else len(importance)

    # Verify our valid_cols matches model's feature count
    print(f"    Model has {num_model_features} features, detected {len(valid_cols)} valid columns")
    if feature_names_from_model:
        feature_names = feature_names_from_model
    elif len(valid_cols) != num_model_features:
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
    print("\n    Top 20 Features by Importance:")
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
        from catboost import CatBoostRegressor, Pool
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
        print(f"    ✓ Prediction shape: {test_pred.shape}")
        print(f"    ✓ Prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

        # 评估
        print("\n[10] Evaluation with selected features...")
        evaluate_model(dataset, test_pred, PROJECT_ROOT, VOLATILITY_WINDOW)

        # 保存模型
        print("\n[11] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"catboost_{args.handler}_{args.stock_pool}_{args.nday}d_topk{args.top_k}_{timestamp}.cbm"
        model_path = MODEL_SAVE_PATH / model_filename
        meta_data = {
            'handler': args.handler,
            'stock_pool': args.stock_pool,
            'nday': args.nday,
            'top_k': args.top_k,
            'feature_names': top_features,
            'train_start': TRAIN_START,
            'train_end': TRAIN_END,
            'valid_start': VALID_START,
            'valid_end': VALID_END,
            'test_start': TEST_START,
            'test_end': TEST_END,
            'use_talib': handler_config['use_talib'],
            'model_type': 'catboost',
        }
        save_model(cb_model, model_path, meta_data)

        # 返回预测结果和模型路径供 backtest 使用
        return model_path, dataset, test_pred, args
    else:
        # 原始模型预测
        print("\n[8] Generating predictions...")
        # Get test features - use DK_L (same as training) to ensure consistent features
        test_data = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_L)
        print(f"    Test data shape: {test_data.shape}")

        # Get training data with same DK_L to ensure same columns
        train_cols = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L).columns.tolist()

        # Make training column names unique in same way
        train_cols_unique = []
        seen = {}
        for col in train_cols:
            if col in seen:
                train_cols_unique.append(f"{col}_{seen[col]}")
                seen[col] += 1
            else:
                train_cols_unique.append(col)
                seen[col] = 1

        # Use the first N columns that match training (N = model features)
        model_n_features = num_model_features
        print(f"    Model expects {model_n_features} features")

        # Select only columns present in both, up to model_n_features
        available_cols = [c for c in test_data.columns if c in train_cols_unique[:model_n_features]]

        # If we have enough columns, use them directly
        if len(available_cols) >= model_n_features:
            # Take only the first model_n_features columns from test_data
            test_data_filtered = test_data.iloc[:, :model_n_features]
        else:
            # Need to add missing columns
            print(f"    ⚠ Only {len(available_cols)} columns available, need {model_n_features}")
            test_data_filtered = test_data.iloc[:, :model_n_features].copy()
            # Pad with NaN if needed
            for i in range(len(available_cols), model_n_features):
                test_data_filtered[f"_missing_{i}"] = np.nan

        print(f"    Filtered test features: {test_data_filtered.shape[1]} (expected: {model_n_features})")

        pred_values = model.model.predict(test_data_filtered.values)
        test_pred = pd.Series(pred_values, index=test_data_filtered.index, name='score')
        print(f"    ✓ Prediction shape: {test_pred.shape}")
        print(f"    ✓ Prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

        # 评估
        print("\n[9] Evaluation...")
        evaluate_model(dataset, test_pred, PROJECT_ROOT, VOLATILITY_WINDOW)

        # 保存模型
        print("\n[10] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"catboost_{args.handler}_{args.stock_pool}_{args.nday}d_{timestamp}.cbm"
        model_path = MODEL_SAVE_PATH / model_filename
        meta_data = {
            'handler': args.handler,
            'stock_pool': args.stock_pool,
            'nday': args.nday,
            'top_k': 0,
            'feature_names': valid_cols,
            'train_start': TRAIN_START,
            'train_end': TRAIN_END,
            'valid_start': VALID_START,
            'valid_end': VALID_END,
            'test_start': TEST_START,
            'test_end': TEST_END,
            'use_talib': handler_config['use_talib'],
            'model_type': 'catboost',
        }
        save_model(model.model, model_path, meta_data)

        # 返回预测结果和模型路径供 backtest 使用
        return model_path, dataset, test_pred, args


def run_backtest(model_path, dataset, pred, args):
    """
    使用 TopkDropoutStrategy 进行回测

    Parameters
    ----------
    model_path : Path or str
        训练好的模型保存路径
    dataset : DatasetH
        数据集
    pred : pd.Series
        预测结果
    args : argparse.Namespace
        命令行参数
    """
    from catboost import CatBoostRegressor

    print("\n" + "=" * 70)
    print("BACKTEST with TopkDropoutStrategy (CatBoost)")
    print("=" * 70)

    # 加载模型和元数据
    model_path = Path(model_path)
    meta_path = model_path.with_suffix('.meta.pkl')

    print(f"\n[BT-0] Loading model from: {model_path}")
    loaded_model = CatBoostRegressor()
    loaded_model.load_model(str(model_path))
    print(f"    ✓ Model loaded successfully")
    # CatBoost 使用不同的方法获取特征数量
    try:
        n_features = loaded_model.get_feature_count() if hasattr(loaded_model, 'get_feature_count') else len(loaded_model.feature_names_)
    except Exception:
        n_features = "N/A"
    print(f"    Model features: {n_features}")

    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            meta_data = pickle.load(f)
        print(f"    ✓ Metadata loaded")
        print(f"    Handler: {meta_data.get('handler', 'N/A')}")
        print(f"    Stock pool: {meta_data.get('stock_pool', 'N/A')}")
        print(f"    N-day: {meta_data.get('nday', 'N/A')}")
        if meta_data.get('top_k', 0) > 0:
            print(f"    Top-k features: {meta_data.get('top_k')}")
    else:
        print(f"    ⚠ Metadata file not found: {meta_path}")
        meta_data = {}

    # 将预测结果转换为 DataFrame 格式
    if isinstance(pred, pd.Series):
        pred_df = pred.to_frame("score")
    else:
        pred_df = pred

    print(f"\n[BT-1] Configuring backtest...")
    print(f"    Topk: {args.topk}")
    print(f"    N_drop: {args.n_drop}")
    print(f"    Account: ${args.account:,.0f}")
    print(f"    Rebalance Freq: every {args.rebalance_freq} day(s)")
    print(f"    Period: {TEST_START} to {TEST_END}")

    # 配置策略
    strategy_config = get_strategy_config(pred_df, args.topk, args.n_drop, args.rebalance_freq)

    # 配置执行器
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    }

    # 配置回测参数（美股市场）
    from data.stock_pools import STOCK_POOLS
    pool_symbols = STOCK_POOLS[args.stock_pool]

    backtest_config = {
        "start_time": TEST_START,
        "end_time": TEST_END,
        "account": args.account,
        "benchmark": pool_symbols,  # 使用股票池平均作为基准
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": None,  # 美股无涨跌停限制
            "deal_price": "close",
            "open_cost": 0.0005,  # 买入成本 0.05%
            "close_cost": 0.0005,  # 卖出成本 0.05%
            "min_cost": 1,  # 最小交易成本 $1
        },
    }

    print(f"\n[BT-2] Running backtest...")
    try:
        portfolio_metric_dict, indicator_dict = qlib_backtest(
            executor=executor_config,
            strategy=strategy_config,
            **backtest_config
        )

        print("    ✓ Backtest completed")

        # 分析结果
        print(f"\n[BT-3] Analyzing results...")

        for freq, (report_df, positions) in portfolio_metric_dict.items():
            print(f"\n    Frequency: {freq}")
            print(f"    Report shape: {report_df.shape}")
            print(f"    Date range: {report_df.index.min()} to {report_df.index.max()}")

            # 计算关键指标
            total_return = (report_df["return"] + 1).prod() - 1

            # 检查是否有 benchmark
            has_bench = "bench" in report_df.columns and not report_df["bench"].isna().all()
            if has_bench:
                bench_return = (report_df["bench"] + 1).prod() - 1
                excess_return = total_return - bench_return
                excess_return_series = report_df["return"] - report_df["bench"]
                analysis = risk_analysis(excess_return_series, freq=freq)
            else:
                bench_return = None
                excess_return = None
                # 对策略收益进行风险分析
                analysis = risk_analysis(report_df["return"], freq=freq)

            print(f"\n    Performance Summary:")
            print(f"    " + "-" * 50)
            print(f"    Total Return:      {total_return:>10.2%}")
            if has_bench:
                print(f"    Benchmark Return:  {bench_return:>10.2%}")
                print(f"    Excess Return:     {excess_return:>10.2%}")
            else:
                print(f"    Benchmark Return:  N/A (no benchmark)")
            print(f"    " + "-" * 50)

            if analysis is not None and not analysis.empty:
                analysis_title = "Risk Analysis (Excess Return)" if has_bench else "Risk Analysis (Strategy Return)"
                print(f"\n    {analysis_title}:")
                print(f"    " + "-" * 50)
                for metric, value in analysis.items():
                    if isinstance(value, (int, float)):
                        print(f"    {metric:<25s}: {value:>10.4f}")
                print(f"    " + "-" * 50)

            # 输出详细报告
            print(f"\n    Daily Returns Statistics:")
            print(f"    " + "-" * 50)
            print(f"    Mean Daily Return:   {report_df['return'].mean():>10.4%}")
            print(f"    Std Daily Return:    {report_df['return'].std():>10.4%}")
            print(f"    Max Daily Return:    {report_df['return'].max():>10.4%}")
            print(f"    Min Daily Return:    {report_df['return'].min():>10.4%}")
            print(f"    Total Trading Days:  {len(report_df):>10d}")
            print(f"    " + "-" * 50)

            # 保存报告
            output_path = PROJECT_ROOT / "outputs" / f"catboost_backtest_report_{freq}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            report_df.to_csv(output_path)
            print(f"\n    ✓ Report saved to: {output_path}")

            # [BT-4] 绘制净值曲线图
            print(f"\n[BT-4] Generating performance chart...")
            plot_backtest_curve(report_df, args, freq, PROJECT_ROOT, model_name="CatBoost")

            # [BT-5] 生成交易记录 CSV
            print(f"\n[BT-5] Generating trade records...")
            generate_trade_records(positions, args, freq, PROJECT_ROOT, model_name="catboost")

        # 输出交易指标
        for freq, (indicator_df, indicator_obj) in indicator_dict.items():
            if indicator_df is not None and not indicator_df.empty:
                print(f"\n    Trading Indicators ({freq}):")
                print(f"    " + "-" * 50)
                print(indicator_df.head(20).to_string(index=True))
                if len(indicator_df) > 20:
                    print(f"    ... ({len(indicator_df)} rows total)")

    except Exception as e:
        print(f"\n    ✗ Backtest failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETED")
    print("=" * 70)


def main():
    result = main_train_impl()
    if result is not None:
        model_path, dataset, pred, args = result
        if args.backtest:
            run_backtest(model_path, dataset, pred, args)


if __name__ == "__main__":
    main()
