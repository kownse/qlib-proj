"""
训练流程共用工具函数

包含命令行参数解析、Qlib 初始化、数据准备、特征验证等共用逻辑
"""

import os

# 关键: 在导入其他库之前设置环境变量，限制线程数避免 TA-Lib 内存冲突
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import argparse
import pickle
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_US
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from utils.talib_ops import TALIB_OPS
from data.stock_pools import STOCK_POOLS

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


def create_argument_parser(model_name: str, example_script: str) -> argparse.ArgumentParser:
    """
    创建通用的命令行参数解析器

    Parameters
    ----------
    model_name : str
        模型名称，用于帮助文本
    example_script : str
        示例脚本名称

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=f'Qlib Stock Price Volatility Prediction ({model_name})',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=get_handler_epilog() + f"""
Examples:
  python {example_script} --stock-pool sp100 --handler alpha158
  python {example_script} --stock-pool sp500 --handler alpha360-pandas
  python {example_script} --stock-pool sp100 --handler alpha158-news --news-features core
  python {example_script} --stock-pool sp100 --max-train  # Max training data for deployment
  python {example_script} --stock-pool sp100 --backtest --rebalance-freq 5  # Rebalance every 5 days
"""
    )

    # 基础参数
    parser.add_argument('--nday', type=int, default=5,
                        help='Volatility prediction window in days (default: 5)')
    parser.add_argument('--handler', type=str, default='alpha158',
                        choices=list(HANDLER_CONFIG.keys()),
                        help='Handler type (default: alpha158)')
    parser.add_argument('--stock-pool', type=str, default='test',
                        choices=['test', 'tech', 'sp100', 'sp500'],
                        help='Stock pool: test (10), tech (~30), sp100 (100), sp500 (~500) (default: test)')
    parser.add_argument('--top-k', type=int, default=0,
                        help='Top features for retraining (0 = disable feature selection)')

    # News 特定参数
    parser.add_argument('--news-features', type=str, default='core',
                        choices=['all', 'sentiment', 'stats', 'core'],
                        help='News feature set (only for alpha158-news handler)')
    parser.add_argument('--news-rolling', action='store_true',
                        help='Add rolling news features (only for alpha158-news handler)')

    # 时间划分参数
    parser.add_argument('--max-train', action='store_true',
                        help='Use maximum training data (train to 2025-09, valid 2025-10-12, no test) for deployment')

    # 回测参数
    parser.add_argument('--backtest', action='store_true',
                        help='Run backtest after training using TopkDropoutStrategy')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to a pre-trained model. If provided, skip training and run backtest directly')
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of stocks to hold in TopkDropoutStrategy (default: 10)')
    parser.add_argument('--n-drop', type=int, default=2,
                        help='Number of stocks to drop/replace each day (default: 2)')
    parser.add_argument('--account', type=float, default=1000000,
                        help='Initial account value for backtest (default: 1000000)')
    parser.add_argument('--rebalance-freq', type=int, default=1,
                        help='Rebalance frequency in days (default: 1, i.e., rebalance every day)')

    # 策略类型参数
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss',
                                 'mvo', 'rp', 'gmv', 'inv'],
                        help='Strategy type: topk (default), dynamic_risk, vol_stoploss, '
                             'mvo (mean-variance), rp (risk parity), gmv (min variance), inv (inverse vol)')

    # Sample weighting 参数
    parser.add_argument('--sample-weight-halflife', type=float, default=0,
                        help='Time-decay sample weight half-life in years. '
                             '0 = no decay (default). Try 5 for 5-year half-life.')

    # 动态风险策略参数
    parser.add_argument('--risk-lookback', type=int, default=20,
                        help='Lookback days for volatility/momentum calculation (default: 20)')
    parser.add_argument('--drawdown-threshold', type=float, default=-0.10,
                        help='[dynamic_risk] Drawdown threshold for high risk (default: -0.10)')
    parser.add_argument('--momentum-threshold', type=float, default=0.03,
                        help='[dynamic_risk] Momentum threshold for trend detection (default: 0.03)')
    parser.add_argument('--risk-high', type=float, default=0.50,
                        help='Position ratio at high risk/volatility (default: 0.50)')
    parser.add_argument('--risk-medium', type=float, default=0.75,
                        help='Position ratio at medium risk/volatility (default: 0.75)')
    parser.add_argument('--risk-normal', type=float, default=0.95,
                        help='Normal position ratio (default: 0.95)')
    parser.add_argument('--market-proxy', type=str, default='AAPL',
                        help='Market proxy symbol for risk calculation (default: AAPL)')

    # vol_stoploss 策略特有参数
    parser.add_argument('--vol-high', type=float, default=0.35,
                        help='[vol_stoploss] High volatility threshold (annualized, default: 0.35)')
    parser.add_argument('--vol-medium', type=float, default=0.25,
                        help='[vol_stoploss] Medium volatility threshold (annualized, default: 0.25)')
    parser.add_argument('--stop-loss', type=float, default=-0.15,
                        help='[vol_stoploss] Stop loss threshold per stock (default: -0.15, i.e., -15%%)')
    parser.add_argument('--no-sell-after-drop', type=float, default=-0.20,
                        help='[vol_stoploss] Do not sell if already dropped more than this (default: -0.20)')

    # Portfolio optimization 策略参数 (mvo/rp/gmv/inv)
    parser.add_argument('--opt-lamb', type=float, default=2.0,
                        help='[mvo] Risk aversion (higher=safer, default: 2.0)')
    parser.add_argument('--opt-delta', type=float, default=0.2,
                        help='[mvo/rp/gmv] Max turnover per rebalance (default: 0.2)')
    parser.add_argument('--opt-alpha', type=float, default=0.01,
                        help='[mvo/rp/gmv] L2 regularization to prevent concentration (default: 0.01)')
    parser.add_argument('--cov-lookback', type=int, default=60,
                        help='[mvo/rp/gmv/inv] Days of history for covariance estimation (default: 60)')
    parser.add_argument('--max-weight', type=float, default=0.0,
                        help='[mvo/rp/gmv/inv] Max weight per stock, 0=no limit (default: 0, try 0.15)')

    return parser


def get_time_splits(use_max_train: bool) -> dict:
    """
    获取时间划分配置

    Parameters
    ----------
    use_max_train : bool
        是否使用最大训练数据模式

    Returns
    -------
    dict
        包含 train_start, train_end, valid_start, valid_end, test_start, test_end 的字典
    """
    if use_max_train:
        return MAX_TRAIN_TIME_SPLITS.copy()
    return DEFAULT_TIME_SPLITS.copy()


def print_training_header(model_name: str, args, symbols: list, handler_config: dict, time_splits: dict):
    """打印训练开始的头部信息"""
    print("=" * 70)
    print(f"{model_name} {args.nday}-Day Stock Price Volatility Prediction")
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
        strategy_info = f", strategy={args.strategy}" if args.strategy != "topk" else ""
        print(f"Backtest: Enabled (topk={args.topk}, n_drop={args.n_drop}{rebalance_info}{strategy_info})")
        if args.strategy == "dynamic_risk":
            print(f"  Dynamic Risk: lookback={args.risk_lookback}, drawdown={args.drawdown_threshold:.0%}, momentum={args.momentum_threshold:.0%}")
    print("=" * 70)


def init_qlib(use_talib: bool):
    """
    初始化 Qlib

    Parameters
    ----------
    use_talib : bool
        是否使用 TA-Lib 自定义算子

    Note
    ----
    当使用 TA-Lib 时，需要设置 kernels=1 和 joblib_backend=None 来避免内存冲突。
    - kernels=1: 禁用多进程数据加载
    - joblib_backend=None: 使用 loky 而非 multiprocessing，loky 使用 spawn 而非 fork
    否则会出现 "free(): invalid pointer" 和 "corrupted size vs. prev_size" 错误。
    """
    print("\n[1] Initializing Qlib...")
    if use_talib:
        # kernels=1 避免多进程数据加载
        # joblib_backend=None 使用 loky (spawn) 而非 multiprocessing (fork)
        # skip_if_reg=True 避免重复初始化
        qlib.init(
            provider_uri=str(QLIB_DATA_PATH),
            region=REG_US,
            custom_ops=TALIB_OPS,
            kernels=1,
            joblib_backend=None,  # 使用 loky 避免 fork 与 TA-Lib C库冲突
            skip_if_reg=True
        )
        print("    ✓ Qlib initialized with TA-Lib (kernels=1, loky backend)")
    else:
        qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US, skip_if_reg=True)
        print("    ✓ Qlib initialized")


def check_data_availability(time_splits: dict):
    """
    检查数据可用性

    Parameters
    ----------
    time_splits : dict
        时间划分配置
    """
    print("\n[2] Checking data availability...")
    instruments = D.instruments(market="all")
    available_instruments = list(D.list_instruments(instruments))
    print(f"    ✓ Available instruments: {len(available_instruments)}")

    # 测试读取一只股票
    test_df = D.features(
        instruments=["AAPL"],
        fields=["$close", "$volume"],
        start_time=time_splits['test_start'],
        end_time=time_splits['test_end']
    )
    print(f"    ✓ AAPL sample data shape: {test_df.shape}")
    print(f"    ✓ Date range: {test_df.index.get_level_values('datetime').min().date()} to {test_df.index.get_level_values('datetime').max().date()}")


def create_data_handler(args, handler_config: dict, symbols: list, time_splits: dict):
    """
    创建 DataHandler

    Parameters
    ----------
    args : argparse.Namespace
        命令行参数
    handler_config : dict
        Handler 配置
    symbols : list
        股票列表
    time_splits : dict
        时间划分配置

    Returns
    -------
    DataHandler
    """
    print(f"\n[3] Creating DataHandler with {args.nday}-day volatility label...")
    print(f"    Features: {handler_config['description']}")
    print(f"    Label: {args.nday}-day realized volatility")

    handler_kwargs = {
        'volatility_window': args.nday,
        'instruments': symbols,
        'start_time': time_splits['train_start'],
        'end_time': time_splits['test_end'],
        'fit_start_time': time_splits['train_start'],
        'fit_end_time': time_splits['train_end'],
        'infer_processors': [],
    }

    # News handler 需要额外参数
    if args.handler == 'alpha158-news':
        handler_kwargs['news_data_path'] = str(NEWS_DATA_PATH) if NEWS_DATA_PATH.exists() else None
        handler_kwargs['news_features'] = args.news_features
        handler_kwargs['add_news_rolling'] = args.news_rolling
        print(f"    News data path: {NEWS_DATA_PATH}")

    # Apply default kwargs from handler registry (e.g., sector_features for sector handlers)
    if 'default_kwargs' in handler_config:
        for key, value in handler_config['default_kwargs'].items():
            if key not in handler_kwargs:
                handler_kwargs[key] = value
        print(f"    Default kwargs: {handler_config['default_kwargs']}")

    # 创建 handler
    HandlerClass = handler_config['class']
    handler = HandlerClass(**handler_kwargs)

    if args.handler == 'alpha158-news':
        print(f"    ✓ DataHandler created with news features: {handler.get_news_feature_names()}")
    else:
        print(f"    ✓ DataHandler created: {args.handler}")

    return handler


def create_dataset(handler, time_splits: dict) -> DatasetH:
    """
    创建 Dataset

    Parameters
    ----------
    handler : DataHandler
        数据处理器
    time_splits : dict
        时间划分配置

    Returns
    -------
    DatasetH
    """
    print("\n[4] Creating Dataset...")
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (time_splits['train_start'], time_splits['train_end']),
            "valid": (time_splits['valid_start'], time_splits['valid_end']),
            "test": (time_splits['test_start'], time_splits['test_end']),
        }
    )
    return dataset


def analyze_features(dataset: DatasetH) -> tuple:
    """
    分析特征，识别有效和无效列

    Parameters
    ----------
    dataset : DatasetH
        数据集

    Returns
    -------
    tuple
        (train_data, valid_cols, dropped_cols)
    """
    train_data = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
    print(f"    ✓ Train features shape: {train_data.shape}")
    print(f"      (samples × features)")

    valid_cols = []
    dropped_cols = []
    for col in train_data.columns:
        col_data = train_data[col]
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
            valid_cols.append(col)

    if dropped_cols:
        print(f"    ⚠ {len(dropped_cols)} features will be dropped (all NaN or constant):")
        for col in dropped_cols[:5]:
            print(f"      - {col}")
        if len(dropped_cols) > 5:
            print(f"      ... and {len(dropped_cols) - 5} more")
    print(f"    ✓ Valid features: {len(valid_cols)}")

    return train_data, valid_cols, dropped_cols


def analyze_label_distribution(dataset: DatasetH):
    """
    分析标签分布

    Parameters
    ----------
    dataset : DatasetH
        数据集
    """
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


def print_feature_importance(importance_df: pd.DataFrame, title: str = "Top 20 Features by Importance"):
    """
    打印特征重要性

    Parameters
    ----------
    importance_df : pd.DataFrame
        特征重要性 DataFrame，包含 'feature' 和 'importance' 列
    title : str
        标题
    """
    print(f"\n    {title}:")
    print("    " + "-" * 50)
    for i, row in importance_df.head(20).iterrows():
        print(f"    {importance_df.index.get_loc(i)+1:3d}. {row['feature']:<40s} {row['importance']:>10.2f}")
    print("    " + "-" * 50)
    print(f"    Total features: {len(importance_df)}")


def save_model_with_meta(model, model_path: Path, meta_data: dict, save_func=None):
    """
    保存模型和元数据

    Parameters
    ----------
    model : object
        训练好的模型
    model_path : Path
        模型保存路径
    meta_data : dict
        元数据字典
    save_func : callable, optional
        自定义保存函数，接受 (model, path) 参数。如果为 None，则使用 model.save_model
    """
    model_path = Path(model_path)

    # 保存模型
    if save_func:
        save_func(model, str(model_path))
    else:
        model.save_model(str(model_path))

    # 保存元数据
    meta_path = model_path.with_suffix('.meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta_data, f)

    print(f"    ✓ Model saved to: {model_path}")
    print(f"    ✓ Metadata saved to: {meta_path}")


def create_meta_data(args, handler_config: dict, time_splits: dict, feature_names: list,
                     model_type: str, top_k: int = 0) -> dict:
    """
    创建模型元数据

    Parameters
    ----------
    args : argparse.Namespace
        命令行参数
    handler_config : dict
        Handler 配置
    time_splits : dict
        时间划分配置
    feature_names : list
        特征名称列表
    model_type : str
        模型类型
    top_k : int
        Top-K 特征数量

    Returns
    -------
    dict
    """
    return {
        'handler': args.handler,
        'stock_pool': args.stock_pool,
        'nday': args.nday,
        'top_k': top_k,
        'feature_names': feature_names,
        'train_start': time_splits['train_start'],
        'train_end': time_splits['train_end'],
        'valid_start': time_splits['valid_start'],
        'valid_end': time_splits['valid_end'],
        'test_start': time_splits['test_start'],
        'test_end': time_splits['test_end'],
        'use_talib': handler_config['use_talib'],
        'model_type': model_type,
    }


def generate_model_filename(model_type: str, args, top_k: int = 0, extension: str = '.txt') -> str:
    """
    生成模型文件名

    Parameters
    ----------
    model_type : str
        模型类型
    args : argparse.Namespace
        命令行参数
    top_k : int
        Top-K 特征数量
    extension : str
        文件扩展名

    Returns
    -------
    str
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if top_k > 0:
        return f"{model_type}_{args.handler}_{args.stock_pool}_{args.nday}d_topk{top_k}_{timestamp}{extension}"
    return f"{model_type}_{args.handler}_{args.stock_pool}_{args.nday}d_{timestamp}{extension}"


def prepare_test_data_for_prediction(dataset: DatasetH, num_model_features: int):
    """
    准备测试数据用于预测

    Parameters
    ----------
    dataset : DatasetH
        数据集
    num_model_features : int
        模型特征数量

    Returns
    -------
    pd.DataFrame
        过滤后的测试数据
    """
    print("\n[8] Generating predictions...")
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

    print(f"    Model expects {num_model_features} features")

    # Select only columns present in both, up to model_n_features
    available_cols = [c for c in test_data.columns if c in train_cols_unique[:num_model_features]]

    # If we have enough columns, use them directly
    if len(available_cols) >= num_model_features:
        test_data_filtered = test_data.iloc[:, :num_model_features]
    else:
        print(f"    ⚠ Only {len(available_cols)} columns available, need {num_model_features}")
        test_data_filtered = test_data.iloc[:, :num_model_features].copy()
        for i in range(len(available_cols), num_model_features):
            test_data_filtered[f"_missing_{i}"] = np.nan

    print(f"    Filtered test features: {test_data_filtered.shape[1]} (expected: {num_model_features})")
    return test_data_filtered


def load_catboost_model(model_path):
    """加载 CatBoost 模型"""
    from catboost import CatBoostRegressor
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


def print_prediction_stats(pred: pd.Series):
    """打印预测结果统计"""
    print(f"    ✓ Prediction shape: {pred.shape}")
    print(f"    ✓ Prediction range: [{pred.min():.4f}, {pred.max():.4f}]")


# ============================================================================
# 树模型通用基础设施
# ============================================================================

def load_lightgbm_model(model_path):
    """加载 LightGBM 模型"""
    import lightgbm as lgb
    return lgb.Booster(model_file=str(model_path))


def get_lightgbm_feature_count(model):
    """获取 LightGBM 模型特征数量"""
    return model.num_feature()


def load_xgboost_model(model_path):
    """加载 XGBoost 模型"""
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model(str(model_path))
    return model


def get_xgboost_feature_count(model):
    """获取 XGBoost 模型特征数量"""
    return model.num_features()


def prepare_top_k_data(dataset, importance_df, top_k):
    """
    从 importance_df 中选择 top-k 特征，准备重训练所需的数据。

    Parameters
    ----------
    dataset : DatasetH
        数据集
    importance_df : pd.DataFrame
        特征重要性 DataFrame
    top_k : int
        要选择的特征数量

    Returns
    -------
    tuple
        (top_features, train_data, valid_data, test_data, train_label, valid_label)
    """
    print(f"\n[8] Feature Selection and Retraining...")
    print(f"    Selecting top {top_k} features...")

    top_features = importance_df.head(top_k)['feature'].tolist()
    print(f"    Selected features: {top_features[:10]}{'...' if len(top_features) > 10 else ''}")

    train_data = dataset.prepare("train", col_set="feature")[top_features]
    valid_data = dataset.prepare("valid", col_set="feature")[top_features]
    test_data = dataset.prepare("test", col_set="feature")[top_features]

    train_label = dataset.prepare("train", col_set="label")
    valid_label = dataset.prepare("valid", col_set="label")

    print(f"    ✓ Selected train features shape: {train_data.shape}")

    return top_features, train_data, valid_data, test_data, train_label, valid_label


def print_retrained_importance(feature_names, importance_values):
    """
    打印重新训练后的特征重要性。

    Parameters
    ----------
    feature_names : list
        特征名称列表
    importance_values : array-like
        特征重要性值

    Returns
    -------
    pd.DataFrame
    """
    print("\n    Feature Importance after Retraining:")
    print("    " + "-" * 50)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=False)

    for i, row in importance_df.iterrows():
        print(f"    {importance_df.index.get_loc(i)+1:3d}. {row['feature']:<40s} {row['importance']:>10.2f}")
    print("    " + "-" * 50)

    return importance_df


@dataclass
class TreeModelAdapter:
    """
    树模型适配器 - 封装模型特定操作，使通用训练流程与模型实现解耦。

    Callbacks
    ---------
    train_model : (dataset, valid_cols, **extra_kwargs) ->
        (model_or_wrapper, feature_names, importance_df, num_features)
    predict : (raw_model, test_data_df) -> np.ndarray
    retrain_predict : (dataset, importance_df, args, **extra_kwargs) ->
        (raw_model, top_features, test_pred)
    load_model : (path) -> raw_model
    get_feature_count : (raw_model) -> int
    unwrap_model : (wrapper) -> raw_model, None means model is already raw
    add_arguments : (parser) -> None, add model-specific CLI arguments
    post_parse_args : (args, handler_config, symbols) -> 5-tuple | dict | None
        5-tuple: short-circuit return (e.g., CatBoost --params-file mode)
        dict: extra_kwargs passed to train_model/retrain_predict
        None: no extra kwargs
    """
    model_name: str
    model_prefix: str
    model_extension: str
    model_type: str
    script_name: str

    train_model: Callable
    predict: Callable
    retrain_predict: Callable
    load_model: Callable
    get_feature_count: Callable

    unwrap_model: Optional[Callable] = None
    add_arguments: Optional[Callable] = None
    post_parse_args: Optional[Callable] = None


def tree_main_train_impl(adapter, args):
    """
    通用树模型训练流程。

    Parameters
    ----------
    adapter : TreeModelAdapter
    args : argparse.Namespace

    Returns
    -------
    tuple
        (model_path, dataset, test_pred, args, time_splits)
    """
    from utils.utils import evaluate_model

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    # 处理 post_parse_args
    extra_kwargs = {}
    if adapter.post_parse_args:
        result = adapter.post_parse_args(args, handler_config, symbols)
        if isinstance(result, tuple) and len(result) == 5:
            return result  # Short-circuit (e.g., CatBoost --params-file)
        elif isinstance(result, dict):
            extra_kwargs = result

    time_splits = get_time_splits(args.max_train)

    # 打印头部信息
    print_training_header(adapter.model_name, args, symbols, handler_config, time_splits)

    # 初始化和数据准备
    init_qlib(handler_config['use_talib'])
    check_data_availability(time_splits)
    handler = create_data_handler(args, handler_config, symbols, time_splits)
    dataset = create_dataset(handler, time_splits)
    train_data, valid_cols, dropped_cols = analyze_features(dataset)
    analyze_label_distribution(dataset)

    # 训练模型
    model_wrapper, feature_names, importance_df, num_features = adapter.train_model(
        dataset, valid_cols, **extra_kwargs)

    # 保存特征重要性
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    importance_filename = generate_model_filename(
        f"{adapter.model_prefix}_importance", args, 0, ".csv")
    importance_path = outputs_dir / importance_filename
    importance_df.to_csv(importance_path, index=False)
    print(f"    Feature importance saved to: {importance_path}")

    # 特征选择和重新训练
    if args.top_k > 0 and args.top_k < len(feature_names):
        retrained_model, top_features, test_pred = adapter.retrain_predict(
            dataset, importance_df, args, **extra_kwargs)

        # 评估
        print(f"\n[10] Evaluation with selected features...")
        evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

        # 保存模型
        print(f"\n[11] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_filename = generate_model_filename(
            adapter.model_prefix, args, args.top_k, adapter.model_extension)
        model_path = MODEL_SAVE_PATH / model_filename
        meta_data = create_meta_data(
            args, handler_config, time_splits, top_features, adapter.model_type, args.top_k)
        save_model_with_meta(retrained_model, model_path, meta_data)

        return model_path, dataset, test_pred, args, time_splits
    else:
        # 原始模型预测
        raw_model = adapter.unwrap_model(model_wrapper) if adapter.unwrap_model else model_wrapper
        test_data_filtered = prepare_test_data_for_prediction(dataset, num_features)

        pred_values = adapter.predict(raw_model, test_data_filtered)
        test_pred = pd.Series(pred_values, index=test_data_filtered.index, name='score')
        print_prediction_stats(test_pred)

        # 评估
        print(f"\n[9] Evaluation...")
        evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

        # 保存模型
        print(f"\n[10] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_filename = generate_model_filename(
            adapter.model_prefix, args, 0, adapter.model_extension)
        model_path = MODEL_SAVE_PATH / model_filename
        meta_data = create_meta_data(
            args, handler_config, time_splits, valid_cols, adapter.model_type, 0)
        save_model_with_meta(raw_model, model_path, meta_data)

        return model_path, dataset, test_pred, args, time_splits


def tree_backtest_only_impl(adapter, args):
    """
    跳过训练，直接加载模型进行预测和回测。

    Parameters
    ----------
    adapter : TreeModelAdapter
    args : argparse.Namespace

    Returns
    -------
    tuple or None
        (model_path, dataset, test_pred, args, time_splits)
    """
    from utils.utils import evaluate_model

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

    # 创建数据集
    print(f"\n[3] Creating dataset for prediction...")
    handler = create_data_handler(args, handler_config, symbols, time_splits)
    dataset = create_dataset(handler, time_splits)

    # 加载模型
    print(f"\n[4] Loading model...")
    model = adapter.load_model(model_path)
    num_features = adapter.get_feature_count(model)
    print(f"    ✓ Model loaded, features: {num_features}")

    # 准备测试数据
    print(f"\n[5] Preparing test data...")
    feature_names = meta_data.get('feature_names', [])
    top_k = meta_data.get('top_k', 0)

    if top_k > 0 and feature_names:
        test_data = dataset.prepare("test", col_set="feature")
        available_features = [f for f in feature_names if f in test_data.columns]
        if len(available_features) < len(feature_names):
            print(f"    ⚠ Some features missing: {len(available_features)}/{len(feature_names)}")
        test_data_filtered = test_data[available_features]
    else:
        test_data_filtered = prepare_test_data_for_prediction(dataset, num_features)

    print(f"    Test data shape: {test_data_filtered.shape}")

    # 生成预测
    print(f"\n[6] Generating predictions...")
    pred_values = adapter.predict(model, test_data_filtered)
    test_pred = pd.Series(pred_values, index=test_data_filtered.index, name='score')
    print_prediction_stats(test_pred)

    # 评估
    print(f"\n[7] Evaluation...")
    evaluate_model(dataset, test_pred, PROJECT_ROOT, meta_data.get('nday', args.nday))

    return model_path, dataset, test_pred, args, time_splits


def tree_main(adapter):
    """
    通用树模型入口函数。

    处理命令行参数解析，分发到训练或回测流程，并在需要时运行回测。

    Parameters
    ----------
    adapter : TreeModelAdapter
    """
    from .backtest import run_backtest

    parser = create_argument_parser(adapter.model_name, adapter.script_name)
    if adapter.add_arguments:
        adapter.add_arguments(parser)
    args = parser.parse_args()

    if args.model_path:
        if not args.backtest:
            print("Warning: --model-path provided but --backtest not set. Enabling backtest automatically.")
            args.backtest = True
        result = tree_backtest_only_impl(adapter, args)
    else:
        result = tree_main_train_impl(adapter, args)

    if result is not None:
        model_path, dataset, pred, args, time_splits = result
        if args.backtest:
            run_backtest(
                model_path, dataset, pred, args, time_splits,
                model_name=adapter.model_name,
                load_model_func=adapter.load_model,
                get_feature_count_func=adapter.get_feature_count
            )
