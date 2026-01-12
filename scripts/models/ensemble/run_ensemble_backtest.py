"""
Ensemble 模型回测脚本

读取 my_models 下的 CatBoost 和 LightGBM 模型，
将预测结果 ensemble 后用于 qlib 回测。

使用 DEFAULT_TIME_SPLITS 中的时间进行 IC 计算和回测。
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
script_dir = Path(__file__).parent.parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

import argparse
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.backtest import backtest as qlib_backtest
from qlib.contrib.evaluate import risk_analysis

from utils.talib_ops import TALIB_OPS
from utils.strategy import get_strategy_config
from utils.backtest_utils import plot_backtest_curve, generate_trade_records
from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG,
    PROJECT_ROOT,
    QLIB_DATA_PATH,
    MODEL_SAVE_PATH,
    DEFAULT_TIME_SPLITS,
)


def find_latest_models(model_dir: Path) -> tuple:
    """
    在 model_dir 中查找最新的 LightGBM 和 CatBoost 模型

    Returns
    -------
    tuple
        (lgb_model_path, catboost_model_path)
    """
    lgb_models = sorted(model_dir.glob("lgb_*.txt"), key=lambda x: x.stat().st_mtime, reverse=True)
    cb_models = sorted(model_dir.glob("catboost_*.cbm"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not lgb_models:
        raise FileNotFoundError(f"No LightGBM models found in {model_dir}")
    if not cb_models:
        raise FileNotFoundError(f"No CatBoost models found in {model_dir}")

    return lgb_models[0], cb_models[0]


def load_model_and_meta(model_path: Path, model_type: str) -> tuple:
    """
    加载模型和元数据

    Parameters
    ----------
    model_path : Path
        模型文件路径
    model_type : str
        'lgb' 或 'catboost'

    Returns
    -------
    tuple
        (model, meta_data)
    """
    meta_path = model_path.with_suffix('.meta.pkl')

    # 加载模型
    if model_type == 'lgb':
        model = lgb.Booster(model_file=str(model_path))
    elif model_type == 'catboost':
        model = CatBoostRegressor()
        model.load_model(str(model_path))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 加载元数据
    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            meta_data = pickle.load(f)
    else:
        meta_data = {}

    return model, meta_data


def create_data_handler(meta_data: dict, symbols: list, time_splits: dict):
    """
    根据元数据创建 DataHandler
    """
    handler_name = meta_data.get('handler', 'alpha158')
    handler_config = HANDLER_CONFIG[handler_name]
    nday = meta_data.get('nday', 2)

    handler_kwargs = {
        'volatility_window': nday,
        'instruments': symbols,
        'start_time': time_splits['train_start'],
        'end_time': time_splits['test_end'],
        'fit_start_time': time_splits['train_start'],
        'fit_end_time': time_splits['train_end'],
        'infer_processors': [],
    }

    HandlerClass = handler_config['class']
    return HandlerClass(**handler_kwargs), handler_config


def predict_with_model(model, test_data: pd.DataFrame, model_type: str,
                       feature_names: list = None) -> pd.Series:
    """
    使用模型进行预测

    Parameters
    ----------
    model : object
        训练好的模型
    test_data : pd.DataFrame
        测试数据
    model_type : str
        'lgb' 或 'catboost'
    feature_names : list, optional
        特征名称列表（用于选择特征）

    Returns
    -------
    pd.Series
        预测结果
    """
    # 获取模型期望的特征数量
    if model_type == 'lgb':
        n_features = model.num_feature()
    else:
        n_features = len(model.feature_names_) if model.feature_names_ else test_data.shape[1]

    # 选择特征 - 优先使用列位置而非名称，以保证特征数量正确
    if feature_names and len(feature_names) == n_features:
        # 检查特征是否都存在
        available_features = [f for f in feature_names if f in test_data.columns]
        if len(available_features) == n_features:
            data_for_pred = test_data[available_features]
        else:
            # 使用前 N 列
            data_for_pred = test_data.iloc[:, :n_features]
    else:
        # 直接使用前 N 列
        data_for_pred = test_data.iloc[:, :n_features]

    print(f"    Using {data_for_pred.shape[1]} features (model expects {n_features})")

    # 预测
    pred_values = model.predict(data_for_pred.values)
    pred = pd.Series(pred_values, index=test_data.index, name='score')

    return pred


def ensemble_predictions(pred_lgb: pd.Series, pred_cb: pd.Series,
                        method: str = 'mean', weights: tuple = None) -> pd.Series:
    """
    Ensemble 两个模型的预测结果

    Parameters
    ----------
    pred_lgb : pd.Series
        LightGBM 预测结果
    pred_cb : pd.Series
        CatBoost 预测结果
    method : str
        ensemble 方法: 'mean', 'weighted', 'rank_mean'
    weights : tuple, optional
        (lgb_weight, cb_weight)，仅当 method='weighted' 时使用

    Returns
    -------
    pd.Series
        Ensemble 后的预测结果
    """
    # 对齐索引
    common_idx = pred_lgb.index.intersection(pred_cb.index)
    pred_lgb_aligned = pred_lgb.loc[common_idx]
    pred_cb_aligned = pred_cb.loc[common_idx]

    if method == 'mean':
        ensemble_pred = (pred_lgb_aligned + pred_cb_aligned) / 2
    elif method == 'weighted':
        if weights is None:
            weights = (0.5, 0.5)
        ensemble_pred = pred_lgb_aligned * weights[0] + pred_cb_aligned * weights[1]
    elif method == 'rank_mean':
        # 将预测值转换为排名，然后平均
        rank_lgb = pred_lgb_aligned.groupby(level='datetime').rank(pct=True)
        rank_cb = pred_cb_aligned.groupby(level='datetime').rank(pct=True)
        ensemble_pred = (rank_lgb + rank_cb) / 2
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    ensemble_pred.name = 'score'
    return ensemble_pred


def calculate_ic(pred: pd.Series, label: pd.Series) -> tuple:
    """
    计算 IC 和 ICIR

    Returns
    -------
    tuple
        (ic_series, mean_ic, ic_std, icir)
    """
    # 对齐索引
    common_idx = pred.index.intersection(label.index)
    pred_aligned = pred.loc[common_idx]
    label_aligned = label.loc[common_idx]

    # 去除 NaN
    valid_idx = ~(pred_aligned.isna() | label_aligned.isna())
    pred_clean = pred_aligned[valid_idx]
    label_clean = label_aligned[valid_idx]

    # 按日期计算 IC
    ic = pred_clean.groupby(level="datetime").apply(
        lambda x: x.corr(label_clean.loc[x.index]) if len(x) > 1 else np.nan
    )
    ic = ic.dropna()

    mean_ic = ic.mean()
    ic_std = ic.std()
    icir = mean_ic / ic_std if ic_std > 0 else 0

    return ic, mean_ic, ic_std, icir


def run_ensemble_backtest(pred: pd.Series, args, time_splits: dict, handler_name: str = "ensemble"):
    """
    执行 ensemble 模型回测
    """
    print("\n" + "=" * 70)
    print("BACKTEST with TopkDropoutStrategy (Ensemble)")
    print("=" * 70)

    # 转换为 DataFrame 格式
    pred_df = pred.to_frame("score")

    print(f"\n[BT-1] Configuring backtest...")
    print(f"    Topk: {args.topk}")
    print(f"    N_drop: {args.n_drop}")
    print(f"    Account: ${args.account:,.0f}")
    print(f"    Rebalance Freq: every {args.rebalance_freq} day(s)")
    print(f"    Period: {time_splits['test_start']} to {time_splits['test_end']}")

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

    # 配置回测参数
    pool_symbols = STOCK_POOLS[args.stock_pool]

    backtest_config = {
        "start_time": time_splits['test_start'],
        "end_time": time_splits['test_end'],
        "account": args.account,
        "benchmark": pool_symbols,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": None,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0005,
            "min_cost": 1,
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
            _analyze_backtest_results(report_df, positions, freq, args, time_splits, handler_name)

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


def _analyze_backtest_results(report_df: pd.DataFrame, positions, freq: str,
                              args, time_splits: dict, handler_name: str = "ensemble"):
    """分析回测结果"""
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
    output_path = PROJECT_ROOT / "outputs" / f"ensemble_backtest_report_{freq}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path)
    print(f"\n    ✓ Report saved to: {output_path}")

    # 绘制净值曲线图
    print(f"\n[BT-4] Generating performance chart...")
    # 为 plot_backtest_curve 添加必需的 handler 属性
    if not hasattr(args, 'handler'):
        args.handler = handler_name
    plot_backtest_curve(report_df, args, freq, PROJECT_ROOT, model_name="Ensemble")

    # 生成交易记录 CSV
    print(f"\n[BT-5] Generating trade records...")
    generate_trade_records(positions, args, freq, PROJECT_ROOT, model_name="ensemble")


def main():
    parser = argparse.ArgumentParser(
        description='Ensemble Model Backtest (LightGBM + CatBoost)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ensemble_backtest.py
  python run_ensemble_backtest.py --ensemble-method rank_mean
  python run_ensemble_backtest.py --lgb-weight 0.6 --cb-weight 0.4
  python run_ensemble_backtest.py --topk 15 --n-drop 3 --rebalance-freq 5
"""
    )

    # 模型路径参数
    parser.add_argument('--lgb-model', type=str, default=None,
                        help='LightGBM model path (default: auto-detect latest)')
    parser.add_argument('--cb-model', type=str, default=None,
                        help='CatBoost model path (default: auto-detect latest)')

    # Ensemble 参数
    parser.add_argument('--ensemble-method', type=str, default='mean',
                        choices=['mean', 'weighted', 'rank_mean'],
                        help='Ensemble method (default: mean)')
    parser.add_argument('--lgb-weight', type=float, default=0.5,
                        help='LightGBM weight for weighted ensemble (default: 0.5)')
    parser.add_argument('--cb-weight', type=float, default=0.5,
                        help='CatBoost weight for weighted ensemble (default: 0.5)')

    # 回测参数
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'],
                        help='Stock pool (default: sp500)')
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of stocks to hold (default: 10)')
    parser.add_argument('--n-drop', type=int, default=2,
                        help='Number of stocks to drop/replace each day (default: 2)')
    parser.add_argument('--account', type=float, default=1000000,
                        help='Initial account value (default: 1000000)')
    parser.add_argument('--rebalance-freq', type=int, default=1,
                        help='Rebalance frequency in days (default: 1)')

    args = parser.parse_args()

    # 使用 DEFAULT_TIME_SPLITS
    time_splits = DEFAULT_TIME_SPLITS.copy()

    print("=" * 70)
    print("Ensemble Model Backtest (LightGBM + CatBoost)")
    print("=" * 70)
    print(f"Ensemble Method: {args.ensemble_method}")
    if args.ensemble_method == 'weighted':
        print(f"Weights: LGB={args.lgb_weight}, CB={args.cb_weight}")
    print(f"Stock Pool: {args.stock_pool}")
    print(f"Time Splits:")
    print(f"  Train: {time_splits['train_start']} to {time_splits['train_end']}")
    print(f"  Valid: {time_splits['valid_start']} to {time_splits['valid_end']}")
    print(f"  Test:  {time_splits['test_start']} to {time_splits['test_end']}")
    print("=" * 70)

    # 查找模型
    print("\n[1] Finding models...")
    if args.lgb_model:
        lgb_path = Path(args.lgb_model)
    else:
        lgb_path, _ = find_latest_models(MODEL_SAVE_PATH)

    if args.cb_model:
        cb_path = Path(args.cb_model)
    else:
        _, cb_path = find_latest_models(MODEL_SAVE_PATH)

    print(f"    LightGBM model: {lgb_path.name}")
    print(f"    CatBoost model: {cb_path.name}")

    # 加载模型和元数据
    print("\n[2] Loading models and metadata...")
    lgb_model, lgb_meta = load_model_and_meta(lgb_path, 'lgb')
    cb_model, cb_meta = load_model_and_meta(cb_path, 'catboost')

    print(f"    LightGBM: handler={lgb_meta.get('handler', 'N/A')}, "
          f"nday={lgb_meta.get('nday', 'N/A')}, features={lgb_model.num_feature()}")
    print(f"    CatBoost: handler={cb_meta.get('handler', 'N/A')}, "
          f"nday={cb_meta.get('nday', 'N/A')}, "
          f"features={len(cb_model.feature_names_) if cb_model.feature_names_ else 'N/A'}")

    # 使用第一个模型的元数据来创建 handler（假设两个模型使用相同配置）
    meta_data = lgb_meta if lgb_meta else cb_meta

    # 初始化 Qlib
    print("\n[3] Initializing Qlib...")
    handler_name = meta_data.get('handler', 'alpha158')
    handler_config = HANDLER_CONFIG[handler_name]

    if handler_config['use_talib']:
        qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US, custom_ops=TALIB_OPS)
        print("    ✓ Qlib initialized with TA-Lib custom operators")
    else:
        qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US)
        print("    ✓ Qlib initialized")

    # 创建数据集
    print("\n[4] Creating dataset...")
    symbols = STOCK_POOLS[args.stock_pool]
    handler, _ = create_data_handler(meta_data, symbols, time_splits)

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (time_splits['train_start'], time_splits['train_end']),
            "valid": (time_splits['valid_start'], time_splits['valid_end']),
            "test": (time_splits['test_start'], time_splits['test_end']),
        }
    )
    print(f"    ✓ Dataset created")

    # 准备测试数据
    print("\n[5] Preparing test data...")
    test_data = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_L)
    test_label = dataset.prepare("test", col_set="label")
    print(f"    Test data shape: {test_data.shape}")
    print(f"    Test label shape: {test_label.shape}")

    # 生成预测
    print("\n[6] Generating predictions...")

    # LightGBM 预测
    lgb_feature_names = lgb_meta.get('feature_names', [])
    pred_lgb = predict_with_model(lgb_model, test_data, 'lgb', lgb_feature_names)
    print(f"    LightGBM predictions: {len(pred_lgb)} samples, "
          f"range=[{pred_lgb.min():.4f}, {pred_lgb.max():.4f}]")

    # CatBoost 预测
    cb_feature_names = cb_meta.get('feature_names', [])
    pred_cb = predict_with_model(cb_model, test_data, 'catboost', cb_feature_names)
    print(f"    CatBoost predictions: {len(pred_cb)} samples, "
          f"range=[{pred_cb.min():.4f}, {pred_cb.max():.4f}]")

    # Ensemble 预测
    print(f"\n[7] Ensembling predictions ({args.ensemble_method})...")
    weights = (args.lgb_weight, args.cb_weight) if args.ensemble_method == 'weighted' else None
    pred_ensemble = ensemble_predictions(pred_lgb, pred_cb, args.ensemble_method, weights)
    print(f"    Ensemble predictions: {len(pred_ensemble)} samples, "
          f"range=[{pred_ensemble.min():.4f}, {pred_ensemble.max():.4f}]")

    # 计算各模型的 IC
    print("\n[8] Calculating IC metrics...")
    label_series = test_label['LABEL0']

    _, lgb_ic, lgb_std, lgb_icir = calculate_ic(pred_lgb, label_series)
    _, cb_ic, cb_std, cb_icir = calculate_ic(pred_cb, label_series)
    _, ens_ic, ens_std, ens_icir = calculate_ic(pred_ensemble, label_series)

    print("\n    ╔════════════════════════════════════════════════════════════╗")
    print("    ║              Information Coefficient (IC) Comparison       ║")
    print("    ╠════════════════════════════════════════════════════════════╣")
    print(f"    ║  Model       │   Mean IC  │   IC Std   │    ICIR    ║")
    print("    ╠════════════════════════════════════════════════════════════╣")
    print(f"    ║  LightGBM    │  {lgb_ic:>8.4f}  │  {lgb_std:>8.4f}  │  {lgb_icir:>8.4f}  ║")
    print(f"    ║  CatBoost    │  {cb_ic:>8.4f}  │  {cb_std:>8.4f}  │  {cb_icir:>8.4f}  ║")
    print(f"    ║  Ensemble    │  {ens_ic:>8.4f}  │  {ens_std:>8.4f}  │  {ens_icir:>8.4f}  ║")
    print("    ╚════════════════════════════════════════════════════════════╝")

    # IC 提升
    ic_improvement = (ens_ic - max(lgb_ic, cb_ic)) / abs(max(lgb_ic, cb_ic)) * 100 if max(lgb_ic, cb_ic) != 0 else 0
    icir_improvement = (ens_icir - max(lgb_icir, cb_icir)) / abs(max(lgb_icir, cb_icir)) * 100 if max(lgb_icir, cb_icir) != 0 else 0

    print(f"\n    IC improvement over best single model:   {ic_improvement:>+.2f}%")
    print(f"    ICIR improvement over best single model: {icir_improvement:>+.2f}%")

    # 运行回测
    print("\n[9] Running backtest...")
    handler_name = meta_data.get('handler', 'ensemble')
    run_ensemble_backtest(pred_ensemble, args, time_splits, handler_name)

    print("\n" + "=" * 70)
    print("✓ Ensemble Backtest Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
