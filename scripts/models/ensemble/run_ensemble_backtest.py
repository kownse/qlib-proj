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
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from utils.talib_ops import TALIB_OPS
from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG,
    QLIB_DATA_PATH,
    MODEL_SAVE_PATH,
    DEFAULT_TIME_SPLITS,
)

from models.common.ensemble import (
    load_model_meta,
    compute_ic,
    ensemble_predictions,
    run_ensemble_backtest as _common_run_ensemble_backtest,
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
    # 加载模型
    if model_type == 'lgb':
        model = lgb.Booster(model_file=str(model_path))
    elif model_type == 'catboost':
        model = CatBoostRegressor()
        model.load_model(str(model_path))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 加载元数据 (use common utility)
    meta_data = load_model_meta(model_path)

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
        print("    Qlib initialized with TA-Lib custom operators")
    else:
        qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US)
        print("    Qlib initialized")

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
    print(f"    Dataset created")

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

    # Ensemble 预测 (use common dict-based ensemble_predictions)
    print(f"\n[7] Ensembling predictions ({args.ensemble_method})...")
    pred_dict = {"LightGBM": pred_lgb, "CatBoost": pred_cb}
    weights = None
    if args.ensemble_method == 'weighted':
        weights = {"LightGBM": args.lgb_weight, "CatBoost": args.cb_weight}
    pred_ensemble = ensemble_predictions(pred_dict, args.ensemble_method, weights)
    print(f"    Ensemble predictions: {len(pred_ensemble)} samples, "
          f"range=[{pred_ensemble.min():.4f}, {pred_ensemble.max():.4f}]")

    # 计算各模型的 IC (common compute_ic returns: mean_ic, ic_std, icir, ic_by_date)
    print("\n[8] Calculating IC metrics...")
    label_series = test_label['LABEL0']

    lgb_ic, lgb_std, lgb_icir, _ = compute_ic(pred_lgb, label_series)
    cb_ic, cb_std, cb_icir, _ = compute_ic(pred_cb, label_series)
    ens_ic, ens_std, ens_icir, _ = compute_ic(pred_ensemble, label_series)

    print("\n    +------------------------------------------------------------+")
    print("    |              Information Coefficient (IC) Comparison       |")
    print("    +------------------------------------------------------------+")
    print(f"    |  Model       |   Mean IC  |   IC Std   |    ICIR    |")
    print("    +------------------------------------------------------------+")
    print(f"    |  LightGBM    |  {lgb_ic:>8.4f}  |  {lgb_std:>8.4f}  |  {lgb_icir:>8.4f}  |")
    print(f"    |  CatBoost    |  {cb_ic:>8.4f}  |  {cb_std:>8.4f}  |  {cb_icir:>8.4f}  |")
    print(f"    |  Ensemble    |  {ens_ic:>8.4f}  |  {ens_std:>8.4f}  |  {ens_icir:>8.4f}  |")
    print("    +------------------------------------------------------------+")

    # IC 提升
    ic_improvement = (ens_ic - max(lgb_ic, cb_ic)) / abs(max(lgb_ic, cb_ic)) * 100 if max(lgb_ic, cb_ic) != 0 else 0
    icir_improvement = (ens_icir - max(lgb_icir, cb_icir)) / abs(max(lgb_icir, cb_icir)) * 100 if max(lgb_icir, cb_icir) != 0 else 0

    print(f"\n    IC improvement over best single model:   {ic_improvement:>+.2f}%")
    print(f"    ICIR improvement over best single model: {icir_improvement:>+.2f}%")

    # 运行回测 (use common run_ensemble_backtest)
    print("\n[9] Running backtest...")
    _common_run_ensemble_backtest(pred_ensemble, args, time_splits, model_name="Ensemble")

    print("\n" + "=" * 70)
    print("Ensemble Backtest Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
