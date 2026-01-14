"""
CatBoost 超参数搜索 - 时间序列交叉验证版本

使用多个时间窗口进行交叉验证，选出更稳健的超参数。
避免在单一验证集上过拟合。

时间窗口设计:
  Fold 1: train 2000-2021, valid 2022
  Fold 2: train 2000-2022, valid 2023
  Fold 3: train 2000-2023, valid 2024
  Test:   2025 (完全独立，不参与超参数选择)

使用方法:
    python scripts/models/tree/run_catboost_hyperopt_cv.py
    python scripts/models/tree/run_catboost_hyperopt_cv.py --max-evals 50
    python scripts/models/tree/run_catboost_hyperopt_cv.py --backtest
"""

# ============================================================================
# 重要: 以下代码必须在任何其他导入之前执行
# ============================================================================

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

import sys
from pathlib import Path
import multiprocessing

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))
project_root = script_dir.parent

import qlib
from qlib.constant import REG_US
from utils.talib_ops import TALIB_OPS

qlib_data_path = project_root / "my_data" / "qlib_us"
qlib.init(
    provider_uri=str(qlib_data_path),
    region=REG_US,
    custom_ops=TALIB_OPS,
    kernels=1,
    joblib_backend=None,
)

# ============================================================================
# 现在可以安全地导入其他模块
# ============================================================================

import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.pyll import scope

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from utils.utils import evaluate_model
from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG, PROJECT_ROOT, MODEL_SAVE_PATH,
    print_training_header,
    init_qlib,
    check_data_availability,
    save_model_with_meta,
    create_meta_data,
    generate_model_filename,
    run_backtest,
)


# ============================================================================
# 时间序列交叉验证的 Fold 配置
# ============================================================================

CV_FOLDS = [
    {
        'name': 'Fold 1 (valid 2022)',
        'train_start': '2000-01-01',
        'train_end': '2021-12-31',
        'valid_start': '2022-01-01',
        'valid_end': '2022-12-31',
    },
    {
        'name': 'Fold 2 (valid 2023)',
        'train_start': '2000-01-01',
        'train_end': '2022-12-31',
        'valid_start': '2023-01-01',
        'valid_end': '2023-12-31',
    },
    {
        'name': 'Fold 3 (valid 2024)',
        'train_start': '2000-01-01',
        'train_end': '2023-12-31',
        'valid_start': '2024-01-01',
        'valid_end': '2024-12-31',
    },
]

# 最终测试集 (完全独立)
FINAL_TEST = {
    'train_start': '2000-01-01',
    'train_end': '2024-12-31',
    'valid_start': '2024-10-01',  # 用最近3个月做早停
    'valid_end': '2024-12-31',
    'test_start': '2025-01-01',
    'test_end': '2025-12-31',
}


# ============================================================================
# 超参数搜索空间
# ============================================================================

SEARCH_SPACE = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'max_depth': scope.int(hp.quniform('max_depth', 4, 10, 1)),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
    'random_strength': hp.uniform('random_strength', 0.1, 2),
    'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.6, 1.0),
    'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 1, 100, 1)),
}


def create_catboost_params(hyperparams: dict) -> dict:
    """将 hyperopt 参数转换为 CatBoost 参数"""
    return {
        'loss_function': 'RMSE',
        'iterations': 1000,
        'learning_rate': hyperparams['learning_rate'],
        'max_depth': int(hyperparams['max_depth']),
        'l2_leaf_reg': hyperparams['l2_leaf_reg'],
        'random_strength': hyperparams['random_strength'],
        'bagging_temperature': hyperparams['bagging_temperature'],
        'subsample': hyperparams['subsample'],
        'colsample_bylevel': hyperparams['colsample_bylevel'],
        'min_data_in_leaf': int(hyperparams['min_data_in_leaf']),
        'thread_count': 16,
        'verbose': False,
        'random_seed': 42,
    }


def create_data_handler_for_fold(args, handler_config, symbols, fold_config):
    """为特定 fold 创建 DataHandler"""
    from data.datahandler_ext import (
        Alpha158_Volatility, Alpha360_Volatility,
        Alpha158_Volatility_TALib, Alpha158_Volatility_TALib_Lite
    )
    from data.datahandler_pandas import Alpha158_Volatility_Pandas, Alpha360_Volatility_Pandas
    from data.datahandler_macro import Alpha158_Volatility_TALib_Macro

    handler_map = {
        'alpha158': Alpha158_Volatility,
        'alpha360': Alpha360_Volatility,
        'alpha158-talib': Alpha158_Volatility_TALib,
        'alpha158-talib-lite': Alpha158_Volatility_TALib_Lite,
        'alpha158-pandas': Alpha158_Volatility_Pandas,
        'alpha360-pandas': Alpha360_Volatility_Pandas,
        'alpha158-talib-macro': Alpha158_Volatility_TALib_Macro,
    }

    HandlerClass = handler_map.get(args.handler)
    if HandlerClass is None:
        raise ValueError(f"Unknown handler: {args.handler}")

    # 确定数据的结束时间
    if 'test_end' in fold_config:
        end_time = fold_config['test_end']
    else:
        end_time = fold_config['valid_end']

    handler = HandlerClass(
        volatility_window=args.nday,
        instruments=symbols,
        start_time=fold_config['train_start'],
        end_time=end_time,
        fit_start_time=fold_config['train_start'],
        fit_end_time=fold_config['train_end'],
        infer_processors=[],
    )

    return handler


def create_dataset_for_fold(handler, fold_config):
    """为特定 fold 创建 Dataset"""
    segments = {
        "train": (fold_config['train_start'], fold_config['train_end']),
        "valid": (fold_config['valid_start'], fold_config['valid_end']),
    }

    if 'test_start' in fold_config:
        segments["test"] = (fold_config['test_start'], fold_config['test_end'])

    return DatasetH(handler=handler, segments=segments)


def compute_ic(pred, label, index):
    """计算 IC (按日期分组的相关系数平均值)"""
    df = pd.DataFrame({'pred': pred, 'label': label}, index=index)
    ic_by_date = df.groupby(level='datetime').apply(
        lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
    )
    ic_by_date = ic_by_date.dropna()
    if len(ic_by_date) == 0:
        return 0.0, 0.0, 0.0
    mean_ic = ic_by_date.mean()
    ic_std = ic_by_date.std()
    icir = mean_ic / ic_std if ic_std > 0 else 0
    return mean_ic, ic_std, icir


class CVHyperoptObjective:
    """时间序列交叉验证的 Hyperopt 目标函数"""

    def __init__(self, args, handler_config, symbols, top_features=None):
        self.args = args
        self.handler_config = handler_config
        self.symbols = symbols
        self.top_features = top_features
        self.trial_count = 0
        self.best_mean_ic = -float('inf')

        # 预先准备所有 fold 的数据
        print("\n[*] Preparing data for all CV folds...")
        self.fold_data = []

        for fold in CV_FOLDS:
            print(f"    Preparing {fold['name']}...")
            handler = create_data_handler_for_fold(args, handler_config, symbols, fold)
            dataset = create_dataset_for_fold(handler, fold)

            if top_features:
                train_data = dataset.prepare("train", col_set="feature")[top_features]
                valid_data = dataset.prepare("valid", col_set="feature")[top_features]
            else:
                train_data = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
                valid_data = dataset.prepare("valid", col_set="feature", data_key=DataHandlerLP.DK_L)

            train_label = dataset.prepare("train", col_set="label").values.ravel()
            valid_label = dataset.prepare("valid", col_set="label").values.ravel()

            # Debug: Print data statistics for first fold
            if len(self.fold_data) == 0:
                print(f"\n      [DEBUG Data Prep - {fold['name']}]")
                print(f"        train_data columns ({len(train_data.columns)}): {train_data.columns.tolist()[:10]}...")
                print(f"        train_data NaN%: {train_data.isna().mean().mean()*100:.2f}%")
                print(f"        valid_data NaN%: {valid_data.isna().mean().mean()*100:.2f}%")
                print(f"        train_label: mean={train_label.mean():.6f}, std={train_label.std():.6f}, NaN={np.isnan(train_label).sum()}")
                print(f"        valid_label: mean={valid_label.mean():.6f}, std={valid_label.std():.6f}, NaN={np.isnan(valid_label).sum()}")
                # Check label distribution per date
                valid_label_df = pd.DataFrame({'label': valid_label}, index=valid_data.index)
                label_std_per_date = valid_label_df.groupby(level='datetime')['label'].std()
                print(f"        valid_label std per date: min={label_std_per_date.min():.6f}, max={label_std_per_date.max():.6f}, mean={label_std_per_date.mean():.6f}")

            self.fold_data.append({
                'name': fold['name'],
                'train_data': train_data,
                'valid_data': valid_data,
                'train_label': train_label,
                'valid_label': valid_label,
                'train_pool': Pool(train_data, label=train_label),
                'valid_pool': Pool(valid_data, label=valid_label),
            })

            print(f"      Train: {train_data.shape}, Valid: {valid_data.shape}")

        print(f"    ✓ All {len(CV_FOLDS)} folds prepared")

    def __call__(self, hyperparams):
        """目标函数: 在所有 fold 上训练并返回平均验证集 IC"""
        self.trial_count += 1
        cb_params = create_catboost_params(hyperparams)

        fold_ics = []
        fold_results = []

        try:
            for fold in self.fold_data:
                # 训练模型
                model = CatBoostRegressor(**cb_params)
                model.fit(
                    fold['train_pool'],
                    eval_set=fold['valid_pool'],
                    early_stopping_rounds=50,
                    verbose_eval=False,
                )

                # 验证集预测
                valid_pred = model.predict(fold['valid_data'])

                # 计算 IC
                mean_ic, ic_std, icir = compute_ic(
                    valid_pred, fold['valid_label'], fold['valid_data'].index
                )

                fold_ics.append(mean_ic)
                fold_results.append({
                    'name': fold['name'],
                    'ic': mean_ic,
                    'icir': icir,
                    'best_iter': model.best_iteration_,
                })

            # 计算平均 IC
            mean_ic_all = np.mean(fold_ics)
            std_ic_all = np.std(fold_ics)

            # 更新最佳
            if mean_ic_all > self.best_mean_ic:
                self.best_mean_ic = mean_ic_all
                is_best = " ★ NEW BEST"
            else:
                is_best = ""

            # 打印进度
            fold_ic_str = ", ".join([f"{r['ic']:.4f}" for r in fold_results])
            print(f"  Trial {self.trial_count:3d}: Mean IC={mean_ic_all:.4f} (±{std_ic_all:.4f}) "
                  f"[{fold_ic_str}] lr={hyperparams['learning_rate']:.4f}{is_best}")

            return {
                'loss': -mean_ic_all,
                'status': STATUS_OK,
                'mean_ic': mean_ic_all,
                'std_ic': std_ic_all,
                'fold_results': fold_results,
                'params': cb_params,
            }

        except Exception as e:
            print(f"  Trial {self.trial_count:3d}: FAILED - {str(e)}")
            return {
                'loss': float('inf'),
                'status': STATUS_FAIL,
                'error': str(e),
            }


def first_pass_feature_selection(args, handler_config, symbols):
    """第一遍训练获取特征重要性"""
    print("\n[*] First pass: feature selection using Fold 2...")

    # 使用中间的 fold 来做特征选择
    fold = CV_FOLDS[1]  # Fold 2
    handler = create_data_handler_for_fold(args, handler_config, symbols, fold)
    dataset = create_dataset_for_fold(handler, fold)

    train_data = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
    valid_data = dataset.prepare("valid", col_set="feature", data_key=DataHandlerLP.DK_L)
    train_label = dataset.prepare("train", col_set="label").values.ravel()
    valid_label = dataset.prepare("valid", col_set="label").values.ravel()

    train_pool = Pool(train_data, label=train_label)
    valid_pool = Pool(valid_data, label=valid_label)

    model = CatBoostRegressor(
        loss_function='RMSE',
        learning_rate=0.05,
        max_depth=6,
        l2_leaf_reg=3,
        random_strength=1,
        thread_count=16,
        verbose=False,
    )

    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    importance = model.get_feature_importance()
    feature_names = train_data.columns.tolist()

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("\n    Top 20 features:")
    print("    " + "-" * 50)
    for i, row in importance_df.head(20).iterrows():
        print(f"    {importance_df.index.get_loc(i)+1:3d}. {row['feature']:<40s} {row['importance']:>10.2f}")
    print("    " + "-" * 50)

    return importance_df


def run_hyperopt_cv_search(args, handler_config, symbols, top_features=None):
    """运行时间序列交叉验证的超参数搜索"""
    print("\n" + "=" * 70)
    print("HYPEROPT SEARCH WITH TIME-SERIES CROSS-VALIDATION")
    print("=" * 70)
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: train {fold['train_start']}~{fold['train_end']}, "
              f"valid {fold['valid_start']}~{fold['valid_end']}")
    print(f"Max evaluations: {args.max_evals}")
    print("=" * 70)

    # 创建目标函数
    objective = CVHyperoptObjective(args, handler_config, symbols, top_features)

    # 运行搜索
    trials = Trials()
    print("\n[*] Running hyperparameter search...")

    best = fmin(
        fn=objective,
        space=SEARCH_SPACE,
        algo=tpe.suggest,
        max_evals=args.max_evals,
        trials=trials,
        show_progressbar=False,
    )

    # 获取最佳结果
    best_params = create_catboost_params(best)
    best_trial_idx = np.argmin([t['result']['loss'] for t in trials.trials])
    best_trial = trials.trials[best_trial_idx]['result']

    print("\n" + "=" * 70)
    print("CV HYPEROPT SEARCH COMPLETE")
    print("=" * 70)
    print(f"Best Mean IC: {best_trial['mean_ic']:.4f} (±{best_trial['std_ic']:.4f})")
    print("\nIC by fold:")
    for r in best_trial['fold_results']:
        print(f"  {r['name']}: IC={r['ic']:.4f}, ICIR={r['icir']:.4f}, iter={r['best_iter']}")
    print("\nBest parameters:")
    for key, value in best_params.items():
        if key not in ['thread_count', 'verbose', 'loss_function', 'random_seed']:
            print(f"  {key}: {value}")
    print("=" * 70)

    return best_params, trials, best_trial


def train_final_model(args, handler_config, symbols, best_params, top_features=None):
    """使用最优参数在完整数据上训练最终模型"""
    print("\n[*] Training final model on full data...")
    print("    Parameters:")
    for key, value in best_params.items():
        if key not in ['thread_count', 'verbose', 'loss_function', 'random_seed']:
            print(f"      {key}: {value}")

    # 创建最终数据集
    handler = create_data_handler_for_fold(args, handler_config, symbols, FINAL_TEST)
    dataset = create_dataset_for_fold(handler, FINAL_TEST)

    if top_features:
        train_data = dataset.prepare("train", col_set="feature")[top_features]
        valid_data = dataset.prepare("valid", col_set="feature")[top_features]
        test_data = dataset.prepare("test", col_set="feature")[top_features]
        feature_names = top_features
    else:
        train_data = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
        valid_data = dataset.prepare("valid", col_set="feature", data_key=DataHandlerLP.DK_L)
        test_data = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_L)
        feature_names = train_data.columns.tolist()

    train_label = dataset.prepare("train", col_set="label").values.ravel()
    valid_label = dataset.prepare("valid", col_set="label").values.ravel()

    print(f"\n    Final training data:")
    print(f"      Train: {train_data.shape} ({FINAL_TEST['train_start']} ~ {FINAL_TEST['train_end']})")
    print(f"      Valid: {valid_data.shape} ({FINAL_TEST['valid_start']} ~ {FINAL_TEST['valid_end']})")
    print(f"      Test:  {test_data.shape} ({FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']})")

    train_pool = Pool(train_data, label=train_label)
    valid_pool = Pool(valid_data, label=valid_label)

    # 训练
    print("\n    Training progress:")
    model = CatBoostRegressor(**best_params)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    print(f"\n    Best iteration: {model.best_iteration_}")

    # 验证集 IC (用于参考)
    valid_pred = model.predict(valid_data)
    valid_ic, valid_ic_std, valid_icir = compute_ic(valid_pred, valid_label, valid_data.index)
    print(f"\n    [Validation Set - for reference]")
    print(f"    Valid IC:   {valid_ic:.4f}")
    print(f"    Valid ICIR: {valid_icir:.4f}")

    # 测试集预测
    test_pred_values = model.predict(test_data)
    test_pred = pd.Series(test_pred_values, index=test_data.index, name='score')

    print(f"\n    Test prediction shape: {test_pred.shape}")
    print(f"    Test prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    return model, feature_names, test_pred, dataset


def main():
    parser = argparse.ArgumentParser(
        description='CatBoost Hyperopt with Time-Series Cross-Validation',
    )

    # 基础参数
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--handler', type=str, default='alpha158-talib-macro',
                        choices=list(HANDLER_CONFIG.keys()))
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--top-k', type=int, default=0)

    # Hyperopt 参数
    parser.add_argument('--max-evals', type=int, default=50)

    # 回测参数
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=1)
    parser.add_argument('--account', type=float, default=10000)
    parser.add_argument('--rebalance-freq', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss'])

    # 策略参数 (保持兼容)
    parser.add_argument('--risk-lookback', type=int, default=20)
    parser.add_argument('--drawdown-threshold', type=float, default=-0.10)
    parser.add_argument('--momentum-threshold', type=float, default=0.03)
    parser.add_argument('--risk-high', type=float, default=0.50)
    parser.add_argument('--risk-medium', type=float, default=0.75)
    parser.add_argument('--risk-normal', type=float, default=0.95)
    parser.add_argument('--market-proxy', type=str, default='AAPL')
    parser.add_argument('--vol-high', type=float, default=0.35)
    parser.add_argument('--vol-medium', type=float, default=0.25)
    parser.add_argument('--stop-loss', type=float, default=-0.15)
    parser.add_argument('--no-sell-after-drop', type=float, default=-0.20)

    args = parser.parse_args()

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    # 打印头部
    print("=" * 70)
    print("CatBoost Hyperopt with Time-Series Cross-Validation")
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Handler: {args.handler}")
    print(f"N-day: {args.nday}")
    print(f"Top-K features: {args.top_k}")
    print(f"Max evaluations: {args.max_evals}")
    print(f"CV Folds: {len(CV_FOLDS)}")
    print("=" * 70)

    # 初始化
    init_qlib(handler_config['use_talib'])

    # 特征选择
    top_features = None
    if args.top_k > 0:
        importance_df = first_pass_feature_selection(args, handler_config, symbols)
        top_features = importance_df.head(args.top_k)['feature'].tolist()
        print(f"\n    Selected top {args.top_k} features")

    # 运行 CV 超参数搜索
    best_params, trials, best_trial = run_hyperopt_cv_search(
        args, handler_config, symbols, top_features
    )

    # 保存搜索结果
    output_dir = PROJECT_ROOT / "outputs" / "hyperopt_cv"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存最佳参数
    params_file = output_dir / f"catboost_cv_best_params_{timestamp}.json"
    params_to_save = {
        'params': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                   for k, v in best_params.items()},
        'cv_results': {
            'mean_ic': best_trial['mean_ic'],
            'std_ic': best_trial['std_ic'],
            'fold_results': best_trial['fold_results'],
        }
    }
    with open(params_file, 'w') as f:
        json.dump(params_to_save, f, indent=2)
    print(f"\nBest parameters saved to: {params_file}")

    # 保存搜索历史
    history = []
    for t in trials.trials:
        if t['result']['status'] == STATUS_OK:
            history.append({
                'mean_ic': t['result']['mean_ic'],
                'std_ic': t['result']['std_ic'],
                **{k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                   for k, v in t['result']['params'].items()
                   if k not in ['thread_count', 'verbose', 'loss_function', 'random_seed']}
            })

    history_df = pd.DataFrame(history)
    history_file = output_dir / f"catboost_cv_history_{timestamp}.csv"
    history_df.to_csv(history_file, index=False)
    print(f"Search history saved to: {history_file}")

    # 训练最终模型
    model, feature_names, test_pred, dataset = train_final_model(
        args, handler_config, symbols, best_params, top_features
    )

    # 评估
    print("\n[*] Final Evaluation on Test Set (2025)...")
    evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

    # 保存模型
    print("\n[*] Saving model...")
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model_filename = generate_model_filename(
        "catboost_cv", args, args.top_k if args.top_k > 0 else 0, ".cbm"
    )
    model_path = MODEL_SAVE_PATH / model_filename

    # 构造 time_splits 用于 meta_data
    time_splits = {
        'train_start': FINAL_TEST['train_start'],
        'train_end': FINAL_TEST['train_end'],
        'valid_start': FINAL_TEST['valid_start'],
        'valid_end': FINAL_TEST['valid_end'],
        'test_start': FINAL_TEST['test_start'],
        'test_end': FINAL_TEST['test_end'],
    }

    meta_data = create_meta_data(args, handler_config, time_splits, feature_names, "catboost_cv", args.top_k)
    meta_data['cv_params'] = best_params
    meta_data['cv_results'] = {
        'mean_ic': best_trial['mean_ic'],
        'std_ic': best_trial['std_ic'],
        'fold_results': best_trial['fold_results'],
    }
    save_model_with_meta(model, model_path, meta_data)

    # 回测
    if args.backtest:
        def load_catboost_model(path):
            m = CatBoostRegressor()
            m.load_model(str(path))
            return m

        def get_catboost_feature_count(m):
            if hasattr(m, 'get_feature_count'):
                return m.get_feature_count()
            elif m.feature_names_:
                return len(m.feature_names_)
            return "N/A"

        run_backtest(
            model_path, dataset, test_pred, args, time_splits,
            model_name="CatBoost (CV Hyperopt)",
            load_model_func=load_catboost_model,
            get_feature_count_func=get_catboost_feature_count
        )

    print("\n" + "=" * 70)
    print("CV HYPEROPT COMPLETE")
    print("=" * 70)
    print(f"CV Mean IC: {best_trial['mean_ic']:.4f} (±{best_trial['std_ic']:.4f})")
    print(f"Model saved to: {model_path}")
    print(f"Best parameters: {params_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
