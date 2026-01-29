"""
CatBoost 超参数搜索 - 时间序列交叉验证版本

使用多个时间窗口进行交叉验证，选出更稳健的超参数。
避免在单一验证集上过拟合。

时间窗口设计:
  Fold 1: train 2000-2020, valid 2021
  Fold 2: train 2000-2021, valid 2022
  Fold 3: train 2000-2022, valid 2023
  Fold 4: train 2000-2023, valid 2024
  Test:   2025 (完全独立，不参与超参数选择)

使用方法:
    # 完整训练流程 (hyperopt + train + backtest)
    python scripts/models/tree/run_catboost_hyperopt_cv.py
    python scripts/models/tree/run_catboost_hyperopt_cv.py --max-evals 50
    python scripts/models/tree/run_catboost_hyperopt_cv.py --backtest

    # 仅加载已有模型并回测
    python scripts/models/tree/run_catboost_hyperopt_cv.py --model-path ./my_models/catboost_cv_xxx.cbm
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
import pickle
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
    # CV utilities
    CV_FOLDS,
    FINAL_TEST,
    create_data_handler_for_fold,
    create_dataset_for_fold,
    compute_ic,
)


# ============================================================================
# 超参数搜索空间
# ============================================================================

SEARCH_SPACE = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.15)),  # 平衡: 0.01-0.15
    'max_depth': scope.int(hp.quniform('max_depth', 4, 8, 1)),  # 平衡: 4-8
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 15),  # 平衡: 1-15
    'random_strength': hp.uniform('random_strength', 0.1, 2),
    'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
    'subsample': hp.uniform('subsample', 0.6, 0.95),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.6, 0.95),
    'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 5, 100, 1)),
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


class CVHyperoptObjective:
    """时间序列交叉验证的 Hyperopt 目标函数"""

    def __init__(self, args, handler_config, symbols, top_features=None, verbose=False):
        self.args = args
        self.handler_config = handler_config
        self.symbols = symbols
        self.top_features = top_features
        self.verbose = verbose
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

            # Preprocess features to match feature selection script
            train_data = train_data.fillna(0).replace([np.inf, -np.inf], 0)
            valid_data = valid_data.fillna(0).replace([np.inf, -np.inf], 0)

            # Use DK_L for labels to ensure consistent processing with feature selection
            train_label_df = dataset.prepare("train", col_set="label", data_key=DataHandlerLP.DK_L)
            valid_label_df = dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
            if isinstance(train_label_df, pd.DataFrame):
                train_label = train_label_df.iloc[:, 0].fillna(0).values
            else:
                train_label = train_label_df.fillna(0).values
            if isinstance(valid_label_df, pd.DataFrame):
                valid_label = valid_label_df.iloc[:, 0].fillna(0).values
            else:
                valid_label = valid_label_df.fillna(0).values

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

        # 打印 Trial 开始信息
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Trial {self.trial_count}/{self.args.max_evals} | Best IC so far: {self.best_mean_ic:.4f}")
            print(f"{'='*60}")
            print(f"  Params: lr={hyperparams['learning_rate']:.4f}, depth={int(hyperparams['max_depth'])}, "
                  f"l2={hyperparams['l2_leaf_reg']:.2f}, subsample={hyperparams['subsample']:.2f}")
            sys.stdout.flush()

        fold_ics = []
        fold_results = []

        try:
            for fold_idx, fold in enumerate(self.fold_data):
                if self.verbose:
                    print(f"\n  [{fold_idx+1}/{len(self.fold_data)}] {fold['name']}...")
                    sys.stdout.flush()

                # 训练模型
                model = CatBoostRegressor(**cb_params)

                # verbose 控制训练日志输出
                model.fit(
                    fold['train_pool'],
                    eval_set=fold['valid_pool'],
                    early_stopping_rounds=50,
                    verbose=100 if self.verbose else False,
                )

                # 训练集预测和 IC
                train_pred = model.predict(fold['train_data'])
                train_ic, train_ic_std, train_icir = compute_ic(
                    train_pred, fold['train_label'], fold['train_data'].index
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

                # 打印 Fold 结果 (仅 verbose 模式)
                if self.verbose:
                    print(f"      Best iter: {model.best_iteration_}")
                    print(f"      Train IC: {train_ic:.4f} (ICIR: {train_icir:.4f})")
                    print(f"      Valid IC: {mean_ic:.4f} (ICIR: {icir:.4f})")
                    sys.stdout.flush()

            # 计算平均 IC
            mean_ic_all = np.mean(fold_ics)
            std_ic_all = np.std(fold_ics)

            # 更新最佳
            if mean_ic_all > self.best_mean_ic:
                self.best_mean_ic = mean_ic_all
                is_best = " ★ NEW BEST"
            else:
                is_best = ""

            # 打印 Trial 汇总 (始终输出，但格式不同)
            fold_ic_str = ", ".join([f"{r['ic']:.4f}" for r in fold_results])
            if self.verbose:
                print(f"\n  {'─'*50}")
                print(f"  Trial {self.trial_count} Summary:")
                print(f"    Mean IC: {mean_ic_all:.4f} (±{std_ic_all:.4f})")
                print(f"    Folds:   [{fold_ic_str}]")
                print(f"    Best Trial IC: {self.best_mean_ic:.4f}{is_best}")
                print(f"  {'─'*50}")
            else:
                # 简洁输出: 一行显示 trial 结果
                print(f"Trial {self.trial_count:3d}: Mean IC={mean_ic_all:.4f} (±{std_ic_all:.4f}) [{fold_ic_str}] lr={hyperparams['learning_rate']:.4f}{is_best}")
            sys.stdout.flush()

            return {
                'loss': -mean_ic_all,
                'status': STATUS_OK,
                'mean_ic': mean_ic_all,
                'std_ic': std_ic_all,
                'fold_results': fold_results,
                'params': cb_params,
            }

        except Exception as e:
            import traceback
            print(f"\n  Trial {self.trial_count} FAILED!")
            print(f"    Error: {str(e)}")
            traceback.print_exc()
            sys.stdout.flush()
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

    # Preprocess features
    train_data = train_data.fillna(0).replace([np.inf, -np.inf], 0)
    valid_data = valid_data.fillna(0).replace([np.inf, -np.inf], 0)

    # Use DK_L for labels
    train_label_df = dataset.prepare("train", col_set="label", data_key=DataHandlerLP.DK_L)
    valid_label_df = dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(train_label_df, pd.DataFrame):
        train_label = train_label_df.iloc[:, 0].fillna(0).values
    else:
        train_label = train_label_df.fillna(0).values
    if isinstance(valid_label_df, pd.DataFrame):
        valid_label = valid_label_df.iloc[:, 0].fillna(0).values
    else:
        valid_label = valid_label_df.fillna(0).values

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
    objective = CVHyperoptObjective(args, handler_config, symbols, top_features, verbose=args.verbose)

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

    # Preprocess features to match feature selection script
    train_data = train_data.fillna(0).replace([np.inf, -np.inf], 0)
    valid_data = valid_data.fillna(0).replace([np.inf, -np.inf], 0)
    test_data = test_data.fillna(0).replace([np.inf, -np.inf], 0)

    # Use DK_L for labels to ensure consistent processing
    train_label_df = dataset.prepare("train", col_set="label", data_key=DataHandlerLP.DK_L)
    valid_label_df = dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(train_label_df, pd.DataFrame):
        train_label = train_label_df.iloc[:, 0].fillna(0).values
    else:
        train_label = train_label_df.fillna(0).values
    if isinstance(valid_label_df, pd.DataFrame):
        valid_label = valid_label_df.iloc[:, 0].fillna(0).values
    else:
        valid_label = valid_label_df.fillna(0).values

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

    # 输出控制
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed training logs for each trial')

    # 模型加载（跳过训练，直接回测）
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to existing model file (.cbm) for backtest-only mode')

    args = parser.parse_args()

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    # =========================================================================
    # Model-path mode: Load existing model and run backtest only
    # =========================================================================
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"Error: Model file not found: {model_path}")
            sys.exit(1)

        print("=" * 70)
        print("CatBoost Backtest-Only Mode (Loading Existing Model)")
        print("=" * 70)
        print(f"Model: {model_path}")
        print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
        print("=" * 70)

        # Load model
        print("\n[*] Loading model...")
        model = CatBoostRegressor()
        model.load_model(str(model_path))
        print(f"    ✓ Model loaded successfully")

        # Load metadata if available
        meta_path = model_path.with_suffix('.meta.pkl')
        meta_data = {}
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                meta_data = pickle.load(f)
            print(f"    ✓ Metadata loaded")
            # Override args from metadata if available
            if 'handler' in meta_data:
                args.handler = meta_data['handler']
                handler_config = HANDLER_CONFIG[args.handler]
                print(f"    Using handler from metadata: {args.handler}")
            if 'nday' in meta_data:
                args.nday = meta_data['nday']
                print(f"    Using nday from metadata: {args.nday}")
        else:
            print(f"    ⚠ Metadata file not found, using command-line args")

        # Initialize qlib
        init_qlib(handler_config['use_talib'])

        # Create dataset for test period
        print("\n[*] Preparing test data...")
        from models.common.handlers import get_handler_class
        HandlerClass = get_handler_class(args.handler)

        handler = HandlerClass(
            volatility_window=args.nday,
            instruments=symbols,
            start_time=FINAL_TEST['train_start'],
            end_time=FINAL_TEST['test_end'],
            fit_start_time=FINAL_TEST['train_start'],
            fit_end_time=FINAL_TEST['train_end'],
            infer_processors=[],
        )

        dataset = DatasetH(
            handler=handler,
            segments={
                "train": (FINAL_TEST['train_start'], FINAL_TEST['train_end']),
                "valid": (FINAL_TEST['valid_start'], FINAL_TEST['valid_end']),
                "test": (FINAL_TEST['test_start'], FINAL_TEST['test_end']),
            }
        )

        # Prepare test data
        test_data = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_L)
        test_data = test_data.fillna(0).replace([np.inf, -np.inf], 0)

        print(f"    Test data shape: {test_data.shape}")
        print(f"    Test period: {FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']}")

        # Make predictions
        print("\n[*] Generating predictions...")
        test_pred_values = model.predict(test_data)
        test_pred = pd.Series(test_pred_values, index=test_data.index, name='score')
        print(f"    Prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

        # Evaluate
        print("\n[*] Evaluation on Test Set...")
        evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

        # Run backtest
        time_splits = {
            'train_start': FINAL_TEST['train_start'],
            'train_end': FINAL_TEST['train_end'],
            'valid_start': FINAL_TEST['valid_start'],
            'valid_end': FINAL_TEST['valid_end'],
            'test_start': FINAL_TEST['test_start'],
            'test_end': FINAL_TEST['test_end'],
        }

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
            model_name="CatBoost (Loaded Model)",
            load_model_func=load_catboost_model,
            get_feature_count_func=get_catboost_feature_count
        )

        print("\n" + "=" * 70)
        print("BACKTEST COMPLETE")
        print("=" * 70)
        return

    # =========================================================================
    # Normal mode: Hyperopt search + Training + Backtest
    # =========================================================================

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
