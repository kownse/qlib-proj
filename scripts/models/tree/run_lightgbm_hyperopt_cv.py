"""
LightGBM 超参数搜索 - 时间序列交叉验证版本

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
    python scripts/models/tree/run_lightgbm_hyperopt_cv.py
    python scripts/models/tree/run_lightgbm_hyperopt_cv.py --max-evals 50
    python scripts/models/tree/run_lightgbm_hyperopt_cv.py --backtest

    # 仅加载已有模型并回测
    python scripts/models/tree/run_lightgbm_hyperopt_cv.py --model-path ./my_models/lightgbm_cv_xxx.txt
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
import lightgbm as lgb
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

# 扩展训练期配置：训练到 2024，测试 2025
# 这样可以让模型学习更近期的市场模式
EXTENDED_FINAL_TEST = {
    'train_start': '2000-01-01',
    'train_end': '2024-12-31',      # 扩展到 2024
    'valid_start': '2025-01-01',    # 2025 H1 作为验证
    'valid_end': '2025-06-30',
    'test_start': '2025-07-01',     # 2025 H2 作为测试
    'test_end': '2025-12-31',
}


# ============================================================================
# 超参数搜索空间
# ============================================================================

# 搜索空间 - 对齐 nested_cv_feature_selection_lightgbm.py 的参数范围
SEARCH_SPACE = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.1)),
    'num_leaves': scope.int(hp.quniform('num_leaves', 64, 256, 1)),  # 对齐: 128
    'max_depth': scope.int(hp.quniform('max_depth', 6, 10, 1)),      # 对齐: 8
    'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 20, 100, 1)),  # 对齐: 50
    'feature_fraction': hp.uniform('feature_fraction', 0.6, 0.95),  # 对齐: 0.8
    'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 0.95),  # 对齐: 0.8
    'bagging_freq': scope.int(hp.quniform('bagging_freq', 1, 10, 1)),  # 对齐: 5
    'lambda_l1': hp.loguniform('lambda_l1', np.log(1e-4), np.log(1)),  # 对齐: 0.0
    'lambda_l2': hp.loguniform('lambda_l2', np.log(0.1), np.log(10)),  # 对齐: 1.0
}


def create_lightgbm_params(hyperparams: dict) -> dict:
    """将 hyperopt 参数转换为 LightGBM 参数 (对齐 nested_cv_feature_selection_lightgbm.py)"""
    return {
        'objective': 'regression',
        'metric': 'mse',  # 对齐: mse (不是 rmse)
        'boosting_type': 'gbdt',
        'learning_rate': hyperparams['learning_rate'],
        'num_leaves': int(hyperparams['num_leaves']),
        'max_depth': int(hyperparams['max_depth']),
        'min_data_in_leaf': int(hyperparams['min_data_in_leaf']),  # 对齐参数名
        'feature_fraction': hyperparams['feature_fraction'],       # 对齐参数名
        'bagging_fraction': hyperparams['bagging_fraction'],       # 对齐参数名
        'bagging_freq': int(hyperparams['bagging_freq']),          # 添加 bagging_freq
        'lambda_l1': hyperparams['lambda_l1'],                     # 对齐参数名
        'lambda_l2': hyperparams['lambda_l2'],                     # 对齐参数名
        'num_threads': 8,
        'verbose': -1,
        'seed': 42,
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
            })

            print(f"      Train: {train_data.shape}, Valid: {valid_data.shape}")

        print(f"    ✓ All {len(CV_FOLDS)} folds prepared")

    def __call__(self, hyperparams):
        """目标函数: 在所有 fold 上训练并返回平均验证集 IC"""
        self.trial_count += 1
        lgb_params = create_lightgbm_params(hyperparams)

        # 打印 Trial 开始信息
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Trial {self.trial_count}/{self.args.max_evals} | Best IC so far: {self.best_mean_ic:.4f}")
            print(f"{'='*60}")
            print(f"  Params: lr={hyperparams['learning_rate']:.4f}, leaves={int(hyperparams['num_leaves'])}, "
                  f"depth={int(hyperparams['max_depth'])}, bagging={hyperparams['bagging_fraction']:.2f}")
            sys.stdout.flush()

        fold_ics = []
        fold_results = []

        try:
            for fold_idx, fold in enumerate(self.fold_data):
                if self.verbose:
                    print(f"\n  [{fold_idx+1}/{len(self.fold_data)}] {fold['name']}...")
                    sys.stdout.flush()

                # 创建 LightGBM 数据集
                train_set = lgb.Dataset(fold['train_data'], label=fold['train_label'])
                valid_set = lgb.Dataset(fold['valid_data'], label=fold['valid_label'], reference=train_set)

                # 训练模型
                callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
                if self.verbose:
                    callbacks.append(lgb.log_evaluation(period=100))

                model = lgb.train(
                    lgb_params,
                    train_set,
                    num_boost_round=500,  # 对齐 nested_cv_feature_selection_lightgbm.py
                    valid_sets=[valid_set],
                    valid_names=['valid'],
                    callbacks=callbacks,
                )

                best_iter = model.best_iteration

                # 训练集预测和 IC
                train_pred = model.predict(fold['train_data'], num_iteration=best_iter)
                train_ic, train_ic_std, train_icir = compute_ic(
                    train_pred, fold['train_label'], fold['train_data'].index
                )

                # 验证集预测
                valid_pred = model.predict(fold['valid_data'], num_iteration=best_iter)

                # 计算 IC
                mean_ic, ic_std, icir = compute_ic(
                    valid_pred, fold['valid_label'], fold['valid_data'].index
                )

                fold_ics.append(mean_ic)
                fold_results.append({
                    'name': fold['name'],
                    'ic': mean_ic,
                    'icir': icir,
                    'best_iter': best_iter,
                })

                # 打印 Fold 结果 (仅 verbose 模式)
                if self.verbose:
                    print(f"      Best iter: {best_iter}")
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

            # 打印 Trial 汇总
            fold_ic_str = ", ".join([f"{r['ic']:.4f}" for r in fold_results])
            if self.verbose:
                print(f"\n  {'─'*50}")
                print(f"  Trial {self.trial_count} Summary:")
                print(f"    Mean IC: {mean_ic_all:.4f} (±{std_ic_all:.4f})")
                print(f"    Folds:   [{fold_ic_str}]")
                print(f"    Best Trial IC: {self.best_mean_ic:.4f}{is_best}")
                print(f"  {'─'*50}")
            else:
                # 简洁输出
                print(f"Trial {self.trial_count:3d}: Mean IC={mean_ic_all:.4f} (±{std_ic_all:.4f}) [{fold_ic_str}] lr={hyperparams['learning_rate']:.4f}{is_best}")
            sys.stdout.flush()

            return {
                'loss': -mean_ic_all,
                'status': STATUS_OK,
                'mean_ic': mean_ic_all,
                'std_ic': std_ic_all,
                'fold_results': fold_results,
                'params': lgb_params,
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

    train_set = lgb.Dataset(train_data, label=train_label)
    valid_set = lgb.Dataset(valid_data, label=valid_label, reference=train_set)

    # 对齐 nested_cv_feature_selection_lightgbm.py 的默认参数
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'max_depth': 8,
        'num_leaves': 128,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.0,
        'lambda_l2': 1.0,
        'num_threads': 8,
        'verbose': -1,
        'seed': 42,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,  # 对齐 nested_cv_feature_selection_lightgbm.py
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    importance = model.feature_importance(importance_type='gain')
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
    best_params = create_lightgbm_params(best)
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
        if key not in ['num_threads', 'verbose', 'objective', 'metric', 'seed', 'boosting_type']:
            print(f"  {key}: {value}")
    print("=" * 70)

    return best_params, trials, best_trial


def train_final_model(args, handler_config, symbols, best_params, final_test_config, top_features=None):
    """使用最优参数在完整数据上训练最终模型"""
    print("\n[*] Training final model on full data...")
    print("    Parameters:")
    for key, value in best_params.items():
        if key not in ['num_threads', 'verbose', 'objective', 'metric', 'seed', 'boosting_type']:
            print(f"      {key}: {value}")

    # 创建最终数据集
    handler = create_data_handler_for_fold(args, handler_config, symbols, final_test_config)
    dataset = create_dataset_for_fold(handler, final_test_config)

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

    # Preprocess features
    train_data = train_data.fillna(0).replace([np.inf, -np.inf], 0)
    valid_data = valid_data.fillna(0).replace([np.inf, -np.inf], 0)
    test_data = test_data.fillna(0).replace([np.inf, -np.inf], 0)

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

    print(f"\n    Final training data:")
    print(f"      Train: {train_data.shape} ({final_test_config['train_start']} ~ {final_test_config['train_end']})")
    print(f"      Valid: {valid_data.shape} ({final_test_config['valid_start']} ~ {final_test_config['valid_end']})")
    print(f"      Test:  {test_data.shape} ({final_test_config['test_start']} ~ {final_test_config['test_end']})")

    train_set = lgb.Dataset(train_data, label=train_label)
    valid_set = lgb.Dataset(valid_data, label=valid_label, reference=train_set)

    # 训练
    print("\n    Training progress:")
    model = lgb.train(
        best_params,
        train_set,
        num_boost_round=500,  # 对齐 nested_cv_feature_selection_lightgbm.py
        valid_sets=[valid_set],
        valid_names=['valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    print(f"\n    Best iteration: {model.best_iteration}")

    # 验证集 IC (用于参考)
    valid_pred = model.predict(valid_data, num_iteration=model.best_iteration)
    valid_ic, valid_ic_std, valid_icir = compute_ic(valid_pred, valid_label, valid_data.index)
    print(f"\n    [Validation Set - for reference]")
    print(f"    Valid IC:   {valid_ic:.4f}")
    print(f"    Valid ICIR: {valid_icir:.4f}")

    # 测试集预测
    test_pred_values = model.predict(test_data, num_iteration=model.best_iteration)
    test_pred = pd.Series(test_pred_values, index=test_data.index, name='score')

    print(f"\n    Test prediction shape: {test_pred.shape}")
    print(f"    Test prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    return model, feature_names, test_pred, dataset


def main():
    parser = argparse.ArgumentParser(
        description='LightGBM Hyperopt with Time-Series Cross-Validation',
    )

    # 基础参数
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--handler', type=str, default='lightgbm-v1',
                        choices=list(HANDLER_CONFIG.keys()))
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--top-k', type=int, default=0)

    # Hyperopt 参数
    parser.add_argument('--max-evals', type=int, default=50)

    # 训练期配置
    parser.add_argument('--extended-train', action='store_true',
                        help='使用扩展训练期 (train到2024, valid/test在2025)，让模型学习更近期的市场模式')

    # 回测参数
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=1)
    parser.add_argument('--account', type=float, default=10000)
    parser.add_argument('--rebalance-freq', type=int, default=5)
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
                        help='Path to existing model file (.txt) for backtest-only mode')

    args = parser.parse_args()

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    # 选择训练期配置
    if args.extended_train:
        final_test_config = EXTENDED_FINAL_TEST
        print("[*] Using EXTENDED training period (train to 2024, test on 2025)")
    else:
        final_test_config = FINAL_TEST
        print("[*] Using STANDARD training period (train to 2023, valid 2024, test 2025)")

    # =========================================================================
    # Model-path mode: Load existing model and run backtest only
    # =========================================================================
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"Error: Model file not found: {model_path}")
            sys.exit(1)

        print("=" * 70)
        print("LightGBM Backtest-Only Mode (Loading Existing Model)")
        print("=" * 70)
        print(f"Model: {model_path}")
        print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
        print("=" * 70)

        # Load model
        print("\n[*] Loading model...")
        model = lgb.Booster(model_file=str(model_path))
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
            start_time=final_test_config['train_start'],
            end_time=final_test_config['test_end'],
            fit_start_time=final_test_config['train_start'],
            fit_end_time=final_test_config['train_end'],
            infer_processors=[],
        )

        dataset = DatasetH(
            handler=handler,
            segments={
                "train": (final_test_config['train_start'], final_test_config['train_end']),
                "valid": (final_test_config['valid_start'], final_test_config['valid_end']),
                "test": (final_test_config['test_start'], final_test_config['test_end']),
            }
        )

        # Prepare test data
        test_data = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_L)
        test_data = test_data.fillna(0).replace([np.inf, -np.inf], 0)

        print(f"    Test data shape: {test_data.shape}")
        print(f"    Test period: {final_test_config['test_start']} ~ {final_test_config['test_end']}")

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
            'train_start': final_test_config['train_start'],
            'train_end': final_test_config['train_end'],
            'valid_start': final_test_config['valid_start'],
            'valid_end': final_test_config['valid_end'],
            'test_start': final_test_config['test_start'],
            'test_end': final_test_config['test_end'],
        }

        def load_lightgbm_model(path):
            return lgb.Booster(model_file=str(path))

        def get_lightgbm_feature_count(m):
            return m.num_feature()

        run_backtest(
            model_path, dataset, test_pred, args, time_splits,
            model_name="LightGBM (Loaded Model)",
            load_model_func=load_lightgbm_model,
            get_feature_count_func=get_lightgbm_feature_count
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
    print("LightGBM Hyperopt with Time-Series Cross-Validation")
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
    params_file = output_dir / f"lightgbm_cv_best_params_{timestamp}.json"
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
                   if k not in ['num_threads', 'verbose', 'objective', 'metric', 'seed', 'boosting_type']}
            })

    history_df = pd.DataFrame(history)
    history_file = output_dir / f"lightgbm_cv_history_{timestamp}.csv"
    history_df.to_csv(history_file, index=False)
    print(f"Search history saved to: {history_file}")

    # 训练最终模型
    model, feature_names, test_pred, dataset = train_final_model(
        args, handler_config, symbols, best_params, final_test_config, top_features
    )

    # 评估
    test_period = f"{final_test_config['test_start']} ~ {final_test_config['test_end']}"
    print(f"\n[*] Final Evaluation on Test Set ({test_period})...")
    evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

    # 保存模型
    print("\n[*] Saving model...")
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model_filename = generate_model_filename(
        "lightgbm_cv", args, args.top_k if args.top_k > 0 else 0, ".txt"
    )
    model_path = MODEL_SAVE_PATH / model_filename

    # 保存 LightGBM 模型
    model.save_model(str(model_path))

    # 构造 time_splits 用于 meta_data
    time_splits = {
        'train_start': final_test_config['train_start'],
        'train_end': final_test_config['train_end'],
        'valid_start': final_test_config['valid_start'],
        'valid_end': final_test_config['valid_end'],
        'test_start': final_test_config['test_start'],
        'test_end': final_test_config['test_end'],
    }

    meta_data = create_meta_data(args, handler_config, time_splits, feature_names, "lightgbm_cv", args.top_k)
    meta_data['cv_params'] = best_params
    meta_data['cv_results'] = {
        'mean_ic': best_trial['mean_ic'],
        'std_ic': best_trial['std_ic'],
        'fold_results': best_trial['fold_results'],
    }

    # 保存元数据
    meta_path = model_path.with_suffix('.meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta_data, f)
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {meta_path}")

    # 回测
    if args.backtest:
        def load_lightgbm_model(path):
            return lgb.Booster(model_file=str(path))

        def get_lightgbm_feature_count(m):
            return m.num_feature()

        run_backtest(
            model_path, dataset, test_pred, args, time_splits,
            model_name="LightGBM (CV Hyperopt)",
            load_model_func=load_lightgbm_model,
            get_feature_count_func=get_lightgbm_feature_count
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
