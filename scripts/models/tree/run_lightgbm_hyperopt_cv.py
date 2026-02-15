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
from hyperopt import hp, STATUS_OK
from hyperopt.pyll import scope

from utils.utils import evaluate_model
from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG, PROJECT_ROOT, MODEL_SAVE_PATH,
    init_qlib,
    save_model_with_meta,
    create_meta_data,
    generate_model_filename,
    run_backtest,
    load_lightgbm_model,
    get_lightgbm_feature_count,
    # CV utilities
    FINAL_TEST,
    create_data_handler_for_fold,
    create_dataset_for_fold,
    prepare_features_and_labels,
    BaseCVHyperoptObjective,
    run_hyperopt_cv_search_generic,
    first_pass_feature_selection_generic,
    train_final_model_generic,
)

# 扩展训练期配置：训练到 2024，测试 2025
# 这样可以让模型学习更近期的市场模式
EXTENDED_FINAL_TEST = {
    'train_start': '2000-01-01',
    'train_end': '2024-12-31',
    'valid_start': '2025-01-01',
    'valid_end': '2025-06-30',
    'test_start': '2025-07-01',
    'test_end': '2025-12-31',
}


# ============================================================================
# 超参数搜索空间
# ============================================================================

SEARCH_SPACE = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.1)),
    'num_leaves': scope.int(hp.quniform('num_leaves', 64, 256, 1)),
    'max_depth': scope.int(hp.quniform('max_depth', 6, 10, 1)),
    'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 20, 100, 1)),
    'feature_fraction': hp.uniform('feature_fraction', 0.6, 0.95),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 0.95),
    'bagging_freq': scope.int(hp.quniform('bagging_freq', 1, 10, 1)),
    'lambda_l1': hp.loguniform('lambda_l1', np.log(1e-4), np.log(1)),
    'lambda_l2': hp.loguniform('lambda_l2', np.log(0.1), np.log(10)),
}

LGB_PARAM_FILTER_KEYS = {
    'num_threads', 'verbose', 'objective', 'metric', 'seed', 'boosting_type'}


def create_lightgbm_params(hyperparams):
    """将 hyperopt 参数转换为 LightGBM 参数"""
    return {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'learning_rate': hyperparams['learning_rate'],
        'num_leaves': int(hyperparams['num_leaves']),
        'max_depth': int(hyperparams['max_depth']),
        'min_data_in_leaf': int(hyperparams['min_data_in_leaf']),
        'feature_fraction': hyperparams['feature_fraction'],
        'bagging_fraction': hyperparams['bagging_fraction'],
        'bagging_freq': int(hyperparams['bagging_freq']),
        'lambda_l1': hyperparams['lambda_l1'],
        'lambda_l2': hyperparams['lambda_l2'],
        'num_threads': 8,
        'verbose': -1,
        'seed': 42,
    }


# ============================================================================
# LightGBM CV Objective (subclass of BaseCVHyperoptObjective)
# ============================================================================

class LightGBMCVObjective(BaseCVHyperoptObjective):
    """LightGBM-specific CV hyperopt objective."""

    def create_model_params(self, hyperparams):
        return create_lightgbm_params(hyperparams)

    def train_and_predict_fold(self, fold_data, params, hyperparams):
        train_set = lgb.Dataset(
            fold_data['train_data'], label=fold_data['train_label'])
        valid_set = lgb.Dataset(
            fold_data['valid_data'], label=fold_data['valid_label'],
            reference=train_set)

        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        if self.verbose:
            callbacks.append(lgb.log_evaluation(period=100))

        model = lgb.train(
            params,
            train_set,
            num_boost_round=500,
            valid_sets=[valid_set],
            valid_names=['valid'],
            callbacks=callbacks,
        )

        best_iter = model.best_iteration
        train_pred = model.predict(
            fold_data['train_data'], num_iteration=best_iter)
        valid_pred = model.predict(
            fold_data['valid_data'], num_iteration=best_iter)
        return train_pred, valid_pred, best_iter

    def format_trial_params(self, hyperparams):
        return (f"lr={hyperparams['learning_rate']:.4f}, "
                f"leaves={int(hyperparams['num_leaves'])}, "
                f"depth={int(hyperparams['max_depth'])}, "
                f"bagging={hyperparams['bagging_fraction']:.2f}")


# ============================================================================
# Callbacks for generic functions
# ============================================================================

def lightgbm_train_and_get_importance(train_data, train_label,
                                       valid_data, valid_label, args):
    """LightGBM feature importance callback for first-pass selection."""
    train_set = lgb.Dataset(train_data, label=train_label)
    valid_set = lgb.Dataset(valid_data, label=valid_label, reference=train_set)

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
        num_boost_round=500,
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    importance = model.feature_importance(importance_type='gain')
    return train_data.columns.tolist(), importance


def lightgbm_train_predict_final(train_data, train_label, valid_data,
                                   valid_label, test_data, best_params):
    """LightGBM training callback for final model."""
    train_set = lgb.Dataset(train_data, label=train_label)
    valid_set = lgb.Dataset(valid_data, label=valid_label, reference=train_set)

    print("\n    Training progress:")
    model = lgb.train(
        best_params,
        train_set,
        num_boost_round=500,
        valid_sets=[valid_set],
        valid_names=['valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    print(f"\n    Best iteration: {model.best_iteration}")

    valid_pred = model.predict(valid_data, num_iteration=model.best_iteration)
    test_pred_values = model.predict(
        test_data, num_iteration=model.best_iteration)
    return model, valid_pred, test_pred_values


# ============================================================================
# Main
# ============================================================================

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
                        help='使用扩展训练期 (train到2024, valid/test在2025)')

    # 回测参数
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=1)
    parser.add_argument('--account', type=float, default=10000)
    parser.add_argument('--rebalance-freq', type=int, default=5)
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss'])

    # 策略参数
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

    # 模型加载
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to existing model file (.txt) for backtest-only mode')

    args = parser.parse_args()

    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    # 选择训练期配置
    if getattr(args, 'extended_train', False):
        final_test_config = EXTENDED_FINAL_TEST
        print("[*] Using EXTENDED training period (train to 2024, test on 2025)")
    else:
        final_test_config = FINAL_TEST
        print("[*] Using STANDARD training period "
              "(train to 2023, valid 2024, test 2025)")

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

        # Load model and metadata
        model = load_lightgbm_model(model_path)
        meta_path = model_path.with_suffix('.meta.pkl')
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                meta_data = pickle.load(f)
            if 'handler' in meta_data:
                args.handler = meta_data['handler']
                handler_config = HANDLER_CONFIG[args.handler]
            if 'nday' in meta_data:
                args.nday = meta_data['nday']

        init_qlib(handler_config['use_talib'])

        # Create dataset and predict
        handler = create_data_handler_for_fold(
            args, handler_config, symbols, final_test_config)
        dataset = create_dataset_for_fold(handler, final_test_config)
        test_data, _ = prepare_features_and_labels(dataset, "test")

        test_pred_values = model.predict(test_data)
        test_pred = pd.Series(
            test_pred_values, index=test_data.index, name='score')

        evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

        time_splits = {k: final_test_config[k] for k in [
            'train_start', 'train_end', 'valid_start', 'valid_end',
            'test_start', 'test_end']}
        run_backtest(
            model_path, dataset, test_pred, args, time_splits,
            model_name="LightGBM (Loaded Model)",
            load_model_func=load_lightgbm_model,
            get_feature_count_func=get_lightgbm_feature_count)
        return

    # =========================================================================
    # Normal mode: Hyperopt search + Training + Backtest
    # =========================================================================
    print("=" * 70)
    print("LightGBM Hyperopt with Time-Series Cross-Validation")
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Handler: {args.handler}")
    print(f"N-day: {args.nday}")
    print(f"Top-K features: {args.top_k}")
    print(f"Max evaluations: {args.max_evals}")
    print("=" * 70)

    init_qlib(handler_config['use_talib'])

    # Feature selection
    top_features = None
    if args.top_k > 0:
        importance_df = first_pass_feature_selection_generic(
            args, handler_config, symbols, lightgbm_train_and_get_importance)
        top_features = importance_df.head(args.top_k)['feature'].tolist()
        print(f"\n    Selected top {args.top_k} features")

    # Run CV hyperparameter search
    objective = LightGBMCVObjective(
        args, handler_config, symbols, top_features, verbose=args.verbose)

    best, best_params, trials, best_trial = run_hyperopt_cv_search_generic(
        objective, SEARCH_SPACE, args.max_evals,
        create_lightgbm_params,
        param_filter_keys=LGB_PARAM_FILTER_KEYS)

    # Save search results
    output_dir = PROJECT_ROOT / "outputs" / "hyperopt_cv"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    params_file = output_dir / f"lightgbm_cv_best_params_{timestamp}.json"
    params_to_save = {
        'params': {k: float(v) if isinstance(v, (np.floating, np.integer))
                   else v for k, v in best_params.items()},
        'cv_results': {
            'mean_ic': best_trial['mean_ic'],
            'std_ic': best_trial['std_ic'],
            'fold_results': best_trial['fold_results'],
        }
    }
    with open(params_file, 'w') as f:
        json.dump(params_to_save, f, indent=2)
    print(f"\nBest parameters saved to: {params_file}")

    history = []
    for t in trials.trials:
        if t['result']['status'] == STATUS_OK:
            history.append({
                'mean_ic': t['result']['mean_ic'],
                'std_ic': t['result']['std_ic'],
                **{k: float(v) if isinstance(v, (np.floating, np.integer))
                   else v for k, v in t['result']['params'].items()
                   if k not in LGB_PARAM_FILTER_KEYS}
            })
    history_df = pd.DataFrame(history)
    history_file = output_dir / f"lightgbm_cv_history_{timestamp}.csv"
    history_df.to_csv(history_file, index=False)
    print(f"Search history saved to: {history_file}")

    # Train final model
    model, feature_names, test_pred, dataset = train_final_model_generic(
        args, handler_config, symbols, best_params, final_test_config,
        lightgbm_train_predict_final,
        top_features=top_features,
        param_filter_keys=LGB_PARAM_FILTER_KEYS)

    # Evaluate
    test_period = (f"{final_test_config['test_start']} ~ "
                   f"{final_test_config['test_end']}")
    print(f"\n[*] Final Evaluation on Test Set ({test_period})...")
    evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

    # Save model
    print("\n[*] Saving model...")
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model_filename = generate_model_filename(
        "lightgbm_cv", args, args.top_k if args.top_k > 0 else 0, ".txt")
    model_path = MODEL_SAVE_PATH / model_filename

    time_splits = {k: final_test_config[k] for k in [
        'train_start', 'train_end', 'valid_start', 'valid_end',
        'test_start', 'test_end']}
    meta_data = create_meta_data(
        args, handler_config, time_splits, feature_names,
        "lightgbm_cv", args.top_k)
    meta_data['cv_params'] = best_params
    meta_data['cv_results'] = {
        'mean_ic': best_trial['mean_ic'],
        'std_ic': best_trial['std_ic'],
        'fold_results': best_trial['fold_results'],
    }
    save_model_with_meta(model, model_path, meta_data,
                         save_func=lambda m, p: m.save_model(p))

    # Backtest
    if args.backtest:
        run_backtest(
            model_path, dataset, test_pred, args, time_splits,
            model_name="LightGBM (CV Hyperopt)",
            load_model_func=load_lightgbm_model,
            get_feature_count_func=get_lightgbm_feature_count)

    print("\n" + "=" * 70)
    print("CV HYPEROPT COMPLETE")
    print("=" * 70)
    print(f"CV Mean IC: {best_trial['mean_ic']:.4f} "
          f"(±{best_trial['std_ic']:.4f})")
    print(f"Model saved to: {model_path}")
    print(f"Best parameters: {params_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
