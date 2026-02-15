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
    load_catboost_model,
    get_catboost_feature_count,
    # CV utilities
    CV_FOLDS,
    FINAL_TEST,
    create_data_handler_for_fold,
    create_dataset_for_fold,
    compute_time_decay_weights,
    prepare_features_and_labels,
    BaseCVHyperoptObjective,
    run_hyperopt_cv_search_generic,
    first_pass_feature_selection_generic,
    train_final_model_generic,
)


# ============================================================================
# 超参数搜索空间
# ============================================================================

SEARCH_SPACE = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.15)),
    'max_depth': scope.int(hp.quniform('max_depth', 4, 8, 1)),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 15),
    'random_strength': hp.uniform('random_strength', 0.1, 2),
    'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
    'subsample': hp.uniform('subsample', 0.6, 0.95),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.6, 0.95),
    'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 5, 100, 1)),
}

# 带 sample weight 搜索的扩展空间
SEARCH_SPACE_WITH_WEIGHT = {
    **SEARCH_SPACE,
    'sample_weight_halflife': hp.choice('sample_weight_halflife', [
        0, 3, 5, 8, 10, 15,
    ]),
}

CB_PARAM_FILTER_KEYS = {'thread_count', 'verbose', 'loss_function', 'random_seed'}


def create_catboost_params(hyperparams, thread_count=16):
    """将 hyperopt 参数转换为 CatBoost 参数（不含 sample_weight_halflife）"""
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
        'thread_count': thread_count,
        'verbose': False,
        'random_seed': 42,
    }


# ============================================================================
# CatBoost CV Objective (subclass of BaseCVHyperoptObjective)
# ============================================================================

class CatBoostCVObjective(BaseCVHyperoptObjective):
    """CatBoost-specific CV hyperopt objective with sample weight support."""

    def _init_extra(self, fixed_halflife=0, search_weight=False, **kwargs):
        self.fixed_halflife = fixed_halflife
        self.search_weight = search_weight

    def _augment_fold_data(self, entry, fold):
        train_weight = None
        valid_weight = None
        if self.fixed_halflife > 0:
            train_weight = compute_time_decay_weights(
                entry['train_data'].index, self.fixed_halflife)
            valid_weight = compute_time_decay_weights(
                entry['valid_data'].index, self.fixed_halflife)
        entry['train_weight'] = train_weight
        entry['valid_weight'] = valid_weight
        entry['train_pool'] = Pool(
            entry['train_data'], label=entry['train_label'],
            weight=train_weight)
        entry['valid_pool'] = Pool(
            entry['valid_data'], label=entry['valid_label'],
            weight=valid_weight)

    def create_model_params(self, hyperparams):
        return create_catboost_params(
            hyperparams, thread_count=self.args.thread_count)

    def train_and_predict_fold(self, fold_data, params, hyperparams):
        trial_halflife = 0
        if self.search_weight:
            trial_halflife = hyperparams.get('sample_weight_halflife', 0)
        elif self.fixed_halflife > 0:
            trial_halflife = self.fixed_halflife

        if self.search_weight and trial_halflife > 0:
            tw = compute_time_decay_weights(
                fold_data['train_data'].index, trial_halflife)
            vw = compute_time_decay_weights(
                fold_data['valid_data'].index, trial_halflife)
            train_pool = Pool(fold_data['train_data'],
                              label=fold_data['train_label'], weight=tw)
            valid_pool = Pool(fold_data['valid_data'],
                              label=fold_data['valid_label'], weight=vw)
        else:
            train_pool = fold_data['train_pool']
            valid_pool = fold_data['valid_pool']

        model = CatBoostRegressor(**params)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=50,
            verbose=100 if self.verbose else False,
        )

        train_pred = model.predict(fold_data['train_data'])
        valid_pred = model.predict(fold_data['valid_data'])
        return train_pred, valid_pred, model.best_iteration_

    def format_trial_params(self, hyperparams):
        return (f"lr={hyperparams['learning_rate']:.4f}, "
                f"depth={int(hyperparams['max_depth'])}, "
                f"l2={hyperparams['l2_leaf_reg']:.2f}, "
                f"subsample={hyperparams['subsample']:.2f}")

    def get_extra_trial_info(self, hyperparams):
        trial_halflife = 0
        if self.search_weight:
            trial_halflife = hyperparams.get('sample_weight_halflife', 0)
        elif self.fixed_halflife > 0:
            trial_halflife = self.fixed_halflife
        return f", halflife={trial_halflife}" if trial_halflife > 0 else ""


# ============================================================================
# Callbacks for generic functions
# ============================================================================

def catboost_train_and_get_importance(train_data, train_label,
                                      valid_data, valid_label, args):
    """CatBoost feature importance callback for first-pass selection."""
    train_pool = Pool(train_data, label=train_label)
    valid_pool = Pool(valid_data, label=valid_label)

    model = CatBoostRegressor(
        loss_function='RMSE',
        learning_rate=0.05,
        max_depth=6,
        l2_leaf_reg=3,
        random_strength=1,
        thread_count=args.thread_count,
        verbose=False,
    )
    model.fit(train_pool, eval_set=valid_pool,
              early_stopping_rounds=50, verbose_eval=100)

    return train_data.columns.tolist(), model.get_feature_importance()


def catboost_train_predict_final(train_data, train_label, valid_data,
                                  valid_label, test_data, best_params,
                                  halflife=0):
    """CatBoost training callback for final model."""
    train_weight = None
    valid_weight = None
    if halflife > 0:
        train_weight = compute_time_decay_weights(train_data.index, halflife)
        valid_weight = compute_time_decay_weights(valid_data.index, halflife)
        print(f"\n    Using sample weights: halflife={halflife} years")

    train_pool = Pool(train_data, label=train_label, weight=train_weight)
    valid_pool = Pool(valid_data, label=valid_label, weight=valid_weight)

    print("\n    Training progress:")
    model = CatBoostRegressor(**best_params)
    model.fit(train_pool, eval_set=valid_pool,
              early_stopping_rounds=50, verbose_eval=100)
    print(f"\n    Best iteration: {model.best_iteration_}")

    valid_pred = model.predict(valid_data)
    test_pred_values = model.predict(test_data)
    return model, valid_pred, test_pred_values


# ============================================================================
# Main
# ============================================================================

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

    # 样本权重参数
    parser.add_argument('--sample-weight-halflife', type=float, default=0,
                        help='Fixed time-decay half-life in years (0=disabled)')
    parser.add_argument('--search-sample-weight', action='store_true',
                        help='Include sample weight halflife in hyperopt search space')

    # 训练参数
    parser.add_argument('--thread-count', type=int, default=16)

    # 回测参数
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=1)
    parser.add_argument('--account', type=float, default=10000)
    parser.add_argument('--rebalance-freq', type=int, default=1)
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
                        help='Path to existing model file (.cbm) for backtest-only mode')

    args = parser.parse_args()

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

        # Load model and metadata
        model = load_catboost_model(model_path)
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
            args, handler_config, symbols, FINAL_TEST)
        dataset = create_dataset_for_fold(handler, FINAL_TEST)
        test_data, _ = prepare_features_and_labels(dataset, "test")

        test_pred_values = model.predict(test_data)
        test_pred = pd.Series(
            test_pred_values, index=test_data.index, name='score')

        evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

        time_splits = {k: FINAL_TEST[k] for k in [
            'train_start', 'train_end', 'valid_start', 'valid_end',
            'test_start', 'test_end']}
        run_backtest(
            model_path, dataset, test_pred, args, time_splits,
            model_name="CatBoost (Loaded Model)",
            load_model_func=load_catboost_model,
            get_feature_count_func=get_catboost_feature_count)
        return

    # =========================================================================
    # Normal mode: Hyperopt search + Training + Backtest
    # =========================================================================
    fixed_halflife = getattr(args, 'sample_weight_halflife', 0)
    search_weight = getattr(args, 'search_sample_weight', False)

    print("=" * 70)
    print("CatBoost Hyperopt with Time-Series Cross-Validation")
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Handler: {args.handler}")
    print(f"N-day: {args.nday}")
    print(f"Top-K features: {args.top_k}")
    print(f"Max evaluations: {args.max_evals}")
    if search_weight:
        print(f"Sample weight: SEARCH [0, 3, 5, 8, 10, 15 years]")
    elif fixed_halflife > 0:
        print(f"Sample weight: fixed halflife={fixed_halflife} years")
    print("=" * 70)

    init_qlib(handler_config['use_talib'])

    # Feature selection
    top_features = None
    if args.top_k > 0:
        importance_df = first_pass_feature_selection_generic(
            args, handler_config, symbols, catboost_train_and_get_importance)
        top_features = importance_df.head(args.top_k)['feature'].tolist()
        print(f"\n    Selected top {args.top_k} features")

    # Run CV hyperparameter search
    objective = CatBoostCVObjective(
        args, handler_config, symbols, top_features, verbose=args.verbose,
        fixed_halflife=fixed_halflife, search_weight=search_weight)

    search_space = SEARCH_SPACE_WITH_WEIGHT if search_weight else SEARCH_SPACE

    extra_header = []
    if search_weight:
        extra_header.append(
            "Sample weight: SEARCH [0, 3, 5, 8, 10, 15 years halflife]")
    elif fixed_halflife > 0:
        extra_header.append(
            f"Sample weight: fixed halflife={fixed_halflife} years")

    best, best_params, trials, best_trial = run_hyperopt_cv_search_generic(
        objective, search_space, args.max_evals,
        lambda b: create_catboost_params(b, thread_count=args.thread_count),
        param_filter_keys=CB_PARAM_FILTER_KEYS,
        extra_header_lines=extra_header)

    # Extract best halflife
    best_halflife = 0
    if search_weight:
        halflife_choices = [0, 3, 5, 8, 10, 15]
        best_halflife = halflife_choices[best.get('sample_weight_halflife', 0)]
        print(f"  sample_weight_halflife: {best_halflife}")
    elif fixed_halflife > 0:
        best_halflife = fixed_halflife

    # Save search results
    output_dir = PROJECT_ROOT / "outputs" / "hyperopt_cv"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    params_file = output_dir / f"catboost_cv_best_params_{timestamp}.json"
    params_to_save = {
        'params': {k: float(v) if isinstance(v, (np.floating, np.integer))
                   else v for k, v in best_params.items()},
        'sample_weight_halflife': best_halflife,
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
                   if k not in CB_PARAM_FILTER_KEYS}
            })
    history_df = pd.DataFrame(history)
    history_file = output_dir / f"catboost_cv_history_{timestamp}.csv"
    history_df.to_csv(history_file, index=False)
    print(f"Search history saved to: {history_file}")

    # Train final model
    model, feature_names, test_pred, dataset = train_final_model_generic(
        args, handler_config, symbols, best_params, FINAL_TEST,
        catboost_train_predict_final,
        top_features=top_features,
        param_filter_keys=CB_PARAM_FILTER_KEYS,
        halflife=best_halflife)

    # Evaluate
    print("\n[*] Final Evaluation on Test Set (2025)...")
    evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

    # Save model
    print("\n[*] Saving model...")
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model_filename = generate_model_filename(
        "catboost_cv", args, args.top_k if args.top_k > 0 else 0, ".cbm")
    model_path = MODEL_SAVE_PATH / model_filename

    time_splits = {k: FINAL_TEST[k] for k in [
        'train_start', 'train_end', 'valid_start', 'valid_end',
        'test_start', 'test_end']}
    meta_data = create_meta_data(
        args, handler_config, time_splits, feature_names,
        "catboost_cv", args.top_k)
    meta_data['cv_params'] = best_params
    if best_halflife > 0:
        meta_data['sample_weight_halflife'] = best_halflife
    meta_data['cv_results'] = {
        'mean_ic': best_trial['mean_ic'],
        'std_ic': best_trial['std_ic'],
        'fold_results': best_trial['fold_results'],
    }
    save_model_with_meta(model, model_path, meta_data)

    # Backtest
    if args.backtest:
        run_backtest(
            model_path, dataset, test_pred, args, time_splits,
            model_name="CatBoost (CV Hyperopt)",
            load_model_func=load_catboost_model,
            get_feature_count_func=get_catboost_feature_count)

    print("\n" + "=" * 70)
    print("CV HYPEROPT COMPLETE")
    print("=" * 70)
    print(f"CV Mean IC: {best_trial['mean_ic']:.4f} "
          f"(±{best_trial['std_ic']:.4f})")
    if best_halflife > 0:
        print(f"Sample weight halflife: {best_halflife} years")
    print(f"Model saved to: {model_path}")
    print(f"Best parameters: {params_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
