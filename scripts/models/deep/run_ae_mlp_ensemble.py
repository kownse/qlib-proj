"""
Multi-Seed AE-MLP Ensemble Training and Evaluation

Train multiple AE-MLP models with different random seeds, then average their
predictions for more robust final prediction.

Usage:
    python scripts/models/deep/run_ae_mlp_ensemble.py \
        --params-file outputs/hyperopt_cv/ae_mlp_cv_best_params_20260116_173601_best.json \
        --stock-pool sp500 --handler alpha158-enhanced-v7 --backtest
"""

# ============================================================================
# 重要: 以下代码必须在任何其他导入之前执行
# ============================================================================

import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

import sys
from pathlib import Path

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
)

# ============================================================================
# 现在可以安全地导入其他模块
# ============================================================================

import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from utils.utils import evaluate_model
from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG, PROJECT_ROOT, MODEL_SAVE_PATH,
    run_backtest,
    FINAL_TEST,
    create_data_handler_for_fold,
    create_dataset_for_fold,
    prepare_data_from_dataset,
    compute_ic,
)
from models.deep.ae_mlp_shared import build_ae_mlp_model, set_random_seed, create_tf_dataset


DEFAULT_SEEDS = [42, 123, 456, 789, 1000]


def load_params_from_json(params_file: str) -> dict:
    """从 JSON 文件加载超参数"""
    with open(params_file, 'r') as f:
        data = json.load(f)

    params = data['params']
    cv_results = data.get('cv_results', {})

    print(f"    hidden_units: {params['hidden_units']}")
    print(f"    lr: {params['lr']:.6f}, batch_size: {params['batch_size']}")
    if cv_results:
        print(f"    Original CV IC: {cv_results['mean_ic']:.4f} (±{cv_results['std_ic']:.4f})")

    return params


def train_single_model(X_train, y_train, X_valid, y_valid, params: dict, seed: int, n_epochs: int, early_stop: int):
    """训练单个模型"""
    tf.keras.backend.clear_session()
    set_random_seed(seed)

    model = build_ae_mlp_model(params)

    batch_size = params['batch_size']
    train_dataset = create_tf_dataset(X_train, y_train, batch_size, shuffle=True)
    valid_dataset = create_tf_dataset(X_valid, y_valid, batch_size, shuffle=False)

    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_action_loss',
            patience=early_stop,
            min_delta=1e-5,
            restore_best_weights=True,
            verbose=0,
            mode='min'
        ),
    ]

    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=n_epochs,
        callbacks=cb_list,
        verbose=0,
    )

    return model


def main():
    parser = argparse.ArgumentParser(description='AE-MLP Ensemble Training')

    parser.add_argument('--params-file', type=str, required=True,
                        help='Path to JSON file with hyperparameters')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated list of random seeds')
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='Number of seeds if --seeds not specified')
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--handler', type=str, default='alpha158-enhanced-v7',
                        choices=list(HANDLER_CONFIG.keys()))
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--n-epochs', type=int, default=50)
    parser.add_argument('--early-stop', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=1)
    parser.add_argument('--account', type=float, default=10000)
    parser.add_argument('--rebalance-freq', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss'])

    args = parser.parse_args()

    # Parse seeds
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        seeds = DEFAULT_SEEDS[:args.n_seeds]

    # Load params
    print(f"\n[1] Loading params: {args.params_file}")
    params = load_params_from_json(args.params_file)

    # Configuration
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    print(f"\n[2] Configuration:")
    print(f"    Handler: {args.handler}")
    print(f"    Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"    Seeds: {seeds}")
    print(f"    Train: {FINAL_TEST['train_start']} ~ {FINAL_TEST['train_end']}")
    print(f"    Valid: {FINAL_TEST['valid_start']} ~ {FINAL_TEST['valid_end']}")
    print(f"    Test:  {FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']}")

    # Setup GPU
    if args.gpu >= 0 and gpus:
        tf.config.set_visible_devices(gpus[args.gpu], 'GPU')

    # Prepare data
    print(f"\n[3] Preparing data...")
    handler = create_data_handler_for_fold(args, handler_config, symbols, FINAL_TEST)
    dataset = create_dataset_for_fold(handler, FINAL_TEST)

    X_train, y_train, _ = prepare_data_from_dataset(dataset, "train")
    X_valid, y_valid, valid_index = prepare_data_from_dataset(dataset, "valid")
    X_test, y_test, test_index = prepare_data_from_dataset(dataset, "test")

    print(f"    Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    # Update params
    params['num_columns'] = X_train.shape[1]

    # Train models
    print(f"\n[4] Training {len(seeds)} models...")
    print(f"    {'Seed':<10} {'Mean IC':<12} {'ICIR':<12}")
    print(f"    {'-'*34}")

    models = []
    individual_preds = []
    individual_ics = []

    for seed in seeds:
        model = train_single_model(
            X_train, y_train, X_valid, y_valid,
            params, seed, args.n_epochs, args.early_stop
        )
        models.append(model)

        # Predict on test
        _, _, pred = model.predict(X_test, batch_size=params['batch_size'], verbose=0)
        pred = pred.flatten()
        individual_preds.append(pred)

        # Compute IC
        mean_ic, ic_std, icir = compute_ic(pred, y_test, test_index)
        individual_ics.append({'seed': seed, 'mean_ic': mean_ic, 'icir': icir})
        print(f"    {seed:<10} {mean_ic:<12.4f} {icir:<12.4f}")

    # Summary
    mean_individual_ic = np.mean([ic['mean_ic'] for ic in individual_ics])
    std_individual_ic = np.std([ic['mean_ic'] for ic in individual_ics])
    print(f"    {'-'*34}")
    print(f"    {'Avg':<10} {mean_individual_ic:<12.4f} (std: {std_individual_ic:.4f})")

    # Ensemble prediction
    print(f"\n[5] Ensemble Prediction...")
    ensemble_pred = np.mean(individual_preds, axis=0)
    ensemble_ic, ensemble_std, ensemble_icir = compute_ic(ensemble_pred, y_test, test_index)

    improvement = ensemble_ic - mean_individual_ic
    improvement_pct = improvement / abs(mean_individual_ic) * 100 if mean_individual_ic != 0 else 0

    print(f"\n    ╔════════════════════════════════════════════════════════╗")
    print(f"    ║  Ensemble vs Individual Comparison                     ║")
    print(f"    ╠════════════════════════════════════════════════════════╣")
    print(f"    ║  Individual Mean IC:  {mean_individual_ic:>8.4f} (std: {std_individual_ic:.4f})     ║")
    print(f"    ║  Ensemble Mean IC:    {ensemble_ic:>8.4f}                    ║")
    print(f"    ║  Improvement:         {improvement:>+8.4f} ({improvement_pct:+.1f}%)             ║")
    print(f"    ║  Ensemble ICIR:       {ensemble_icir:>8.4f}                    ║")
    print(f"    ╚════════════════════════════════════════════════════════╝")

    # Create prediction Series for evaluation
    ensemble_pred_series = pd.Series(ensemble_pred, index=test_index, name='score')

    # Evaluate
    print("\n[6] Evaluation...")
    evaluate_model(dataset, ensemble_pred_series, PROJECT_ROOT, args.nday)

    # Save results
    print("\n[7] Saving Results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / "ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_path = output_dir / f"ae_mlp_ensemble_pred_{args.handler}_{args.stock_pool}_{timestamp}.parquet"
    ensemble_pred_series.to_frame("score").to_parquet(pred_path)
    print(f"    Predictions: {pred_path}")

    results = {
        'params_file': str(args.params_file),
        'seeds': seeds,
        'handler': args.handler,
        'stock_pool': args.stock_pool,
        'individual_results': individual_ics,
        'individual_mean_ic': float(mean_individual_ic),
        'ensemble_mean_ic': float(ensemble_ic),
        'ensemble_icir': float(ensemble_icir),
        'improvement': float(improvement),
    }

    results_path = output_dir / f"ae_mlp_ensemble_results_{args.handler}_{args.stock_pool}_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"    Results: {results_path}")

    # Backtest
    if args.backtest:
        print("\n[8] Backtest...")
        pred_df = ensemble_pred_series.to_frame("score")

        # Save model for backtest
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_SAVE_PATH / f"ae_mlp_ensemble_{args.handler}_{args.stock_pool}_{timestamp}.keras"
        models[0].save(str(model_path))

        time_splits = {
            'train_start': FINAL_TEST['train_start'],
            'train_end': FINAL_TEST['train_end'],
            'valid_start': FINAL_TEST['valid_start'],
            'valid_end': FINAL_TEST['valid_end'],
            'test_start': FINAL_TEST['test_start'],
            'test_end': FINAL_TEST['test_end'],
        }

        def load_model(path):
            return keras.models.load_model(str(path))

        def get_feature_count(m):
            return m.input_shape[1]

        run_backtest(
            model_path, dataset, pred_df, args, time_splits,
            model_name="AE-MLP Ensemble",
            load_model_func=load_model,
            get_feature_count_func=get_feature_count
        )

    print("\n" + "=" * 70)
    print("AE-MLP Ensemble Completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
