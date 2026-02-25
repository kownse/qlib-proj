"""
AE-MLP + AE-MLP (mkt-neutral) + AE-MLP-V9R + CatBoost + DoubleAdapt-FiLM Ensemble (V7)

Five-model ensemble:
  1. AE-MLP (alpha158-enhanced-v7, train 2000-2023)
  2. AE-MLP (v9-mkt-neutral, train 2000-2023)
  3. AE-MLP V9R (alpha158-enhanced-v9, train 2015-2023)
  4. CatBoost (catboost-v1)
  5. DoubleAdapt-FiLM (alpha300 + macro, meta-learning incremental)

Based on run_ae_cb_ensemble_v5.py with DoubleAdapt-FiLM model added.

Usage:
    python scripts/models/ensemble/run_ae_cb_ensemble_v7.py
    python scripts/models/ensemble/run_ae_cb_ensemble_v7.py --ensemble-method rank_mean
    python scripts/models/ensemble/run_ae_cb_ensemble_v7.py --stacking
    python scripts/models/ensemble/run_ae_cb_ensemble_v7.py --backtest --topk 10 --n-drop 2
"""

import os

# Set thread limits before any other imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
from pathlib import Path
import multiprocessing

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Add scripts directory to path
script_dir = Path(__file__).parent.parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

import argparse
import json
import numpy as np
import pandas as pd
import torch

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from utils.talib_ops import TALIB_OPS
from utils.ai_filter import apply_ai_affinity_filter
from data.stock_pools import STOCK_POOLS

from models.common import (
    PROJECT_ROOT,
    QLIB_DATA_PATH,
    MODEL_SAVE_PATH,
    CV_FOLDS,
    FINAL_TEST,
)
from models.common.ensemble import (
    load_ae_mlp_model,
    load_model_meta,
    create_ensemble_data_handler,
    create_ensemble_dataset,
    predict_with_ae_mlp,
    predict_with_catboost,
    zscore_by_day,
    calculate_pairwise_correlations,
    compute_ic,
    ensemble_predictions,
    learn_optimal_weights,
    run_ensemble_backtest,
)
from models.common.training import load_catboost_model
from models.common.handlers import get_handler_class
from models.common.macro_features import load_macro_df, get_macro_cols


# ── V7-specific helpers ──────────────────────────────────────────────

def align_features_for_catboost(X, model):
    """Align data features to match CatBoost model's expected feature names"""
    model_features = model.feature_names_
    if not model_features:
        return X

    if isinstance(X.columns, pd.MultiIndex):
        X = X.copy()
        X.columns = [col[1] if isinstance(col, tuple) else col for col in X.columns]

    data_cols = set(X.columns)
    missing = [c for c in model_features if c not in data_cols]
    if missing:
        for c in missing:
            X[c] = 0.0

    return X[list(model_features)]


def predict_catboost_raw(model, dataset, segment):
    """Predict with CatBoost using raw data, with feature alignment"""
    data = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
    data = data.fillna(0).replace([np.inf, -np.inf], 0)
    data = align_features_for_catboost(data, model)
    pred_values = model.predict(data)
    return pd.Series(pred_values, index=data.index, name='score')


# ── DoubleAdapt prediction ───────────────────────────────────────────

def generate_doubleadapt_predictions(checkpoint_path, meta_json_path, symbols):
    """Generate DoubleAdapt-FiLM predictions from checkpoint.

    This runs the full DoubleAdapt inference pipeline:
    1. Load macro data
    2. Create alpha300 dataset
    3. Build rolling tasks
    4. Load model from checkpoint
    5. Run incremental learning inference

    Returns:
        pd.Series with MultiIndex (datetime, instrument)
    """
    from models.deep.doubleadapt.net import GRUFiLM
    from models.deep.doubleadapt.model import DoubleAdaptFiLMManager
    from models.deep.doubleadapt.utils import (
        TimeAdjuster, organize_all_tasks, get_rolling_data_with_macro,
    )

    # Load metadata
    with open(meta_json_path) as f:
        meta = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Constants
    d_feat = 5
    seq_len = 60
    x_dim = 300
    factor_num = d_feat

    macro_set = meta.get('macro_set', 'minimal')
    n_macro = meta.get('n_macro', 6)
    hidden_size = meta.get('hidden_size', 64)
    num_layers = meta.get('num_layers', 2)
    film_hidden = meta.get('film_hidden', 32)
    num_head = meta.get('num_head', 8)
    tau = meta.get('tau', 10.0)
    step = meta.get('step', 20)
    nday = meta.get('nday', 5)
    lr = meta.get('lr', 0.01)
    lr_da = meta.get('lr_da', 0.01)
    lr_ma = meta.get('lr_ma', 0.01)

    print(f"    Config: hidden={hidden_size}, layers={num_layers}, macro={macro_set}({n_macro})")
    print(f"    Step={step}, nday={nday}, lr={lr}")

    # [1] Load macro data
    macro_df = load_macro_df()
    macro_cols = get_macro_cols(macro_set)
    available_macro = [c for c in macro_cols if c in macro_df.columns]
    n_macro = len(available_macro)

    # [2] Create alpha300 dataset
    HandlerClass = get_handler_class('alpha300')
    handler = HandlerClass(
        volatility_window=nday,
        instruments=symbols,
        start_time=FINAL_TEST['train_start'],
        end_time=FINAL_TEST['test_end'],
        fit_start_time=FINAL_TEST['train_start'],
        fit_end_time=FINAL_TEST['train_end'],
    )
    dataset = DatasetH(handler=handler, segments={
        "train": (FINAL_TEST['train_start'], FINAL_TEST['train_end']),
        "valid": (FINAL_TEST['valid_start'], FINAL_TEST['valid_end']),
        "test": (FINAL_TEST['test_start'], FINAL_TEST['test_end']),
    })

    # [3] Extract full dataframe
    print("    Extracting full dataframe...")
    features = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
    labels = dataset.prepare("train", col_set="label", data_key=DataHandlerLP.DK_L)
    features_v = dataset.prepare("valid", col_set="feature", data_key=DataHandlerLP.DK_L)
    labels_v = dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
    features_t = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_L)
    labels_t = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)

    all_features = pd.concat([features, features_v, features_t])
    all_labels = pd.concat([labels, labels_v, labels_t])
    all_features = all_features.fillna(0).replace([np.inf, -np.inf], 0)
    all_labels = all_labels.fillna(0)

    stock_data = pd.concat(
        {"feature": all_features, "label": all_labels},
        axis=1,
    )
    print(f"    Full data: {stock_data.shape}")

    # [4] Build rolling tasks
    from qlib.data import D
    cal = D.calendar(start_time='2000-01-01', end_time='2026-12-31', freq='day')
    calendar = list(cal)

    ta = TimeAdjuster(calendar)
    segments = {
        'train': (FINAL_TEST['train_start'], FINAL_TEST['train_end']),
        'valid': (FINAL_TEST['valid_start'], FINAL_TEST['valid_end']),
        'test': (FINAL_TEST['test_start'], FINAL_TEST['test_end']),
    }
    rolling_tasks = organize_all_tasks(segments, ta, step=step, trunc_days=2)
    print(f"    Test tasks: {len(rolling_tasks['test'])}")

    # [5] Prepare task data with macro
    print("    Preparing task data...")
    meta_tasks = {}
    for phase in ['train', 'valid', 'test']:
        meta_tasks[phase] = get_rolling_data_with_macro(
            rolling_tasks[phase],
            stock_data,
            macro_df=macro_df,
            macro_cols=available_macro,
            factor_num=factor_num,
            horizon=nday,
            sequence_last_dim=True,
            macro_lag=1,
        )

    # [6] Create and load model
    print("    Loading model from checkpoint...")
    gru_film = GRUFiLM(
        d_feat=d_feat,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.0,
        n_macro=n_macro,
        film_hidden=film_hidden,
    )

    manager = DoubleAdaptFiLMManager(
        model=gru_film,
        lr_model=lr,
        lr_da=lr_da,
        lr_ma=lr_ma,
        reg=0.5,
        adapt_x=meta.get('adapt_x', True),
        adapt_y=meta.get('adapt_y', True),
        first_order=meta.get('first_order', True),
        factor_num=factor_num,
        x_dim=x_dim,
        need_permute=False,
        num_head=num_head,
        temperature=tau,
        begin_valid_epoch=30,
    )

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    manager.load_state_dict(state_dict)

    # [7] Inference on test set
    # The checkpoint already contains the final trained state (after validation pass),
    # so we go directly to online incremental learning on test data.
    print("    Running test inference (incremental learning)...")
    pred_y = manager.inference(meta_tasks['test'])

    # Extract predictions as Series
    pred_series = pred_y["pred"]
    pred_series.name = 'score'

    return pred_series


def load_or_generate_da_predictions(checkpoint_path, meta_json_path, symbols, cache_path):
    """Load cached DoubleAdapt predictions or generate them.

    Args:
        checkpoint_path: path to .pt checkpoint
        meta_json_path: path to .json metadata
        symbols: list of stock symbols
        cache_path: path to pickle cache file

    Returns:
        pd.Series with MultiIndex (datetime, instrument)
    """
    cache_path = Path(cache_path)

    if cache_path.exists():
        print(f"    Loading cached predictions from {cache_path}")
        pred_series = pd.read_pickle(cache_path)
        print(f"    Loaded {len(pred_series)} predictions")
        return pred_series

    print("    No cached predictions found, running full DoubleAdapt inference...")
    print("    (This may take 30-60 minutes for SP500)")
    pred_series = generate_doubleadapt_predictions(
        checkpoint_path, meta_json_path, symbols,
    )

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pred_series.to_pickle(cache_path)
    print(f"    Saved predictions cache to {cache_path}")

    return pred_series


# ── CV evaluation ─────────────────────────────────────────────────────

def run_cv_evaluation(models: dict, handlers: dict, args, symbols, MODEL_CONFIG,
                      da_pred_test=None):
    """Evaluate ensemble IC on CV folds.

    Note: DoubleAdapt predictions are only available for test period (2025),
    not for CV folds. CV evaluation uses the 4 non-DA models only,
    then reports full 5-model test IC separately.
    """
    # Models that participate in CV (all except DoubleAdapt)
    cv_models = [k for k in models if k != 'da']
    all_models = list(models.keys())
    display_names = {k: cfg['display'] for k, cfg in MODEL_CONFIG.items()}

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION EVALUATION (5-Model Ensemble V7)")
    print("=" * 80)
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: valid {fold['valid_start']}~{fold['valid_end']}")
    print(f"Test Set: {FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']} (2025)")
    print(f"Ensemble Method: {args.ensemble_method}")
    print(f"Note: DA-FiLM only available for test period (incremental learning)")
    print("=" * 80)

    # Prepare 2025 test set
    print("\n[CV] Preparing 2025 test data...")
    test_time = {
        'train_start': FINAL_TEST['train_start'],
        'train_end': FINAL_TEST['train_end'],
        'valid_start': FINAL_TEST['valid_start'],
        'valid_end': FINAL_TEST['valid_end'],
        'test_start': FINAL_TEST['test_start'],
        'test_end': FINAL_TEST['test_end'],
    }

    # Create test datasets for non-DA models
    test_datasets = {}
    for key in cv_models:
        h = create_ensemble_data_handler(handlers[key], symbols, test_time, args.nday,
                                         include_valid=True)
        test_datasets[key] = create_ensemble_dataset(h, test_time, include_valid=True)

    # Test predictions
    test_preds = {}
    for key in cv_models:
        if key == 'cb':
            test_preds[key] = predict_catboost_raw(models[key], test_datasets[key], "test")
        else:
            test_preds[key] = predict_with_ae_mlp(models[key], test_datasets[key], segment="test")

    # Add DA predictions for test set
    if da_pred_test is not None:
        test_preds['da'] = da_pred_test

    weights = None
    if args.ensemble_method in ['weighted', 'zscore_weighted']:
        weights = {n: getattr(args, f'{n}_weight') for n in all_models}

    # Full ensemble with all 5 models (test only)
    test_pred_ens = ensemble_predictions(test_preds, args.ensemble_method, weights)

    # Also compute 4-model ensemble (without DA) for comparison
    test_pred_ens_v5 = ensemble_predictions(
        {k: v for k, v in test_preds.items() if k != 'da'},
        args.ensemble_method,
        {k: v for k, v in weights.items() if k != 'da'} if weights else None,
    )

    # Labels from first AE-MLP dataset
    test_label_df = test_datasets['ae'].prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)
    test_label = test_label_df.iloc[:, 0] if isinstance(test_label_df, pd.DataFrame) else test_label_df

    test_ics = {}
    for key in all_models:
        if key in test_preds:
            ic, _, _, _ = compute_ic(test_preds[key], test_label)
            test_ics[key] = ic
    test_ens_ic, _, _, _ = compute_ic(test_pred_ens, test_label)
    test_ens_v5_ic, _, _, _ = compute_ic(test_pred_ens_v5, test_label)

    parts = ", ".join(f"{display_names[k]} IC={test_ics[k]:.4f}" for k in all_models if k in test_ics)
    print(f"    Test (2025): {parts}")
    print(f"    V5 Ensemble IC={test_ens_v5_ic:.4f}, V7 Ensemble IC={test_ens_ic:.4f}")

    # CV folds (without DA)
    fold_results = []
    for fold in CV_FOLDS:
        print(f"\n[CV] Evaluating on {fold['name']}...")
        fold_time = {
            'train_start': fold['train_start'],
            'train_end': fold['train_end'],
            'valid_start': fold['valid_start'],
            'valid_end': fold['valid_end'],
            'test_start': fold['valid_start'],
            'test_end': fold['valid_end'],
        }

        fold_datasets = {}
        for key in cv_models:
            h = create_ensemble_data_handler(handlers[key], symbols, fold_time, args.nday,
                                             include_valid=True)
            fold_datasets[key] = create_ensemble_dataset(h, fold_time, include_valid=True)

        val_preds = {}
        for key in cv_models:
            if key == 'cb':
                val_preds[key] = predict_catboost_raw(models[key], fold_datasets[key], "valid")
            else:
                val_preds[key] = predict_with_ae_mlp(models[key], fold_datasets[key], segment="valid")

        cv_weights = None
        if weights:
            cv_weights = {k: v for k, v in weights.items() if k != 'da'}
        val_pred_ens = ensemble_predictions(val_preds, args.ensemble_method, cv_weights)

        val_label_df = fold_datasets['ae'].prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
        val_label = val_label_df.iloc[:, 0] if isinstance(val_label_df, pd.DataFrame) else val_label_df

        fold_ic = {}
        for key in cv_models:
            ic, _, _, _ = compute_ic(val_preds[key], val_label)
            fold_ic[key] = ic
        ens_ic, _, ens_icir, _ = compute_ic(val_pred_ens, val_label)

        result = {'name': fold['name'], 'ens_ic': ens_ic, 'ens_icir': ens_icir}
        for key in cv_models:
            result[f'{key}_ic'] = fold_ic[key]
        fold_results.append(result)

        parts = ", ".join(f"{display_names[k]}={fold_ic[k]:.4f}" for k in cv_models)
        print(f"    {fold['name']}: {parts}, Ensemble(4)={ens_ic:.4f}")

    # Summary table
    header_parts = "".join(f" {display_names[k]:>12s}" for k in cv_models)
    print("\n" + "=" * 80)
    print("CV EVALUATION COMPLETE (Ensemble V7)")
    print("=" * 80)
    print(f"{'':25s}{header_parts} {'Ens(4-mdl)':>12s}")
    print(f"{'-'*25}" + " ".join([f"{'-'*12}"] * (len(cv_models) + 1)))
    for r in fold_results:
        vals = "".join(f" {r[f'{k}_ic']:>12.4f}" for k in cv_models)
        print(f"{r['name']:<25s}{vals} {r['ens_ic']:>12.4f}")
    print(f"{'-'*25}" + " ".join([f"{'-'*12}"] * (len(cv_models) + 1)))

    mean_vals = "".join(f" {np.mean([r[f'{k}_ic'] for r in fold_results]):>12.4f}" for k in cv_models)
    std_vals = "".join(f" {np.std([r[f'{k}_ic'] for r in fold_results]):>12.4f}" for k in cv_models)
    test_vals = "".join(f" {test_ics.get(k, 0):>12.4f}" for k in cv_models)

    ens_ics = [r['ens_ic'] for r in fold_results]
    print(f"{'Valid Mean IC':<25s}{mean_vals} {np.mean(ens_ics):>12.4f}")
    print(f"{'Valid IC Std':<25s}{std_vals} {np.std(ens_ics):>12.4f}")
    print(f"{'Test IC (2025)':<25s}{test_vals} {test_ens_v5_ic:>12.4f}")
    print(f"\n  Full 5-model Test IC (V7): {test_ens_ic:.4f}  (vs 4-model: {test_ens_v5_ic:.4f})")
    if 'da' in test_ics:
        print(f"  DA-FiLM Test IC: {test_ics['da']:.4f}")
    print("=" * 80)

    return fold_results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='AE-MLP + AE-MLP(mkt-neutral) + AE-MLP-V9R + CatBoost + DA-FiLM Ensemble (V7)',
    )

    # Model paths (4 original models)
    parser.add_argument('--ae-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'ae_mlp_cv_alpha158-enhanced-v7_sp500_5d_best.keras'),
                        help='AE-MLP V7 model path (.keras)')
    parser.add_argument('--ae-mn-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'ae_mlp_cv_v9-mkt-neutral_sp500_5d.keras'),
                        help='AE-MLP market-neutral model path (.keras)')
    parser.add_argument('--ae-v9r-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'ae_mlp_cv_alpha158-enhanced-v9_sp500_5d_20_from_2015.keras'),
                        help='AE-MLP V9 rolling window model path (.keras)')
    parser.add_argument('--cb-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'catboost_cv_catboost-v1_sp500_5d_20260129_141353_best.cbm'),
                        help='CatBoost model path (.cbm)')

    # DoubleAdapt model
    parser.add_argument('--da-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'doubleadapt_da_film_sp500_minimal_5d_best.pt'),
                        help='DoubleAdapt-FiLM checkpoint path (.pt)')
    parser.add_argument('--da-meta', type=str,
                        default=str(MODEL_SAVE_PATH / 'doubleadapt_da_film_sp500_minimal_5d_best.json'),
                        help='DoubleAdapt-FiLM metadata path (.json)')
    parser.add_argument('--da-preds', type=str, default='',
                        help='Pre-computed DA predictions pickle (skip inference if provided)')

    # Handler configuration
    parser.add_argument('--ae-handler', type=str, default='alpha158-enhanced-v7')
    parser.add_argument('--ae-mn-handler', type=str, default='v9-mkt-neutral')
    parser.add_argument('--ae-v9r-handler', type=str, default='alpha158-enhanced-v9')
    parser.add_argument('--cb-handler', type=str, default='catboost-v1')

    # Ensemble parameters
    parser.add_argument('--ensemble-method', type=str, default='zscore_mean',
                        choices=['mean', 'weighted', 'rank_mean', 'zscore_mean', 'zscore_weighted'],
                        help='Ensemble method (default: zscore_mean)')
    parser.add_argument('--ae-weight', type=float, default=0.20)
    parser.add_argument('--ae-mn-weight', type=float, default=0.20)
    parser.add_argument('--ae-v9r-weight', type=float, default=0.20)
    parser.add_argument('--cb-weight', type=float, default=0.20)
    parser.add_argument('--da-weight', type=float, default=0.20)

    # Stacking parameters
    parser.add_argument('--stacking', action='store_true',
                        help='Learn optimal weights from validation set (Stacking)')
    parser.add_argument('--stacking-method', type=str, default='grid_search',
                        choices=['grid_search', 'grid_search_icir'],
                        help='Stacking weight learning method (default: grid_search)')
    parser.add_argument('--min-weight', type=float, default=0.05)
    parser.add_argument('--diversity-bonus', type=float, default=0.1)

    # Data parameters
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])

    # Backtest parameters
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=2)
    parser.add_argument('--account', type=float, default=100000)
    parser.add_argument('--rebalance-freq', type=int, default=5)
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss',
                                 'mvo', 'rp', 'gmv', 'inv'])

    # Skip CV evaluation
    parser.add_argument('--skip-cv', action='store_true')

    # Strategy parameters
    parser.add_argument('--risk-lookback', type=int, default=20)
    parser.add_argument('--drawdown-threshold', type=float, default=-0.10)
    parser.add_argument('--momentum-threshold', type=float, default=0.03)
    parser.add_argument('--risk-high', type=float, default=0.50)
    parser.add_argument('--risk-medium', type=float, default=0.75)
    parser.add_argument('--risk-normal', type=float, default=0.95)
    parser.add_argument('--market-proxy', type=str, default='SPY')
    parser.add_argument('--vol-high', type=float, default=0.35)
    parser.add_argument('--vol-medium', type=float, default=0.25)
    parser.add_argument('--stop-loss', type=float, default=-0.15)
    parser.add_argument('--no-sell-after-drop', type=float, default=-0.20)

    # Portfolio optimization parameters
    parser.add_argument('--opt-lamb', type=float, default=2.0)
    parser.add_argument('--opt-delta', type=float, default=0.2)
    parser.add_argument('--opt-alpha', type=float, default=0.01)
    parser.add_argument('--cov-lookback', type=int, default=60)
    parser.add_argument('--max-weight', type=float, default=0.0)

    # AI affinity filter
    parser.add_argument('--ai-filter', type=str, default='none',
                        choices=['none', 'penalty', 'exclude'])
    parser.add_argument('--ai-penalty-weight', type=float, default=0.5)
    parser.add_argument('--ai-bonus-weight', type=float, default=0.0)
    parser.add_argument('--ai-exclude-threshold', type=int, default=-1)
    parser.add_argument('--no-ai-time-scale', action='store_true')

    args = parser.parse_args()

    # Use FINAL_TEST time splits
    time_splits = {
        'train_start': FINAL_TEST['train_start'],
        'train_end': FINAL_TEST['train_end'],
        'valid_start': FINAL_TEST['valid_start'],
        'valid_end': FINAL_TEST['valid_end'],
        'test_start': FINAL_TEST['test_start'],
        'test_end': FINAL_TEST['test_end'],
    }

    # Model config (4 original + DoubleAdapt)
    MODEL_CONFIG = {
        'ae': {
            'path': Path(args.ae_model),
            'handler_arg': 'ae_handler',
            'handler': args.ae_handler,
            'display': 'AE-MLP-V7',
            'type': 'ae_mlp',
        },
        'ae_mn': {
            'path': Path(args.ae_mn_model),
            'handler_arg': 'ae_mn_handler',
            'handler': args.ae_mn_handler,
            'display': 'AE-MLP-MN',
            'type': 'ae_mlp',
        },
        'ae_v9r': {
            'path': Path(args.ae_v9r_model),
            'handler_arg': 'ae_v9r_handler',
            'handler': args.ae_v9r_handler,
            'display': 'AE-MLP-V9R',
            'type': 'ae_mlp',
        },
        'cb': {
            'path': Path(args.cb_model),
            'handler_arg': 'cb_handler',
            'handler': args.cb_handler,
            'display': 'CatBoost',
            'type': 'catboost',
        },
        'da': {
            'path': Path(args.da_model),
            'handler_arg': None,
            'handler': 'alpha300',
            'display': 'DA-FiLM',
            'type': 'doubleadapt',
        },
    }

    print("=" * 80)
    print("AE-MLP-V7 + AE-MLP-MN + AE-MLP-V9R + CatBoost + DA-FiLM Ensemble (V7)")
    print("=" * 80)
    for key, cfg in MODEL_CONFIG.items():
        print(f"{cfg['display']} Model:   {cfg['path']}")
        if cfg.get('handler'):
            print(f"{cfg['display']} Handler: {cfg['handler']}")
    print(f"Stock Pool: {args.stock_pool}")
    print(f"Prediction Horizon: {args.nday} days")
    print(f"Ensemble Method: {args.ensemble_method}")
    if args.ensemble_method in ['weighted', 'zscore_weighted']:
        parts = ", ".join(f"{cfg['display']}={getattr(args, f'{key}_weight')}"
                          for key, cfg in MODEL_CONFIG.items())
        print(f"Weights: {parts}")
    print(f"Test Period: {time_splits['test_start']} to {time_splits['test_end']}")
    print("=" * 80)

    # Check model files exist
    for key, cfg in MODEL_CONFIG.items():
        if not cfg['path'].exists():
            print(f"Error: {cfg['display']} model not found: {cfg['path']}")
            sys.exit(1)

    # [1] Load metadata
    print("\n[1] Loading model metadata...")
    for key, cfg in MODEL_CONFIG.items():
        if cfg['type'] == 'doubleadapt':
            meta_path = Path(args.da_meta)
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                print(f"    {cfg['display']} metadata: macro={meta.get('macro_set', 'N/A')}, "
                      f"test_ic={meta.get('test_ic', 'N/A'):.4f}")
        else:
            meta = load_model_meta(cfg['path'])
            if meta:
                print(f"    {cfg['display']} metadata: handler={meta.get('handler', 'N/A')}, "
                      f"nday={meta.get('nday', 'N/A')}")
                if 'handler' in meta and cfg['handler_arg']:
                    cfg['handler'] = meta['handler']
                    setattr(args, cfg['handler_arg'], meta['handler'])
            else:
                print(f"    {cfg['display']} metadata not found, using default: {cfg['handler']}")

    # [2] Initialize Qlib
    print("\n[2] Initializing Qlib...")
    qlib.init(
        provider_uri=str(QLIB_DATA_PATH),
        region=REG_US,
        custom_ops=TALIB_OPS,
        kernels=1,
        joblib_backend=None,
    )
    print("    Qlib initialized with TA-Lib support")

    # [3] Stock pool
    symbols = STOCK_POOLS[args.stock_pool]
    print(f"\n[3] Using stock pool: {args.stock_pool} ({len(symbols)} stocks)")

    # [4] Create datasets (for non-DA models)
    print("\n[4] Creating datasets...")
    datasets = {}
    handlers_dict = {}
    for key, cfg in MODEL_CONFIG.items():
        if cfg['type'] == 'doubleadapt':
            continue  # DA has its own data pipeline
        print(f"    Creating {cfg['handler']} dataset for {cfg['display']}...")
        h = create_ensemble_data_handler(cfg['handler'], symbols, time_splits, args.nday,
                                         include_valid=True)
        datasets[key] = create_ensemble_dataset(h, time_splits, include_valid=True)
        handlers_dict[key] = cfg['handler']

    # [5] Load models
    print("\n[5] Loading models...")
    models = {}
    for key, cfg in MODEL_CONFIG.items():
        if cfg['type'] == 'ae_mlp':
            models[key] = load_ae_mlp_model(cfg['path'])
        elif cfg['type'] == 'catboost':
            models[key] = load_catboost_model(cfg['path'])
        elif cfg['type'] == 'doubleadapt':
            models[key] = None  # DA model is loaded during inference

    # [6] Generate DoubleAdapt predictions
    print("\n[6] Loading/generating DoubleAdapt-FiLM predictions...")
    da_cache_path = MODEL_SAVE_PATH / 'doubleadapt_da_film_sp500_minimal_5d_preds.pkl'
    if args.da_preds:
        da_cache_path = Path(args.da_preds)

    da_pred_test = load_or_generate_da_predictions(
        checkpoint_path=str(args.da_model),
        meta_json_path=str(args.da_meta),
        symbols=symbols,
        cache_path=da_cache_path,
    )

    # [7] CV evaluation
    if args.skip_cv:
        print("\n[7] Skipping cross-validation evaluation (--skip-cv)")
    else:
        print("\n[7] Cross-validation evaluation...")
        run_cv_evaluation(models, handlers_dict, args, symbols, MODEL_CONFIG,
                          da_pred_test=da_pred_test)

    # [8] Generate predictions
    print("\n[8] Generating predictions...")
    pred_dict = {}
    for key, cfg in MODEL_CONFIG.items():
        if cfg['type'] == 'doubleadapt':
            pred_dict[key] = da_pred_test
            print(f"    {cfg['display']} predictions (from inference)...")
        elif cfg['type'] == 'ae_mlp':
            print(f"    {cfg['display']} predictions...")
            pred_dict[key] = predict_with_ae_mlp(models[key], datasets[key])
        elif cfg['type'] == 'catboost':
            print(f"    {cfg['display']} predictions...")
            pred_dict[key] = predict_with_catboost(models[key], datasets[key])
        print(f"      Shape: {len(pred_dict[key])}, Range: [{pred_dict[key].min():.4f}, {pred_dict[key].max():.4f}]")

    # Prediction statistics
    model_names = list(MODEL_CONFIG.keys())
    print("\n    Prediction Statistics Comparison:")
    header = f"    {'Metric':<15s}"
    for cfg in MODEL_CONFIG.values():
        header += f" | {cfg['display']:>12s}"
    print("    " + "=" * (15 + 17 * len(MODEL_CONFIG)))
    print(header)
    print("    " + "-" * (15 + 17 * len(MODEL_CONFIG)))

    for metric_name, metric_fn in [('Mean', lambda s: s.mean()), ('Std', lambda s: s.std()),
                                    ('Median', lambda s: s.median()), ('Abs Mean', lambda s: s.abs().mean()),
                                    ('Min', lambda s: s.min()), ('Max', lambda s: s.max())]:
        row = f"    {metric_name:<15s}"
        for key in MODEL_CONFIG:
            row += f" | {metric_fn(pred_dict[key]):>12.6f}"
        print(row)
    print("    " + "=" * (15 + 17 * len(MODEL_CONFIG)))

    # [9] Correlation
    print("\n[9] Calculating pairwise correlations...")
    overall_corr = calculate_pairwise_correlations(pred_dict)

    print("\n    Overall Correlation Matrix:")
    print(overall_corr.to_string())

    # Compute daily correlations
    names = list(pred_dict.keys())
    common_idx = pred_dict[names[0]].index
    for name in names[1:]:
        common_idx = common_idx.intersection(pred_dict[name].index)
    corr_df = pd.DataFrame({name: pred_dict[name].loc[common_idx] for name in names})

    # Helper: map model key to display name (replace longest keys first to avoid partial matches)
    display_names = {k: cfg['display'] for k, cfg in MODEL_CONFIG.items()}
    sorted_keys = sorted(display_names.keys(), key=len, reverse=True)

    def format_pair(pair_key):
        result = pair_key
        for k in sorted_keys:
            result = result.replace(k, display_names[k])
        return result

    daily_corrs = {}
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j <= i:
                continue
            pair_key = f"{n1} vs {n2}"
            dc = corr_df.groupby(level='datetime').apply(
                lambda x, _n1=n1, _n2=n2: x[_n1].corr(x[_n2]) if len(x) > 1 else np.nan
            ).dropna()
            daily_corrs[pair_key] = (dc.mean(), dc.std())

    print("\n    Daily Correlation (mean +/- std):")
    print("    " + "-" * 55)
    for pair, (mean_c, std_c) in daily_corrs.items():
        print(f"    {format_pair(pair):<35s}: {mean_c:.4f} +/- {std_c:.4f}")
    print("    " + "-" * 55)

    # Highlight DA correlations
    print("\n    DA-FiLM correlations with other models:")
    for pair, (mean_c, std_c) in daily_corrs.items():
        if 'da' in pair:
            diversity = "HIGH diversity" if abs(mean_c) < 0.3 else \
                        "MODERATE diversity" if abs(mean_c) < 0.5 else "LOW diversity"
            print(f"    {format_pair(pair):<35s}: {mean_c:.4f} ({diversity})")

    # [10] Stacking (optional)
    learned_weights = None
    if args.stacking:
        print(f"\n[10] Stacking: Learning optimal weights from validation set...")
        print(f"    Method: {args.stacking_method}")
        print(f"    Note: DA-FiLM excluded from stacking (no validation predictions)")

        val_preds = {}
        for key, cfg in MODEL_CONFIG.items():
            if cfg['type'] == 'doubleadapt':
                continue
            if cfg['type'] == 'ae_mlp':
                val_preds[key] = predict_with_ae_mlp(models[key], datasets[key], segment="valid")
            elif cfg['type'] == 'catboost':
                val_preds[key] = predict_with_catboost(models[key], datasets[key], segment="valid")
            print(f"      {cfg['display']} valid: {len(val_preds[key])} samples")

        val_label = datasets['ae'].prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(val_label, pd.DataFrame):
            val_label = val_label.iloc[:, 0]

        learned_weights_4, learn_info = learn_optimal_weights(
            val_preds, val_label,
            method=args.stacking_method,
            min_weight=args.min_weight,
            diversity_bonus=args.diversity_bonus,
        )

        # Scale 4-model weights to sum to 0.8, give DA 0.2
        total_4 = sum(learned_weights_4.values())
        learned_weights = {}
        for k, w in learned_weights_4.items():
            learned_weights[k] = w / total_4 * 0.8
        learned_weights['da'] = 0.20

        print(f"\n    Learned Weights (4-model scaled + DA fixed 0.20):")
        print(f"    " + "-" * 50)
        for key, cfg in MODEL_CONFIG.items():
            print(f"    {cfg['display']:15s}: {learned_weights[key]:>10.4f}")
        print(f"    " + "-" * 50)

        for k, v in learn_info.items():
            if k != 'method':
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

        args.ensemble_method = 'zscore_weighted'
        for key in MODEL_CONFIG:
            setattr(args, f'{key}_weight', learned_weights[key])
        print(f"\n    Using zscore_weighted ensemble with learned weights")

    # Ensemble
    step_num = 11 if args.stacking else 10
    print(f"\n[{step_num}] Ensembling predictions ({args.ensemble_method})...")

    weights = None
    if args.ensemble_method in ['weighted', 'zscore_weighted']:
        if learned_weights:
            weights = learned_weights
        else:
            weights = {key: getattr(args, f'{key}_weight') for key in MODEL_CONFIG}

    pred_ensemble = ensemble_predictions(pred_dict, args.ensemble_method, weights)
    print(f"    Ensemble shape: {len(pred_ensemble)}, Range: [{pred_ensemble.min():.4f}, {pred_ensemble.max():.4f}]")

    # AI affinity filter
    pred_raw = None
    if args.ai_filter != 'none':
        step_num += 1
        print(f"\n[{step_num}] Applying AI affinity filter ({args.ai_filter})...")
        pred_raw = pred_ensemble.copy()
        pred_ensemble = apply_ai_affinity_filter(
            pred_ensemble,
            mode=args.ai_filter,
            penalty_weight=args.ai_penalty_weight,
            bonus_weight=args.ai_bonus_weight,
            exclude_threshold=args.ai_exclude_threshold,
            time_scale=not args.no_ai_time_scale,
        )
        print(f"    Filtered shape: {len(pred_ensemble)}, Range: [{pred_ensemble.min():.4f}, {pred_ensemble.max():.4f}]")

    # IC metrics
    step_num += 1
    print(f"\n[{step_num}] Calculating IC metrics...")
    test_label = datasets['ae'].prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(test_label, pd.DataFrame):
        label = test_label.iloc[:, 0]
    else:
        label = test_label

    ic_results = {}
    for key, cfg in MODEL_CONFIG.items():
        ic, std, icir, ic_series = compute_ic(pred_dict[key], label)
        ic_results[key] = {'ic': ic, 'std': std, 'icir': icir}

    ens_ic, ens_std, ens_icir, ens_ic_series = compute_ic(pred_ensemble, label)

    # Also compute V5-style 4-model ensemble for comparison
    pred_dict_v5 = {k: v for k, v in pred_dict.items() if k != 'da'}
    weights_v5 = None
    if weights:
        weights_v5 = {k: v for k, v in weights.items() if k != 'da'}
    pred_ens_v5 = ensemble_predictions(pred_dict_v5, args.ensemble_method, weights_v5)
    v5_ic, v5_std, v5_icir, _ = compute_ic(pred_ens_v5, label)

    print("\n    +" + "=" * 70 + "+")
    print("    |" + " " * 10 + "Information Coefficient (IC) Comparison (V7)" + " " * 14 + "|")
    print("    +" + "=" * 70 + "+")
    print(f"    |  {'Model':<20s} | {'Mean IC':>10s} | {'IC Std':>10s} | {'ICIR':>10s} |")
    print("    +" + "-" * 70 + "+")
    for key, cfg in MODEL_CONFIG.items():
        r = ic_results[key]
        print(f"    |  {cfg['display']:<20s} | {r['ic']:>10.4f} | {r['std']:>10.4f} | {r['icir']:>10.4f} |")
    print("    +" + "-" * 70 + "+")
    print(f"    |  {'Ensemble V5 (4-mdl)':<20s} | {v5_ic:>10.4f} | {v5_std:>10.4f} | {v5_icir:>10.4f} |")
    print(f"    |  {'Ensemble V7 (5-mdl)':<20s} | {ens_ic:>10.4f} | {ens_std:>10.4f} | {ens_icir:>10.4f} |")
    print("    +" + "=" * 70 + "+")

    # Improvement
    best_single_ic = max(r['ic'] for r in ic_results.values())
    best_single_icir = max(r['icir'] for r in ic_results.values())

    ic_improvement = (ens_ic - best_single_ic) / abs(best_single_ic) * 100 if best_single_ic != 0 else 0
    icir_improvement = (ens_icir - best_single_icir) / abs(best_single_icir) * 100 if best_single_icir != 0 else 0
    v7_vs_v5_ic = (ens_ic - v5_ic) / abs(v5_ic) * 100 if v5_ic != 0 else 0
    v7_vs_v5_icir = (ens_icir - v5_icir) / abs(v5_icir) * 100 if v5_icir != 0 else 0

    print(f"\n    Ensemble Performance vs Best Single Model:")
    print(f"    IC improvement:   {ic_improvement:>+.2f}%")
    print(f"    ICIR improvement: {icir_improvement:>+.2f}%")
    print(f"\n    V7 vs V5 Ensemble:")
    print(f"    IC change:   {v7_vs_v5_ic:>+.2f}% ({v5_ic:.4f} -> {ens_ic:.4f})")
    print(f"    ICIR change: {v7_vs_v5_icir:>+.2f}% ({v5_icir:.4f} -> {ens_icir:.4f})")

    # Monthly IC comparison
    print(f"\n    Monthly IC (V5 vs V7):")
    print(f"    {'Month':<12s} {'V5 Ens':>10s} {'V7 Ens':>10s} {'DA-FiLM':>10s} {'V7-V5':>10s}")
    print(f"    {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    _, _, _, v5_monthly = compute_ic(pred_ens_v5, label)
    _, _, _, da_monthly = compute_ic(pred_dict['da'], label)

    if len(ens_ic_series) > 0:
        ens_monthly = ens_ic_series.groupby(ens_ic_series.index.to_period('M')).mean()
        v5_monthly_grp = v5_monthly.groupby(v5_monthly.index.to_period('M')).mean()
        da_monthly_grp = da_monthly.groupby(da_monthly.index.to_period('M')).mean()

        for month in ens_monthly.index:
            v5_val = v5_monthly_grp.get(month, np.nan)
            v7_val = ens_monthly.get(month, np.nan)
            da_val = da_monthly_grp.get(month, np.nan)
            delta = v7_val - v5_val if not (np.isnan(v7_val) or np.isnan(v5_val)) else np.nan
            print(f"    {str(month):<12s} {v5_val:>10.4f} {v7_val:>10.4f} {da_val:>10.4f} {delta:>+10.4f}")

    # Dual IC comparison when AI filter is active
    if pred_raw is not None:
        raw_ic, raw_std, raw_icir, _ = compute_ic(pred_raw, label)
        print(f"\n    AI Filter Impact on IC:")
        print(f"      Before filter: IC={raw_ic:.4f}, ICIR={raw_icir:.4f}")
        print(f"      After filter:  IC={ens_ic:.4f}, ICIR={ens_icir:.4f}")
        ic_delta = ens_ic - raw_ic
        icir_delta = ens_icir - raw_icir
        print(f"      Delta:         IC={ic_delta:+.4f}, ICIR={icir_delta:+.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("ENSEMBLE V7 ANALYSIS COMPLETE")
    print("=" * 80)
    print("Pairwise Correlations (daily mean):")
    for pair, (mean_c, _) in daily_corrs.items():
        print(f"  {format_pair(pair)}: {mean_c:.4f}")
    if learned_weights:
        print("Stacking Weights:")
        for key, cfg in MODEL_CONFIG.items():
            print(f"  {cfg['display']}: {learned_weights[key]:.3f}")
    for key, cfg in MODEL_CONFIG.items():
        r = ic_results[key]
        print(f"{cfg['display']:15s} IC: {r['ic']:.4f} (ICIR: {r['icir']:.4f})")
    print(f"{'Ensemble V5':15s} IC: {v5_ic:.4f} (ICIR: {v5_icir:.4f})")
    print(f"{'Ensemble V7':15s} IC: {ens_ic:.4f} (ICIR: {ens_icir:.4f})")
    print("=" * 80)

    # Backtest
    if args.backtest:
        run_ensemble_backtest(pred_ensemble, args, time_splits, model_name="Ensemble_V7")

        print("\n" + "=" * 80)
        print("ENSEMBLE V7 BACKTEST COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    main()
