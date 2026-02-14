"""
AE-MLP + AE-MLP (mkt-neutral) + CatBoost Ensemble (V4)

Three-model ensemble:
  1. AE-MLP (alpha158-enhanced-v7)
  2. AE-MLP (v9-mkt-neutral)
  3. CatBoost (catboost-v1)

Based on run_ae_catboost_ensemble.py with a third model added.

Usage:
    python scripts/models/ensemble/run_ae_cb_ensemble_v4.py
    python scripts/models/ensemble/run_ae_cb_ensemble_v4.py --ensemble-method rank_mean
    python scripts/models/ensemble/run_ae_cb_ensemble_v4.py --stacking
    python scripts/models/ensemble/run_ae_cb_ensemble_v4.py --backtest --topk 10 --n-drop 2
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
import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from utils.talib_ops import TALIB_OPS
from utils.strategy import get_strategy_config
from utils.backtest_utils import plot_backtest_curve, generate_trade_records
from utils.ai_filter import apply_ai_affinity_filter
from data.stock_pools import STOCK_POOLS

from models.common import (
    PROJECT_ROOT,
    QLIB_DATA_PATH,
    MODEL_SAVE_PATH,
    CV_FOLDS,
    FINAL_TEST,
)
from models.deep.ae_mlp_model import AEMLP


# ── Model loading ──────────────────────────────────────────────────────

def load_ae_mlp_model(model_path: Path):
    """Load AE-MLP (.keras) model"""
    print(f"    Loading AE-MLP model from: {model_path}")
    model = AEMLP.load(str(model_path))
    return model


def load_catboost_model(model_path: Path):
    """Load CatBoost (.cbm) model"""
    print(f"    Loading CatBoost model from: {model_path}")
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return model


def load_model_meta(model_path: Path) -> dict:
    """Load model metadata"""
    meta_path = model_path.with_suffix('.meta.pkl')
    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            return pickle.load(f)

    stem = model_path.stem
    if stem.endswith('_best'):
        alt_meta_path = model_path.parent / (stem[:-5] + '.meta.pkl')
        if alt_meta_path.exists():
            with open(alt_meta_path, 'rb') as f:
                return pickle.load(f)

    return {}


# ── Data handling ──────────────────────────────────────────────────────

def create_data_handler(handler_name: str, symbols: list, time_splits: dict, nday: int):
    """Create DataHandler for a specific handler type"""
    from models.common.handlers import get_handler_class

    HandlerClass = get_handler_class(handler_name)

    handler = HandlerClass(
        volatility_window=nday,
        instruments=symbols,
        start_time=time_splits['train_start'],
        end_time=time_splits['test_end'],
        fit_start_time=time_splits['train_start'],
        fit_end_time=time_splits['train_end'],
        infer_processors=[],
    )

    return handler


def create_dataset(handler, time_splits: dict) -> DatasetH:
    """Create Qlib DatasetH"""
    return DatasetH(
        handler=handler,
        segments={
            "train": (time_splits['train_start'], time_splits['train_end']),
            "valid": (time_splits['valid_start'], time_splits['valid_end']),
            "test": (time_splits['test_start'], time_splits['test_end']),
        }
    )


# ── Prediction ─────────────────────────────────────────────────────────

def predict_with_ae_mlp(model: AEMLP, dataset: DatasetH, segment: str = "test") -> pd.Series:
    """Generate predictions with AE-MLP model"""
    pred = model.predict(dataset, segment=segment)
    pred.name = 'score'
    return pred


def predict_with_catboost(model: CatBoostRegressor, dataset: DatasetH, segment: str = "test") -> pd.Series:
    """Generate predictions with CatBoost model"""
    data = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
    data = data.fillna(0).replace([np.inf, -np.inf], 0)
    pred_values = model.predict(data.values)
    pred = pd.Series(pred_values, index=data.index, name='score')
    return pred


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


# ── Correlation ────────────────────────────────────────────────────────

def calculate_correlation(pred_dict: dict) -> pd.DataFrame:
    """Calculate pairwise correlation between all model predictions.

    Parameters
    ----------
    pred_dict : dict
        {model_name: pd.Series}

    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    names = list(pred_dict.keys())
    # Find common index across all models
    common_idx = pred_dict[names[0]].index
    for name in names[1:]:
        common_idx = common_idx.intersection(pred_dict[name].index)

    df = pd.DataFrame({name: pred_dict[name].loc[common_idx] for name in names})

    # Overall correlation
    overall_corr = df.corr()

    # Daily correlation mean
    daily_corrs = {}
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j <= i:
                continue
            pair_key = f"{n1} vs {n2}"
            dc = df.groupby(level='datetime').apply(
                lambda x: x[n1].corr(x[n2]) if len(x) > 1 else np.nan
            ).dropna()
            daily_corrs[pair_key] = (dc.mean(), dc.std())

    return overall_corr, daily_corrs


# ── Ensemble ───────────────────────────────────────────────────────────

def zscore_by_day(x):
    mean = x.groupby(level='datetime').transform('mean')
    std = x.groupby(level='datetime').transform('std')
    return (x - mean) / (std + 1e-8)


def ensemble_predictions_multi(pred_dict: dict, method: str = 'zscore_mean',
                                weights: dict = None) -> pd.Series:
    """
    Ensemble multiple model predictions.

    Parameters
    ----------
    pred_dict : dict
        {model_name: pd.Series}
    method : str
        'mean', 'weighted', 'rank_mean', 'zscore_mean', 'zscore_weighted'
    weights : dict, optional
        {model_name: weight} for weighted methods

    Returns
    -------
    pd.Series
        Ensembled predictions
    """
    names = list(pred_dict.keys())

    # Find common index
    common_idx = pred_dict[names[0]].index
    for name in names[1:]:
        common_idx = common_idx.intersection(pred_dict[name].index)

    preds = {name: pred_dict[name].loc[common_idx] for name in names}

    if method == 'mean':
        ensemble_pred = sum(preds.values()) / len(preds)
    elif method == 'weighted':
        if weights is None:
            weights = {n: 1.0 / len(names) for n in names}
        total = sum(weights.values())
        ensemble_pred = sum(preds[n] * weights[n] for n in names) / total
    elif method == 'rank_mean':
        ranks = {n: preds[n].groupby(level='datetime').rank(pct=True) for n in names}
        ensemble_pred = sum(ranks.values()) / len(ranks)
    elif method == 'zscore_mean':
        zscores = {n: zscore_by_day(preds[n]) for n in names}
        ensemble_pred = sum(zscores.values()) / len(zscores)
    elif method == 'zscore_weighted':
        if weights is None:
            weights = {n: 1.0 / len(names) for n in names}
        total = sum(weights.values())
        zscores = {n: zscore_by_day(preds[n]) for n in names}
        ensemble_pred = sum(zscores[n] * weights[n] for n in names) / total
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    ensemble_pred.name = 'score'
    return ensemble_pred


# ── IC ─────────────────────────────────────────────────────────────────

def compute_ic(pred: pd.Series, label: pd.Series) -> tuple:
    """Calculate IC (Information Coefficient)"""
    common_idx = pred.index.intersection(label.index)
    pred_aligned = pred.loc[common_idx]
    label_aligned = label.loc[common_idx]

    valid_idx = ~(pred_aligned.isna() | label_aligned.isna())
    pred_clean = pred_aligned[valid_idx]
    label_clean = label_aligned[valid_idx]

    df = pd.DataFrame({'pred': pred_clean, 'label': label_clean})
    ic_by_date = df.groupby(level='datetime').apply(
        lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
    )
    ic_by_date = ic_by_date.dropna()

    if len(ic_by_date) == 0:
        return 0.0, 0.0, 0.0, pd.Series()

    mean_ic = ic_by_date.mean()
    ic_std = ic_by_date.std()
    icir = mean_ic / ic_std if ic_std > 0 else 0

    return mean_ic, ic_std, icir, ic_by_date


# ── Stacking (weight learning) ────────────────────────────────────────

def learn_optimal_weights_multi(pred_dict: dict, label: pd.Series,
                                 method: str = 'grid_search',
                                 min_weight: float = 0.05,
                                 diversity_bonus: float = 0.1) -> tuple:
    """
    Learn optimal ensemble weights for N models using validation data.

    Parameters
    ----------
    pred_dict : dict
        {model_name: pd.Series} predictions on validation set
    label : pd.Series
        True labels on validation set
    method : str
        'grid_search' or 'grid_search_icir'
    min_weight : float
        Minimum weight for each model
    diversity_bonus : float
        Bonus for balanced weights

    Returns
    -------
    tuple
        (weights_dict, info_dict)
    """
    names = list(pred_dict.keys())
    n_models = len(names)

    # Align all series
    common_idx = label.index
    for name in names:
        common_idx = common_idx.intersection(pred_dict[name].index)

    preds = {n: pred_dict[n].loc[common_idx] for n in names}
    y = label.loc[common_idx]

    # Remove NaN
    valid_mask = ~y.isna()
    for n in names:
        valid_mask &= ~preds[n].isna()

    preds = {n: preds[n][valid_mask] for n in names}
    y = y[valid_mask]

    # Z-score normalize
    z_preds = {n: zscore_by_day(preds[n]) for n in names}

    if method in ('grid_search', 'grid_search_icir'):
        # Grid search over weight space for 3 models
        # w1 + w2 + w3 = 1, each >= min_weight
        step = 0.05
        best_score = -np.inf
        best_weights = {n: 1.0 / n_models for n in names}
        best_ic = 0

        # Generate weight grid for 3 models
        weight_grid = []
        for w1 in np.arange(min_weight, 1.0 - (n_models - 1) * min_weight + 0.001, step):
            for w2 in np.arange(min_weight, 1.0 - w1 - (n_models - 2) * min_weight + 0.001, step):
                w3 = 1.0 - w1 - w2
                if w3 >= min_weight - 0.001:
                    weight_grid.append((w1, w2, w3))

        for ws in weight_grid:
            w_dict = {names[i]: ws[i] for i in range(n_models)}
            ensemble = sum(z_preds[n] * w_dict[n] for n in names)

            df = pd.DataFrame({'pred': ensemble, 'label': y})
            ic_by_date = df.groupby(level='datetime').apply(
                lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
            )
            ic_series = ic_by_date.dropna()
            mean_ic = ic_series.mean()

            if method == 'grid_search':
                # Diversity: how balanced are the weights (max when all equal)
                diversity = 1 - np.std(list(ws)) * np.sqrt(n_models)
                score = mean_ic + diversity_bonus * diversity
            else:  # grid_search_icir
                ic_std = ic_series.std()
                score = mean_ic / ic_std if ic_std > 0 else 0

            if score > best_score:
                best_score = score
                best_weights = w_dict
                best_ic = mean_ic

        info = {
            'method': method,
            'best_ic': best_ic,
            'grid_size': len(weight_grid),
        }
        if method == 'grid_search':
            info['diversity_bonus'] = diversity_bonus

        return best_weights, info

    else:
        raise ValueError(f"Unknown method: {method}. Use 'grid_search' or 'grid_search_icir'.")


# ── CV evaluation ─────────────────────────────────────────────────────

def run_cv_evaluation(models: dict, handlers: dict, args, symbols):
    """
    Evaluate ensemble IC on CV folds.

    Parameters
    ----------
    models : dict
        {'ae': AEMLP, 'ae_mn': AEMLP, 'cb': CatBoostRegressor}
    handlers : dict
        {'ae': handler_name, 'ae_mn': handler_name, 'cb': handler_name}
    """
    model_names = list(models.keys())
    display_names = {
        'ae': 'AE-MLP',
        'ae_mn': 'AE-MLP-MN',
        'cb': 'CatBoost',
    }

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION EVALUATION (3-Model Ensemble V4)")
    print("=" * 80)
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: valid {fold['valid_start']}~{fold['valid_end']}")
    print(f"Test Set: {FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']} (2025)")
    print(f"Ensemble Method: {args.ensemble_method}")
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

    # Create test datasets for each handler
    test_datasets = {}
    for key in model_names:
        h = create_data_handler(handlers[key], symbols, test_time, args.nday)
        test_datasets[key] = create_dataset(h, test_time)

    # Test predictions
    test_preds = {}
    for key in model_names:
        if key == 'cb':
            test_preds[key] = predict_catboost_raw(models[key], test_datasets[key], "test")
        else:
            test_preds[key] = predict_with_ae_mlp(models[key], test_datasets[key], segment="test")

    weights = None
    if args.ensemble_method in ['weighted', 'zscore_weighted']:
        weights = {n: getattr(args, f'{n}_weight') for n in model_names}

    test_pred_ens = ensemble_predictions_multi(test_preds, args.ensemble_method, weights)

    # Labels from first AE-MLP dataset
    test_label_df = test_datasets['ae'].prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)
    test_label = test_label_df.iloc[:, 0] if isinstance(test_label_df, pd.DataFrame) else test_label_df

    test_ics = {}
    for key in model_names:
        ic, _, _, _ = compute_ic(test_preds[key], test_label)
        test_ics[key] = ic
    test_ens_ic, _, _, _ = compute_ic(test_pred_ens, test_label)

    parts = ", ".join(f"{display_names[k]} IC={test_ics[k]:.4f}" for k in model_names)
    print(f"    Test (2025): {parts}, Ensemble IC={test_ens_ic:.4f}")

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

        # Create fold datasets
        fold_datasets = {}
        for key in model_names:
            h = create_data_handler(handlers[key], symbols, fold_time, args.nday)
            fold_datasets[key] = create_dataset(h, fold_time)

        # Validation predictions
        val_preds = {}
        for key in model_names:
            if key == 'cb':
                val_preds[key] = predict_catboost_raw(models[key], fold_datasets[key], "valid")
            else:
                val_preds[key] = predict_with_ae_mlp(models[key], fold_datasets[key], segment="valid")

        val_pred_ens = ensemble_predictions_multi(val_preds, args.ensemble_method, weights)

        # Labels
        val_label_df = fold_datasets['ae'].prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
        val_label = val_label_df.iloc[:, 0] if isinstance(val_label_df, pd.DataFrame) else val_label_df

        # IC
        fold_ic = {}
        for key in model_names:
            ic, _, _, _ = compute_ic(val_preds[key], val_label)
            fold_ic[key] = ic
        ens_ic, _, ens_icir, _ = compute_ic(val_pred_ens, val_label)

        result = {'name': fold['name'], 'ens_ic': ens_ic, 'ens_icir': ens_icir}
        for key in model_names:
            result[f'{key}_ic'] = fold_ic[key]
        fold_results.append(result)

        parts = ", ".join(f"{display_names[k]}={fold_ic[k]:.4f}" for k in model_names)
        print(f"    {fold['name']}: {parts}, Ensemble={ens_ic:.4f}")

    # Summary table
    header_parts = "".join(f" {display_names[k]:>12s}" for k in model_names)
    print("\n" + "=" * 80)
    print("CV EVALUATION COMPLETE (3-Model Ensemble V4)")
    print("=" * 80)
    print(f"{'':25s}{header_parts} {'Ensemble':>12s}")
    print(f"{'-'*25}" + " ".join([f"{'-'*12}"] * (len(model_names) + 1)))
    for r in fold_results:
        vals = "".join(f" {r[f'{k}_ic']:>12.4f}" for k in model_names)
        print(f"{r['name']:<25s}{vals} {r['ens_ic']:>12.4f}")
    print(f"{'-'*25}" + " ".join([f"{'-'*12}"] * (len(model_names) + 1)))

    mean_vals = "".join(f" {np.mean([r[f'{k}_ic'] for r in fold_results]):>12.4f}" for k in model_names)
    std_vals = "".join(f" {np.std([r[f'{k}_ic'] for r in fold_results]):>12.4f}" for k in model_names)
    test_vals = "".join(f" {test_ics[k]:>12.4f}" for k in model_names)

    ens_ics = [r['ens_ic'] for r in fold_results]
    print(f"{'Valid Mean IC':<25s}{mean_vals} {np.mean(ens_ics):>12.4f}")
    print(f"{'Valid IC Std':<25s}{std_vals} {np.std(ens_ics):>12.4f}")
    print(f"{'Test IC (2025)':<25s}{test_vals} {test_ens_ic:>12.4f}")
    print("=" * 80)

    return fold_results


# ── Backtest ──────────────────────────────────────────────────────────

def run_ensemble_backtest(pred: pd.Series, args, time_splits: dict):
    """Run backtest with ensembled predictions"""
    from qlib.backtest import backtest as qlib_backtest
    from qlib.contrib.evaluate import risk_analysis

    strategy_label = args.strategy.upper() if args.strategy in ('mvo', 'rp', 'gmv', 'inv') else args.strategy
    print("\n" + "=" * 70)
    print(f"BACKTEST with {strategy_label} Strategy (3-Model Ensemble V4)")
    print("=" * 70)

    pred_df = pred.to_frame("score")

    print(f"\n[BT-1] Configuring backtest...")
    print(f"    Strategy: {args.strategy}")
    print(f"    Topk: {args.topk}")
    if args.strategy not in ('mvo', 'rp', 'gmv', 'inv'):
        print(f"    N_drop: {args.n_drop}")
    print(f"    Account: ${args.account:,.0f}")
    print(f"    Rebalance Freq: every {args.rebalance_freq} day(s)")
    print(f"    Period: {time_splits['test_start']} to {time_splits['test_end']}")

    # Portfolio optimization params
    optimizer_params = None
    if args.strategy in ("mvo", "rp", "gmv", "inv"):
        optimizer_params = {
            "lamb": args.opt_lamb,
            "delta": args.opt_delta,
            "alpha": args.opt_alpha,
            "cov_lookback": args.cov_lookback,
            "max_weight": args.max_weight,
        }
        print(f"    Portfolio Optimization:")
        if args.strategy == "mvo":
            print(f"      Risk aversion (lamb): {args.opt_lamb}")
        print(f"      Turnover limit (delta): {args.opt_delta:.0%}")
        print(f"      L2 regularization (alpha): {args.opt_alpha}")
        print(f"      Covariance lookback: {args.cov_lookback} days")
        if args.max_weight > 0:
            print(f"      Max weight per stock: {args.max_weight:.0%}")
        else:
            print(f"      Max weight per stock: unlimited")

    dynamic_risk_params = None
    if args.strategy == "vol_stoploss":
        dynamic_risk_params = {
            "lookback": args.risk_lookback,
            "vol_threshold_high": args.vol_high,
            "vol_threshold_medium": args.vol_medium,
            "stop_loss_threshold": args.stop_loss,
            "no_sell_after_drop": args.no_sell_after_drop,
            "risk_degree_high": args.risk_high,
            "risk_degree_medium": args.risk_medium,
            "risk_degree_normal": args.risk_normal,
            "market_proxy": args.market_proxy,
        }
    elif args.strategy == "dynamic_risk":
        dynamic_risk_params = {
            "lookback": args.risk_lookback,
            "drawdown_threshold": args.drawdown_threshold,
            "momentum_threshold": args.momentum_threshold,
            "risk_degree_high": args.risk_high,
            "risk_degree_medium": args.risk_medium,
            "risk_degree_normal": args.risk_normal,
            "market_proxy": args.market_proxy,
        }

    strategy_config = get_strategy_config(
        pred_df, args.topk, args.n_drop, args.rebalance_freq,
        strategy_type=args.strategy,
        dynamic_risk_params=dynamic_risk_params,
        optimizer_params=optimizer_params,
    )

    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    }

    pool_symbols = STOCK_POOLS[args.stock_pool]

    backtest_config = {
        "start_time": time_splits['test_start'],
        "end_time": time_splits['test_end'],
        "account": args.account,
        "benchmark": "SPY",
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": None,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0005,
            "min_cost": 1,
            "trade_unit": None,
            "codes": pool_symbols,
        },
    }

    print(f"\n[BT-2] Running backtest...")
    try:
        portfolio_metric_dict, indicator_dict = qlib_backtest(
            executor=executor_config,
            strategy=strategy_config,
            **backtest_config
        )

        print("    Backtest completed")

        print(f"\n[BT-3] Analyzing results...")

        for freq, (report_df, positions) in portfolio_metric_dict.items():
            _analyze_backtest_results(report_df, positions, freq, args, time_splits)

        for freq, (indicator_df, indicator_obj) in indicator_dict.items():
            if indicator_df is not None and not indicator_df.empty:
                print(f"\n    Trading Indicators ({freq}):")
                print(f"    " + "-" * 50)
                print(indicator_df.head(20).to_string(index=True))
                if len(indicator_df) > 20:
                    print(f"    ... ({len(indicator_df)} rows total)")

        return portfolio_metric_dict

    except Exception as e:
        print(f"\n    Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _analyze_backtest_results(report_df: pd.DataFrame, positions, freq: str,
                              args, time_splits: dict):
    """Analyze and report backtest results"""
    from qlib.contrib.evaluate import risk_analysis

    print(f"\n    Frequency: {freq}")
    print(f"    Report shape: {report_df.shape}")
    print(f"    Date range: {report_df.index.min()} to {report_df.index.max()}")

    total_return = (report_df["return"] + 1).prod() - 1

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

    print(f"\n    Daily Returns Statistics:")
    print(f"    " + "-" * 50)
    print(f"    Mean Daily Return:   {report_df['return'].mean():>10.4%}")
    print(f"    Std Daily Return:    {report_df['return'].std():>10.4%}")
    print(f"    Max Daily Return:    {report_df['return'].max():>10.4%}")
    print(f"    Min Daily Return:    {report_df['return'].min():>10.4%}")
    print(f"    Total Trading Days:  {len(report_df):>10d}")
    print(f"    " + "-" * 50)

    # Position concentration analysis
    print(f"\n    Position Concentration Analysis:")
    print(f"    " + "-" * 50)
    if positions is not None:
        try:
            # Sample positions at regular intervals
            pos_dates = sorted(positions.keys()) if isinstance(positions, dict) else []
            if not pos_dates:
                # positions might be a list or other structure
                pos_dates = []

            if pos_dates:
                daily_top1 = []
                daily_top3 = []
                daily_top5 = []
                daily_n_stocks = []
                stock_freq = {}

                for date in pos_dates:
                    pos = positions[date]
                    if hasattr(pos, 'get_stock_weight_dict'):
                        weights = pos.get_stock_weight_dict(only_stock=True)
                    elif isinstance(pos, dict):
                        weights = {k: v for k, v in pos.items() if k != 'cash'}
                    else:
                        continue

                    if not weights:
                        continue

                    sorted_w = sorted(weights.values(), reverse=True)
                    total_w = sum(sorted_w)
                    if total_w <= 0:
                        continue

                    sorted_w = [w / total_w for w in sorted_w]
                    daily_top1.append(sorted_w[0])
                    daily_top3.append(sum(sorted_w[:3]))
                    daily_top5.append(sum(sorted_w[:5]))
                    daily_n_stocks.append(len([w for w in sorted_w if w > 0.01]))

                    for stock in weights:
                        stock_freq[stock] = stock_freq.get(stock, 0) + 1

                if daily_top1:
                    import numpy as _np
                    print(f"    Avg stocks held (>1% weight): {_np.mean(daily_n_stocks):.1f}")
                    print(f"    Top-1 stock weight:  {_np.mean(daily_top1):>8.1%} (max {_np.max(daily_top1):.1%})")
                    print(f"    Top-3 stocks weight: {_np.mean(daily_top3):>8.1%} (max {_np.max(daily_top3):.1%})")
                    print(f"    Top-5 stocks weight: {_np.mean(daily_top5):>8.1%} (max {_np.max(daily_top5):.1%})")
                    print(f"    Unique stocks traded: {len(stock_freq)}")

                    # HHI (Herfindahl index) - measure of concentration
                    # HHI = 1/N means equal weight, HHI = 1 means single stock
                    # Not computed here for simplicity

                    # Most frequently held stocks
                    top_held = sorted(stock_freq.items(), key=lambda x: -x[1])[:5]
                    print(f"    Most frequently held:")
                    for stock, days in top_held:
                        print(f"      {stock}: {days}/{len(pos_dates)} days ({days/len(pos_dates):.0%})")
                else:
                    print(f"    Unable to extract position weights")
            else:
                print(f"    Position data format not supported for analysis")
        except Exception as e:
            print(f"    Position analysis failed: {e}")
    print(f"    " + "-" * 50)

    output_path = PROJECT_ROOT / "outputs" / f"ensemble_v4_backtest_report_{freq}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path)
    print(f"\n    Report saved to: {output_path}")

    print(f"\n[BT-4] Generating performance chart...")
    if not hasattr(args, 'handler'):
        args.handler = "ensemble_v4"
    plot_backtest_curve(report_df, args, freq, PROJECT_ROOT, model_name="Ensemble_V4")

    print(f"\n[BT-5] Generating trade records...")
    generate_trade_records(positions, args, freq, PROJECT_ROOT, model_name="ensemble_v4")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='AE-MLP + AE-MLP(mkt-neutral) + CatBoost Ensemble (V4)',
    )

    # Model paths
    parser.add_argument('--ae-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'ae_mlp_cv_alpha158-enhanced-v7_sp500_5d_best.keras'),
                        help='AE-MLP model path (.keras)')
    parser.add_argument('--ae-mn-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'ae_mlp_cv_v9-mkt-neutral_sp500_5d.keras'),
                        help='AE-MLP market-neutral model path (.keras)')
    parser.add_argument('--cb-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'catboost_cv_catboost-v1_sp500_5d_20260129_141353_best.cbm'),
                        help='CatBoost model path (.cbm)')

    # Handler configuration (override metadata)
    parser.add_argument('--ae-handler', type=str, default='alpha158-enhanced-v7',
                        help='Handler for AE-MLP model')
    parser.add_argument('--ae-mn-handler', type=str, default='v9-mkt-neutral',
                        help='Handler for AE-MLP market-neutral model')
    parser.add_argument('--cb-handler', type=str, default='catboost-v1',
                        help='Handler for CatBoost model')

    # Ensemble parameters
    parser.add_argument('--ensemble-method', type=str, default='zscore_mean',
                        choices=['mean', 'weighted', 'rank_mean', 'zscore_mean', 'zscore_weighted'],
                        help='Ensemble method (default: zscore_mean)')
    parser.add_argument('--ae-weight', type=float, default=0.34,
                        help='AE-MLP weight (default: 0.34)')
    parser.add_argument('--ae-mn-weight', type=float, default=0.33,
                        help='AE-MLP mkt-neutral weight (default: 0.33)')
    parser.add_argument('--cb-weight', type=float, default=0.33,
                        help='CatBoost weight (default: 0.33)')

    # Stacking parameters
    parser.add_argument('--stacking', action='store_true',
                        help='Learn optimal weights from validation set (Stacking)')
    parser.add_argument('--stacking-method', type=str, default='grid_search',
                        choices=['grid_search', 'grid_search_icir'],
                        help='Stacking weight learning method (default: grid_search)')
    parser.add_argument('--min-weight', type=float, default=0.05,
                        help='Minimum weight for each model (default: 0.05)')
    parser.add_argument('--diversity-bonus', type=float, default=0.1,
                        help='Bonus for balanced weights (default: 0.1)')

    # Data parameters
    parser.add_argument('--nday', type=int, default=5,
                        help='Prediction horizon (default: 5)')
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'],
                        help='Stock pool (default: sp500)')

    # Backtest parameters
    parser.add_argument('--backtest', action='store_true',
                        help='Run backtest after ensemble')
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of stocks to hold (default: 10)')
    parser.add_argument('--n-drop', type=int, default=2,
                        help='Number of stocks to drop/replace each day (default: 2)')
    parser.add_argument('--account', type=float, default=100000,
                        help='Initial account value (default: 100000)')
    parser.add_argument('--rebalance-freq', type=int, default=5,
                        help='Rebalance frequency in days (default: 5)')
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss',
                                 'mvo', 'rp', 'gmv', 'inv'],
                        help='Trading strategy (default: topk)')

    # Skip CV evaluation
    parser.add_argument('--skip-cv', action='store_true',
                        help='Skip cross-validation evaluation, go directly to ensemble + backtest')

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

    # Portfolio optimization parameters (for mvo/rp/gmv/inv strategies)
    parser.add_argument('--opt-lamb', type=float, default=2.0,
                        help='[mvo] Risk aversion (higher=safer, default: 2.0)')
    parser.add_argument('--opt-delta', type=float, default=0.2,
                        help='[mvo/rp/gmv] Max turnover per rebalance (default: 0.2)')
    parser.add_argument('--opt-alpha', type=float, default=0.01,
                        help='[mvo/rp/gmv] L2 regularization (default: 0.01)')
    parser.add_argument('--cov-lookback', type=int, default=60,
                        help='[mvo/rp/gmv/inv] Covariance lookback days (default: 60)')
    parser.add_argument('--max-weight', type=float, default=0.0,
                        help='[mvo/rp/gmv/inv] Max weight per stock, 0=no limit (default: 0, try 0.15)')

    # AI affinity filter
    parser.add_argument('--ai-filter', type=str, default='none',
                        choices=['none', 'penalty', 'exclude'],
                        help='AI affinity filter mode (default: none)')
    parser.add_argument('--ai-penalty-weight', type=float, default=0.5,
                        help='Penalty multiplier for negative-affinity stocks (default: 0.5)')
    parser.add_argument('--ai-bonus-weight', type=float, default=0.0,
                        help='Bonus multiplier for positive-affinity stocks (default: 0.0)')
    parser.add_argument('--ai-exclude-threshold', type=int, default=-1,
                        help='Affinity threshold for exclude mode, drop if <= this (default: -1)')
    parser.add_argument('--no-ai-time-scale', action='store_true',
                        help='Disable AI affinity time scaling (ramp 2020-2024)')

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

    # Model config: key -> (path, handler_arg, handler_name, display_name, model_type)
    MODEL_CONFIG = {
        'ae': {
            'path': Path(args.ae_model),
            'handler_arg': 'ae_handler',
            'handler': args.ae_handler,
            'display': 'AE-MLP',
            'type': 'ae_mlp',
        },
        'ae_mn': {
            'path': Path(args.ae_mn_model),
            'handler_arg': 'ae_mn_handler',
            'handler': args.ae_mn_handler,
            'display': 'AE-MLP-MN',
            'type': 'ae_mlp',
        },
        'cb': {
            'path': Path(args.cb_model),
            'handler_arg': 'cb_handler',
            'handler': args.cb_handler,
            'display': 'CatBoost',
            'type': 'catboost',
        },
    }

    print("=" * 80)
    print("AE-MLP + AE-MLP(mkt-neutral) + CatBoost Ensemble (V4)")
    print("=" * 80)
    for key, cfg in MODEL_CONFIG.items():
        print(f"{cfg['display']} Model:   {cfg['path']}")
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
        meta = load_model_meta(cfg['path'])
        if meta:
            print(f"    {cfg['display']} metadata: handler={meta.get('handler', 'N/A')}, nday={meta.get('nday', 'N/A')}")
            if 'handler' in meta:
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

    # [4] Create datasets
    print("\n[4] Creating datasets...")
    datasets = {}
    handlers_dict = {}
    for key, cfg in MODEL_CONFIG.items():
        print(f"    Creating {cfg['handler']} dataset for {cfg['display']}...")
        h = create_data_handler(cfg['handler'], symbols, time_splits, args.nday)
        datasets[key] = create_dataset(h, time_splits)
        handlers_dict[key] = cfg['handler']

    # [5] Load models
    print("\n[5] Loading models...")
    models = {}
    for key, cfg in MODEL_CONFIG.items():
        if cfg['type'] == 'ae_mlp':
            models[key] = load_ae_mlp_model(cfg['path'])
        elif cfg['type'] == 'catboost':
            models[key] = load_catboost_model(cfg['path'])

    # [6] CV evaluation
    if args.skip_cv:
        print("\n[6] Skipping cross-validation evaluation (--skip-cv)")
    else:
        print("\n[6] Cross-validation evaluation...")
        run_cv_evaluation(models, handlers_dict, args, symbols)

    # [7] Generate predictions
    print("\n[7] Generating predictions...")
    pred_dict = {}
    for key, cfg in MODEL_CONFIG.items():
        print(f"    {cfg['display']} predictions...")
        if cfg['type'] == 'ae_mlp':
            pred = predict_with_ae_mlp(models[key], datasets[key])
        elif cfg['type'] == 'catboost':
            pred = predict_with_catboost(models[key], datasets[key])
        pred_dict[key] = pred
        print(f"      Shape: {len(pred)}, Range: [{pred.min():.4f}, {pred.max():.4f}]")

    # Prediction statistics
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

    # [8] Correlation
    print("\n[8] Calculating pairwise correlations...")
    overall_corr, daily_corrs = calculate_correlation(pred_dict)

    print("\n    Overall Correlation Matrix:")
    print(overall_corr.to_string())

    print("\n    Daily Correlation (mean +/- std):")
    print("    " + "-" * 50)
    for pair, (mean_c, std_c) in daily_corrs.items():
        print(f"    {pair:<30s}: {mean_c:.4f} +/- {std_c:.4f}")
    print("    " + "-" * 50)

    # [9] Stacking (optional)
    learned_weights = None
    if args.stacking:
        print(f"\n[9] Stacking: Learning optimal weights from validation set...")
        print(f"    Method: {args.stacking_method}")

        val_preds = {}
        for key, cfg in MODEL_CONFIG.items():
            if cfg['type'] == 'ae_mlp':
                val_preds[key] = predict_with_ae_mlp(models[key], datasets[key], segment="valid")
            elif cfg['type'] == 'catboost':
                val_preds[key] = predict_with_catboost(models[key], datasets[key], segment="valid")
            print(f"      {cfg['display']} valid: {len(val_preds[key])} samples")

        val_label = datasets['ae'].prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(val_label, pd.DataFrame):
            val_label = val_label.iloc[:, 0]

        learned_weights, learn_info = learn_optimal_weights_multi(
            val_preds, val_label,
            method=args.stacking_method,
            min_weight=args.min_weight,
            diversity_bonus=args.diversity_bonus,
        )

        print(f"\n    Learned Weights:")
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
    step_num = 10 if args.stacking else 9
    print(f"\n[{step_num}] Ensembling predictions ({args.ensemble_method})...")

    weights = None
    if args.ensemble_method in ['weighted', 'zscore_weighted']:
        if learned_weights:
            weights = learned_weights
        else:
            weights = {key: getattr(args, f'{key}_weight') for key in MODEL_CONFIG}

    pred_ensemble = ensemble_predictions_multi(pred_dict, args.ensemble_method, weights)
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

    print("\n    +" + "=" * 70 + "+")
    print("    |" + " " * 10 + "Information Coefficient (IC) Comparison (V4)" + " " * 14 + "|")
    print("    +" + "=" * 70 + "+")
    print(f"    |  {'Model':<20s} | {'Mean IC':>10s} | {'IC Std':>10s} | {'ICIR':>10s} |")
    print("    +" + "-" * 70 + "+")
    for key, cfg in MODEL_CONFIG.items():
        r = ic_results[key]
        print(f"    |  {cfg['display']:<20s} | {r['ic']:>10.4f} | {r['std']:>10.4f} | {r['icir']:>10.4f} |")
    print("    +" + "-" * 70 + "+")
    print(f"    |  {'Ensemble (V4)':<20s} | {ens_ic:>10.4f} | {ens_std:>10.4f} | {ens_icir:>10.4f} |")
    print("    +" + "=" * 70 + "+")

    # Improvement
    best_single_ic = max(r['ic'] for r in ic_results.values())
    best_single_icir = max(r['icir'] for r in ic_results.values())

    ic_improvement = (ens_ic - best_single_ic) / abs(best_single_ic) * 100 if best_single_ic != 0 else 0
    icir_improvement = (ens_icir - best_single_icir) / abs(best_single_icir) * 100 if best_single_icir != 0 else 0

    print(f"\n    Ensemble Performance vs Best Single Model:")
    print(f"    IC improvement:   {ic_improvement:>+.2f}%")
    print(f"    ICIR improvement: {icir_improvement:>+.2f}%")

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
    print("ENSEMBLE V4 ANALYSIS COMPLETE")
    print("=" * 80)
    print("Pairwise Correlations (daily mean):")
    for pair, (mean_c, _) in daily_corrs.items():
        print(f"  {pair}: {mean_c:.4f}")
    if learned_weights:
        print("Stacking Weights:")
        for key, cfg in MODEL_CONFIG.items():
            print(f"  {cfg['display']}: {learned_weights[key]:.3f}")
    for key, cfg in MODEL_CONFIG.items():
        r = ic_results[key]
        print(f"{cfg['display']:15s} IC: {r['ic']:.4f} (ICIR: {r['icir']:.4f})")
    print(f"{'Ensemble V4':15s} IC: {ens_ic:.4f} (ICIR: {ens_icir:.4f})")
    print("=" * 80)

    # Backtest
    if args.backtest:
        run_ensemble_backtest(pred_ensemble, args, time_splits)

        print("\n" + "=" * 80)
        print("ENSEMBLE V4 BACKTEST COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    main()
