"""
Common utilities for ensemble model scripts.

Shared functions for model loading, prediction, ensemble combining,
IC evaluation, weight learning, and backtesting from predictions.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Utility Functions
# ============================================================================

def zscore_by_day(series):
    """Cross-sectional z-score normalization within each day."""
    mean = series.groupby(level='datetime').transform('mean')
    std = series.groupby(level='datetime').transform('std')
    return (series - mean) / (std + 1e-8)


# ============================================================================
# Model Loading Functions
# ============================================================================

def load_ae_mlp_model(model_path: Path):
    """Load AE-MLP (.keras) model"""
    from models.deep.ae_mlp_model import AEMLP
    print(f"    Loading AE-MLP model from: {model_path}")
    model = AEMLP.load(str(model_path))
    return model


def load_model_meta(model_path: Path) -> dict:
    """Load model metadata (.meta.pkl) file"""
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


# ============================================================================
# Data Preparation Functions
# ============================================================================

def create_ensemble_data_handler(handler_name: str, symbols: list,
                                 time_splits: dict, nday: int,
                                 include_valid: bool = False):
    """Create DataHandler for ensemble prediction.

    Args:
        handler_name: Handler registry name
        symbols: List of stock symbols
        time_splits: Dict with train/valid/test start/end dates
        nday: Prediction horizon
        include_valid: If True, load data from valid_start; otherwise from test_start
    """
    from models.common.handlers import get_handler_class

    HandlerClass = get_handler_class(handler_name)

    start_time = time_splits['valid_start'] if include_valid else time_splits['test_start']

    handler = HandlerClass(
        volatility_window=nday,
        instruments=symbols,
        start_time=start_time,
        end_time=time_splits['test_end'],
        fit_start_time=start_time,
        fit_end_time=time_splits['test_end'],
        infer_processors=[],
    )

    return handler


def create_ensemble_dataset(handler, time_splits: dict,
                            include_valid: bool = False):
    """Create Qlib DatasetH for ensemble prediction.

    Args:
        handler: DataHandler instance
        time_splits: Dict with train/valid/test start/end dates
        include_valid: If True, include valid segment; otherwise only test
    """
    from qlib.data.dataset import DatasetH

    segments = {"test": (time_splits['test_start'], time_splits['test_end'])}
    if include_valid:
        segments["valid"] = (time_splits['valid_start'], time_splits['valid_end'])

    return DatasetH(handler=handler, segments=segments)


# ============================================================================
# Prediction Functions
# ============================================================================

def predict_with_ae_mlp(model, dataset, segment: str = "test") -> pd.Series:
    """Generate predictions with AE-MLP model"""
    pred = model.predict(dataset, segment=segment)
    pred.name = 'score'
    return pred


def predict_with_catboost(model, dataset, segment: str = "test") -> pd.Series:
    """Generate predictions with CatBoost model"""
    from qlib.data.dataset.handler import DataHandlerLP

    data = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
    data = data.fillna(0).replace([np.inf, -np.inf], 0)
    pred_values = model.predict(data.values)
    pred = pd.Series(pred_values, index=data.index, name='score')
    return pred


# ============================================================================
# Evaluation Functions
# ============================================================================

def compute_ic(pred: pd.Series, label: pd.Series) -> tuple:
    """Calculate IC (Information Coefficient) from prediction and label Series.

    Returns:
        tuple: (mean_ic, ic_std, icir, ic_by_date)
    """
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


def calculate_pairwise_correlations(preds: dict) -> pd.DataFrame:
    """Calculate pairwise correlations between model predictions.

    Args:
        preds: Dict of {model_name: prediction_series}

    Returns:
        pd.DataFrame: Correlation matrix
    """
    model_names = list(preds.keys())
    n_models = len(model_names)

    common_idx = preds[model_names[0]].index
    for name in model_names[1:]:
        common_idx = common_idx.intersection(preds[name].index)

    corr_matrix = np.zeros((n_models, n_models))
    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                p_i = preds[name_i].loc[common_idx]
                p_j = preds[name_j].loc[common_idx]
                corr_matrix[i, j] = p_i.corr(p_j)

    return pd.DataFrame(corr_matrix, index=model_names, columns=model_names)


# ============================================================================
# Ensemble Methods
# ============================================================================

def ensemble_predictions(pred_dict: dict, method: str = 'zscore_mean',
                         weights: dict = None) -> pd.Series:
    """
    Ensemble multiple model predictions.

    Parameters
    ----------
    pred_dict : dict
        Dict of {model_name: prediction_series}
    method : str
        'mean', 'weighted', 'rank_mean', 'zscore_mean', 'zscore_weighted'
    weights : dict, optional
        {model_name: weight} for weighted ensemble

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

    aligned = {name: pred_dict[name].loc[common_idx] for name in names}

    if method == 'mean':
        ensemble_pred = sum(aligned[name] for name in names) / len(names)

    elif method == 'weighted':
        if weights is None:
            weights = {name: 1.0 / len(names) for name in names}
        total = sum(weights.values())
        ensemble_pred = sum(aligned[name] * weights[name] for name in names) / total

    elif method == 'rank_mean':
        ranks = {name: aligned[name].groupby(level='datetime').rank(pct=True)
                 for name in names}
        ensemble_pred = sum(ranks[name] for name in names) / len(names)

    elif method == 'zscore_mean':
        zscores = {name: zscore_by_day(aligned[name]) for name in names}
        ensemble_pred = sum(zscores[name] for name in names) / len(names)

    elif method == 'zscore_weighted':
        if weights is None:
            weights = {name: 1.0 / len(names) for name in names}
        zscores = {name: zscore_by_day(aligned[name]) for name in names}
        total = sum(weights.values())
        ensemble_pred = sum(zscores[name] * weights[name] for name in names) / total

    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    ensemble_pred.name = 'score'
    return ensemble_pred


# ============================================================================
# Weight Learning
# ============================================================================

def _generate_weight_combinations(n, remaining=1.0, min_w=0.05, step=0.05):
    """Generate all weight combinations that sum to remaining."""
    if n == 1:
        if remaining >= min_w - 1e-6:
            yield [remaining]
        return
    for w in np.arange(min_w, remaining - min_w * (n - 1) + 0.01, step):
        for rest in _generate_weight_combinations(n - 1, remaining - w, min_w, step):
            yield [w] + rest


def learn_optimal_weights(preds: dict, label: pd.Series,
                          method: str = 'grid_search',
                          min_weight: float = 0.05,
                          diversity_bonus: float = 0.05) -> tuple:
    """
    Learn optimal ensemble weights for multiple models.

    Parameters
    ----------
    preds : dict
        Dict of {model_name: prediction_series}
    label : pd.Series
        True labels
    method : str
        'grid_search', 'grid_search_icir', 'regression', 'ridge', 'equal'
    min_weight : float
        Minimum weight for each model
    diversity_bonus : float
        Bonus for balanced weights

    Returns
    -------
    tuple
        (weights_dict, info_dict)
    """
    model_names = list(preds.keys())
    n_models = len(model_names)

    # Find common index
    common_idx = label.index
    for name in model_names:
        common_idx = common_idx.intersection(preds[name].index)

    # Align all series
    aligned_preds = {name: preds[name].loc[common_idx] for name in model_names}
    y = label.loc[common_idx]

    # Remove NaN
    valid_mask = ~y.isna()
    for name in model_names:
        valid_mask &= ~aligned_preds[name].isna()

    for name in model_names:
        aligned_preds[name] = aligned_preds[name][valid_mask]
    y = y[valid_mask]

    # Z-score normalize within each day
    normalized_preds = {name: zscore_by_day(aligned_preds[name]) for name in model_names}

    if method == 'equal':
        weights = {name: 1.0 / n_models for name in model_names}
        return weights, {'method': 'equal'}

    elif method in ['grid_search', 'grid_search_icir']:
        best_score = -np.inf
        best_weights = {name: 1.0 / n_models for name in model_names}
        best_ic = 0

        step = 0.05

        for weight_list in _generate_weight_combinations(n_models, 1.0, min_weight, step):
            weights = {name: w for name, w in zip(model_names, weight_list)}

            ensemble = sum(normalized_preds[name] * weights[name] for name in model_names)

            df = pd.DataFrame({'pred': ensemble, 'label': y})
            ic_by_date = df.groupby(level='datetime').apply(
                lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
            )
            ic_series = ic_by_date.dropna()

            if len(ic_series) == 0:
                continue

            mean_ic = ic_series.mean()
            ic_std = ic_series.std()
            icir = mean_ic / ic_std if ic_std > 0 else 0

            if method == 'grid_search':
                weight_variance = np.var(weight_list)
                max_variance = ((1 - min_weight * n_models) / n_models) ** 2 * (n_models - 1)
                diversity = 1 - weight_variance / (max_variance + 1e-8)
                score = mean_ic + diversity_bonus * diversity
            else:  # grid_search_icir
                score = icir

            if score > best_score:
                best_score = score
                best_weights = weights.copy()
                best_ic = mean_ic

        return best_weights, {
            'method': method,
            'best_ic': best_ic,
        }

    elif method == 'regression':
        from sklearn.linear_model import LinearRegression

        X = np.column_stack([normalized_preds[name].values for name in model_names])
        reg = LinearRegression(fit_intercept=False, positive=True)
        reg.fit(X, y.values)

        raw_weights = reg.coef_
        total = sum(abs(w) for w in raw_weights)
        if total > 0:
            weights = {name: abs(raw_weights[i]) / total for i, name in enumerate(model_names)}
        else:
            weights = {name: 1.0 / n_models for name in model_names}

        return weights, {'method': 'regression', 'r2': reg.score(X, y.values)}

    elif method == 'ridge':
        from sklearn.linear_model import Ridge

        X = np.column_stack([normalized_preds[name].values for name in model_names])
        reg = Ridge(alpha=1.0, fit_intercept=False)
        reg.fit(X, y.values)

        raw_weights = reg.coef_
        total = sum(abs(w) for w in raw_weights)
        if total > 0:
            weights = {name: abs(raw_weights[i]) / total for i, name in enumerate(model_names)}
        else:
            weights = {name: 1.0 / n_models for name in model_names}

        return weights, {'method': 'ridge', 'r2': reg.score(X, y.values)}

    else:
        raise ValueError(f"Unknown weight learning method: {method}")


# ============================================================================
# Backtest Functions
# ============================================================================

def run_ensemble_backtest(pred: pd.Series, args, time_splits: dict,
                          model_name: str = "Ensemble"):
    """Run backtest with ensembled predictions.

    Supports all strategy types: topk, dynamic_risk, vol_stoploss, mvo, rp, gmv, inv.

    Args:
        pred: Ensembled prediction Series
        args: Parsed arguments (must include strategy, topk, n_drop, etc.)
        time_splits: Dict with test_start/test_end
        model_name: Display name for outputs
    """
    from qlib.backtest import backtest as qlib_backtest

    from utils.strategy import get_strategy_config
    from utils.backtest_utils import plot_backtest_curve, generate_trade_records
    from data.stock_pools import STOCK_POOLS
    from models.common.config import PROJECT_ROOT

    strategy_label = model_name
    if hasattr(args, 'strategy') and args.strategy in ('mvo', 'rp', 'gmv', 'inv'):
        strategy_label = f"{args.strategy.upper()} Strategy ({model_name})"

    print("\n" + "=" * 70)
    print(f"BACKTEST with {strategy_label}")
    print("=" * 70)

    pred_df = pred.to_frame("score")

    print(f"\n[BT-1] Configuring backtest...")
    if hasattr(args, 'strategy'):
        print(f"    Strategy: {args.strategy}")
    print(f"    Topk: {args.topk}")
    if not hasattr(args, 'strategy') or args.strategy not in ('mvo', 'rp', 'gmv', 'inv'):
        print(f"    N_drop: {args.n_drop}")
    print(f"    Account: ${args.account:,.0f}")
    print(f"    Rebalance Freq: every {args.rebalance_freq} day(s)")
    print(f"    Period: {time_splits['test_start']} to {time_splits['test_end']}")

    # Build strategy params
    strategy_type = getattr(args, 'strategy', 'topk')

    optimizer_params = None
    if strategy_type in ("mvo", "rp", "gmv", "inv"):
        optimizer_params = {
            "lamb": getattr(args, 'opt_lamb', 1.0),
            "delta": getattr(args, 'opt_delta', 0.3),
            "alpha": getattr(args, 'opt_alpha', 0.01),
            "cov_lookback": getattr(args, 'cov_lookback', 60),
            "max_weight": getattr(args, 'max_weight', 0),
        }

    dynamic_risk_params = None
    if strategy_type == "vol_stoploss":
        dynamic_risk_params = {
            "lookback": getattr(args, 'risk_lookback', 20),
            "vol_threshold_high": getattr(args, 'vol_high', 0.25),
            "vol_threshold_medium": getattr(args, 'vol_medium', 0.18),
            "stop_loss_threshold": getattr(args, 'stop_loss', -0.08),
            "no_sell_after_drop": getattr(args, 'no_sell_after_drop', -0.05),
            "risk_degree_high": getattr(args, 'risk_high', 0.5),
            "risk_degree_medium": getattr(args, 'risk_medium', 0.75),
            "risk_degree_normal": getattr(args, 'risk_normal', 0.95),
            "market_proxy": getattr(args, 'market_proxy', "SPY"),
        }
    elif strategy_type == "dynamic_risk":
        dynamic_risk_params = {
            "lookback": getattr(args, 'risk_lookback', 20),
            "drawdown_threshold": getattr(args, 'drawdown_threshold', -0.05),
            "momentum_threshold": getattr(args, 'momentum_threshold', -0.02),
            "risk_degree_high": getattr(args, 'risk_high', 0.5),
            "risk_degree_medium": getattr(args, 'risk_medium', 0.75),
            "risk_degree_normal": getattr(args, 'risk_normal', 0.95),
            "market_proxy": getattr(args, 'market_proxy', "SPY"),
        }

    strategy_config = get_strategy_config(
        pred_df, args.topk, args.n_drop, args.rebalance_freq,
        strategy_type=strategy_type,
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

        from models.common.backtest import _analyze_backtest_results
        for freq, (report_df, positions) in portfolio_metric_dict.items():
            _analyze_backtest_results(report_df, positions, freq, args,
                                     time_splits, model_name, PROJECT_ROOT)

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
