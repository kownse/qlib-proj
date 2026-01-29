"""
AE-MLP + CatBoost Ensemble

Load pre-trained AE-MLP and CatBoost models, generate predictions on test set,
calculate correlation between outputs, ensemble them, and compute IC.

Usage:
    python scripts/models/ensemble/run_ae_catboost_ensemble.py
    python scripts/models/ensemble/run_ae_catboost_ensemble.py --ensemble-method rank_mean
    python scripts/models/ensemble/run_ae_catboost_ensemble.py --ae-weight 0.6 --cb-weight 0.4
    python scripts/models/ensemble/run_ae_catboost_ensemble.py --backtest --topk 10 --n-drop 2
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
from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG,
    PROJECT_ROOT,
    QLIB_DATA_PATH,
    MODEL_SAVE_PATH,
    CV_FOLDS,
    FINAL_TEST,
)
from models.deep.ae_mlp_model import AEMLP


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
    # Try with .meta.pkl suffix
    meta_path = model_path.with_suffix('.meta.pkl')
    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            return pickle.load(f)

    # Try replacing _best suffix
    stem = model_path.stem
    if stem.endswith('_best'):
        alt_meta_path = model_path.parent / (stem[:-5] + '.meta.pkl')
        if alt_meta_path.exists():
            with open(alt_meta_path, 'rb') as f:
                return pickle.load(f)

    return {}


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


def predict_with_ae_mlp(model: AEMLP, dataset: DatasetH, segment: str = "test") -> pd.Series:
    """Generate predictions with AE-MLP model"""
    pred = model.predict(dataset, segment=segment)
    pred.name = 'score'
    return pred


def predict_with_catboost(model: CatBoostRegressor, dataset: DatasetH, segment: str = "test") -> pd.Series:
    """Generate predictions with CatBoost model"""
    # Get data with DK_L for consistent preprocessing
    data = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)

    # Fill NaN and handle inf values
    data = data.fillna(0).replace([np.inf, -np.inf], 0)

    # Predict
    pred_values = model.predict(data.values)
    pred = pd.Series(pred_values, index=data.index, name='score')

    return pred


def calculate_correlation(pred1: pd.Series, pred2: pd.Series) -> tuple:
    """Calculate correlation between two prediction series"""
    # Find common index
    common_idx = pred1.index.intersection(pred2.index)
    p1 = pred1.loc[common_idx]
    p2 = pred2.loc[common_idx]

    # Overall correlation
    overall_corr = p1.corr(p2)

    # Daily correlation (by datetime)
    daily_corr = pd.DataFrame({'p1': p1, 'p2': p2}).groupby(level='datetime').apply(
        lambda x: x['p1'].corr(x['p2']) if len(x) > 1 else np.nan
    )
    daily_corr = daily_corr.dropna()

    mean_daily_corr = daily_corr.mean()
    std_daily_corr = daily_corr.std()

    return overall_corr, mean_daily_corr, std_daily_corr, daily_corr


def learn_optimal_weights(pred1: pd.Series, pred2: pd.Series, label: pd.Series,
                          method: str = 'grid_search', use_zscore: bool = True,
                          min_weight: float = 0.1, diversity_bonus: float = 0.1) -> tuple:
    """
    Learn optimal ensemble weights using validation data (Stacking).

    Parameters
    ----------
    pred1, pred2 : pd.Series
        Prediction series from each model on validation set
    label : pd.Series
        True labels on validation set
    method : str
        Learning method: 'grid_search', 'grid_search_icir', 'regression', 'ridge', 'analytical'
    use_zscore : bool
        Whether to zscore normalize predictions before learning weights
    min_weight : float
        Minimum weight for each model (prevents extreme weights, default 0.1)
    diversity_bonus : float
        Bonus for balanced weights to encourage diversity (default 0.1)
        Final score = IC + diversity_bonus * (1 - |w1 - 0.5| * 2)

    Returns
    -------
    tuple
        (weight1, weight2) optimal weights
    """
    # Align all series
    common_idx = pred1.index.intersection(pred2.index).intersection(label.index)
    p1 = pred1.loc[common_idx]
    p2 = pred2.loc[common_idx]
    y = label.loc[common_idx]

    # Remove NaN
    valid_mask = ~(p1.isna() | p2.isna() | y.isna())
    p1 = p1[valid_mask]
    p2 = p2[valid_mask]
    y = y[valid_mask]

    # Optionally zscore normalize within each day
    if use_zscore:
        def zscore_by_day(x):
            mean = x.groupby(level='datetime').transform('mean')
            std = x.groupby(level='datetime').transform('std')
            return (x - mean) / (std + 1e-8)
        p1 = zscore_by_day(p1)
        p2 = zscore_by_day(p2)

    if method == 'grid_search':
        # Grid search to maximize IC with diversity bonus
        # Prevents overfitting to one model by encouraging balanced weights
        best_score = -np.inf
        best_w1 = 0.5
        best_ic = 0

        for w1 in np.arange(min_weight, 1.0 - min_weight + 0.01, 0.05):
            w2 = 1 - w1
            ensemble = p1 * w1 + p2 * w2

            # Calculate daily IC
            df = pd.DataFrame({'pred': ensemble, 'label': y})
            ic_by_date = df.groupby(level='datetime').apply(
                lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
            )
            mean_ic = ic_by_date.dropna().mean()

            # Diversity bonus: highest when w1 = 0.5, zero when w1 = 0 or 1
            diversity = 1 - abs(w1 - 0.5) * 2
            score = mean_ic + diversity_bonus * diversity

            if score > best_score:
                best_score = score
                best_w1 = w1
                best_ic = mean_ic

        return (best_w1, 1 - best_w1), {
            'method': 'grid_search',
            'best_ic': best_ic,
            'diversity_bonus': diversity_bonus
        }

    elif method == 'grid_search_icir':
        # Grid search to maximize ICIR (more stable than IC)
        best_icir = -np.inf
        best_w1 = 0.5
        best_ic = 0

        for w1 in np.arange(min_weight, 1.0 - min_weight + 0.01, 0.05):
            w2 = 1 - w1
            ensemble = p1 * w1 + p2 * w2

            # Calculate daily IC
            df = pd.DataFrame({'pred': ensemble, 'label': y})
            ic_by_date = df.groupby(level='datetime').apply(
                lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
            )
            ic_series = ic_by_date.dropna()
            mean_ic = ic_series.mean()
            ic_std = ic_series.std()
            icir = mean_ic / ic_std if ic_std > 0 else 0

            if icir > best_icir:
                best_icir = icir
                best_w1 = w1
                best_ic = mean_ic

        return (best_w1, 1 - best_w1), {
            'method': 'grid_search_icir',
            'best_ic': best_ic,
            'best_icir': best_icir
        }

    elif method == 'regression':
        # Linear regression: y = w1*p1 + w2*p2
        from sklearn.linear_model import LinearRegression

        X = np.column_stack([p1.values, p2.values])
        reg = LinearRegression(fit_intercept=False, positive=True)
        reg.fit(X, y.values)

        w1, w2 = reg.coef_
        # Normalize weights to sum to 1
        total = w1 + w2
        if total > 0:
            w1, w2 = w1 / total, w2 / total
        else:
            w1, w2 = 0.5, 0.5

        return (w1, w2), {'method': 'regression', 'r2': reg.score(X, y.values)}

    elif method == 'ridge':
        # Ridge regression with regularization
        from sklearn.linear_model import Ridge

        X = np.column_stack([p1.values, p2.values])
        reg = Ridge(alpha=1.0, fit_intercept=False)
        reg.fit(X, y.values)

        w1, w2 = reg.coef_
        # Normalize weights to sum to 1
        total = abs(w1) + abs(w2)
        if total > 0:
            w1, w2 = w1 / total, w2 / total
        else:
            w1, w2 = 0.5, 0.5

        return (w1, w2), {'method': 'ridge', 'r2': reg.score(X, y.values)}

    elif method == 'analytical':
        # Analytical solution considering correlation
        # Maximize IC of ensemble: w1*IC1 + w2*IC2 considering covariance
        # This is similar to Markowitz portfolio optimization

        # Calculate daily ICs for each model
        df = pd.DataFrame({'p1': p1, 'p2': p2, 'label': y})
        daily_stats = df.groupby(level='datetime').apply(
            lambda x: pd.Series({
                'ic1': x['p1'].corr(x['label']),
                'ic2': x['p2'].corr(x['label']),
                'corr': x['p1'].corr(x['p2'])
            })
        ).dropna()

        ic1 = daily_stats['ic1'].mean()
        ic2 = daily_stats['ic2'].mean()
        var1 = daily_stats['ic1'].var()
        var2 = daily_stats['ic2'].var()
        cov12 = daily_stats['ic1'].cov(daily_stats['ic2'])

        # Optimal weight for maximum Sharpe-like ratio
        # w1 = (ic1*var2 - ic2*cov12) / (ic1*var2 + ic2*var1 - (ic1+ic2)*cov12)
        denom = ic1 * var2 + ic2 * var1 - (ic1 + ic2) * cov12
        if abs(denom) > 1e-8:
            w1 = (ic1 * var2 - ic2 * cov12) / denom
            w1 = np.clip(w1, 0, 1)
        else:
            w1 = 0.5

        return (w1, 1 - w1), {
            'method': 'analytical',
            'ic1': ic1, 'ic2': ic2,
            'corr': daily_stats['corr'].mean()
        }

    else:
        raise ValueError(f"Unknown weight learning method: {method}")


def ensemble_predictions(pred1: pd.Series, pred2: pd.Series,
                         method: str = 'mean', weights: tuple = None) -> pd.Series:
    """
    Ensemble two model predictions

    Parameters
    ----------
    pred1, pred2 : pd.Series
        Prediction series from each model
    method : str
        Ensemble method: 'mean', 'weighted', 'rank_mean', 'zscore_mean', 'ic_weighted'
    weights : tuple, optional
        (weight1, weight2) for weighted ensemble

    Returns
    -------
    pd.Series
        Ensembled predictions
    """
    # Find common index
    common_idx = pred1.index.intersection(pred2.index)
    p1 = pred1.loc[common_idx]
    p2 = pred2.loc[common_idx]

    if method == 'mean':
        ensemble_pred = (p1 + p2) / 2
    elif method == 'weighted':
        if weights is None:
            weights = (0.5, 0.5)
        w1, w2 = weights
        total = w1 + w2
        ensemble_pred = (p1 * w1 + p2 * w2) / total
    elif method == 'rank_mean':
        # Convert to rank percentiles within each day, then average
        rank1 = p1.groupby(level='datetime').rank(pct=True)
        rank2 = p2.groupby(level='datetime').rank(pct=True)
        ensemble_pred = (rank1 + rank2) / 2
    elif method == 'zscore_mean':
        # Normalize to z-scores within each day, then average
        # This preserves relative magnitude information better than rank
        def zscore_by_day(x):
            mean = x.groupby(level='datetime').transform('mean')
            std = x.groupby(level='datetime').transform('std')
            return (x - mean) / (std + 1e-8)
        z1 = zscore_by_day(p1)
        z2 = zscore_by_day(p2)
        ensemble_pred = (z1 + z2) / 2
    elif method == 'zscore_weighted':
        # Z-score normalize then weighted average
        if weights is None:
            weights = (0.5, 0.5)
        w1, w2 = weights
        total = w1 + w2
        def zscore_by_day(x):
            mean = x.groupby(level='datetime').transform('mean')
            std = x.groupby(level='datetime').transform('std')
            return (x - mean) / (std + 1e-8)
        z1 = zscore_by_day(p1)
        z2 = zscore_by_day(p2)
        ensemble_pred = (z1 * w1 + z2 * w2) / total
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    ensemble_pred.name = 'score'
    return ensemble_pred


def compute_ic(pred: pd.Series, label: pd.Series) -> tuple:
    """Calculate IC (Information Coefficient)"""
    # Align indices
    common_idx = pred.index.intersection(label.index)
    pred_aligned = pred.loc[common_idx]
    label_aligned = label.loc[common_idx]

    # Remove NaN
    valid_idx = ~(pred_aligned.isna() | label_aligned.isna())
    pred_clean = pred_aligned[valid_idx]
    label_clean = label_aligned[valid_idx]

    # Calculate daily IC
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


def run_ensemble_backtest(pred: pd.Series, args, time_splits: dict):
    """Run backtest with ensembled predictions"""
    from qlib.backtest import backtest as qlib_backtest
    from qlib.contrib.evaluate import risk_analysis

    print("\n" + "=" * 70)
    print("BACKTEST with TopkDropoutStrategy (AE-MLP + CatBoost Ensemble)")
    print("=" * 70)

    pred_df = pred.to_frame("score")

    print(f"\n[BT-1] Configuring backtest...")
    print(f"    Strategy: {args.strategy}")
    print(f"    Topk: {args.topk}")
    print(f"    N_drop: {args.n_drop}")
    print(f"    Account: ${args.account:,.0f}")
    print(f"    Rebalance Freq: every {args.rebalance_freq} day(s)")
    print(f"    Period: {time_splits['test_start']} to {time_splits['test_end']}")

    # Dynamic risk strategy parameters
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
        dynamic_risk_params=dynamic_risk_params
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

    # Save report
    output_path = PROJECT_ROOT / "outputs" / f"ae_catboost_ensemble_backtest_report_{freq}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path)
    print(f"\n    Report saved to: {output_path}")

    # Plot performance chart
    print(f"\n[BT-4] Generating performance chart...")
    # Ensure handler attribute exists for plot function
    if not hasattr(args, 'handler'):
        args.handler = "ensemble"
    plot_backtest_curve(report_df, args, freq, PROJECT_ROOT, model_name="AE_CatBoost_Ensemble")

    # Generate trade records
    print(f"\n[BT-5] Generating trade records...")
    generate_trade_records(positions, args, freq, PROJECT_ROOT, model_name="ae_catboost_ensemble")


def main():
    parser = argparse.ArgumentParser(
        description='AE-MLP + CatBoost Ensemble',
    )

    # Model paths
    parser.add_argument('--ae-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'ae_mlp_cv_alpha158-enhanced-v7_sp500_5d_best.keras'),
                        help='AE-MLP model path (.keras)')
    parser.add_argument('--cb-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'catboost_cv_catboost-v1_test_5d_20260129_105915_best.cbm'),
                        help='CatBoost model path (.cbm)')

    # Handler configuration (override metadata)
    parser.add_argument('--ae-handler', type=str, default='alpha158-enhanced-v7',
                        help='Handler for AE-MLP model')
    parser.add_argument('--cb-handler', type=str, default='catboost-v1',
                        help='Handler for CatBoost model')

    # Ensemble parameters
    parser.add_argument('--ensemble-method', type=str, default='zscore_mean',
                        choices=['mean', 'weighted', 'rank_mean', 'zscore_mean', 'zscore_weighted'],
                        help='Ensemble method (default: zscore_mean)')
    parser.add_argument('--ae-weight', type=float, default=0.5,
                        help='AE-MLP weight for weighted ensemble (default: 0.5)')
    parser.add_argument('--cb-weight', type=float, default=0.5,
                        help='CatBoost weight for weighted ensemble (default: 0.5)')

    # Stacking parameters
    parser.add_argument('--stacking', action='store_true',
                        help='Learn optimal weights from validation set (Stacking)')
    parser.add_argument('--stacking-method', type=str, default='grid_search',
                        choices=['grid_search', 'grid_search_icir', 'regression', 'ridge', 'analytical'],
                        help='Stacking weight learning method (default: grid_search)')
    parser.add_argument('--min-weight', type=float, default=0.1,
                        help='Minimum weight for each model (default: 0.1, prevents extreme weights)')
    parser.add_argument('--diversity-bonus', type=float, default=0.1,
                        help='Bonus for balanced weights (default: 0.1, set to 0 to disable)')

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
                        choices=['topk', 'dynamic_risk', 'vol_stoploss'],
                        help='Trading strategy (default: topk)')

    # Strategy parameters (for dynamic_risk and vol_stoploss)
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

    print("=" * 70)
    print("AE-MLP + CatBoost Ensemble")
    print("=" * 70)
    print(f"AE-MLP Model: {args.ae_model}")
    print(f"CatBoost Model: {args.cb_model}")
    print(f"AE-MLP Handler: {args.ae_handler}")
    print(f"CatBoost Handler: {args.cb_handler}")
    print(f"Stock Pool: {args.stock_pool}")
    print(f"Prediction Horizon: {args.nday} days")
    print(f"Ensemble Method: {args.ensemble_method}")
    if args.ensemble_method == 'weighted':
        print(f"Weights: AE-MLP={args.ae_weight}, CatBoost={args.cb_weight}")
    print(f"Test Period: {time_splits['test_start']} to {time_splits['test_end']}")
    print("=" * 70)

    # Check model files exist
    ae_path = Path(args.ae_model)
    cb_path = Path(args.cb_model)

    if not ae_path.exists():
        print(f"Error: AE-MLP model not found: {ae_path}")
        sys.exit(1)
    if not cb_path.exists():
        print(f"Error: CatBoost model not found: {cb_path}")
        sys.exit(1)

    # Load metadata
    print("\n[1] Loading model metadata...")
    ae_meta = load_model_meta(ae_path)
    cb_meta = load_model_meta(cb_path)

    if ae_meta:
        print(f"    AE-MLP metadata found: handler={ae_meta.get('handler', 'N/A')}, nday={ae_meta.get('nday', 'N/A')}")
        if 'handler' in ae_meta:
            args.ae_handler = ae_meta['handler']
    else:
        print(f"    AE-MLP metadata not found, using default handler: {args.ae_handler}")

    if cb_meta:
        print(f"    CatBoost metadata found: handler={cb_meta.get('handler', 'N/A')}, nday={cb_meta.get('nday', 'N/A')}")
        if 'handler' in cb_meta:
            args.cb_handler = cb_meta['handler']
    else:
        print(f"    CatBoost metadata not found, using default handler: {args.cb_handler}")

    # Initialize Qlib
    print("\n[2] Initializing Qlib...")
    qlib.init(
        provider_uri=str(QLIB_DATA_PATH),
        region=REG_US,
        custom_ops=TALIB_OPS,
        kernels=1,
        joblib_backend=None,
    )
    print("    Qlib initialized with TA-Lib support")

    # Get symbols
    symbols = STOCK_POOLS[args.stock_pool]
    print(f"\n[3] Using stock pool: {args.stock_pool} ({len(symbols)} stocks)")

    # Create datasets for each model (different handlers)
    print("\n[4] Creating datasets...")

    print(f"    Creating {args.ae_handler} dataset for AE-MLP...")
    ae_handler = create_data_handler(args.ae_handler, symbols, time_splits, args.nday)
    ae_dataset = create_dataset(ae_handler, time_splits)

    print(f"    Creating {args.cb_handler} dataset for CatBoost...")
    cb_handler = create_data_handler(args.cb_handler, symbols, time_splits, args.nday)
    cb_dataset = create_dataset(cb_handler, time_splits)

    # Load models
    print("\n[5] Loading models...")
    ae_model = load_ae_mlp_model(ae_path)
    cb_model = load_catboost_model(cb_path)

    # Generate predictions
    print("\n[6] Generating predictions...")

    print("    AE-MLP predictions...")
    pred_ae = predict_with_ae_mlp(ae_model, ae_dataset)
    print(f"      Shape: {len(pred_ae)}, Range: [{pred_ae.min():.4f}, {pred_ae.max():.4f}]")

    print("    CatBoost predictions...")
    pred_cb = predict_with_catboost(cb_model, cb_dataset)
    print(f"      Shape: {len(pred_cb)}, Range: [{pred_cb.min():.4f}, {pred_cb.max():.4f}]")

    # Compare prediction statistics
    print("\n    Prediction Statistics Comparison:")
    print("    " + "=" * 60)
    print(f"    {'Metric':<20s} | {'AE-MLP':>15s} | {'CatBoost':>15s} | {'Ratio':>10s}")
    print("    " + "-" * 60)

    ae_mean, cb_mean = pred_ae.mean(), pred_cb.mean()
    ae_std, cb_std = pred_ae.std(), pred_cb.std()
    ae_median, cb_median = pred_ae.median(), pred_cb.median()
    ae_abs_mean, cb_abs_mean = pred_ae.abs().mean(), pred_cb.abs().mean()

    # Calculate ratios (AE-MLP / CatBoost)
    mean_ratio = ae_mean / cb_mean if cb_mean != 0 else float('inf')
    std_ratio = ae_std / cb_std if cb_std != 0 else float('inf')
    abs_mean_ratio = ae_abs_mean / cb_abs_mean if cb_abs_mean != 0 else float('inf')

    print(f"    {'Mean':<20s} | {ae_mean:>15.6f} | {cb_mean:>15.6f} | {mean_ratio:>10.2f}x")
    print(f"    {'Std':<20s} | {ae_std:>15.6f} | {cb_std:>15.6f} | {std_ratio:>10.2f}x")
    print(f"    {'Median':<20s} | {ae_median:>15.6f} | {cb_median:>15.6f} | {'-':>10s}")
    print(f"    {'Abs Mean':<20s} | {ae_abs_mean:>15.6f} | {cb_abs_mean:>15.6f} | {abs_mean_ratio:>10.2f}x")
    print(f"    {'Min':<20s} | {pred_ae.min():>15.6f} | {pred_cb.min():>15.6f} | {'-':>10s}")
    print(f"    {'Max':<20s} | {pred_ae.max():>15.6f} | {pred_cb.max():>15.6f} | {'-':>10s}")
    print("    " + "=" * 60)

    # Warning if scales are very different
    if abs(std_ratio) > 10 or abs(std_ratio) < 0.1:
        print(f"\n    WARNING: Prediction scales differ significantly ({std_ratio:.1f}x)!")
        print(f"    Consider using 'rank_mean' ensemble method or normalizing predictions.")

    # Calculate correlation between predictions
    print("\n[7] Calculating correlation between model outputs...")
    overall_corr, mean_daily_corr, std_daily_corr, daily_corr = calculate_correlation(pred_ae, pred_cb)

    print(f"\n    Prediction Correlation:")
    print(f"    " + "-" * 50)
    print(f"    Overall Correlation:     {overall_corr:>10.4f}")
    print(f"    Mean Daily Correlation:  {mean_daily_corr:>10.4f}")
    print(f"    Std Daily Correlation:   {std_daily_corr:>10.4f}")
    print(f"    " + "-" * 50)

    # Stacking: Learn optimal weights from validation set
    learned_weights = None
    if args.stacking:
        print(f"\n[8] Stacking: Learning optimal weights from validation set...")
        print(f"    Method: {args.stacking_method}")

        # Generate predictions on validation set
        print("    Generating validation predictions...")
        val_pred_ae = predict_with_ae_mlp(ae_model, ae_dataset, segment="valid")
        val_pred_cb = predict_with_catboost(cb_model, cb_dataset, segment="valid")
        print(f"      AE-MLP valid: {len(val_pred_ae)} samples")
        print(f"      CatBoost valid: {len(val_pred_cb)} samples")

        # Get validation labels
        val_label = ae_dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(val_label, pd.DataFrame):
            val_label = val_label.iloc[:, 0]

        # Learn optimal weights
        learned_weights, learn_info = learn_optimal_weights(
            val_pred_ae, val_pred_cb, val_label,
            method=args.stacking_method,
            use_zscore=True,  # Always zscore for fair comparison
            min_weight=args.min_weight,
            diversity_bonus=args.diversity_bonus
        )

        print(f"\n    Learned Weights:")
        print(f"    " + "-" * 50)
        print(f"    AE-MLP weight:   {learned_weights[0]:>10.4f}")
        print(f"    CatBoost weight: {learned_weights[1]:>10.4f}")
        print(f"    " + "-" * 50)

        # Print additional info from learning
        for k, v in learn_info.items():
            if k != 'method':
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

        # Override ensemble method to use learned weights
        args.ensemble_method = 'zscore_weighted'
        args.ae_weight, args.cb_weight = learned_weights
        print(f"\n    Using zscore_weighted ensemble with learned weights")

    # Ensemble predictions
    step_num = 9 if args.stacking else 8
    print(f"\n[{step_num}] Ensembling predictions ({args.ensemble_method})...")
    weights = (args.ae_weight, args.cb_weight) if args.ensemble_method in ['weighted', 'zscore_weighted'] else None
    pred_ensemble = ensemble_predictions(pred_ae, pred_cb, args.ensemble_method, weights)
    print(f"    Ensemble shape: {len(pred_ensemble)}, Range: [{pred_ensemble.min():.4f}, {pred_ensemble.max():.4f}]")

    # Get labels (use AE-MLP dataset's label as reference)
    step_num += 1
    print(f"\n[{step_num}] Calculating IC metrics...")
    test_label = ae_dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(test_label, pd.DataFrame):
        label = test_label.iloc[:, 0]
    else:
        label = test_label

    # Calculate IC for each model and ensemble
    ae_ic, ae_std, ae_icir, ae_ic_series = compute_ic(pred_ae, label)
    cb_ic, cb_std, cb_icir, cb_ic_series = compute_ic(pred_cb, label)
    ens_ic, ens_std, ens_icir, ens_ic_series = compute_ic(pred_ensemble, label)

    print("\n    +" + "=" * 60 + "+")
    print("    |" + " " * 10 + "Information Coefficient (IC) Comparison" + " " * 10 + "|")
    print("    +" + "=" * 60 + "+")
    print(f"    |  {'Model':<15s} | {'Mean IC':>10s} | {'IC Std':>10s} | {'ICIR':>10s} |")
    print("    +" + "-" * 60 + "+")
    print(f"    |  {'AE-MLP':<15s} | {ae_ic:>10.4f} | {ae_std:>10.4f} | {ae_icir:>10.4f} |")
    print(f"    |  {'CatBoost':<15s} | {cb_ic:>10.4f} | {cb_std:>10.4f} | {cb_icir:>10.4f} |")
    print("    +" + "-" * 60 + "+")
    print(f"    |  {'Ensemble':<15s} | {ens_ic:>10.4f} | {ens_std:>10.4f} | {ens_icir:>10.4f} |")
    print("    +" + "=" * 60 + "+")

    # Calculate improvement
    best_single_ic = max(ae_ic, cb_ic)
    best_single_icir = max(ae_icir, cb_icir)

    if best_single_ic != 0:
        ic_improvement = (ens_ic - best_single_ic) / abs(best_single_ic) * 100
    else:
        ic_improvement = 0

    if best_single_icir != 0:
        icir_improvement = (ens_icir - best_single_icir) / abs(best_single_icir) * 100
    else:
        icir_improvement = 0

    print(f"\n    Ensemble Performance vs Best Single Model:")
    print(f"    IC improvement:   {ic_improvement:>+.2f}%")
    print(f"    ICIR improvement: {icir_improvement:>+.2f}%")

    # Summary
    print("\n" + "=" * 70)
    print("ENSEMBLE ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Prediction Correlation: {overall_corr:.4f} (daily mean: {mean_daily_corr:.4f})")
    if learned_weights:
        print(f"Stacking Weights: AE-MLP={learned_weights[0]:.3f}, CatBoost={learned_weights[1]:.3f}")
    print(f"AE-MLP IC:     {ae_ic:.4f} (ICIR: {ae_icir:.4f})")
    print(f"CatBoost IC:   {cb_ic:.4f} (ICIR: {cb_icir:.4f})")
    print(f"Ensemble IC:   {ens_ic:.4f} (ICIR: {ens_icir:.4f})")
    print("=" * 70)

    # Run backtest if requested
    if args.backtest:
        run_ensemble_backtest(pred_ensemble, args, time_splits)

        print("\n" + "=" * 70)
        print("ENSEMBLE BACKTEST COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    main()
