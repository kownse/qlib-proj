"""
Deep Model Ensemble Backtest

Ensemble AE-MLP (.keras) and TCN (.pt) model predictions for backtesting.

Note: AE-MLP uses alpha158 handler, TCN uses alpha360 handler.
Each model requires its own dataset with the corresponding handler.

Usage:
    python run_deep_ensemble.py --stock-pool sp500
    python run_deep_ensemble.py --ensemble-method weighted --ae-weight 0.6 --tcn-weight 0.4
    python run_deep_ensemble.py --ensemble-method rank_mean --topk 15 --n-drop 3
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
script_dir = Path(__file__).parent.parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.backtest import backtest as qlib_backtest
from qlib.contrib.evaluate import risk_analysis

from utils.talib_ops import TALIB_OPS
from utils.strategy import get_strategy_config
from utils.backtest_utils import plot_backtest_curve, generate_trade_records
from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG,
    PROJECT_ROOT,
    QLIB_DATA_PATH,
    MODEL_SAVE_PATH,
    DEFAULT_TIME_SPLITS,
)
from models.deep.ae_mlp_model import AEMLP


def load_ae_mlp_model(model_path: Path):
    """Load AE-MLP (.keras) model"""
    print(f"    Loading AE-MLP model from: {model_path}")
    model = AEMLP.load(str(model_path))
    return model


def load_tcn_model(model_path: Path):
    """Load TCN (.pt) model"""
    print(f"    Loading TCN model from: {model_path}")
    model = torch.load(str(model_path), weights_only=False)
    return model


def create_data_handler_for_model(handler_name: str, symbols: list, time_splits: dict, nday: int):
    """
    Create DataHandler for a specific handler type

    Parameters
    ----------
    handler_name : str
        Handler type: 'alpha158' or 'alpha360'
    symbols : list
        Stock symbols
    time_splits : dict
        Time split configuration
    nday : int
        Volatility window (prediction horizon)

    Returns
    -------
    DataHandler
    """
    handler_config = HANDLER_CONFIG[handler_name]
    HandlerClass = handler_config['class']

    handler_kwargs = {
        'volatility_window': nday,
        'instruments': symbols,
        'start_time': time_splits['train_start'],
        'end_time': time_splits['test_end'],
        'fit_start_time': time_splits['train_start'],
        'fit_end_time': time_splits['train_end'],
        'infer_processors': [],
    }

    return HandlerClass(**handler_kwargs), handler_config


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


def predict_with_ae_mlp(model: AEMLP, dataset: DatasetH) -> pd.Series:
    """Generate predictions with AE-MLP model"""
    pred = model.predict(dataset, segment="test")
    pred.name = 'score'
    return pred


def predict_with_tcn(model, dataset: DatasetH) -> pd.Series:
    """
    Generate predictions with TCN model

    This function replicates the prediction logic from run_tcn.py exactly,
    including the normalization for data with NaN values.
    """
    # Get test data - NOTE: no data_key parameter, same as run_tcn.py
    test_data = dataset.prepare("test", col_set="feature")

    test_nan_count = test_data.isna().sum().sum()
    test_nan_pct = test_nan_count / test_data.size * 100

    # Fill NaN first, then check min/max
    test_data_clean = test_data.fillna(0)
    test_abs_max = np.abs(test_data_clean.values).max()

    # Check if normalization is needed (same logic as run_tcn.py)
    need_normalize = test_nan_count > 0 or test_abs_max > 1e6

    if need_normalize:
        if test_abs_max > 1e6:
            print(f"    WARNING: Test data has extreme values (max abs: {test_abs_max:.2e})")

        # Normalize test data (same as run_tcn.py)
        test_data_normalized = test_data.fillna(0)
        for col in test_data_normalized.columns:
            col_mean = test_data_normalized[col].mean()
            col_std = test_data_normalized[col].std()
            if col_std > 0:
                lower = col_mean - 3 * col_std
                upper = col_mean + 3 * col_std
                test_data_normalized[col] = test_data_normalized[col].clip(lower, upper)
                test_data_normalized[col] = (test_data_normalized[col] - col_mean) / col_std
        test_data_normalized = test_data_normalized.replace([np.inf, -np.inf], 0).fillna(0)

        # Generate predictions with normalized data (manual batch processing)
        model.tcn_model.eval()
        test_values = test_data_normalized.values
        preds = []
        batch_size = getattr(model, 'batch_size', 2000)

        with torch.no_grad():
            for i in range(0, len(test_values), batch_size):
                batch = test_values[i:i+batch_size]
                batch_tensor = torch.from_numpy(batch).float().to(model.device)
                batch_pred = model.tcn_model(batch_tensor)
                preds.append(batch_pred.cpu().numpy())

        pred_values = np.concatenate(preds).flatten()
        pred = pd.Series(pred_values, index=test_data.index, name='score')
    else:
        # Normal prediction flow
        pred = model.predict(dataset)
        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[:, 0]
        pred.name = 'score'

    return pred


def ensemble_predictions(pred1: pd.Series, pred2: pd.Series,
                        method: str = 'mean', weights: tuple = None) -> pd.Series:
    """
    Ensemble two model predictions

    Parameters
    ----------
    pred1 : pd.Series
        First model predictions
    pred2 : pd.Series
        Second model predictions
    method : str
        Ensemble method: 'mean', 'weighted', 'rank_mean'
    weights : tuple, optional
        (weight1, weight2) for weighted ensemble

    Returns
    -------
    pd.Series
        Ensembled predictions
    """
    # Align indices
    common_idx = pred1.index.intersection(pred2.index)
    pred1_aligned = pred1.loc[common_idx]
    pred2_aligned = pred2.loc[common_idx]

    if method == 'mean':
        ensemble_pred = (pred1_aligned + pred2_aligned) / 2
    elif method == 'weighted':
        if weights is None:
            weights = (0.5, 0.5)
        ensemble_pred = pred1_aligned * weights[0] + pred2_aligned * weights[1]
    elif method == 'rank_mean':
        # Convert to rank percentiles, then average
        rank1 = pred1_aligned.groupby(level='datetime').rank(pct=True)
        rank2 = pred2_aligned.groupby(level='datetime').rank(pct=True)
        ensemble_pred = (rank1 + rank2) / 2
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    ensemble_pred.name = 'score'
    return ensemble_pred


def calculate_ic(pred: pd.Series, label: pd.Series) -> tuple:
    """
    Calculate IC and ICIR

    Returns
    -------
    tuple
        (ic_series, mean_ic, ic_std, icir)
    """
    # Align indices
    common_idx = pred.index.intersection(label.index)
    pred_aligned = pred.loc[common_idx]
    label_aligned = label.loc[common_idx]

    # Remove NaN
    valid_idx = ~(pred_aligned.isna() | label_aligned.isna())
    pred_clean = pred_aligned[valid_idx]
    label_clean = label_aligned[valid_idx]

    # Calculate IC by date
    ic = pred_clean.groupby(level="datetime").apply(
        lambda x: x.corr(label_clean.loc[x.index]) if len(x) > 1 else np.nan
    )
    ic = ic.dropna()

    mean_ic = ic.mean()
    ic_std = ic.std()
    icir = mean_ic / ic_std if ic_std > 0 else 0

    return ic, mean_ic, ic_std, icir


def plot_ic_comparison(ic_dict: dict, output_path: Path):
    """
    Plot IC comparison chart

    Parameters
    ----------
    ic_dict : dict
        {model_name: ic_series}
    output_path : Path
        Output directory
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Daily IC
    ax1 = axes[0]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    for i, (name, ic) in enumerate(ic_dict.items()):
        ax1.plot(ic.index, ic.values, label=name, alpha=0.7, color=colors[i % len(colors)])
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Daily IC Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('IC')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative IC
    ax2 = axes[1]
    for i, (name, ic) in enumerate(ic_dict.items()):
        cumsum_ic = ic.cumsum()
        ax2.plot(cumsum_ic.index, cumsum_ic.values, label=name, alpha=0.8,
                color=colors[i % len(colors)], linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Cumulative IC Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative IC')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / 'deep_ensemble_ic_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    IC comparison chart saved to: {fig_path}")


def run_ensemble_backtest(pred: pd.Series, args, time_splits: dict, handler_name: str = "ensemble"):
    """
    Run backtest with ensembled predictions
    """
    print("\n" + "=" * 70)
    print("BACKTEST with TopkDropoutStrategy (Deep Ensemble)")
    print("=" * 70)

    # Convert to DataFrame
    pred_df = pred.to_frame("score")

    print(f"\n[BT-1] Configuring backtest...")
    print(f"    Topk: {args.topk}")
    print(f"    N_drop: {args.n_drop}")
    print(f"    Account: ${args.account:,.0f}")
    print(f"    Rebalance Freq: every {args.rebalance_freq} day(s)")
    print(f"    Period: {time_splits['test_start']} to {time_splits['test_end']}")

    # Configure strategy
    strategy_config = get_strategy_config(pred_df, args.topk, args.n_drop, args.rebalance_freq)

    # Configure executor
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    }

    # Configure backtest parameters
    pool_symbols = STOCK_POOLS[args.stock_pool]

    backtest_config = {
        "start_time": time_splits['test_start'],
        "end_time": time_splits['test_end'],
        "account": args.account,
        "benchmark": pool_symbols,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": None,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0005,
            "min_cost": 1,
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

        # Analyze results
        print(f"\n[BT-3] Analyzing results...")

        for freq, (report_df, positions) in portfolio_metric_dict.items():
            _analyze_backtest_results(report_df, positions, freq, args, time_splits, handler_name)

        # Output trading indicators
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
                              args, time_splits: dict, handler_name: str = "ensemble"):
    """Analyze and report backtest results"""
    print(f"\n    Frequency: {freq}")
    print(f"    Report shape: {report_df.shape}")
    print(f"    Date range: {report_df.index.min()} to {report_df.index.max()}")

    # Calculate key metrics
    total_return = (report_df["return"] + 1).prod() - 1

    # Check for benchmark
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

    # Daily returns statistics
    print(f"\n    Daily Returns Statistics:")
    print(f"    " + "-" * 50)
    print(f"    Mean Daily Return:   {report_df['return'].mean():>10.4%}")
    print(f"    Std Daily Return:    {report_df['return'].std():>10.4%}")
    print(f"    Max Daily Return:    {report_df['return'].max():>10.4%}")
    print(f"    Min Daily Return:    {report_df['return'].min():>10.4%}")
    print(f"    Total Trading Days:  {len(report_df):>10d}")
    print(f"    " + "-" * 50)

    # Save report
    output_path = PROJECT_ROOT / "outputs" / f"deep_ensemble_backtest_report_{freq}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path)
    print(f"\n    Report saved to: {output_path}")

    # Plot equity curve
    print(f"\n[BT-4] Generating performance chart...")
    if not hasattr(args, 'handler'):
        args.handler = handler_name
    plot_backtest_curve(report_df, args, freq, PROJECT_ROOT, model_name="DeepEnsemble")

    # Generate trade records
    print(f"\n[BT-5] Generating trade records...")
    generate_trade_records(positions, args, freq, PROJECT_ROOT, model_name="deep_ensemble")


def main():
    parser = argparse.ArgumentParser(
        description='Deep Model Ensemble Backtest (AE-MLP + TCN)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_deep_ensemble.py
  python run_deep_ensemble.py --ensemble-method rank_mean
  python run_deep_ensemble.py --ae-weight 0.6 --tcn-weight 0.4
  python run_deep_ensemble.py --topk 15 --n-drop 3 --rebalance-freq 5
"""
    )

    # Model path arguments
    parser.add_argument('--ae-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'ae_mlp_alpha158_sp500_5d.keras'),
                        help='AE-MLP model path (.keras)')
    parser.add_argument('--tcn-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'tcn_alpha360_sp500_5d.pt'),
                        help='TCN model path (.pt)')

    # Ensemble parameters
    parser.add_argument('--ensemble-method', type=str, default='mean',
                        choices=['mean', 'weighted', 'rank_mean'],
                        help='Ensemble method (default: mean)')
    parser.add_argument('--ae-weight', type=float, default=0.5,
                        help='AE-MLP weight for weighted ensemble (default: 0.5)')
    parser.add_argument('--tcn-weight', type=float, default=0.5,
                        help='TCN weight for weighted ensemble (default: 0.5)')

    # Volatility window (should match model training)
    parser.add_argument('--nday', type=int, default=5,
                        help='Volatility window / prediction horizon (default: 5)')

    # Backtest parameters
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'],
                        help='Stock pool (default: sp500)')
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of stocks to hold (default: 10)')
    parser.add_argument('--n-drop', type=int, default=2,
                        help='Number of stocks to drop/replace each day (default: 2)')
    parser.add_argument('--account', type=float, default=1000000,
                        help='Initial account value (default: 1000000)')
    parser.add_argument('--rebalance-freq', type=int, default=1,
                        help='Rebalance frequency in days (default: 1)')

    args = parser.parse_args()

    # Use default time splits
    time_splits = DEFAULT_TIME_SPLITS.copy()

    print("=" * 70)
    print("Deep Model Ensemble Backtest (AE-MLP + TCN)")
    print("=" * 70)
    print(f"AE-MLP Model: {args.ae_model}")
    print(f"TCN Model: {args.tcn_model}")
    print(f"Ensemble Method: {args.ensemble_method}")
    if args.ensemble_method == 'weighted':
        print(f"Weights: AE-MLP={args.ae_weight}, TCN={args.tcn_weight}")
    print(f"Stock Pool: {args.stock_pool}")
    print(f"Prediction Horizon: {args.nday} days")
    print(f"Time Splits:")
    print(f"  Train: {time_splits['train_start']} to {time_splits['train_end']}")
    print(f"  Valid: {time_splits['valid_start']} to {time_splits['valid_end']}")
    print(f"  Test:  {time_splits['test_start']} to {time_splits['test_end']}")
    print("=" * 70)

    # Check model files exist
    ae_path = Path(args.ae_model)
    tcn_path = Path(args.tcn_model)

    if not ae_path.exists():
        raise FileNotFoundError(f"AE-MLP model not found: {ae_path}")
    if not tcn_path.exists():
        raise FileNotFoundError(f"TCN model not found: {tcn_path}")

    # Initialize Qlib (without TA-Lib for alpha158/alpha360)
    print("\n[1] Initializing Qlib...")
    qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US)
    print("    Qlib initialized")

    # Get symbols
    symbols = STOCK_POOLS[args.stock_pool]
    print(f"\n[2] Loading stock pool: {args.stock_pool} ({len(symbols)} stocks)")

    # Create datasets for each model (different handlers)
    print("\n[3] Creating datasets...")

    # AE-MLP uses alpha158
    print("    Creating alpha158 dataset for AE-MLP...")
    handler_158, _ = create_data_handler_for_model('alpha158', symbols, time_splits, args.nday)
    dataset_158 = create_dataset(handler_158, time_splits)

    # TCN uses alpha360
    print("    Creating alpha360 dataset for TCN...")
    handler_360, _ = create_data_handler_for_model('alpha360', symbols, time_splits, args.nday)
    dataset_360 = create_dataset(handler_360, time_splits)

    # Load models
    print("\n[4] Loading models...")
    ae_model = load_ae_mlp_model(ae_path)
    tcn_model = load_tcn_model(tcn_path)

    # Generate predictions
    print("\n[5] Generating predictions...")

    # AE-MLP predictions
    print("    Generating AE-MLP predictions...")
    pred_ae = predict_with_ae_mlp(ae_model, dataset_158)
    print(f"    AE-MLP predictions: {len(pred_ae)} samples, "
          f"range=[{pred_ae.min():.4f}, {pred_ae.max():.4f}]")

    # TCN predictions
    print("    Generating TCN predictions...")
    pred_tcn = predict_with_tcn(tcn_model, dataset_360)
    print(f"    TCN predictions: {len(pred_tcn)} samples, "
          f"range=[{pred_tcn.min():.4f}, {pred_tcn.max():.4f}]")

    # Ensemble predictions
    print(f"\n[6] Ensembling predictions ({args.ensemble_method})...")
    weights = (args.ae_weight, args.tcn_weight) if args.ensemble_method == 'weighted' else None
    pred_ensemble = ensemble_predictions(pred_ae, pred_tcn, args.ensemble_method, weights)
    print(f"    Ensemble predictions: {len(pred_ensemble)} samples, "
          f"range=[{pred_ensemble.min():.4f}, {pred_ensemble.max():.4f}]")

    # Calculate IC metrics
    print("\n[7] Calculating IC metrics...")

    # Get labels from both datasets
    # Labels should be identical since they're computed from the same underlying data
    test_label_158 = dataset_158.prepare("test", col_set="label")
    test_label_360 = dataset_360.prepare("test", col_set="label")
    label_158 = test_label_158['LABEL0']
    label_360 = test_label_360['LABEL0']

    # Calculate IC for each model with its corresponding label
    # AE-MLP uses alpha158 dataset
    ic_ae, ae_ic, ae_std, ae_icir = calculate_ic(pred_ae, label_158)

    # TCN uses alpha360 dataset
    ic_tcn, tcn_ic, tcn_std, tcn_icir = calculate_ic(pred_tcn, label_360)

    # Ensemble uses alpha360 label (TCN's label covers all ensemble samples)
    ic_ens, ens_ic, ens_std, ens_icir = calculate_ic(pred_ensemble, label_360)

    print("\n    +" + "=" * 64 + "+")
    print("    |" + " " * 12 + "Information Coefficient (IC) Comparison" + " " * 12 + "|")
    print("    +" + "=" * 64 + "+")
    print(f"    |  {'Model':<12s} | {'Mean IC':>10s} | {'IC Std':>10s} | {'ICIR':>10s}  |")
    print("    +" + "-" * 64 + "+")
    print(f"    |  {'AE-MLP':<12s} | {ae_ic:>10.4f} | {ae_std:>10.4f} | {ae_icir:>10.4f}  |")
    print(f"    |  {'TCN':<12s} | {tcn_ic:>10.4f} | {tcn_std:>10.4f} | {tcn_icir:>10.4f}  |")
    print(f"    |  {'Ensemble':<12s} | {ens_ic:>10.4f} | {ens_std:>10.4f} | {ens_icir:>10.4f}  |")
    print("    +" + "=" * 64 + "+")

    # IC improvement
    best_single_ic = max(ae_ic, tcn_ic)
    best_single_icir = max(ae_icir, tcn_icir)
    ic_improvement = (ens_ic - best_single_ic) / abs(best_single_ic) * 100 if best_single_ic != 0 else 0
    icir_improvement = (ens_icir - best_single_icir) / abs(best_single_icir) * 100 if best_single_icir != 0 else 0

    print(f"\n    IC improvement over best single model:   {ic_improvement:>+.2f}%")
    print(f"    ICIR improvement over best single model: {icir_improvement:>+.2f}%")

    # Plot IC comparison
    print("\n[8] Plotting IC comparison...")
    output_path = PROJECT_ROOT / "outputs"
    ic_dict = {
        'AE-MLP': ic_ae,
        'TCN': ic_tcn,
        'Ensemble': ic_ens,
    }
    plot_ic_comparison(ic_dict, output_path)

    # Run backtest
    print("\n[9] Running backtest...")
    run_ensemble_backtest(pred_ensemble, args, time_splits, "deep_ensemble")

    print("\n" + "=" * 70)
    print("Deep Ensemble Backtest Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
