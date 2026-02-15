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
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import qlib
from qlib.constant import REG_US

from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG,
    PROJECT_ROOT,
    QLIB_DATA_PATH,
    MODEL_SAVE_PATH,
    DEFAULT_TIME_SPLITS,
)
from models.common.ensemble import (
    load_ae_mlp_model,
    create_ensemble_dataset,
    predict_with_ae_mlp,
    ensemble_predictions,
    compute_ic,
    run_ensemble_backtest,
)


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


def predict_with_tcn(model, dataset) -> pd.Series:
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
    dataset_158 = create_ensemble_dataset(handler_158, time_splits)

    # TCN uses alpha360
    print("    Creating alpha360 dataset for TCN...")
    handler_360, _ = create_data_handler_for_model('alpha360', symbols, time_splits, args.nday)
    dataset_360 = create_ensemble_dataset(handler_360, time_splits)

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
    weights = {'AE-MLP': args.ae_weight, 'TCN': args.tcn_weight} if args.ensemble_method == 'weighted' else None
    pred_ensemble = ensemble_predictions(
        {'AE-MLP': pred_ae, 'TCN': pred_tcn},
        method=args.ensemble_method,
        weights=weights,
    )
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
    # compute_ic returns (mean_ic, ic_std, icir, ic_by_date)
    # AE-MLP uses alpha158 dataset
    ae_ic, ae_std, ae_icir, ic_ae = compute_ic(pred_ae, label_158)

    # TCN uses alpha360 dataset
    tcn_ic, tcn_std, tcn_icir, ic_tcn = compute_ic(pred_tcn, label_360)

    # Ensemble uses alpha360 label (TCN's label covers all ensemble samples)
    ens_ic, ens_std, ens_icir, ic_ens = compute_ic(pred_ensemble, label_360)

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
    run_ensemble_backtest(pred_ensemble, args, time_splits, model_name="DeepEnsemble")

    print("\n" + "=" * 70)
    print("Deep Ensemble Backtest Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
