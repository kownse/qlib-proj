"""
Multi-Seed AE-MLP Ensemble Training and Evaluation

Train multiple AE-MLP models with different random seeds, then average their
predictions to get a more robust final prediction. This reduces variance from
random initialization and typically improves model performance.

Usage:
    python scripts/models/deep/run_ae_mlp_ensemble.py \
        --params-file outputs/hyperopt_cv/ae_mlp_cv_best_params_20260116_173601_best.json \
        --stock-pool sp500 --handler alpha158 --backtest

    python scripts/models/deep/run_ae_mlp_ensemble.py \
        --params-file outputs/hyperopt_cv/ae_mlp_cv_best_params_20260116_173601_best.json \
        --seeds 42,123,456,789,1000 --n-epochs 50 --backtest
"""

import sys
import json
from pathlib import Path
from datetime import datetime

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import numpy as np
import pandas as pd

from utils.utils import evaluate_model
from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG, PROJECT_ROOT, MODEL_SAVE_PATH,
    create_argument_parser,
    get_time_splits,
    print_training_header,
    init_qlib,
    check_data_availability,
    create_data_handler,
    create_dataset,
    analyze_features,
    analyze_label_distribution,
    print_prediction_stats,
    run_backtest,
)

from models.deep.ae_mlp_model import AEMLP


# ============================================================================
# Default Parameters
# ============================================================================

DEFAULT_SEEDS = [42, 123, 456, 789, 1000]

DEFAULT_AE_MLP_PARAMS = {
    'hidden_units': None,
    'dropout_rates': None,
    'lr': 0.001,
    'batch_size': 4096,
    'loss_weights': {'decoder': 0.1, 'ae_action': 0.1, 'action': 1.0},
}


def load_params_from_file(params_file: str) -> dict:
    """
    Load AE-MLP params from JSON file (hyperopt CV output)

    Parameters
    ----------
    params_file : str
        Path to params JSON file

    Returns
    -------
    dict
        AE-MLP parameters
    """
    with open(params_file, 'r') as f:
        data = json.load(f)

    # Support two formats:
    # 1. hyperopt CV format: {"params": {...}, "cv_results": {...}}
    # 2. Direct params format: {"hidden_units": [...], "lr": ...}
    if 'params' in data:
        params = data['params']
        print(f"    Loaded params from CV hyperopt file")
        if 'cv_results' in data:
            cv = data['cv_results']
            print(f"    CV Mean IC: {cv.get('mean_ic', 'N/A'):.4f} (±{cv.get('std_ic', 'N/A'):.4f})")
    else:
        params = data

    # Build final params
    final_params = DEFAULT_AE_MLP_PARAMS.copy()

    if 'hidden_units' in params:
        final_params['hidden_units'] = params['hidden_units']
    if 'dropout_rates' in params:
        final_params['dropout_rates'] = params['dropout_rates']
    if 'lr' in params:
        final_params['lr'] = params['lr']
    if 'batch_size' in params:
        final_params['batch_size'] = params['batch_size']
    if 'loss_weights' in params:
        final_params['loss_weights'] = params['loss_weights']

    return final_params


def add_ensemble_args(parser):
    """Add ensemble-specific arguments"""
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated list of random seeds (default: 42,123,456,789,1000)')
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='Number of random seeds to use if --seeds not specified (default: 5)')
    parser.add_argument('--hidden-units', type=str, default=None,
                        help='Hidden units per layer, comma-separated')
    parser.add_argument('--dropout-rates', type=str, default=None,
                        help='Dropout rates per layer, comma-separated')
    parser.add_argument('--n-epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--early-stop', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--params-file', type=str, default=None,
                        help='Path to JSON file with AE-MLP params (from hyperopt CV search)')
    parser.add_argument('--save-individual', action='store_true',
                        help='Save individual models (not just ensemble prediction)')
    return parser


def parse_list_arg(arg_str, dtype=float):
    """Parse comma-separated list argument"""
    if arg_str is None:
        return None
    return [dtype(x.strip()) for x in arg_str.split(',')]


def train_single_model(dataset, model_params: dict, seed: int, model_idx: int, total_models: int):
    """
    Train a single AE-MLP model with given seed

    Parameters
    ----------
    dataset : DatasetH
        Qlib dataset
    model_params : dict
        Model parameters
    seed : int
        Random seed
    model_idx : int
        Model index (1-based)
    total_models : int
        Total number of models

    Returns
    -------
    AEMLP
        Trained model
    """
    print(f"\n    {'='*60}")
    print(f"    Training Model {model_idx}/{total_models} (seed={seed})")
    print(f"    {'='*60}")

    model = AEMLP(
        num_columns=model_params['num_columns'],
        hidden_units=model_params['hidden_units'],
        dropout_rates=model_params['dropout_rates'],
        lr=model_params['lr'],
        n_epochs=model_params['n_epochs'],
        batch_size=model_params['batch_size'],
        early_stop=model_params['early_stop'],
        loss_weights=model_params['loss_weights'],
        GPU=model_params['gpu'],
        seed=seed,
    )

    model.fit(dataset)
    return model


def ensemble_predict(models: list, dataset, segment: str = "test") -> pd.Series:
    """
    Generate ensemble predictions by averaging individual model predictions

    Parameters
    ----------
    models : list
        List of trained AEMLP models
    dataset : DatasetH
        Qlib dataset
    segment : str
        Data segment to predict on

    Returns
    -------
    pd.Series
        Averaged predictions
    """
    print(f"\n    Generating ensemble predictions from {len(models)} models...")

    all_preds = []
    for i, model in enumerate(models):
        pred = model.predict(dataset, segment=segment)
        all_preds.append(pred)
        print(f"      Model {i+1}: pred range [{pred.min():.4f}, {pred.max():.4f}]")

    # Stack predictions and compute mean
    pred_df = pd.concat(all_preds, axis=1)
    ensemble_pred = pred_df.mean(axis=1)
    ensemble_pred.name = 'score'

    print(f"\n    Ensemble prediction range: [{ensemble_pred.min():.4f}, {ensemble_pred.max():.4f}]")
    print(f"    Ensemble prediction std (model variance): {pred_df.std(axis=1).mean():.6f}")

    return ensemble_pred


def compute_ic_metrics(pred: pd.Series, dataset, segment: str = "test"):
    """
    Compute IC metrics for predictions

    Parameters
    ----------
    pred : pd.Series
        Predictions
    dataset : DatasetH
        Qlib dataset
    segment : str
        Data segment

    Returns
    -------
    dict
        IC metrics
    """
    label_df = dataset.prepare(segment, col_set="label")
    pred_aligned = pred.reindex(label_df.index)

    # Remove NaN values
    valid_idx = ~(pred_aligned.isna() | label_df["LABEL0"].isna())
    pred_clean = pred_aligned[valid_idx]
    label_clean = label_df["LABEL0"][valid_idx]

    # Compute daily IC
    ic = pred_clean.groupby(level="datetime").apply(
        lambda x: x.corr(label_clean.loc[x.index]) if len(x) > 1 else np.nan
    )
    ic = ic.dropna()

    return {
        'mean_ic': ic.mean(),
        'std_ic': ic.std(),
        'icir': ic.mean() / ic.std() if ic.std() > 0 else 0,
        'ic_series': ic,
    }


def main():
    # Parse arguments
    parser = create_argument_parser("AE-MLP Ensemble", "run_ae_mlp_ensemble.py")
    parser = add_ensemble_args(parser)
    args = parser.parse_args()

    # Check for params file
    if args.params_file is None:
        print("ERROR: --params-file is required")
        print("Example: --params-file outputs/hyperopt_cv/ae_mlp_cv_best_params_20260116_173601_best.json")
        return

    # Load params from file
    print(f"\n[*] Loading AE-MLP params from: {args.params_file}")
    file_params = load_params_from_file(args.params_file)

    # Parse seeds
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        # Generate default seeds
        seeds = DEFAULT_SEEDS[:args.n_seeds]

    print(f"\n[*] Ensemble Configuration:")
    print(f"    Seeds: {seeds}")
    print(f"    Number of models: {len(seeds)}")

    # Parse list arguments (command line overrides file)
    hidden_units = parse_list_arg(args.hidden_units, int)
    dropout_rates = parse_list_arg(args.dropout_rates, float)

    if hidden_units is None and file_params.get('hidden_units'):
        hidden_units = file_params['hidden_units']
    if dropout_rates is None and file_params.get('dropout_rates'):
        dropout_rates = file_params['dropout_rates']

    # Merge params: CLI > file > default
    lr = args.lr if args.lr is not None else file_params['lr']
    batch_size = args.batch_size if args.batch_size is not None else file_params['batch_size']
    loss_weights = file_params.get('loss_weights', DEFAULT_AE_MLP_PARAMS['loss_weights'])

    # Get configuration
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = get_time_splits(args.max_train)

    # Print header
    print_training_header("AE-MLP Ensemble", args, symbols, handler_config, time_splits)
    print(f"Params File: {args.params_file}")
    print(f"Seeds: {seeds}")

    # Initialize Qlib and prepare data
    init_qlib(handler_config['use_talib'])
    check_data_availability(time_splits)
    handler = create_data_handler(args, handler_config, symbols, time_splits)
    dataset = create_dataset(handler, time_splits)
    train_data, valid_cols, dropped_cols = analyze_features(dataset)
    analyze_label_distribution(dataset)

    # Get actual feature count
    actual_train_data = dataset.prepare("train", col_set="feature")
    total_features = actual_train_data.shape[1]
    print(f"\n    Actual training data shape: {actual_train_data.shape}")

    print(f"\n[6] Model Configuration:")
    print(f"    Total features: {total_features}")
    print(f"    Hidden units: {hidden_units}")
    print(f"    Dropout rates: {dropout_rates}")
    print(f"    Learning rate: {lr}")
    print(f"    Batch size: {batch_size}")
    print(f"    Epochs: {args.n_epochs}")
    print(f"    Early stop: {args.early_stop}")
    print(f"    GPU: {args.gpu}")
    print(f"    Loss weights: decoder={loss_weights['decoder']}, ae={loss_weights['ae_action']}, main={loss_weights['action']}")

    # Prepare model parameters
    model_params = {
        'num_columns': total_features,
        'hidden_units': hidden_units,
        'dropout_rates': dropout_rates,
        'lr': lr,
        'n_epochs': args.n_epochs,
        'batch_size': batch_size,
        'early_stop': args.early_stop,
        'loss_weights': loss_weights,
        'gpu': args.gpu,
    }

    # ========================================================================
    # Train multiple models with different seeds
    # ========================================================================
    print(f"\n[7] Training {len(seeds)} AE-MLP models with different seeds...")

    models = []
    individual_predictions = []
    individual_ics = []

    for i, seed in enumerate(seeds):
        # Train model
        model = train_single_model(dataset, model_params, seed, i + 1, len(seeds))
        models.append(model)

        # Get individual prediction
        pred = model.predict(dataset, segment="test")
        individual_predictions.append(pred)

        # Compute individual IC
        ic_metrics = compute_ic_metrics(pred, dataset, "test")
        individual_ics.append({
            'seed': seed,
            'mean_ic': ic_metrics['mean_ic'],
            'icir': ic_metrics['icir'],
        })
        print(f"      Individual IC: {ic_metrics['mean_ic']:.4f}, ICIR: {ic_metrics['icir']:.4f}")

        # Save individual model if requested
        if args.save_individual:
            MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
            model_path = MODEL_SAVE_PATH / f"ae_mlp_ensemble_{args.handler}_{args.stock_pool}_{args.nday}d_seed{seed}.keras"
            model.save(str(model_path))

    # ========================================================================
    # Print individual model performance
    # ========================================================================
    print(f"\n[8] Individual Model Performance:")
    print(f"    {'Seed':<10} {'Mean IC':<12} {'ICIR':<12}")
    print(f"    {'-'*34}")
    for ic_info in individual_ics:
        print(f"    {ic_info['seed']:<10} {ic_info['mean_ic']:<12.4f} {ic_info['icir']:<12.4f}")

    mean_individual_ic = np.mean([ic['mean_ic'] for ic in individual_ics])
    std_individual_ic = np.std([ic['mean_ic'] for ic in individual_ics])
    print(f"    {'-'*34}")
    print(f"    {'Average':<10} {mean_individual_ic:<12.4f} (std: {std_individual_ic:.4f})")

    # ========================================================================
    # Generate ensemble predictions
    # ========================================================================
    print(f"\n[9] Generating Ensemble Predictions...")
    ensemble_pred = ensemble_predict(models, dataset, segment="test")
    print_prediction_stats(ensemble_pred)

    # ========================================================================
    # Evaluate ensemble
    # ========================================================================
    print("\n[10] Ensemble Evaluation...")
    evaluate_model(dataset, ensemble_pred, PROJECT_ROOT, args.nday)

    # Compare ensemble vs individual
    ensemble_ic = compute_ic_metrics(ensemble_pred, dataset, "test")

    print(f"\n    ╔════════════════════════════════════════════════════════╗")
    print(f"    ║  Ensemble vs Individual Comparison                     ║")
    print(f"    ╠════════════════════════════════════════════════════════╣")
    print(f"    ║  Individual Mean IC:  {mean_individual_ic:>8.4f} (std: {std_individual_ic:.4f})     ║")
    print(f"    ║  Ensemble Mean IC:    {ensemble_ic['mean_ic']:>8.4f}                    ║")
    print(f"    ║  Improvement:         {(ensemble_ic['mean_ic'] - mean_individual_ic):>8.4f} ({(ensemble_ic['mean_ic'] - mean_individual_ic) / mean_individual_ic * 100:+.1f}%)             ║")
    print(f"    ║  Ensemble ICIR:       {ensemble_ic['icir']:>8.4f}                    ║")
    print(f"    ╚════════════════════════════════════════════════════════╝")

    # ========================================================================
    # Save ensemble prediction
    # ========================================================================
    print("\n[11] Saving Ensemble Results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / "ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save ensemble prediction
    pred_path = output_dir / f"ae_mlp_ensemble_pred_{args.handler}_{args.stock_pool}_{timestamp}.parquet"
    ensemble_pred.to_frame("score").to_parquet(pred_path)
    print(f"    Ensemble predictions saved to: {pred_path}")

    # Save results summary
    results = {
        'params_file': str(args.params_file),
        'seeds': seeds,
        'handler': args.handler,
        'stock_pool': args.stock_pool,
        'n_epochs': args.n_epochs,
        'individual_results': individual_ics,
        'individual_mean_ic': mean_individual_ic,
        'individual_std_ic': std_individual_ic,
        'ensemble_mean_ic': ensemble_ic['mean_ic'],
        'ensemble_std_ic': ensemble_ic['std_ic'],
        'ensemble_icir': ensemble_ic['icir'],
        'ic_improvement': ensemble_ic['mean_ic'] - mean_individual_ic,
        'ic_improvement_pct': (ensemble_ic['mean_ic'] - mean_individual_ic) / mean_individual_ic * 100,
    }

    results_path = output_dir / f"ae_mlp_ensemble_results_{args.handler}_{args.stock_pool}_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"    Results summary saved to: {results_path}")

    # ========================================================================
    # Backtest
    # ========================================================================
    if args.backtest:
        print("\n[12] Running Backtest...")

        # Create a dummy model path for backtest (we'll pass pred directly)
        # The backtest function expects a model path but we can work around it
        pred_df = ensemble_pred.to_frame("score")

        # Save a dummy model for backtest compatibility
        dummy_model_path = MODEL_SAVE_PATH / f"ae_mlp_ensemble_{args.handler}_{args.stock_pool}_{args.nday}d_{timestamp}.keras"
        models[0].save(str(dummy_model_path))  # Save first model as reference

        def load_model(path):
            return AEMLP.load(str(path))

        def get_feature_count(m):
            return m.num_columns

        run_backtest(
            dummy_model_path, dataset, pred_df, args, time_splits,
            model_name="AE-MLP Ensemble",
            load_model_func=load_model,
            get_feature_count_func=get_feature_count
        )

    print("\n" + "=" * 70)
    print("AE-MLP Ensemble Training and Evaluation Completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
