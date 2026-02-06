"""
Run KAN (Kolmogorov-Arnold Network) Model for Stock Return Prediction

KAN replaces standard MLP linear layers with learnable B-spline activation
functions, providing more expressive function approximation for stock prediction.

Usage:
    # Basic test
    python scripts/models/deep/run_kan.py --stock-pool test --handler alpha158

    # SP500 with default settings
    python scripts/models/deep/run_kan.py --stock-pool sp500 --handler alpha158

    # With TA-Lib features
    python scripts/models/deep/run_kan.py --stock-pool sp500 --handler alpha158-talib

    # With macro features
    python scripts/models/deep/run_kan.py --stock-pool sp500 --handler alpha158-macro

    # IC loss for better ranking
    python scripts/models/deep/run_kan.py --stock-pool sp500 --handler alpha158 --loss-type ic

    # Smaller/faster model
    python scripts/models/deep/run_kan.py --stock-pool sp500 --handler alpha158 --hidden-sizes 128 64

    # With backtest
    python scripts/models/deep/run_kan.py --stock-pool sp500 --handler alpha158 --backtest

    # Load pre-trained model
    python scripts/models/deep/run_kan.py --model-path ./my_models/kan_alpha158_sp500_2d.pt --backtest
"""

import sys
from pathlib import Path

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

from models.deep.kan_model import KANStock


# ============================================================================
# KAN-Specific Arguments
# ============================================================================

def add_kan_args(parser):
    """Add KAN-specific arguments."""
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[256, 128],
                        help='Hidden layer sizes (default: 256 128)')
    parser.add_argument('--grid-size', type=int, default=8,
                        help='B-spline grid segments (default: 8)')
    parser.add_argument('--spline-order', type=int, default=3,
                        help='B-spline order, 3=cubic (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='AdamW weight decay (default: 1e-4)')
    parser.add_argument('--reg-lambda', type=float, default=1e-5,
                        help='KAN regularization weight (default: 1e-5)')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Batch size (default: 2048)')
    parser.add_argument('--n-epochs', type=int, default=100,
                        help='Training epochs (default: 100)')
    parser.add_argument('--early-stop', type=int, default=15,
                        help='Early stopping patience on val IC (default: 15)')
    parser.add_argument('--grid-update-freq', type=int, default=20,
                        help='Update B-spline grids every N epochs (default: 20)')
    parser.add_argument('--loss-type', type=str, default='mse',
                        choices=['mse', 'ic'],
                        help='Loss type: mse or ic (default: mse)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID, -1 for CPU (default: 0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    return parser


# ============================================================================
# Main
# ============================================================================

def main():
    # Parse arguments
    parser = create_argument_parser("KAN", "run_kan.py")
    parser = add_kan_args(parser)
    args = parser.parse_args()

    # Configuration
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = get_time_splits(args.max_train)

    # Print header
    print_training_header("KAN", args, symbols, handler_config, time_splits)

    # Initialize Qlib
    init_qlib(handler_config['use_talib'])
    check_data_availability(time_splits)

    # Create data handler and dataset
    handler = create_data_handler(args, handler_config, symbols, time_splits)
    dataset = create_dataset(handler, time_splits)

    # Analyze data
    train_data, valid_cols, dropped_cols = analyze_features(dataset)
    analyze_label_distribution(dataset)

    # Determine d_feat from actual data
    d_feat = train_data.shape[1]

    # KAN parameter summary
    # Estimate parameter count
    sizes = [d_feat] + args.hidden_sizes + [1]
    est_params = 0
    n_spline = args.grid_size + args.spline_order
    for i in range(len(sizes) - 1):
        # spline_weight + base_weight + spline_scaler
        est_params += sizes[i + 1] * sizes[i] * n_spline  # spline
        est_params += sizes[i + 1] * sizes[i]              # base
        est_params += sizes[i + 1] * sizes[i]              # scaler

    print(f"\n[6] KAN Configuration:")
    print(f"    d_feat: {d_feat}")
    print(f"    hidden_sizes: {args.hidden_sizes}")
    print(f"    architecture: {' -> '.join(str(s) for s in sizes)}")
    print(f"    grid_size: {args.grid_size}, spline_order: {args.spline_order}")
    print(f"    dropout: {args.dropout}")
    print(f"    loss_type: {args.loss_type}")
    print(f"    reg_lambda: {args.reg_lambda}")
    print(f"    batch_size: {args.batch_size}")
    print(f"    n_epochs: {args.n_epochs}, early_stop: {args.early_stop}")
    print(f"    grid_update_freq: {args.grid_update_freq}")
    print(f"    estimated parameters: {est_params:,}")
    print(f"    GPU: {args.gpu}")

    # Model loading helpers
    def load_model(path):
        return KANStock.load(str(path), GPU=args.gpu)

    def get_feature_count(m):
        return m.d_feat

    # Train or load
    if args.model_path:
        model_path = Path(args.model_path)
        print(f"\n[7] Loading pre-trained model from: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = load_model(model_path)
    else:
        print("\n[7] Training KAN model...")

        model = KANStock(
            d_feat=d_feat,
            hidden_sizes=args.hidden_sizes,
            grid_size=args.grid_size,
            spline_order=args.spline_order,
            dropout=args.dropout,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            reg_lambda=args.reg_lambda,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            early_stop=args.early_stop,
            grid_update_freq=args.grid_update_freq,
            loss_type=args.loss_type,
            GPU=args.gpu,
            seed=args.seed,
        )

        model.fit(dataset)

        # Save model
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_SAVE_PATH / f"kan_{args.handler}_{args.stock_pool}_{args.nday}d.pt"
        model.save(str(model_path))

    # Prediction
    print("\n[8] Generating predictions...")
    test_pred = model.predict(dataset, segment="test")

    print(f"    Predictions: {len(test_pred):,} samples")
    print(f"    NaN count: {test_pred.isna().sum()}")
    if not test_pred.isna().all():
        print(f"    Range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
    print_prediction_stats(test_pred)

    # Evaluation
    print("\n[9] Evaluation...")
    evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

    # Backtest
    if args.backtest:
        pred_df = test_pred.to_frame("score")

        run_backtest(
            model_path, dataset, pred_df, args, time_splits,
            model_name="KAN",
            load_model_func=load_model,
            get_feature_count_func=get_feature_count,
        )


if __name__ == "__main__":
    main()
