"""
Run TKAN (Temporal Kolmogorov-Arnold Network) Model for Stock Return Prediction

TKAN combines LSTM-style gating with KAN (learnable B-spline) sub-layers,
replacing the output gate with multi-scale learnable activation functions.

Works with both flat handlers (alpha158-talib-macro) and sequential handlers (alpha360).

Usage:
    # Basic test with flat handler
    python scripts/models/deep/run_tkan.py --stock-pool test --handler alpha158

    # SP500 with macro features (flat, seq_len=1)
    python scripts/models/deep/run_tkan.py --stock-pool sp500 --handler alpha158-talib-macro

    # With sequential handler (alpha360, 6 features x 60 timesteps)
    python scripts/models/deep/run_tkan.py --stock-pool sp500 --handler alpha360 --d-feat 6

    # With macro + sequential
    python scripts/models/deep/run_tkan.py --stock-pool sp500 --handler alpha360-macro --d-feat 29

    # IC loss for ranking
    python scripts/models/deep/run_tkan.py --stock-pool sp500 --handler alpha158-talib-macro --loss-type ic

    # More KAN sub-layers
    python scripts/models/deep/run_tkan.py --stock-pool sp500 --handler alpha158-talib-macro --num-sub-layers 3

    # With backtest
    python scripts/models/deep/run_tkan.py --stock-pool sp500 --handler alpha158-talib-macro --backtest

    # Load pre-trained model
    python scripts/models/deep/run_tkan.py --model-path ./my_models/tkan_alpha158-talib-macro_sp500_2d.pt --backtest
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

from models.deep.tkan_model import TKANStock


from models.common.ts_model_utils import resolve_d_feat_and_seq_len


# ============================================================================
# TKAN-Specific Arguments
# ============================================================================

def add_tkan_args(parser):
    """Add TKAN-specific arguments."""
    parser.add_argument('--d-feat', type=int, default=0,
                        help='Features per timestep (0=auto-detect from handler)')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='TKAN hidden units (default: 64)')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of stacked TKAN layers (default: 2)')
    parser.add_argument('--num-sub-layers', type=int, default=2,
                        help='Number of KAN sub-layers per cell (default: 2)')
    parser.add_argument('--sub-kan-dim', type=int, default=0,
                        help='KAN sub-layer dim (0=auto, default: 0)')
    parser.add_argument('--sub-type', type=str, default='kan',
                        choices=['kan', 'relu', 'mixed'],
                        help='Sub-layer type: kan, relu, or mixed (default: kan)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--recurrent-dropout', type=float, default=0.0,
                        help='Recurrent dropout rate (default: 0.0)')
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
    parser.add_argument('--grid-size', type=int, default=5,
                        help='B-spline grid segments (default: 5)')
    parser.add_argument('--spline-order', type=int, default=3,
                        help='B-spline order, 3=cubic (default: 3)')
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


def build_sub_kan_configs(args):
    """Build sub_kan_configs list from args."""
    n = args.num_sub_layers
    if args.sub_type == 'kan':
        return [None] * n
    elif args.sub_type == 'relu':
        return ['relu'] * n
    elif args.sub_type == 'mixed':
        # Alternate KAN and relu
        configs = []
        for i in range(n):
            configs.append(None if i % 2 == 0 else 'relu')
        return configs
    return [None] * n


# ============================================================================
# Main
# ============================================================================

def main():
    # Parse arguments
    parser = create_argument_parser("TKAN", "run_tkan.py")
    parser = add_tkan_args(parser)
    args = parser.parse_args()

    # Configuration
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = get_time_splits(args.max_train)

    # Print header
    print_training_header("TKAN", args, symbols, handler_config, time_splits)

    # Initialize Qlib
    init_qlib(handler_config['use_talib'])
    check_data_availability(time_splits)

    # Create data handler and dataset
    handler = create_data_handler(args, handler_config, symbols, time_splits)
    dataset = create_dataset(handler, time_splits)

    # Analyze data
    train_data, valid_cols, dropped_cols = analyze_features(dataset)
    analyze_label_distribution(dataset)

    # Determine d_feat and seq_len
    total_features = train_data.shape[1]

    d_feat, seq_len = resolve_d_feat_and_seq_len(
        args.handler, total_features, args.d_feat if args.d_feat > 0 else None
    )

    # Build sub-layer configs
    sub_kan_configs = build_sub_kan_configs(args)
    sub_kan_dim = args.sub_kan_dim if args.sub_kan_dim > 0 else None

    # Print configuration
    print(f"\n[6] TKAN Configuration:")
    print(f"    Total features: {total_features}")
    print(f"    d_feat (per timestep): {d_feat}")
    print(f"    Sequence length: {seq_len}")
    print(f"    hidden_size: {args.hidden_size}")
    print(f"    num_layers: {args.num_layers}")
    print(f"    sub_kan_configs: {sub_kan_configs}")
    print(f"    sub_kan_dim: {sub_kan_dim or 'auto'}")
    print(f"    dropout: {args.dropout}")
    print(f"    recurrent_dropout: {args.recurrent_dropout}")
    print(f"    loss_type: {args.loss_type}")
    print(f"    reg_lambda: {args.reg_lambda}")
    print(f"    grid_size: {args.grid_size}, spline_order: {args.spline_order}")
    print(f"    batch_size: {args.batch_size}")
    print(f"    n_epochs: {args.n_epochs}, early_stop: {args.early_stop}")
    print(f"    grid_update_freq: {args.grid_update_freq}")
    print(f"    GPU: {args.gpu}")

    # Model loading helpers
    def load_model(path):
        return TKANStock.load(str(path), GPU=args.gpu)

    def get_feature_count(m):
        return m.d_feat * m.seq_len

    # Train or load
    if args.model_path:
        model_path = Path(args.model_path)
        print(f"\n[7] Loading pre-trained model from: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = load_model(model_path)
    else:
        print("\n[7] Training TKAN model...")

        model = TKANStock(
            d_feat=d_feat,
            seq_len=seq_len,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            sub_kan_configs=sub_kan_configs,
            sub_kan_output_dim=sub_kan_dim,
            sub_kan_input_dim=sub_kan_dim,
            dropout=args.dropout,
            recurrent_dropout=args.recurrent_dropout,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            reg_lambda=args.reg_lambda,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            early_stop=args.early_stop,
            grid_update_freq=args.grid_update_freq,
            grid_size=args.grid_size,
            spline_order=args.spline_order,
            loss_type=args.loss_type,
            GPU=args.gpu,
            seed=args.seed,
        )

        model.fit(dataset)

        # Save model
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_SAVE_PATH / f"tkan_{args.handler}_{args.stock_pool}_{args.nday}d.pt"
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
            model_name="TKAN",
            load_model_func=load_model,
            get_feature_count_func=get_feature_count,
        )


if __name__ == "__main__":
    main()
