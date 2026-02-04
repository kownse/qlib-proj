"""
Run StockMixer (AAAI 2024) Model

StockMixer is an MLP-based architecture that processes ALL stocks simultaneously
per date, enabling cross-stock relationship learning through stock-dimension mixing.

Key features:
1. Multi-scale time mixing with causal (TriU) constraints
2. Stock-dimension mixing (NoGraphMixer) for cross-stock patterns
3. Pairwise ranking loss for better relative predictions

Usage:
    # Basic test (small stock pool)
    python scripts/models/deep/run_stockmixer.py --stock-pool test --handler alpha300

    # Full training with backtest
    python scripts/models/deep/run_stockmixer.py --stock-pool sp500 --handler alpha300 --backtest

    # Adjust ranking loss weight
    python scripts/models/deep/run_stockmixer.py --stock-pool sp500 --handler alpha300 --alpha 0.2

    # Load pre-trained model
    python scripts/models/deep/run_stockmixer.py --model-path ./my_models/stockmixer.pt --backtest

Paper: StockMixer: A Simple yet Strong MLP-based Architecture for Stock Price Forecasting
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

from models.deep.stockmixer_model import StockMixer, create_stockmixer


# ============================================================================
# Default StockMixer Parameters
# ============================================================================

DEFAULT_STOCKMIXER_PARAMS = {
    'time_steps': 60,       # Lookback window (60 for Alpha300)
    'channels': 5,          # Features per time step (OHLCV)
    'market_num': 20,       # NoGraphMixer hidden dimension
    'scale_factor': 3,      # Multi-scale factor
    'learning_rate': 0.001,
    'alpha': 0.1,           # Ranking loss weight
    'n_epochs': 100,
    'early_stop': 10,
}


def add_stockmixer_args(parser):
    """Add StockMixer-specific arguments."""
    parser.add_argument('--time-steps', type=int, default=None,
                        help='Lookback window (default: 60 for alpha300)')
    parser.add_argument('--fea-num', type=int, default=None,
                        help='Features per time step (default: 5 for OHLCV)')
    parser.add_argument('--market-num', type=int, default=20,
                        help='NoGraphMixer hidden dimension (default: 20)')
    parser.add_argument('--scale-factor', type=int, default=3,
                        help='Multi-scale factor (default: 3)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Ranking loss weight (default: 0.1)')
    parser.add_argument('--n-epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--early-stop', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    return parser


def infer_handler_config(handler_name: str) -> dict:
    """
    Infer time_steps and channels from handler name.

    Args:
        handler_name: Handler name (e.g., 'alpha300', 'alpha360')

    Returns:
        dict with 'time_steps' and 'channels'
    """
    configs = {
        'alpha300': {'time_steps': 60, 'channels': 5},
        'alpha300-ts': {'time_steps': 60, 'channels': 5},
        'alpha300-macro': {'time_steps': 60, 'channels': 11},  # 5 OHLCV + 6 macro
        'alpha360': {'time_steps': 60, 'channels': 6},
        'alpha360-macro': {'time_steps': 60, 'channels': 6},  # varies with macro
        'alpha180': {'time_steps': 30, 'channels': 6},
        'alpha180-macro': {'time_steps': 30, 'channels': 6},
    }

    return configs.get(handler_name, {'time_steps': 60, 'channels': 5})


def main():
    # Parse command line arguments
    parser = create_argument_parser("StockMixer", "run_stockmixer.py")
    parser = add_stockmixer_args(parser)
    args = parser.parse_args()

    # Get configuration
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = get_time_splits(args.max_train)

    # Infer handler-specific config
    handler_params = infer_handler_config(args.handler)
    time_steps = args.time_steps or handler_params['time_steps']
    channels = args.fea_num or handler_params['channels']

    # Print header
    print_training_header("StockMixer", args, symbols, handler_config, time_splits)

    # Initialize Qlib
    init_qlib(handler_config['use_talib'])
    check_data_availability(time_splits)

    # Create data handler and dataset
    handler = create_data_handler(args, handler_config, symbols, time_splits)
    dataset = create_dataset(handler, time_splits)

    # Analyze data
    train_data, valid_cols, dropped_cols = analyze_features(dataset)
    analyze_label_distribution(dataset)

    # Get actual data dimensions
    actual_train_data = dataset.prepare("train", col_set="feature")
    total_features = actual_train_data.shape[1]

    # Infer actual time_steps and channels from data
    if total_features != time_steps * channels:
        # Try to find correct configuration
        possible_channels = [5, 6, 11, 29, 115]  # Common channel counts
        for ch in possible_channels:
            if total_features % ch == 0:
                inferred_ts = total_features // ch
                if inferred_ts > 0 and inferred_ts <= 120:
                    time_steps = inferred_ts
                    channels = ch
                    print(f"\n    Auto-adjusted: time_steps={time_steps}, channels={channels}")
                    break

    # Count stocks
    num_stocks = len(symbols)
    print(f"\n[6] StockMixer Configuration:")
    print(f"    Stock universe: {num_stocks} stocks")
    print(f"    Time steps (lookback): {time_steps}")
    print(f"    Channels (features/step): {channels}")
    print(f"    Total features: {total_features} (should be {time_steps * channels})")
    print(f"    Market num (stock mixer hidden): {args.market_num}")
    print(f"    Scale factor: {args.scale_factor}")
    print(f"    Learning rate: {args.lr}")
    print(f"    Ranking loss alpha: {args.alpha}")
    print(f"    Epochs: {args.n_epochs}")
    print(f"    Early stop patience: {args.early_stop}")
    print(f"    GPU: {args.gpu}")

    # Model loading function
    def load_model(path):
        return StockMixer.load(str(path), GPU=args.gpu)

    def get_feature_count(m):
        return m.time_steps * m.channels

    # Check if loading pre-trained model
    if args.model_path:
        model_path = Path(args.model_path)
        print(f"\n[7] Loading pre-trained model from: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = load_model(model_path)
        print("    Model loaded successfully")
    else:
        # Training
        print("\n[7] Training StockMixer model...")

        # Create model
        model = StockMixer(
            num_stocks=num_stocks,
            time_steps=time_steps,
            channels=channels,
            market_num=args.market_num,
            scale_factor=args.scale_factor,
            learning_rate=args.lr,
            alpha=args.alpha,
            n_epochs=args.n_epochs,
            early_stop=args.early_stop,
            GPU=args.gpu,
            seed=args.seed,
        )

        # Train
        model.fit(dataset)
        print("    Training completed")

        # Save model
        print("\n[10] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_SAVE_PATH / f"stockmixer_{args.handler}_{args.stock_pool}_{args.nday}d.pt"
        model.save(str(model_path))

    # Prediction
    print("\n[8] Generating predictions...")
    test_pred = model.predict(dataset, segment="test")

    # Debug: check predictions
    print(f"    Predictions shape: {len(test_pred)}")
    print(f"    Predictions NaN count: {test_pred.isna().sum()} ({test_pred.isna().sum() / len(test_pred) * 100:.2f}%)")
    if not test_pred.isna().all():
        print(f"    Predictions min/max: {test_pred.min():.4f} / {test_pred.max():.4f}")
    print_prediction_stats(test_pred)

    # Evaluation
    print("\n[9] Evaluation...")
    evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

    # Backtest
    if args.backtest:
        pred_df = test_pred.to_frame("score")

        run_backtest(
            model_path, dataset, pred_df, args, time_splits,
            model_name="StockMixer",
            load_model_func=load_model,
            get_feature_count_func=get_feature_count
        )


if __name__ == "__main__":
    main()
