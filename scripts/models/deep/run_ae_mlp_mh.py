"""
运行 Multi-Horizon AE-MLP 模型

在 AE-MLP 基础上增加 multi-horizon auxiliary tasks。
共享编码器同时预测 2天、5天、10天 forward returns，
辅助目标提供额外梯度信号，起到隐式正则化作用，提升主目标的预测质量。

推理时只使用主目标 (默认5天) 的 prediction head。

使用方法:
    # 基本用法（默认 horizons=[2,5,10], primary=5d）
    python scripts/models/deep/run_ae_mlp_mh.py --stock-pool sp500 --handler alpha158-mh

    # 自定义 horizons 和辅助权重
    python scripts/models/deep/run_ae_mlp_mh.py --handler alpha158-mh --horizons 2,5,10,20 --primary-horizon 5 --aux-weight 0.3

    # 使用 TA-Lib 特征
    python scripts/models/deep/run_ae_mlp_mh.py --handler alpha158-talib-lite-mh --stock-pool sp100

    # 加载已有模型进行回测
    python scripts/models/deep/run_ae_mlp_mh.py --model-path ./my_models/ae_mlp_mh.keras --backtest
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import numpy as np
import pandas as pd

from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG, MODEL_SAVE_PATH,
    create_argument_parser,
    get_time_splits,
    print_training_header,
    init_qlib,
    check_data_availability,
    create_dataset,
    analyze_features,
    print_prediction_stats,
    run_backtest,
)

from models.deep.ae_mlp_multihorizon import AEMLPMultiHorizon


# ============================================================================
# 默认参数
# ============================================================================

DEFAULT_MH_PARAMS = {
    'hidden_units': None,
    'dropout_rates': None,
    'lr': 0.001,
    'batch_size': 4096,
    'horizons': [2, 5, 10],
    'primary_horizon': 5,
    'aux_horizon_weight': 0.3,
    'loss_weights_decoder': 0.1,
    'loss_weights_ae': 0.1,
}


def parse_list_arg(arg_str, dtype=float):
    """解析逗号分隔的列表参数"""
    if arg_str is None:
        return None
    return [dtype(x.strip()) for x in arg_str.split(',')]


def add_mh_args(parser):
    """添加 Multi-Horizon AE-MLP 特定参数"""
    # 模型结构
    parser.add_argument('--hidden-units', type=str, default=None,
                        help='Hidden units per layer, comma-separated (e.g., "96,96,512,256,128")')
    parser.add_argument('--dropout-rates', type=str, default=None,
                        help='Dropout rates per layer, comma-separated')
    parser.add_argument('--n-epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: 4096)')
    parser.add_argument('--early-stop', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # Multi-horizon 特有参数
    parser.add_argument('--horizons', type=str, default='2,5,10',
                        help='Prediction horizons, comma-separated (default: "2,5,10")')
    parser.add_argument('--primary-horizon', type=int, default=5,
                        help='Primary prediction horizon in days (default: 5)')
    parser.add_argument('--aux-weight', type=float, default=0.3,
                        help='Loss weight for auxiliary horizon heads (default: 0.3)')

    # 损失权重
    parser.add_argument('--loss-decoder', type=float, default=0.1,
                        help='Loss weight for decoder (default: 0.1)')
    parser.add_argument('--loss-ae', type=float, default=0.1,
                        help='Loss weight for ae_action (default: 0.1)')
    parser.add_argument('--loss-main', type=float, default=1.0,
                        help='Loss weight for primary horizon head (default: 1.0)')

    # Macro 特征参数（仅 alpha158-talib-lite-macro-mh handler 使用）
    parser.add_argument('--macro-features', type=str, default='core',
                        choices=['all', 'core', 'vix_only', 'none'],
                        help='Macro feature set (default: core, ~23 features)')

    # 参数文件
    parser.add_argument('--params-file', type=str, default=None,
                        help='Path to JSON file with model params')

    return parser


def create_mh_data_handler(args, handler_config, symbols, time_splits, horizons, primary_horizon):
    """
    创建 Multi-Horizon DataHandler。

    不使用通用的 create_data_handler，因为 multi-horizon handler
    需要传入 horizons 和 primary_horizon 参数。
    """
    print(f"\n[3] Creating Multi-Horizon DataHandler...")
    print(f"    Features: {handler_config['description']}")
    print(f"    Horizons: {horizons}")
    print(f"    Primary horizon: {primary_horizon}d")

    handler_kwargs = {
        'horizons': horizons,
        'primary_horizon': primary_horizon,
        'instruments': symbols,
        'start_time': time_splits['train_start'],
        'end_time': time_splits['test_end'],
        'fit_start_time': time_splits['train_start'],
        'fit_end_time': time_splits['train_end'],
        'infer_processors': [],
    }

    # Macro handler 需要额外参数
    if 'macro' in args.handler:
        handler_kwargs['macro_features'] = args.macro_features
        print(f"    Macro features: {args.macro_features}")

    HandlerClass = handler_config['class']
    handler = HandlerClass(**handler_kwargs)

    print(f"    DataHandler created: {args.handler}")
    return handler


def main():
    # 解析命令行参数
    parser = create_argument_parser("Multi-Horizon AE-MLP", "run_ae_mlp_mh.py")
    parser = add_mh_args(parser)
    args = parser.parse_args()

    # 解析 horizons
    horizons = [int(x.strip()) for x in args.horizons.split(',')]
    primary_horizon = args.primary_horizon

    # 解析列表参数
    hidden_units = parse_list_arg(args.hidden_units, int)
    dropout_rates = parse_list_arg(args.dropout_rates, float)

    # 合并参数
    lr = args.lr or DEFAULT_MH_PARAMS['lr']
    batch_size = args.batch_size or DEFAULT_MH_PARAMS['batch_size']

    # 构建 loss_weights
    loss_weights = {
        'decoder': args.loss_decoder,
        'ae_action': args.loss_ae,
    }
    for h in horizons:
        key = f'action_{h}d'
        if h == primary_horizon:
            loss_weights[key] = args.loss_main
        else:
            loss_weights[key] = args.aux_weight

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = get_time_splits(args.max_train)

    # 打印头部信息
    print_training_header("Multi-Horizon AE-MLP", args, symbols, handler_config, time_splits)
    print(f"Horizons: {horizons}")
    print(f"Primary Horizon: {primary_horizon}d")
    print(f"Aux Weight: {args.aux_weight}")

    # 初始化和数据准备
    init_qlib(handler_config['use_talib'])
    check_data_availability(time_splits)

    # Multi-horizon handler 需要特殊创建
    handler = create_mh_data_handler(args, handler_config, symbols, time_splits, horizons, primary_horizon)
    dataset = create_dataset(handler, time_splits)
    train_data, valid_cols, dropped_cols = analyze_features(dataset)

    # Multi-horizon 标签分析（不用通用的 analyze_label_distribution，因为它硬编码了 LABEL0）
    print("\n[5] Analyzing multi-horizon label distribution...")
    train_label = dataset.prepare("train", col_set="label")
    valid_label = dataset.prepare("valid", col_set="label")
    if isinstance(train_label, pd.DataFrame):
        for col in train_label.columns:
            col_name = col[1] if isinstance(col, tuple) else col
            print(f"    {col_name} (train): mean={train_label[col].mean():.4f}, "
                  f"std={train_label[col].std():.4f}, "
                  f"min={train_label[col].min():.4f}, max={train_label[col].max():.4f}")

    # 获取实际的特征数量
    actual_train_data = dataset.prepare("train", col_set="feature")
    total_features = actual_train_data.shape[1]
    print(f"\n    Actual training data shape: {actual_train_data.shape}")
    print(f"    Label columns: {list(train_label.columns) if hasattr(train_label, 'columns') else 'single'}")
    print(f"    Label shape: {train_label.shape}")

    print(f"\n[6] Model Configuration:")
    print(f"    Total features: {total_features}")
    print(f"    Hidden units: {hidden_units or 'auto'}")
    print(f"    Dropout rates: {dropout_rates or 'auto'}")
    print(f"    Learning rate: {lr}")
    print(f"    Batch size: {batch_size}")
    print(f"    Epochs: {args.n_epochs}")
    print(f"    Early stop: {args.early_stop}")
    print(f"    GPU: {args.gpu}")
    print(f"    Loss weights: {loss_weights}")

    # 模型加载/训练
    def load_model(path):
        return AEMLPMultiHorizon.load(str(path), horizons=horizons, primary_horizon=primary_horizon)

    def get_feature_count(m):
        return m.num_columns

    if args.model_path:
        model_path = Path(args.model_path)
        print(f"\n[7] Loading pre-trained model from: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = load_model(model_path)
    else:
        print("\n[7] Training Multi-Horizon AE-MLP model...")

        model = AEMLPMultiHorizon(
            num_columns=total_features,
            horizons=horizons,
            primary_horizon=primary_horizon,
            hidden_units=hidden_units or [96, 96, 512, 256, 128],
            dropout_rates=dropout_rates or [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
            lr=lr,
            n_epochs=args.n_epochs,
            batch_size=batch_size,
            early_stop=args.early_stop,
            loss_weights=loss_weights,
            aux_horizon_weight=args.aux_weight,
            GPU=args.gpu,
            seed=args.seed,
        )

        model.fit(dataset)
        print("    Model training completed")

        # 保存模型
        print("\n[10] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        horizons_str = '_'.join(str(h) for h in horizons)
        model_path = MODEL_SAVE_PATH / f"ae_mlp_mh_{args.handler}_{args.stock_pool}_{horizons_str}.keras"
        model.save(str(model_path))

    # 预测 (仅主 horizon)
    print("\n[8] Generating predictions (primary horizon only)...")
    test_pred = model.predict(dataset, segment="test")

    print(f"    Predictions NaN count: {test_pred.isna().sum()} ({test_pred.isna().sum() / len(test_pred) * 100:.2f}%)")
    if not test_pred.isna().all():
        print(f"    Predictions min/max: {test_pred.min():.4f} / {test_pred.max():.4f}")
    print_prediction_stats(test_pred)

    # 所有 horizon 的预测（用于对比）
    print("\n[8b] Generating predictions for all horizons...")
    all_horizon_preds = model.predict_all_horizons(dataset, segment="test")
    for h, pred in all_horizon_preds.items():
        marker = " (PRIMARY)" if h == primary_horizon else ""
        print(f"    {h}d{marker}: mean={pred.mean():.6f}, std={pred.std():.6f}, "
              f"min={pred.min():.6f}, max={pred.max():.6f}")

    # 评估所有 horizons 的 IC/ICIR
    print("\n[9] Evaluation (all horizons)...")
    label_data = dataset.prepare("test", col_set="label")

    # 处理 MultiIndex 列名
    if isinstance(label_data, pd.DataFrame) and isinstance(label_data.columns, pd.MultiIndex):
        label_data.columns = [col[1] if isinstance(col, tuple) else col for col in label_data.columns]

    if isinstance(label_data, pd.DataFrame):
        for h, pred in all_horizon_preds.items():
            col_name = f'LABEL_{h}d'
            matched_col = None
            if col_name in label_data.columns:
                matched_col = col_name
            else:
                # 尝试匹配可能的列名变体
                for c in label_data.columns:
                    cn = c[1] if isinstance(c, tuple) else c
                    if cn == col_name:
                        matched_col = c
                        break

            if matched_col is not None:
                label_series = label_data[matched_col]
                test_pred_aligned = pred.reindex(label_data.index)

                valid_idx = ~(test_pred_aligned.isna() | label_series.isna())
                pred_clean = test_pred_aligned[valid_idx]
                label_clean = label_series[valid_idx]

                if len(pred_clean) > 0:
                    # IC
                    daily_ic = pred_clean.groupby(level="datetime").apply(
                        lambda x: x.corr(label_clean.loc[x.index]) if len(x) > 1 else np.nan
                    ).dropna()

                    # Error metrics
                    mse = ((pred_clean - label_clean) ** 2).mean()
                    mae = (pred_clean - label_clean).abs().mean()
                    rmse = np.sqrt(mse)

                    marker = " <<< PRIMARY" if h == primary_horizon else ""
                    print(f"\n    --- {h}d{marker} ---")
                    print(f"    Valid samples: {len(pred_clean)}")
                    if len(daily_ic) > 0:
                        print(f"    IC:   {daily_ic.mean():.4f}")
                        print(f"    ICIR: {daily_ic.mean()/daily_ic.std():.4f}")
                        print(f"    IC_std: {daily_ic.std():.4f}")
                    print(f"    MSE:  {mse:.6f}")
                    print(f"    MAE:  {mae:.6f}")
                    print(f"    RMSE: {rmse:.6f}")

    # 回测
    if args.backtest:
        pred_df = test_pred.to_frame("score")

        run_backtest(
            model_path, dataset, pred_df, args, time_splits,
            model_name="MH-AE-MLP",
            load_model_func=load_model,
            get_feature_count_func=get_feature_count,
        )


if __name__ == "__main__":
    main()
