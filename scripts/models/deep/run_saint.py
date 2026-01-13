"""
运行 SAINT (Self-Attention and Intersample Attention Transformer) 模型

SAINT 使用双重注意力机制:
- 行注意力 (Row Attention): 捕捉特征之间的关系
- 列注意力 (Column/Intersample Attention): 捕捉股票之间的关系

这种设计特别适合股票预测，因为:
1. 不同技术指标之间存在复杂的非线性关系
2. 不同股票之间存在联动效应（行业、市场情绪等）

使用方法:
    python scripts/models/deep/run_saint.py --stock-pool sp500 --handler alpha158 --backtest
    python scripts/models/deep/run_saint.py --stock-pool sp100 --handler alpha360 --n-epochs 200
    python scripts/models/deep/run_saint.py --model-path ./my_models/saint.pt --backtest
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import torch
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

from models.deep.saint_model import SAINT, create_saint_for_handler


def add_saint_args(parser):
    """添加 SAINT 特定参数"""
    parser.add_argument('--d-model', type=int, default=64,
                        help='Model hidden dimension (default: 64)')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads (default: 4)')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of SAINT blocks (default: 3)')
    parser.add_argument('--dim-feedforward', type=int, default=256,
                        help='FFN dimension (default: 256)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--col-sample-ratio', type=float, default=1.0,
                        help='Column attention sampling ratio (default: 1.0)')
    parser.add_argument('--n-epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size (default: 512)')
    parser.add_argument('--grad-accum-steps', type=int, default=2,
                        help='Gradient accumulation steps (default: 2)')
    parser.add_argument('--early-stop', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--reg', type=float, default=1e-3,
                        help='L2 regularization weight (default: 1e-3)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision training (AMP)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--auto-config', action='store_true',
                        help='Auto configure model based on handler type')
    return parser


def main():
    # 解析命令行参数
    parser = create_argument_parser("SAINT", "run_saint.py")
    parser = add_saint_args(parser)
    args = parser.parse_args()

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = get_time_splits(args.max_train)

    # 打印头部信息
    print_training_header("SAINT", args, symbols, handler_config, time_splits)

    # 初始化和数据准备
    init_qlib(handler_config['use_talib'])
    check_data_availability(time_splits)
    handler = create_data_handler(args, handler_config, symbols, time_splits)
    dataset = create_dataset(handler, time_splits)
    train_data, valid_cols, dropped_cols = analyze_features(dataset)
    analyze_label_distribution(dataset)

    # 获取实际的特征数量
    actual_train_data = dataset.prepare("train", col_set="feature")
    total_features = actual_train_data.shape[1]
    print(f"\n    Actual training data shape: {actual_train_data.shape}")

    print(f"\n[6] Model Configuration:")
    print(f"    Total features: {total_features}")

    # 定义模型加载函数
    def load_model(path):
        return SAINT.load(str(path), GPU=args.gpu)

    def get_feature_count(m):
        return m.num_features

    # 检查是否提供了预训练模型路径
    if args.model_path:
        # 加载预训练模型，跳过训练
        model_path = Path(args.model_path)
        print(f"\n[7] Loading pre-trained model from: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = load_model(model_path)
        print("    Model loaded successfully")
    else:
        # 正常训练流程
        use_amp = not args.no_amp
        print(f"    d_model: {args.d_model}")
        print(f"    Attention heads: {args.nhead}")
        print(f"    Num layers: {args.num_layers}")
        print(f"    FFN dimension: {args.dim_feedforward}")
        print(f"    Column sample ratio: {args.col_sample_ratio}")
        print(f"    Dropout: {args.dropout}")
        print(f"    Learning rate: {args.lr}")
        print(f"    L2 regularization: {args.reg}")
        print(f"    Batch size: {args.batch_size}")
        print(f"    Grad accum steps: {args.grad_accum_steps}")
        print(f"    Effective batch: {args.batch_size * args.grad_accum_steps}")
        print(f"    Epochs: {args.n_epochs}")
        print(f"    Early stop: {args.early_stop}")
        print(f"    Mixed precision (AMP): {use_amp}")
        print(f"    GPU: {args.gpu}")

        # 创建模型
        print("\n[7] Training SAINT model...")

        if args.auto_config:
            # 根据 handler 自动配置
            print("    Using auto configuration based on handler type...")
            model = create_saint_for_handler(
                args.handler,
                lr=args.lr,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum_steps,
                early_stop=args.early_stop,
                reg=args.reg,
                use_amp=use_amp,
                GPU=args.gpu if torch.cuda.is_available() else -1,
                seed=args.seed,
            )
            # 更新实际特征数
            model.num_features = total_features
        else:
            model = SAINT(
                num_features=total_features,
                d_model=args.d_model,
                nhead=args.nhead,
                num_layers=args.num_layers,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                col_sample_ratio=args.col_sample_ratio,
                lr=args.lr,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum_steps,
                early_stop=args.early_stop,
                reg=args.reg,
                use_amp=use_amp,
                GPU=args.gpu if torch.cuda.is_available() else -1,
                seed=args.seed,
            )

        # 训练
        model.fit(dataset)
        print("    Model training completed")

        # 保存模型
        print("\n[10] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_SAVE_PATH / f"saint_{args.handler}_{args.stock_pool}_{args.nday}d.pt"
        model.save(str(model_path))

    # 预测
    print("\n[8] Generating predictions...")

    # Debug: 检查测试数据
    test_data = dataset.prepare("test", col_set="feature")
    test_label = dataset.prepare("test", col_set="label")
    print(f"    Test data shape: {test_data.shape}")

    test_nan_count = test_data.isna().sum().sum()
    test_nan_pct = test_nan_count / test_data.size * 100
    print(f"    Test data NaN count: {test_nan_count} ({test_nan_pct:.2f}%)")

    # 预测
    test_pred = model.predict(dataset, segment="test")

    # Debug: 检查预测结果
    print(f"    Predictions NaN count: {test_pred.isna().sum()} ({test_pred.isna().sum() / len(test_pred) * 100:.2f}%)")
    if not test_pred.isna().all():
        print(f"    Predictions min/max: {test_pred.min():.4f} / {test_pred.max():.4f}")
    print_prediction_stats(test_pred)

    # 评估
    print("\n[9] Evaluation...")
    evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

    # 回测
    if args.backtest:
        pred_df = test_pred.to_frame("score")

        run_backtest(
            model_path, dataset, pred_df, args, time_splits,
            model_name="SAINT",
            load_model_func=load_model,
            get_feature_count_func=get_feature_count
        )


if __name__ == "__main__":
    main()
