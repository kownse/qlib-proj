"""
运行 Transformer 模型

Transformer 使用自注意力机制，能够更好地捕捉长距离依赖关系。

使用方法:
    python scripts/models/run_transformer.py --stock-pool sp500 --handler alpha360 --nday 5 --backtest
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import torch
import numpy as np
import pandas as pd

from qlib.contrib.model.pytorch_transformer import TransformerModel

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


def add_transformer_args(parser):
    """添加 Transformer 特定参数"""
    parser.add_argument('--d-feat', type=int, default=6,
                        help='Base features per timestep (default: 6 for Alpha360)')
    parser.add_argument('--d-model', type=int, default=64,
                        help='Model dimension (default: 64)')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads (default: 4)')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of transformer layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--n-epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Batch size (default: 2048)')
    parser.add_argument('--early-stop', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--reg', type=float, default=1e-3,
                        help='L2 regularization weight (default: 1e-3)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    return parser


from models.common.ts_model_utils import resolve_d_feat_and_seq_len


def main():
    # 解析命令行参数
    parser = create_argument_parser("Transformer", "run_transformer.py")
    parser = add_transformer_args(parser)
    args = parser.parse_args()

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = get_time_splits(args.max_train)

    # 打印头部信息
    print_training_header("Transformer", args, symbols, handler_config, time_splits)

    # 初始化和数据准备
    init_qlib(handler_config['use_talib'])
    check_data_availability(time_splits)
    handler = create_data_handler(args, handler_config, symbols, time_splits)
    dataset = create_dataset(handler, time_splits)
    train_data, valid_cols, dropped_cols = analyze_features(dataset)
    analyze_label_distribution(dataset)

    # 获取实际的特征数量（从训练数据中）
    # valid_cols 是去掉全 NaN 列后的数量，但模型实际接收的是原始特征
    actual_train_data = dataset.prepare("train", col_set="feature")
    total_features = actual_train_data.shape[1]
    print(f"\n    Actual training data shape: {actual_train_data.shape}")

    d_feat, seq_len = resolve_d_feat_and_seq_len(args.handler, total_features, args.d_feat)

    print(f"\n[6] Model Configuration:")
    print(f"    Total features: {total_features}")
    print(f"    d_feat (per timestep): {d_feat}")
    print(f"    Sequence length: {seq_len}")

    # 定义模型加载函数
    def load_model(path):
        return torch.load(path, weights_only=False)

    def get_feature_count(m):
        return total_features

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
        print(f"    Model dimension: {args.d_model}")
        print(f"    Attention heads: {args.nhead}")
        print(f"    Num layers: {args.num_layers}")
        print(f"    Dropout: {args.dropout}")
        print(f"    Learning rate: {args.lr}")
        print(f"    L2 regularization: {args.reg}")
        print(f"    Batch size: {args.batch_size}")
        print(f"    Epochs: {args.n_epochs}")
        print(f"    Early stop: {args.early_stop}")
        print(f"    GPU: {args.gpu}")

        # 创建模型
        print("\n[7] Training Transformer model...")
        model = TransformerModel(
            d_feat=d_feat,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            n_epochs=args.n_epochs,
            lr=args.lr,
            reg=args.reg,
            early_stop=args.early_stop,
            batch_size=args.batch_size,
            metric="loss",
            loss="mse",
            GPU=args.gpu if torch.cuda.is_available() else -1,
        )

        # 训练
        model.fit(dataset)
        print("    Model training completed")

        # 保存模型
        print("\n[10] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_SAVE_PATH / f"transformer_{args.handler}_{args.stock_pool}_{args.nday}d.pt"
        torch.save(model, model_path)
        print(f"    Model saved to: {model_path}")

    # 预测
    print("\n[8] Generating predictions...")

    # Debug: 检查测试数据
    test_data = dataset.prepare("test", col_set="feature")
    test_label = dataset.prepare("test", col_set="label")
    print(f"    Test data shape: {test_data.shape}")

    test_nan_count = test_data.isna().sum().sum()
    test_nan_pct = test_nan_count / test_data.size * 100
    print(f"    Test data NaN count: {test_nan_count} ({test_nan_pct:.2f}%)")

    # 先处理 NaN，再计算 min/max
    test_data_clean = test_data.fillna(0)
    test_min = test_data_clean.values.min()
    test_max = test_data_clean.values.max()
    test_abs_max = np.abs(test_data_clean.values).max()
    print(f"    Test data min/max (after fillna): {test_min:.4f} / {test_max:.4f}")

    # 检查是否需要归一化（有 NaN 或有极端值）
    need_normalize = test_nan_count > 0 or test_abs_max > 1e6
    if need_normalize:
        if test_abs_max > 1e6:
            print(f"    WARNING: Test data has extreme values (max abs: {test_abs_max:.2e})")
        print(f"    Applying normalization to test data...")

        # 对测试数据应用相同的归一化
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
        print(f"    After normalization - min/max: {test_data_normalized.values.min():.4f} / {test_data_normalized.values.max():.4f}")

        # 使用归一化后的数据进行预测
        print("    Generating predictions with normalized test data...")
        model.model.eval()
        test_values = test_data_normalized.values
        preds = []
        batch_size = getattr(model, 'batch_size', 2048)

        with torch.no_grad():
            for i in range(0, len(test_values), batch_size):
                batch = test_values[i:i+batch_size]
                batch_tensor = torch.from_numpy(batch).float().to(model.device)
                batch_pred = model.model(batch_tensor)
                preds.append(batch_pred.cpu().numpy())

        pred_values = np.concatenate(preds)
        test_pred = pd.Series(pred_values, index=test_data.index, name='score')
    else:
        # 正常预测流程
        pred = model.predict(dataset)

        if isinstance(pred, pd.DataFrame):
            test_pred = pred.iloc[:, 0]
        else:
            test_pred = pred

        test_pred.name = 'score'

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
            model_name="Transformer",
            load_model_func=load_model,
            get_feature_count_func=get_feature_count
        )


if __name__ == "__main__":
    main()
