"""
运行 ALSTM 模型 (v2 - 自定义实现)

使用方法:
    # 使用 Alpha360 (推荐，时间序列特征)
    python scripts/models/deep/run_alstm_v2.py --stock-pool sp500 --handler alpha360

    # 使用 Alpha158 (横截面特征)
    python scripts/models/deep/run_alstm_v2.py --stock-pool sp500 --handler alpha158 --d-feat 158
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import argparse
import torch
import numpy as np

from data.stock_pools import STOCK_POOLS
from models.common import (
    HANDLER_CONFIG, PROJECT_ROOT, MODEL_SAVE_PATH,
    get_time_splits,
    print_training_header,
    init_qlib,
    check_data_availability,
    create_data_handler,
)

from qlib.data.dataset import DatasetH

from models.deep.alstm_model import ALSTMModel


def create_argument_parser():
    parser = argparse.ArgumentParser(description="ALSTM Model Training")

    # 数据参数
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'],
                        help='Stock pool to use')
    parser.add_argument('--handler', type=str, default='alpha360',
                        help='Data handler (alpha360, alpha158, alpha158-master)')
    parser.add_argument('--nday', type=int, default=2,
                        help='Prediction horizon in days')
    parser.add_argument('--max-train', action='store_true',
                        help='Use maximum training data')

    # 模型参数
    parser.add_argument('--d-feat', type=int, default=6,
                        help='Feature dimension per timestep (6 for alpha360, 158 for alpha158)')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Hidden size')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--rnn-type', type=str, default='GRU',
                        choices=['GRU', 'LSTM'],
                        help='RNN type')

    # 训练参数
    parser.add_argument('--n-epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=2000,
                        help='Batch size')
    parser.add_argument('--early-stop', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # 其他
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')

    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    # 根据 handler 自动设置 d_feat
    if args.handler.startswith('alpha360'):
        if args.d_feat == 6:  # 默认值，没有手动设置
            args.d_feat = 6
    elif args.handler.startswith('alpha158'):
        if args.d_feat == 6:  # 默认值，需要调整
            if 'master' in args.handler:
                args.d_feat = 205  # 142 + 63 market features
            else:
                args.d_feat = 158

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = get_time_splits(args.max_train)

    print("=" * 70)
    print("ALSTM Model Training (v2)")
    print("=" * 70)
    print(f"Stock pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Handler: {args.handler}")
    print(f"d_feat: {args.d_feat}")
    print(f"Time splits: {time_splits['train_start']} to {time_splits['test_end']}")

    # 初始化 Qlib
    init_qlib(handler_config['use_talib'])
    check_data_availability(time_splits)

    # 创建 DataHandler
    handler = create_data_handler(args, handler_config, symbols, time_splits)

    # 创建 Dataset
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (time_splits['train_start'], time_splits['train_end']),
            "valid": (time_splits['valid_start'], time_splits['valid_end']),
            "test": (time_splits['test_start'], time_splits['test_end']),
        }
    )

    print(f"\n[Model Configuration]")
    print(f"  d_feat: {args.d_feat}")
    print(f"  hidden_size: {args.hidden_size}")
    print(f"  num_layers: {args.num_layers}")
    print(f"  dropout: {args.dropout}")
    print(f"  rnn_type: {args.rnn_type}")
    print(f"  lr: {args.lr}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  n_epochs: {args.n_epochs}")
    print(f"  early_stop: {args.early_stop}")

    # 创建模型
    model = ALSTMModel(
        d_feat=args.d_feat,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        rnn_type=args.rnn_type,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        early_stop=args.early_stop,
        GPU=args.gpu if torch.cuda.is_available() else -1,
        seed=args.seed,
    )

    # 训练
    model.fit(dataset, debug=args.debug)

    # 预测
    print("\n[Prediction]")
    test_pred = model.predict(dataset, "test")
    print(f"  Prediction shape: {test_pred.shape}")
    print(f"  Prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    # 评估
    print("\n[Evaluation]")
    try:
        test_label = handler.fetch(col_set="label")
        test_label = test_label.loc[time_splits['test_start']:time_splits['test_end']]

        common_idx = test_pred.index.intersection(test_label.index)
        pred_aligned = test_pred.loc[common_idx]
        label_aligned = test_label.loc[common_idx]

        if isinstance(label_aligned, np.ndarray):
            pass
        elif hasattr(label_aligned, 'iloc'):
            label_aligned = label_aligned.iloc[:, 0] if len(label_aligned.shape) > 1 else label_aligned

        from scipy import stats
        valid_mask = ~label_aligned.isna()
        if valid_mask.sum() > 0:
            ic, _ = stats.spearmanr(
                pred_aligned[valid_mask].values,
                label_aligned[valid_mask].values
            )
            print(f"  Test IC (Spearman): {ic:.4f}")
    except Exception as e:
        print(f"  Warning: Could not compute metrics: {e}")

    # 保存模型
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_SAVE_PATH / f"alstm_v2_{args.handler}_{args.stock_pool}_{args.nday}d.pt"
    model.save(str(model_path))


if __name__ == "__main__":
    main()
