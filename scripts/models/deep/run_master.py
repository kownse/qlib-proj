"""
运行 MASTER 模型

MASTER (Market-Guided Stock Transformer) 使用市场信息来引导个股预测。
通过门控机制融合市场状态和个股特征，实现市场感知的股票预测。

特点:
1. 双编码器架构：分别处理个股特征(158)和市场信息(63)
2. 市场引导门控：用市场状态调制个股预测
3. 可选的交叉注意力融合

使用方法:
    # 基本用法
    python scripts/models/deep/run_master.py --stock-pool sp500 --handler alpha158-master --backtest

    # 使用大模型
    python scripts/models/deep/run_master.py --stock-pool sp500 --handler alpha158-master --preset large

    # 自定义参数
    python scripts/models/deep/run_master.py --stock-pool sp500 --handler alpha158-master \
        --stock-hidden 512 --stock-emb 256 --market-hidden 256 --market-emb 128

    # 加载预训练模型
    python scripts/models/deep/run_master.py --model-path ./my_models/master.pt --backtest

数据准备:
    1. 下载 IWM 数据: python scripts/data/download_macro_data.py --symbols IWM --force
    2. 处理市场信息: python scripts/data/process_master_market_info.py
    3. 运行模型
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

from models.deep.master_model import MASTERModel, create_master_model, PRESET_CONFIGS


def add_master_args(parser):
    """添加 MASTER 特定参数"""
    # 模型架构参数
    parser.add_argument('--preset', type=str, default='default',
                        choices=['default', 'large', 'lite'],
                        help='Model preset (default, large, lite)')
    parser.add_argument('--stock-hidden', type=int, default=None,
                        help='Stock encoder hidden dim (default: 256)')
    parser.add_argument('--stock-emb', type=int, default=None,
                        help='Stock embedding dim (default: 128)')
    parser.add_argument('--market-hidden', type=int, default=None,
                        help='Market encoder hidden dim (default: 128)')
    parser.add_argument('--market-emb', type=int, default=None,
                        help='Market embedding dim (default: 64)')
    parser.add_argument('--fusion-hidden', type=int, default=None,
                        help='Fusion hidden dim (default: 128)')
    parser.add_argument('--stock-layers', type=int, default=None,
                        help='Number of stock encoder layers (default: 2)')
    parser.add_argument('--market-layers', type=int, default=None,
                        help='Number of market encoder layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--no-attention', action='store_true',
                        help='Disable cross-attention fusion')

    # 训练参数
    parser.add_argument('--n-epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                        help='Weight decay (default: 1e-3)')
    parser.add_argument('--batch-size', type=int, default=4096,
                        help='Batch size (default: 4096)')
    parser.add_argument('--early-stop', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # 损失函数参数
    parser.add_argument('--use-ic-loss', action='store_true',
                        help='Use IC loss in addition to MSE')
    parser.add_argument('--ic-loss-weight', type=float, default=0.5,
                        help='IC loss weight (default: 0.5)')

    return parser


def get_preset_config(preset: str) -> dict:
    """获取预设配置"""
    if preset == 'large':
        return PRESET_CONFIGS['alpha158_master_large']
    elif preset == 'lite':
        return PRESET_CONFIGS['alpha158_master_lite']
    else:
        return PRESET_CONFIGS['alpha158_master']


def main():
    # 解析命令行参数
    parser = create_argument_parser("MASTER", "run_master.py")
    parser = add_master_args(parser)
    args = parser.parse_args()

    # 检查 handler
    if args.handler != 'alpha158-master':
        print(f"Warning: MASTER model is designed for alpha158-master handler, but got {args.handler}")
        print("         Continuing anyway, but results may not be optimal.")

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = get_time_splits(args.max_train)

    # 打印头部信息
    print_training_header("MASTER", args, symbols, handler_config, time_splits)

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

    # 获取预设配置
    preset_config = get_preset_config(args.preset)

    # 从 handler 获取 MASTER 配置
    try:
        from data.datahandler_master import Alpha158_Master
        master_config = Alpha158_Master.get_model_config()
        gate_input_start = master_config['gate_input_start_index']
        gate_input_end = master_config['gate_input_end_index']
        stock_feat_dim = master_config['n_stock_features']
        market_feat_dim = master_config['n_market_features']
        print(f"\n    Using MASTER config from handler:")
        print(f"    - Stock features: {stock_feat_dim}")
        print(f"    - Market features: {market_feat_dim}")
        print(f"    - Gate input range: [{gate_input_start}, {gate_input_end})")
    except Exception as e:
        print(f"    Warning: Could not load MASTER config from handler: {e}")
        # 使用默认值
        gate_input_start = 158
        gate_input_end = total_features
        stock_feat_dim = gate_input_start
        market_feat_dim = gate_input_end - gate_input_start

    # 构建模型参数
    model_params = {
        'd_feat': total_features,
        'gate_input_start_index': gate_input_start,
        'gate_input_end_index': gate_input_end,
        'stock_hidden_dim': args.stock_hidden or preset_config['stock_hidden_dim'],
        'stock_emb_dim': args.stock_emb or preset_config['stock_emb_dim'],
        'market_hidden_dim': args.market_hidden or preset_config['market_hidden_dim'],
        'market_emb_dim': args.market_emb or preset_config['market_emb_dim'],
        'fusion_hidden_dim': args.fusion_hidden or preset_config['fusion_hidden_dim'],
        'num_stock_layers': args.stock_layers or preset_config['num_stock_layers'],
        'num_market_layers': args.market_layers or preset_config['num_market_layers'],
        'dropout': args.dropout,
        'use_attention': not args.no_attention,
        'use_ic_loss': args.use_ic_loss,
        'ic_loss_weight': args.ic_loss_weight,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'early_stop': args.early_stop,
        'GPU': args.gpu if torch.cuda.is_available() else -1,
        'seed': args.seed,
    }

    print(f"\n[6] Model Configuration:")
    print(f"    Preset: {args.preset}")
    print(f"    Total features: {total_features}")
    print(f"    Stock features: {stock_feat_dim}")
    print(f"    Market features: {market_feat_dim}")
    print(f"    Stock encoder: hidden={model_params['stock_hidden_dim']}, emb={model_params['stock_emb_dim']}, layers={model_params['num_stock_layers']}")
    print(f"    Market encoder: hidden={model_params['market_hidden_dim']}, emb={model_params['market_emb_dim']}, layers={model_params['num_market_layers']}")
    print(f"    Fusion: hidden={model_params['fusion_hidden_dim']}, attention={model_params['use_attention']}")
    print(f"    Dropout: {model_params['dropout']}")
    print(f"    Learning rate: {model_params['lr']}")
    print(f"    Weight decay: {model_params['weight_decay']}")
    print(f"    Batch size: {model_params['batch_size']}")
    print(f"    Epochs: {model_params['n_epochs']}")
    print(f"    Early stop: {model_params['early_stop']}")
    print(f"    IC loss: {model_params['use_ic_loss']} (weight={model_params['ic_loss_weight']})")
    print(f"    GPU: {model_params['GPU']}")

    # 定义模型加载函数
    def load_model(path):
        return MASTERModel.load(str(path), GPU=args.gpu if torch.cuda.is_available() else -1)

    def get_feature_count(m):
        return m.d_feat

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
        print("\n[7] Training MASTER model...")

        # 创建模型
        model = MASTERModel(**model_params)

        # 训练
        model.fit(dataset)
        print("    Model training completed")

        # 保存模型
        print("\n[10] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_SAVE_PATH / f"master_{args.handler}_{args.stock_pool}_{args.nday}d.pt"
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
            model_name="MASTER",
            load_model_func=load_model,
            get_feature_count_func=get_feature_count
        )


if __name__ == "__main__":
    main()
