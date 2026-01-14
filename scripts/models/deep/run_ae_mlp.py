"""
运行 AE-MLP (Autoencoder-enhanced MLP) 模型

AE-MLP 是一种结合自编码器和多层感知机的模型，源自 Kaggle 金融竞赛。
通过自编码器辅助任务学习更鲁棒的特征表示，适合噪声大、信噪比低的金融数据。

使用方法:
    python scripts/models/deep/run_ae_mlp.py --stock-pool sp500 --handler alpha158 --backtest
    python scripts/models/deep/run_ae_mlp.py --stock-pool sp100 --handler alpha360 --n-epochs 200
    python scripts/models/deep/run_ae_mlp.py --model-path ./my_models/ae_mlp.keras --backtest
    python scripts/models/deep/run_ae_mlp.py --params-file ./outputs/hyperopt_cv/ae_mlp_cv_best_params_xxx.json
"""

import sys
import json
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

from models.deep.ae_mlp_model import AEMLP, create_ae_mlp_for_handler


# ============================================================================
# 默认 AE-MLP 参数
# ============================================================================

DEFAULT_AE_MLP_PARAMS = {
    'hidden_units': None,  # 根据 handler 自动配置
    'dropout_rates': None,
    'lr': 0.001,
    'batch_size': 4096,
    'loss_weights': {'decoder': 0.1, 'ae_action': 0.1, 'action': 1.0},
}


def load_params_from_file(params_file: str) -> dict:
    """
    从 JSON 文件加载 AE-MLP 参数

    Parameters
    ----------
    params_file : str
        参数文件路径 (hyperopt CV 输出的 JSON 文件)

    Returns
    -------
    dict
        AE-MLP 参数字典
    """
    with open(params_file, 'r') as f:
        data = json.load(f)

    # 支持两种格式:
    # 1. hyperopt CV 格式: {"params": {...}, "cv_results": {...}}
    # 2. 直接参数格式: {"hidden_units": [...], "lr": ...}
    if 'params' in data:
        params = data['params']
        print(f"    Loaded params from CV hyperopt file")
        if 'cv_results' in data:
            cv = data['cv_results']
            print(f"    CV Mean IC: {cv.get('mean_ic', 'N/A'):.4f} (±{cv.get('std_ic', 'N/A'):.4f})")
    else:
        params = data

    # 构建最终参数
    final_params = DEFAULT_AE_MLP_PARAMS.copy()

    # 网络结构参数
    if 'hidden_units' in params:
        final_params['hidden_units'] = params['hidden_units']
    if 'dropout_rates' in params:
        final_params['dropout_rates'] = params['dropout_rates']

    # 训练参数
    if 'lr' in params:
        final_params['lr'] = params['lr']
    if 'batch_size' in params:
        final_params['batch_size'] = params['batch_size']

    # 损失权重
    if 'loss_weights' in params:
        final_params['loss_weights'] = params['loss_weights']

    return final_params


def add_ae_mlp_args(parser):
    """添加 AE-MLP 特定参数"""
    parser.add_argument('--hidden-units', type=str, default=None,
                        help='Hidden units per layer, comma-separated (e.g., "96,96,512,256,128")')
    parser.add_argument('--dropout-rates', type=str, default=None,
                        help='Dropout rates per layer, comma-separated (e.g., "0.03,0.03,0.03,0.03,0.03")')
    parser.add_argument('--n-epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 0.001, or from params-file)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: 4096, or from params-file)')
    parser.add_argument('--early-stop', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--loss-decoder', type=float, default=None,
                        help='Loss weight for decoder (default: 0.1, or from params-file)')
    parser.add_argument('--loss-ae', type=float, default=None,
                        help='Loss weight for ae_action (default: 0.1, or from params-file)')
    parser.add_argument('--loss-main', type=float, default=None,
                        help='Loss weight for main action (default: 1.0, or from params-file)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--params-file', type=str, default=None,
                        help='Path to JSON file with AE-MLP params (from hyperopt CV search)')
    return parser


def parse_list_arg(arg_str, dtype=float):
    """解析逗号分隔的列表参数"""
    if arg_str is None:
        return None
    return [dtype(x.strip()) for x in arg_str.split(',')]


def main():
    # 解析命令行参数
    parser = create_argument_parser("AE-MLP", "run_ae_mlp.py")
    parser = add_ae_mlp_args(parser)
    args = parser.parse_args()

    # 加载参数文件 (如果提供)
    file_params = None
    if args.params_file:
        print(f"\n[*] Loading AE-MLP params from: {args.params_file}")
        file_params = load_params_from_file(args.params_file)

    # 解析列表参数 (命令行优先于文件)
    hidden_units = parse_list_arg(args.hidden_units, int)
    dropout_rates = parse_list_arg(args.dropout_rates, float)

    # 如果命令行没有指定，则使用文件参数
    if file_params:
        if hidden_units is None and file_params.get('hidden_units'):
            hidden_units = file_params['hidden_units']
        if dropout_rates is None and file_params.get('dropout_rates'):
            dropout_rates = file_params['dropout_rates']

    # 合并参数: 命令行 > 文件 > 默认值
    lr = args.lr if args.lr is not None else (
        file_params['lr'] if file_params else DEFAULT_AE_MLP_PARAMS['lr']
    )
    batch_size = args.batch_size if args.batch_size is not None else (
        file_params['batch_size'] if file_params else DEFAULT_AE_MLP_PARAMS['batch_size']
    )

    # 损失权重
    if file_params and 'loss_weights' in file_params:
        loss_weights = file_params['loss_weights'].copy()
    else:
        loss_weights = DEFAULT_AE_MLP_PARAMS['loss_weights'].copy()

    # 命令行覆盖损失权重
    if args.loss_decoder is not None:
        loss_weights['decoder'] = args.loss_decoder
    if args.loss_ae is not None:
        loss_weights['ae_action'] = args.loss_ae
    if args.loss_main is not None:
        loss_weights['action'] = args.loss_main

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = get_time_splits(args.max_train)

    # 打印头部信息
    print_training_header("AE-MLP", args, symbols, handler_config, time_splits)
    if args.params_file:
        print(f"Params File: {args.params_file}")

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
    print(f"    Hidden units: {hidden_units or 'auto (based on handler)'}")
    print(f"    Dropout rates: {dropout_rates or 'auto'}")
    print(f"    Learning rate: {lr}")
    print(f"    Batch size: {batch_size}")
    print(f"    Epochs: {args.n_epochs}")
    print(f"    Early stop: {args.early_stop}")
    print(f"    GPU: {args.gpu}")
    print(f"    Loss weights: decoder={loss_weights['decoder']}, ae={loss_weights['ae_action']}, main={loss_weights['action']}")

    # 定义模型加载函数
    def load_model(path):
        return AEMLP.load(str(path))

    def get_feature_count(m):
        return m.num_columns

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
        print("\n[7] Training AE-MLP model...")

        # 创建模型
        if hidden_units is not None or dropout_rates is not None:
            # 使用自定义参数 (来自命令行或参数文件)
            model = AEMLP(
                num_columns=total_features,
                hidden_units=hidden_units,
                dropout_rates=dropout_rates,
                lr=lr,
                n_epochs=args.n_epochs,
                batch_size=batch_size,
                early_stop=args.early_stop,
                loss_weights=loss_weights,
                GPU=args.gpu,
                seed=args.seed,
            )
        else:
            # 根据 handler 自动配置
            model = create_ae_mlp_for_handler(
                args.handler,
                lr=lr,
                n_epochs=args.n_epochs,
                batch_size=batch_size,
                early_stop=args.early_stop,
                loss_weights=loss_weights,
                GPU=args.gpu,
                seed=args.seed,
            )
            # 更新实际特征数
            model.num_columns = total_features

        # 训练
        model.fit(dataset)
        print("    ✓ Model training completed")

        # 保存模型
        print("\n[10] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_SAVE_PATH / f"ae_mlp_{args.handler}_{args.stock_pool}_{args.nday}d.keras"
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
            model_name="AE-MLP",
            load_model_func=load_model,
            get_feature_count_func=get_feature_count
        )


if __name__ == "__main__":
    main()
