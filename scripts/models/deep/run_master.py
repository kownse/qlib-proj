"""
运行 MASTER 模型

MASTER (Market-Guided Stock Transformer) 使用市场信息来引导个股预测。
通过门控机制融合市场状态和个股特征，实现市场感知的股票预测。

特点:
1. 3D 时间序列输入: (N, T, F) = (股票数, 时间步, 特征数)
2. TAttention: 时间维度自注意力 (intra-stock)
3. SAttention: 股票维度自注意力 (inter-stock, 跨股票建模)
4. 市场引导门控: 用市场状态的 softmax 权重调制股票特征

使用方法:
    # 基本用法 (使用 US 市场配置)
    python scripts/models/deep/run_master.py --stock-pool sp500 --handler alpha158-master

    # 完整训练 + 回测
    python scripts/models/deep/run_master.py --stock-pool sp500 --handler alpha158-master --backtest

    # 自定义参数
    python scripts/models/deep/run_master.py --stock-pool sp500 --handler alpha158-master \
        --d-model 256 --seq-len 8 --beta 5 --n-epochs 40

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
    run_backtest,
)

from qlib.contrib.data.dataset import MTSDatasetH

from models.deep.master_model import MASTERModel, PRESET_CONFIGS


def add_master_args(parser):
    """添加 MASTER 特定参数"""
    # 模型架构参数
    parser.add_argument('--d-model', type=int, default=256,
                        help='Hidden dimension (default: 256)')
    parser.add_argument('--t-nhead', type=int, default=4,
                        help='Number of temporal attention heads (default: 4)')
    parser.add_argument('--s-nhead', type=int, default=2,
                        help='Number of stock attention heads (default: 2)')
    parser.add_argument('--seq-len', type=int, default=8,
                        help='Sequence length / lookback window (default: 8)')
    parser.add_argument('--beta', type=float, default=5.0,
                        help='Gate temperature parameter (default: 5.0)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate for attention layers (default: 0.5)')

    # 训练参数
    parser.add_argument('--n-epochs', type=int, default=40,
                        help='Number of training epochs (default: 40)')
    parser.add_argument('--lr', type=float, default=8e-6,
                        help='Learning rate (default: 8e-6)')
    parser.add_argument('--early-stop', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--train-stop-loss', type=float, default=None,
                        help='Stop training when loss reaches this threshold')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # 调试参数
    parser.add_argument('--debug', action='store_true',
                        help='Enable detailed debug output for first training iteration')

    return parser


def main():
    # 解析命令行参数
    parser = create_argument_parser("MASTER", "run_master.py")
    parser = add_master_args(parser)
    args = parser.parse_args()

    # 检查 handler (MASTER 需要包含市场信息的 handler)
    if 'master' not in args.handler:
        print(f"Warning: MASTER model is designed for *-master handlers, but got {args.handler}")
        print("         Consider using alpha158-master for best results.")

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = get_time_splits(args.max_train)

    # 打印头部信息
    print_training_header("MASTER", args, symbols, handler_config, time_splits)

    # 初始化 Qlib
    init_qlib(handler_config['use_talib'])
    check_data_availability(time_splits)

    # 创建 DataHandler
    handler = create_data_handler(args, handler_config, symbols, time_splits)

    # 从 handler 获取 MASTER 配置
    try:
        from data.datahandler_master import Alpha158_Master
        master_config = Alpha158_Master.get_model_config()
        d_feat = master_config['n_stock_features']
        gate_input_start = master_config['gate_input_start_index']
        gate_input_end = master_config['gate_input_end_index']
        print(f"\n[5] MASTER Configuration from handler:")
        print(f"    Stock features (d_feat): {d_feat}")
        print(f"    Market features: {gate_input_end - gate_input_start}")
        print(f"    Gate input range: [{gate_input_start}, {gate_input_end})")
    except Exception as e:
        print(f"    Warning: Could not load MASTER config from handler: {e}")
        # 使用 US market 默认值
        d_feat = 142
        gate_input_start = 142
        gate_input_end = 205

    # Debug: 检查原始 handler 数据
    print(f"\n[5.1] Raw handler data statistics:")
    try:
        raw_features = handler.fetch(col_set="feature")
        raw_labels = handler.fetch(col_set="label")
        print(f"    Feature shape: {raw_features.shape}")
        print(f"    Label shape: {raw_labels.shape}")
        print(f"    Feature columns: {len(raw_features.columns)}")

        # Feature statistics
        feat_vals = raw_features.values
        print(f"\n    Feature stats:")
        print(f"      mean={np.nanmean(feat_vals):.6f}, std={np.nanstd(feat_vals):.6f}")
        print(f"      min={np.nanmin(feat_vals):.6f}, max={np.nanmax(feat_vals):.6f}")
        print(f"      NaN%: {np.isnan(feat_vals).mean()*100:.2f}%")

        # Check for extreme values
        extreme_mask = np.abs(feat_vals) > 100
        if extreme_mask.any():
            print(f"      WARNING: {extreme_mask.sum()} values with |x| > 100 ({extreme_mask.mean()*100:.4f}%)")

        # Stock features vs Market features
        if raw_features.shape[1] >= gate_input_end:
            stock_vals = feat_vals[:, :d_feat]
            market_vals = feat_vals[:, d_feat:gate_input_end]
            print(f"\n    Stock features [0:{d_feat}]:")
            print(f"      mean={np.nanmean(stock_vals):.6f}, std={np.nanstd(stock_vals):.6f}")
            print(f"      min={np.nanmin(stock_vals):.6f}, max={np.nanmax(stock_vals):.6f}")
            print(f"    Market features [{d_feat}:{gate_input_end}]:")
            print(f"      mean={np.nanmean(market_vals):.6f}, std={np.nanstd(market_vals):.6f}")
            print(f"      min={np.nanmin(market_vals):.6f}, max={np.nanmax(market_vals):.6f}")

        # Label statistics
        label_vals = raw_labels.values
        print(f"\n    Label stats:")
        print(f"      mean={np.nanmean(label_vals):.6f}, std={np.nanstd(label_vals):.6f}")
        print(f"      min={np.nanmin(label_vals):.6f}, max={np.nanmax(label_vals):.6f}")
        print(f"      NaN%: {np.isnan(label_vals).mean()*100:.2f}%")

    except Exception as e:
        print(f"    Warning: Could not fetch raw handler data: {e}")

    # 创建 MTSDatasetH (时间序列数据集)
    print(f"\n[6] Creating time-series dataset...")
    print(f"    seq_len: {args.seq_len}")
    print(f"    This enables (N, T, F) format for MASTER model")

    dataset = MTSDatasetH(
        handler=handler,
        segments={
            "train": (time_splits['train_start'], time_splits['train_end']),
            "valid": (time_splits['valid_start'], time_splits['valid_end']),
            "test": (time_splits['test_start'], time_splits['test_end']),
        },
        seq_len=args.seq_len,
        horizon=args.nday,  # prediction horizon
        batch_size=-1,  # -1 means daily sampling (required for cross-stock attention)
        shuffle=True,
        drop_last=True,
    )

    # 准备数据
    dl_train = dataset.prepare("train")
    dl_valid = dataset.prepare("valid")
    dl_test = dataset.prepare("test")

    # 检查数据格式和统计信息
    print(f"\n[7] Data format check and statistics:")
    batch_count = 0
    all_data_stats = {'mean': [], 'std': [], 'min': [], 'max': [], 'nan_pct': []}
    all_label_stats = {'mean': [], 'std': [], 'min': [], 'max': [], 'nan_pct': []}

    for batch in dl_train:
        if isinstance(batch, dict):
            data = batch['data']
            label = batch['label']

            if batch_count == 0:
                print(f"    Data shape: {data.shape}")
                print(f"    Label shape: {label.shape}")
                print(f"    Expected: data=(N_stocks, T={args.seq_len}, F={gate_input_end}), label=(N_stocks,)")

                # Detailed first batch analysis
                print(f"\n    [First batch detailed analysis]")
                print(f"    Data dtype: {data.dtype}")
                print(f"    Label dtype: {label.dtype}")

                # Overall data statistics (move to CPU if on GPU)
                data_np = data.cpu().numpy() if data.is_cuda else data.numpy()
                label_np = label.cpu().numpy() if label.is_cuda else label.numpy()
                print(f"\n    Data overall: mean={np.nanmean(data_np):.6f}, std={np.nanstd(data_np):.6f}, "
                      f"min={np.nanmin(data_np):.6f}, max={np.nanmax(data_np):.6f}")
                print(f"    Data NaN%: {np.isnan(data_np).mean()*100:.2f}%")

                # Stock features vs Market features
                stock_feat = data_np[:, :, :d_feat]
                market_feat = data_np[:, :, d_feat:gate_input_end]
                print(f"\n    Stock features [0:{d_feat}]:")
                print(f"      mean={np.nanmean(stock_feat):.6f}, std={np.nanstd(stock_feat):.6f}, "
                      f"min={np.nanmin(stock_feat):.6f}, max={np.nanmax(stock_feat):.6f}")
                print(f"      NaN%: {np.isnan(stock_feat).mean()*100:.2f}%")

                print(f"\n    Market features [{d_feat}:{gate_input_end}]:")
                print(f"      mean={np.nanmean(market_feat):.6f}, std={np.nanstd(market_feat):.6f}, "
                      f"min={np.nanmin(market_feat):.6f}, max={np.nanmax(market_feat):.6f}")
                print(f"      NaN%: {np.isnan(market_feat).mean()*100:.2f}%")

                # Check last timestep market features (used for gate)
                gate_input = data_np[:, -1, d_feat:gate_input_end]
                print(f"\n    Gate input (last timestep market features):")
                print(f"      shape: {gate_input.shape}")
                print(f"      mean={np.nanmean(gate_input):.6f}, std={np.nanstd(gate_input):.6f}, "
                      f"min={np.nanmin(gate_input):.6f}, max={np.nanmax(gate_input):.6f}")

                # Label statistics
                print(f"\n    Label: mean={np.nanmean(label_np):.6f}, std={np.nanstd(label_np):.6f}, "
                      f"min={np.nanmin(label_np):.6f}, max={np.nanmax(label_np):.6f}")
                print(f"    Label NaN%: {np.isnan(label_np).mean()*100:.2f}%")

            # Collect statistics for aggregation (move to CPU if on GPU)
            data_np = data.cpu().numpy() if data.is_cuda else data.numpy()
            label_np = label.cpu().numpy() if label.is_cuda else label.numpy()
            all_data_stats['mean'].append(np.nanmean(data_np))
            all_data_stats['std'].append(np.nanstd(data_np))
            all_data_stats['nan_pct'].append(np.isnan(data_np).mean()*100)
            all_label_stats['mean'].append(np.nanmean(label_np))
            all_label_stats['std'].append(np.nanstd(label_np))
            all_label_stats['nan_pct'].append(np.isnan(label_np).mean()*100)

            batch_count += 1
            if batch_count >= 10:  # Sample first 10 batches for statistics
                break
        else:
            data = torch.squeeze(batch, dim=0)
            print(f"    Data shape: {data.shape}")
            break

    if batch_count > 1:
        print(f"\n    [Aggregated statistics over {batch_count} batches]")
        print(f"    Data mean: avg={np.mean(all_data_stats['mean']):.6f}, std across batches={np.std(all_data_stats['mean']):.6f}")
        print(f"    Data std: avg={np.mean(all_data_stats['std']):.6f}")
        print(f"    Data NaN%: avg={np.mean(all_data_stats['nan_pct']):.2f}%")
        print(f"    Label mean: avg={np.mean(all_label_stats['mean']):.6f}, std across batches={np.std(all_label_stats['mean']):.6f}")
        print(f"    Label std: avg={np.mean(all_label_stats['std']):.6f}")

    # 构建模型参数
    model_params = {
        'd_feat': d_feat,
        'd_model': args.d_model,
        't_nhead': args.t_nhead,
        's_nhead': args.s_nhead,
        'gate_input_start_index': gate_input_start,
        'gate_input_end_index': gate_input_end,
        'T_dropout_rate': args.dropout,
        'S_dropout_rate': args.dropout,
        'beta': args.beta,
        'seq_len': args.seq_len,
        'n_epochs': args.n_epochs,
        'lr': args.lr,
        'early_stop': args.early_stop,
        'train_stop_loss_thred': args.train_stop_loss,
        'GPU': args.gpu if torch.cuda.is_available() else -1,
        'seed': args.seed,
    }

    print(f"\n[8] Model Configuration:")
    print(f"    d_feat: {model_params['d_feat']}")
    print(f"    d_model: {model_params['d_model']}")
    print(f"    t_nhead: {model_params['t_nhead']}, s_nhead: {model_params['s_nhead']}")
    print(f"    seq_len: {model_params['seq_len']}")
    print(f"    beta: {model_params['beta']}")
    print(f"    dropout: {model_params['T_dropout_rate']}")
    print(f"    lr: {model_params['lr']}")
    print(f"    n_epochs: {model_params['n_epochs']}")
    print(f"    early_stop: {model_params['early_stop']}")
    print(f"    debug: {args.debug}")

    # 定义模型加载函数
    def load_model(path):
        return MASTERModel.load(str(path), GPU=args.gpu if torch.cuda.is_available() else -1)

    def get_feature_count(m):
        return m.d_feat

    # 检查是否提供了预训练模型路径
    if args.model_path:
        # 加载预训练模型
        model_path = Path(args.model_path)
        print(f"\n[9] Loading pre-trained model from: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = load_model(model_path)
        print("    Model loaded successfully")
    else:
        # 训练模型
        print("\n[9] Training MASTER model...")
        model = MASTERModel(**model_params)
        model.fit(dl_train, dl_valid, debug=args.debug)
        print("    Model training completed")

        # 保存模型
        print("\n[10] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_SAVE_PATH / f"master_{args.handler}_{args.stock_pool}_{args.nday}d.pt"
        model.save(str(model_path))

    # 预测
    print("\n[11] Generating predictions...")
    test_pred = model.predict(dl_test)
    print(f"    Prediction shape: {test_pred.shape}")
    print(f"    Prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    # 评估
    print("\n[12] Evaluation...")

    # 获取标签用于评估
    # MTSDatasetH 的标签需要从原始 handler 获取
    try:
        test_label = handler.fetch(col_set="label")
        test_label = test_label.loc[time_splits['test_start']:time_splits['test_end']]

        # 对齐预测和标签
        common_idx = test_pred.index.intersection(test_label.index)
        pred_aligned = test_pred.loc[common_idx]
        label_aligned = test_label.loc[common_idx]

        if isinstance(label_aligned, pd.DataFrame):
            label_aligned = label_aligned.iloc[:, 0]

        # 计算 IC
        from scipy import stats
        valid_mask = ~label_aligned.isna()
        if valid_mask.sum() > 0:
            ic, _ = stats.spearmanr(
                pred_aligned[valid_mask].values,
                label_aligned[valid_mask].values
            )
            print(f"\n    Overall Test IC: {ic:.4f}")
    except Exception as e:
        print(f"    Warning: Could not compute evaluation metrics: {e}")

    # 回测
    if args.backtest:
        pred_df = test_pred.to_frame("score")

        # 由于 MTSDatasetH 格式不同，需要调整回测逻辑
        print("\n[13] Running backtest...")
        try:
            run_backtest(
                model_path, dataset, pred_df, args, time_splits,
                model_name="MASTER",
                load_model_func=load_model,
                get_feature_count_func=get_feature_count
            )
        except Exception as e:
            print(f"    Warning: Backtest failed: {e}")
            print("    This may be due to MTSDatasetH format differences.")


if __name__ == "__main__":
    main()
