"""
Transformer V2 Improved 训练脚本

使用改进的 Transformer 模型，针对60天时序数据优化：
- Time2Vec 可学习位置编码
- 局部时序卷积
- CLS Token 聚合
- 相对位置偏置

使用方法:
    # 基础训练
    python scripts/models/deep/run_transformer_v2_improved.py --handler alpha300

    # 完整训练 + 回测
    python scripts/models/deep/run_transformer_v2_improved.py --handler alpha300 --stock-pool sp500 --backtest

    # 加载已有模型并运行回测
    python scripts/models/deep/run_transformer_v2_improved.py --handler alpha300 --stock-pool sp500 --backtest \
        --model-path my_models/transformer_v2_improved_alpha300_sp500_5d_20260122_141429.pt

    # 禁用某些特性进行对比实验
    python scripts/models/deep/run_transformer_v2_improved.py --no-local-conv --no-relative-bias
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))
project_root = script_dir.parent

import qlib
from qlib.constant import REG_US
from utils.talib_ops import TALIB_OPS

qlib_data_path = project_root / "my_data" / "qlib_us"
qlib.init(
    provider_uri=str(qlib_data_path),
    region=REG_US,
    custom_ops=TALIB_OPS,
)

import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from utils.utils import evaluate_model
from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG, PROJECT_ROOT, MODEL_SAVE_PATH,
    init_qlib, run_backtest,
)
from models.common.handlers import get_handler_class

from models.deep.transformer_v2_model import TransformerModelV2, create_model_v2, PRESET_CONFIGS_V2


# 时间配置
TIME_SPLITS = {
    'train_start': '2000-01-01',
    'train_end': '2024-09-30',
    'valid_start': '2024-10-01',
    'valid_end': '2024-12-31',
    'test_start': '2025-01-01',
    'test_end': '2025-12-31',
}


def compute_ic(pred, label, index):
    """计算 IC"""
    df = pd.DataFrame({'pred': pred, 'label': label}, index=index)
    ic_by_date = df.groupby(level='datetime').apply(
        lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
    )
    ic_by_date = ic_by_date.dropna()
    if len(ic_by_date) == 0:
        return 0.0, 0.0, 0.0
    mean_ic = ic_by_date.mean()
    ic_std = ic_by_date.std()
    icir = mean_ic / ic_std if ic_std > 0 else 0
    return mean_ic, ic_std, icir


def main():
    parser = argparse.ArgumentParser(description='Transformer V2 Improved Training')

    # 数据参数
    parser.add_argument('--handler', type=str, default='alpha300',
                        choices=list(HANDLER_CONFIG.keys()),
                        help='Handler type')
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--nday', type=int, default=5,
                        help='Prediction horizon (days)')

    # 模型参数
    parser.add_argument('--preset', type=str, default='alpha300',
                        choices=list(PRESET_CONFIGS_V2.keys()),
                        help='Model preset configuration')
    parser.add_argument('--d-model', type=int, default=None,
                        help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=None,
                        help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Number of transformer layers')
    parser.add_argument('--dim-feedforward', type=int, default=None,
                        help='FFN hidden dimension')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate')
    parser.add_argument('--use-ic-loss', action='store_true',
                        help='Use IC loss function')

    # V2 特有参数
    parser.add_argument('--no-local-conv', action='store_true',
                        help='Disable local temporal convolution')
    parser.add_argument('--no-relative-bias', action='store_true',
                        help='Disable relative position bias')

    # 训练参数
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--n-epochs', type=int, default=100)
    parser.add_argument('--early-stop', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    # 回测参数
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to a pre-trained model. If provided, skip training and run backtest directly')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=1)
    parser.add_argument('--rebalance-freq', type=int, default=5)
    parser.add_argument('--account', type=float, default=1000000,
                        help='Initial account value for backtest')
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss'])

    # 动态风险策略参数
    parser.add_argument('--risk-lookback', type=int, default=20,
                        help='Lookback days for volatility/momentum calculation')
    parser.add_argument('--drawdown-threshold', type=float, default=-0.10,
                        help='[dynamic_risk] Drawdown threshold for high risk')
    parser.add_argument('--momentum-threshold', type=float, default=0.03,
                        help='[dynamic_risk] Momentum threshold for trend detection')
    parser.add_argument('--risk-high', type=float, default=0.50,
                        help='Position ratio at high risk/volatility')
    parser.add_argument('--risk-medium', type=float, default=0.75,
                        help='Position ratio at medium risk/volatility')
    parser.add_argument('--risk-normal', type=float, default=0.95,
                        help='Normal position ratio')
    parser.add_argument('--market-proxy', type=str, default='AAPL',
                        help='Market proxy symbol for risk calculation')

    # vol_stoploss 策略特有参数
    parser.add_argument('--vol-high', type=float, default=0.35,
                        help='[vol_stoploss] High volatility threshold (annualized)')
    parser.add_argument('--vol-medium', type=float, default=0.25,
                        help='[vol_stoploss] Medium volatility threshold (annualized)')
    parser.add_argument('--stop-loss', type=float, default=-0.15,
                        help='[vol_stoploss] Stop loss threshold per stock')
    parser.add_argument('--no-sell-after-drop', type=float, default=-0.20,
                        help='[vol_stoploss] Do not sell if already dropped more than this')

    args = parser.parse_args()

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    # 打印配置
    print("\n" + "=" * 70)
    if args.model_path:
        print("Transformer V2 Improved - Load & Backtest")
        print("=" * 70)
        print(f"Model Path: {args.model_path}")
    else:
        print("Transformer V2 Improved Training")
        print("=" * 70)
    print(f"Handler: {args.handler}")
    print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"N-day: {args.nday}")
    print(f"Preset: {args.preset}")
    print(f"IC Loss: {args.use_ic_loss}")
    print(f"Local Conv: {not args.no_local_conv}")
    print(f"Relative Bias: {not args.no_relative_bias}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size}")
    print(f"GPU: {args.gpu}")
    print("=" * 70)

    # 初始化
    init_qlib(handler_config['use_talib'])

    # 创建数据集
    print("\n[*] Creating dataset...")
    HandlerClass = get_handler_class(args.handler)
    handler = HandlerClass(
        volatility_window=args.nday,
        instruments=symbols,
        start_time=TIME_SPLITS['train_start'],
        end_time=TIME_SPLITS['test_end'],
        fit_start_time=TIME_SPLITS['train_start'],
        fit_end_time=TIME_SPLITS['train_end'],
        infer_processors=[],
    )

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (TIME_SPLITS['train_start'], TIME_SPLITS['train_end']),
            "valid": (TIME_SPLITS['valid_start'], TIME_SPLITS['valid_end']),
            "test": (TIME_SPLITS['test_start'], TIME_SPLITS['test_end']),
        }
    )

    # 获取特征数量
    features = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
    num_columns = features.shape[1]
    print(f"    Feature count: {num_columns}")

    # 确定时序维度
    if 'alpha300' in args.handler.lower():
        seq_len, d_feat = 60, 5
    elif 'alpha360' in args.handler.lower():
        seq_len, d_feat = 60, 6
    else:
        # 自动推断
        for ts in [60, 50, 40, 30, 20]:
            if num_columns % ts == 0:
                seq_len = ts
                d_feat = num_columns // ts
                break
        else:
            seq_len = num_columns
            d_feat = 1

    # 如果特征数不匹配，调整 d_feat
    if num_columns % seq_len == 0:
        d_feat = num_columns // seq_len

    print(f"    Sequence length: {seq_len}, Features per step: {d_feat}")

    # 根据是否提供 model-path 决定加载还是训练
    if args.model_path:
        # 加载已有模型
        print(f"\n[*] Loading pre-trained model from: {args.model_path}")
        model = TransformerModelV2.load(args.model_path, GPU=args.gpu)
        model_path = Path(args.model_path)

        # 打印模型配置
        print(f"    d_model: {model.d_model}")
        print(f"    nhead: {model.nhead}")
        print(f"    num_layers: {model.num_layers}")
        print(f"    dim_feedforward: {model.dim_feedforward}")
        print(f"    dropout: {model.dropout}")
        print(f"    seq_len: {model.seq_len}")
        print(f"    d_feat: {model.d_feat}")
    else:
        # 创建模型参数
        model_kwargs = {
            'd_feat': d_feat,
            'seq_len': seq_len,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'early_stop': args.early_stop,
            'use_ic_loss': args.use_ic_loss,
            'use_local_conv': not args.no_local_conv,
            'use_relative_bias': not args.no_relative_bias,
            'GPU': args.gpu,
            'seed': args.seed,
        }

        # 添加可选参数
        if args.d_model is not None:
            model_kwargs['d_model'] = args.d_model
        if args.nhead is not None:
            model_kwargs['nhead'] = args.nhead
        if args.num_layers is not None:
            model_kwargs['num_layers'] = args.num_layers
        if args.dim_feedforward is not None:
            model_kwargs['dim_feedforward'] = args.dim_feedforward
        if args.dropout is not None:
            model_kwargs['dropout'] = args.dropout

        # 创建模型
        print("\n[*] Creating model...")
        model = create_model_v2(preset=args.preset, **model_kwargs)

        # 打印模型配置
        print(f"    d_model: {model.d_model}")
        print(f"    nhead: {model.nhead}")
        print(f"    num_layers: {model.num_layers}")
        print(f"    dim_feedforward: {model.dim_feedforward}")
        print(f"    dropout: {model.dropout}")

        # 训练
        history = model.fit(dataset)

    # 验证集评估
    print("\n[*] Evaluating on validation set...")
    valid_pred = model.predict(dataset, "valid")
    valid_labels = dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(valid_labels, pd.DataFrame):
        valid_labels = valid_labels.iloc[:, 0]

    valid_ic, valid_ic_std, valid_icir = compute_ic(
        valid_pred.values, valid_labels.values, valid_pred.index
    )
    print(f"    Valid IC: {valid_ic:.4f} (+-{valid_ic_std:.4f})")
    print(f"    Valid ICIR: {valid_icir:.4f}")

    # 测试集评估
    print("\n[*] Evaluating on test set (2025)...")
    test_pred = model.predict(dataset, "test")
    test_labels = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(test_labels, pd.DataFrame):
        test_labels = test_labels.iloc[:, 0]

    test_ic, test_ic_std, test_icir = compute_ic(
        test_pred.values, test_labels.values, test_pred.index
    )
    print(f"    Test IC: {test_ic:.4f} (+-{test_ic_std:.4f})")
    print(f"    Test ICIR: {test_icir:.4f}")

    # 使用 evaluate_model 显示详细指标
    evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

    # 保存模型（仅当训练新模型时）
    if not args.model_path:
        print("\n[*] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"transformer_v2_improved_{args.handler}_{args.stock_pool}_{args.nday}d_{timestamp}.pt"
        model_path = MODEL_SAVE_PATH / model_filename
        model.save(str(model_path))

    # 回测
    if args.backtest:
        print("\n[*] Running backtest...")
        pred_df = test_pred.to_frame("score")

        def load_model_func(path):
            return TransformerModelV2.load(str(path), GPU=args.gpu)

        def get_feature_count_func(m):
            return m.seq_len * m.d_feat

        run_backtest(
            model_path, dataset, pred_df, args, TIME_SPLITS,
            model_name="Transformer V2 Improved",
            load_model_func=load_model_func,
            get_feature_count_func=get_feature_count_func
        )

    # 汇总
    print("\n" + "=" * 70)
    if args.model_path:
        print("BACKTEST COMPLETE")
    else:
        print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Valid IC: {valid_ic:.4f}, Test IC: {test_ic:.4f}")
    print(f"Model: {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
