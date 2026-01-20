"""
CNN-AE-MLP V2 训练脚本

使用改进版 CNN-AE-MLP 模型训练 Alpha300/Alpha360 数据。

使用方法:
    # 基础训练
    python scripts/models/deep/run_cnn_ae_mlp_v2.py --handler alpha300

    # 使用 alpha300-ts (时序标准化)
    python scripts/models/deep/run_cnn_ae_mlp_v2.py --handler alpha300-ts --stock-pool sp500

    # 使用 IC 损失
    python scripts/models/deep/run_cnn_ae_mlp_v2.py --handler alpha300 --use-ic-loss

    # 轻量版 (更快训练)
    python scripts/models/deep/run_cnn_ae_mlp_v2.py --handler alpha300 --preset alpha300_lite

    # 完整训练 + 回测
    python scripts/models/deep/run_cnn_ae_mlp_v2.py --handler alpha300-ts --stock-pool sp500 --backtest
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

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

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from utils.utils import evaluate_model
from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG, PROJECT_ROOT, MODEL_SAVE_PATH,
    init_qlib, run_backtest,
)
from models.common.handlers import get_handler_class

from models.deep.cnn_ae_mlp_v2_model import CNNAEMLPV2, create_model, PRESET_CONFIGS


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
    parser = argparse.ArgumentParser(description='CNN-AE-MLP V2 Training')

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
                        choices=list(PRESET_CONFIGS.keys()),
                        help='Model preset configuration')
    parser.add_argument('--use-attention', action='store_true', default=True,
                        help='Use temporal attention')
    parser.add_argument('--no-attention', action='store_true',
                        help='Disable temporal attention')
    parser.add_argument('--use-feature-interaction', action='store_true', default=True,
                        help='Use feature interaction layer')
    parser.add_argument('--no-feature-interaction', action='store_true',
                        help='Disable feature interaction')
    parser.add_argument('--use-ic-loss', action='store_true',
                        help='Use IC loss function')
    parser.add_argument('--num-residual-blocks', type=int, default=2,
                        help='Number of residual blocks')

    # 训练参数
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Batch size (reduce to save GPU memory, e.g., 512 or 256)')
    parser.add_argument('--n-epochs', type=int, default=100)
    parser.add_argument('--early-stop', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    # 回测参数
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=1)
    parser.add_argument('--rebalance-freq', type=int, default=5)
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss'])

    args = parser.parse_args()

    # 处理 attention 和 feature interaction 参数
    use_attention = args.use_attention and not args.no_attention
    use_feature_interaction = args.use_feature_interaction and not args.no_feature_interaction

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    # 打印配置
    print("\n" + "=" * 70)
    print("CNN-AE-MLP V2 Training")
    print("=" * 70)
    print(f"Handler: {args.handler}")
    print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"N-day: {args.nday}")
    print(f"Preset: {args.preset}")
    print(f"Attention: {use_attention}")
    print(f"Feature Interaction: {use_feature_interaction}")
    print(f"IC Loss: {args.use_ic_loss}")
    print(f"Residual Blocks: {args.num_residual_blocks}")
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
        time_steps, features_per_step = 60, 5
    elif 'alpha360' in args.handler.lower():
        time_steps, features_per_step = 60, 6
    else:
        # 自动推断
        for ts in [60, 50, 40, 30, 20]:
            if num_columns % ts == 0:
                time_steps = ts
                features_per_step = num_columns // ts
                break
        else:
            time_steps = num_columns
            features_per_step = 1

    print(f"    Time steps: {time_steps}, Features per step: {features_per_step}")

    # 创建模型
    print("\n[*] Creating model...")
    model = create_model(
        preset=args.preset,
        num_columns=num_columns,
        time_steps=time_steps,
        features_per_step=features_per_step,
        use_attention=use_attention,
        use_feature_interaction=use_feature_interaction,
        use_ic_loss=args.use_ic_loss,
        num_residual_blocks=args.num_residual_blocks,
        lr=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        early_stop=args.early_stop,
        GPU=args.gpu,
        seed=args.seed,
    )

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
    print(f"    Valid IC: {valid_ic:.4f} (±{valid_ic_std:.4f})")
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
    print(f"    Test IC: {test_ic:.4f} (±{test_ic_std:.4f})")
    print(f"    Test ICIR: {test_icir:.4f}")

    # 使用 evaluate_model 显示详细指标
    evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

    # 保存模型
    print("\n[*] Saving model...")
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"cnn_ae_mlp_v2_{args.handler}_{args.stock_pool}_{args.nday}d_{timestamp}.keras"
    model_path = MODEL_SAVE_PATH / model_filename
    model.save(str(model_path))

    # 回测
    if args.backtest:
        print("\n[*] Running backtest...")
        pred_df = test_pred.to_frame("score")

        from tensorflow import keras

        def load_model_func(path):
            return keras.models.load_model(str(path), compile=False)

        def get_feature_count_func(m):
            return m.input_shape[1]

        run_backtest(
            model_path, dataset, pred_df, args, TIME_SPLITS,
            model_name="CNN-AE-MLP V2",
            load_model_func=load_model_func,
            get_feature_count_func=get_feature_count_func
        )

    # 汇总
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Valid IC: {valid_ic:.4f}, Test IC: {test_ic:.4f}")
    print(f"Model saved to: {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
