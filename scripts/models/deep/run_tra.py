"""
运行 TRA (Temporal Routing Adaptor) + ALSTM 模型

TRA 使用多个独立 predictor + router 自动路由样本到不同 predictor，
直接对标 regime shift 问题。

使用方法:
    # 快速测试
    python scripts/models/deep/run_tra.py --stock-pool test --handler alpha300 --n-epochs 20

    # SP500 完整训练
    python scripts/models/deep/run_tra.py --stock-pool sp500 --handler alpha300 \
        --num-states 3 --pretrain --n-epochs 100 --early-stop 20

    # 不同 num_states 对比
    python scripts/models/deep/run_tra.py --stock-pool sp500 --handler alpha300 --num-states 5
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import torch
import numpy as np
import pandas as pd
from datetime import datetime

from qlib.contrib.model.pytorch_tra import TRAModel
from qlib.contrib.data.dataset import MTSDatasetH

from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG, MODEL_SAVE_PATH,
    create_argument_parser,
    get_time_splits,
    print_training_header,
    init_qlib,
    check_data_availability,
    create_data_handler,
    print_prediction_stats,
    run_backtest,
)
from models.common.ts_model_utils import resolve_d_feat_and_seq_len


def add_tra_args(parser):
    """添加 TRA 特定参数"""
    # Backbone (ALSTM) args
    parser.add_argument('--d-feat', type=int, default=5,
                        help='Base features per timestep (default: 5 for alpha300)')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Hidden size of backbone LSTM (default: 64)')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of backbone LSTM layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (default: 0.0)')

    # TRA 核心参数
    parser.add_argument('--num-states', type=int, default=3,
                        help='Number of predictors/trading patterns (default: 3)')
    parser.add_argument('--tra-hidden', type=int, default=32,
                        help='Router RNN hidden size (default: 32)')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='Gumbel softmax temperature (default: 1.0)')
    parser.add_argument('--src-info', type=str, default='LR_TPE',
                        choices=['LR', 'TPE', 'LR_TPE'],
                        help='Router input: LR=latent repr, TPE=temporal prediction error, LR_TPE=both')
    parser.add_argument('--transport-method', type=str, default='router',
                        choices=['none', 'router', 'oracle'],
                        help='Transport method (default: router)')
    parser.add_argument('--memory-mode', type=str, default='sample',
                        choices=['sample', 'daily'],
                        help='Memory mode (default: sample)')
    parser.add_argument('--lamb', type=float, default=1.0,
                        help='Router regularization weight (default: 1.0)')
    parser.add_argument('--rho', type=float, default=0.99,
                        help='Lambda decay rate (default: 0.99)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Current vs historical loss blend (default: 0.5)')
    parser.add_argument('--pretrain', action='store_true',
                        help='Pretrain backbone before TRA training')

    # 训练参数
    parser.add_argument('--n-epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Batch size (default: 1024, negative for daily batching)')
    parser.add_argument('--early-stop', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    return parser


def main():
    # 解析命令行参数
    parser = create_argument_parser("TRA-ALSTM", "run_tra.py")
    parser = add_tra_args(parser)
    args = parser.parse_args()

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = get_time_splits(args.max_train)

    # 打印头部信息
    print_training_header("TRA-ALSTM", args, symbols, handler_config, time_splits)

    # 初始化和数据准备
    init_qlib(handler_config['use_talib'])
    check_data_availability(time_splits)
    handler = create_data_handler(args, handler_config, symbols, time_splits)

    # 获取 d_feat 和 seq_len
    # 需要先从 handler 获取总特征数
    from qlib.data.dataset import DatasetH
    temp_dataset = DatasetH(
        handler=handler,
        segments={
            "train": (time_splits['train_start'], time_splits['train_end']),
            "valid": (time_splits['valid_start'], time_splits['valid_end']),
            "test": (time_splits['test_start'], time_splits['test_end']),
        },
    )
    actual_train_data = temp_dataset.prepare("train", col_set="feature")
    total_features = actual_train_data.shape[1]
    del temp_dataset, actual_train_data

    d_feat, seq_len = resolve_d_feat_and_seq_len(args.handler, total_features, args.d_feat)

    print(f"\n[4] Feature Configuration:")
    print(f"    Total features: {total_features}")
    print(f"    d_feat (per timestep): {d_feat}")
    print(f"    Sequence length: {seq_len}")

    # 创建 MTSDatasetH
    print(f"\n[5] Creating MTSDatasetH...")
    dataset = MTSDatasetH(
        handler=handler,
        segments={
            "train": (time_splits['train_start'], time_splits['train_end']),
            "valid": (time_splits['valid_start'], time_splits['valid_end']),
            "test": (time_splits['test_start'], time_splits['test_end']),
        },
        seq_len=seq_len,
        horizon=args.nday,
        input_size=d_feat,
        num_states=args.num_states,
        batch_size=args.batch_size,
        memory_mode=args.memory_mode,
        drop_last=True,
    )
    print(f"    ✓ MTSDatasetH created (seq_len={seq_len}, horizon={args.nday}, "
          f"num_states={args.num_states}, batch_size={args.batch_size})")

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = MODEL_SAVE_PATH / f"tra_{args.handler}_{args.stock_pool}_{args.nday}d_ns{args.num_states}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打印模型配置
    print(f"\n[6] Model Configuration:")
    print(f"    === Backbone (ALSTM) ===")
    print(f"    Input size: {d_feat}")
    print(f"    Hidden size: {args.hidden_size}")
    print(f"    Num layers: {args.num_layers}")
    print(f"    Dropout: {args.dropout}")
    print(f"    === TRA Router ===")
    print(f"    Num states: {args.num_states}")
    print(f"    Router hidden: {args.tra_hidden}")
    print(f"    Tau: {args.tau}")
    print(f"    Source info: {args.src_info}")
    print(f"    Transport method: {args.transport_method}")
    print(f"    Memory mode: {args.memory_mode}")
    print(f"    Lambda: {args.lamb} (decay: {args.rho})")
    print(f"    Alpha: {args.alpha}")
    print(f"    Pretrain: {args.pretrain}")
    print(f"    === Training ===")
    print(f"    Learning rate: {args.lr}")
    print(f"    Epochs: {args.n_epochs}")
    print(f"    Early stop: {args.early_stop}")
    print(f"    Seed: {args.seed}")
    print(f"    Output: {output_dir}")

    # GPU 设置
    if args.gpu >= 0 and torch.cuda.is_available():
        print(f"    GPU: cuda:{args.gpu}")
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        print(f"    GPU: CPU mode")

    # 创建模型
    print(f"\n[7] Training TRA-ALSTM model...")
    model = TRAModel(
        model_config={
            "input_size": d_feat,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "rnn_arch": "LSTM",
            "use_attn": True,
            "dropout": args.dropout,
        },
        tra_config={
            "num_states": args.num_states,
            "hidden_size": args.tra_hidden,
            "rnn_arch": "LSTM",
            "num_layers": 1,
            "tau": args.tau,
            "src_info": args.src_info,
        },
        model_type="RNN",
        lr=args.lr,
        n_epochs=args.n_epochs,
        early_stop=args.early_stop,
        lamb=args.lamb,
        rho=args.rho,
        alpha=args.alpha,
        transport_method=args.transport_method,
        memory_mode=args.memory_mode,
        pretrain=args.pretrain,
        eval_test=True,
        seed=args.seed,
        logdir=str(output_dir),
    )

    # 训练 — TRAModel.fit() 内部处理所有训练逻辑
    model.fit(dataset)
    print("    ✓ Model training completed")

    # 预测各 segment
    print(f"\n[8] Generating predictions...")

    test_pred = None
    for segment in ["train", "valid", "test"]:
        preds = model.predict(dataset, segment=segment)
        if isinstance(preds, pd.DataFrame) and "score" in preds.columns:
            score = preds["score"]
            label = preds["label"] if "label" in preds.columns else None

            print(f"\n    --- {segment.upper()} ---")
            print(f"    Samples: {len(score)}")
            print(f"    Score range: [{score.min():.4f}, {score.max():.4f}]")
            print(f"    Score NaN: {score.isna().sum()}")

            if label is not None and not label.isna().all():
                # 按天计算 IC
                ic_by_day = score.groupby(level="datetime").apply(
                    lambda x: x.corr(label.loc[x.index]) if len(x) > 1 else np.nan
                ).dropna()
                if len(ic_by_day) > 0:
                    print(f"    IC: {ic_by_day.mean():.4f} (std: {ic_by_day.std():.4f})")
                    print(f"    ICIR: {ic_by_day.mean() / ic_by_day.std():.4f}")
                    print(f"    IC > 0: {(ic_by_day > 0).mean():.1%}")

            if segment == "test":
                test_pred = score
                test_pred.name = "score"
        else:
            print(f"    WARNING: unexpected prediction format for {segment}")
            if segment == "test":
                test_pred = preds

    if test_pred is None:
        print("    ERROR: no test predictions generated")
        return
    print_prediction_stats(test_pred)

    # 额外：月度 IC 分析
    print(f"\n[9] Monthly IC Analysis (Test)...")
    test_preds_df = model.predict(dataset, segment="test")
    if isinstance(test_preds_df, pd.DataFrame) and "score" in test_preds_df.columns:
        score = test_preds_df["score"]
        label = test_preds_df["label"]

        ic_by_day = score.groupby(level="datetime").apply(
            lambda x: x.corr(label.loc[x.index]) if len(x) > 1 else np.nan
        ).dropna()

        if len(ic_by_day) > 0:
            ic_by_day.index = pd.to_datetime(ic_by_day.index)
            monthly_ic = ic_by_day.groupby(ic_by_day.index.to_period('M')).agg(['mean', 'std', 'count'])
            monthly_ic.columns = ['IC_mean', 'IC_std', 'days']
            monthly_ic['ICIR'] = monthly_ic['IC_mean'] / monthly_ic['IC_std']
            print(f"\n    {'Month':<12} {'IC':>8} {'ICIR':>8} {'Days':>6}")
            print(f"    {'-'*36}")
            for idx, row in monthly_ic.iterrows():
                print(f"    {str(idx):<12} {row['IC_mean']:>8.4f} {row['ICIR']:>8.4f} {int(row['days']):>6}")

    # 回测
    if args.backtest:
        pred_df = test_pred.to_frame("score")

        # TRA model 用 logdir 保存，构造兼容的 load 函数
        model_bin_path = output_dir / "model.bin"

        def load_tra_model(path):
            return model  # 直接返回已训练的模型

        def get_feature_count(m):
            return total_features

        run_backtest(
            model_bin_path, dataset, pred_df, args, time_splits,
            model_name="TRA-ALSTM",
            load_model_func=load_tra_model,
            get_feature_count_func=get_feature_count,
        )

    print(f"\n{'='*70}")
    print(f"✓ TRA-ALSTM Training Complete")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
