"""
TCN Cross-Validation 训练脚本 - 复现 Qlib Benchmark Alpha158 配置

复现 qlib-src/examples/benchmarks/TCN/workflow_config_tcn_Alpha158.yaml
在 US 数据上使用时间序列交叉验证训练 TCN 模型。

关键配置:
    - 20 个特征 (从 Alpha158 筛选)
    - step_len = 20 (时间序列长度)
    - d_feat = 20 (每个时间步的特征数)
    - TSDatasetH (时间序列数据集)

使用方法:
    # 基础 CV 训练
    python scripts/models/deep/run_tcn_qlib_alpha158_cv.py --stock-pool sp500

    # CV 训练并回测
    python scripts/models/deep/run_tcn_qlib_alpha158_cv.py --stock-pool sp500 --backtest

    # 只运行 CV 训练，不训练最终模型
    python scripts/models/deep/run_tcn_qlib_alpha158_cv.py --stock-pool sp500 --cv-only

    # 评估已训练模型
    python scripts/models/deep/run_tcn_qlib_alpha158_cv.py --stock-pool sp500 --eval-only \
        --model-path my_models/tcn_qlib_alpha158_cv_xxx.pt
"""

import os
import sys
from pathlib import Path

# 设置环境变量
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))
project_root = script_dir.parent

import argparse
import copy
from datetime import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP

from utils.utils import evaluate_model
from data.stock_pools import STOCK_POOLS
from data.datahandler_tcn_v1 import TCN_V1_Handler

from models.common import (
    PROJECT_ROOT, MODEL_SAVE_PATH,
    init_qlib,
    run_backtest,
    # CV utilities
    CV_FOLDS,
    FINAL_TEST,
)

# 导入 qlib 的 TCN 模型组件
from qlib.contrib.model.tcn import TemporalConvNet


# ============================================================================
# TCN 模型定义
# ============================================================================

class TCNModel(nn.Module):
    """TCN 模型 - 与 Qlib benchmark 一致"""

    def __init__(self, num_input, output_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.num_input = num_input
        self.tcn = TemporalConvNet(num_input, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        output = self.tcn(x)
        # 取最后一个时间步的输出
        output = self.linear(output[:, :, -1])
        return output.squeeze()


class TCNTrainer:
    """TCN 训练器 - 复现 Qlib benchmark 训练逻辑"""

    def __init__(
        self,
        d_feat=20,
        n_chans=32,
        kernel_size=7,
        num_layers=5,
        dropout=0.5,
        n_epochs=200,
        lr=1e-4,
        batch_size=2000,
        early_stop=20,
        gpu=0,
        seed=None,
    ):
        self.d_feat = d_feat
        self.n_chans = n_chans
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.seed = seed

        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        self.model = None
        self.fitted = False

    def _init_model(self):
        """初始化模型"""
        self.model = TCNModel(
            num_input=self.d_feat,
            output_size=1,
            num_channels=[self.n_chans] * self.num_layers,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _mse_loss(self, pred, label):
        """MSE 损失函数"""
        mask = ~torch.isnan(label)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        return ((pred[mask] - label[mask]) ** 2).mean()

    def _train_epoch(self, data_loader):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0

        for data in data_loader:
            # data shape: (batch, step_len, features+1)  最后一列是 label
            data = data.to(self.device).float()

            # 转置为 (batch, features+1, step_len)
            data = torch.transpose(data, 1, 2)

            # 分离特征和标签
            feature = data[:, :-1, :]  # (batch, features, step_len)
            label = data[:, -1, -1]    # 最后一个时间步的 label

            pred = self.model(feature)
            loss = self._mse_loss(pred, label)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _eval_epoch(self, data_loader):
        """评估一个 epoch"""
        self.model.eval()
        preds = []
        labels = []

        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device).float()
                data = torch.transpose(data, 1, 2)

                feature = data[:, :-1, :]
                label = data[:, -1, -1]

                pred = self.model(feature)
                preds.append(pred.cpu().numpy())
                labels.append(label.cpu().numpy())

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

        # 计算损失
        mask = ~np.isnan(labels)
        if mask.sum() == 0:
            return float('inf'), preds

        loss = np.mean((preds[mask] - labels[mask]) ** 2)
        return loss, preds

    def fit(self, train_loader, valid_loader, verbose=True):
        """训练模型"""
        self._init_model()

        best_loss = float('inf')
        best_epoch = 0
        best_params = None
        stop_steps = 0

        for epoch in range(self.n_epochs):
            train_loss = self._train_epoch(train_loader)
            valid_loss, _ = self._eval_epoch(valid_loader)

            if verbose:
                print(f"    Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}")

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_params = copy.deepcopy(self.model.state_dict())
                stop_steps = 0
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    if verbose:
                        print(f"    Early stop at epoch {epoch+1}")
                    break

        if best_params is not None:
            self.model.load_state_dict(best_params)

        self.fitted = True
        return best_epoch + 1, best_loss

    def predict(self, data_loader):
        """预测"""
        if not self.fitted:
            raise ValueError("Model not fitted yet")

        self.model.eval()
        preds = []

        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device).float()
                data = torch.transpose(data, 1, 2)
                feature = data[:, :-1, :]
                pred = self.model(feature)
                preds.append(pred.cpu().numpy())

        return np.concatenate(preds)

    def save(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'd_feat': self.d_feat,
                'n_chans': self.n_chans,
                'kernel_size': self.kernel_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
            }
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        config = checkpoint['config']

        self.d_feat = config['d_feat']
        self.n_chans = config['n_chans']
        self.kernel_size = config['kernel_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']

        self._init_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.fitted = True


# ============================================================================
# 数据准备函数
# ============================================================================

def create_ts_handler_for_fold(args, fold_config):
    """为特定 fold 创建 TSDatasetH 的 Handler"""
    end_time = fold_config.get('test_end', fold_config['valid_end'])

    handler = TCN_V1_Handler(
        volatility_window=args.nday,
        instruments=STOCK_POOLS[args.stock_pool],
        start_time=fold_config['train_start'],
        end_time=end_time,
        fit_start_time=fold_config['train_start'],
        fit_end_time=fold_config['train_end'],
    )

    return handler


def create_ts_dataset_for_fold(handler, fold_config, step_len=20):
    """为特定 fold 创建 TSDatasetH"""
    segments = {
        "train": (fold_config['train_start'], fold_config['train_end']),
        "valid": (fold_config['valid_start'], fold_config['valid_end']),
    }

    if 'test_start' in fold_config:
        segments["test"] = (fold_config['test_start'], fold_config['test_end'])

    return TSDatasetH(
        handler=handler,
        segments=segments,
        step_len=step_len,
    )


def compute_ic_from_pred(pred, dataset, segment):
    """从预测结果计算 IC"""
    # 获取标签
    dl = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    dl.config(fillna_type="ffill+bfill")

    # 获取索引
    index = dl.get_index()

    # 创建 DataFrame
    df = pd.DataFrame({'pred': pred, 'label': np.nan}, index=index)

    # 填充标签 - 需要从 TSDataSampler 中获取
    labels = []
    for i in range(len(dl)):
        data = dl[i]
        label = data[-1, -1]  # 最后一个时间步的最后一个值 (label)
        labels.append(label)

    df['label'] = labels

    # 按日期计算 IC
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


# ============================================================================
# CV 训练函数
# ============================================================================

def run_cv_training(args):
    """运行 CV 训练"""
    print("\n" + "=" * 70)
    print("TCN CROSS-VALIDATION TRAINING (Qlib Alpha158 Benchmark)")
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool} ({len(STOCK_POOLS[args.stock_pool])} stocks)")
    print(f"N-day: {args.nday}")
    print(f"Step Length: {args.step_len}")
    print(f"d_feat: {args.d_feat}")
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: train {fold['train_start']}~{fold['train_end']}, "
              f"valid {fold['valid_start']}~{fold['valid_end']}")
    print(f"Test: {FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']}")
    print("=" * 70)

    # 预先准备 2025 测试集
    print("\n[*] Preparing 2025 test data...")
    test_handler = create_ts_handler_for_fold(args, FINAL_TEST)
    test_dataset = create_ts_dataset_for_fold(test_handler, FINAL_TEST, step_len=args.step_len)

    test_dl = test_dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    test_dl.config(fillna_type="ffill+bfill")
    test_loader = DataLoader(test_dl, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_index = test_dl.get_index()
    print(f"    Test samples: {len(test_dl)}")

    fold_results = []
    fold_ics = []
    fold_test_ics = []

    for fold_idx, fold in enumerate(CV_FOLDS):
        print(f"\n[*] Training {fold['name']}...")

        # 设置随机种子
        if args.seed is not None:
            seed = args.seed + fold_idx
            np.random.seed(seed)
            torch.manual_seed(seed)

        # 准备数据
        handler = create_ts_handler_for_fold(args, fold)
        dataset = create_ts_dataset_for_fold(handler, fold, step_len=args.step_len)

        train_dl = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        train_dl.config(fillna_type="ffill+bfill")

        valid_dl = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        valid_dl.config(fillna_type="ffill+bfill")

        print(f"    Train samples: {len(train_dl)}, Valid samples: {len(valid_dl)}")

        train_loader = DataLoader(
            train_dl, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True
        )
        valid_loader = DataLoader(
            valid_dl, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        # 创建并训练模型
        trainer = TCNTrainer(
            d_feat=args.d_feat,
            n_chans=args.n_chans,
            kernel_size=args.kernel_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            n_epochs=args.n_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            early_stop=args.early_stop,
            gpu=args.gpu,
            seed=args.seed + fold_idx if args.seed else None,
        )

        best_epoch, best_loss = trainer.fit(train_loader, valid_loader, verbose=args.verbose)

        # 验证集预测
        valid_pred = trainer.predict(valid_loader)
        valid_ic, valid_ic_std, valid_icir = compute_ic_from_pred(valid_pred, dataset, "valid")

        # 测试集预测
        test_pred = trainer.predict(test_loader)
        test_ic, test_ic_std, test_icir = compute_ic_from_pred(test_pred, test_dataset, "test")

        fold_ics.append(valid_ic)
        fold_test_ics.append(test_ic)
        fold_results.append({
            'name': fold['name'],
            'valid_ic': valid_ic,
            'valid_icir': valid_icir,
            'test_ic': test_ic,
            'test_icir': test_icir,
            'best_epoch': best_epoch,
        })

        print(f"    {fold['name']}: Valid IC={valid_ic:.4f}, Test IC (2025)={test_ic:.4f}, epoch={best_epoch}")

    # 汇总结果
    mean_valid_ic = np.mean(fold_ics)
    std_valid_ic = np.std(fold_ics)
    mean_test_ic = np.mean(fold_test_ics)
    std_test_ic = np.std(fold_test_ics)

    print("\n" + "=" * 70)
    print("CV TRAINING COMPLETE")
    print("=" * 70)
    print(f"Valid Mean IC: {mean_valid_ic:.4f} (±{std_valid_ic:.4f})")
    print(f"Test Mean IC (2025): {mean_test_ic:.4f} (±{std_test_ic:.4f})")
    print("\nIC by fold:")
    print(f"  {'Fold':<25s} {'Valid IC':>10s} {'Test IC':>10s} {'Epoch':>8s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")
    for r in fold_results:
        print(f"  {r['name']:<25s} {r['valid_ic']:>10.4f} {r['test_ic']:>10.4f} {r['best_epoch']:>8d}")
    print("=" * 70)

    return fold_results, mean_valid_ic, std_valid_ic, test_dataset


def train_final_model(args, test_dataset):
    """训练最终模型"""
    print("\n[*] Training final model on full data...")

    # 准备数据
    handler = create_ts_handler_for_fold(args, FINAL_TEST)
    dataset = create_ts_dataset_for_fold(handler, FINAL_TEST, step_len=args.step_len)

    train_dl = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    train_dl.config(fillna_type="ffill+bfill")

    valid_dl = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    valid_dl.config(fillna_type="ffill+bfill")

    test_dl = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    test_dl.config(fillna_type="ffill+bfill")

    print(f"    Train: {len(train_dl)}, Valid: {len(valid_dl)}, Test: {len(test_dl)}")

    train_loader = DataLoader(train_dl, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    valid_loader = DataLoader(valid_dl, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dl, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 训练
    trainer = TCNTrainer(
        d_feat=args.d_feat,
        n_chans=args.n_chans,
        kernel_size=args.kernel_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        early_stop=args.early_stop,
        gpu=args.gpu,
        seed=args.seed,
    )

    best_epoch, best_loss = trainer.fit(train_loader, valid_loader, verbose=True)
    print(f"    Best epoch: {best_epoch}, Best loss: {best_loss:.6f}")

    # 预测
    test_pred = trainer.predict(test_loader)
    test_index = test_dl.get_index()

    test_pred_series = pd.Series(test_pred, index=test_index, name='score')

    # 保存模型
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_SAVE_PATH / f"tcn_qlib_alpha158_cv_{args.stock_pool}_{args.nday}d_{timestamp}.pt"
    trainer.save(model_path)
    print(f"    Model saved to: {model_path}")

    return trainer, test_pred_series, dataset, model_path


def main():
    parser = argparse.ArgumentParser(
        description='TCN Cross-Validation Training (Qlib Alpha158 Benchmark)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 基础参数
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--nday', type=int, default=5,
                        help='Label prediction horizon (default: 5)')

    # TCN 参数 (与 benchmark 一致)
    parser.add_argument('--d-feat', type=int, default=20,
                        help='Features per timestep (default: 20)')
    parser.add_argument('--step-len', type=int, default=20,
                        help='Time series length (default: 20)')
    parser.add_argument('--n-chans', type=int, default=32,
                        help='Number of channels (default: 32)')
    parser.add_argument('--kernel-size', type=int, default=7,
                        help='Kernel size (default: 7)')
    parser.add_argument('--num-layers', type=int, default=5,
                        help='Number of TCN layers (default: 5)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')

    # 训练参数
    parser.add_argument('--n-epochs', type=int, default=200,
                        help='Max epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=2000,
                        help='Batch size (default: 2000)')
    parser.add_argument('--early-stop', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')

    # 模式选择
    parser.add_argument('--cv-only', action='store_true',
                        help='Only run CV training, skip final model')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate pre-trained model')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to pre-trained model for evaluation')
    parser.add_argument('--verbose', action='store_true',
                        help='Show training progress for each epoch')

    # 回测参数
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=1)
    parser.add_argument('--account', type=float, default=10000)
    parser.add_argument('--rebalance-freq', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss'])

    args = parser.parse_args()

    # 验证参数
    if args.eval_only and not args.model_path:
        parser.error("--eval-only requires --model-path")

    # 初始化 Qlib
    qlib_data_path = project_root / "my_data" / "qlib_us"
    qlib.init(provider_uri=str(qlib_data_path), region=REG_US)

    print("\n" + "=" * 70)
    print("TCN Qlib Alpha158 Benchmark - US Data")
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool}")
    print(f"N-day: {args.nday}")
    print(f"Step Length: {args.step_len}")
    print(f"d_feat: {args.d_feat}")
    print(f"Channels: {args.n_chans}")
    print(f"Kernel Size: {args.kernel_size}")
    print(f"Num Layers: {args.num_layers}")
    print(f"Dropout: {args.dropout}")
    print(f"GPU: {args.gpu}")
    print("=" * 70)

    # ========== 评估模式 ==========
    if args.eval_only:
        print("\n[*] Evaluation mode...")
        # TODO: 实现评估模式
        print("    Not implemented yet")
        return

    # ========== CV 训练 ==========
    fold_results, mean_ic, std_ic, test_dataset = run_cv_training(args)

    if args.cv_only:
        print("\n[*] CV-only mode, skipping final model training.")
        return

    # ========== 训练最终模型 ==========
    trainer, test_pred, dataset, model_path = train_final_model(args, test_dataset)

    # ========== 评估 ==========
    print("\n[*] Final Evaluation on Test Set (2025)...")
    evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

    # ========== 回测 ==========
    if args.backtest:
        pred_df = test_pred.to_frame("score")

        time_splits = {
            'train_start': FINAL_TEST['train_start'],
            'train_end': FINAL_TEST['train_end'],
            'valid_start': FINAL_TEST['valid_start'],
            'valid_end': FINAL_TEST['valid_end'],
            'test_start': FINAL_TEST['test_start'],
            'test_end': FINAL_TEST['test_end'],
        }

        def load_model(path):
            t = TCNTrainer(gpu=args.gpu)
            t.load(path)
            return t

        def get_feature_count(m):
            return args.d_feat * args.step_len

        run_backtest(
            model_path, dataset, pred_df, args, time_splits,
            model_name="TCN (Qlib Alpha158 CV)",
            load_model_func=load_model,
            get_feature_count_func=get_feature_count
        )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"CV Valid Mean IC: {mean_ic:.4f} (±{std_ic:.4f})")
    print(f"Model saved to: {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
