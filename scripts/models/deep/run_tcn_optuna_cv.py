"""
TCN 超参数搜索 - Optuna 版本 (带 Pruning)

使用 Optuna 进行超参数搜索，支持：
- 早期剪枝 (Pruning): 表现差的 trial 会被提前终止
- 时间序列交叉验证
- 断点续传

时间窗口设计:
  Fold 1: train 2000-2021, valid 2022
  Fold 2: train 2000-2022, valid 2023
  Fold 3: train 2000-2023, valid 2024
  Test:   2025 (完全独立)

使用方法:
    python scripts/models/deep/run_tcn_optuna_cv.py
    python scripts/models/deep/run_tcn_optuna_cv.py --n-trials 50
    python scripts/models/deep/run_tcn_optuna_cv.py --stock-pool sp100 --backtest

    # 断点续传 (使用相同的 study-name)
    python scripts/models/deep/run_tcn_optuna_cv.py --study-name tcn_search --resume
"""

import os
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
import json
from datetime import datetime
import numpy as np
import pandas as pd

import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler

import torch
import torch.nn as nn

from utils.utils import evaluate_model
from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG, PROJECT_ROOT, MODEL_SAVE_PATH,
    init_qlib,
    run_backtest,
    CV_FOLDS,
    FINAL_TEST,
    create_data_handler_for_fold,
    create_dataset_for_fold,
    prepare_data_from_dataset,
    compute_ic,
)


# Handler d_feat 配置
# Alpha360-Macro has (6 + M) features × 60 timesteps where M = macro features
HANDLER_D_FEAT = {
    'alpha360': 6,
    'alpha360-macro': 29,    # (6 + 23 core macro) × 60 = 1740 total
    'alpha158': 158,
    'alpha158-talib': 158,
    'alpha158-talib-lite': 158,
}


def create_tcn_model(d_feat, n_chans, num_layers, kernel_size, dropout, device):
    """创建 TCN 模型"""
    from qlib.contrib.model.tcn import TemporalConvNet

    class TCNModel(nn.Module):
        def __init__(self, num_input, output_size, num_channels, kernel_size, dropout):
            super().__init__()
            self.num_input = num_input
            self.tcn = TemporalConvNet(num_input, num_channels, kernel_size, dropout=dropout)
            self.linear = nn.Linear(num_channels[-1], output_size)

        def forward(self, x):
            batch_size = x.size(0)
            seq_len = x.size(1) // self.num_input
            x = x.view(batch_size, seq_len, self.num_input)
            x = x.permute(0, 2, 1)
            y = self.tcn(x)
            return self.linear(y[:, :, -1])

    model = TCNModel(
        num_input=d_feat,
        output_size=1,
        num_channels=[n_chans] * num_layers,
        kernel_size=kernel_size,
        dropout=dropout,
    )
    model.to(device)
    return model


class OptunaObjective:
    """Optuna 目标函数，支持 Pruning"""

    def __init__(self, args, handler_config, symbols, n_epochs=50, early_stop=10, gpu=0):
        self.args = args
        self.handler_config = handler_config
        self.symbols = symbols
        self.n_epochs = n_epochs
        self.early_stop = early_stop
        self.gpu = gpu

        # 设置设备
        if gpu >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu}')
        else:
            self.device = torch.device('cpu')

        # 预先准备所有 fold 的数据
        print("\n[*] Preparing data for all CV folds...")
        self.fold_data = []
        self.d_feat = None
        self.total_features = None

        for fold in CV_FOLDS:
            print(f"    Preparing {fold['name']}...")
            handler = create_data_handler_for_fold(args, handler_config, symbols, fold)
            dataset = create_dataset_for_fold(handler, fold)

            X_train, y_train, _ = prepare_data_from_dataset(dataset, "train")
            X_valid, y_valid, valid_index = prepare_data_from_dataset(dataset, "valid")

            if self.total_features is None:
                self.total_features = X_train.shape[1]
                self.d_feat = HANDLER_D_FEAT.get(args.handler, self.total_features)
                if self.total_features % self.d_feat != 0:
                    self.d_feat = self.total_features

            self.fold_data.append({
                'name': fold['name'],
                'X_train': X_train,
                'y_train': y_train,
                'X_valid': X_valid,
                'y_valid': y_valid,
                'valid_index': valid_index,
            })

            print(f"      Train: {X_train.shape}, Valid: {X_valid.shape}")

        print(f"    ✓ All {len(CV_FOLDS)} folds prepared")
        print(f"    Total features: {self.total_features}, d_feat: {self.d_feat}")

    def __call__(self, trial: optuna.Trial):
        """目标函数：支持中间汇报和剪枝"""

        # 动态采样超参数
        n_chans = trial.suggest_categorical('n_chans', [32, 64, 128, 256])
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
        num_layers = trial.suggest_int('num_layers', 2, 6)
        dropout = trial.suggest_float('dropout', 0.1, 0.7)
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048, 4096])

        # 打印当前 trial 的参数
        print(f"\n{'='*60}")
        print(f"Trial {trial.number}: chans={n_chans}, layers={num_layers}, "
              f"kernel={kernel_size}, dropout={dropout:.3f}, lr={lr:.6f}, batch={batch_size}")
        print(f"{'='*60}")

        fold_ics = []

        for fold_idx, fold in enumerate(self.fold_data):
            print(f"\n  [{fold['name']}]")

            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 创建模型
            model = create_tcn_model(
                self.d_feat, n_chans, num_layers, kernel_size, dropout, self.device
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # 训练循环 (带中间汇报)
            best_loss = float('inf')
            best_state = None
            best_val_ic = 0.0
            stop_steps = 0

            for epoch in range(self.n_epochs):
                # 训练一个 epoch
                model.train()
                indices = np.arange(len(fold['X_train']))
                np.random.shuffle(indices)

                train_losses = []
                for i in range(0, len(indices), batch_size):
                    if len(indices) - i < batch_size:
                        break
                    batch_idx = indices[i:i + batch_size]
                    feature = torch.from_numpy(fold['X_train'][batch_idx]).float().to(self.device)
                    label = torch.from_numpy(fold['y_train'][batch_idx]).float().to(self.device)

                    pred = model(feature).squeeze()
                    loss = torch.mean((pred - label) ** 2)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
                    optimizer.step()

                    train_losses.append(loss.item())

                train_loss = np.mean(train_losses)

                # 验证：计算 loss 和 IC
                model.eval()
                with torch.no_grad():
                    # 训练集 IC (采样计算，避免太慢)
                    sample_size = min(50000, len(fold['X_train']))
                    sample_idx = np.random.choice(len(fold['X_train']), sample_size, replace=False)
                    train_preds = []
                    for i in range(0, len(sample_idx), batch_size):
                        end = min(i + batch_size, len(sample_idx))
                        idx = sample_idx[i:end]
                        feature = torch.from_numpy(fold['X_train'][idx]).float().to(self.device)
                        pred = model(feature).squeeze()
                        train_preds.append(pred.cpu().numpy())
                    train_pred = np.concatenate(train_preds)
                    # 简单相关系数作为训练 IC 近似
                    train_ic = np.corrcoef(train_pred, fold['y_train'][sample_idx])[0, 1]
                    if np.isnan(train_ic):
                        train_ic = 0.0

                    # 验证集预测
                    val_preds = []
                    for i in range(0, len(fold['X_valid']), batch_size):
                        end = min(i + batch_size, len(fold['X_valid']))
                        feature = torch.from_numpy(fold['X_valid'][i:end]).float().to(self.device)
                        pred = model(feature).squeeze()
                        val_preds.append(pred.cpu().numpy())
                    val_pred = np.concatenate(val_preds)
                    val_loss = np.mean((val_pred - fold['y_valid']) ** 2)

                    # 验证集 IC
                    val_ic, _, _ = compute_ic(val_pred, fold['y_valid'], fold['valid_index'])

                # 打印每个 epoch 的进度
                is_best = ""
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_val_ic = val_ic
                    stop_steps = 0
                    is_best = " *"
                else:
                    stop_steps += 1

                print(f"    Epoch {epoch+1:2d}/{self.n_epochs}: "
                      f"train_loss={train_loss:.6f} train_IC={train_ic:.4f} | "
                      f"val_loss={val_loss:.6f} val_IC={val_ic:.4f}{is_best}")

                if stop_steps >= self.early_stop:
                    print(f"    Early stopping at epoch {epoch+1} (best val_IC={best_val_ic:.4f})")
                    break

                # ========== 关键：中间汇报给 Optuna ==========
                # 每 5 个 epoch 汇报一次，让 Optuna 决定是否剪枝
                if epoch % 5 == 0 and epoch > 0:
                    intermediate_value = best_val_ic if fold_idx == 0 else np.mean(fold_ics + [best_val_ic])
                    trial.report(intermediate_value, epoch + fold_idx * self.n_epochs)

                    # 检查是否应该剪枝
                    if trial.should_prune():
                        print(f"    >>> PRUNED by Optuna (intermediate IC={intermediate_value:.4f} 低于中位数)")
                        raise optuna.TrialPruned()

            # 加载最佳模型并计算最终 IC
            if best_state is not None:
                model.load_state_dict(best_state)

            model.eval()
            with torch.no_grad():
                val_preds = []
                for i in range(0, len(fold['X_valid']), batch_size):
                    end = min(i + batch_size, len(fold['X_valid']))
                    feature = torch.from_numpy(fold['X_valid'][i:end]).float().to(self.device)
                    pred = model(feature).squeeze()
                    val_preds.append(pred.cpu().numpy())
                val_pred = np.concatenate(val_preds)

            mean_ic, _, icir = compute_ic(val_pred, fold['y_valid'], fold['valid_index'])
            fold_ics.append(mean_ic)

            print(f"  [{fold['name']}] Final IC={mean_ic:.4f}, ICIR={icir:.4f}")

            # 清理
            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 返回平均 IC
        mean_ic_all = np.mean(fold_ics)
        std_ic_all = np.std(fold_ics)

        print(f"\n  Trial {trial.number} Result: Mean IC={mean_ic_all:.4f} (±{std_ic_all:.4f})")

        # 保存额外信息
        trial.set_user_attr('fold_ics', fold_ics)
        trial.set_user_attr('std_ic', std_ic_all)

        return mean_ic_all


def run_optuna_search(args, handler_config, symbols):
    """运行 Optuna 超参数搜索"""
    print("\n" + "=" * 70)
    print("OPTUNA SEARCH WITH PRUNING (TCN)")
    print("=" * 70)
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: train {fold['train_start']}~{fold['train_end']}, "
              f"valid {fold['valid_start']}~{fold['valid_end']}")
    print(f"Max trials: {args.n_trials}")
    print(f"Epochs per fold: {args.cv_epochs}")
    print(f"Pruner: MedianPruner (启动后第10个epoch开始剪枝)")
    print("=" * 70)

    # 创建目标函数
    objective = OptunaObjective(
        args, handler_config, symbols,
        n_epochs=args.cv_epochs,
        early_stop=args.cv_early_stop,
        gpu=args.gpu
    )

    # 创建 Optuna study
    # MedianPruner: 如果中间值低于之前 trials 的中位数，则剪枝
    pruner = MedianPruner(
        n_startup_trials=5,      # 前5个 trial 不剪枝，收集基准
        n_warmup_steps=10,       # 每个 trial 前10步不剪枝
        interval_steps=5,        # 每5步检查一次
    )

    sampler = TPESampler(seed=42)

    # 支持断点续传
    storage = f"sqlite:///{PROJECT_ROOT}/outputs/optuna_tcn.db"

    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',  # 最大化 IC
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=args.resume,
    )

    # 回调函数：打印进度
    def print_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            fold_ics = trial.user_attrs.get('fold_ics', [])
            fold_ic_str = ", ".join([f"{ic:.4f}" for ic in fold_ics])
            print(f"  Trial {trial.number:3d}: IC={trial.value:.4f} [{fold_ic_str}] "
                  f"lr={trial.params['lr']:.5f} chans={trial.params['n_chans']} "
                  f"layers={trial.params['num_layers']}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"  Trial {trial.number:3d}: PRUNED (表现差，提前终止)")

    # 运行优化
    print("\n[*] Running optimization...")
    study.optimize(
        objective,
        n_trials=args.n_trials,
        callbacks=[print_callback],
        show_progress_bar=True,
    )

    # 获取最佳结果
    best_trial = study.best_trial
    best_params = best_trial.params
    best_params['d_feat'] = objective.d_feat

    print("\n" + "=" * 70)
    print("OPTUNA SEARCH COMPLETE")
    print("=" * 70)
    print(f"Total trials: {len(study.trials)}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"\nBest IC: {best_trial.value:.4f}")
    if 'fold_ics' in best_trial.user_attrs:
        fold_ics = best_trial.user_attrs['fold_ics']
        print(f"Fold ICs: {[f'{ic:.4f}' for ic in fold_ics]}")
    print(f"\nBest parameters:")
    print(f"  d_feat: {best_params['d_feat']}")
    print(f"  n_chans: {best_params['n_chans']}")
    print(f"  kernel_size: {best_params['kernel_size']}")
    print(f"  num_layers: {best_params['num_layers']}")
    print(f"  dropout: {best_params['dropout']:.4f}")
    print(f"  lr: {best_params['lr']:.6f}")
    print(f"  batch_size: {best_params['batch_size']}")
    print("=" * 70)

    return best_params, study, objective.d_feat, objective.total_features


def train_final_model(args, handler_config, symbols, best_params, d_feat, total_features):
    """使用最优参数训练最终模型"""
    print("\n[*] Training final model on full data...")

    # 创建数据集
    handler = create_data_handler_for_fold(args, handler_config, symbols, FINAL_TEST)
    dataset = create_dataset_for_fold(handler, FINAL_TEST)

    X_train, y_train, _ = prepare_data_from_dataset(dataset, "train")
    X_valid, y_valid, valid_index = prepare_data_from_dataset(dataset, "valid")
    X_test, y_test, test_index = prepare_data_from_dataset(dataset, "test")

    print(f"    Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    # 创建模型
    model = create_tcn_model(
        best_params['d_feat'],
        best_params['n_chans'],
        best_params['num_layers'],
        best_params['kernel_size'],
        best_params['dropout'],
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    batch_size = best_params['batch_size']

    # 训练
    best_loss = float('inf')
    best_state = None
    stop_steps = 0

    print("    Training...")
    for epoch in range(args.n_epochs):
        model.train()
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            if len(indices) - i < batch_size:
                break
            batch_idx = indices[i:i + batch_size]
            feature = torch.from_numpy(X_train[batch_idx]).float().to(device)
            label = torch.from_numpy(y_train[batch_idx]).float().to(device)

            pred = model(feature).squeeze()
            loss = torch.mean((pred - label) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
            optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_preds = []
            for i in range(0, len(X_valid), batch_size):
                end = min(i + batch_size, len(X_valid))
                feature = torch.from_numpy(X_valid[i:end]).float().to(device)
                pred = model(feature).squeeze()
                val_preds.append(pred.cpu().numpy())
            val_pred = np.concatenate(val_preds)
            val_loss = np.mean((val_pred - y_valid) ** 2)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stop_steps = 0
            if epoch % 10 == 0:
                print(f"      Epoch {epoch+1}: val_loss={val_loss:.6f} (best)")
        else:
            stop_steps += 1
            if stop_steps >= args.early_stop:
                print(f"      Early stopping at epoch {epoch+1}")
                break

    # 加载最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    # 测试集预测
    model.eval()
    with torch.no_grad():
        test_preds = []
        for i in range(0, len(X_test), batch_size):
            end = min(i + batch_size, len(X_test))
            feature = torch.from_numpy(X_test[i:end]).float().to(device)
            pred = model(feature).squeeze()
            test_preds.append(pred.cpu().numpy())
        test_pred_values = np.concatenate(test_preds)

    test_pred = pd.Series(test_pred_values, index=test_index, name='score')

    # 验证集 IC
    valid_ic, _, valid_icir = compute_ic(val_pred, y_valid, valid_index)
    print(f"\n    Valid IC: {valid_ic:.4f}, ICIR: {valid_icir:.4f}")
    print(f"    Test prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    # 包装模型
    class TCNWrapper:
        def __init__(self, tcn_model, device, batch_size):
            self.tcn_model = tcn_model
            self.device = device
            self.batch_size = batch_size

        def predict(self, X_data):
            self.tcn_model.eval()
            preds = []
            with torch.no_grad():
                for i in range(0, len(X_data), self.batch_size):
                    end = min(i + self.batch_size, len(X_data))
                    feature = torch.from_numpy(X_data[i:end]).float().to(self.device)
                    pred = self.tcn_model(feature).squeeze()
                    preds.append(pred.cpu().numpy())
            return np.concatenate(preds)

    model_wrapper = TCNWrapper(model, device, batch_size)

    return model_wrapper, test_pred, dataset, total_features


def main():
    parser = argparse.ArgumentParser(description='TCN Optuna with Pruning')

    # 基础参数
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--handler', type=str, default='alpha180',
                        choices=list(HANDLER_CONFIG.keys()))
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])

    # Optuna 参数
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials')
    parser.add_argument('--cv-epochs', type=int, default=30,
                        help='Max epochs per CV fold (会被 pruning 提前终止)')
    parser.add_argument('--cv-early-stop', type=int, default=8,
                        help='Early stopping patience for CV')
    parser.add_argument('--study-name', type=str, default='tcn_cv_search',
                        help='Optuna study name (用于断点续传)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous study')

    # 最终训练参数
    parser.add_argument('--n-epochs', type=int, default=100,
                        help='Epochs for final model')
    parser.add_argument('--early-stop', type=int, default=15,
                        help='Early stopping for final model')
    parser.add_argument('--gpu', type=int, default=0)

    # 回测参数
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--n-drop', type=int, default=1)
    parser.add_argument('--account', type=float, default=10000)
    parser.add_argument('--rebalance-freq', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss'])

    args = parser.parse_args()

    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    print("=" * 70)
    print("TCN Hyperparameter Search with Optuna + Pruning")
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Handler: {args.handler}")
    print(f"N-day: {args.nday}")
    print(f"Max trials: {args.n_trials}")
    print(f"CV epochs: {args.cv_epochs}")
    print(f"Study name: {args.study_name}")
    print(f"Resume: {args.resume}")
    print("=" * 70)

    init_qlib(handler_config['use_talib'])

    # 运行搜索
    best_params, study, d_feat, total_features = run_optuna_search(
        args, handler_config, symbols
    )

    # 保存结果
    output_dir = PROJECT_ROOT / "outputs" / "optuna_cv"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    params_file = output_dir / f"tcn_optuna_best_{timestamp}.json"
    with open(params_file, 'w') as f:
        json.dump({
            'best_ic': study.best_value,
            'params': {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                      for k, v in best_params.items()},
            'n_trials': len(study.trials),
            'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        }, f, indent=2)
    print(f"\nBest parameters saved to: {params_file}")

    # 训练最终模型
    model, test_pred, dataset, total_features = train_final_model(
        args, handler_config, symbols, best_params, d_feat, total_features
    )

    # 评估
    print("\n[*] Final Evaluation...")
    evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

    # 保存模型
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_SAVE_PATH / f"tcn_optuna_{args.handler}_{args.stock_pool}_{args.nday}d.pt"
    torch.save(model, model_path)
    print(f"Model saved to: {model_path}")

    # 回测
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
            return torch.load(path, weights_only=False)

        def get_feature_count(m):
            return total_features

        run_backtest(
            model_path, dataset, pred_df, args, time_splits,
            model_name="TCN (Optuna)",
            load_model_func=load_model,
            get_feature_count_func=get_feature_count
        )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
