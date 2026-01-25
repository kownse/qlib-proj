"""
AE-MLP PyTorch 交叉验证训练和评估脚本

支持两种模式:
1. 训练模式: 使用预先搜索好的超参数进行训练
2. 评估模式: 加载已训练模型，在 CV folds 上计算 IC

时间窗口设计:
  Fold 1: train 2000-2021, valid 2022
  Fold 2: train 2000-2022, valid 2023
  Fold 3: train 2000-2023, valid 2024
  Test:   2025 (完全独立)

使用方法:
    # ===== 评估模式 (加载已训练模型) =====
    python scripts/models/deep/run_ae_mlp_pytorch_cv.py \
        --eval-only \
        --model-path my_models/ae_mlp_pytorch_cv_xxx.pt \
        --handler alpha158-enhanced-v7

    # ===== 训练模式 =====
    # 使用参数文件训练
    python scripts/models/deep/run_ae_mlp_pytorch_cv.py \
        --params-file outputs/hyperopt_cv/ae_mlp_cv_best_params_20260117_151024.json

    # 只运行 CV 训练，不训练最终模型
    python scripts/models/deep/run_ae_mlp_pytorch_cv.py \
        --params-file outputs/hyperopt_cv/ae_mlp_cv_best_params_20260117_151024.json \
        --cv-only

    # 多种子训练，寻找最佳结果
    python scripts/models/deep/run_ae_mlp_pytorch_cv.py \
        --params-file outputs/hyperopt_cv/ae_mlp_cv_best_params_20260117_151024.json \
        --cv-only --num-seeds 10

    # 训练最终模型并回测
    python scripts/models/deep/run_ae_mlp_pytorch_cv.py \
        --params-file outputs/hyperopt_cv/ae_mlp_cv_best_params_20260117_151024.json \
        --backtest
"""

# ============================================================================
# 设置环境变量
# ============================================================================

import os

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import random
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

# ============================================================================
# 现在可以安全地导入其他模块
# ============================================================================

import argparse
import copy
import json
from datetime import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from utils.utils import evaluate_model
from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG, PROJECT_ROOT, MODEL_SAVE_PATH,
    print_training_header,
    init_qlib,
    check_data_availability,
    save_model_with_meta,
    create_meta_data,
    generate_model_filename,
    run_backtest,
    # CV utilities
    CV_FOLDS,
    FINAL_TEST,
    create_data_handler_for_fold,
    create_dataset_for_fold,
    prepare_data_from_dataset,
    compute_ic,
)

# 导入 PyTorch AE-MLP 模型
from ae_mlp_model_pytorch import AEMLPNetwork, EarlyStopping


def set_random_seed(seed: int):
    """设置随机种子以提高可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"    Random seed set to: {seed}")


def load_params_from_json(params_file: str) -> dict:
    """从 JSON 文件加载超参数"""
    with open(params_file, 'r') as f:
        data = json.load(f)

    params = data['params']
    cv_results = data.get('cv_results', {})

    print(f"\n[*] Loaded parameters from: {params_file}")
    print(f"    hidden_units: {params['hidden_units']}")
    print(f"    learning_rate: {params['lr']:.6f}")
    print(f"    batch_size: {params['batch_size']}")
    print(f"    dropout: {params['dropout_rates'][1]:.4f}")
    print(f"    noise_std: {params['dropout_rates'][0]:.4f}")
    print(f"    loss_weights: {params['loss_weights']}")

    if cv_results:
        print(f"\n    Original CV results:")
        print(f"      Mean IC: {cv_results['mean_ic']:.4f} (±{cv_results['std_ic']:.4f})")
        if 'fold_results' in cv_results:
            for fold in cv_results['fold_results']:
                print(f"      {fold['name']}: IC={fold['ic']:.4f}, ICIR={fold['icir']:.4f}")

    return params, cv_results


def setup_device(gpu: int) -> torch.device:
    """配置 GPU/CPU"""
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        print(f"    Using GPU: cuda:{gpu}")
        # 打印 GPU 信息
        print(f"    GPU Name: {torch.cuda.get_device_name(gpu)}")
        print(f"    GPU Memory: {torch.cuda.get_device_properties(gpu).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("    Using CPU")
    return device


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray = None,
    batch_size: int = 4096,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """创建 DataLoader"""
    X_tensor = torch.FloatTensor(X)

    if y is not None:
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)
    else:
        dataset = TensorDataset(X_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )


def build_ae_mlp_model(params: dict, device: torch.device) -> AEMLPNetwork:
    """构建 AE-MLP 模型"""
    model = AEMLPNetwork(
        num_columns=params['num_columns'],
        hidden_units=params['hidden_units'],
        dropout_rates=params['dropout_rates'],
    )
    return model.to(device)


def train_one_epoch(
    model: AEMLPNetwork,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_weights: dict,
    device: torch.device,
) -> dict:
    """训练一个 epoch"""
    model.train()
    mse_loss = nn.MSELoss()
    losses = {'decoder': 0, 'ae_action': 0, 'action': 0, 'total': 0}
    num_batches = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        # 前向传播
        decoder_out, ae_action_out, action_out = model(batch_X)

        # 计算损失
        loss_decoder = mse_loss(decoder_out, batch_X)
        loss_ae_action = mse_loss(ae_action_out, batch_y)
        loss_action = mse_loss(action_out, batch_y)

        # 加权总损失
        total_loss = (
            loss_weights['decoder'] * loss_decoder +
            loss_weights['ae_action'] * loss_ae_action +
            loss_weights['action'] * loss_action
        )

        # 反向传播
        total_loss.backward()
        optimizer.step()

        losses['decoder'] += loss_decoder.item()
        losses['ae_action'] += loss_ae_action.item()
        losses['action'] += loss_action.item()
        losses['total'] += total_loss.item()
        num_batches += 1

    # 计算平均损失
    for key in losses:
        losses[key] /= num_batches

    return losses


def validate(
    model: AEMLPNetwork,
    valid_loader: DataLoader,
    loss_weights: dict,
    device: torch.device,
) -> dict:
    """验证模型"""
    model.eval()
    mse_loss = nn.MSELoss()
    losses = {'decoder': 0, 'ae_action': 0, 'action': 0, 'total': 0}
    num_batches = 0

    with torch.no_grad():
        for batch_X, batch_y in valid_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            decoder_out, ae_action_out, action_out = model(batch_X)

            loss_decoder = mse_loss(decoder_out, batch_X)
            loss_ae_action = mse_loss(ae_action_out, batch_y)
            loss_action = mse_loss(action_out, batch_y)

            total_loss = (
                loss_weights['decoder'] * loss_decoder +
                loss_weights['ae_action'] * loss_ae_action +
                loss_weights['action'] * loss_action
            )

            losses['decoder'] += loss_decoder.item()
            losses['ae_action'] += loss_ae_action.item()
            losses['action'] += loss_action.item()
            losses['total'] += total_loss.item()
            num_batches += 1

    for key in losses:
        losses[key] /= num_batches

    return losses


def predict_batch(
    model: AEMLPNetwork,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """批量预测"""
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)

    predictions = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            _, _, pred = model(batch)
            predictions.append(pred.cpu().numpy())

    return np.concatenate(predictions, axis=0).flatten()


def save_pytorch_model(model: AEMLPNetwork, params: dict, path: str):
    """保存 PyTorch 模型"""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'num_columns': params['num_columns'],
        'hidden_units': params['hidden_units'],
        'dropout_rates': params['dropout_rates'],
        'loss_weights': params['loss_weights'],
    }
    torch.save(save_dict, path)
    print(f"    Model saved to: {path}")


def load_pytorch_model(path: str, device: torch.device) -> tuple:
    """加载 PyTorch 模型"""
    checkpoint = torch.load(path, map_location=device)

    model = AEMLPNetwork(
        num_columns=checkpoint['num_columns'],
        hidden_units=checkpoint['hidden_units'],
        dropout_rates=checkpoint['dropout_rates'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    params = {
        'num_columns': checkpoint['num_columns'],
        'hidden_units': checkpoint['hidden_units'],
        'dropout_rates': checkpoint['dropout_rates'],
        'loss_weights': checkpoint['loss_weights'],
    }

    print(f"    Model loaded from: {path}")
    return model, params


def run_cv_evaluation(args, handler_config, symbols, model_path):
    """加载预训练模型并在 CV folds 和2025测试集上评估 IC"""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION EVALUATION (AE-MLP PyTorch)")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: valid {fold['valid_start']}~{fold['valid_end']}")
    print(f"Test Set: {FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']} (2025)")
    print("=" * 70)

    # 设置设备
    device = setup_device(args.gpu)

    # 加载模型
    print(f"\n[*] Loading model from: {model_path}")
    model, params = load_pytorch_model(model_path, device)
    print(f"    Model input shape: {params['num_columns']}")
    print(f"    Model loaded successfully")

    # 预先准备2025测试集数据
    print("\n[*] Preparing 2025 test data for evaluation...")
    test_handler = create_data_handler_for_fold(args, handler_config, symbols, FINAL_TEST)
    test_dataset = create_dataset_for_fold(test_handler, FINAL_TEST)
    X_test, y_test, test_index = prepare_data_from_dataset(test_dataset, "test")
    print(f"    Test (2025): {X_test.shape}")

    fold_results = []
    fold_ics = []
    fold_test_ics = []
    batch_size = 4096

    for fold_idx, fold in enumerate(CV_FOLDS):
        print(f"\n[*] Evaluating on {fold['name']}...")

        # 准备数据
        handler = create_data_handler_for_fold(args, handler_config, symbols, fold)
        dataset = create_dataset_for_fold(handler, fold)

        X_valid, y_valid, valid_index = prepare_data_from_dataset(dataset, "valid")

        print(f"    Valid: {X_valid.shape}")

        # 预测验证集
        valid_pred = predict_batch(model, X_valid, device, batch_size)

        # 计算验证集 IC
        mean_ic, ic_std, icir = compute_ic(valid_pred, y_valid, valid_index)

        # ========== 2025 测试集评估 ==========
        test_pred = predict_batch(model, X_test, device, batch_size)
        test_ic, test_ic_std, test_icir = compute_ic(test_pred, y_test, test_index)

        fold_ics.append(mean_ic)
        fold_test_ics.append(test_ic)
        fold_results.append({
            'name': fold['name'],
            'ic': mean_ic,
            'icir': icir,
            'test_ic': test_ic,
            'test_icir': test_icir,
        })

        print(f"    {fold['name']}: Valid IC={mean_ic:.4f}, Test IC (2025)={test_ic:.4f}")

    # 汇总结果
    mean_ic_all = np.mean(fold_ics)
    std_ic_all = np.std(fold_ics)
    mean_test_ic_all = np.mean(fold_test_ics)
    std_test_ic_all = np.std(fold_test_ics)

    print("\n" + "=" * 70)
    print("CV EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Valid Mean IC: {mean_ic_all:.4f} (±{std_ic_all:.4f})")
    print(f"Test Mean IC (2025): {mean_test_ic_all:.4f} (±{std_test_ic_all:.4f})")
    print("\nIC by fold:")
    print(f"  {'Fold':<25s} {'Valid IC':>10s} {'Test IC':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    for r in fold_results:
        print(f"  {r['name']:<25s} {r['ic']:>10.4f} {r['test_ic']:>10.4f}")
    print("=" * 70)

    # 返回测试集预测（用于backtest）
    test_pred_final = predict_batch(model, X_test, device, batch_size)
    test_pred_series = pd.Series(test_pred_final, index=test_index, name='score')

    return fold_results, mean_ic_all, std_ic_all, test_pred_series, test_dataset


def run_cv_training(args, handler_config, symbols, params):
    """运行 CV 训练以复现 IC，同时在2025测试集上评估"""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION TRAINING (AE-MLP PyTorch)")
    print("=" * 70)
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: train {fold['train_start']}~{fold['train_end']}, "
              f"valid {fold['valid_start']}~{fold['valid_end']}")
    print(f"Test Set: {FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']} (2025)")
    print(f"Epochs: {args.cv_epochs}")
    print(f"Early stop patience: {args.cv_early_stop}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print("=" * 70)

    # 设置设备
    device = setup_device(args.gpu)

    # 预先准备2025测试集数据（只需准备一次）
    print("\n[*] Preparing 2025 test data for evaluation...")
    test_handler = create_data_handler_for_fold(args, handler_config, symbols, FINAL_TEST)
    test_dataset = create_dataset_for_fold(test_handler, FINAL_TEST)
    X_test, y_test, test_index = prepare_data_from_dataset(test_dataset, "test")
    print(f"    Test (2025): {X_test.shape}")

    fold_results = []
    fold_ics = []
    fold_test_ics = []

    for fold_idx, fold in enumerate(CV_FOLDS):
        print(f"\n[*] Training {fold['name']}...")

        # 为每个 fold 设置随机种子
        if args.seed is not None:
            set_random_seed(args.seed + fold_idx)

        # 准备数据
        handler = create_data_handler_for_fold(args, handler_config, symbols, fold)
        dataset = create_dataset_for_fold(handler, fold)

        X_train, y_train, _ = prepare_data_from_dataset(dataset, "train")
        X_valid, y_valid, valid_index = prepare_data_from_dataset(dataset, "valid")

        print(f"    Train: {X_train.shape}, Valid: {X_valid.shape}")

        # 更新 num_columns
        fold_params = params.copy()
        fold_params['num_columns'] = X_train.shape[1]

        # 构建模型
        model = build_ae_mlp_model(fold_params, device)

        # 打印模型信息
        if fold_idx == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"    Total parameters: {total_params:,}")

        # 创建 DataLoader
        batch_size = params['batch_size']
        pin_memory = device.type == 'cuda'
        train_loader = create_dataloader(X_train, y_train, batch_size, shuffle=True, pin_memory=pin_memory)
        valid_loader = create_dataloader(X_valid, y_valid, batch_size, shuffle=False, pin_memory=pin_memory)

        # 优化器和调度器
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        # 早停
        early_stopping = EarlyStopping(
            patience=args.cv_early_stop,
            min_delta=1e-5,
            restore_best=True,
        )

        # 训练循环
        loss_weights = params['loss_weights']
        best_epoch = 0

        for epoch in range(args.cv_epochs):
            train_losses = train_one_epoch(model, train_loader, optimizer, loss_weights, device)
            val_losses = validate(model, valid_loader, loss_weights, device)

            scheduler.step(val_losses['action'])

            if args.verbose:
                print(f"      Epoch {epoch+1:3d}: train_loss={train_losses['total']:.4f}, "
                      f"val_action_loss={val_losses['action']:.4f}")

            if early_stopping(val_losses['action'], model):
                best_epoch = epoch + 1 - args.cv_early_stop
                break
            else:
                best_epoch = epoch + 1

        # 恢复最佳模型
        early_stopping.restore(model)

        # 验证集预测
        valid_pred = predict_batch(model, X_valid, device, batch_size)

        # 计算验证集 IC
        mean_ic, ic_std, icir = compute_ic(valid_pred, y_valid, valid_index)

        # ========== 2025 测试集评估 ==========
        test_pred = predict_batch(model, X_test, device, batch_size)
        test_ic, test_ic_std, test_icir = compute_ic(test_pred, y_test, test_index)

        fold_ics.append(mean_ic)
        fold_test_ics.append(test_ic)
        fold_results.append({
            'name': fold['name'],
            'ic': mean_ic,
            'icir': icir,
            'test_ic': test_ic,
            'test_icir': test_icir,
            'best_epoch': best_epoch,
        })

        print(f"    {fold['name']}: Valid IC={mean_ic:.4f}, Test IC (2025)={test_ic:.4f}, epoch={best_epoch}")

    # 汇总结果
    mean_ic_all = np.mean(fold_ics)
    std_ic_all = np.std(fold_ics)
    mean_test_ic_all = np.mean(fold_test_ics)
    std_test_ic_all = np.std(fold_test_ics)

    print("\n" + "=" * 70)
    print("CV TRAINING COMPLETE")
    print("=" * 70)
    print(f"Valid Mean IC: {mean_ic_all:.4f} (±{std_ic_all:.4f})")
    print(f"Test Mean IC (2025): {mean_test_ic_all:.4f} (±{std_test_ic_all:.4f})")
    print("\nIC by fold:")
    print(f"  {'Fold':<25s} {'Valid IC':>10s} {'Test IC':>10s} {'Epoch':>8s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")
    for r in fold_results:
        print(f"  {r['name']:<25s} {r['ic']:>10.4f} {r['test_ic']:>10.4f} {r['best_epoch']:>8d}")
    print("=" * 70)

    return fold_results, mean_ic_all, std_ic_all


def train_final_model(args, handler_config, symbols, params):
    """使用参数在完整数据上训练最终模型"""
    print("\n[*] Training final model on full data...")
    print("    Parameters:")
    print(f"      hidden_units: {params['hidden_units']}")
    print(f"      learning_rate: {params['lr']:.6f}")
    print(f"      batch_size: {params['batch_size']}")

    # 设置设备
    device = setup_device(args.gpu)

    # 创建最终数据集
    handler = create_data_handler_for_fold(args, handler_config, symbols, FINAL_TEST)
    dataset = create_dataset_for_fold(handler, FINAL_TEST)

    X_train, y_train, _ = prepare_data_from_dataset(dataset, "train")
    X_valid, y_valid, valid_index = prepare_data_from_dataset(dataset, "valid")
    X_test, _, test_index = prepare_data_from_dataset(dataset, "test")

    print(f"\n    Final training data:")
    print(f"      Train: {X_train.shape} ({FINAL_TEST['train_start']} ~ {FINAL_TEST['train_end']})")
    print(f"      Valid: {X_valid.shape} ({FINAL_TEST['valid_start']} ~ {FINAL_TEST['valid_end']})")
    print(f"      Test:  {X_test.shape} ({FINAL_TEST['test_start']} ~ {FINAL_TEST['test_end']})")

    # 更新特征数
    final_params = params.copy()
    final_params['num_columns'] = X_train.shape[1]

    # 设置随机种子
    if args.seed is not None:
        set_random_seed(args.seed + 100)  # 使用不同于 CV 的种子

    # 构建模型
    model = build_ae_mlp_model(final_params, device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Total parameters: {total_params:,}")

    # 创建 DataLoader
    batch_size = params['batch_size']
    pin_memory = device.type == 'cuda'
    train_loader = create_dataloader(X_train, y_train, batch_size, shuffle=True, pin_memory=pin_memory)
    valid_loader = create_dataloader(X_valid, y_valid, batch_size, shuffle=False, pin_memory=pin_memory)

    # 优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # 早停
    early_stopping = EarlyStopping(
        patience=args.early_stop,
        min_delta=1e-5,
        restore_best=True,
    )

    # 训练循环
    loss_weights = params['loss_weights']
    print("\n    Training progress:")

    for epoch in range(args.n_epochs):
        train_losses = train_one_epoch(model, train_loader, optimizer, loss_weights, device)
        val_losses = validate(model, valid_loader, loss_weights, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_losses['action'])

        print(f"    Epoch {epoch+1:3d}/{args.n_epochs}: "
              f"train_loss={train_losses['total']:.4f}, "
              f"val_loss={val_losses['total']:.4f}, "
              f"val_action_loss={val_losses['action']:.4f}, "
              f"lr={current_lr:.2e}")

        if early_stopping(val_losses['action'], model):
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # 恢复最佳模型
    early_stopping.restore(model)

    # 验证集 IC
    valid_pred = predict_batch(model, X_valid, device, batch_size)
    valid_ic, valid_ic_std, valid_icir = compute_ic(valid_pred, y_valid, valid_index)
    print(f"\n    [Validation Set - for reference]")
    print(f"    Valid IC:   {valid_ic:.4f}")
    print(f"    Valid ICIR: {valid_icir:.4f}")

    # 测试集预测
    test_pred_values = predict_batch(model, X_test, device, batch_size)
    test_pred = pd.Series(test_pred_values, index=test_index, name='score')

    print(f"\n    Test prediction shape: {test_pred.shape}")
    print(f"    Test prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    return model, final_params, test_pred, dataset


def main():
    parser = argparse.ArgumentParser(
        description='AE-MLP PyTorch Cross-Validation Training with Loaded Parameters',
    )

    # 参数文件
    parser.add_argument('--params-file', type=str, default=None,
                        help='Path to JSON file with hyperparameters')

    # 模型评估模式
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to pre-trained model for evaluation (e.g., my_models/ae_mlp_pytorch_cv_xxx.pt)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate pre-trained model on CV folds, no training')

    # 基础参数
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--handler', type=str, default='alpha158-talib-macro',
                        choices=list(HANDLER_CONFIG.keys()))
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])

    # CV 训练参数
    parser.add_argument('--cv-epochs', type=int, default=50,
                        help='Epochs per CV fold')
    parser.add_argument('--cv-early-stop', type=int, default=10,
                        help='Early stopping patience for CV folds')
    parser.add_argument('--cv-only', action='store_true',
                        help='Only run CV training, skip final model')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (try different seeds like 42, 123, 2024)')
    parser.add_argument('--num-seeds', type=int, default=1,
                        help='Run CV with multiple seeds and report best result (e.g., --num-seeds 5)')

    # 最终训练参数
    parser.add_argument('--n-epochs', type=int, default=100,
                        help='Epochs for final model training')
    parser.add_argument('--early-stop', type=int, default=10,
                        help='Early stopping patience for final model')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')

    # 训练参数
    parser.add_argument('--verbose', action='store_true',
                        help='Show training progress for each fold')

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
    if not args.eval_only and not args.params_file:
        parser.error("--params-file is required for training mode")

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    # 初始化
    init_qlib(handler_config['use_talib'])

    # ========== 评估模式 ==========
    if args.eval_only:
        print("\n" + "=" * 70)
        print("AE-MLP PyTorch Cross-Validation EVALUATION Mode")
        print("=" * 70)
        print(f"Model: {args.model_path}")
        print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
        print(f"Handler: {args.handler}")
        print(f"N-day: {args.nday}")
        print(f"GPU: {args.gpu}")
        print(f"Backtest: {'ON' if args.backtest else 'OFF'}")
        print("=" * 70)

        fold_results, mean_ic, std_ic, test_pred, test_dataset = run_cv_evaluation(
            args, handler_config, symbols, args.model_path
        )

        # 回测（评估模式）
        if args.backtest:
            print("\n[*] Running backtest on 2025 test set...")
            pred_df = test_pred.to_frame("score")

            time_splits = {
                'train_start': FINAL_TEST['train_start'],
                'train_end': FINAL_TEST['train_end'],
                'valid_start': FINAL_TEST['valid_start'],
                'valid_end': FINAL_TEST['valid_end'],
                'test_start': FINAL_TEST['test_start'],
                'test_end': FINAL_TEST['test_end'],
            }

            device = setup_device(args.gpu)

            def load_model_func(path):
                model, _ = load_pytorch_model(str(path), device)
                return model

            def get_feature_count_func(m):
                return m.num_columns

            run_backtest(
                args.model_path, test_dataset, pred_df, args, time_splits,
                model_name="AE-MLP PyTorch (CV Eval)",
                load_model_func=load_model_func,
                get_feature_count_func=get_feature_count_func
            )

        return

    # ========== 训练模式 ==========
    # 加载超参数
    params, original_cv_results = load_params_from_json(args.params_file)

    # 打印头部
    print("\n" + "=" * 70)
    print("AE-MLP PyTorch Cross-Validation Training")
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Handler: {args.handler}")
    print(f"N-day: {args.nday}")
    print(f"CV epochs: {args.cv_epochs}")
    print(f"CV Folds: {len(CV_FOLDS)}")
    print(f"GPU: {args.gpu}")
    if args.num_seeds > 1:
        print(f"Num seeds: {args.num_seeds}")
    print("=" * 70)

    # 运行 CV 训练 (支持多种子)
    if args.num_seeds > 1:
        # 多种子模式：尝试多个种子，找到最好的结果
        base_seed = args.seed if args.seed is not None else 42
        all_results = []

        print(f"\n[*] Running CV with {args.num_seeds} different seeds...")
        for seed_idx in range(args.num_seeds):
            seed = base_seed + seed_idx * 1000
            args.seed = seed
            print(f"\n{'='*70}")
            print(f"SEED {seed_idx + 1}/{args.num_seeds}: seed={seed}")
            print(f"{'='*70}")

            fold_results, mean_ic, std_ic = run_cv_training(
                args, handler_config, symbols, params
            )
            # 计算测试集平均IC
            mean_test_ic = np.mean([r['test_ic'] for r in fold_results])
            all_results.append({
                'seed': seed,
                'mean_ic': mean_ic,
                'std_ic': std_ic,
                'mean_test_ic': mean_test_ic,
                'fold_results': fold_results,
            })

        # 找到最佳种子（基于验证集IC）
        best_result = max(all_results, key=lambda x: x['mean_ic'])
        # 也找出测试集最佳种子
        best_test_result = max(all_results, key=lambda x: x['mean_test_ic'])
        args.seed = best_result['seed']
        fold_results = best_result['fold_results']
        mean_ic = best_result['mean_ic']
        std_ic = best_result['std_ic']

        print("\n" + "=" * 70)
        print("MULTI-SEED SUMMARY")
        print("=" * 70)
        print(f"  {'Seed':<10s} {'Valid IC':>12s} {'Test IC (2025)':>16s}")
        print(f"  {'-'*10} {'-'*12} {'-'*16}")
        for r in all_results:
            valid_marker = " *" if r['seed'] == best_result['seed'] else ""
            test_marker = " #" if r['seed'] == best_test_result['seed'] else ""
            print(f"  {r['seed']:<10d} {r['mean_ic']:>10.4f}{valid_marker:<2s} {r['mean_test_ic']:>14.4f}{test_marker:<2s}")
        print(f"\n* Best valid seed: {best_result['seed']} (Valid IC={best_result['mean_ic']:.4f}, Test IC={best_result['mean_test_ic']:.4f})")
        print(f"# Best test seed:  {best_test_result['seed']} (Valid IC={best_test_result['mean_ic']:.4f}, Test IC={best_test_result['mean_test_ic']:.4f})")
        print("=" * 70)
    else:
        fold_results, mean_ic, std_ic = run_cv_training(
            args, handler_config, symbols, params
        )

    # 比较结果
    if original_cv_results:
        print("\n[*] Comparison with original results:")
        print(f"    Original Mean IC: {original_cv_results['mean_ic']:.4f}")
        print(f"    Current Mean IC:  {mean_ic:.4f}")
        diff = mean_ic - original_cv_results['mean_ic']
        print(f"    Difference: {diff:+.4f}")

    if args.cv_only:
        print("\n[*] CV-only mode, skipping final model training.")
        return

    # 训练最终模型
    model, final_params, test_pred, dataset = train_final_model(
        args, handler_config, symbols, params
    )

    # 评估
    print("\n[*] Final Evaluation on Test Set (2025)...")
    evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

    # 保存模型
    print("\n[*] Saving model...")
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_SAVE_PATH / f"ae_mlp_pytorch_cv_{args.handler}_{args.stock_pool}_{args.nday}d_{timestamp}.pt"
    save_pytorch_model(model, final_params, str(model_path))

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

        device = setup_device(args.gpu)

        def load_model(path):
            m, _ = load_pytorch_model(str(path), device)
            return m

        def get_feature_count(m):
            return m.num_columns

        run_backtest(
            model_path, dataset, pred_df, args, time_splits,
            model_name="AE-MLP PyTorch (CV)",
            load_model_func=load_model,
            get_feature_count_func=get_feature_count
        )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"CV Mean IC: {mean_ic:.4f} (±{std_ic:.4f})")
    print(f"Model saved to: {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
