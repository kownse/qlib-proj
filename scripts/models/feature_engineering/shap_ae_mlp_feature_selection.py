"""
SHAP-based Feature Selection using AE-MLP with alpha158-talib-macro handler

基于 SHAP 值分析 AE-MLP 模型的特征重要性，使用全量宏观特征数据。
支持多种特征筛选策略：
1. 按 SHAP importance 排序
2. 按稳定性 (importance / std) 筛选
3. 排除负贡献特征

使用方法:
    python scripts/models/feature_engineering/shap_ae_mlp_feature_selection.py --stock-pool sp500
    python scripts/models/feature_engineering/shap_ae_mlp_feature_selection.py --stock-pool sp500 --top-k 50
    python scripts/models/feature_engineering/shap_ae_mlp_feature_selection.py --stock-pool sp500 --min-importance 0.001

    # 使用预训练模型
    python scripts/models/feature_engineering/shap_ae_mlp_feature_selection.py --stock-pool sp500 \
        --model-path my_models/ae_mlp_xxx.keras
"""

# ============================================================================
# 环境设置 (必须在其他导入之前)
# ============================================================================

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

import sys
from pathlib import Path
import multiprocessing

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

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
    kernels=1,
    joblib_backend=None,
)

# ============================================================================
# 其他导入
# ============================================================================

import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import shap

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, callbacks

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from data.stock_pools import STOCK_POOLS
from models.common import (
    HANDLER_CONFIG, PROJECT_ROOT, MODEL_SAVE_PATH,
    get_time_splits,
)
from models.deep.ae_mlp_shared import build_ae_mlp_model, setup_gpu


# ============================================================================
# 默认配置
# ============================================================================

# AE-MLP 默认参数 (来自 hyperopt 搜索)
DEFAULT_AE_MLP_PARAMS = {
    'hidden_units': [48, 112, 256, 224, 48],
    'dropout_rates': [0.091, 0.148, 0.148, 0.148, 0.148, 0.148, 0.148],
    'lr': 0.00853,
    'batch_size': 4096,
    'loss_weights': {
        'decoder': 0.151,
        'ae_action': 0.115,
        'action': 1.0
    }
}

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "feature_selection"


# ============================================================================
# 数据加载
# ============================================================================

def load_data(
    handler_name: str,
    symbols: List[str],
    time_splits: Dict[str, str],
    nday: int = 5,
) -> Tuple[DatasetH, List[str]]:
    """
    加载数据集

    Args:
        handler_name: handler 名称
        symbols: 股票代码列表
        time_splits: 时间划分配置
        nday: 预测天数

    Returns:
        (dataset, feature_names)
    """
    handler_config = HANDLER_CONFIG[handler_name]
    handler_class = handler_config['class']

    print(f"\n[1] Creating DataHandler: {handler_name}")
    print(f"    Description: {handler_config['description']}")
    print(f"    Symbols: {len(symbols)}")
    print(f"    N-day: {nday}")

    handler = handler_class(
        volatility_window=nday,
        instruments=symbols,
        start_time=time_splits['train_start'],
        end_time=time_splits['test_end'],
        fit_start_time=time_splits['train_start'],
        fit_end_time=time_splits['train_end'],
    )

    print(f"\n[2] Creating Dataset...")
    segments = {
        "train": (time_splits['train_start'], time_splits['train_end']),
        "valid": (time_splits['valid_start'], time_splits['valid_end']),
        "test": (time_splits['test_start'], time_splits['test_end']),
    }
    dataset = DatasetH(handler=handler, segments=segments)

    # 获取特征名
    train_data = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
    feature_names = train_data.columns.tolist()

    print(f"    Train samples: {len(train_data)}")
    print(f"    Features: {len(feature_names)}")

    return dataset, feature_names


def prepare_data(
    dataset: DatasetH,
    segment: str,
) -> Tuple[np.ndarray, np.ndarray, pd.MultiIndex]:
    """
    准备训练/验证数据

    Args:
        dataset: 数据集
        segment: 数据段 ("train", "valid", "test")

    Returns:
        (X, y, index)
    """
    X = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
    y = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # 处理缺失值和异常值
    X = X.fillna(0).replace([np.inf, -np.inf], 0).clip(-10, 10)
    y = y.fillna(0)

    index = X.index
    return X.values, y.values, index



def create_prediction_model(full_model: Model) -> Model:
    """
    创建只输出主预测的模型 (用于 SHAP)

    Args:
        full_model: 完整的 AE-MLP 模型 (3个输出)

    Returns:
        只有主输出的模型
    """
    # 获取主输出 (action)
    action_output = full_model.get_layer('action').output
    pred_model = Model(inputs=full_model.input, outputs=action_output, name='AE_MLP_Pred')
    return pred_model


# ============================================================================
# AE-MLP 训练
# ============================================================================

def train_ae_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    params: dict = None,
    n_epochs: int = 50,
    early_stop: int = 10,
) -> Model:
    """
    训练 AE-MLP 模型

    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_valid: 验证特征
        y_valid: 验证标签
        params: 模型参数
        n_epochs: 训练轮数
        early_stop: 早停耐心值

    Returns:
        训练好的模型
    """
    if params is None:
        params = DEFAULT_AE_MLP_PARAMS.copy()

    num_columns = X_train.shape[1]

    print("\n[3] Training AE-MLP model...")
    print("    Parameters:")
    print(f"      hidden_units: {params['hidden_units']}")
    print(f"      learning_rate: {params['lr']}")
    print(f"      batch_size: {params['batch_size']}")
    print(f"      epochs: {n_epochs}")

    model = build_ae_mlp_model({**params, 'num_columns': num_columns})

    # 回调
    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_action_loss',
            patience=early_stop,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_action_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0,
            mode='min'
        ),
    ]

    # 训练数据
    train_outputs = {
        'decoder': X_train,
        'ae_action': y_train,
        'action': y_train,
    }
    valid_outputs = {
        'decoder': X_valid,
        'ae_action': y_valid,
        'action': y_valid,
    }

    batch_size = params['batch_size']

    model.fit(
        X_train,
        train_outputs,
        validation_data=(X_valid, valid_outputs),
        epochs=n_epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1,
    )

    return model


# ============================================================================
# SHAP 分析
# ============================================================================

def compute_shap_importance(
    model: Model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    background_size: int = 500,
    explain_size: int = 2000,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    计算 SHAP 特征重要性

    使用 DeepExplainer 或 GradientExplainer 计算深度学习模型的 SHAP 值

    Args:
        model: 训练好的模型 (只有主输出)
        X_background: 背景数据 (用于计算期望值)
        X_explain: 要解释的数据
        feature_names: 特征名列表
        background_size: 背景样本大小
        explain_size: 解释样本大小

    Returns:
        (shap_df, shap_values, X_sample)
    """
    print("\n[4] Computing SHAP values...")

    # 采样背景数据
    if len(X_background) > background_size:
        bg_indices = np.random.choice(len(X_background), background_size, replace=False)
        X_bg = X_background[bg_indices]
    else:
        X_bg = X_background

    # 采样解释数据
    if len(X_explain) > explain_size:
        exp_indices = np.random.choice(len(X_explain), explain_size, replace=False)
        X_exp = X_explain[exp_indices]
    else:
        X_exp = X_explain

    print(f"    Background samples: {len(X_bg)}")
    print(f"    Explain samples: {len(X_exp)}")

    # 使用 GradientExplainer (更稳定，适合复杂模型)
    print("    Using GradientExplainer...")
    try:
        explainer = shap.GradientExplainer(model, X_bg)
        shap_values = explainer.shap_values(X_exp)

        # GradientExplainer 返回列表，取第一个
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # 如果是 3D (samples, features, 1)，压缩最后一维
        if len(shap_values.shape) == 3:
            shap_values = shap_values.squeeze(-1)

    except Exception as e:
        print(f"    GradientExplainer failed: {e}")
        print("    Falling back to KernelExplainer (slower)...")

        # 使用 KernelExplainer 作为后备
        def model_predict(x):
            return model.predict(x, verbose=0).flatten()

        # 进一步减少样本以加速
        if len(X_bg) > 100:
            X_bg = X_bg[:100]
        if len(X_exp) > 500:
            X_exp = X_exp[:500]

        explainer = shap.KernelExplainer(model_predict, X_bg)
        shap_values = explainer.shap_values(X_exp, nsamples=100)

    print(f"    SHAP values shape: {shap_values.shape}")

    # 计算特征重要性指标
    importance = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)
    std_shap = shap_values.std(axis=0)
    stability = importance / (std_shap + 1e-8)

    # 构建 DataFrame
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': importance,
        'shap_mean': mean_shap,
        'shap_std': std_shap,
        'shap_stability': stability,
    })

    # 排序
    shap_df = shap_df.sort_values('shap_importance', ascending=False).reset_index(drop=True)

    return shap_df, shap_values, X_exp


def select_features_by_shap(
    shap_df: pd.DataFrame,
    top_k: int = None,
    min_importance: float = None,
    min_stability: float = None,
    exclude_negative: bool = False,
) -> pd.DataFrame:
    """
    基于 SHAP 值筛选特征

    Args:
        shap_df: SHAP 重要性 DataFrame
        top_k: 选择前 k 个特征
        min_importance: 最小重要性阈值
        min_stability: 最小稳定性阈值
        exclude_negative: 是否排除平均贡献为负的特征

    Returns:
        筛选后的 DataFrame
    """
    selected = shap_df.copy()

    if exclude_negative:
        selected = selected[selected['shap_mean'] >= 0]
        print(f"    After excluding negative contribution: {len(selected)} features")

    if min_importance is not None:
        selected = selected[selected['shap_importance'] >= min_importance]
        print(f"    After min_importance filter: {len(selected)} features")

    if min_stability is not None:
        selected = selected[selected['shap_stability'] >= min_stability]
        print(f"    After min_stability filter: {len(selected)} features")

    if top_k is not None and len(selected) > top_k:
        selected = selected.head(top_k)
        print(f"    After top-k selection: {len(selected)} features")

    return selected.reset_index(drop=True)


# ============================================================================
# 评估
# ============================================================================

def evaluate_with_features(
    dataset: DatasetH,
    selected_features: List[str],
    params: dict = None,
    n_epochs: int = 50,
    early_stop: int = 10,
) -> Dict:
    """
    使用选定特征重新训练并评估

    Args:
        dataset: 数据集
        selected_features: 选定的特征列表
        params: AE-MLP 参数
        n_epochs: 训练轮数
        early_stop: 早停耐心值

    Returns:
        评估结果字典
    """
    if params is None:
        params = DEFAULT_AE_MLP_PARAMS.copy()

    print(f"\n[6] Evaluating with {len(selected_features)} selected features...")

    # 准备数据
    train_data = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
    valid_data = dataset.prepare("valid", col_set="feature", data_key=DataHandlerLP.DK_L)
    test_data = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_L)

    train_label = dataset.prepare("train", col_set="label", data_key=DataHandlerLP.DK_L)
    valid_label = dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
    test_label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)

    if isinstance(train_label, pd.DataFrame):
        train_label = train_label.iloc[:, 0]
    if isinstance(valid_label, pd.DataFrame):
        valid_label = valid_label.iloc[:, 0]
    if isinstance(test_label, pd.DataFrame):
        test_label = test_label.iloc[:, 0]

    # 筛选特征
    X_train = train_data[selected_features].fillna(0).replace([np.inf, -np.inf], 0).clip(-10, 10).values
    X_valid = valid_data[selected_features].fillna(0).replace([np.inf, -np.inf], 0).clip(-10, 10).values
    X_test = test_data[selected_features].fillna(0).replace([np.inf, -np.inf], 0).clip(-10, 10).values

    y_train = train_label.fillna(0).values
    y_valid = valid_label.fillna(0).values
    y_test = test_label.fillna(0).values

    # 更新特征数
    eval_params = params.copy()
    num_columns = len(selected_features)

    # 训练模型
    tf.keras.backend.clear_session()
    model = build_ae_mlp_model({**eval_params, 'num_columns': num_columns})

    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_action_loss',
            patience=early_stop,
            restore_best_weights=True,
            verbose=0,
            mode='min'
        ),
    ]

    train_outputs = {'decoder': X_train, 'ae_action': y_train, 'action': y_train}
    valid_outputs = {'decoder': X_valid, 'ae_action': y_valid, 'action': y_valid}

    model.fit(
        X_train, train_outputs,
        validation_data=(X_valid, valid_outputs),
        epochs=n_epochs,
        batch_size=params['batch_size'],
        callbacks=cb_list,
        verbose=0,
    )

    # 预测
    _, _, valid_pred = model.predict(X_valid, batch_size=params['batch_size'], verbose=0)
    _, _, test_pred = model.predict(X_test, batch_size=params['batch_size'], verbose=0)
    valid_pred = valid_pred.flatten()
    test_pred = test_pred.flatten()

    # 计算 IC
    def compute_ic(pred, label, index):
        df = pd.DataFrame({'pred': pred, 'label': label}, index=index)
        ic_by_date = df.groupby(level='datetime').apply(
            lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
        )
        ic_by_date = ic_by_date.dropna()
        return ic_by_date.mean() if len(ic_by_date) > 0 else 0.0

    valid_ic = compute_ic(valid_pred, y_valid, valid_data.index)
    test_ic = compute_ic(test_pred, y_test, test_data.index)

    print(f"    Valid IC: {valid_ic:.4f}")
    print(f"    Test IC:  {test_ic:.4f}")

    return {
        'valid_ic': valid_ic,
        'test_ic': test_ic,
        'num_features': len(selected_features),
    }


# ============================================================================
# 可视化
# ============================================================================

def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_names: List[str],
    output_path: Path,
    top_n: int = 30,
):
    """
    绘制 SHAP 摘要图

    Args:
        shap_values: SHAP 值
        X_sample: 采样数据
        feature_names: 特征名
        output_path: 输出路径
        top_n: 显示前 n 个特征
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print(f"\n[5] Generating SHAP plots...")

    # Summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=top_n,
        show=False,
    )
    plt.tight_layout()
    summary_path = output_path / "shap_ae_mlp_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {summary_path}")

    # Bar plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=top_n,
        show=False,
    )
    plt.tight_layout()
    bar_path = output_path / "shap_ae_mlp_importance_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {bar_path}")


# ============================================================================
# 保存结果
# ============================================================================

def save_results(
    output_dir: Path,
    shap_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    eval_results: Dict,
    args: argparse.Namespace,
):
    """
    保存分析结果

    Args:
        output_dir: 输出目录
        shap_df: 完整 SHAP 重要性 DataFrame
        selected_df: 筛选后的特征 DataFrame
        eval_results: 评估结果
        args: 命令行参数
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存完整 SHAP 重要性
    full_path = output_dir / f"shap_ae_mlp_full_{timestamp}.csv"
    shap_df.to_csv(full_path, index=False)
    print(f"    Full SHAP importance: {full_path}")

    # 保存筛选后的特征
    selected_path = output_dir / f"shap_ae_mlp_selected_{timestamp}.csv"
    selected_df.to_csv(selected_path, index=False)
    print(f"    Selected features: {selected_path}")

    # 保存元数据
    meta = {
        'timestamp': timestamp,
        'handler': args.handler,
        'stock_pool': args.stock_pool,
        'nday': args.nday,
        'top_k': args.top_k,
        'min_importance': args.min_importance,
        'min_stability': args.min_stability,
        'exclude_negative': args.exclude_negative,
        'total_features': len(shap_df),
        'selected_features': len(selected_df),
        'eval_results': eval_results,
        'selected_feature_names': selected_df['feature'].tolist(),
    }

    meta_path = output_dir / f"shap_ae_mlp_meta_{timestamp}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"    Metadata: {meta_path}")

    return timestamp


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SHAP-based Feature Selection using AE-MLP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 数据参数
    parser.add_argument('--handler', type=str, default='alpha158-talib-macro',
                        help='DataHandler name (default: alpha158-talib-macro)')
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'],
                        help='Stock pool (default: sp500)')
    parser.add_argument('--nday', type=int, default=5,
                        help='Prediction window in days (default: 5)')
    parser.add_argument('--max-train', action='store_true',
                        help='Use max training time splits')

    # 模型参数
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to pre-trained AE-MLP model')
    parser.add_argument('--n-epochs', type=int, default=50,
                        help='Training epochs (default: 50)')
    parser.add_argument('--early-stop', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')

    # SHAP 参数
    parser.add_argument('--background-size', type=int, default=500,
                        help='Background sample size for SHAP (default: 500)')
    parser.add_argument('--explain-size', type=int, default=2000,
                        help='Explain sample size for SHAP (default: 2000)')

    # 特征筛选参数
    parser.add_argument('--top-k', type=int, default=None,
                        help='Select top-k features')
    parser.add_argument('--min-importance', type=float, default=None,
                        help='Minimum SHAP importance threshold')
    parser.add_argument('--min-stability', type=float, default=None,
                        help='Minimum stability threshold (importance/std)')
    parser.add_argument('--exclude-negative', action='store_true',
                        help='Exclude features with negative mean SHAP')

    # 输出参数
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip SHAP plots')
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip evaluation with selected features')

    args = parser.parse_args()

    # 设置 GPU
    setup_gpu(args.gpu)

    # 获取配置
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = get_time_splits(args.max_train)

    print("=" * 70)
    print("SHAP-based Feature Selection using AE-MLP")
    print("=" * 70)
    print(f"Handler:     {args.handler}")
    print(f"Stock Pool:  {args.stock_pool} ({len(symbols)} stocks)")
    print(f"N-day:       {args.nday}")
    print(f"Train:       {time_splits['train_start']} to {time_splits['train_end']}")
    print(f"Valid:       {time_splits['valid_start']} to {time_splits['valid_end']}")
    print(f"Test:        {time_splits['test_start']} to {time_splits['test_end']}")
    print("=" * 70)

    # 加载数据
    dataset, feature_names = load_data(
        args.handler,
        symbols,
        time_splits,
        args.nday,
    )

    # 准备数据
    X_train, y_train, train_index = prepare_data(dataset, "train")
    X_valid, y_valid, valid_index = prepare_data(dataset, "valid")

    print(f"\n    Train shape: {X_train.shape}")
    print(f"    Valid shape: {X_valid.shape}")

    # 加载或训练模型
    if args.model_path:
        print(f"\n[3] Loading pre-trained model: {args.model_path}")
        full_model = keras.models.load_model(args.model_path)
    else:
        full_model = train_ae_mlp(
            X_train, y_train,
            X_valid, y_valid,
            n_epochs=args.n_epochs,
            early_stop=args.early_stop,
        )

    # 创建只有主输出的模型 (用于 SHAP)
    pred_model = create_prediction_model(full_model)

    # 计算 SHAP 值
    shap_df, shap_values, X_sample = compute_shap_importance(
        pred_model,
        X_train,
        X_valid,
        feature_names,
        background_size=args.background_size,
        explain_size=args.explain_size,
    )

    # 打印 SHAP 重要性
    print("\n[*] Top 30 Features by SHAP Importance:")
    print("-" * 70)
    for i, row in shap_df.head(30).iterrows():
        sign = "+" if row['shap_mean'] >= 0 else "-"
        print(f"    {i+1:3d}. {row['feature']:<45s} "
              f"imp={row['shap_importance']:.4f} {sign} "
              f"stab={row['shap_stability']:.2f}")
    print("-" * 70)

    # 创建输出目录
    output_dir = OUTPUT_DIR / "shap_ae_mlp"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 绘制 SHAP 图
    if not args.no_plot:
        plot_shap_summary(shap_values, X_sample, feature_names, output_dir)

    # 筛选特征
    print("\n[*] Selecting features...")
    selected_df = select_features_by_shap(
        shap_df,
        top_k=args.top_k,
        min_importance=args.min_importance,
        min_stability=args.min_stability,
        exclude_negative=args.exclude_negative,
    )

    print(f"\n    Selected {len(selected_df)} features")

    # 评估
    eval_results = {}
    if not args.no_eval and len(selected_df) > 0:
        selected_features = selected_df['feature'].tolist()
        eval_results = evaluate_with_features(
            dataset, selected_features,
            n_epochs=args.n_epochs,
            early_stop=args.early_stop,
        )

    # 保存结果
    print("\n[7] Saving results...")
    timestamp = save_results(output_dir, shap_df, selected_df, eval_results, args)

    # 打印最终选定的特征
    print("\n" + "=" * 70)
    print("SELECTED FEATURES")
    print("=" * 70)
    for i, row in selected_df.iterrows():
        print(f"  {i+1:3d}. {row['feature']}")
    print("=" * 70)
    print(f"Total: {len(selected_df)} features")

    if eval_results:
        print(f"\nEvaluation Results:")
        print(f"  Valid IC: {eval_results['valid_ic']:.4f}")
        print(f"  Test IC:  {eval_results['test_ic']:.4f}")

    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
