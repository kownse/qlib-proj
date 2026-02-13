"""
时间序列交叉验证的公共工具函数。

提供 CV fold 配置、数据准备和 IC 计算等功能，
供 TCN Optuna 和 AE-MLP Hyperopt 等超参数搜索脚本使用。
"""

import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


# ============================================================================
# 时间序列交叉验证的 Fold 配置
# ============================================================================

# 4-fold CV 配置（与特征选择脚本一致）
# 包含 2021-2024 四年的验证集
CV_FOLDS = [
    {
        'name': 'Fold 1 (valid 2021)',
        'train_start': '2000-01-01',
        'train_end': '2020-12-31',
        'valid_start': '2021-01-01',
        'valid_end': '2021-12-31',
    },
    {
        'name': 'Fold 2 (valid 2022)',
        'train_start': '2000-01-01',
        'train_end': '2021-12-31',
        'valid_start': '2022-01-01',
        'valid_end': '2022-12-31',
    },
    {
        'name': 'Fold 3 (valid 2023)',
        'train_start': '2000-01-01',
        'train_end': '2022-12-31',
        'valid_start': '2023-01-01',
        'valid_end': '2023-12-31',
    },
    {
        'name': 'Fold 4 (valid 2024)',
        'train_start': '2000-01-01',
        'train_end': '2023-12-31',
        'valid_start': '2024-01-01',
        'valid_end': '2024-12-31',
    },
]

# 最终测试集 (完全独立)
FINAL_TEST = {
    'train_start': '2000-01-01',
    'train_end': '2023-12-31',
    'valid_start': '2024-01-01',
    'valid_end': '2024-12-31',
    'test_start': '2025-01-01',
    'test_end': '2025-12-31',
}


def create_data_handler_for_fold(args, handler_config, symbols, fold_config):
    """
    为特定 fold 创建 DataHandler。

    Args:
        args: 命令行参数，需要包含 handler 和 nday 属性
        handler_config: handler 配置字典
        symbols: 股票代码列表
        fold_config: fold 配置字典，包含 train_start, train_end, valid_start, valid_end
                     以及可选的 test_start, test_end

    Returns:
        DataHandler 实例
    """
    from models.common.handlers import get_handler_class

    HandlerClass = get_handler_class(args.handler)

    # 确定数据的结束时间
    end_time = fold_config.get('test_end', fold_config['valid_end'])

    handler_kwargs = {
        'volatility_window': args.nday,
        'instruments': symbols,
        'start_time': fold_config['train_start'],
        'end_time': end_time,
        'fit_start_time': fold_config['train_start'],
        'fit_end_time': fold_config['train_end'],
        'infer_processors': [],
    }

    # Apply default kwargs from handler registry (e.g., sector_features for sector handlers)
    if 'default_kwargs' in handler_config:
        for key, value in handler_config['default_kwargs'].items():
            if key not in handler_kwargs:
                handler_kwargs[key] = value

    handler = HandlerClass(**handler_kwargs)

    return handler


def create_dataset_for_fold(handler, fold_config):
    """
    为特定 fold 创建 Dataset。

    Args:
        handler: DataHandler 实例
        fold_config: fold 配置字典

    Returns:
        DatasetH 实例
    """
    segments = {
        "train": (fold_config['train_start'], fold_config['train_end']),
        "valid": (fold_config['valid_start'], fold_config['valid_end']),
    }

    if 'test_start' in fold_config:
        segments["test"] = (fold_config['test_start'], fold_config['test_end'])

    return DatasetH(handler=handler, segments=segments)


def prepare_data_from_dataset(dataset: DatasetH, segment: str):
    """
    从 Dataset 准备数据，处理 NaN 和异常值。

    Args:
        dataset: DatasetH 实例
        segment: 数据段名称 ("train", "valid", "test")

    Returns:
        tuple: (features_array, labels_array, index)
               如果没有 labels，labels_array 为 None
    """
    features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)

    # 处理 NaN 和异常值
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)
    features = features.clip(-10, 10)

    try:
        labels = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(labels, pd.DataFrame):
            labels = labels.iloc[:, 0]
        labels = labels.fillna(0).values
        return features.values, labels, features.index
    except Exception:
        return features.values, None, features.index


def compute_time_decay_weights(index, half_life_years, min_weight=0.1):
    """计算时间衰减权重"""
    if isinstance(index, pd.MultiIndex):
        datetimes = index.get_level_values(0)
    else:
        datetimes = index
    ref_date = datetimes.max()
    days_ago = (ref_date - datetimes).days
    years_ago = days_ago / 365.25
    decay_rate = np.log(2) / half_life_years
    weights = np.exp(-decay_rate * years_ago.values.astype(float))
    if min_weight > 0:
        weights = np.maximum(weights, min_weight)
    return weights


def prepare_features_and_labels(dataset, segment, top_features=None):
    """
    从 Dataset 准备特征 DataFrame 和标签 array。

    Args:
        dataset: DatasetH 实例
        segment: 数据段名称 ("train", "valid", "test")
        top_features: 可选的特征列表，用于特征选择

    Returns:
        tuple: (features_df: pd.DataFrame, labels: np.ndarray)
    """
    if top_features:
        features = dataset.prepare(segment, col_set="feature")[top_features]
    else:
        features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
    features = features.fillna(0).replace([np.inf, -np.inf], 0)

    label_df = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(label_df, pd.DataFrame):
        labels = label_df.iloc[:, 0].fillna(0).values
    else:
        labels = label_df.fillna(0).values

    return features, labels


def compute_ic(pred, label, index):
    """
    计算 IC (Information Coefficient)。

    按日期分组计算预测值与真实值的相关系数，然后取平均。

    Args:
        pred: 预测值数组
        label: 真实值数组
        index: MultiIndex，包含 datetime 级别

    Returns:
        tuple: (mean_ic, ic_std, icir)
    """
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


def print_cv_info(cv_folds=None, final_test=None):
    """
    打印 CV 配置信息。

    Args:
        cv_folds: CV fold 配置列表，默认使用 CV_FOLDS
        final_test: 最终测试配置，默认使用 FINAL_TEST
    """
    if cv_folds is None:
        cv_folds = CV_FOLDS
    if final_test is None:
        final_test = FINAL_TEST

    print(f"CV Folds: {len(cv_folds)}")
    for fold in cv_folds:
        print(f"  - {fold['name']}: train {fold['train_start']}~{fold['train_end']}, "
              f"valid {fold['valid_start']}~{fold['valid_end']}")

    print(f"Final Test: {final_test['test_start']}~{final_test['test_end']}")


def prepare_cv_fold_data(args, handler_config, symbols, cv_folds=None, verbose=True):
    """
    预先准备所有 CV fold 的数据。

    Args:
        args: 命令行参数
        handler_config: handler 配置
        symbols: 股票代码列表
        cv_folds: CV fold 配置列表，默认使用 CV_FOLDS
        verbose: 是否打印进度信息

    Returns:
        tuple: (fold_data_list, num_features)
            fold_data_list: 包含每个 fold 数据的字典列表
            num_features: 特征数量
    """
    if cv_folds is None:
        cv_folds = CV_FOLDS

    if verbose:
        print("\n[*] Preparing data for all CV folds...")

    fold_data = []
    num_features = None

    for fold in cv_folds:
        if verbose:
            print(f"    Preparing {fold['name']}...")

        handler = create_data_handler_for_fold(args, handler_config, symbols, fold)
        dataset = create_dataset_for_fold(handler, fold)

        X_train, y_train, _ = prepare_data_from_dataset(dataset, "train")
        X_valid, y_valid, valid_index = prepare_data_from_dataset(dataset, "valid")

        if num_features is None:
            num_features = X_train.shape[1]

        fold_data.append({
            'name': fold['name'],
            'X_train': X_train,
            'y_train': y_train,
            'X_valid': X_valid,
            'y_valid': y_valid,
            'valid_index': valid_index,
        })

        if verbose:
            print(f"      Train: {X_train.shape}, Valid: {X_valid.shape}")

    if verbose:
        print(f"    ✓ All {len(cv_folds)} folds prepared")
        print(f"    Feature count: {num_features}")

    return fold_data, num_features
