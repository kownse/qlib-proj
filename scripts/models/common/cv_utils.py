"""
时间序列交叉验证的公共工具函数。

提供 CV fold 配置、数据准备和 IC 计算等功能，
供 TCN Optuna 和 AE-MLP Hyperopt 等超参数搜索脚本使用。
"""

import sys
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


# ============================================================================
# Hyperopt CV 通用基础设施
# ============================================================================


class BaseCVHyperoptObjective:
    """时间序列交叉验证的 Hyperopt 目标函数基类。

    Subclasses must implement:
        create_model_params(hyperparams) -> dict
        train_and_predict_fold(fold_data, params, hyperparams)
            -> (train_pred, valid_pred, best_iter)
        format_trial_params(hyperparams) -> str

    Optional overrides:
        _init_extra(**kwargs): extra initialization
        _augment_fold_data(entry, fold): augment fold entry (e.g., create Pool)
        get_extra_trial_info(hyperparams) -> str: extra info for trial output
    """

    def __init__(self, args, handler_config, symbols,
                 top_features=None, verbose=False, **kwargs):
        self.args = args
        self.handler_config = handler_config
        self.symbols = symbols
        self.top_features = top_features
        self.verbose = verbose
        self.trial_count = 0
        self.best_mean_ic = -float('inf')

        self._init_extra(**kwargs)
        self._prepare_fold_data()

    def _init_extra(self, **kwargs):
        """Override for subclass-specific initialization."""
        pass

    def _prepare_fold_data(self):
        """Prepare data for all CV folds."""
        print("\n[*] Preparing data for all CV folds...")
        self.fold_data = []

        for fold in CV_FOLDS:
            print(f"    Preparing {fold['name']}...")
            handler = create_data_handler_for_fold(
                self.args, self.handler_config, self.symbols, fold)
            dataset = create_dataset_for_fold(handler, fold)

            train_data, train_label = prepare_features_and_labels(
                dataset, "train", self.top_features)
            valid_data, valid_label = prepare_features_and_labels(
                dataset, "valid", self.top_features)

            if len(self.fold_data) == 0:
                self._print_debug_stats(
                    fold, train_data, valid_data, train_label, valid_label)

            entry = {
                'name': fold['name'],
                'train_data': train_data,
                'valid_data': valid_data,
                'train_label': train_label,
                'valid_label': valid_label,
            }

            self._augment_fold_data(entry, fold)
            self.fold_data.append(entry)
            print(f"      Train: {train_data.shape}, Valid: {valid_data.shape}")

        print(f"    ✓ All {len(CV_FOLDS)} folds prepared")

    def _print_debug_stats(self, fold, train_data, valid_data,
                           train_label, valid_label):
        """Print debug statistics for first fold."""
        print(f"\n      [DEBUG Data Prep - {fold['name']}]")
        print(f"        train_data columns ({len(train_data.columns)}): "
              f"{train_data.columns.tolist()[:10]}...")
        print(f"        train_data NaN%: "
              f"{train_data.isna().mean().mean()*100:.2f}%")
        print(f"        valid_data NaN%: "
              f"{valid_data.isna().mean().mean()*100:.2f}%")
        print(f"        train_label: mean={train_label.mean():.6f}, "
              f"std={train_label.std():.6f}, "
              f"NaN={np.isnan(train_label).sum()}")
        print(f"        valid_label: mean={valid_label.mean():.6f}, "
              f"std={valid_label.std():.6f}, "
              f"NaN={np.isnan(valid_label).sum()}")
        valid_label_df = pd.DataFrame(
            {'label': valid_label}, index=valid_data.index)
        label_std_per_date = valid_label_df.groupby(
            level='datetime')['label'].std()
        print(f"        valid_label std per date: "
              f"min={label_std_per_date.min():.6f}, "
              f"max={label_std_per_date.max():.6f}, "
              f"mean={label_std_per_date.mean():.6f}")

    def _augment_fold_data(self, entry, fold):
        """Override to add model-specific data to fold entry."""
        pass

    def create_model_params(self, hyperparams):
        """Convert hyperopt hyperparams to model parameters. Must return dict."""
        raise NotImplementedError

    def train_and_predict_fold(self, fold_data, params, hyperparams):
        """Train on fold and return (train_pred, valid_pred, best_iter)."""
        raise NotImplementedError

    def format_trial_params(self, hyperparams):
        """Format hyperparams as concise string for logging."""
        raise NotImplementedError

    def get_extra_trial_info(self, hyperparams):
        """Return extra info string for trial output (e.g., weight info)."""
        return ""

    def __call__(self, hyperparams):
        """Objective function: train on all folds and return mean validation IC."""
        from hyperopt import STATUS_OK, STATUS_FAIL

        self.trial_count += 1
        params = self.create_model_params(hyperparams)
        extra_info = self.get_extra_trial_info(hyperparams)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Trial {self.trial_count}/{self.args.max_evals} | "
                  f"Best IC so far: {self.best_mean_ic:.4f}")
            print(f"{'='*60}")
            print(f"  Params: {self.format_trial_params(hyperparams)}"
                  f"{extra_info}")
            sys.stdout.flush()

        fold_ics = []
        fold_results = []

        try:
            for fold_idx, fold in enumerate(self.fold_data):
                if self.verbose:
                    print(f"\n  [{fold_idx+1}/{len(self.fold_data)}] "
                          f"{fold['name']}...")
                    sys.stdout.flush()

                train_pred, valid_pred, best_iter = \
                    self.train_and_predict_fold(fold, params, hyperparams)

                train_ic, _, train_icir = compute_ic(
                    train_pred, fold['train_label'],
                    fold['train_data'].index)
                mean_ic, ic_std, icir = compute_ic(
                    valid_pred, fold['valid_label'],
                    fold['valid_data'].index)

                fold_ics.append(mean_ic)
                fold_results.append({
                    'name': fold['name'],
                    'ic': mean_ic,
                    'icir': icir,
                    'best_iter': best_iter,
                })

                if self.verbose:
                    print(f"      Best iter: {best_iter}")
                    print(f"      Train IC: {train_ic:.4f} "
                          f"(ICIR: {train_icir:.4f})")
                    print(f"      Valid IC: {mean_ic:.4f} "
                          f"(ICIR: {icir:.4f})")
                    sys.stdout.flush()

            mean_ic_all = np.mean(fold_ics)
            std_ic_all = np.std(fold_ics)

            if mean_ic_all > self.best_mean_ic:
                self.best_mean_ic = mean_ic_all
                is_best = " ★ NEW BEST"
            else:
                is_best = ""

            fold_ic_str = ", ".join(
                [f"{r['ic']:.4f}" for r in fold_results])
            if self.verbose:
                print(f"\n  {'─'*50}")
                print(f"  Trial {self.trial_count} Summary:")
                print(f"    Mean IC: {mean_ic_all:.4f} "
                      f"(±{std_ic_all:.4f})")
                print(f"    Folds:   [{fold_ic_str}]")
                print(f"    Best Trial IC: "
                      f"{self.best_mean_ic:.4f}{is_best}")
                print(f"  {'─'*50}")
            else:
                print(f"Trial {self.trial_count:3d}: "
                      f"Mean IC={mean_ic_all:.4f} "
                      f"(±{std_ic_all:.4f}) [{fold_ic_str}] "
                      f"{self.format_trial_params(hyperparams)}"
                      f"{extra_info}{is_best}")
            sys.stdout.flush()

            return {
                'loss': -mean_ic_all,
                'status': STATUS_OK,
                'mean_ic': mean_ic_all,
                'std_ic': std_ic_all,
                'fold_results': fold_results,
                'params': params,
            }

        except Exception as e:
            import traceback
            print(f"\n  Trial {self.trial_count} FAILED!")
            print(f"    Error: {str(e)}")
            traceback.print_exc()
            sys.stdout.flush()
            return {
                'loss': float('inf'),
                'status': STATUS_FAIL,
                'error': str(e),
            }


def run_hyperopt_cv_search_generic(objective, search_space, max_evals,
                                   create_params_func,
                                   param_filter_keys=None,
                                   extra_header_lines=None):
    """Run hyperopt search with time-series CV.

    Parameters
    ----------
    objective : BaseCVHyperoptObjective
    search_space : dict
    max_evals : int
    create_params_func : callable
        (best_raw_dict) -> model_params_dict
    param_filter_keys : set, optional
        Parameter keys to exclude from result printing
    extra_header_lines : list of str, optional
        Extra lines to print in the header

    Returns
    -------
    tuple
        (best_raw, best_params, trials, best_trial)
    """
    from hyperopt import fmin, tpe, Trials

    if param_filter_keys is None:
        param_filter_keys = set()

    print("\n" + "=" * 70)
    print("HYPEROPT SEARCH WITH TIME-SERIES CROSS-VALIDATION")
    print("=" * 70)
    print(f"CV Folds: {len(CV_FOLDS)}")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: train {fold['train_start']}~"
              f"{fold['train_end']}, "
              f"valid {fold['valid_start']}~{fold['valid_end']}")
    print(f"Max evaluations: {max_evals}")
    if extra_header_lines:
        for line in extra_header_lines:
            print(line)
    print("=" * 70)

    trials = Trials()
    print("\n[*] Running hyperparameter search...")

    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        show_progressbar=False,
    )

    best_params = create_params_func(best)
    best_trial_idx = np.argmin(
        [t['result']['loss'] for t in trials.trials])
    best_trial = trials.trials[best_trial_idx]['result']

    print("\n" + "=" * 70)
    print("CV HYPEROPT SEARCH COMPLETE")
    print("=" * 70)
    print(f"Best Mean IC: {best_trial['mean_ic']:.4f} "
          f"(±{best_trial['std_ic']:.4f})")
    print("\nIC by fold:")
    for r in best_trial['fold_results']:
        print(f"  {r['name']}: IC={r['ic']:.4f}, "
              f"ICIR={r['icir']:.4f}, iter={r['best_iter']}")
    print("\nBest parameters:")
    for key, value in best_params.items():
        if key not in param_filter_keys:
            print(f"  {key}: {value}")
    print("=" * 70)

    return best, best_params, trials, best_trial


def first_pass_feature_selection_generic(args, handler_config, symbols,
                                         train_and_get_importance):
    """Generic first-pass feature selection.

    Parameters
    ----------
    train_and_get_importance : callable
        (train_data, train_label, valid_data, valid_label, args)
            -> (feature_names: list, importance_values: array)

    Returns
    -------
    pd.DataFrame
        Feature importance sorted by importance descending
    """
    print("\n[*] First pass: feature selection using Fold 2...")

    fold = CV_FOLDS[1]
    handler = create_data_handler_for_fold(args, handler_config, symbols, fold)
    dataset = create_dataset_for_fold(handler, fold)

    train_data, train_label = prepare_features_and_labels(dataset, "train")
    valid_data, valid_label = prepare_features_and_labels(dataset, "valid")

    feature_names, importance = train_and_get_importance(
        train_data, train_label, valid_data, valid_label, args
    )

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("\n    Top 20 features:")
    print("    " + "-" * 50)
    for i, row in importance_df.head(20).iterrows():
        print(f"    {importance_df.index.get_loc(i)+1:3d}. "
              f"{row['feature']:<40s} {row['importance']:>10.2f}")
    print("    " + "-" * 50)

    return importance_df


def train_final_model_generic(args, handler_config, symbols, best_params,
                              final_test_config, train_predict_func,
                              top_features=None, param_filter_keys=None,
                              **kwargs):
    """Generic final model training with best hyperparams.

    Parameters
    ----------
    train_predict_func : callable
        (train_data, train_label, valid_data, valid_label,
         test_data, best_params, **kwargs)
            -> (model, valid_pred, test_pred_values)
    param_filter_keys : set, optional
        Parameter keys to exclude from printing

    Returns
    -------
    tuple
        (model, feature_names, test_pred, dataset)
    """
    if param_filter_keys is None:
        param_filter_keys = set()

    print("\n[*] Training final model on full data...")
    print("    Parameters:")
    for key, value in best_params.items():
        if key not in param_filter_keys:
            print(f"      {key}: {value}")

    handler = create_data_handler_for_fold(
        args, handler_config, symbols, final_test_config)
    dataset = create_dataset_for_fold(handler, final_test_config)

    train_data, train_label = prepare_features_and_labels(
        dataset, "train", top_features)
    valid_data, valid_label = prepare_features_and_labels(
        dataset, "valid", top_features)
    test_data, _ = prepare_features_and_labels(
        dataset, "test", top_features)
    feature_names = (top_features if top_features
                     else train_data.columns.tolist())

    print(f"\n    Final training data:")
    print(f"      Train: {train_data.shape} "
          f"({final_test_config['train_start']} ~ "
          f"{final_test_config['train_end']})")
    print(f"      Valid: {valid_data.shape} "
          f"({final_test_config['valid_start']} ~ "
          f"{final_test_config['valid_end']})")
    print(f"      Test:  {test_data.shape} "
          f"({final_test_config['test_start']} ~ "
          f"{final_test_config['test_end']})")

    model, valid_pred, test_pred_values = train_predict_func(
        train_data, train_label, valid_data, valid_label,
        test_data, best_params, **kwargs
    )

    valid_ic, valid_ic_std, valid_icir = compute_ic(
        valid_pred, valid_label, valid_data.index)
    print(f"\n    [Validation Set - for reference]")
    print(f"    Valid IC:   {valid_ic:.4f}")
    print(f"    Valid ICIR: {valid_icir:.4f}")

    test_pred = pd.Series(
        test_pred_values, index=test_data.index, name='score')
    print(f"\n    Test prediction shape: {test_pred.shape}")
    print(f"    Test prediction range: "
          f"[{test_pred.min():.4f}, {test_pred.max():.4f}]")

    return model, feature_names, test_pred, dataset
