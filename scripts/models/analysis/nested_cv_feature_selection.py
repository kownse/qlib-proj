"""
嵌套交叉验证特征选择 - 避免测试集泄漏

设计原理:
- 外层: 2025年作为最终测试集，完全独立，不参与任何特征选择决策
- 内层: 在2000-2024年数据上做4折CV特征选择
  - Fold 1: train 2000-2020, valid 2021
  - Fold 2: train 2000-2021, valid 2022
  - Fold 3: train 2000-2022, valid 2023
  - Fold 4: train 2000-2023, valid 2024
- 特征选择标准: 内层4折的平均验证IC

这样确保forward selection过程完全不接触2025年测试数据。

使用方法:
    python scripts/models/analysis/nested_cv_feature_selection.py
    python scripts/models/analysis/nested_cv_feature_selection.py --stock-pool sp500 --max-features 20
"""

import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 关闭 Qlib 的 INFO 日志
logging.getLogger('qlib').setLevel(logging.WARNING)

import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))
project_root = script_dir.parent

import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

# Qlib 初始化
qlib_data_path = project_root / "my_data" / "qlib_us"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks

from utils.talib_ops import TALIB_OPS
from data.stock_pools import STOCK_POOLS

# ============================================================================
# 内层CV Folds (用于特征选择，不包含2025)
# ============================================================================

INNER_CV_FOLDS = [
    {
        'name': 'Inner Fold 1 (valid 2021)',
        'train_start': '2000-01-01',
        'train_end': '2020-12-31',
        'valid_start': '2021-01-01',
        'valid_end': '2021-12-31',
    },
    {
        'name': 'Inner Fold 2 (valid 2022)',
        'train_start': '2000-01-01',
        'train_end': '2021-12-31',
        'valid_start': '2022-01-01',
        'valid_end': '2022-12-31',
    },
    {
        'name': 'Inner Fold 3 (valid 2023)',
        'train_start': '2000-01-01',
        'train_end': '2022-12-31',
        'valid_start': '2023-01-01',
        'valid_end': '2023-12-31',
    },
    {
        'name': 'Inner Fold 4 (valid 2024)',
        'train_start': '2000-01-01',
        'train_end': '2023-12-31',
        'valid_start': '2024-01-01',
        'valid_end': '2024-12-31',
    },
]

# 外层测试集 (完全独立)
OUTER_TEST = {
    'train_start': '2000-01-01',
    'train_end': '2024-09-30',
    'valid_start': '2024-10-01',
    'valid_end': '2024-12-31',
    'test_start': '2025-01-01',
    'test_end': '2025-12-31',
}

# ============================================================================
# 候选特征池 (来自V7和其他有潜力的特征)
# ============================================================================

# 基础特征 (初始集)
BASE_STOCK_FEATURES = {
    "MOMENTUM_QUALITY": "($close/Ref($close, 20) - 1) / (Std($close/Ref($close,1)-1, 20) + 1e-12)",
    "PCT_FROM_52W_HIGH": "($close - Max($high, 252)) / (Max($high, 252) + 1e-12)",
    "MAX60": "Max($high, 60)/$close",
}

# 候选股票特征 (用于forward selection)
CANDIDATE_STOCK_FEATURES = {
    # 波动率特征
    "TALIB_NATR14": "TALIB_NATR($high, $low, $close, 14)",
    "PARKINSON_VOL_20": "Power(Log($high/$low), 2) / (4*0.693)",
    "GARMAN_KLASS_VOL": "0.5*Power(Log($high/$low), 2) - 0.386*Power(Log($close/$open), 2)",

    # 动量特征
    "ROC5": "Ref($close, 5)/$close",
    "ROC20": "Ref($close, 20)/$close",
    "ROC60": "Ref($close, 60)/$close",
    "TALIB_RSI14": "TALIB_RSI($close, 14)",
    "TALIB_WILLR14": "TALIB_WILLR($high, $low, $close, 14)",

    # 均值回归特征
    "RESI5": "Resi($close, 5)/$close",
    "RESI10": "Resi($close, 10)/$close",
    "RESI20": "Resi($close, 20)/$close",
    "MA_RATIO_5_20": "Mean($close, 5)/Mean($close, 20)",
    "MA_RATIO_20_60": "Mean($close, 20)/Mean($close, 60)",

    # 极值特征
    "MIN60": "Min($low, 60)/$close",
    "MAX_DRAWDOWN_60": "(Min($close, 60) - Max($high, 60)) / (Max($high, 60) + 1e-12)",
    "PCT_FROM_52W_LOW": "($close - Min($low, 252)) / (Min($low, 252) + 1e-12)",

    # 成交量特征
    "VOLUME_RATIO_5_20": "Mean($volume, 5)/(Mean($volume, 20)+1e-12)",
    "VOLUME_STD_20": "Std($volume, 20)/(Mean($volume, 20)+1e-12)",

    # 趋势特征
    "TALIB_ADX14": "TALIB_ADX($high, $low, $close, 14)",
    "SLOPE20": "Slope($close, 20)/$close",

    # 价格位置
    "CLOSE_POSITION_20": "($close - Min($low, 20))/(Max($high, 20) - Min($low, 20) + 1e-12)",
    "CLOSE_POSITION_60": "($close - Min($low, 60))/(Max($high, 60) - Min($low, 60) + 1e-12)",
}

# 候选宏观特征 (使用实际的列名)
CANDIDATE_MACRO_FEATURES = [
    # VIX 相关
    "macro_vix_zscore20",
    "macro_vix_regime",
    "macro_vix_pct_5d",
    # 信用/风险
    "macro_hy_spread_zscore",
    "macro_credit_stress",
    "macro_hyg_pct_5d",
    # 利率/债券
    "macro_yield_curve",
    "macro_tlt_pct_5d",
    "macro_tlt_pct_20d",
    # 商品
    "macro_gld_pct_5d",
    "macro_gld_pct_20d",
    "macro_uso_pct_5d",
    "macro_uso_pct_20d",
    # 美元
    "macro_uup_pct_5d",
    # 市场
    "macro_spy_pct_5d",
    "macro_spy_vol20",
    "macro_risk_on_off",
]

# AE-MLP 最佳超参数 (从之前的CV搜索)
BEST_HYPERPARAMS = {
    "hidden_units": [112, 64, 128, 224, 48],
    "dropout_rates": [0.05, 0.096, 0.096, 0.096, 0.096, 0.096, 0.096],
    "lr": 0.000534,
    "batch_size": 2048,
    "loss_weights": {"decoder": 0.267, "ae_action": 0.072, "action": 1.0}
}


def validate_all_features(symbols, verbose: bool = True) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    """
    在流程开始前验证所有候选特征是否被Qlib支持。

    Returns:
        valid_base: 有效的基础特征
        valid_candidates: 有效的候选股票特征
        valid_macro: 有效的宏观特征名称列表
        (如果有无效特征会打印警告并从返回结果中排除)
    """
    from qlib.data import D

    print("\n" + "=" * 70)
    print("VALIDATING ALL CANDIDATE FEATURES")
    print("=" * 70)

    # 获取一小段测试数据的时间范围
    test_start = "2024-01-01"
    test_end = "2024-01-10"
    test_symbols = symbols[:5] if len(symbols) > 5 else symbols

    valid_base = {}
    valid_candidates = {}
    invalid_features = []

    # 验证基础特征
    print("\n[1/3] Validating base features...")
    for name, expr in BASE_STOCK_FEATURES.items():
        try:
            df = D.features(test_symbols, [expr], start_time=test_start, end_time=test_end)
            if df is not None and len(df) > 0:
                valid_base[name] = expr
                if verbose:
                    print(f"  ✓ {name}")
            else:
                invalid_features.append((name, "empty result"))
                print(f"  ✗ {name}: empty result")
        except Exception as e:
            invalid_features.append((name, str(e)))
            print(f"  ✗ {name}: {e}")

    # 验证候选股票特征
    print(f"\n[2/3] Validating candidate stock features ({len(CANDIDATE_STOCK_FEATURES)})...")
    for name, expr in CANDIDATE_STOCK_FEATURES.items():
        try:
            df = D.features(test_symbols, [expr], start_time=test_start, end_time=test_end)
            if df is not None and len(df) > 0:
                valid_candidates[name] = expr
                if verbose:
                    print(f"  ✓ {name}")
            else:
                invalid_features.append((name, "empty result"))
                print(f"  ✗ {name}: empty result")
        except Exception as e:
            invalid_features.append((name, str(e)))
            print(f"  ✗ {name}: {e}")

    # 验证宏观特征 (检查parquet文件中是否存在)
    print(f"\n[3/3] Validating macro features ({len(CANDIDATE_MACRO_FEATURES)})...")
    valid_macro = []
    macro_path = project_root / "my_data" / "macro_processed" / "macro_features.parquet"

    if macro_path.exists():
        try:
            macro_df = pd.read_parquet(macro_path)
            available_cols = set(macro_df.columns)

            for name in CANDIDATE_MACRO_FEATURES:
                if name in available_cols:
                    valid_macro.append(name)
                    if verbose:
                        print(f"  ✓ {name}")
                else:
                    invalid_features.append((name, "not in macro data"))
                    print(f"  ✗ {name}: not found in macro data")
        except Exception as e:
            print(f"  ! Error loading macro data: {e}")
            print("    Macro features will be skipped")
    else:
        print(f"  ! Macro data file not found: {macro_path}")
        print("    Macro features will be skipped")

    # 汇总
    print("\n" + "-" * 70)
    print("VALIDATION SUMMARY")
    print("-" * 70)
    print(f"Base features:      {len(valid_base)}/{len(BASE_STOCK_FEATURES)} valid")
    print(f"Candidate features: {len(valid_candidates)}/{len(CANDIDATE_STOCK_FEATURES)} valid")
    print(f"Macro features:     {len(valid_macro)}/{len(CANDIDATE_MACRO_FEATURES)} valid")

    if invalid_features:
        print(f"\n⚠ {len(invalid_features)} invalid features will be skipped:")
        for name, reason in invalid_features:
            print(f"  - {name}: {reason}")

    if len(valid_base) == 0:
        raise ValueError("No valid base features! Cannot proceed.")

    print("=" * 70)

    return valid_base, valid_candidates, valid_macro


def build_ae_mlp_model(num_columns: int, params: dict = None) -> Model:
    """构建 AE-MLP 模型"""
    if params is None:
        params = BEST_HYPERPARAMS

    hidden_units = params['hidden_units']
    dropout_rates = params['dropout_rates']
    lr = params['lr']
    loss_weights = params['loss_weights']

    inp = layers.Input(shape=(num_columns,), name='input')

    x0 = layers.BatchNormalization(name='input_bn')(inp)

    # Encoder
    encoder = layers.GaussianNoise(dropout_rates[0], name='noise')(x0)
    encoder = layers.Dense(hidden_units[0], name='encoder_dense')(encoder)
    encoder = layers.BatchNormalization(name='encoder_bn')(encoder)
    encoder = layers.Activation('swish', name='encoder_act')(encoder)

    # Decoder
    decoder = layers.Dropout(dropout_rates[1], name='decoder_dropout')(encoder)
    decoder = layers.Dense(num_columns, dtype='float32', name='decoder')(decoder)

    # Auxiliary branch
    x_ae = layers.Dense(hidden_units[1], name='ae_dense1')(decoder)
    x_ae = layers.BatchNormalization(name='ae_bn1')(x_ae)
    x_ae = layers.Activation('swish', name='ae_act1')(x_ae)
    x_ae = layers.Dropout(dropout_rates[2], name='ae_dropout1')(x_ae)
    out_ae = layers.Dense(1, dtype='float32', name='ae_action')(x_ae)

    # Main branch
    x = layers.Concatenate(name='concat')([x0, encoder])
    x = layers.BatchNormalization(name='main_bn0')(x)
    x = layers.Dropout(dropout_rates[3], name='main_dropout0')(x)

    for i in range(2, len(hidden_units)):
        dropout_idx = min(i + 2, len(dropout_rates) - 1)
        x = layers.Dense(hidden_units[i], name=f'main_dense{i-1}')(x)
        x = layers.BatchNormalization(name=f'main_bn{i-1}')(x)
        x = layers.Activation('swish', name=f'main_act{i-1}')(x)
        x = layers.Dropout(dropout_rates[dropout_idx], name=f'main_dropout{i-1}')(x)

    out = layers.Dense(1, dtype='float32', name='action')(x)

    model = Model(inputs=inp, outputs=[decoder, out_ae, out], name='AE_MLP')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss={'decoder': 'mse', 'ae_action': 'mse', 'action': 'mse'},
        loss_weights=loss_weights,
    )

    return model


def compute_ic(pred: np.ndarray, label: np.ndarray, index: pd.MultiIndex) -> float:
    """计算IC"""
    df = pd.DataFrame({'pred': pred, 'label': label}, index=index)
    ic_by_date = df.groupby(level='datetime').apply(
        lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
    )
    ic_by_date = ic_by_date.dropna()
    return ic_by_date.mean() if len(ic_by_date) > 0 else 0.0


class DynamicFeatureHandler(DataHandlerLP):
    """动态特征Handler，支持运行时指定特征"""

    def __init__(
        self,
        feature_config: Dict[str, str],
        macro_features: List[str] = None,
        volatility_window: int = 5,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq: str = "day",
        infer_processors=[],
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        macro_lag: int = 1,
        **kwargs,
    ):
        self.feature_config = feature_config
        self.macro_features = macro_features or []
        self.volatility_window = volatility_window
        self.macro_lag = macro_lag

        # 加载宏观数据
        macro_path = project_root / "my_data" / "macro_processed" / "macro_features.parquet"
        self._macro_df = None
        if macro_path.exists() and self.macro_features:
            try:
                self._macro_df = pd.read_parquet(macro_path)
            except Exception as e:
                print(f"Warning: Failed to load macro data: {e}")

        from qlib.contrib.data.handler import check_transform_proc, _DEFAULT_LEARN_PROCESSORS

        if learn_processors is None:
            learn_processors = _DEFAULT_LEARN_PROCESSORS

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self._get_feature_config(),
                    "label": kwargs.pop("label", self._get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def _get_feature_config(self):
        fields = list(self.feature_config.values())
        names = list(self.feature_config.keys())
        return fields, names

    def _get_label_config(self):
        label_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [label_expr], ["LABEL0"]

    def process_data(self, with_fit: bool = False):
        super().process_data(with_fit=with_fit)

        if self._macro_df is not None and self.macro_features:
            self._add_macro_features()

    def _add_macro_features(self):
        """添加宏观特征"""
        available_cols = [c for c in self.macro_features if c in self._macro_df.columns]
        if not available_cols:
            return

        for attr in ['_learn', '_infer']:
            df = getattr(self, attr, None)
            if df is None:
                continue

            datetime_col = df.index.names[0]
            main_datetimes = df.index.get_level_values(datetime_col)
            has_multi_columns = isinstance(df.columns, pd.MultiIndex)

            macro_data = {}
            for col in available_cols:
                macro_series = self._macro_df[col].shift(self.macro_lag)
                aligned_values = macro_series.reindex(main_datetimes).values

                new_name = f"{col}_lag{self.macro_lag}"
                if has_multi_columns:
                    macro_data[('feature', new_name)] = aligned_values
                else:
                    macro_data[new_name] = aligned_values

            macro_df = pd.DataFrame(macro_data, index=df.index)
            merged = pd.concat([df, macro_df], axis=1, copy=False)
            setattr(self, attr, merged.copy())


def prepare_fold_data(symbols, fold_config, feature_config, macro_features, nday=5):
    """为单个fold准备数据"""
    handler = DynamicFeatureHandler(
        feature_config=feature_config,
        macro_features=macro_features,
        volatility_window=nday,
        instruments=symbols,
        start_time=fold_config['train_start'],
        end_time=fold_config['valid_end'],
        fit_start_time=fold_config['train_start'],
        fit_end_time=fold_config['train_end'],
        infer_processors=[],
    )

    segments = {
        "train": (fold_config['train_start'], fold_config['train_end']),
        "valid": (fold_config['valid_start'], fold_config['valid_end']),
    }

    dataset = DatasetH(handler=handler, segments=segments)

    # 准备数据
    X_train = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
    X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0).clip(-10, 10)

    y_train = dataset.prepare("train", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    y_train = y_train.fillna(0).values

    X_valid = dataset.prepare("valid", col_set="feature", data_key=DataHandlerLP.DK_L)
    X_valid = X_valid.fillna(0).replace([np.inf, -np.inf], 0).clip(-10, 10)
    valid_index = X_valid.index

    y_valid = dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(y_valid, pd.DataFrame):
        y_valid = y_valid.iloc[:, 0]
    y_valid = y_valid.fillna(0).values

    return X_train.values, y_train, X_valid.values, y_valid, valid_index


def evaluate_feature_set_inner_cv(
    symbols,
    feature_config: Dict[str, str],
    macro_features: List[str],
    nday: int = 5,
    epochs: int = 15,
    early_stop: int = 5,
    batch_size: int = 2048,
) -> Tuple[float, List[float]]:
    """
    在内层CV上评估特征集。

    返回: (平均IC, 各fold IC列表)
    """
    fold_ics = []

    for fold in INNER_CV_FOLDS:
        tf.keras.backend.clear_session()

        # 准备数据
        X_train, y_train, X_valid, y_valid, valid_index = prepare_fold_data(
            symbols, fold, feature_config, macro_features, nday
        )

        num_features = X_train.shape[1]

        # 构建模型
        model = build_ae_mlp_model(num_features)

        # 训练
        train_outputs = {'decoder': X_train, 'ae_action': y_train, 'action': y_train}
        valid_outputs = {'decoder': X_valid, 'ae_action': y_valid, 'action': y_valid}

        cb_list = [
            callbacks.EarlyStopping(
                monitor='val_action_loss',
                patience=early_stop,
                restore_best_weights=True,
                verbose=0,
                mode='min'
            ),
        ]

        model.fit(
            X_train, train_outputs,
            validation_data=(X_valid, valid_outputs),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb_list,
            verbose=0,
        )

        # 预测并计算IC
        _, _, valid_pred = model.predict(X_valid, batch_size=batch_size, verbose=0)
        ic = compute_ic(valid_pred.flatten(), y_valid, valid_index)
        fold_ics.append(ic)

    mean_ic = np.mean(fold_ics)
    return mean_ic, fold_ics


def run_nested_cv_forward_selection(
    symbols,
    valid_base_features: Dict[str, str],
    valid_candidate_features: Dict[str, str],
    valid_macro_features: List[str],
    nday: int = 5,
    max_features: int = 25,
    min_improvement: float = 0.001,
    epochs: int = 15,
    early_stop: int = 5,
    batch_size: int = 2048,
    output_dir: Path = None,
):
    """
    嵌套CV Forward Selection。

    使用内层4折CV来选择特征，外层测试集(2025)完全不参与。

    Args:
        valid_base_features: 已验证的基础特征
        valid_candidate_features: 已验证的候选股票特征
        valid_macro_features: 已验证的宏观特征列表
    """
    print("\n" + "=" * 70)
    print("NESTED CV FORWARD FEATURE SELECTION")
    print("=" * 70)
    print(f"Inner CV Folds: {len(INNER_CV_FOLDS)}")
    for fold in INNER_CV_FOLDS:
        print(f"  - {fold['name']}")
    print(f"Outer Test: 2025 (completely held out)")
    print(f"Max features: {max_features}")
    print(f"Min improvement: {min_improvement}")
    print(f"Valid base features: {len(valid_base_features)}")
    print(f"Valid candidate features: {len(valid_candidate_features)}")
    print(f"Valid macro features: {len(valid_macro_features)}")
    print("=" * 70)

    # 初始化 (使用已验证的特征)
    selected_stock_features = dict(valid_base_features)
    selected_macro_features = []

    remaining_stock = dict(valid_candidate_features)
    remaining_macro = list(valid_macro_features)

    history = []

    # 基线评估
    print("\n[*] Evaluating baseline features...")
    print(f"    Base features: {list(selected_stock_features.keys())}")

    baseline_ic, baseline_fold_ics = evaluate_feature_set_inner_cv(
        symbols, selected_stock_features, selected_macro_features,
        nday, epochs, early_stop, batch_size
    )

    print(f"    Baseline Inner CV IC: {baseline_ic:.4f}")
    print(f"    Fold ICs: {[f'{ic:.4f}' for ic in baseline_fold_ics]}")

    history.append({
        'round': 0,
        'feature': 'BASELINE',
        'type': 'base',
        'inner_cv_ic': baseline_ic,
        'fold_ics': baseline_fold_ics,
        'improvement': 0,
        'total_features': len(selected_stock_features) + len(selected_macro_features),
    })

    current_best_ic = baseline_ic
    round_num = 0

    # Forward selection
    while len(selected_stock_features) + len(selected_macro_features) < max_features:
        round_num += 1
        print(f"\n[Round {round_num}] Current best IC: {current_best_ic:.4f}")
        print(f"    Testing {len(remaining_stock)} stock + {len(remaining_macro)} macro candidates...")

        best_candidate = None
        best_candidate_type = None
        best_candidate_ic = current_best_ic
        best_fold_ics = None

        # 测试每个候选股票特征
        for name, expr in remaining_stock.items():
            test_features = dict(selected_stock_features)
            test_features[name] = expr

            try:
                ic, fold_ics = evaluate_feature_set_inner_cv(
                    symbols, test_features, selected_macro_features,
                    nday, epochs, early_stop, batch_size
                )

                if ic > best_candidate_ic:
                    best_candidate = name
                    best_candidate_type = 'stock'
                    best_candidate_ic = ic
                    best_fold_ics = fold_ics
                    print(f"      + {name}: IC={ic:.4f} (new best)")
                else:
                    print(f"        {name}: IC={ic:.4f}")

            except Exception as e:
                print(f"      ! {name}: ERROR - {e}")

        # 测试每个候选宏观特征
        for name in remaining_macro:
            test_macro = selected_macro_features + [name]

            try:
                ic, fold_ics = evaluate_feature_set_inner_cv(
                    symbols, selected_stock_features, test_macro,
                    nday, epochs, early_stop, batch_size
                )

                if ic > best_candidate_ic:
                    best_candidate = name
                    best_candidate_type = 'macro'
                    best_candidate_ic = ic
                    best_fold_ics = fold_ics
                    print(f"      + {name}: IC={ic:.4f} (new best)")
                else:
                    print(f"        {name}: IC={ic:.4f}")

            except Exception as e:
                print(f"      ! {name}: ERROR - {e}")

        # 检查是否有改进
        improvement = best_candidate_ic - current_best_ic

        if best_candidate is None or improvement < min_improvement:
            print(f"\n[!] Stopping: No candidate improved IC by >= {min_improvement}")
            print(f"    Best improvement found: {improvement:.4f}")
            break

        # 添加最佳候选
        if best_candidate_type == 'stock':
            selected_stock_features[best_candidate] = remaining_stock.pop(best_candidate)
        else:
            selected_macro_features.append(best_candidate)
            remaining_macro.remove(best_candidate)

        current_best_ic = best_candidate_ic

        history.append({
            'round': round_num,
            'feature': best_candidate,
            'type': best_candidate_type,
            'inner_cv_ic': best_candidate_ic,
            'fold_ics': best_fold_ics,
            'improvement': improvement,
            'total_features': len(selected_stock_features) + len(selected_macro_features),
        })

        print(f"\n    ✓ Added {best_candidate} ({best_candidate_type})")
        print(f"      Inner CV IC: {best_candidate_ic:.4f} (+{improvement:.4f})")
        print(f"      Fold ICs: {[f'{ic:.4f}' for ic in best_fold_ics]}")
        print(f"      Total features: {len(selected_stock_features)} stock + {len(selected_macro_features)} macro")

    # 最终结果
    print("\n" + "=" * 70)
    print("NESTED CV FORWARD SELECTION COMPLETE")
    print("=" * 70)
    print(f"Final Inner CV IC: {current_best_ic:.4f}")
    print(f"\nSelected Stock Features ({len(selected_stock_features)}):")
    for name in selected_stock_features:
        print(f"  - {name}")
    print(f"\nSelected Macro Features ({len(selected_macro_features)}):")
    for name in selected_macro_features:
        print(f"  - {name}")

    # 保存结果
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        result = {
            'timestamp': timestamp,
            'method': 'nested_cv_forward_selection',
            'inner_cv_folds': len(INNER_CV_FOLDS),
            'outer_test': '2025 (held out)',
            'max_features': max_features,
            'min_improvement': min_improvement,
            'epochs': epochs,
            'selected_stock_features': selected_stock_features,
            'selected_macro_features': selected_macro_features,
            'history': history,
            'final_inner_cv_ic': current_best_ic,
        }

        result_file = output_dir / f"nested_cv_selection_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {result_file}")

    return selected_stock_features, selected_macro_features, history


def evaluate_on_outer_test(
    symbols,
    feature_config: Dict[str, str],
    macro_features: List[str],
    nday: int = 5,
    epochs: int = 50,
    early_stop: int = 10,
    batch_size: int = 2048,
):
    """
    在外层测试集(2025)上评估最终特征集。

    这个函数只在特征选择完成后调用一次。
    """
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON OUTER TEST SET (2025)")
    print("=" * 70)

    tf.keras.backend.clear_session()

    # 准备完整数据
    handler = DynamicFeatureHandler(
        feature_config=feature_config,
        macro_features=macro_features,
        volatility_window=nday,
        instruments=symbols,
        start_time=OUTER_TEST['train_start'],
        end_time=OUTER_TEST['test_end'],
        fit_start_time=OUTER_TEST['train_start'],
        fit_end_time=OUTER_TEST['train_end'],
        infer_processors=[],
    )

    segments = {
        "train": (OUTER_TEST['train_start'], OUTER_TEST['train_end']),
        "valid": (OUTER_TEST['valid_start'], OUTER_TEST['valid_end']),
        "test": (OUTER_TEST['test_start'], OUTER_TEST['test_end']),
    }

    dataset = DatasetH(handler=handler, segments=segments)

    # 准备数据
    X_train = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
    X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0).clip(-10, 10)
    y_train = dataset.prepare("train", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    y_train = y_train.fillna(0).values

    X_valid = dataset.prepare("valid", col_set="feature", data_key=DataHandlerLP.DK_L)
    X_valid = X_valid.fillna(0).replace([np.inf, -np.inf], 0).clip(-10, 10)
    valid_index = X_valid.index
    y_valid = dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(y_valid, pd.DataFrame):
        y_valid = y_valid.iloc[:, 0]
    y_valid = y_valid.fillna(0).values

    X_test = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_L)
    X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0).clip(-10, 10)
    test_index = X_test.index
    y_test = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]
    y_test = y_test.fillna(0).values

    print(f"Train: {X_train.shape}")
    print(f"Valid: {X_valid.shape}")
    print(f"Test:  {X_test.shape}")

    # 构建模型
    model = build_ae_mlp_model(X_train.shape[1])

    # 训练
    train_outputs = {'decoder': X_train.values, 'ae_action': y_train, 'action': y_train}
    valid_outputs = {'decoder': X_valid.values, 'ae_action': y_valid, 'action': y_valid}

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
            verbose=1,
            mode='min'
        ),
    ]

    print("\nTraining final model...")
    model.fit(
        X_train.values, train_outputs,
        validation_data=(X_valid.values, valid_outputs),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1,
    )

    # 评估
    _, _, valid_pred = model.predict(X_valid.values, batch_size=batch_size, verbose=0)
    valid_ic = compute_ic(valid_pred.flatten(), y_valid, valid_index)

    _, _, test_pred = model.predict(X_test.values, batch_size=batch_size, verbose=0)
    test_ic = compute_ic(test_pred.flatten(), y_test, test_index)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Valid IC (2024 Q4): {valid_ic:.4f}")
    print(f"Test IC (2025):     {test_ic:.4f}")
    print("=" * 70)

    return valid_ic, test_ic


def main():
    parser = argparse.ArgumentParser(description='Nested CV Forward Feature Selection')
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--max-features', type=int, default=25)
    parser.add_argument('--min-improvement', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=15,
                        help='Epochs per evaluation during selection')
    parser.add_argument('--early-stop', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--final-eval', action='store_true',
                        help='Run final evaluation on outer test set')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip feature validation (use with caution)')
    parser.add_argument('--no-countdown', action='store_true',
                        help='Skip the 3-second countdown before starting')

    args = parser.parse_args()

    # GPU 设置
    gpus = tf.config.list_physical_devices('GPU')
    if args.gpu >= 0 and gpus:
        tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)

    # Qlib 初始化
    qlib.init(
        provider_uri=str(qlib_data_path),
        region=REG_US,
        custom_ops=TALIB_OPS,
    )

    symbols = STOCK_POOLS[args.stock_pool]
    output_dir = project_root / "outputs" / "feature_selection"

    print(f"Stock pool: {args.stock_pool} ({len(symbols)} stocks)")

    # 第一步: 验证所有候选特征
    if args.skip_validation:
        print("\n[!] Skipping feature validation (--skip-validation)")
        valid_base = dict(BASE_STOCK_FEATURES)
        valid_candidates = dict(CANDIDATE_STOCK_FEATURES)
        valid_macro = list(CANDIDATE_MACRO_FEATURES)
    else:
        valid_base, valid_candidates, valid_macro = validate_all_features(symbols, verbose=True)

    # 询问用户是否继续
    if not args.no_countdown:
        print("\nProceed with feature selection? (Press Ctrl+C to abort)")
        try:
            import time
            for i in range(3, 0, -1):
                print(f"  Starting in {i}...")
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nAborted by user.")
            return

    # 运行嵌套CV特征选择
    selected_stock, selected_macro, history = run_nested_cv_forward_selection(
        symbols,
        valid_base_features=valid_base,
        valid_candidate_features=valid_candidates,
        valid_macro_features=valid_macro,
        nday=args.nday,
        max_features=args.max_features,
        min_improvement=args.min_improvement,
        epochs=args.epochs,
        early_stop=args.early_stop,
        batch_size=args.batch_size,
        output_dir=output_dir,
    )

    # 可选: 在外层测试集上评估
    if args.final_eval:
        evaluate_on_outer_test(
            symbols,
            selected_stock,
            selected_macro,
            nday=args.nday,
            epochs=50,
            early_stop=10,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
