"""
嵌套交叉验证 Forward Selection - 基于 Backward Elimination 结果扩展特征

设计原理:
- 从 backward elimination 的 protected_features 作为基线
- 逐个测试候选特征，如果能提高IC则加入
- 维护排除列表：如果加入某特征导致IC下降，则排除该特征
- 后续轮次跳过被排除的特征

内层CV (只用2000-2024年数据):
  Fold 1: train 2000-2020, valid 2021
  Fold 2: train 2000-2021, valid 2022
  Fold 3: train 2000-2022, valid 2023
  Fold 4: train 2000-2023, valid 2024

使用方法:
    python scripts/models/analysis/nested_cv_feature_selection.py --baseline backward_elimination_20260118_224446.json
    python scripts/models/analysis/nested_cv_feature_selection.py --resume
"""

import os
import logging
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

# ============================================================================
# 候选特征池 (不在 baseline 中的特征)
# ============================================================================

# 所有可能的股票特征
ALL_STOCK_FEATURES = {
    # 动量特征
    "MOMENTUM_QUALITY": "($close/Ref($close, 20) - 1) / (Std($close/Ref($close,1)-1, 20) + 1e-12)",
    "ROC5": "Ref($close, 5)/$close",
    "ROC20": "Ref($close, 20)/$close",
    "ROC60": "Ref($close, 60)/$close",
    "TALIB_RSI14": "TALIB_RSI($close, 14)",
    "TALIB_WILLR14": "TALIB_WILLR($high, $low, $close, 14)",

    # 波动率特征
    "TALIB_NATR14": "TALIB_NATR($high, $low, $close, 14)",
    "STD20": "Std($close, 20)/$close",
    "STD60": "Std($close, 60)/$close",
    "TALIB_ATR14": "TALIB_ATR($high, $low, $close, 14)/$close",

    # 价格位置特征
    "PCT_FROM_52W_HIGH": "($close - Max($high, 252)) / (Max($high, 252) + 1e-12)",
    "PCT_FROM_52W_LOW": "($close - Min($low, 252)) / (Min($low, 252) + 1e-12)",
    "MAX60": "Max($high, 60)/$close",
    "MIN60": "Min($low, 60)/$close",
    "CLOSE_POSITION_60": "($close - Min($low, 60))/(Max($high, 60) - Min($low, 60) + 1e-12)",
    "CLOSE_POSITION_20": "($close - Min($low, 20))/(Max($high, 20) - Min($low, 20) + 1e-12)",

    # 均值回归
    "RESI5": "Resi($close, 5)/$close",
    "RESI10": "Resi($close, 10)/$close",
    "RESI20": "Resi($close, 20)/$close",
    "MA_RATIO_5_20": "Mean($close, 5)/Mean($close, 20)",
    "MA_RATIO_20_60": "Mean($close, 20)/Mean($close, 60)",

    # 成交量
    "VOLUME_RATIO_5_20": "Mean($volume, 5)/(Mean($volume, 20)+1e-12)",
    "VOLUME_STD_20": "Std($volume, 20)/(Mean($volume, 20)+1e-12)",
    "VWAP_BIAS": "($close*$volume)/Sum($volume, 20) - Mean($close, 20)",

    # 趋势
    "TALIB_ADX14": "TALIB_ADX($high, $low, $close, 14)",
    "SLOPE20": "Slope($close, 20)/$close",
    "SLOPE60": "Slope($close, 60)/$close",

    # Drawdown
    "MAX_DRAWDOWN_20": "(Min($close, 20) - Max($high, 20)) / (Max($high, 20) + 1e-12)",
    "MAX_DRAWDOWN_60": "(Min($close, 60) - Max($high, 60)) / (Max($high, 60) + 1e-12)",

    # 技术指标
    "TALIB_CCI14": "TALIB_CCI($high, $low, $close, 14)",
    "TALIB_MFI14": "TALIB_MFI($high, $low, $close, $volume, 14)",

    # 额外候选特征
    "TALIB_MACD_HIST": "TALIB_MACD_HIST($close, 12, 26, 9)",
    "TALIB_BBANDS_UPPER": "TALIB_BBANDS_UPPER($close, 20, 2, 2)/$close",
    "TALIB_BBANDS_LOWER": "TALIB_BBANDS_LOWER($close, 20, 2, 2)/$close",
    "HIGH_LOW_RATIO": "$high/$low",
    "CLOSE_OPEN_RATIO": "$close/$open",
}

# 所有可能的宏观特征
ALL_MACRO_FEATURES = [
    # VIX 相关
    "macro_vix_zscore20",
    "macro_vix_regime",
    "macro_vix_pct_5d",
    "macro_vix_term_structure",
    # 信用/风险
    "macro_hy_spread_zscore",
    "macro_credit_stress",
    "macro_hyg_pct_5d",
    "macro_hyg_pct_20d",
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
    "macro_uup_pct_20d",
    # 市场
    "macro_spy_pct_5d",
    "macro_spy_pct_20d",
    "macro_spy_vol20",
    "macro_risk_on_off",
]

# AE-MLP 最佳超参数
BEST_HYPERPARAMS = {
    "hidden_units": [112, 64, 128, 224, 48],
    "dropout_rates": [0.05, 0.096, 0.096, 0.096, 0.096, 0.096, 0.096],
    "lr": 0.000534,
    "batch_size": 2048,
    "loss_weights": {"decoder": 0.267, "ae_action": 0.072, "action": 1.0}
}


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
    """动态特征Handler"""

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
    """在内层CV上评估特征集"""
    fold_ics = []

    for fold in INNER_CV_FOLDS:
        tf.keras.backend.clear_session()
        gc.collect()

        X_train, y_train, X_valid, y_valid, valid_index = prepare_fold_data(
            symbols, fold, feature_config, macro_features, nday
        )

        num_features = X_train.shape[1]
        model = build_ae_mlp_model(num_features)

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

        _, _, valid_pred = model.predict(X_valid, batch_size=batch_size, verbose=0)
        ic = compute_ic(valid_pred.flatten(), y_valid, valid_index)
        fold_ics.append(ic)

        del X_train, y_train, X_valid, y_valid, valid_index, model
        del train_outputs, valid_outputs
        gc.collect()

    mean_ic = np.mean(fold_ics)
    return mean_ic, fold_ics


def load_baseline_from_file(filepath: Path) -> Tuple[Dict[str, str], List[str], set]:
    """
    从 backward elimination 结果文件加载 baseline 特征集

    Returns:
        baseline_stock: 基线股票特征 dict
        baseline_macro: 基线宏观特征 list
        protected_features: 受保护特征 set (用作 baseline)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    # 优先使用 protected_features 作为 baseline
    protected = set(data.get('protected_features', []))

    # 获取完整的特征表达式
    all_stock = data.get('final_stock_features', {})
    all_macro = data.get('final_macro_features', [])

    # 筛选出受保护的特征
    baseline_stock = {}
    for name, expr in all_stock.items():
        if name in protected:
            baseline_stock[name] = expr

    baseline_macro = [m for m in all_macro if m in protected]

    return baseline_stock, baseline_macro, protected


def validate_candidate_features(symbols, candidates: Dict[str, str]) -> Dict[str, str]:
    """验证候选特征是否有效"""
    from qlib.data import D

    test_start = "2024-01-01"
    test_end = "2024-01-10"
    test_symbols = symbols[:5] if len(symbols) > 5 else symbols

    valid = {}
    for name, expr in candidates.items():
        try:
            df = D.features(test_symbols, [expr], start_time=test_start, end_time=test_end)
            if df is not None and len(df) > 0:
                valid[name] = expr
        except:
            pass

    return valid


def validate_macro_features(candidates: List[str]) -> List[str]:
    """验证宏观特征是否存在"""
    macro_path = project_root / "my_data" / "macro_processed" / "macro_features.parquet"
    if not macro_path.exists():
        return []

    try:
        macro_df = pd.read_parquet(macro_path)
        available = set(macro_df.columns)
        return [m for m in candidates if m in available]
    except:
        return []


def _save_checkpoint(output_dir: Path, round_num: int, current_ic: float,
                     current_stock: Dict, current_macro: List,
                     excluded_features: set, history: List):
    """保存 checkpoint"""
    checkpoint = {
        'round': round_num,
        'current_ic': current_ic,
        'current_stock_features': current_stock,
        'current_macro_features': current_macro,
        'excluded_features': list(excluded_features),
        'history': history,
    }
    checkpoint_file = output_dir / "forward_selection_checkpoint.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def run_forward_selection(
    symbols,
    baseline_stock: Dict[str, str],
    baseline_macro: List[str],
    candidate_stock: Dict[str, str],
    candidate_macro: List[str],
    nday: int = 5,
    max_features: int = 50,
    min_improvement: float = 0.0005,
    epochs: int = 15,
    early_stop: int = 5,
    batch_size: int = 2048,
    output_dir: Path = None,
    excluded_features: set = None,
):
    """
    Forward Selection - 从 baseline 开始逐步添加特征

    维护排除列表：如果加入某特征导致 IC 下降，则排除该特征
    """
    print("\n" + "=" * 70)
    print("NESTED CV FORWARD SELECTION")
    print("=" * 70)
    print(f"Baseline features: {len(baseline_stock)} stock + {len(baseline_macro)} macro")
    print(f"Candidate features: {len(candidate_stock)} stock + {len(candidate_macro)} macro")
    print(f"Max features: {max_features}")
    print(f"Min improvement: {min_improvement}")
    print(f"Inner CV Folds: {len(INNER_CV_FOLDS)}")
    print("=" * 70)

    current_stock = dict(baseline_stock)
    current_macro = list(baseline_macro)
    history = []

    if excluded_features is None:
        excluded_features = set()
    else:
        excluded_features = set(excluded_features)

    if excluded_features:
        print(f"\nExcluded features from previous runs: {len(excluded_features)}")
        for f in sorted(excluded_features):
            print(f"  - {f}")

    # 基线评估
    print("\n[*] Evaluating baseline features...")
    baseline_ic, baseline_fold_ics = evaluate_feature_set_inner_cv(
        symbols, current_stock, current_macro,
        nday, epochs, early_stop, batch_size
    )

    print(f"    Baseline Inner CV IC: {baseline_ic:.4f}")
    print(f"    Fold ICs: {[f'{ic:.4f}' for ic in baseline_fold_ics]}")

    history.append({
        'round': 0,
        'action': 'BASELINE',
        'feature': None,
        'type': None,
        'inner_cv_ic': baseline_ic,
        'fold_ics': baseline_fold_ics,
        'ic_change': 0,
        'stock_count': len(current_stock),
        'macro_count': len(current_macro),
    })

    current_ic = baseline_ic
    round_num = 0

    # Forward selection
    while len(current_stock) + len(current_macro) < max_features:
        round_num += 1
        total_features = len(current_stock) + len(current_macro)

        # 计算可测试的候选特征（排除已选中的和被排除的）
        testable_stock = {k: v for k, v in candidate_stock.items()
                         if k not in current_stock and k not in excluded_features}
        testable_macro = [m for m in candidate_macro
                         if m not in current_macro and m not in excluded_features]

        if not testable_stock and not testable_macro:
            print(f"\n[!] No more candidates to test. Stopping.")
            break

        print(f"\n[Round {round_num}] Current IC: {current_ic:.4f}, Features: {total_features}")
        print(f"    Excluded features: {len(excluded_features)} (will skip)")
        print(f"    Testing {len(testable_stock)} stock + {len(testable_macro)} macro candidates...")

        candidates = []
        newly_excluded = []

        # 测试每个候选股票特征
        for name, expr in testable_stock.items():
            test_stock = dict(current_stock)
            test_stock[name] = expr

            for attempt in range(3):
                try:
                    tf.keras.backend.clear_session()
                    gc.collect()

                    ic, fold_ics = evaluate_feature_set_inner_cv(
                        symbols, test_stock, current_macro,
                        nday, epochs, early_stop, batch_size
                    )
                    ic_change = ic - current_ic
                    candidates.append({
                        'name': name,
                        'type': 'stock',
                        'expr': expr,
                        'ic': ic,
                        'fold_ics': fold_ics,
                        'ic_change': ic_change,
                    })

                    # 如果加入后 IC 下降，排除该特征
                    if ic_change < 0:
                        excluded_features.add(name)
                        newly_excluded.append(name)
                        symbol = "🚫"

                        # 立即保存
                        if output_dir:
                            _save_checkpoint(output_dir, round_num, current_ic,
                                           current_stock, current_macro,
                                           excluded_features, history)
                    else:
                        symbol = "+" if ic_change >= min_improvement else ""

                    print(f"      +{name}: IC={ic:.4f} ({symbol}{ic_change:+.4f})")
                    break

                except Exception as e:
                    if attempt < 2:
                        print(f"      +{name}: Retry {attempt+1}/3 after error...")
                        tf.keras.backend.clear_session()
                        gc.collect()
                    else:
                        print(f"      +{name}: ERROR - {e}")

        # 测试每个候选宏观特征
        for name in testable_macro:
            test_macro = current_macro + [name]

            for attempt in range(3):
                try:
                    tf.keras.backend.clear_session()
                    gc.collect()

                    ic, fold_ics = evaluate_feature_set_inner_cv(
                        symbols, current_stock, test_macro,
                        nday, epochs, early_stop, batch_size
                    )
                    ic_change = ic - current_ic
                    candidates.append({
                        'name': name,
                        'type': 'macro',
                        'expr': None,
                        'ic': ic,
                        'fold_ics': fold_ics,
                        'ic_change': ic_change,
                    })

                    # 如果加入后 IC 下降，排除该特征
                    if ic_change < 0:
                        excluded_features.add(name)
                        newly_excluded.append(name)
                        symbol = "🚫"

                        # 立即保存
                        if output_dir:
                            _save_checkpoint(output_dir, round_num, current_ic,
                                           current_stock, current_macro,
                                           excluded_features, history)
                    else:
                        symbol = "+" if ic_change >= min_improvement else ""

                    print(f"      +{name}: IC={ic:.4f} ({symbol}{ic_change:+.4f})")
                    break

                except Exception as e:
                    if attempt < 2:
                        print(f"      +{name}: Retry {attempt+1}/3 after error...")
                        tf.keras.backend.clear_session()
                        gc.collect()
                    else:
                        print(f"      +{name}: ERROR - {e}")

        # 打印本轮新排除的特征
        if newly_excluded:
            print(f"\n    🚫 Newly excluded features (IC dropped):")
            for f in newly_excluded:
                print(f"       - {f}")

        if not candidates:
            print("    No valid candidates, stopping.")
            break

        # 找到加入后 IC 提升最大的特征
        positive_candidates = [c for c in candidates if c['ic_change'] >= min_improvement]

        if not positive_candidates:
            print(f"\n[!] Stopping: No candidate improved IC by >= {min_improvement}")
            best_change = max(c['ic_change'] for c in candidates) if candidates else 0
            print(f"    Best improvement found: {best_change:+.4f}")
            break

        positive_candidates.sort(key=lambda x: x['ic_change'], reverse=True)
        best = positive_candidates[0]

        # 添加该特征
        if best['type'] == 'stock':
            current_stock[best['name']] = best['expr']
        else:
            current_macro.append(best['name'])

        current_ic = best['ic']

        history.append({
            'round': round_num,
            'action': 'ADD',
            'feature': best['name'],
            'type': best['type'],
            'inner_cv_ic': best['ic'],
            'fold_ics': best['fold_ics'],
            'ic_change': best['ic_change'],
            'stock_count': len(current_stock),
            'macro_count': len(current_macro),
        })

        print(f"\n    ✓ Added {best['name']} ({best['type']})")
        print(f"      IC: {best['ic']:.4f} (+{best['ic_change']:.4f})")
        print(f"      Features: {len(current_stock)} stock + {len(current_macro)} macro")

        # 保存 checkpoint
        if output_dir:
            _save_checkpoint(output_dir, round_num, current_ic,
                           current_stock, current_macro,
                           excluded_features, history)
            print(f"      Checkpoint saved (excluded: {len(excluded_features)} features)")

    # 最终结果
    print("\n" + "=" * 70)
    print("FORWARD SELECTION COMPLETE")
    print("=" * 70)
    print(f"Baseline features: {len(baseline_stock)} stock + {len(baseline_macro)} macro")
    print(f"Final features:    {len(current_stock)} stock + {len(current_macro)} macro")
    print(f"Baseline IC: {baseline_ic:.4f}")
    print(f"Final IC:    {current_ic:.4f} ({'+' if current_ic >= baseline_ic else ''}{current_ic - baseline_ic:.4f})")

    print(f"\nFinal Stock Features ({len(current_stock)}):")
    for name in sorted(current_stock.keys()):
        print(f"  - {name}")

    print(f"\nFinal Macro Features ({len(current_macro)}):")
    for name in current_macro:
        print(f"  - {name}")

    print(f"\nExcluded Features ({len(excluded_features)}):")
    for name in sorted(excluded_features):
        print(f"  - {name}")

    # 保存最终结果
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        result = {
            'timestamp': timestamp,
            'method': 'nested_cv_forward_selection',
            'baseline_stock_count': len(baseline_stock),
            'baseline_macro_count': len(baseline_macro),
            'final_stock_count': len(current_stock),
            'final_macro_count': len(current_macro),
            'baseline_ic': baseline_ic,
            'final_ic': current_ic,
            'final_stock_features': current_stock,
            'final_macro_features': current_macro,
            'excluded_features': list(excluded_features),
            'history': history,
        }

        result_file = output_dir / f"forward_selection_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {result_file}")

    return current_stock, current_macro, history, excluded_features


def main():
    parser = argparse.ArgumentParser(description='Nested CV Forward Feature Selection')
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--max-features', type=int, default=50,
                        help='Maximum number of features')
    parser.add_argument('--min-improvement', type=float, default=0.0005,
                        help='Minimum IC improvement to add a feature')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--early-stop', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--no-countdown', action='store_true')
    parser.add_argument('--baseline', type=str, default=None,
                        help='Backward elimination result file to use as baseline')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint file')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from a specific result file')

    args = parser.parse_args()

    # GPU 设置
    gpus = tf.config.list_physical_devices('GPU')
    if args.gpu >= 0 and gpus:
        tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)

    # Qlib 初始化
    qlib_data_path = project_root / "my_data" / "qlib_us"
    qlib.init(
        provider_uri=str(qlib_data_path),
        region=REG_US,
        custom_ops=TALIB_OPS,
    )

    symbols = STOCK_POOLS[args.stock_pool]
    output_dir = project_root / "outputs" / "feature_selection"

    print(f"Stock pool: {args.stock_pool} ({len(symbols)} stocks)")

    # 加载 baseline 或从 checkpoint 恢复
    excluded_features = set()
    baseline_stock = {}
    baseline_macro = []

    if args.resume_from:
        # 从指定文件恢复
        resume_path = Path(args.resume_from)
        if not resume_path.is_absolute():
            resume_path = output_dir / resume_path

        print(f"\n[*] Resuming from: {resume_path}")
        with open(resume_path, 'r') as f:
            checkpoint = json.load(f)

        if 'current_stock_features' in checkpoint:
            baseline_stock = checkpoint['current_stock_features']
            baseline_macro = checkpoint['current_macro_features']
        else:
            baseline_stock = checkpoint['final_stock_features']
            baseline_macro = checkpoint['final_macro_features']

        excluded_features = set(checkpoint.get('excluded_features', []))
        print(f"    Features: {len(baseline_stock)} stock + {len(baseline_macro)} macro")
        print(f"    Excluded: {len(excluded_features)}")

    elif args.resume:
        # 从默认 checkpoint 恢复
        checkpoint_file = output_dir / "forward_selection_checkpoint.json"
        if checkpoint_file.exists():
            print(f"\n[*] Resuming from checkpoint")
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)

            baseline_stock = checkpoint['current_stock_features']
            baseline_macro = checkpoint['current_macro_features']
            excluded_features = set(checkpoint.get('excluded_features', []))
            print(f"    Features: {len(baseline_stock)} stock + {len(baseline_macro)} macro")
            print(f"    Excluded: {len(excluded_features)}")
        else:
            print("ERROR: No checkpoint file found")
            return

    elif args.baseline:
        # 从 backward elimination 结果加载 baseline
        baseline_path = Path(args.baseline)
        if not baseline_path.is_absolute():
            baseline_path = output_dir / baseline_path

        print(f"\n[*] Loading baseline from: {baseline_path}")
        baseline_stock, baseline_macro, protected = load_baseline_from_file(baseline_path)
        print(f"    Baseline (from protected_features): {len(baseline_stock)} stock + {len(baseline_macro)} macro")

    else:
        print("ERROR: Must specify --baseline, --resume, or --resume-from")
        return

    # 获取候选特征（不在 baseline 中的特征）
    print("\n[*] Preparing candidate features...")

    candidate_stock = {k: v for k, v in ALL_STOCK_FEATURES.items()
                       if k not in baseline_stock}
    candidate_macro = [m for m in ALL_MACRO_FEATURES
                       if m not in baseline_macro]

    # 验证候选特征
    print(f"    Validating {len(candidate_stock)} stock candidates...")
    candidate_stock = validate_candidate_features(symbols, candidate_stock)
    print(f"    Valid: {len(candidate_stock)}")

    print(f"    Validating {len(candidate_macro)} macro candidates...")
    candidate_macro = validate_macro_features(candidate_macro)
    print(f"    Valid: {len(candidate_macro)}")

    # 倒计时
    if not args.no_countdown:
        print("\nProceed with forward selection? (Press Ctrl+C to abort)")
        try:
            import time
            for i in range(3, 0, -1):
                print(f"  Starting in {i}...")
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nAborted.")
            return

    # 运行 forward selection
    final_stock, final_macro, history, final_excluded = run_forward_selection(
        symbols,
        baseline_stock,
        baseline_macro,
        candidate_stock,
        candidate_macro,
        nday=args.nday,
        max_features=args.max_features,
        min_improvement=args.min_improvement,
        epochs=args.epochs,
        early_stop=args.early_stop,
        batch_size=args.batch_size,
        output_dir=output_dir,
        excluded_features=excluded_features,
    )

    print(f"\n✓ Forward selection complete")
    print(f"  Final features: {len(final_stock)} stock + {len(final_macro)} macro")
    print(f"  Excluded features: {len(final_excluded)}")


if __name__ == "__main__":
    main()
