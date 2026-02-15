"""
嵌套交叉验证 Backward Elimination - 从V7开始逐步精简特征

设计原理:
- 从V7的40个特征开始（已验证 Test IC 0.0446）
- 每轮删除1个对IC影响最小的特征
- 使用内层4折CV评估，外层测试集(2025)完全独立
- 保留特征间的协同效应

内层CV (只用2000-2024年数据):
  Fold 1: train 2000-2020, valid 2021
  Fold 2: train 2000-2021, valid 2022
  Fold 3: train 2000-2022, valid 2023
  Fold 4: train 2000-2023, valid 2024

使用方法:
    python scripts/models/feature_engineering/nested_cv_backward_elimination.py
    python scripts/models/feature_engineering/nested_cv_backward_elimination.py --stock-pool sp500 --min-features 15
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
from tensorflow.keras import callbacks

from utils.talib_ops import TALIB_OPS
from data.stock_pools import STOCK_POOLS
from models.deep.ae_mlp_shared import build_ae_mlp_model

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
# V7 完整特征集 (40个特征)
# ============================================================================

V7_STOCK_FEATURES = {
    # === 核心价格动量 (5) ===
    "MOMENTUM_QUALITY": "($close/Ref($close, 20) - 1) / (Std($close/Ref($close,1)-1, 20) + 1e-12)",
    "ROC5": "Ref($close, 5)/$close",
    "ROC20": "Ref($close, 20)/$close",
    "ROC60": "Ref($close, 60)/$close",
    "TALIB_RSI14": "TALIB_RSI($close, 14)",

    # === 波动率 (4) ===
    "TALIB_NATR14": "TALIB_NATR($high, $low, $close, 14)",
    "STD20": "Std($close, 20)/$close",
    "STD60": "Std($close, 60)/$close",
    "TALIB_ATR14": "TALIB_ATR($high, $low, $close, 14)/$close",

    # === 价格位置 (5) ===
    "PCT_FROM_52W_HIGH": "($close - Max($high, 252)) / (Max($high, 252) + 1e-12)",
    "PCT_FROM_52W_LOW": "($close - Min($low, 252)) / (Min($low, 252) + 1e-12)",
    "MAX60": "Max($high, 60)/$close",
    "MIN60": "Min($low, 60)/$close",
    "CLOSE_POSITION_60": "($close - Min($low, 60))/(Max($high, 60) - Min($low, 60) + 1e-12)",

    # === 均值回归 (4) ===
    "RESI5": "Resi($close, 5)/$close",
    "RESI10": "Resi($close, 10)/$close",
    "RESI20": "Resi($close, 20)/$close",
    "MA_RATIO_5_20": "Mean($close, 5)/Mean($close, 20)",

    # === 成交量 (3) ===
    "VOLUME_RATIO_5_20": "Mean($volume, 5)/(Mean($volume, 20)+1e-12)",
    "VOLUME_STD_20": "Std($volume, 20)/(Mean($volume, 20)+1e-12)",
    "VWAP_BIAS": "($close*$volume)/Sum($volume, 20) - Mean($close, 20)",

    # === 趋势强度 (3) ===
    "TALIB_ADX14": "TALIB_ADX($high, $low, $close, 14)",
    "SLOPE20": "Slope($close, 20)/$close",
    "SLOPE60": "Slope($close, 60)/$close",

    # === Drawdown (2) ===
    "MAX_DRAWDOWN_20": "(Min($close, 20) - Max($high, 20)) / (Max($high, 20) + 1e-12)",
    "MAX_DRAWDOWN_60": "(Min($close, 60) - Max($high, 60)) / (Max($high, 60) + 1e-12)",

    # === 技术指标 (3) ===
    "TALIB_WILLR14": "TALIB_WILLR($high, $low, $close, 14)",
    "TALIB_CCI14": "TALIB_CCI($high, $low, $close, 14)",
    "TALIB_MFI14": "TALIB_MFI($high, $low, $close, $volume, 14)",
}

V7_MACRO_FEATURES = [
    # VIX 相关 (3)
    "macro_vix_zscore20",
    "macro_vix_regime",
    "macro_vix_pct_5d",

    # 信用/风险 (3)
    "macro_hy_spread_zscore",
    "macro_credit_stress",
    "macro_hyg_pct_5d",

    # 利率/债券 (2)
    "macro_yield_curve",
    "macro_tlt_pct_20d",

    # 商品 (2)
    "macro_gld_pct_20d",
    "macro_uso_pct_20d",

    # 美元 (1)
    "macro_uup_pct_5d",

    # 市场 (2)
    "macro_spy_pct_5d",
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
        # 强制清理内存
        tf.keras.backend.clear_session()
        gc.collect()

        X_train, y_train, X_valid, y_valid, valid_index = prepare_fold_data(
            symbols, fold, feature_config, macro_features, nday
        )

        num_features = X_train.shape[1]
        model = build_ae_mlp_model({**BEST_HYPERPARAMS, 'num_columns': num_features})

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

        # 清理当前fold的数据
        del X_train, y_train, X_valid, y_valid, valid_index, model
        del train_outputs, valid_outputs
        gc.collect()

    mean_ic = np.mean(fold_ics)
    return mean_ic, fold_ics


def validate_v7_features(symbols) -> Tuple[Dict[str, str], List[str]]:
    """验证V7特征是否都可用"""
    from qlib.data import D

    print("\n" + "=" * 70)
    print("VALIDATING V7 FEATURES")
    print("=" * 70)

    test_start = "2024-01-01"
    test_end = "2024-01-10"
    test_symbols = symbols[:5] if len(symbols) > 5 else symbols

    valid_stock = {}
    invalid_features = []

    print(f"\n[1/2] Validating stock features ({len(V7_STOCK_FEATURES)})...")
    for name, expr in V7_STOCK_FEATURES.items():
        try:
            df = D.features(test_symbols, [expr], start_time=test_start, end_time=test_end)
            if df is not None and len(df) > 0:
                valid_stock[name] = expr
                print(f"  ✓ {name}")
            else:
                invalid_features.append((name, "empty result"))
                print(f"  ✗ {name}: empty result")
        except Exception as e:
            invalid_features.append((name, str(e)))
            print(f"  ✗ {name}: {e}")

    print(f"\n[2/2] Validating macro features ({len(V7_MACRO_FEATURES)})...")
    valid_macro = []
    macro_path = project_root / "my_data" / "macro_processed" / "macro_features.parquet"

    if macro_path.exists():
        try:
            macro_df = pd.read_parquet(macro_path)
            available_cols = set(macro_df.columns)

            for name in V7_MACRO_FEATURES:
                if name in available_cols:
                    valid_macro.append(name)
                    print(f"  ✓ {name}")
                else:
                    invalid_features.append((name, "not in macro data"))
                    print(f"  ✗ {name}: not found in macro data")
        except Exception as e:
            print(f"  ! Error loading macro data: {e}")

    print("\n" + "-" * 70)
    print(f"Stock features: {len(valid_stock)}/{len(V7_STOCK_FEATURES)} valid")
    print(f"Macro features: {len(valid_macro)}/{len(V7_MACRO_FEATURES)} valid")
    print(f"Total: {len(valid_stock) + len(valid_macro)} features")

    if invalid_features:
        print(f"\n⚠ {len(invalid_features)} invalid features:")
        for name, reason in invalid_features:
            print(f"  - {name}: {reason}")

    print("=" * 70)
    return valid_stock, valid_macro


def _save_checkpoint(output_dir: Path, round_num: int, current_ic: float,
                     current_stock: Dict, current_macro: List,
                     protected_features: set, history: List):
    """保存 checkpoint 到文件"""
    checkpoint = {
        'round': round_num,
        'current_ic': current_ic,
        'current_stock_features': current_stock,
        'current_macro_features': current_macro,
        'protected_features': list(protected_features),
        'history': history,
    }
    checkpoint_file = output_dir / "backward_elimination_checkpoint.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def run_backward_elimination(
    symbols,
    stock_features: Dict[str, str],
    macro_features: List[str],
    nday: int = 5,
    min_features: int = 10,
    max_ic_drop: float = 0.005,
    protect_threshold: float = -0.002,
    epochs: int = 15,
    early_stop: int = 5,
    batch_size: int = 2048,
    output_dir: Path = None,
    protected_features: set = None,
):
    """
    Backward Elimination - 从V7开始逐步精简特征

    每轮删除对IC影响最小的1个特征，直到：
    1. 特征数达到 min_features，或
    2. 删除任何特征都会导致IC下降超过 max_ic_drop

    优化：如果某特征删除后IC变化 < protect_threshold，则将其标记为"受保护"，
    后续轮次不再测试删除该特征。
    """
    print("\n" + "=" * 70)
    print("NESTED CV BACKWARD ELIMINATION")
    print("=" * 70)
    print(f"Starting features: {len(stock_features)} stock + {len(macro_features)} macro")
    print(f"Min features: {min_features}")
    print(f"Max IC drop per round: {max_ic_drop}")
    print(f"Protection threshold: {protect_threshold} (features with IC drop < this will be protected)")
    print(f"Inner CV Folds: {len(INNER_CV_FOLDS)}")
    print("=" * 70)

    current_stock = dict(stock_features)
    current_macro = list(macro_features)
    history = []

    # 受保护的特征集合（删除后IC下降明显的特征）
    if protected_features is None:
        protected_features = set()
    else:
        protected_features = set(protected_features)

    if protected_features:
        print(f"\nProtected features from previous runs: {len(protected_features)}")
        for f in sorted(protected_features):
            print(f"  - {f}")

    # 基线评估
    print("\n[*] Evaluating baseline (full V7 features)...")
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

    # Backward elimination
    while len(current_stock) + len(current_macro) > min_features:
        round_num += 1
        total_features = len(current_stock) + len(current_macro)

        # 计算可测试的特征数（排除受保护的）
        testable_stock = [n for n in current_stock.keys() if n not in protected_features]
        testable_macro = [n for n in current_macro if n not in protected_features]

        print(f"\n[Round {round_num}] Current IC: {current_ic:.4f}, Features: {total_features}")
        print(f"    Protected features: {len(protected_features)} (will skip)")
        print(f"    Testing removal of {len(testable_stock)} stock + {len(testable_macro)} macro features...")

        candidates = []
        newly_protected = []  # 本轮新发现的重要特征

        # 测试删除每个股票特征（跳过受保护的）
        for name in testable_stock:
            test_stock = {k: v for k, v in current_stock.items() if k != name}

            # 尝试最多3次
            success = False
            for attempt in range(3):
                try:
                    # 每次尝试前清理内存
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
                        'ic': ic,
                        'fold_ics': fold_ics,
                        'ic_change': ic_change,
                    })

                    # 检查是否应该保护这个特征
                    if ic_change < protect_threshold:
                        protected_features.add(name)
                        newly_protected.append(name)
                        symbol = "🛡️"  # 标记为受保护

                        # 立即保存受保护特征到 checkpoint
                        if output_dir:
                            _save_checkpoint(output_dir, round_num, current_ic,
                                           current_stock, current_macro,
                                           protected_features, history)
                    else:
                        symbol = "+" if ic_change >= 0 else ""

                    print(f"      -{name}: IC={ic:.4f} ({symbol}{ic_change:.4f})")
                    success = True
                    break

                except Exception as e:
                    if attempt < 2:
                        print(f"      -{name}: Retry {attempt+1}/3 after error: {str(e)[:50]}...")
                        tf.keras.backend.clear_session()
                        gc.collect()
                    else:
                        print(f"      -{name}: ERROR - {e}")

        # 测试删除每个宏观特征（跳过受保护的）
        for name in testable_macro:
            test_macro = [m for m in current_macro if m != name]

            # 尝试最多3次
            success = False
            for attempt in range(3):
                try:
                    # 每次尝试前清理内存
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
                        'ic': ic,
                        'fold_ics': fold_ics,
                        'ic_change': ic_change,
                    })

                    # 检查是否应该保护这个特征
                    if ic_change < protect_threshold:
                        protected_features.add(name)
                        newly_protected.append(name)
                        symbol = "🛡️"  # 标记为受保护

                        # 立即保存受保护特征到 checkpoint
                        if output_dir:
                            _save_checkpoint(output_dir, round_num, current_ic,
                                           current_stock, current_macro,
                                           protected_features, history)
                    else:
                        symbol = "+" if ic_change >= 0 else ""

                    print(f"      -{name}: IC={ic:.4f} ({symbol}{ic_change:.4f})")
                    success = True
                    break

                except Exception as e:
                    if attempt < 2:
                        print(f"      -{name}: Retry {attempt+1}/3 after error: {str(e)[:50]}...")
                        tf.keras.backend.clear_session()
                        gc.collect()
                    else:
                        print(f"      -{name}: ERROR - {e}")

        # 打印本轮新保护的特征
        if newly_protected:
            print(f"\n    🛡️ Newly protected features (IC drop < {protect_threshold}):")
            for f in newly_protected:
                print(f"       - {f}")

        if not candidates:
            print("    No valid candidates, stopping.")
            break

        # 找到删除后IC下降最小（或提升最大）的特征
        candidates.sort(key=lambda x: x['ic_change'], reverse=True)
        best = candidates[0]

        # 检查是否应该停止
        if best['ic_change'] < -max_ic_drop:
            print(f"\n[!] Stopping: Best removal would drop IC by {-best['ic_change']:.4f} (> {max_ic_drop})")
            break

        # 删除该特征
        if best['type'] == 'stock':
            del current_stock[best['name']]
        else:
            current_macro.remove(best['name'])

        current_ic = best['ic']

        history.append({
            'round': round_num,
            'action': 'REMOVE',
            'feature': best['name'],
            'type': best['type'],
            'inner_cv_ic': best['ic'],
            'fold_ics': best['fold_ics'],
            'ic_change': best['ic_change'],
            'stock_count': len(current_stock),
            'macro_count': len(current_macro),
        })

        symbol = "+" if best['ic_change'] >= 0 else ""
        print(f"\n    ✓ Removed {best['name']} ({best['type']})")
        print(f"      IC: {best['ic']:.4f} ({symbol}{best['ic_change']:.4f})")
        print(f"      Remaining: {len(current_stock)} stock + {len(current_macro)} macro")

        # 每轮结束后保存中间结果
        if output_dir:
            checkpoint = {
                'round': round_num,
                'current_ic': current_ic,
                'current_stock_features': current_stock,
                'current_macro_features': current_macro,
                'protected_features': list(protected_features),
                'history': history,
            }
            checkpoint_file = output_dir / "backward_elimination_checkpoint.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            print(f"      Checkpoint saved (protected: {len(protected_features)} features)")

    # 最终结果
    print("\n" + "=" * 70)
    print("BACKWARD ELIMINATION COMPLETE")
    print("=" * 70)
    print(f"Initial features: {len(stock_features)} stock + {len(macro_features)} macro = {len(stock_features) + len(macro_features)}")
    print(f"Final features:   {len(current_stock)} stock + {len(current_macro)} macro = {len(current_stock) + len(current_macro)}")
    print(f"Baseline IC: {baseline_ic:.4f}")
    print(f"Final IC:    {current_ic:.4f} ({'+' if current_ic >= baseline_ic else ''}{current_ic - baseline_ic:.4f})")

    print(f"\nFinal Stock Features ({len(current_stock)}):")
    for name in sorted(current_stock.keys()):
        print(f"  - {name}")

    print(f"\nFinal Macro Features ({len(current_macro)}):")
    for name in current_macro:
        print(f"  - {name}")

    print(f"\nProtected Features ({len(protected_features)}):")
    for name in sorted(protected_features):
        print(f"  - {name}")

    # 保存结果
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        result = {
            'timestamp': timestamp,
            'method': 'nested_cv_backward_elimination',
            'initial_stock_count': len(stock_features),
            'initial_macro_count': len(macro_features),
            'final_stock_count': len(current_stock),
            'final_macro_count': len(current_macro),
            'baseline_ic': baseline_ic,
            'final_ic': current_ic,
            'final_stock_features': current_stock,
            'final_macro_features': current_macro,
            'protected_features': list(protected_features),
            'history': history,
        }

        result_file = output_dir / f"backward_elimination_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {result_file}")

    return current_stock, current_macro, history, protected_features


def main():
    parser = argparse.ArgumentParser(description='Nested CV Backward Elimination')
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--min-features', type=int, default=10,
                        help='Minimum number of features to keep')
    parser.add_argument('--max-ic-drop', type=float, default=0.005,
                        help='Stop if removing any feature drops IC by more than this')
    parser.add_argument('--protect-threshold', type=float, default=-0.002,
                        help='Protect features if removing them drops IC by more than this (default: -0.002)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--early-stop', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--skip-validation', action='store_true')
    parser.add_argument('--no-countdown', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint file')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from a specific result file (e.g., backward_elimination_20260118_032017.json)')

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

    # 检查是否从 checkpoint 或指定文件恢复
    resume_file = None
    if args.resume_from:
        # 从指定文件恢复
        resume_path = Path(args.resume_from)
        if not resume_path.is_absolute():
            resume_path = output_dir / resume_path
        if resume_path.exists():
            resume_file = resume_path
        else:
            print(f"ERROR: Resume file not found: {resume_path}")
            return
    elif args.resume:
        # 从默认 checkpoint 恢复
        checkpoint_file = output_dir / "backward_elimination_checkpoint.json"
        if checkpoint_file.exists():
            resume_file = checkpoint_file

    protected_features = set()

    if resume_file:
        print(f"\n[*] Resuming from: {resume_file}")
        with open(resume_file, 'r') as f:
            checkpoint = json.load(f)

        # 支持两种格式：checkpoint 格式和最终结果格式
        if 'current_stock_features' in checkpoint:
            valid_stock = checkpoint['current_stock_features']
            valid_macro = checkpoint['current_macro_features']
            resume_round = checkpoint.get('round', 0)
            resume_ic = checkpoint.get('current_ic', 0)
        else:
            # 最终结果格式
            valid_stock = checkpoint['final_stock_features']
            valid_macro = checkpoint['final_macro_features']
            resume_round = len(checkpoint.get('history', [])) - 1
            resume_ic = checkpoint.get('final_ic', 0)

        # 加载受保护的特征
        if 'protected_features' in checkpoint:
            protected_features = set(checkpoint['protected_features'])

        print(f"    Resuming from round {resume_round}")
        print(f"    Current IC: {resume_ic:.4f}")
        print(f"    Features: {len(valid_stock)} stock + {len(valid_macro)} macro")
        print(f"    Protected features: {len(protected_features)}")
    else:
        # 验证特征
        if args.skip_validation:
            print("\n[!] Skipping validation")
            valid_stock = dict(V7_STOCK_FEATURES)
            valid_macro = list(V7_MACRO_FEATURES)
        else:
            valid_stock, valid_macro = validate_v7_features(symbols)

    # 倒计时
    if not args.no_countdown:
        print("\nProceed with backward elimination? (Press Ctrl+C to abort)")
        try:
            import time
            for i in range(3, 0, -1):
                print(f"  Starting in {i}...")
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nAborted.")
            return

    # 运行 backward elimination
    final_stock, final_macro, history, final_protected = run_backward_elimination(
        symbols,
        valid_stock,
        valid_macro,
        nday=args.nday,
        min_features=args.min_features,
        max_ic_drop=args.max_ic_drop,
        protect_threshold=args.protect_threshold,
        epochs=args.epochs,
        early_stop=args.early_stop,
        batch_size=args.batch_size,
        output_dir=output_dir,
        protected_features=protected_features,
    )

    print(f"\n✓ Backward elimination complete")
    print(f"  Final features: {len(final_stock)} stock + {len(final_macro)} macro")
    print(f"  Protected features: {len(final_protected)}")


if __name__ == "__main__":
    main()
