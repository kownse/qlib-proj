"""
TCN 嵌套交叉验证 Forward Selection - 基于 Alpha180 逐步添加 Macro 和 TALib 特征

设计原理:
- 从 Alpha180 (6 OHLCV × 30 days = 180 features) 作为基线
- 逐个测试候选特征（macro 和 TALib），每个特征加入过去30天的值
- 如果某特征加入后导致 IC 变差，则加入排除列表
- 后续轮次跳过被排除的特征

内层CV (只用2000-2024年数据):
  Fold 1: train 2000-2020, valid 2021
  Fold 2: train 2000-2021, valid 2022
  Fold 3: train 2000-2022, valid 2023
  Fold 4: train 2000-2023, valid 2024

使用方法:
    python scripts/models/analysis/nested_cv_feature_selection_tcn.py --stock-pool sp500
    python scripts/models/analysis/nested_cv_feature_selection_tcn.py --resume
    python scripts/models/analysis/nested_cv_feature_selection_tcn.py --resume-from forward_selection_tcn_xxx.json
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

import torch
import torch.nn as nn

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
# 候选特征池
# ============================================================================

# 所有可能的宏观特征 (每个都会扩展为30天的历史)
ALL_MACRO_FEATURES = [
    # VIX 相关
    "macro_vix_level",
    "macro_vix_zscore20",
    "macro_vix_regime",
    "macro_vix_pct_5d",
    "macro_vix_term_structure",
    # 信用/风险
    "macro_hy_spread_zscore",
    "macro_credit_stress",
    "macro_hyg_pct_5d",
    "macro_hyg_pct_20d",
    "macro_hyg_vs_lqd",
    # 利率/债券
    "macro_yield_curve",
    "macro_tlt_pct_5d",
    "macro_tlt_pct_20d",
    "macro_yield_10y",
    "macro_yield_2s10s",
    "macro_yield_inversion",
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
    "macro_qqq_vs_spy",
    # 跨资产
    "macro_risk_on_off",
    "macro_market_stress",
    "macro_global_risk",
    "macro_eem_vs_spy",
]

# 所有可能的 TALib 特征 (每个都会扩展为30天的历史)
# 这些是 Qlib 表达式，会为每个时间步计算
ALL_TALIB_FEATURES = {
    # 动量指标
    "TALIB_RSI14": "TALIB_RSI($close, 14)",
    "TALIB_MOM10": "TALIB_MOM($close, 10)/$close",
    "TALIB_ROC10": "TALIB_ROC($close, 10)",
    "TALIB_CMO14": "TALIB_CMO($close, 14)",
    "TALIB_WILLR14": "TALIB_WILLR($high, $low, $close, 14)",

    # MACD
    "TALIB_MACD": "TALIB_MACD_MACD($close, 12, 26, 9)/$close",
    "TALIB_MACD_SIGNAL": "TALIB_MACD_SIGNAL($close, 12, 26, 9)/$close",
    "TALIB_MACD_HIST": "TALIB_MACD_HIST($close, 12, 26, 9)/$close",

    # 移动平均
    "TALIB_EMA20": "TALIB_EMA($close, 20)/$close",
    "TALIB_SMA20": "TALIB_SMA($close, 20)/$close",

    # 布林带
    "TALIB_BB_UPPER_DIST": "(TALIB_BBANDS_UPPER($close, 20, 2, 2) - $close)/$close",
    "TALIB_BB_LOWER_DIST": "($close - TALIB_BBANDS_LOWER($close, 20, 2, 2))/$close",
    "TALIB_BB_WIDTH": "(TALIB_BBANDS_UPPER($close, 20, 2, 2) - TALIB_BBANDS_LOWER($close, 20, 2, 2))/$close",

    # 波动率
    "TALIB_ATR14": "TALIB_ATR($high, $low, $close, 14)/$close",
    "TALIB_NATR14": "TALIB_NATR($high, $low, $close, 14)",

    # 趋势指标
    "TALIB_ADX14": "TALIB_ADX($high, $low, $close, 14)",
    "TALIB_PLUS_DI14": "TALIB_PLUS_DI($high, $low, $close, 14)",
    "TALIB_MINUS_DI14": "TALIB_MINUS_DI($high, $low, $close, 14)",

    # 随机指标
    "TALIB_STOCH_K": "TALIB_STOCH_K($high, $low, $close, 5, 3, 3)",
    "TALIB_STOCH_D": "TALIB_STOCH_D($high, $low, $close, 5, 3, 3)",

    # 统计
    "TALIB_STDDEV20": "TALIB_STDDEV($close, 20, 1)/$close",

    # CCI
    "TALIB_CCI14": "TALIB_CCI($high, $low, $close, 14)",

    # MFI
    "TALIB_MFI14": "TALIB_MFI($high, $low, $close, $volume, 14)",
}

# TCN 默认超参数
DEFAULT_TCN_PARAMS = {
    "n_chans": 64,
    "num_layers": 3,
    "kernel_size": 5,
    "dropout": 0.3,
    "lr": 0.001,
    "batch_size": 2048,
}


# ============================================================================
# TCN Model
# ============================================================================

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
            x = x.permute(0, 2, 1)  # (batch, features, seq_len)
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


def compute_ic(pred: np.ndarray, label: np.ndarray, index: pd.MultiIndex) -> float:
    """计算IC"""
    df = pd.DataFrame({'pred': pred, 'label': label}, index=index)
    ic_by_date = df.groupby(level='datetime').apply(
        lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
    )
    ic_by_date = ic_by_date.dropna()
    return ic_by_date.mean() if len(ic_by_date) > 0 else 0.0


# ============================================================================
# Dynamic Alpha180 Handler with Incremental Features
# ============================================================================

class DynamicAlpha180Handler(DataHandlerLP):
    """
    动态 Alpha180 Handler，支持增量添加 Macro 和 TALib 特征。

    基线: Alpha180 (6 OHLCV × 30 days = 180 features)
    增量特征: 每个 macro/talib 特征扩展为 30 天的历史
    """

    def __init__(
        self,
        talib_features: Dict[str, str] = None,  # {name: expression}
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
        **kwargs,
    ):
        self.talib_features = talib_features or {}
        self.macro_features = macro_features or []
        self.volatility_window = volatility_window

        # Load macro data
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

    def _get_alpha180_features(self):
        """获取 Alpha180 基线特征 (6 OHLCV × 30 days)"""
        fields = []
        names = []

        # CLOSE: 30 天历史收盘价
        for i in range(29, 0, -1):
            fields.append(f"Ref($close, {i})/$close")
            names.append(f"CLOSE{i}")
        fields.append("$close/$close")
        names.append("CLOSE0")

        # OPEN: 30 天历史开盘价
        for i in range(29, 0, -1):
            fields.append(f"Ref($open, {i})/$close")
            names.append(f"OPEN{i}")
        fields.append("$open/$close")
        names.append("OPEN0")

        # HIGH: 30 天历史最高价
        for i in range(29, 0, -1):
            fields.append(f"Ref($high, {i})/$close")
            names.append(f"HIGH{i}")
        fields.append("$high/$close")
        names.append("HIGH0")

        # LOW: 30 天历史最低价
        for i in range(29, 0, -1):
            fields.append(f"Ref($low, {i})/$close")
            names.append(f"LOW{i}")
        fields.append("$low/$close")
        names.append("LOW0")

        # VWAP: 30 天历史成交均价
        for i in range(29, 0, -1):
            fields.append(f"Ref($vwap, {i})/$close")
            names.append(f"VWAP{i}")
        fields.append("$vwap/$close")
        names.append("VWAP0")

        # VOLUME: 30 天历史成交量
        for i in range(29, 0, -1):
            fields.append(f"Ref($volume, {i})/($volume+1e-12)")
            names.append(f"VOLUME{i}")
        fields.append("$volume/($volume+1e-12)")
        names.append("VOLUME0")

        return fields, names

    def _get_talib_features_config(self):
        """获取 TALib 特征配置 (每个特征扩展为30天)"""
        fields = []
        names = []

        for feat_name, expr in self.talib_features.items():
            # 扩展为30天历史
            for i in range(29, 0, -1):
                # 使用 Ref 来获取历史值
                ref_expr = f"Ref({expr}, {i})"
                fields.append(ref_expr)
                names.append(f"{feat_name}_{i}")
            # 当天的值
            fields.append(expr)
            names.append(f"{feat_name}_0")

        return fields, names

    def _get_feature_config(self):
        """获取完整特征配置"""
        # Alpha180 基线
        fields, names = self._get_alpha180_features()

        # TALib 特征
        talib_fields, talib_names = self._get_talib_features_config()
        fields.extend(talib_fields)
        names.extend(talib_names)

        return fields, names

    def _get_label_config(self):
        """返回N天波动率标签"""
        label_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [label_expr], ["LABEL0"]

    def process_data(self, with_fit: bool = False):
        """处理数据，添加 macro 特征"""
        super().process_data(with_fit=with_fit)
        if self._macro_df is not None and self.macro_features:
            self._add_macro_features()

    def _add_macro_features(self):
        """添加时间对齐的 macro 特征 (每个扩展为30天)"""
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
                base_series = self._macro_df[col]
                # 扩展为30天历史
                for i in range(29, -1, -1):
                    col_name = f"{col}_{i}"
                    shifted = base_series.shift(i + 1)  # +1 for look-ahead prevention
                    aligned_values = shifted.reindex(main_datetimes).values
                    if has_multi_columns:
                        macro_data[('feature', col_name)] = aligned_values
                    else:
                        macro_data[col_name] = aligned_values

            macro_df = pd.DataFrame(macro_data, index=df.index)
            merged = pd.concat([df, macro_df], axis=1, copy=False)
            setattr(self, attr, merged.copy())


def prepare_fold_data(symbols, fold_config, talib_features, macro_features, nday=5):
    """为单个fold准备数据"""
    handler = DynamicAlpha180Handler(
        talib_features=talib_features,
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
    talib_features: Dict[str, str],
    macro_features: List[str],
    nday: int = 5,
    epochs: int = 20,
    early_stop: int = 8,
    params: dict = None,
    device=None,
) -> Tuple[float, List[float]]:
    """在内层CV上评估特征集"""
    if params is None:
        params = DEFAULT_TCN_PARAMS
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fold_ics = []

    for fold in INNER_CV_FOLDS:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        X_train, y_train, X_valid, y_valid, valid_index = prepare_fold_data(
            symbols, fold, talib_features, macro_features, nday
        )

        total_features = X_train.shape[1]
        # d_feat = number of features per timestep (total / 30 timesteps)
        d_feat = total_features // 30

        model = create_tcn_model(
            d_feat=d_feat,
            n_chans=params['n_chans'],
            num_layers=params['num_layers'],
            kernel_size=params['kernel_size'],
            dropout=params['dropout'],
            device=device,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        batch_size = params['batch_size']

        # Training loop
        best_loss = float('inf')
        best_state = None
        stop_steps = 0

        for epoch in range(epochs):
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

            # Validation
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
            else:
                stop_steps += 1
                if stop_steps >= early_stop:
                    break

        # Load best model and compute IC
        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            val_preds = []
            for i in range(0, len(X_valid), batch_size):
                end = min(i + batch_size, len(X_valid))
                feature = torch.from_numpy(X_valid[i:end]).float().to(device)
                pred = model(feature).squeeze()
                val_preds.append(pred.cpu().numpy())
            val_pred = np.concatenate(val_preds)

        ic = compute_ic(val_pred, y_valid, valid_index)
        fold_ics.append(ic)

        del X_train, y_train, X_valid, y_valid, valid_index, model
        gc.collect()

    mean_ic = np.mean(fold_ics)
    return mean_ic, fold_ics


def validate_talib_features(symbols, candidates: Dict[str, str]) -> Dict[str, str]:
    """验证 TALib 特征是否有效"""
    from qlib.data import D

    test_start = "2024-01-01"
    test_end = "2024-01-10"
    test_symbols = symbols[:5] if len(symbols) > 5 else symbols

    valid = {}
    for name, expr in candidates.items():
        try:
            df = D.features(test_symbols, [expr], start_time=test_start, end_time=test_end)
            if df is not None and len(df) > 0 and df.notna().any().any():
                valid[name] = expr
        except Exception:
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
                     current_talib: Dict, current_macro: List,
                     excluded_features: set, history: List):
    """保存 checkpoint"""
    checkpoint = {
        'round': round_num,
        'current_ic': current_ic,
        'current_talib_features': current_talib,
        'current_macro_features': current_macro,
        'excluded_features': list(excluded_features),
        'history': history,
    }
    checkpoint_file = output_dir / "forward_selection_tcn_checkpoint.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def run_forward_selection(
    symbols,
    candidate_talib: Dict[str, str],
    candidate_macro: List[str],
    nday: int = 5,
    max_features: int = 30,
    min_improvement: float = 0.0005,
    epochs: int = 20,
    early_stop: int = 8,
    params: dict = None,
    output_dir: Path = None,
    excluded_features: set = None,
    baseline_talib: Dict[str, str] = None,
    baseline_macro: List[str] = None,
    device=None,
):
    """
    Forward Selection - 从 Alpha180 基线开始逐步添加特征

    每个特征都会扩展为30天的历史值，与 Alpha180 的时间结构对齐
    """
    print("\n" + "=" * 70)
    print("TCN NESTED CV FORWARD SELECTION")
    print("=" * 70)
    print("Baseline: Alpha180 (6 OHLCV × 30 days = 180 features)")
    print(f"Candidate features: {len(candidate_talib)} TALib + {len(candidate_macro)} macro")
    print(f"Each feature expands to 30 timesteps")
    print(f"Max additional features: {max_features}")
    print(f"Min improvement: {min_improvement}")
    print(f"Inner CV Folds: {len(INNER_CV_FOLDS)}")
    print("=" * 70)

    current_talib = dict(baseline_talib) if baseline_talib else {}
    current_macro = list(baseline_macro) if baseline_macro else []
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
    print("\n[*] Evaluating baseline (Alpha180" +
          (f" + {len(current_talib)} TALib + {len(current_macro)} macro" if current_talib or current_macro else "") +
          ")...")
    baseline_ic, baseline_fold_ics = evaluate_feature_set_inner_cv(
        symbols, current_talib, current_macro,
        nday, epochs, early_stop, params, device
    )

    current_features_count = 6 + len(current_talib) + len(current_macro)
    print(f"    Baseline Inner CV IC: {baseline_ic:.4f}")
    print(f"    Fold ICs: {[f'{ic:.4f}' for ic in baseline_fold_ics]}")
    print(f"    d_feat: {current_features_count} (features per timestep)")

    history.append({
        'round': 0,
        'action': 'BASELINE',
        'feature': None,
        'type': None,
        'inner_cv_ic': baseline_ic,
        'fold_ics': baseline_fold_ics,
        'ic_change': 0,
        'd_feat': current_features_count,
        'talib_count': len(current_talib),
        'macro_count': len(current_macro),
    })

    current_ic = baseline_ic
    round_num = 0

    # Forward selection
    while len(current_talib) + len(current_macro) < max_features:
        round_num += 1
        total_added = len(current_talib) + len(current_macro)

        # 计算可测试的候选特征
        testable_talib = {k: v for k, v in candidate_talib.items()
                         if k not in current_talib and k not in excluded_features}
        testable_macro = [m for m in candidate_macro
                         if m not in current_macro and m not in excluded_features]

        if not testable_talib and not testable_macro:
            print(f"\n[!] No more candidates to test. Stopping.")
            break

        print(f"\n[Round {round_num}] Current IC: {current_ic:.4f}, Added features: {total_added}")
        print(f"    Excluded features: {len(excluded_features)} (will skip)")
        print(f"    Testing {len(testable_talib)} TALib + {len(testable_macro)} macro candidates...")

        candidates = []
        newly_excluded = []

        # 测试每个候选 TALib 特征
        for name, expr in testable_talib.items():
            test_talib = dict(current_talib)
            test_talib[name] = expr

            for attempt in range(3):
                try:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    ic, fold_ics = evaluate_feature_set_inner_cv(
                        symbols, test_talib, current_macro,
                        nday, epochs, early_stop, params, device
                    )
                    ic_change = ic - current_ic
                    candidates.append({
                        'name': name,
                        'type': 'talib',
                        'expr': expr,
                        'ic': ic,
                        'fold_ics': fold_ics,
                        'ic_change': ic_change,
                    })

                    # 如果加入后 IC 下降，排除该特征
                    if ic_change < 0:
                        excluded_features.add(name)
                        newly_excluded.append(name)
                        symbol = "X"

                        if output_dir:
                            _save_checkpoint(output_dir, round_num, current_ic,
                                           current_talib, current_macro,
                                           excluded_features, history)
                    else:
                        symbol = "+" if ic_change >= min_improvement else ""

                    print(f"      +{name}: IC={ic:.4f} ({symbol}{ic_change:+.4f})")
                    break

                except Exception as e:
                    if attempt < 2:
                        print(f"      +{name}: Retry {attempt+1}/3 after error...")
                        gc.collect()
                    else:
                        print(f"      +{name}: ERROR - {e}")

        # 测试每个候选宏观特征
        for name in testable_macro:
            test_macro = current_macro + [name]

            for attempt in range(3):
                try:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    ic, fold_ics = evaluate_feature_set_inner_cv(
                        symbols, current_talib, test_macro,
                        nday, epochs, early_stop, params, device
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
                        symbol = "X"

                        if output_dir:
                            _save_checkpoint(output_dir, round_num, current_ic,
                                           current_talib, current_macro,
                                           excluded_features, history)
                    else:
                        symbol = "+" if ic_change >= min_improvement else ""

                    print(f"      +{name}: IC={ic:.4f} ({symbol}{ic_change:+.4f})")
                    break

                except Exception as e:
                    if attempt < 2:
                        print(f"      +{name}: Retry {attempt+1}/3 after error...")
                        gc.collect()
                    else:
                        print(f"      +{name}: ERROR - {e}")

        # 打印本轮新排除的特征
        if newly_excluded:
            print(f"\n    X Newly excluded features (IC dropped):")
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
        if best['type'] == 'talib':
            current_talib[best['name']] = best['expr']
        else:
            current_macro.append(best['name'])

        current_ic = best['ic']
        current_features_count = 6 + len(current_talib) + len(current_macro)

        history.append({
            'round': round_num,
            'action': 'ADD',
            'feature': best['name'],
            'type': best['type'],
            'inner_cv_ic': best['ic'],
            'fold_ics': best['fold_ics'],
            'ic_change': best['ic_change'],
            'd_feat': current_features_count,
            'talib_count': len(current_talib),
            'macro_count': len(current_macro),
        })

        print(f"\n    + Added {best['name']} ({best['type']})")
        print(f"      IC: {best['ic']:.4f} (+{best['ic_change']:.4f})")
        print(f"      d_feat: {current_features_count} ({len(current_talib)} TALib + {len(current_macro)} macro)")

        # 保存 checkpoint
        if output_dir:
            _save_checkpoint(output_dir, round_num, current_ic,
                           current_talib, current_macro,
                           excluded_features, history)
            print(f"      Checkpoint saved (excluded: {len(excluded_features)} features)")

    # 最终结果
    print("\n" + "=" * 70)
    print("FORWARD SELECTION COMPLETE")
    print("=" * 70)
    print(f"Baseline: Alpha180 (6 features × 30 days)")
    print(f"Final: Alpha180 + {len(current_talib)} TALib + {len(current_macro)} macro")
    print(f"Final d_feat: {6 + len(current_talib) + len(current_macro)}")
    print(f"Total features: {(6 + len(current_talib) + len(current_macro)) * 30}")
    print(f"Baseline IC: {baseline_ic:.4f}")
    print(f"Final IC:    {current_ic:.4f} ({'+' if current_ic >= baseline_ic else ''}{current_ic - baseline_ic:.4f})")

    print(f"\nFinal TALib Features ({len(current_talib)}):")
    for name in sorted(current_talib.keys()):
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
            'method': 'nested_cv_tcn_forward_selection',
            'baseline': 'alpha180',
            'final_talib_count': len(current_talib),
            'final_macro_count': len(current_macro),
            'final_d_feat': 6 + len(current_talib) + len(current_macro),
            'baseline_ic': baseline_ic,
            'final_ic': current_ic,
            'final_talib_features': current_talib,
            'final_macro_features': current_macro,
            'excluded_features': list(excluded_features),
            'history': history,
        }

        result_file = output_dir / f"forward_selection_tcn_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {result_file}")

    return current_talib, current_macro, history, excluded_features


def main():
    parser = argparse.ArgumentParser(description='TCN Nested CV Forward Feature Selection')
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'])
    parser.add_argument('--nday', type=int, default=5)
    parser.add_argument('--max-features', type=int, default=30,
                        help='Maximum number of additional features (TALib + macro)')
    parser.add_argument('--min-improvement', type=float, default=0.0005,
                        help='Minimum IC improvement to add a feature')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--early-stop', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--no-countdown', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint file')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from a specific result file')

    args = parser.parse_args()

    # GPU 设置
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

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
    baseline_talib = {}
    baseline_macro = []

    if args.resume_from:
        # 从指定文件恢复
        resume_path = Path(args.resume_from)
        if not resume_path.is_absolute():
            resume_path = output_dir / resume_path

        print(f"\n[*] Resuming from: {resume_path}")
        with open(resume_path, 'r') as f:
            checkpoint = json.load(f)

        if 'current_talib_features' in checkpoint:
            baseline_talib = checkpoint['current_talib_features']
            baseline_macro = checkpoint['current_macro_features']
        else:
            baseline_talib = checkpoint.get('final_talib_features', {})
            baseline_macro = checkpoint.get('final_macro_features', [])

        excluded_features = set(checkpoint.get('excluded_features', []))
        print(f"    Features: {len(baseline_talib)} TALib + {len(baseline_macro)} macro")
        print(f"    Excluded: {len(excluded_features)}")

    elif args.resume:
        # 从默认 checkpoint 恢复
        checkpoint_file = output_dir / "forward_selection_tcn_checkpoint.json"
        if checkpoint_file.exists():
            print(f"\n[*] Resuming from checkpoint")
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)

            baseline_talib = checkpoint['current_talib_features']
            baseline_macro = checkpoint['current_macro_features']
            excluded_features = set(checkpoint.get('excluded_features', []))
            print(f"    Features: {len(baseline_talib)} TALib + {len(baseline_macro)} macro")
            print(f"    Excluded: {len(excluded_features)}")
        else:
            print("No checkpoint file found, starting fresh")

    # 获取候选特征（不在 baseline 中的特征）
    print("\n[*] Preparing candidate features...")

    candidate_talib = {k: v for k, v in ALL_TALIB_FEATURES.items()
                       if k not in baseline_talib}
    candidate_macro = [m for m in ALL_MACRO_FEATURES
                       if m not in baseline_macro]

    # 验证候选特征
    print(f"    Validating {len(candidate_talib)} TALib candidates...")
    candidate_talib = validate_talib_features(symbols, candidate_talib)
    print(f"    Valid: {len(candidate_talib)}")

    print(f"    Validating {len(candidate_macro)} macro candidates...")
    candidate_macro = validate_macro_features(candidate_macro)
    print(f"    Valid: {len(candidate_macro)}")

    # TCN 参数
    params = dict(DEFAULT_TCN_PARAMS)
    params['batch_size'] = args.batch_size

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
    final_talib, final_macro, history, final_excluded = run_forward_selection(
        symbols,
        candidate_talib,
        candidate_macro,
        nday=args.nday,
        max_features=args.max_features,
        min_improvement=args.min_improvement,
        epochs=args.epochs,
        early_stop=args.early_stop,
        params=params,
        output_dir=output_dir,
        excluded_features=excluded_features,
        baseline_talib=baseline_talib,
        baseline_macro=baseline_macro,
        device=device,
    )

    print(f"\n[+] Forward selection complete")
    print(f"  Final d_feat: {6 + len(final_talib) + len(final_macro)}")
    print(f"  Excluded features: {len(final_excluded)}")


if __name__ == "__main__":
    main()
