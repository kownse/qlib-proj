"""
TCN 嵌套交叉验证 Forward Selection - 基于 Alpha300 逐步添加 Macro 和 TALib 特征

设计原理:
- 从 Alpha300 (5 OHLCV × 60 days = 300 features) 作为基线 (去掉了全是NaN的VWAP)
- 逐个测试候选特征（macro 和 TALib），每个特征加入过去60天的值
- 如果某特征加入后导致 IC 变差，则加入排除列表
- 后续轮次跳过被排除的特征

内层CV (只用2000-2024年数据):
  Fold 1: train 2000-2020, valid 2021
  Fold 2: train 2000-2021, valid 2022
  Fold 3: train 2000-2022, valid 2023
  Fold 4: train 2000-2023, valid 2024

使用方法:
    python scripts/models/feature_engineering/nested_cv_feature_selection_tcn.py --stock-pool sp500
    python scripts/models/feature_engineering/nested_cv_feature_selection_tcn.py --resume
    python scripts/models/feature_engineering/nested_cv_feature_selection_tcn.py --resume-from forward_selection_tcn_xxx.json
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
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

import torch

from qlib.contrib.model.pytorch_tcn import TCN

from utils.talib_ops import TALIB_OPS
from data.stock_pools import STOCK_POOLS

# 导入共享工具
from models.feature_engineering.feature_selection_utils import (
    INNER_CV_FOLDS,
    ALL_TALIB_FEATURES,
    ALL_MACRO_FEATURES,
    PROJECT_ROOT,
    compute_ic,
    validate_qlib_features,
    validate_macro_features,
    load_macro_data,
    load_checkpoint,
    ForwardSelectionBase,
    add_common_args,
    countdown,
)


# ============================================================================
# TCN 模型配置
# ============================================================================

# TCN 默认参数（和 run_tcn.py 保持一致）
DEFAULT_TCN_PARAMS = {
    "n_chans": 128,      # qlib TCN 默认值
    "num_layers": 5,     # qlib TCN 默认值
    "kernel_size": 5,    # qlib TCN 默认值
    "dropout": 0.5,      # qlib TCN 默认值
    "lr": 0.0001,        # qlib TCN 默认值
    "batch_size": 2000,  # qlib TCN 默认值
}

# Alpha300 配置: 5 features × 60 timesteps = 300 (去掉了全是NaN的VWAP)
ALPHA300_SEQ_LEN = 60
ALPHA300_BASE_FEATURES = 5  # CLOSE, OPEN, HIGH, LOW, VOLUME (no VWAP)


# ============================================================================
# 数据归一化 (借鉴 run_tcn.py)
# ============================================================================

def normalize_data(X, fit_stats=None):
    """
    对数据进行归一化处理：3σ clip + zscore

    借鉴自 run_tcn.py 的数据处理方式。

    Args:
        X: numpy array 或 DataFrame
        fit_stats: 训练集统计量 (mean, std)，如果为 None 则从 X 计算

    Returns:
        normalized_X: 归一化后的数据 (numpy array)
        stats: 统计量 (mean, std)，可用于测试数据
    """
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X)

    # 处理 NaN 和 inf
    X_df = X_df.fillna(0)
    X_df = X_df.replace([np.inf, -np.inf], 0)

    if fit_stats is None:
        # 计算统计量
        means = X_df.mean().values
        stds = X_df.std().values
        stds = np.where(stds == 0, 1, stds)  # 避免除以0
    else:
        means, stds = fit_stats

    # 按列进行 3σ clip + zscore 归一化
    X_values = X_df.values.copy()
    for i in range(X_values.shape[1]):
        col_mean = means[i]
        col_std = stds[i]
        if col_std > 0:
            lower = col_mean - 3 * col_std
            upper = col_mean + 3 * col_std
            X_values[:, i] = np.clip(X_values[:, i], lower, upper)
            X_values[:, i] = (X_values[:, i] - col_mean) / col_std

    # 最终处理
    X_values = np.nan_to_num(X_values, nan=0.0, posinf=0.0, neginf=0.0)

    return X_values, (means, stds)


def print_data_quality(X, name="Data"):
    """打印数据质量诊断信息"""
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"      {name} shape: {X.shape}")
    print(f"      {name} NaN: {nan_count} ({nan_count / X.size * 100:.2f}%)")
    valid_values = X[~np.isnan(X) & ~np.isinf(X)]
    if len(valid_values) > 0:
        print(f"      {name} range: [{valid_values.min():.4f}, {valid_values.max():.4e}]")


# ============================================================================
# Dynamic Alpha300 Handler with Incremental Features
# ============================================================================

class DynamicAlpha300Handler(DataHandlerLP):
    """
    动态 Alpha300 Handler，支持增量添加 Macro 和 TALib 特征。

    基线: Alpha300 (5 OHLCV × 60 days = 300 features)
    - 去掉了全是 NaN 的 VWAP (US data 没有 VWAP)
    - 使用 60 天历史而不是 30 天
    增量特征: 每个 macro/talib 特征扩展为 60 天的历史
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
        self.seq_len = ALPHA300_SEQ_LEN  # 60 天

        self._macro_df = load_macro_data() if self.macro_features else None

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

    def _get_alpha300_features(self):
        """获取 Alpha300 基线特征 (5 OHLCV × 60 days, 去掉 VWAP)"""
        fields = []
        names = []

        # CLOSE: 60 天历史收盘价
        for i in range(self.seq_len - 1, 0, -1):
            fields.append(f"Ref($close, {i})/$close")
            names.append(f"CLOSE{i}")
        fields.append("$close/$close")
        names.append("CLOSE0")

        # OPEN: 60 天历史开盘价
        for i in range(self.seq_len - 1, 0, -1):
            fields.append(f"Ref($open, {i})/$close")
            names.append(f"OPEN{i}")
        fields.append("$open/$close")
        names.append("OPEN0")

        # HIGH: 60 天历史最高价
        for i in range(self.seq_len - 1, 0, -1):
            fields.append(f"Ref($high, {i})/$close")
            names.append(f"HIGH{i}")
        fields.append("$high/$close")
        names.append("HIGH0")

        # LOW: 60 天历史最低价
        for i in range(self.seq_len - 1, 0, -1):
            fields.append(f"Ref($low, {i})/$close")
            names.append(f"LOW{i}")
        fields.append("$low/$close")
        names.append("LOW0")

        # 注意: 去掉 VWAP (US data 中全是 NaN)

        # VOLUME: 60 天历史成交量
        for i in range(self.seq_len - 1, 0, -1):
            fields.append(f"Ref($volume, {i})/($volume+1e-12)")
            names.append(f"VOLUME{i}")
        fields.append("$volume/($volume+1e-12)")
        names.append("VOLUME0")

        return fields, names

    def _get_talib_features_config(self):
        """获取 TALib 特征配置 (每个特征扩展为60天)"""
        fields = []
        names = []

        for feat_name, expr in self.talib_features.items():
            # 扩展为60天历史
            for i in range(self.seq_len - 1, 0, -1):
                ref_expr = f"Ref({expr}, {i})"
                fields.append(ref_expr)
                names.append(f"{feat_name}_{i}")
            # 当天的值
            fields.append(expr)
            names.append(f"{feat_name}_0")

        return fields, names

    def _get_feature_config(self):
        """获取完整特征配置"""
        # Alpha300 基线
        fields, names = self._get_alpha300_features()

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
        """添加时间对齐的 macro 特征 (每个扩展为60天)"""
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
                # 扩展为60天历史
                for i in range(self.seq_len - 1, -1, -1):
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


# ============================================================================
# TCN Forward Selection 实现
# ============================================================================

class TCNForwardSelection(ForwardSelectionBase):
    """TCN 模型的 Forward Selection 实现"""

    def __init__(
        self,
        symbols: List[str],
        baseline_talib: Dict[str, str],
        baseline_macro: List[str],
        candidate_talib: Dict[str, str],
        candidate_macro: List[str],
        nday: int = 5,
        max_features: int = 30,
        min_improvement: float = 0.0005,
        epochs: int = 20,
        early_stop: int = 8,
        params: dict = None,
        output_dir: Path = None,
        gpu: int = 0,
    ):
        super().__init__(
            symbols=symbols,
            nday=nday,
            max_features=max_features,
            min_improvement=min_improvement,
            output_dir=output_dir,
            checkpoint_name="forward_selection_tcn_checkpoint",
            result_prefix="forward_selection_tcn",
        )

        self.current_talib = dict(baseline_talib)
        self.current_macro = list(baseline_macro)
        self.candidate_talib = candidate_talib
        self.candidate_macro = candidate_macro

        self.epochs = epochs
        self.early_stop = early_stop
        self.params = params or DEFAULT_TCN_PARAMS
        self.gpu = gpu  # GPU ID for qlib TCN

    def prepare_fold_data(self, fold_config: Dict) -> Tuple:
        """准备单个fold的数据"""
        handler = DynamicAlpha300Handler(
            talib_features=self.current_talib,
            macro_features=self.current_macro,
            volatility_window=self.nday,
            instruments=self.symbols,
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

        # 获取原始数据
        X_train_raw = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)

        y_train = dataset.prepare("train", col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0]
        y_train = y_train.fillna(0).values

        X_valid_raw = dataset.prepare("valid", col_set="feature", data_key=DataHandlerLP.DK_L)
        valid_index = X_valid_raw.index

        y_valid = dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(y_valid, pd.DataFrame):
            y_valid = y_valid.iloc[:, 0]
        y_valid = y_valid.fillna(0).values

        # 使用 3σ clip + zscore 归一化 (借鉴 run_tcn.py)
        X_train, train_stats = normalize_data(X_train_raw)
        X_valid, _ = normalize_data(X_valid_raw, fit_stats=train_stats)

        return X_train, y_train, X_valid, y_valid, valid_index

    def evaluate_feature_set(self) -> Tuple[float, List[float]]:
        """在内层CV上评估特征集"""
        return self._evaluate_with_features(self.current_talib, self.current_macro)

    def _evaluate_with_features(
        self,
        talib_features: Dict[str, str],
        macro_features: List[str],
    ) -> Tuple[float, List[float]]:
        """使用指定特征集评估（使用 qlib TCN 实现）"""
        fold_ics = []

        # 临时保存当前特征
        orig_talib = self.current_talib
        orig_macro = self.current_macro
        self.current_talib = talib_features
        self.current_macro = macro_features

        try:
            for fold in INNER_CV_FOLDS:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                X_train, y_train, X_valid, y_valid, valid_index = self.prepare_fold_data(fold)

                total_features = X_train.shape[1]
                # d_feat = number of features per timestep (total / 60 timesteps)
                d_feat = total_features // ALPHA300_SEQ_LEN

                # 使用 qlib 的 TCN 实现（和 run_tcn.py 一样）
                gpu_id = self.gpu if torch.cuda.is_available() else -1
                model = TCN(
                    d_feat=d_feat,
                    n_chans=self.params['n_chans'],
                    kernel_size=self.params['kernel_size'],
                    num_layers=self.params['num_layers'],
                    dropout=self.params['dropout'],
                    n_epochs=self.epochs,
                    lr=self.params['lr'],
                    early_stop=self.early_stop,
                    batch_size=self.params['batch_size'],
                    metric="loss",
                    loss="mse",
                    GPU=gpu_id,
                )

                batch_size = self.params['batch_size']

                # 使用 qlib TCN 内部模型进行训练（手动训练循环以使用归一化后的数据）
                tcn_model = model.tcn_model
                optimizer = model.train_optimizer

                best_loss = float('inf')
                best_state = None
                stop_steps = 0

                for epoch in range(self.epochs):
                    tcn_model.train()
                    indices = np.arange(len(X_train))
                    np.random.shuffle(indices)

                    for i in range(0, len(indices), batch_size):
                        if len(indices) - i < batch_size:
                            break
                        batch_idx = indices[i:i + batch_size]
                        feature = torch.from_numpy(X_train[batch_idx]).float().to(model.device)
                        label = torch.from_numpy(y_train[batch_idx]).float().to(model.device)

                        pred = tcn_model(feature).squeeze()
                        loss = torch.mean((pred - label) ** 2)

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_value_(tcn_model.parameters(), 3.0)
                        optimizer.step()

                    # Validation
                    tcn_model.eval()
                    with torch.no_grad():
                        val_preds = []
                        for i in range(0, len(X_valid), batch_size):
                            end = min(i + batch_size, len(X_valid))
                            feature = torch.from_numpy(X_valid[i:end]).float().to(model.device)
                            pred = tcn_model(feature).squeeze()
                            val_preds.append(pred.cpu().numpy())
                        val_pred = np.concatenate(val_preds)
                        val_loss = np.mean((val_pred - y_valid) ** 2)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_state = {k: v.cpu().clone() for k, v in tcn_model.state_dict().items()}
                        stop_steps = 0
                    else:
                        stop_steps += 1
                        if stop_steps >= self.early_stop:
                            break

                # Load best model and compute IC
                if best_state is not None:
                    tcn_model.load_state_dict(best_state)

                tcn_model.eval()
                with torch.no_grad():
                    val_preds = []
                    for i in range(0, len(X_valid), batch_size):
                        end = min(i + batch_size, len(X_valid))
                        feature = torch.from_numpy(X_valid[i:end]).float().to(model.device)
                        pred = tcn_model(feature).squeeze()
                        val_preds.append(pred.cpu().numpy())
                    val_pred = np.concatenate(val_preds)

                ic = compute_ic(val_pred, y_valid, valid_index)
                fold_ics.append(ic)

                del X_train, y_train, X_valid, y_valid, valid_index, model, tcn_model
                gc.collect()

        finally:
            # 恢复原特征
            self.current_talib = orig_talib
            self.current_macro = orig_macro

        return np.mean(fold_ics), fold_ics

    def get_feature_counts(self) -> Dict[str, int]:
        return {
            'base': ALPHA300_BASE_FEATURES,  # Alpha300: 5 OHLCV features (no VWAP)
            'talib': len(self.current_talib),
            'macro': len(self.current_macro),
        }

    def add_feature(self, name: str, feature_type: str, expr: str = None):
        if feature_type == 'feature':
            self.current_talib[name] = expr
        else:
            self.current_macro.append(name)

    def get_current_features_dict(self) -> Dict[str, Any]:
        return {
            'current_talib_features': self.current_talib,
            'current_macro_features': self.current_macro,
        }

    def get_testable_candidates(self) -> Tuple[Dict[str, str], List[str]]:
        testable_talib = {k: v for k, v in self.candidate_talib.items()
                         if k not in self.current_talib and k not in self.excluded_features}
        testable_macro = [m for m in self.candidate_macro
                         if m not in self.current_macro and m not in self.excluded_features]
        return testable_talib, testable_macro

    def test_feature(
        self, name: str, feature_type: str, expr: str = None
    ) -> Tuple[float, List[float]]:
        if feature_type == 'feature':
            test_talib = dict(self.current_talib)
            test_talib[name] = expr
            return self._evaluate_with_features(test_talib, self.current_macro)
        else:
            test_macro = self.current_macro + [name]
            return self._evaluate_with_features(self.current_talib, test_macro)

    def cleanup_after_evaluation(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _print_final_result(self):
        """打印最终结果（覆盖基类方法，添加 TCN 特有信息）"""
        print("\n" + "=" * 70)
        print("TCN FORWARD SELECTION COMPLETE")
        print("=" * 70)
        d_feat = ALPHA300_BASE_FEATURES + len(self.current_talib) + len(self.current_macro)
        total_features = d_feat * ALPHA300_SEQ_LEN
        print(f"Baseline: Alpha300 ({ALPHA300_BASE_FEATURES} features × {ALPHA300_SEQ_LEN} days, no VWAP)")
        print(f"Final: Alpha300 + {len(self.current_talib)} TALib + {len(self.current_macro)} macro")
        print(f"Final d_feat: {d_feat}")
        print(f"Total features: {total_features}")
        print(f"Baseline IC: {self.baseline_ic:.4f}")
        ic_diff = self.current_ic - self.baseline_ic
        print(f"Final IC:    {self.current_ic:.4f} ({'+' if ic_diff >= 0 else ''}{ic_diff:.4f})")

        print(f"\nFinal TALib Features ({len(self.current_talib)}):")
        for name in sorted(self.current_talib.keys()):
            print(f"  - {name}")

        print(f"\nFinal Macro Features ({len(self.current_macro)}):")
        for name in self.current_macro:
            print(f"  - {name}")

        print(f"\nExcluded Features ({len(self.excluded_features)}):")
        for name in sorted(self.excluded_features):
            print(f"  - {name}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='TCN Nested CV Forward Feature Selection')
    add_common_args(parser)

    args = parser.parse_args()

    # GPU 设置
    if args.gpu >= 0 and torch.cuda.is_available():
        print(f"Using GPU: cuda:{args.gpu}")
    else:
        print("Using CPU")

    # Qlib 初始化
    qlib_data_path = PROJECT_ROOT / "my_data" / "qlib_us"
    qlib.init(
        provider_uri=str(qlib_data_path),
        region=REG_US,
        custom_ops=TALIB_OPS,
    )

    symbols = STOCK_POOLS[args.stock_pool]
    output_dir = PROJECT_ROOT / "outputs" / "feature_selection"

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
        checkpoint = load_checkpoint(resume_path)

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
            checkpoint = load_checkpoint(checkpoint_file)

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
    candidate_talib = validate_qlib_features(symbols, candidate_talib)
    print(f"    Valid: {len(candidate_talib)}")

    print(f"    Validating {len(candidate_macro)} macro candidates...")
    candidate_macro = validate_macro_features(candidate_macro)
    print(f"    Valid: {len(candidate_macro)}")

    # TCN 参数
    params = dict(DEFAULT_TCN_PARAMS)
    params['batch_size'] = args.batch_size

    # 倒计时
    if not args.no_countdown:
        if not countdown(3):
            return

    # 创建 Forward Selection 实例并运行
    selector = TCNForwardSelection(
        symbols=symbols,
        baseline_talib=baseline_talib,
        baseline_macro=baseline_macro,
        candidate_talib=candidate_talib,
        candidate_macro=candidate_macro,
        nday=args.nday,
        max_features=args.max_features,
        min_improvement=args.min_improvement,
        epochs=args.epochs,
        early_stop=args.early_stop,
        params=params,
        output_dir=output_dir,
        gpu=args.gpu,
    )

    final_features, history, final_excluded = selector.run(
        excluded_features=excluded_features,
        method_name='nested_cv_tcn_forward_selection',
    )

    print(f"\n[+] Forward selection complete")
    d_feat = ALPHA300_BASE_FEATURES + len(final_features.get('current_talib_features', {})) + len(final_features.get('current_macro_features', []))
    print(f"  Final d_feat: {d_feat}")
    print(f"  Total features: {d_feat * ALPHA300_SEQ_LEN}")
    print(f"  Excluded features: {len(final_excluded)}")


if __name__ == "__main__":
    main()
