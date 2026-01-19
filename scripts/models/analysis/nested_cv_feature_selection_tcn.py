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
from typing import Dict, List, Tuple, Any

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

# 导入共享工具
from models.analysis.feature_selection_utils import (
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
        device=None,
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
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_fold_data(self, fold_config: Dict) -> Tuple:
        """准备单个fold的数据"""
        handler = DynamicAlpha180Handler(
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

    def evaluate_feature_set(self) -> Tuple[float, List[float]]:
        """在内层CV上评估特征集"""
        return self._evaluate_with_features(self.current_talib, self.current_macro)

    def _evaluate_with_features(
        self,
        talib_features: Dict[str, str],
        macro_features: List[str],
    ) -> Tuple[float, List[float]]:
        """使用指定特征集评估"""
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
                # d_feat = number of features per timestep (total / 30 timesteps)
                d_feat = total_features // 30

                model = create_tcn_model(
                    d_feat=d_feat,
                    n_chans=self.params['n_chans'],
                    num_layers=self.params['num_layers'],
                    kernel_size=self.params['kernel_size'],
                    dropout=self.params['dropout'],
                    device=self.device,
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=self.params['lr'])
                batch_size = self.params['batch_size']

                # Training loop
                best_loss = float('inf')
                best_state = None
                stop_steps = 0

                for epoch in range(self.epochs):
                    model.train()
                    indices = np.arange(len(X_train))
                    np.random.shuffle(indices)

                    for i in range(0, len(indices), batch_size):
                        if len(indices) - i < batch_size:
                            break
                        batch_idx = indices[i:i + batch_size]
                        feature = torch.from_numpy(X_train[batch_idx]).float().to(self.device)
                        label = torch.from_numpy(y_train[batch_idx]).float().to(self.device)

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
                            feature = torch.from_numpy(X_valid[i:end]).float().to(self.device)
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
                        if stop_steps >= self.early_stop:
                            break

                # Load best model and compute IC
                if best_state is not None:
                    model.load_state_dict(best_state)

                model.eval()
                with torch.no_grad():
                    val_preds = []
                    for i in range(0, len(X_valid), batch_size):
                        end = min(i + batch_size, len(X_valid))
                        feature = torch.from_numpy(X_valid[i:end]).float().to(self.device)
                        pred = model(feature).squeeze()
                        val_preds.append(pred.cpu().numpy())
                    val_pred = np.concatenate(val_preds)

                ic = compute_ic(val_pred, y_valid, valid_index)
                fold_ics.append(ic)

                del X_train, y_train, X_valid, y_valid, valid_index, model
                gc.collect()

        finally:
            # 恢复原特征
            self.current_talib = orig_talib
            self.current_macro = orig_macro

        return np.mean(fold_ics), fold_ics

    def get_feature_counts(self) -> Dict[str, int]:
        return {
            'base': 6,  # Alpha180 OHLCV features
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
        print(f"Baseline: Alpha180 (6 features × 30 days)")
        print(f"Final: Alpha180 + {len(self.current_talib)} TALib + {len(self.current_macro)} macro")
        print(f"Final d_feat: {6 + len(self.current_talib) + len(self.current_macro)}")
        print(f"Total features: {(6 + len(self.current_talib) + len(self.current_macro)) * 30}")
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
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

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
        device=device,
    )

    final_features, history, final_excluded = selector.run(
        excluded_features=excluded_features,
        method_name='nested_cv_tcn_forward_selection',
    )

    print(f"\n[+] Forward selection complete")
    print(f"  Final d_feat: {6 + len(final_features.get('current_talib_features', {})) + len(final_features.get('current_macro_features', []))}")
    print(f"  Excluded features: {len(final_excluded)}")


if __name__ == "__main__":
    main()
