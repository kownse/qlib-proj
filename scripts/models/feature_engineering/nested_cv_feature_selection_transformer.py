"""
Transformer 嵌套交叉验证 Forward Selection - 基于 Alpha300 逐步添加 Macro 和 TALib 特征

设计原理:
- 从 Alpha300 (5 OHLCV × 60 days = 300 features) 作为基线
- 逐个测试候选特征（macro 和 TALib），每个特征加入过去60天的值
- 如果某特征加入后导致 IC 变差，则加入排除列表并立即保存
- 一轮中所有能提升 IC 的特征都会被加入下一轮的特征集

内层CV (只用2000-2024年数据):
  Fold 1: train 2000-2020, valid 2021
  Fold 2: train 2000-2021, valid 2022
  Fold 3: train 2000-2022, valid 2023
  Fold 4: train 2000-2023, valid 2024

使用方法:
    python scripts/models/feature_engineering/nested_cv_feature_selection_transformer.py --stock-pool sp500
    python scripts/models/feature_engineering/nested_cv_feature_selection_transformer.py --resume
    python scripts/models/feature_engineering/nested_cv_feature_selection_transformer.py --resume-from forward_selection_transformer_xxx.json
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

from utils.talib_ops import TALIB_OPS
from data.stock_pools import STOCK_POOLS

# 导入自定义 Transformer 模型
from models.deep.transformer_model import TransformerModel, TransformerNet

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
    save_checkpoint,
    save_final_result,
    ForwardSelectionBase,
    add_common_args,
    countdown,
)


# ============================================================================
# Transformer 模型配置
# ============================================================================

DEFAULT_TRANSFORMER_PARAMS = {
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "lr": 1e-4,
    "weight_decay": 1e-3,
    "batch_size": 2048,
}

# Alpha300 配置: 5 features × 60 timesteps = 300
ALPHA300_SEQ_LEN = 60
ALPHA300_BASE_FEATURES = 5  # CLOSE, OPEN, HIGH, LOW, VOLUME (no VWAP)


# ============================================================================
# 数据归一化
# ============================================================================

def normalize_data(X, fit_stats=None):
    """
    对数据进行归一化处理：3σ clip + zscore
    """
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X)

    X_df = X_df.fillna(0)
    X_df = X_df.replace([np.inf, -np.inf], 0)

    if fit_stats is None:
        means = X_df.mean().values
        stds = X_df.std().values
        stds = np.where(stds == 0, 1, stds)
    else:
        means, stds = fit_stats

    X_values = X_df.values.copy()
    for i in range(X_values.shape[1]):
        col_mean = means[i]
        col_std = stds[i]
        if col_std > 0:
            lower = col_mean - 3 * col_std
            upper = col_mean + 3 * col_std
            X_values[:, i] = np.clip(X_values[:, i], lower, upper)
            X_values[:, i] = (X_values[:, i] - col_mean) / col_std

    X_values = np.nan_to_num(X_values, nan=0.0, posinf=0.0, neginf=0.0)

    return X_values, (means, stds)


# ============================================================================
# Dynamic Alpha300 Handler with Incremental Features
# ============================================================================

class DynamicAlpha300Handler(DataHandlerLP):
    """
    动态 Alpha300 Handler，支持增量添加 Macro 和 TALib 特征。
    """

    def __init__(
        self,
        talib_features: Dict[str, str] = None,
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
        self.seq_len = ALPHA300_SEQ_LEN

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
        """获取 Alpha300 基线特征 (5 OHLCV × 60 days)"""
        fields = []
        names = []

        # CLOSE
        for i in range(self.seq_len - 1, 0, -1):
            fields.append(f"Ref($close, {i})/$close")
            names.append(f"CLOSE{i}")
        fields.append("$close/$close")
        names.append("CLOSE0")

        # OPEN
        for i in range(self.seq_len - 1, 0, -1):
            fields.append(f"Ref($open, {i})/$close")
            names.append(f"OPEN{i}")
        fields.append("$open/$close")
        names.append("OPEN0")

        # HIGH
        for i in range(self.seq_len - 1, 0, -1):
            fields.append(f"Ref($high, {i})/$close")
            names.append(f"HIGH{i}")
        fields.append("$high/$close")
        names.append("HIGH0")

        # LOW
        for i in range(self.seq_len - 1, 0, -1):
            fields.append(f"Ref($low, {i})/$close")
            names.append(f"LOW{i}")
        fields.append("$low/$close")
        names.append("LOW0")

        # VOLUME
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
            for i in range(self.seq_len - 1, 0, -1):
                ref_expr = f"Ref({expr}, {i})"
                fields.append(ref_expr)
                names.append(f"{feat_name}_{i}")
            fields.append(expr)
            names.append(f"{feat_name}_0")

        return fields, names

    def _get_feature_config(self):
        """获取完整特征配置"""
        fields, names = self._get_alpha300_features()
        talib_fields, talib_names = self._get_talib_features_config()
        fields.extend(talib_fields)
        names.extend(talib_names)
        return fields, names

    def _get_label_config(self):
        """返回N天收益标签"""
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
                for i in range(self.seq_len - 1, -1, -1):
                    col_name = f"{col}_{i}"
                    shifted = base_series.shift(i + 1)
                    aligned_values = shifted.reindex(main_datetimes).values
                    if has_multi_columns:
                        macro_data[('feature', col_name)] = aligned_values
                    else:
                        macro_data[col_name] = aligned_values

            macro_df = pd.DataFrame(macro_data, index=df.index)
            merged = pd.concat([df, macro_df], axis=1, copy=False)
            setattr(self, attr, merged.copy())


# ============================================================================
# Transformer Forward Selection 实现
# ============================================================================

class TransformerForwardSelection(ForwardSelectionBase):
    """Transformer 模型的 Forward Selection 实现"""

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
            checkpoint_name="forward_selection_transformer_checkpoint",
            result_prefix="forward_selection_transformer",
        )

        self.current_talib = dict(baseline_talib)
        self.current_macro = list(baseline_macro)
        self.candidate_talib = candidate_talib
        self.candidate_macro = candidate_macro

        self.epochs = epochs
        self.early_stop = early_stop
        self.params = params or DEFAULT_TRANSFORMER_PARAMS
        self.gpu = gpu

        # 设置设备
        if gpu >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu}')
        else:
            self.device = torch.device('cpu')

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

        # 归一化
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
        """使用指定特征集评估"""
        fold_ics = []

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
                d_feat = total_features // ALPHA300_SEQ_LEN

                # 创建 Transformer 模型
                model = TransformerNet(
                    d_feat=d_feat,
                    d_model=self.params['d_model'],
                    nhead=self.params['nhead'],
                    num_layers=self.params['num_layers'],
                    dim_feedforward=self.params['dim_feedforward'],
                    dropout=self.params['dropout'],
                    seq_len=ALPHA300_SEQ_LEN,
                ).to(self.device)

                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=self.params['lr'],
                    weight_decay=self.params['weight_decay']
                )

                batch_size = self.params['batch_size']
                best_loss = float('inf')
                best_state = None
                stop_steps = 0

                # 训练循环
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

                        pred = model(feature)
                        loss = torch.mean((pred - label) ** 2)

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                    # 验证
                    model.eval()
                    with torch.no_grad():
                        val_preds = []
                        for i in range(0, len(X_valid), batch_size):
                            end = min(i + batch_size, len(X_valid))
                            feature = torch.from_numpy(X_valid[i:end]).float().to(self.device)
                            pred = model(feature)
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

                # 加载最佳模型并计算 IC
                if best_state is not None:
                    model.load_state_dict(best_state)

                model.eval()
                with torch.no_grad():
                    val_preds = []
                    for i in range(0, len(X_valid), batch_size):
                        end = min(i + batch_size, len(X_valid))
                        feature = torch.from_numpy(X_valid[i:end]).float().to(self.device)
                        pred = model(feature)
                        val_preds.append(pred.cpu().numpy())
                    val_pred = np.concatenate(val_preds)

                ic = compute_ic(val_pred, y_valid, valid_index)
                fold_ics.append(ic)

                del X_train, y_train, X_valid, y_valid, valid_index, model
                gc.collect()

        finally:
            self.current_talib = orig_talib
            self.current_macro = orig_macro

        return np.mean(fold_ics), fold_ics

    def get_feature_counts(self) -> Dict[str, int]:
        return {
            'base': ALPHA300_BASE_FEATURES,
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
        """打印最终结果"""
        print("\n" + "=" * 70)
        print("TRANSFORMER FORWARD SELECTION COMPLETE")
        print("=" * 70)
        d_feat = ALPHA300_BASE_FEATURES + len(self.current_talib) + len(self.current_macro)
        total_features = d_feat * ALPHA300_SEQ_LEN
        print(f"Baseline: Alpha300 ({ALPHA300_BASE_FEATURES} features × {ALPHA300_SEQ_LEN} days)")
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
    parser = argparse.ArgumentParser(description='Transformer Nested CV Forward Feature Selection')
    add_common_args(parser)

    # Transformer 特有参数
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dim-feedforward', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)

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
        checkpoint_file = output_dir / "forward_selection_transformer_checkpoint.json"
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

    # 获取候选特征
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

    # Transformer 参数
    params = dict(DEFAULT_TRANSFORMER_PARAMS)
    params['d_model'] = args.d_model
    params['nhead'] = args.nhead
    params['num_layers'] = args.num_layers
    params['dim_feedforward'] = args.dim_feedforward
    params['dropout'] = args.dropout
    params['lr'] = args.lr
    params['weight_decay'] = args.weight_decay
    params['batch_size'] = args.batch_size

    print(f"\n[*] Transformer config:")
    print(f"    d_model: {params['d_model']}")
    print(f"    nhead: {params['nhead']}")
    print(f"    num_layers: {params['num_layers']}")
    print(f"    dim_feedforward: {params['dim_feedforward']}")
    print(f"    dropout: {params['dropout']}")
    print(f"    lr: {params['lr']}")
    print(f"    epochs: {args.epochs}")
    print(f"    early_stop: {args.early_stop}")

    # 倒计时
    if not args.no_countdown:
        if not countdown(3):
            return

    # 创建 Forward Selection 实例并运行
    selector = TransformerForwardSelection(
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
        method_name='nested_cv_transformer_forward_selection',
    )

    print(f"\n[+] Forward selection complete")
    d_feat = ALPHA300_BASE_FEATURES + len(final_features.get('current_talib_features', {})) + len(final_features.get('current_macro_features', []))
    print(f"  Final d_feat: {d_feat}")
    print(f"  Total features: {d_feat * ALPHA300_SEQ_LEN}")
    print(f"  Excluded features: {len(final_excluded)}")


if __name__ == "__main__":
    main()
