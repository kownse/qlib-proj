"""
嵌套交叉验证 Forward Selection - 基于 AE-MLP 模型

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
    python scripts/models/feature_engineering/nested_cv_feature_selection.py --baseline backward_elimination_20260118_224446.json
    python scripts/models/feature_engineering/nested_cv_feature_selection.py --resume
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
from typing import Dict, List, Tuple, Set, Any

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

# 导入共享工具
from models.feature_engineering.feature_selection_utils import (
    INNER_CV_FOLDS,
    ALL_STOCK_FEATURES,
    ALL_MACRO_FEATURES,
    PROJECT_ROOT,
    compute_ic,
    validate_qlib_features,
    validate_macro_features,
    load_macro_data,
    save_checkpoint,
    load_checkpoint,
    save_final_result,
    prepare_dataset_data,
    ForwardSelectionBase,
    add_common_args,
    countdown,
)


# ============================================================================
# AE-MLP 超参数
# ============================================================================

BEST_HYPERPARAMS = {
    "hidden_units": [112, 64, 128, 224, 48],
    "dropout_rates": [0.05, 0.096, 0.096, 0.096, 0.096, 0.096, 0.096],
    "lr": 0.000534,
    "batch_size": 2048,
    "loss_weights": {"decoder": 0.267, "ae_action": 0.072, "action": 1.0}
}


# ============================================================================
# 动态特征 Handler
# ============================================================================

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


# ============================================================================
# AE-MLP Forward Selection 实现
# ============================================================================

class AEMLPForwardSelection(ForwardSelectionBase):
    """AE-MLP 模型的 Forward Selection 实现"""

    def __init__(
        self,
        symbols: List[str],
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
    ):
        super().__init__(
            symbols=symbols,
            nday=nday,
            max_features=max_features,
            min_improvement=min_improvement,
            output_dir=output_dir,
            checkpoint_name="forward_selection_checkpoint",
            result_prefix="forward_selection",
        )

        self.current_stock = dict(baseline_stock)
        self.current_macro = list(baseline_macro)
        self.candidate_stock = candidate_stock
        self.candidate_macro = candidate_macro

        self.epochs = epochs
        self.early_stop = early_stop
        self.batch_size = batch_size

    def prepare_fold_data(self, fold_config: Dict) -> Tuple:
        """准备单个fold的数据"""
        handler = DynamicFeatureHandler(
            feature_config=self.current_stock,
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

        X_train, y_train, _ = prepare_dataset_data(dataset, "train")
        X_valid, y_valid, valid_index = prepare_dataset_data(dataset, "valid")

        return X_train, y_train, X_valid, y_valid, valid_index

    def evaluate_feature_set(self) -> Tuple[float, List[float]]:
        """在内层CV上评估特征集"""
        return self._evaluate_with_features(self.current_stock, self.current_macro)

    def _evaluate_with_features(
        self,
        stock_features: Dict[str, str],
        macro_features: List[str],
    ) -> Tuple[float, List[float]]:
        """使用指定特征集评估"""
        fold_ics = []

        # 临时保存当前特征
        orig_stock = self.current_stock
        orig_macro = self.current_macro
        self.current_stock = stock_features
        self.current_macro = macro_features

        try:
            for fold in INNER_CV_FOLDS:
                tf.keras.backend.clear_session()
                gc.collect()

                X_train, y_train, X_valid, y_valid, valid_index = self.prepare_fold_data(fold)

                num_features = X_train.shape[1]
                model = build_ae_mlp_model({**BEST_HYPERPARAMS, 'num_columns': num_features})

                train_outputs = {'decoder': X_train, 'ae_action': y_train, 'action': y_train}
                valid_outputs = {'decoder': X_valid, 'ae_action': y_valid, 'action': y_valid}

                cb_list = [
                    callbacks.EarlyStopping(
                        monitor='val_action_loss',
                        patience=self.early_stop,
                        restore_best_weights=True,
                        verbose=0,
                        mode='min'
                    ),
                ]

                model.fit(
                    X_train, train_outputs,
                    validation_data=(X_valid, valid_outputs),
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    callbacks=cb_list,
                    verbose=0,
                )

                _, _, valid_pred = model.predict(X_valid, batch_size=self.batch_size, verbose=0)
                ic = compute_ic(valid_pred.flatten(), y_valid, valid_index)
                fold_ics.append(ic)

                del X_train, y_train, X_valid, y_valid, valid_index, model
                del train_outputs, valid_outputs
                gc.collect()

        finally:
            # 恢复原特征
            self.current_stock = orig_stock
            self.current_macro = orig_macro

        return np.mean(fold_ics), fold_ics

    def get_feature_counts(self) -> Dict[str, int]:
        return {
            'stock': len(self.current_stock),
            'macro': len(self.current_macro),
        }

    def add_feature(self, name: str, feature_type: str, expr: str = None):
        if feature_type == 'feature':
            self.current_stock[name] = expr
        else:
            self.current_macro.append(name)

    def get_current_features_dict(self) -> Dict[str, Any]:
        return {
            'current_stock_features': self.current_stock,
            'current_macro_features': self.current_macro,
        }

    def get_testable_candidates(self) -> Tuple[Dict[str, str], List[str]]:
        testable_stock = {k: v for k, v in self.candidate_stock.items()
                         if k not in self.current_stock and k not in self.excluded_features}
        testable_macro = [m for m in self.candidate_macro
                         if m not in self.current_macro and m not in self.excluded_features]
        return testable_stock, testable_macro

    def test_feature(
        self, name: str, feature_type: str, expr: str = None
    ) -> Tuple[float, List[float]]:
        if feature_type == 'feature':
            test_stock = dict(self.current_stock)
            test_stock[name] = expr
            return self._evaluate_with_features(test_stock, self.current_macro)
        else:
            test_macro = self.current_macro + [name]
            return self._evaluate_with_features(self.current_stock, test_macro)

    def cleanup_after_evaluation(self):
        tf.keras.backend.clear_session()
        gc.collect()


# ============================================================================
# 工具函数
# ============================================================================

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


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='AE-MLP Nested CV Forward Feature Selection')
    add_common_args(parser)
    parser.add_argument('--baseline', type=str, default=None,
                        help='Backward elimination result file to use as baseline')

    args = parser.parse_args()

    # GPU 设置
    gpus = tf.config.list_physical_devices('GPU')
    if args.gpu >= 0 and gpus:
        tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)

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
    baseline_stock = {}
    baseline_macro = []

    if args.resume_from:
        # 从指定文件恢复
        resume_path = Path(args.resume_from)
        if not resume_path.is_absolute():
            resume_path = output_dir / resume_path

        print(f"\n[*] Resuming from: {resume_path}")
        checkpoint = load_checkpoint(resume_path)

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
            checkpoint = load_checkpoint(checkpoint_file)

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
    candidate_stock = validate_qlib_features(symbols, candidate_stock)
    print(f"    Valid: {len(candidate_stock)}")

    print(f"    Validating {len(candidate_macro)} macro candidates...")
    candidate_macro = validate_macro_features(candidate_macro)
    print(f"    Valid: {len(candidate_macro)}")

    # 倒计时
    if not args.no_countdown:
        if not countdown(3):
            return

    # 创建 Forward Selection 实例并运行
    selector = AEMLPForwardSelection(
        symbols=symbols,
        baseline_stock=baseline_stock,
        baseline_macro=baseline_macro,
        candidate_stock=candidate_stock,
        candidate_macro=candidate_macro,
        nday=args.nday,
        max_features=args.max_features,
        min_improvement=args.min_improvement,
        epochs=args.epochs,
        early_stop=args.early_stop,
        batch_size=args.batch_size,
        output_dir=output_dir,
    )

    final_features, history, final_excluded = selector.run(
        excluded_features=excluded_features,
        method_name='nested_cv_forward_selection_ae_mlp',
    )

    print(f"\n[+] Forward selection complete")
    print(f"  Final features: {len(final_features.get('current_stock_features', {}))} stock + "
          f"{len(final_features.get('current_macro_features', []))} macro")
    print(f"  Excluded features: {len(final_excluded)}")


if __name__ == "__main__":
    main()
