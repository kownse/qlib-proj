"""
LightGBM 嵌套交叉验证 Forward Selection

设计原理:
- 从用户指定的初始特征集开始（基于 importance 分析）
- 逐个测试候选特征（stock 和 macro），如果某特征加入后导致 IC 变差则排除
- 后续轮次跳过被排除的特征
- 每轮只添加 1 个 IC 提升最大的特征（贪婪策略，与 CatBoost 版本一致）

内层CV (只用2000-2024年数据):
  Fold 1: train 2000-2020, valid 2021
  Fold 2: train 2000-2021, valid 2022
  Fold 3: train 2000-2022, valid 2023
  Fold 4: train 2000-2023, valid 2024

使用方法:
    python scripts/models/feature_engineering/nested_cv_feature_selection_lightgbm.py --stock-pool sp500
    python scripts/models/feature_engineering/nested_cv_feature_selection_lightgbm.py --resume
    python scripts/models/feature_engineering/nested_cv_feature_selection_lightgbm.py --resume-from forward_selection_lightgbm_xxx.json
"""

import os
import logging

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

logging.getLogger('qlib').setLevel(logging.WARNING)

import sys
from pathlib import Path
import gc

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))
project_root = script_dir.parent

import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

import lightgbm as lgb

from utils.talib_ops import TALIB_OPS
from data.stock_pools import STOCK_POOLS

from models.feature_engineering.feature_selection_utils import (
    INNER_CV_FOLDS,
    ALL_STOCK_FEATURES,
    ALL_MACRO_FEATURES,
    PROJECT_ROOT,
    compute_ic,
    validate_qlib_features,
    validate_macro_features,
    load_checkpoint,
    ForwardSelectionBase,
    add_common_args,
    countdown,
)
from models.common.dynamic_handlers import DynamicTabularHandler


# ============================================================================
# LightGBM 模型配置
# ============================================================================

DEFAULT_LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_iterations': 500,
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 128,
    'min_data_in_leaf': 50,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.0,
    'lambda_l2': 1.0,
    'num_threads': 8,
    'verbose': -1,
    'seed': 42,
}


# ============================================================================
# 初始特征集 (基于 importance 分析)
# ============================================================================

# 初始股票/TALib 特征
BASELINE_STOCK_FEATURES = {
    "TALIB_ATR14": "TALIB_ATR($high, $low, $close, 14)/$close",
}

# 初始宏观特征
BASELINE_MACRO_FEATURES = [
    "macro_vix_term_structure",
    "macro_gld_vol20",
    "macro_vix_level",
    "macro_xly_pct_20d",
    "macro_uso_pct_20d",
    "macro_hy_spread",
    "macro_gld_pct_20d",
    "macro_vix_term_zscore",
    "macro_uup_ma20_ratio",
]


# 使用共享的 DynamicTabularHandler (从 models.common.dynamic_handlers 导入)


# ============================================================================
# LightGBM Forward Selection 实现
# ============================================================================

class LightGBMForwardSelection(ForwardSelectionBase):
    """LightGBM 模型的 Forward Selection 实现"""

    def __init__(
        self,
        symbols: List[str],
        baseline_stock: Dict[str, str],
        baseline_macro: List[str],
        candidate_stock: Dict[str, str],
        candidate_macro: List[str],
        nday: int = 5,
        max_features: int = 30,
        min_improvement: float = 0.0005,
        params: dict = None,
        output_dir: Path = None,
        quiet: bool = False,
    ):
        super().__init__(
            symbols=symbols,
            nday=nday,
            max_features=max_features,
            min_improvement=min_improvement,
            output_dir=output_dir,
            checkpoint_name="forward_selection_lightgbm_checkpoint",
            result_prefix="forward_selection_lightgbm",
            quiet=quiet,
        )

        self.current_stock = dict(baseline_stock)
        self.current_macro = list(baseline_macro)
        self.candidate_stock = candidate_stock
        self.candidate_macro = candidate_macro

        self.params = params or DEFAULT_LIGHTGBM_PARAMS

    def prepare_fold_data(self, fold_config: Dict) -> Tuple:
        """准备单个fold的数据"""
        handler = DynamicTabularHandler(
            stock_features=self.current_stock,
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

        # 获取数据 - LightGBM 需要填充 NaN
        X_train = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
        X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)

        y_train = dataset.prepare("train", col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0]
        y_train = y_train.fillna(0).values

        X_valid = dataset.prepare("valid", col_set="feature", data_key=DataHandlerLP.DK_L)
        X_valid = X_valid.fillna(0).replace([np.inf, -np.inf], 0)
        valid_index = X_valid.index

        y_valid = dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(y_valid, pd.DataFrame):
            y_valid = y_valid.iloc[:, 0]
        y_valid = y_valid.fillna(0).values

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
                gc.collect()

                X_train, y_train, X_valid, y_valid, valid_index = self.prepare_fold_data(fold)

                # 创建 LightGBM Dataset
                train_set = lgb.Dataset(X_train, label=y_train)
                valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

                # 训练参数
                lgb_params = {
                    'objective': self.params.get('objective', 'regression'),
                    'metric': self.params.get('metric', 'mse'),
                    'boosting_type': self.params.get('boosting_type', 'gbdt'),
                    'learning_rate': self.params.get('learning_rate', 0.05),
                    'max_depth': int(self.params.get('max_depth', 8)),
                    'num_leaves': int(self.params.get('num_leaves', 128)),
                    'min_data_in_leaf': int(self.params.get('min_data_in_leaf', 50)),
                    'feature_fraction': self.params.get('feature_fraction', 0.8),
                    'bagging_fraction': self.params.get('bagging_fraction', 0.8),
                    'bagging_freq': int(self.params.get('bagging_freq', 5)),
                    'lambda_l1': self.params.get('lambda_l1', 0.0),
                    'lambda_l2': self.params.get('lambda_l2', 1.0),
                    'num_threads': int(self.params.get('num_threads', 8)),
                    'verbose': -1,
                    'seed': self.params.get('seed', 42),
                }

                # 训练模型
                model = lgb.train(
                    lgb_params,
                    train_set,
                    num_boost_round=self.params.get('num_iterations', 500),
                    valid_sets=[valid_set],
                    valid_names=['valid'],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                    ],
                )

                # 预测并计算 IC
                val_pred = model.predict(X_valid)
                ic = compute_ic(val_pred, y_valid, valid_index)
                fold_ics.append(ic)

                del X_train, y_train, X_valid, y_valid, valid_index, model, train_set, valid_set
                gc.collect()

        finally:
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
        gc.collect()

    def _print_final_result(self):
        """打印最终结果"""
        print("\n" + "=" * 70)
        print("LIGHTGBM FORWARD SELECTION COMPLETE")
        print("=" * 70)
        total_features = len(self.current_stock) + len(self.current_macro)
        print(f"Final: {len(self.current_stock)} stock + {len(self.current_macro)} macro = {total_features} features")
        print(f"Baseline IC: {self.baseline_ic:.4f}")
        ic_diff = self.current_ic - self.baseline_ic
        print(f"Final IC:    {self.current_ic:.4f} ({'+' if ic_diff >= 0 else ''}{ic_diff:.4f})")

        print(f"\nFinal Stock Features ({len(self.current_stock)}):")
        for name in sorted(self.current_stock.keys()):
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
    parser = argparse.ArgumentParser(description='LightGBM Nested CV Forward Feature Selection')
    add_common_args(parser)

    args = parser.parse_args()

    # Qlib 初始化
    qlib_data_path = PROJECT_ROOT / "my_data" / "qlib_us"
    qlib.init(
        provider_uri=str(qlib_data_path),
        region=REG_US,
        custom_ops=TALIB_OPS,
        kernels=1,
        joblib_backend=None,
    )

    symbols = STOCK_POOLS[args.stock_pool]
    output_dir = PROJECT_ROOT / "outputs" / "feature_selection"

    print(f"Stock pool: {args.stock_pool} ({len(symbols)} stocks)")

    # 加载 baseline 或从 checkpoint 恢复
    excluded_features = set()
    baseline_stock = dict(BASELINE_STOCK_FEATURES)
    baseline_macro = list(BASELINE_MACRO_FEATURES)

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
            baseline_stock = checkpoint.get('final_stock_features', dict(BASELINE_STOCK_FEATURES))
            baseline_macro = checkpoint.get('final_macro_features', list(BASELINE_MACRO_FEATURES))

        excluded_features = set(checkpoint.get('excluded_features', []))
        print(f"    Features: {len(baseline_stock)} stock + {len(baseline_macro)} macro")
        print(f"    Excluded: {len(excluded_features)}")

    elif args.resume:
        # 从默认 checkpoint 恢复
        checkpoint_file = output_dir / "forward_selection_lightgbm_checkpoint.json"
        if checkpoint_file.exists():
            print(f"\n[*] Resuming from checkpoint")
            checkpoint = load_checkpoint(checkpoint_file)

            baseline_stock = checkpoint['current_stock_features']
            baseline_macro = checkpoint['current_macro_features']
            excluded_features = set(checkpoint.get('excluded_features', []))
            print(f"    Features: {len(baseline_stock)} stock + {len(baseline_macro)} macro")
            print(f"    Excluded: {len(excluded_features)}")
        else:
            print("No checkpoint file found, starting fresh")

    # 获取候选特征（不在 baseline 中的特征）
    candidate_stock = {k: v for k, v in ALL_STOCK_FEATURES.items()
                       if k not in baseline_stock}
    candidate_macro = [m for m in ALL_MACRO_FEATURES
                       if m not in baseline_macro]

    # 验证候选特征
    candidate_stock = validate_qlib_features(symbols, candidate_stock)
    candidate_macro = validate_macro_features(candidate_macro)
    print(f"\n[*] Baseline: {len(baseline_stock)} stock + {len(baseline_macro)} macro")
    print(f"[*] Candidates: {len(candidate_stock)} stock + {len(candidate_macro)} macro")

    # LightGBM 参数
    params = dict(DEFAULT_LIGHTGBM_PARAMS)

    # 倒计时
    if not args.no_countdown:
        if not countdown(3):
            return

    # 创建 Forward Selection 实例并运行
    selector = LightGBMForwardSelection(
        symbols=symbols,
        baseline_stock=baseline_stock,
        baseline_macro=baseline_macro,
        candidate_stock=candidate_stock,
        candidate_macro=candidate_macro,
        nday=args.nday,
        max_features=args.max_features,
        min_improvement=args.min_improvement,
        params=params,
        output_dir=output_dir,
        quiet=args.quiet,
    )

    final_features, history, final_excluded = selector.run(
        excluded_features=excluded_features,
        method_name='nested_cv_lightgbm_forward_selection',
    )

    print(f"\n[+] Forward selection complete")
    print(f"  Final stock features: {len(final_features.get('current_stock_features', {}))}")
    print(f"  Final macro features: {len(final_features.get('current_macro_features', []))}")
    print(f"  Excluded features: {len(final_excluded)}")


if __name__ == "__main__":
    main()
