"""
Transformer Forward Selection 共享基础模块

包含:
- TransformerForwardSelectionBase: Forward Selection 基类
- run_forward_selection_main: 共享的 main 函数逻辑
- 共享常量

注意: normalize_data 和 DynamicTimeSeriesHandler 已移至 models.common.dynamic_handlers
"""

import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Any, Callable

import numpy as np
import pandas as pd
import torch

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from models.feature_engineering.feature_selection_utils import (
    INNER_CV_FOLDS,
    compute_ic,
    save_final_result,
    ForwardSelectionBase,
)

# 从共享模块导入
from models.common.dynamic_handlers import (
    DynamicTimeSeriesHandler,
    normalize_data,
    DEFAULT_SEQ_LEN,
    ALPHA300_BASE_FEATURES,
)

# 兼容性别名
ALPHA300_SEQ_LEN = DEFAULT_SEQ_LEN
DynamicAlpha300Handler = DynamicTimeSeriesHandler


# ============================================================================
# Transformer Forward Selection 基类
# ============================================================================

class TransformerForwardSelectionBase(ForwardSelectionBase, ABC):
    """Transformer 模型 Forward Selection 的抽象基类

    子类需要实现:
    - create_model(): 创建具体的 Transformer 模型
    - get_model_name(): 返回模型名称用于日志输出
    """

    def __init__(
        self,
        symbols: List[str],
        baseline_talib: Dict[str, str],
        baseline_macro: List[str],
        candidate_talib: Dict[str, str],
        candidate_macro: List[str],
        nday: int = 5,
        max_features: int = 30,
        min_improvement: float = 0.002,
        exclude_threshold: float = -0.001,
        epochs: int = 20,
        early_stop: int = 8,
        params: dict = None,
        output_dir: Path = None,
        checkpoint_name: str = "forward_selection_transformer_checkpoint",
        result_prefix: str = "forward_selection_transformer",
        gpu: int = 0,
    ):
        super().__init__(
            symbols=symbols,
            nday=nday,
            max_features=max_features,
            min_improvement=min_improvement,
            output_dir=output_dir,
            checkpoint_name=checkpoint_name,
            result_prefix=result_prefix,
        )

        self.exclude_threshold = exclude_threshold
        self.current_talib = dict(baseline_talib)
        self.current_macro = list(baseline_macro)
        self.candidate_talib = candidate_talib
        self.candidate_macro = candidate_macro

        self.epochs = epochs
        self.early_stop = early_stop
        self.params = params or {}
        self.gpu = gpu

        # 设置设备
        if gpu >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu}')
        else:
            self.device = torch.device('cpu')

    @abstractmethod
    def create_model(self, d_feat: int) -> torch.nn.Module:
        """创建 Transformer 模型

        Args:
            d_feat: 每个时间步的特征数量

        Returns:
            torch.nn.Module: Transformer 模型实例
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """返回模型名称用于日志输出"""
        pass

    def get_extra_config_str(self) -> str:
        """返回额外的配置信息字符串，子类可覆盖"""
        return ""

    def run(
        self,
        excluded_features: set = None,
        method_name: str = None,
    ) -> Tuple[Dict, List, set]:
        """运行贪婪 forward selection"""
        if method_name is None:
            method_name = f"nested_cv_{self.get_model_name().lower().replace(' ', '_')}_forward_selection"

        if excluded_features is not None:
            self.excluded_features = set(excluded_features)

        counts = self.get_feature_counts()
        counts_str = " + ".join([f"{v} {k}" for k, v in counts.items()])

        print("\n" + "=" * 70)
        print(f"NESTED CV FORWARD SELECTION - {self.get_model_name().upper()} (Greedy Mode)")
        print("=" * 70)
        print(f"Current features: {counts_str}")
        print(f"Max features: {self.max_features}")
        print(f"Min IC improvement to add: {self.min_improvement}")
        print(f"Exclude threshold: {self.exclude_threshold} (only exclude if IC drops more)")
        print(f"Inner CV Folds: {len(INNER_CV_FOLDS)}")
        extra_config = self.get_extra_config_str()
        if extra_config:
            print(extra_config)
        print("=" * 70)

        if self.excluded_features:
            print(f"\nExcluded features from previous runs: {len(self.excluded_features)}")

        # 基线评估
        print("\n[*] Evaluating baseline features...")
        self.baseline_ic, baseline_fold_ics = self.evaluate_feature_set()
        self.current_ic = self.baseline_ic

        print(f"    Baseline Inner CV IC: {self.baseline_ic:.4f}")
        print(f"    Fold ICs: {[f'{ic:.4f}' for ic in baseline_fold_ics]}")

        counts = self.get_feature_counts()
        self.history.append({
            'round': 0,
            'action': 'BASELINE',
            'feature': None,
            'type': None,
            'inner_cv_ic': self.baseline_ic,
            'fold_ics': baseline_fold_ics,
            'ic_change': 0,
            **{f'{k}_count': v for k, v in counts.items()},
        })

        # 保存初始 checkpoint
        self._save_checkpoint(0)

        round_num = 0
        features_added_this_session = 0

        # 贪婪 forward selection loop
        while sum(self.get_feature_counts().values()) < self.max_features:
            round_num += 1

            testable_talib, testable_macro = self.get_testable_candidates()

            if not testable_talib and not testable_macro:
                print(f"\n[!] No more candidates to test. Stopping.")
                break

            counts = self.get_feature_counts()
            counts_str = " + ".join([f"{v} {k}" for k, v in counts.items()])
            print(f"\n[Round {round_num}] Current IC: {self.current_ic:.4f}, Features: {counts_str}")
            print(f"    Excluded: {len(self.excluded_features)}, Candidates: {len(testable_talib)} TALib + {len(testable_macro)} macro")

            found_improvement = False

            # 测试 TALib 特征
            talib_items = list(testable_talib.items())
            for name, expr in talib_items:
                if name in self.excluded_features:
                    continue

                try:
                    self.cleanup_after_evaluation()
                    ic, fold_ics = self.test_feature(name, 'feature', expr)
                    ic_change = ic - self.current_ic

                    if ic_change < self.exclude_threshold:
                        self.excluded_features.add(name)
                        self._save_checkpoint(round_num)
                        print(f"    X {name}: IC={ic:.4f} ({ic_change:+.4f}) -> excluded (< {self.exclude_threshold})")

                    elif ic_change >= self.min_improvement:
                        self.add_feature(name, 'feature', expr)
                        self.current_ic = ic
                        features_added_this_session += 1

                        counts = self.get_feature_counts()
                        self.history.append({
                            'round': round_num,
                            'action': 'ADD',
                            'feature': name,
                            'type': 'talib',
                            'inner_cv_ic': ic,
                            'fold_ics': fold_ics,
                            'ic_change': ic_change,
                            **{f'{k}_count': v for k, v in counts.items()},
                        })

                        self._save_checkpoint(round_num)
                        print(f"    + {name}: IC={ic:.4f} (+{ic_change:.4f}) -> ADDED")
                        found_improvement = True

                    else:
                        print(f"      {name}: IC={ic:.4f} ({ic_change:+.4f}) -> skipped")

                except Exception as e:
                    print(f"    ! {name}: ERROR - {e}")

            # 测试 Macro 特征
            macro_items = list(testable_macro)
            for name in macro_items:
                if name in self.excluded_features:
                    continue

                try:
                    self.cleanup_after_evaluation()
                    ic, fold_ics = self.test_feature(name, 'macro', None)
                    ic_change = ic - self.current_ic

                    if ic_change < self.exclude_threshold:
                        self.excluded_features.add(name)
                        self._save_checkpoint(round_num)
                        print(f"    X {name}: IC={ic:.4f} ({ic_change:+.4f}) -> excluded (< {self.exclude_threshold})")

                    elif ic_change >= self.min_improvement:
                        self.add_feature(name, 'macro', None)
                        self.current_ic = ic
                        features_added_this_session += 1

                        counts = self.get_feature_counts()
                        self.history.append({
                            'round': round_num,
                            'action': 'ADD',
                            'feature': name,
                            'type': 'macro',
                            'inner_cv_ic': ic,
                            'fold_ics': fold_ics,
                            'ic_change': ic_change,
                            **{f'{k}_count': v for k, v in counts.items()},
                        })

                        self._save_checkpoint(round_num)
                        print(f"    + {name}: IC={ic:.4f} (+{ic_change:.4f}) -> ADDED")
                        found_improvement = True

                    else:
                        print(f"      {name}: IC={ic:.4f} ({ic_change:+.4f}) -> skipped")

                except Exception as e:
                    print(f"    ! {name}: ERROR - {e}")

            if not found_improvement:
                print(f"\n[!] No feature improved IC by >= {self.min_improvement} in this round. Stopping.")
                break

        # 打印最终结果
        self._print_final_result()
        print(f"\nFeatures added this session: {features_added_this_session}")

        # 保存最终结果
        if self.output_dir:
            result_file = save_final_result(
                self.output_dir,
                self.result_prefix,
                method_name,
                self.baseline_ic,
                self.current_ic,
                self.get_current_features_dict(),
                self.excluded_features,
                self.history,
            )
            print(f"Results saved to: {result_file}")

        return self.get_current_features_dict(), self.history, self.excluded_features

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

                # 创建模型（由子类实现）
                model = self.create_model(d_feat)
                model = model.to(self.device)

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
        print(f"{self.get_model_name().upper()} FORWARD SELECTION COMPLETE")
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
# 共享的 main 函数辅助
# ============================================================================

def run_forward_selection_main(
    selector_class,
    model_name: str,
    default_params: dict,
    extra_args_func: Callable = None,
    extra_params_func: Callable = None,
):
    """
    共享的 main 函数逻辑

    Args:
        selector_class: ForwardSelection 类
        model_name: 模型名称 (用于日志)
        default_params: 默认参数字典
        extra_args_func: 添加额外参数的函数 (parser) -> None
        extra_params_func: 处理额外参数的函数 (args, params) -> None
    """
    import argparse
    import qlib
    from qlib.constant import REG_US
    from utils.talib_ops import TALIB_OPS
    from data.stock_pools import STOCK_POOLS
    from models.feature_engineering.feature_selection_utils import (
        ALL_TALIB_FEATURES,
        ALL_MACRO_FEATURES,
        PROJECT_ROOT,
        validate_qlib_features,
        validate_macro_features,
        load_checkpoint,
        add_common_args,
        countdown,
    )

    parser = argparse.ArgumentParser(description=f'{model_name} Nested CV Forward Feature Selection (Greedy Mode)')
    add_common_args(parser)

    # 覆盖 min_improvement 默认值为 0.002
    parser.set_defaults(min_improvement=0.002)

    # 排除阈值参数
    parser.add_argument('--exclude-threshold', type=float, default=-0.001,
                        help='IC drop threshold to exclude a feature (default: -0.001)')

    # Transformer 通用参数
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dim-feedforward', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)

    # 额外参数
    if extra_args_func:
        extra_args_func(parser)

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

    # 根据 selector 类确定 checkpoint 文件名
    checkpoint_name = getattr(selector_class, 'DEFAULT_CHECKPOINT_NAME', 'forward_selection_transformer_checkpoint')

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
        print(f"    Current IC: {checkpoint.get('current_ic', 'N/A')}")
        print(f"    Included TALib: {len(baseline_talib)}")
        if baseline_talib:
            for name in sorted(baseline_talib.keys()):
                print(f"      - {name}")
        print(f"    Included Macro: {len(baseline_macro)}")
        if baseline_macro:
            for name in baseline_macro:
                print(f"      - {name}")
        print(f"    Excluded: {len(excluded_features)}")

    elif args.resume:
        checkpoint_file = output_dir / f"{checkpoint_name}.json"
        if checkpoint_file.exists():
            print(f"\n[*] Resuming from checkpoint: {checkpoint_file}")
            checkpoint = load_checkpoint(checkpoint_file)

            baseline_talib = checkpoint.get('current_talib_features', {})
            baseline_macro = checkpoint.get('current_macro_features', [])
            excluded_features = set(checkpoint.get('excluded_features', []))
            print(f"    Current IC: {checkpoint.get('current_ic', 'N/A')}")
            print(f"    Included TALib: {len(baseline_talib)}")
            if baseline_talib:
                for name in sorted(baseline_talib.keys()):
                    print(f"      - {name}")
            print(f"    Included Macro: {len(baseline_macro)}")
            if baseline_macro:
                for name in baseline_macro:
                    print(f"      - {name}")
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

    # 构建参数
    params = dict(default_params)
    params['d_model'] = args.d_model
    params['nhead'] = args.nhead
    params['num_layers'] = args.num_layers
    params['dim_feedforward'] = args.dim_feedforward
    params['dropout'] = args.dropout
    params['lr'] = args.lr
    params['weight_decay'] = args.weight_decay
    params['batch_size'] = args.batch_size

    # 处理额外参数
    if extra_params_func:
        extra_params_func(args, params)

    print(f"\n[*] {model_name} config:")
    print(f"    d_model: {params['d_model']}")
    print(f"    nhead: {params['nhead']}")
    print(f"    num_layers: {params['num_layers']}")
    print(f"    dim_feedforward: {params['dim_feedforward']}")
    print(f"    dropout: {params['dropout']}")
    print(f"    lr: {params['lr']}")
    # 打印额外参数
    for key in ['use_local_conv', 'use_relative_bias']:
        if key in params:
            print(f"    {key}: {params[key]}")
    print(f"    epochs: {args.epochs}")
    print(f"    early_stop: {args.early_stop}")
    print(f"    min_improvement: {args.min_improvement} (IC threshold to add feature)")
    print(f"    exclude_threshold: {args.exclude_threshold} (IC drop threshold to exclude)")

    # 倒计时
    if not args.no_countdown:
        if not countdown(3):
            return

    # 创建 Forward Selection 实例并运行
    selector = selector_class(
        symbols=symbols,
        baseline_talib=baseline_talib,
        baseline_macro=baseline_macro,
        candidate_talib=candidate_talib,
        candidate_macro=candidate_macro,
        nday=args.nday,
        max_features=args.max_features,
        min_improvement=args.min_improvement,
        exclude_threshold=args.exclude_threshold,
        epochs=args.epochs,
        early_stop=args.early_stop,
        params=params,
        output_dir=output_dir,
        gpu=args.gpu,
    )

    final_features, history, final_excluded = selector.run(
        excluded_features=excluded_features,
    )

    print(f"\n[+] Forward selection complete")
    d_feat = ALPHA300_BASE_FEATURES + len(final_features.get('current_talib_features', {})) + len(final_features.get('current_macro_features', []))
    print(f"  Final d_feat: {d_feat}")
    print(f"  Total features: {d_feat * ALPHA300_SEQ_LEN}")
    print(f"  Excluded features: {len(final_excluded)}")
