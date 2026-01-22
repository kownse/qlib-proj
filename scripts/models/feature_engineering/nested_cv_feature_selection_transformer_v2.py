"""
Transformer V2 嵌套交叉验证 Forward Selection (贪婪模式) - 基于 Alpha300 逐步添加 Macro 和 TALib 特征

使用改进的 Transformer V2 模型:
- Time2Vec 可学习位置编码
- 局部时序卷积
- CLS Token 聚合
- 相对位置偏置

设计原理:
- 从 Alpha300 (5 OHLCV × 60 days = 300 features) 作为基线
- 逐个测试候选特征（macro 和 TALib），每个特征加入过去60天的值
- 贪婪添加：当发现一个特征能提升IC超过阈值(默认0.002)时，立即添加到特征集
- 排除逻辑：只有当IC下降超过阈值(默认-0.001)时才排除，避免过于激进
- 每次添加或排除特征后立即保存checkpoint，支持中断恢复

内层CV (只用2000-2024年数据):
  Fold 1: train 2000-2020, valid 2021
  Fold 2: train 2000-2021, valid 2022
  Fold 3: train 2000-2022, valid 2023
  Fold 4: train 2000-2023, valid 2024

使用方法:
    # 从头开始
    python scripts/models/feature_engineering/nested_cv_feature_selection_transformer_v2.py --stock-pool sp500

    # 从checkpoint恢复（读取已添加和已排除的特征）
    python scripts/models/feature_engineering/nested_cv_feature_selection_transformer_v2.py --resume

    # 从指定文件恢复
    python scripts/models/feature_engineering/nested_cv_feature_selection_transformer_v2.py --resume-from forward_selection_transformer_v2_xxx.json

    # 指定阈值
    python scripts/models/feature_engineering/nested_cv_feature_selection_transformer_v2.py --min-improvement 0.003 --exclude-threshold -0.002

    # 禁用V2特有功能进行对比
    python scripts/models/feature_engineering/nested_cv_feature_selection_transformer_v2.py --no-local-conv --no-relative-bias
"""

import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('qlib').setLevel(logging.WARNING)

import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

from typing import Dict, List

import torch

# 导入 Transformer V2 模型
from models.deep.transformer_v2_model import TransformerNetV2

# 导入共享基类
from models.feature_engineering.transformer_forward_selection_base import (
    ALPHA300_SEQ_LEN,
    TransformerForwardSelectionBase,
    run_forward_selection_main,
)


# ============================================================================
# Transformer V2 模型配置
# ============================================================================

DEFAULT_TRANSFORMER_V2_PARAMS = {
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "lr": 1e-4,
    "weight_decay": 1e-3,
    "batch_size": 2048,
    "use_local_conv": True,
    "use_relative_bias": True,
}


# ============================================================================
# Transformer V2 Forward Selection 实现
# ============================================================================

class TransformerV2ForwardSelection(TransformerForwardSelectionBase):
    """Transformer V2 模型的 Forward Selection 实现

    V2 特有功能:
    - Time2Vec 可学习位置编码
    - 局部时序卷积 (可通过 use_local_conv 开关)
    - CLS Token 聚合
    - 相对位置偏置 (可通过 use_relative_bias 开关)
    """

    DEFAULT_CHECKPOINT_NAME = "forward_selection_transformer_v2_checkpoint"

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
        gpu: int = 0,
    ):
        super().__init__(
            symbols=symbols,
            baseline_talib=baseline_talib,
            baseline_macro=baseline_macro,
            candidate_talib=candidate_talib,
            candidate_macro=candidate_macro,
            nday=nday,
            max_features=max_features,
            min_improvement=min_improvement,
            exclude_threshold=exclude_threshold,
            epochs=epochs,
            early_stop=early_stop,
            params=params or DEFAULT_TRANSFORMER_V2_PARAMS,
            output_dir=output_dir,
            checkpoint_name="forward_selection_transformer_v2_checkpoint",
            result_prefix="forward_selection_transformer_v2",
            gpu=gpu,
        )

    def create_model(self, d_feat: int) -> torch.nn.Module:
        """创建 Transformer V2 模型"""
        return TransformerNetV2(
            d_feat=d_feat,
            d_model=self.params['d_model'],
            nhead=self.params['nhead'],
            num_layers=self.params['num_layers'],
            dim_feedforward=self.params['dim_feedforward'],
            dropout=self.params['dropout'],
            seq_len=ALPHA300_SEQ_LEN,
            use_local_conv=self.params.get('use_local_conv', True),
            use_relative_bias=self.params.get('use_relative_bias', True),
        )

    def get_model_name(self) -> str:
        return "Transformer V2"

    def get_extra_config_str(self) -> str:
        """返回 V2 特有配置信息"""
        use_local_conv = self.params.get('use_local_conv', True)
        use_relative_bias = self.params.get('use_relative_bias', True)
        return f"V2 Features: LocalConv={use_local_conv}, RelativeBias={use_relative_bias}"


# ============================================================================
# Main
# ============================================================================

def add_v2_args(parser):
    """添加 V2 特有参数"""
    parser.add_argument('--no-local-conv', action='store_true',
                        help='Disable local temporal convolution')
    parser.add_argument('--no-relative-bias', action='store_true',
                        help='Disable relative position bias')


def process_v2_params(args, params):
    """处理 V2 特有参数"""
    params['use_local_conv'] = not args.no_local_conv
    params['use_relative_bias'] = not args.no_relative_bias


def main():
    run_forward_selection_main(
        selector_class=TransformerV2ForwardSelection,
        model_name="Transformer V2",
        default_params=DEFAULT_TRANSFORMER_V2_PARAMS,
        extra_args_func=add_v2_args,
        extra_params_func=process_v2_params,
    )


if __name__ == "__main__":
    main()
