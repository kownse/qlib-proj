"""
共享配置和常量

包含所有模型训练脚本共用的配置、常量和路径配置。
Handler 配置已统一到 handlers.py 中管理。
"""

from pathlib import Path

# Import handler utilities from centralized registry
from models.common.handlers import (
    HANDLER_CONFIG,
    get_handler_class,
    get_handler_config,
    get_available_handlers,
    get_handler_epilog,
    handler_uses_talib,
)


# ========== 路径配置 ==========
# scripts/models/common/config.py -> scripts/models/common -> scripts/models -> scripts -> qlib-proj
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # 项目根目录
QLIB_DATA_PATH = PROJECT_ROOT / "my_data" / "qlib_us"
NEWS_DATA_PATH = PROJECT_ROOT / "my_data" / "news_processed" / "news_features.parquet"
MACRO_DATA_PATH = PROJECT_ROOT / "my_data" / "macro_processed" / "macro_features.parquet"
MODEL_SAVE_PATH = PROJECT_ROOT / "my_models"


# ========== 默认时间划分 ==========
DEFAULT_TIME_SPLITS = {
    'train_start': "2000-01-01",
    'train_end': "2022-12-31",
    'valid_start': "2023-01-01",
    'valid_end': "2023-12-31",
    'test_start': "2024-01-01",
    'test_end': "2025-12-31",
}

MAX_TRAIN_TIME_SPLITS = {
    'train_start': "2000-01-01",
    'train_end': "2025-09-30",
    'valid_start': "2025-10-01",
    'valid_end': "2025-12-31",
    'test_start': "2025-10-01",
    'test_end': "2025-12-31",
}
