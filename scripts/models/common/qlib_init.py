"""
Qlib 初始化模块

这个模块必须在任何其他 qlib 相关导入之前被导入。
它解决了 qlib 自动初始化与 TA-Lib 多进程冲突的问题。

问题描述:
- 某些 qlib 模块在导入时会触发 auto_init
- auto_init 使用默认配置（多进程）
- 多进程与 TA-Lib 的内存管理冲突，导致:
  - "free(): invalid pointer"
  - "corrupted size vs. prev_size"

解决方案:
- 在任何 qlib 模块导入之前，先用 kernels=1 初始化 qlib
"""

import sys
import os
from pathlib import Path


def _ensure_correct_qlib_import():
    """
    确保正确导入 qlib（避免被项目中的 qlib 子目录干扰）

    项目中有一个 qlib/ 子目录（git submodule），它会干扰 Python 的导入。
    需要确保 Python 导入的是从这个子目录安装的 pyqlib 包，而不是把
    子目录当作普通 Python 包导入。
    """
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent.parent
    qlib_subdir = project_root / "qlib"

    # 从 sys.path 中移除可能导致错误导入的路径
    paths_to_remove = [
        '',
        '.',
        str(project_root),
        str(qlib_subdir),
    ]

    # 过滤掉问题路径
    sys.path = [p for p in sys.path if p not in paths_to_remove]


def pre_init_qlib(use_talib: bool = True):
    """
    在导入任何其他 qlib 模块之前预初始化 qlib

    Parameters
    ----------
    use_talib : bool
        是否使用 TA-Lib（如果使用，则设置 kernels=1）
    """
    # 确保正确导入
    _ensure_correct_qlib_import()

    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent.parent
    qlib_data_path = project_root / "my_data" / "qlib_us"

    # 导入 qlib
    import qlib
    from qlib.constant import REG_US

    # 如果使用 TA-Lib，需要设置 kernels=1
    if use_talib:
        from utils.talib_ops import TALIB_OPS
        qlib.init(
            provider_uri=str(qlib_data_path),
            region=REG_US,
            custom_ops=TALIB_OPS,
            kernels=1,
            skip_if_reg=True  # 如果已经初始化则跳过
        )
    else:
        qlib.init(
            provider_uri=str(qlib_data_path),
            region=REG_US,
            skip_if_reg=True
        )
