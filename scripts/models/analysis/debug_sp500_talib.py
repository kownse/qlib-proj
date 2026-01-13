"""
调试 sp500 + alpha158-talib 的 DataSet 初始化问题

重现错误：
- free(): invalid pointer
- corrupted size vs. prev_size

这通常与 Qlib 的多进程数据加载与 TA-Lib 的内存管理冲突有关。
"""

import sys
import os
import traceback
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "scripts"))

import pandas as pd
import numpy as np


def test_with_original_flow():
    """使用与 run_catboost_nd.py 相同的流程测试"""
    print("\n" + "=" * 70)
    print("测试: 使用原始训练流程 (与 run_catboost_nd.py 相同)")
    print("=" * 70)

    from models.common import (
        HANDLER_CONFIG, PROJECT_ROOT, MODEL_SAVE_PATH,
        get_time_splits,
        init_qlib,
        check_data_availability,
        create_data_handler,
        create_dataset,
        analyze_features,
    )
    from data.stock_pools import STOCK_POOLS

    # 模拟命令行参数
    class Args:
        nday = 5
        handler = 'alpha158-talib'
        stock_pool = 'sp500'
        max_train = False
        news_features = 'core'
        news_rolling = False

    args = Args()
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = get_time_splits(args.max_train)

    print(f"股票池: {args.stock_pool} ({len(symbols)} 只股票)")
    print(f"Handler: {args.handler}")
    print(f"时间范围: {time_splits['train_start']} ~ {time_splits['test_end']}")

    try:
        # 初始化 Qlib
        init_qlib(handler_config['use_talib'])

        # 检查数据可用性
        check_data_availability(time_splits)

        # 创建 DataHandler - 这里可能出问题
        print("\n正在创建 DataHandler...")
        handler = create_data_handler(args, handler_config, symbols, time_splits)
        print("✓ DataHandler 创建成功")

        # 创建 Dataset
        print("\n正在创建 Dataset...")
        dataset = create_dataset(handler, time_splits)
        print("✓ Dataset 创建成功")

        # 分析特征
        print("\n正在分析特征...")
        train_data, valid_cols, dropped_cols = analyze_features(dataset)
        print("✓ 特征分析完成")

        return True
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        traceback.print_exc()
        return False


def test_with_reduced_workers():
    """使用减少的 worker 数量测试"""
    print("\n" + "=" * 70)
    print("测试: 设置环境变量限制并行度")
    print("=" * 70)

    # 设置环境变量限制并行
    os.environ['NUMEXPR_MAX_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    print("已设置环境变量:")
    print("  NUMEXPR_MAX_THREADS=1")
    print("  OMP_NUM_THREADS=1")
    print("  MKL_NUM_THREADS=1")
    print("  OPENBLAS_NUM_THREADS=1")

    return test_with_original_flow()


def test_qlib_kernels():
    """测试 Qlib 的 kernel 数量设置"""
    print("\n" + "=" * 70)
    print("测试: 设置 Qlib kernels=1 (单进程)")
    print("=" * 70)

    import qlib
    from qlib.constant import REG_US
    from utils.talib_ops import TALIB_OPS
    from data.stock_pools import SP500_SYMBOLS
    from data.datahandler_ext import Alpha158_Volatility_TALib
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    qlib_data_path = project_root / "my_data" / "qlib_us"

    try:
        # 使用 kernels=1 初始化
        qlib.init(
            provider_uri=str(qlib_data_path),
            region=REG_US,
            custom_ops=TALIB_OPS,
            kernels=1  # 单进程
        )
        print("✓ Qlib 初始化成功 (kernels=1)")

        # 获取可用的 SP500 股票
        from qlib.data import D
        instruments = D.instruments(market="all")
        available = set(D.list_instruments(instruments))
        symbols = [s for s in SP500_SYMBOLS if s in available]
        print(f"可用股票: {len(symbols)}")

        # 创建 Handler
        print("\n正在创建 DataHandler...")
        handler = Alpha158_Volatility_TALib(
            volatility_window=5,
            instruments=symbols,
            start_time="2000-01-01",
            end_time="2025-12-31",
            fit_start_time="2000-01-01",
            fit_end_time="2022-12-31",
            infer_processors=[],
        )
        print("✓ DataHandler 创建成功")

        # 创建 Dataset
        print("\n正在创建 Dataset...")
        dataset = DatasetH(
            handler=handler,
            segments={
                "train": ("2000-01-01", "2022-12-31"),
                "valid": ("2023-01-01", "2023-12-31"),
                "test": ("2024-01-01", "2025-12-31"),
            }
        )
        print("✓ Dataset 创建成功")

        # 准备数据
        print("\n正在准备训练数据...")
        train_features = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
        print(f"✓ Train 特征: {train_features.shape}")

        return True
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        traceback.print_exc()
        return False


def test_sequential_loading():
    """测试顺序加载（无并行）"""
    print("\n" + "=" * 70)
    print("测试: 手动顺序加载特征")
    print("=" * 70)

    import qlib
    from qlib.constant import REG_US
    from qlib.data import D
    from utils.talib_ops import TALIB_OPS
    from data.stock_pools import SP500_SYMBOLS
    from data.datahandler_ext import Alpha158_Volatility_TALib

    qlib_data_path = project_root / "my_data" / "qlib_us"

    try:
        # 强制重新初始化
        qlib.init(
            provider_uri=str(qlib_data_path),
            region=REG_US,
            custom_ops=TALIB_OPS,
            kernels=1,
        )
        print("✓ Qlib 初始化成功")

        # 获取可用股票
        instruments = D.instruments(market="all")
        available = set(D.list_instruments(instruments))
        symbols = [s for s in SP500_SYMBOLS if s in available][:50]  # 只用50只股票
        print(f"测试股票数: {len(symbols)}")

        # 获取特征配置
        handler = Alpha158_Volatility_TALib.__new__(Alpha158_Volatility_TALib)
        handler.volatility_window = 5
        fields, names = handler.get_feature_config()
        print(f"特征数: {len(fields)}")

        # 分批加载特征
        batch_size = 50
        print(f"\n分批加载特征 (每批 {batch_size} 个)...")

        for i in range(0, len(fields), batch_size):
            batch_fields = fields[i:i+batch_size]
            batch_names = names[i:i+batch_size]

            try:
                df = D.features(
                    instruments=symbols,
                    fields=batch_fields,
                    start_time="2024-01-01",
                    end_time="2024-03-31"
                )
                print(f"  批次 {i//batch_size + 1}: {len(batch_fields)} 特征, shape={df.shape}")
            except Exception as e:
                print(f"  批次 {i//batch_size + 1}: 失败 - {e}")
                # 找出哪个特征有问题
                for j, (field, name) in enumerate(zip(batch_fields, batch_names)):
                    try:
                        df = D.features(
                            instruments=symbols[:5],
                            fields=[field],
                            start_time="2024-01-01",
                            end_time="2024-01-31"
                        )
                    except Exception as e2:
                        print(f"    问题特征: {name} - {e2}")

        print("✓ 分批加载完成")
        return True
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        traceback.print_exc()
        return False


def test_spawn_multiprocessing():
    """测试使用 spawn 而不是 fork"""
    print("\n" + "=" * 70)
    print("测试: 设置 multiprocessing start_method='spawn'")
    print("=" * 70)

    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("✓ 设置 multiprocessing start_method='spawn'")
    except RuntimeError as e:
        print(f"⚠ 无法设置 start_method: {e}")

    return test_qlib_kernels()


def main():
    print("=" * 70)
    print("调试 sp500 + alpha158-talib DataSet 初始化")
    print("目标: 重现 'free(): invalid pointer' 错误")
    print("=" * 70)

    # 显示系统信息
    import platform
    print(f"\n系统信息:")
    print(f"  Python: {platform.python_version()}")
    print(f"  Platform: {platform.platform()}")

    # 检查 TA-Lib
    try:
        import talib
        print(f"  TA-Lib: {talib.__version__}")
    except ImportError:
        print("  TA-Lib: 未安装")

    # 测试1: 原始流程
    print("\n" + "=" * 70)
    print("运行测试...")
    print("=" * 70)

    # 首先尝试 kernels=1 的方式
    success = test_qlib_kernels()

    if success:
        print("\n" + "=" * 70)
        print("✓ kernels=1 测试成功")
        print("=" * 70)
        print("\n建议: 在初始化 Qlib 时使用 kernels=1 参数")
        print("修改 scripts/models/common/training.py 中的 init_qlib 函数")
    else:
        print("\n" + "=" * 70)
        print("✗ kernels=1 测试也失败，问题可能更严重")
        print("=" * 70)

    # 测试顺序加载
    print("\n" + "-" * 70)
    test_sequential_loading()


if __name__ == "__main__":
    main()
