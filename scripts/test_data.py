"""
测试 Qlib 数据是否正确加载
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "qlib"))

import qlib
from qlib.data import D
from qlib.constant import REG_US


if __name__ == '__main__':
    # 初始化 Qlib
    data_path = PROJECT_ROOT / "my_data" / "qlib_us"
    qlib.init(provider_uri=str(data_path), region=REG_US)

    print("=" * 60)
    print("Qlib Data Test")
    print("=" * 60)

    # 测试1: 获取交易日历
    print("\n1. Trading Calendar (first 10 days):")
    calendar = D.calendar(start_time='2000-01-01', end_time='2000-01-31', freq='day')
    print(calendar[:10])

    # 测试2: 获取股票列表
    print("\n2. Available Instruments:")
    instruments = D.instruments('all')
    stocks = D.list_instruments(instruments=instruments, start_time='2024-01-01', end_time='2024-12-31', as_list=True)
    print(f"Total stocks: {len(stocks)}")
    print(f"First 10: {stocks[:10]}")

    # 测试3: 读取 AAPL 的数据
    print("\n3. AAPL Data (latest 5 days):")
    instruments = ['AAPL']
    fields = ['$close', '$volume', 'Ref($close, 1)', '$high/$low']
    df = D.features(instruments, fields, start_time='2024-12-01', end_time='2024-12-31', freq='day')
    print(df.tail())

    # 测试4: 读取多只股票
    print("\n4. Multiple Stocks (AAPL, MSFT, NVDA) close prices:")
    instruments = ['AAPL', 'MSFT', 'NVDA']
    fields = ['$close']
    df = D.features(instruments, fields, start_time='2024-12-01', end_time='2024-12-31', freq='day')
    print(df.tail(15))

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
