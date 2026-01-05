"""
新闻特征使用示例

演示如何:
1. 下载新闻数据
2. 处理新闻生成特征
3. 使用带新闻特征的数据处理器
"""

import sys
from pathlib import Path

# 项目路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def example_download_news():
    """
    示例 1: 下载新闻数据

    需要先设置环境变量: export FINNHUB_API_KEY="your_api_key"
    获取免费 API 密钥: https://finnhub.io/register
    """
    print("\n" + "=" * 50)
    print("示例 1: 下载新闻数据")
    print("=" * 50)

    from news.download_news import download_all_news, DEFAULT_SYMBOLS

    # 只下载几只测试股票
    test_symbols = ["AAPL", "MSFT", "NVDA"]

    print(f"下载股票: {test_symbols}")
    print("日期范围: 2024-01-01 至 2024-01-31 (测试)")

    # 下载新闻
    # 注意: 需要设置 FINNHUB_API_KEY 环境变量
    try:
        df = download_all_news(
            symbols=test_symbols,
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
        print(f"下载完成! 共 {len(df)} 条新闻")
        if not df.empty:
            print("\n前 5 条新闻:")
            print(df[["datetime", "symbol", "headline"]].head())
    except Exception as e:
        print(f"下载失败 (可能是 API 密钥未设置): {e}")
        print("请设置环境变量: export FINNHUB_API_KEY='your_api_key'")


def example_process_news():
    """
    示例 2: 处理新闻数据生成特征

    需要先运行示例 1 下载新闻数据
    """
    print("\n" + "=" * 50)
    print("示例 2: 处理新闻数据")
    print("=" * 50)

    from news.process_news import process_all_news, load_news_data
    from news.process_news import NEWS_CSV_DIR, NEWS_PROCESSED_DIR

    # 检查是否有新闻数据
    news_df = load_news_data(NEWS_CSV_DIR)
    if news_df.empty:
        print("未找到新闻数据，请先运行示例 1")
        return

    print(f"加载 {len(news_df)} 条新闻")

    # 处理新闻生成特征
    # 使用 VADER (快速) 进行情感分析
    features_df = process_all_news(
        input_dir=NEWS_CSV_DIR,
        output_dir=NEWS_PROCESSED_DIR,
        sentiment_method="vader",  # 或 "finbert" (更准确但需要 GPU)
    )

    if not features_df.empty:
        print(f"\n生成特征完成! 形状: {features_df.shape}")
        print("\n特征列:")
        for col in features_df.columns:
            print(f"  - {col}")
        print("\n前 5 行:")
        print(features_df.head())


def example_use_handler():
    """
    示例 3: 使用带新闻特征的数据处理器

    需要先运行示例 2 处理新闻数据
    """
    print("\n" + "=" * 50)
    print("示例 3: 使用带新闻特征的数据处理器")
    print("=" * 50)

    import qlib
    from qlib.constant import REG_US

    # 初始化 Qlib
    qlib_data_path = PROJECT_ROOT / "my_data" / "qlib_us"
    if not qlib_data_path.exists():
        print(f"Qlib 数据目录不存在: {qlib_data_path}")
        print("请先下载市场数据")
        return

    # 注册 TA-Lib 操作符
    from utils.talib_ops import TALIB_OPS
    qlib.init(provider_uri=str(qlib_data_path), region=REG_US, custom_ops=TALIB_OPS)

    # 创建数据处理器
    from data.datahandler_news import Alpha158_Volatility_TALib_News

    news_path = PROJECT_ROOT / "my_data" / "news_processed" / "news_features.parquet"

    handler = Alpha158_Volatility_TALib_News(
        volatility_window=2,
        instruments=["AAPL", "MSFT", "NVDA"],
        start_time="2024-01-01",
        end_time="2024-12-31",
        fit_start_time="2024-01-01",
        fit_end_time="2024-06-30",
        news_data_path=str(news_path) if news_path.exists() else None,
        news_features="core",  # 使用核心新闻特征
        add_news_rolling=True,  # 添加滚动特征
    )

    # 获取数据
    print("\n数据处理器创建成功!")
    print(f"新闻特征: {handler.get_news_feature_names()}")

    # 创建数据集
    from qlib.data.dataset import DatasetH

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": ("2024-01-01", "2024-06-30"),
            "valid": ("2024-07-01", "2024-09-30"),
            "test": ("2024-10-01", "2024-12-31"),
        },
    )

    # 获取训练数据
    train_data = dataset.prepare("train", col_set=["feature", "label"])
    print(f"\n训练数据形状: {train_data.shape}")
    print(f"特征数量: {train_data.shape[1] - 1}")  # 减去标签列


def example_train_model():
    """
    示例 4: 使用带新闻特征的数据训练模型
    """
    print("\n" + "=" * 50)
    print("示例 4: 训练模型 (简化版)")
    print("=" * 50)

    print("""
要使用带新闻特征的数据训练模型，可以修改现有的训练脚本:

1. 在 run_lgb_nd.py 中:

   # 导入新数据处理器
   from data.datahandler_news import Alpha158_Volatility_TALib_News

   # 替换原来的处理器
   handler = Alpha158_Volatility_TALib_News(
       volatility_window=VOLATILITY_WINDOW,
       instruments=TEST_SYMBOLS,
       start_time=TRAIN_START,
       end_time=TEST_END,
       fit_start_time=TRAIN_START,
       fit_end_time=TRAIN_END,
       news_data_path="my_data/news_processed/news_features.parquet",
       news_features="core",  # 或 "all"
       add_news_rolling=True,
   )

2. 其他代码保持不变，模型会自动学习新闻特征

3. 可以通过特征重要性分析查看新闻特征的贡献:
   - news_sentiment_score: 情感分数
   - news_count_log: 新闻数量
   - news_bull_bear_ratio: 多空关键词比率
    """)


def main():
    """运行所有示例"""
    print("=" * 60)
    print("新闻特征使用示例")
    print("=" * 60)

    print("""
本示例演示如何在量化交易系统中集成新闻数据。

运行顺序:
1. 下载新闻数据 (需要 Finnhub API 密钥)
2. 处理新闻生成特征
3. 使用带新闻特征的数据处理器
4. 训练模型

请根据需要运行各个示例函数。
    """)

    # 可以取消注释以运行各个示例
    # example_download_news()
    # example_process_news()
    # example_use_handler()
    example_train_model()


if __name__ == "__main__":
    main()
