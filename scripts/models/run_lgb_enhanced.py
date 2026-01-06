"""
增强版 LightGBM 训练脚本

解决数据不足问题的5种策略:
1. 增强正则化，防止过拟合
2. 扩大股票池
3. 特征选择，减少维度
4. 时间序列交叉验证
5. 迁移学习：两阶段训练
"""

import sys
from pathlib import Path
import argparse
import pickle
import warnings

warnings.filterwarnings("ignore")

# Add scripts directory to path for imports
script_dir = Path(__file__).parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_US
from qlib.data import D
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

# Import TA-Lib custom operators
from utils.talib_ops import TALIB_OPS
from utils.utils import evaluate_model

# Import extended data handlers
from data.datahandler_ext import Alpha158_Volatility_TALib
from data.datahandler_news import Alpha158_Volatility_TALib_News


# ========== 配置 ==========

PROJECT_ROOT = Path(__file__).parent.parent.parent
QLIB_DATA_PATH = PROJECT_ROOT / "my_data" / "qlib_us"
NEWS_DATA_PATH = PROJECT_ROOT / "my_data" / "news_processed" / "news_features.parquet"
MODEL_SAVE_PATH = PROJECT_ROOT / "my_models"

# ========== 股票池配置 ==========

# 原始小股票池 (10只)
SMALL_POOL = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]

# 扩展股票池 (30只) - 包含更多行业
MEDIUM_POOL = [
    # 科技股 (Magnificent 7 + 半导体)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "INTC", "AVGO", "QCOM", "MU",
    # 金融
    "JPM", "V", "MA", "BAC", "GS",
    # 医疗
    "JNJ", "UNH", "PFE", "ABBV", "MRK",
    # 消费
    "WMT", "PG", "KO", "PEP", "MCD",
    # 通信/媒体
    "DIS", "NFLX", "CMCSA",
]

# 大股票池 (SP100 - 100只)
LARGE_POOL = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "C", "CAT",
    "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX",
    "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX", "GD",
    "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC",
    "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD",
    "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE",
    "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM",
    "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA",
    "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM",
]

# 时间划分 (适用于1年新闻数据)
TRAIN_START = "2025-01-01"
TRAIN_END = "2025-09-30"
VALID_START = "2025-10-01"
VALID_END = "2025-11-30"
TEST_START = "2025-12-01"
TEST_END = "2025-12-31"

# 预训练时间段 (无新闻数据，用于迁移学习)
PRETRAIN_START = "2015-01-01"
PRETRAIN_END = "2024-12-31"


# ========== 1. 正则化配置 ==========

def get_regularized_model_params(regularization_level="medium"):
    """
    获取正则化的模型参数

    Args:
        regularization_level: "light", "medium", "strong"

    Returns:
        dict: LightGBM 参数
    """
    base_params = {
        "loss": "mse",
        "num_threads": 4,
        "verbose": -1,
    }

    if regularization_level == "light":
        # 轻度正则化 - 适用于数据量适中的情况
        return {
            **base_params,
            "learning_rate": 0.03,
            "max_depth": 6,
            "num_leaves": 48,
            "min_data_in_leaf": 30,
            "lambda_l1": 0.01,
            "lambda_l2": 0.1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.9,
            "bagging_freq": 3,
            "n_estimators": 300,
            "early_stopping_rounds": 40,
        }
    elif regularization_level == "medium":
        # 中度正则化 - 适用于数据量较少的情况
        return {
            **base_params,
            "learning_rate": 0.01,
            "max_depth": 4,
            "num_leaves": 16,
            "min_data_in_leaf": 50,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "n_estimators": 500,
            "early_stopping_rounds": 50,
        }
    else:  # strong
        # 强正则化 - 适用于数据量很少的情况
        return {
            **base_params,
            "learning_rate": 0.005,
            "max_depth": 3,
            "num_leaves": 8,
            "min_data_in_leaf": 100,
            "lambda_l1": 0.5,
            "lambda_l2": 5.0,
            "feature_fraction": 0.4,
            "bagging_fraction": 0.6,
            "bagging_freq": 5,
            "n_estimators": 800,
            "early_stopping_rounds": 80,
        }


# ========== 2. 特征选择 ==========

class FeatureSelector:
    """基于模型重要性的特征选择器"""

    def __init__(self, n_features=50, importance_type="gain"):
        """
        Args:
            n_features: 保留的特征数量
            importance_type: "gain" 或 "split"
        """
        self.n_features = n_features
        self.importance_type = importance_type
        self.selected_features = None
        self.feature_importance = None

    def fit(self, model, feature_names):
        """
        基于训练好的模型选择特征

        Args:
            model: 训练好的 LGBModel
            feature_names: 训练时使用的特征名称列表
        """
        # 获取特征重要性
        importance = model.model.feature_importance(importance_type=self.importance_type)

        # 验证特征数量匹配
        if len(feature_names) != len(importance):
            raise ValueError(
                f"Feature names ({len(feature_names)}) and importance ({len(importance)}) length mismatch"
            )

        # 创建重要性 DataFrame
        self.feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)

        # 选择 top N 特征
        self.selected_features = self.feature_importance.head(self.n_features)["feature"].tolist()

        return self

    def transform(self, df):
        """筛选特征"""
        if self.selected_features is None:
            raise ValueError("必须先调用 fit() 方法")

        available_features = [f for f in self.selected_features if f in df.columns]
        return df[available_features]

    def get_importance_report(self, top_n=20):
        """获取特征重要性报告"""
        if self.feature_importance is None:
            return "未执行特征选择"

        report = self.feature_importance.head(top_n).to_string()
        return report


# ========== 3. 时间序列交叉验证 ==========

class TimeSeriesCrossValidator:
    """时间序列交叉验证器"""

    def __init__(self, n_splits=3, test_ratio=0.15, gap_days=5):
        """
        Args:
            n_splits: 交叉验证折数
            test_ratio: 每折测试集占比
            gap_days: 训练集和测试集之间的间隔天数（避免信息泄露）
        """
        self.n_splits = n_splits
        self.test_ratio = test_ratio
        self.gap_days = gap_days

    def split(self, df):
        """
        生成时间序列交叉验证的分割

        Args:
            df: 带有 datetime index 的 DataFrame

        Yields:
            (train_idx, val_idx): 训练集和验证集的索引
        """
        # 获取唯一日期
        dates = df.index.get_level_values("datetime").unique().sort_values()
        n_dates = len(dates)

        # 每折的测试集大小
        test_size = max(int(n_dates * self.test_ratio), 5)

        # 计算起始位置，确保每折都有足够的训练数据
        min_train_size = int(n_dates * 0.4)

        for i in range(self.n_splits):
            # 计算当前折的测试集结束位置
            test_end_idx = n_dates - i * test_size
            test_start_idx = test_end_idx - test_size
            train_end_idx = test_start_idx - self.gap_days

            if train_end_idx < min_train_size:
                break

            # 获取日期范围
            train_dates = dates[:train_end_idx]
            test_dates = dates[test_start_idx:test_end_idx]

            # 获取索引
            train_mask = df.index.get_level_values("datetime").isin(train_dates)
            test_mask = df.index.get_level_values("datetime").isin(test_dates)

            yield np.where(train_mask)[0], np.where(test_mask)[0]

    def cross_validate(self, model_params, X, y, feature_names=None):
        """
        执行交叉验证

        Returns:
            dict: 交叉验证结果
        """
        results = {
            "fold_scores": [],
            "fold_models": [],
            "feature_importance": None,
        }

        all_importance = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.split(X)):
            print(f"\n    Fold {fold_idx + 1}/{self.n_splits}:")
            print(f"      Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # 创建临时 dataset 结构供 LGBModel 使用
            # 这里使用简化的训练方式
            import lightgbm as lgb

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            # 训练
            lgb_params = {k: v for k, v in model_params.items()
                        if k not in ["n_estimators", "early_stopping_rounds"]}
            lgb_params["objective"] = "regression"
            lgb_params["metric"] = "mse"

            callbacks = [lgb.early_stopping(model_params.get("early_stopping_rounds", 50))]

            model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=model_params.get("n_estimators", 500),
                valid_sets=[val_data],
                callbacks=callbacks,
            )

            # 预测和评估
            val_pred = model.predict(X_val)
            mse = np.mean((val_pred - y_val.values.flatten()) ** 2)
            mae = np.mean(np.abs(val_pred - y_val.values.flatten()))

            # IC
            ic = pd.Series(val_pred, index=y_val.index).groupby(level="datetime").apply(
                lambda x: x.corr(y_val.loc[x.index].iloc[:, 0]) if len(x) > 1 else np.nan
            ).mean()

            print(f"      MSE: {mse:.6f}, MAE: {mae:.6f}, IC: {ic:.4f}")

            results["fold_scores"].append({
                "mse": mse,
                "mae": mae,
                "ic": ic,
            })
            results["fold_models"].append(model)

            # 累积特征重要性
            if feature_names is not None:
                importance = model.feature_importance(importance_type="gain")
                all_importance.append(importance)

        # 计算平均特征重要性
        if all_importance:
            avg_importance = np.mean(all_importance, axis=0)
            results["feature_importance"] = pd.DataFrame({
                "feature": feature_names,
                "importance": avg_importance
            }).sort_values("importance", ascending=False)

        # 汇总统计
        avg_mse = np.mean([s["mse"] for s in results["fold_scores"]])
        avg_mae = np.mean([s["mae"] for s in results["fold_scores"]])
        avg_ic = np.mean([s["ic"] for s in results["fold_scores"]])

        print(f"\n    Cross-validation Summary:")
        print(f"      Avg MSE: {avg_mse:.6f} (+/- {np.std([s['mse'] for s in results['fold_scores']]):.6f})")
        print(f"      Avg MAE: {avg_mae:.6f}")
        print(f"      Avg IC: {avg_ic:.4f}")

        results["avg_scores"] = {"mse": avg_mse, "mae": avg_mae, "ic": avg_ic}

        return results


# ========== 4. 迁移学习 ==========

class TransferLearningTrainer:
    """两阶段迁移学习训练器"""

    def __init__(self, pretrain_params, finetune_params, n_top_features=80):
        """
        Args:
            pretrain_params: 预训练阶段的模型参数
            finetune_params: 微调阶段的模型参数
            n_top_features: 微调时保留的 top 特征数
        """
        self.pretrain_params = pretrain_params
        self.finetune_params = finetune_params
        self.n_top_features = n_top_features
        self.pretrained_model = None
        self.finetuned_model = None
        self.selected_features = None

    def pretrain(self, dataset):
        """
        阶段1: 使用历史数据预训练

        Args:
            dataset: Qlib DatasetH (长期历史数据，无新闻)
        """
        print("\n" + "=" * 60)
        print("阶段1: 预训练 (使用长期历史数据)")
        print("=" * 60)

        model = LGBModel(**self.pretrain_params)
        model.fit(dataset)

        self.pretrained_model = model

        # 获取特征重要性，用于特征选择
        train_data = dataset.prepare("train", col_set="feature")
        feature_names = train_data.columns.tolist()
        importance = model.model.feature_importance(importance_type="gain")

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)

        # 选择 top 特征
        self.selected_features = importance_df.head(self.n_top_features)["feature"].tolist()

        print(f"\n预训练完成，选择了 {len(self.selected_features)} 个重要特征")
        print(f"Top 10 特征: {self.selected_features[:10]}")

        return self

    def finetune(self, dataset, use_pretrain_init=True):
        """
        阶段2: 使用目标数据微调

        Args:
            dataset: Qlib DatasetH (包含新闻的短期数据)
            use_pretrain_init: 是否使用预训练模型初始化
        """
        print("\n" + "=" * 60)
        print("阶段2: 微调 (使用目标域数据)")
        print("=" * 60)

        # 准备数据
        train_features = dataset.prepare("train", col_set="feature")
        train_label = dataset.prepare("train", col_set="label")
        valid_features = dataset.prepare("valid", col_set="feature")
        valid_label = dataset.prepare("valid", col_set="label")

        # 筛选预训练选出的特征 (如果有)
        if self.selected_features:
            # 保留预训练选出的特征 + 所有新闻特征
            selected = [f for f in self.selected_features if f in train_features.columns]
            news_features = [f for f in train_features.columns if f.startswith("news_")]
            final_features = list(set(selected + news_features))

            print(f"使用 {len(selected)} 个预训练特征 + {len(news_features)} 个新闻特征")

            train_features = train_features[final_features]
            valid_features = valid_features[final_features]

        print(f"微调数据形状: {train_features.shape}")

        # 使用 lightgbm 直接训练（更灵活）
        import lightgbm as lgb

        train_data = lgb.Dataset(train_features, label=train_label.values.flatten())
        val_data = lgb.Dataset(valid_features, label=valid_label.values.flatten(), reference=train_data)

        lgb_params = {k: v for k, v in self.finetune_params.items()
                     if k not in ["n_estimators", "early_stopping_rounds", "loss"]}
        lgb_params["objective"] = "regression"
        lgb_params["metric"] = "mse"

        callbacks = [
            lgb.early_stopping(self.finetune_params.get("early_stopping_rounds", 50)),
            lgb.log_evaluation(period=50),
        ]

        # 如果使用预训练初始化
        init_model = None
        if use_pretrain_init and self.pretrained_model is not None:
            # 注意：直接使用预训练模型需要特征完全匹配
            # 这里我们选择不使用 init_model，而是依靠特征选择
            pass

        self.finetuned_model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=self.finetune_params.get("n_estimators", 500),
            valid_sets=[val_data],
            callbacks=callbacks,
            init_model=init_model,
        )

        # 验证集评估
        val_pred = self.finetuned_model.predict(valid_features)
        mse = np.mean((val_pred - valid_label.values.flatten()) ** 2)
        mae = np.mean(np.abs(val_pred - valid_label.values.flatten()))

        print(f"\n微调完成:")
        print(f"  验证集 MSE: {mse:.6f}")
        print(f"  验证集 MAE: {mae:.6f}")

        return self

    def predict(self, dataset, segment="test"):
        """使用微调后的模型预测"""
        features = dataset.prepare(segment, col_set="feature")

        # 筛选特征
        if self.selected_features:
            selected = [f for f in self.selected_features if f in features.columns]
            news_features = [f for f in features.columns if f.startswith("news_")]
            final_features = list(set(selected + news_features))
            features = features[final_features]

        pred = self.finetuned_model.predict(features)
        return pd.Series(pred, index=features.index)

    def save(self, path):
        """保存模型"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "transfer_model.pkl", "wb") as f:
            pickle.dump({
                "pretrained_model": self.pretrained_model,
                "finetuned_model": self.finetuned_model,
                "selected_features": self.selected_features,
            }, f)

        print(f"模型已保存到: {path}")

    @classmethod
    def load(cls, path):
        """加载模型"""
        with open(Path(path) / "transfer_model.pkl", "rb") as f:
            data = pickle.load(f)

        trainer = cls({}, {})
        trainer.pretrained_model = data["pretrained_model"]
        trainer.finetuned_model = data["finetuned_model"]
        trainer.selected_features = data["selected_features"]

        return trainer


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description="Enhanced LightGBM Training with Data Scarcity Solutions")

    # 基本参数
    parser.add_argument("--nday", type=int, default=2, help="Volatility prediction window")
    parser.add_argument("--use-news", action="store_true", help="Use news features")
    parser.add_argument("--news-features", type=str, default="core", choices=["all", "sentiment", "stats", "core"])

    # 改进策略
    parser.add_argument("--regularization", type=str, default="medium",
                       choices=["light", "medium", "strong"],
                       help="Regularization level")
    parser.add_argument("--stock-pool", type=str, default="small",
                       choices=["small", "medium", "large"],
                       help="Stock pool size")
    parser.add_argument("--feature-selection", action="store_true",
                       help="Enable feature selection")
    parser.add_argument("--n-features", type=int, default=50,
                       help="Number of features to select")
    parser.add_argument("--cross-validation", action="store_true",
                       help="Enable time series cross-validation")
    parser.add_argument("--cv-folds", type=int, default=3,
                       help="Number of CV folds")
    parser.add_argument("--transfer-learning", action="store_true",
                       help="Enable two-stage transfer learning")

    args = parser.parse_args()

    # 选择股票池
    if args.stock_pool == "small":
        symbols = SMALL_POOL
    elif args.stock_pool == "medium":
        symbols = MEDIUM_POOL
    else:
        symbols = LARGE_POOL

    print("=" * 70)
    print("Enhanced LightGBM Training")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  - Volatility window: {args.nday} days")
    print(f"  - Stock pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"  - Regularization: {args.regularization}")
    print(f"  - Feature selection: {args.feature_selection} (n={args.n_features})")
    print(f"  - Cross-validation: {args.cross_validation} (folds={args.cv_folds})")
    print(f"  - Transfer learning: {args.transfer_learning}")
    print(f"  - News features: {args.use_news} ({args.news_features})")

    # 初始化 Qlib
    print("\n[1] Initializing Qlib...")
    qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US, custom_ops=TALIB_OPS)

    # 检查可用股票
    available_instruments = list(D.list_instruments(D.instruments(market="all")))
    symbols = [s for s in symbols if s in available_instruments]
    print(f"    Available stocks: {len(symbols)}")

    # 获取模型参数
    model_params = get_regularized_model_params(args.regularization)
    print(f"\n[2] Model Parameters ({args.regularization} regularization):")
    for k, v in model_params.items():
        print(f"    {k}: {v}")

    # ========== 策略5: 迁移学习 ==========
    if args.transfer_learning:
        print("\n" + "=" * 70)
        print("Running Transfer Learning (Two-Stage Training)")
        print("=" * 70)

        # 阶段1: 预训练 (使用长期历史数据，无新闻)
        print("\n[3] Creating pretrain dataset (2020-2024, no news)...")

        pretrain_handler = Alpha158_Volatility_TALib(
            volatility_window=args.nday,
            instruments=symbols,
            start_time=PRETRAIN_START,
            end_time=PRETRAIN_END,
            fit_start_time=PRETRAIN_START,
            fit_end_time="2023-12-31",
            infer_processors=[],
        )

        pretrain_dataset = DatasetH(
            handler=pretrain_handler,
            segments={
                "train": (PRETRAIN_START, "2023-12-31"),
                "valid": ("2024-01-01", "2024-06-30"),
                "test": ("2024-07-01", PRETRAIN_END),
            }
        )

        pretrain_data = pretrain_dataset.prepare("train", col_set="feature")
        print(f"    Pretrain data shape: {pretrain_data.shape}")

        # 阶段2: 微调数据集 (2025年，包含新闻)
        print("\n[4] Creating finetune dataset (2025, with news)...")

        if args.use_news:
            finetune_handler = Alpha158_Volatility_TALib_News(
                volatility_window=args.nday,
                instruments=symbols,
                start_time=TRAIN_START,
                end_time=TEST_END,
                fit_start_time=TRAIN_START,
                fit_end_time=TRAIN_END,
                infer_processors=[],
                news_data_path=str(NEWS_DATA_PATH) if NEWS_DATA_PATH.exists() else None,
                news_features=args.news_features,
            )
        else:
            finetune_handler = Alpha158_Volatility_TALib(
                volatility_window=args.nday,
                instruments=symbols,
                start_time=TRAIN_START,
                end_time=TEST_END,
                fit_start_time=TRAIN_START,
                fit_end_time=TRAIN_END,
                infer_processors=[],
            )

        finetune_dataset = DatasetH(
            handler=finetune_handler,
            segments={
                "train": (TRAIN_START, TRAIN_END),
                "valid": (VALID_START, VALID_END),
                "test": (TEST_START, TEST_END),
            }
        )

        finetune_data = finetune_dataset.prepare("train", col_set="feature")
        print(f"    Finetune data shape: {finetune_data.shape}")

        # 迁移学习训练
        pretrain_params = get_regularized_model_params("light")  # 预训练用轻度正则化
        finetune_params = get_regularized_model_params(args.regularization)

        trainer = TransferLearningTrainer(
            pretrain_params=pretrain_params,
            finetune_params=finetune_params,
            n_top_features=args.n_features,
        )

        trainer.pretrain(pretrain_dataset)
        trainer.finetune(finetune_dataset)

        # 预测
        print("\n[5] Generating predictions...")
        test_pred = trainer.predict(finetune_dataset, segment="test")

        # 评估
        evaluate_model(finetune_dataset, test_pred, PROJECT_ROOT, args.nday)

        # 保存模型
        trainer.save(MODEL_SAVE_PATH / "transfer_learning")

        return

    # ========== 非迁移学习模式 ==========

    # 创建数据集
    print("\n[3] Creating dataset...")

    if args.use_news:
        handler = Alpha158_Volatility_TALib_News(
            volatility_window=args.nday,
            instruments=symbols,
            start_time=TRAIN_START,
            end_time=TEST_END,
            fit_start_time=TRAIN_START,
            fit_end_time=TRAIN_END,
            infer_processors=[],
            news_data_path=str(NEWS_DATA_PATH) if NEWS_DATA_PATH.exists() else None,
            news_features=args.news_features,
        )
    else:
        handler = Alpha158_Volatility_TALib(
            volatility_window=args.nday,
            instruments=symbols,
            start_time=TRAIN_START,
            end_time=TEST_END,
            fit_start_time=TRAIN_START,
            fit_end_time=TRAIN_END,
            infer_processors=[],
        )

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (TRAIN_START, TRAIN_END),
            "valid": (VALID_START, VALID_END),
            "test": (TEST_START, TEST_END),
        }
    )

    # 使用 DK_L 与模型训练时保持一致
    train_features = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
    print(f"    Train data shape: {train_features.shape}")
    print(f"    Features: {train_features.shape[1]}")

    # ========== 策略4: 时间序列交叉验证 ==========
    if args.cross_validation:
        print("\n[4] Running Time Series Cross-Validation...")

        cv = TimeSeriesCrossValidator(n_splits=args.cv_folds, test_ratio=0.15, gap_days=5)

        # 合并训练和验证数据用于 CV
        all_features = pd.concat([
            dataset.prepare("train", col_set="feature"),
            dataset.prepare("valid", col_set="feature"),
        ])
        all_labels = pd.concat([
            dataset.prepare("train", col_set="label"),
            dataset.prepare("valid", col_set="label"),
        ])

        cv_results = cv.cross_validate(
            model_params,
            all_features,
            all_labels,
            feature_names=all_features.columns.tolist()
        )

        # 使用 CV 选出的特征重要性进行特征选择
        if args.feature_selection and cv_results["feature_importance"] is not None:
            print(f"\n[5] Feature Selection based on CV importance...")
            top_features = cv_results["feature_importance"].head(args.n_features)["feature"].tolist()
            print(f"    Selected {len(top_features)} features")
            print(f"    Top 10: {top_features[:10]}")

    # ========== 常规训练 ==========
    print("\n[6] Training final model...")

    model = LGBModel(**model_params)
    model.fit(dataset)

    # ========== 策略3: 特征选择 (非CV模式) ==========
    if args.feature_selection and not args.cross_validation:
        print("\n[7] Feature Selection...")
        selector = FeatureSelector(n_features=args.n_features)
        selector.fit(model, train_features.columns.tolist())

        print(f"    Selected {len(selector.selected_features)} features")
        print(f"\n    Top 20 features by importance:")
        print(selector.get_importance_report(20))

    # 预测
    print("\n[8] Generating predictions...")

    # 获取训练时实际使用的特征数量
    n_train_features = model.model.num_feature()

    # 准备测试数据
    x_test = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)

    # 检查特征数量差异
    if x_test.shape[1] != n_train_features:
        print(f"    Warning: Test features ({x_test.shape[1]}) != Training features ({n_train_features})")
        print(f"    Using training feature columns to filter test data...")
        # 使用训练数据的列名来筛选测试数据
        # train_features 在前面已经获取，包含训练时的特征列
        x_test = x_test[train_features.columns]

    # 使用过滤后的特征进行预测
    test_pred = pd.Series(model.model.predict(x_test.values), index=x_test.index)
    test_pred = test_pred.loc[TEST_START:TEST_END]

    # 评估
    evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

    print("\n" + "=" * 70)
    print("Training Completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
