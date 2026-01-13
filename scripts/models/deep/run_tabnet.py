"""
运行 TabNet (Attentive Interpretable Tabular Learning) 模型

TabNet 是 Google 提出的专门为表格数据设计的深度学习模型，特点：
1. 内置注意力机制进行特征选择
2. 稀疏性学习，提高可解释性
3. 端到端学习，无需特征工程

使用方法:
    python scripts/models/deep/run_tabnet.py --stock-pool sp500 --handler alpha158 --nday 5 --backtest

安装依赖:
    pip install pytorch-tabnet
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import pickle
import torch
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor

from qlib.data.dataset.handler import DataHandlerLP

from utils.utils import evaluate_model
from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG, PROJECT_ROOT, MODEL_SAVE_PATH,
    create_argument_parser,
    get_time_splits,
    print_training_header,
    init_qlib,
    check_data_availability,
    create_data_handler,
    create_dataset,
    analyze_features,
    analyze_label_distribution,
    print_prediction_stats,
    run_backtest,
)


def add_tabnet_args(parser):
    """添加 TabNet 特定参数"""
    # 模型结构参数
    parser.add_argument('--n-d', type=int, default=64,
                        help='Width of the decision prediction layer (default: 64)')
    parser.add_argument('--n-a', type=int, default=64,
                        help='Width of the attention embedding (default: 64)')
    parser.add_argument('--n-steps', type=int, default=5,
                        help='Number of decision steps (default: 5)')
    parser.add_argument('--gamma', type=float, default=1.5,
                        help='Coefficient for feature reusage in attention (default: 1.5)')
    parser.add_argument('--n-independent', type=int, default=2,
                        help='Number of independent GLU layers (default: 2)')
    parser.add_argument('--n-shared', type=int, default=2,
                        help='Number of shared GLU layers (default: 2)')

    # 正则化参数
    parser.add_argument('--lambda-sparse', type=float, default=1e-4,
                        help='Sparsity regularization coefficient (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.3,
                        help='Momentum for batch normalization (default: 0.3)')

    # 训练参数
    parser.add_argument('--n-epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='Learning rate (default: 0.02)')
    parser.add_argument('--batch-size', type=int, default=4096,
                        help='Batch size (default: 4096)')
    parser.add_argument('--virtual-batch-size', type=int, default=512,
                        help='Virtual batch size for Ghost Batch Normalization (default: 512)')
    parser.add_argument('--early-stop', type=int, default=30,
                        help='Early stopping patience (default: 30)')

    # 其他参数
    parser.add_argument('--mask-type', type=str, default='entmax',
                        choices=['sparsemax', 'entmax'],
                        help='Mask type for attention (default: entmax)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')

    return parser


def prepare_data_for_tabnet(dataset, segment: str):
    """
    准备 TabNet 所需的数据格式

    Parameters
    ----------
    dataset : DatasetH
        Qlib 数据集
    segment : str
        数据段: 'train', 'valid', 'test'

    Returns
    -------
    tuple
        (X, y, index) - 特征矩阵, 标签数组 (2D), 索引
    """
    # 获取特征和标签
    features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
    labels = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)

    # 保存索引
    index = features.index

    # 处理 NaN 和异常值
    X = features.fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

    # TabNet 需要 2D 标签: (n_samples, n_outputs)
    y = labels.values.astype(np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # 处理标签中的 NaN
    y = np.nan_to_num(y, nan=0.0)

    return X, y, index


class TabNetWrapper:
    """
    TabNet 模型包装器，适配 Qlib 接口
    """

    def __init__(
        self,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
        n_independent: int = 2,
        n_shared: int = 2,
        lambda_sparse: float = 1e-4,
        momentum: float = 0.3,
        lr: float = 0.02,
        n_epochs: int = 200,
        batch_size: int = 4096,
        virtual_batch_size: int = 512,
        early_stop: int = 30,
        mask_type: str = 'entmax',
        device: str = 'auto',
        seed: int = 42,
    ):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lambda_sparse = lambda_sparse
        self.momentum = momentum
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.early_stop = early_stop
        self.mask_type = mask_type
        self.device = device
        self.seed = seed

        self.model = None
        self.fitted = False
        self.feature_importances_ = None

    def _create_model(self):
        """创建 TabNet 模型"""
        return TabNetRegressor(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            lambda_sparse=self.lambda_sparse,
            momentum=self.momentum,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=self.lr),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size": 20, "gamma": 0.9},
            mask_type=self.mask_type,
            device_name=self.device,
            seed=self.seed,
            verbose=1,
        )

    def fit(self, dataset):
        """
        训练模型

        Parameters
        ----------
        dataset : DatasetH
            Qlib 数据集
        """
        print("\n    Preparing training data...")
        X_train, y_train, _ = prepare_data_for_tabnet(dataset, "train")
        X_valid, y_valid, _ = prepare_data_for_tabnet(dataset, "valid")

        print(f"    Train shape: X={X_train.shape}, y={y_train.shape}")
        print(f"    Valid shape: X={X_valid.shape}, y={y_valid.shape}")

        # 创建模型
        print("\n    Creating TabNet model...")
        self.model = self._create_model()

        # 打印模型配置
        print(f"    Model config:")
        print(f"      n_d (decision width): {self.n_d}")
        print(f"      n_a (attention width): {self.n_a}")
        print(f"      n_steps: {self.n_steps}")
        print(f"      gamma: {self.gamma}")
        print(f"      lambda_sparse: {self.lambda_sparse}")
        print(f"      mask_type: {self.mask_type}")

        # 训练
        print("\n    Training...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_name=['valid'],
            eval_metric=['rmse', 'mae'],
            max_epochs=self.n_epochs,
            patience=self.early_stop,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
        )

        self.fitted = True
        self.feature_importances_ = self.model.feature_importances_

        print("\n    Training completed")
        print(f"    Best epoch: {self.model.best_epoch}")
        print(f"    Best valid RMSE: {self.model.best_cost:.6f}")

        return self

    def predict(self, dataset, segment: str = "test") -> pd.Series:
        """
        预测

        Parameters
        ----------
        dataset : DatasetH
            Qlib 数据集
        segment : str
            数据段

        Returns
        -------
        pd.Series
            预测结果
        """
        if not self.fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X, _, index = prepare_data_for_tabnet(dataset, segment)

        # 预测
        pred_values = self.model.predict(X)

        # 转换为 Series
        pred = pd.Series(pred_values.flatten(), index=index, name='score')

        return pred

    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        获取特征重要性

        Parameters
        ----------
        feature_names : list, optional
            特征名称列表

        Returns
        -------
        pd.DataFrame
            特征重要性 DataFrame
        """
        if self.feature_importances_ is None:
            raise ValueError("Model not fitted or feature importances not available.")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importances_))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df

    def save(self, path: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("No model to save")

        # TabNet 使用 zip 格式保存
        self.model.save_model(path)

        # 保存包装器的参数
        wrapper_params = {
            'n_d': self.n_d,
            'n_a': self.n_a,
            'n_steps': self.n_steps,
            'gamma': self.gamma,
            'n_independent': self.n_independent,
            'n_shared': self.n_shared,
            'lambda_sparse': self.lambda_sparse,
            'momentum': self.momentum,
            'lr': self.lr,
            'n_epochs': self.n_epochs,
            'batch_size': self.batch_size,
            'virtual_batch_size': self.virtual_batch_size,
            'early_stop': self.early_stop,
            'mask_type': self.mask_type,
            'feature_importances_': self.feature_importances_,
        }
        params_path = Path(path).with_suffix('.params.pkl')
        with open(params_path, 'wb') as f:
            pickle.dump(wrapper_params, f)

        print(f"    Model saved to: {path}")

    @classmethod
    def load(cls, path: str) -> 'TabNetWrapper':
        """加载模型"""
        path = Path(path)

        # 处理路径：TabNet 的 load_model 需要完整的 .zip 路径
        # 但 save_model 会自动添加 .zip，所以我们需要确保路径正确
        if path.suffix == '.zip':
            zip_path = path
            base_path = path.with_suffix('')  # 去掉 .zip
        else:
            base_path = path
            zip_path = Path(str(path) + '.zip')

        # 加载包装器参数
        params_path = base_path.with_suffix('.params.pkl')
        if params_path.exists():
            with open(params_path, 'rb') as f:
                wrapper_params = pickle.load(f)
        else:
            wrapper_params = {}

        # 创建实例
        instance = cls(**{k: v for k, v in wrapper_params.items()
                         if k not in ['feature_importances_']})

        # 加载 TabNet 模型
        instance.model = TabNetRegressor()
        instance.model.load_model(str(zip_path))
        instance.fitted = True
        instance.feature_importances_ = wrapper_params.get('feature_importances_')

        print(f"    Model loaded from: {zip_path}")
        return instance


def print_feature_importance(importance_df: pd.DataFrame, top_n: int = 20):
    """打印特征重要性"""
    print(f"\n    Top {top_n} Features by Importance:")
    print("    " + "-" * 60)
    print(f"    {'Rank':<6} {'Feature':<40} {'Importance':>12}")
    print("    " + "-" * 60)

    for rank, (_, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
        print(f"    {rank:<6} {row['feature']:<40} {row['importance']:>12.4f}")

    print("    " + "-" * 60)
    print(f"    Total features: {len(importance_df)}")

    # 统计非零重要性特征
    nonzero_count = (importance_df['importance'] > 0).sum()
    print(f"    Features with non-zero importance: {nonzero_count} ({nonzero_count/len(importance_df)*100:.1f}%)")


def main():
    # 解析命令行参数
    parser = create_argument_parser("TabNet", "run_tabnet.py")
    parser = add_tabnet_args(parser)
    args = parser.parse_args()

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]
    time_splits = get_time_splits(args.max_train)

    # 打印头部信息
    print_training_header("TabNet", args, symbols, handler_config, time_splits)

    # 初始化和数据准备
    init_qlib(handler_config['use_talib'])
    check_data_availability(time_splits)
    handler = create_data_handler(args, handler_config, symbols, time_splits)
    dataset = create_dataset(handler, time_splits)
    train_data, valid_cols, dropped_cols = analyze_features(dataset)
    analyze_label_distribution(dataset)

    # 获取实际的特征数量
    actual_train_data = dataset.prepare("train", col_set="feature")
    total_features = actual_train_data.shape[1]
    feature_names = actual_train_data.columns.tolist()
    print(f"\n    Actual training data shape: {actual_train_data.shape}")

    print(f"\n[6] Model Configuration:")
    print(f"    Total features: {total_features}")
    print(f"    n_d (decision width): {args.n_d}")
    print(f"    n_a (attention width): {args.n_a}")
    print(f"    n_steps: {args.n_steps}")
    print(f"    gamma: {args.gamma}")
    print(f"    lambda_sparse: {args.lambda_sparse}")
    print(f"    mask_type: {args.mask_type}")
    print(f"    Learning rate: {args.lr}")
    print(f"    Batch size: {args.batch_size}")
    print(f"    Virtual batch size: {args.virtual_batch_size}")
    print(f"    Epochs: {args.n_epochs}")
    print(f"    Early stop: {args.early_stop}")

    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'
    print(f"    Device: {device}")

    # 定义模型加载函数
    def load_model(path):
        return TabNetWrapper.load(path)

    def get_feature_count(m):
        return total_features

    # 检查是否提供了预训练模型路径
    if args.model_path:
        # 加载预训练模型，跳过训练
        model_path = Path(args.model_path)
        print(f"\n[7] Loading pre-trained model from: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = load_model(model_path)
        print("    Model loaded successfully")
    else:
        # 正常训练流程
        print("\n[7] Training TabNet model...")
        model = TabNetWrapper(
            n_d=args.n_d,
            n_a=args.n_a,
            n_steps=args.n_steps,
            gamma=args.gamma,
            n_independent=args.n_independent,
            n_shared=args.n_shared,
            lambda_sparse=args.lambda_sparse,
            momentum=args.momentum,
            lr=args.lr,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            virtual_batch_size=args.virtual_batch_size,
            early_stop=args.early_stop,
            mask_type=args.mask_type,
            device=device,
        )

        # 训练
        model.fit(dataset)
        print("    Model training completed")

        # 特征重要性分析
        print("\n[8] Feature Importance Analysis...")
        importance_df = model.get_feature_importance(feature_names)
        print_feature_importance(importance_df)

        # 保存模型
        print("\n[9] Saving model...")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_SAVE_PATH / f"tabnet_{args.handler}_{args.stock_pool}_{args.nday}d"
        model.save(str(model_path))

        # 保存特征重要性
        importance_path = MODEL_SAVE_PATH / f"tabnet_{args.handler}_{args.stock_pool}_{args.nday}d_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"    Feature importance saved to: {importance_path}")

    # 预测
    print("\n[10] Generating predictions...")
    test_pred = model.predict(dataset, segment="test")

    # 检查预测结果
    print(f"    Predictions shape: {test_pred.shape}")
    print(f"    Predictions NaN count: {test_pred.isna().sum()} ({test_pred.isna().sum() / len(test_pred) * 100:.2f}%)")
    if not test_pred.isna().all():
        print(f"    Predictions min/max: {test_pred.min():.4f} / {test_pred.max():.4f}")
    print_prediction_stats(test_pred)

    # 评估
    print("\n[11] Evaluation...")
    evaluate_model(dataset, test_pred, PROJECT_ROOT, args.nday)

    # 回测
    if args.backtest:
        pred_df = test_pred.to_frame("score")

        run_backtest(
            model_path, dataset, pred_df, args, time_splits,
            model_name="TabNet",
            load_model_func=load_model,
            get_feature_count_func=get_feature_count
        )


if __name__ == "__main__":
    main()
