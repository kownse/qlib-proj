"""
XGBoost 快速评估脚本 - 使用固定超参数

用于快速迭代特征工程:
1. 使用已知的好超参数，跳过超参数搜索
2. 运行 CV 训练，输出 CV Mean IC
3. 输出特征重要性报告

使用方法:
    # 使用默认超参数
    python scripts/models/tree/run_xgboost_quick_eval.py --handler alpha158-enhanced-v3

    # 指定超参数
    python scripts/models/tree/run_xgboost_quick_eval.py --handler alpha158-enhanced-v3 \
        --lr 0.05 --depth 6 --l2 1.0

    # 从 JSON 文件加载超参数
    python scripts/models/tree/run_xgboost_quick_eval.py --handler alpha158-enhanced-v3 \
        --params-file outputs/hyperopt_cv/xgboost_cv_best_params_xxx.json
"""

# 导入公共模块（会设置环境变量和初始化路径）
from quick_eval_common import (
    init_qlib_for_quick_eval,
    CV_FOLDS,
    compute_ic,
    print_feature_importance,
    print_final_summary,
    add_common_args,
    load_params_from_file,
    save_importance_to_csv,
    run_cv_evaluation_generic,
    run_shuffle_test,
)

# 初始化 Qlib
init_qlib_for_quick_eval()

# 重置 OMP_NUM_THREADS 以允许 XGBoost 使用多线程
# (quick_eval_common 为了 TA-Lib 兼容性设置为 '1'，但 XGBoost 需要多线程)
import os
import multiprocessing
NUM_THREADS = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)

# 现在可以安全地导入其他模块
import argparse
import numpy as np
import xgboost as xgb

from data.stock_pools import STOCK_POOLS
from models.common import HANDLER_CONFIG, init_qlib


# ============================================================================
# 默认超参数 (从之前的 hyperopt 搜索中获得)
# ============================================================================

DEFAULT_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 1,
    'reg_lambda': 1.0,  # L2 regularization
    'reg_alpha': 0.0,   # L1 regularization
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'nthread': NUM_THREADS,
    'verbosity': 0,
    'seed': 42,
}

DEFAULT_NUM_BOOST_ROUND = 1000
DEFAULT_EARLY_STOPPING_ROUNDS = 50


# ============================================================================
# XGBoost 特定函数
# ============================================================================

def train_and_predict_xgboost(fold_data: dict, xgb_params: dict):
    """XGBoost 训练和预测函数"""
    # 创建 DMatrix
    dtrain = xgb.DMatrix(fold_data['train_data'], label=fold_data['train_label'])
    dvalid = xgb.DMatrix(fold_data['valid_data'], label=fold_data['valid_label'])

    # 设置 eval list
    evals = [(dtrain, 'train'), (dvalid, 'valid')]

    # 提取训练参数
    num_boost_round = xgb_params.pop('num_boost_round', DEFAULT_NUM_BOOST_ROUND)
    early_stopping_rounds = xgb_params.pop('early_stopping_rounds', DEFAULT_EARLY_STOPPING_ROUNDS)

    # 训练模型
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=100,
    )

    # 恢复参数（因为 pop 会修改原字典）
    xgb_params['num_boost_round'] = num_boost_round
    xgb_params['early_stopping_rounds'] = early_stopping_rounds

    print(f"      Best iter: {model.best_iteration}")

    # 预测
    dtrain_pred = xgb.DMatrix(fold_data['train_data'])
    dvalid_pred = xgb.DMatrix(fold_data['valid_data'])
    train_pred = model.predict(dtrain_pred)
    valid_pred = model.predict(dvalid_pred)

    return model, train_pred, valid_pred


def get_xgboost_importance(model) -> np.ndarray:
    """获取 XGBoost 特征重要性"""
    # 获取 gain 类型的重要性
    importance_dict = model.get_score(importance_type='gain')

    # 获取特征名列表
    feature_names = model.feature_names
    num_features = len(feature_names) if feature_names else model.num_features()

    # 构建重要性数组（按特征名顺序）
    importance_array = np.zeros(num_features)

    if feature_names:
        # 如果有特征名，按名称匹配
        for i, feat_name in enumerate(feature_names):
            if feat_name in importance_dict:
                importance_array[i] = importance_dict[feat_name]
    else:
        # 回退到 f0, f1, f2... 格式
        for feat_name, importance in importance_dict.items():
            if feat_name.startswith('f'):
                try:
                    feat_idx = int(feat_name[1:])
                    if feat_idx < num_features:
                        importance_array[feat_idx] = importance
                except ValueError:
                    pass

    return importance_array


# ============================================================================
# 参数处理
# ============================================================================

def apply_cli_params(xgb_params: dict, args) -> dict:
    """应用 CLI 参数覆盖"""
    if args.lr is not None:
        xgb_params['learning_rate'] = args.lr
    if args.depth is not None:
        xgb_params['max_depth'] = args.depth
    if args.l2 is not None:
        xgb_params['reg_lambda'] = args.l2
    if args.l1 is not None:
        xgb_params['reg_alpha'] = args.l1
    if args.subsample is not None:
        xgb_params['subsample'] = args.subsample
    if args.colsample is not None:
        xgb_params['colsample_bytree'] = args.colsample
    if args.min_child_weight is not None:
        xgb_params['min_child_weight'] = args.min_child_weight
    return xgb_params


def print_header(args, symbols, xgb_params):
    """打印头部信息"""
    print("=" * 70)
    if args.shuffle:
        print("XGBoost SHUFFLE TEST")
    else:
        print("XGBoost Quick Evaluation")
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Handler: {args.handler}")
    print(f"N-day: {args.nday}")
    print(f"Threads: {xgb_params.get('nthread', 'N/A')} (OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'N/A')})")
    if args.shuffle:
        print(f"Shuffle runs: {args.shuffle_runs}")
    print(f"\nParameters:")
    for key in ['learning_rate', 'max_depth', 'reg_lambda', 'reg_alpha', 'subsample', 'colsample_bytree', 'min_child_weight']:
        print(f"  {key}: {xgb_params.get(key, 'N/A')}")
    print("=" * 70)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='XGBoost Quick Evaluation with Fixed Parameters',
    )

    # 添加通用参数
    add_common_args(parser)

    # XGBoost 特定超参数
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--depth', type=int, default=None, help='Max depth')
    parser.add_argument('--l2', type=float, default=None, help='L2 regularization (reg_lambda)')
    parser.add_argument('--l1', type=float, default=None, help='L1 regularization (reg_alpha)')
    parser.add_argument('--subsample', type=float, default=None, help='Subsample ratio')
    parser.add_argument('--colsample', type=float, default=None, help='Column sample ratio (colsample_bytree)')
    parser.add_argument('--min-child-weight', type=int, default=None, help='Min child weight')

    args = parser.parse_args()

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    # 构建超参数
    xgb_params = DEFAULT_PARAMS.copy()
    xgb_params['num_boost_round'] = DEFAULT_NUM_BOOST_ROUND
    xgb_params['early_stopping_rounds'] = DEFAULT_EARLY_STOPPING_ROUNDS

    # 从文件加载
    if args.params_file:
        xgb_params = load_params_from_file(args.params_file, xgb_params)

    # CLI 参数覆盖
    xgb_params = apply_cli_params(xgb_params, args)

    # 打印头部
    print_header(args, symbols, xgb_params)

    # 初始化 Qlib（如果需要）
    init_qlib(handler_config['use_talib'])

    if args.shuffle:
        # Shuffle Test 模式
        run_shuffle_test(
            args, handler_config, symbols, xgb_params,
            train_and_predict_xgboost, get_xgboost_importance,
            fill_na=False,  # XGBoost 可以处理 NaN
            shuffle_runs=args.shuffle_runs,
        )
    else:
        # 正常评估模式
        mean_ic, std_ic, importance_df = run_cv_evaluation_generic(
            args, handler_config, symbols, xgb_params,
            train_and_predict_xgboost, get_xgboost_importance,
            fill_na=False,  # XGBoost 可以处理 NaN
        )

        # 打印特征重要性
        print_feature_importance(importance_df, args.top_n)

        # 保存特征重要性
        if args.save_importance:
            save_importance_to_csv(importance_df, "xgboost", args.handler)

        # 最终汇总
        print_final_summary(args.handler, importance_df, mean_ic, std_ic)


if __name__ == "__main__":
    main()
