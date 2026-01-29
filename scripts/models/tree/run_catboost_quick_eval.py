"""
CatBoost 快速评估脚本 - 使用固定超参数

用于快速迭代特征工程:
1. 使用已知的好超参数，跳过超参数搜索
2. 运行 CV 训练，输出 CV Mean IC
3. 输出特征重要性报告

使用方法:
    # 使用默认超参数
    python scripts/models/tree/run_catboost_quick_eval.py --handler alpha158-enhanced-v3

    # 指定超参数
    python scripts/models/tree/run_catboost_quick_eval.py --handler alpha158-enhanced-v3 \
        --lr 0.05 --depth 6 --l2 3.0

    # 从 JSON 文件加载超参数
    python scripts/models/tree/run_catboost_quick_eval.py --handler alpha158-enhanced-v3 \
        --params-file outputs/hyperopt_cv/catboost_cv_best_params_xxx.json
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

# 现在可以安全地导入其他模块
import argparse
import numpy as np
from catboost import CatBoostRegressor, Pool

from data.stock_pools import STOCK_POOLS
from models.common import HANDLER_CONFIG, init_qlib


# ============================================================================
# 默认超参数 (从之前的 hyperopt 搜索中获得)
# ============================================================================

DEFAULT_PARAMS = {
    'loss_function': 'RMSE',
    'iterations': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'l2_leaf_reg': 3.0,
    'random_strength': 1.0,
    'bagging_temperature': 0.5,
    'subsample': 0.8,
    'colsample_bylevel': 0.8,
    'min_data_in_leaf': 50,
    'thread_count': 16,
    'verbose': False,
    'random_seed': 42,
}


# ============================================================================
# CatBoost 特定函数
# ============================================================================

def train_and_predict_catboost(fold_data: dict, cb_params: dict):
    """CatBoost 训练和预测函数"""
    # 创建 Pool
    train_pool = Pool(fold_data['train_data'], label=fold_data['train_label'])
    valid_pool = Pool(fold_data['valid_data'], label=fold_data['valid_label'])

    # 训练模型
    model = CatBoostRegressor(**cb_params)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=50,
        verbose=100,
    )

    print(f"      Best iter: {model.best_iteration_}")

    # 预测
    train_pred = model.predict(fold_data['train_data'])
    valid_pred = model.predict(fold_data['valid_data'])

    return model, train_pred, valid_pred


def get_catboost_importance(model) -> np.ndarray:
    """获取 CatBoost 特征重要性"""
    return model.get_feature_importance()


# ============================================================================
# 参数处理
# ============================================================================

def apply_cli_params(cb_params: dict, args) -> dict:
    """应用 CLI 参数覆盖"""
    if args.lr is not None:
        cb_params['learning_rate'] = args.lr
    if args.depth is not None:
        cb_params['max_depth'] = args.depth
    if args.l2 is not None:
        cb_params['l2_leaf_reg'] = args.l2
    if args.subsample is not None:
        cb_params['subsample'] = args.subsample
    if args.colsample is not None:
        cb_params['colsample_bylevel'] = args.colsample
    if args.min_leaf is not None:
        cb_params['min_data_in_leaf'] = args.min_leaf
    return cb_params


def print_header(args, symbols, cb_params):
    """打印头部信息"""
    print("=" * 70)
    if args.shuffle:
        print("CatBoost SHUFFLE TEST")
    else:
        print("CatBoost Quick Evaluation")
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Handler: {args.handler}")
    print(f"N-day: {args.nday}")
    if args.shuffle:
        print(f"Shuffle runs: {args.shuffle_runs}")
    print(f"\nParameters:")
    for key in ['learning_rate', 'max_depth', 'l2_leaf_reg', 'subsample', 'colsample_bylevel', 'min_data_in_leaf']:
        print(f"  {key}: {cb_params[key]}")
    print("=" * 70)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CatBoost Quick Evaluation with Fixed Parameters',
    )

    # 添加通用参数
    add_common_args(parser)

    # CatBoost 特定超参数
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--depth', type=int, default=None, help='Max depth')
    parser.add_argument('--l2', type=float, default=None, help='L2 regularization')
    parser.add_argument('--subsample', type=float, default=None, help='Subsample ratio')
    parser.add_argument('--colsample', type=float, default=None, help='Column sample ratio')
    parser.add_argument('--min-leaf', type=int, default=None, help='Min data in leaf')

    args = parser.parse_args()

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    # 构建超参数
    cb_params = DEFAULT_PARAMS.copy()

    # 从文件加载
    if args.params_file:
        cb_params = load_params_from_file(args.params_file, cb_params)

    # CLI 参数覆盖
    cb_params = apply_cli_params(cb_params, args)

    # 打印头部
    print_header(args, symbols, cb_params)

    # 初始化 Qlib（如果需要）
    init_qlib(handler_config['use_talib'])

    if args.shuffle:
        # Shuffle Test 模式
        run_shuffle_test(
            args, handler_config, symbols, cb_params,
            train_and_predict_catboost, get_catboost_importance,
            fill_na=False,  # CatBoost 可以处理 NaN
            shuffle_runs=args.shuffle_runs,
        )
    else:
        # 正常评估模式
        mean_ic, std_ic, importance_df = run_cv_evaluation_generic(
            args, handler_config, symbols, cb_params,
            train_and_predict_catboost, get_catboost_importance,
            fill_na=False,  # CatBoost 可以处理 NaN
        )

        # 打印特征重要性
        print_feature_importance(importance_df, args.top_n)

        # 保存特征重要性
        if args.save_importance:
            save_importance_to_csv(importance_df, "catboost", args.handler)

        # 最终汇总
        print_final_summary(args.handler, importance_df, mean_ic, std_ic)


if __name__ == "__main__":
    main()
