"""
LightGBM 快速评估脚本 - 使用固定超参数

用于快速迭代特征工程:
1. 使用已知的好超参数，跳过超参数搜索
2. 运行 CV 训练，输出 CV Mean IC
3. 输出特征重要性报告

使用方法:
    # 使用默认超参数
    python scripts/models/tree/run_lightgbm_quick_eval.py --handler alpha158-enhanced-v3

    # 指定超参数
    python scripts/models/tree/run_lightgbm_quick_eval.py --handler alpha158-enhanced-v3 \
        --lr 0.05 --depth 6 --num-leaves 64

    # 从 JSON 文件加载超参数
    python scripts/models/tree/run_lightgbm_quick_eval.py --handler alpha158-enhanced-v3 \
        --params-file outputs/hyperopt_cv/lightgbm_cv_best_params_xxx.json
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
import lightgbm as lgb

from data.stock_pools import STOCK_POOLS
from models.common import HANDLER_CONFIG, init_qlib


# ============================================================================
# 默认超参数
# ============================================================================

DEFAULT_PARAMS = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_iterations': 1000,
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 128,
    'min_data_in_leaf': 50,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.0,
    'lambda_l2': 1.0,
    'num_threads': 16,
    'verbose': -1,
    'seed': 42,
}


# ============================================================================
# LightGBM 特定函数
# ============================================================================

def train_and_predict_lightgbm(fold_data: dict, lgb_params: dict):
    """LightGBM 训练和预测函数"""
    # 创建 Dataset
    train_set = lgb.Dataset(fold_data['train_data'], label=fold_data['train_label'])
    valid_set = lgb.Dataset(fold_data['valid_data'], label=fold_data['valid_label'], reference=train_set)

    # 训练模型
    model = lgb.train(
        lgb_params,
        train_set,
        valid_sets=[valid_set],
        valid_names=['valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    print(f"      Best iter: {model.best_iteration}")

    # 预测
    train_pred = model.predict(fold_data['train_data'])
    valid_pred = model.predict(fold_data['valid_data'])

    return model, train_pred, valid_pred


def get_lightgbm_importance(model) -> np.ndarray:
    """获取 LightGBM 特征重要性"""
    return model.feature_importance(importance_type='gain')


# ============================================================================
# 参数处理
# ============================================================================

def apply_cli_params(lgb_params: dict, args) -> dict:
    """应用 CLI 参数覆盖"""
    if args.lr is not None:
        lgb_params['learning_rate'] = args.lr
    if args.depth is not None:
        lgb_params['max_depth'] = args.depth
    if args.num_leaves is not None:
        lgb_params['num_leaves'] = args.num_leaves
    if args.l1 is not None:
        lgb_params['lambda_l1'] = args.l1
    if args.l2 is not None:
        lgb_params['lambda_l2'] = args.l2
    if args.feature_fraction is not None:
        lgb_params['feature_fraction'] = args.feature_fraction
    if args.bagging_fraction is not None:
        lgb_params['bagging_fraction'] = args.bagging_fraction
    if args.min_leaf is not None:
        lgb_params['min_data_in_leaf'] = args.min_leaf
    return lgb_params


def print_header(args, symbols, lgb_params):
    """打印头部信息"""
    print("=" * 70)
    if args.shuffle:
        print("LightGBM SHUFFLE TEST")
    else:
        print("LightGBM Quick Evaluation")
    print("=" * 70)
    print(f"Stock Pool: {args.stock_pool} ({len(symbols)} stocks)")
    print(f"Handler: {args.handler}")
    print(f"N-day: {args.nday}")
    if args.shuffle:
        print(f"Shuffle runs: {args.shuffle_runs}")
    print(f"\nParameters:")
    for key in ['learning_rate', 'max_depth', 'num_leaves', 'feature_fraction', 'bagging_fraction', 'min_data_in_leaf', 'lambda_l1', 'lambda_l2']:
        print(f"  {key}: {lgb_params[key]}")
    print("=" * 70)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='LightGBM Quick Evaluation with Fixed Parameters',
    )

    # 添加通用参数
    add_common_args(parser)

    # LightGBM 特定超参数
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--depth', type=int, default=None, help='Max depth')
    parser.add_argument('--num-leaves', type=int, default=None, help='Number of leaves')
    parser.add_argument('--l1', type=float, default=None, help='L1 regularization')
    parser.add_argument('--l2', type=float, default=None, help='L2 regularization')
    parser.add_argument('--feature-fraction', type=float, default=None, help='Feature fraction')
    parser.add_argument('--bagging-fraction', type=float, default=None, help='Bagging fraction')
    parser.add_argument('--min-leaf', type=int, default=None, help='Min data in leaf')

    args = parser.parse_args()

    # 获取配置
    handler_config = HANDLER_CONFIG[args.handler]
    symbols = STOCK_POOLS[args.stock_pool]

    # 构建超参数
    lgb_params = DEFAULT_PARAMS.copy()

    # 从文件加载
    if args.params_file:
        lgb_params = load_params_from_file(args.params_file, lgb_params)

    # CLI 参数覆盖
    lgb_params = apply_cli_params(lgb_params, args)

    # 打印头部
    print_header(args, symbols, lgb_params)

    # 初始化 Qlib（如果需要）
    init_qlib(handler_config['use_talib'])

    if args.shuffle:
        # Shuffle Test 模式
        run_shuffle_test(
            args, handler_config, symbols, lgb_params,
            train_and_predict_lightgbm, get_lightgbm_importance,
            fill_na=True,  # LightGBM 需要填充 NaN
            shuffle_runs=args.shuffle_runs,
        )
    else:
        # 正常评估模式
        mean_ic, std_ic, importance_df = run_cv_evaluation_generic(
            args, handler_config, symbols, lgb_params,
            train_and_predict_lightgbm, get_lightgbm_importance,
            fill_na=True,  # LightGBM 需要填充 NaN
        )

        # 打印特征重要性
        print_feature_importance(importance_df, args.top_n)

        # 保存特征重要性
        if args.save_importance:
            save_importance_to_csv(importance_df, "lgb", args.handler)

        # 最终汇总
        print_final_summary(args.handler, importance_df, mean_ic, std_ic)


if __name__ == "__main__":
    main()
