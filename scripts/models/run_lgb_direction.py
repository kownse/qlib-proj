"""
运行 LightGBM 二分类模型，预测N天后股票涨跌方向

标签定义：
- 1: N天后价格上涨
- 0: N天后价格下跌或持平

扩展特征：包含 Alpha158 默认指标 + TA-Lib 技术指标
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
script_dir = Path(__file__).parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

import qlib
from qlib.constant import REG_US
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

# Import TA-Lib custom operators
from utils.talib_ops import TALIB_OPS

# Import extended data handlers
from data.datahandler_ext import Alpha158_Direction_TALib


# ========== 配置 ==========

# 数据路径
PROJECT_ROOT = Path(__file__).parent.parent.parent  # 项目根目录
QLIB_DATA_PATH = PROJECT_ROOT / "my_data" / "qlib_us"

# 股票池
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]

# 时间划分
TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
VALID_START = "2024-01-01"
VALID_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

# 预测天数
PREDICTION_DAYS = 1


def evaluate_classification(y_true, y_pred, y_prob, output_dir, prediction_days):
    """评估二分类模型性能"""
    print("\n" + "=" * 70)
    print("Classification Evaluation Results")
    print("=" * 70)

    # 基本指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n  Accuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    # 与随机猜测比较
    baseline_accuracy = max(y_true.mean(), 1 - y_true.mean())
    improvement = (accuracy - 0.5) / 0.5 * 100
    print(f"\n  Baseline (majority class): {baseline_accuracy:.4f}")
    print(f"  Improvement over 50%:      {improvement:+.2f}%")

    # AUC-ROC
    try:
        auc = roc_auc_score(y_true, y_prob)
        print(f"  AUC-ROC:   {auc:.4f}")
    except:
        auc = None
        print("  AUC-ROC:   N/A (single class)")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Down    Up")
    print(f"    Actual Down  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"    Actual Up    {cm[1,0]:5d}  {cm[1,1]:5d}")

    # 详细分类报告
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Down', 'Up']))

    # 按股票分析
    print("\n  Per-Stock Accuracy:")
    print("  " + "-" * 40)

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 混淆矩阵热图
    ax1 = axes[0, 0]
    im = ax1.imshow(cm, cmap='Blues')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Down', 'Up'])
    ax1.set_yticklabels(['Down', 'Up'])
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)
    plt.colorbar(im, ax=ax1)

    # 2. ROC 曲线
    ax2 = axes[0, 1]
    if auc is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax2.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {auc:.4f})')
        ax2.plot([0, 1], [0, 1], 'r--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'ROC not available', ha='center', va='center')
        ax2.set_title('ROC Curve')

    # 3. 预测概率分布
    ax3 = axes[1, 0]
    ax3.hist(y_prob[y_true == 0], bins=50, alpha=0.5, label='Actual Down', color='red')
    ax3.hist(y_prob[y_true == 1], bins=50, alpha=0.5, label='Actual Up', color='green')
    ax3.axvline(0.5, color='black', linestyle='--', label='Threshold')
    ax3.set_xlabel('Predicted Probability of Up')
    ax3.set_ylabel('Count')
    ax3.set_title('Prediction Probability Distribution')
    ax3.legend()

    # 4. 指标汇总
    ax4 = axes[1, 1]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    values = [accuracy, precision, recall, f1, auc if auc else 0]
    colors = ['steelblue' if v > 0.5 else 'salmon' for v in values]
    bars = ax4.bar(metrics, values, color=colors)
    ax4.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
    ax4.set_ylim(0, 1)
    ax4.set_ylabel('Score')
    ax4.set_title(f'{prediction_days}-Day Direction Prediction Metrics')
    ax4.legend()
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # 保存图表
    output_path = output_dir / "outputs" / f"direction_prediction_{prediction_days}d.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\n  Visualization saved to: {output_path}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
    }


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Stock Direction Prediction (Binary Classification)')
    parser.add_argument('--nday', type=int, default=1, help='Prediction horizon in days (default: 1)')
    parser.add_argument('--top-k', type=int, default=0,
                        help='Number of top features to select for retraining (0 = use all)')
    args = parser.parse_args()

    # 更新全局变量
    global PREDICTION_DAYS
    PREDICTION_DAYS = args.nday

    print("=" * 70)
    print(f"Stock Direction Prediction ({PREDICTION_DAYS}-Day)")
    print("Binary Classification: Up (1) vs Down (0)")
    print("Features: Alpha158 + TA-Lib Technical Indicators")
    print("=" * 70)

    # 1. 初始化 Qlib
    print("\n[1] Initializing Qlib...")
    qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US, custom_ops=TALIB_OPS)
    print("    ✓ Qlib initialized with TA-Lib custom operators")

    # 2. 检查数据
    print("\n[2] Checking data availability...")
    test_df = D.features(
        instruments=["AAPL"],
        fields=["$close", "$volume"],
        start_time=TEST_START,
        end_time=TEST_END
    )
    print(f"    ✓ AAPL sample data shape: {test_df.shape}")
    print(f"    ✓ Date range: {test_df.index.get_level_values('datetime').min().date()} to {test_df.index.get_level_values('datetime').max().date()}")

    # 3. 创建 DataHandler
    print(f"\n[3] Creating DataHandler with {PREDICTION_DAYS}-day direction label...")
    print(f"    Features: Alpha158 + TA-Lib (~300+ technical indicators)")
    print(f"    Label: 1 = Up, 0 = Down (after {PREDICTION_DAYS} day(s))")

    handler = Alpha158_Direction_TALib(
        prediction_days=PREDICTION_DAYS,
        instruments=TEST_SYMBOLS,
        start_time=TRAIN_START,
        end_time=TEST_END,
        fit_start_time=TRAIN_START,
        fit_end_time=TRAIN_END,
        infer_processors=[],
    )
    print("    ✓ DataHandler created")

    # 4. 创建 Dataset
    print("\n[4] Creating Dataset...")
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (TRAIN_START, TRAIN_END),
            "valid": (VALID_START, VALID_END),
            "test": (TEST_START, TEST_END),
        }
    )

    # 准备数据
    train_data = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
    train_label = dataset.prepare("train", col_set="label", data_key=DataHandlerLP.DK_L)
    valid_data = dataset.prepare("valid", col_set="feature", data_key=DataHandlerLP.DK_L)
    valid_label = dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
    test_data = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
    test_label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_I)

    print(f"    ✓ Train features shape: {train_data.shape}")
    print(f"    ✓ Valid features shape: {valid_data.shape}")
    print(f"    ✓ Test features shape:  {test_data.shape}")

    # 检查有效特征
    valid_cols = []
    for col in train_data.columns:
        col_data = train_data[col]
        if not col_data.isna().all() and col_data.nunique(dropna=True) > 1:
            valid_cols.append(col)
    print(f"    ✓ Valid features: {len(valid_cols)}")

    # 5. 分析标签分布
    print("\n[5] Analyzing label distribution...")
    train_up_ratio = train_label['LABEL0'].mean()
    valid_up_ratio = valid_label['LABEL0'].mean()
    test_up_ratio = test_label['LABEL0'].mean()

    print(f"    Train: {train_up_ratio*100:.1f}% Up, {(1-train_up_ratio)*100:.1f}% Down (n={len(train_label)})")
    print(f"    Valid: {valid_up_ratio*100:.1f}% Up, {(1-valid_up_ratio)*100:.1f}% Down (n={len(valid_label)})")
    print(f"    Test:  {test_up_ratio*100:.1f}% Up, {(1-test_up_ratio)*100:.1f}% Down (n={len(test_label)})")

    # 6. 训练 LightGBM 二分类模型
    print("\n[6] Training LightGBM binary classifier...")
    import lightgbm as lgb

    # 过滤有效特征
    train_data_filtered = train_data[valid_cols]
    valid_data_filtered = valid_data[valid_cols]
    test_data_filtered = test_data[valid_cols]

    # 处理测试集中缺失的特征
    for col in valid_cols:
        if col not in test_data_filtered.columns:
            test_data_filtered[col] = np.nan
    test_data_filtered = test_data_filtered[valid_cols]

    train_set = lgb.Dataset(train_data_filtered, label=train_label.values.ravel())
    valid_set = lgb.Dataset(valid_data_filtered, label=valid_label.values.ravel())

    # 计算类别权重以平衡不均衡数据
    n_up = train_label.values.sum()
    n_down = len(train_label) - n_up
    scale_pos_weight = n_down / n_up if n_up > 0 else 1.0

    lgb_params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'learning_rate': 0.005,  # 更小的学习率
        'max_depth': 4,  # 更浅的树防止过拟合
        'num_leaves': 16,
        'min_child_samples': 200,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'lambda_l1': 1.0,  # 更强的正则化
        'lambda_l2': 1.0,
        'scale_pos_weight': scale_pos_weight,  # 类别平衡
        'num_threads': 4,
        'verbose': -1,
        'seed': 42,
    }

    print("    Model parameters:")
    print(f"      - objective: binary")
    print(f"      - learning_rate: {lgb_params['learning_rate']}")
    print(f"      - max_depth: {lgb_params['max_depth']}")
    print(f"      - num_leaves: {lgb_params['num_leaves']}")

    model = lgb.train(
        lgb_params,
        train_set,
        num_boost_round=1000,
        valid_sets=[train_set, valid_set],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200)
        ]
    )
    print("    ✓ Model training completed")
    print(f"    ✓ Best iteration: {model.best_iteration}")

    # 7. 特征重要性分析
    print("\n[7] Feature Importance Analysis...")
    importance = model.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': valid_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("\n    Top 20 Features by Importance (gain):")
    print("    " + "-" * 50)
    for idx, (_, row) in enumerate(importance_df.head(20).iterrows()):
        print(f"    {idx+1:3d}. {row['feature']:<40s} {row['importance']:>10.2f}")
    print("    " + "-" * 50)

    # 8. 预测
    print("\n[8] Generating predictions...")
    y_prob = model.predict(test_data_filtered, num_iteration=model.best_iteration)
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = test_label.values.ravel()

    print(f"    ✓ Predictions generated for {len(y_pred)} samples")
    print(f"    ✓ Predicted Up ratio: {y_pred.mean()*100:.1f}%")

    # 9. 评估
    print("\n[9] Evaluation...")
    metrics = evaluate_classification(y_true, y_pred, y_prob, PROJECT_ROOT, PREDICTION_DAYS)

    # 10. 按股票分析
    print("\n[10] Per-Stock Analysis...")
    test_df = test_data_filtered.copy()
    test_df['y_true'] = y_true
    test_df['y_pred'] = y_pred
    test_df['y_prob'] = y_prob

    # 获取股票代码
    test_df = test_df.reset_index()
    if 'instrument' in test_df.columns:
        stock_col = 'instrument'
    else:
        stock_col = test_df.columns[1]  # 通常第二列是股票代码

    print(f"\n    {'Stock':<8s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'Samples':>8s}")
    print("    " + "-" * 50)

    for stock in TEST_SYMBOLS:
        stock_mask = test_df[stock_col] == stock
        if stock_mask.sum() > 0:
            stock_y_true = test_df.loc[stock_mask, 'y_true']
            stock_y_pred = test_df.loc[stock_mask, 'y_pred']
            acc = accuracy_score(stock_y_true, stock_y_pred)
            prec = precision_score(stock_y_true, stock_y_pred, zero_division=0)
            rec = recall_score(stock_y_true, stock_y_pred, zero_division=0)
            n = len(stock_y_true)
            print(f"    {stock:<8s} {acc:>10.3f} {prec:>10.3f} {rec:>10.3f} {n:>8d}")

    print("    " + "-" * 50)

    # 结论
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    accuracy = metrics['accuracy']
    if accuracy > 0.55:
        print(f"✓ Model shows predictive power: {accuracy*100:.1f}% accuracy")
        print("  Consider using this for trading signals with proper risk management")
    elif accuracy > 0.52:
        print(f"△ Model shows weak predictive power: {accuracy*100:.1f}% accuracy")
        print("  May be useful with careful position sizing and risk control")
    else:
        print(f"✗ Model shows no significant predictive power: {accuracy*100:.1f}% accuracy")
        print("  Technical indicators alone may not be sufficient for direction prediction")


if __name__ == "__main__":
    main()
