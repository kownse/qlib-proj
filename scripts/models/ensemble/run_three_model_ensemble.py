"""
Three-Model Ensemble: AE-MLP + CatBoost + TCN

Load pre-trained AE-MLP, CatBoost, and TCN models, generate predictions on test set,
learn optimal ensemble weights, and compute IC.

Usage:
    # Basic ensemble with auto-learned weights
    python scripts/models/ensemble/run_three_model_ensemble.py

    # With specific ensemble method
    python scripts/models/ensemble/run_three_model_ensemble.py --ensemble-method zscore_weighted

    # With backtest
    python scripts/models/ensemble/run_three_model_ensemble.py --backtest --topk 10

    # Custom model paths
    python scripts/models/ensemble/run_three_model_ensemble.py \
        --ae-model my_models/ae_mlp.keras \
        --cb-model my_models/catboost.cbm \
        --tcn-model my_models/tcn.pt
"""

import os

# Set thread limits before any other imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
from pathlib import Path
import multiprocessing

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Add scripts directory to path
script_dir = Path(__file__).parent.parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

import argparse
import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import torch
from torch.utils.data import DataLoader

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from utils.talib_ops import TALIB_OPS
from utils.strategy import get_strategy_config
from utils.backtest_utils import plot_backtest_curve, generate_trade_records
from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG,
    PROJECT_ROOT,
    QLIB_DATA_PATH,
    MODEL_SAVE_PATH,
    CV_FOLDS,
    FINAL_TEST,
)
from models.deep.ae_mlp_model import AEMLP


# ============================================================================
# TCN Model Components (imported from run_tcn_qlib_alpha158_cv.py)
# ============================================================================

from qlib.contrib.model.tcn import TemporalConvNet
import torch.nn as nn


class TCNModel(nn.Module):
    """TCN Model - consistent with Qlib benchmark"""

    def __init__(self, num_input, output_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.num_input = num_input
        self.tcn = TemporalConvNet(num_input, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        output = self.tcn(x)
        output = self.linear(output[:, :, -1])
        return output.squeeze()


class TCNTrainer:
    """TCN Trainer for loading and predicting"""

    def __init__(self, gpu=0):
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")
        self.model = None
        self.fitted = False
        self.d_feat = None
        self.n_chans = None
        self.kernel_size = None
        self.num_layers = None
        self.dropout = None

    def _init_model(self):
        """Initialize model"""
        self.model = TCNModel(
            num_input=self.d_feat,
            output_size=1,
            num_channels=[self.n_chans] * self.num_layers,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )
        self.model.to(self.device)

    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        config = checkpoint['config']

        self.d_feat = config['d_feat']
        self.n_chans = config['n_chans']
        self.kernel_size = config['kernel_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']

        self._init_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.fitted = True

    def predict(self, data_loader):
        """Predict"""
        if not self.fitted:
            raise ValueError("Model not fitted yet")

        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch_data in data_loader:
                feature, _ = batch_data
                feature = feature.to(self.device)
                pred = self.model(feature)
                preds.append(pred.cpu().numpy())

        return np.concatenate(preds)


class ReshapedTCNDataset(torch.utils.data.Dataset):
    """Dataset for Alpha300/Alpha360 with time-series structure"""

    def __init__(self, features, labels, d_feat, step_len):
        self.features = features
        self.labels = labels
        self.d_feat = d_feat
        self.step_len = step_len

        expected_features = d_feat * step_len
        if features.shape[1] != expected_features:
            raise ValueError(
                f"Feature dimension mismatch: got {features.shape[1]}, "
                f"expected {expected_features} (d_feat={d_feat} x step_len={step_len})"
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        feat_reshaped = feat.reshape(self.d_feat, self.step_len)
        label = self.labels[idx] if self.labels is not None else 0.0
        return torch.tensor(feat_reshaped, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# ============================================================================
# Model Loading Functions
# ============================================================================

def load_ae_mlp_model(model_path: Path):
    """Load AE-MLP (.keras) model"""
    print(f"    Loading AE-MLP model from: {model_path}")
    model = AEMLP.load(str(model_path))
    return model


def load_catboost_model(model_path: Path):
    """Load CatBoost (.cbm) model"""
    print(f"    Loading CatBoost model from: {model_path}")
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return model


def load_tcn_model(model_path: Path, gpu: int = 0):
    """Load TCN (.pt) model"""
    print(f"    Loading TCN model from: {model_path}")
    trainer = TCNTrainer(gpu=gpu)
    trainer.load(str(model_path))
    print(f"      Config: d_feat={trainer.d_feat}, n_chans={trainer.n_chans}, "
          f"kernel_size={trainer.kernel_size}, num_layers={trainer.num_layers}")
    return trainer


def load_model_meta(model_path: Path) -> dict:
    """Load model metadata"""
    meta_path = model_path.with_suffix('.meta.pkl')
    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            return pickle.load(f)

    stem = model_path.stem
    if stem.endswith('_best'):
        alt_meta_path = model_path.parent / (stem[:-5] + '.meta.pkl')
        if alt_meta_path.exists():
            with open(alt_meta_path, 'rb') as f:
                return pickle.load(f)

    return {}


# ============================================================================
# Data Preparation Functions
# ============================================================================

def create_data_handler(handler_name: str, symbols: list, time_splits: dict, nday: int):
    """Create DataHandler for a specific handler type"""
    from models.common.handlers import get_handler_class

    HandlerClass = get_handler_class(handler_name)

    handler = HandlerClass(
        volatility_window=nday,
        instruments=symbols,
        start_time=time_splits['train_start'],
        end_time=time_splits['test_end'],
        fit_start_time=time_splits['train_start'],
        fit_end_time=time_splits['train_end'],
        infer_processors=[],
    )

    return handler


def create_dataset(handler, time_splits: dict) -> DatasetH:
    """Create Qlib DatasetH"""
    return DatasetH(
        handler=handler,
        segments={
            "train": (time_splits['train_start'], time_splits['train_end']),
            "valid": (time_splits['valid_start'], time_splits['valid_end']),
            "test": (time_splits['test_start'], time_splits['test_end']),
        }
    )


# ============================================================================
# Prediction Functions
# ============================================================================

def predict_with_ae_mlp(model: AEMLP, dataset: DatasetH, segment: str = "test") -> pd.Series:
    """Generate predictions with AE-MLP model"""
    pred = model.predict(dataset, segment=segment)
    pred.name = 'score'
    return pred


def predict_with_catboost(model: CatBoostRegressor, dataset: DatasetH, segment: str = "test") -> pd.Series:
    """Generate predictions with CatBoost model"""
    data = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
    data = data.fillna(0).replace([np.inf, -np.inf], 0)
    pred_values = model.predict(data.values)
    pred = pd.Series(pred_values, index=data.index, name='score')
    return pred


def predict_with_tcn(model: TCNTrainer, dataset: DatasetH, segment: str = "test",
                     d_feat: int = 5, step_len: int = 60, batch_size: int = 2000) -> pd.Series:
    """Generate predictions with TCN model"""
    # Get features and labels
    features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
    labels = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)

    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]
    labels = labels.fillna(0)

    index = features.index

    # Create dataset and dataloader
    tcn_dataset = ReshapedTCNDataset(
        features.values,
        labels.values,
        d_feat=d_feat,
        step_len=step_len,
    )

    data_loader = DataLoader(tcn_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Predict
    pred_values = model.predict(data_loader)
    pred = pd.Series(pred_values, index=index, name='score')

    return pred


# ============================================================================
# Correlation and Weight Learning Functions
# ============================================================================

def calculate_pairwise_correlations(preds: dict) -> pd.DataFrame:
    """Calculate pairwise correlations between model predictions"""
    model_names = list(preds.keys())
    n_models = len(model_names)

    # Find common index
    common_idx = preds[model_names[0]].index
    for name in model_names[1:]:
        common_idx = common_idx.intersection(preds[name].index)

    # Build correlation matrix
    corr_matrix = np.zeros((n_models, n_models))
    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                p_i = preds[name_i].loc[common_idx]
                p_j = preds[name_j].loc[common_idx]
                corr_matrix[i, j] = p_i.corr(p_j)

    return pd.DataFrame(corr_matrix, index=model_names, columns=model_names)


def learn_optimal_weights_multi(preds: dict, label: pd.Series,
                                method: str = 'grid_search',
                                min_weight: float = 0.05,
                                diversity_bonus: float = 0.05) -> tuple:
    """
    Learn optimal ensemble weights for multiple models.

    Parameters
    ----------
    preds : dict
        Dict of {model_name: prediction_series}
    label : pd.Series
        True labels
    method : str
        'grid_search', 'grid_search_icir', 'regression', 'ridge', 'equal'
    min_weight : float
        Minimum weight for each model
    diversity_bonus : float
        Bonus for balanced weights

    Returns
    -------
    tuple
        (weights_dict, info_dict)
    """
    model_names = list(preds.keys())
    n_models = len(model_names)

    # Find common index
    common_idx = label.index
    for name in model_names:
        common_idx = common_idx.intersection(preds[name].index)

    # Align all series
    aligned_preds = {name: preds[name].loc[common_idx] for name in model_names}
    y = label.loc[common_idx]

    # Remove NaN
    valid_mask = ~y.isna()
    for name in model_names:
        valid_mask &= ~aligned_preds[name].isna()

    for name in model_names:
        aligned_preds[name] = aligned_preds[name][valid_mask]
    y = y[valid_mask]

    # Z-score normalize within each day
    def zscore_by_day(x):
        mean = x.groupby(level='datetime').transform('mean')
        std = x.groupby(level='datetime').transform('std')
        return (x - mean) / (std + 1e-8)

    normalized_preds = {name: zscore_by_day(aligned_preds[name]) for name in model_names}

    if method == 'equal':
        weights = {name: 1.0 / n_models for name in model_names}
        return weights, {'method': 'equal'}

    elif method in ['grid_search', 'grid_search_icir']:
        # Grid search over weight combinations
        best_score = -np.inf
        best_weights = {name: 1.0 / n_models for name in model_names}
        best_ic = 0

        # Generate weight combinations (sum to 1, each >= min_weight)
        step = 0.05
        weight_range = np.arange(min_weight, 1.0 - min_weight * (n_models - 1) + 0.01, step)

        def generate_weight_combinations(n, remaining=1.0, min_w=0.05, step=0.05):
            """Generate all weight combinations that sum to remaining"""
            if n == 1:
                if remaining >= min_w - 1e-6:
                    yield [remaining]
                return
            for w in np.arange(min_w, remaining - min_w * (n - 1) + 0.01, step):
                for rest in generate_weight_combinations(n - 1, remaining - w, min_w, step):
                    yield [w] + rest

        for weight_list in generate_weight_combinations(n_models, 1.0, min_weight, step):
            weights = {name: w for name, w in zip(model_names, weight_list)}

            # Compute ensemble
            ensemble = sum(normalized_preds[name] * weights[name] for name in model_names)

            # Calculate daily IC
            df = pd.DataFrame({'pred': ensemble, 'label': y})
            ic_by_date = df.groupby(level='datetime').apply(
                lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
            )
            ic_series = ic_by_date.dropna()

            if len(ic_series) == 0:
                continue

            mean_ic = ic_series.mean()
            ic_std = ic_series.std()
            icir = mean_ic / ic_std if ic_std > 0 else 0

            if method == 'grid_search':
                # Diversity bonus: reward balanced weights
                weight_variance = np.var(weight_list)
                max_variance = ((1 - min_weight * n_models) / n_models) ** 2 * (n_models - 1)
                diversity = 1 - weight_variance / (max_variance + 1e-8)
                score = mean_ic + diversity_bonus * diversity
            else:  # grid_search_icir
                score = icir

            if score > best_score:
                best_score = score
                best_weights = weights.copy()
                best_ic = mean_ic

        return best_weights, {
            'method': method,
            'best_ic': best_ic,
        }

    elif method == 'regression':
        from sklearn.linear_model import LinearRegression

        X = np.column_stack([normalized_preds[name].values for name in model_names])
        reg = LinearRegression(fit_intercept=False, positive=True)
        reg.fit(X, y.values)

        raw_weights = reg.coef_
        total = sum(abs(w) for w in raw_weights)
        if total > 0:
            weights = {name: abs(raw_weights[i]) / total for i, name in enumerate(model_names)}
        else:
            weights = {name: 1.0 / n_models for name in model_names}

        return weights, {'method': 'regression', 'r2': reg.score(X, y.values)}

    elif method == 'ridge':
        from sklearn.linear_model import Ridge

        X = np.column_stack([normalized_preds[name].values for name in model_names])
        reg = Ridge(alpha=1.0, fit_intercept=False)
        reg.fit(X, y.values)

        raw_weights = reg.coef_
        total = sum(abs(w) for w in raw_weights)
        if total > 0:
            weights = {name: abs(raw_weights[i]) / total for i, name in enumerate(model_names)}
        else:
            weights = {name: 1.0 / n_models for name in model_names}

        return weights, {'method': 'ridge', 'r2': reg.score(X, y.values)}

    else:
        raise ValueError(f"Unknown weight learning method: {method}")


def ensemble_predictions_multi(preds: dict, method: str = 'zscore_mean',
                               weights: dict = None) -> pd.Series:
    """
    Ensemble multiple model predictions.

    Parameters
    ----------
    preds : dict
        Dict of {model_name: prediction_series}
    method : str
        'mean', 'weighted', 'rank_mean', 'zscore_mean', 'zscore_weighted'
    weights : dict, optional
        {model_name: weight} for weighted ensemble

    Returns
    -------
    pd.Series
        Ensembled predictions
    """
    model_names = list(preds.keys())

    # Find common index
    common_idx = preds[model_names[0]].index
    for name in model_names[1:]:
        common_idx = common_idx.intersection(preds[name].index)

    aligned = {name: preds[name].loc[common_idx] for name in model_names}

    if method == 'mean':
        ensemble_pred = sum(aligned[name] for name in model_names) / len(model_names)

    elif method == 'weighted':
        if weights is None:
            weights = {name: 1.0 / len(model_names) for name in model_names}
        total = sum(weights.values())
        ensemble_pred = sum(aligned[name] * weights[name] for name in model_names) / total

    elif method == 'rank_mean':
        ranks = {name: aligned[name].groupby(level='datetime').rank(pct=True) for name in model_names}
        ensemble_pred = sum(ranks[name] for name in model_names) / len(model_names)

    elif method == 'zscore_mean':
        def zscore_by_day(x):
            mean = x.groupby(level='datetime').transform('mean')
            std = x.groupby(level='datetime').transform('std')
            return (x - mean) / (std + 1e-8)
        zscores = {name: zscore_by_day(aligned[name]) for name in model_names}
        ensemble_pred = sum(zscores[name] for name in model_names) / len(model_names)

    elif method == 'zscore_weighted':
        if weights is None:
            weights = {name: 1.0 / len(model_names) for name in model_names}
        def zscore_by_day(x):
            mean = x.groupby(level='datetime').transform('mean')
            std = x.groupby(level='datetime').transform('std')
            return (x - mean) / (std + 1e-8)
        zscores = {name: zscore_by_day(aligned[name]) for name in model_names}
        total = sum(weights.values())
        ensemble_pred = sum(zscores[name] * weights[name] for name in model_names) / total

    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    ensemble_pred.name = 'score'
    return ensemble_pred


def compute_ic(pred: pd.Series, label: pd.Series) -> tuple:
    """Calculate IC (Information Coefficient)"""
    common_idx = pred.index.intersection(label.index)
    pred_aligned = pred.loc[common_idx]
    label_aligned = label.loc[common_idx]

    valid_idx = ~(pred_aligned.isna() | label_aligned.isna())
    pred_clean = pred_aligned[valid_idx]
    label_clean = label_aligned[valid_idx]

    df = pd.DataFrame({'pred': pred_clean, 'label': label_clean})
    ic_by_date = df.groupby(level='datetime').apply(
        lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
    )
    ic_by_date = ic_by_date.dropna()

    if len(ic_by_date) == 0:
        return 0.0, 0.0, 0.0, pd.Series()

    mean_ic = ic_by_date.mean()
    ic_std = ic_by_date.std()
    icir = mean_ic / ic_std if ic_std > 0 else 0

    return mean_ic, ic_std, icir, ic_by_date


# ============================================================================
# Backtest Functions
# ============================================================================

def run_ensemble_backtest(pred: pd.Series, args, time_splits: dict):
    """Run backtest with ensembled predictions"""
    from qlib.backtest import backtest as qlib_backtest
    from qlib.contrib.evaluate import risk_analysis

    print("\n" + "=" * 70)
    print("BACKTEST with TopkDropoutStrategy (Three-Model Ensemble)")
    print("=" * 70)

    pred_df = pred.to_frame("score")

    print(f"\n[BT-1] Configuring backtest...")
    print(f"    Strategy: {args.strategy}")
    print(f"    Topk: {args.topk}")
    print(f"    N_drop: {args.n_drop}")
    print(f"    Account: ${args.account:,.0f}")
    print(f"    Rebalance Freq: every {args.rebalance_freq} day(s)")
    print(f"    Period: {time_splits['test_start']} to {time_splits['test_end']}")

    strategy_config = get_strategy_config(
        pred_df, args.topk, args.n_drop, args.rebalance_freq,
        strategy_type=args.strategy
    )

    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    }

    pool_symbols = STOCK_POOLS[args.stock_pool]

    backtest_config = {
        "start_time": time_splits['test_start'],
        "end_time": time_splits['test_end'],
        "account": args.account,
        "benchmark": "SPY",
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": None,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0005,
            "min_cost": 1,
            "trade_unit": None,
            "codes": pool_symbols,
        },
    }

    print(f"\n[BT-2] Running backtest...")
    try:
        portfolio_metric_dict, indicator_dict = qlib_backtest(
            executor=executor_config,
            strategy=strategy_config,
            **backtest_config
        )

        print("    Backtest completed")

        print(f"\n[BT-3] Analyzing results...")

        for freq, (report_df, positions) in portfolio_metric_dict.items():
            _analyze_backtest_results(report_df, positions, freq, args, time_splits)

        return portfolio_metric_dict

    except Exception as e:
        print(f"\n    Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _analyze_backtest_results(report_df: pd.DataFrame, positions, freq: str,
                              args, time_splits: dict):
    """Analyze and report backtest results"""
    from qlib.contrib.evaluate import risk_analysis

    print(f"\n    Frequency: {freq}")
    print(f"    Report shape: {report_df.shape}")
    print(f"    Date range: {report_df.index.min()} to {report_df.index.max()}")

    total_return = (report_df["return"] + 1).prod() - 1

    has_bench = "bench" in report_df.columns and not report_df["bench"].isna().all()
    if has_bench:
        bench_return = (report_df["bench"] + 1).prod() - 1
        excess_return = total_return - bench_return
        excess_return_series = report_df["return"] - report_df["bench"]
        analysis = risk_analysis(excess_return_series, freq=freq)
    else:
        bench_return = None
        excess_return = None
        analysis = risk_analysis(report_df["return"], freq=freq)

    print(f"\n    Performance Summary:")
    print(f"    " + "-" * 50)
    print(f"    Total Return:      {total_return:>10.2%}")
    if has_bench:
        print(f"    Benchmark Return:  {bench_return:>10.2%}")
        print(f"    Excess Return:     {excess_return:>10.2%}")
    print(f"    " + "-" * 50)

    if analysis is not None and not analysis.empty:
        analysis_title = "Risk Analysis (Excess Return)" if has_bench else "Risk Analysis (Strategy Return)"
        print(f"\n    {analysis_title}:")
        print(f"    " + "-" * 50)
        for metric, value in analysis.items():
            if isinstance(value, (int, float)):
                print(f"    {metric:<25s}: {value:>10.4f}")
        print(f"    " + "-" * 50)

    # Save report
    output_path = PROJECT_ROOT / "outputs" / f"three_model_ensemble_backtest_report_{freq}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path)
    print(f"\n    Report saved to: {output_path}")

    # Plot performance chart
    print(f"\n[BT-4] Generating performance chart...")
    if not hasattr(args, 'handler'):
        args.handler = "ensemble"
    plot_backtest_curve(report_df, args, freq, PROJECT_ROOT, model_name="Three_Model_Ensemble")

    # Generate trade records
    print(f"\n[BT-5] Generating trade records...")
    generate_trade_records(positions, args, freq, PROJECT_ROOT, model_name="three_model_ensemble")


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Three-Model Ensemble: AE-MLP + CatBoost + TCN',
    )

    # Model paths
    parser.add_argument('--ae-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'ae_mlp_cv_alpha158-enhanced-v7_sp500_5d_best.keras'),
                        help='AE-MLP model path (.keras)')
    parser.add_argument('--cb-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'catboost_cv_catboost-v1_test_5d_20260129_105915_best.cbm'),
                        help='CatBoost model path (.cbm)')
    parser.add_argument('--tcn-model', type=str,
                        default=str(MODEL_SAVE_PATH / 'tcn_cv_alpha300_sp500_5d_20260130_152436_best.pt'),
                        help='TCN model path (.pt)')

    # Handler configuration
    parser.add_argument('--ae-handler', type=str, default='alpha158-enhanced-v7',
                        help='Handler for AE-MLP model')
    parser.add_argument('--cb-handler', type=str, default='catboost-v1',
                        help='Handler for CatBoost model')
    parser.add_argument('--tcn-handler', type=str, default='alpha300',
                        help='Handler for TCN model')

    # TCN specific parameters
    parser.add_argument('--tcn-d-feat', type=int, default=5,
                        help='TCN d_feat (default: 5 for alpha300)')
    parser.add_argument('--tcn-step-len', type=int, default=60,
                        help='TCN step_len (default: 60 for alpha300)')

    # Ensemble parameters
    parser.add_argument('--ensemble-method', type=str, default='zscore_weighted',
                        choices=['mean', 'weighted', 'rank_mean', 'zscore_mean', 'zscore_weighted'],
                        help='Ensemble method (default: zscore_weighted)')

    # Weight learning parameters
    parser.add_argument('--learn-weights', action='store_true', default=True,
                        help='Learn optimal weights from validation set (default: True)')
    parser.add_argument('--no-learn-weights', dest='learn_weights', action='store_false',
                        help='Use equal weights instead of learning')
    parser.add_argument('--weight-method', type=str, default='grid_search',
                        choices=['grid_search', 'grid_search_icir', 'regression', 'ridge', 'equal'],
                        help='Weight learning method (default: grid_search)')
    parser.add_argument('--min-weight', type=float, default=0.1,
                        help='Minimum weight for each model (default: 0.1)')
    parser.add_argument('--diversity-bonus', type=float, default=0.05,
                        help='Bonus for balanced weights (default: 0.05)')

    # Manual weights (if not learning)
    parser.add_argument('--ae-weight', type=float, default=None,
                        help='Manual AE-MLP weight')
    parser.add_argument('--cb-weight', type=float, default=None,
                        help='Manual CatBoost weight')
    parser.add_argument('--tcn-weight', type=float, default=None,
                        help='Manual TCN weight')

    # Data parameters
    parser.add_argument('--nday', type=int, default=5,
                        help='Prediction horizon (default: 5)')
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'],
                        help='Stock pool (default: sp500)')
    parser.add_argument('--batch-size', type=int, default=2000,
                        help='Batch size for TCN (default: 2000)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device (-1 for CPU)')

    # Backtest parameters
    parser.add_argument('--backtest', action='store_true',
                        help='Run backtest after ensemble')
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of stocks to hold (default: 10)')
    parser.add_argument('--n-drop', type=int, default=2,
                        help='Number of stocks to drop/replace each day (default: 2)')
    parser.add_argument('--account', type=float, default=100000,
                        help='Initial account value (default: 100000)')
    parser.add_argument('--rebalance-freq', type=int, default=5,
                        help='Rebalance frequency in days (default: 5)')
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'dynamic_risk', 'vol_stoploss'],
                        help='Trading strategy (default: topk)')

    args = parser.parse_args()

    # Use FINAL_TEST time splits
    time_splits = {
        'train_start': FINAL_TEST['train_start'],
        'train_end': FINAL_TEST['train_end'],
        'valid_start': FINAL_TEST['valid_start'],
        'valid_end': FINAL_TEST['valid_end'],
        'test_start': FINAL_TEST['test_start'],
        'test_end': FINAL_TEST['test_end'],
    }

    print("=" * 70)
    print("Three-Model Ensemble: AE-MLP + CatBoost + TCN")
    print("=" * 70)
    print(f"AE-MLP Model:   {args.ae_model}")
    print(f"CatBoost Model: {args.cb_model}")
    print(f"TCN Model:      {args.tcn_model}")
    print(f"Handlers:       AE-MLP={args.ae_handler}, CatBoost={args.cb_handler}, TCN={args.tcn_handler}")
    print(f"Stock Pool:     {args.stock_pool}")
    print(f"Prediction Horizon: {args.nday} days")
    print(f"Ensemble Method: {args.ensemble_method}")
    print(f"Learn Weights:  {args.learn_weights} (method: {args.weight_method})")
    print(f"Test Period:    {time_splits['test_start']} to {time_splits['test_end']}")
    print("=" * 70)

    # Check model files exist
    ae_path = Path(args.ae_model)
    cb_path = Path(args.cb_model)
    tcn_path = Path(args.tcn_model)

    for path, name in [(ae_path, 'AE-MLP'), (cb_path, 'CatBoost'), (tcn_path, 'TCN')]:
        if not path.exists():
            print(f"Error: {name} model not found: {path}")
            sys.exit(1)

    # Initialize Qlib
    print("\n[1] Initializing Qlib...")
    qlib.init(
        provider_uri=str(QLIB_DATA_PATH),
        region=REG_US,
        custom_ops=TALIB_OPS,
        kernels=1,
        joblib_backend=None,
    )
    print("    Qlib initialized with TA-Lib support")

    # Get symbols
    symbols = STOCK_POOLS[args.stock_pool]
    print(f"\n[2] Using stock pool: {args.stock_pool} ({len(symbols)} stocks)")

    # Create datasets for each model
    print("\n[3] Creating datasets...")

    print(f"    Creating {args.ae_handler} dataset for AE-MLP...")
    ae_handler = create_data_handler(args.ae_handler, symbols, time_splits, args.nday)
    ae_dataset = create_dataset(ae_handler, time_splits)

    print(f"    Creating {args.cb_handler} dataset for CatBoost...")
    cb_handler = create_data_handler(args.cb_handler, symbols, time_splits, args.nday)
    cb_dataset = create_dataset(cb_handler, time_splits)

    print(f"    Creating {args.tcn_handler} dataset for TCN...")
    tcn_handler = create_data_handler(args.tcn_handler, symbols, time_splits, args.nday)
    tcn_dataset = create_dataset(tcn_handler, time_splits)

    # Load models
    print("\n[4] Loading models...")
    ae_model = load_ae_mlp_model(ae_path)
    cb_model = load_catboost_model(cb_path)
    tcn_model = load_tcn_model(tcn_path, args.gpu)

    # Generate test predictions
    print("\n[5] Generating test predictions...")

    print("    AE-MLP predictions...")
    pred_ae = predict_with_ae_mlp(ae_model, ae_dataset)
    print(f"      Shape: {len(pred_ae)}, Range: [{pred_ae.min():.4f}, {pred_ae.max():.4f}]")

    print("    CatBoost predictions...")
    pred_cb = predict_with_catboost(cb_model, cb_dataset)
    print(f"      Shape: {len(pred_cb)}, Range: [{pred_cb.min():.4f}, {pred_cb.max():.4f}]")

    print("    TCN predictions...")
    pred_tcn = predict_with_tcn(tcn_model, tcn_dataset, d_feat=args.tcn_d_feat,
                                step_len=args.tcn_step_len, batch_size=args.batch_size)
    print(f"      Shape: {len(pred_tcn)}, Range: [{pred_tcn.min():.4f}, {pred_tcn.max():.4f}]")

    # Store predictions in dict
    preds = {
        'AE-MLP': pred_ae,
        'CatBoost': pred_cb,
        'TCN': pred_tcn,
    }

    # Calculate pairwise correlations
    print("\n[6] Calculating pairwise correlations...")
    corr_matrix = calculate_pairwise_correlations(preds)

    print("\n    Prediction Correlation Matrix:")
    print("    " + "=" * 50)
    print(corr_matrix.to_string())
    print("    " + "=" * 50)

    # Learn optimal weights
    weights = None
    if args.learn_weights:
        print(f"\n[7] Learning optimal weights from validation set...")
        print(f"    Method: {args.weight_method}")
        print(f"    Min weight: {args.min_weight}")

        # Generate validation predictions
        print("    Generating validation predictions...")
        val_pred_ae = predict_with_ae_mlp(ae_model, ae_dataset, segment="valid")
        val_pred_cb = predict_with_catboost(cb_model, cb_dataset, segment="valid")
        val_pred_tcn = predict_with_tcn(tcn_model, tcn_dataset, segment="valid",
                                         d_feat=args.tcn_d_feat, step_len=args.tcn_step_len,
                                         batch_size=args.batch_size)

        val_preds = {
            'AE-MLP': val_pred_ae,
            'CatBoost': val_pred_cb,
            'TCN': val_pred_tcn,
        }

        print(f"      AE-MLP valid: {len(val_pred_ae)} samples")
        print(f"      CatBoost valid: {len(val_pred_cb)} samples")
        print(f"      TCN valid: {len(val_pred_tcn)} samples")

        # Get validation labels
        val_label = ae_dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(val_label, pd.DataFrame):
            val_label = val_label.iloc[:, 0]

        # Learn weights
        weights, learn_info = learn_optimal_weights_multi(
            val_preds, val_label,
            method=args.weight_method,
            min_weight=args.min_weight,
            diversity_bonus=args.diversity_bonus
        )

        print(f"\n    Learned Weights:")
        print(f"    " + "-" * 50)
        for name, w in weights.items():
            print(f"    {name:<15s}: {w:>10.4f}")
        print(f"    " + "-" * 50)

        for k, v in learn_info.items():
            if k != 'method':
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    elif args.ae_weight is not None and args.cb_weight is not None and args.tcn_weight is not None:
        # Use manual weights
        total = args.ae_weight + args.cb_weight + args.tcn_weight
        weights = {
            'AE-MLP': args.ae_weight / total,
            'CatBoost': args.cb_weight / total,
            'TCN': args.tcn_weight / total,
        }
        print(f"\n[7] Using manual weights: AE-MLP={weights['AE-MLP']:.3f}, "
              f"CatBoost={weights['CatBoost']:.3f}, TCN={weights['TCN']:.3f}")
    else:
        # Equal weights
        weights = {'AE-MLP': 1/3, 'CatBoost': 1/3, 'TCN': 1/3}
        print(f"\n[7] Using equal weights: 1/3 each")

    # Ensemble predictions
    step_num = 8
    print(f"\n[{step_num}] Ensembling predictions ({args.ensemble_method})...")
    pred_ensemble = ensemble_predictions_multi(preds, args.ensemble_method, weights)
    print(f"    Ensemble shape: {len(pred_ensemble)}, Range: [{pred_ensemble.min():.4f}, {pred_ensemble.max():.4f}]")

    # Get labels
    step_num += 1
    print(f"\n[{step_num}] Calculating IC metrics...")
    test_label = ae_dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)
    if isinstance(test_label, pd.DataFrame):
        label = test_label.iloc[:, 0]
    else:
        label = test_label

    # Calculate IC for each model and ensemble
    ae_ic, ae_std, ae_icir, _ = compute_ic(pred_ae, label)
    cb_ic, cb_std, cb_icir, _ = compute_ic(pred_cb, label)
    tcn_ic, tcn_std, tcn_icir, _ = compute_ic(pred_tcn, label)
    ens_ic, ens_std, ens_icir, _ = compute_ic(pred_ensemble, label)

    print("\n    +" + "=" * 68 + "+")
    print("    |" + " " * 12 + "Information Coefficient (IC) Comparison" + " " * 16 + "|")
    print("    +" + "=" * 68 + "+")
    print(f"    |  {'Model':<15s} | {'Weight':>8s} | {'Mean IC':>10s} | {'IC Std':>10s} | {'ICIR':>10s} |")
    print("    +" + "-" * 68 + "+")
    print(f"    |  {'AE-MLP':<15s} | {weights['AE-MLP']:>8.3f} | {ae_ic:>10.4f} | {ae_std:>10.4f} | {ae_icir:>10.4f} |")
    print(f"    |  {'CatBoost':<15s} | {weights['CatBoost']:>8.3f} | {cb_ic:>10.4f} | {cb_std:>10.4f} | {cb_icir:>10.4f} |")
    print(f"    |  {'TCN':<15s} | {weights['TCN']:>8.3f} | {tcn_ic:>10.4f} | {tcn_std:>10.4f} | {tcn_icir:>10.4f} |")
    print("    +" + "-" * 68 + "+")
    print(f"    |  {'ENSEMBLE':<15s} | {'1.000':>8s} | {ens_ic:>10.4f} | {ens_std:>10.4f} | {ens_icir:>10.4f} |")
    print("    +" + "=" * 68 + "+")

    # Calculate improvement
    best_single_ic = max(ae_ic, cb_ic, tcn_ic)
    best_single_icir = max(ae_icir, cb_icir, tcn_icir)
    best_model = 'AE-MLP' if ae_ic == best_single_ic else ('CatBoost' if cb_ic == best_single_ic else 'TCN')

    if best_single_ic != 0:
        ic_improvement = (ens_ic - best_single_ic) / abs(best_single_ic) * 100
    else:
        ic_improvement = 0

    if best_single_icir != 0:
        icir_improvement = (ens_icir - best_single_icir) / abs(best_single_icir) * 100
    else:
        icir_improvement = 0

    print(f"\n    Ensemble Performance vs Best Single Model ({best_model}):")
    print(f"    IC improvement:   {ic_improvement:>+.2f}%")
    print(f"    ICIR improvement: {icir_improvement:>+.2f}%")

    # Summary
    print("\n" + "=" * 70)
    print("THREE-MODEL ENSEMBLE COMPLETE")
    print("=" * 70)
    print(f"Weights: AE-MLP={weights['AE-MLP']:.3f}, CatBoost={weights['CatBoost']:.3f}, TCN={weights['TCN']:.3f}")
    print(f"AE-MLP IC:   {ae_ic:.4f} (ICIR: {ae_icir:.4f})")
    print(f"CatBoost IC: {cb_ic:.4f} (ICIR: {cb_icir:.4f})")
    print(f"TCN IC:      {tcn_ic:.4f} (ICIR: {tcn_icir:.4f})")
    print(f"Ensemble IC: {ens_ic:.4f} (ICIR: {ens_icir:.4f})")
    print("=" * 70)

    # Run backtest if requested
    if args.backtest:
        run_ensemble_backtest(pred_ensemble, args, time_splits)

        print("\n" + "=" * 70)
        print("ENSEMBLE BACKTEST COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    main()
