"""
Three-Model Ensemble: AE-MLP + CatBoost + TCN-FiLM

Load pre-trained AE-MLP, CatBoost, and TCN-FiLM models, generate predictions on test set,
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
        --tcn-model my_models/tcn_film.pt
"""

import os

# Set thread limits before any other imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['JOBLIB_START_METHOD'] = 'fork'
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

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
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import qlib
from qlib.constant import REG_US
from qlib.data.dataset.handler import DataHandlerLP

from utils.talib_ops import TALIB_OPS
from data.stock_pools import STOCK_POOLS

from models.common import (
    HANDLER_CONFIG,
    PROJECT_ROOT,
    QLIB_DATA_PATH,
    MODEL_SAVE_PATH,
    CV_FOLDS,
    FINAL_TEST,
)
from models.common.ensemble import (
    load_ae_mlp_model,
    load_model_meta,
    create_ensemble_data_handler,
    create_ensemble_dataset,
    predict_with_ae_mlp,
    predict_with_catboost,
    calculate_pairwise_correlations,
    compute_ic,
    ensemble_predictions,
    learn_optimal_weights,
    run_ensemble_backtest,
)
from models.common.training import load_catboost_model
from models.deep.tcn_film import TCNFiLM


# ============================================================================
# TCN-FiLM Components (using macro conditioning)
# ============================================================================

# Macro configuration
DEFAULT_MACRO_PATH = PROJECT_ROOT / "my_data" / "macro_processed" / "macro_features.parquet"

CORE_MACRO_FEATURES = [
    "macro_vix_level", "macro_vix_zscore20", "macro_vix_pct_5d",
    "macro_vix_regime", "macro_vix_term_structure",
    "macro_gld_pct_5d", "macro_tlt_pct_5d", "macro_yield_curve",
    "macro_uup_pct_5d", "macro_uso_pct_5d",
    "macro_spy_pct_5d", "macro_spy_vol20",
    "macro_hyg_vs_lqd", "macro_credit_stress", "macro_hy_spread_zscore",
    "macro_eem_vs_spy", "macro_global_risk",
    "macro_yield_10y", "macro_yield_2s10s", "macro_yield_inversion",
    "macro_risk_on_off", "macro_market_stress", "macro_hy_spread",
]

FEATURES_NEED_ZSCORE = [
    "macro_tlt_pct_20d", "macro_tlt_pct_5d", "macro_uso_pct_5d",
    "macro_gld_pct_5d", "macro_uup_pct_5d", "macro_spy_pct_5d",
]


def load_macro_df(path=None):
    """Load macro features DataFrame"""
    path = path or DEFAULT_MACRO_PATH
    df = pd.read_parquet(path)
    return df


def prepare_macro(index, macro_df, macro_cols, lag=1):
    """Prepare macro features aligned to stock index"""
    dates = index.get_level_values('datetime')
    available = [c for c in macro_cols if c in macro_df.columns]
    macro = macro_df[available].copy()

    # Z-score normalize momentum features
    for col in available:
        if col in FEATURES_NEED_ZSCORE:
            roll_mean = macro[col].rolling(60, min_periods=20).mean()
            roll_std = macro[col].rolling(60, min_periods=20).std()
            macro[col] = ((macro[col] - roll_mean) / (roll_std + 1e-8)).clip(-5, 5)

    if lag > 0:
        macro = macro.shift(lag)

    return macro.reindex(dates).fillna(0).values


class TCNMacroDataset(torch.utils.data.Dataset):
    """Dataset for TCN-FiLM with macro conditioning"""

    def __init__(self, stock_features, macro_features, labels, d_feat, step_len):
        self.stock = stock_features.reshape(-1, d_feat, step_len)
        self.macro = macro_features
        self.labels = labels

    def __len__(self):
        return len(self.stock)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.stock[idx], dtype=torch.float32),
            torch.tensor(self.macro[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class TCNFiLMTrainer:
    """TCN-FiLM Trainer for loading and predicting"""

    def __init__(self, gpu=0):
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")
        self.model = None
        self.config = None
        self.fitted = False

    def _init_model(self):
        """Initialize model from config"""
        self.model = TCNFiLM(**self.config).to(self.device)

    def load(self, path):
        """Load model checkpoint"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # Filter config to only include TCNFiLM parameters
        valid_keys = {'d_feat', 'n_macro', 'n_chans', 'num_layers', 'kernel_size', 'dropout', 'film_hidden'}
        self.config = {k: v for k, v in ckpt['config'].items() if k in valid_keys}
        self._init_model()
        # Support both 'state_dict' and 'model_state_dict' keys
        state_key = 'state_dict' if 'state_dict' in ckpt else 'model_state_dict'
        self.model.load_state_dict(ckpt[state_key])
        self.fitted = True

    def predict(self, data_loader):
        """Predict with model"""
        if not self.fitted:
            raise ValueError("Model not fitted yet")

        self.model.eval()
        preds = []

        with torch.no_grad():
            for stock, macro, _ in data_loader:
                pred = self.model(stock.to(self.device), macro.to(self.device))
                preds.append(pred.cpu().numpy())

        return np.concatenate(preds)


# ============================================================================
# TCN-specific Model Loading and Prediction
# ============================================================================

def load_tcn_model(model_path: Path, gpu: int = 0):
    """Load TCN-FiLM (.pt) model"""
    print(f"    Loading TCN-FiLM model from: {model_path}")
    trainer = TCNFiLMTrainer(gpu=gpu)
    trainer.load(str(model_path))
    cfg = trainer.config
    print(f"      Config: d_feat={cfg['d_feat']}, n_macro={cfg['n_macro']}, "
          f"n_chans={cfg['n_chans']}, num_layers={cfg['num_layers']}")
    return trainer


def predict_with_tcn(model: TCNFiLMTrainer, dataset, segment: str = "test",
                     d_feat: int = 5, step_len: int = 60, batch_size: int = 2000,
                     macro_df: pd.DataFrame = None, macro_cols: list = None,
                     macro_lag: int = 1) -> pd.Series:
    """Generate predictions with TCN-FiLM model using macro conditioning"""
    # Get features and labels
    features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L)
    labels = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)

    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]
    labels = labels.fillna(0)

    index = features.index

    # Prepare macro features
    macro = prepare_macro(index, macro_df, macro_cols, macro_lag)

    # Create dataset and dataloader
    tcn_dataset = TCNMacroDataset(
        features.values,
        macro,
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
                        default=str(MODEL_SAVE_PATH / 'tcn_macro_cv_sp500_5d_m23_film_20260202_090518_best.pt'),
                        help='TCN-FiLM model path (.pt)')

    # Handler configuration
    parser.add_argument('--ae-handler', type=str, default='alpha158-enhanced-v7',
                        help='Handler for AE-MLP model')
    parser.add_argument('--cb-handler', type=str, default='catboost-v1',
                        help='Handler for CatBoost model')
    parser.add_argument('--tcn-handler', type=str, default='alpha300',
                        help='Handler for TCN model')

    # TCN-FiLM specific parameters
    parser.add_argument('--tcn-d-feat', type=int, default=5,
                        help='TCN d_feat (default: 5 for alpha300)')
    parser.add_argument('--tcn-step-len', type=int, default=60,
                        help='TCN step_len (default: 60 for alpha300)')
    parser.add_argument('--macro-lag', type=int, default=1,
                        help='Macro feature lag in days (default: 1)')

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
    print("Three-Model Ensemble: AE-MLP + CatBoost + TCN-FiLM")
    print("=" * 70)
    print(f"AE-MLP Model:    {args.ae_model}")
    print(f"CatBoost Model:  {args.cb_model}")
    print(f"TCN-FiLM Model:  {args.tcn_model}")
    print(f"Handlers:        AE-MLP={args.ae_handler}, CatBoost={args.cb_handler}, TCN={args.tcn_handler}")
    print(f"Macro Lag:       {args.macro_lag} day(s)")
    print(f"Stock Pool:      {args.stock_pool}")
    print(f"Prediction Horizon: {args.nday} days")
    print(f"Ensemble Method: {args.ensemble_method}")
    print(f"Learn Weights:   {args.learn_weights} (method: {args.weight_method})")
    print(f"Test Period:     {time_splits['test_start']} to {time_splits['test_end']}")
    print("=" * 70)

    # Check model files exist
    ae_path = Path(args.ae_model)
    cb_path = Path(args.cb_model)
    tcn_path = Path(args.tcn_model)

    for path, name in [(ae_path, 'AE-MLP'), (cb_path, 'CatBoost'), (tcn_path, 'TCN-FiLM')]:
        if not path.exists():
            print(f"Error: {name} model not found: {path}")
            sys.exit(1)

    # Load macro data for TCN-FiLM
    print("\n[0] Loading macro data...")
    macro_df = load_macro_df()
    macro_cols = [c for c in CORE_MACRO_FEATURES if c in macro_df.columns]
    print(f"    Loaded macro: {macro_df.shape}, {len(macro_cols)} features")
    print(f"    Date range: {macro_df.index.min()} ~ {macro_df.index.max()}")

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
    # Only load validation data if learning weights, otherwise just test data
    include_valid = args.learn_weights
    data_start = time_splits['valid_start'] if include_valid else time_splits['test_start']
    print(f"\n[3] Creating datasets (from {data_start})...")

    print(f"    Creating {args.ae_handler} dataset for AE-MLP...")
    ae_handler = create_ensemble_data_handler(args.ae_handler, symbols, time_splits, args.nday, include_valid)
    ae_dataset = create_ensemble_dataset(ae_handler, time_splits, include_valid)

    print(f"    Creating {args.cb_handler} dataset for CatBoost...")
    cb_handler = create_ensemble_data_handler(args.cb_handler, symbols, time_splits, args.nday, include_valid)
    cb_dataset = create_ensemble_dataset(cb_handler, time_splits, include_valid)

    print(f"    Creating {args.tcn_handler} dataset for TCN...")
    tcn_handler = create_ensemble_data_handler(args.tcn_handler, symbols, time_splits, args.nday, include_valid)
    tcn_dataset = create_ensemble_dataset(tcn_handler, time_splits, include_valid)

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

    print("    TCN-FiLM predictions...")
    pred_tcn = predict_with_tcn(tcn_model, tcn_dataset, d_feat=args.tcn_d_feat,
                                step_len=args.tcn_step_len, batch_size=args.batch_size,
                                macro_df=macro_df, macro_cols=macro_cols,
                                macro_lag=args.macro_lag)
    print(f"      Shape: {len(pred_tcn)}, Range: [{pred_tcn.min():.4f}, {pred_tcn.max():.4f}]")

    # Store predictions in dict
    preds = {
        'AE-MLP': pred_ae,
        'CatBoost': pred_cb,
        'TCN-FiLM': pred_tcn,
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
                                         batch_size=args.batch_size,
                                         macro_df=macro_df, macro_cols=macro_cols,
                                         macro_lag=args.macro_lag)

        val_preds = {
            'AE-MLP': val_pred_ae,
            'CatBoost': val_pred_cb,
            'TCN-FiLM': val_pred_tcn,
        }

        print(f"      AE-MLP valid: {len(val_pred_ae)} samples")
        print(f"      CatBoost valid: {len(val_pred_cb)} samples")
        print(f"      TCN-FiLM valid: {len(val_pred_tcn)} samples")

        # Get validation labels
        val_label = ae_dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(val_label, pd.DataFrame):
            val_label = val_label.iloc[:, 0]

        # Learn weights
        weights, learn_info = learn_optimal_weights(
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
            'TCN-FiLM': args.tcn_weight / total,
        }
        print(f"\n[7] Using manual weights: AE-MLP={weights['AE-MLP']:.3f}, "
              f"CatBoost={weights['CatBoost']:.3f}, TCN-FiLM={weights['TCN-FiLM']:.3f}")
    else:
        # Equal weights
        weights = {'AE-MLP': 1/3, 'CatBoost': 1/3, 'TCN-FiLM': 1/3}
        print(f"\n[7] Using equal weights: 1/3 each")

    # Ensemble predictions
    step_num = 8
    print(f"\n[{step_num}] Ensembling predictions ({args.ensemble_method})...")
    pred_ensemble = ensemble_predictions(preds, args.ensemble_method, weights)
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
    print(f"    |  {'TCN-FiLM':<15s} | {weights['TCN-FiLM']:>8.3f} | {tcn_ic:>10.4f} | {tcn_std:>10.4f} | {tcn_icir:>10.4f} |")
    print("    +" + "-" * 68 + "+")
    print(f"    |  {'ENSEMBLE':<15s} | {'1.000':>8s} | {ens_ic:>10.4f} | {ens_std:>10.4f} | {ens_icir:>10.4f} |")
    print("    +" + "=" * 68 + "+")

    # Calculate improvement
    best_single_ic = max(ae_ic, cb_ic, tcn_ic)
    best_single_icir = max(ae_icir, cb_icir, tcn_icir)
    best_model = 'AE-MLP' if ae_ic == best_single_ic else ('CatBoost' if cb_ic == best_single_ic else 'TCN-FiLM')

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
    print(f"Weights: AE-MLP={weights['AE-MLP']:.3f}, CatBoost={weights['CatBoost']:.3f}, TCN-FiLM={weights['TCN-FiLM']:.3f}")
    print(f"AE-MLP IC:   {ae_ic:.4f} (ICIR: {ae_icir:.4f})")
    print(f"CatBoost IC: {cb_ic:.4f} (ICIR: {cb_icir:.4f})")
    print(f"TCN-FiLM IC: {tcn_ic:.4f} (ICIR: {tcn_icir:.4f})")
    print(f"Ensemble IC: {ens_ic:.4f} (ICIR: {ens_icir:.4f})")
    print("=" * 70)

    # Run backtest if requested
    if args.backtest:
        run_ensemble_backtest(pred_ensemble, args, time_splits,
                              model_name="Three_Model_Ensemble")

        print("\n" + "=" * 70)
        print("ENSEMBLE BACKTEST COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    main()
