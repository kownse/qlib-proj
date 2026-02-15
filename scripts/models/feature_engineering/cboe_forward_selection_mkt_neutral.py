"""
CBOE Forward Selection on v9-mkt-neutral baseline.

Tests each of 10 CBOE options-derived features individually on top of the
v9-mkt-neutral baseline (7 stock + 3 macro + sector_dispersion + mkt-neutral label).
Only features that improve IC by >= min_improvement are retained.

Usage:
    conda activate qlib310
    python scripts/models/feature_engineering/cboe_forward_selection_mkt_neutral.py \
        --stock-pool sp500 --min-improvement 0.0005
"""

import os
import sys
import gc
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")

import qlib
from qlib.constant import REG_US
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import DatasetH

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.talib_ops import TALIB_OPS
from data.stock_pools import STOCK_POOLS
from models.feature_engineering.feature_selection_utils import (
    ForwardSelectionBase,
    INNER_CV_FOLDS,
    compute_ic,
    prepare_dataset_data,
    add_common_args,
    countdown,
    load_checkpoint,
)
from models.deep.ae_mlp_shared import build_ae_mlp_model

# ---------------------------------------------------------------------------
# V9 baseline features (hardcoded from datahandler_enhanced_v9.py)
# ---------------------------------------------------------------------------
V9_STOCK_FEATURES = {
    "MOMENTUM_QUALITY": "($close/Ref($close, 20) - 1) / (Std($close/Ref($close,1)-1, 20) + 1e-12)",
    "PCT_FROM_52W_HIGH": "($close - Max($high, 252)) / (Max($high, 252) + 1e-12)",
    "MAX60": "Max($high, 60)/$close",
    "TALIB_NATR14": "TALIB_NATR($high, $low, $close, 14)",
    "MAX_DRAWDOWN_60": "(Min($close, 60) - Max($high, 60)) / (Max($high, 60) + 1e-12)",
    "RESI20": "Resi($close, 20)/$close",
    "ROC60": "Ref($close, 60)/$close",
}

V9_MACRO_FEATURES = ["macro_vix_zscore20", "macro_hy_spread_zscore", "macro_credit_stress"]
# + computed sector_dispersion

ALL_CBOE_FEATURES = [
    "cboe_skew_level",
    "cboe_skew_zscore20",
    "cboe_skew_pct_5d",
    "cboe_skew_regime",
    "cboe_vvix_level",
    "cboe_vvix_zscore20",
    "cboe_vvix_regime",
    "cboe_vvix_vs_vix",
    "cboe_vix9d_vs_vix",
    "cboe_vix9d_spike",
]

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "my_data")
HYPERPARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "outputs", "hyperopt_cv", "ae_mlp_cv_best_params_v9-mkt-neutral.json",
)


def _load_hyperparams():
    """Load optimized hyperparams from hyperopt CV result file."""
    import json
    with open(HYPERPARAMS_PATH, "r") as f:
        data = json.load(f)
    return data["params"]


MKT_NEUTRAL_HYPERPARAMS = _load_hyperparams()
MACRO_PATH = os.path.join(DATA_DIR, "macro_processed", "macro_features.parquet")
CBOE_PATH = os.path.join(DATA_DIR, "cboe_processed", "cboe_features.parquet")
SPY_PATH = os.path.join(DATA_DIR, "spy_forward_returns.parquet")

_DEFAULT_LEARN_PROCESSORS = [
    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
]


# ---------------------------------------------------------------------------
# DynamicMktNeutralCBOEHandler
# ---------------------------------------------------------------------------
class DynamicMktNeutralCBOEHandler(DataHandlerLP):
    """V9 features + macro + dynamic CBOE features + market-neutral label."""

    def __init__(
        self,
        instruments,
        start_time,
        end_time,
        feature_config,
        macro_features,
        cboe_features=None,
        nday=5,
        macro_lag=1,
        cboe_lag=1,
        fit_start_time=None,
        fit_end_time=None,
    ):
        self.feature_config = feature_config
        self.macro_feature_names = macro_features
        self.cboe_feature_names = cboe_features or []
        self.nday = nday
        self.macro_lag = macro_lag
        self.cboe_lag = cboe_lag

        # Load external data
        self._macro_df = pd.read_parquet(MACRO_PATH)
        self._cboe_df = pd.read_parquet(CBOE_PATH) if self.cboe_feature_names else None
        self._spy_df = pd.read_parquet(SPY_PATH)

        # Build Qlib feature config
        fields, names = self._build_feature_config()
        label_fields, label_names = self._build_label_config()

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": (fields, names),
                    "label": (label_fields, label_names),
                },
                "freq": "day",
            },
        }

        from qlib.contrib.data.handler import check_transform_proc

        learn_processors = check_transform_proc(
            _DEFAULT_LEARN_PROCESSORS, fit_start_time, fit_end_time
        )
        infer_processors = check_transform_proc([], fit_start_time, fit_end_time)

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            learn_processors=learn_processors,
            infer_processors=infer_processors,
        )

    def _build_feature_config(self):
        fields = []
        names = []
        for name, expr in self.feature_config.items():
            fields.append(expr)
            names.append(name)
        return fields, names

    def _build_label_config(self):
        n = self.nday
        expr = f"Ref($close, -{n})/Ref($close, -1) - 1"
        return [expr], ["LABEL0"]

    def process_data(self, **kwargs):
        super().process_data(**kwargs)
        self._add_lagged_macro_features()
        self._add_computed_macro_features()
        if self.cboe_feature_names:
            self._add_cboe_features()
        self._subtract_spy_returns()

    def _add_lagged_macro_features(self):
        available = [c for c in self.macro_feature_names if c in self._macro_df.columns]
        if not available:
            print("[WARN] No macro features found in macro data")
            return
        for dataset_attr in ["_learn", "_infer"]:
            df = getattr(self, dataset_attr, None)
            if df is None:
                continue
            self._merge_external(df, self._macro_df, available, self.macro_lag)

    def _add_computed_macro_features(self):
        """Compute sector_dispersion from sector ETF 20d returns."""
        sector_cols = [c for c in self._macro_df.columns if c.startswith("macro_xl") and c.endswith("_pct_20d")]
        if len(sector_cols) < 5:
            print(f"[WARN] Only {len(sector_cols)} sector columns found, skipping sector_dispersion")
            return
        sector_disp = self._macro_df[sector_cols].std(axis=1)
        sector_disp.name = "macro_sector_dispersion"
        disp_df = sector_disp.to_frame()

        for dataset_attr in ["_learn", "_infer"]:
            df = getattr(self, dataset_attr, None)
            if df is None:
                continue
            self._merge_external(df, disp_df, ["macro_sector_dispersion"], self.macro_lag)

    def _add_cboe_features(self):
        available = [c for c in self.cboe_feature_names if c in self._cboe_df.columns]
        if not available:
            return
        for dataset_attr in ["_learn", "_infer"]:
            df = getattr(self, dataset_attr, None)
            if df is None:
                continue
            self._merge_external(df, self._cboe_df, available, self.cboe_lag)

    def _merge_external(self, main_df, ext_df, columns, lag):
        """Merge external time-series features into the main dataframe."""
        dates = main_df.index.get_level_values("datetime").unique()

        for col in columns:
            series = ext_df[col].copy()
            if lag > 0:
                series = series.shift(lag)
            series = series.reindex(dates)
            # Build aligned array for all rows (broadcast to all instruments)
            date_to_val = series.to_dict()
            col_data = main_df.index.get_level_values("datetime").map(
                lambda d, _m=date_to_val: _m.get(d, np.nan)
            )
            if isinstance(main_df.columns, pd.MultiIndex):
                main_df[("feature", col)] = col_data.values
            else:
                main_df[col] = col_data.values

    def _subtract_spy_returns(self):
        """Subtract SPY forward return from LABEL0 to create market-neutral label."""
        spy_col = f"spy_fwd_return_{self.nday}d"
        if spy_col not in self._spy_df.columns:
            print(f"[WARN] {spy_col} not found in SPY data")
            return

        spy_returns = self._spy_df[spy_col]
        spy_dict = spy_returns.to_dict()

        for dataset_attr in ["_learn", "_infer"]:
            df = getattr(self, dataset_attr, None)
            if df is None:
                continue
            dates = df.index.get_level_values("datetime")

            if isinstance(df.columns, pd.MultiIndex):
                label_col = ("label", "LABEL0")
            else:
                label_col = "LABEL0"

            if label_col not in df.columns:
                continue

            spy_aligned = dates.map(lambda d, _m=spy_dict: _m.get(d, np.nan))
            df[label_col] = df[label_col].values - spy_aligned.values



# ---------------------------------------------------------------------------
# CBOEForwardSelection
# ---------------------------------------------------------------------------
class CBOEForwardSelection(ForwardSelectionBase):
    """Forward selection of CBOE features on v9-mkt-neutral baseline.

    Maps CBOE features into the base class's 'macro' candidate slot since
    they are external time-series features (not Qlib expressions).
    """

    def __init__(
        self,
        symbols,
        nday=5,
        max_features=21,
        min_improvement=0.0005,
        hyperparams=None,
        quiet=False,
    ):
        self.hyperparams = hyperparams or MKT_NEUTRAL_HYPERPARAMS
        self.current_cboe_features = []  # CBOE features selected so far

        super().__init__(
            symbols=symbols,
            nday=nday,
            max_features=max_features,
            min_improvement=min_improvement,
            output_dir=Path("outputs/feature_selection"),
            checkpoint_name="cboe_fwd_sel_mkt_neutral",
            result_prefix="cboe_fwd_sel_mkt_neutral",
            quiet=quiet,
        )

    def prepare_fold_data(self, fold_config, cboe_features=None):
        """Prepare data for one CV fold with given CBOE features."""
        if cboe_features is None:
            cboe_features = self.current_cboe_features

        handler = DynamicMktNeutralCBOEHandler(
            instruments=self.symbols,
            start_time=fold_config["train_start"],
            end_time=fold_config["valid_end"],
            feature_config=V9_STOCK_FEATURES,
            macro_features=V9_MACRO_FEATURES,
            cboe_features=cboe_features,
            nday=self.nday,
            fit_start_time=fold_config["train_start"],
            fit_end_time=fold_config["train_end"],
        )

        dataset = DatasetH(
            handler=handler,
            segments={
                "train": (fold_config["train_start"], fold_config["train_end"]),
                "valid": (fold_config["valid_start"], fold_config["valid_end"]),
            },
        )

        X_train, y_train, _ = prepare_dataset_data(dataset, "train")
        X_valid, y_valid, valid_index = prepare_dataset_data(dataset, "valid")

        del handler, dataset
        gc.collect()

        return X_train, y_train, X_valid, y_valid, valid_index

    def _train_and_evaluate(self, cboe_features, feature_info=None):
        """Train AE-MLP on 4-fold inner CV and return (mean_ic, fold_ics)."""
        fold_ics = []

        for fold_idx, fold in enumerate(INNER_CV_FOLDS):
            if not self.quiet:
                fold_name = fold.get("name", f"Fold {fold_idx + 1}")
                if feature_info:
                    print(f"  [{feature_info}] {fold_name}...", end=" ", flush=True)
                else:
                    print(f"  {fold_name}...", end=" ", flush=True)

            X_train, y_train, X_valid, y_valid, valid_index = self.prepare_fold_data(
                fold, cboe_features=cboe_features
            )

            input_dim = X_train.shape[1]
            model = build_ae_mlp_model({**self.hyperparams, 'num_columns': input_dim})

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_action_loss",
                mode="min",
                patience=5,
                restore_best_weights=True,
                verbose=0,
            )

            model.fit(
                X_train,
                {"decoder": X_train, "ae_action": y_train, "action": y_train},
                validation_data=(
                    X_valid,
                    {"decoder": X_valid, "ae_action": y_valid, "action": y_valid},
                ),
                epochs=30,
                batch_size=self.hyperparams["batch_size"],
                callbacks=[early_stop],
                verbose=0,
            )

            preds = model.predict(X_valid, verbose=0)
            pred_action = preds[2].flatten()

            fold_ic = compute_ic(pred_action, y_valid, valid_index)
            fold_ics.append(fold_ic)

            if not self.quiet:
                print(f"IC={fold_ic:.4f}")

            del model, X_train, y_train, X_valid, y_valid
            tf.keras.backend.clear_session()
            gc.collect()

        mean_ic = float(np.mean(fold_ics))
        if not self.quiet:
            print(f"  -> Mean IC: {mean_ic:.4f} (std: {np.std(fold_ics):.4f})")
        return mean_ic, fold_ics

    def evaluate_feature_set(self):
        """Evaluate current feature set. Returns (mean_ic, fold_ics)."""
        return self._train_and_evaluate(self.current_cboe_features)

    def get_feature_counts(self):
        """Return dict of feature counts by type.

        sum(values) is used by the base class while loop to check against max_features.
        """
        return {
            "stock": len(V9_STOCK_FEATURES),
            "macro": len(V9_MACRO_FEATURES) + 1,  # +1 for sector_dispersion
            "cboe": len(self.current_cboe_features),
        }

    def add_feature(self, name, feature_type, expr=None):
        """Add a CBOE feature to the selected set."""
        self.current_cboe_features.append(name)

    def get_current_features_dict(self):
        """Return dict describing current feature set."""
        return {
            "stock_features": list(V9_STOCK_FEATURES.keys()),
            "macro_features": V9_MACRO_FEATURES + ["macro_sector_dispersion"],
            "cboe_features": list(self.current_cboe_features),
        }

    def get_testable_candidates(self):
        """Return (features_dict, macro_list) matching base class interface.

        CBOE features are external time-series, so they go in the 'macro' slot.
        Stock features slot is empty (no new Qlib expressions to test).
        """
        cboe_candidates = [
            f for f in ALL_CBOE_FEATURES
            if f not in self.current_cboe_features and f not in self.excluded_features
        ]
        # No stock feature candidates; CBOE features go in macro slot
        return {}, cboe_candidates

    def test_feature(self, name, feature_type, expr=None):
        """Test adding one CBOE feature to the current set.

        Returns (mean_ic, fold_ics).
        """
        test_cboe = self.current_cboe_features + [name]
        return self._train_and_evaluate(test_cboe, feature_info=name)

    def cleanup_after_evaluation(self):
        """Clean up TF session and memory."""
        tf.keras.backend.clear_session()
        gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CBOE Forward Selection on v9-mkt-neutral")
    add_common_args(parser)
    args = parser.parse_args()

    pool = args.stock_pool
    symbols = STOCK_POOLS[pool]
    print(f"\n{'='*60}")
    print(f"CBOE Forward Selection on v9-mkt-neutral baseline")
    print(f"{'='*60}")
    print(f"Stock pool: {pool} ({len(symbols)} symbols)")
    print(f"N-day: {args.nday}")
    print(f"Min improvement: {args.min_improvement}")
    print(f"Max features: {args.max_features}")
    print(f"\nBaseline: V9 stock ({len(V9_STOCK_FEATURES)}) + macro ({len(V9_MACRO_FEATURES)}+1) + mkt-neutral label")
    print(f"Candidates: {len(ALL_CBOE_FEATURES)} CBOE features")
    print(f"Hyperparams: mkt-neutral optimized (lr={MKT_NEUTRAL_HYPERPARAMS['lr']:.4f}, "
          f"batch={MKT_NEUTRAL_HYPERPARAMS['batch_size']})")

    # Init Qlib
    qlib.init(
        provider_uri="./my_data/qlib_us",
        region=REG_US,
        custom_ops=TALIB_OPS,
        kernels=1,
    )

    countdown(3)

    selector = CBOEForwardSelection(
        symbols=symbols,
        nday=args.nday,
        max_features=args.max_features,
        min_improvement=args.min_improvement,
    )

    if args.resume:
        ckpt_path = selector.output_dir / f"{selector.checkpoint_name}.json"
        checkpoint = load_checkpoint(ckpt_path)
        if checkpoint:
            selector.current_cboe_features = checkpoint.get("cboe_features", [])
            selector.excluded_features = set(checkpoint.get("excluded_features", []))
            selector.baseline_ic = checkpoint.get("baseline_ic", 0.0)
            selector.current_ic = checkpoint.get("current_ic", 0.0)
            print(f"\nResumed from checkpoint:")
            print(f"  Selected CBOE: {selector.current_cboe_features}")
            print(f"  Excluded: {selector.excluded_features}")
            print(f"  Baseline IC: {selector.baseline_ic}")
            print(f"  Current IC: {selector.current_ic}")

    selector.run(method_name="cboe_forward_selection_mkt_neutral")


if __name__ == "__main__":
    main()
