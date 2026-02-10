"""
Market-Neutral Target Variants for Tree-Based Models.

Applies market-neutral label transformation (stock return - SPY return)
to CatBoost V1 and LightGBM V1 handlers. Same features, different target.

Usage:
    python scripts/models/tree/run_catboost_nd.py --handler catboost-v1-mkt-neutral --stock-pool sp500
    python scripts/models/tree/run_lgb_nd.py --handler lightgbm-v1-mkt-neutral --stock-pool sp500
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from data.datahandler_catboost_v1 import Alpha158_CatBoost_V1
from data.datahandler_lightgbm_v1 import Alpha158_LightGBM_V1

PROJECT_ROOT = script_dir.parent
DEFAULT_SPY_PATH = PROJECT_ROOT / "my_data" / "spy_forward_returns.parquet"


class _MarketNeutralMixin:
    """Mixin that subtracts SPY forward return from label after processing."""

    def _init_spy(self, spy_data_path=None):
        self.spy_data_path = Path(spy_data_path) if spy_data_path else DEFAULT_SPY_PATH
        self._spy_returns = None

    def _apply_market_neutral(self):
        """Load SPY returns and subtract from label."""
        self._spy_returns = self._load_spy_returns()
        if self._spy_returns is None:
            print("Warning: SPY forward returns not available, using raw labels")
            return
        self._make_market_neutral()

    def _load_spy_returns(self) -> Optional[pd.Series]:
        if not self.spy_data_path.exists():
            print(f"Warning: SPY forward returns not found: {self.spy_data_path}")
            print("Run: python scripts/data/download_spy_forward_returns.py")
            return None
        try:
            df = pd.read_parquet(self.spy_data_path)
            col_name = f"spy_fwd_return_{self.volatility_window}d"
            if col_name not in df.columns:
                print(f"Warning: {col_name} not in SPY data. Available: {list(df.columns)}")
                return None
            series = df[col_name]
            print(f"Loaded SPY forward returns ({col_name}): {series.notna().sum()} valid values")
            return series
        except Exception as e:
            print(f"Warning: Failed to load SPY returns: {e}")
            return None

    def _make_market_neutral(self):
        for df_attr in ['_learn', '_infer']:
            df = getattr(self, df_attr, None)
            if df is None:
                continue
            has_multi_columns = isinstance(df.columns, pd.MultiIndex)
            label_col = ('label', 'LABEL0') if has_multi_columns else 'LABEL0'
            if label_col not in df.columns:
                print(f"Warning: label column {label_col} not found in {df_attr}")
                continue
            dates = df.index.get_level_values('datetime')
            spy_aligned = self._spy_returns.reindex(dates).values
            original_mean = df[label_col].mean()
            df[label_col] = df[label_col] - spy_aligned
            new_mean = df[label_col].mean()
            valid_count = (~np.isnan(spy_aligned)).sum()
            print(f"Market-neutral {df_attr}: mean {original_mean:.6f} -> {new_mean:.6f} "
                  f"({valid_count}/{len(dates)} dates matched)")


class Alpha158_CatBoost_V1_MktNeutral(_MarketNeutralMixin, Alpha158_CatBoost_V1):
    """CatBoost V1 features (14) with market-neutral label."""

    def __init__(self, spy_data_path=None, **kwargs):
        self._init_spy(spy_data_path)
        super().__init__(**kwargs)

    def process_data(self, with_fit: bool = False):
        super().process_data(with_fit=with_fit)
        self._apply_market_neutral()


class Alpha158_LightGBM_V1_MktNeutral(_MarketNeutralMixin, Alpha158_LightGBM_V1):
    """LightGBM V1 features (12) with market-neutral label."""

    def __init__(self, spy_data_path=None, **kwargs):
        self._init_spy(spy_data_path)
        super().__init__(**kwargs)

    def process_data(self, with_fit: bool = False):
        super().process_data(with_fit=with_fit)
        self._apply_market_neutral()
