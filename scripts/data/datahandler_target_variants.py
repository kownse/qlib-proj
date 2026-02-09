"""
Target Engineering Variants for Alpha158 Enhanced V9.

Three handlers that use the same V9 features but different prediction targets:

1. Alpha158_V9_MarketNeutral: label = raw return - SPY return (alpha only)
2. Alpha158_V9_RankTarget: label = cross-sectional rank percentile (0-1)
3. Alpha158_V9_VolScaled: label = return / realized volatility (risk-adjusted)

All inherit from Alpha158_Enhanced_V9 without modifying the original.

Usage:
    python scripts/models/deep/run_ae_mlp_cv.py \
        --params-file outputs/hyperopt_cv/ae_mlp_cv_best_params_v9_best.json \
        --handler v9-mkt-neutral --cv-only --seed 42
"""

import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from data.datahandler_enhanced_v9 import Alpha158_Enhanced_V9
from qlib.data.dataset.handler import DataHandlerLP

PROJECT_ROOT = script_dir.parent
DEFAULT_SPY_PATH = PROJECT_ROOT / "my_data" / "spy_forward_returns.parquet"


class Alpha158_V9_MarketNeutral(Alpha158_Enhanced_V9):
    """
    V9 features with market-neutral label.

    Label = stock_return - SPY_return (excess return / alpha).

    When the market rises 2% and a stock rises 3%, raw label = 3%,
    but market-neutral label = 1%. This removes systematic market noise
    and improves signal-to-noise ratio for alpha prediction.
    """

    def __init__(
        self,
        spy_data_path: Union[str, Path] = None,
        **kwargs,
    ):
        self.spy_data_path = Path(spy_data_path) if spy_data_path else DEFAULT_SPY_PATH
        self._spy_returns = None
        super().__init__(**kwargs)

    def process_data(self, with_fit: bool = False):
        """Override to subtract SPY forward return from label after parent processing."""
        super().process_data(with_fit=with_fit)

        # Load SPY forward returns
        self._spy_returns = self._load_spy_returns()
        if self._spy_returns is None:
            print("Warning: SPY forward returns not available, using raw labels")
            return

        self._make_market_neutral()

    def _load_spy_returns(self) -> Optional[pd.Series]:
        """Load SPY forward returns matching the volatility window."""
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
        """Subtract SPY forward return from stock label."""
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


class Alpha158_V9_RankTarget(Alpha158_Enhanced_V9):
    """
    V9 features with cross-sectional rank target.

    Label = daily rank percentile (0-1) of stock returns.

    Benefits:
    - Uniform distribution, no outliers
    - Robust to extreme returns (earnings surprises, etc.)
    - Directly aligned with Spearman IC metric
    - More stable training signal
    """

    def process_data(self, with_fit: bool = False):
        """Override to rank-transform labels after parent processing."""
        super().process_data(with_fit=with_fit)
        self._apply_rank_transform()

    def _apply_rank_transform(self):
        """Transform labels to cross-sectional rank percentiles."""
        for df_attr in ['_learn', '_infer']:
            df = getattr(self, df_attr, None)
            if df is None:
                continue

            has_multi_columns = isinstance(df.columns, pd.MultiIndex)
            label_col = ('label', 'LABEL0') if has_multi_columns else 'LABEL0'

            if label_col not in df.columns:
                print(f"Warning: label column {label_col} not found in {df_attr}")
                continue

            original_std = df[label_col].std()

            # Cross-sectional rank: for each day, rank all stocks as percentile
            df[label_col] = df[label_col].groupby(level='datetime').rank(pct=True)

            # Center around 0 and scale for better gradient behavior
            # rank in [0, 1] -> centered in [-1.73, 1.73] (approx std=1)
            df[label_col] = (df[label_col] - 0.5) * 3.46

            new_std = df[label_col].std()
            print(f"Rank transform {df_attr}: std {original_std:.6f} -> {new_std:.6f}")


class Alpha158_V9_VolScaled(Alpha158_Enhanced_V9):
    """
    V9 features with volatility-scaled return target.

    Label = return / realized_volatility.

    A +1% return for a low-vol utility stock (vol=10%) is very different
    from a +1% return for a high-vol biotech (vol=50%). Dividing by
    volatility makes signals more comparable across stocks.
    """

    def get_label_config(self):
        """Override label to be volatility-scaled return."""
        N = self.volatility_window
        # return / realized_vol_20d
        # Ref($close, -N)/Ref($close, -1) - 1 is the raw return
        # Std($close/$Ref($close,1)-1, 20) is the 20-day realized volatility of daily returns
        label_expr = (
            f"(Ref($close, -{N})/Ref($close, -1) - 1) / "
            f"(Std($close/Ref($close, 1) - 1, 20) + 1e-8)"
        )
        return [label_expr], ["LABEL0"]
