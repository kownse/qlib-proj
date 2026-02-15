"""
Macro feature constants and helpers for FiLM-conditioned models.

Provides shared macro feature lists, loading, and alignment utilities
used by Transformer-FiLM, TCN-FiLM, and related scripts.
"""

import pandas as pd

from models.common.config import MACRO_DATA_PATH


# Default path to processed macro features parquet
DEFAULT_MACRO_PATH = MACRO_DATA_PATH

# 6 key macro features (minimal set)
MINIMAL_MACRO_FEATURES = [
    "macro_vix_zscore20", "macro_hy_spread_zscore", "macro_credit_stress",
    "macro_tlt_pct_20d", "macro_uso_pct_5d", "macro_risk_on_off",
]

# 23 core macro features (recommended)
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

# Momentum features that need rolling z-score normalization
FEATURES_NEED_ZSCORE = [
    "macro_tlt_pct_20d", "macro_tlt_pct_5d", "macro_uso_pct_5d",
    "macro_gld_pct_5d", "macro_uup_pct_5d", "macro_spy_pct_5d",
]


def load_macro_df(path=None):
    """Load macro features dataframe."""
    path = path or DEFAULT_MACRO_PATH
    df = pd.read_parquet(path)
    print(f"Loaded macro: {df.shape}, {df.index.min()} ~ {df.index.max()}")
    return df


def get_macro_cols(macro_set):
    """Get macro feature columns based on set name."""
    return CORE_MACRO_FEATURES if macro_set == "core" else MINIMAL_MACRO_FEATURES


def prepare_macro(index, macro_df, macro_cols, lag=1):
    """Prepare macro features aligned with stock data index.

    Args:
        index: MultiIndex with 'datetime' level from stock data
        macro_df: DataFrame of macro features (date-indexed)
        macro_cols: list of column names to use
        lag: number of days to shift macro data (avoid lookahead bias)

    Returns:
        numpy array of shape (len(index), n_available_cols)
    """
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
