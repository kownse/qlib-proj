"""
Time-series model utilities for deep learning runners.

Provides handler → d_feat mapping and d_feat/seq_len resolution logic
used by TCN, ALSTM, Transformer, TKAN, and related scripts.
"""

# Canonical handler → d_feat mapping (union of all scripts)
HANDLER_D_FEAT = {
    'alpha360': 6,           # 6 features × 60 timesteps (includes VWAP)
    'alpha300': 5,           # 5 features × 60 timesteps (no VWAP - recommended for US data)
    'alpha300-ts': 5,        # 5 features × 60 timesteps (time-series norm)
    'alpha360-macro': 29,    # (6 + 23 core macro) × 60 = 1740 total
    'alpha300-macro': 11,    # (5 + 6 minimal macro) × 60
    'alpha180': 6,           # 6 features × 30 timesteps
    'alpha180-macro': 6,
    'alpha158': 158,         # No temporal structure
    'alpha158_vol': 158,
    'alpha158_vol_talib': 158,
    'alpha158-talib': 158,
    'alpha158-talib-lite': 158,
}


def resolve_d_feat_and_seq_len(handler_name, total_features, user_d_feat=None):
    """Resolve d_feat and seq_len from handler name and total feature count.

    Args:
        handler_name: Name of the handler (e.g. 'alpha360', 'alpha300-ts')
        total_features: Total number of features in the dataset
        user_d_feat: User-specified d_feat override (from --d-feat arg), or None/0

    Returns:
        tuple: (d_feat, seq_len)
    """
    if user_d_feat:
        d_feat = user_d_feat
    else:
        d_feat = HANDLER_D_FEAT.get(handler_name, total_features)

    if total_features % d_feat == 0:
        seq_len = total_features // d_feat
    else:
        print(f"    WARNING: total_features ({total_features}) not divisible by d_feat ({d_feat})")
        print(f"    Falling back to d_feat = total_features (no temporal structure)")
        d_feat = total_features
        seq_len = 1

    return d_feat, seq_len
