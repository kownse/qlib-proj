"""
AI Affinity Filter for ensemble predictions.

Penalizes or excludes stocks based on AI disruption risk scores,
preventing the model from selecting structurally declining stocks
(e.g., CHGG) as "buy the dip" candidates.

Two modes:
  - penalty: Softly adjust prediction scores based on affinity
  - exclude: Remove stocks below an affinity threshold
"""

import numpy as np
import pandas as pd

from data.ai_affinity_labels import AI_AFFINITY_SCORES

# Time scaling: AI impact ramped from 2020 to 2024
AI_AFFINITY_RAMP_START = pd.Timestamp("2020-01-01")
AI_AFFINITY_RAMP_END = pd.Timestamp("2024-01-01")


def _get_time_scale_factors(dates: pd.DatetimeIndex) -> np.ndarray:
    """Compute time scaling factors for AI affinity (0 before 2020, linear ramp to 2024, 1 after)."""
    ramp_start = AI_AFFINITY_RAMP_START.value
    ramp_end = AI_AFFINITY_RAMP_END.value
    ramp_duration = ramp_end - ramp_start

    dt_values = dates.values.astype("int64")
    scale = (dt_values - ramp_start) / ramp_duration
    return np.clip(scale, 0.0, 1.0)


def _build_affinity_map() -> dict:
    """Build lowercase symbol -> affinity score mapping."""
    return {sym.lower(): score for sym, score in AI_AFFINITY_SCORES.items()}


def apply_ai_affinity_filter(
    pred: pd.Series,
    mode: str = "penalty",
    penalty_weight: float = 0.5,
    bonus_weight: float = 0.0,
    exclude_threshold: int = -1,
    time_scale: bool = True,
    verbose: bool = True,
) -> pd.Series:
    """
    Apply AI affinity filter to ensemble predictions.

    Parameters
    ----------
    pred : pd.Series
        Prediction scores with MultiIndex(datetime, instrument).
    mode : str
        'penalty' to adjust scores, 'exclude' to drop stocks.
    penalty_weight : float
        Penalty multiplier for negative-affinity stocks (in units of daily std).
    bonus_weight : float
        Bonus multiplier for positive-affinity stocks (in units of daily std).
    exclude_threshold : int
        Raw affinity threshold for exclude mode (drop if affinity <= threshold).
    time_scale : bool
        Apply temporal scaling (0 before 2020, ramp to 2024).
    verbose : bool
        Print summary of affected stocks.

    Returns
    -------
    pd.Series
        Filtered/adjusted predictions.
    """
    if mode not in ("penalty", "exclude"):
        raise ValueError(f"Unknown AI filter mode: {mode}")

    affinity_map = _build_affinity_map()

    # Extract instrument names from MultiIndex level 1
    instruments = pred.index.get_level_values(1)
    # Map instruments to raw affinity scores (0 for unlabeled)
    raw_affinity = instruments.map(lambda s: affinity_map.get(s.lower(), 0)).values.astype(float)

    # Count affected stocks for summary
    unique_instruments = instruments.unique()
    neg_stocks = [s for s in unique_instruments if affinity_map.get(s.lower(), 0) < 0]
    pos_stocks = [s for s in unique_instruments if affinity_map.get(s.lower(), 0) > 0]

    if mode == "exclude":
        mask = raw_affinity > exclude_threshold
        if time_scale:
            dates = pred.index.get_level_values(0)
            scale = _get_time_scale_factors(dates)
            # Only exclude when time scale is significant (> 0.5)
            # For dates before ramp, don't exclude
            mask = mask | (scale < 0.5)

        filtered = pred[mask]

        if verbose:
            n_removed = len(pred) - len(filtered)
            removed_stocks = [s for s in unique_instruments
                              if affinity_map.get(s.lower(), 0) <= exclude_threshold]
            print(f"\n    AI Affinity Filter (exclude mode, threshold={exclude_threshold}):")
            print(f"      Removed {n_removed} predictions ({n_removed/len(pred)*100:.1f}%)")
            print(f"      Excluded stocks ({len(removed_stocks)}): {', '.join(sorted(removed_stocks)[:10])}")
            if len(neg_stocks) > len(removed_stocks):
                kept_neg = [s for s in neg_stocks if s not in removed_stocks]
                print(f"      Negative-affinity stocks kept: {', '.join(sorted(kept_neg)[:10])}")

        return filtered

    # Penalty mode
    result = pred.copy()

    # Compute daily cross-sectional std for scale-invariant adjustment
    dates = pred.index.get_level_values(0)
    daily_std = pred.groupby(level=0).std()
    std_per_row = dates.map(daily_std).values

    # Normalize affinity to [-1, 1] range (raw scores are -2 to +2)
    affinity_normalized = raw_affinity / 2.0

    # Compute time scale factors
    if time_scale:
        scale = _get_time_scale_factors(dates)
    else:
        scale = np.ones(len(pred))

    # Apply penalty (negative affinity) and bonus (positive affinity) separately
    adjustment = np.zeros(len(pred))
    neg_mask = affinity_normalized < 0
    pos_mask = affinity_normalized > 0

    if penalty_weight > 0:
        # Negative affinity -> negative adjustment (penalize)
        adjustment[neg_mask] = affinity_normalized[neg_mask] * penalty_weight * std_per_row[neg_mask] * scale[neg_mask]

    if bonus_weight > 0:
        # Positive affinity -> positive adjustment (bonus)
        adjustment[pos_mask] = affinity_normalized[pos_mask] * bonus_weight * std_per_row[pos_mask] * scale[pos_mask]

    result = result + adjustment

    if verbose:
        print(f"\n    AI Affinity Filter (penalty mode, weight={penalty_weight}, bonus={bonus_weight}):")
        print(f"      Time scaling: {'enabled' if time_scale else 'disabled'}")
        print(f"      Negative-affinity stocks ({len(neg_stocks)}): {', '.join(sorted(neg_stocks)[:10])}")
        print(f"      Positive-affinity stocks ({len(pos_stocks)}): {', '.join(sorted(pos_stocks)[:10])}")
        if len(neg_stocks) > 0:
            neg_adj = adjustment[neg_mask]
            if len(neg_adj) > 0:
                print(f"      Avg penalty (neg stocks): {neg_adj.mean():.6f}")
        if len(pos_stocks) > 0 and bonus_weight > 0:
            pos_adj = adjustment[pos_mask]
            if len(pos_adj) > 0:
                print(f"      Avg bonus (pos stocks): {pos_adj.mean():.6f}")

    return result
