"""
Time-decay sample reweighter for Qlib models.

Assigns higher weights to recent samples using exponential decay,
so models emphasize recent market patterns while still learning from history.
"""

import numpy as np
import pandas as pd
from qlib.data.dataset.weight import Reweighter


class TimeDecayReweighter(Reweighter):
    """
    Exponential time-decay sample weights.

    weight(t) = max(exp(-ln(2) * years_ago(t) / half_life), min_weight)

    Examples (half_life=5 years, min_weight=0.1):
        2024 data: weight = 1.0
        2019 data: weight = 0.5
        2014 data: weight = 0.25
        2009 data: weight = 0.125
        2004 data: weight = 0.1  (clamped by min_weight)

    Usage:
        reweighter = TimeDecayReweighter(half_life_years=5)
        model.fit(dataset, reweighter=reweighter)
    """

    def __init__(self, half_life_years: float = 5.0, reference_date: str = None, min_weight: float = 0.1):
        """
        Args:
            half_life_years: Half-life in years. After this many years,
                sample weight drops to 50%. Default: 5 years.
            reference_date: Date where weight = 1.0. Default: latest date in data.
            min_weight: Minimum weight floor to prevent extreme down-weighting.
                Default: 0.1 (oldest data retains at least 10% weight).
        """
        self.half_life_years = half_life_years
        self.reference_date = pd.Timestamp(reference_date) if reference_date else None
        self.min_weight = min_weight

    def reweight(self, data):
        """
        Compute time-decay weights for a DataFrame with datetime index.

        Args:
            data: DataFrame with datetime as first level of MultiIndex,
                  or as the index itself.

        Returns:
            pd.Series of weights aligned with data's index.
        """
        # Extract datetime from index
        if isinstance(data.index, pd.MultiIndex):
            datetimes = data.index.get_level_values(0)
        else:
            datetimes = data.index

        # Reference date: latest date in data if not specified
        ref_date = self.reference_date if self.reference_date else datetimes.max()

        # Compute years ago (as float)
        days_ago = (ref_date - datetimes).days
        years_ago = days_ago / 365.25

        # Exponential decay: w = exp(-ln(2) * years_ago / half_life)
        decay_rate = np.log(2) / self.half_life_years
        weights = np.exp(-decay_rate * years_ago.values.astype(float))

        # Apply minimum weight floor
        if self.min_weight > 0:
            weights = np.maximum(weights, self.min_weight)

        return pd.Series(weights, index=data.index)
