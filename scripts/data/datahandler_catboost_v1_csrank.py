"""
CatBoost DataHandler V1 + CSRankNorm on Stock Features.

Extends Alpha158_CatBoost_V1 by adding CSRankNorm (Cross-Sectional Rank Normalization)
to infer_processors. This converts the 8 stock features into daily cross-sectional
percentile ranks before training/inference.

Macro features (6) are added post-processing via process_data() and are NOT affected
by CSRankNorm — correct behavior since macro features are market-level (same value
for all stocks on a given day, ranking is meaningless).

Total: 14 features (8 CS-ranked stock + 6 raw macro)
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from data.datahandler_catboost_v1 import Alpha158_CatBoost_V1


class Alpha158_CatBoost_V1_CSRank(Alpha158_CatBoost_V1):
    """CatBoost V1 with CSRankNorm applied to stock features."""

    def __init__(self, infer_processors=None, **kwargs):
        if not infer_processors:
            infer_processors = [
                {"class": "CSRankNorm", "kwargs": {"fields_group": "feature"}},
            ]
        super().__init__(infer_processors=infer_processors, **kwargs)
