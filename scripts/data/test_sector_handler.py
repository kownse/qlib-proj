"""
Unit tests for sector/AI feature integration in Alpha158_Volatility_TALib_Macro handler.

Validates:
1. Sector parquet data quality (one-hot encoding, ai_affinity range)
2. Case-insensitive instrument matching (qlib lowercase vs parquet uppercase)
3. AI affinity time-scaling (ramp from 2020-2024)
4. Feature merge into handler _learn/_infer data
5. End-to-end handler output with sector features
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

SECTOR_PATH = PROJECT_ROOT / "my_data" / "sector_data" / "sector_features.parquet"
MACRO_PATH = PROJECT_ROOT / "my_data" / "macro_processed" / "macro_features.parquet"

SECTOR_COLS = [
    "sector_technology", "sector_healthcare", "sector_financials",
    "sector_consumer_discretionary", "sector_consumer_staples",
    "sector_communication_services", "sector_industrials",
    "sector_energy", "sector_utilities", "sector_real_estate", "sector_materials",
]


# ============================================================================
# 1. Sector parquet data quality
# ============================================================================

class TestSectorParquetData:
    """Test the raw sector_features.parquet data quality."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        if not SECTOR_PATH.exists():
            pytest.skip(f"Sector data not found: {SECTOR_PATH}")
        self.df = pd.read_parquet(SECTOR_PATH)

    def test_expected_columns_exist(self):
        """All 11 sector one-hot columns and ai_affinity should exist."""
        expected = SECTOR_COLS + ["ai_affinity"]
        for col in expected:
            assert col in self.df.columns, f"Missing column: {col}"

    def test_index_is_symbol(self):
        """Index should be stock symbols."""
        assert self.df.index.name == "symbol"
        assert len(self.df) > 0

    def test_sector_onehot_values_binary(self):
        """Sector columns should contain only 0.0 or 1.0."""
        for col in SECTOR_COLS:
            unique_vals = set(self.df[col].unique())
            assert unique_vals <= {0.0, 1.0}, f"{col} has non-binary values: {unique_vals}"

    def test_most_stocks_have_one_sector(self):
        """At least 95% of stocks should belong to exactly one sector."""
        sector_sums = self.df[SECTOR_COLS].sum(axis=1)
        pct_valid = (sector_sums == 1.0).mean()
        assert pct_valid >= 0.95, f"Only {pct_valid:.1%} stocks have exactly 1 sector (need >= 95%)"

    def test_ai_affinity_range(self):
        """ai_affinity should be in [-1, 1] range."""
        assert self.df["ai_affinity"].min() >= -1.0, "ai_affinity below -1"
        assert self.df["ai_affinity"].max() <= 1.0, "ai_affinity above 1"

    def test_no_nan_values(self):
        """No NaN values in any column."""
        nan_counts = self.df.isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        assert len(cols_with_nan) == 0, f"Columns with NaN: {cols_with_nan.to_dict()}"

    def test_known_stocks_have_correct_sector(self):
        """Spot-check known stock-sector mappings."""
        known = {
            "AAPL": "sector_technology",
            "JPM": "sector_financials",
            "JNJ": "sector_healthcare",
            "XOM": "sector_energy",
        }
        for symbol, expected_sector in known.items():
            if symbol in self.df.index:
                assert self.df.loc[symbol, expected_sector] == 1.0, \
                    f"{symbol} should be in {expected_sector}"

    def test_known_ai_stocks_have_positive_affinity(self):
        """Core AI stocks should have positive ai_affinity."""
        ai_stocks = ["NVDA", "MSFT", "GOOG", "META", "AMZN"]
        for symbol in ai_stocks:
            if symbol in self.df.index:
                assert self.df.loc[symbol, "ai_affinity"] > 0, \
                    f"{symbol} should have positive ai_affinity, got {self.df.loc[symbol, 'ai_affinity']}"


# ============================================================================
# 2. Case-insensitive instrument matching
# ============================================================================

class TestInstrumentCaseMatching:
    """Test that sector features correctly match qlib's lowercase instrument names."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        if not SECTOR_PATH.exists():
            pytest.skip(f"Sector data not found: {SECTOR_PATH}")
        self.df_raw = pd.read_parquet(SECTOR_PATH)

    def test_raw_parquet_is_uppercase(self):
        """Verify raw parquet uses uppercase symbols."""
        sample = self.df_raw.index[:10].tolist()
        assert all(s == s.upper() for s in sample), \
            f"Expected uppercase index, got: {sample}"

    def test_lowercase_reindex_matches(self):
        """After lowercasing, qlib-style instruments should match."""
        df = self.df_raw.copy()
        df.index = df.index.str.lower()

        qlib_instruments = ["aapl", "msft", "nvda", "goog", "meta"]
        aligned = df.reindex(qlib_instruments, fill_value=0.0)

        # These should NOT all be zero
        assert not (aligned.values == 0).all(), \
            "All values are zero after reindex — case matching still broken"

    def test_uppercase_reindex_without_fix_fails(self):
        """Without the fix, lowercase instruments produce all zeros."""
        qlib_instruments = ["aapl", "msft", "nvda"]
        aligned = self.df_raw.reindex(qlib_instruments, fill_value=0.0)
        # Without fix, everything is 0
        assert (aligned.values == 0).all(), \
            "Raw uppercase parquet should NOT match lowercase qlib instruments"

    def test_ai_affinity_values_after_lowercase(self):
        """Specific AI stocks should have correct values after lowercase fix."""
        df = self.df_raw.copy()
        df.index = df.index.str.lower()

        # NVDA should have ai_affinity=1.0
        assert df.loc["nvda", "ai_affinity"] == 1.0
        # MSFT should have ai_affinity=1.0
        assert df.loc["msft", "ai_affinity"] == 1.0
        # JPM should have ai_affinity=0.0 (not an AI stock)
        if "jpm" in df.index:
            assert df.loc["jpm", "ai_affinity"] == 0.0


# ============================================================================
# 3. AI affinity time-scaling
# ============================================================================

class TestAIAffinityTimeScaling:
    """Test the time-scaling logic for AI affinity (ramp 2020-2024)."""

    def _make_mock_sector_df(self):
        """Create a mock sector DataFrame with known values."""
        data = {
            "sector_technology": [1.0, 0.0, 1.0],
            "ai_affinity": [1.0, 0.0, 0.5],
        }
        return pd.DataFrame(data, index=pd.Index(["nvda", "jpm", "tsla"], name="symbol"))

    def _make_mock_main_df(self, dates, instruments):
        """Create a mock main DataFrame with MultiIndex (datetime, instrument)."""
        idx = pd.MultiIndex.from_product(
            [pd.DatetimeIndex(dates), instruments],
            names=["datetime", "instrument"]
        )
        return pd.DataFrame({"feature_x": np.random.randn(len(idx))}, index=idx)

    def test_before_2020_ai_affinity_is_zero(self):
        """AI affinity should be 0 for dates before 2020."""
        from data.datahandler_macro import Alpha158_Volatility_TALib_Macro as Handler

        sector_df = self._make_mock_sector_df()
        main_df = self._make_mock_main_df(["2019-06-01"], ["nvda", "jpm", "tsla"])

        # Manually call the merge logic
        handler = object.__new__(Handler)
        handler._sector_df = sector_df
        handler.sector_features = "ai_only"
        handler.AI_AFFINITY_FEATURE = "ai_affinity"
        handler.AI_AFFINITY_RAMP_START = pd.Timestamp("2020-01-01")
        handler.AI_AFFINITY_RAMP_END = pd.Timestamp("2024-01-01")

        result = handler._merge_sector_to_df(main_df, ["ai_affinity"])
        ai_col = result["ai_affinity"] if "ai_affinity" in result.columns else result[("feature", "ai_affinity")]

        # Before 2020, all AI affinity should be 0
        assert (ai_col == 0).all(), f"AI affinity should be 0 before 2020, got:\n{ai_col}"

    def test_after_2024_ai_affinity_is_full(self):
        """AI affinity should be at full value for dates after 2024."""
        from data.datahandler_macro import Alpha158_Volatility_TALib_Macro as Handler

        sector_df = self._make_mock_sector_df()
        main_df = self._make_mock_main_df(["2025-01-15"], ["nvda", "jpm", "tsla"])

        handler = object.__new__(Handler)
        handler._sector_df = sector_df
        handler.sector_features = "ai_only"
        handler.AI_AFFINITY_FEATURE = "ai_affinity"
        handler.AI_AFFINITY_RAMP_START = pd.Timestamp("2020-01-01")
        handler.AI_AFFINITY_RAMP_END = pd.Timestamp("2024-01-01")

        result = handler._merge_sector_to_df(main_df, ["ai_affinity"])
        ai_col = result["ai_affinity"] if "ai_affinity" in result.columns else result[("feature", "ai_affinity")]

        # NVDA: 1.0 * 1.0 = 1.0 (full scale)
        nvda_val = ai_col.loc[pd.Timestamp("2025-01-15"), "nvda"]
        assert abs(nvda_val - 1.0) < 0.01, f"NVDA ai_affinity should be 1.0 after 2024, got {nvda_val}"

        # JPM: 0.0 * 1.0 = 0.0 (no AI affinity)
        jpm_val = ai_col.loc[pd.Timestamp("2025-01-15"), "jpm"]
        assert abs(jpm_val - 0.0) < 0.01, f"JPM ai_affinity should be 0.0, got {jpm_val}"

    def test_mid_ramp_ai_affinity_is_scaled(self):
        """AI affinity should be ~50% at midpoint of ramp (2022-01-01)."""
        from data.datahandler_macro import Alpha158_Volatility_TALib_Macro as Handler

        sector_df = self._make_mock_sector_df()
        main_df = self._make_mock_main_df(["2022-01-01"], ["nvda"])

        handler = object.__new__(Handler)
        handler._sector_df = sector_df
        handler.sector_features = "ai_only"
        handler.AI_AFFINITY_FEATURE = "ai_affinity"
        handler.AI_AFFINITY_RAMP_START = pd.Timestamp("2020-01-01")
        handler.AI_AFFINITY_RAMP_END = pd.Timestamp("2024-01-01")

        result = handler._merge_sector_to_df(main_df, ["ai_affinity"])
        ai_col = result["ai_affinity"] if "ai_affinity" in result.columns else result[("feature", "ai_affinity")]

        nvda_val = ai_col.iloc[0]
        # ~50% of ramp (2 years out of 4)
        assert 0.4 < nvda_val < 0.6, f"NVDA ai_affinity at 2022-01-01 should be ~0.5, got {nvda_val}"


# ============================================================================
# 4. End-to-end handler integration
# ============================================================================

class TestHandlerIntegration:
    """End-to-end test: initialize handler and verify sector features appear in output."""

    @pytest.fixture(autouse=True)
    def check_data(self):
        if not SECTOR_PATH.exists():
            pytest.skip("Sector data not available")
        if not MACRO_PATH.exists():
            pytest.skip("Macro data not available")

    def _init_qlib(self):
        """Initialize qlib."""
        import qlib
        from qlib.constant import REG_US
        from utils.talib_ops import TALIB_OPS
        qlib.init(
            provider_uri=str(PROJECT_ROOT / "my_data" / "qlib_us"),
            region=REG_US,
            custom_ops=TALIB_OPS,
            kernels=1,
        )

    def test_sector_features_in_handler_output(self):
        """sector+ai handler should produce non-zero sector features in output."""
        self._init_qlib()
        from data.datahandler_macro import Alpha158_Volatility_TALib_Macro

        handler = Alpha158_Volatility_TALib_Macro(
            instruments="sp500",
            start_time="2024-01-01",
            end_time="2024-01-31",
            fit_start_time="2024-01-01",
            fit_end_time="2024-01-31",
            sector_features="sector+ai",
            macro_features="none",
        )

        # Get learn data
        learn_df = handler.fetch(col_set="feature", data_key="learn")
        cols = learn_df.columns
        if isinstance(cols, pd.MultiIndex):
            feature_names = [c[1] if len(c) > 1 else c[0] for c in cols]
        else:
            feature_names = cols.tolist()

        # Check sector columns exist
        for sc in SECTOR_COLS:
            assert sc in feature_names, f"Sector column '{sc}' missing from handler output"
        assert "ai_affinity" in feature_names, "ai_affinity missing from handler output"

        # Check sector values are NOT all zeros
        sector_data = learn_df[[c for c in learn_df.columns
                                if (c[1] if isinstance(c, tuple) else c) in SECTOR_COLS]]
        sector_sum = sector_data.sum().sum()
        assert sector_sum > 0, \
            "All sector features are zero — instrument case matching likely broken"

        # Check ai_affinity has non-zero values (for 2024 data, should be at/near full scale)
        ai_col_name = [c for c in learn_df.columns
                       if (c[1] if isinstance(c, tuple) else c) == "ai_affinity"][0]
        ai_data = learn_df[ai_col_name]
        assert ai_data.abs().sum() > 0, \
            "ai_affinity is all zeros — instrument case matching likely broken"

        # NVDA should have ai_affinity close to 1.0 in 2024
        if "nvda" in learn_df.index.get_level_values("instrument"):
            nvda_ai = ai_data.loc[:, "nvda"].mean()
            assert nvda_ai > 0.8, f"NVDA ai_affinity in 2024 should be ~1.0, got {nvda_ai:.4f}"

    def test_ai_only_handler_has_single_feature(self):
        """ai_only handler should add only the ai_affinity feature, no sector one-hots."""
        self._init_qlib()
        from data.datahandler_macro import Alpha158_Volatility_TALib_Macro

        handler = Alpha158_Volatility_TALib_Macro(
            instruments="sp500",
            start_time="2024-01-01",
            end_time="2024-01-31",
            fit_start_time="2024-01-01",
            fit_end_time="2024-01-31",
            sector_features="ai_only",
            macro_features="none",
        )

        learn_df = handler.fetch(col_set="feature", data_key="learn")
        cols = learn_df.columns
        if isinstance(cols, pd.MultiIndex):
            feature_names = [c[1] if len(c) > 1 else c[0] for c in cols]
        else:
            feature_names = cols.tolist()

        assert "ai_affinity" in feature_names, "ai_affinity should be in ai_only handler"
        for sc in SECTOR_COLS:
            assert sc not in feature_names, f"Sector column '{sc}' should NOT be in ai_only handler"


# ============================================================================
# 5. Macro features sanity check
# ============================================================================

class TestMacroFeatures:
    """Sanity check that macro features are loaded and merged correctly."""

    @pytest.fixture(autouse=True)
    def check_data(self):
        if not MACRO_PATH.exists():
            pytest.skip("Macro data not available")

    def test_macro_parquet_not_empty(self):
        df = pd.read_parquet(MACRO_PATH)
        assert len(df) > 100, f"Macro data too small: {len(df)} rows"

    def test_macro_parquet_has_expected_columns(self):
        df = pd.read_parquet(MACRO_PATH)
        expected = ["macro_vix_level", "macro_spy_pct_5d", "macro_yield_10y"]
        for col in expected:
            assert col in df.columns, f"Missing macro column: {col}"

    def test_macro_index_is_datetime(self):
        df = pd.read_parquet(MACRO_PATH)
        assert isinstance(df.index, pd.DatetimeIndex), \
            f"Macro index should be DatetimeIndex, got {type(df.index)}"

    def test_macro_features_in_handler(self):
        """Macro features should appear in handler output with non-zero values."""
        import qlib
        from qlib.constant import REG_US
        from utils.talib_ops import TALIB_OPS
        qlib.init(
            provider_uri=str(PROJECT_ROOT / "my_data" / "qlib_us"),
            region=REG_US,
            custom_ops=TALIB_OPS,
            kernels=1,
        )
        from data.datahandler_macro import Alpha158_Volatility_TALib_Macro

        handler = Alpha158_Volatility_TALib_Macro(
            instruments="sp500",
            start_time="2024-01-01",
            end_time="2024-01-31",
            fit_start_time="2024-01-01",
            fit_end_time="2024-01-31",
            sector_features="none",
            macro_features="core",
        )

        learn_df = handler.fetch(col_set="feature", data_key="learn")
        cols = learn_df.columns
        if isinstance(cols, pd.MultiIndex):
            feature_names = [c[1] if len(c) > 1 else c[0] for c in cols]
        else:
            feature_names = cols.tolist()

        assert "macro_vix_level" in feature_names, "macro_vix_level missing from handler"
        macro_cols = [c for c in learn_df.columns
                      if (c[1] if isinstance(c, tuple) else c).startswith("macro_")]
        macro_data = learn_df[macro_cols]
        assert macro_data.abs().sum().sum() > 0, "All macro features are zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
