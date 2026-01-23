"""
DataHandler for MASTER model integration

Combines Alpha158 stock features with market information features for MASTER model.
MASTER paper: https://arxiv.org/abs/2312.15235

Features:
- Stock features (Alpha158): 158 features per stock
- Market info features: 63 features (21 per index × 3 indices)
- Total: 221 features

Market information features (from SPY, QQQ, IWM):
- Daily return (1 per index)
- Return mean over 5, 10, 20, 30, 60 days (5 per index)
- Return std over 5, 10, 20, 30, 60 days (5 per index)
- Volume mean ratio over 5, 10, 20, 30, 60 days (5 per index)
- Volume std ratio over 5, 10, 20, 30, 60 days (5 per index)

Usage:
    handler = Alpha158_Master(
        volatility_window=2,
        instruments=["AAPL", "MSFT", "NVDA"],
        start_time="2020-01-01",
        end_time="2024-12-31",
    )

    # For MASTER model configuration:
    # gate_input_start_index = 158  (start of market info features)
    # gate_input_end_index = 221    (end of market info features)
"""

import sys
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

# Add scripts directory to path for imports
script_dir = Path(__file__).parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

from qlib.contrib.data.handler import Alpha158, check_transform_proc, _DEFAULT_LEARN_PROCESSORS
from qlib.contrib.data.loader import Alpha158DL
from qlib.data.dataset.handler import DataHandlerLP

# Project root
PROJECT_ROOT = script_dir.parent

# Default market info features path
DEFAULT_MARKET_INFO_PATH = PROJECT_ROOT / "my_data" / "macro_processed" / "master_market_info.parquet"

# Market indices
MARKET_INDICES = ["SPY", "QQQ", "IWM"]

# Rolling windows for market info features
WINDOWS = [5, 10, 20, 30, 60]


def get_master_market_info_features() -> List[str]:
    """
    Get list of all 63 MASTER market information feature names.

    Returns:
        List of 63 feature names
    """
    names = []
    for symbol in MARKET_INDICES:
        sym_lower = symbol.lower()
        # Daily return (1)
        names.append(f"mkt_{sym_lower}_ret_1d")
        # Return mean (5)
        for w in WINDOWS:
            names.append(f"mkt_{sym_lower}_ret_mean_{w}d")
        # Return std (5)
        for w in WINDOWS:
            names.append(f"mkt_{sym_lower}_ret_std_{w}d")
        # Volume mean ratio (5)
        for w in WINDOWS:
            names.append(f"mkt_{sym_lower}_vol_mean_{w}d")
        # Volume std ratio (5)
        for w in WINDOWS:
            names.append(f"mkt_{sym_lower}_vol_std_{w}d")
    return names


# Feature name list for external reference
MASTER_MARKET_INFO_FEATURES = get_master_market_info_features()


class Alpha158_Master(DataHandlerLP):
    """
    Alpha158 features + MASTER market information features.

    Designed for MASTER model integration which uses market-wide information
    to gate individual stock predictions.

    Features:
    - Alpha158 stock features: 158 features
    - Market info features: 63 features (shared across all stocks)
    - Total: 221 features

    MASTER model configuration:
    - gate_input_start_index: 158 (start of market info features)
    - gate_input_end_index: 221 (end of market info features)

    Usage:
        handler = Alpha158_Master(
            volatility_window=2,
            instruments="sp500",
            start_time="2020-01-01",
            end_time="2024-12-31",
        )

        # Get data for model training
        data = handler.fetch()  # Shape: (N_samples, 221)

        # MASTER model config
        model = MASTER(
            d_feat=221,
            gate_input_start_index=158,
            gate_input_end_index=221,
            ...
        )
    """

    # Number of Alpha158 features (excluding VWAP and volume-based problematic features)
    # Original 158 - VWAP(1) - VMA(5) - VSTD(5) - WVMA(5) = 142
    N_STOCK_FEATURES = 142

    # Number of market info features (21 per index × 3 indices)
    N_MARKET_FEATURES = 63

    # Total features
    D_FEAT = N_STOCK_FEATURES + N_MARKET_FEATURES  # 142 + 63 = 205

    # Gate input indices for MASTER model
    GATE_INPUT_START_INDEX = N_STOCK_FEATURES  # 142
    GATE_INPUT_END_INDEX = D_FEAT  # 205

    def __init__(
        self,
        volatility_window: int = 2,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq: str = "day",
        infer_processors=[],
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        # Market info parameters
        market_info_path: Union[str, Path] = None,
        **kwargs,
    ):
        """
        Initialize Alpha158 + MASTER market info DataHandler.

        Args:
            volatility_window: Prediction window (days) for label
            instruments: Stock pool (e.g., "sp500", list of symbols)
            start_time: Data start time
            end_time: Data end time
            freq: Data frequency ("day")
            infer_processors: Processors for inference
            learn_processors: Processors for training
            fit_start_time: Fit start time for processors
            fit_end_time: Fit end time for processors
            process_type: Processing type
            filter_pipe: Data filter pipe
            inst_processors: Instrument processors
            market_info_path: Path to market info parquet file
            **kwargs: Additional arguments
        """
        self.volatility_window = volatility_window
        self.market_info_path = Path(market_info_path) if market_info_path else DEFAULT_MARKET_INFO_PATH

        # Load market info features
        self._market_info_df = self._load_market_info()

        if learn_processors is None:
            learn_processors = _DEFAULT_LEARN_PROCESSORS

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        # Use Alpha158 config (excluding problematic features)
        # NOTE: Must use Alpha158DL.get_feature_config() which accepts config parameter
        #       Alpha158.get_feature_config() is an instance method that ignores config!
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW"],  # Exclude VWAP (not available in US data)
            },
            "rolling": {
                # Exclude volume-based features that cause extreme values after CSZScoreNorm
                # These features have very small std on some days, causing z-score explosion
                "exclude": ["VMA", "VSTD", "WVMA"],
            },
        }
        fields, names = Alpha158DL.get_feature_config(conf)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": (fields, names),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def _load_market_info(self) -> Optional[pd.DataFrame]:
        """Load market info features from parquet file."""
        if not self.market_info_path.exists():
            print(f"Warning: Market info file not found: {self.market_info_path}")
            print("Run: python scripts/data/process_master_market_info.py")
            return None

        try:
            df = pd.read_parquet(self.market_info_path)
            print(f"Loaded market info features: {df.shape}, "
                  f"date range: {df.index.min().date()} to {df.index.max().date()}")
            return df
        except Exception as e:
            print(f"Warning: Failed to load market info features: {e}")
            return None

    def process_data(self, with_fit: bool = False):
        """
        Override process_data to add market info features AFTER processors run.
        """
        # First, call parent's process_data
        super().process_data(with_fit=with_fit)

        # Fill NaN values in Alpha158 features (from CSZScoreNorm when std=0)
        self._fill_nan_in_processed_data()

        # Then add market info features to _learn and _infer
        if self._market_info_df is not None:
            self._add_market_info_to_processed_data()

    def _fill_nan_in_processed_data(self):
        """Fill NaN values in processed data with 0."""
        if hasattr(self, "_learn") and self._learn is not None:
            nan_count = self._learn.isna().sum().sum()
            if nan_count > 0:
                self._learn = self._learn.fillna(0)
                print(f"    Note: Filled {nan_count} NaN values in learn data with 0")

        if hasattr(self, "_infer") and self._infer is not None:
            nan_count = self._infer.isna().sum().sum()
            if nan_count > 0:
                self._infer = self._infer.fillna(0)
                print(f"    Note: Filled {nan_count} NaN values in infer data with 0")

    def _add_market_info_to_processed_data(self):
        """Add market info features to _learn and _infer after processors run."""
        try:
            available_cols = [c for c in MASTER_MARKET_INFO_FEATURES if c in self._market_info_df.columns]

            if not available_cols:
                print("Warning: No market info features available")
                return

            # Add to _learn
            if hasattr(self, "_learn") and self._learn is not None:
                self._learn = self._merge_market_info_to_df(self._learn, available_cols)
                print(f"    Added {len(available_cols)} market info features to learn data")

            # Add to _infer
            if hasattr(self, "_infer") and self._infer is not None:
                self._infer = self._merge_market_info_to_df(self._infer, available_cols)

        except Exception as e:
            print(f"Warning: Error adding market info features: {e}")
            import traceback
            traceback.print_exc()

    def _merge_market_info_to_df(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """
        Merge market info features into a DataFrame.

        Market info features are the same for all stocks on the same day,
        so we align by date and broadcast to all instruments.

        Args:
            df: DataFrame with (datetime, instrument) MultiIndex
            cols: List of market info column names to add

        Returns:
            DataFrame with added market info columns
        """
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        # Build all market info columns at once to avoid fragmentation
        market_data = {}
        nan_count = 0
        for col in cols:
            market_series = self._market_info_df[col]
            aligned_values = market_series.reindex(main_datetimes).values

            # Fill NaN with 0 (for dates not in market info)
            col_nan = pd.isna(aligned_values).sum()
            if col_nan > 0:
                nan_count += col_nan
                aligned_values = pd.Series(aligned_values).fillna(0).values

            if has_multi_columns:
                market_data[('feature', col)] = aligned_values
            else:
                market_data[col] = aligned_values

        if nan_count > 0:
            print(f"    Note: Filled {nan_count} NaN values in market info features with 0")

        # Create DataFrame with all market info columns
        market_df = pd.DataFrame(market_data, index=df.index)

        # Use pd.concat to merge all columns at once (avoids fragmentation warning)
        merged = pd.concat([df, market_df], axis=1, copy=False)

        # Return a copy to ensure defragmentation
        return merged.copy()

    def get_label_config(self):
        """Return N-day return label."""
        return_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [return_expr], ["LABEL0"]

    @classmethod
    def get_model_config(cls) -> dict:
        """
        Get MASTER model configuration for this handler.

        Returns:
            Dict with d_feat, gate_input_start_index, gate_input_end_index
        """
        return {
            "d_feat": cls.D_FEAT,
            "gate_input_start_index": cls.GATE_INPUT_START_INDEX,
            "gate_input_end_index": cls.GATE_INPUT_END_INDEX,
            "n_stock_features": cls.N_STOCK_FEATURES,
            "n_market_features": cls.N_MARKET_FEATURES,
        }


class Alpha360_Master(DataHandlerLP):
    """
    Alpha360 features + MASTER market information features (time-aligned).

    For MASTER model with sequential input (LSTM/Transformer based).

    Features per timestep:
    - Alpha360 stock features: 6 (CLOSE, OPEN, HIGH, LOW, VWAP, VOLUME)
    - Market info features: 63 features per timestep
    - Total per timestep: 69

    Structure: (60 timesteps, 69 features) = 4140 total features

    MASTER model configuration:
    - d_feat: 69 (features per timestep)
    - gate_input_start_index: 6 (start of market info within each timestep)
    - gate_input_end_index: 69 (end of market info within each timestep)
    """

    # Features per timestep
    N_STOCK_FEATURES_PER_STEP = 6  # CLOSE, OPEN, HIGH, LOW, VWAP, VOLUME
    N_MARKET_FEATURES = 63
    D_FEAT_PER_STEP = N_STOCK_FEATURES_PER_STEP + N_MARKET_FEATURES  # 69

    # Total features (60 timesteps)
    SEQ_LEN = 60
    D_FEAT_TOTAL = D_FEAT_PER_STEP * SEQ_LEN  # 4140

    # Gate input indices (within each timestep)
    GATE_INPUT_START_INDEX = N_STOCK_FEATURES_PER_STEP  # 6
    GATE_INPUT_END_INDEX = D_FEAT_PER_STEP  # 69

    def __init__(
        self,
        volatility_window: int = 2,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq: str = "day",
        infer_processors=[],
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        # Market info parameters
        market_info_path: Union[str, Path] = None,
        **kwargs,
    ):
        """
        Initialize Alpha360 + MASTER market info DataHandler.

        Args:
            volatility_window: Prediction window (days) for label
            market_info_path: Path to market info parquet file
            **kwargs: Additional arguments for parent class
        """
        self.volatility_window = volatility_window
        self.market_info_path = Path(market_info_path) if market_info_path else DEFAULT_MARKET_INFO_PATH

        # Load market info features
        self._market_info_df = self._load_market_info()

        if learn_processors is None:
            learn_processors = _DEFAULT_LEARN_PROCESSORS

        from qlib.contrib.data.loader import Alpha360DL

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        # Use Alpha360's feature config
        fields, names = Alpha360DL.get_feature_config()

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": (fields, names),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    _load_market_info = Alpha158_Master._load_market_info

    def process_data(self, with_fit: bool = False):
        """Override process_data to add time-aligned market info features."""
        super().process_data(with_fit=with_fit)

        if self._market_info_df is not None:
            self._add_market_info_to_processed_data()

    def _add_market_info_to_processed_data(self):
        """Add time-aligned market info features to _learn and _infer."""
        try:
            available_cols = [c for c in MASTER_MARKET_INFO_FEATURES if c in self._market_info_df.columns]

            if not available_cols:
                print("Warning: No market info features available")
                return

            # Add to _learn with temporal expansion
            if hasattr(self, "_learn") and self._learn is not None:
                self._learn = self._expand_market_info_temporally(self._learn, available_cols)
                num_expanded = len(available_cols) * 60
                print(f"Added {num_expanded} market info features to learn data "
                      f"({len(available_cols)} features × 60 timesteps)")

            # Add to _infer with temporal expansion
            if hasattr(self, "_infer") and self._infer is not None:
                self._infer = self._expand_market_info_temporally(self._infer, available_cols)

        except Exception as e:
            print(f"Warning: Error adding market info features: {e}")
            import traceback
            traceback.print_exc()

    def _expand_market_info_temporally(self, df: pd.DataFrame, market_cols: list) -> pd.DataFrame:
        """
        Expand market info features temporally to align with Alpha360's 60-day structure.

        For each market info feature col, creates 60 columns: col_59, col_58, ..., col_0
        where col_i contains the market info value from i days ago.

        Args:
            df: DataFrame with Alpha360 features (index: datetime, instrument)
            market_cols: List of market info feature column names

        Returns:
            DataFrame with additional market info columns for each timestep
        """
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        # Build all expanded market info columns at once
        expanded_data = {}
        nan_count = 0
        for col in market_cols:
            base_series = self._market_info_df[col]
            for i in range(59, -1, -1):
                col_name = f"{col}_{i}"
                # Shift market info data by i days (shift(i) means value from i days ago)
                shifted = base_series.shift(i)
                aligned_values = shifted.reindex(main_datetimes).values

                # Fill NaN with 0 (for dates not in market info or from shifting)
                col_nan = pd.isna(aligned_values).sum()
                if col_nan > 0:
                    nan_count += col_nan
                    aligned_values = pd.Series(aligned_values).fillna(0).values

                if has_multi_columns:
                    expanded_data[('feature', col_name)] = aligned_values
                else:
                    expanded_data[col_name] = aligned_values

        if nan_count > 0:
            print(f"    Note: Filled {nan_count} NaN values in expanded market info features with 0")

        # Create DataFrame with all expanded market info columns
        expanded_df = pd.DataFrame(expanded_data, index=df.index)

        # Use pd.concat to merge all columns at once
        merged = pd.concat([df, expanded_df], axis=1, copy=False)

        # Return a copy to ensure defragmentation
        return merged.copy()

    def get_label_config(self):
        """Return N-day return label."""
        return_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [return_expr], ["LABEL0"]

    @classmethod
    def get_model_config(cls) -> dict:
        """
        Get MASTER model configuration for this handler.

        Returns:
            Dict with model configuration
        """
        return {
            "d_feat": cls.D_FEAT_PER_STEP,  # Features per timestep
            "seq_len": cls.SEQ_LEN,
            "gate_input_start_index": cls.GATE_INPUT_START_INDEX,
            "gate_input_end_index": cls.GATE_INPUT_END_INDEX,
            "n_stock_features": cls.N_STOCK_FEATURES_PER_STEP,
            "n_market_features": cls.N_MARKET_FEATURES,
            "total_features": cls.D_FEAT_TOTAL,
        }
