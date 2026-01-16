# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantitative trading research project built on Microsoft's Qlib platform. Focus: US stock market analysis using ML models for alpha signal generation, volatility prediction, and backtesting.

## Key Directories

- `scripts/models/common/` - Shared config, training utils, backtest logic (start here for understanding architecture)
- `scripts/data/datahandler_*.py` - All feature engineering handlers
- `scripts/models/tree/` - Tree-based models (LightGBM, CatBoost, XGBoost)
- `scripts/models/deep/` - Deep learning models (ALSTM, TCN, Transformer, AE-MLP)
- `my_data/qlib_us/` - Qlib binary format data (gitignored)
- `my_models/` - Trained model checkpoints

## Common Commands

### Data Pipeline

```bash
# Download US stock data and convert to Qlib format
python scripts/data/download_us_data.py

# Test data loading
python scripts/data/test_data.py

# Download macro data (VIX, bonds, sectors, FRED yields/spreads)
python scripts/data/download_macro_data.py

# Process macro data into features (~109 features)
python scripts/data/process_macro_data.py

# Download news data (requires FINNHUB_API_KEY in .env)
python scripts/news/download_news.py

# Process news into features
python scripts/news/process_news.py
```

### Training Models

```bash
# Tree-based models
python scripts/models/tree/run_baseline.py                              # Basic test
python scripts/models/tree/run_lgb_nd.py --stock-pool sp500 --backtest  # LightGBM
python scripts/models/tree/run_catboost_nd.py --stock-pool sp500        # CatBoost
python scripts/models/tree/run_xgboost_nd.py --stock-pool sp500         # XGBoost

# Tree-based with hyperparameter optimization (Hyperopt + CV)
python scripts/models/tree/run_catboost_hyperopt_cv.py --stock-pool sp500

# Deep learning models
python scripts/models/deep/run_alstm.py --stock-pool sp500 --handler alpha360
python scripts/models/deep/run_tcn.py --stock-pool sp500 --handler alpha360
python scripts/models/deep/run_transformer.py --stock-pool sp500 --handler alpha360

# Deep learning with macro features (time-aligned)
python scripts/models/deep/run_alstm.py --handler alpha360-macro --d-feat 29
python scripts/models/deep/run_tcn.py --handler alpha360-macro --d-feat 29

# Deep learning with hyperparameter optimization (Optuna + CV)
python scripts/models/deep/run_tcn_optuna_cv.py --stock-pool sp500

# AutoGluon models
python scripts/models/autogluon/run_ag_ts.py
python scripts/models/autogluon/run_ag_tft.py

# Ensemble backtest
python scripts/models/ensemble/run_ensemble_backtest.py --stock-pool sp500

# Analysis
python scripts/models/analysis/param_search.py --model-path ./my_models/model.txt
```

### Qlib Initialization

```python
import qlib
from qlib.constant import REG_US
from scripts.utils.talib_ops import TALIB_OPS

qlib.init(
    provider_uri="./my_data/qlib_us",
    region=REG_US,
    custom_ops=TALIB_OPS  # Enable TA-Lib indicators
)
```

## Key Architecture

### Data Handlers

Extended `DataHandlerLP` classes in `scripts/data/`:

**Base Handlers** (`datahandler_ext.py`):
- **Alpha158_Volatility**: Alpha158 features (~158)
- **Alpha360_Volatility**: Alpha360 features (6 OHLCV × 60 days = 360)
- **Alpha158_Volatility_TALib**: Alpha158 + TA-Lib indicators (~170)

**Macro Handlers** (`datahandler_macro.py`):
- **Alpha158_Macro**: Alpha158 + macro features (~263 total)
- **Alpha158_Volatility_TALib_Macro**: Alpha158 + TA-Lib + macro (~275 total)
- **Alpha360_Macro**: Alpha360 + time-aligned macro (for ALSTM/TCN/Transformer)
  - Structure: (60 timesteps, 6+M features) where M = macro features
  - `macro_features="core"`: 23 macro → 1740 total, d_feat=29
  - `macro_features="all"`: 109 macro → 6900 total, d_feat=115

**News Handler** (`datahandler_news.py`):
- **Alpha158_Volatility_TALib_News**: Adds news sentiment features

Label config uses N-day forward returns: `Ref($close, -N)/Ref($close, -1) - 1`

### TA-Lib Integration

Custom operators in `scripts/utils/talib_ops.py` registered via `custom_ops` parameter:
- `TALIB_RSI`, `TALIB_MACD_*`, `TALIB_ATR`, `TALIB_BBANDS_*`, `TALIB_ADX`, etc.
- Installation:
  ```bash
  # macOS
  brew install ta-lib && pip install TA-Lib

  # Ubuntu/Debian
  sudo apt-get install libta-lib-dev && pip install TA-Lib
  ```

**Threading Conflict**: TA-Lib C library has memory issues with multiprocessing (fork). Training scripts set these env vars automatically:
```python
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
```
When using TA-Lib handlers, Qlib init requires: `kernels=1, joblib_backend=None`

### Model Training Features

All training scripts in `scripts/models/` share common utilities from `models/common/`:

1. **Stock pools**: test (10), tech (~30), sp100 (100), sp500 (~500 stocks)
2. **Handlers** (use `--handler` flag) - see `scripts/models/common/handlers.py` for full registry:
   - `alpha158`, `alpha360` - Base handlers
   - `alpha158-talib`, `alpha158-talib-lite` - With TA-Lib indicators
   - `alpha158-macro`, `alpha158-talib-macro` - With macro features
   - `alpha360-macro` - Alpha360 + time-aligned macro (for deep learning)
   - `alpha158-news` - With news sentiment
   - `alpha158-enhanced-v1` through `v8` - Feature-selected handlers based on importance analysis
3. **Backtest integration**: Use `--backtest` flag to run TopkDropoutStrategy
4. **Feature selection**: Use `--top-k N` for top-N feature retraining
5. **Strategies**: topk, dynamic_risk, vol_stoploss
6. **Hyperparameter optimization**: Hyperopt (tree) and Optuna (deep) with CV

### News Pipeline

1. `download_news.py`: Fetches from Finnhub API by symbol/date range
2. `process_news.py`: Extracts sentiment (TextBlob/VADER) and statistics
3. Features: `news_sentiment_mean`, `news_count`, `news_sentiment_std`, etc.

### Macro Data Pipeline

1. `download_macro_data.py`: Downloads from Yahoo Finance and FRED API
2. `process_macro_data.py`: Engineers ~109 macro features

**Data Sources:**
- **Yahoo Finance**: VIX, GLD, TLT, UUP, USO, sector ETFs (XLK, XLF, etc.), HYG, SPY, QQQ
- **FRED**: Treasury yields (DGS2, DGS10, DGS30), yield spreads (T10Y2Y), credit spreads

**Feature Categories (~109 total):**
- VIX features: level, zscore, regime, term structure (13)
- Asset class momentum: gold, bonds, dollar, oil (21)
- Sector ETFs: 11 sectors × 3 features (33)
- Credit/risk indicators: HYG spreads, stress (8)
- Treasury yields: levels, curve slope, inversion (10)
- Cross-asset: risk-on/off, correlations (5)

**Feature Sets:**
- `macro_features="all"`: All 109 features
- `macro_features="core"`: 23 key features (recommended)
- `macro_features="vix_only"`: 13 VIX-related features

## Stock Pools

Defined in `scripts/data/stock_pools.py`:

- **TEST_SYMBOLS**: 10 test stocks
- **TECH_SYMBOLS**: ~30 tech stocks (Mag 7, semis, software, internet)
- **SP100_SYMBOLS**: S&P 100 components
- **SP500_SYMBOLS**: S&P 500 components

## Time Periods

Default splits (see `scripts/models/common/config.py`):
```python
DEFAULT_TIME_SPLITS = {
    'train_start': "2000-01-01", 'train_end': "2022-12-31",
    'valid_start': "2023-01-01", 'valid_end': "2023-12-31",
    'test_start': "2024-01-01", 'test_end': "2025-12-31",
}

MAX_TRAIN_TIME_SPLITS = {  # Use --max-train flag for deployment
    'train_start': "2000-01-01", 'train_end': "2025-09-30",
    'valid_start': "2025-10-01", 'valid_end': "2025-12-31",
    'test_start': "2025-10-01", 'test_end': "2025-12-31",
}
```

## Evaluation Metrics

`scripts/utils/utils.py` provides:
- IC (Information Coefficient) and ICIR
- MSE, MAE, RMSE, MAPE
- Visualization: time series plots, scatter plots, error analysis

## AutoGluon Integration

AutoGluon provides AutoML for tabular and time series prediction. See `autogluon/CLAUDE.md` for details.

```bash
# Install AutoGluon
pip install autogluon

# Or install from local source
cd autogluon && ./full_install.sh
```

**TabularPredictor** for cross-sectional stock ranking:
```python
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label="future_return").fit(train_df, presets="best_quality")
```

**TimeSeriesPredictor** for volatility/return forecasting:
```python
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
predictor = TimeSeriesPredictor(prediction_length=5, target="return").fit(ts_data)
```

## Qlib Development

```bash
# Install Qlib in editable mode
cd qlib-src && pip install -e .[dev]

# If modifying Cython extensions
make prerequisite
```

## Environment Variables

`.env` file (gitignored):
```
FINNHUB_API_KEY=your_key_here
FRED_API_KEY=your_fred_key_here  # For FRED macro data (treasury yields, spreads)
```

## Adding New Handlers

1. Create `datahandler_*.py` in `scripts/data/` extending `DataHandlerLP`
2. Register in `scripts/models/common/handlers.py` HANDLER_REGISTRY
3. Use `get_feature_config()` to return `(fields, names)` tuples with Qlib expression syntax

Example feature expression: `"($close - Mean($close, 60)) / (Std($close, 60) + 1e-12)"`

## Feature Importance Workflow

Enhanced handlers V4-V8 use permutation importance analysis:

```bash
# Train base model
python scripts/models/deep/run_ae_mlp.py --handler alpha158-enhanced-v7

# Analyze feature importance
python scripts/models/analysis/permutation_importance_ae_mlp.py --model-path ./my_models/model.pt

# Create new handler keeping only positive-importance features
# See datahandler_enhanced_v8.py for example
```

## Important Notes

- **Region**: Always use `REG_US`, not `REG_CN` (most Qlib examples use China data)
- **Custom ops**: Must pass `custom_ops=TALIB_OPS` to `qlib.init()` for TA-Lib features
- **Date format**: Yahoo Finance data uses YYYY-MM-DD, timezone stripped
- **Instruments format**: Tab-separated `SYMBOL\tSTART_DATE\tEND_DATE` in instruments files
- **Macro lag**: Macro features should be lagged by 1 day to avoid look-ahead bias
