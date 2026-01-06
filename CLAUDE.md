# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantitative trading research project built on Microsoft's Qlib platform. Focus: US stock market analysis using ML models for alpha signal generation, volatility prediction, and backtesting.

## Project Structure

```
qlib-proj/
├── qlib/                    # Qlib library source (submodule)
├── autogluon/               # AutoGluon library (submodule, see autogluon/CLAUDE.md)
├── scripts/
│   ├── data/                # Data download and handlers
│   │   ├── download_us_data.py   # Yahoo Finance data downloader
│   │   ├── datahandler_ext.py    # Extended handlers with TA-Lib
│   │   └── datahandler_news.py   # Handlers with news features
│   ├── models/              # Training scripts
│   │   ├── run_baseline.py       # Basic LightGBM training
│   │   └── run_lgb_enhanced.py   # Enhanced training with regularization, CV, transfer learning
│   ├── news/                # News data acquisition
│   │   ├── download_news.py      # Finnhub news downloader
│   │   └── process_news.py       # Sentiment analysis pipeline
│   └── utils/
│       ├── utils.py              # Evaluation and plotting utilities
│       └── talib_ops.py          # TA-Lib custom operators for Qlib
├── my_data/                 # Data storage (gitignored)
│   ├── csv_us/              # Raw Yahoo Finance CSV data
│   ├── qlib_us/             # Qlib binary format data
│   ├── news_csv/            # Raw news data
│   └── news_processed/      # Processed news features
├── my_configs/              # Workflow YAML configs
├── my_models/               # Trained model checkpoints
├── outputs/                 # Experiment results and figures
└── mlruns/                  # MLflow tracking
```

## Common Commands

### Data Pipeline

```bash
# Download US stock data and convert to Qlib format
python scripts/data/download_us_data.py

# Test data loading
python scripts/data/test_data.py

# Download news data (requires FINNHUB_API_KEY in .env)
python scripts/news/download_news.py

# Process news into features
python scripts/news/process_news.py
```

### Training Models

```bash
# Basic LightGBM baseline
python scripts/models/run_baseline.py

# Enhanced training with options
python scripts/models/run_lgb_enhanced.py --help

# Examples:
python scripts/models/run_lgb_enhanced.py --nday 2 --stock-pool medium --regularization medium
python scripts/models/run_lgb_enhanced.py --use-news --transfer-learning --feature-selection
python scripts/models/run_lgb_enhanced.py --cross-validation --cv-folds 3
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

- **Alpha158_Volatility_TALib**: Alpha158 features + TA-Lib indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
- **Alpha158_Volatility_TALib_News**: Adds news sentiment features

Label config uses N-day forward returns: `Ref($close, -N)/Ref($close, -1) - 1`

### TA-Lib Integration

Custom operators in `scripts/utils/talib_ops.py` registered via `custom_ops` parameter:
- `TALIB_RSI`, `TALIB_MACD_*`, `TALIB_ATR`, `TALIB_BBANDS_*`, `TALIB_ADX`, etc.
- Requires TA-Lib C library: `brew install ta-lib && pip install TA-Lib`

### Enhanced Training Features (`run_lgb_enhanced.py`)

1. **Regularization levels**: light/medium/strong presets for different data sizes
2. **Stock pools**: small (10), medium (30), large (100 S&P stocks)
3. **Feature selection**: Based on LightGBM feature importance
4. **Time series CV**: Walk-forward validation with gap days
5. **Transfer learning**: Pretrain on historical data, finetune with news

### News Pipeline

1. `download_news.py`: Fetches from Finnhub API by symbol/date range
2. `process_news.py`: Extracts sentiment (TextBlob/VADER) and statistics
3. Features: `news_sentiment_mean`, `news_count`, `news_sentiment_std`, etc.

## Stock Pools

Defined in `scripts/data/download_us_data.py` and `scripts/models/run_lgb_enhanced.py`:

- **TECH_SYMBOLS**: ~30 tech stocks (Mag 7, semis, software, internet)
- **SP100_SYMBOLS**: S&P 100 components

## Time Periods

Default splits in training scripts:
- Pretrain: 2015-2024 (no news, for transfer learning)
- Train: 2025-01-01 to 2025-09-30
- Valid: 2025-10-01 to 2025-11-30
- Test: 2025-12-01 to 2025-12-31

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
cd qlib && pip install -e .[dev]

# If modifying Cython extensions
make prerequisite
```

## Environment Variables

`.env` file (gitignored):
```
FINNHUB_API_KEY=your_key_here
```

## Important Notes

- **Region**: Always use `REG_US`, not `REG_CN` (most Qlib examples use China data)
- **Custom ops**: Must pass `custom_ops=TALIB_OPS` to `qlib.init()` for TA-Lib features
- **Date format**: Yahoo Finance data uses YYYY-MM-DD, timezone stripped
- **Instruments format**: Tab-separated `SYMBOL\tSTART_DATE\tEND_DATE` in instruments files
