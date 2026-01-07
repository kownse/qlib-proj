# Qlib-Proj

Quantitative trading research project built on [Microsoft Qlib](https://github.com/microsoft/qlib) platform, focusing on US stock market analysis using machine learning models for alpha signal generation, volatility prediction, and backtesting.

## Features

- **Data Pipeline**: Automated download from Yahoo Finance with conversion to Qlib binary format
- **Technical Analysis**: Integrated TA-Lib indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
- **News Sentiment**: Finnhub news data with sentiment analysis (TextBlob/VADER)
- **Multiple Models**: LightGBM, CatBoost, Transformer, TFT (Temporal Fusion Transformer)
- **AutoML Integration**: AutoGluon TabularPredictor and TimeSeriesPredictor support
- **Flexible Stock Pools**: Test (10), Tech (30), S&P 100, S&P 500

## Project Structure

```
qlib-proj/
├── qlib/                    # Qlib library (submodule)
├── autogluon/               # AutoGluon library (submodule)
├── scripts/
│   ├── data/                # Data handlers and downloaders
│   ├── models/              # Training scripts
│   ├── news/                # News data pipeline
│   └── utils/               # Evaluation and TA-Lib operators
├── my_data/                 # Data storage (gitignored)
├── my_configs/              # Workflow YAML configs
├── my_models/               # Trained model checkpoints
├── outputs/                 # Experiment results
├── mlruns/                  # MLflow tracking
└── notebooks/               # Jupyter notebooks
```

## Installation

### Prerequisites

- Python 3.8+
- TA-Lib C library

```bash
# macOS
brew install ta-lib

# Ubuntu/Debian
sudo apt-get install libta-lib-dev

# Then install Python binding
pip install TA-Lib
```

### Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/your-repo/qlib-proj.git
cd qlib-proj

# Install Qlib
cd qlib && pip install -e . && cd ..

# Install dependencies
pip install pandas numpy yfinance python-dotenv mlflow lightgbm catboost

# Optional: Install AutoGluon
pip install autogluon
```

## Quick Start

### 1. Download Data

```bash
# Download US stock data and convert to Qlib format
python scripts/data/download_us_data.py

# Verify data
python scripts/data/test_data.py
```

### 2. Train Models

```bash
# Basic LightGBM baseline
python scripts/models/run_baseline.py

# Enhanced training with options
python scripts/models/run_lgb_enhanced.py --nday 2 --stock-pool tech --regularization medium

# Multi-day prediction with LightGBM
python scripts/models/run_lgb_nd.py

# CatBoost model
python scripts/models/run_catboost_nd.py

# AutoGluon time series
python scripts/models/run_ag_timeseries.py
```

### 3. Qlib Initialization in Code

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

## Data Handlers

Extended `DataHandlerLP` classes with custom features:

| Handler | Features |
|---------|----------|
| `Alpha158_Volatility_TALib` | Alpha158 + TA-Lib indicators |
| `Alpha158_Volatility_TALib_News` | Above + news sentiment features |

## Stock Pools

| Pool | Size | Description |
|------|------|-------------|
| `test` | 10 | Core stocks for testing |
| `tech` | ~30 | Tech sector (Mag 7, semis, software) |
| `sp100` | 100 | S&P 100 components |
| `sp500` | ~500 | S&P 500 components |

## News Pipeline

Requires Finnhub API key in `.env`:

```bash
FINNHUB_API_KEY=your_key_here
```

```bash
# Download news
python scripts/news/download_news.py

# Process into features
python scripts/news/process_news.py
```

## Training Options

`run_lgb_enhanced.py` supports:

```bash
--nday N              # N-day forward return prediction
--stock-pool POOL     # test/tech/sp100/sp500
--regularization LVL  # light/medium/strong
--use-news            # Include news sentiment features
--transfer-learning   # Pretrain on historical data
--feature-selection   # Feature importance based selection
--cross-validation    # Time series cross-validation
--cv-folds N          # Number of CV folds
```

## Evaluation Metrics

- IC (Information Coefficient) and ICIR
- MSE, MAE, RMSE, MAPE
- Visualization: time series plots, scatter plots, error analysis

## License

MIT

## References

- [Microsoft Qlib](https://github.com/microsoft/qlib)
- [AutoGluon](https://github.com/autogluon/autogluon)
- [TA-Lib](https://ta-lib.org/)
