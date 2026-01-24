# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantitative trading research project built on Microsoft's Qlib platform. Focus: US stock market analysis using ML models for alpha signal generation, volatility prediction, and backtesting.

**Key Technologies**: Python, PyTorch, LightGBM, CatBoost, XGBoost, TA-Lib, Qlib, AutoGluon, Optuna, Hyperopt

## Directory Structure

```
qlib-proj/
├── scripts/                    # Main application code
│   ├── data/                   # Data handlers & pipelines (30 files)
│   ├── models/                 # Model training & inference
│   │   ├── common/             # Shared utilities (handlers.py, training.py, backtest.py)
│   │   ├── tree/               # Tree-based models (LightGBM, CatBoost, XGBoost)
│   │   ├── deep/               # Deep learning (ALSTM, TCN, Transformer, AE-MLP, MASTER)
│   │   ├── ensemble/           # Ensemble methods
│   │   ├── autogluon/          # AutoML integration
│   │   ├── analysis/           # Model analysis & diagnostics (22 files)
│   │   ├── feature_engineering/# Feature selection workflows
│   │   └── kaggle/             # Competition code
│   ├── utils/                  # Utilities (TA-Lib ops, metrics, strategies)
│   ├── news/                   # News data pipeline
│   └── analysis/               # High-level analysis scripts
├── MASTER/                     # MASTER model research implementation
├── qlib-src/                   # Qlib source (git submodule)
├── autogluon/                  # AutoGluon source (git submodule)
├── my_data/qlib_us/            # Qlib binary format data (gitignored)
└── my_models/                  # Trained model checkpoints (gitignored)
```

## Common Commands

### Data Pipeline

```bash
# Download US stock data and convert to Qlib format
python scripts/data/download_us_data.py

# Download with date range control
python scripts/data/download_us_data_to_date.py --end-date 2025-01-15

# Test data loading
python scripts/data/test_data.py

# Download macro data (VIX, bonds, sectors, FRED yields/spreads)
python scripts/data/download_macro_data.py
python scripts/data/download_macro_data_to_date.py --end-date 2025-01-15

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
python scripts/models/deep/run_transformer_v2.py --stock-pool sp500 --handler alpha360-macro
python scripts/models/deep/run_ae_mlp.py --stock-pool sp500 --handler alpha158
python scripts/models/deep/run_tabnet.py --stock-pool sp500
python scripts/models/deep/run_saint.py --stock-pool sp500

# MASTER framework
python scripts/models/deep/run_master.py --stock-pool sp500
python scripts/models/deep/run_master_official.py --stock-pool sp500

# Deep learning with macro features (time-aligned)
python scripts/models/deep/run_alstm.py --handler alpha360-macro --d-feat 29
python scripts/models/deep/run_tcn.py --handler alpha360-macro --d-feat 29

# Deep learning with hyperparameter optimization (Optuna/Hyperopt + CV)
python scripts/models/deep/run_tcn_optuna_cv.py --stock-pool sp500
python scripts/models/deep/run_ae_mlp_optuna_cv.py --stock-pool sp500
python scripts/models/deep/run_ae_mlp_hyperopt_cv.py --stock-pool sp500

# AutoGluon models
python scripts/models/autogluon/run_ag_ts.py
python scripts/models/autogluon/run_ag_tft.py

# Ensemble backtest
python scripts/models/ensemble/run_ensemble_backtest.py --stock-pool sp500
python scripts/models/ensemble/run_deep_ensemble.py --stock-pool sp500
python scripts/models/ensemble/run_triple_ensemble.py --stock-pool sp500

# Analysis
python scripts/models/analysis/param_search.py --model-path ./my_models/model.txt
python scripts/models/analysis/permutation_importance_ae_mlp.py --model-path ./my_models/model.pt
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
- **Alpha180_Volatility**: Alpha180 features (~180)
- **Alpha300_Volatility**: Alpha300 features (~300)
- **Alpha360_Volatility**: Alpha360 features (6 OHLCV × 60 days = 360)
- **Alpha158_Volatility_TALib**: Alpha158 + TA-Lib indicators (~300+)
- **Alpha158_Volatility_TALib_Lite**: Alpha158 + subset of TA-Lib (~20 indicators)

**Pandas Handlers** (`datahandler_pandas.py`) - No TA-Lib C library dependency:
- **Alpha158_Pandas**: Pure pandas implementation
- **Alpha360_Pandas**: Pure pandas implementation

**Macro Handlers** (`datahandler_macro.py`):
- **Alpha158_Macro**: Alpha158 + macro features (~263 total)
- **Alpha158_Volatility_TALib_Macro**: Alpha158 + TA-Lib + macro (~275 total)
- **Alpha360_Macro**: Alpha360 + time-aligned macro (for ALSTM/TCN/Transformer)
  - Structure: (60 timesteps, 6+M features) where M = macro features
  - `macro_features="core"`: 23 macro → 1740 total, d_feat=29
  - `macro_features="all"`: 109 macro → 6900 total, d_feat=115

**News Handler** (`datahandler_news.py`):
- **Alpha158_Volatility_TALib_News**: Adds news sentiment features

**MASTER Handler** (`datahandler_master.py`):
- **Alpha158_MASTER**: Integrated with MASTER framework

**Enhanced Handlers** (`datahandler_enhanced.py` through `datahandler_enhanced_v11.py`):
- **Alpha158_Enhanced_V1** through **V11**: Progressive feature selection based on importance analysis
- Each version retains only features with positive permutation importance

Label config uses N-day forward returns: `Ref($close, -N)/Ref($close, -1) - 1`

### Handler Registry

See `scripts/models/common/handlers.py` for full registry (~30+ handlers):

| Handler Key | Description | Features |
|-------------|-------------|----------|
| `alpha158` | Base Alpha158 | ~158 |
| `alpha180` | Extended Alpha180 | ~180 |
| `alpha300` | Extended Alpha300 | ~300 |
| `alpha360` | 60-day OHLCV rolling | 360 |
| `alpha158-talib` | Alpha158 + TA-Lib | ~300+ |
| `alpha158-talib-lite` | Alpha158 + TA-Lib subset | ~178 |
| `alpha158-pandas` | Pure pandas (no C lib) | ~158 |
| `alpha360-pandas` | Pure pandas rolling | 360 |
| `alpha158-macro` | Alpha158 + macro | ~263 |
| `alpha158-talib-macro` | Full feature set | ~380 |
| `alpha360-macro` | For deep learning | 29-115 d_feat |
| `alpha158-news` | With news sentiment | ~170 |
| `alpha158-enhanced-v1` to `v11` | Feature-selected | Varies |

### TA-Lib Integration

Custom operators in `scripts/utils/talib_ops.py` registered via `custom_ops` parameter:
- `TALIB_RSI`, `TALIB_MACD_*`, `TALIB_ATR`, `TALIB_BBANDS_*`, `TALIB_ADX`, etc.
- 50+ technical indicators available
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

### Model Types

#### Tree-Based Models (`scripts/models/tree/`)
- **LightGBM**: `run_lgb_nd.py` - Fast gradient boosting
- **CatBoost**: `run_catboost_nd.py`, `run_catboost_hyperopt_cv.py` - Handles categoricals
- **XGBoost**: `run_xgboost_nd.py` - Extreme gradient boosting

#### Deep Learning Models (`scripts/models/deep/`)
- **ALSTM**: `run_alstm.py` - Attention LSTM for sequence modeling
- **TCN**: `run_tcn.py` - Temporal Convolutional Network
- **Transformer**: `run_transformer.py`, `run_transformer_v2.py` - Self-attention models
- **AE-MLP**: `run_ae_mlp.py` - AutoEncoder + MLP architecture
- **TabNet**: `run_tabnet.py` - Attention-based tabular learning
- **SAINT**: `run_saint.py` - Self-Attention and Intersample Attention
- **MASTER**: `run_master.py`, `run_master_official.py` - Market-guided framework
- **TFT**: `run_tft_volatility.py` - Temporal Fusion Transformer

#### Ensemble Methods (`scripts/models/ensemble/`)
- `run_ensemble_backtest.py` - Combine multiple model predictions
- `run_deep_ensemble.py` - Deep learning ensemble
- `run_triple_ensemble.py` - Three-model strategy

### Model Training Features

All training scripts share common utilities from `scripts/models/common/`:

1. **Stock pools**: test (10), tech (~30), sp100 (100), sp500 (~500 stocks)
2. **Handlers**: Use `--handler` flag (see Handler Registry above)
3. **Backtest integration**: Use `--backtest` flag to run TopkDropoutStrategy
4. **Feature selection**: Use `--top-k N` for top-N feature retraining
5. **Strategies**: topk, dynamic_risk, vol_stoploss
6. **Hyperparameter optimization**:
   - Hyperopt (tree models): `run_catboost_hyperopt_cv.py`
   - Optuna (deep learning): `run_tcn_optuna_cv.py`, `run_ae_mlp_optuna_cv.py`
7. **Cross-validation**: Time-series aware CV in `cv_utils.py`

### Feature Engineering Workflow

Located in `scripts/models/feature_engineering/`:

```bash
# Nested CV with backward elimination
python scripts/models/feature_engineering/nested_cv_backward_elimination.py

# Feature selection for specific models
python scripts/models/feature_engineering/nested_cv_feature_selection_tcn.py
python scripts/models/feature_engineering/nested_cv_feature_selection_transformer.py
```

**Workflow:**
1. Train base model with full feature set
2. Run permutation importance analysis
3. Create new handler keeping only positive-importance features
4. Iterate to create enhanced versions (v1 → v11)

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

### MASTER Framework

Research implementation in `/MASTER/` directory:

```bash
# Train MASTER model
python scripts/models/deep/run_master.py --stock-pool sp500

# Official MASTER implementation
python scripts/models/deep/run_master_official.py --stock-pool sp500
```

Features market-guided stock representation learning with:
- Intra-stock aggregation
- Inter-stock aggregation
- Market information integration

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

## Analysis Scripts

Located in `scripts/models/analysis/` (22 files):

**Feature Analysis:**
```bash
python scripts/models/analysis/permutation_importance_ae_mlp.py --model-path ./my_models/model.pt
python scripts/models/analysis/check_cross_sectional_ic.py
python scripts/models/analysis/print_catboost_feature_importance.py
```

**Data Quality:**
```bash
python scripts/models/analysis/check_price_predictability.py
python scripts/models/analysis/check_label_autocorr.py
python scripts/models/analysis/check_smoothed_predictability.py
```

**Diagnostics:**
```bash
python scripts/models/analysis/debug_sp500_talib.py
python scripts/models/analysis/check_talib_lite_anomalies.py
python scripts/models/analysis/estimate_strategy_return.py
```

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

Enhanced handlers V1-V11 use permutation importance analysis:

```bash
# Train base model
python scripts/models/deep/run_ae_mlp.py --handler alpha158-enhanced-v7

# Analyze feature importance
python scripts/models/analysis/permutation_importance_ae_mlp.py --model-path ./my_models/model.pt

# Create new handler keeping only positive-importance features
# See datahandler_enhanced_v8.py for example
```

## Architecture Overview

**Data Flow:**
```
Yahoo Finance / Finnhub / FRED
    ↓
download_*.py → Parquet/CSV
    ↓
process_*.py → Feature engineering (~109 macro + 158-360 alpha)
    ↓
datahandler_*.py → DataHandlerLP (Qlib integration)
    ↓
run_*.py (Tree/Deep/Ensemble models)
    ↓
Predictions → run_ensemble_backtest.py
    ↓
Backtest results (IC, ICIR, PnL, etc.)
```

**Handler Hierarchy:**
```
Base: Alpha158_Volatility (158 features)
  ├── Extended: Alpha360_Volatility (360 features - 60 days OHLCV)
  ├── Technical: Alpha158_Volatility_TALib (~300+ indicators)
  │     └── Lite: Alpha158_Volatility_TALib_Lite (~20 indicators)
  ├── Macro: Adds 23-109 macro features
  ├── News: Adds sentiment features
  └── Enhanced v1-v11: Progressive feature selection
```

## Important Notes

- **Region**: Always use `REG_US`, not `REG_CN` (most Qlib examples use China data)
- **Custom ops**: Must pass `custom_ops=TALIB_OPS` to `qlib.init()` for TA-Lib features
- **Date format**: Yahoo Finance data uses YYYY-MM-DD, timezone stripped
- **Instruments format**: Tab-separated `SYMBOL\tSTART_DATE\tEND_DATE` in instruments files
- **Macro lag**: Macro features should be lagged by 1 day to avoid look-ahead bias
- **Threading**: TA-Lib handlers require single-threaded execution (`kernels=1`)
- **GPU**: Deep learning models auto-detect CUDA availability

## Common CLI Arguments

Most training scripts support these arguments:

| Argument | Description | Example |
|----------|-------------|---------|
| `--stock-pool` | Stock universe | `sp500`, `sp100`, `tech`, `test` |
| `--handler` | Feature handler | `alpha158`, `alpha360-macro` |
| `--backtest` | Run backtest after training | Flag |
| `--max-train` | Use MAX_TRAIN_TIME_SPLITS | Flag |
| `--top-k` | Top-K feature selection | `--top-k 50` |
| `--d-feat` | Feature dimension (deep) | `--d-feat 29` |
| `--n-days` | Forward return days | `--n-days 5` |
| `--strategy` | Backtest strategy | `topk`, `dynamic_risk` |

## Troubleshooting

**TA-Lib import errors:**
```bash
# Ensure C library is installed first
brew install ta-lib  # macOS
sudo apt-get install libta-lib-dev  # Ubuntu
pip install TA-Lib
```

**Memory issues with large stock pools:**
- Use `--stock-pool sp100` instead of `sp500` for testing
- Enable `kernels=1` in qlib.init() for TA-Lib handlers

**Qlib data not found:**
```bash
# Download and process data first
python scripts/data/download_us_data.py
```

**FRED API errors:**
- Ensure `FRED_API_KEY` is set in `.env` file
- Get free API key from https://fred.stlouisfed.org/docs/api/
