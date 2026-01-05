# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantitative trading research project built on Microsoft's Qlib platform. The project focuses on US stock market analysis, particularly technology stocks, using machine learning models for alpha signal generation and backtesting.

## Project Structure

```
qlib-proj/
├── qlib/              # Qlib library source code (see qlib/CLAUDE.md for details)
├── scripts/           # Custom data download and processing scripts
├── my_data/           # Market data storage
│   ├── csv_us/        # Raw CSV data from Yahoo Finance
│   └── qlib_us/       # Processed Qlib binary format data
├── my_configs/        # Custom workflow configuration files (YAML)
├── my_models/         # Trained model checkpoints and artifacts
├── notebooks/         # Jupyter notebooks for analysis and experiments
├── outputs/           # Experiment outputs and results
└── logs/              # Application logs
```

## Common Commands

### Data Download and Preparation

```bash
# Download US stock data from Yahoo Finance and convert to Qlib format
python scripts/download_us_data.py

# Note: The script downloads tech stocks (AAPL, MSFT, NVDA, etc.) by default
# Edit TECH_SYMBOLS or SP100_SYMBOLS in the script to customize stock selection
```

### Running Experiments

```bash
# Initialize Qlib with US data (in Python scripts/notebooks)
import qlib
from qlib.constant import REG_US

qlib.init(
    provider_uri="./my_data/qlib_us",
    region=REG_US
)

# Run a workflow from config file
cd qlib/examples
qrun ../../my_configs/your_config.yaml

# Run workflow in debug mode
python -m pdb qlib/cli/run.py ../my_configs/your_config.yaml
```

### Working with Notebooks

```bash
# Start Jupyter notebook server
jupyter notebook notebooks/

# Install analysis dependencies if needed
pip install -e qlib/.[analysis]
```

## Key Implementation Details

### Data Pipeline

**CSV to Qlib Binary Conversion**: The project uses a custom script `scripts/download_us_data.py` to:
1. Download daily OHLCV data from Yahoo Finance using yfinance
2. Convert to Qlib's binary format for efficient access
3. Create instrument files defining stock universes

**Stock Universe**: Currently focused on US tech stocks including:
- Magnificent 7 (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA)
- Semiconductor stocks (AMD, INTC, AVGO, QCOM, MU, AMAT)
- Software/Cloud (CRM, ORCL, ADBE, NOW, SNOW, PLTR)
- Internet/Consumer tech (NFLX, UBER, ABNB, SHOP, PYPL, SPOT)

### Data Paths

Always use these paths when initializing Qlib:

```python
# US stock data (tech-focused)
provider_uri = "./my_data/qlib_us"  # or absolute path

# Region setting
from qlib.constant import REG_US
region = REG_US
```

### Workflow Configuration

Config files should be placed in `my_configs/` directory. Standard Qlib workflow config structure:

```yaml
qlib_init:
    provider_uri: "./my_data/qlib_us"
    region: us

market: sp100  # or 'all' for all stocks

data_handler_config:
    # Define features, labels, and data processing

model:
    # Model class and hyperparameters

strategy:
    # Trading strategy configuration

backtest:
    # Backtesting parameters
```

### Model Storage

- Save trained models to `my_models/` directory
- Use descriptive names with timestamps (e.g., `lightgbm_tech_20260105.pkl`)
- Track experiments using MLflow (default location: `qlib/mlruns/`)

### Outputs and Logs

- **outputs/**: Backtest results, analysis reports, figures
- **logs/**: Application logs, debugging information

## Known Issues and Workarounds

### DumpDataAll API Change

The `scripts/download_us_data.py` script currently fails at the CSV-to-binary conversion step due to API changes in Qlib's `DumpDataAll` class. The correct usage is:

```python
from qlib.scripts.dump_bin import DumpDataUpdate

dumper = DumpDataUpdate(
    csv_path=str(csv_dir),
    qlib_dir=str(qlib_dir),
    freq="day",
    date_field_name="date",
    file_suffix=".csv",
    include_fields="open,high,low,close,adj_close,volume",
)
dumper.dump()
```

### Yahoo Finance Data Quality

Yahoo Finance data may have:
- Missing data for certain dates (holidays, delistings)
- Stock splits not always handled correctly
- Timezone issues (handled in the script by removing timezone info)

For production use, consider professional data providers.

### Qlib Library Development Mode

The `qlib/` subdirectory contains the Qlib source code. If you modify Qlib internals:

```bash
cd qlib
pip install -e .[dev]  # Install in editable mode

# Rebuild Cython extensions if you modify data operations
make prerequisite
```

## Development Workflow

### Adding New Stocks

1. Edit `scripts/download_us_data.py`, modify `TECH_SYMBOLS` or `SP100_SYMBOLS`
2. Run `python scripts/download_us_data.py` to download new data
3. Update instrument files in `my_data/qlib_us/instruments/`

### Creating New Experiments

1. Create config file in `my_configs/` based on examples in `qlib/examples/benchmarks/`
2. Customize:
   - Market/instrument universe
   - Features (Alpha158, Alpha360, or custom)
   - Model (LightGBM, LSTM, Transformer, etc.)
   - Strategy parameters
3. Run with `qrun my_configs/your_config.yaml`
4. Analyze results in Jupyter notebooks

### Custom Feature Engineering

Define custom features in your dataset handler:

```python
from qlib.contrib.data.handler import DataHandlerLP

class MyDataHandler(DataHandlerLP):
    def setup_data(self):
        # Define your features and labels
        self.fields = [
            "$close / Ref($close, 1) - 1",  # Daily return
            "Mean($close, 5) / $close - 1",  # MA5 deviation
            # ... custom features
        ]
```

### Backtesting Custom Strategies

1. Create strategy class inheriting from `qlib.strategy.base.BaseStrategy`
2. Implement `generate_trade_decision()` method
3. Add to workflow config or use programmatically
4. Results will be in `outputs/` directory

## Testing

Since this is a research project, testing primarily involves:

```bash
# Test Qlib installation and data access
cd qlib
pytest tests/test_all_pipeline.py -v

# Test data health
python qlib/scripts/check_data_health.py check_data --qlib_dir ./my_data/qlib_us

# Validate workflow config before long runs
python qlib/cli/run.py my_configs/your_config.yaml --dry-run  # (if supported)
```

## Performance Optimization

- **Enable caching**: Qlib uses multi-layer caching. Ensure `my_data/qlib_us/.cache/` is writable
- **Parallel processing**: Use `n_jobs` parameter in model trainers
- **GPU acceleration**: For PyTorch models (LSTM, Transformer, etc.), ensure CUDA is available

## Important Notes

- **Region Settings**: This project uses US data (`REG_US`), not China data (`REG_CN`) used in most Qlib examples
- **Date Format**: Yahoo Finance returns data in YYYY-MM-DD format with timezone info removed
- **Market Calendar**: US market calendar is used for trading day calculations
- **Instruments Format**: Files in `my_data/qlib_us/instruments/` use format: `SYMBOL\tSTART_DATE\tEND_DATE`

## Reference

- Full Qlib documentation: See `qlib/CLAUDE.md` and https://qlib.readthedocs.io
- Example workflows: `qlib/examples/benchmarks/`
- Tutorial notebooks: `qlib/examples/tutorial/`
