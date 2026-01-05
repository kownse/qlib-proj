# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Qlib is an AI-oriented quantitative investment platform developed by Microsoft. It provides a complete ML pipeline for quantitative trading, from data processing and model training to backtesting and portfolio optimization.

## Development Setup

### Installation from source

```bash
# Install dependencies first
pip install numpy
pip install --upgrade cython

# Install in editable mode for development
pip install -e .[dev]
```

### Build Cython extensions

The project uses Cython for performance-critical operations (rolling/expanding window calculations):

```bash
# Cython extensions are built automatically during installation
# To rebuild manually:
make prerequisite
```

### Pre-commit hooks

```bash
pip install -e .[dev]
pre-commit install
```

## Common Commands

### Running tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_all_pipeline.py

# Run tests excluding slow tests
pytest -m "not slow"

# Run tests with coverage
pytest --cov=qlib tests/
```

### Linting and formatting

```bash
# Format code with black (120 char line length)
make black

# Run all linters (black, pylint, flake8, mypy)
make lint

# Individual linters
make pylint
make flake8
make mypy

# Check notebooks
make nbqa
```

### Building documentation

```bash
# Install doc dependencies
make docs

# Build documentation
make docs-gen
# Output in public/ directory (or $READTHEDOCS_OUTPUT/html on RTD)
```

### Data preparation

```bash
# Download data (from community source - official source temporarily disabled)
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1

# Or using the official method (when available):
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

# Check data health
python scripts/check_data_health.py check_data --qlib_dir ~/.qlib/qlib_data/cn_data
```

### Running experiments

```bash
# Run a workflow from config file
cd examples
qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml

# Run workflow in debug mode
python -m pdb qlib/cli/run.py examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml

# Run workflow by code
python examples/workflow_by_code.py
```

### Running single models

```bash
# Run specific model benchmark
cd examples
python run_all_model.py run --models=lightgbm

# Run multiple models for N iterations
python run_all_model.py run 10
```

## Architecture Overview

### Core Modules

- **`qlib/data/`** - Data infrastructure (providers, cache, storage, operations)
  - Supports both local file-based and client-server modes
  - Multi-layer caching (memory, disk, redis)
  - Compiled Cython operations for performance
  - Provider pattern for calendar, instruments, features, expressions, datasets

- **`qlib/model/`** - Model framework (base classes, trainers, ensembles, meta-learning)
  - Base classes: `BaseModel`, `Model`, `ModelFT`
  - `Trainer` classes orchestrate model training with experiment tracking

- **`qlib/backtest/`** - Backtesting engine (exchange, executor, account, position)
  - `Exchange` simulates market with transaction costs and limits
  - `Executor` controls order execution timing and generates metrics
  - `Account` tracks positions and portfolio value

- **`qlib/strategy/`** - Trading strategy framework
  - `BaseStrategy` receives signals and generates orders
  - Supports nested strategies for hierarchical optimization

- **`qlib/workflow/`** - Experiment management (MLflow integration, task management)
  - `Recorder` logs experiments to MLflow
  - `TaskManager` handles task queuing and execution
  - Online serving support for production deployment

- **`qlib/rl/`** - Reinforcement learning framework
  - `Simulator` bridges backtest engine and RL agents
  - `StateInterpreter`/`ActionInterpreter` define observation/action spaces
  - Order execution optimization using RL

- **`qlib/contrib/`** - Community models and tools
  - PyTorch models: LSTM, GRU, Transformer, HIST, ADARNN, etc.
  - Traditional ML: LightGBM, XGBoost, CatBoost
  - Meta-learning and concept drift handling (DDG-DA)

### Key Design Patterns

**Provider Pattern**: All data access goes through providers (Calendar, Instrument, Feature, Expression, Dataset). This enables swapping between local/remote data sources.

**Factory Pattern**: `init_instance_by_config(config_dict)` creates objects from configuration dictionaries. This is used throughout for dynamic instantiation.

**Configuration-Driven**: Entire workflows can be defined in YAML configs and run with `qrun config.yaml`. All classes support dict-based configuration.

**Layered Architecture**: Clear separation between layers:
```
Data Layer → Model Layer → Strategy Layer → Backtest Layer → Analysis Layer
```

**Context Manager Pattern**: Experiments use context managers:
```python
with R.start(experiment_name="exp1"):
    R.log_params(...)
    R.save_objects(model=model)
```

### Important Implementation Details

**Cython Extensions**: Critical performance paths (`qlib/data/_libs/rolling.pyx`, `expanding.pyx`) are implemented in Cython. Changes require rebuilding with `make prerequisite`.

**Expression Evaluation**: Features are defined as expressions (e.g., `"$close/Ref($close, 1)"`) and evaluated by the expression engine in `qlib/data/ops.py`.

**Point-in-Time (PIT) Database**: Supports historical data that changes over time (e.g., financial statements). Ensures no look-ahead bias.

**Multi-level Execution**: `NestedExecutor` allows optimizing strategies at different granularities (e.g., portfolio management + order execution) simultaneously.

**Cache Hierarchy**: Three cache levels - MemCache (LRU), DiskCache (expressions/datasets), RedisCache (distributed). Configure via `qlib.init()`.

## Code Standards

### Docstring style

Use Numpydoc format:
```python
def function(param1, param2):
    """
    Short description.

    Parameters
    ----------
    param1 : type
        Description
    param2 : type
        Description

    Returns
    -------
    type
        Description
    """
```

### Line length and formatting

- Maximum line length: 120 characters
- Use `black . -l 120` for auto-formatting
- Pre-commit hooks enforce black and flake8

### Ignoring linter warnings

When necessary, disable specific warnings inline:
```python
return -ICLoss()(pred, target, index)  # pylint: disable=E1130
```

### Imports

- Use absolute imports from `qlib` package
- Avoid circular imports (common between data/model layers)
- Group imports: stdlib, third-party, qlib

## Testing Practices

### Test organization

```
tests/
├── backtest/           # Backtest module tests
├── data_mid_layer_tests/
├── dataset_tests/
├── rl/                 # RL framework tests
├── ops/                # Data operations tests
└── test_*.py          # Integration tests
```

### Marking slow tests

```python
import pytest

@pytest.mark.slow
def test_long_running():
    ...
```

### Data requirements

Many tests require downloaded data in `~/.qlib/qlib_data/cn_data`. Use `GetData().qlib_data()` in test setup or download manually.

## Common Workflows

### Adding a new model

1. Create model class inheriting from `qlib.model.base.Model`
2. Implement `fit()` and `predict()` methods
3. Add config file in `examples/benchmarks/YourModel/`
4. Add model to `qlib/contrib/model/` if contributing
5. Update `examples/benchmarks/README.md` with performance metrics

### Adding a new dataset handler

1. Inherit from `qlib.data.dataset.DataHandlerLP`
2. Override `setup_data()` to define features/labels
3. Define processors in `setup_processors()` if needed
4. Register in `qlib/contrib/data/handler.py`

### Adding a new strategy

1. Inherit from `qlib.strategy.base.BaseStrategy`
2. Implement `generate_trade_decision()` method
3. Handle position tracking in `post_trade_decision()`
4. Add configuration example to `examples/`

### Working with RL

1. Define state space via `StateInterpreter`
2. Define action space via `ActionInterpreter`
3. Create `Simulator` with backtest environment
4. Use standard RL library (Tianshou) for training
5. See `examples/rl_order_execution/` for reference

## Important Configuration

### Initialization

```python
import qlib
from qlib.constant import REG_CN

# Local mode
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

# Client mode (connects to qlib-server)
qlib.init(provider_uri="client", region=REG_CN)
```

### MLflow configuration

Experiments are tracked via MLflow. Default location: `mlruns/` directory.

Configure via:
```python
qlib.init(
    provider_uri="~/.qlib/qlib_data/cn_data",
    region=REG_CN,
    exp_manager={
        "class": "MLflowExpManager",
        "module_path": "qlib.workflow.expm",
        "kwargs": {"uri": "mlruns"}
    }
)
```

## Known Issues and Gotchas

### Pandas compatibility

Qlib sets `group_keys=False` in groupby operations for pandas 2.0+ compatibility. Some older scripts may not work correctly:
- `examples/rl_order_execution/scripts/gen_training_orders.py`
- `examples/benchmarks/TRA/src/dataset.MTSDatasetH.py`
- `examples/benchmarks/TFT/tft.py`

### Python version constraints

- Core library: Python 3.8-3.12
- RL module: Requires `numpy<2.0.0` due to PyTorch compatibility
- TFT model: Only supports Python 3.6-3.7 (tensorflow 1.15 limitation)

### macOS M1 issues

Installing LightGBM on M1 Macs requires OpenMP:
```bash
brew install libomp
pip install .
```

### Data security

Recent security fixes restrict pickle deserialization to safe classes. Custom model classes may need whitelisting in `qlib/utils/__init__.py`.

## Security Considerations

- **Pickle safety**: Qlib restricts unpickling to known-safe classes. Add custom classes to allowlist carefully.
- **SQL injection**: Expression evaluation uses AST parsing, not eval(). Safe by design.
- **Path traversal**: Data paths validated before filesystem access.
- **Credentials**: Never commit API keys. Use environment variables or separate config files.

## Performance Tips

- Enable ExpressionCache and DatasetCache for repeated experiments
- Use Cython-compiled operations for custom features when possible
- Profile with `scripts/collect_info.py` to identify bottlenecks
- Use `multiprocessing` in Trainer for parallel model training
- Consider client-server mode for shared data access across experiments
