"""
Model training and analysis scripts.

Directory structure:
- common/     : Shared utilities (config, training, backtest)
- tree/       : Tree-based models (LightGBM, CatBoost, XGBoost)
- deep/       : Deep learning models (ALSTM, TCN, Transformer, TFT)
- autogluon/  : AutoGluon-based models
- ensemble/   : Ensemble models and backtesting
- analysis/   : Analysis and diagnostic scripts
"""

# Re-export common module for backward compatibility
from . import common
