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

# 注意: 不自动导入 common，避免过早触发 qlib 初始化
# 这会导致 TA-Lib 与 qlib 多进程的内存冲突
# 请使用: from models.common import ...
# 而不是: from models import common
