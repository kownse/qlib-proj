python scripts/models/run_lgb_nd.py --stock-pool sp500 --handler alpha360 --backtest --topk 10 --n-drop 1 --account 10000 --rebalance-freq 5
python scripts/models/run_catboost_nd.py --stock-pool sp500 --handler alpha360 --backtest --topk 10 --n-drop 1 --account 10000 --rebalance-freq 5

python scripts/models/run_ensemble_backtest.py --stock-pool sp500 --topk 10 --n-drop 1 --rebalance-freq 5