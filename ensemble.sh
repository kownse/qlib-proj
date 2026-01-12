python scripts/models/run_lgb_nd.py --stock-pool sp500 --handler alpha360 --backtest --topk 10 --n-drop 1 --account 10000 --rebalance-freq 5
python scripts/models/run_catboost_nd.py --stock-pool sp500 --handler alpha360 --backtest --topk 10 --n-drop 1 --account 10000 --rebalance-freq 5

python scripts/models/run_ensemble_backtest.py --stock-pool sp500 --topk 10 --n-drop 1 --rebalance-freq 5


python ./scripts/models/run_lgb_nd.py --stock-pool sp500 --handler alpha360       --backtest --topk 10 --n-drop 1 --rebalance-freq 5 --nday 5 --strategy dynamic_risk --risk-lookback 20 --drawdown-threshold -0.08       --momentum-threshold 0.05       --risk-high 0.40       --risk-medium 0.70       --risk-normal 0.95       --market-proxy AAPL --model-path my_models/lgb_alpha360_sp500_5d_20260109_222239.txt