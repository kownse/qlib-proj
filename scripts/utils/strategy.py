"""
Custom trading strategies for backtesting.

Extends Qlib's built-in strategies with additional features like
configurable rebalance frequency.
"""


class TopkDropoutStrategyWithRebalance:
    """
    TopkDropoutStrategy with configurable rebalance frequency.

    This is a wrapper that creates a strategy class dynamically with
    the specified rebalance frequency.
    """

    @staticmethod
    def create_strategy_class(rebalance_freq=1):
        """
        Create a TopkDropoutStrategy subclass with rebalance frequency control.

        Parameters
        ----------
        rebalance_freq : int
            Rebalance every N days. 1 = daily rebalance (default behavior).

        Returns
        -------
        type
            A strategy class with rebalance frequency control.
        """
        from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
        from qlib.backtest.decision import TradeDecisionWO

        class TopkDropoutStrategyRebalanced(TopkDropoutStrategy):
            """TopkDropoutStrategy with rebalance frequency control."""

            def __init__(self, *, rebalance_freq=1, **kwargs):
                super().__init__(**kwargs)
                self._rebalance_freq = rebalance_freq
                self._last_rebalance_step = None

            def generate_trade_decision(self, execute_result=None):
                trade_step = self.trade_calendar.get_trade_step()

                # Check if we should rebalance today
                if self._rebalance_freq > 1:
                    # Only rebalance on step 0, rebalance_freq, 2*rebalance_freq, etc.
                    if trade_step % self._rebalance_freq != 0:
                        # Not a rebalance day, return empty decision (hold current position)
                        return TradeDecisionWO([], self)

                # Rebalance day - call parent's implementation
                return super().generate_trade_decision(execute_result)

        return TopkDropoutStrategyRebalanced


def get_strategy_config(pred_df, topk, n_drop, rebalance_freq=1):
    """
    Get strategy configuration for backtest.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Prediction DataFrame with 'score' column
    topk : int
        Number of stocks to hold
    n_drop : int
        Number of stocks to drop/replace each rebalance
    rebalance_freq : int
        Rebalance frequency in days (default: 1)

    Returns
    -------
    dict
        Strategy configuration for qlib backtest
    """
    if rebalance_freq == 1:
        # Default daily rebalance, use original strategy
        return {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": pred_df,
                "topk": topk,
                "n_drop": n_drop,
            },
        }
    else:
        # Use custom strategy with rebalance frequency control
        RebalancedStrategy = TopkDropoutStrategyWithRebalance.create_strategy_class(rebalance_freq)
        return {
            "class": RebalancedStrategy,
            "kwargs": {
                "signal": pred_df,
                "topk": topk,
                "n_drop": n_drop,
                "rebalance_freq": rebalance_freq,
            },
        }
