"""
Custom trading strategies for backtesting.

Extends Qlib's built-in strategies with additional features like
configurable rebalance frequency and dynamic risk control.
"""

import numpy as np
import pandas as pd
from qlib.data import D


class MomentumVolatilityRiskStrategy:
    """
    动量+波动率动态风险控制策略

    根据市场趋势和回撤情况动态调整仓位：
    - 下跌趋势 + 大回撤 → 大幅降仓
    - 下跌趋势 或 小回撤 → 小幅降仓
    - 上涨/横盘 → 正常仓位

    这样可以在下跌时降低风险，同时不会在上涨时错过收益。
    """

    @staticmethod
    def create_strategy_class(
        rebalance_freq=1,
        lookback=20,
        drawdown_threshold=-0.10,
        momentum_threshold=0.03,
        risk_degree_high=0.50,
        risk_degree_medium=0.75,
        risk_degree_normal=0.95,
        market_proxy="SPY",
    ):
        """
        创建带动态风险控制的策略类

        Parameters
        ----------
        rebalance_freq : int
            换仓频率（天）
        lookback : int
            计算动量和波动率的回看天数
        drawdown_threshold : float
            触发大幅降仓的回撤阈值（负数，如 -0.10 表示 10% 回撤）
        momentum_threshold : float
            判断趋势的动量阈值
        risk_degree_high : float
            高风险时的仓位比例
        risk_degree_medium : float
            中风险时的仓位比例
        risk_degree_normal : float
            正常情况的仓位比例
        market_proxy : str
            市场代理股票代码（如 SPY）

        Returns
        -------
        type
            策略类
        """
        from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
        from qlib.backtest.decision import TradeDecisionWO

        class MomentumVolatilityTopkStrategy(TopkDropoutStrategy):
            """带动量+波动率风险控制的TopK策略"""

            def __init__(self, *, rebalance_freq=1, **kwargs):
                super().__init__(**kwargs)
                self._rebalance_freq = rebalance_freq
                self._lookback = lookback
                self._drawdown_threshold = drawdown_threshold
                self._momentum_threshold = momentum_threshold
                self._risk_degree_high = risk_degree_high
                self._risk_degree_medium = risk_degree_medium
                self._risk_degree_normal = risk_degree_normal
                self._market_proxy = market_proxy
                self._market_state_cache = {}

            def _get_market_state(self, trade_date):
                """
                获取市场状态

                Returns
                -------
                tuple : (trend, drawdown)
                    trend: "up" / "down" / "neutral"
                    drawdown: 当前回撤幅度（负数）
                """
                # 使用缓存避免重复计算
                date_key = str(trade_date)[:10]
                if date_key in self._market_state_cache:
                    return self._market_state_cache[date_key]

                try:
                    start_date = pd.Timestamp(trade_date) - pd.Timedelta(days=self._lookback * 2)

                    # 尝试获取市场代理的价格数据
                    # 如果指定的代理不可用，尝试备选列表
                    market_proxies = [self._market_proxy, "AAPL", "MSFT", "GOOGL"]
                    data = pd.DataFrame()

                    for proxy in market_proxies:
                        data = D.features(
                            [proxy],
                            ["$close"],
                            start_time=start_date,
                            end_time=trade_date
                        )
                        if not data.empty and len(data) >= self._lookback:
                            if proxy != self._market_proxy:
                                print(f"[DynamicRisk] Using {proxy} as market proxy (original {self._market_proxy} not available)")
                            break

                    if data.empty or len(data) < self._lookback:
                        return "neutral", 0.0

                    close = data.iloc[:, 0].dropna().tail(self._lookback)

                    if len(close) < 5:
                        return "neutral", 0.0

                    # 计算动量（期间涨跌幅）
                    momentum = (close.iloc[-1] / close.iloc[0]) - 1

                    # 计算当前回撤（相对于期间最高点）
                    rolling_max = close.expanding().max()
                    max_price = rolling_max.iloc[-1]
                    current_price = close.iloc[-1]
                    current_drawdown = (current_price - max_price) / max_price

                    # 判断趋势
                    if momentum > self._momentum_threshold:
                        trend = "up"
                    elif momentum < -self._momentum_threshold:
                        trend = "down"
                    else:
                        trend = "neutral"

                    # 缓存结果
                    self._market_state_cache[date_key] = (trend, current_drawdown)
                    return trend, current_drawdown

                except Exception as e:
                    # 出错时返回中性状态
                    return "neutral", 0.0

            def get_risk_degree(self, trade_step=None):
                """
                根据市场状态动态调整仓位

                Returns
                -------
                float
                    仓位比例 (0-1)
                """
                try:
                    trade_date, _ = self.trade_calendar.get_step_time(trade_step)
                    trend, drawdown = self._get_market_state(trade_date)

                    # 决策逻辑：
                    # 1. 下跌趋势 + 显著回撤 → 大幅降仓
                    # 2. 下跌趋势 或 轻微回撤 → 小幅降仓
                    # 3. 上涨或横盘 → 正常仓位

                    if trend == "down" and drawdown < self._drawdown_threshold:
                        risk_degree = self._risk_degree_high
                    elif trend == "down" or drawdown < self._drawdown_threshold * 0.5:
                        risk_degree = self._risk_degree_medium
                    else:
                        risk_degree = self._risk_degree_normal

                    # 调试打印
                    print(f"[DynamicRisk] {str(trade_date)[:10]} | trend={trend:>7} | drawdown={drawdown:>7.2%} | risk_degree={risk_degree:.0%}")

                    return risk_degree

                except Exception as e:
                    print(f"[DynamicRisk] Error: {e}, using normal risk degree")
                    return self._risk_degree_normal

            def generate_trade_decision(self, execute_result=None):
                """
                生成交易决策，支持换仓频率控制和动态风险

                - 当 risk_degree 降低时，会主动卖出部分持仓来降低总仓位
                - 当 risk_degree 恢复时，会加仓补回仓位
                """
                from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
                import copy

                trade_step = self.trade_calendar.get_trade_step()
                trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)

                # 获取当前风险度
                current_risk_degree = self.get_risk_degree(trade_step)

                # 获取当前持仓信息
                current_position = copy.deepcopy(self.trade_position)
                current_stock_list = current_position.get_stock_list()
                total_value = current_position.calculate_value()
                cash = current_position.get_cash()
                stock_value = total_value - cash

                # 计算当前仓位比例
                current_position_ratio = stock_value / total_value if total_value > 0 else 0

                # 目标仓位比例
                target_position_ratio = current_risk_degree

                print(f"[DynamicRisk] Position: current={current_position_ratio:.1%}, target={target_position_ratio:.1%}, stocks={len(current_stock_list)}")

                # 如果当前仓位超过目标仓位，需要减仓
                if current_position_ratio > target_position_ratio + 0.05:  # 5% 容忍度
                    # 计算需要减仓的金额
                    excess_ratio = current_position_ratio - target_position_ratio
                    reduce_value = total_value * excess_ratio

                    print(f"[DynamicRisk] REDUCING POSITION: need to sell ${reduce_value:,.0f} ({excess_ratio:.1%} of portfolio)")

                    # 生成减仓卖出订单（按比例卖出每只股票）
                    sell_order_list = []
                    reduce_ratio = excess_ratio / current_position_ratio if current_position_ratio > 0 else 0

                    for code in current_stock_list:
                        if not self.trade_exchange.is_stock_tradable(
                            stock_id=code,
                            start_time=trade_start_time,
                            end_time=trade_end_time,
                        ):
                            continue

                        # 计算该股票需要卖出的数量
                        current_amount = current_position.get_stock_amount(code=code)
                        sell_amount = current_amount * reduce_ratio

                        if sell_amount > 0:
                            # 获取交易单位
                            factor = self.trade_exchange.get_factor(
                                stock_id=code,
                                start_time=trade_start_time,
                                end_time=trade_end_time
                            )
                            sell_amount = self.trade_exchange.round_amount_by_trade_unit(sell_amount, factor)

                            if sell_amount > 0:
                                sell_order = Order(
                                    stock_id=code,
                                    amount=sell_amount,
                                    start_time=trade_start_time,
                                    end_time=trade_end_time,
                                    direction=Order.SELL,
                                )
                                if self.trade_exchange.check_order(sell_order):
                                    sell_order_list.append(sell_order)
                                    print(f"[DynamicRisk] Sell {code}: {sell_amount:.0f} shares")

                    if sell_order_list:
                        return TradeDecisionWO(sell_order_list, self)

                # 如果当前仓位低于目标仓位，需要加仓
                elif current_position_ratio < target_position_ratio - 0.05 and len(current_stock_list) > 0:
                    # 计算需要加仓的金额
                    deficit_ratio = target_position_ratio - current_position_ratio
                    add_value = total_value * deficit_ratio

                    # 确保不超过可用现金
                    add_value = min(add_value, cash * 0.95)  # 保留5%现金

                    if add_value > 100:  # 最小加仓金额
                        print(f"[DynamicRisk] INCREASING POSITION: need to buy ${add_value:,.0f} ({deficit_ratio:.1%} of portfolio)")

                        # 按比例加仓现有持仓（每只股票等比例加仓）
                        buy_order_list = []
                        value_per_stock = add_value / len(current_stock_list)

                        for code in current_stock_list:
                            if not self.trade_exchange.is_stock_tradable(
                                stock_id=code,
                                start_time=trade_start_time,
                                end_time=trade_end_time,
                                direction=OrderDir.BUY,
                            ):
                                continue

                            # 获取股票价格
                            buy_price = self.trade_exchange.get_deal_price(
                                stock_id=code,
                                start_time=trade_start_time,
                                end_time=trade_end_time,
                                direction=OrderDir.BUY
                            )

                            if buy_price is None or buy_price <= 0:
                                continue

                            buy_amount = value_per_stock / buy_price

                            # 获取交易单位
                            factor = self.trade_exchange.get_factor(
                                stock_id=code,
                                start_time=trade_start_time,
                                end_time=trade_end_time
                            )
                            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)

                            if buy_amount > 0:
                                buy_order = Order(
                                    stock_id=code,
                                    amount=buy_amount,
                                    start_time=trade_start_time,
                                    end_time=trade_end_time,
                                    direction=Order.BUY,
                                )
                                buy_order_list.append(buy_order)
                                print(f"[DynamicRisk] Buy {code}: {buy_amount:.0f} shares @ ${buy_price:.2f}")

                        if buy_order_list:
                            return TradeDecisionWO(buy_order_list, self)

                # 检查是否为换仓日
                if self._rebalance_freq > 1:
                    if trade_step % self._rebalance_freq != 0:
                        return TradeDecisionWO([], self)

                # 正常换仓逻辑
                self.risk_degree = current_risk_degree
                return super().generate_trade_decision(execute_result)

        return MomentumVolatilityTopkStrategy


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


def get_strategy_config(
    pred_df,
    topk,
    n_drop,
    rebalance_freq=1,
    strategy_type="topk",
    dynamic_risk_params=None,
):
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
    strategy_type : str
        Strategy type: "topk" (default) or "dynamic_risk"
    dynamic_risk_params : dict, optional
        Parameters for dynamic risk strategy:
        - lookback: int, lookback days for momentum/drawdown calculation (default: 20)
        - drawdown_threshold: float, drawdown threshold for high risk (default: -0.10)
        - momentum_threshold: float, momentum threshold for trend (default: 0.03)
        - risk_degree_high: float, position ratio at high risk (default: 0.50)
        - risk_degree_medium: float, position ratio at medium risk (default: 0.75)
        - risk_degree_normal: float, normal position ratio (default: 0.95)
        - market_proxy: str, market proxy symbol (default: "SPY")

    Returns
    -------
    dict
        Strategy configuration for qlib backtest
    """
    if strategy_type == "dynamic_risk":
        # 使用动态风险控制策略
        params = dynamic_risk_params or {}
        DynamicStrategy = MomentumVolatilityRiskStrategy.create_strategy_class(
            rebalance_freq=rebalance_freq,
            lookback=params.get("lookback", 20),
            drawdown_threshold=params.get("drawdown_threshold", -0.10),
            momentum_threshold=params.get("momentum_threshold", 0.03),
            risk_degree_high=params.get("risk_degree_high", 0.50),
            risk_degree_medium=params.get("risk_degree_medium", 0.75),
            risk_degree_normal=params.get("risk_degree_normal", 0.95),
            market_proxy=params.get("market_proxy", "SPY"),
        )
        return {
            "class": DynamicStrategy,
            "kwargs": {
                "signal": pred_df,
                "topk": topk,
                "n_drop": n_drop,
                "rebalance_freq": rebalance_freq,
            },
        }
    elif rebalance_freq == 1:
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
