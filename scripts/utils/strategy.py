"""
Custom trading strategies for backtesting.

Extends Qlib's built-in strategies with additional features like
configurable rebalance frequency, dynamic risk control,
and portfolio optimization (MVO, Risk Parity, etc.).
"""

import copy
import warnings
import numpy as np
import pandas as pd
from qlib.data import D


class VolatilityStopLossStrategy:
    """
    波动率预警 + 个股止损策略

    核心思路：
    1. 用波动率预警：波动率上升时提前降仓，不等确认下跌
    2. 个股止损：单只股票亏损超过阈值就卖出
    3. 大跌后不追卖：已经大跌的不再卖（可能是反弹机会）

    优点：
    - 提前预警，不是追涨杀跌
    - 止损明确，控制单票风险
    - 不会在底部恐慌抛售
    """

    @staticmethod
    def create_strategy_class(
        rebalance_freq=1,
        lookback=20,
        vol_threshold_high=0.35,      # 年化波动率超过35%为高波动
        vol_threshold_medium=0.25,    # 年化波动率超过25%为中波动
        stop_loss_threshold=-0.15,    # 单股止损线：亏损15%
        no_sell_after_drop=-0.20,     # 已跌超过20%不再卖出（等反弹）
        risk_degree_high_vol=0.60,    # 高波动时仓位
        risk_degree_medium_vol=0.80,  # 中波动时仓位
        risk_degree_normal=0.95,      # 正常仓位
        market_proxy="SPY",
    ):
        """
        创建波动率+止损策略类

        Parameters
        ----------
        vol_threshold_high : float
            高波动率阈值（年化），超过此值大幅降仓
        vol_threshold_medium : float
            中波动率阈值（年化）
        stop_loss_threshold : float
            个股止损阈值，如 -0.15 表示亏损15%止损
        no_sell_after_drop : float
            已跌超过此值不再卖出（避免底部抛售）
        """
        from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
        from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO

        class VolatilityStopLossTopkStrategy(TopkDropoutStrategy):
            """波动率预警+止损策略"""

            def __init__(self, *, rebalance_freq=1, **kwargs):
                super().__init__(**kwargs)
                self._rebalance_freq = rebalance_freq
                self._lookback = lookback
                self._vol_threshold_high = vol_threshold_high
                self._vol_threshold_medium = vol_threshold_medium
                self._stop_loss_threshold = stop_loss_threshold
                self._no_sell_after_drop = no_sell_after_drop
                self._risk_degree_high_vol = risk_degree_high_vol
                self._risk_degree_medium_vol = risk_degree_medium_vol
                self._risk_degree_normal = risk_degree_normal
                self._market_proxy = market_proxy
                self._cache = {}
                # 记录每只股票的买入成本
                self._cost_basis = {}

            def _get_market_volatility(self, trade_date):
                """获取市场波动率（年化）"""
                date_key = f"vol_{str(trade_date)[:10]}"
                if date_key in self._cache:
                    return self._cache[date_key]

                try:
                    start_date = pd.Timestamp(trade_date) - pd.Timedelta(days=self._lookback * 2)

                    # 尝试获取市场代理数据
                    market_proxies = [self._market_proxy, "AAPL", "MSFT", "GOOGL"]
                    for proxy in market_proxies:
                        data = D.features(
                            [proxy],
                            ["$close"],
                            start_time=start_date,
                            end_time=trade_date
                        )
                        if not data.empty and len(data) >= self._lookback:
                            break

                    if data.empty or len(data) < self._lookback:
                        return 0.20  # 默认20%波动率

                    close = data.iloc[:, 0].dropna().tail(self._lookback)
                    if len(close) < 5:
                        return 0.20

                    # 计算日收益率的标准差，年化
                    returns = close.pct_change().dropna()
                    daily_vol = returns.std()
                    annual_vol = daily_vol * np.sqrt(252)

                    self._cache[date_key] = annual_vol
                    return annual_vol

                except Exception:
                    return 0.20

            def _get_stock_return(self, code, trade_date):
                """获取个股从买入到现在的收益率"""
                if code not in self._cost_basis:
                    return 0.0

                try:
                    current_price = D.features(
                        [code], ["$close"],
                        start_time=trade_date,
                        end_time=trade_date
                    )
                    if current_price.empty:
                        return 0.0

                    price = current_price.iloc[-1, 0]
                    cost = self._cost_basis[code]
                    return (price - cost) / cost if cost > 0 else 0.0
                except Exception:
                    return 0.0

            def _update_cost_basis(self, code, price, amount, direction):
                """更新成本基础"""
                if direction == Order.BUY:
                    if code in self._cost_basis:
                        # 简化：用新价格更新（实际应该加权平均）
                        self._cost_basis[code] = price
                    else:
                        self._cost_basis[code] = price
                elif direction == Order.SELL:
                    # 如果全部卖出，移除成本记录
                    pass  # 保留成本记录以备后用

            def get_risk_degree(self, trade_step=None):
                """根据波动率动态调整仓位"""
                try:
                    trade_date, _ = self.trade_calendar.get_step_time(trade_step)
                    vol = self._get_market_volatility(trade_date)

                    if vol > self._vol_threshold_high:
                        risk_degree = self._risk_degree_high_vol
                        vol_level = "HIGH"
                    elif vol > self._vol_threshold_medium:
                        risk_degree = self._risk_degree_medium_vol
                        vol_level = "MEDIUM"
                    else:
                        risk_degree = self._risk_degree_normal
                        vol_level = "LOW"

                    print(f"[VolStopLoss] {str(trade_date)[:10]} | volatility={vol:.1%} ({vol_level}) | risk_degree={risk_degree:.0%}")
                    return risk_degree

                except Exception as e:
                    print(f"[VolStopLoss] Error: {e}")
                    return self._risk_degree_normal

            def generate_trade_decision(self, execute_result=None):
                """生成交易决策：波动率控制 + 个股止损"""
                import copy

                trade_step = self.trade_calendar.get_trade_step()
                trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)

                current_risk_degree = self.get_risk_degree(trade_step)

                current_position = copy.deepcopy(self.trade_position)
                current_stock_list = current_position.get_stock_list()
                total_value = current_position.calculate_value()
                cash = current_position.get_cash()
                stock_value = total_value - cash
                current_position_ratio = stock_value / total_value if total_value > 0 else 0

                # 初始化成本基础（如果没有）
                for code in current_stock_list:
                    if code not in self._cost_basis:
                        try:
                            price_data = D.features(
                                [code], ["$close"],
                                start_time=trade_start_time,
                                end_time=trade_end_time
                            )
                            if not price_data.empty:
                                self._cost_basis[code] = price_data.iloc[-1, 0]
                        except Exception:
                            pass

                # === 第1步：检查个股止损 ===
                stop_loss_sells = []
                for code in current_stock_list:
                    stock_return = self._get_stock_return(code, trade_start_time)

                    # 止损条件：亏损超过阈值，但没有跌太多（避免底部抛售）
                    if stock_return < self._stop_loss_threshold and stock_return > self._no_sell_after_drop:
                        if not self.trade_exchange.is_stock_tradable(
                            stock_id=code, start_time=trade_start_time, end_time=trade_end_time
                        ):
                            continue

                        sell_amount = current_position.get_stock_amount(code=code)
                        factor = self.trade_exchange.get_factor(
                            stock_id=code, start_time=trade_start_time, end_time=trade_end_time
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
                                stop_loss_sells.append(sell_order)
                                print(f"[VolStopLoss] STOP LOSS {code}: return={stock_return:.1%}, selling {sell_amount:.0f} shares")

                if stop_loss_sells:
                    return TradeDecisionWO(stop_loss_sells, self)

                # === 第2步：波动率仓位控制 ===
                target_position_ratio = current_risk_degree

                print(f"[VolStopLoss] Position: current={current_position_ratio:.1%}, target={target_position_ratio:.1%}")

                # 如果仓位过高，减仓
                if current_position_ratio > target_position_ratio + 0.05:
                    excess_ratio = current_position_ratio - target_position_ratio
                    reduce_ratio = excess_ratio / current_position_ratio

                    print(f"[VolStopLoss] REDUCING: {excess_ratio:.1%} of portfolio")

                    sell_order_list = []
                    for code in current_stock_list:
                        # 检查该股票是否已经大跌（不卖）
                        stock_return = self._get_stock_return(code, trade_start_time)
                        if stock_return < self._no_sell_after_drop:
                            print(f"[VolStopLoss] Skip {code}: already down {stock_return:.1%}, waiting for rebound")
                            continue

                        if not self.trade_exchange.is_stock_tradable(
                            stock_id=code, start_time=trade_start_time, end_time=trade_end_time
                        ):
                            continue

                        current_amount = current_position.get_stock_amount(code=code)
                        sell_amount = current_amount * reduce_ratio

                        factor = self.trade_exchange.get_factor(
                            stock_id=code, start_time=trade_start_time, end_time=trade_end_time
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
                                print(f"[VolStopLoss] Sell {code}: {sell_amount:.0f} shares")

                    if sell_order_list:
                        return TradeDecisionWO(sell_order_list, self)

                # 如果仓位过低且波动率已降，加仓
                elif current_position_ratio < target_position_ratio - 0.05 and len(current_stock_list) > 0:
                    deficit_ratio = target_position_ratio - current_position_ratio
                    add_value = min(total_value * deficit_ratio, cash * 0.95)

                    if add_value > 100:
                        print(f"[VolStopLoss] INCREASING: ${add_value:,.0f}")

                        buy_order_list = []
                        value_per_stock = add_value / len(current_stock_list)

                        for code in current_stock_list:
                            if not self.trade_exchange.is_stock_tradable(
                                stock_id=code, start_time=trade_start_time, end_time=trade_end_time,
                                direction=OrderDir.BUY
                            ):
                                continue

                            buy_price = self.trade_exchange.get_deal_price(
                                stock_id=code, start_time=trade_start_time, end_time=trade_end_time,
                                direction=OrderDir.BUY
                            )
                            if buy_price is None or buy_price <= 0:
                                continue

                            buy_amount = value_per_stock / buy_price
                            factor = self.trade_exchange.get_factor(
                                stock_id=code, start_time=trade_start_time, end_time=trade_end_time
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
                                self._cost_basis[code] = buy_price
                                print(f"[VolStopLoss] Buy {code}: {buy_amount:.0f} @ ${buy_price:.2f}")

                        if buy_order_list:
                            return TradeDecisionWO(buy_order_list, self)

                # 检查是否为换仓日
                if self._rebalance_freq > 1:
                    if trade_step % self._rebalance_freq != 0:
                        return TradeDecisionWO([], self)

                self.risk_degree = current_risk_degree
                return super().generate_trade_decision(execute_result)

        return VolatilityStopLossTopkStrategy


class MomentumVolatilityRiskStrategy:
    """
    动量+波动率动态风险控制策略（旧版本，保留兼容）

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


class OptimizedWeightStrategy:
    """
    基于 PortfolioOptimizer 的权重优化策略。

    支持 4 种优化方法:
    - mvo: Mean-Variance Optimization (利用模型预测 + 协方差矩阵)
    - rp: Risk Parity (等风险贡献)
    - gmv: Global Minimum Variance (最小化组合波动)
    - inv: Inverse Volatility (反波动率加权)

    支持 rebalance_freq 控制调仓频率。
    """

    @staticmethod
    def create_strategy_class(
        opt_method="mvo",
        lamb=2.0,
        delta=0.2,
        alpha=0.01,
        scale_return=True,
        cov_lookback=60,
        rebalance_freq=1,
        risk_degree=0.95,
        max_weight=0.0,
    ):
        """
        Create a WeightStrategyBase subclass with portfolio optimization.

        Parameters
        ----------
        opt_method : str
            Optimization method: "mvo", "rp", "gmv", "inv"
        lamb : float
            Risk aversion for MVO (higher = more risk-averse)
        delta : float
            Turnover limit per rebalance (0.2 = max 20% turnover)
        alpha : float
            L2 regularization to prevent weight concentration
        scale_return : bool
            Scale prediction scores to match covariance magnitude
        cov_lookback : int
            Days of history for covariance estimation
        rebalance_freq : int
            Rebalance every N days
        risk_degree : float
            Fraction of capital to deploy (default 0.95)
        max_weight : float
            Maximum weight per stock (0 = no limit, 0.15 = max 15% per stock).
            Excess weight is redistributed proportionally to other stocks.

        Returns
        -------
        type
            A strategy class with portfolio optimization
        """
        from qlib.contrib.strategy.signal_strategy import WeightStrategyBase
        from qlib.contrib.strategy.optimizer import PortfolioOptimizer
        from qlib.backtest.decision import TradeDecisionWO

        _opt_method = opt_method
        _lamb = lamb
        _delta = delta
        _alpha = alpha
        _scale_return = scale_return
        _cov_lookback = cov_lookback
        _rebalance_freq = rebalance_freq
        _risk_degree = risk_degree
        _max_weight = max_weight

        class OptimizedWeightStrategyImpl(WeightStrategyBase):
            def __init__(self, *, topk=30, rebalance_freq=1, **kwargs):
                super().__init__(**kwargs)
                self._topk = topk
                self._rebalance_freq = _rebalance_freq
                print(f"[MVO-INIT] OptimizedWeightStrategyImpl created: topk={topk}, rebalance_freq={_rebalance_freq}, method={_opt_method}")
                self._risk_degree = _risk_degree
                self._max_weight = _max_weight
                self._optimizer = PortfolioOptimizer(
                    method=_opt_method,
                    lamb=_lamb,
                    delta=_delta,
                    alpha=_alpha,
                    scale_return=_scale_return,
                )
                self._cov_lookback = _cov_lookback
                self._prev_weights = {}

            def get_risk_degree(self, trade_step=None):
                return self._risk_degree

            def generate_trade_decision(self, execute_result=None):
                trade_step = self.trade_calendar.get_trade_step()

                # Rebalance frequency control
                if self._rebalance_freq > 1 and trade_step % self._rebalance_freq != 0:
                    return TradeDecisionWO([], self)

                # On rebalance days, replicate WeightStrategyBase logic with debug
                trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
                pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
                pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)

                if pred_score is None:
                    print(f"[MVO-DEBUG] Step={trade_step}: pred_score is None, no trade")
                    return TradeDecisionWO([], self)

                if isinstance(pred_score, pd.DataFrame):
                    pred_score = pred_score.iloc[:, 0]

                current_temp = copy.deepcopy(self.trade_position)

                target_weight_position = self.generate_target_weight_position(
                    score=pred_score, current=current_temp,
                    trade_start_time=trade_start_time, trade_end_time=trade_end_time
                )

                if trade_step < 8:
                    print(f"[MVO-DEBUG] Step={trade_step}: target_weights={len(target_weight_position)} stocks, "
                          f"weights={dict(list(target_weight_position.items())[:3])}...")

                order_list = self.order_generator.generate_order_list_from_target_weight_position(
                    current=current_temp,
                    trade_exchange=self.trade_exchange,
                    risk_degree=self.get_risk_degree(trade_step),
                    target_weight_position=target_weight_position,
                    pred_start_time=pred_start_time,
                    pred_end_time=pred_end_time,
                    trade_start_time=trade_start_time,
                    trade_end_time=trade_end_time,
                )

                if trade_step < 8:
                    print(f"[MVO-DEBUG] Step={trade_step}: order_list={len(order_list)} orders")

                return TradeDecisionWO(order_list, self)

            def generate_target_weight_position(self, score, current,
                                                 trade_start_time, trade_end_time):
                if score is None or len(score) == 0:
                    return {}

                # Handle DataFrame (multiple signals) - use first column
                if isinstance(score, pd.DataFrame):
                    score = score.iloc[:, 0]

                # 1. Select top-K stocks by prediction score
                topk_scores = score.nlargest(self._topk)
                stock_ids = list(topk_scores.index)

                if len(stock_ids) < 2:
                    # Too few stocks for optimization, equal weight
                    return {sid: 1.0 / max(len(stock_ids), 1) for sid in stock_ids}

                # 2. Get covariance matrix from historical returns
                try:
                    cov_matrix = self._compute_covariance(stock_ids, trade_start_time)
                except Exception as e:
                    warnings.warn(f"Covariance computation failed: {e}, using equal weight")
                    return {sid: 1.0 / len(stock_ids) for sid in stock_ids}

                # 3. Build current weight vector for turnover constraint
                cur_weights = current.get_stock_weight_dict(only_stock=True)
                if cur_weights and any(sid in cur_weights for sid in stock_ids):
                    w0 = pd.Series(
                        [cur_weights.get(sid, 0.0) for sid in stock_ids],
                        index=stock_ids
                    )
                    total = w0.sum()
                    if total > 0:
                        w0 = w0 / total
                    else:
                        w0 = None
                else:
                    w0 = None

                # 4. Run optimizer
                try:
                    r = topk_scores if _opt_method == "mvo" else None
                    weights = self._optimizer(S=cov_matrix, r=r, w0=w0)

                    if isinstance(weights, pd.Series):
                        opt_weights = {sid: weights[sid] if sid in weights.index else 0.0
                                       for sid in stock_ids}
                    else:
                        opt_weights = dict(zip(stock_ids, weights))

                    # Normalize optimizer weights
                    total_w = sum(opt_weights.values())
                    if total_w > 0:
                        opt_weights = {k: v / total_w for k, v in opt_weights.items()}
                    else:
                        opt_weights = {k: 1.0 / len(stock_ids) for k in stock_ids}

                    # Blend with equal weight to ensure diversification
                    if self._max_weight > 0:
                        eq_w = 1.0 / len(stock_ids)
                        # blend factor: how much optimizer to trust
                        # max_weight determines the blend: higher cap = more optimizer trust
                        # At max_weight=1/topk (equal weight), blend=0
                        # At max_weight=1.0 (no cap), blend=1
                        blend = min(1.0, self._max_weight * len(stock_ids) - 1.0)
                        blend = max(0.0, blend)
                        result = {}
                        for sid in stock_ids:
                            w = blend * opt_weights.get(sid, 0.0) + (1 - blend) * eq_w
                            result[sid] = w
                        # Hard cap after blending
                        result = self._apply_max_weight_cap(result)
                    else:
                        # No cap: use optimizer weights directly, drop near-zero
                        result = {k: v for k, v in opt_weights.items() if v > 1e-6}
                        total_w = sum(result.values())
                        if total_w > 0:
                            result = {k: v / total_w for k, v in result.items()}

                    self._prev_weights = result
                    return result

                except Exception as e:
                    warnings.warn(f"Optimization failed: {e}, using equal weight")
                    return {sid: 1.0 / len(stock_ids) for sid in stock_ids}

            def _apply_max_weight_cap(self, weights):
                """Cap individual stock weights and redistribute excess.
                Iteratively caps and redistributes until all weights are within limit."""
                cap = self._max_weight
                n = len(weights)
                if n == 0 or cap <= 0 or cap >= 1.0:
                    return weights

                # If cap * n < 1, cap is too tight; use equal weight
                if cap * n < 1.0:
                    return {k: 1.0 / n for k in weights}

                weights = dict(weights)
                for _ in range(20):
                    over = {k: v for k, v in weights.items() if v > cap + 1e-9}
                    if not over:
                        break
                    excess = sum(v - cap for v in over.values())
                    under = {k: v for k, v in weights.items() if v <= cap + 1e-9}
                    # Cap the overweight stocks
                    for k in over:
                        weights[k] = cap
                    # Redistribute excess equally among underweight stocks
                    if under:
                        per_stock = excess / len(under)
                        for k in under:
                            weights[k] = weights[k] + per_stock
                return weights

            def _compute_covariance(self, stock_ids, trade_date):
                """Compute covariance matrix from historical daily returns."""
                end_date = pd.Timestamp(trade_date) - pd.Timedelta(days=1)
                start_date = end_date - pd.Timedelta(days=self._cov_lookback * 2)

                close_data = D.features(
                    stock_ids,
                    fields=["$close"],
                    start_time=start_date.strftime("%Y-%m-%d"),
                    end_time=end_date.strftime("%Y-%m-%d"),
                )

                if close_data is None or close_data.empty:
                    raise ValueError("No historical data available")

                close_df = close_data["$close"].unstack(level=0)

                # Drop stocks with insufficient data
                min_obs = max(self._cov_lookback // 2, 20)
                close_df = close_df.dropna(axis=1, thresh=min_obs)

                if close_df.shape[1] < 2:
                    raise ValueError("Too few stocks with sufficient history")

                # Use last cov_lookback days
                close_df = close_df.tail(self._cov_lookback)

                # Compute daily returns
                returns = close_df.pct_change().dropna()

                if len(returns) < min_obs:
                    raise ValueError(f"Only {len(returns)} return observations, need {min_obs}")

                # Shrinkage estimator: blend sample cov with diagonal (Ledoit-Wolf lite)
                sample_cov = returns.cov()
                shrink_factor = 0.3
                diag = pd.DataFrame(
                    np.diag(np.diag(sample_cov.values)),
                    index=sample_cov.index,
                    columns=sample_cov.columns,
                )
                cov_matrix = (1 - shrink_factor) * sample_cov + shrink_factor * diag

                # Align to stock_ids order, fill missing with average variance
                available = [s for s in stock_ids if s in cov_matrix.index]
                missing = [s for s in stock_ids if s not in cov_matrix.index]

                if len(available) < 2:
                    raise ValueError("Too few stocks in covariance matrix")

                if missing:
                    avg_var = np.mean(np.diag(cov_matrix.loc[available, available].values))
                    full_index = available + missing
                    full_cov = pd.DataFrame(0.0, index=full_index, columns=full_index)
                    full_cov.loc[available, available] = cov_matrix.loc[available, available]
                    for s in missing:
                        full_cov.loc[s, s] = avg_var
                    cov_matrix = full_cov

                # Reorder to match stock_ids
                ordered = [s for s in stock_ids if s in cov_matrix.index]
                return cov_matrix.loc[ordered, ordered]

        return OptimizedWeightStrategyImpl


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
    optimizer_params=None,
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
        Strategy type: "topk", "dynamic_risk", "vol_stoploss",
        "mvo", "rp", "gmv", "inv"
    dynamic_risk_params : dict, optional
        Parameters for risk strategies
    optimizer_params : dict, optional
        Parameters for portfolio optimization strategies

    Returns
    -------
    dict
        Strategy configuration for qlib backtest
    """
    # Portfolio optimization strategies
    if strategy_type in ("mvo", "rp", "gmv", "inv"):
        params = optimizer_params or {}
        OptStrategy = OptimizedWeightStrategy.create_strategy_class(
            opt_method=strategy_type,
            lamb=params.get("lamb", 2.0),
            delta=params.get("delta", 0.2),
            alpha=params.get("alpha", 0.01),
            scale_return=params.get("scale_return", True),
            cov_lookback=params.get("cov_lookback", 60),
            rebalance_freq=rebalance_freq,
            risk_degree=params.get("risk_degree", 0.95),
            max_weight=params.get("max_weight", 0.0),
        )
        return {
            "class": OptStrategy,
            "kwargs": {
                "signal": pred_df,
                "topk": topk,
                "rebalance_freq": rebalance_freq,
            },
        }
    elif strategy_type == "vol_stoploss":
        # 使用波动率预警+止损策略（推荐）
        params = dynamic_risk_params or {}
        VolStrategy = VolatilityStopLossStrategy.create_strategy_class(
            rebalance_freq=rebalance_freq,
            lookback=params.get("lookback", 20),
            vol_threshold_high=params.get("vol_threshold_high", 0.35),
            vol_threshold_medium=params.get("vol_threshold_medium", 0.25),
            stop_loss_threshold=params.get("stop_loss_threshold", -0.15),
            no_sell_after_drop=params.get("no_sell_after_drop", -0.20),
            risk_degree_high_vol=params.get("risk_degree_high", 0.60),
            risk_degree_medium_vol=params.get("risk_degree_medium", 0.80),
            risk_degree_normal=params.get("risk_degree_normal", 0.95),
            market_proxy=params.get("market_proxy", "SPY"),
        )
        return {
            "class": VolStrategy,
            "kwargs": {
                "signal": pred_df,
                "topk": topk,
                "n_drop": n_drop,
                "rebalance_freq": rebalance_freq,
            },
        }
    elif strategy_type == "dynamic_risk":
        # 使用动量+回撤策略（旧版）
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
