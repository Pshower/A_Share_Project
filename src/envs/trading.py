"""Daily close decisions, next open execution, next close rewards."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .broker import BrokerSimulator
from .portfolio import Portfolio


class TargetWeightSpace(spaces.Box):
    """Box-shaped long-only weights with unallocated weight held as cash."""

    def sample(self, mask=None, probability=None):
        weights = super().sample(mask=mask, probability=probability)
        return (weights / max(1.0, float(weights.sum()))).astype(self.dtype)

    def contains(self, x):
        return super().contains(x) and np.asarray(x).sum() <= 1 + 1e-7


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, market, initial_capital=1e6, broker_config=None, max_steps=None):
        super().__init__()
        if max_steps is not None and (not isinstance(max_steps, int) or max_steps < 1):
            raise ValueError("max_steps must be a positive integer")
        self.market = market
        self.portfolio = Portfolio(initial_capital, market.stock_codes)
        self.broker = BrokerSimulator(broker_config)
        self.max_steps = max_steps
        n = market.n_stocks
        self.action_space = TargetWeightSpace(0.0, self.broker.config.max_weight, (n,), np.float32)
        self.observation_space = spaces.Dict({
            "stock_features": spaces.Box(-np.inf, np.inf,
                                         (n, market.lookback, len(market.feature_cols)), np.float32),
            "portfolio": spaces.Box(0.0, 1.0, (n + 1,), np.float32),
            "receivable_ratio": spaces.Box(0.0, 1.0, (1,), np.float32),
            "valid_mask": spaces.MultiBinary(n),
            "buy_mask": spaces.MultiBinary(n),
            "sell_mask": spaces.MultiBinary(n),
        })
        self._done = True
        self.history = []
        self.assumptions = list(market.assumptions) + [
            "One rebalance per day; deterministic stock-order cash allocation, no intraday order book.",
            "Uniform configurable buy lot; odd-lot sells allowed. Board-specific order rules are not modeled.",
            "Fees are configurable fixed scenario rates, not historical statutory schedules.",
        ]
        self.assumptions.append(
            "Volume cap uses previous-day shares, not same-day realized volume."
            if self.broker.config.volume_fraction is not None else "Volume participation constraint disabled.")

    def _observation(self):
        features, valid, buy, sell = self.market.observation()
        prices = self.market.prices.iloc[self.market.current_index].to_numpy()
        weights = self.portfolio.get_weights(prices)
        total = self.portfolio.value(prices)
        return dict(stock_features=features, valid_mask=valid.astype(np.int8),
                    receivable_ratio=np.array([self.portfolio.receivable_value / total if total else 0], dtype=np.float32),
                    buy_mask=buy.astype(np.int8), sell_mask=sell.astype(np.int8),
                    portfolio=np.append(weights, self.portfolio.get_cash_ratio(prices)).astype(np.float32))

    def _info(self, orders=None):
        prices = self.market.prices.iloc[self.market.current_index].to_numpy()
        return dict(date=self.market.current_date, equity=self.portfolio.value(prices),
                    cash=self.portfolio.cash, positions=self.portfolio.positions.copy(),
                    available_positions=self.portfolio.available_positions.copy(),
                    receivables=self.portfolio.receivable_value, event_cash=self.portfolio.event_cash,
                    event_shares=self.portfolio.event_shares.copy(),
                    total_fees=self.portfolio.total_fees, orders=orders or [],
                    price_basis=self.market.price_basis, assumptions=list(self.assumptions))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.market.reset()
        self.portfolio.reset()
        self.portfolio.start_day(self.market.current_date)
        self.broker.reset()
        self._steps = 0
        self._done = False
        info = self._info()
        self.history = [info]
        return self._observation(), info

    def step(self, action):
        weights = self.broker.validate_weights(action, self.market.n_stocks)
        return self._advance(weights)

    def advance(self):
        """Advance one day without placing orders, for hold/rebalance schedules."""
        return self._advance(None)

    def _advance(self, weights):
        if self._done:
            raise RuntimeError("Call reset before stepping or after episode end")
        idx = self.market.current_index
        previous_value = self.portfolio.value(self.market.prices.iloc[idx].to_numpy())
        valid = self.market.observation()[1]
        quote = self.market.execution(idx + 1)
        events = self.market.events_for(idx + 1)
        for event in events:
            asset = self.market.stock_codes.index(event["code"])
            bonus = self.portfolio.positions[asset] * (event["share_multiplier"] - 1)
            if not np.isclose(bonus, round(bonus), rtol=0, atol=1e-8):
                raise ValueError("Fractional bonus shares require an explicit cash-in-lieu policy")
        self.portfolio.start_day(self.market.dates[idx + 1])
        for event in events:
            asset = self.market.stock_codes.index(event["code"])
            self.portfolio.apply_corporate_action(asset, event["cash_per_share"], event["share_multiplier"],
                                                  event["payment_date"], event["share_available_date"])
        quote["buy_mask"] &= valid
        for i in np.flatnonzero(~valid):
            quote["reasons"][i].append("invalid_decision_features")
        if weights is None:
            self.portfolio.start_day(self.market.dates[idx + 1])
            orders = []
        else:
            orders = self.broker.rebalance(self.portfolio, weights, date=self.market.dates[idx + 1], **quote)
        self.market.step()
        self._steps += 1
        info = self._info(orders)
        info["corporate_actions"] = events
        reward = info["equity"] / previous_value - 1 if previous_value > 0 else 0.0
        terminated = self.market.current_index == self.market.n_dates - 1 or info["equity"] <= 0
        truncated = not terminated and self.max_steps is not None and self._steps >= self.max_steps
        self._done = terminated or truncated
        info["reward"] = float(reward)
        self.history.append(info)
        return self._observation(), float(reward), bool(terminated), bool(truncated), info
