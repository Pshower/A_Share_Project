"""Conditional whole-lot trade intents with independently checked execution."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd

from .delta_mapping import ALLOCATION_FIELDS, EXECUTION_FIELDS, enrich_delta_observation


HOLD, BUY, REDUCE, EXIT = range(4)
DIRECTION_NAMES = ("hold", "buy", "reduce", "exit")
TRADE_MAPPING_VERSION = "stock_trade_lots_hold_buy_reduce_exit_residual_cash_v4"
TRADE_ALLOCATION_FIELDS = (*ALLOCATION_FIELDS, "minimum_trade_value_ratio")
SHARE_FIELDS = ("current_lots", "available_lots")
LOT_BOUND_FIELDS = ("buy_min", "buy_max", "reduce_min", "reduce_max")


@dataclass
class TradeOrder:
    shares: np.ndarray
    freeze_mask: np.ndarray
    exit_mask: np.ndarray
    priority: np.ndarray
    top_n: int
    diagnostics: dict


def _available(portfolio, execution_date):
    if execution_date is not None and pd.Timestamp(execution_date) > portfolio.current_date:
        available = portfolio.positions.copy()
        for due, asset, quantity in portfolio.share_locks:
            if due > pd.Timestamp(execution_date):
                available[asset] -= quantity
    else:
        available = portfolio.available_positions.copy()
    return np.maximum(0., np.minimum(available, portfolio.positions))


def enrich_trade_observation(observation, portfolio, prices, broker_config, top_n,
                             max_trade_weight=.05, max_commission_ratio=.001, execution_date=None):
    if (not np.isfinite([max_trade_weight, max_commission_ratio]).all()
            or not 0 < max_trade_weight <= 1 or not 0 < max_commission_ratio <= 1
            or max_commission_ratio < broker_config.commission):
        raise ValueError("Invalid trade limits or commission-ratio guarantee")
    result = enrich_delta_observation(observation, portfolio, prices, broker_config, top_n, execution_date)
    prices = np.asarray(prices, dtype=float)
    equity = portfolio.value(prices)
    if not np.isfinite(equity) or equity <= 0:
        raise ValueError("Trade observations require positive finite equity")
    quantity = portfolio.positions.copy()
    available = _available(portfolio, execution_date)
    lot = broker_config.lot_size
    known = np.isfinite(prices) & (prices > 0)
    valid = np.asarray(observation["valid_mask"], dtype=bool) & known
    frozen = result["execution_features"][:, 2].astype(bool) | ((quantity > 0) & ~known)
    result["execution_features"][:, 2] = frozen
    sellable = valid & ~frozen & np.asarray(observation["sell_mask"], dtype=bool)
    legal = np.zeros((portfolio.n_stocks, 4), dtype=bool)
    legal[:, HOLD] = True
    legal[:, EXIT] = sellable & (quantity > 0) & (available == quantity)
    bounds = np.zeros((portfolio.n_stocks, 4), dtype=np.int64)
    hurdle = broker_config.min_commission / max_commission_ratio
    possible_proceeds = np.zeros(portfolio.n_stocks)
    for i in np.flatnonzero(sellable):
        price = prices[i] * (1 - broker_config.slippage)
        minimum = max(1, int(np.ceil(hurdle / (lot * price))))
        maximum = max(0, int(np.floor(min(available[i], max(0., quantity[i] - 1),
                                         max_trade_weight * equity / price) / lot)))
        bounds[i, 2:] = [minimum, maximum]
        legal[i, REDUCE] = maximum >= minimum
        if legal[i, EXIT]:
            possible_proceeds[i] = quantity[i] * price
        elif legal[i, REDUCE]:
            possible_proceeds[i] = maximum * lot * price
        amount = possible_proceeds[i]
        if amount > 0:
            possible_proceeds[i] = max(0., amount - max(broker_config.min_commission,
                amount * broker_config.commission) - amount * broker_config.sell_tax)
    potential_cash = portfolio.cash + float(possible_proceeds.sum())
    unavoidable_names = int(((quantity > 0) & ~legal[:, EXIT]).sum())
    for i in np.flatnonzero(valid & ~frozen & np.asarray(observation["buy_mask"], dtype=bool)):
        price = prices[i] * (1 + broker_config.slippage)
        minimum = max(1, int(np.ceil(hurdle / (lot * price))))
        room = max(0., broker_config.max_weight * equity / prices[i] - quantity[i])
        cash_room = max(0., min(potential_cash - broker_config.min_commission,
                                potential_cash / (1 + broker_config.commission)))
        maximum = max(0, int(np.floor(min(room, max_trade_weight * equity / price, cash_room / price) / lot)))
        bounds[i, :2] = [minimum, maximum]
        slot_possible = quantity[i] > 0 or unavoidable_names < top_n
        legal[i, BUY] = top_n > 0 and maximum >= minimum and slot_possible
    result.update(share_state=np.column_stack([quantity / lot, available / lot]).astype(np.float64),
                  trade_lot_bounds=bounds, legal_directions=legal.astype(np.int8),
                  allocation_context=np.append(result["allocation_context"], hurdle / equity).astype(np.float32))
    return result


def trade_size_active(observation, directions):
    directions = np.asarray(directions)
    bounds = np.asarray(observation["trade_lot_bounds"])
    return (directions == BUY) | ((directions == REDUCE) & (bounds[:, 2] < bounds[:, 3]))


def map_trade_action(raw, observation, bound=8., lot_size=100, stock_codes=None):
    n = len(observation["portfolio"]) - 1
    raw = np.asarray(raw, dtype=float)
    if (raw.shape != (2 * n,) or not np.isfinite(raw).all() or not np.isfinite(bound) or bound <= 0
            or type(lot_size) is not int or lot_size < 1):
        raise ValueError("Expected finite packed directions/sizes and valid trade limits")
    if np.any(raw[:n] != np.rint(raw[:n])) or np.any((raw[:n] < HOLD) | (raw[:n] > EXIT)):
        raise ValueError("Trade directions must be integer codes 0..3")
    directions = raw[:n].astype(np.int64)
    legal_values = np.asarray(observation["legal_directions"])
    legal = legal_values.astype(bool)
    bounds = np.asarray(observation["trade_lot_bounds"])
    state = np.asarray(observation["share_state"], dtype=float)
    if (legal.shape != (n, 4) or not np.isin(legal_values, [0, 1]).all()
            or bounds.shape != (n, 4) or state.shape != (n, 2)
            or not np.isfinite(state).all() or np.any(state < 0) or np.any(bounds < 0)
            or not np.isfinite(bounds).all() or np.any(bounds != np.floor(bounds))):
        raise ValueError("Invalid registered trade-state fields")
    if np.any(~legal[np.arange(n), directions]):
        raise ValueError("Direction is not legal in the observed state")
    for direction, first in ((BUY, 0), (REDUCE, 2)):
        selected = bounds[directions == direction, first:first + 2]
        if np.any(selected[:, 0] < 1) or np.any(selected[:, 1] < selected[:, 0]):
            raise ValueError("Legal BUY/REDUCE needs a positive feasible lot interval")
    active = trade_size_active(observation, directions)
    if np.any(raw[n:][~active] != 0):
        raise ValueError("Inactive HOLD/EXIT/fixed-REDUCE size must be zero")
    top = float(observation["top_n"][0])
    if not top.is_integer() or not 0 <= top <= 500:
        raise ValueError("Invalid top_n condition")
    top = int(top)
    if stock_codes is not None and (len(stock_codes) != n or len(set(stock_codes)) != n):
        raise ValueError("Stock codes must match the unique trade pool")
    sizes = raw[n:].copy()
    clipped = np.clip(sizes, -bound, bound)
    fractions = 1 / (1 + np.exp(-clipped))
    quantity = state[:, 0] * lot_size
    shares = np.zeros(n)
    selected_lots = np.zeros(n, dtype=np.int64)
    for direction, first in ((BUY, 0), (REDUCE, 2)):
        indices = np.flatnonzero(directions == direction)
        low, high = bounds[indices, first], bounds[indices, first + 1]
        selected = np.minimum(high, low + np.floor(fractions[indices] * (high - low + 1)).astype(np.int64))
        selected_lots[indices] = selected
        shares[indices] = selected * lot_size * (1 if direction == BUY else -1)
    exits = directions == EXIT
    shares[exits] = -quantity[exits]
    proposed = shares.copy()
    # SB3 clips Box actions before stepping; inference must use the same priority.
    priority = np.where(directions == BUY, clipped, 0.)
    slots = max(0, top - int((quantity > 0).sum()) + int(exits.sum()))
    budget = max(0., float(observation["portfolio"][-1])
                 - float((shares[shares < 0] / lot_size * observation["execution_features"][shares < 0, 0]).sum()))
    buys = list(np.flatnonzero(directions == BUY))
    buys.sort(key=lambda i: (-priority[i], stock_codes[i] if stock_codes is not None else i))
    top_filtered, cash_filtered = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for i in buys:
        new = quantity[i] == 0
        if new and slots == 0:
            shares[i], top_filtered[i] = 0., True
            continue
        lot_value = float(observation["execution_features"][i, 0])
        affordable = max(0, int(np.floor((budget + 1e-8) / lot_value))) if lot_value > 0 else 0
        lots = min(selected_lots[i], affordable)
        if lots < bounds[i, 0]:
            shares[i], cash_filtered[i] = 0., True
            continue
        shares[i] = lots * lot_size
        budget = max(0., budget - lots * lot_value)
        if new:
            slots -= 1
    delta_weights = shares / lot_size * observation["execution_features"][:, 0]
    weights = np.maximum(0., np.asarray(observation["portfolio"][:-1], dtype=float) + delta_weights)
    weights[exits] = 0.
    cash = max(0., float(observation["portfolio"][-1]) - float(delta_weights.sum()))
    frozen = np.asarray(observation["execution_features"][:, 2], dtype=bool)
    return dict(directions=directions, size_latent=sizes, size_active=active, latent=raw.copy(),
                clipped_latent=np.concatenate([directions, clipped]), requested_lots=np.abs(proposed) / lot_size,
                selected_lots=selected_lots, proposed_shares=proposed, shares=shares, priority=priority,
                freeze_mask=frozen, exit_mask=exits, legal_directions=legal, lot_bounds=bounds.copy(),
                share_state=state.copy(), top_n=top, top_n_filtered=top_filtered, cash_filtered=cash_filtered,
                top_n_excess=max(0, int((quantity > 0).sum()) - top), weights=weights.astype(np.float32),
                cash_weight=cash, requested_cash_reference=cash,
                delta_weights=delta_weights.astype(np.float32),
                requested_notional_l1=float(np.abs(delta_weights).sum()),
                frozen_weight=float(np.asarray(observation["portfolio"][:-1])[frozen].sum()),
                locked_weights=np.maximum(0., np.asarray(observation["portfolio"][:-1])
                                           - observation["execution_features"][:, 1]).astype(np.float32),
                eligible=legal[:, 1:].any(axis=1), saturation_fraction=float(np.mean(np.abs(sizes) >= bound)))


class TradeActionSpace(gym.spaces.Box):
    def __init__(self, n, bound=8.):
        super().__init__(np.concatenate([np.zeros(n), np.full(n, -bound)]).astype(np.float32),
                         np.concatenate([np.full(n, 3.), np.full(n, bound)]).astype(np.float32), dtype=np.float32)
        self.n = n
        self.legal = np.zeros((n, 4), dtype=bool)
        self.legal[:, HOLD] = True
        self.bounds = np.zeros((n, 4), dtype=np.int64)

    def set_state(self, observation):
        self.legal = np.asarray(observation["legal_directions"], dtype=bool).copy()
        self.bounds = np.asarray(observation["trade_lot_bounds"], dtype=np.int64).copy()

    def sample(self, mask=None, probability=None):
        if mask is not None or probability is not None:
            raise ValueError("TradeActionSpace uses its observed legal state")
        directions = np.array([self.np_random.choice(np.flatnonzero(row)) for row in self.legal])
        active = (directions == BUY) | ((directions == REDUCE) & (self.bounds[:, 2] < self.bounds[:, 3]))
        sizes = np.zeros(self.n)
        sizes[active] = self.np_random.uniform(self.low[self.n:][active], self.high[self.n:][active])
        return np.concatenate([directions, sizes]).astype(np.float32)

    def contains(self, value):
        if not super().contains(value):
            return False
        value = np.asarray(value)
        directions = value[:self.n]
        return bool(np.all(directions == np.rint(directions))
                    and np.all(value[self.n:][(directions == HOLD) | (directions == EXIT)] == 0))


class TradeActionWrapper(gym.Wrapper):
    def __init__(self, env, bound=8., max_trade_weight=.05, max_commission_ratio=.001,
                 top_n=20, top_n_choices=None):
        super().__init__(env)
        self.bound, self.max_trade_weight, self.max_commission_ratio = bound, max_trade_weight, max_commission_ratio
        self.top_n, self.top_n_choices = top_n, tuple(top_n_choices or ())
        if any(type(k) is not int or not 0 <= k <= 500 for k in (top_n, *self.top_n_choices)):
            raise ValueError("Invalid trade top_n choices")
        self._top_rng = np.random.default_rng(0)
        n = env.market.n_stocks
        self.action_space = TradeActionSpace(n, bound)
        fields = dict(env.observation_space.spaces)
        fields.update(execution_features=gym.spaces.Box(0., np.inf, (n, 3), np.float32),
                      allocation_context=gym.spaces.Box(0., np.inf, (7,), np.float32),
                      top_n=gym.spaces.Box(0., 500., (1,), np.float32),
                      share_state=gym.spaces.Box(0., np.inf, (n, 2), np.float64),
                      trade_lot_bounds=gym.spaces.Box(0, np.iinfo(np.int64).max, (n, 4), np.int64),
                      legal_directions=gym.spaces.MultiBinary((n, 4)))
        self.observation_space = gym.spaces.Dict(fields)
        self._observation = None

    def enrich(self, observation):
        market = self.env.market
        following = market.dates[market.current_index + 1] if market.current_index + 1 < market.n_dates else None
        result = enrich_trade_observation(observation, self.env.portfolio,
            market.prices.iloc[market.current_index].to_numpy(), self.env.broker.config, self.top_n,
            self.max_trade_weight, self.max_commission_ratio, following)
        self.action_space.set_state(result)
        return result

    def reset(self, **kwargs):
        if kwargs.get("seed") is not None:
            self._top_rng = np.random.default_rng(int(kwargs["seed"]) ^ 0x7A43)
        observation, info = self.env.reset(**kwargs)
        if self.top_n_choices:
            self.top_n = int(self._top_rng.choice(self.top_n_choices))
        self._observation = self.enrich(observation)
        info["top_n"] = self.top_n
        return self._observation, info

    def step_order(self, order):
        if self._observation is None:
            raise RuntimeError("Call reset before stepping")
        if order.top_n != self.top_n:
            raise ValueError("Trade order top_n differs from the observed condition")
        portfolio, market = self.env.portfolio, self.env.market
        marks = market.prices.iloc[market.current_index].to_numpy()
        following = market.dates[market.current_index + 1] if market.current_index + 1 < market.n_dates else None
        snapshot = dict(date=market.current_date, codes=list(market.stock_codes), positions=portfolio.positions.copy(),
                        available_positions=portfolio.available_positions.copy(), cash=float(portfolio.cash),
                        receivables=float(portfolio.receivable_value), equity=float(portfolio.value(marks)), marks=marks.copy(),
                        projected_available_positions=_available(portfolio, following))
        observation, reward, terminated, truncated, info = self.env.step_shares(
            order.shares, freeze_mask=order.freeze_mask, exit_mask=order.exit_mask, top_n=order.top_n,
            buy_priority=order.priority, max_trade_weight=self.max_trade_weight, max_commission_ratio=self.max_commission_ratio)
        self._observation = self.enrich(observation)
        info.update(action_diagnostics=order.diagnostics, top_n=self.top_n, trade_action=True,
                    decision_snapshot=snapshot)
        return self._observation, reward, terminated, truncated, info

    def step(self, raw):
        if self._observation is None:
            raise RuntimeError("Call reset before stepping")
        result = map_trade_action(raw, self._observation, self.bound, self.env.broker.config.lot_size,
                                  self.env.market.stock_codes)
        return self.step_order(TradeOrder(result["shares"], result["freeze_mask"], result["exit_mask"],
                                         result["priority"], result["top_n"], result))

    def advance(self):
        observation, reward, terminated, truncated, info = self.env.advance()
        self._observation = self.enrich(observation)
        info["top_n"] = self.top_n
        return self._observation, reward, terminated, truncated, info
