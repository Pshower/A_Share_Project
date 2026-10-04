"""Stock-only trade increments, with state constraints shared by rollout and inference."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd


DELTA_MAPPING_VERSION = "stock_delta_frozen_residual_cash_topn_v3"
EXECUTION_FIELDS = ("lot_value_ratio", "sellable_weight", "freeze_flag")
ALLOCATION_FIELDS = ("top_n_per_pool", "max_weight", "minimum_fee_ratio", "held_count_per_pool",
                     "locked_count_per_pool", "new_slot_slack_per_pool")


@dataclass
class DeltaOrder:
    deltas: np.ndarray
    freeze_mask: np.ndarray
    exit_mask: np.ndarray
    top_n: int
    diagnostics: dict


def enrich_delta_observation(observation, portfolio, prices, broker_config, top_n, execution_date=None):
    n = portfolio.n_stocks
    if type(top_n) is not int or not 0 <= top_n <= 500:
        raise ValueError("top_n must be an integer in [0, 500]")
    prices = np.asarray(prices, dtype=float)
    equity = portfolio.value(prices)
    if equity <= 0 or prices.shape != (n,):
        raise ValueError("Delta observations require positive equity and matching marks")
    known = np.isfinite(prices) & (prices > 0)
    marks = np.where(known, prices, 0.)
    # Ordinary T+1 buys release next trading day; longer corporate locks do not.
    if execution_date is not None and pd.Timestamp(execution_date) > portfolio.current_date:
        available = portfolio.positions.copy()
        for due, asset, quantity in portfolio.share_locks:
            if due > pd.Timestamp(execution_date):
                available[asset] -= quantity
    else:
        available = portfolio.available_positions.copy()
    available = np.maximum(0., np.minimum(available, portfolio.positions))
    held = portfolio.positions > 0
    frozen = held & (~np.asarray(observation["valid_mask"], dtype=bool)
                     | ~np.asarray(observation["sell_mask"], dtype=bool) | (available <= 0))
    auxiliary = np.column_stack([broker_config.lot_size * marks / equity, available * marks / equity, frozen])
    locked = held & ((available < portfolio.positions) | frozen)
    divisor = max(1, n)
    result = {name: value.copy() for name, value in observation.items()}
    result.update(execution_features=auxiliary.astype(np.float32),
                  allocation_context=np.array([top_n / divisor, broker_config.max_weight,
                                               broker_config.min_commission / equity,
                                               held.sum() / divisor, locked.sum() / divisor,
                                               max(0, top_n - held.sum()) / divisor], dtype=np.float32),
                  top_n=np.array([top_n], dtype=np.float32))
    return result


def active_delta_coordinates(observation):
    """Only state-known impossible actions are removed from the policy density."""
    current = np.asarray(observation["portfolio"][:-1], dtype=float)
    auxiliary = np.asarray(observation["execution_features"], dtype=float)
    valid = np.asarray(observation["valid_mask"], dtype=bool)
    free = auxiliary[:, 2] == 0
    cap = float(observation["allocation_context"][1])
    buy = (np.asarray(observation["buy_mask"], dtype=bool)
           & (cap - current >= auxiliary[:, 0] - 1e-8) & (auxiliary[:, 0] > 0)
           & (float(observation["top_n"][0]) > 0))
    sell = np.asarray(observation["sell_mask"], dtype=bool) & (auxiliary[:, 1] > 0)
    return valid & free & (buy | sell)


def map_delta_action(latent, observation, max_weight, max_delta_weight, bound=8.):
    current = np.asarray(observation["portfolio"][:-1], dtype=float)
    n = len(current)
    raw = np.asarray(latent, dtype=float)
    if raw.shape != (n,) or not np.isfinite(raw).all():
        raise ValueError("Expected finite N-stock delta latents")
    if not 0 < max_delta_weight <= 1 or not 0 < max_weight <= 1 or not np.isfinite(bound) or bound <= 0:
        raise ValueError("Invalid delta action limits")
    auxiliary = np.asarray(observation["execution_features"], dtype=float)
    if auxiliary.shape != (n, 3) or not np.isfinite(auxiliary).all():
        raise ValueError("Invalid execution features")
    top = float(observation["top_n"][0])
    if not top.is_integer() or not 0 <= top <= 500:
        raise ValueError("Invalid conditional top_n")
    top = int(top)
    active = active_delta_coordinates(observation)
    clipped = np.clip(raw, -bound, bound)
    proposed = max_delta_weight * np.tanh(clipped)
    valid = np.asarray(observation["valid_mask"], dtype=bool)
    frozen = auxiliary[:, 2].astype(bool)
    buys = valid & np.asarray(observation["buy_mask"], dtype=bool) & ~frozen & (top > 0)
    sells = valid & np.asarray(observation["sell_mask"], dtype=bool) & ~frozen
    deltas = np.where(proposed > 0, np.minimum(proposed, np.maximum(0., max_weight - current)),
                      np.maximum(proposed, -np.minimum(current, auxiliary[:, 1])))
    deltas[(deltas > 0) & ~buys] = 0
    deltas[(deltas < 0) & ~sells] = 0
    deltas[~active] = 0
    exit_mask = (proposed < 0) & (proposed <= -current) & (current > 0) & sells
    # Planned exits may fail or be partial: the broker rechecks actual free slots.
    clear = exit_mask & (auxiliary[:, 1] >= current - 1e-7)
    slots = max(0, top - int((current > 0).sum()) + int(clear.sum()))
    new = np.flatnonzero((current == 0) & (deltas > 0))
    ranked = new[np.argsort(-deltas[new], kind="stable")]
    deltas[ranked[slots:]] = 0
    budget = max(0., float(observation["portfolio"][-1]) - float(deltas[deltas < 0].sum()))
    requested_buys = float(deltas[deltas > 0].sum())
    if requested_buys > budget:
        deltas[deltas > 0] *= budget / requested_buys
    # Sub-lot buys at decision marks are not executable; future prices remain unknown.
    deltas[(deltas > 0) & (deltas < auxiliary[:, 0])] = 0
    weights = np.maximum(0., current + deltas)
    cash = max(0., float(observation["portfolio"][-1]) - float(deltas.sum()))
    return dict(weights=weights.astype(np.float32), cash_weight=cash,
                delta_weights=deltas.astype(np.float32), freeze_mask=frozen, exit_mask=exit_mask,
                top_n=top, frozen_weight=float(current[frozen].sum()),
                locked_weights=np.maximum(0., current - auxiliary[:, 1]).astype(np.float32),
                eligible=active, latent=raw.copy(), clipped_latent=clipped,
                saturation_fraction=float(np.mean(np.abs(raw) >= bound)),
                top_n_excess=max(0, int((current > 0).sum()) - top),
                target_delta_l1=float(np.abs(deltas).sum()))


class DeltaActionWrapper(gym.Wrapper):
    def __init__(self, env, bound=8., max_delta_weight=.05, top_n=20, top_n_choices=None):
        super().__init__(env)
        self.bound, self.max_delta_weight = float(bound), float(max_delta_weight)
        self.top_n, self.top_n_choices = int(top_n), tuple(top_n_choices or ())
        if any(type(k) is not int or not 0 <= k <= 500 for k in (top_n, *self.top_n_choices)):
            raise ValueError("Invalid top_n choices")
        self._top_rng = np.random.default_rng(0)
        n = env.market.n_stocks
        self.action_space = gym.spaces.Box(-self.bound, self.bound, (n,), np.float32)
        fields = dict(env.observation_space.spaces)
        fields.update(execution_features=gym.spaces.Box(0., np.inf, (n, 3), np.float32),
                      allocation_context=gym.spaces.Box(0., np.inf, (6,), np.float32),
                      top_n=gym.spaces.Box(0., 500., (1,), np.float32))
        self.observation_space = gym.spaces.Dict(fields)
        self._observation = None

    def enrich(self, observation):
        market = self.env.market
        following = market.dates[market.current_index + 1] if market.current_index + 1 < market.n_dates else None
        return enrich_delta_observation(observation, self.env.portfolio,
            market.prices.iloc[market.current_index].to_numpy(), self.env.broker.config, self.top_n, following)

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self._top_rng = np.random.default_rng(int(seed) ^ 0x5D31)
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
            raise ValueError("Delta order top_n differs from the observed condition")
        observation, reward, terminated, truncated, info = self.env.step_deltas(
            order.deltas, freeze_mask=order.freeze_mask, exit_mask=order.exit_mask, top_n=order.top_n)
        self._observation = self.enrich(observation)
        info.update(action_diagnostics=order.diagnostics, top_n=self.top_n,
                    delta_action=True)
        return self._observation, reward, terminated, truncated, info

    def step(self, latent):
        if self._observation is None:
            raise RuntimeError("Call reset before stepping")
        result = map_delta_action(latent, self._observation, self.env.broker.config.max_weight,
                                  self.max_delta_weight, self.bound)
        return self.step_order(DeltaOrder(result["delta_weights"], result["freeze_mask"],
                                         result["exit_mask"], self.top_n, result))

    def advance(self):
        observation, reward, terminated, truncated, info = self.env.advance()
        self._observation = self.enrich(observation)
        info["top_n"] = self.top_n
        return self._observation, reward, terminated, truncated, info
