"""Isolated 3N request mapping; stock quantity and BUY ranking are separate."""

import gymnasium as gym
import numpy as np

from .priority_policy import validate_actor_variant
from .trade_mapping import BUY, TradeActionSpace, TradeActionWrapper, TradeOrder, map_trade_action


PRIORITY_MAPPING_VERSION = "stock_priority_lots_residual_cash_coupled_decoupled_v1"
PRIORITY_ACTION_LAYOUT = "N_direction_codes_then_N_conditional_size_latents_then_N_buy_priority_latents"
PRIORITY_SEMANTICS = "BUY_quantity_copy_atom_coupled_or_independent_normal_decoupled_nonBUY_zero_atom"


def map_priority_trade_action(raw, observation, bound=8., lot_size=100,
                             stock_codes=None, actor_variant="coupled"):
    validate_actor_variant(actor_variant)
    n = len(observation["portfolio"]) - 1
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (3 * n,) or not np.isfinite(raw).all():
        raise ValueError("Expected finite packed 3N directions/quantities/priorities")
    # Reuse the registered quantity, support, odd-share and economic checks.
    result = map_trade_action(raw[:2 * n], observation, bound, lot_size, stock_codes)
    directions = result["directions"]
    priority_latent = raw[2 * n:].copy()
    expected = np.where(directions == BUY, raw[n:2 * n], 0.)
    if actor_variant == "coupled" and np.any(priority_latent != expected):
        raise ValueError("Coupled BUY priority must be the exact quantity copy atom")
    if np.any(priority_latent[directions != BUY] != 0):
        raise ValueError("Inactive non-BUY priority must be zero")
    priority = np.where(directions == BUY, np.clip(priority_latent, -bound, bound), 0.)
    shares = result["proposed_shares"].copy()
    quantity = np.asarray(observation["share_state"], dtype=float)[:, 0] * lot_size
    exits = result["exit_mask"]
    top = result["top_n"]
    slots = max(0, top - int((quantity > 0).sum()) + int(exits.sum()))
    lot_values = np.asarray(observation["execution_features"], dtype=float)[:, 0]
    budget = max(0., float(observation["portfolio"][-1])
                 - float((shares[shares < 0] / lot_size * lot_values[shares < 0]).sum()))
    buys = list(np.flatnonzero(directions == BUY))
    buys.sort(key=lambda i: (-priority[i], stock_codes[i] if stock_codes is not None else i))
    top_filtered, cash_filtered = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for i in buys:
        new = quantity[i] == 0
        if new and slots == 0:
            shares[i], top_filtered[i] = 0., True
            continue
        lot_value = lot_values[i]
        affordable = max(0, int(np.floor((budget + 1e-8) / lot_value))) if lot_value > 0 else 0
        lots = min(result["selected_lots"][i], affordable)
        if lots < result["lot_bounds"][i, 0]:
            shares[i], cash_filtered[i] = 0., True
            continue
        shares[i] = lots * lot_size
        budget = max(0., budget - lots * lot_value)
        if new:
            slots -= 1
    deltas = shares / lot_size * lot_values
    weights = np.maximum(0., np.asarray(observation["portfolio"][:-1], dtype=float) + deltas)
    weights[exits] = 0.
    cash = max(0., float(observation["portfolio"][-1]) - float(deltas.sum()))
    result.update(latent=raw.copy(), raw_packed_action=raw.copy(),
        clipped_latent=np.concatenate([directions, np.clip(raw[n:2 * n], -bound, bound), priority]),
        priority_latent=priority_latent, raw_priorities=priority_latent.copy(),
        priority_active=directions == BUY, priority=priority, actor_variant=actor_variant,
        shares=shares, top_n_filtered=top_filtered, cash_filtered=cash_filtered,
        weights=weights.astype(np.float32), cash_weight=cash, requested_cash_reference=cash,
        delta_weights=deltas.astype(np.float32), requested_notional_l1=float(np.abs(deltas).sum()),
        priority_saturation_fraction=float(np.mean(np.abs(priority_latent) >= bound)))
    return result


class PriorityTradeActionSpace(gym.spaces.Box):
    def __init__(self, n, bound=8., actor_variant="coupled"):
        validate_actor_variant(actor_variant)
        self.quantity_space = TradeActionSpace(n, bound)
        super().__init__(np.concatenate([self.quantity_space.low, np.full(n, -bound)]).astype(np.float32),
                         np.concatenate([self.quantity_space.high, np.full(n, bound)]).astype(np.float32),
                         dtype=np.float32)
        self.n, self.actor_variant = n, actor_variant

    def seed(self, seed=None):
        result = super().seed(seed)
        self.quantity_space.seed(seed)
        return result

    def set_state(self, observation):
        self.quantity_space.set_state(observation)

    def sample(self, mask=None, probability=None):
        if mask is not None or probability is not None:
            raise ValueError("PriorityTradeActionSpace uses its observed legal state")
        quantities = self.quantity_space.sample()
        priorities = (quantities[self.n:].copy() if self.actor_variant == "coupled"
                      else self.np_random.uniform(-self.high[-1], self.high[-1], self.n))
        priorities[quantities[:self.n] != BUY] = 0.
        return np.concatenate([quantities, priorities]).astype(np.float32)

    def contains(self, value):
        if not super().contains(value):
            return False
        value = np.asarray(value)
        if not self.quantity_space.contains(value[:2 * self.n]):
            return False
        expected = np.where(value[:self.n] == BUY, value[self.n:2 * self.n], 0.)
        return bool(np.all(value[2 * self.n:] == expected) if self.actor_variant == "coupled"
                    else np.all(value[2 * self.n:][value[:self.n] != BUY] == 0))


class PriorityTradeActionWrapper(TradeActionWrapper):
    def __init__(self, env, *, actor_variant="coupled", **kwargs):
        validate_actor_variant(actor_variant)
        self.actor_variant = actor_variant
        super().__init__(env, **kwargs)
        self.action_space = PriorityTradeActionSpace(env.market.n_stocks, self.bound, actor_variant)

    def step(self, raw):
        if self._observation is None:
            raise RuntimeError("Call reset before stepping")
        result = map_priority_trade_action(raw, self._observation, self.bound,
            self.env.broker.config.lot_size, self.env.market.stock_codes, self.actor_variant)
        return self.step_order(TradeOrder(result["shares"], result["freeze_mask"], result["exit_mask"],
                                         result["priority"], result["top_n"], result))
