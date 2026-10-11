"""Isolated ranking/quantity, slot/cash and frozen inventory mapping checks."""

import pickle

import numpy as np
import pandas as pd
import pytest

from src.agents.priority_mapping import PriorityTradeActionSpace, PriorityTradeActionWrapper, map_priority_trade_action
from src.agents.trade_mapping import BUY, EXIT, HOLD, REDUCE, TradeActionWrapper, map_trade_action
from src.envs.broker import BrokerConfig
from src.envs.trading import TradingEnv
from tests.test_delta_execution import account
from tests.test_trade_mapping import observed
from tests.test_trading_env import market


@pytest.mark.parametrize("directions,sizes", [([HOLD, HOLD], [0., 0.]), ([BUY, BUY], [1., 2.]),
    ([BUY, BUY], [100., 200.]), ([REDUCE, BUY], [.3, .2]), ([EXIT, BUY], [0., .4])])
def test_coupled_mapping_is_exact_legacy_positive_control(directions, sizes):
    obs = observed(account(200000, (1000, 0)), top_n=1)
    if directions == [BUY, BUY]:
        obs = observed(top_n=1)
    raw = np.array([*directions, *sizes])
    priorities = np.where(raw[:2] == BUY, raw[2:], 0.)
    before = map_trade_action(raw, obs, stock_codes=["a", "b"])
    after = map_priority_trade_action(np.r_[raw, priorities], obs, stock_codes=["a", "b"])
    for name in ["shares", "selected_lots", "proposed_shares", "priority", "weights",
                 "top_n_filtered", "cash_filtered", "delta_weights", "exit_mask", "freeze_mask"]:
        np.testing.assert_array_equal(before[name], after[name])
    assert before["cash_weight"] == after["cash_weight"]


def test_priority_changes_selection_without_changing_requested_quantities():
    obs = observed(top_n=1)
    first = map_priority_trade_action([BUY, BUY, 2., 1., -2., 3.], obs, actor_variant="decoupled")
    other = map_priority_trade_action([BUY, BUY, 2., 1., 3., -2.], obs, actor_variant="decoupled")
    np.testing.assert_array_equal(first["selected_lots"], other["selected_lots"])
    np.testing.assert_array_equal(first["proposed_shares"], other["proposed_shares"])
    assert first["shares"][0] == 0 and first["shares"][1] > 0
    assert other["shares"][0] > 0 and other["shares"][1] == 0


def test_quantity_changes_not_priority_when_budget_is_unconstrained():
    obs = observed(top_n=2)
    first = map_priority_trade_action([BUY, BUY, -2., -2., 1., -1.], obs, actor_variant="decoupled")
    other = map_priority_trade_action([BUY, BUY, 2., 2., 1., -1.], obs, actor_variant="decoupled")
    np.testing.assert_array_equal(first["priority"], other["priority"])
    assert (other["shares"] > first["shares"]).all()


def test_cash_filter_uses_priority_and_keeps_economic_minimum():
    obs = observed(account(200000, (19100, 0)), BrokerConfig(min_commission=5, max_weight=1.))
    result = map_priority_trade_action([BUY, BUY, 1., 0., -1., 1.], obs, actor_variant="decoupled")
    assert result["shares"][1] >= 500 and result["shares"][0] == 0
    assert result["cash_filtered"][0]
    assert result["cash_weight"] >= 0


def test_selected_exit_releases_slot_but_hold_does_not():
    obs = observed(account(200000, (20000, 0)), top_n=1)
    held = map_priority_trade_action([HOLD, BUY, 0., 0., 0., 2.], obs, actor_variant="decoupled")
    exited = map_priority_trade_action([EXIT, BUY, 0., 0., 0., 2.], obs, actor_variant="decoupled")
    assert held["shares"][1] == 0 and held["top_n_filtered"][1]
    assert exited["shares"][0] == -20000 and exited["shares"][1] > 0


@pytest.mark.parametrize("restriction", ["invalid", "sell", "available"])
def test_frozen_inventory_cannot_be_modified(restriction):
    portfolio = account(200000, (1000, 0))
    kwargs = {"valid": [0, 1]} if restriction == "invalid" else {"sell": [0, 1]} if restriction == "sell" else {}
    if restriction == "available":
        portfolio.available_positions[0] = 0
    obs = observed(portfolio, **kwargs)
    result = map_priority_trade_action([HOLD, HOLD, 0., 0., 0., 0.], obs)
    assert result["freeze_mask"][0] and result["shares"][0] == 0
    with pytest.raises(ValueError, match="not legal"):
        map_priority_trade_action([EXIT, HOLD, 0., 0., 0., 0.], obs)


@pytest.mark.parametrize("variant", ["coupled", "decoupled"])
def test_sb3_clipping_preserves_execution_but_not_original_latents(variant):
    obs = observed(top_n=1)
    raw = np.array([BUY, BUY, 100., 200., 100. if variant == "coupled" else -100., 200.])
    clipped = raw.copy()
    clipped[2:] = np.clip(clipped[2:], -8, 8)
    before = map_priority_trade_action(raw, obs, actor_variant=variant)
    after = map_priority_trade_action(clipped, obs, actor_variant=variant)
    np.testing.assert_array_equal(before["shares"], after["shares"])
    np.testing.assert_array_equal(before["priority"], after["priority"])
    np.testing.assert_array_equal(before["latent"], raw)


@pytest.mark.parametrize("raw,variant", [([BUY, HOLD, 1., 0., 2., 0.], "coupled"),
    ([BUY, HOLD, 1., 0., 2., 1.], "decoupled"), ([BUY, HOLD, 1., 0., float("nan"), 0.], "decoupled"),
    ([BUY, HOLD, 1., 0.], "coupled"), ([HOLD, HOLD, .1, 0., 0., 0.], "decoupled")])
def test_noncanonical_packed_actions_are_rejected(raw, variant):
    with pytest.raises(ValueError):
        map_priority_trade_action(raw, observed(), actor_variant=variant)


@pytest.mark.parametrize("variant", ["coupled", "decoupled"])
def test_action_space_seeded_sampling_and_pickle(variant):
    obs = observed(account(200000, (1000, 0)))
    left, right = PriorityTradeActionSpace(2, actor_variant=variant), PriorityTradeActionSpace(2, actor_variant=variant)
    for value in [left, right]:
        value.set_state(obs)
        value.seed(42)
    for _ in range(25):
        raw = left.sample()
        np.testing.assert_array_equal(raw, right.sample())
        assert left.contains(raw)
        map_priority_trade_action(raw, obs, actor_variant=variant)
    restored = pickle.loads(pickle.dumps(left))
    assert restored.actor_variant == variant and restored.shape == (6,)


def test_coupled_wrapper_matches_legacy_execution_and_fee_ledger():
    settings = dict(initial_capital=200000, broker_config=BrokerConfig(min_commission=5, max_weight=.5))
    old = TradeActionWrapper(TradingEnv(market(), **settings), top_n=1)
    new = PriorityTradeActionWrapper(TradingEnv(market(), **settings), top_n=1)
    try:
        old.reset(seed=42)
        obs, _ = new.reset(seed=42)
        assert new.observation_space.contains(obs)
        old_step = old.step(np.array([BUY, HOLD, 1., 0.], dtype=np.float32))
        new_step = new.step(np.array([BUY, HOLD, 1., 0., 1., 0.], dtype=np.float32))
        assert old_step[1:4] == new_step[1:4]
        for name in old_step[0]:
            np.testing.assert_array_equal(old_step[0][name], new_step[0][name])
        assert old_step[4]["orders"] == new_step[4]["orders"]
        assert new_step[4]["action_diagnostics"]["latent"].shape == (6,)
    finally:
        old.close()
        new.close()


@pytest.mark.parametrize("n", [1, 7, 300])
@pytest.mark.parametrize("top", [0, 1, 20])
def test_coupled_multi_pool_randomized_requests_match_legacy(n, top):
    rng = np.random.default_rng(42)
    quantity = np.where(np.arange(n) % 5 == 0, 1000., 0.)
    quantity[0] = 1000.
    lots = quantity / 100
    lot_value = np.full(n, .001)
    obs = dict(portfolio=np.r_[quantity * .00001, .05], valid_mask=np.ones(n),
        execution_features=np.c_[lot_value, quantity * .00001, np.zeros(n)],
        share_state=np.c_[lots, lots], trade_lot_bounds=np.tile([5, 20, 5, 9], (n, 1)),
        legal_directions=np.tile([1, int(top > 0), 0, 0], (n, 1)), top_n=np.array([top]))
    obs["legal_directions"][quantity > 0, 2:] = 1
    obs["execution_features"][::11, 2] = 1
    obs["legal_directions"][::11] = [1, 0, 0, 0]
    codes = [f"{i:06d}" for i in range(n)]
    for _ in range(12):
        directions = np.array([rng.choice(np.flatnonzero(row)) for row in obs["legal_directions"]])
        sizes = rng.normal(0., 10., n)
        sizes[(directions == HOLD) | (directions == EXIT)] = 0
        packed = np.r_[directions, sizes]
        before = map_trade_action(packed, obs, stock_codes=codes)
        after = map_priority_trade_action(np.r_[packed, np.where(directions == BUY, sizes, 0.)], obs, stock_codes=codes)
        for name in ["shares", "priority", "selected_lots", "top_n_filtered", "cash_filtered", "weights"]:
            np.testing.assert_array_equal(before[name], after[name])
        assert after["cash_weight"] == before["cash_weight"]


def test_equal_priorities_resolve_by_stock_code_not_quantity_or_input_order():
    obs = observed(top_n=1)
    raw = np.array([BUY, BUY, -2., 2., 0., 0.])
    original = map_priority_trade_action(raw, obs, stock_codes=["z", "a"], actor_variant="decoupled")
    assert original["shares"][0] == 0 and original["shares"][1] > 0
    other = {name: value.copy() for name, value in obs.items()}
    for name in ["stock_features", "valid_mask", "buy_mask", "sell_mask", "execution_features",
                 "share_state", "trade_lot_bounds", "legal_directions"]:
        other[name] = other[name][[1, 0]]
    other["portfolio"] = other["portfolio"][[1, 0, 2]]
    reordered = np.r_[raw[:2][[1, 0]], raw[2:4][[1, 0]], raw[4:][[1, 0]]]
    result = map_priority_trade_action(reordered, other, stock_codes=["a", "z"], actor_variant="decoupled")
    np.testing.assert_array_equal(result["shares"], original["shares"][[1, 0]])


def test_partial_t1_lock_allows_reduce_but_not_exit_or_slot_release():
    portfolio = account(200000, (1001, 0))
    portfolio.share_locks = [(pd.Timestamp("2024-01-05"), 0, 401)]
    portfolio.start_day("2024-01-03")
    obs = observed(portfolio, top_n=1)
    result = map_priority_trade_action([REDUCE, HOLD, .5, 0., 0., 0.], obs)
    assert result["shares"][0] == -600 and not result["exit_mask"][0]
    assert result["locked_weights"][0] > 0
    with pytest.raises(ValueError, match="not legal"):
        map_priority_trade_action([EXIT, HOLD, 0., 0., 0., 0.], obs)


def test_complete_odd_share_exit_does_not_acquire_priority_or_quantity():
    obs = observed(account(200000, (.4, 0)))
    result = map_priority_trade_action([EXIT, HOLD, 0., 0., 0., 0.], obs)
    assert result["shares"][0] == -.4 and not result["size_active"].any()
    assert not result["priority_active"].any()
