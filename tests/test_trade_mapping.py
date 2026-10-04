"""Trade masks and direct lot requests are derived from the exact account book."""

import pickle

import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.env_checker import check_env

from src.agents.trade_mapping import (
    BUY, EXIT, HOLD, REDUCE, TRADE_ALLOCATION_FIELDS, TradeActionSpace,
    TradeActionWrapper, enrich_trade_observation, map_trade_action, trade_size_active,
)
from src.envs.broker import BrokerConfig
from src.envs.trading import TradingEnv
from tests.test_delta_execution import account
from tests.test_delta_policy import observation
from tests.test_trading_env import market, panels


def observed(portfolio=None, config=None, prices=(10., 10.), top_n=2, **kwargs):
    portfolio = portfolio or account(200000)
    config = config or BrokerConfig(min_commission=5, max_weight=.5)
    weights = portfolio.get_weights(prices)
    base = observation(weights, cash=portfolio.get_cash_ratio(prices), top_n=top_n, **kwargs)
    return enrich_trade_observation(base, portfolio, prices, config, top_n)


def test_raw_book_bounds_use_side_slippage_and_keep_odd_share_state():
    portfolio = account(200000, (1001, 0))
    obs = observed(portfolio, BrokerConfig(min_commission=5, slippage=.001, max_weight=.5))
    np.testing.assert_array_equal(obs["share_state"][0], [10.01, 10.01])
    np.testing.assert_array_equal(obs["trade_lot_bounds"][0], [5, 9, 6, 10])
    assert obs["legal_directions"][0].tolist() == [1, 1, 1, 1]
    assert len(TRADE_ALLOCATION_FIELDS) == 7
    assert obs["allocation_context"][-1] == pytest.approx(.025)


def test_hold_is_exact_zero_and_exit_uses_raw_odd_inventory():
    obs = observed(account(200000, (155, 0)))
    held = map_trade_action([HOLD, HOLD, 0, 0], obs)
    np.testing.assert_array_equal(held["shares"], [0, 0])
    exited = map_trade_action([EXIT, HOLD, 0, 0], obs)
    np.testing.assert_array_equal(exited["shares"], [-155, 0])
    assert not exited["size_active"].any()


def test_exit_does_not_round_a_small_true_holding_to_zero():
    obs = observed(account(200000, (.4, 0)))
    mapped = map_trade_action([EXIT, HOLD, 0, 0], obs)
    assert mapped["shares"][0] == -.4
    assert mapped["exit_mask"][0]


def test_zero_buy_size_is_economic_midpoint_not_zero_trade():
    obs = observed()
    mapped = map_trade_action([BUY, HOLD, 0, 0], obs)
    assert mapped["shares"][0] == 800
    assert mapped["shares"][0] * 10 >= 5000
    assert mapped["size_active"].tolist() == [True, False]


def test_fixed_buy_priority_active_fixed_reduce_canonical_zero():
    portfolio = account(100000, (600, 0))
    obs = observed(portfolio)
    assert obs["trade_lot_bounds"][0, 2:].tolist() == [5, 5]
    assert obs["trade_lot_bounds"][1, :2].tolist() == [5, 5]
    assert trade_size_active(obs, [REDUCE, BUY]).tolist() == [False, True]
    mapped = map_trade_action([REDUCE, BUY, 0, 1.2], obs)
    np.testing.assert_array_equal(mapped["shares"], [-500, 500])
    assert mapped["priority"][1] == 1.2
    with pytest.raises(ValueError, match="Inactive"):
        map_trade_action([REDUCE, BUY, .1, 1.2], obs)


def test_partial_lock_masks_exit_without_float_weight_comparison():
    portfolio = account(200000, (1001, 0))
    portfolio.share_locks = [(pd.Timestamp("2024-01-05"), 0, 401)]
    portfolio.start_day("2024-01-03")
    obs = observed(portfolio, top_n=1)
    assert obs["share_state"][0].tolist() == [10.01, 6.]
    assert obs["legal_directions"][0, EXIT] == 0
    assert obs["legal_directions"][0, REDUCE] == 1
    assert obs["legal_directions"][1, BUY] == 0
    with pytest.raises(ValueError, match="not legal"):
        map_trade_action([EXIT, HOLD, 0, 0], obs)


@pytest.mark.parametrize("restriction", ["invalid", "sell", "available"])
def test_frozen_held_name_has_only_hold(restriction):
    portfolio = account(200000, (1000, 0))
    kwargs = {"valid": [0, 1]} if restriction == "invalid" else {"sell": [0, 1]} if restriction == "sell" else {}
    if restriction == "available":
        portfolio.available_positions[0] = 0
    obs = observed(portfolio, **kwargs)
    assert obs["legal_directions"][0].tolist() == [1, 0, 0, 0]
    assert obs["execution_features"][0, 2] == 1


def test_buy_blocked_can_reduce_or_exit_without_forced_trade():
    obs = observed(account(200000, (1000, 0)), buy=[0, 1])
    assert obs["legal_directions"][0].tolist() == [1, 0, 1, 1]
    assert map_trade_action([HOLD, HOLD, 0, 0], obs)["shares"][0] == 0


def test_potential_exit_funds_mask_and_slot_but_only_selected_exit_releases():
    portfolio = account(200000, (20000, 0))
    obs = observed(portfolio, top_n=1)
    assert portfolio.cash == 0 and obs["legal_directions"][1, BUY] == 1
    hold = map_trade_action([HOLD, BUY, 0, 0], obs)
    assert hold["shares"][1] == 0 and hold["top_n_filtered"][1]
    exit_buy = map_trade_action([EXIT, BUY, 0, 0], obs)
    assert exit_buy["shares"][0] == -20000 and exit_buy["shares"][1] > 0


def test_ranking_is_size_priority_not_high_price_or_notional():
    obs = observed(prices=(10., 20.), top_n=1)
    mapped = map_trade_action([BUY, BUY, 2, 1], obs, stock_codes=["a", "b"])
    assert mapped["shares"][0] > 0 and mapped["shares"][1] == 0
    assert mapped["top_n_filtered"][1]


def test_cash_projection_never_scales_below_economic_minimum():
    obs = observed(account(200000, (19100, 0)), BrokerConfig(min_commission=5, max_weight=1.))
    mapped = map_trade_action([BUY, BUY, 1, 0], obs)
    assert mapped["shares"][0] == 800
    assert mapped["shares"][1] == 0 and mapped["cash_filtered"][1]
    assert mapped["proposed_shares"][1] >= 500


def test_extreme_mean_and_sb3_clipped_action_have_identical_allocation():
    obs = observed(top_n=1)
    raw = np.array([BUY, BUY, 100, 200.])
    bounded = np.clip(raw, [0, 0, -8, -8], [3, 3, 8, 8])
    first, second = map_trade_action(raw, obs, stock_codes=["a", "b"]), map_trade_action(bounded, obs, stock_codes=["a", "b"])
    np.testing.assert_array_equal(first["shares"], second["shares"])
    np.testing.assert_array_equal(first["priority"], second["priority"])


@pytest.mark.parametrize("raw", [[.5, 0, 0, 0], [0, 0, .1, 0], [0, 0, np.nan, 0], [0, 0]])
def test_noncanonical_actions_fail_closed(raw):
    with pytest.raises(ValueError):
        map_trade_action(raw, observed())


@pytest.mark.parametrize("corruption", ["nonboolean", "empty_interval"])
def test_corrupt_registered_mask_or_interval_fails_closed(corruption):
    obs = observed()
    if corruption == "nonboolean":
        obs["legal_directions"][0, BUY] = 2
    else:
        obs["trade_lot_bounds"][0, :2] = [5, 4]
    with pytest.raises(ValueError):
        map_trade_action([BUY, HOLD, 0, 0], obs)


def test_action_space_seeded_samples_are_legal_and_do_not_capture_env():
    obs = observed(account(200000, (1000, 0)))
    left, right = TradeActionSpace(2), TradeActionSpace(2)
    left.set_state(obs)
    right.set_state(obs)
    left.seed(42)
    right.seed(42)
    for _ in range(50):
        raw = left.sample()
        np.testing.assert_array_equal(raw, right.sample())
        assert left.contains(raw)
        map_trade_action(raw, obs)
    assert not left.contains(np.array([.5, 0, 0, 0], dtype=np.float32))
    assert not left.contains(np.array([0, 0, 1, 0], dtype=np.float32))
    restored = pickle.loads(pickle.dumps(left))
    assert restored.__dict__.keys() == left.__dict__.keys()
    assert all(not isinstance(value, TradingEnv) for value in left.__dict__.values())


def test_wrapper_check_env_and_exact_decision_opening_snapshots():
    env = TradeActionWrapper(TradingEnv(market(), initial_capital=200000,
        broker_config=BrokerConfig(min_commission=5, max_weight=.5)), top_n=1)
    try:
        check_env(env, warn=True)
        obs, info = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        _, _, _, _, info = env.step(np.array([BUY, HOLD, 0, 0], dtype=np.float32))
        assert info["decision_snapshot"]["cash"] == 200000
        assert info["opening_snapshot"]["cash"] == 200000
        assert info["orders"][0]["filled_shares"] > 0
        assert info["action_diagnostics"]["directions"].tolist() == [BUY, HOLD]
    finally:
        env.close()


def test_wrapper_sampling_and_top_n_rng_do_not_consume_account_rng():
    features, prices = panels()
    settings = dict(initial_capital=200000, broker_config=BrokerConfig(min_commission=5),
                    initial_portfolio="mixed", random_start=True, max_steps=2)
    baseline = TradingEnv(market(features, prices), **settings)
    wrapped = TradeActionWrapper(TradingEnv(market(features, prices), **settings), top_n_choices=[1, 2])
    for i in range(8):
        _, old = baseline.reset(seed=42 if i == 0 else None)
        _, new = wrapped.reset(seed=42 if i == 0 else None)
        for _ in range(20):
            wrapped.action_space.sample()
        assert old["date"] == new["date"] and old["cash"] == new["cash"]
        np.testing.assert_array_equal(old["positions"], new["positions"])
        assert baseline.np_random.bit_generator.state == wrapped.unwrapped.np_random.bit_generator.state
    baseline.close()
    wrapped.close()
