"""Whole-lot trade execution checks economics after real execution constraints."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.envs.broker import BrokerConfig, BrokerSimulator
from src.envs.trading import TradingEnv
from tests.test_delta_execution import account
from tests.test_trading_env import market, panels


def execute(portfolio, shares, config=None, prices=(10., 10.), **kwargs):
    broker = BrokerSimulator(config or BrokerConfig(min_commission=5))
    orders = broker.trade_shares(portfolio, shares, prices, prices, "2024-01-03", **kwargs)
    return broker, orders


def test_hold_preserves_over_cap_shares_despite_price_drift():
    portfolio = account(200000, (15000, 0))
    broker, orders = execute(portfolio, [0, 0], BrokerConfig(max_weight=.05), prices=(20., 10.), top_n=0)
    assert orders == broker.orders == []
    assert portfolio.cash == 50000 and portfolio.positions[0] == 15000


def test_reduce_retains_a_share_and_uses_whole_lots():
    portfolio = account(200000, (1000, 0))
    _, orders = execute(portfolio, [-1000, 0])
    assert orders[0]["filled_shares"] == -900
    assert orders[0]["position_after"] == 100
    assert orders[0]["trade_direction"] == 2
    assert orders[0]["exit_completed"] is False


def test_full_exit_bypasses_small_notional_lot_and_five_percent_limits():
    portfolio = account(200000, (155, 0))
    _, orders = execute(portfolio, [-1, 0], prices=(7., 10.), exit_mask=[True, False])
    assert orders[0]["requested_shares"] == orders[0]["filled_shares"] == -155
    assert orders[0]["exit_completed"] is True
    assert orders[0]["trade_direction"] == 3


def test_exit_placeholder_preserves_true_small_holding_without_rounding():
    portfolio = account(200000, (.4, 0))
    _, orders = execute(portfolio, [-.4, 0], exit_mask=[True, False])
    assert orders[0]["requested_shares"] == -.4
    assert orders[0]["filled_shares"] == 0
    assert orders[0]["exit_completed"] is False
    assert orders[0]["reasons"] == ["unsupported_fractional_position"]
    assert portfolio.positions[0] == .4


def test_full_exit_can_exceed_five_percent():
    portfolio = account(200000, (15000, 0))
    _, orders = execute(portfolio, [-1, 0], exit_mask=[True, False])
    assert orders[0]["filled_shares"] == -15000
    assert orders[0]["exit_completed"] is True


@pytest.mark.parametrize("constraint", ["available", "volume"])
def test_partial_exit_has_no_small_notional_exception_or_slot_release(constraint):
    portfolio = account(200000, (1000, 0))
    config = BrokerConfig(min_commission=5, volume_fraction=.1 if constraint == "volume" else None)
    kwargs = {"volume": [3500, 10000]} if constraint == "volume" else {}
    if constraint == "available":
        portfolio.share_locks = [(pd.Timestamp("2024-01-05"), 0, 650)]
    _, orders = execute(portfolio, [-1, 500], config, exit_mask=[True, False], top_n=1, **kwargs)
    assert orders[0]["filled_shares"] == 0
    assert "minimum_trade_notional" in orders[0]["reasons"]
    assert orders[0]["exit_completed"] is False
    assert orders[1]["filled_shares"] == 0 and "top_n_limit" in orders[1]["reasons"]
    np.testing.assert_array_equal(portfolio.positions, [1000, 0])


def test_partial_exit_can_execute_economic_lots_but_keeps_name():
    portfolio = account(200000, (1500, 0))
    portfolio.share_locks = [(pd.Timestamp("2024-01-05"), 0, 900)]
    _, orders = execute(portfolio, [-1, 0], exit_mask=[True, False])
    assert orders[0]["filled_shares"] == -600
    assert orders[0]["exit_completed"] is False
    assert portfolio.positions[0] == 900


def test_sell_side_slippage_is_used_for_minimum_notional():
    portfolio = account(200000, (2000, 0))
    _, orders = execute(portfolio, [-500, 0], BrokerConfig(min_commission=5, slippage=.001))
    assert orders[0]["price"] == pytest.approx(9.99)
    assert orders[0]["filled_shares"] == 0
    assert "minimum_trade_notional" in orders[0]["reasons"]


def test_buy_side_slippage_is_used_for_five_percent_maximum():
    portfolio = account(200000)
    _, orders = execute(portfolio, [1000, 0], BrokerConfig(min_commission=5, slippage=.001))
    assert orders[0]["filled_shares"] == 900
    assert "trade_weight_limit" in orders[0]["reasons"]
    assert abs(orders[0]["filled_shares"]) * orders[0]["price"] <= .05 * orders[0]["opening_equity"]


def test_cash_clipping_does_not_create_a_subminimum_buy():
    portfolio = account(200000, (19600, 0))
    _, orders = execute(portfolio, [0, 500])
    assert orders[0]["filled_shares"] == 0
    assert "insufficient_cash" in orders[0]["reasons"]
    assert "minimum_trade_notional" in orders[0]["reasons"]
    assert portfolio.cash == 4000


def test_raw_priority_decides_funding_and_top_n_admission():
    portfolio = account(200000)
    _, orders = execute(portfolio, [500, 500], buy_priority=[-1, 1], top_n=1)
    assert [row["code"] for row in orders] == ["600000", "000001"]
    assert [row["filled_shares"] for row in orders] == [500, 0]
    assert orders[1]["reasons"] == ["top_n_limit"]


def test_sell_funds_later_buy_and_only_actual_exit_releases_name():
    portfolio = account(200000, (20000, 0))
    _, orders = execute(portfolio, [-1, 1000], exit_mask=[True, False], top_n=1)
    assert orders[0]["exit_completed"] is True
    assert orders[1]["filled_shares"] == 1000
    assert portfolio.cash >= 0


def test_tiny_full_exit_cannot_overdraw_cash_or_release_slot():
    portfolio = account(10, (1, 0))
    portfolio.cash = 0
    _, orders = execute(portfolio, [-1, 100], prices=(1., 1.), exit_mask=[True, False], top_n=1)
    assert orders[0]["filled_shares"] == 0
    assert orders[0]["reasons"] == ["insufficient_cash_for_sell_fee"]
    assert orders[0]["fee"] == 0 and portfolio.total_fees == 0
    assert orders[1]["filled_shares"] == 0 and "top_n_limit" in orders[1]["reasons"]
    assert portfolio.cash == 0 and portfolio.positions[0] == 1


@pytest.mark.parametrize("quantity", [-600, 600])
def test_held_execution_sell_mask_freezes_both_directions(quantity):
    portfolio = account(200000, (1000, 0))
    _, orders = execute(portfolio, [quantity, 0], sell_mask=[False, True])
    assert orders[0]["reasons"] == ["frozen_holdings"]
    assert portfolio.positions[0] == 1000 and portfolio.total_fees == 0


@pytest.mark.parametrize("shares,options", [
    ([1.5, 0], {}), ([np.nan, 0], {}), ([1, 0], {}), ([-1, 0], {}),
    ([0, 0], {"exit_mask": [True, False]}),
    ([100, 0], {"buy_priority": [np.nan, 0]}),
    ([100, 0], {"max_commission_ratio": .0001}),
    ([100, 0], {"max_trade_weight": 0}),
])
def test_invalid_share_intent_is_atomic(shares, options):
    portfolio = account(200000)
    before = deepcopy(portfolio.__dict__)
    broker = BrokerSimulator(BrokerConfig(min_commission=5))
    with pytest.raises(ValueError):
        broker.trade_shares(portfolio, shares, [10, 10], [10, 10], "2024-01-03", **options)
    assert portfolio.current_date == before["current_date"]
    assert portfolio.cash == before["cash"] and broker.orders == []


def test_environment_exit_reloads_quantity_after_corporate_action():
    features, prices = panels()
    prices.loc[prices.index[1]:, ("000001", "open")] = 5.
    prices.loc[prices.index[1]:, ("000001", "close")] = 5.
    prices.loc[prices.index[1]:, ("000001", "limit_up")] = 5.5
    prices.loc[prices.index[1]:, ("000001", "limit_down")] = 4.5
    events = pd.DataFrame([dict(date=features.index[1], code="000001", cash_per_share=0.,
        share_multiplier=2., payment_date=features.index[1], share_available_date=features.index[1])])
    env = TradingEnv(market(features, prices, corporate_actions=events), initial_capital=200000,
                     broker_config=BrokerConfig(min_commission=5))
    env.reset()
    env.portfolio.positions[0] = env.portfolio.available_positions[0] = 155
    env.portfolio.cash = 198450
    _, _, _, _, info = env.step_shares([-155, 0], exit_mask=[True, False])
    assert info["orders"][0]["requested_shares"] == -310
    assert info["orders"][0]["filled_shares"] == -310
    assert info["opening_snapshot"]["positions"][0] == 310
    assert info["positions"][0] == 0


def test_environment_invalid_held_features_freeze_exit_after_events():
    features, prices = panels()
    features.loc[features.index[0], ("000001", "signal")] = np.nan
    env = TradingEnv(market(features, prices), initial_capital=200000,
                     broker_config=BrokerConfig(min_commission=5))
    env.reset()
    env.portfolio.positions[0] = env.portfolio.available_positions[0] = 500
    env.portfolio.cash = 195000
    _, _, _, _, info = env.step_shares([-500, 0], exit_mask=[True, False])
    assert info["positions"][0] == 500
    assert "invalid_decision_features" in info["orders"][0]["reasons"]
