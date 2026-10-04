"""Signed execution requests preserve holdings, constraints and legacy rebalances."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.envs.broker import BrokerConfig, BrokerSimulator
from src.envs.portfolio import Portfolio
from src.envs.trading import TradingEnv
from tests.test_trading_env import market, panels


def account(capital=10000, positions=(0, 0), price=10.0):
    portfolio = Portfolio(capital, ["000001", "600000"])
    portfolio.start_day("2024-01-02")
    portfolio.positions[:] = positions
    portfolio.available_positions[:] = positions
    portfolio.cost_basis[:] = price
    portfolio.cash = capital - sum(positions) * price
    return portfolio


def execute(broker, portfolio, deltas, date="2024-01-02", prices=(10.0, 10.0), **kwargs):
    return broker.trade_deltas(portfolio, deltas, prices, prices, date, **kwargs)


def test_zero_delta_is_no_order_despite_opening_price_drift_and_existing_over_cap():
    portfolio = account(positions=(500, 0))
    broker = BrokerSimulator(BrokerConfig(max_weight=0.2))
    original_cash = portfolio.cash

    orders = execute(broker, portfolio, [0, 0], prices=(20.0, 10.0), top_n=1)

    assert orders == broker.orders == []
    np.testing.assert_array_equal(portfolio.positions, [500, 0])
    assert portfolio.cash == original_cash and portfolio.total_fees == 0
    assert portfolio.get_weights([20, 10])[0] > broker.config.max_weight


def test_signed_trade_uses_opening_nav_and_rounds_buys_to_lots_but_sells_to_shares():
    portfolio = account(positions=(500, 0))
    broker = BrokerSimulator(BrokerConfig(commission=0))
    # Opening NAV is 500 * 20 + 5000 = 15000, not yesterday's 10000.
    orders = execute(broker, portfolio, [-0.1, 0.1], prices=(20.0, 10.0))

    assert [order["side"] for order in orders] == ["sell", "buy"]
    assert [order["requested_shares"] for order in orders] == [-75, 150]
    assert [order["filled_shares"] for order in orders] == [-75, 100]
    assert "buy_lot" in orders[1]["reasons"]
    assert all(order["opening_equity"] == 15000 for order in orders)
    np.testing.assert_array_equal(portfolio.positions, [425, 100])
    assert portfolio.cash == 5500


def test_sells_first_fund_buys_and_costs_cannot_overdraw_cash():
    portfolio = account(positions=(1000, 0))
    broker = BrokerSimulator(BrokerConfig(commission=0.001, min_commission=5))

    orders = execute(broker, portfolio, [-1, 1], exit_mask=[True, False], top_n=1)

    assert [order["filled_shares"] for order in orders] == [-1000, 900]
    assert orders[0]["exit_requested"] is True
    assert "insufficient_cash" in orders[1]["reasons"]
    assert portfolio.cash == pytest.approx(981)
    assert portfolio.total_fees == pytest.approx(19)
    assert portfolio.value([10, 10]) == pytest.approx(9981)


@pytest.mark.parametrize("delta", [-0.1, 0.1])
def test_explicit_freeze_prevents_buy_and_sell_by_share_count(delta):
    portfolio = account(positions=(500, 0))
    broker = BrokerSimulator(BrokerConfig(commission=0))

    orders = execute(broker, portfolio, [delta, 0], prices=(20.0, 10.0), freeze_mask=[True, False])

    np.testing.assert_array_equal(portfolio.positions, [500, 0])
    assert portfolio.cash == 5000
    assert orders[0]["filled_shares"] == 0
    assert orders[0]["reasons"] == ["frozen_holdings"]


@pytest.mark.parametrize("delta", [-0.1, 0.1])
def test_execution_sell_restriction_freezes_an_existing_position_in_both_directions(delta):
    portfolio = account(positions=(500, 0))
    broker = BrokerSimulator(BrokerConfig(commission=0))
    orders = execute(broker, portfolio, [delta, 0], buy_mask=[True, True], sell_mask=[False, True],
                     reasons=[["limit_down_or_unknown"], []])
    assert orders[0]["filled_shares"] == 0
    assert "frozen_holdings" in orders[0]["reasons"]
    assert "limit_down_or_unknown" in orders[0]["reasons"]
    np.testing.assert_array_equal(portfolio.positions, [500, 0])


def test_buy_restriction_does_not_force_a_sell_and_still_allows_reduction():
    portfolio = account(positions=(500, 0))
    broker = BrokerSimulator(BrokerConfig(commission=0))
    masks = dict(buy_mask=[False, True], sell_mask=[True, True])
    assert execute(broker, portfolio, [0, 0], **masks) == []
    assert portfolio.positions[0] == 500
    rejected = execute(broker, portfolio, [0.1, 0], **masks)
    assert rejected[0]["reasons"] == ["buy_mask"] and portfolio.positions[0] == 500
    sold = execute(broker, portfolio, [-0.1, 0], **masks)
    assert sold[0]["filled_shares"] == -100 and portfolio.positions[0] == 400


def test_partial_locked_holdings_are_not_sold_or_counted_as_a_freed_name():
    portfolio = account(positions=(155, 0))
    portfolio.share_locks = [(pd.Timestamp("2024-01-04"), 0, 55)]
    broker = BrokerSimulator(BrokerConfig(commission=0))

    orders = execute(broker, portfolio, [-0.5, 0.1], date="2024-01-03",
                     exit_mask=[True, False], top_n=1)

    assert orders[0]["requested_shares"] == -155
    assert orders[0]["filled_shares"] == -100
    assert "t_plus_one" in orders[0]["reasons"]
    assert orders[1]["filled_shares"] == 0 and "top_n_limit" in orders[1]["reasons"]
    np.testing.assert_array_equal(portfolio.positions, [55, 0])


def test_new_name_can_use_a_slot_only_after_actual_full_exit_even_after_a_price_gap():
    portfolio = account(positions=(155, 0))
    broker = BrokerSimulator(BrokerConfig(commission=0))

    orders = execute(broker, portfolio, [-0.001, 0.2], prices=(7.0, 10.0),
                     exit_mask=[True, False], top_n=1)

    assert orders[0]["requested_shares"] == orders[0]["filled_shares"] == -155
    assert orders[1]["filled_shares"] == 100
    assert "top_n_limit" not in orders[1]["reasons"]
    assert portfolio.positions[0] == 0
    assert portfolio.positions[1] == 100 and portfolio.cash == 8535


def test_partial_volume_exit_does_not_release_a_top_n_slot_and_volume_budget_is_shared():
    portfolio = account(positions=(500, 0))
    broker = BrokerSimulator(BrokerConfig(commission=0, volume_fraction=0.1))
    orders = execute(broker, portfolio, [-0.5, 0.2], exit_mask=[True, False], top_n=1, volume=[1000, 1000])
    assert orders[0]["filled_shares"] == -100 and "volume_limit" in orders[0]["reasons"]
    assert orders[1]["filled_shares"] == 0 and "top_n_limit" in orders[1]["reasons"]
    repeated = execute(broker, portfolio, [-0.5, 0], exit_mask=[True, False], volume=[1000, 1000])
    assert repeated[0]["filled_shares"] == 0 and "volume_limit" in repeated[0]["reasons"]
    assert portfolio.positions[0] == 400


def test_existing_names_above_top_n_are_grandfathered_without_implicit_liquidation():
    portfolio = account(positions=(100, 100))
    broker = BrokerSimulator(BrokerConfig(commission=0))
    assert execute(broker, portfolio, [0, 0], top_n=1) == []
    np.testing.assert_array_equal(portfolio.positions, [100, 100])
    orders = execute(broker, portfolio, [0.1, 0], top_n=1)
    assert orders[0]["filled_shares"] == 100
    np.testing.assert_array_equal(portfolio.positions, [200, 100])


def test_zero_top_n_keeps_zero_delta_holdings_blocks_new_names_and_allows_exits():
    portfolio = account(positions=(100, 0))
    broker = BrokerSimulator(BrokerConfig(commission=0))
    assert execute(broker, portfolio, [0, 0], top_n=0) == []
    assert portfolio.positions[0] == 100
    rejected = execute(broker, portfolio, [0, 0.1], top_n=0)
    assert rejected[0]["filled_shares"] == 0 and "top_n_limit" in rejected[0]["reasons"]
    sold = execute(broker, portfolio, [-0.001, 0], top_n=0, exit_mask=[True, False])
    assert sold[0]["filled_shares"] == -100
    np.testing.assert_array_equal(portfolio.positions, [0, 0])
    rejected = execute(broker, portfolio, [0.1, 0], top_n=0)
    assert rejected[0]["filled_shares"] == 0 and "top_n_limit" in rejected[0]["reasons"]
    assert portfolio.cash == 10000


def test_positive_deltas_respect_opening_position_cap_without_selling_existing_excess():
    portfolio = account()
    broker = BrokerSimulator(BrokerConfig(commission=0, max_weight=0.2))
    orders = execute(broker, portfolio, [0.5, 0])
    assert orders[0]["filled_shares"] == 200 and "position_cap" in orders[0]["reasons"]
    assert portfolio.cash == 8000
    held = account(positions=(500, 0))
    orders = execute(broker, held, [0.1, 0])
    assert orders[0]["filled_shares"] == 0 and "position_cap" in orders[0]["reasons"]
    assert execute(broker, held, [0, 0]) == []
    sold = execute(broker, held, [-0.1, 0])
    assert sold[0]["filled_shares"] == -100 and held.positions[0] == 400


@pytest.mark.parametrize("delta", [-0.1, 0.1])
def test_missing_open_rejects_nonzero_requests_and_preserves_signed_delta_diagnostics(delta):
    portfolio = account(positions=(500, 0))
    broker = BrokerSimulator(BrokerConfig(commission=0))
    orders = broker.trade_deltas(portfolio, [delta, 0], [np.nan, 10], [10, 10], "2024-01-02")
    assert orders[0]["status"] == "rejected"
    assert orders[0]["reasons"] == ["missing_open"]
    assert orders[0]["requested_delta_weight"] == delta
    assert orders[0]["side"] == ("sell" if delta < 0 else "buy")
    assert portfolio.positions[0] == 500 and portfolio.cash == 5000


@pytest.mark.parametrize("deltas,options", [
    ([np.nan, 0], {}), ([np.inf, 0], {}), ([0], {}),
    ([0, 0], {"freeze_mask": [True]}), ([0, 0], {"freeze_mask": [2, 0]}),
    ([0, 0], {"exit_mask": [True, False]}), ([0.1, 0], {"exit_mask": [True, False]}),
    ([0, 0], {"top_n": -1}), ([0, 0], {"top_n": True}),
])
def test_invalid_delta_requests_leave_portfolio_and_broker_unchanged(deltas, options):
    portfolio = account(positions=(100, 0))
    broker = BrokerSimulator()
    before = deepcopy(portfolio.__dict__)
    with pytest.raises(ValueError):
        execute(broker, portfolio, deltas, date="2024-01-03", **options)
    assert portfolio.cash == before["cash"] and portfolio.current_date == before["current_date"]
    np.testing.assert_array_equal(portfolio.positions, before["positions"])
    assert broker.orders == [] and broker._date is None


def test_delta_environment_zero_action_preserves_shares_across_next_open_and_close():
    features, prices = panels()
    prices.loc[prices.index[1], ("000001", "open")] = 20.0
    prices.loc[prices.index[1], ("000001", "close")] = 25.0
    env = TradingEnv(market(features, prices), initial_capital=10000, initial_portfolio="equal",
                     broker_config=BrokerConfig(commission=0, max_weight=0.6))
    try:
        _, initial = env.reset(seed=42)
        _, reward, _, _, info = env.step_deltas([0, 0], top_n=1)
        np.testing.assert_array_equal(info["positions"], initial["positions"])
        assert info["orders"] == [] and info["total_fees"] == 0
        assert info["cash"] == initial["cash"]
        assert reward == pytest.approx(info["equity"] / initial["equity"] - 1)
    finally:
        env.close()


def test_delta_environment_freezes_invalid_features_after_corporate_actions():
    features, prices = panels()
    features.loc[features.index[0], ("000001", "signal")] = np.nan
    prices.loc[prices.index[1]:, ("000001", "open")] = 5.0
    prices.loc[prices.index[1]:, ("000001", "close")] = 5.0
    prices.loc[prices.index[1]:, ("000001", "limit_up")] = 5.5
    prices.loc[prices.index[1]:, ("000001", "limit_down")] = 4.5
    events = pd.DataFrame([dict(date=features.index[1], code="000001", cash_per_share=0.0,
                               share_multiplier=2.0, payment_date=features.index[1],
                               share_available_date=features.index[1])])
    env = TradingEnv(market(features, prices, corporate_actions=events), initial_capital=10000,
                     broker_config=BrokerConfig(commission=0))
    try:
        env.reset()
        env.portfolio.positions[0] = env.portfolio.available_positions[0] = 100
        env.portfolio.cash = 9000
        _, _, _, _, info = env.step_deltas([-0.1, 0], exit_mask=[True, False])
        assert info["positions"][0] == 200
        assert info["orders"][0]["filled_shares"] == 0
        assert "frozen_holdings" in info["orders"][0]["reasons"]
        assert "invalid_decision_features" in info["orders"][0]["reasons"]
        assert info["cash"] == 9000 and info["total_fees"] == 0
    finally:
        env.close()


def test_legacy_rebalance_audit_fields_and_fill_amounts_are_unchanged():
    portfolio = account()
    broker = BrokerSimulator(BrokerConfig(commission=0.001, min_commission=5))
    orders = broker.rebalance(portfolio, [1, 0], [10, 10], [10, 10], "2024-01-03")
    assert portfolio.cash == pytest.approx(991) and portfolio.positions[0] == 900
    assert orders == [dict(date=pd.Timestamp("2024-01-03"), code="000001", side="buy", target_weight=1.0,
                           requested_shares=1000.0, filled_shares=900.0, price=10.0, fee=9.0,
                           slippage_cost=0.0, status="partial", reasons=["insufficient_cash"],
                           cash_after=991.0, position_after=900.0)]
