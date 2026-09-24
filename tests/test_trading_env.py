import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envs.broker import BrokerConfig, BrokerSimulator
from src.envs.market import MarketDataProvider
from src.envs.portfolio import Portfolio
from src.envs.trading import TradingEnv


def panels():
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    features = pd.DataFrame({("000001", "signal"): [0.1] * 5,
                             ("600000", "signal"): [-0.2] * 5}, index=dates)
    prices = pd.DataFrame({(code, field): [value] * 5
                           for code in ["000001", "600000"]
                           for field, value in [("open", 10.0), ("close", 10.0),
                                                ("volume", 10000.0), ("suspended", 0),
                                                ("limit_up", 11.0), ("limit_down", 9.0)]}, index=dates)
    return features, prices


def market(features=None, prices=None, **kwargs):
    default_features, default_prices = panels()
    return MarketDataProvider(default_features if features is None else features,
                              price_path=default_prices if prices is None else prices,
                              price_basis="unadjusted", **kwargs)


def account(capital=10000):
    result = Portfolio(capital, ["000001", "600000"])
    result.start_day("2024-01-02")
    return result


def test_equal_value_rotation_charges_both_sides():
    p = account()
    p.update_positions([500, 0], [10, 10])
    p.start_day("2024-01-03")
    fee, net = p.update_positions([0, 500], [10, 10], transaction_cost=0.001)
    assert net == 0
    assert fee == pytest.approx(10)
    assert p.cash == pytest.approx(4990)
    assert p.value([10, 10]) == pytest.approx(9990)
    np.testing.assert_array_equal(p.positions, [0, 500])


def test_t_plus_one_and_failed_update_are_atomic():
    p = account()
    p.fill(0, 100, 10, 0)
    p.start_day("2024-01-02")
    with pytest.raises(ValueError, match="T\\+1"):
        p.fill(0, -100, 10, 0)
    assert p.cash == 9000
    np.testing.assert_array_equal(p.positions, [100, 0])
    with pytest.raises(ValueError, match="Insufficient cash"):
        p.update_positions([1000, 1000], [10, 10])
    assert p.cash == 9000
    p.start_day("2024-01-03")
    p.fill(0, -100, 10, 0)
    assert p.value([10, 10]) == 10000
    with pytest.raises(ValueError, match="nondecreasing"):
        p.start_day("2024-01-02")


def test_full_weight_includes_fees_and_lot_size():
    p = account()
    b = BrokerSimulator(BrokerConfig(commission=0.001, min_commission=5))
    orders = b.rebalance(p, [1, 0], [10, 10], [10, 10], "2024-01-03")
    assert p.positions[0] == 900
    assert p.cash == pytest.approx(991)
    assert orders[0]["fee"] == pytest.approx(9)
    assert "insufficient_cash" in orders[0]["reasons"]
    assert orders[0]["status"] == "partial"
    assert p.value([10, 10]) == pytest.approx(9991)


def test_sells_fund_buys_and_t1_is_logged():
    p = account()
    b = BrokerSimulator(BrokerConfig(commission=0))
    b.rebalance(p, [1, 0], [10, 10], [10, 10], "2024-01-02")
    same_day = b.rebalance(p, [0, 1], [10, 10], [10, 10], "2024-01-02")
    assert same_day[0]["reasons"] == ["t_plus_one"]
    assert same_day[1]["filled_shares"] == 0
    next_day = b.rebalance(p, [0, 1], [10, 10], [10, 10], "2024-01-03")
    assert [o["side"] for o in next_day] == ["sell", "buy"]
    np.testing.assert_array_equal(p.positions, [0, 1000])
    assert p.cash == 0


@pytest.mark.parametrize("weights", [[-0.1, 0.5], [0.8, 0.8], [np.nan, 0], [np.inf, 0], [0.2]])
def test_invalid_weights_do_not_mutate_account(weights):
    p = account()
    b = BrokerSimulator()
    with pytest.raises(ValueError, match="Weights"):
        b.rebalance(p, weights, [10, 10], [10, 10], "2024-01-03")
    assert p.cash == 10000
    assert p.current_date == pd.Timestamp("2024-01-02")


def test_slippage_tax_volume_and_daily_volume_budget():
    p = account()
    b = BrokerSimulator(BrokerConfig(commission=0, sell_tax=0.001,
                                    slippage=0.01, volume_fraction=0.1))
    first = b.rebalance(p, [1, 0], [10, 10], [10, 10], "2024-01-02", volume=[2500, 2500])
    assert first[0]["filled_shares"] == 200
    assert first[0]["price"] == pytest.approx(10.1)
    second = b.rebalance(p, [1, 0], [10, 10], [10, 10], "2024-01-02", volume=[2500, 2500])
    assert second[0]["filled_shares"] == 0
    sold = b.rebalance(p, [0, 0], [10, 10], [10, 10], "2024-01-03", volume=[2500, 2500])
    assert sold[0]["price"] == pytest.approx(9.9)
    assert sold[0]["fee"] == pytest.approx(1.98)
    assert p.cash == pytest.approx(9958.02)


def test_market_separates_prices_aligns_codes_and_filters_dates(tmp_path):
    features, prices = panels()
    prices = prices[prices.columns[::-1]]
    path = tmp_path / "prices.parquet"
    prices.to_parquet(path)
    m = MarketDataProvider(features, price_path=path, price_basis="unadjusted",
                           start_date="2024-01-03", end_date="2024-01-05", lookback=2)
    state, price = m.get_state()
    np.testing.assert_allclose(state[:, 0], [0.1, -0.2])
    np.testing.assert_array_equal(price, [10, 10])
    assert m.observation()[0].shape == (2, 2, 1)
    assert m.observation()[1].all()
    assert list(m.dates) == list(pd.date_range("2024-01-03", "2024-01-05"))


def test_market_rejects_standardized_price_and_missing_price_source():
    f, p = panels()
    p.loc[p.index[0], ("000001", "close")] = -0.2
    with pytest.raises(ValueError, match="positive"):
        market(f, p)
    with pytest.raises(TypeError):
        MarketDataProvider(f)


def test_missing_features_are_finite_and_cannot_trigger_buy():
    f, p = panels()
    f.loc[f.index[0], ("000001", "signal")] = np.nan
    env = TradingEnv(market(f, p), broker_config=BrokerConfig(commission=0))
    obs, _ = env.reset()
    assert np.isfinite(obs["stock_features"]).all()
    np.testing.assert_array_equal(obs["valid_mask"], [0, 1])
    _, _, _, _, info = env.step([0.5, 0])
    assert info["positions"][0] == 0
    assert "invalid_decision_features" in info["orders"][0]["reasons"]


@pytest.mark.parametrize("field,value,reason", [
    ("suspended", 1, "suspended_or_unknown"),
    ("suspended", np.nan, "suspended_or_unknown"),
    ("open", 11, "limit_up_or_unknown"),
    ("open", np.nan, "missing_open"),
])
def test_next_day_buy_restrictions(field, value, reason):
    f, p = panels()
    p.loc[p.index[1], ("000001", field)] = value
    env = TradingEnv(market(f, p))
    env.reset()
    _, reward, _, _, info = env.step([1, 0])
    assert info["positions"].sum() == 0
    assert reward == 0
    assert reason in info["orders"][0]["reasons"]


def test_limit_down_blocks_sale_and_slippage_respects_limit():
    p = account()
    b = BrokerSimulator(BrokerConfig(commission=0, slippage=0.1))
    buy = b.rebalance(p, [0.5, 0], [10, 10], [10, 10], "2024-01-02", limit_up=[10.5, 11])
    assert buy[0]["price"] == 10.5
    sell = b.rebalance(p, [0, 0], [9, 10], [9, 10], "2024-01-03", limit_down=[9, 9])
    assert sell[0]["filled_shares"] == 0
    assert "limit_down" in sell[0]["reasons"]


def test_missing_open_uses_previous_close_for_held_asset():
    f, p = panels()
    p.loc[p.index[2], ("000001", "open")] = np.nan
    p.loc[p.index[2], ("000001", "close")] = np.nan
    env = TradingEnv(market(f, p), initial_capital=10000, broker_config=BrokerConfig(commission=0))
    env.reset()
    env.step([1, 0])
    _, reward, _, _, info = env.step([0, 0])
    assert info["positions"][0] == 1000
    assert reward == 0
    assert info["equity"] == 10000


def test_next_open_fills_next_close_rewards_and_end():
    f, p = panels()
    p.loc[p.index[1], ("000001", "open")] = 10.5
    p.loc[p.index[1], ("000001", "close")] = 11
    env = TradingEnv(market(f, p, end_date="2024-01-03"), initial_capital=10000,
                     broker_config=BrokerConfig(commission=0.001))
    obs, info = env.reset(seed=42)
    assert info["date"] == f.index[0]
    obs, reward, terminated, truncated, info = env.step([1, 0])
    assert info["orders"][0]["price"] == 10.5
    assert info["positions"][0] == 900
    assert info["cash"] == pytest.approx(540.55)
    assert info["equity"] == pytest.approx(10440.55)
    assert reward == pytest.approx(0.044055)
    assert terminated and not truncated
    assert env.observation_space.contains(obs)
    with pytest.raises(RuntimeError, match="reset"):
        env.step([0, 0])


def test_constant_price_loss_equals_cost_and_cash_hold_is_flat():
    env = TradingEnv(market(), initial_capital=10000, broker_config=BrokerConfig(commission=0.001))
    env.reset()
    _, reward, _, _, info = env.step([0.5, 0])
    assert info["total_fees"] == 5
    assert reward == pytest.approx(-0.0005)
    env.reset()
    _, reward, _, _, info = env.step([0, 0])
    assert reward == 0 and info["equity"] == 10000


def test_no_future_features_close_or_volume_used_for_decision_or_fill():
    f, p = panels()
    env = TradingEnv(market(f, p), broker_config=BrokerConfig(volume_fraction=0.1))
    before, _ = env.reset()
    f.loc[f.index[1]:, :] = 123
    p.loc[p.index[1], ("000001", "close")] = 50
    p.loc[p.index[1], ("000001", "volume")] = 0
    other = TradingEnv(market(f, p), broker_config=BrokerConfig(volume_fraction=0.1))
    after, _ = other.reset()
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    first_info = env.step([1, 0])[-1]
    second_info = other.step([1, 0])[-1]
    assert first_info["orders"] == second_info["orders"]
    assert first_info["equity"] != second_info["equity"]


def test_gymnasium_checker_determinism_truncation_and_action_sampling():
    env = TradingEnv(market(), max_steps=2)
    check_env(env, skip_render_check=True)
    env.reset(seed=17)
    for _ in range(30):
        assert env.action_space.contains(env.action_space.sample())
    env.step([0, 0])
    _, _, terminated, truncated, _ = env.step([0, 0])
    assert not terminated and truncated
    obs, info = env.reset(seed=17)
    assert info["total_fees"] == 0 and info["positions"].sum() == 0
    assert env.observation_space.contains(obs)


def test_hfq_mode_and_missing_constraints_are_disclosed():
    f, p = panels()
    p = p.loc[:, p.columns.get_level_values(1).isin(["open", "close"])]
    m = MarketDataProvider(f, price_path=p, price_basis="hfq_research")
    _, info = TradingEnv(m).reset()
    assert info["price_basis"] == "hfq_research"
    assert any("No suspended" in a for a in info["assumptions"])
    assert any("not real execution" in a for a in info["assumptions"])


def test_dividend_receivable_payment_and_bonus_share_unlock():
    f, p = panels()
    p.loc[p.index[2]:, ("000001", "open")] = 4.5
    p.loc[p.index[2]:, ("000001", "close")] = 4.5
    p.loc[p.index[2]:, ("000001", "limit_up")] = 5.0
    p.loc[p.index[2]:, ("000001", "limit_down")] = 4.0
    events = pd.DataFrame([dict(date=f.index[2], code="000001", cash_per_share=1.0,
                                share_multiplier=2.0, payment_date=f.index[3], share_available_date=f.index[4])])
    env = TradingEnv(market(f, p, corporate_actions=events), initial_capital=1000,
                     broker_config=BrokerConfig(commission=0))
    env.reset()
    env.step([1, 0])
    obs, reward, _, _, info = env.advance()
    assert info["positions"][0] == 200
    assert info["available_positions"][0] == 100
    assert info["receivables"] == 100 and info["cash"] == 0
    assert info["equity"] == 1000 and reward == 0
    assert obs["receivable_ratio"][0] == pytest.approx(0.1)
    _, reward, _, _, info = env.advance()
    assert info["cash"] == 100 and info["receivables"] == 0
    assert info["available_positions"][0] == 100
    assert info["equity"] == 1000 and reward == 0
    _, _, _, _, info = env.advance()
    assert info["available_positions"][0] == 200
    env.reset()
    assert env.portfolio.receivable_value == 0
    assert env.portfolio.share_locks == []


def test_corporate_actions_reject_hfq_double_counting_and_fractional_shares():
    f, p = panels()
    with pytest.raises(ValueError, match="second time"):
        MarketDataProvider(f, price_path=p, price_basis="hfq_research", corporate_actions=pd.DataFrame())
    portfolio = account()
    portfolio.fill(0, 1, 10, 0)
    before = portfolio.cash
    with pytest.raises(ValueError, match="Fractional"):
        portfolio.apply_corporate_action(0, 1, 1.5, "2024-01-02", "2024-01-02")
    assert portfolio.cash == before and portfolio.positions[0] == 1
