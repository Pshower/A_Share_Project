"""Trade decisions use the current book, not next-day quotes or new share locks."""

import numpy as np
import pandas as pd

from src.agents.trade_mapping import BUY, EXIT, HOLD, TradeActionWrapper
from src.envs.broker import BrokerConfig
from src.envs.trading import TradingEnv
from tests.test_trading_env import market, panels


def wrapper(features, prices, **kwargs):
    base = TradingEnv(market(features, prices, **kwargs), initial_capital=1000000,
                      broker_config=BrokerConfig(min_commission=5, slippage=.001,
                                                 max_weight=.05, volume_fraction=.01))
    return TradeActionWrapper(base, top_n=2)


def test_future_quotes_features_and_volume_do_not_change_decision_support():
    features, prices = panels()
    prices = prices.drop(columns=[(code, field) for code in ("000001", "600000")
                                  for field in ("limit_up", "limit_down")])
    for code in ("000001", "600000"):
        prices[(code, "volume")] = 1000000
    future_features, future_prices = features.copy(), prices.copy()
    future_features.iloc[1] = 123
    future_prices.loc[prices.index[1], ("000001", "open")] = 20
    future_prices.loc[prices.index[1], ("000001", "close")] = 23
    future_prices.loc[prices.index[1], ("000001", "volume")] = 0
    original = wrapper(features, prices)
    perturbed = wrapper(future_features, future_prices)
    original_obs, _ = original.reset(seed=42)
    perturbed_obs, _ = perturbed.reset(seed=42)
    for name in original_obs:
        np.testing.assert_array_equal(original_obs[name], perturbed_obs[name], err_msg=name)
    _, _, _, _, original_info = original.step(np.array([BUY, HOLD, 0, 0], dtype=np.float32))
    _, _, _, _, perturbed_info = perturbed.step(np.array([BUY, HOLD, 0, 0], dtype=np.float32))
    np.testing.assert_array_equal(original_info["action_diagnostics"]["shares"],
                                  perturbed_info["action_diagnostics"]["shares"])
    assert original_info["positions"][0] > perturbed_info["positions"][0] > 0
    fill = perturbed_info["orders"][0]
    assert 5000 <= fill["filled_shares"] * fill["price"] <= .05 * fill["opening_equity"]


def test_next_day_locked_bonus_prevents_full_exit_and_new_top_n_slot():
    features, prices = panels()
    prices.loc[prices.index[1]:, ("000001", "open")] = 5
    prices.loc[prices.index[1]:, ("000001", "close")] = 5
    prices.loc[prices.index[1]:, ("000001", "limit_up")] = 5.5
    prices.loc[prices.index[1]:, ("000001", "limit_down")] = 4.5
    events = pd.DataFrame([dict(date=features.index[1], code="000001", cash_per_share=0,
                                share_multiplier=2, payment_date=features.index[1],
                                share_available_date=features.index[3])])
    base = TradingEnv(market(features, prices, corporate_actions=events), initial_capital=200000,
                      broker_config=BrokerConfig(min_commission=5, max_weight=.05))
    env = TradeActionWrapper(base, top_n=1)
    env.reset(seed=42)
    base.portfolio.positions[0] = base.portfolio.available_positions[0] = 1000
    base.portfolio.cash = 190000
    env._observation = env.enrich(base._observation())
    assert env._observation["legal_directions"][0, EXIT]
    _, _, _, _, info = env.step(np.array([EXIT, BUY, 0, 0], dtype=np.float32))
    np.testing.assert_array_equal(info["decision_snapshot"]["positions"], [1000, 0])
    np.testing.assert_array_equal(info["opening_snapshot"]["positions"], [2000, 0])
    np.testing.assert_array_equal(info["positions"], [1000, 0])
    np.testing.assert_array_equal(info["available_positions"], [0, 0])
    sold, blocked_buy = info["orders"]
    assert sold["trade_direction"] == EXIT and sold["filled_shares"] == -1000
    assert sold["position_after"] == 1000 and sold["status"] == "partial"
    assert blocked_buy["filled_shares"] == 0 and "top_n_limit" in blocked_buy["reasons"]
