import numpy as np
import pytest

from src.envs.broker import BrokerConfig
from src.envs.trading import TradingEnv
from tests.test_trading_env import market


@pytest.mark.parametrize("mode", ["cash", "equal", "concentrated", "random", "mixed"])
def test_initial_holdings_conserve_equity_and_reproduce(mode):
    env = TradingEnv(market(), 100000, BrokerConfig(max_weight=0.6), initial_portfolio=mode)
    first, info = env.reset(seed=42)
    assert info["equity"] == pytest.approx(100000)
    assert info["cash"] >= 0 and info["total_fees"] == 0
    assert np.all(info["positions"] % 100 == 0)
    np.testing.assert_array_equal(info["positions"], info["available_positions"])
    again, repeat = env.reset(seed=42)
    np.testing.assert_array_equal(first["portfolio"], again["portfolio"])
    assert repeat["initial_account"] == info["initial_account"]
    _, _, _, _, sold = env.step(np.zeros(2))
    assert sold["positions"].sum() == 0


def test_mixed_cycle_covers_all_modes_and_random_start_stays_in_training_market():
    env = TradingEnv(market(), 100000, initial_portfolio="mixed", max_steps=2, random_start=True)
    modes = []
    for i in range(8):
        _, info = env.reset(seed=42 if i == 0 else None)
        modes.append(info["initial_account"]["mode"])
        assert 0 <= env.market.current_index < env.market.n_dates - 2
        assert info["date"] <= env.market.dates[-3]
    assert set(modes[:4]) == set(modes[4:]) == {"cash", "equal", "concentrated", "random"}


def test_equal_account_marks_actual_positions_in_backtest():
    from src.backtest.run import run_strategy
    settings = dict(initial_capital=100000, seed=42, initial_portfolio="equal", broker={}, rebalance_every=20, momentum_top_k=2)
    daily, positions, _, _ = run_strategy(market(), "buy_hold", settings)
    assert daily.iloc[0].holding_count == 2
    assert positions.iloc[0].sum() > 0
    assert daily.iloc[0].equity == 100000
