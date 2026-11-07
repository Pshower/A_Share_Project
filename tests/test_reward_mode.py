"""Reward shaping against a benchmark, and the per-checkpoint backtest row.

The identity under test is narrow and checkable: whatever the fees and lots do to the raw
return, the shaped reward must be that raw return minus an equal-weight benchmark measured
over the same close-to-close interval.
"""

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.metrics import performance
from src.envs.broker import BrokerConfig
from src.envs.market import MarketDataProvider
from src.envs.trading import REWARD_MODES, TradingEnv
from src.training.train import checkpoint_row


def panels(features=None):
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    if features is None:
        features = pd.DataFrame({("000001", "signal"): [0.1] * 4,
                                 ("600000", "signal"): [-0.2] * 4}, index=dates)
    closes = {"000001": [10.0, 11.0, 11.0, 10.0], "600000": [20.0, 20.0, 22.0, 22.0]}
    prices = pd.DataFrame({(code, field): values
                           for code, values in closes.items()
                           for field in ("open", "close")}, index=dates)
    for code in closes:
        prices[(code, "volume")] = 1e6
    return features, prices


def market(features=None):
    default_features, prices = panels(features)
    return MarketDataProvider(default_features, price_path=prices,
                              price_basis="unadjusted")


def broker():
    return BrokerConfig(commission=0.0, min_commission=0.0, sell_tax=0.0, slippage=0.0,
                        lot_size=100, max_weight=0.5, volume_fraction=None)


def test_reward_modes_are_declared():
    assert set(REWARD_MODES) == {"return", "excess_pool"}


def test_unknown_reward_mode_is_rejected():
    with pytest.raises(ValueError):
        TradingEnv(market(), 10000, broker(), reward_mode="excess_index")


def test_default_reward_is_the_raw_equity_return():
    env = TradingEnv(market(), 10000, broker())
    env.reset(seed=1)
    _, reward, _, _, info = env.step(np.array([0.5, 0.5], dtype=np.float32))
    assert reward == pytest.approx(info["raw_return"])
    assert reward == pytest.approx(info["equity"] / 10000 - 1)


def test_excess_reward_subtracts_the_equal_weight_pool():
    env = TradingEnv(market(), 10000, broker(), reward_mode="excess_pool")
    env.reset(seed=1)
    _, reward, _, _, info = env.step(np.array([0.5, 0.5], dtype=np.float32))
    # Day one closes: 000001 10 -> 11 (+10%), 600000 20 -> 20 (0%). Pool mean is +5%.
    assert info["benchmark_return"] == pytest.approx(0.05)
    # The account was still in cash over that interval, so the raw return is zero and the
    # whole benchmark is a shortfall -- which is the point of the shaping.
    assert info["raw_return"] == pytest.approx(0.0)
    assert reward == pytest.approx(-0.05)
    assert reward == pytest.approx(info["excess_return"])


def test_a_partly_invested_book_earns_less_than_the_pool_when_it_rises():
    env = TradingEnv(market(), 10000, broker(), reward_mode="excess_pool")
    env.reset(seed=1)
    weights = np.array([0.5, 0.5], dtype=np.float32)
    env.step(weights)
    _, reward, _, _, info = env.step(weights)
    # Day two closes: 000001 11 -> 11 (0%), 600000 20 -> 22 (+10%). Pool mean is +5%.
    assert info["benchmark_return"] == pytest.approx(0.05)
    # Lot rounding leaves part of the account in cash, which cannot earn the pool return.
    assert 0.0 < info["raw_return"] < 0.05
    assert reward == pytest.approx(info["raw_return"] - info["benchmark_return"])
    assert reward < 0


def test_benchmark_ignores_a_stock_the_agent_could_not_hold():
    """A name without a complete feature row is invalid, so it must leave the benchmark too."""
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    features = pd.DataFrame({("000001", "signal"): [0.1] * 4,
                             ("600000", "signal"): [np.nan] + [-0.2] * 3}, index=dates)
    env = TradingEnv(market(features), 10000, broker(), reward_mode="excess_pool")
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.array([0.5, 0.5], dtype=np.float32))
    # Only 000001 is valid on day one, so the benchmark is its own +10% return.
    assert info["benchmark_return"] == pytest.approx(0.10)


def test_full_cash_reward_is_the_negative_benchmark():
    env = TradingEnv(market(), 10000, broker(), reward_mode="excess_pool")
    env.reset(seed=1)
    _, reward, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["raw_return"] == pytest.approx(0.0)
    assert reward == pytest.approx(-info["benchmark_return"])


def daily_frame(equities, start="2024-01-02"):
    dates = pd.date_range(start, periods=len(equities), freq="B")
    return pd.DataFrame(dict(date=dates, equity=equities, cash=0.0, daily_return=0.0,
                             fee=0.0, slippage_cost=0.0, turnover=0.0))


def test_checkpoint_row_measures_excess_over_the_reference():
    daily = daily_frame([100.0, 110.0, 120.0])
    reference = daily_frame([100.0, 100.0, 100.0])
    row = checkpoint_row(512, daily, reference, performance(daily))
    assert row["step"] == 512
    assert row["policy_return"] == pytest.approx(0.20)
    assert row["reference_return"] == pytest.approx(0.0)
    assert row["excess_return"] == pytest.approx(0.20)
    assert row["matched_steps"] == 2
    assert row["tracking_error"] > 0


def test_checkpoint_row_is_zero_when_the_policy_holds_the_pool():
    daily = daily_frame([100.0, 105.0, 103.0])
    row = checkpoint_row(256, daily, deepcopy(daily), performance(daily))
    assert row["excess_return"] == pytest.approx(0.0, abs=1e-12)
    assert row["tracking_error"] == pytest.approx(0.0, abs=1e-12)
    assert row["information_ratio"] is None


def test_reward_mode_reaches_the_contract():
    from src.training.env_factory import validate_config
    base = dict(research_config="configs/research.json", stock_codes=None, lookback=1, seed=42,
                device="cpu", torch_threads=1, hidden_sizes=[8, 8], logit_bound=8.0,
                validation_rollouts=10, output_root="reports/runs/tmp",
                ppo=dict(n_steps=8, batch_size=4, learning_rate=1e-3, n_epochs=1, gamma=0.99,
                         gae_lambda=0.95, clip_range=0.2, target_kl=0.03, ent_coef=0.0,
                         vf_coef=0.5, max_grad_norm=0.5))
    assert validate_config(dict(base, reward_mode="excess_pool", checkpoint_every=64))["reward_mode"] == "excess_pool"
    with pytest.raises(ValueError):
        validate_config(dict(base, reward_mode="sharpe"))
    with pytest.raises(ValueError):
        validate_config(dict(base, checkpoint_every=0))


def test_checkpoint_summary_reports_first_and_final_separately():
    from tools.checkpoint_compare import summarise
    frame = pd.DataFrame({
        "step": [256, 512, 768], "policy_return": [0.01, 0.02, 0.03],
        "reference_return": [0.05, 0.05, 0.05], "excess_return": [-0.04, -0.03, -0.02],
        "sharpe": [0.1, 0.2, 0.3], "max_drawdown": [-0.1, -0.1, -0.1],
        "mean_cash_ratio": [0.3, 0.2, 0.1], "mean_daily_turnover": [0.01, 0.01, 0.01],
        "total_fees": [1.0, 2.0, 3.0], "tracking_error": [0.01, 0.01, 0.01],
        "information_ratio": [-1.0, -1.0, -1.0], "run": ["x", "x", "x"]})
    row = summarise(frame)
    assert row["checkpoints"] == 3
    assert row["first_excess"] == pytest.approx(-0.04)
    assert row["final_excess"] == pytest.approx(-0.02)
    # Best and worst are reported but never selected on.
    assert row["best_excess"] == pytest.approx(-0.02)
    assert row["best_step"] == 768
    assert row["cash_first"] == pytest.approx(0.3)
    assert row["cash_final"] == pytest.approx(0.1)
