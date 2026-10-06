"""Synthetic checks; no network, real history or training updates."""

import numpy as np
import pandas as pd
import pytest

from src.training.exploration_env import DateSampler, ExplorationMarket, ExplorationEnv, EconomicRule
from src.envs.broker import BrokerConfig
from src.agents.trade_mapping import TradeActionWrapper


def market(lookback=1, missing=False, sampler=False):
    dates = pd.bdate_range("2020-01-01", periods=30)
    cols = pd.MultiIndex.from_product([["A", "B"], ["momentum_20"]])
    features = pd.DataFrame(np.arange(60).reshape(30, 2) / 100, index=dates, columns=cols)
    if missing:
        features.iloc[12, 0] = np.nan
    prices = pd.DataFrame(10., index=dates, columns=pd.MultiIndex.from_product([["A", "B"], ["open", "close", "volume"]]))
    prices.loc[:, pd.IndexSlice[:, "volume"]] = 1e8
    prices.iloc[-1, prices.columns.get_level_values(1) == "close"] = 11.
    result = ExplorationMarket(features, ["momentum_20"], price_path=prices, price_basis="hfq_research",
                               start_date=dates[4], lookback=lookback, common_lookback=5)
    if sampler:
        result.sampler = DateSampler(result.dates, 8)
    return result


def test_common_mask_is_equal_for_different_lookbacks_and_no_future_leak():
    one, five = market(1, True), market(5, True)
    for index in range(one.n_dates):
        for field in (1, 2, 3):
            np.testing.assert_array_equal(one.observation(index)[field], five.observation(index)[field])
    assert not one.observation(8)[1][0]
    assert one.observation(13)[1][0]
    before = one.observation(5)
    one._features[one._indices[6]:] = 999.
    for old, new in zip(before, one.observation(5)):
        np.testing.assert_array_equal(old, new)


def test_quarter_sampler_balances_buckets_and_reproduces_without_global_rng():
    dates = pd.bdate_range("2018-01-01", "2022-12-31")
    a, b = DateSampler(dates, 256, "quarter_balanced"), DateSampler(dates, 256, "quarter_balanced")
    a.seed(42)
    b.seed(42)
    size = len(a.buckets)
    indices = [a.next_index() for _ in range(size)]
    assert indices == [b.next_index() for _ in range(size)]
    assert len(set(dates[indices].to_period("Q"))) == size
    assert max(indices) + 256 < len(dates)


def test_recent_sampler_keeps_every_fourth_recent_episode():
    sample = DateSampler(pd.bdate_range("2018-01-01", periods=600), 256)
    values = [sample.next_index() for _ in range(9)]
    assert values[0] == values[4] == values[8] == sample.high - 1
    with pytest.raises(ValueError):
        DateSampler(pd.bdate_range("2020-01-01", periods=10), 256)


def test_sampler_selects_before_initial_account_and_reset_seed_reproduces():
    env = ExplorationEnv(market(sampler=True), 1e6, BrokerConfig(), initial_portfolio="mixed", max_steps=8)
    _, info = env.reset(seed=42)
    first = (info["date"], info["cash"], info["positions"].copy())
    for _ in range(4):
        env.reset()
    _, info = env.reset(seed=42)
    assert info["date"] == first[0] and info["cash"] == first[1]
    np.testing.assert_array_equal(info["positions"], first[2])


def test_account_mode_flow_does_not_depend_on_date_sampler_or_eligible_count():
    a, b = market(sampler=True), market(sampler=True)
    a.sampler = DateSampler(a.dates, 8, "recent")
    b.sampler = DateSampler(b.dates, 8, "quarter_balanced")
    b.common_valid[:, 0] = False
    env_a = ExplorationEnv(a, 1e6, BrokerConfig(), initial_portfolio="mixed", max_steps=8)
    env_b = ExplorationEnv(b, 1e6, BrokerConfig(), initial_portfolio="mixed", max_steps=8)
    modes_a, modes_b = [], []
    for i in range(32):
        _, left = env_a.reset(seed=42 if i == 0 else None)
        _, right = env_b.reset(seed=42 if i == 0 else None)
        modes_a.append(left["initial_account"]["mode"])
        modes_b.append(right["initial_account"]["mode"])
    assert modes_a == modes_b
    assert len(set(modes_a)) == 4


@pytest.mark.parametrize("basis", ["simple_net", "log_net"])
def test_rewards_preserve_identical_book_and_fees_once(basis):
    env = ExplorationEnv(market(), 1e6, BrokerConfig(), initial_portfolio="equal", reward_basis=basis)
    _, initial = env.reset(seed=7)
    while True:
        _, reward, done, truncated, info = env.advance()
        ratio = info["equity"] / initial["equity"] - 1
        assert info["raw_net_return"] == pytest.approx(ratio)
        assert reward == pytest.approx(np.log1p(ratio) if basis == "log_net" else ratio)
        assert info["total_fees"] == 0
        initial = info
        if done or truncated:
            break


def test_true_hold_rule_has_zero_order_requests():
    env = TradeActionWrapper(ExplorationEnv(market(), 1e6, BrokerConfig(), initial_portfolio="equal"), top_n=1)
    obs, initial = env.reset(seed=3)
    contract = dict(stock_codes=["A", "B"], feature_cols=["momentum_20"], logit_bound=8., broker=dict(lot_size=100))
    rule = EconomicRule(contract)
    order = rule.act(obs, 0)
    np.testing.assert_array_equal(order.shares, [0, 0])
    _, _, _, _, info = env.step_order(order)
    np.testing.assert_array_equal(info["positions"], initial["positions"])
    assert info["orders"] == []


def test_ranking_baseline_uses_mapper_support_and_interval():
    env = TradeActionWrapper(ExplorationEnv(market(), 1e6, BrokerConfig()), top_n=1)
    obs, _ = env.reset(seed=3)
    contract = dict(stock_codes=["A", "B"], feature_cols=["momentum_20"], logit_bound=8., broker=dict(lot_size=100))
    rule = EconomicRule(contract, "momentum_20", 20)
    order = rule.act(obs, 0)
    assert order.shares[0] == 0 and order.shares[1] > 0
    np.testing.assert_array_equal(rule.act(obs, 1).shares, [0, 0])
