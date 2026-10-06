"""Independent synthetic review: no history diagnostics or parameter updates."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.exploration_policy import ExplorationTradePolicy
from src.agents.trade_mapping import TradeActionWrapper
from src.envs.broker import BrokerConfig
from src.training.exploration_env import (
    ACCOUNT_SEED_SALT,
    DateSampler,
    ExplorationEnv,
    ExplorationMarket,
)
from tests.test_ppo_framework import no_parameter_updates


pytestmark = pytest.mark.usefixtures("no_parameter_updates")


def synthetic_market(*, lookback=1, length=620, sampler=None, hold_only=False,
                     codes=("B", "A"), fields=("momentum_20", "return")):
    dates = pd.bdate_range("2020-01-01", periods=length)
    feature_columns = pd.MultiIndex.from_product([codes, fields])
    features = pd.DataFrame(index=dates, columns=feature_columns, dtype=float)
    price_columns = pd.MultiIndex.from_product([tuple(reversed(codes)),
                                                ["close", "volume", "open", "suspended"]])
    prices = pd.DataFrame(index=dates, columns=price_columns, dtype=float)
    for stock, code in enumerate(codes):
        for feature, field in enumerate(fields):
            features[(code, field)] = stock * 10 + feature + np.arange(length) / 100
        prices[(code, "close")] = 10 + stock * 2 + np.arange(length) / 1000
        prices[(code, "open")] = prices[(code, "close")]
        prices[(code, "volume")] = 1e9
        prices[(code, "suspended")] = int(hold_only)
    result = ExplorationMarket(features, list(fields), price_path=prices,
        price_basis="hfq_research", start_date=dates[4], lookback=lookback,
        common_lookback=5)
    if sampler:
        result.sampler = DateSampler(result.dates, 8, sampler)
    return result


class RawRolloutCapture(BaseCallback):
    def __init__(self):
        super().__init__()
        self.raw_rewards, self.info, self.terminal_dates = [], [], []

    def _on_step(self):
        self.raw_rewards.append(float(self.locals["rewards"][0]))
        info = deepcopy(self.locals["infos"][0])
        self.info.append(info)
        if "terminal_observation" in info:
            self.terminal_dates.append(pd.Timestamp(info["date"]))
        return True


@pytest.mark.parametrize("reward_basis", ["simple_net", "log_net"])
@pytest.mark.parametrize("natural_terminal", [False, True])
def test_actual_sb3_timeout_bootstrap_and_gae_keep_raw_rewards_separate(
        reward_basis, natural_terminal, monkeypatch):
    market = synthetic_market(length=7 if natural_terminal else 20, hold_only=True)
    base = ExplorationEnv(market, 1e6, BrokerConfig(), initial_portfolio="cash",
        max_steps=None if natural_terminal else 2, reward_basis=reward_basis)
    wrapped = TradeActionWrapper(base, top_n=1)
    vector = DummyVecEnv([lambda: Monitor(wrapped)])
    model = PPO(ExplorationTradePolicy, vector, n_steps=2, batch_size=2,
        n_epochs=1, gamma=.9, gae_lambda=.8, seed=42, device="cpu",
        policy_kwargs=dict(hidden_sizes=(16, 8), architecture="baseline"))
    before = {name: value.clone() for name, value in model.policy.state_dict().items()}
    terminal_value = 2.5
    monkeypatch.setattr(model.policy, "predict_values", lambda observation:
        torch.full((observation["portfolio"].shape[0], 1), terminal_value))
    callback = RawRolloutCapture()
    try:
        _, initialized = model._setup_learn(total_timesteps=2, callback=callback)
        assert initialized is callback
        assert model.collect_rollouts(vector, callback, model.rollout_buffer,
                                      n_rollout_steps=2)
        buffer = model.rollout_buffer
        assert model.num_timesteps == 2
        np.testing.assert_array_equal(callback.raw_rewards, [0., 0.])
        assert len(callback.terminal_dates) == 1
        expected_date = market.dates[2]
        assert callback.terminal_dates[0] == expected_date
        assert callback.info[-1]["TimeLimit.truncated"] is (not natural_terminal)
        for info in callback.info:
            assert info["raw_net_return"] == info["learning_reward"] == info["reward"] == 0.
            assert info["equity"] == 1e6
            assert info["total_fees"] == 0.
        augmented = 0. if natural_terminal else model.gamma * terminal_value
        np.testing.assert_allclose(buffer.rewards[:, 0], [0., augmented])
        last_advantage = augmented - buffer.values[1, 0]
        first_delta = model.gamma * buffer.values[1, 0] - buffer.values[0, 0]
        first_advantage = first_delta + model.gamma * model.gae_lambda * last_advantage
        np.testing.assert_allclose(buffer.advantages[:, 0],
            [first_advantage, last_advantage], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(buffer.returns[:, 0],
            [first_advantage + buffer.values[0, 0], augmented], rtol=1e-6, atol=1e-6)
        for name, value in model.policy.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        assert not model.policy.optimizer.state
    finally:
        vector.close()


def test_all_three_episode_random_streams_are_independent_of_eligible_draws():
    first = synthetic_market(sampler="recent")
    second = synthetic_market(sampler="quarter_balanced")
    # Account initialization now consumes a different number of eligible draws.
    second.common_valid[:, 0] = False
    left = TradeActionWrapper(ExplorationEnv(first, 1e6, BrokerConfig(),
        initial_portfolio="mixed", max_steps=8), top_n_choices=[10, 20, 40])
    right = TradeActionWrapper(ExplorationEnv(second, 1e6, BrokerConfig(),
        initial_portfolio="mixed", max_steps=8), top_n_choices=[10, 20, 40])
    modes, ks = [], []
    try:
        for episode in range(40):
            a, ia = left.reset(seed=123 if episode == 0 else None)
            b, ib = right.reset(seed=123 if episode == 0 else None)
            expected_seed = int(np.random.SeedSequence(
                [123, ACCOUNT_SEED_SALT, episode]).generate_state(1)[0])
            assert left.unwrapped._np_random_seed == right.unwrapped._np_random_seed == expected_seed
            assert ia["initial_account"]["mode"] == ib["initial_account"]["mode"]
            np.testing.assert_array_equal(a["top_n"], b["top_n"])
            modes.append(ia["initial_account"]["mode"])
            ks.append(int(a["top_n"][0]))
            assert first.current_index + 8 < first.n_dates
            assert second.current_index + 8 < second.n_dates
        assert set(modes) == {"cash", "equal", "concentrated", "random"}
        assert set(ks) == {10, 20, 40}
        for start in range(0, 40, 4):
            assert len(set(modes[start:start + 4])) == 4
    finally:
        left.close()
        right.close()


def test_provider_order_preserves_codes_features_and_common_support():
    one, five = synthetic_market(), synthetic_market(lookback=5)
    assert one.stock_codes == five.stock_codes == ["B", "A"]
    assert one.feature_cols == five.feature_cols == ["momentum_20", "return"]
    for index in (0, 20, 200, one.n_dates - 1):
        l1, l5 = one.observation(index), five.observation(index)
        np.testing.assert_array_equal(l1[0][:, -1], l5[0][:, -1])
        for field in (1, 2, 3):
            np.testing.assert_array_equal(l1[field], l5[field])
        j = one._indices[index]
        np.testing.assert_allclose(l1[0][:, -1],
            [[j / 100, 1 + j / 100], [10 + j / 100, 11 + j / 100]])
        np.testing.assert_allclose(one.execution(index)["prices"],
                                   [10 + j / 1000, 12 + j / 1000])


def test_final_full_episode_fits_but_cannot_cross_price_boundary():
    dates = pd.bdate_range("2020-01-01", periods=9)
    for mode in ("recent", "quarter_balanced"):
        sampler = DateSampler(dates, episode_steps=8, mode=mode)
        sampler.seed(42)
        assert {sampler.next_index() for _ in range(10)} == {0}
        assert dates[0 + 8] == dates[-1]
    with pytest.raises(ValueError, match="fit"):
        DateSampler(dates[:8], episode_steps=8)
