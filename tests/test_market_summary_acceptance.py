"""Tiny synthetic PPO updates; no market data or financial evaluation."""

import random

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from src.agents.market_summary_policy import MarketSummaryPolicy
from src.training.research_market_summary import parameter_participation
from tests.test_exploration_policy import observation, spaces


class SyntheticStocks(gym.Env):
    def __init__(self):
        self.template = observation(n=3)
        self.observation_space, self.action_space = spaces(self.template)
        self.index = 0

    def _observation(self):
        values = {name: value[0].numpy().copy() for name, value in self.template.items()}
        values["stock_features"] += np.float32(.03 * self.index)
        return values

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return self._observation(), {}

    def step(self, action):
        self.index += 1
        reward = .001 * float(action[0] == 1) - .0001 * float(np.count_nonzero(action[:3]))
        return self._observation(), reward, self.index == 4, False, {}


@pytest.mark.parametrize("pooling", ["mean", "attention", "dispersion"])
def test_tiny_ppo_updates_train_active_pooling_only_and_roundtrip(tmp_path, pooling):
    previous_threads = torch.get_num_threads()
    numpy_state, python_state = np.random.get_state(), random.getstate()
    env = SyntheticStocks()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            model = PPO(
                MarketSummaryPolicy, env, device="cpu", seed=25,
                n_steps=8, batch_size=4, n_epochs=1, learning_rate=1e-3,
                policy_kwargs=dict(pooling=pooling, hidden_sizes=(16, 8)),
            )
            initial = {name: value.clone() for name, value in model.policy.state_dict().items()}
            model.learn(total_timesteps=16)
            assert model.num_timesteps == 16
            assert all(torch.isfinite(value).all() for value in model.policy.parameters())
            assert not torch.equal(initial["network.value_head.2.weight"], model.policy.network.value_head[2].weight)
            participation = parameter_participation(model)
            assert participation["inactive_adam_state_names"] == []
            assert participation["inactive_nonzero_names"] == []
            assert participation["adam_state_parameters"] == participation["expected_functionally_active_parameters"]
            if pooling == "attention":
                assert torch.count_nonzero(model.policy.network.summary_query).item() > 0
            elif pooling == "dispersion":
                assert torch.count_nonzero(model.policy.network.variance_projection.weight).item() > 0
            obs, _ = env.reset(seed=25)
            before, _ = model.predict(obs, deterministic=True)
            model.save(tmp_path / pooling)
            restored = PPO.load(tmp_path / pooling, device="cpu")
            after, _ = restored.predict(obs, deterministic=True)
            np.testing.assert_array_equal(before, after)
            for name, value in model.policy.state_dict().items():
                torch.testing.assert_close(value, restored.policy.state_dict()[name], rtol=0, atol=0)
    finally:
        env.close()
        torch.set_num_threads(previous_threads)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
