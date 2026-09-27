from copy import deepcopy

import numpy as np
import pytest
import torch

from src.agents.action_mapping import map_action
from src.agents.ppo_agent import PPOAgent, make_model
from src.agents.ppo_policy import SharedStockNetwork
from src.training.env_factory import ResearchData
from src.envs.trading import TradingEnv
from tests.test_ppo_framework import local_data, no_parameter_updates
from tests.test_trading_env import market


def test_top_k_respects_masks_caps_and_cash():
    obs = dict(valid_mask=np.ones(4), buy_mask=np.array([1, 0, 1, 1]))
    result = map_action([1, 8, 3, 2, 0], obs, 0.4, top_k=2)
    assert result["weights"][0] == result["weights"][1] == 0
    assert (result["weights"] > 0).sum() == 2
    assert result["weights"].max() <= 0.4 and result["weights"].sum() <= 1
    obs["buy_mask"][:] = 0
    assert map_action([1, 8, 3, 2, 0], obs, 0.4, top_k=2)["cash_weight"] == 1
    with pytest.raises(ValueError):
        map_action([1, 8, 3, 2, 0], obs, 0.4, top_k=0)


def test_prior_uses_current_eligible_ranks_and_equal_ties():
    net = SharedStockNetwork(1, 1, (8, 4), initial_log_std=-3, prior_feature_index=0)
    obs = dict(stock_features=torch.tensor([[[[1.]], [[2.]], [[2.]], [[999.]]]]),
               portfolio=torch.tensor([[0., 0., 0., 0., 1.]]), valid_mask=torch.tensor([[1, 1, 1, 0]]),
               buy_mask=torch.ones((1, 4)), sell_mask=torch.ones((1, 4)), receivable_ratio=torch.zeros((1, 1)))
    mean, _, std = net(obs)
    assert mean[0, 0] < mean[0, 1] == mean[0, 2]
    assert mean[0, 3] == mean[0, 4] == 0
    assert torch.all(std == -3)
    perm = [2, 0, 3, 1]
    moved = {k: v[:, perm] if k not in {"portfolio", "receivable_ratio"} else v for k, v in obs.items()}
    moved["portfolio"] = torch.cat([obs["portfolio"][:, perm], obs["portfolio"][:, -1:]], 1)
    changed = net(moved)[0]
    torch.testing.assert_close(changed[:, :-1], mean[:, perm])


def test_sparse_prior_save_load_rollout_and_inference_agree(local_data, tmp_path, no_parameter_updates):
    config = deepcopy(local_data.config)
    config.update(action_top_k=1, initial_log_std=-3, prior_feature_index=14, prior_strength=4., residual_scale=.25)
    data = ResearchData(config)
    env = data.vector_env()
    try:
        agent = PPOAgent(make_model(env, config), data.contract)
        obs = data.env("train")
        initial, _ = obs.reset(seed=42)
        predicted = agent.inspect_action(initial)
        latent, _ = agent.model.predict(initial, deterministic=True)
        _, _, _, _, info = obs.step(latent)
        np.testing.assert_array_equal(predicted["weights"], info["action_diagnostics"]["weights"])
        assert (predicted["weights"] > 0).sum() <= 1
        agent.save(tmp_path / "sparse", config)
        restored = PPOAgent.load(tmp_path / "sparse", data.contract, "cpu")
        np.testing.assert_array_equal(agent.act(initial), restored.act(initial))
        wrong = deepcopy(data.contract)
        wrong["prior"]["strength"] = 2
        with pytest.raises(ValueError, match="prior"):
            PPOAgent(agent.model, wrong)
        obs.close()
    finally:
        env.close()


def test_recent_window_is_in_training_range_and_reproducible():
    env = TradingEnv(market(), max_steps=2, random_start=True, recent_episode_every=4)
    starts = []
    for i in range(8):
        _, info = env.reset(seed=42 if i == 0 else None)
        starts.append(env.market.current_index)
    assert starts[0] == starts[4] == 2
    assert all(0 <= i <= 2 for i in starts)
    env.reset(seed=42)
    assert env.market.current_index == starts[0]
