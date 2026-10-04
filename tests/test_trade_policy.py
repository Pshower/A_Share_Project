"""Hybrid probability and account-aware integration checks without learning or gradients."""

from copy import deepcopy
import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.buffers import DictRolloutBuffer
from torch.distributions import Categorical, Normal

from src.agents.trade_mapping import HOLD, BUY, REDUCE, EXIT, TradeOrder, trade_size_active
from src.agents.trade_policy import SharedTradeNetwork, StockTradeDistribution
from src.agents.ppo_agent import PPOAgent, make_model
from src.inference.live import build_daily_plan
from src.inference.predict import predict_action
from src.training.env_factory import ResearchData, validate_config
from tests.test_ppo_framework import local_data, no_parameter_updates


pytestmark = pytest.mark.usefixtures("no_parameter_updates")


def distribution(batch=1):
    logits = torch.tensor([[[.2, -.3, .4, .1], [.1, .2, -.2, .3], [4., 5., 6., 7.]]]).expand(batch, -1, -1)
    mean = torch.tensor([[.25, -.3, .8]]).expand(batch, -1)
    legal = torch.tensor([[[1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 0]]], dtype=torch.int8).expand(batch, -1, -1)
    bounds = torch.tensor([[[2, 2, 1, 3], [0, 0, 2, 2], [0, 0, 0, 0]]], dtype=torch.int64).expand(batch, -1, -1)
    log_std = torch.tensor([-.7])
    result = StockTradeDistribution(3).proba_distribution(logits, mean, log_std, legal, bounds)
    return result, logits, mean, legal, bounds


def test_joint_probability_keeps_fixed_buy_priority_normal_and_fixed_reduce_atom():
    result, logits, mean, legal, bounds = distribution()
    action = torch.tensor([[BUY, REDUCE, HOLD, .4, 0., 0.]])
    expected = Categorical(logits=logits[0, 0]).log_prob(torch.tensor(BUY))
    expected += Categorical(logits=logits[0, 1].masked_fill(~legal[0, 1].bool(), -torch.inf)).log_prob(torch.tensor(REDUCE))
    expected += Normal(mean[0, 0], math.exp(-.7)).log_prob(action[0, 3])
    torch.testing.assert_close(result.log_prob(action)[0], expected)
    assert result.size_supports[0, 0, BUY] and not result.size_supports[0, 1, REDUCE]
    for code in [HOLD, BUY, REDUCE, EXIT]:
        codes = np.array([code, REDUCE, HOLD])
        actual = trade_size_active({"trade_lot_bounds": bounds[0].numpy()}, codes)
        expected_active = result._size_active(torch.tensor(codes).unsqueeze(0))[0].numpy()
        np.testing.assert_array_equal(actual, expected_active)


def test_true_entropy_weights_size_by_direction_probability_not_sampled_mask():
    result, _, _, _, _ = distribution()
    probabilities = result.direction_distribution.probs
    expected_category = result.direction_distribution.entropy().sum(dim=-1)
    expected_size = (probabilities[0, 0, BUY] + probabilities[0, 0, REDUCE]) * Normal(0., math.exp(-.7)).entropy()
    torch.testing.assert_close(result.categorical_entropy(), expected_category)
    torch.testing.assert_close(result.conditional_size_entropy()[0], expected_size)
    torch.testing.assert_close(result.entropy()[0], expected_category[0] + expected_size)


@pytest.mark.parametrize("action", [[1.5, 2, 0, .4, 0, 0], [4, 2, 0, .4, 0, 0],
    [0, 2, 0, .4, 0, 0], [3, 2, 0, .4, 0, 0], [1, 2, 0, .4, .1, 0],
    [1, 1, 0, .4, 0, 0], [1, 2, 3, .4, 0, 0], [1, 2, 0, float("nan"), 0, 0]])
def test_invalid_direction_or_inactive_size_is_outside_support(action):
    result, _, _, _, _ = distribution()
    assert torch.isneginf(result.log_prob(torch.tensor([action], dtype=torch.float32))).all()


def test_sampling_canonical_hold_exit_and_frozen_sizes():
    torch.manual_seed(123)
    result, _, _, _, _ = distribution(256)
    actions = result.sample()
    directions, sizes = actions[:, :3], actions[:, 3:]
    assert (directions == directions.round()).all()
    assert (directions[:, 2] == HOLD).all() and (sizes[:, 2] == 0).all()
    assert (sizes[(directions == HOLD) | (directions == EXIT)] == 0).all()
    assert (sizes[:, 1] == 0).all()
    assert torch.isfinite(result.log_prob(actions)).all() and torch.isfinite(result.entropy()).all()


@pytest.mark.parametrize("change", ["no_hold", "fractional_bounds", "nonbinary", "nan_logits"])
def test_malformed_authoritative_support_is_rejected(change):
    _, logits, mean, legal, bounds = distribution()
    logits, legal, bounds = logits.clone(), legal.clone().float(), bounds.clone().float()
    if change == "no_hold":
        legal[0, 0, HOLD] = 0
    elif change == "fractional_bounds":
        bounds[0, 0, 0] = 1.5
    elif change == "nonbinary":
        legal[0, 0, 0] = .5
    else:
        logits[0, 0, 0] = float("nan")
    with pytest.raises((ValueError, FloatingPointError)):
        StockTradeDistribution(3).proba_distribution(logits, mean, torch.tensor([-.7]), legal, bounds)


def network_observation(features=20):
    torch.manual_seed(13)
    return dict(stock_features=torch.randn(1, 3, 1, features), portfolio=torch.tensor([[.1, .2, .0, .7]]),
        receivable_ratio=torch.tensor([[0.]]), valid_mask=torch.ones(1, 3), buy_mask=torch.ones(1, 3),
        sell_mask=torch.ones(1, 3), execution_features=torch.tensor([[[.001, .1, 0.], [.002, .2, 0.], [.003, 0., 0.]]]),
        allocation_context=torch.tensor([[1., .5, .000005, 2 / 3, 0., 1 / 3, .005]]),
        trade_lot_bounds=torch.tensor([[[5, 20, 5, 10], [3, 10, 3, 15], [2, 15, 0, 0]]], dtype=torch.int64))


def test_network_shape_permutation_parameter_count_and_actor_only_gain():
    obs = network_observation()
    torch.manual_seed(42)
    network = SharedTradeNetwork(1, 20)
    assert sum(parameter.numel() for parameter in network.parameters()) == 126343
    logits, size, value, sigma = network(obs)
    assert logits.shape == (1, 3, 4) and size.shape == (1, 3) and value.shape == (1, 1) and sigma.shape == (1,)
    perm = [2, 0, 1]
    other = {key: item.clone() for key, item in obs.items()}
    for key in ["stock_features", "valid_mask", "buy_mask", "sell_mask", "execution_features", "trade_lot_bounds"]:
        other[key] = obs[key][:, perm]
    other["portfolio"] = obs["portfolio"][:, [2, 0, 1, 3]]
    other_logits, other_size, other_value, _ = network(other)
    torch.testing.assert_close(other_logits, logits[:, perm], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(other_size, size[:, perm], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(other_value, value, rtol=1e-5, atol=1e-6)
    for head in [network.direction_head, network.size_head]:
        torch.testing.assert_close(head.weight @ head.weight.T, torch.eye(head.out_features) * .01 ** 2, atol=1e-8, rtol=1e-5)
        assert torch.count_nonzero(head.bias) == 0
    torch.manual_seed(42)
    larger_gain = SharedTradeNetwork(1, 20, actor_output_gain=.1)
    for name, tensor in network.value_head.state_dict().items():
        torch.testing.assert_close(tensor, larger_gain.value_head.state_dict()[name])


def trade_config(data):
    config = deepcopy(data.config)
    config.update(action_mode="stock_trade", prior_feature_index=None, action_top_k=None,
                  max_trade_weight=.05, max_commission_ratio=.001, actor_output_gain=.01,
                  initial_log_std=-.7, top_n=2, top_n_choices=[1, 2], device="cpu")
    return config


def test_agent_save_load_typed_order_and_legacy_transfer_rejection(local_data, tmp_path):
    config = trade_config(local_data)
    data = ResearchData(config)
    env, old_env = data.vector_env(), local_data.vector_env()
    try:
        model = make_model(env, config)
        agent = PPOAgent(model, data.contract)
        obs = {key: value[0] for key, value in env.reset().items()}
        before = {key: value.clone() for key, value in model.policy.state_dict().items()}
        inspected = agent.inspect_action(obs)
        assert inspected["latent"].shape == (2 * len(data.codes),)
        assert inspected["direction_probabilities"].shape == (len(data.codes), 4)
        assert inspected["size_sigma"] == pytest.approx(math.exp(-.7))
        assert isinstance(agent.act(obs), TradeOrder)
        with torch.no_grad():
            tensor, _ = model.policy.obs_to_tensor(obs)
            _, value, _ = model.policy(tensor, deterministic=True)
            torch.testing.assert_close(value, model.policy.predict_values(tensor))
        agent.save(tmp_path / "trade_model", config)
        restored = PPOAgent.load(tmp_path / "trade_model", data.contract, "cpu")
        np.testing.assert_allclose(restored.inspect_action(obs)["latent"], inspected["latent"], rtol=1e-5, atol=1e-6)
        for key, value in model.policy.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        assert model.num_timesteps == 0 and restored.model.num_timesteps == 0
        with pytest.raises(ValueError, match="action_mode"):
            agent.transfer_to(old_env, local_data.config, local_data.contract)
        wrong = deepcopy(data.contract)
        wrong["trade"]["actor_output_gain"] = .1
        with pytest.raises(ValueError, match="contract"):
            PPOAgent(model, wrong)
        assert data.contract["trade"]["quantity_units"] == "integer_lots_buy_reduce_full_book_shares_exit"
    finally:
        env.close()
        old_env.close()


def test_rollout_buffer_retains_exact_integer_bounds_and_fractional_odd_lots(local_data):
    config = trade_config(local_data)
    data = ResearchData(config)
    env = data.vector_env()
    try:
        model = make_model(env, config)
        obs = env.reset()
        obs["trade_lot_bounds"][0, 0] = [16777217, 16777218, 0, 0]
        obs["share_state"][0, 0] = [2.53, 1.99]
        buffer = DictRolloutBuffer(2, env.observation_space, env.action_space, device="cpu")
        buffer.add(obs, np.zeros((1, env.action_space.shape[0]), dtype=np.float32), np.array([0.]), np.array([1.]),
                   torch.tensor([0.]), torch.tensor([0.]))
        assert buffer.observations["trade_lot_bounds"].dtype == np.int64
        assert buffer.observations["share_state"].dtype == np.float64
        np.testing.assert_array_equal(buffer.observations["trade_lot_bounds"][0, 0, 0], obs["trade_lot_bounds"][0, 0])
        np.testing.assert_array_equal(buffer.observations["share_state"][0, 0, 0], obs["share_state"][0, 0])
        assert model.num_timesteps == 0
    finally:
        env.close()


def test_actual_policy_forward_evaluate_identity_and_raw_size_support(local_data):
    config = trade_config(local_data)
    data = ResearchData(config)
    env = data.vector_env()
    try:
        model = make_model(env, config)
        obs = env.reset()
        obs["valid_mask"][:] = 1
        obs["execution_features"][..., 2] = 0
        obs["legal_directions"][0] = [[1, 1, 1, 1], [1, 0, 1, 1]]
        obs["trade_lot_bounds"][0] = [[2, 2, 1, 3], [0, 0, 2, 2]]
        tensor = {key: torch.as_tensor(value) for key, value in obs.items()}
        with torch.no_grad():
            actions, value, old_probability = model.policy(tensor)
            new_value, probability, _ = model.policy.evaluate_actions(tensor, actions)
            torch.testing.assert_close(old_probability, probability, rtol=0, atol=0)
            torch.testing.assert_close(value, new_value, rtol=0, atol=0)
            raw = torch.tensor([[BUY, REDUCE, 12., 0.]])
            expected = model.policy.get_distribution(tensor).log_prob(raw)
            _, actual, _ = model.policy.evaluate_actions(tensor, raw)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert torch.isfinite(actual).all()
            raw[0, 1] = 1.5
            assert torch.isneginf(model.policy.evaluate_actions(tensor, raw)[1]).all()
            raw[0] = torch.tensor([BUY, REDUCE, 12., .1])
            assert torch.isneginf(model.policy.evaluate_actions(tensor, raw)[1]).all()
            obs["legal_directions"][:] = [1, 0, 0, 0]
            frozen = {key: torch.as_tensor(value) for key, value in obs.items()}
            action, _, probability = model.policy(frozen)
            assert (action == 0).all() and (probability == 0).all()
            assert (model.policy.get_distribution(frozen).entropy() == 0).all()
        assert model.num_timesteps == 0
    finally:
        env.close()


def test_agent_rejects_wrong_policy_shape_and_trade_units(local_data):
    config = trade_config(local_data)
    data = ResearchData(config)
    env, old_env = data.vector_env(), local_data.vector_env()
    try:
        model = make_model(env, config)
        old = make_model(old_env, local_data.config)
        with pytest.raises(ValueError, match="contract"):
            PPOAgent(old, data.contract)
        malformed = SimpleNamespace(policy=model.policy, observation_space=model.observation_space,
            action_space=gym.spaces.Box(-8., 8., (2 * len(data.codes) + 1,), dtype=np.float32))
        with pytest.raises(ValueError, match="contract"):
            PPOAgent(malformed, data.contract)
        wrong = deepcopy(data.contract)
        wrong["trade"]["quantity_units"] = "weights"
        with pytest.raises(ValueError, match="economic contract"):
            PPOAgent(model, wrong)
        wrong = deepcopy(data.contract)
        wrong["trade"]["max_commission_ratio"] = data.contract["broker"]["commission"] / 2
        with pytest.raises(ValueError, match="economic contract"):
            PPOAgent(model, wrong)
        wrong = deepcopy(data.contract)
        wrong["broker"]["commission"] = 0
        wrong["trade"]["max_commission_ratio"] = 0
        with pytest.raises(ValueError, match="economic contract"):
            PPOAgent(model, wrong)
        wrong = deepcopy(data.contract)
        wrong["trade"]["execution_fields"] = list(reversed(wrong["trade"]["execution_fields"]))
        with pytest.raises(ValueError, match="economic contract"):
            PPOAgent(model, wrong)
        wrong = deepcopy(data.contract)
        wrong["hidden_sizes"][0] += 1
        with pytest.raises(ValueError, match="contract"):
            PPOAgent(model, wrong)
    finally:
        env.close()
        old_env.close()


def test_account_inference_requires_known_sellable_shares_and_blocks_weight_only_live(local_data):
    config = trade_config(local_data)
    data = ResearchData(config)
    env = data.vector_env()
    try:
        agent = PPOAgent(make_model(env, config), data.contract)
        date = data.features.index[-1]
        account = dict(cash=10000., positions=[dict(code=data.codes[0], shares=253)])
        with pytest.raises(ValueError, match="explicit available_shares"):
            predict_action(agent, data, date, account)
        account["positions"][0]["available_shares"] = 0
        result = predict_action(agent, data, date, account)
        first = result["rows"][0]
        assert result["action_mode"] == "stock_trade"
        assert first["direction"] == HOLD and first["projected_request_shares"] == 0
        assert first["shares"] == 253 and first["available_shares"] == 0
        assert "target_cash_ratio" not in result
        assert len(result["rows"]) == len(data.codes)
        with pytest.raises(ValueError, match="weight-only"):
            build_daily_plan(agent, data, "not-read", model_id="stock-trade")
        with pytest.raises(ValueError, match="registered"):
            predict_action(agent, data, date, top_n=3)
    finally:
        env.close()


@pytest.mark.parametrize("change", [{"prior_feature_index": 14}, {"action_top_k": 20},
    {"max_trade_weight": 0.}, {"max_commission_ratio": float("nan")}, {"actor_output_gain": 0.},
    {"top_n_choices": [1, 1]}])
def test_trade_config_rejects_incompatible_or_invalid_fields(local_data, change):
    config = trade_config(local_data)
    config.update(change)
    with pytest.raises(ValueError):
        validate_config(config)
