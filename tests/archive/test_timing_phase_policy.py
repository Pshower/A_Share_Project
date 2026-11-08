"""Synthetic tools-only phase Actor tests; no market data or stock execution."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from src.agents.timing_policy import CalendarTimingPolicy
from tests.archive.test_timing_policy import SyntheticTimingEnv
from tools.archive.timing_phase_policy import DuePhaseTimingPolicy, mapped_baseline_state_dict


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def policy(kind, seed=42):
    torch.manual_seed(seed)
    return kind(gym.spaces.Box(-np.inf, np.inf, (129,), np.float32), gym.spaces.Discrete(2), lambda _: 3e-4)


def observations():
    values = torch.arange(20 * 109, dtype=torch.float32).reshape(20, 109).sin()
    return torch.cat((torch.eye(20), values), dim=1)


def test_zero_residual_preserves_shared_tensors_rng_probabilities_and_sampled_actions():
    base = policy(CalendarTimingPolicy)
    base_rng = torch.get_rng_state().clone()
    candidate = policy(DuePhaseTimingPolicy)
    assert torch.equal(base_rng, torch.get_rng_state())
    expected, actual = mapped_baseline_state_dict(base), mapped_baseline_state_dict(candidate)
    assert set(expected) == set(actual)
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    assert base.parameter_count == 12805 and candidate.parameter_count == 12914
    assert torch.count_nonzero(candidate.action_net.residual.weight) == 0
    vector = observations()
    assert torch.equal(base.action_net(vector), candidate.action_net(vector))
    assert torch.equal(base.get_distribution(vector).distribution.probs, candidate.get_distribution(vector).distribution.probs)
    for deterministic in (False, True):
        torch.manual_seed(2026)
        expected_output = base(vector, deterministic=deterministic)
        torch.manual_seed(2026)
        actual_output = candidate(vector, deterministic=deterministic)
        assert all(torch.equal(left, right) for left, right in zip(expected_output, actual_output))


def test_every_sb3_actor_entry_uses_the_same_due_only_head():
    candidate = policy(DuePhaseTimingPolicy)
    vector = observations()
    with torch.no_grad():
        candidate.action_net.residual.weight.fill_(.1)
    logits = candidate.action_net(vector)
    original = candidate.action_net.base(vector)
    assert torch.equal(logits[1:], original[1:])
    assert logits[0, 1] != original[0, 1]
    distribution = candidate.get_distribution(vector)
    action, value, log_prob = candidate(vector, deterministic=True)
    evaluated_value, evaluated_log_prob, entropy = candidate.evaluate_actions(vector, action)
    assert torch.equal(value, evaluated_value)
    assert torch.equal(log_prob, evaluated_log_prob)
    assert torch.equal(log_prob, distribution.log_prob(action))
    assert torch.equal(entropy, distribution.entropy())


@pytest.mark.parametrize("due", [True, False])
def test_residual_gradient_is_gated_but_optimizer_registers_all_nine_tensors(due):
    candidate = policy(DuePhaseTimingPolicy)
    parameters = list(candidate.parameters())
    optimized = [p for group in candidate.optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len(optimized) == 9
    assert {id(p) for p in parameters} == {id(p) for p in optimized}
    assert not candidate.optimizer.state
    vector = observations()[[0 if due else 1]]
    candidate.get_distribution(vector).log_prob(torch.tensor([0])).sum().backward()
    gradient = candidate.action_net.residual.weight.grad
    assert gradient is not None
    assert bool(torch.count_nonzero(gradient)) is due


def test_tiny_nonfinancial_ppo_updates_new_tensor_and_rest_of_actor_critic():
    env = SyntheticTimingEnv()
    model = PPO(DuePhaseTimingPolicy, env, seed=42, device="cpu", n_steps=8, batch_size=4, n_epochs=2,
                gamma=.995, gae_lambda=1.)
    before = model.policy.action_net.residual.weight.detach().clone()
    try:
        model.learn(16)
        assert not torch.equal(before, model.policy.action_net.residual.weight)
        assert len(model.policy.optimizer.state) == 9
        assert all(int(state["step"]) == 8 for state in model.policy.optimizer.state.values())
        assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
    finally:
        env.close()


@pytest.mark.parametrize("dimension", [20, 128, 130])
def test_wrong_observation_width_is_rejected(dimension):
    with pytest.raises(ValueError, match="129D"):
        DuePhaseTimingPolicy(gym.spaces.Box(-np.inf, np.inf, (dimension,), np.float32), gym.spaces.Discrete(2), lambda _: 3e-4)


@pytest.mark.parametrize("margin", [True, .5, 1., float("nan")])
def test_margin_is_not_a_hidden_new_factor(margin):
    with pytest.raises(ValueError, match="margin"):
        DuePhaseTimingPolicy(gym.spaces.Box(-np.inf, np.inf, (129,), np.float32), gym.spaces.Discrete(2),
                             lambda _: 3e-4, calendar_margin=margin)


def test_invalid_phase_is_rejected_before_distribution():
    candidate = policy(DuePhaseTimingPolicy)
    vector = observations()
    vector[0, 0] = .5
    with pytest.raises(ValueError, match="one-hot"):
        candidate.get_distribution(vector)
