"""Pure vector/controller checks and tiny non-financial CPU PPO updates."""

from copy import deepcopy
import math

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import CategoricalDistribution

from src.agents.timing_policy import CalendarTimingPolicy, PHASE_PERIOD, TIMING_NET_ARCH


class SyntheticTimingEnv(gym.Env):
    """Synthetic vectors and binary rewards; no stocks, market data or broker."""

    def __init__(self, dimension=129):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (dimension,), np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.steps = 0

    def observation(self):
        vector = np.zeros(self.observation_space.shape, dtype=np.float32)
        vector[self.steps % PHASE_PERIOD] = 1.
        if len(vector) > PHASE_PERIOD:
            vector[PHASE_PERIOD:] = np.sin(self.steps + np.arange(len(vector) - PHASE_PERIOD))
        return vector

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self.observation(), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Synthetic controller action must be 0 or 1")
        target = int(self.observation()[-1] > 0)
        reward = 1. if int(action) == target else -1.
        self.steps += 1
        return self.observation(), reward, False, self.steps >= 32, {}


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def make_policy(dimension=129, margin=2., **kwargs):
    return CalendarTimingPolicy(gym.spaces.Box(-np.inf, np.inf, (dimension,), np.float32),
        gym.spaces.Discrete(2), lambda _: 3e-4, calendar_margin=margin, **kwargs)


def phase_batch(dimension=129):
    torch.manual_seed(17)
    observations = torch.randn(PHASE_PERIOD, dimension)
    observations[:, :PHASE_PERIOD] = torch.eye(PHASE_PERIOD)
    return observations


@pytest.mark.parametrize("dimension", [20, 23, 109, 129])
def test_one_time_actor_initialization_matches_calendar_for_every_phase(dimension):
    policy = make_policy(dimension)
    observation = phase_batch(dimension)
    expected = torch.zeros(PHASE_PERIOD, dtype=torch.long)
    expected[0] = 1
    actions, values, log_probability = policy(observation, deterministic=True)
    torch.testing.assert_close(actions, expected, rtol=0, atol=0)
    assert values.shape == (PHASE_PERIOD, 1) and log_probability.shape == (PHASE_PERIOD,)
    assert isinstance(policy.action_dist, CategoricalDistribution)
    assert policy.net_arch == TIMING_NET_ARCH
    assert policy.parameter_count == 66 * dimension + 4291
    assert policy.action_net.weight.requires_grad and policy.action_net.bias.requires_grad
    assert torch.count_nonzero(policy.action_net.bias) == 0
    assert torch.count_nonzero(policy.action_net.weight[:, PHASE_PERIOD:]) == 0


def test_initial_probabilities_are_finite_categorical_not_stock_gaussian():
    policy = make_policy()
    observations = phase_batch()
    actions, _, probability = policy(observations, deterministic=True)
    distribution = policy.get_distribution(observations)
    expected = 1 / (1 + math.exp(-2))
    assert distribution.distribution.probs[0, 1].item() == pytest.approx(expected)
    assert distribution.distribution.probs[1, 0].item() == pytest.approx(expected)
    torch.testing.assert_close(probability, distribution.log_prob(actions), rtol=0, atol=0)
    assert torch.isfinite(distribution.entropy()).all()
    assert not hasattr(policy, "log_std")


def test_forward_and_evaluate_actions_retain_sampled_macro_log_probability():
    torch.manual_seed(42)
    policy = make_policy()
    observations = phase_batch()
    actions, values, old_probability = policy(observations)
    new_values, probability, entropy = policy.evaluate_actions(observations, actions)
    torch.testing.assert_close(values, new_values, rtol=0, atol=0)
    torch.testing.assert_close(probability, old_probability, rtol=0, atol=0)
    assert actions.shape == (PHASE_PERIOD,) and ((actions == 0) | (actions == 1)).all()
    assert torch.isfinite(entropy).all()


def test_forward_does_not_reset_calendar_weights_or_add_persistent_prior():
    policy = make_policy()
    observation = phase_batch()[1:2]
    observation[:, PHASE_PERIOD] = 1.
    assert policy(observation, deterministic=True)[0].item() == 0
    with torch.no_grad():
        policy.action_net.weight[1, PHASE_PERIOD] = 3.
        policy.action_net.weight[0, 1] = .5
    before = policy.action_net.weight.clone()
    for _ in range(3):
        assert policy(observation, deterministic=True)[0].item() == 1
    torch.testing.assert_close(policy.action_net.weight, before, rtol=0, atol=0)


def test_calendar_actor_is_seed_invariant_but_critic_retains_normal_initialization():
    torch.manual_seed(42)
    first = make_policy()
    torch.manual_seed(123)
    second = make_policy()
    for name, value in first.action_net.state_dict().items():
        torch.testing.assert_close(value, second.action_net.state_dict()[name], rtol=0, atol=0)
    assert not torch.equal(first.value_net.weight, second.value_net.weight)
    assert torch.count_nonzero(first.value_net.weight) > 0
    assert all(parameter.requires_grad for parameter in first.parameters())


@pytest.mark.parametrize("corruption", ["zero", "two", "fraction", "nan", "wrong_shape"])
def test_invalid_phase_vectors_fail_closed(corruption):
    policy = make_policy()
    observation = phase_batch()[:1]
    if corruption == "zero":
        observation[:, :20] = 0
    elif corruption == "two":
        observation[0, 1] = 1
    elif corruption == "fraction":
        observation[0, :2] = .5
    elif corruption == "nan":
        observation[0, -1] = float("nan")
    else:
        observation = observation[:, :-1]
    with pytest.raises(ValueError):
        policy(observation)


@pytest.mark.parametrize("bad", [-1., .5, 2., float("nan"), float("inf")])
def test_invalid_macro_codes_are_not_given_finite_log_probability(bad):
    policy = make_policy()
    with pytest.raises(ValueError, match="integer code"):
        policy.evaluate_actions(phase_batch()[:1], torch.tensor([bad]))


@pytest.mark.parametrize("margin", [0, -1, float("nan"), float("inf"), True, "2"])
def test_bad_initial_margins_are_rejected(margin):
    with pytest.raises(ValueError, match="calendar_margin"):
        make_policy(margin=margin)


@pytest.mark.parametrize("space", [gym.spaces.Discrete(3), gym.spaces.Discrete(2, start=1),
    gym.spaces.Box(-1., 1., (2,), np.float32)])
def test_wrong_action_spaces_are_rejected(space):
    with pytest.raises(ValueError, match="Discrete"):
        CalendarTimingPolicy(gym.spaces.Box(-np.inf, np.inf, (129,), np.float32), space, lambda _: 3e-4)


@pytest.mark.parametrize("space", [gym.spaces.Box(-1., 1., (19,), np.float32),
    gym.spaces.Box(-1., 1., (129,), np.float64), gym.spaces.Box(-1., 1., (1, 129), np.float32)])
def test_wrong_observation_spaces_are_rejected(space):
    with pytest.raises(ValueError, match="vector Box"):
        CalendarTimingPolicy(space, gym.spaces.Discrete(2), lambda _: 3e-4)


def test_nonregistered_architecture_is_rejected():
    with pytest.raises(ValueError, match="linear pi"):
        make_policy(net_arch=dict(pi=[64], vf=[64, 64]))


def test_tiny_synthetic_cpu_update_changes_both_actor_and_critic():
    env = SyntheticTimingEnv()
    model = PPO(CalendarTimingPolicy, env, seed=42, device="cpu", n_steps=8, batch_size=4,
                n_epochs=2, learning_rate=3e-3)
    before = deepcopy(model.policy.state_dict())
    model.learn(total_timesteps=16)
    assert model.num_timesteps == 16
    assert not torch.equal(model.policy.action_net.weight, before["action_net.weight"])
    assert not torch.equal(model.policy.value_net.weight, before["value_net.weight"])
    assert len(model.policy.optimizer.state) == 8
    steps = {int(state["step"].item()) for state in model.policy.optimizer.state.values()}
    assert steps == {8}
    assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
    env.close()
