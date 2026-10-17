"""Pure synthetic SB3 engineering checks; no financial dataset or live study."""

from copy import deepcopy
import random

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.timing_policy import CalendarTimingPolicy
from tools import critic_initialization_probe as tool


class SyntheticTiming(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (129,), np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.step_index = 0

    def observation(self):
        result = np.zeros(129, dtype=np.float32)
        result[self.step_index % 20] = 1
        result[20:] = np.sin(np.arange(109) * .07 + self.step_index * .031).astype(np.float32)
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_index = 0
        return self.observation(), {}

    def step(self, action):
        reward = float(.004 * np.cos(self.step_index * .13) + .002 * (2 * int(action) - 1))
        self.step_index += 1
        return self.observation(), reward, False, self.step_index == 96, dict(raw_reward=reward)


def make_model():
    torch.set_num_threads(1)
    env = DummyVecEnv([SyntheticTiming])
    model = PPO(CalendarTimingPolicy, env, seed=42, device="cpu", verbose=0,
        policy_kwargs=dict(calendar_margin=2., net_arch={"pi": [], "vf": [64, 64]}),
        n_steps=512, batch_size=128, n_epochs=2, gamma=.995, gae_lambda=1.,
        learning_rate=3e-4, target_kl=None)
    return model, env


def rng_state():
    return random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone()


def assert_rng_equal(left, right):
    assert left[0] == right[0]
    assert left[1][0] == right[1][0] and left[1][2:] == right[1][2:]
    np.testing.assert_array_equal(left[1][1], right[1][1])
    assert torch.equal(left[2], right[2])


def assert_tree_equal(left, right):
    assert type(left) is type(right)
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_tree_equal(first, second)
    else:
        assert left == right


@pytest.mark.parametrize("gain", [1., .01, 0.])
def test_fresh_head_scaling_is_once_rng_free_and_only_changes_final_head(gain):
    model, env = make_model()
    try:
        before = {name: tensor.detach().clone() for name, tensor in model.policy.state_dict().items()}
        rng = rng_state()
        record = tool.initialize_critic_head(model, gain)
        assert_rng_equal(rng, rng_state())
        assert len(record["before_tensor_sha256"]) == len(record["after_tensor_sha256"]) == 8
        assert record["only_value_head_may_change"] is True
        for name, value in model.policy.state_dict().items():
            expected = before[name] * gain if name in {"value_net.weight", "value_net.bias"} else before[name]
            assert torch.equal(value, expected)
            assert record["before_tensor_sha256"][name] == tool.tensor_sha256(before[name])
            assert record["after_tensor_sha256"][name] == tool.tensor_sha256(value)
        with pytest.raises(ValueError, match="already"):
            tool.initialize_critic_head(model, gain)
    finally:
        env.close()


@pytest.mark.parametrize("gain", [True, False, -1., .1, 2., np.float32(.01), "0", float("nan")])
def test_initializer_rejects_unregistered_gain_without_touching_weights(gain):
    model, env = make_model()
    try:
        before = deepcopy(model.policy.state_dict())
        with pytest.raises(ValueError):
            tool.initialize_critic_head(model, gain)
        assert_tree_equal(before, model.policy.state_dict())
    finally:
        env.close()


@pytest.mark.parametrize("mutation", ["clock", "epochs", "buffer_pos", "buffer_full", "gradient", "adam"])
def test_initializer_rejects_nonfresh_model(mutation):
    model, env = make_model()
    try:
        if mutation == "clock":
            model.num_timesteps = 1
        elif mutation == "epochs":
            model._n_updates = 1
        elif mutation == "buffer_pos":
            model.rollout_buffer.pos = 1
        elif mutation == "buffer_full":
            model.rollout_buffer.full = True
        elif mutation == "gradient":
            next(model.policy.parameters()).grad = torch.ones_like(next(model.policy.parameters()))
        else:
            model.policy.optimizer.state[next(model.policy.parameters())] = {"step": torch.tensor(1.)}
        with pytest.raises(ValueError, match="fresh"):
            tool.initialize_critic_head(model, .01)
    finally:
        env.close()


def test_probe_preserves_original_order_kwargs_return_and_gradients(monkeypatch):
    model, env = make_model()
    original = torch.nn.utils.clip_grad_norm_
    calls = []
    def observe_original(parameters, *args, **kwargs):
        ordered = list(parameters)
        calls.append((tuple(id(p) for p in ordered), args, kwargs))
        return original(ordered, *args, **kwargs)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", observe_original)
    try:
        for i, parameter in enumerate(model.policy.parameters()):
            parameter.grad = torch.full_like(parameter, (i + 1) * .03)
        expected = [parameter.grad.clone() for parameter in model.policy.parameters()]
        standalone = [torch.nn.Parameter(torch.zeros_like(value)) for value in expected]
        for parameter, value in zip(standalone, expected):
            parameter.grad = value.clone()
        expected_return = original(standalone, max_norm=.5, norm_type=2., error_if_nonfinite=True, foreach=False)
        rng = rng_state()
        with tool.ClipGradientProbe(model) as probe:
            returned = torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=.5,
                norm_type=2., error_if_nonfinite=True, foreach=False)
        assert_rng_equal(rng, rng_state())
        assert torch.equal(returned, expected_return)
        assert calls[0][0] == tuple(id(p) for p in model.policy.parameters())
        assert calls[0][1] == () and calls[0][2] == dict(max_norm=.5, norm_type=2., error_if_nonfinite=True, foreach=False)
        assert torch.nn.utils.clip_grad_norm_ is observe_original
        for actual, expected in zip(model.policy.parameters(), standalone):
            assert torch.equal(actual.grad, expected.grad)
        assert probe.records[0]["after"]["global_norm"] <= .500001
        assert len(probe.gradient_snapshots) == 1
    finally:
        env.close()


def test_probe_restores_original_after_error_and_rejects_nested_installation():
    model, env = make_model()
    original = torch.nn.utils.clip_grad_norm_
    try:
        with pytest.raises(RuntimeError, match="synthetic failure"):
            with tool.ClipGradientProbe(model):
                with pytest.raises(ValueError, match="one scoped"):
                    with tool.ClipGradientProbe(model):
                        pass
                raise RuntimeError("synthetic failure")
        assert torch.nn.utils.clip_grad_norm_ is original
    finally:
        env.close()


@pytest.mark.parametrize("norm_type", [1., float("inf")])
def test_probe_preserves_positional_arguments_and_non_l2_native_clipping(norm_type):
    model, env = make_model()
    original = torch.nn.utils.clip_grad_norm_
    try:
        expected = []
        for index, parameter in enumerate(model.policy.parameters()):
            parameter.grad = torch.full_like(parameter, (index + 1) * .03)
            other = torch.nn.Parameter(torch.zeros_like(parameter))
            other.grad = parameter.grad.clone()
            expected.append(other)
        expected_return = original(expected, .5, norm_type, True, False)
        with tool.ClipGradientProbe(model) as probe:
            returned = torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5, norm_type, True, False)
        assert torch.equal(returned, expected_return)
        assert probe.records[0]["norm_type"] == norm_type
        assert probe.records[0]["error_if_nonfinite"] is True
        assert probe.records[0]["foreach"] is False
        for actual, other in zip(model.policy.parameters(), expected):
            assert torch.equal(actual.grad, other.grad)
    finally:
        env.close()


def test_probe_forwards_original_nonfinite_error_and_restores_hook():
    model, env = make_model()
    original = torch.nn.utils.clip_grad_norm_
    try:
        next(model.policy.parameters()).grad = torch.full_like(next(model.policy.parameters()), float("nan"))
        probe = tool.ClipGradientProbe(model)
        with pytest.raises(RuntimeError, match="non-finite"):
            with probe:
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5, error_if_nonfinite=True)
        assert torch.nn.utils.clip_grad_norm_ is original
        assert probe.records[0]["status"] == "raised"
        assert probe.records[0]["error_type"] == "RuntimeError"
        assert probe.summary()["completed_calls"] == 0
        assert probe.gradient_snapshots == []
    finally:
        env.close()


def test_probe_passes_foreign_parameter_calls_through_without_claiming_evidence():
    model, env = make_model()
    original = torch.nn.utils.clip_grad_norm_
    try:
        foreign = torch.nn.Parameter(torch.zeros(3))
        foreign.grad = torch.tensor([1., 2., 3.])
        expected = torch.nn.Parameter(torch.zeros(3))
        expected.grad = foreign.grad.clone()
        expected_return = original(expected, .5)
        with tool.ClipGradientProbe(model) as probe:
            returned = torch.nn.utils.clip_grad_norm_(foreign, .5)
        assert torch.equal(returned, expected_return)
        assert torch.equal(foreign.grad, expected.grad)
        assert probe.foreign_clip_calls == 1
        assert probe.records == probe.gradient_snapshots == []
    finally:
        env.close()


class Capture(BaseCallback):
    def __init__(self):
        super().__init__()
        self.raw = []
        self.buffers = []

    def _on_step(self):
        self.raw.append(dict(observation=self.model._last_obs.copy(), actions=self.locals["actions"].copy(),
            raw_reward=np.asarray([info["raw_reward"] for info in self.locals["infos"]]),
            log_prob=self.locals["log_probs"].detach().cpu().numpy().copy(), dones=self.locals["dones"].copy()))
        return True

    def _on_rollout_end(self):
        self.buffers.append({name: getattr(self.model.rollout_buffer, name).copy() for name in
            ("observations", "actions", "rewards", "values", "log_probs", "returns", "advantages", "episode_starts")})


def train(gain, instrument):
    model, env = make_model()
    before = deepcopy(model.policy.state_dict())
    initialization = tool.initialize_critic_head(model, gain)
    initial = deepcopy(model.policy.state_dict())
    capture, probe = Capture(), tool.ClipGradientProbe(model)
    try:
        if instrument:
            with probe:
                model.learn(total_timesteps=1024, callback=capture)
        else:
            model.learn(total_timesteps=1024, callback=capture)
        return dict(before=before, initial=initial, initialization=initialization,
            parameters=deepcopy(model.policy.state_dict()), optimizer=deepcopy(model.policy.optimizer.state_dict()),
            rng=rng_state(), raw=capture.raw, buffers=capture.buffers, probe=probe)
    finally:
        env.close()


@pytest.fixture(scope="module")
def trained():
    return {"plain": train(1., False), "instrumented": train(1., True),
            "small": train(.01, True), "zero": train(0., True)}


def test_plain_and_instrumented_real_sb3_training_are_bitwise_identical(trained):
    plain, observed = trained["plain"], trained["instrumented"]
    for key in ("before", "initial", "parameters", "optimizer", "raw", "buffers"):
        assert_tree_equal(plain[key], observed[key])
    assert_rng_equal(plain["rng"], observed["rng"])
    probe = observed["probe"]
    assert len(probe.records) == 16
    assert [row["call_index"] for row in probe.records] == list(range(1, 17))
    assert [snapshot["rollout_clock"] for snapshot in probe.gradient_snapshots] == [512, 1024]
    assert sum(row["first_minibatch_of_observed_rollout"] for row in probe.records) == 2
    assert probe.summary()["new_forward_calls"] == probe.summary()["new_backward_calls"] == 0
    assert probe.foreign_clip_calls == 0


def test_gain_variants_share_first_512_actual_collector_inputs_actions_logp_and_raw_reward(trained):
    reference = trained["plain"]
    for name in ("small", "zero"):
        other = trained[name]
        assert_tree_equal(reference["before"], other["before"])
        for parameter in reference["initial"]:
            if not parameter.startswith("value_net."):
                assert torch.equal(reference["initial"][parameter], other["initial"][parameter])
        assert_tree_equal(reference["raw"][:512], other["raw"][:512])
        for key in ("observations", "actions", "log_probs", "episode_starts"):
            np.testing.assert_array_equal(reference["buffers"][0][key], other["buffers"][0][key])
        assert not np.array_equal(reference["buffers"][0]["values"], other["buffers"][0]["values"])
        assert not np.array_equal(reference["buffers"][0]["advantages"], other["buffers"][0]["advantages"])
    assert np.count_nonzero(trained["zero"]["buffers"][0]["values"]) == 0


def test_zero_head_first_clip_blocks_hidden_critic_gradients_but_later_hidden_layers_update(trained):
    result = trained["zero"]
    first, later = result["probe"].gradient_snapshots
    hidden = [name for name in first["before"] if name.startswith("mlp_extractor.value_net.")]
    assert len(hidden) == 4
    assert all(torch.count_nonzero(first["before"][name]) == 0 for name in hidden)
    assert torch.count_nonzero(first["before"]["value_net.weight"]) > 0
    assert any(torch.count_nonzero(later["before"][name]) > 0 for name in hidden)
    assert all(not torch.equal(result["initial"][name], result["parameters"][name]) for name in hidden)
    states = result["optimizer"]["state"]
    assert len(states) == 8 and {int(item["step"].item()) for item in states.values()} == {16}
