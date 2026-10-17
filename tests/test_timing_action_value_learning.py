"""Synthetic-only action-value fitting and ordinary-Actor transfer checks."""

from copy import deepcopy
import json

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from src.agents.timing_policy import CalendarTimingPolicy
from tools import timing_action_value_learning as learning


def sample(count=240, seed=7):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(count, 129))
    features[:, :20] = np.eye(20)[np.arange(count) % 20]
    features[:, -1] = 3.0
    targets = (0.003 * features[:, 20] - 0.002 * features[:, 21]
               + 0.001 * features[:, 0] + rng.normal(0, 0.0001, count))
    return features, targets


class SyntheticTimingEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (129,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observation = np.zeros(129, dtype=np.float32)
        observation[0] = 1
        return observation, {}

    def step(self, action):
        raise AssertionError("Synthetic transfer tests do not collect trajectories")


def model(policy=CalendarTimingPolicy, **policy_kwargs):
    return PPO(policy, SyntheticTimingEnv(), seed=42, device="cpu", n_steps=8,
        batch_size=4, n_epochs=1, learning_rate=3e-4, policy_kwargs=policy_kwargs)


def test_ridge_matches_independent_augmented_normal_equations():
    X, y = sample()
    fit = learning.fit_action_value(X, y)
    Z = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1, X.std(axis=0))
    design = np.column_stack((Z, np.ones(len(X))))
    penalty = np.diag([1.0] * 129 + [0.0])
    expected = np.linalg.solve(design.T @ design / len(X) + penalty,
                               design.T @ (y / np.sqrt(np.mean(y * y))) / len(X))
    np.testing.assert_allclose(fit["coef_normalized"], expected[:-1], atol=1e-12)
    assert fit["intercept_normalized"] == pytest.approx(expected[-1], abs=1e-12)
    assert fit["training"]["coefficient_gradient_max_abs"] < 1e-12
    assert fit["training"]["intercept_gradient_abs"] < 1e-12
    assert fit["counters"]["model_fits"] == 1
    assert fit["counters"]["training_label_exposures"] == len(X)
    assert fit["counters"]["neural_forwards"] == 0


def test_observation_coordinate_mapping_and_json_round_trip():
    X, y = sample()
    fit = json.loads(json.dumps(learning.fit_action_value(X, y)))
    q = learning.predict(fit, X, normalized=True)
    mapped = X @ np.array(fit["coef_observation"]) + fit["intercept_observation"]
    np.testing.assert_allclose(q, mapped, atol=1e-12)
    np.testing.assert_allclose(learning.predict(fit, X), q * fit["target_rms"])
    assert fit["constant_feature_mask"][-1] is True
    assert fit["x_scale"][-1] == 1


def test_fit_and_diagnostics_do_not_mutate_training_data_or_fit():
    X, y = sample()
    original_X, original_y = X.copy(), y.copy()
    fit = learning.fit_action_value(X, y)
    saved = deepcopy(fit)
    hold_X, hold_y = sample(40, seed=99)
    learning.diagnostics(fit, hold_X, hold_y)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)
    assert fit == saved
    assert fit["x_mean"] == original_X.mean(axis=0).tolist()


def test_inexact_binary_fraction_constant_column_is_still_exactly_constant():
    X, y = sample()
    X[:, -1] = 0.1
    fit = learning.fit_action_value(X, y)
    assert fit["constant_feature_mask"][-1] is True
    assert fit["x_mean"][-1] == 0.1
    assert fit["x_scale"][-1] == 1
    assert fit["coef_normalized"][-1] == 0


def test_intercept_not_penalized_and_zero_target_floor():
    X, _ = sample()
    fit = learning.fit_action_value(X, np.full(len(X), 0.02))
    np.testing.assert_allclose(learning.predict(fit, X), 0.02, atol=1e-14)
    np.testing.assert_allclose(fit["coef_normalized"], 0, atol=1e-14)
    zero = learning.fit_action_value(X, np.zeros(len(X)))
    assert zero["target_rms"] == 1e-8
    assert zero["raw_target_rms"] == 0
    np.testing.assert_array_equal(learning.predict(zero, X), np.zeros(len(X)))


def test_diagnostics_use_actual_deadband_greedy_decision_not_raw_sign():
    X, _ = sample()
    fit = learning.fit_action_value(X, np.full(len(X), 0.001))
    hold_X, _ = sample(40)
    hold_y = np.linspace(-0.02, 0.01, len(hold_X))
    result = learning.diagnostics(fit, hold_X, hold_y, np.arange(40) % 20)
    q = learning.predict(fit, hold_X, normalized=True)
    calendar = hold_X[:, 0] == 1
    decisions = 8 * (q + 0.25 * (2 * calendar.astype(float) - 1)) > 0
    expected = np.mean((decisions.astype(float) - calendar) * hold_y)
    assert result["mean_single_decision_calendar_difference"] == pytest.approx(expected)
    assert result["extra_rebalances"] == 38
    assert result["skipped_due_rebalances"] == 0
    assert sum(result["phase_counts"].values()) == 40
    assert result["strategy_roi"] is False
    assert result["on_policy_advantage"] is False
    assert result["actor_mapping"]["deadband_is_statistical_confidence"] is False
    assert result["counters"]["model_fits"] == 0


@pytest.mark.parametrize("damage", ["shape", "empty", "nonfinite", "complex", "phase_fraction", "phase_two", "phase_zero"])
def test_bad_features_rejected(damage):
    X, y = sample()
    if damage == "shape":
        X = X[:, :-1]
    elif damage == "empty":
        X, y = X[:0], y[:0]
    elif damage == "nonfinite":
        X[0, -1] = np.nan
    elif damage == "complex":
        X = X.astype(complex) + 1j
    elif damage == "phase_fraction":
        X[0, 0] = 0.5
    elif damage == "phase_two":
        X[0, 1] = 1
    else:
        X[0, :20] = 0
    with pytest.raises(ValueError):
        learning.fit_action_value(X, y)


@pytest.mark.parametrize("bad", [0, 0.1, 2, True, np.nan])
def test_other_alpha_not_silently_accepted(bad):
    X, y = sample()
    with pytest.raises(ValueError):
        learning.fit_action_value(X, y, alpha=bad)


@pytest.mark.parametrize("bad", ["short", "column", "nonfinite", "complex"])
def test_bad_targets_rejected(bad):
    X, y = sample()
    if bad == "short":
        y = y[:-1]
    elif bad == "column":
        y = y[:, None]
    elif bad == "complex":
        y = y.astype(complex) + 1j
    else:
        y[0] = np.inf
    with pytest.raises(ValueError):
        learning.fit_action_value(X, y)


def test_one_training_row_and_inconsistent_diagnostic_phase_rejected():
    X, y = sample()
    with pytest.raises(ValueError, match="two training rows"):
        learning.fit_action_value(X[:1], y[:1])
    fit = learning.fit_action_value(X, y)
    with pytest.raises(ValueError, match="phase"):
        learning.diagnostics(fit, X, y, (np.arange(len(X)) + 1) % 20)
    with pytest.raises(ValueError, match="explicit boolean"):
        learning.predict(fit, X, normalized=1)


def test_fit_tampering_is_rejected_before_prediction_or_transfer():
    X, y = sample()
    fit = learning.fit_action_value(X, y)
    fit["coef_observation"][20] += 1
    with pytest.raises(ValueError, match="SHA"):
        learning.predict(fit, X)
    learner = model()
    before = deepcopy(learner.policy.state_dict())
    with pytest.raises(ValueError, match="SHA"):
        learning.apply_actor_warmup(learner, fit)
    assert all(torch.equal(before[name], value) for name, value in learner.policy.state_dict().items())


def test_actor_transfer_matches_mapping_and_preserves_critic_optimizer_clocks_rng():
    X, y = sample()
    fit = learning.fit_action_value(X, y)
    learner = model()
    before = deepcopy(learner.policy.state_dict())
    rng_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()
    receipt = learning.apply_actor_warmup(learner, fit)
    expected = 8 * (learning.predict(fit, X, normalized=True) + 0.25 * (2 * X[:, 0] - 1))
    with torch.no_grad():
        logits = learner.policy.action_net(torch.tensor(X, dtype=torch.float32))
        margin = (logits[:, 1] - logits[:, 0]).numpy()
    np.testing.assert_allclose(margin, expected, atol=3e-6, rtol=2e-6)
    torch.testing.assert_close(logits[:, 0], -logits[:, 1], rtol=0, atol=0)
    for name, tensor in learner.policy.state_dict().items():
        if not name.startswith("action_net."):
            assert torch.equal(tensor, before[name])
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert receipt["critic_and_all_nonactor_tensors_bitwise_unchanged"] is True
    assert receipt["parameter_count"] == 12805
    assert receipt["ppo_optimizer_state_entries"] == 0
    assert receipt["ppo_num_timesteps"] == receipt["ppo_update_count"] == 0
    assert not learner.policy.optimizer.state
    assert receipt["counters"]["neural_forwards"] == 0


def test_zero_q_transfer_preserves_calendar_probabilities_not_tensor_gauge():
    X, _ = sample()
    fit = learning.fit_action_value(X, np.zeros(len(X)))
    learner = model()
    with torch.no_grad():
        before = torch.softmax(learner.policy.action_net(torch.tensor(X, dtype=torch.float32)), dim=1)
    learning.apply_actor_warmup(learner, fit)
    with torch.no_grad():
        after = torch.softmax(learner.policy.action_net(torch.tensor(X, dtype=torch.float32)), dim=1)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize("damage", ["clock", "updates", "buffer", "optimizer", "coverage", "gradients", "margin", "actor"])
def test_transfer_rejects_nonfresh_or_nonregistered_model_without_mutation(damage):
    X, y = sample()
    fit = learning.fit_action_value(X, y)
    learner = model(calendar_margin=3 if damage == "margin" else 2)
    if damage == "clock":
        learner.num_timesteps = 1
    elif damage == "updates":
        learner._n_updates = 1
    elif damage == "buffer":
        learner.rollout_buffer.pos = 1
    elif damage == "optimizer":
        learner.policy.optimizer.state[next(learner.policy.parameters())]["step"] = torch.tensor(1.0)
    elif damage == "coverage":
        learner.policy.optimizer.param_groups[0]["params"].pop()
    elif damage == "gradients":
        next(learner.policy.parameters()).grad = torch.zeros_like(next(learner.policy.parameters()))
    elif damage == "actor":
        with torch.no_grad():
            learner.policy.action_net.weight[0, 21] = 0.1
    before = deepcopy(learner.policy.state_dict())
    with pytest.raises(ValueError):
        learning.apply_actor_warmup(learner, fit)
    assert all(torch.equal(before[name], tensor) for name, tensor in learner.policy.state_dict().items())


def test_other_policy_rejected():
    X, y = sample()
    fit = learning.fit_action_value(X, y)
    with pytest.raises(ValueError, match="CalendarTimingPolicy"):
        learning.apply_actor_warmup(model("MlpPolicy"), fit)


def test_repeated_warmup_is_rejected_even_with_zero_ppo_steps():
    X, y = sample()
    fit = learning.fit_action_value(X, y)
    learner = model()
    learning.apply_actor_warmup(learner, fit)
    with pytest.raises(ValueError, match="original calendar Actor"):
        learning.apply_actor_warmup(learner, fit)


def test_ordinary_ppo_can_update_actor_and_critic_after_transfer_on_synthetic_buffer():
    X, y = sample()
    fit = learning.fit_action_value(X, y)
    learner = model()
    learning.apply_actor_warmup(learner, fit)
    learner.set_logger(configure(format_strings=[]))
    before = deepcopy(learner.policy.state_dict())
    observations = X[:8].astype(np.float32)
    actions = torch.arange(8) % 2
    with torch.no_grad():
        values, log_probabilities, _ = learner.policy.evaluate_actions(
            torch.tensor(observations), actions)
    for index in range(8):
        learner.rollout_buffer.add(observations[index:index + 1], actions[index:index + 1].numpy(),
            np.array([0.002 + 0.01 * (-1) ** index], dtype=np.float32),
            np.array([index == 0]), values[index:index + 1], log_probabilities[index:index + 1])
    learner.rollout_buffer.compute_returns_and_advantage(torch.tensor([0.01]), np.array([False]))
    learner.train()
    after = learner.policy.state_dict()
    assert any(not torch.equal(value, after[name]) for name, value in before.items() if name.startswith("action_net."))
    assert any(not torch.equal(value, after[name]) for name, value in before.items() if not name.startswith("action_net."))
    assert len(learner.policy.optimizer.state) == len(list(learner.policy.parameters())) == 8
    assert all(state["step"].item() == 2 for state in learner.policy.optimizer.state.values())
    assert all(torch.isfinite(value).all() for value in after.values())
