"""Regression contract for critic diagnostics captured before SB3 resets a rollout.

These synthetic checks target the deferred fix, not the frozen experiment source.
No model is trained and no market data is loaded.
"""

from time import perf_counter
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from src.training import research_matrix
from src.training.train import ValidationCallback


class FakePolicy:
    def obs_to_tensor(self, observations):
        return {name: torch.as_tensor(values) for name, values in observations.items()}, True

    def predict_values(self, observations):
        return observations["features"][:, 0, 0, 0].reshape(-1, 1)


class FakeBuffer:
    def __init__(self):
        self.returns = np.arange(1, 5, dtype=np.float32).reshape(4, 1)
        self.values = np.zeros((4, 1), dtype=np.float32)
        predictions = np.array([0, 2, 2, 4], dtype=np.float32)
        self.observations = {"features": predictions.reshape(4, 1, 1, 1, 1)}

    def reset(self):
        self.returns.fill(0)
        self.values.fill(0)
        for values in self.observations.values():
            values.fill(0)


def fake_model():
    return SimpleNamespace(
        num_timesteps=4,
        batch_size=2,
        policy=FakePolicy(),
        observation_space={"features": gym.spaces.Box(-np.inf, np.inf, (1, 1, 1))},
        rollout_buffer=FakeBuffer(),
    )


def copy_rollout(buffer):
    return SimpleNamespace(
        returns=buffer.returns.copy(),
        values=buffer.values.copy(),
        observations={name: values.copy() for name, values in buffer.observations.items()},
    )


def assert_real_rollout_fit(result):
    assert result["critic/target_std"] == pytest.approx(np.sqrt(1.25))
    assert result["critic/old_value_mse"] == pytest.approx(7.5)
    assert result["critic/post_update_mse"] == pytest.approx(0.5)
    assert result["critic/post_update_pred_std"] == pytest.approx(np.sqrt(2))
    assert result["critic/post_update_explained_variance"] == pytest.approx(0.8)
    assert result["critic/post_update_normalized_mse"] == pytest.approx(0.4)


def make_callback(model, output):
    data = SimpleNamespace(config={"validation_rollouts": 1, "ppo": {"n_steps": 4}}, codes=["000001"])
    callback = research_matrix.MatrixCallback(
        data, output, SimpleNamespace(stats=lambda: {}), {},
        {"short": 4, "long": 8}, output / "status.json", "synthetic",
    )
    callback.model = model
    callback.rollout_started = perf_counter()
    return callback


def test_critic_fit_explicit_buffer_ignores_cleared_live_rollout():
    model = fake_model()
    captured = copy_rollout(model.rollout_buffer)
    model.rollout_buffer.reset()

    result = research_matrix.critic_fit(model, buffer=captured)

    assert_real_rollout_fit(result)
    assert not model.rollout_buffer.returns.any()
    assert not model.rollout_buffer.observations["features"].any()
    assert research_matrix.critic_fit(model)["critic/post_update_explained_variance"] is None


def test_rollout_end_cache_survives_reset_and_record_update_uses_it(tmp_path, monkeypatch):
    model = fake_model()
    callback = make_callback(model, tmp_path)
    expected = copy_rollout(model.rollout_buffer)

    callback._on_rollout_end()
    captured = callback.critic_rollout

    assert isinstance(captured, SimpleNamespace)
    assert not np.shares_memory(captured.returns, model.rollout_buffer.returns)
    assert not np.shares_memory(captured.values, model.rollout_buffer.values)
    for name in expected.observations:
        assert not np.shares_memory(captured.observations[name], model.rollout_buffer.observations[name])
    model.rollout_buffer.reset()
    np.testing.assert_array_equal(captured.returns, expected.returns)
    np.testing.assert_array_equal(captured.values, expected.values)
    np.testing.assert_array_equal(captured.observations["features"], expected.observations["features"])

    def record_parent_update(self):
        self.update_records.append(dict(timesteps=model.num_timesteps, updates=1))

    calls = []
    fit = research_matrix.critic_fit

    def inspect_buffer(passed_model, buffer=None):
        assert passed_model is model
        assert buffer is captured
        calls.append(buffer)
        return fit(passed_model, buffer=buffer)

    monkeypatch.setattr(ValidationCallback, "record_update", record_parent_update)
    monkeypatch.setattr(research_matrix, "critic_fit", inspect_buffer)
    callback.record_update()

    assert calls == [captured]
    assert_real_rollout_fit(callback.update_records[-1])
    assert (tmp_path / "training_updates.csv").exists()
    assert (tmp_path / "status.json").exists()


def test_rollout_end_replaces_cache_with_the_latest_real_rollout(tmp_path):
    model = fake_model()
    callback = make_callback(model, tmp_path)
    callback._on_rollout_end()
    first = callback.critic_rollout

    model.rollout_buffer.returns += 10
    model.rollout_buffer.observations["features"] += 10
    callback._on_rollout_end()
    latest = callback.critic_rollout
    model.rollout_buffer.reset()

    assert latest is not first
    np.testing.assert_array_equal(first.returns.reshape(-1), [1, 2, 3, 4])
    np.testing.assert_array_equal(latest.returns.reshape(-1), [11, 12, 13, 14])
    result = research_matrix.critic_fit(model, buffer=latest)
    assert result["critic/post_update_mse"] == pytest.approx(0.5)
    assert result["critic/post_update_explained_variance"] == pytest.approx(0.8)
