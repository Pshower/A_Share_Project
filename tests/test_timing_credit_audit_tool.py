"""Pure estimator arrays and identical-parameter checks, never model execution."""

from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3.common.buffers import RolloutBuffer

from tools import audit_timing_credit as audit


def sample(gae_lambda, final_done=False):
    native_rewards = np.array([.01, -.02, .03, .005], dtype=np.float64).reshape(4, 1)
    context = dict(gamma=np.array(.995), gae_lambda=np.array(gae_lambda),
        last_values=np.array([.7], dtype=np.float32), last_dones=np.array([final_done]),
        dones=np.array([False, True, False, final_done]).reshape(4, 1),
        truncated=np.array([False, True, False, False]).reshape(4, 1),
        terminal_values=np.array([0., .3, 0., 0.], dtype=np.float32).reshape(4, 1),
        raw_rewards=native_rewards.astype(np.float32), native_learning_rewards=native_rewards)
    rewards = context["raw_rewards"].copy()
    rewards[1] += np.float32(.995) * context["terminal_values"][1]
    values = np.array([.2, .4, -.1, .6], dtype=np.float32).reshape(4, 1)
    starts = np.array([1., 0., 1., 0.], dtype=np.float32).reshape(4, 1)
    # Exercise the real library recurrence without any environment or policy.
    library = RolloutBuffer(4, gym.spaces.Box(-1., 1., (1,)), gym.spaces.Discrete(2),
                            gamma=.995, gae_lambda=gae_lambda, n_envs=1, device="cpu")
    library.rewards, library.values, library.episode_starts = rewards, values, starts
    library.compute_returns_and_advantage(torch.from_numpy(context["last_values"]), context["last_dones"])
    buffer = dict(rewards=rewards, values=values, episode_starts=starts,
                  advantages=library.advantages, returns=library.returns)
    native = dict(done=context["dones"].reshape(-1), action_learning_reward=native_rewards.reshape(-1))
    return buffer, context, native


@pytest.mark.parametrize("value", [.95, 1.])
@pytest.mark.parametrize("final_done", [False, True])
def test_all_returns_match_actual_library_including_tail_and_timeout(value, final_done):
    buffer, context, native = sample(value, final_done)
    result = audit.estimator_arrays(buffer, context, native, value)
    assert result["steps"] == 4 and result["timeout_bootstraps"] == 1
    assert result["natural_terminals"] == int(final_done)
    assert result["maximum_advantage_error"] == result["maximum_return_error"] == 0.
    expected_last = buffer["rewards"][-1] - buffer["values"][-1]
    if not final_done:
        expected_last += np.float32(.995) * context["last_values"]
    np.testing.assert_allclose(buffer["advantages"][-1], expected_last)


def test_fabricated_tail_value_rejected_without_network():
    buffer, context, native = sample(1.)
    context["last_values"] += .1
    with pytest.raises(ValueError, match="GAE recurrence"):
        audit.estimator_arrays(buffer, context, native, 1.)


def test_nonterminal_bootstrap_reward_rejected():
    buffer, context, native = sample(.95)
    buffer["rewards"][0] += .1
    with pytest.raises(ValueError, match="terminal bootstrap"):
        audit.estimator_arrays(buffer, context, native, .95)


def test_return_identity_alone_does_not_prove_credit_estimator():
    buffer, context, native = sample(.95)
    buffer["advantages"][0] += .1
    buffer["returns"][0] += .1
    np.testing.assert_allclose(buffer["returns"], buffer["advantages"] + buffer["values"])
    with pytest.raises(ValueError, match="GAE recurrence"):
        audit.estimator_arrays(buffer, context, native, .95)


def test_wrong_actual_lambda_and_episode_reset_rejected():
    buffer, context, native = sample(.95)
    with pytest.raises(ValueError, match="gamma/GAE"):
        audit.estimator_arrays(buffer, context, native, 1.)
    buffer["episode_starts"][2] = 0.
    with pytest.raises(ValueError, match="Episode boundary"):
        audit.estimator_arrays(buffer, context, native, .95)


def test_fractional_episode_start_is_not_a_boolean_boundary():
    buffer, context, native = sample(1.)
    buffer["episode_starts"][2] = .5
    with pytest.raises(ValueError, match="Episode boundary"):
        audit.estimator_arrays(buffer, context, native, 1.)


def test_four_cells_require_identical_actor_as_well_as_critic():
    theta = {"actor": torch.tensor([2.]), "critic": torch.tensor([.3])}
    cells = {(reward, value): deepcopy(theta) for reward, value in audit.runner.CELLS}
    actual = audit.shared_initials({("2023", "screen", 42, cell): value for cell, value in cells.items()})
    assert actual["checked_cells"] == 4 and actual["full_initial_actor_and_critic_equal"]
    cells["log_net", 1.]["actor"] += .1
    with pytest.raises(ValueError, match="Actor or Critic"):
        audit.shared_initials({("2023", "screen", 42, cell): value for cell, value in cells.items()})


def test_actual_callback_sidecar_feeds_independent_recurrence_without_predictions(tmp_path):
    buffer, context, native = sample(1.)
    original = {**buffer, "observations": np.zeros((4, 1, 129), dtype=np.float32),
                "actions": np.zeros((4, 1, 1), dtype=np.float32),
                "log_probs": np.full((4, 1), -.5, dtype=np.float32)}

    def forbidden_prediction(*_args, **_kwargs):
        raise AssertionError("The callback must not request additional predictions")

    callback = object.__new__(audit.runner.CreditCallback)
    callback.model = SimpleNamespace(n_steps=4, num_timesteps=4,
        rollout_buffer=SimpleNamespace(**original, gamma=.995, gae_lambda=1.),
        policy=SimpleNamespace(predict_values=forbidden_prediction))
    callback.data = SimpleNamespace(config={"ppo": {"gae_lambda": 1.},
                                            "timing": {"reward_basis": "log_net"}})
    callback.output, callback.rollouts, callback.estimator_buffers = tmp_path, [], []
    callback.boundary_rows = [dict(raw_reward=context["raw_rewards"][i, 0],
        native_learning_reward=float(context["native_learning_rewards"][i, 0]),
        done=bool(context["dones"][i, 0]), truncated=bool(context["truncated"][i, 0]),
        terminal_value=np.float32(0.)) for i in range(4)]
    callback.pending_timeout = 1
    callback.locals = dict(terminal_value=torch.tensor(.3), values=torch.from_numpy(context["last_values"]),
                          dones=context["last_dones"])
    callback._on_rollout_end()
    with np.load(tmp_path / callback.estimator_buffers[0]["path"], allow_pickle=False) as saved:
        actual_context = {key: saved[key] for key in saved.files}
    with np.load(tmp_path / callback.rollouts[0]["path"], allow_pickle=False) as saved:
        assert set(saved.files) == set(original)
        actual_buffer = {key: saved[key] for key in saved.files}
    assert all(np.array_equal(actual_buffer[key], value) for key, value in original.items())
    assert callback.pending_timeout is None
    assert audit.estimator_arrays(actual_buffer, actual_context, native, 1.)["maximum_advantage_error"] == 0.
    trace = tmp_path / "training_trace"
    trace.mkdir()
    native_path = trace / "steps_001_004.npz"
    np.savez_compressed(native_path, **native)
    update = {"timesteps": 4, "credit/gamma": .995, "credit/gae_lambda": 1.,
              **{"credit/" + name + "_std": float(np.std(original[field])) for name, field in (
                  ("returns", "returns"), ("advantages", "advantages"), ("buffer_rewards", "rewards"))}}
    pd.DataFrame([update]).to_csv(tmp_path / "training_updates.csv", index=False)
    evidence = dict(actual_model_gamma=.995, actual_buffer_gamma=.995,
        actual_model_gae_lambda=1., actual_buffer_gae_lambda=1., credit_estimator_buffers=callback.estimator_buffers,
        rollout_buffers=callback.rollouts,
        training_trace_prefix={"chunks": [dict(path=native_path.name, first_step=1, last_step=4, steps=4)]})
    assert audit.estimator_audit(tmp_path, evidence, {"ppo": {"gae_lambda": 1.}})["steps"] == 4
    update["credit/gae_lambda"] = .95
    pd.DataFrame([update]).to_csv(tmp_path / "training_updates.csv", index=False)
    with pytest.raises(ValueError, match="update logs"):
        audit.estimator_audit(tmp_path, evidence, {"ppo": {"gae_lambda": 1.}})
