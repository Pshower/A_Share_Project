"""Artificial reward/GAE factorial checks; no real registration or market training."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from tools.archive import research_timing_credit as study
from src.agents.timing_agent import TimingAgent
from src.data.preprocess import read_config, file_hash
from src.envs.initial_account import MODES
from src.training.freeze import source_hashes
from tests.test_ppo_framework import local_data
from tests.test_research_timing import timing_config


@pytest.fixture(autouse=True)
def artificial_updates_only(request, monkeypatch):
    if request.node.name == "test_lambda_one_relative_reward_eight_artificial_steps_and_archive":
        return
    def forbidden(*args, **kwargs):
        raise AssertionError("Only the named eight-step artificial integration may optimize")
    monkeypatch.setattr(study.PPO, "learn", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def protocol():
    return dict(base_config={"ppo": {}}, folds={fold: {"research_config": "data.json"} for fold in ("2023", "2024", "2025", "2026")},
                tool_sha256="synthetic runner", helper_sha256="synthetic helper", reference_final_audit_sha256="synthetic audit")


def credit_config(local_data, reward="log_net", gae_lambda=.95):
    config = timing_config(local_data, reward)
    config["ppo"].update(learning_rate=3e-4, gamma=.995, gae_lambda=gae_lambda, n_steps=8, batch_size=4, n_epochs=1)
    config["credit_study"] = dict(study="timing_credit", reward_basis=reward, gae_lambda=gae_lambda,
        label=study.cell_label(reward, gae_lambda), protocol_sha256="synthetic", tool_sha256="synthetic runner",
        helper_sha256="synthetic helper", reference_audit_sha256="synthetic audit")
    return config


@pytest.mark.parametrize("reward,value", study.CELLS)
def test_only_reward_and_gae_change_not_actor_gamma_or_execution(tmp_path, reward, value):
    config = study.config_for(protocol(), tmp_path, "2023", reward, value, "screen", 42)
    assert config["ppo"]["gae_lambda"] == value and config["ppo"]["gamma"] == .995
    assert config["ppo"]["learning_rate"] == 3e-4 and config["ppo"]["n_steps"] == 512
    assert config["timing"]["calendar_margin"] == 2 and config["timing"]["reward_basis"] == reward
    assert config["hidden_sizes"] == [64, 64] and config["top_n_choices"] == []
    assert config["training_episode_steps"] == 240 and "margin_study" not in config


@pytest.mark.parametrize("reward,value", [("other", .95), ("log_net", True), ("log_net", .9), ("log_net", np.nan)])
def test_unregistered_credit_cell_fails_before_reading_data(reward, value):
    with pytest.raises(ValueError):
        study.cell_label(reward, value)


def metric(roi=.02):
    return dict(cumulative_return=roi, max_drawdown=-.03, total_fees=1., total_slippage_cost=0.)


def rows(gap=0., changed=False):
    return [dict(fold=fold, reward_basis=reward, gae_lambda=value, method=reward, phase="screen", seed=42, steps=8192,
        metrics={mode: dict(trained=metric(.02 + gap), initialized=metric(), benchmark=metric()) for mode in MODES},
        changes={mode: dict(greedy_changed=changed, economic_ledger_changed=changed) for mode in MODES})
        for fold in study.original.SCREEN_FOLDS for reward, value in study.CELLS]


def test_zero_screen_tie_prioritizes_shorter_trace_then_absolute_reward():
    selected = study.select_screen(rows())
    assert selected["selected_lambda"] == .95 and selected["selected_reward"] == "log_net"
    assert selected["status"] == "screen_failed" and selected["confirmation_steps_authorized"] == 0
    assert selected["actual_screen_steps"] == 65536


@pytest.mark.parametrize("bad", ["one_negative_fold", "no_macro", "no_book", "sub_bp", "missing", "early_budget", "flag_type"])
def test_admission_and_complete_grid_cannot_be_bypassed(bad):
    data = rows(.001, True)
    if bad == "missing":
        data.pop()
    elif bad == "early_budget":
        data[0]["steps"] = 4096
    else:
        for row in data:
            for mode in MODES:
                if bad == "one_negative_fold" and row["fold"] == "2023":
                    row["metrics"][mode]["trained"] = metric(.019)
                elif bad == "no_macro":
                    row["changes"][mode]["greedy_changed"] = False
                elif bad == "no_book":
                    row["changes"][mode]["economic_ledger_changed"] = False
                elif bad == "sub_bp":
                    row["metrics"][mode]["trained"] = metric(.020099)
                elif bad == "flag_type":
                    row["changes"][mode]["greedy_changed"] = "False"
    if bad in ("missing", "early_budget", "flag_type"):
        with pytest.raises(ValueError):
            study.select_screen(data)
    else:
        assert not study.select_screen(data)["admitted"]


def test_eligible_trace_cell_selection_and_no_fallback():
    data = rows(0., False)
    for row in data:
        if row["gae_lambda"] == 1 and row["reward_basis"] == "relative_log":
            for mode in MODES:
                row["metrics"][mode]["trained"] = metric(.03)
                row["changes"][mode] = dict(greedy_changed=True, economic_ledger_changed=True)
    selected = study.select_screen(data)
    assert selected["admitted"] and selected["selected_lambda"] == 1 and selected["selected_reward"] == "relative_log"
    assert selected["confirmation_steps_authorized"] == 98304


def test_all_factorial_cells_share_identical_initial_actor_and_critic(local_data):
    states = []
    for reward, value in study.CELLS:
        config = credit_config(local_data, reward, value)
        data = study.CreditData(config)
        env = data.env("train")
        try:
            model = study.make_model(env, config)
            states.append({key: tensor.detach().clone() for key, tensor in model.policy.state_dict().items()})
            assert model.policy.parameter_count == 12805
            assert data.contract["timing"]["reward_basis"] == reward and data.contract["credit_study"]["gae_lambda"] == value
        finally:
            env.close()
    for state in states[1:]:
        for key, tensor in state.items():
            torch.testing.assert_close(tensor, states[0][key], rtol=0., atol=0.)


@pytest.mark.parametrize("reward,value", study.CELLS)
def test_initial_calendar_and_both_native_books_are_unchanged(local_data, tmp_path, reward, value):
    config = credit_config(local_data, reward, value)
    data = study.CreditData(config)
    data.settings["initial_portfolio"] = "cash"
    env = data.env("val")
    model = study.make_model(env, config)
    try:
        result = study.original.evaluate_case(TimingAgent(model, data.contract), data, tmp_path / "case", "cash", "deterministic_initialized")
        assert result["metrics"] == result["benchmark_metrics"]
        assert not any(study.shared.case_changes(tmp_path / "case", result).values())
    finally:
        env.close()


def test_lambda_one_relative_reward_eight_artificial_steps_and_archive(local_data, tmp_path):
    config = credit_config(local_data, "relative_log", 1.)
    data = study.CreditData(config)
    experience = study.original.TimingExperience(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    env.seed(42)
    model = study.make_model(env, config)
    initial = {key: value.detach().clone() for key, value in model.policy.state_dict().items()}
    callback = study.CreditCallback(data, tmp_path, experience, initial, 8, "synthetic credit")
    try:
        model.learn(8, callback=callback)
        TimingAgent(model, data.contract, dict(credit_study=config["credit_study"])).save(tmp_path / "model", config)
        meta = read_config(tmp_path / "model" / "metadata.json")
        study.verify_model(tmp_path / "model", meta, config, dict(source_hashes=source_hashes()), data.contract, 8)
        restored = TimingAgent.load(tmp_path / "model", data.contract)
        assert restored.model.gae_lambda == 1 and restored.model.gamma == .995
        for key, tensor in model.policy.state_dict().items():
            torch.testing.assert_close(tensor, restored.model.policy.state_dict()[key], rtol=0., atol=0.)
        assert experience.stats()["total_steps"] == experience.stats()["benchmark_steps"] == 8
        assert callback.estimator_statistics["credit/gae_lambda"] == 1.
        assert callback.update_records[-1]["credit/gae_lambda"] == 1.
        study.original.verify_rollouts(tmp_path, callback.rollouts, 8, callback.trace.manifest())
        study.verify_estimator_buffers(tmp_path, dict(credit_estimator_buffers=callback.estimator_buffers,
            rollout_buffers=callback.rollouts, training_trace_prefix=callback.trace.manifest()), config)
        wrong = deepcopy(config)
        wrong["ppo"]["gae_lambda"] = .95
        with pytest.raises(ValueError):
            study.verify_model(tmp_path / "model", meta, wrong, dict(source_hashes=source_hashes()), data.contract, 8)
    finally:
        callback.trace.close()
        env.close()


def test_unpinned_margin_audit_forbids_prepare_without_output(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "REFERENCE_AUDIT_SHA256", None)
    with pytest.raises(ValueError, match="pinned"):
        study.prepare(tmp_path / "new", tmp_path / "unsealed")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("natural_terminal,steps", [(False, 8), (True, 8), (False, 16)])
def test_existing_sb3_timeout_and_tail_values_are_captured_without_extra_predictions(local_data, tmp_path, monkeypatch, natural_terminal, steps):
    config = credit_config(local_data, "log_net", 1.)
    config["ppo"]["n_steps"] = steps
    data = study.CreditData(config)
    base = data.env("train")
    dates = base.env.unwrapped.market.dates
    start = len(dates) - 9 if natural_terminal else 0
    class FixedArtificialSampler:
        def seed(self, seed):
            pass
        def next(self):
            return dict(episode=0, initial_mode="cash", date_stratum="latest" if natural_terminal else "early",
                phase_bin=0, phase=0, start_index=start, start_date=str(dates[start].date()))
    base.sampler = FixedArtificialSampler()
    experience = study.original.TimingExperience(base)
    env = DummyVecEnv([lambda: Monitor(experience)])
    env.seed(42)
    model = study.make_model(env, config)
    predicted_calls = []
    predict = model.policy.predict_values
    def count_existing_predictions(obs):
        value = predict(obs)
        predicted_calls.append(value.detach().cpu().numpy().copy())
        return value
    monkeypatch.setattr(model.policy, "predict_values", count_existing_predictions)
    initial = {key: value.detach().clone() for key, value in model.policy.state_dict().items()}
    callback = study.CreditCallback(data, tmp_path, experience, initial, steps, "synthetic bootstrap boundary")
    try:
        _, prepared = model._setup_learn(steps, callback)
        assert model.collect_rollouts(model.env, prepared, model.rollout_buffer, n_rollout_steps=steps)
        callback.trace.close()
        assert len(predicted_calls) == (1 if natural_terminal else steps // 8 + 1)
        with np.load(tmp_path / callback.estimator_buffers[0]["path"], allow_pickle=False) as evidence:
            assert evidence["dones"][-1, 0] and evidence["last_dones"][0]
            assert bool(evidence["truncated"][-1, 0]) != natural_terminal
            np.testing.assert_array_equal(evidence["last_values"], predicted_calls[-1].reshape(1))
            if natural_terminal:
                assert not evidence["terminal_values"].any()
            else:
                for index in range(steps // 8):
                    np.testing.assert_array_equal(evidence["terminal_values"][8 * index + 7], predicted_calls[index].reshape(1))
        study.verify_estimator_buffers(tmp_path, dict(credit_estimator_buffers=callback.estimator_buffers,
            rollout_buffers=callback.rollouts, training_trace_prefix=callback.trace.manifest()), config)
        assert study.original._optimizer_counts(model) == {}
    finally:
        callback.trace.close()
        env.close()


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_actual_callback_buffer_lambda_rejects_boolean_before_float_conversion(monkeypatch, value):
    monkeypatch.setattr(study.original.TimingCallback, "_on_rollout_end", lambda self: None)
    callback = object.__new__(study.CreditCallback)
    callback.pending_timeout = None
    callback.data = SimpleNamespace(config={"timing": {"reward_basis": "log_net"}, "ppo": {"gae_lambda": 1.}})
    callback.model = SimpleNamespace(rollout_buffer=SimpleNamespace(gamma=.995, gae_lambda=value))
    with pytest.raises(ValueError, match="GAE lambda"):
        callback._on_rollout_end()


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_sidecar_config_lambda_rejects_boolean_without_io(tmp_path, value):
    with pytest.raises(ValueError, match="GAE lambda"):
        study.verify_estimator_buffers(tmp_path, {}, {"timing": {"reward_basis": "log_net"}, "ppo": {"gae_lambda": value}})


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_sidecar_scalar_lambda_rejects_boolean_before_numeric_cast(tmp_path, value):
    side = tmp_path / "side.npz"
    buffer, native = tmp_path / "buffer.npz", tmp_path / "native.npz"
    np.savez_compressed(side, gamma=np.asarray(.995), gae_lambda=np.asarray(value),
        last_values=np.zeros(1, np.float32), last_dones=np.zeros(1, bool),
        dones=np.zeros((1, 1), bool), truncated=np.zeros((1, 1), bool),
        terminal_values=np.zeros((1, 1), np.float32), raw_rewards=np.zeros((1, 1), np.float32), native_learning_rewards=np.zeros((1, 1)))
    np.savez_compressed(buffer, unused=np.zeros(1))
    (tmp_path / "training_trace").mkdir()
    np.savez_compressed(tmp_path / "training_trace" / native.name, unused=np.zeros(1))
    clock = dict(first_step=1, last_step=1, steps=1)
    audit = dict(credit_estimator_buffers=[dict(clock, path=side.name, sha256=file_hash(side))],
        rollout_buffers=[dict(clock, path=buffer.name)], training_trace_prefix=dict(chunks=[dict(clock, path=native.name)]))
    with pytest.raises(ValueError, match="GAE lambda"):
        study.verify_estimator_buffers(tmp_path, audit, {"timing": {"reward_basis": "log_net"}, "ppo": {"gae_lambda": 1.}})
