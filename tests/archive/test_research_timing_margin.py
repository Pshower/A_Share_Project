"""Synthetic margin-study contracts and predeclared admission; no real market runs."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from tools.archive import research_timing_margin as study
from src.agents.timing_agent import TimingAgent
from src.data.preprocess import read_config, write_json, file_hash
from src.envs.initial_account import MODES
from src.training.freeze import source_hashes, json_digest
from tests.test_ppo_framework import local_data
from tests.test_research_timing import timing_config


@pytest.fixture(autouse=True)
def no_real_training(request, monkeypatch):
    if request.node.name == "test_eight_synthetic_steps_use_shared_callback_and_margin_save_load":
        return
    def forbidden(*args, **kwargs):
        raise AssertionError("Only the named artificial eight-step integration may update parameters")
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
        tool_sha256="runner", reference_final_audit_sha256="audit")


def margin_config(local_data, margin):
    value = timing_config(local_data)
    value["ppo"].update(gamma=.995, learning_rate=3e-4, n_steps=8, batch_size=4, n_epochs=1)
    value["timing"]["calendar_margin"] = margin
    value["margin_study"] = dict(study="timing_margin", margin=margin, label=study.LABELS[margin],
        protocol_sha256="synthetic", tool_sha256="synthetic runner", reference_audit_sha256="synthetic reference")
    return value


@pytest.mark.parametrize("margin", study.MARGINS)
@pytest.mark.parametrize("phase,fold,seed", [("screen", "2023", 42), ("confirm", "2026", 123)])
def test_only_margin_changes_policy_initialization_not_other_hyperparameters(tmp_path, margin, phase, fold, seed):
    config = study.config_for(protocol(), tmp_path, fold, margin, phase, seed)
    assert config["timing"]["calendar_margin"] == margin
    assert config["ppo"]["gamma"] == .995 and config["ppo"]["gae_lambda"] == .95
    assert config["ppo"]["learning_rate"] == 3e-4 and config["ppo"]["n_steps"] == 512
    assert config["device"] == "cpu" and config["training_episode_steps"] == 240
    assert config["margin_study"]["protocol_sha256"] == json_digest(protocol())


@pytest.mark.parametrize("margin,phase,fold,seed", [(True, "screen", "2023", 42), (.1, "screen", "2023", 42),
    (1., "screen", "2023", 123), (1., "confirm", "2023", 42)])
def test_unregistered_condition_rejected_without_file_reads(tmp_path, margin, phase, fold, seed):
    with pytest.raises(ValueError, match="Unregistered"):
        study.config_for({}, tmp_path, fold, margin, phase, seed)


def metric(roi=.02):
    return dict(cumulative_return=roi, max_drawdown=-.03, total_fees=1., total_slippage_cost=0.)


def rows(gap=0., changed=False):
    return [dict(fold=fold, margin=margin, phase="screen", seed=42, steps=8192,
        metrics={mode: dict(trained=metric(.02 + gap), initialized=metric(), benchmark=metric()) for mode in MODES},
        changes={mode: dict(greedy_changed=changed, economic_ledger_changed=changed) for mode in MODES})
        for fold in study.original.SCREEN_FOLDS for margin in study.MARGINS]


def test_zero_grid_seals_failed_screen_and_larger_margin_tie():
    selected = study.select_screen(rows())
    assert selected["selected_margin"] == 2. and not selected["admitted"]
    assert selected["status"] == "screen_failed" and selected["confirmation_steps_authorized"] == 0
    assert selected["actual_screen_steps"] == 49152


@pytest.mark.parametrize("bad", ["one_negative_fold", "no_macro_change", "no_economic_change", "sub_one_bp"])
def test_confirmation_admission_requires_both_fold_gap_and_real_behavior(bad):
    data = rows(.0002, True)
    for row in data:
        if bad == "one_negative_fold" and row["fold"] == "2023":
            for mode in MODES:
                row["metrics"][mode]["trained"] = metric(.019)
        elif bad in ("no_macro_change", "no_economic_change"):
            for mode in MODES:
                row["changes"][mode]["greedy_changed" if bad == "no_macro_change" else "economic_ledger_changed"] = False
        elif bad == "sub_one_bp":
            for mode in MODES:
                row["metrics"][mode]["trained"] = metric(.020099)
    assert not study.select_screen(data)["admitted"]


def test_behavior_changes_must_be_macro_and_economic_in_same_fold():
    data = rows(.001, True)
    for row in data:
        for mode in MODES:
            row["changes"][mode]["economic_ledger_changed" if row["fold"] == "2023" else "greedy_changed"] = False
    assert not study.select_screen(data)["admitted"]


def test_margin_grid_positive_admission_and_fixed_budget():
    selected = study.select_screen(rows(.001, True))
    assert selected["admitted"] and selected["confirmation_steps_authorized"] == 98304
    data = rows(.001, True)
    data[0]["steps"] = 4096
    with pytest.raises(ValueError):
        study.select_screen(data)


@pytest.mark.parametrize("margin", study.MARGINS)
def test_shared_policy_keeps_greedy_calendar_and_exact_contract(local_data, margin):
    config = margin_config(local_data, margin)
    data = study.MarginData(config)
    data.settings["initial_portfolio"] = "cash"
    env = data.env("val")
    model = study.make_model(env, config)
    agent = TimingAgent(model, data.contract)
    try:
        vector, _ = env.reset(seed=42)
        for phase in range(20):
            vector[:20] = 0
            vector[phase] = 1
            predicted = agent.inspect_action(vector)
            assert predicted["macro_action"] == int(phase == 0)
            assert predicted["action_probabilities"][int(phase == 0)] == pytest.approx(1 / (1 + np.exp(-margin)), abs=1e-6)
        assert model.policy.parameter_count == 12805 and data.contract["timing"]["calendar_margin"] == margin
    finally:
        env.close()


def test_initial_critic_and_noncalendar_tensors_equal_across_margins(local_data):
    states = []
    for margin in study.MARGINS:
        config = margin_config(local_data, margin)
        data = study.MarginData(config)
        env = data.env("train")
        try:
            model = study.make_model(env, config)
            states.append({key: value.detach().clone() for key, value in model.policy.state_dict().items()})
        finally:
            env.close()
    for state in states[1:]:
        for key in states[0]:
            if key != "action_net.weight":
                torch.testing.assert_close(state[key], states[0][key], atol=0., rtol=0.)


def test_eight_synthetic_steps_use_shared_callback_and_margin_save_load(local_data, tmp_path):
    config = margin_config(local_data, .5)
    data = study.MarginData(config)
    experience = study.original.TimingExperience(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    env.seed(42)
    model = study.make_model(env, config)
    initial = {key: value.detach().clone() for key, value in model.policy.state_dict().items()}
    agent = TimingAgent(model, data.contract, dict(margin_study=config["margin_study"]))
    callback = study.original.TimingCallback(data, tmp_path, experience, initial, 8, "synthetic margin")
    try:
        agent.save(tmp_path / "initial_model", config)
        model.learn(8, callback=callback)
        agent.save(tmp_path / "final_model", config)
        restored = TimingAgent.load(tmp_path / "final_model", data.contract)
        for key, value in model.policy.state_dict().items():
            torch.testing.assert_close(value, restored.model.policy.state_dict()[key], atol=0., rtol=0.)
        meta = read_config(tmp_path / "final_model" / "metadata.json")
        study.verify_model(tmp_path / "final_model", meta, config, dict(source_hashes=source_hashes()), data.contract, 8)
        assert experience.stats()["total_steps"] == experience.stats()["benchmark_steps"] == 8
        assert max(study.original._optimizer_counts(model).values()) > 0
        study.original.verify_rollouts(tmp_path, callback.rollouts, 8, callback.trace.manifest())
        meta["provenance"]["margin_study"]["tool_sha256"] = "different"
        with pytest.raises(ValueError, match="runner"):
            study.verify_model(tmp_path / "final_model", meta, config, dict(source_hashes=source_hashes()), data.contract, 8)
    finally:
        callback.trace.close()
        env.close()


def test_unpinned_predecessor_prevents_prepare_without_output(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "REFERENCE_AUDIT_SHA256", None)
    with pytest.raises(ValueError, match="pinned"):
        study.prepare(tmp_path / "new", tmp_path / "unsealed")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("workspace_relative", [True, False])
def test_reference_accepts_only_current_workspace_relative_signed_schema(tmp_path, monkeypatch, workspace_relative):
    directory = tmp_path / "reports" / "runs" / "timing"
    directory.mkdir(parents=True)
    sources = {f'source_{index}.py': str(index) for index in range(96)}
    registered = dict(source_hashes=sources)
    write_json(directory / "protocol.json", registered)
    key = "reports/runs/timing/protocol.json" if workspace_relative else "protocol.json"
    write_json(directory / "final_audit.json", dict(status="passed", signed_files_sha256={key: file_hash(directory / "protocol.json")}))
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "source_hashes", lambda: sources)
    monkeypatch.setattr(study, "REFERENCE_AUDIT_SHA256", file_hash(directory / "final_audit.json"))
    monkeypatch.setattr(study.original, "check_frozen", lambda _: None)
    if workspace_relative:
        assert study.reference_protocol(directory) == registered
    else:
        with pytest.raises(ValueError, match="Predecessor"):
            study.reference_protocol(directory)
