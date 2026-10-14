"""Artificial phase/budget contracts; one explicit eight-step synthetic update."""

from copy import deepcopy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.data.preprocess import file_hash, read_config, write_json
from src.training.freeze import json_digest
from tests.test_ppo_framework import local_data
from tests.test_research_timing import timing_config
from tools import research_timing_phase_budget as study
from tools.timing_phase_agent import PhaseTimingAgent
from tests.test_timing_phase_agent import fixture as phase_fixture


@pytest.fixture(autouse=True)
def synthetic_updates_only(request, monkeypatch):
    if request.node.name.startswith("test_eight_artificial_steps"):
        return
    def forbidden(*args, **kwargs):
        raise AssertionError("Only the explicitly named eight-step artificial integration may optimize")
    monkeypatch.setattr(study.PPO, "learn", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def protocol():
    return dict(base_config={"ppo": {}}, folds={fold: {"research_config": "synthetic.json"} for fold in
        (*study.original.SCREEN_FOLDS, *study.original.CONFIRM_FOLDS)}, tool_sha256="synthetic runner",
        component_hashes={name: "synthetic" for name in study.COMPONENT_FILES}, helper_hashes={name: "synthetic" for name in study.HELPER_FILES})


def metric(roi=0.):
    return dict(cumulative_return=roi, max_drawdown=-.03, total_fees=1., total_slippage_cost=0.)


def rows(gap=0., changed=False):
    return [dict(fold=fold, architecture=architecture, steps=steps, seed=42, phase="screen",
        method=study.cell_label(architecture, steps), physical_run_id=study.run_id(fold, architecture, "screen", 42),
        metrics={mode: dict(trained=metric(gap), initialized=metric(), benchmark=metric()) for mode in study.MODES},
        changes={mode: dict(greedy_changed=changed, economic_ledger_changed=changed) for mode in study.MODES},
        test_evaluated=False, actionable=False) for fold in study.original.SCREEN_FOLDS for architecture, steps in study.CELLS]


@pytest.mark.parametrize("architecture", study.ARCHITECTURES)
@pytest.mark.parametrize("phase,fold,seed,budget", [("screen", "2023", 42, 32768), ("screen", "2024", 42, 32768),
    ("confirm", "2025", 123, 8192), ("confirm", "2026", 2026, 32768)])
def test_config_keeps_execution_and_ppo_constant_and_separates_physical_budget(tmp_path, architecture, phase, fold, seed, budget):
    registered = protocol()
    before = deepcopy(registered)
    config = study.config_for(registered, tmp_path, fold, architecture, phase, seed, budget)
    assert registered == before
    identity = config["phase_budget_study"]
    assert identity["physical_budget"] == budget
    assert identity["snapshot_steps"] == ([8192, 32768] if phase == "screen" else [budget])
    assert config["timing"]["reward_basis"] == "relative_log" and config["timing"]["calendar_margin"] == 2.
    assert config["ppo"]["gamma"] == .995 and config["ppo"]["gae_lambda"] == 1.
    assert config["ppo"]["learning_rate"] == 3e-4 and config["ppo"]["clip_range"] == .2
    assert config["ppo"]["n_steps"] == 512 and config["ppo"]["batch_size"] == 128 and config["ppo"]["n_epochs"] == 5
    assert config["training_episode_steps"] == 240 and config["top_n"] == 20 and not config["top_n_choices"]
    assert config["device"] == "cpu" and config["torch_threads"] == 1


@pytest.mark.parametrize("kwargs", [dict(architecture="unknown"), dict(seed=True), dict(seed=123), dict(budget=8192),
    dict(budget=True), dict(budget=32768.), dict(fold="2025"), dict(phase="unknown")])
def test_unregistered_screen_request_fails_before_data_or_training(tmp_path, kwargs):
    request = dict(fold="2023", architecture="linear", phase="screen", seed=42, budget=32768)
    request.update(kwargs)
    with pytest.raises(ValueError):
        study.config_for(protocol(), tmp_path, **request)


def test_grid_selection_counts_physical_experience_once_and_ties_linear_short():
    selected = study.select_screen(rows())
    assert selected["actual_training_steps"] == 131072
    assert selected["snapshot_steps_sum"] == 163840
    assert selected["shared_short_prefix_steps"] == 32768
    assert selected["physical_runs"] == 4 and selected["snapshots"] == 8
    assert (selected["selected_architecture"], selected["selected_budget"]) == ("linear", 8192)
    assert not selected["admitted"] and selected["confirmation_steps_authorized"] == 0


@pytest.mark.parametrize("gap,admitted", [(.0001, True), (.0001 - 1e-12, False), (0., False)])
def test_one_basis_point_gate_uses_unrounded_registered_gap(gap, admitted):
    selected = study.select_screen(rows(gap, True))
    assert selected["admitted"] is admitted


@pytest.mark.parametrize("bad", ["missing", "duplicate", "nonfinite", "typed_flag", "budget_bool", "wrong_seed"])
def test_snapshot_grid_and_metrics_cannot_be_fabricated(bad):
    data = rows(.001, True)
    if bad == "missing":
        data.pop()
    elif bad == "duplicate":
        data[-1] = deepcopy(data[0])
    elif bad == "nonfinite":
        data[0]["metrics"][study.MODES[0]]["trained"]["cumulative_return"] = np.nan
    elif bad == "typed_flag":
        data[0]["changes"][study.MODES[0]]["greedy_changed"] = 1
    elif bad == "budget_bool":
        data[0]["steps"] = True
    else:
        data[0]["seed"] = 123
    with pytest.raises(ValueError):
        study.select_screen(data)


@pytest.mark.parametrize("budget", study.SNAPSHOT_STEPS)
def test_confirmation_gate_reuses_same_financial_thresholds_at_either_selected_budget(budget):
    confirmed = []
    for fold in study.original.CONFIRM_FOLDS:
        for seed in study.original.SEEDS:
            row = deepcopy(rows(.01, True)[0])
            row.update(fold=fold, seed=seed, phase="confirm", steps=budget, method=study.cell_label("linear", budget))
            confirmed.append(row)
    result = study.confirmation_summary(confirmed, ("linear", budget))
    assert result["qualified"]
    assert result["paired_across_fold_gap"]["mean"] == pytest.approx(.01)
    assert result["qualified_candidate"]["budget"] == budget
    for row in confirmed:
        row["metrics"][study.MODES[0]]["trained"]["max_drawdown"] = -.11
    assert not study.confirmation_summary(confirmed, ("linear", budget))["qualified"]


def fake_state():
    generator = np.random.default_rng(42)
    book = SimpleNamespace(portfolio=SimpleNamespace(cash=100., positions=np.array([100.]),
        available_positions=np.array([0.]), cost_basis=np.array([1.]), receivables=[("future", 4.)], share_locks=[("future", 0, 100.)]),
        broker=SimpleNamespace(config="synthetic", _used_volume=np.array([100.])), market=SimpleNamespace(current_index=7), _steps=5, _done=False, history=[])
    controller = SimpleNamespace(env=SimpleNamespace(env=book, _observation=np.ones(3)),
        benchmark=SimpleNamespace(env=deepcopy(book), _observation=np.ones(3)), phase=5,
        stock_observation=np.ones(3), benchmark_stock_observation=np.ones(3),
        schedule_audit=lambda: dict(rng=deepcopy(generator.bit_generator.state)))
    policy = SimpleNamespace(training=True)
    policy.set_training_mode = lambda value: setattr(policy, "training", value)
    model = SimpleNamespace(policy=policy, device=torch.device("cpu"), _last_obs=np.ones((1, 129)),
        _last_episode_starts=np.zeros(1, bool), num_timesteps=8192, _total_timesteps=32768, _n_updates=80, _current_progress_remaining=.75)
    return model, SimpleNamespace(env=controller), generator


def test_isolation_restores_rng_and_policy_mode_on_exception():
    model, experience, _ = fake_state()
    python, numpy, tensor = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        with study.isolated_checkpoint(model, experience):
            random.random()
            np.random.random()
            torch.rand(1)
            model.policy.set_training_mode(False)
            raise RuntimeError("synthetic diagnostic failure")
    assert random.getstate() == python and np.array_equal(np.random.get_state()[1], numpy[1])
    assert torch.equal(torch.get_rng_state(), tensor) and model.policy.training


@pytest.mark.parametrize("field", ["cost_basis", "receivables", "share_locks", "used_volume"])
def test_state_fingerprint_includes_nonaggregate_account_and_broker_state(field):
    model, experience, _ = fake_state()
    with pytest.raises(ValueError, match="Checkpoint changed"):
        with study.isolated_checkpoint(model, experience):
            base = experience.env.env.env
            if field == "used_volume":
                base.broker._used_volume[0] += 1
            elif field == "cost_basis":
                base.portfolio.cost_basis[0] += 1
            else:
                getattr(base.portfolio, field).append(("changed", 1))


def test_callback_captures_only_registered_after_optimizer_prefix(monkeypatch):
    events = []
    callback = study.PhaseBudgetCallback.__new__(study.PhaseBudgetCallback)
    callback.budget, callback.model = 32768, SimpleNamespace(num_timesteps=8192)
    monkeypatch.setattr(study.credit.CreditCallback, "_on_rollout_start", lambda self: events.append("previous Adam recorded"))
    monkeypatch.setattr(study, "capture_snapshot", lambda self, steps: events.append(("capture", steps)))
    callback._on_rollout_start()
    assert events == ["previous Adam recorded", ("capture", 8192)]
    events.clear()
    callback.model.num_timesteps = 7680
    callback._on_rollout_start()
    assert events == ["previous Adam recorded"]


@pytest.mark.parametrize("change", ["learning_rate", "clip_range"])
def test_nonconstant_schedule_cannot_claim_budget_independent_prefix(change):
    model = SimpleNamespace(learning_rate=3e-4, lr_schedule=lambda _: 3e-4, clip_range=lambda _: .2)
    assert study.constant_schedules(model)["learning_rate"] == 3e-4
    if change == "learning_rate":
        model.lr_schedule = lambda progress: progress * 3e-4
    else:
        model.clip_range = lambda progress: .2 * progress
    with pytest.raises(ValueError, match="constant"):
        study.constant_schedules(model)


def test_failed_confirmation_request_cannot_create_output_or_data(tmp_path, monkeypatch):
    path = tmp_path / "selection.json"
    selected = dict(protocol_sha256=json_digest(protocol()), **study.select_screen(rows()), completed_sha256={})
    write_json(path, selected)
    monkeypatch.setattr(study, "check_frozen", lambda _: None)
    monkeypatch.setattr(study, "load_stage", lambda *_args, **_kwargs: rows())
    monkeypatch.setattr(study, "selection_record", lambda *_: selected)
    monkeypatch.setattr(study, "PhaseData", lambda *_: pytest.fail("Failed admission cannot read data"))
    with pytest.raises(ValueError, match="Failed"):
        study.run_trial(tmp_path, protocol(), "2025", "linear", "confirm", 42, 8192, file_hash(path))
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("architecture", study.ARCHITECTURES)
def test_eight_artificial_steps_keep_native_and_actual_bootstrap_evidence(local_data, tmp_path, architecture):
    config = timing_config(local_data, "relative_log")
    config["ppo"].update(learning_rate=3e-4, gamma=.995, gae_lambda=1., n_steps=8, batch_size=4, n_epochs=1, clip_range=.2)
    config["phase_budget_study"] = dict(study="timing_phase_budget", architecture=architecture, physical_budget=32768,
        snapshot_steps=[8192, 32768], protocol_sha256="artificial", tool_sha256="artificial",
        component_hashes={}, helper_hashes={}, reference_audit_sha256="artificial")
    data = study.PhaseData(config)
    experience = study.original.TimingExperience(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    output = tmp_path / architecture
    output.mkdir()
    model = study.make_model(env, config)
    initial = {key: value.detach().cpu().clone() for key, value in model.policy.state_dict().items()}
    callback = study.PhaseBudgetCallback(data, output, experience, initial, 8, "eight_artificial_only", protocol=protocol())
    try:
        model.learn(8, callback=callback)
        assert model.num_timesteps == 8 and not callback.snapshots
        assert len(study.original._optimizer_counts(model)) == (8 if architecture == "linear" else 9)
        audit = dict(credit_estimator_buffers=callback.estimator_buffers, rollout_buffers=callback.rollouts,
            training_trace_prefix=callback.trace.manifest())
        study.original.verify_rollouts(output, callback.rollouts, 8, audit["training_trace_prefix"])
        study.credit.verify_estimator_buffers(output, audit, config)
        artifact = output / "synthetic_model"
        PhaseTimingAgent(model, data.contract).save(artifact, config)
        restored = PhaseTimingAgent.load(artifact, data.contract)
        assert all(torch.equal(value, restored.model.policy.state_dict()[key]) for key, value in model.policy.state_dict().items())
        with pytest.raises(ValueError, match="registered integer"):
            study.verify_model(artifact, config, data.contract, 8)
    finally:
        callback.trace.close()
        env.close()


@pytest.mark.parametrize("entry", ["cached_trial", "load_stage", "require_confirmation"])
def test_public_read_apis_reject_unfrozen_registration_before_data_or_cache(tmp_path, monkeypatch, entry):
    called = []
    def reject(value):
        called.append(value)
        raise ValueError("synthetic unfrozen registration")
    monkeypatch.setattr(study, "check_frozen", reject)
    monkeypatch.setattr(study, "PhaseData", lambda *_: pytest.fail("Frozen check must precede data"))
    with pytest.raises(ValueError, match="unfrozen registration"):
        if entry == "cached_trial":
            study.cached_trial(tmp_path / "missing", {}, protocol())
        elif entry == "load_stage":
            study.load_stage(tmp_path, protocol(), "screen")
        else:
            study.require_confirmation(tmp_path, protocol(), ("linear", 8192), None)
    assert len(called) == 1
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("clock", [torch.tensor(True), torch.tensor(1., dtype=torch.float64),
    torch.tensor(1, dtype=torch.int64), torch.tensor([1.], dtype=torch.float32)])
def test_saved_adam_clock_requires_actual_cpu_float32_scalar(tmp_path, clock):
    model, contract, env = phase_fixture()
    try:
        # Artificial archive fields only: no optimizer or rollout is executed.
        model.num_timesteps = 8192
        model.phase_constant_schedules = study.constant_schedules(model)
        for parameter in model.policy.parameters():
            model.policy.optimizer.state[parameter] = dict(step=clock, exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
        folder = tmp_path / "typed_archive"
        PhaseTimingAgent(model, contract).save(folder, {"synthetic_only": True})
        audit = dict(optimizer_steps_min=1, optimizer_steps_max=1, actual_optimizer_updates=1,
            optimizer_steps_by_parameter={name: 1 for name, _ in model.policy.named_parameters()}, constant_schedules=model.phase_constant_schedules)
        with pytest.raises(ValueError, match="Adam state"):
            study.verify_model(folder, {"synthetic_only": True}, contract, 8192, audit)
    finally:
        env.close()
