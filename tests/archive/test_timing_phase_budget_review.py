"""Independent synthetic shared-prefix accounting and archive guard review."""

from copy import deepcopy
import io
import json
import random
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.preprocess import file_hash, write_json
from tests.archive.test_timing_phase_agent import fixture as phase_fixture
from tools.archive import research_timing_phase_budget as runner
from tools.archive.timing_phase_agent import PhaseTimingAgent


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def snapshot_records():
    metric = dict(cumulative_return=.05, max_drawdown=-.03, total_fees=2., total_slippage_cost=1.)
    rows = []
    for fold in runner.original.SCREEN_FOLDS:
        for architecture, steps in runner.CELLS:
            rows.append(dict(fold=fold, architecture=architecture, steps=steps, seed=42,
                phase="screen", method=runner.cell_label(architecture, steps),
                metrics={mode: {role: deepcopy(metric) for role in ("trained", "initialized", "benchmark")}
                         for mode in runner.MODES},
                changes={mode: dict(greedy_changed=False, economic_ledger_changed=False) for mode in runner.MODES},
                test_evaluated=False, actionable=False))
    return rows


def set_gap(rows, architecture, budget, fold, gap, *, greedy=True, ledger=True):
    for row in rows:
        if (row["architecture"], row["steps"], row["fold"]) == (architecture, budget, fold):
            for mode in runner.MODES:
                row["metrics"][mode]["trained"]["cumulative_return"] += gap
                row["changes"][mode] = dict(greedy_changed=greedy, economic_ledger_changed=ledger)


def fake_training_state():
    generator = np.random.default_rng(42)
    book = SimpleNamespace(portfolio=SimpleNamespace(cash=100.), broker=SimpleNamespace(config="synthetic"),
        market=SimpleNamespace(current_index=7), _steps=5, _done=False, history=[])
    controller = SimpleNamespace(env=SimpleNamespace(env=book, _observation=np.ones(3)),
        benchmark=SimpleNamespace(env=deepcopy(book), _observation=np.ones(3)), phase=5,
        stock_observation=np.ones(3), benchmark_stock_observation=np.ones(3),
        schedule_audit=lambda: dict(rng=deepcopy(generator.bit_generator.state), phase_bag=[1, 2]))
    policy = SimpleNamespace(training=True)
    policy.set_training_mode = lambda value: setattr(policy, "training", value)
    model = SimpleNamespace(policy=policy, device=torch.device("cpu"), _last_obs=np.ones((1, 129)),
        _last_episode_starts=np.zeros(1, dtype=bool), num_timesteps=8192, _total_timesteps=32768,
        _n_updates=80, _current_progress_remaining=.75)
    return model, SimpleNamespace(env=controller), generator


def rewrite_archive(folder, member, mutate):
    path = folder / "model.zip"
    with zipfile.ZipFile(path) as zipped:
        content = {name: zipped.read(name) for name in zipped.namelist()}
    original = torch.load(io.BytesIO(content[member]), map_location="cpu", weights_only=True)
    mutate(original)
    output = io.BytesIO()
    torch.save(original, output)
    content[member] = output.getvalue()
    with zipfile.ZipFile(path, "w") as zipped:
        for name, value in content.items():
            zipped.writestr(name, value)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    metadata["model_sha256"] = file_hash(path)
    write_json(folder / "metadata.json", metadata)


def test_complete_matrix_counts_four_physical_paths_not_eight_snapshot_budgets():
    result = runner.select_screen(snapshot_records())
    assert result["physical_runs"] == 4 and result["snapshots"] == 8
    assert result["actual_training_steps"] == 4 * 32768 == 131072
    assert result["snapshot_steps_sum"] == 4 * (8192 + 32768) == 163840
    assert result["shared_short_prefix_steps"] == 4 * 8192
    assert result["selected_architecture"] == "linear" and result["selected_budget"] == 8192
    assert not result["admitted"] and result["confirmation_steps_authorized"] == 0


@pytest.mark.parametrize("change", ["missing", "duplicate", "unregistered_budget", "nonphysical_fold"])
def test_selection_requires_every_registered_fold_architecture_snapshot_once(change):
    rows = snapshot_records()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif change == "unregistered_budget":
        rows[0]["steps"] = 16384
    else:
        rows[0]["fold"] = "2025"
    with pytest.raises(ValueError):
        runner.select_screen(rows)


@pytest.mark.parametrize("steps", [True, 8192., np.int64(8192), 16384])
def test_snapshot_budget_is_an_explicit_registered_integer(steps):
    with pytest.raises(ValueError, match="snapshot budget"):
        runner.cell_label("linear", steps)


def test_one_fold_profit_does_not_authorize_confirmation():
    rows = snapshot_records()
    set_gap(rows, "due_phase_residual", 32768, "2023", .01)
    result = runner.select_screen(rows)
    assert (result["selected_architecture"], result["selected_budget"]) == ("due_phase_residual", 32768)
    assert not result["admitted"] and result["confirmation_steps_authorized"] == 0


def test_macro_and_economic_changes_must_occur_on_the_same_account_path():
    rows = snapshot_records()
    for fold in runner.original.SCREEN_FOLDS:
        set_gap(rows, "linear", 8192, fold, .001, greedy=False, ledger=False)
    for row in rows:
        if row["architecture"] == "linear" and row["steps"] == 8192:
            row["changes"][runner.MODES[0]]["greedy_changed"] = True
            row["changes"][runner.MODES[1]]["economic_ledger_changed"] = True
    result = runner.select_screen(rows)
    assert not result["admitted"]


def test_ineligible_first_choice_has_no_fallback_to_lower_scoring_cell():
    rows = snapshot_records()
    for fold in runner.original.SCREEN_FOLDS:
        set_gap(rows, "linear", 8192, fold, .002, greedy=False, ledger=False)
        set_gap(rows, "due_phase_residual", 32768, fold, .001)
    result = runner.select_screen(rows)
    assert (result["selected_architecture"], result["selected_budget"]) == ("linear", 8192)
    assert not result["admitted"] and result["confirmation_steps_authorized"] == 0


@pytest.mark.parametrize("budget", [8192, 32768])
def test_admitted_confirmation_preserves_selected_budget_and_maximum_total(budget):
    rows = snapshot_records()
    for fold in runner.original.SCREEN_FOLDS:
        set_gap(rows, "due_phase_residual", budget, fold, .001)
    result = runner.select_screen(rows)
    assert result["admitted"] and result["selected_budget"] == budget
    assert result["confirmation_steps_authorized"] == 6 * budget
    assert result["actual_training_steps"] + result["confirmation_steps_authorized"] == (180224 if budget == 8192 else 327680)


def test_checkpoint_isolation_restores_global_rng_and_policy_mode():
    model, experience, _generator = fake_training_state()
    before = runner.training_state_digest(model, experience)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with runner.isolated_checkpoint(model, experience):
        random.random()
        np.random.random(3)
        torch.rand(3)
        model.policy.set_training_mode(False)
    assert random.getstate() == python_state
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_state[0] and np.array_equal(actual_numpy[1], numpy_state[1])
    assert actual_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert model.policy.training is True and runner.training_state_digest(model, experience) == before


@pytest.mark.parametrize("change", ["cursor", "pending_observation", "independent_schedule_rng"])
def test_checkpoint_cannot_change_training_state_or_independent_schedule(change):
    model, experience, generator = fake_training_state()
    with pytest.raises(ValueError, match="Checkpoint changed"):
        with runner.isolated_checkpoint(model, experience):
            if change == "cursor":
                experience.env.env.env.market.current_index += 1
            elif change == "pending_observation":
                model._last_obs[0, 20] += 1
            else:
                generator.integers(100)


@pytest.mark.parametrize("architecture", ["linear", "due_phase_residual"])
def test_valid_initial_archive_can_be_checked_without_model_load(tmp_path, architecture):
    model, contract, env = phase_fixture(architecture)
    try:
        folder = tmp_path / "artifact"
        PhaseTimingAgent(model, contract).save(folder, {"synthetic_only": True})
        theta = runner.verify_model(folder, {"synthetic_only": True}, contract, 0)
        assert len(theta) == (8 if architecture == "linear" else 9)
    finally:
        env.close()


@pytest.mark.parametrize("change", ["tensor_dtype", "missing_optimizer_coordinate", "duplicate_optimizer_coordinate"])
def test_resigned_initial_archive_cannot_change_tensor_precision_or_optimizer_mapping(tmp_path, change):
    model, contract, env = phase_fixture()
    try:
        folder = tmp_path / "artifact"
        PhaseTimingAgent(model, contract).save(folder, {"synthetic_only": True})
        if change == "tensor_dtype":
            rewrite_archive(folder, "policy.pth", lambda theta: theta.update(
                {"action_net.residual.weight": theta["action_net.residual.weight"].double()}))
        elif change == "missing_optimizer_coordinate":
            rewrite_archive(folder, "policy.optimizer.pth", lambda optimizer: optimizer["param_groups"][0]["params"].pop())
        else:
            rewrite_archive(folder, "policy.optimizer.pth", lambda optimizer: optimizer["param_groups"][0]["params"].append(
                optimizer["param_groups"][0]["params"][0]))
        with pytest.raises(ValueError):
            runner.verify_model(folder, {"synthetic_only": True}, contract, 0)
    finally:
        env.close()


@pytest.mark.parametrize("wrong_delta", [123., float("nan")])
def test_cold_cache_recomputes_claimed_parameter_growth_without_model_load(tmp_path, monkeypatch, wrong_delta):
    name = runner.run_id("2025", "due_phase_residual", "confirm", 42)
    directory = tmp_path / name
    directory.mkdir()
    write_json(directory / "completed.json", {})
    config = dict(seed=42, timing=dict(fold="2025", phase="confirm"), phase_budget_study=dict(
        architecture="due_phase_residual", physical_budget=8192, snapshot_steps=[8192]))
    contract = dict(phase_policy=dict(parameter_count=12914))
    data = SimpleNamespace(stock_contract={"synthetic": True}, contract=contract)
    protocol = dict(folds={"2025": {"stock_contract_sha256": runner.json_digest(data.stock_contract)}},
                    source_hashes={}, tool_sha256="0" * 64)
    baseline = dict(cumulative_return=.05, max_drawdown=-.03, total_fees=2., total_slippage_cost=1.)
    # CPU tensors stand in for the already cold-read archive output. No policy
    # load or prediction is allowed while validating the cache's delta claim.
    initial = {"synthetic.weight": torch.tensor([0.], dtype=torch.float32)}
    final = {"synthetic.weight": torch.tensor([.25], dtype=torch.float32)}
    audit = dict(actual_timesteps=8192, experience=dict(total_steps=8192, benchmark_steps=8192),
        training_schedule=dict(agent_environment_steps=8192, benchmark_extra_environment_steps=8192),
        source_hashes={}, tool_sha256=protocol["tool_sha256"], parameters=12914,
        tensor_delta_l2={"synthetic.weight": wrong_delta}, actual_optimizer_updates=4,
        optimizer_steps_min=4, optimizer_steps_max=4,
        rollout_buffers=[], credit_estimator_buffers=[], training_trace_prefix={},
        training_updates_path="snapshot_updates.csv", training_updates_sha256="0" * 64,
        save_load_parameters_equal=True, save_load_macro_action_equal=True,
        checkpoint_training_state_unchanged=True, all_parameters_finite=True)
    row = dict(architecture="due_phase_residual", fold="2025", phase="confirm", seed=42,
        physical_run_id=name, steps=8192, method=runner.cell_label("due_phase_residual", 8192),
        test_evaluated=False, actionable=False, audit=deepcopy(audit), model_path=str(directory / "final_model"),
        metrics={mode: {role: deepcopy(baseline) for role in ("trained", "initialized", "benchmark")} for mode in runner.MODES},
        changes={mode: dict(greedy_changed=False, economic_ledger_changed=False) for mode in runner.MODES},
        evaluation_steps={role: {mode: 59 for mode in runner.MODES} for role in ("initialized", "trained")})
    completed = dict(protocol_sha256=runner.json_digest(protocol), config_sha256=runner.json_digest(config),
        tool_sha256=protocol["tool_sha256"], selection_sha256=None, physical_steps=8192, files={}, records=[row])

    def read(path):
        path = runner.Path(path)
        if path.name == "completed.json":
            return completed
        if path.name == "config.json":
            return config
        if path.name == "final_audit.json":
            return audit
        if path.name == "result.json":
            return dict(contract=contract, metrics=baseline, benchmark_metrics=baseline, evaluation_steps=59)
        raise AssertionError("Unexpected financial artifact read: " + str(path))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Cold cache review must not instantiate a model or financial environment")

    monkeypatch.setattr(runner, "read_config", read)
    monkeypatch.setattr(runner, "check_frozen", lambda _protocol: None)
    monkeypatch.setattr(runner, "file_hash", lambda _path: "0" * 64)
    monkeypatch.setattr(runner, "PhaseData", lambda _config: data)
    monkeypatch.setattr(runner, "make_model", forbidden)
    monkeypatch.setattr(runner.PhaseTimingAgent, "load", forbidden)
    monkeypatch.setattr(runner, "verify_model", lambda _path, _config, _contract, steps, *args: initial if steps == 0 else final)
    monkeypatch.setattr(runner, "baseline_metrics", lambda *_args: {mode: baseline for mode in runner.MODES})
    for helper in ("verify_leaf_files", "verify_timing_trace", "verify_rollouts", "_verify_result"):
        monkeypatch.setattr(runner.original, helper, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner.original, "validation_dates", lambda _data: [])
    monkeypatch.setattr(runner.credit, "verify_estimator_buffers", lambda *_args: None)
    monkeypatch.setattr(runner.shared, "case_changes", lambda *_args: dict(greedy_changed=False, economic_ledger_changed=False))
    updates = pd.DataFrame(dict(timesteps=np.arange(512, 8193, 512), actual_optimizer_steps=[4] * 16,
                               **{"credit/gamma": [.995] * 16, "credit/gae_lambda": [1.] * 16}))
    monkeypatch.setattr(runner.pd, "read_csv", lambda *_args, **_kwargs: updates)
    with pytest.raises(ValueError):
        runner.cached_trial(directory, config, protocol)


@pytest.mark.parametrize("entry", ["cached_trial", "load_stage", "require_confirmation"])
def test_public_cold_entry_checks_frozen_protocol_before_cache_or_provider(tmp_path, monkeypatch, entry):
    protocol, checked = {"synthetic_review_only": True}, []

    def frozen(value):
        checked.append(value)
        raise ValueError("Independent frozen guard sentinel")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("No provider, file parsing or model may precede the frozen protocol guard")

    monkeypatch.setattr(runner, "check_frozen", frozen)
    monkeypatch.setattr(runner, "PhaseData", forbidden)
    monkeypatch.setattr(runner, "read_config", forbidden)
    monkeypatch.setattr(runner, "file_hash", forbidden)
    monkeypatch.setattr(runner, "make_model", forbidden)
    monkeypatch.setattr(runner.PhaseTimingAgent, "load", forbidden)
    with pytest.raises(ValueError, match="Independent frozen guard sentinel"):
        if entry == "cached_trial":
            runner.cached_trial(tmp_path / "absent", {}, protocol)
        elif entry == "load_stage":
            runner.load_stage(tmp_path / "absent", protocol, "screen")
        else:
            runner.require_confirmation(tmp_path / "absent", protocol, ("linear", 8192), "0" * 64)
    assert checked == [protocol]
