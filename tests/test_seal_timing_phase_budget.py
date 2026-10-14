"""Pure temporary seal manifests; no Tensor load, prediction or market steps."""

from copy import deepcopy
import json
from pathlib import PureWindowsPath

import pytest

from tools import seal_timing_phase_budget as tool


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def trace(steps):
    return dict(steps=steps, agent_rebalances=10, benchmark_rebalances=10,
        agent_rebalance_decoder_samples=2, benchmark_rebalance_decoder_samples=2)


def evidence(fold, architecture, phase="screen", budget=32768, seed=42):
    coordinates = 9 if architecture == "due_phase_residual" else 8
    parameters = 12914 if coordinates == 9 else 12805
    snapshots = {}
    for steps in (8192, 32768) if phase == "screen" else (budget,):
        final = dict(parameters=parameters, tensors=coordinates, active_adam_tensors=coordinates,
            adam_updates=steps // 512 * 20, changed_tensors=coordinates, actor_changed=True, critic_changed=True)
        size = steps // 512
        snapshots[str(steps)] = dict(final=final, estimator=dict(steps=steps, gamma=.995, gae_lambda=1.,
            original_buffers=size, timeout_bootstraps=steps // 240, natural_terminals=0, active_rollout_tails=size, done_rollout_tails=0,
            diagnostics=[dict(maximum_advantage_error=0., maximum_return_error=0.) for _ in range(size)]),
            trained={mode: trace(59) for mode in tool.audit.base.MODES})
    return dict(fold=fold, architecture=architecture, phase=phase, seed=seed, physical_budget=budget,
        actual_rl_steps=budget, separate_benchmark_training_steps=budget, registered_leaves=1,
        initial=dict(parameters=parameters, tensors=coordinates, active_adam_tensors=0, adam_updates=0),
        physical_final=deepcopy(snapshots[str(budget)]["final"]), training=trace(budget),
        initialized={mode: trace(59) for mode in tool.audit.base.MODES}, snapshots=snapshots,
        shared_prefix=dict(shared_steps=8192, shared_original_buffers=16, shared_native_chunks=16,
            extra_rl_steps=0, counted_twice=False) if phase == "screen" else None,
        reproduction=dict(policy_tensors_equal={"initial_model": True, "short_model": True}, agent_native_arrays_equal=True,
            benchmark_native_arrays_equal=True, original_eight_key_ppo_buffers_equal=True, no_extra_environment_steps=True,
            reused_as_sample=False) if phase == "screen" and architecture == "linear" else None)


def trial_set(confirmation_budget=None):
    result = {tool.audit.runner.run_id(fold, architecture, "screen", 42): evidence(fold, architecture)
        for fold in tool.audit.runner.original.SCREEN_FOLDS for architecture in tool.audit.runner.ARCHITECTURES}
    if confirmation_budget:
        result.update({tool.audit.runner.run_id(fold, "linear", "confirm", seed): evidence(fold, "linear", "confirm", confirmation_budget, seed)
            for fold in tool.audit.runner.original.CONFIRM_FOLDS for seed in tool.audit.runner.original.SEEDS})
    return result


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    study = tmp_path / tool.STUDY_DIRECTORY
    study.mkdir(parents=True)
    parent, source = tmp_path / "parent", {}
    for index in range(96):
        path = tmp_path / f"src/frozen_{index:03d}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# immutable\n", encoding="utf-8")
        source[path.relative_to(tmp_path).as_posix()] = tool.sha(path)
    write(parent / "final_audit.json", dict(status="passed", signed_files_sha256=source))
    pin = tool.sha(parent / "final_audit.json")
    monkeypatch.setattr(tool, "PARENT_SHA256", pin)
    monkeypatch.setattr(tool, "PARENT_SIGNED_FILES", 96)
    protocol = dict(reference_root=str(parent), reference_final_audit_sha256=pin, source_hashes=source)
    write(study / "protocol.json", protocol)
    selected = dict(admitted=False, selected_architecture="linear", selected_budget=8192)
    write(study / "selection.json", selected)
    registration = dict(source_files=96, original_sources_unchanged=True)
    monkeypatch.setattr(tool.audit, "registration", lambda *_: registration)
    trials = trial_set()
    for name, row in trials.items():
        directory = study / "runs" / name
        directory.mkdir(parents=True)
        (directory / "artifact.bin").write_bytes(b"synthetic accepted evidence")
        write(directory / "completed.json", dict(files={"artifact.bin": tool.sha(directory / "artifact.bin")}, records=[]))
        row["completion_sha256"] = tool.sha(directory / "completed.json")
    counters = tool.audit.experience_counters(trials)
    monkeypatch.setattr(tool.audit, "final_scope", lambda *_: (selected, counters))
    monkeypatch.setattr(tool.audit, "sealed_trial", lambda *_: pytest.fail("Cannot rerun the financial/model audit"))
    monkeypatch.setattr(tool.audit, "estimator_audit", lambda *_: pytest.fail("Cannot rerun GAE audit"))
    progress = dict(status="passed", scope="complete_authorized_sealed_physical_study", registration=registration,
        selection=selected, counters=counters, sealed_trials=trials, sealed_physical_count=4,
        shared_initials=dict(mapped_full_actor_and_critic_equal=True, checked_physical_models=4, groups=2),
        **{key: 0 for key in tool.ZERO_OPERATIONS})
    write(study / "independent_audit_progress.json", progress)
    write(study / "independent_audit_running.json", dict(status="passed", completed_physical_runs=list(trials),
        progress_sha256=tool.sha(study / "independent_audit_progress.json"), network_forwards=0, environment_steps=0, registered_leaf_writes=0))
    write(study / "study_summary.json", dict(status="screen_failed", actual_confirmation_steps=0, qualified=False, qualified_candidate=None))
    write(study / "summary.json", [])
    write(study / "status.json", dict(status="screen_failed"))
    for fold in tool.audit.runner.original.SCREEN_FOLDS:
        write(study / f"worker_screen_{fold}.json", dict(status="complete"))
    base = dict(command="actual completed command", session_id=None, execution_result_id="actual_chunk", exit_code=0)
    acceptance = dict(status="passed", new_downloads=0, protected_period_evaluated=False, default_model_promoted=False,
        tests=[dict(base, scope="synthetic", passed=12, elapsed_seconds=1.2)],
        formal_processes=[dict(base, stage=stage, fold=fold) for stage, fold in
            (("prepare", None), ("screen", None), ("report", None), ("screen", "2023"), ("screen", "2024"))],
        independent_audit=dict(base, path="independent_audit_progress.json", sha256=tool.sha(study / "independent_audit_progress.json")),
        analysis_processes=[dict(base, tool="tools/analyze_timing_phase_budget.py")])
    write(study / "acceptance.json", acceptance)
    for name in tool.CODE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# registered artifact\n", encoding="utf-8")
    folder = study / "diagnostics/phase_budget_all"
    write(folder / "diagnostics.json", dict(status="complete_for_sealed_stage", stage="all",
        counters=dict(actual_agent_rl_steps=131072, separate_training_benchmark_steps=131072,
            physical_training_runs=4, evaluation_snapshots=8), protocol_sha256=tool.sha(study / "protocol.json"),
        selection_sha256=tool.sha(study / "selection.json"), tool_sha256=tool.sha(tmp_path / "tools/analyze_timing_phase_budget.py"),
        **{key: True for key in ("no_model_load", "no_network_forwards", "no_environment_steps", "no_gradient_updates", "no_trial_leaf_writes")}))
    for name in ("phase_mode.csv", "rollout_scales.csv", "paired_effects.csv"):
        (folder / name).write_text("value\n1\n", encoding="utf-8")
    return study, progress, acceptance


def test_seal_preserves_inputs_counts_physical_work_once_and_refuses_overwrite(frozen):
    study, _, _ = frozen
    before = tool.sha(study / "runs/screen_2023_linear_42/completed.json")
    result = tool.seal(study)
    assert result["counters"]["physical_training_runs"] == 4
    assert result["counters"]["actual_rl_steps"] == 131072 and result["counters"]["snapshot_step_sum"] == 163840
    assert result["counters"]["original_gae_buffers"] == 256
    assert result["audited_proof_statistics"]["snapshot_gae_buffer_views"] == 320
    assert result["counters"]["evaluation_initialized_paths"] == 16 and result["counters"]["evaluation_trained_paths"] == 32
    assert len(result["fresh_linear_prefix_reproduced_trials"]) == 2
    assert tool.sha(study / "runs/screen_2023_linear_42/completed.json") == before
    assert study.relative_to(tool.ROOT).as_posix() + "/final_audit.json" not in result["signed_files_sha256"]
    with pytest.raises(ValueError, match="already exists"):
        tool.seal(study)


@pytest.mark.parametrize("budget,total,buffers,views", [(8192, 180224, 352, 416), (32768, 327680, 640, 704)])
def test_conditional_actual_and_repeated_snapshot_counts_are_separate(budget, total, buffers, views):
    trials = trial_set(budget)
    counters = tool.audit.experience_counters(trials)
    statistics, controls = tool.proof_statistics(trials)
    assert counters["physical_training_runs"] == 10 and counters["evaluated_snapshots"] == 14
    assert counters["actual_rl_steps"] == total and counters["original_gae_buffers"] == buffers
    assert statistics["snapshot_gae_buffer_views"] == views
    assert len(controls) == 2


@pytest.mark.parametrize("field,value", [("status", "partial"), ("scope", "frozen_registration"), ("network_forwards", 1)])
def test_incomplete_or_executing_audit_cannot_seal(frozen, field, value):
    study, progress, _ = frozen
    progress[field] = value
    write(study / "independent_audit_progress.json", progress)
    with pytest.raises(ValueError, match="incomplete"):
        tool.seal(study)
    assert not (study / "final_audit.json").exists()


@pytest.mark.parametrize("change", ["missing_worker_exit", "failed_exit", "missing_identifier", "wrong_progress_sha", "missing_analysis"])
def test_actual_exit_evidence_is_not_inferred_from_file_existence(frozen, change):
    study, _, acceptance = frozen
    if change == "missing_worker_exit":
        acceptance["formal_processes"].pop()
    elif change == "failed_exit":
        acceptance["formal_processes"][0]["exit_code"] = 1
    elif change == "missing_identifier":
        acceptance["formal_processes"][0]["execution_result_id"] = None
    elif change == "wrong_progress_sha":
        acceptance["independent_audit"]["sha256"] = "0" * 64
    else:
        acceptance["analysis_processes"] = []
    write(study / "acceptance.json", acceptance)
    with pytest.raises(ValueError):
        tool.seal(study)
    assert not (study / "final_audit.json").exists()


@pytest.mark.parametrize("change", ["parent", "source", "extra_source", "leaf", "marker", "running_worker", "missing_diagnostic", "wrong_root"])
def test_inventory_and_complete_receipts_are_mandatory(frozen, change):
    study, _, _ = frozen
    if change == "parent":
        (tool.ROOT / "parent/final_audit.json").write_text("{}", encoding="utf-8")
    elif change == "source":
        (tool.ROOT / "src/frozen_000.py").write_text("changed", encoding="utf-8")
    elif change == "extra_source":
        (tool.ROOT / "src/extra.py").write_text("extra", encoding="utf-8")
    elif change == "leaf":
        (study / "runs/screen_2023_linear_42/artifact.bin").write_bytes(b"changed")
    elif change == "marker":
        (study / "runs/screen_2023_linear_42/completed.json").write_text("{}", encoding="utf-8")
    elif change == "running_worker":
        write(study / "worker_screen_2024.json", dict(status="running"))
    elif change == "missing_diagnostic":
        (study / "diagnostics/phase_budget_all/paired_effects.csv").unlink()
    else:
        study = study / "not_registered"
    with pytest.raises((ValueError, FileNotFoundError)):
        tool.seal(study)
    assert not (study / "final_audit.json").exists()


def test_native_windows_source_registry_is_compared_without_protocol_rewrite(frozen):
    study, _, _ = frozen
    protocol = tool.read(study / "protocol.json")
    protocol["source_hashes"] = {str(PureWindowsPath(name)): value for name, value in protocol["source_hashes"].items()}
    write(study / "protocol.json", protocol)
    path = study / "diagnostics/phase_budget_all/diagnostics.json"
    diagnostic = tool.read(path)
    diagnostic["protocol_sha256"] = tool.sha(study / "protocol.json")
    write(path, diagnostic)
    before = (study / "protocol.json").read_bytes()
    tool.seal(study)
    assert (study / "protocol.json").read_bytes() == before


@pytest.mark.parametrize("change", ["duplicated_prefix", "missing_control", "typed_tensor_flag", "nine_adam_missing", "gae_error_nan"])
def test_cold_proof_claims_cannot_hide_missing_prefix_or_parameter_evidence(change):
    trials = trial_set()
    row = trials["screen_2023_linear_42"]
    if change == "duplicated_prefix":
        row["shared_prefix"]["counted_twice"] = True
    elif change == "missing_control":
        row["reproduction"] = None
    elif change == "typed_tensor_flag":
        row["reproduction"]["policy_tensors_equal"]["short_model"] = 1
    elif change == "nine_adam_missing":
        trials["screen_2023_due_phase_residual_42"]["snapshots"]["8192"]["final"]["active_adam_tensors"] = 8
    else:
        row["snapshots"]["8192"]["estimator"]["diagnostics"][0]["maximum_advantage_error"] = float("nan")
    with pytest.raises(ValueError):
        tool.proof_statistics(trials)
