"""Pure temporary-manifest admission tests: no models, gradients or trading."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from tools.archive import seal_timing_credit as tool


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def evidence(phase="screen", value=.95, fold="2023", reward="log_net"):
    steps = 8192 if phase == "screen" else 16384
    size = steps // 512
    trace = dict(steps=steps, agent_rebalances=13, benchmark_rebalances=10,
                 agent_rebalance_decoder_samples=4, benchmark_rebalance_decoder_samples=4)
    case = dict(trace, steps=59)
    proof = dict(policy_tensors_equal={"initial_model": True, "final_model": True},
        training_native_arrays_equal=True, benchmark_native_arrays_equal=True, rollout_buffers_equal=True,
        reused_as_sample=False, no_extra_forward=True, no_extra_environment_steps=True)
    return dict(phase=phase, fold=fold, gae_lambda=value, reward_basis=reward,
        actual_rl_steps=steps, separate_benchmark_training_steps=steps, registered_leaves=size + 1,
        initial=dict(parameters=12805), final=dict(parameters=12805, active_adam_tensors=8, adam_updates=size * 20),
        training=trace, cases={str(index): deepcopy(case) for index in range(8)},
        estimator=dict(gamma=.995, gae_lambda=value, original_buffers=size, steps=steps,
            timeout_bootstraps=30, natural_terminals=4,
            diagnostics=[dict(maximum_advantage_error=0., maximum_return_error=0.) for _ in range(size)]),
        reproduction=proof if phase == "screen" and value == .95 else None)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    study = tmp_path / tool.STUDY_DIRECTORY
    study.mkdir(parents=True)
    parent = tmp_path / "parent"
    source = {}
    for index in range(96):
        path = tmp_path / f"src/frozen_{index:03d}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# immutable\n", encoding="utf-8")
        source[path.relative_to(tmp_path).as_posix()] = tool.sha(path)
    write(parent / "final_audit.json", dict(status="passed", signed_files_sha256=source))
    pin = tool.sha(parent / "final_audit.json")
    monkeypatch.setattr(tool, "PARENT_SHA256", pin)
    monkeypatch.setattr(tool, "PARENT_SIGNED_FILES", len(source))
    protocol = dict(reference_root=str(parent), reference_final_audit_sha256=pin,
        source_hashes=source, parameters=12805, feature_names=[str(i) for i in range(129)])
    write(study / "protocol.json", protocol)
    selected = dict(admitted=False, selected_reward="log_net", selected_lambda=.95)
    write(study / "selection.json", selected)
    registration = dict(source_files=96, original_sources_unchanged=True)
    monkeypatch.setattr(tool.audit, "registration", lambda *_: registration)
    monkeypatch.setattr(tool.audit, "final_scope", lambda *_: selected)
    monkeypatch.setattr(tool.audit, "sealed_trial", lambda *_: pytest.fail("Cannot rerun native/model audit"))
    trials = {}
    for fold in ("2023", "2024"):
        for reward in ("log_net", "relative_log"):
            for value in (.95, 1.):
                name = f"screen_{fold}_{reward}_{value}"
                directory = study / "runs" / name
                row = evidence(fold=fold, reward=reward, value=value)
                context_files, records = {}, []
                for index in range(16):
                    path = directory / f"boundaries/{index}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(path, last_dones=np.array([index % 4 == 0], dtype=np.bool_))
                    relative = path.relative_to(directory).as_posix()
                    context_files[relative] = tool.sha(path)
                    records.append(dict(path=relative))
                write(directory / "final_audit.json", dict(credit_estimator_buffers=records))
                context_files["final_audit.json"] = tool.sha(directory / "final_audit.json")
                write(directory / "completed.json", dict(files=context_files, record=dict(steps=8192)))
                row["completion_sha256"] = tool.sha(directory / "completed.json")
                trials[name] = row
    progress = dict(status="passed", scope="complete_authorized_sealed_native_study", registration=registration,
        selection=selected, sealed_trials=trials, sealed_trial_count=8, actual_rl_steps=65536,
        shared_initials=dict(full_initial_actor_and_critic_equal=True, checked_cells=8, groups=2),
        **{key: 0 for key in tool.ZERO_OPERATIONS})
    write(study / "independent_audit_progress.json", progress)
    write(study / "independent_audit_running.json", dict(status="passed", progress_sha256=tool.sha(study / "independent_audit_progress.json"),
        completed_trials=list(trials), network_forwards=0, environment_steps=0, registered_leaf_writes=0))
    summary = dict(status="screen_failed", actual_training_steps=65536, final_snapshots=8,
        qualified=False, qualified_candidate=None, test_evaluated=False, actionable=False, actual_confirmation_steps=0)
    write(study / "study_summary.json", summary)
    write(study / "status.json", {key: summary[key] for key in
        ("status", "actual_training_steps", "final_snapshots", "qualified", "test_evaluated", "actionable")})
    write(study / "summary.json", [])
    for fold in ("2023", "2024"):
        write(study / f"worker_screen_{fold}.json", dict(status="complete"))
    base = dict(command="actual command", session_id=None, execution_result_id="actual_chunk", exit_code=0)
    formal = [dict(base, stage=stage, fold=fold) for stage, fold in
              (("prepare", None), ("screen", None), ("report", None), ("screen", "2023"), ("screen", "2024"))]
    acceptance = dict(status="passed", new_downloads=0, protected_period_evaluated=False, default_model_promoted=False,
        tests=[dict(base, scope="synthetic", passed=12, elapsed_seconds=1.5)], formal_processes=formal,
        independent_audit=dict(base, path="independent_audit_progress.json", sha256=tool.sha(study / "independent_audit_progress.json")),
        analysis_processes=[dict(base, tool=name) for name in ("tools/archive/analyze_timing_credit.py", "tools/archive/analyze_credit_counterfactual.py")])
    write(study / "acceptance.json", acceptance)
    for name in tool.CODE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# registered\n", encoding="utf-8")
    for stage in ("all",):
        folder = study / f"diagnostics/credit_{stage}"
        write(folder / "diagnostics.json", dict(status="complete_for_sealed_stage", stage=stage, runs=8,
            actual_training_steps=65536, protocol_sha256=tool.sha(study / "protocol.json"),
            selection_sha256=tool.sha(study / "selection.json"), tool_sha256=tool.sha(tmp_path / "tools/archive/analyze_timing_credit.py"),
            **{key: True for key in ("no_model_load", "no_tensor_deserialization", "no_forward", "no_environment_steps", "no_gradient_updates", "no_trial_leaf_writes")}))
        for name in ("phase_mode.csv", "rollout_scales.csv"):
            (folder / name).write_text("value\n1\n", encoding="utf-8")
    folder = study / "diagnostics/credit_counterfactual"
    write(folder / "diagnostics.json", dict(status="complete_for_sealed_screen", registered_trials=8,
        original_training_steps=65536, analyzed_original_buffers=128, protocol_sha256=tool.sha(study / "protocol.json"),
        selection_sha256=tool.sha(study / "selection.json"), tool_sha256=tool.sha(tmp_path / "tools/archive/analyze_credit_counterfactual.py"),
        no_tensor_deserialization=True, no_source_or_signed_leaf_writes=True,
        **{key: 0 for key in ("added_rl_experience", "added_environment_steps", "added_forward_calls", "added_gradient_updates")}))
    (folder / "same_path_scales.csv").write_text("value\n1\n", encoding="utf-8")
    return study, progress, acceptance


def test_seal_signs_exact_inputs_without_models_or_leaf_writes(frozen):
    study, _, _ = frozen
    before = tool.sha(study / "runs/screen_2023_log_net_0.95/completed.json")
    result = tool.seal(study)
    assert result["counters"]["actual_agent_rl_steps"] == 65536
    assert result["counters"]["actual_adam_updates"] == 2560
    assert result["counters"]["original_gae_buffers"] == 128
    assert result["counters"]["active_rollout_tails"] == 96
    assert result["counters"]["done_rollout_tails"] == 32
    assert len(result["fresh_lambda95_reproduced_trials"]) == 4
    assert result["predecessor_signed_files_verified"] == 96
    assert tool.sha(study / "runs/screen_2023_log_net_0.95/completed.json") == before
    assert study.relative_to(tool.ROOT).as_posix() + "/final_audit.json" not in result["signed_files_sha256"]
    assert all(not name.endswith("human_report.md") for name in result["signed_files_sha256"])
    with pytest.raises(ValueError, match="already exists"):
        tool.seal(study)


@pytest.mark.parametrize("field,value", [("status", "partial"), ("scope", "frozen_registration"), ("network_forwards", 1)])
def test_partial_or_executing_audit_cannot_seal(frozen, field, value):
    study, progress, _ = frozen
    progress[field] = value
    write(study / "independent_audit_progress.json", progress)
    with pytest.raises(ValueError, match="not complete"):
        tool.seal(study)
    assert not (study / "final_audit.json").exists()


@pytest.mark.parametrize("change", ["missing_worker", "extra_confirm", "nonzero_exit", "missing_identifier", "wrong_audit_sha"])
def test_root_actual_exit_evidence_is_required(frozen, change):
    study, _, acceptance = frozen
    if change == "missing_worker":
        acceptance["formal_processes"].pop()
    elif change == "extra_confirm":
        acceptance["formal_processes"].append(dict(acceptance["formal_processes"][0], stage="confirm"))
    elif change == "nonzero_exit":
        acceptance["formal_processes"][0]["exit_code"] = 1
    elif change == "missing_identifier":
        acceptance["formal_processes"][0]["execution_result_id"] = None
    else:
        acceptance["independent_audit"]["sha256"] = "0" * 64
    write(study / "acceptance.json", acceptance)
    with pytest.raises(ValueError):
        tool.seal(study)
    assert not (study / "final_audit.json").exists()


@pytest.mark.parametrize("change", ["parent", "source", "extra_source", "leaf", "marker", "missing_diagnostics", "wrong_root"])
def test_inventory_drift_refuses_new_seal(frozen, change):
    study, _, _ = frozen
    root = tool.ROOT
    if change == "parent":
        (root / "parent/final_audit.json").write_text("{}", encoding="utf-8")
    elif change == "source":
        (root / "src/frozen_000.py").write_text("changed", encoding="utf-8")
    elif change == "extra_source":
        (root / "src/unregistered.py").write_text("new", encoding="utf-8")
    elif change == "leaf":
        (study / "runs/screen_2023_log_net_0.95/boundaries/0.npz").write_bytes(b"changed")
    elif change == "marker":
        (study / "runs/screen_2023_log_net_0.95/completed.json").write_text("{}", encoding="utf-8")
    elif change == "missing_diagnostics":
        (study / "diagnostics/credit_all/phase_mode.csv").unlink()
    else:
        study = study / "wrong"
    with pytest.raises((ValueError, FileNotFoundError)):
        tool.seal(study)
    assert not (study / "final_audit.json").exists()


@pytest.mark.parametrize("field,value", [("maximum_advantage_error", -1e-6), ("maximum_return_error", float("inf"))])
def test_unaccepted_actual_boundary_error_refused(field, value):
    rows = {str(index): evidence(value=.95 if index < 4 else 1.) for index in range(8)}
    rows["0"]["estimator"]["diagnostics"][0][field] = value
    with pytest.raises(ValueError, match="statistic is invalid"):
        tool.counters(rows)


def test_confirmation_counters_are_actual_not_screen_prefixes():
    rows = {str(index): evidence(value=.95 if index < 4 else 1.) for index in range(8)}
    rows.update({"confirm_" + str(index): evidence(phase="confirm") for index in range(6)})
    result, _, _ = tool.counters(rows)
    assert result["actual_agent_rl_steps"] == 163840
    assert result["confirmation_trials"] == 6
    assert result["actual_adam_updates"] == 6400
    assert result["original_gae_buffers"] == 320


@pytest.mark.parametrize("name", ["../escape", "src\\frozen_000.py", "/absolute", "src/./frozen_000.py", "C:/file"])
def test_workspace_inventory_rejects_noncanonical_paths(frozen, name):
    with pytest.raises(ValueError):
        tool.workspace_path(name)


def test_conflicting_sha_and_postwrite_recheck(frozen, monkeypatch):
    study, _, _ = frozen
    files = {}
    path = tool.ROOT / "src/frozen_000.py"
    tool.add_file(files, path)
    files["src/frozen_000.py"] = "0" * 64
    with pytest.raises(ValueError, match="Conflicting"):
        tool.add_file(files, path)
    original = tool.verify_files
    calls = []
    def check(mapping):
        calls.append(len(mapping))
        return original(mapping)
    monkeypatch.setattr(tool, "verify_files", check)
    tool.seal(study)
    assert len(calls) == 3
    assert calls[-1] == calls[-2]


def test_incomplete_worker_receipt_cannot_seal(frozen):
    study, _, _ = frozen
    write(study / "worker_screen_2024.json", dict(status="running"))
    with pytest.raises(ValueError, match="Worker receipt is not complete"):
        tool.seal(study)
    assert not (study / "final_audit.json").exists()


def test_native_windows_source_keys_match_parent_posix_without_rewriting_protocol(frozen):
    study, _, _ = frozen
    protocol = tool.read(study / "protocol.json")
    protocol["source_hashes"] = {str(tool.PureWindowsPath(name)): digest for name, digest in protocol["source_hashes"].items()}
    write(study / "protocol.json", protocol)
    before = (study / "protocol.json").read_bytes()
    for folder in ("credit_all", "credit_counterfactual"):
        path = study / "diagnostics" / folder / "diagnostics.json"
        diagnostic = tool.read(path)
        diagnostic["protocol_sha256"] = tool.sha(study / "protocol.json")
        write(path, diagnostic)
    result = tool.seal(study)
    assert result["source_count"] == 96
    assert (study / "protocol.json").read_bytes() == before
    assert all("\\" not in name for name in result["signed_files_sha256"])


@pytest.mark.parametrize("name", [r"C:\src\agents\a.py", r"\src\agents\a.py", r".\src\agents\a.py",
    r"src\..\agents\a.py", r"src\.\agents\a.py", r"src\\agents\a.py", r"src\agents/a.py",
    "src/./agents/a.py", "../src/agents/a.py", "/src/agents/a.py", "src/agents/a.py:stream"])
def test_registered_source_paths_reject_absolute_drive_dot_and_noncanonical_forms(name):
    with pytest.raises(ValueError):
        tool.canonical_source_hashes({name: "0" * 64})


def test_registered_source_alias_conflicts_are_rejected():
    with pytest.raises(ValueError, match="alias"):
        tool.canonical_source_hashes({r"src\agents\a.py": "0" * 64, "src/agents/a.py": "1" * 64})
    with pytest.raises(ValueError, match="alias"):
        tool.canonical_source_hashes({r"src\agents\a.py": "0" * 64, "src/agents/A.py": "0" * 64})
