"""Seal a completed Credit study using existing read-only audit evidence."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.archive import audit_timing_credit as audit

PARENT_SHA256 = "1e8b7363582b452f6f1b42a79049cdd94f067b21a244abf8b8734e4a390453df"
PARENT_SIGNED_FILES = 3244
STUDY_DIRECTORY = "reports/runs/ppo_timing_credit_20261015"
CODE_FILES = (
    "tools/archive/research_timing_credit.py", "tools/archive/audit_timing_credit.py",
    "tools/archive/analyze_timing_credit.py", "tools/archive/analyze_credit_counterfactual.py",
    "tools/archive/seal_timing_credit.py", "tests/test_research_timing_credit.py",
    "tests/test_timing_credit_review.py", "tests/test_timing_credit_audit_tool.py",
    "tests/test_analyze_timing_credit.py", "tests/test_credit_counterfactual.py",
    "tests/test_seal_timing_credit.py", "doc/ppo_timing_credit_protocol_20261015.md",
)
ROOT_FILES = (
    "protocol.json", "selection.json", "summary.json", "study_summary.json", "status.json",
    "acceptance.json", "independent_audit_progress.json", "independent_audit_running.json",
)
ZERO_OPERATIONS = (
    "model_instances", "network_forwards", "environment_steps", "gradient_calls",
    "optimizer_steps", "registered_leaf_writes", "frozen_source_writes",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def workspace_path(name):
    require(isinstance(name, str), "Signed file path must be text")
    value = PurePosixPath(name)
    require("\\" not in name and not value.is_absolute()
            and value.as_posix() == name and ":" not in name
            and not any(part in (".", "..") for part in value.parts),
            "Signed paths must be canonical workspace-relative POSIX paths")
    path = (ROOT / name).resolve()
    require(path.is_relative_to(ROOT.resolve()) and path.is_file(), "Signed file escaped workspace or is missing")
    return path


def add_file(files, path, expected=None):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()), "Cannot sign a file outside the workspace")
    name = path.relative_to(ROOT.resolve()).as_posix()
    digest = sha(workspace_path(name))
    require(expected is None or digest == expected, "Registered file SHA differs: " + name)
    require(name not in files or files[name] == digest, "Conflicting signed path SHA: " + name)
    files[name] = digest


def verify_files(files):
    aliases = {}
    for name, expected in files.items():
        path = workspace_path(name)
        alias = str(path).casefold()
        require(alias not in aliases or aliases[alias] == expected, "Conflicting signed alias SHA: " + name)
        aliases[alias] = expected
        require(sha(path) == expected, "Signed file changed: " + name)


def canonical_source_hashes(registered):
    """Normalize the frozen native-Windows source registry, never its bytes."""
    require(isinstance(registered, dict), "Registered source hashes must be a mapping")
    result, aliases = {}, {}
    for name, digest in registered.items():
        require(isinstance(name, str), "Registered source path must be text")
        path = PureWindowsPath(name) if "\\" in name else PurePosixPath(name)
        require(not path.is_absolute() and not getattr(path, "drive", "") and not path.root
                and ":" not in name and str(path) == name
                and not any(part in (".", "..") for part in path.parts),
                "Registered source path must be canonical and workspace-relative")
        canonical = path.as_posix()
        require(canonical.startswith("src/") and canonical.endswith(".py"), "Registered source is not a src Python file")
        alias = canonical.casefold()
        require(alias not in aliases or aliases[alias] == (canonical, digest), "Conflicting registered source alias")
        aliases[alias] = (canonical, digest)
        require(canonical not in result or result[canonical] == digest, "Conflicting registered source SHA")
        result[canonical] = digest
    return result


def process_evidence(row):
    require(isinstance(row, dict) and type(row.get("exit_code")) is int and row["exit_code"] == 0
            and isinstance(row.get("command"), str) and bool(row["command"].strip()),
            "Missing successful root-observed command exit")
    identifiers = (row.get("session_id"), row.get("execution_result_id"))
    require(any(type(value) in (str, int) and bool(str(value)) for value in identifiers),
            "Root process must identify its actual session or execution result")


def verify_acceptance(acceptance, selected, progress_path):
    require(acceptance.get("status") == "passed" and acceptance.get("new_downloads") == 0
            and acceptance.get("protected_period_evaluated") is False
            and acceptance.get("default_model_promoted") is False,
            "Root acceptance is absent, incomplete or changed protected-period/promotion flags")
    tests = acceptance.get("tests", [])
    require(bool(tests), "Root must record actual completed tests")
    for row in tests:
        process_evidence(row)
        require(type(row.get("passed")) is int and row["passed"] > 0
                and type(row.get("elapsed_seconds")) in (float, int)
                and math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] >= 0,
                "Actual test count or duration is missing")
    expected = {("prepare", None), ("screen", None), ("report", None),
                *(('screen', fold) for fold in audit.runner.original.SCREEN_FOLDS)}
    if selected["admitted"]:
        expected |= {("confirm", None), *(('confirm', fold) for fold in audit.runner.original.CONFIRM_FOLDS)}
    processes = acceptance.get("formal_processes", [])
    require(len(processes) == len(expected)
            and {(row.get("stage"), row.get("fold")) for row in processes} == expected,
            "Actual formal process exits do not cover exactly the authorized stages and workers")
    for row in processes:
        process_evidence(row)
    independent = acceptance.get("independent_audit", {})
    process_evidence(independent)
    require(independent.get("path") == "independent_audit_progress.json"
            and independent.get("sha256") == sha(progress_path), "Acceptance does not bind the actual final audit progress")
    analyses = acceptance.get("analysis_processes", [])
    required = {"tools/archive/analyze_timing_credit.py", "tools/archive/analyze_credit_counterfactual.py"}
    require(len(analyses) == len(required) and {row.get("tool") for row in analyses} == required,
            "Actual sealed-study diagnostic process exits are missing")
    for row in analyses:
        process_evidence(row)


def counters(trials):
    require(len(trials) in (8, 14), "Unregistered Credit trial count")
    result = dict(sealed_trials=len(trials), final_snapshots=len(trials), confirmation_trials=0,
        actual_agent_rl_steps=0, separate_training_benchmark_steps=0, actual_adam_updates=0,
        agent_native_steps=0, benchmark_native_steps=0, registered_leaf_files=0,
        agent_rebalance_rows=0, benchmark_rebalance_rows=0, agent_decoder_samples=0,
        benchmark_decoder_samples=0, original_gae_buffers=0, audited_gae_steps=0,
        timeout_bootstraps=0, natural_terminals=0, active_rollout_tails=0, done_rollout_tails=0)
    errors = dict(maximum_advantage_error=0., maximum_return_error=0.)
    reproduction = []
    for name, row in trials.items():
        result["confirmation_trials"] += row["phase"] == "confirm"
        result["actual_agent_rl_steps"] += row["actual_rl_steps"]
        result["separate_training_benchmark_steps"] += row["separate_benchmark_training_steps"]
        result["actual_adam_updates"] += row["final"]["adam_updates"]
        result["registered_leaf_files"] += row["registered_leaves"]
        require(row["initial"]["parameters"] == row["final"]["parameters"] == 12805
                and row["final"]["active_adam_tensors"] == 8, "Accepted model parameter/Adam schema differs")
        for trace in (row["training"], *row["cases"].values()):
            result["agent_native_steps"] += trace["steps"]
            result["benchmark_native_steps"] += trace["steps"]
            for target, source in (("agent_rebalance_rows", "agent_rebalances"),
                                   ("benchmark_rebalance_rows", "benchmark_rebalances"),
                                   ("agent_decoder_samples", "agent_rebalance_decoder_samples"),
                                   ("benchmark_decoder_samples", "benchmark_rebalance_decoder_samples")):
                result[target] += trace[source]
        estimator = row["estimator"]
        require(estimator["gamma"] == .995 and estimator["gae_lambda"] == row["gae_lambda"]
                and estimator["steps"] == row["actual_rl_steps"]
                and estimator["original_buffers"] == len(estimator["diagnostics"]), "Accepted GAE scope differs")
        result["original_gae_buffers"] += estimator["original_buffers"]
        result["audited_gae_steps"] += estimator["steps"]
        for key in ("timeout_bootstraps", "natural_terminals"):
            result[key] += estimator[key]
        for diagnostic in estimator["diagnostics"]:
            for key in errors:
                value = diagnostic[key]
                require(type(value) in (float, int) and math.isfinite(value) and value >= 0,
                        "Accepted actual-boundary GAE error statistic is invalid")
                errors[key] = max(errors[key], value)
        if row["phase"] == "screen" and row["gae_lambda"] == .95:
            proof = row["reproduction"]
            tensor_flags = proof.get("policy_tensors_equal", {}) if proof else {}
            require(proof and set(tensor_flags) == {"initial_model", "final_model"}
                    and all(value is True for value in tensor_flags.values())
                    and all(proof.get(key) is True for key in ("training_native_arrays_equal", "benchmark_native_arrays_equal",
                                                               "rollout_buffers_equal", "no_extra_forward", "no_extra_environment_steps"))
                    and proof.get("reused_as_sample") is False, "A fresh lambda95 control lacks complete reproduction evidence")
            reproduction.append(name)
    require(len(reproduction) == 4, "The four fresh reward/fold lambda95 controls are missing")
    return result, errors, sorted(reproduction)


def diagnostic_files(study, protocol, selected, trials, files):
    expected_steps = sum(row["actual_rl_steps"] for row in trials.values())
    for stage in ("screen", "confirm", "all"):
        folder = study / "diagnostics" / ("credit_" + stage)
        if not folder.exists():
            require(stage != "all", "Complete actual-study diagnostics are missing")
            continue
        require(stage != "confirm" or selected["admitted"], "Unauthorized confirmation diagnostics")
        report = read(folder / "diagnostics.json")
        count = 8 if stage == "screen" else 6 if stage == "confirm" else len(trials)
        steps = 65536 if stage == "screen" else 98304 if stage == "confirm" else expected_steps
        require(report.get("status") == "complete_for_sealed_stage" and report.get("stage") == stage
                and report.get("runs") == count and report.get("actual_training_steps") == steps
                and report.get("protocol_sha256") == sha(study / "protocol.json")
                and report.get("selection_sha256") == sha(study / "selection.json")
                and report.get("tool_sha256") == sha(ROOT / "tools/archive/analyze_timing_credit.py")
                and all(report.get(key) is True for key in ("no_model_load", "no_tensor_deserialization", "no_forward",
                    "no_environment_steps", "no_gradient_updates", "no_trial_leaf_writes")), "Diagnostic scope/identity differs")
        for name in ("diagnostics.json", "phase_mode.csv", "rollout_scales.csv"):
            add_file(files, folder / name)
    folder = study / "diagnostics/credit_counterfactual"
    report = read(folder / "diagnostics.json")
    require(report.get("status") == "complete_for_sealed_screen" and report.get("registered_trials") == 8
            and report.get("original_training_steps") == 65536 and report.get("analyzed_original_buffers") == 128
            and report.get("protocol_sha256") == sha(study / "protocol.json")
            and report.get("selection_sha256") == sha(study / "selection.json")
            and report.get("tool_sha256") == sha(ROOT / "tools/archive/analyze_credit_counterfactual.py")
            and report.get("no_tensor_deserialization") is True
            and report.get("no_source_or_signed_leaf_writes") is True
            and all(type(report.get(key)) is int and report[key] == 0 for key in
                    ("added_rl_experience", "added_environment_steps", "added_forward_calls", "added_gradient_updates")),
            "Same-path diagnostics are absent or not the registered eight original screen paths")
    for name in ("diagnostics.json", "same_path_scales.csv"):
        add_file(files, folder / name)


def build_seal(study):
    study = Path(study).resolve()
    require(study == (ROOT / STUDY_DIRECTORY).resolve(), "Seal only the registered Credit root")
    require(not (study / "final_audit.json").exists(), "Final audit already exists; no overwrite or reseal")
    protocol = read(study / "protocol.json")
    parent_path = Path(protocol["reference_root"]) / "final_audit.json"
    require(sha(parent_path) == PARENT_SHA256 and protocol["reference_final_audit_sha256"] == PARENT_SHA256,
            "Predecessor final audit does not match the hard pin")
    parent = read(parent_path)
    files = dict(parent["signed_files_sha256"])
    require(parent.get("status") == "passed" and len(files) == PARENT_SIGNED_FILES, "Incomplete predecessor signed inventory")
    verify_files(files)
    add_file(files, parent_path, PARENT_SHA256)
    sources = {key: value for key, value in files.items() if key.startswith("src/") and key.endswith(".py")}
    actual_sources = {path.relative_to(ROOT).as_posix() for path in (ROOT / "src").rglob("*.py")}
    require(len(sources) == 96 and set(sources) == actual_sources
            and sources == canonical_source_hashes(protocol["source_hashes"]),
            "Frozen 96-source set or bytes changed")
    progress_path = study / "independent_audit_progress.json"
    progress = read(progress_path)
    require(progress.get("status") == "passed" and progress.get("scope") == "complete_authorized_sealed_native_study"
            and all(type(progress.get(key)) is int and progress[key] == 0 for key in ZERO_OPERATIONS),
            "Independent native audit is not complete or performed extra execution")
    require(progress["registration"] == audit.registration(study, protocol), "Final audit registration differs")
    trials = progress["sealed_trials"]
    selected = audit.final_scope(study, protocol, trials)
    require(progress["selection"] == selected and progress["sealed_trial_count"] == len(trials)
            and progress["actual_rl_steps"] == sum(row["actual_rl_steps"] for row in trials.values()),
            "Final audit selection or complete-scope counters differ")
    running = read(study / "independent_audit_running.json")
    require(running.get("status") == "passed" and running.get("progress_sha256") == sha(progress_path)
            and set(running.get("completed_trials", [])) == set(trials)
            and all(type(running.get(key)) is int and running[key] == 0 for key in
                    ("network_forwards", "environment_steps", "registered_leaf_writes")), "Final auditor running receipt is incomplete")
    shared = progress["shared_initials"]
    require(shared.get("full_initial_actor_and_critic_equal") is True and shared.get("checked_cells") == len(trials),
            "Full initial Actor/Critic equivalence is missing")
    acceptance = read(study / "acceptance.json")
    verify_acceptance(acceptance, selected, progress_path)
    result, errors, reproduced = counters(trials)
    markers = {}
    for name, row in trials.items():
        directory = study / "runs" / name
        marker = directory / "completed.json"
        require(sha(marker) == row["completion_sha256"], "Accepted completion marker changed")
        completed = read(marker)
        require(row["registered_leaves"] == len(completed["files"]), "Accepted registered leaf count differs")
        audit.base.verify_leaf_files(directory, completed["files"])
        add_file(files, marker, row["completion_sha256"])
        markers[name] = row["completion_sha256"]
        for leaf, digest in completed["files"].items():
            require(PurePosixPath(leaf).as_posix() == leaf and "\\" not in leaf
                    and not PurePosixPath(leaf).is_absolute() and ".." not in PurePosixPath(leaf).parts,
                    "Completion leaf path is not canonical")
            add_file(files, directory / leaf, digest)
        runtime = read(directory / "final_audit.json")
        require(len(runtime["credit_estimator_buffers"]) == row["estimator"]["original_buffers"],
                "Actual captured boundary buffer count differs")
        for record in runtime["credit_estimator_buffers"]:
            require(record["path"] in completed["files"], "Captured boundary is not a registered trial leaf")
            with np.load(directory / record["path"], allow_pickle=False) as context:
                require(context["last_dones"].dtype == np.bool_ and context["last_dones"].shape == (1,),
                        "Captured tail done flag differs")
                result["done_rollout_tails"] += int(context["last_dones"][0])
                result["active_rollout_tails"] += int(not context["last_dones"][0])
    for name in ROOT_FILES:
        add_file(files, study / name)
    for phase, folds in (("screen", audit.runner.original.SCREEN_FOLDS),
                         ("confirm", audit.runner.original.CONFIRM_FOLDS if selected["admitted"] else ())):
        for fold in folds:
            worker_path = study / f"worker_{phase}_{fold}.json"
            require(read(worker_path).get("status") == "complete", "Worker receipt is not complete")
            add_file(files, worker_path)
    for name in CODE_FILES:
        add_file(files, ROOT / name)
    diagnostic_files(study, protocol, selected, trials, files)
    summary = read(study / "study_summary.json")
    require(summary["status"] == ("complete" if selected["admitted"] else "screen_failed"), "Final report is unfinished")
    require(read(study / "status.json") == {key: summary[key] for key in
        ("status", "actual_training_steps", "final_snapshots", "qualified", "test_evaluated", "actionable")},
        "Final status is not the completed report")
    verify_files(files)
    return dict(schema_version=1, status="passed", study="timing_credit_20261015",
        created_utc=datetime.now(timezone.utc).isoformat(), source_count=len(sources),
        predecessor_signed_files_verified=PARENT_SIGNED_FILES, predecessor_final_audit_sha256=PARENT_SHA256,
        scope=progress["scope"], counters=result, actual_boundary_gae_maximum_errors=errors,
        actual_boundary_gae_auditor_tolerances=dict(absolute=1e-6, relative=1e-6),
        full_initial_actor_and_critic_equal=True, fresh_lambda95_reproduced_trials=reproduced,
        shared_initials=shared, initial_equivalence_scope="Within each registered fold/phase/seed group",
        policy_parameters=protocol["parameters"], policy_input_dimensions=len(protocol["feature_names"]),
        decoder_scope="fixed first and last true REBALANCE per native chunk; all HOLD coordinates checked",
        independent_audit_operations={key: progress[key] for key in ZERO_OPERATIONS},
        seal_operations=dict(network_forwards=0, environment_steps=0, gradient_calls=0, model_loads=0, registered_leaf_writes=0),
        actual_audit_process_exit_code=acceptance["independent_audit"]["exit_code"],
        audit_progress_sha256=sha(progress_path), root_process_evidence_sha256=sha(study / "acceptance.json"),
        root_process_evidence="Signed acceptance.json records actual observed exits; sealer did not rerun those processes",
        screen_admitted=selected["admitted"], confirmation_steps_authorized=summary["actual_confirmation_steps"],
        gate_passed=summary["qualified"], qualified_candidate=summary["qualified_candidate"],
        protected_period_evaluated=False, default_model_promoted=False, sealed_markers=markers,
        signed_files_sha256=dict(sorted(files.items())))


def seal(study):
    result = build_seal(study)
    output = Path(study) / "final_audit.json"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    verify_files(result["signed_files_sha256"])
    print(json.dumps(dict(status="passed", final_audit=str(output), sha256=sha(output),
        signed_files=len(result["signed_files_sha256"]), counters=result["counters"]), sort_keys=True))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / STUDY_DIRECTORY))
    args = parser.parse_args(argv)
    return seal(Path(args.root).resolve())


if __name__ == "__main__":
    main()
