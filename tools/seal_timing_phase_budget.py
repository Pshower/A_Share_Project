"""Seal accepted physical PhaseBudget evidence; never rerun its financial audit."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_timing_phase_budget as audit
from tools.seal_timing_credit import sha, read, process_evidence, canonical_source_hashes, ZERO_OPERATIONS

PARENT_SHA256 = "fd3449e3d0fcb2df87fa9cd3d87274fa3d93290bb77a3b8d111c5c04751e8955"
PARENT_SIGNED_FILES = 4676
STUDY_DIRECTORY = "reports/runs/ppo_timing_phase_budget_20261015"
CODE_FILES = (*audit.runner.COMPONENT_FILES, "tools/research_timing_phase_budget.py",
    "tests/test_research_timing_phase_budget.py", "tests/test_timing_phase_budget_review.py",
    "doc/ppo_timing_phase_budget_protocol_20261015.md", "tools/audit_timing_phase_budget.py",
    "tests/test_timing_phase_budget_audit.py", "tools/analyze_timing_phase_budget.py",
    "tests/test_analyze_timing_phase_budget.py", "tools/seal_timing_phase_budget.py", "tests/test_seal_timing_phase_budget.py")
ROOT_FILES = ("protocol.json", "selection.json", "summary.json", "study_summary.json", "status.json",
              "acceptance.json", "independent_audit_progress.json", "independent_audit_running.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def workspace_path(name):
    require(isinstance(name, str), "Signed path must be text")
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name and "\\" not in name and ":" not in name
            and ".." not in path.parts, "Signed path must be canonical workspace-relative POSIX")
    actual = (ROOT / name).resolve()
    require(actual.is_relative_to(ROOT.resolve()) and actual.is_file(), "Signed file is missing or escaped workspace")
    return actual


def add_file(files, path, expected=None):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()), "Cannot sign outside workspace")
    name = path.relative_to(ROOT.resolve()).as_posix()
    digest = sha(workspace_path(name))
    require(expected is None or digest == expected, "Registered file SHA changed: " + name)
    require(name not in files or files[name] == digest, "Conflicting signed file SHA: " + name)
    files[name] = digest


def verify_files(files):
    aliases = {}
    for name, digest in files.items():
        path = workspace_path(name)
        alias = str(path).casefold()
        require(alias not in aliases or aliases[alias] == digest, "Conflicting workspace alias SHA")
        aliases[alias] = digest
        require(sha(path) == digest, "Signed file changed: " + name)


def verify_acceptance(value, selected, progress_path):
    require(value.get("status") == "passed" and value.get("new_downloads") == 0
            and value.get("protected_period_evaluated") is False and value.get("default_model_promoted") is False,
            "Root actual acceptance is missing or changes protected-period/promotion flags")
    require(bool(value.get("tests")), "Root must record actual completed tests")
    for row in value["tests"]:
        process_evidence(row)
        require(type(row.get("passed")) is int and row["passed"] > 0
                and type(row.get("elapsed_seconds")) in (int, float) and 0 <= row["elapsed_seconds"] < float("inf"),
                "Root actual test count or duration is invalid")
    expected = {("prepare", None), ("screen", None), ("report", None),
                *(('screen', fold) for fold in audit.runner.original.SCREEN_FOLDS)}
    if selected["admitted"]:
        expected |= {("confirm", None), *(('confirm', fold) for fold in audit.runner.original.CONFIRM_FOLDS)}
    processes = value.get("formal_processes", [])
    require(len(processes) == len(expected) and {(row.get("stage"), row.get("fold")) for row in processes} == expected,
            "Root exits do not cover exactly the authorized physical workers and stages")
    for row in processes:
        process_evidence(row)
    independent = value.get("independent_audit", {})
    process_evidence(independent)
    require(independent.get("path") == "independent_audit_progress.json" and independent.get("sha256") == sha(progress_path),
            "Acceptance does not bind the actual final independent audit")
    analyses = value.get("analysis_processes", [])
    require(len(analyses) == 1 and analyses[0].get("tool") == "tools/analyze_timing_phase_budget.py",
            "Actual complete-stage analysis exit is missing")
    process_evidence(analyses[0])


def proof_statistics(trials):
    statistics = dict(snapshot_gae_buffer_views=0, snapshot_gae_step_views=0, unique_timeout_bootstraps=0,
        unique_natural_terminals=0, unique_active_rollout_tails=0, unique_done_rollout_tails=0,
        agent_native_steps=0, benchmark_native_steps=0, agent_rebalance_rows=0, benchmark_rebalance_rows=0,
        agent_decoder_samples=0, benchmark_decoder_samples=0, registered_leaf_files=0,
        maximum_advantage_error=0., maximum_return_error=0.)
    reproduced = []
    for name, row in trials.items():
        require(row["architecture"] in audit.runner.ARCHITECTURES and type(row["physical_budget"]) is int,
                "Audited physical architecture or budget is invalid")
        budget = row["physical_budget"]
        audit.runner.cell_label(row["architecture"], budget)
        coordinates = 9 if row["architecture"] == "due_phase_residual" else 8
        parameters = 12914 if coordinates == 9 else 12805
        initial = row["initial"]
        require(initial["parameters"] == parameters and initial["tensors"] == coordinates
                and initial["active_adam_tensors"] == initial["adam_updates"] == 0,
                "Audited initial parameter architecture or empty optimizer differs")
        expected = {"8192", "32768"} if row["phase"] == "screen" else {str(budget)}
        require(set(row["snapshots"]) == expected and audit.runner.same_json(row["physical_final"], row["snapshots"][str(budget)]["final"]),
                "Audited snapshots or physical final differ")
        for step, snapshot in row["snapshots"].items():
            final = snapshot["final"]
            require(final["parameters"] == parameters and final["tensors"] == final["active_adam_tensors"] == coordinates
                    and type(final["adam_updates"]) is int and final["adam_updates"] > 0 and final["changed_tensors"] > 0
                    and type(final["actor_changed"]) is bool and type(final["critic_changed"]) is bool,
                    "Actual audited snapshot updates or tensor architecture differ")
            estimator = snapshot["estimator"]
            require(estimator["steps"] == int(step) and estimator["original_buffers"] == int(step) // 512
                    and len(estimator["diagnostics"]) == estimator["original_buffers"]
                    and estimator["gamma"] == .995 and type(estimator["gae_lambda"]) is float and estimator["gae_lambda"] == 1.,
                    "Audited estimator snapshot scope differs")
            statistics["snapshot_gae_buffer_views"] += estimator["original_buffers"]
            statistics["snapshot_gae_step_views"] += estimator["steps"]
            for diagnostic in estimator["diagnostics"]:
                for key in ("maximum_advantage_error", "maximum_return_error"):
                    value = diagnostic[key]
                    require(type(value) in (int, float) and 0 <= value < float("inf"), "Audited GAE error statistic is invalid")
                    statistics[key] = max(statistics[key], value)
        complete = row["snapshots"][str(budget)]["estimator"]
        for target, source in (("unique_timeout_bootstraps", "timeout_bootstraps"), ("unique_natural_terminals", "natural_terminals"),
                               ("unique_active_rollout_tails", "active_rollout_tails"), ("unique_done_rollout_tails", "done_rollout_tails")):
            statistics[target] += complete[source]
        traces = [row["training"], *row["initialized"].values(), *(case for snapshot in row["snapshots"].values() for case in snapshot["trained"].values())]
        for trace in traces:
            statistics["agent_native_steps"] += trace["steps"]
            statistics["benchmark_native_steps"] += trace["steps"]
            for target, source in (("agent_rebalance_rows", "agent_rebalances"), ("benchmark_rebalance_rows", "benchmark_rebalances"),
                ("agent_decoder_samples", "agent_rebalance_decoder_samples"), ("benchmark_decoder_samples", "benchmark_rebalance_decoder_samples")):
                statistics[target] += trace[source]
        statistics["registered_leaf_files"] += row["registered_leaves"]
        if row["phase"] == "screen":
            prefix = row["shared_prefix"]
            require(budget == 32768 and prefix["shared_steps"] == 8192 and prefix["shared_original_buffers"] == prefix["shared_native_chunks"] == 16
                    and prefix["extra_rl_steps"] == 0 and prefix["counted_twice"] is False,
                    "Short snapshot was not proven to be the same physical prefix")
            if row["architecture"] == "linear":
                proof = row["reproduction"]
                flags = proof.get("policy_tensors_equal", {}) if proof else {}
                require(proof and set(flags) == {"initial_model", "short_model"} and all(value is True for value in flags.values())
                        and all(proof.get(key) is True for key in ("agent_native_arrays_equal", "benchmark_native_arrays_equal",
                            "original_eight_key_ppo_buffers_equal", "no_extra_environment_steps")) and proof.get("reused_as_sample") is False,
                        "A fresh linear 8K control lacks its Credit reproduction proof")
                reproduced.append(name)
    require(len(reproduced) == 2, "Both fresh linear 8K controls are required")
    return statistics, sorted(reproduced)


def diagnostic_files(study, selected, counters, files):
    for stage in ("screen", "confirm", "all"):
        folder = study / "diagnostics" / ("phase_budget_" + stage)
        if not folder.exists():
            require(stage != "all", "Actual complete-stage machine diagnostics are missing")
            continue
        require(stage != "confirm" or selected["admitted"], "Unauthorized confirmation diagnostic output")
        result = read(folder / "diagnostics.json")
        expected_steps = 131072 if stage == "screen" else counters["actual_rl_steps"] - 131072 if stage == "confirm" else counters["actual_rl_steps"]
        actual = result["counters"]
        require(result.get("status") == "complete_for_sealed_stage" and result.get("stage") == stage
                and actual["actual_agent_rl_steps"] == actual["separate_training_benchmark_steps"] == expected_steps
                and actual["physical_training_runs"] == (4 if stage == "screen" else 6 if stage == "confirm" else counters["physical_training_runs"])
                and actual["evaluation_snapshots"] == (8 if stage == "screen" else 6 if stage == "confirm" else counters["evaluated_snapshots"])
                and result.get("protocol_sha256") == sha(study / "protocol.json") and result.get("selection_sha256") == sha(study / "selection.json")
                and result.get("tool_sha256") == sha(ROOT / "tools/analyze_timing_phase_budget.py")
                and all(result.get(key) is True for key in ("no_model_load", "no_network_forwards", "no_environment_steps", "no_gradient_updates", "no_trial_leaf_writes")),
                "Machine diagnostics have a different complete physical scope or identity")
        for name in ("diagnostics.json", "phase_mode.csv", "rollout_scales.csv", "paired_effects.csv"):
            add_file(files, folder / name)


def build_seal(study):
    study = Path(study).resolve()
    require(study == (ROOT / STUDY_DIRECTORY).resolve(), "Seal only the registered PhaseBudget root")
    require(not (study / "final_audit.json").exists(), "Final audit already exists; no overwrite or reseal")
    protocol = read(study / "protocol.json")
    parent_path = Path(protocol["reference_root"]) / "final_audit.json"
    require(sha(parent_path) == PARENT_SHA256 and protocol["reference_final_audit_sha256"] == PARENT_SHA256,
            "Credit final audit does not match its hard pin")
    parent = read(parent_path)
    files = dict(parent["signed_files_sha256"])
    require(parent.get("status") == "passed" and len(files) == PARENT_SIGNED_FILES, "Incomplete predecessor signed inventory")
    verify_files(files)
    add_file(files, parent_path, PARENT_SHA256)
    sources = {key: value for key, value in files.items() if key.startswith("src/") and key.endswith(".py")}
    current = {path.relative_to(ROOT).as_posix() for path in (ROOT / "src").rglob("*.py")}
    require(len(sources) == 96 and set(sources) == current and sources == canonical_source_hashes(protocol["source_hashes"]),
            "Frozen 96-source inventory or actual bytes changed")
    progress_path = study / "independent_audit_progress.json"
    progress = read(progress_path)
    require(progress.get("status") == "passed" and progress.get("scope") == "complete_authorized_sealed_physical_study"
            and all(type(progress.get(key)) is int and progress[key] == 0 for key in ZERO_OPERATIONS),
            "Independent physical/native audit is incomplete or executed extra operations")
    require(audit.runner.same_json(progress["registration"], audit.registration(study, protocol)), "Audited frozen registration changed")
    trials = progress["sealed_trials"]
    selected, counters = audit.final_scope(study, protocol, trials)
    require(audit.runner.same_json(progress["selection"], selected) and progress["sealed_physical_count"] == len(trials)
            and audit.runner.same_json(progress["counters"], counters), "Final audit physical scope or selection differs")
    running = read(study / "independent_audit_running.json")
    require(running.get("status") == "passed" and running.get("progress_sha256") == sha(progress_path)
            and set(running.get("completed_physical_runs", [])) == set(trials)
            and all(type(running.get(key)) is int and running[key] == 0 for key in ("network_forwards", "environment_steps", "registered_leaf_writes")),
            "Actual final auditor receipt is incomplete")
    shared = progress["shared_initials"]
    require(shared.get("mapped_full_actor_and_critic_equal") is True and shared.get("checked_physical_models") == len(trials),
            "Mapped initial Actor/Critic equality and zero due-vector audit are missing")
    acceptance = read(study / "acceptance.json")
    verify_acceptance(acceptance, selected, progress_path)
    statistics, reproduced = proof_statistics(trials)
    markers = {}
    for name, row in trials.items():
        directory = study / "runs" / name
        marker = directory / "completed.json"
        require(sha(marker) == row["completion_sha256"], "An audited physical completion marker changed")
        completed = read(marker)
        audit.base.verify_leaf_files(directory, completed["files"])
        require(row["registered_leaves"] == len(completed["files"]), "Audited leaf count differs from the completed path")
        add_file(files, marker, row["completion_sha256"])
        markers[name] = row["completion_sha256"]
        for leaf, digest in completed["files"].items():
            relative = PurePosixPath(leaf)
            require(not relative.is_absolute() and relative.as_posix() == leaf and "\\" not in leaf and ".." not in relative.parts,
                    "Completed leaf path is not canonical")
            add_file(files, directory / leaf, digest)
    for name in ROOT_FILES:
        add_file(files, study / name)
    for phase, folds in (("screen", audit.runner.original.SCREEN_FOLDS),
        ("confirm", audit.runner.original.CONFIRM_FOLDS if selected["admitted"] else ())):
        for fold in folds:
            path = study / f"worker_{phase}_{fold}.json"
            require(read(path).get("status") == "complete", "Physical worker receipt is incomplete")
            add_file(files, path)
    for name in CODE_FILES:
        add_file(files, ROOT / name)
    diagnostic_files(study, selected, counters, files)
    summary = read(study / "study_summary.json")
    verify_files(files)
    return dict(schema_version=1, status="passed", study="timing_phase_budget_20261015", created_utc=datetime.now(timezone.utc).isoformat(),
        source_count=96, predecessor_signed_files_verified=PARENT_SIGNED_FILES, predecessor_final_audit_sha256=PARENT_SHA256,
        scope=progress["scope"], counters=counters, audited_proof_statistics=statistics, shared_initials=shared,
        initial_due_vector_zero="Verified by the independent shared_initials cold audit, not rerun by this sealer",
        fresh_linear_prefix_reproduced_trials=reproduced, decoder_scope="fixed first and last true REBALANCE per native chunk; all HOLD coordinates checked",
        audit_progress_sha256=sha(progress_path), root_process_evidence_sha256=sha(study / "acceptance.json"),
        root_process_evidence="Actual observed exits in signed acceptance.json; this tool did not rerun those processes",
        actual_independent_audit_exit_code=acceptance["independent_audit"]["exit_code"],
        independent_audit_operations={key: progress[key] for key in ZERO_OPERATIONS},
        seal_operations=dict(model_instances=0, network_forwards=0, environment_steps=0, gradient_calls=0, optimizer_steps=0, registered_leaf_writes=0),
        screen_admitted=selected["admitted"], confirmation_steps_authorized=summary["actual_confirmation_steps"],
        gate_passed=summary["qualified"], qualified_candidate=summary["qualified_candidate"],
        protected_period_evaluated=False, default_model_promoted=False, sealed_markers=markers, signed_files_sha256=dict(sorted(files.items())))


def seal(study):
    result = build_seal(study)
    path = Path(study) / "final_audit.json"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    verify_files(result["signed_files_sha256"])
    print(json.dumps(dict(status="passed", final_audit=str(path), sha256=sha(path), signed_files=len(result["signed_files_sha256"]), counters=result["counters"]), sort_keys=True))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / STUDY_DIRECTORY))
    return seal(Path(parser.parse_args(argv).root).resolve())


if __name__ == "__main__":
    main()
