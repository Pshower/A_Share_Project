"""Pure offline audit of sealed margin trials; no model or environment execution."""

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_timing_study as base
from tools import research_timing_margin as runner
from src.training.timing_env import build_timing_contract


SHAPES = {
    "mlp_extractor.value_net.0.weight": (64, 129), "mlp_extractor.value_net.0.bias": (64,),
    "mlp_extractor.value_net.2.weight": (64, 64), "mlp_extractor.value_net.2.bias": (64,),
    "action_net.weight": (2, 129), "action_net.bias": (2,),
    "value_net.weight": (1, 64), "value_net.bias": (1,),
}


def initial_model_audit(directory, metadata, margin):
    base.require(type(margin) in (int, float) and margin in runner.MARGINS,
                 "Unregistered calendar margin")
    with zipfile.ZipFile(Path(directory) / "model.zip") as archive:
        base.require(archive.testzip() is None, "Initial model ZIP CRC failed")
        saved = json.loads(archive.read("data"))
        policy_bytes = archive.read("policy.pth")
        theta = torch.load(io.BytesIO(policy_bytes), weights_only=True, map_location="cpu")
        optimizer = torch.load(io.BytesIO(archive.read("policy.optimizer.pth")),
                               weights_only=True, map_location="cpu")
    base.require(set(theta) == set(SHAPES) and all(
        tuple(theta[name].shape) == shape and theta[name].dtype == torch.float32
        and torch.isfinite(theta[name]).all() for name, shape in SHAPES.items()),
        "Initial policy parameter coordinates/dtype/finiteness differ")
    base.require(sum(t.numel() for t in theta.values()) == 12805
                 and not optimizer["state"]
                 and len([i for group in optimizer["param_groups"] for i in group["params"]]) == 8
                 and len({i for group in optimizer["param_groups"] for i in group["params"]}) == 8,
                 "Initial policy already has Adam updates or wrong parameter count")
    base.require(metadata["num_timesteps"] == saved["num_timesteps"] == 0
                 and metadata["actual_device"] == "cpu"
                 and saved["timing_registered_contract"] == metadata["contract"],
                 "Initial archive timing registration/device differs")
    expected = torch.zeros_like(theta["action_net.weight"])
    expected[0, 1:20], expected[1, 0] = float(margin), float(margin)
    base.require(torch.equal(theta["action_net.weight"], expected)
                 and torch.count_nonzero(theta["action_net.bias"]) == 0,
                 "Actor is not the exact registered trainable margin calendar")
    return theta, dict(parameters=12805, tensors=8, active_adam_tensors=0, adam_updates=0,
                       calendar_margin=float(margin),
                       policy_pth_sha256=hashlib.sha256(policy_bytes).hexdigest())


def check_shared_initials(initials):
    groups = {}
    for key, tensors in initials.items():
        fold, phase, seed, _margin = key
        group = groups.setdefault((fold, phase, seed), tensors)
        base.require(set(group) == set(tensors) and all(torch.equal(group[name], tensors[name])
                     for name in group if name != "action_net.weight"),
                     "Margin cells changed Critic or other noncalendar initial tensors")
    return dict(groups=len(groups), checked_cells=len(initials),
                critic_and_noncalendar_tensors_equal=True)


def registration(study, protocol):
    runner.check_frozen(protocol)
    previous = base.read(Path(protocol["reference_root"]) / "protocol.json")
    original = base.registration(Path(protocol["reference_root"]), previous)
    return dict(source_files=96, original_source_files=91, original_sources_unchanged=True,
                protocol_sha256=base.sha(study / "protocol.json"),
                document_sha256=protocol["document_sha256"], runner_sha256=protocol["tool_sha256"],
                reference_final_audit_sha256=runner.REFERENCE_AUDIT_SHA256,
                registered_git=protocol["git"], datasets=original["datasets"])


def reference_read(protocol, path):
    path = Path(path).resolve()
    receipt = base.read(Path(protocol["reference_root"]) / "final_audit.json")
    key = path.relative_to(ROOT).as_posix()
    base.require(base.sha(path) == receipt["signed_files_sha256"].get(key),
                 "Read predecessor input no longer matches its final signature")
    return base.read(path)


def expected_contract(protocol, config):
    spec = config["timing"]
    previous = Path(protocol["reference_root"]) / "runs" / (
        f'{spec["phase"]}_{spec["fold"]}_log_net_{config["seed"]}') / "initial_model" / "metadata.json"
    contract = build_timing_contract(reference_read(protocol, previous)["contract"], "log_net",
                                     spec["calendar_margin"])
    contract["timing_research"] = deepcopy(spec)
    contract["margin_study"] = deepcopy(config["margin_study"])
    return contract


def verify_selection(study, protocol):
    selected = base.read(study / "selection.json")
    records, markers = [], {}
    for fold in runner.original.SCREEN_FOLDS:
        runner.verify_worker_receipt(study, protocol, "screen", fold)
        for margin in runner.MARGINS:
            name = runner.run_id(fold, margin, "screen", 42)
            path = study / "runs" / name / "completed.json"
            completed = base.read(path)
            base.require(completed["protocol_sha256"] == base.json_digest(protocol)
                         and completed["tool_sha256"] == protocol["tool_sha256"],
                         "Screen selection borrows another registration")
            records.append(completed["record"])
            markers[name] = base.sha(path)
    recomputed = dict(protocol_sha256=base.json_digest(protocol), **runner.select_screen(records),
                      completed_sha256=markers)
    base.require(selected == recomputed, "Six-trial fixed admission/selection differs")
    return selected


def sealed_trial(study, protocol, directory):
    marker = directory / "completed.json"
    base.require(marker.exists(), "Unsealed trial may not be audited")
    completed = base.read(marker)
    base.verify_leaf_files(directory, completed["files"])
    config, row = base.read(directory / "config.json"), completed["record"]
    spec = config["timing"]
    margin, phase, fold, seed = spec["calendar_margin"], spec["phase"], spec["fold"], config["seed"]
    base.require(config == runner.config_for(protocol, study, fold, margin, phase, seed)
                 and directory.name == runner.run_id(fold, margin, phase, seed)
                 and completed["protocol_sha256"] == base.json_digest(protocol)
                 and completed["config_sha256"] == base.json_digest(config)
                 and completed["tool_sha256"] == protocol["tool_sha256"],
                 "Trial registration/config/margin/tool identity differs")
    base.require((row["fold"], row["margin"], row["phase"], row["seed"], row["steps"], row["method"]) ==
                 (fold, margin, phase, seed, protocol["budgets"][phase], "log_net")
                 and row["test_evaluated"] is False and row["actionable"] is False,
                 "Trial final-budget identity/protection differs")
    if phase == "confirm":
        selected = verify_selection(study, protocol)
        base.require(selected["admitted"] is True and selected["selected_margin"] == margin
                     and completed["selection_sha256"] == base.sha(study / "selection.json"),
                     "Confirmation is not authorized by the frozen screen result")
    else:
        base.require(completed["selection_sha256"] is None, "Screen borrowed confirmation identity")
    contract = expected_contract(protocol, config)
    for name, steps in (("initial_model", 0), ("final_model", row["steps"])):
        runner.verify_model(directory / name, base.read(directory / name / "metadata.json"),
                            config, protocol, contract, steps)
    initial, initial_result = initial_model_audit(directory / "initial_model",
        base.read(directory / "initial_model/metadata.json"), margin)
    audit = base.read(directory / "final_audit.json")
    base.require(audit == row["audit"] and audit["actual_timesteps"] == row["steps"]
                 and audit["source_hashes"] == protocol["source_hashes"]
                 and audit["tool_sha256"] == protocol["tool_sha256"], "Actual training audit identity differs")
    base.require(audit["experience"]["total_steps"] == audit["experience"]["benchmark_steps"] == row["steps"]
                 and audit["training_schedule"]["agent_environment_steps"] ==
                 audit["training_schedule"]["benchmark_extra_environment_steps"] == row["steps"],
                 "Separate agent/benchmark training counts differ")
    base.verify_rollouts(directory, audit["rollout_buffers"], row["steps"], audit["training_trace_prefix"])
    _, final_result = base.model_audit(directory / "final_model",
        base.read(directory / "final_model/metadata.json"), initial, audit)
    market = base.market_arrays(protocol, fold, contract["stock_codes"])
    schedule = base.schedule_audit(audit, config, market, protocol["folds"][fold])
    training = base.trace_audit(directory / "training_trace", audit["training_trace_prefix"], market, contract, True)
    cases = {}
    dates = market.evaluation_dates(protocol["folds"][fold])
    for mode in base.MODES:
        frozen_baseline = reference_read(protocol, Path(protocol["reference_root"]) / "rules" / fold / mode / "result.json")["metrics"]
        for role in ("initialized", "trained"):
            folder = directory / role / mode
            result = base.read(folder / "result.json")
            base.require(result["contract"] == contract and result["initial_mode"] == mode
                         and result["metrics"] == row["metrics"][mode][role]
                         and result["benchmark_metrics"] == row["metrics"][mode]["benchmark"] == frozen_baseline
                         and result["evaluation_steps"] == row["evaluation_steps"][role][mode],
                         "Native evaluation metrics/identity/benchmark differs")
            changes = runner.case_changes(folder, result)
            base.require(changes == row["changes"][mode] if role == "trained" else not any(changes.values()),
                         "Original macro/economic behavior differs from registered flags")
            base.verify_evaluation_dates(folder, result, dates)
            base.case_metrics(folder, result)
            cases[role + "/" + mode] = base.trace_audit(folder / "trace", result["trace"], market, contract, False)
    reproduction = None
    if margin == 2.:
        previous = Path(protocol["reference_root"]) / "runs" / f'{phase}_{fold}_log_net_{seed}'
        reproduction = runner.compare_reproduction(directory, previous, audit)
        base.require(base.read(directory / "reproduction.json") == reproduction,
                     "Margin2 native/tensor reproduction evidence differs")
    evidence = dict(completion_sha256=base.sha(marker), registered_leaves=len(completed["files"]),
        actual_rl_steps=row["steps"], separate_benchmark_training_steps=row["steps"],
        margin=margin, phase=phase, fold=fold, seed=seed, initial=initial_result,
        final=final_result, training=training, schedule=schedule, cases=cases, reproduction=reproduction)
    return evidence, initial


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / "reports/runs/ppo_timing_margin_20261014"))
    parser.add_argument("--sealed", action="store_true")
    parser.add_argument("--final", action="store_true", help="Require the entire authorized study, not a partial audit")
    args = parser.parse_args(argv)
    base.require(not args.final or args.sealed, "Final audit requires sealed native trials")
    study = Path(args.root).resolve()
    protocol = base.read(study / "protocol.json")
    result = dict(status="passed", scope="frozen_registration", registration=registration(study, protocol),
        model_instances=0, network_forwards=0, environment_steps=0, gradient_calls=0, optimizer_steps=0,
        registered_leaf_writes=0, frozen_source_writes=0)
    if args.sealed:
        trials, initials = {}, {}
        running = study / "independent_audit_running.json"
        for marker in sorted((study / "runs").glob("*/completed.json")):
            evidence, tensors = sealed_trial(study, protocol, marker.parent)
            trials[marker.parent.name] = evidence
            initials[(evidence["fold"], evidence["phase"], evidence["seed"], evidence["margin"])] = tensors
            print("AUDITED " + marker.parent.name, flush=True)
            running.write_text(json.dumps(dict(status="running", completed_trials=list(trials),
                audited_rl_steps=sum(row["actual_rl_steps"] for row in trials.values()),
                network_forwards=0, environment_steps=0, registered_leaf_writes=0), indent=2) + "\n", encoding="utf-8")
        result.update(status="partial", scope="registration_and_listed_sealed_native_trials",
            sealed_trials=trials, sealed_trial_count=len(trials),
            actual_rl_steps=sum(row["actual_rl_steps"] for row in trials.values()),
            shared_initials=check_shared_initials(initials))
        if args.final:
            selected = verify_selection(study, protocol)
            count = 12 if selected["admitted"] else 6
            expected_steps = 147456 if selected["admitted"] else 49152
            expected_names = {runner.run_id(fold, margin, "screen", 42)
                              for fold in runner.original.SCREEN_FOLDS for margin in runner.MARGINS}
            if selected["admitted"]:
                expected_names |= {runner.run_id(fold, selected["selected_margin"], "confirm", seed)
                                   for fold in runner.original.CONFIRM_FOLDS for seed in runner.original.SEEDS}
                for fold in runner.original.CONFIRM_FOLDS:
                    runner.verify_worker_receipt(study, protocol, "confirm", fold, selected["selected_margin"])
            base.require(len(trials) == count and set(trials) == expected_names
                         and {path.name for path in (study / "runs").iterdir() if path.is_dir()} == expected_names
                         and result["actual_rl_steps"] == expected_steps,
                         "Full audit is missing trials or includes unauthorized experience")
            summary = base.read(study / "study_summary.json")
            evaluation_steps = sum(case["steps"] for trial in trials.values() for case in trial["cases"].values())
            base.require(summary["actual_training_steps"] == expected_steps and summary["final_snapshots"] == count
                         and summary["actual_screen_steps"] == 49152 and summary["screen"] == selected
                         and summary["actual_confirmation_steps"] == (98304 if selected["admitted"] else 0)
                         and summary["actual_optimizer_updates"] == sum(t["final"]["adam_updates"] for t in trials.values())
                         and summary["independent_training_benchmark_steps"] == expected_steps
                         and summary["evaluation_agent_steps"] == summary["evaluation_parallel_benchmark_steps"] == evaluation_steps
                         and summary["evaluation_initialized_paths"] == summary["evaluation_trained_paths"] == 4 * count
                         and summary["new_rule_control_steps"] == summary["additional_reproduction_environment_steps"] == 0
                         and summary["test_evaluated"] is False and summary["actionable"] is False,
                         "Final machine summary budget/Adam/protected-period identity differs")
            native_rows = {name: base.read(study / "runs" / name / "completed.json")["record"] for name in expected_names}
            supplied = base.read(study / "summary.json")
            supplied_rows = {runner.run_id(row["fold"], row["margin"], row["phase"], row["seed"]): row for row in supplied}
            base.require(len(supplied) == count and supplied_rows == native_rows, "Summary borrows or changes native rows")
            confirmed = [native_rows[runner.run_id(fold, selected["selected_margin"], "confirm", seed)]
                         for fold in runner.original.CONFIRM_FOLDS for seed in runner.original.SEEDS] if selected["admitted"] else []
            economics = runner.original.summarize_confirmation(confirmed, "log_net") if confirmed else dict(qualified=False, qualified_candidate=None)
            base.require(all(summary.get(key) == value for key, value in economics.items()),
                         "Final qualification/financial summary differs from native confirmation records")
            result.update(status="passed", scope="complete_authorized_sealed_native_study", selection=selected)
    output = study / "independent_audit_progress.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.sealed:
        running.write_text(json.dumps(dict(status=result["status"], completed_trials=list(trials),
            progress_sha256=base.sha(output), network_forwards=0, environment_steps=0,
            registered_leaf_writes=0), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("registration", "sealed_trials")}, sort_keys=True))


if __name__ == "__main__":
    main()
