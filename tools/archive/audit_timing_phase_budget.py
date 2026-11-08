"""Cold audit of physical timing paths and their shared-prefix snapshots."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.archive import audit_timing_study as base
from tools.archive import audit_timing_credit as credit_audit
from tools.archive import research_timing_phase_budget as runner
from tools.archive.timing_phase_agent import phase_policy_contract
from src.training.timing_env import build_timing_contract


def registration(study, protocol):
    runner.check_frozen(protocol)
    previous = runner.reference_protocol(protocol["reference_root"], full=True)
    inherited = credit_audit.registration(Path(protocol["reference_root"]), previous)
    return dict(source_files=96, original_sources_unchanged=True, predecessor_signed_files_verified=4676,
        predecessor_final_audit_sha256=runner.REFERENCE_AUDIT_SHA256,
        protocol_sha256=base.sha(study / "protocol.json"), document_sha256=protocol["document_sha256"],
        runner_sha256=protocol["tool_sha256"], auditor_sha256=base.sha(Path(__file__)),
        registered_git=protocol["git"], datasets=inherited["datasets"])


def reference_read(protocol, path):
    path = Path(path).resolve()
    inherited = base.read(Path(protocol["reference_root"]) / "final_audit.json")["signed_files_sha256"]
    base.require(path.is_relative_to(ROOT.resolve())
        and inherited.get(path.relative_to(ROOT.resolve()).as_posix()) == base.sha(path),
        "Read predecessor artifact no longer matches its final signature")
    return base.read(path)


def expected_contract(protocol, config):
    spec = config["timing"]
    initial = Path(protocol["baseline_reference_root"]) / "runs" / (
        f'{spec["phase"]}_{spec["fold"]}_log_net_{config["seed"]}') / "initial_model/metadata.json"
    contract = build_timing_contract(reference_read(protocol, initial)["contract"], "relative_log", 2.)
    contract["timing_research"] = deepcopy(spec)
    contract["phase_budget_study"] = deepcopy(config["phase_budget_study"])
    return phase_policy_contract(contract, config["phase_budget_study"]["architecture"])


def mapped_tensors(theta):
    return {key.replace("action_net.base.", "action_net."): value for key, value in theta.items()
            if key != "action_net.residual.weight"}


def shared_initials(initials):
    groups = {}
    for (fold, phase, seed, _architecture), theta in initials.items():
        mapped = mapped_tensors(theta)
        first = groups.setdefault((fold, phase, seed), mapped)
        base.require(set(first) == set(mapped) and all(torch.equal(first[key], mapped[key]) for key in first),
                     "Initial mapped Actor or Critic differs across architectures")
        if "action_net.residual.weight" in theta:
            base.require(torch.count_nonzero(theta["action_net.residual.weight"]) == 0,
                         "Initial due vector is not zero")
    return dict(groups=len(groups), checked_physical_models=len(initials), mapped_full_actor_and_critic_equal=True,
                equality_scope="Within each registered fold/phase/seed group")


def parameter_changes(initial, theta, audit):
    base.require(set(initial) == set(theta) == set(audit["tensor_delta_l2"]), "Parameter delta coordinates differ")
    changed = {}
    for key, value in theta.items():
        actual = float(torch.linalg.vector_norm(value - initial[key]))
        claimed = audit["tensor_delta_l2"][key]
        base.require(type(claimed) in (float, int) and np.isfinite(claimed), "Claimed parameter change is not finite")
        base.close(claimed, actual, "Actual initial-to-snapshot parameter change differs", atol=1e-7, rtol=1e-6)
        changed[key] = not torch.equal(value, initial[key])
    base.require(any(changed.values()), "Snapshot performed no actual parameter update")
    return dict(actor_changed=any(value for key, value in changed.items() if key.startswith("action_net.")),
                critic_changed=any(value for key, value in changed.items() if not key.startswith("action_net.")),
                changed_tensors=sum(changed.values()))


def model_audit(directory, config, contract, steps, initial=None, audit=None):
    theta = runner.verify_model(directory, config, contract, steps, audit)
    metadata = base.read(Path(directory) / "metadata.json")
    base.require(metadata["actual_device"] == "cpu", "Actual model device is not the registered CPU")
    with zipfile.ZipFile(Path(directory) / "model.zip") as zipped:
        digest = hashlib.sha256(zipped.read("policy.pth")).hexdigest()
    counts = audit["optimizer_steps_by_parameter"] if audit is not None else {}
    result = dict(parameters=sum(value.numel() for value in theta.values()), tensors=len(theta),
                  active_adam_tensors=len(counts), adam_updates=max(counts.values(), default=0), policy_pth_sha256=digest)
    if initial is not None:
        result.update(parameter_changes(initial, theta, audit))
    return theta, result


def estimator_audit(directory, audit, config):
    base.require(audit["actual_model_gamma"] == audit["actual_buffer_gamma"] == .995
        and type(audit["actual_model_gae_lambda"]) is float and type(audit["actual_buffer_gae_lambda"]) is float
        and type(config["ppo"]["gae_lambda"]) is float
        and audit["actual_model_gae_lambda"] == audit["actual_buffer_gae_lambda"] == config["ppo"]["gae_lambda"] == 1.,
        "Actual PhaseBudget model/buffer gamma or GAE lambda differs")
    records, rollouts = audit["credit_estimator_buffers"], audit["rollout_buffers"]
    chunks = audit["training_trace_prefix"]["chunks"]
    base.require(len(records) == len(rollouts) == len(chunks), "Estimator evidence omits a snapshot buffer")
    updates_path = (directory / audit["training_updates_path"]).resolve()
    base.require(updates_path.is_relative_to(directory.resolve())
        and base.sha(updates_path) == audit["training_updates_sha256"], "Snapshot update-prefix path/hash differs")
    updates = pd.read_csv(updates_path, float_precision="round_trip")
    base.require(len(updates) == len(records) and np.all(updates["credit/gamma"] == .995)
        and np.all(updates["credit/gae_lambda"] == 1.), "Snapshot actual update logs use another estimator")
    previous, diagnostics, active_tails, done_tails = True, [], 0, 0
    for index, (record, rollout, chunk) in enumerate(zip(records, rollouts, chunks)):
        path = (directory / record["path"]).resolve()
        base.require(path.is_relative_to(directory.resolve()) and base.sha(path) == record["sha256"]
            and all(record[key] == rollout[key] == chunk[key] for key in ("first_step", "last_step", "steps")),
            "Snapshot estimator prefix/hash differs")
        with zipfile.ZipFile(path) as zipped:
            base.require(zipped.testzip() is None, "Estimator sidecar CRC failed")
        with np.load(path, allow_pickle=False) as zipped:
            context = {key: zipped[key] for key in zipped.files}
        with np.load(directory / rollout["path"], allow_pickle=False) as zipped:
            buffer = {key: zipped[key] for key in zipped.files}
        with np.load(directory / "training_trace" / chunk["path"], allow_pickle=False) as zipped:
            native = {key: zipped[key] for key in ("done", "action_learning_reward")}
        diagnostics.append(credit_audit.estimator_arrays(buffer, context, native, 1., previous))
        previous = bool(context["last_dones"][0])
        done_tails += int(previous)
        active_tails += int(not previous)
        row = updates.iloc[index]
        base.require(row["timesteps"] == record["last_step"], "Snapshot logger step differs from actual buffer")
        for key, field in (("credit/returns_std", "returns"), ("credit/advantages_std", "advantages"),
                           ("credit/buffer_rewards_std", "rewards")):
            base.close(row[key], float(np.std(buffer[field])), "Snapshot actual estimator-scale logger differs", atol=1e-7)
    steps = sum(item["steps"] for item in diagnostics)
    base.require(steps == audit["actual_timesteps"], "Estimator audit omitted actual snapshot steps")
    return dict(gamma=.995, gae_lambda=1., original_buffers=len(records), steps=steps,
        timeout_bootstraps=sum(item["timeout_bootstraps"] for item in diagnostics),
        natural_terminals=sum(item["natural_terminals"] for item in diagnostics),
        active_rollout_tails=active_tails, done_rollout_tails=done_tails,
        scope="all original PPO targets including actual tails and TimeLimit values", diagnostics=diagnostics)


def prefix_audit(short, long, directory):
    base.require(short["actual_timesteps"] == 8192 and long["actual_timesteps"] == 32768,
                 "Shared prefix requires the two registered snapshot clocks")
    for name in ("rollout_buffers", "credit_estimator_buffers"):
        base.require(len(short[name]) == 16 and len(long[name]) == 64
            and runner.same_json(short[name], long[name][:16]), "Short PPO/estimator evidence is not the physical prefix")
    for first, second in ((short["training_trace_prefix"], long["training_trace_prefix"]),
                         (short["training_trace_prefix"]["benchmark_trace"], long["training_trace_prefix"]["benchmark_trace"])):
        base.require(runner.same_json(first["identity"], second["identity"])
            and len(first["chunks"]) == 16 and len(second["chunks"]) == 64
            and runner.same_json(first["chunks"], second["chunks"][:16]), "Short native account evidence is not the physical prefix")
    first = pd.read_csv(directory / short["training_updates_path"], float_precision="round_trip")
    second = pd.read_csv(directory / long["training_updates_path"], float_precision="round_trip")
    base.require(len(first) == 16 and len(second) == 64 and set(first.columns).issubset(second.columns),
                 "Short after-Adam logger lacks its physical prefix")
    for key in first:
        base.require(np.array_equal(first[key].to_numpy(), second[key].iloc[:16].to_numpy(), equal_nan=True),
                     "Short after-Adam update row changed during continuation")
    return dict(shared_steps=8192, shared_original_buffers=16, shared_native_chunks=16,
        extra_rl_steps=0, counted_twice=False, binding="exact original arrays/sidecar hashes and update rows")


def verify_selection(study, protocol):
    records = []
    for fold in runner.original.SCREEN_FOLDS:
        runner.verify_worker_receipt(study, protocol, "screen", fold)
        for architecture in runner.ARCHITECTURES:
            name = runner.run_id(fold, architecture, "screen", 42)
            completed = base.read(study / "runs" / name / "completed.json")
            base.require(completed["protocol_sha256"] == base.json_digest(protocol)
                and completed["tool_sha256"] == protocol["tool_sha256"], "Matrix selection borrows another registration")
            records.extend(completed["records"])
    selected = base.read(study / "selection.json")
    base.require(runner.same_json(selected, runner.selection_record(study, protocol, records)),
                 "Fixed complete eight-snapshot selection differs")
    return selected


def sealed_trial(study, protocol, directory):
    completed = base.read(directory / "completed.json")
    base.verify_leaf_files(directory, completed["files"])
    config = base.read(directory / "config.json")
    spec, identity = config["timing"], config["phase_budget_study"]
    fold, phase, architecture, seed, budget = spec["fold"], spec["phase"], identity["architecture"], config["seed"], identity["physical_budget"]
    base.require(runner.same_json(config, runner.config_for(protocol, study, fold, architecture, phase, seed, budget))
        and directory.name == runner.run_id(fold, architecture, phase, seed)
        and completed["protocol_sha256"] == base.json_digest(protocol)
        and completed["config_sha256"] == base.json_digest(config) and completed["tool_sha256"] == protocol["tool_sha256"]
        and type(completed["physical_steps"]) is int and completed["physical_steps"] == budget,
        "Physical run registration, config, class or budget differs")
    if phase == "confirm":
        selected = verify_selection(study, protocol)
        base.require(selected["admitted"] is True
            and (selected["selected_architecture"], selected["selected_budget"]) == (architecture, budget)
            and completed["selection_sha256"] == base.sha(study / "selection.json"), "Unauthorized confirmation path")
    else:
        base.require(budget == 32768 and completed["selection_sha256"] is None, "Screen is not a full registered physical path")
    contract = expected_contract(protocol, config)
    initial, cold = model_audit(directory / "initial_model", config, contract, 0)
    market = base.market_arrays(protocol, fold, contract["stock_codes"])
    records = completed["records"]
    expected_budgets = [8192, 32768] if phase == "screen" else [budget]
    base.require([row["steps"] for row in records] == identity["snapshot_steps"] == expected_budgets,
                 "Physical path omitted or duplicated a registered snapshot")
    initialized = {}
    for mode in base.MODES:
        folder = directory / "initialized" / mode
        result = base.read(folder / "result.json")
        baseline_path = Path(protocol["baseline_reference_root"]) / "rules" / fold / mode / "result.json"
        baseline = reference_read(protocol, baseline_path)["metrics"]
        base.require(runner.same_json(result["contract"], contract) and result["initial_mode"] == mode
            and result["metrics"] == result["benchmark_metrics"] == baseline
            and not any(runner.shared.case_changes(folder, result).values()), "Common initialization changed the unchanged baseline")
        base.verify_evaluation_dates(folder, result, market.evaluation_dates(protocol["folds"][fold]))
        base.case_metrics(folder, result)
        initialized[mode] = base.trace_audit(folder / "trace", result["trace"], market, contract, False)
    snapshots, raw_audits, finals = {}, {}, {}
    for row, steps in zip(records, expected_budgets):
        base.require((row["fold"], row["phase"], row["architecture"], row["seed"], row["physical_run_id"]) ==
            (fold, phase, architecture, seed, directory.name) and type(row["seed"]) is int
            and row["method"] == runner.cell_label(architecture, steps)
            and row["test_evaluated"] is False and row["actionable"] is False, "Snapshot row identity differs")
        folder = directory / "snapshots" / str(steps)
        audit = base.read(folder / "final_audit.json")
        base.require(runner.same_json(audit, row["audit"]) and type(audit["actual_timesteps"]) is int
            and audit["actual_timesteps"] == steps and audit["source_hashes"] == protocol["source_hashes"]
            and audit["tool_sha256"] == protocol["tool_sha256"] and audit["parameters"] == contract["phase_policy"]["parameter_count"]
            and audit["experience"]["total_steps"] == audit["experience"]["benchmark_steps"] == steps
            and audit["training_schedule"]["agent_environment_steps"] == audit["training_schedule"]["benchmark_extra_environment_steps"] == steps
            and all(audit.get(key) is True for key in ("save_load_parameters_equal", "save_load_macro_action_equal",
                "checkpoint_training_state_unchanged", "all_parameters_finite")), "Snapshot actual independent books or after-optimizer clock differs")
        base.require(runner.same_json(audit["constant_schedules"], dict(learning_rate=3e-4, clip_range=.2,
            checked_progress=[0., .25, .5, .75, 1.])), "Shared prefix changed learning-rate or clipping schedules")
        model_folder = directory / ("short_model" if steps == 8192 and budget != 8192 else "final_model")
        base.require(row["model_path"] == str(model_folder), "Snapshot model path differs")
        _, final = model_audit(model_folder, config, contract, steps, initial, audit)
        base.verify_rollouts(directory, audit["rollout_buffers"], steps, audit["training_trace_prefix"])
        estimator = estimator_audit(directory, audit, config)
        schedule = base.schedule_audit(audit, config, market, protocol["folds"][fold])
        trained = {}
        for mode in base.MODES:
            path = folder / "trained" / mode
            result = base.read(path / "result.json")
            init_result = base.read(directory / "initialized" / mode / "result.json")
            base.require(runner.same_json(result["contract"], contract) and result["initial_mode"] == mode
                and row["metrics"][mode]["trained"] == result["metrics"]
                and row["metrics"][mode]["initialized"] == init_result["metrics"]
                and row["metrics"][mode]["benchmark"] == result["benchmark_metrics"] == init_result["benchmark_metrics"]
                and row["evaluation_steps"]["initialized"][mode] == init_result["evaluation_steps"]
                and row["evaluation_steps"]["trained"][mode] == result["evaluation_steps"]
                and row["changes"][mode] == runner.shared.case_changes(path, result), "Snapshot native metrics, dates or macro/economic flags differ")
            base.verify_evaluation_dates(path, result, market.evaluation_dates(protocol["folds"][fold]))
            base.case_metrics(path, result)
            trained[mode] = base.trace_audit(path / "trace", result["trace"], market, contract, False)
        snapshots[str(steps)] = dict(final=final, estimator=estimator, schedule=schedule, trained=trained)
        raw_audits[steps], finals[steps] = audit, final
    prefix = prefix_audit(raw_audits[8192], raw_audits[32768], directory) if phase == "screen" else None
    training = base.trace_audit(directory / "training_trace", raw_audits[budget]["training_trace_prefix"], market, contract, True)
    reproduction = None
    if phase == "screen" and architecture == "linear":
        reproduction = runner.compare_linear_prefix(directory, protocol, raw_audits[8192])
        base.require(runner.same_json(base.read(directory / "reproduction.json"), reproduction), "Fresh linear prefix reproduction differs")
    return dict(completion_sha256=base.sha(directory / "completed.json"), registered_leaves=len(completed["files"]),
        fold=fold, phase=phase, architecture=architecture, seed=seed, physical_budget=budget,
        actual_rl_steps=budget, separate_benchmark_training_steps=budget, initial=cold, physical_final=finals[budget],
        training=training, initialized=initialized, snapshots=snapshots, shared_prefix=prefix, reproduction=reproduction), initial


def experience_counters(trials):
    values = list(trials.values())
    return dict(physical_training_runs=len(values), evaluated_snapshots=sum(len(row["snapshots"]) for row in values),
        actual_rl_steps=sum(row["actual_rl_steps"] for row in values),
        separate_training_benchmark_steps=sum(row["separate_benchmark_training_steps"] for row in values),
        actual_adam_updates=sum(row["physical_final"]["adam_updates"] for row in values),
        original_gae_buffers=sum(row["snapshots"][str(row["physical_budget"])]["estimator"]["original_buffers"] for row in values),
        unique_audited_gae_steps=sum(row["snapshots"][str(row["physical_budget"])]["estimator"]["steps"] for row in values),
        snapshot_step_sum=sum(int(steps) for row in values for steps in row["snapshots"]),
        shared_short_prefix_steps=sum(row["shared_prefix"]["shared_steps"] for row in values if row["shared_prefix"]),
        evaluation_initialized_paths=sum(len(row["initialized"]) for row in values),
        evaluation_trained_paths=sum(len(snapshot["trained"]) for row in values for snapshot in row["snapshots"].values()),
        evaluation_agent_steps=sum(case["steps"] for row in values for case in row["initialized"].values())
            + sum(case["steps"] for row in values for snapshot in row["snapshots"].values() for case in snapshot["trained"].values()))


def final_scope(study, protocol, trials):
    selected = verify_selection(study, protocol)
    cell = selected["selected_architecture"], selected["selected_budget"]
    expected = {runner.run_id(fold, architecture, "screen", 42)
                for fold in runner.original.SCREEN_FOLDS for architecture in runner.ARCHITECTURES}
    if selected["admitted"]:
        expected |= {runner.run_id(fold, cell[0], "confirm", seed)
                     for fold in runner.original.CONFIRM_FOLDS for seed in runner.original.SEEDS}
        for fold in runner.original.CONFIRM_FOLDS:
            runner.verify_worker_receipt(study, protocol, "confirm", fold, cell)
    base.require(set(trials) == expected and {path.name for path in (study / "runs").iterdir() if path.is_dir()} == expected,
                 "Missing or unauthorized physical training path")
    records = [record for name in sorted(expected) for record in base.read(study / "runs" / name / "completed.json")["records"]]
    supplied = base.read(study / "summary.json")
    key = lambda row: (row["fold"], row["phase"], row["architecture"], row["steps"], row["seed"])
    base.require(len(supplied) == len(records) and len({key(row) for row in supplied}) == len(records)
        and runner.same_json({str(key(row)): row for row in supplied}, {str(key(row)): row for row in records}),
        "Final snapshot summary changed or borrowed completed rows")
    counters = experience_counters(trials)
    expected_steps = 131072 + (6 * cell[1] if selected["admitted"] else 0)
    expected_runs = 10 if selected["admitted"] else 4
    expected_snapshots = 14 if selected["admitted"] else 8
    base.require(counters["actual_rl_steps"] == counters["separate_training_benchmark_steps"]
        == counters["unique_audited_gae_steps"] == expected_steps
        and counters["physical_training_runs"] == expected_runs and counters["evaluated_snapshots"] == expected_snapshots
        and counters["original_gae_buffers"] == expected_steps // 512
        and counters["shared_short_prefix_steps"] == 32768, "Shared snapshots were counted as new RL experience")
    summary = base.read(study / "study_summary.json")
    fields = {"actual_training_steps": counters["actual_rl_steps"],
        "independent_training_benchmark_steps": counters["separate_training_benchmark_steps"],
        "actual_optimizer_updates": counters["actual_adam_updates"], "physical_training_runs": expected_runs,
        "final_snapshots": expected_snapshots, "actual_screen_steps": 131072,
        "actual_confirmation_steps": 6 * cell[1] if selected["admitted"] else 0,
        "shared_short_prefix_steps": 32768, "snapshot_steps_sum": counters["snapshot_step_sum"],
        "evaluation_initialized_paths": counters["evaluation_initialized_paths"],
        "evaluation_trained_paths": counters["evaluation_trained_paths"],
        "evaluation_agent_steps": counters["evaluation_agent_steps"],
        "evaluation_parallel_benchmark_steps": counters["evaluation_agent_steps"],
        "new_rule_control_steps": 0, "additional_reproduction_environment_steps": 0,
        "screen": selected, "test_evaluated": False, "actionable": False,
        "status": "complete" if selected["admitted"] else "screen_failed"}
    base.require(all(runner.same_json(summary.get(name), value) for name, value in fields.items()),
                 "Final physical/experience/evaluation report differs from actual evidence")
    confirmed = [row for row in records if row["phase"] == "confirm"]
    financial = runner.confirmation_summary(confirmed, cell) if selected["admitted"] else dict(qualified=False, qualified_candidate=None)
    base.require(all(runner.same_json(summary.get(name), value) for name, value in financial.items()),
                 "Final qualification differs from all native confirmation seeds")
    base.require(runner.same_json(base.read(study / "status.json"), {name: summary[name] for name in (
        "status", "actual_training_steps", "final_snapshots", "physical_training_runs", "qualified", "test_evaluated", "actionable")}),
        "Final status is not the completed report")
    return selected, counters


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / "reports/runs/ppo_timing_phase_budget_20261015"))
    parser.add_argument("--sealed", action="store_true")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args(argv)
    base.require(not args.final or args.sealed, "Final audit requires sealed native physical runs")
    study = Path(args.root).resolve()
    base.require(not (study / "final_audit.json").exists(), "Final seal already exists; never rewrite signed progress")
    torch.set_num_threads(1)
    protocol = base.read(study / "protocol.json")
    result = dict(status="passed", scope="frozen_registration", registration=registration(study, protocol),
        model_instances=0, network_forwards=0, environment_steps=0, gradient_calls=0,
        optimizer_steps=0, registered_leaf_writes=0, frozen_source_writes=0)
    if args.sealed:
        trials, initials = {}, {}
        running = study / "independent_audit_running.json"
        for marker in sorted((study / "runs").glob("*/completed.json")):
            row, theta = sealed_trial(study, protocol, marker.parent)
            trials[marker.parent.name] = row
            initials[row["fold"], row["phase"], row["seed"], row["architecture"]] = theta
            print("AUDITED " + marker.parent.name, flush=True)
            running.write_text(json.dumps(dict(status="running", completed_physical_runs=list(trials),
                counters=experience_counters(trials), network_forwards=0, environment_steps=0,
                registered_leaf_writes=0), indent=2) + "\n", encoding="utf-8")
        result.update(status="partial", scope="registration_and_listed_sealed_physical_runs", sealed_trials=trials,
            sealed_physical_count=len(trials), counters=experience_counters(trials), shared_initials=shared_initials(initials))
        if args.final:
            selection, counters = final_scope(study, protocol, trials)
            result.update(status="passed", scope="complete_authorized_sealed_physical_study", selection=selection, counters=counters)
    output = study / "independent_audit_progress.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.sealed:
        running.write_text(json.dumps(dict(status=result["status"], completed_physical_runs=list(trials),
            progress_sha256=base.sha(output), network_forwards=0, environment_steps=0,
            registered_leaf_writes=0), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("registration", "sealed_trials")}, sort_keys=True))


if __name__ == "__main__":
    main()
