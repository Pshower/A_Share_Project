"""Offline credit-estimator audit; no neural execution or financial environment."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_timing_study as base
from tools import audit_timing_margin as margin
from tools import research_timing_credit as runner
from src.training.timing_env import build_timing_contract


def gae_targets(rewards, values, episode_starts, last_values, last_dones, gamma, gae_lambda):
    """Match the float32 SB3 backward recurrence using captured actual boundary values."""
    advantages = np.zeros_like(values)
    carry = 0.
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            alive, next_values = 1. - last_dones.astype(np.float32), last_values
        else:
            alive, next_values = 1. - episode_starts[step + 1], values[step + 1]
        delta = rewards[step] + gamma * next_values * alive - values[step]
        carry = delta + gamma * gae_lambda * alive * carry
        advantages[step] = carry
    return advantages, advantages + values


def estimator_arrays(buffer, context, native, gae_lambda, previous_done=True):
    size = len(buffer["rewards"])
    base.require(set(context) == {"gamma", "gae_lambda", "last_values", "last_dones", "dones", "truncated",
                                  "terminal_values", "raw_rewards", "native_learning_rewards"},
                 "Estimator sidecar schema differs")
    for key in ("rewards", "values", "advantages", "returns", "episode_starts"):
        base.require(buffer[key].shape == (size, 1) and buffer[key].dtype == np.float32
                     and np.isfinite(buffer[key]).all(), "Original credit buffer layout/precision differs")
    for key in ("gamma", "gae_lambda"):
        base.require(np.asarray(context[key]).shape == () and np.asarray(context[key]).dtype == np.float64,
                     "Estimator parameter must be a float64 scalar")
    base.require(type(gae_lambda) in (float, int) and gae_lambda in runner.LAMBDAS
                 and float(context["gamma"]) == .995 and float(context["gae_lambda"]) == gae_lambda,
                 "Actual buffer gamma/GAE lambda differs")
    for key in ("dones", "truncated", "terminal_values", "raw_rewards", "native_learning_rewards"):
        base.require(np.asarray(context[key]).shape == (size, 1) and np.isfinite(context[key]).all(),
                     "Estimator context shape/finiteness differs")
    base.require(context["dones"].dtype == context["truncated"].dtype == np.bool_
                 and context["terminal_values"].dtype == context["raw_rewards"].dtype == np.float32
                 and context["native_learning_rewards"].dtype == np.float64,
                 "Estimator context precision or termination mask differs")
    base.require(context["last_values"].shape == context["last_dones"].shape == (1,)
                 and context["last_values"].dtype == np.float32 and context["last_dones"].dtype == np.bool_
                 and np.isfinite(context["last_values"]).all()
                 and np.array_equal(context["last_dones"], context["dones"][-1]),
                 "Captured actual rollout boundary differs")
    dones, truncated = context["dones"], context["truncated"]
    base.require(np.all(~truncated | dones) and np.all(context["terminal_values"][~truncated] == 0.),
                 "TimeLimit bootstrap applied to a non-timeout transition")
    base.require(np.array_equal(dones.reshape(-1), np.asarray(native["done"]).reshape(-1))
                 and np.array_equal(context["native_learning_rewards"].reshape(-1), native["action_learning_reward"])
                 and np.array_equal(context["raw_rewards"], context["native_learning_rewards"].astype(np.float32)),
                 "Original realized reward/done is not the actual context input")
    starts = buffer["episode_starts"]
    base.require(np.isin(starts, [0., 1.]).all() and bool(starts[0, 0]) == bool(previous_done)
                 and np.array_equal(starts[1:].astype(bool), dones[:-1]),
                 "Episode boundary cuts differ from actual done transitions")
    expected_rewards = context["raw_rewards"].copy()
    expected_rewards[truncated] += np.float32(.995) * context["terminal_values"][truncated]
    base.close(buffer["rewards"], expected_rewards, "Raw reward/TimeLimit terminal bootstrap differs", atol=1e-7, rtol=1e-6)
    advantages, returns = gae_targets(buffer["rewards"], buffer["values"], starts,
        context["last_values"], context["last_dones"], .995, gae_lambda)
    base.close(buffer["advantages"], advantages, "Full GAE recurrence including captured tail differs", atol=1e-6, rtol=1e-6)
    base.close(buffer["returns"], returns, "Credit return targets differ from actual GAE and old values", atol=1e-6, rtol=1e-6)
    return dict(steps=size, timeout_bootstraps=int(truncated.sum()), natural_terminals=int((dones & ~truncated).sum()),
        maximum_advantage_error=float(np.max(np.abs(buffer["advantages"] - advantages))),
        maximum_return_error=float(np.max(np.abs(buffer["returns"] - returns))),
        boundary_values="actual existing SB3 outputs; no additional forward")


def estimator_audit(directory, audit, config):
    value = config["ppo"]["gae_lambda"]
    base.require(audit["actual_model_gamma"] == audit["actual_buffer_gamma"] == .995
                 and audit["actual_model_gae_lambda"] == audit["actual_buffer_gae_lambda"] == value,
                 "Actual model/buffer estimator parameters differ")
    records, rollouts, chunks = audit["credit_estimator_buffers"], audit["rollout_buffers"], audit["training_trace_prefix"]["chunks"]
    base.require(len(records) == len(rollouts) == len(chunks), "Estimator evidence omits a PPO buffer")
    previous, diagnostics = True, []
    updates = pd.read_csv(directory / "training_updates.csv", float_precision="round_trip")
    base.require(len(updates) == len(records) and np.all(updates["credit/gamma"] == .995)
                 and np.all(updates["credit/gae_lambda"] == value), "Actual update logs use another estimator")
    for index, (record, rollout, chunk) in enumerate(zip(records, rollouts, chunks)):
        path = (directory / record["path"]).resolve()
        base.require(path.is_relative_to(directory.resolve()) and base.sha(path) == record["sha256"]
                     and all(record[key] == rollout[key] == chunk[key] for key in ("first_step", "last_step", "steps")),
                     "Estimator sidecar prefix/hash differs")
        with np.load(path, allow_pickle=False) as zipped:
            context = {key: zipped[key] for key in zipped.files}
        with np.load(directory / rollout["path"], allow_pickle=False) as zipped:
            buffer = {key: zipped[key] for key in zipped.files}
        with np.load(directory / "training_trace" / chunk["path"], allow_pickle=False) as zipped:
            native = {key: zipped[key] for key in ("done", "action_learning_reward")}
        diagnostics.append(estimator_arrays(buffer, context, native, value, previous))
        previous = bool(context["last_dones"][0])
        row = updates.iloc[index]
        base.require(row["timesteps"] == record["last_step"], "Credit logger step differs from actual buffer")
        for key, field in (("credit/returns_std", "returns"), ("credit/advantages_std", "advantages"),
                           ("credit/buffer_rewards_std", "rewards")):
            base.close(row[key], float(np.std(buffer[field])), "Actual estimator-scale logger differs", atol=1e-7)
    return dict(gamma=.995, gae_lambda=value, original_buffers=len(records),
        steps=sum(item["steps"] for item in diagnostics), timeout_bootstraps=sum(item["timeout_bootstraps"] for item in diagnostics),
        natural_terminals=sum(item["natural_terminals"] for item in diagnostics),
        scope="all original PPO targets including actual tails and TimeLimit values", diagnostics=diagnostics)


def registration(study, protocol):
    runner.check_frozen(protocol)
    previous = base.read(Path(protocol["reference_root"]) / "protocol.json")
    original = margin.registration(Path(protocol["reference_root"]), previous)
    return dict(source_files=96, original_sources_unchanged=True, protocol_sha256=base.sha(study / "protocol.json"),
        document_sha256=protocol["document_sha256"], runner_sha256=protocol["tool_sha256"],
        reference_final_audit_sha256=runner.REFERENCE_AUDIT_SHA256, registered_git=protocol["git"], datasets=original["datasets"])


def expected_contract(protocol, config):
    spec = config["timing"]
    previous = Path(protocol["reproduction_reference_root"]) / "runs" / (
        f'{spec["phase"]}_{spec["fold"]}_log_net_{config["seed"]}') / "initial_model" / "metadata.json"
    contract = build_timing_contract(margin.reference_read(protocol, previous)["contract"], spec["reward_basis"], 2.)
    contract["timing_research"] = deepcopy(spec)
    contract["credit_study"] = deepcopy(config["credit_study"])
    return contract


def shared_initials(initials):
    groups = {}
    for (fold, phase, seed, _cell), theta in initials.items():
        first = groups.setdefault((fold, phase, seed), theta)
        base.require(set(first) == set(theta) and all(torch.equal(first[key], theta[key]) for key in first),
                     "Reward/GAE cells changed initial Actor or Critic tensors")
    return dict(groups=len(groups), checked_cells=len(initials), full_initial_actor_and_critic_equal=True)


def verify_selection(study, protocol):
    records, markers = [], {}
    for fold in runner.original.SCREEN_FOLDS:
        runner.verify_worker_receipt(study, protocol, "screen", fold)
        for reward, value in runner.CELLS:
            name = runner.run_id(fold, reward, value, "screen", 42)
            path = study / "runs" / name / "completed.json"
            completed = base.read(path)
            base.require(completed["protocol_sha256"] == base.json_digest(protocol)
                         and completed["tool_sha256"] == protocol["tool_sha256"], "Screen registration differs")
            records.append(completed["record"])
            markers[name] = base.sha(path)
    selected = base.read(study / "selection.json")
    base.require(selected == dict(protocol_sha256=base.json_digest(protocol), **runner.select_screen(records),
                                  completed_sha256=markers), "Eight-trial fixed selection differs")
    return selected


def sealed_trial(study, protocol, directory):
    completed = base.read(directory / "completed.json")
    base.verify_leaf_files(directory, completed["files"])
    config, row = base.read(directory / "config.json"), completed["record"]
    spec, value, seed = config["timing"], config["ppo"]["gae_lambda"], config["seed"]
    fold, reward, phase = spec["fold"], spec["reward_basis"], spec["phase"]
    base.require(config == runner.config_for(protocol, study, fold, reward, value, phase, seed)
                 and directory.name == runner.run_id(fold, reward, value, phase, seed)
                 and completed["protocol_sha256"] == base.json_digest(protocol)
                 and completed["config_sha256"] == base.json_digest(config)
                 and completed["tool_sha256"] == protocol["tool_sha256"], "Credit trial identity differs")
    base.require((row["fold"], row["reward_basis"], row["gae_lambda"], row["phase"], row["seed"], row["steps"]) ==
                 (fold, reward, value, phase, seed, protocol["budgets"][phase])
                 and row["method"] == reward and row["test_evaluated"] is False and row["actionable"] is False,
                 "Credit budget, condition or protected-period flag differs")
    if phase == "confirm":
        selected = verify_selection(study, protocol)
        base.require(selected["admitted"] and (selected["selected_reward"], selected["selected_lambda"]) == (reward, value)
                     and completed["selection_sha256"] == base.sha(study / "selection.json"), "Unauthorized confirmation")
    else:
        base.require(completed["selection_sha256"] is None, "Screen borrowed confirmation identity")
    contract = expected_contract(protocol, config)
    for name, steps in (("initial_model", 0), ("final_model", row["steps"])):
        runner.verify_model(directory / name, base.read(directory / name / "metadata.json"), config, protocol, contract, steps)
    initial, cold = margin.initial_model_audit(directory / "initial_model", base.read(directory / "initial_model/metadata.json"), 2.)
    audit = base.read(directory / "final_audit.json")
    base.require(audit == row["audit"] and audit["actual_timesteps"] == row["steps"]
                 and audit["source_hashes"] == protocol["source_hashes"] and audit["tool_sha256"] == protocol["tool_sha256"]
                 and audit["experience"]["total_steps"] == audit["experience"]["benchmark_steps"] == row["steps"]
                 and audit["training_schedule"]["agent_environment_steps"] ==
                 audit["training_schedule"]["benchmark_extra_environment_steps"] == row["steps"], "Actual independent books/budget differ")
    base.verify_rollouts(directory, audit["rollout_buffers"], row["steps"], audit["training_trace_prefix"])
    estimator = estimator_audit(directory, audit, config)
    _, trained = base.model_audit(directory / "final_model", base.read(directory / "final_model/metadata.json"), initial, audit)
    market = base.market_arrays(protocol, fold, contract["stock_codes"])
    schedule = base.schedule_audit(audit, config, market, protocol["folds"][fold])
    training = base.trace_audit(directory / "training_trace", audit["training_trace_prefix"], market, contract, True)
    cases = {}
    for mode in base.MODES:
        baseline = margin.reference_read(protocol, Path(protocol["reproduction_reference_root"]) / "rules" / fold / mode / "result.json")["metrics"]
        for role in ("initialized", "trained"):
            folder = directory / role / mode
            result = base.read(folder / "result.json")
            base.require(result["contract"] == contract and result["initial_mode"] == mode
                         and result["metrics"] == row["metrics"][mode][role]
                         and result["benchmark_metrics"] == row["metrics"][mode]["benchmark"] == baseline
                         and result["evaluation_steps"] == row["evaluation_steps"][role][mode], "Original evaluation identity differs")
            changes = runner.shared.case_changes(folder, result)
            base.require(changes == row["changes"][mode] if role == "trained" else not any(changes.values()), "Greedy/economic flags differ")
            base.verify_evaluation_dates(folder, result, market.evaluation_dates(protocol["folds"][fold]))
            base.case_metrics(folder, result)
            cases[role + "/" + mode] = base.trace_audit(folder / "trace", result["trace"], market, contract, False)
    reproduction = None
    if value == .95 and phase == "screen":
        previous = Path(protocol["reproduction_reference_root"]) / "runs" / f'{phase}_{fold}_{reward}_{seed}'
        reproduction = runner.compare_control(directory, previous, audit)
        base.require(base.read(directory / "reproduction.json") == reproduction, "Fresh lambda95 control reproduction differs")
    return dict(completion_sha256=base.sha(directory / "completed.json"), registered_leaves=len(completed["files"]),
        actual_rl_steps=row["steps"], separate_benchmark_training_steps=row["steps"], fold=fold, phase=phase,
        reward_basis=reward, gae_lambda=value, seed=seed, initial=cold, final=trained, estimator=estimator,
        training=training, schedule=schedule, cases=cases, reproduction=reproduction), initial


def final_scope(study, protocol, trials):
    selected = verify_selection(study, protocol)
    cell = selected["selected_reward"], selected["selected_lambda"]
    expected = {runner.run_id(fold, reward, value, "screen", 42)
                for fold in runner.original.SCREEN_FOLDS for reward, value in runner.CELLS}
    if selected["admitted"]:
        expected |= {runner.run_id(fold, *cell, "confirm", seed) for fold in runner.original.CONFIRM_FOLDS for seed in runner.original.SEEDS}
        for fold in runner.original.CONFIRM_FOLDS:
            runner.verify_worker_receipt(study, protocol, "confirm", fold, cell)
    base.require(set(trials) == expected and {p.name for p in (study / "runs").iterdir() if p.is_dir()} == expected,
                 "Missing or unauthorized credit trial folder")
    rows = {name: base.read(study / "runs" / name / "completed.json")["record"] for name in expected}
    supplied = base.read(study / "summary.json")
    base.require(len(supplied) == len(rows) and {runner.run_id(r["fold"], r["reward_basis"], r["gae_lambda"], r["phase"], r["seed"]): r for r in supplied} == rows,
                 "Summary changed or borrowed completed native rows")
    summary = base.read(study / "study_summary.json")
    steps, eval_steps = sum(r["actual_rl_steps"] for r in trials.values()), sum(c["steps"] for r in trials.values() for c in r["cases"].values())
    base.require(steps == (163840 if selected["admitted"] else 65536)
                 and summary["actual_training_steps"] == summary["independent_training_benchmark_steps"] == steps
                 and summary["final_snapshots"] == len(rows) and summary["actual_screen_steps"] == 65536
                 and summary["actual_confirmation_steps"] == (98304 if selected["admitted"] else 0)
                 and summary["actual_optimizer_updates"] == sum(r["final"]["adam_updates"] for r in trials.values())
                 and summary["evaluation_agent_steps"] == summary["evaluation_parallel_benchmark_steps"] == eval_steps
                 and summary["new_rule_control_steps"] == summary["additional_reproduction_environment_steps"] == 0
                 and summary["screen"] == selected and summary["test_evaluated"] is False and summary["actionable"] is False,
                 "Final credit financial/experience summary differs")
    confirmed = [rows[runner.run_id(fold, *cell, "confirm", seed)] for fold in runner.original.CONFIRM_FOLDS
                 for seed in runner.original.SEEDS] if selected["admitted"] else []
    financial = runner.original.summarize_confirmation(confirmed, cell[0]) if confirmed else dict(qualified=False, qualified_candidate=None)
    base.require(all(summary.get(key) == value for key, value in financial.items()), "Credit qualification differs from native confirmation")
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / "reports/runs/ppo_timing_credit_20261015"))
    parser.add_argument("--sealed", action="store_true")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args(argv)
    base.require(not args.final or args.sealed, "Final audit requires sealed native trials")
    torch.set_num_threads(1)
    study = Path(args.root).resolve()
    protocol = base.read(study / "protocol.json")
    result = dict(status="passed", scope="frozen_registration", registration=registration(study, protocol),
        model_instances=0, network_forwards=0, environment_steps=0, gradient_calls=0, optimizer_steps=0,
        registered_leaf_writes=0, frozen_source_writes=0)
    if args.sealed:
        trials, initials = {}, {}
        running = study / "independent_audit_running.json"
        for marker in sorted((study / "runs").glob("*/completed.json")):
            row, theta = sealed_trial(study, protocol, marker.parent)
            trials[marker.parent.name] = row
            initials[row["fold"], row["phase"], row["seed"], (row["reward_basis"], row["gae_lambda"])] = theta
            print("AUDITED " + marker.parent.name, flush=True)
            running.write_text(json.dumps(dict(status="running", completed_trials=list(trials),
                audited_rl_steps=sum(r["actual_rl_steps"] for r in trials.values()), network_forwards=0,
                environment_steps=0, registered_leaf_writes=0), indent=2) + "\n", encoding="utf-8")
        result.update(status="partial", scope="registration_and_listed_sealed_native_trials", sealed_trials=trials,
            sealed_trial_count=len(trials), actual_rl_steps=sum(r["actual_rl_steps"] for r in trials.values()),
            shared_initials=shared_initials(initials))
        if args.final:
            result.update(status="passed", scope="complete_authorized_sealed_native_study", selection=final_scope(study, protocol, trials))
    output = study / "independent_audit_progress.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.sealed:
        running.write_text(json.dumps(dict(status=result["status"], completed_trials=list(trials), progress_sha256=base.sha(output),
            network_forwards=0, environment_steps=0, registered_leaf_writes=0), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("registration", "sealed_trials")}, sort_keys=True))


if __name__ == "__main__":
    main()
