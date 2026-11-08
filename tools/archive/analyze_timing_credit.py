"""Sealed credit-cell diagnostics from arrays and provenance, never neural execution."""

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import file_hash, read_config, write_json
from src.training.freeze import json_digest
from tools.archive import research_timing_credit as study
from tools.archive import analyze_timing as common
from tools.archive.analyze_timing_margin import margin_metrics
from tools.archive.audit_timing_credit import estimator_audit


def scale_metrics(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Credit scales require finite saved samples")
    return dict(count=values.size, mean=float(values.mean()), std=float(values.std()), minimum=float(values.min()), maximum=float(values.max()))


def read_stage(root, protocol, phase, selected):
    """Verify cold evidence without the control reader's tensor deserialization."""
    rows = []
    cell = (selected["selected_reward"], selected["selected_lambda"])
    for fold in study.original.SCREEN_FOLDS if phase == "screen" else study.original.CONFIRM_FOLDS:
        study.verify_worker_receipt(root, protocol, phase, fold, cell if phase == "confirm" else None)
        for reward, value in study.CELLS if phase == "screen" else (cell,):
            for seed in (42,) if phase == "screen" else study.original.SEEDS:
                identifier = study.run_id(fold, reward, value, phase, seed)
                directory = root / "runs" / identifier
                marker = directory / "completed.json"
                saved = read_config(marker)
                config = study.config_for(protocol, root, fold, reward, value, phase, seed)
                if (saved["protocol_sha256"] != json_digest(protocol) or saved["config_sha256"] != json_digest(config)
                        or saved["tool_sha256"] != protocol["tool_sha256"]
                        or saved["selection_sha256"] != (file_hash(root / "selection.json") if phase == "confirm" else None)
                        or phase == "screen" and file_hash(marker) != selected["completed_sha256"][identifier]):
                    raise ValueError("Credit diagnostic condition differs from frozen selection")
                study.original.verify_leaf_files(directory, saved["files"])
                row, data = saved["record"], study.CreditData(config)  # Tables only, never env().
                if (row["fold"], row["reward_basis"], row["gae_lambda"], row["phase"], row["seed"], row["steps"]) != (fold, reward, value, phase, seed, study.BUDGETS[phase]):
                    raise ValueError("Credit diagnostic row is not the registered fixed-budget cell")
                for name, steps in (("initial_model", 0), ("final_model", row["steps"])):
                    study.verify_model(directory / name, read_config(directory / name / "metadata.json"), config, protocol, data.contract, steps)
                audit = read_config(directory / "final_audit.json")
                if audit != row["audit"] or audit["actual_timesteps"] != row["steps"] or audit["experience"]["total_steps"] != row["steps"]:
                    raise ValueError("Credit diagnostic counters differ from native trial")
                study.original.verify_timing_trace(directory / "training_trace", audit["training_trace_prefix"], steps=row["steps"], expected_contract_sha256=json_digest(data.contract))
                study.original.verify_rollouts(directory, audit["rollout_buffers"], row["steps"], audit["training_trace_prefix"])
                study.verify_estimator_buffers(directory, audit, config)
                estimator_audit(directory, audit, config)
                for mode in row["metrics"]:
                    for label in ("initialized", "trained"):
                        folder = directory / label / mode
                        result = read_config(folder / "result.json")
                        study.original._verify_result(folder, result, data.contract, mode, expected_dates=study.original.validation_dates(data))
                        if result["metrics"] != row["metrics"][mode][label] or result["benchmark_metrics"] != row["metrics"][mode]["benchmark"]:
                            raise ValueError("Credit diagnostic ROI differs from sealed economic curves")
                        if label == "trained" and study.shared.case_changes(folder, result) != row["changes"][mode]:
                            raise ValueError("Credit diagnostic behavior flags differ from original books")
                rows.append(row)
    return study.validate_records(rows, phase, cell if phase == "confirm" else None)


def estimator_scales(directory, audit, config):
    checked = estimator_audit(directory, audit, config)
    scales = []
    for record, boundary in zip(audit["rollout_buffers"], audit["credit_estimator_buffers"]):
        with np.load(directory / record["path"], allow_pickle=False) as buffer, np.load(directory / boundary["path"], allow_pickle=False) as side:
            scales.append(dict(first_step=record["first_step"], last_step=record["last_step"], steps=record["steps"],
                gamma=float(side["gamma"]), gae_lambda=float(side["gae_lambda"]),
                returns=scale_metrics(buffer["returns"]), advantages=scale_metrics(buffer["advantages"]), old_values=scale_metrics(buffer["values"]),
                raw_rewards=scale_metrics(side["raw_rewards"]), corrected_rewards=scale_metrics(buffer["rewards"]),
                timeout_bootstraps=int(side["truncated"].sum()), natural_terminals=int((side["dones"] & ~side["truncated"]).sum()),
                actual_tail_value=float(side["last_values"][0]), actual_tail_done=bool(side["last_dones"][0])))
    return checked, scales


def analyze_trial(root, row):
    identifier = study.run_id(row["fold"], row["reward_basis"], row["gae_lambda"], row["phase"], row["seed"])
    directory = root / "runs" / identifier
    frame = common.read_native(directory / "training_trace", row["audit"]["training_trace_prefix"])
    groups = [dict(run_id=identifier, scope="training", phase=int(phase), mode=mode, **margin_metrics(part, 2.))
              for (phase, mode), part in frame.groupby(["phase", "mode"], sort=True)]
    config = read_config(directory / "config.json")
    checked, scales = estimator_scales(directory, row["audit"], config)
    for scale in scales:
        scale["run_id"] = identifier
    report = dict(run_id=identifier, reward_basis=row["reward_basis"], gae_lambda=row["gae_lambda"], fold=row["fold"], phase=row["phase"], seed=row["seed"],
        completed_sha256=file_hash(directory / "completed.json"), training=margin_metrics(frame, 2.), experience=row["audit"]["experience"],
        actual_optimizer_updates=row["audit"]["actual_optimizer_updates"], critic=common.critic_metrics(directory, row["audit"]["rollout_buffers"]),
        estimator_audit=checked, estimator_scales=scales, tensor_delta_l2=row["audit"]["tensor_delta_l2"],
        theta_source="Saved runtime initial-to-final tensor norms; no weights were deserialized",
        initial_policy_tensor_payload_sha256=hashlib.sha256(study.original._policy_bytes(directory / "initial_model")).hexdigest(),
        final_policy_tensor_payload_sha256=hashlib.sha256(study.original._policy_bytes(directory / "final_model")).hexdigest(), evaluations={})
    for mode in row["metrics"]:
        folder = directory / "trained" / mode
        result = read_config(folder / "result.json")
        sample = common.read_native(folder / "trace", result["trace"])
        report["evaluations"][mode] = dict(**margin_metrics(sample, 2.), metrics=result["metrics"],
            paired_net_roi_gap=result["metrics"]["cumulative_return"] - result["benchmark_metrics"]["cumulative_return"], changes=row["changes"][mode])
        groups.extend(dict(run_id=identifier, scope="deterministic_evaluation", phase=int(phase), mode=mode, **margin_metrics(part, 2.))
                      for phase, part in sample.groupby("phase", sort=True))
    return report, groups, scales


def cell_summary(trials):
    cells = {}
    for report in trials:
        label = study.cell_label(report["reward_basis"], report["gae_lambda"])
        group = cells.setdefault(label, dict(reward_basis=report["reward_basis"], gae_lambda=report["gae_lambda"], runs=0,
            actual_optimizer_updates=0, timeout_bootstraps=0, natural_terminals=0, greedy_eval_flips=0, fold_account_mean_gaps={}))
        group["runs"] += 1
        group["actual_optimizer_updates"] += report["actual_optimizer_updates"]
        group["timeout_bootstraps"] += report["estimator_audit"]["timeout_bootstraps"]
        group["natural_terminals"] += report["estimator_audit"]["natural_terminals"]
        group["greedy_eval_flips"] += sum(case["greedy_calendar_flips"] for case in report["evaluations"].values())
        group["fold_account_mean_gaps"][f'{report["phase"]}_{report["fold"]}_{report["seed"]}'] = float(np.mean([case["paired_net_roi_gap"] for case in report["evaluations"].values()]))
    for label, group in cells.items():
        matching = [report for report in trials if study.cell_label(report["reward_basis"], report["gae_lambda"]) == label]
        rollouts = [rollout for report in matching for rollout in report.get("estimator_scales", [])]
        group["rollouts_with_captured_boundaries"] = len(rollouts)
        group["tail_done_rollouts"] = sum(rollout["actual_tail_done"] for rollout in rollouts)
        group["active_tail_rollouts"] = sum(not rollout["actual_tail_done"] for rollout in rollouts)
        for name in ("returns", "advantages", "raw_rewards", "corrected_rewards"):
            stds = [rollout[name]["std"] for rollout in rollouts]
            group["mean_rollout_" + name + "_std"] = float(np.mean(stds)) if stds else None
        group["theta_delta_l2_by_trial"] = {report.get("run_id", str(index)): report.get("tensor_delta_l2", {}) for index, report in enumerate(matching)}
    return cells


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reports/runs/ppo_timing_credit_20261015")
    parser.add_argument("--stage", choices=("screen", "confirm", "all"), required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    protocol = read_config(root / "protocol.json")
    study.check_frozen(protocol)
    if not (root / "selection.json").exists():
        raise ValueError("Credit diagnostics require all eight sealed screens and root's locked selection")
    selected = read_config(root / "selection.json")
    screen = read_stage(root, protocol, "screen", selected)
    if selected["protocol_sha256"] != json_digest(protocol) or any(selected.get(key) != value for key, value in study.select_screen(screen).items()):
        raise ValueError("Credit diagnostic selection differs from the predeclared rule")
    rows = screen if args.stage != "confirm" else []
    if args.stage in ("confirm", "all") and selected["admitted"]:
        rows += read_stage(root, protocol, "confirm", selected)
    elif args.stage == "confirm":
        raise ValueError("Failed screen authorizes zero confirmation diagnostics")
    output = Path(args.output or root / "diagnostics" / ("credit_" + args.stage)).resolve()
    if not output.is_relative_to(root / "diagnostics") or output == root / "diagnostics" or output.exists():
        raise ValueError("Output must be a new diagnostics child, never a signed leaf or existing result")
    reports, groups, scales = [], [], []
    for row in rows:
        report, grouped, sampled = analyze_trial(root, row)
        reports.append(report)
        groups.extend(grouped)
        scales.extend(sampled)
    result = dict(schema_version=1, status="complete_for_sealed_stage", stage=args.stage, runs=len(rows),
        actual_training_steps=sum(row["steps"] for row in rows), actual_optimizer_updates=sum(row["actual_optimizer_updates"] for row in reports),
        actual_confirmation_runs=sum(row["phase"] == "confirm" for row in rows), screen_admitted=selected["admitted"],
        protocol_sha256=file_hash(root / "protocol.json"), selection_sha256=file_hash(root / "selection.json"), tool_sha256=file_hash(Path(__file__)),
        no_model_load=True, no_tensor_deserialization=True, no_forward=True, no_environment_steps=True, no_gradient_updates=True, no_trial_leaf_writes=True,
        trials=reports, cells=cell_summary(reports),
        interpretation=["Return/advantage targets include actual captured SB3 tail and TimeLimit values; no new or inferred Critic predictions were used.",
            "Raw and bootstrap-corrected reward scales differ; relative_log reward is not net ROI.",
            "Old Critic error is pre-update against stored GAE targets, not post-fit forecasting accuracy.",
            "Training distributions mix successive policies and sampled accounts; action-day conditional profit is not a general causal effect.",
            "Raw policy payload hashes and saved theta-delta norms do not require weight deserialization."])
    output.mkdir(parents=True)
    write_json(output / "diagnostics.json", result)
    pd.DataFrame(groups).to_csv(output / "phase_mode.csv", index=False)
    pd.DataFrame(scales).to_csv(output / "rollout_scales.csv", index=False)
    print(f'CREDIT_DIAGNOSTICS {args.stage} sealed_runs={len(rows)} RL_steps={result["actual_training_steps"]} Adam={result["actual_optimizer_updates"]} output={output}', flush=True)
    return result


if __name__ == "__main__":
    main()
