"""Thin margin-aware diagnostics of sealed evidence; no tensors, models or replay."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import file_hash, read_config, write_json
from src.training.freeze import json_digest
from tools import research_timing_margin as study
from tools import analyze_timing as common


def margin_metrics(frame, margin):
    if isinstance(margin, bool) or margin not in study.MARGINS:
        raise ValueError("Only the registered numeric margins are supported")
    result = common.timing_metrics(frame)
    if not len(frame):
        return dict(result, margin=margin)
    p = np.clip(frame.p_rebalance.to_numpy(dtype=float), 1e-15, 1 - 1e-15)
    phase = frame.phase.to_numpy(dtype=int)
    initial_gap = np.where(phase == 0, margin, -margin)
    initial = 1. / (1. + np.exp(-initial_gap))
    gap = np.log(p) - np.log1p(-p)
    result.update(margin=margin, initial_preferred_probability=float(1. / (1 + np.exp(-margin))),
        logit_gap_delta_from_calendar=common.stats(gap - initial_gap),
        categorical_entropy=common.stats(-p * np.log(p) - (1 - p) * np.log1p(-p)),
        initial_to_current_kl=common.stats(initial * np.log(initial / p) + (1 - initial) * np.log((1 - initial) / (1 - p))),
        current_to_initial_kl=common.stats(p * np.log(p / initial) + (1 - p) * np.log((1 - p) / (1 - initial))))
    return result


def read_stage(root, protocol, phase, selected):
    """Validate signed cold data, avoiding the runner's tensor-reproduction reader."""
    rows = []
    margins = study.MARGINS if phase == "screen" else (selected["selected_margin"],)
    seeds = (42,) if phase == "screen" else study.original.SEEDS
    folds = study.original.SCREEN_FOLDS if phase == "screen" else study.original.CONFIRM_FOLDS
    for fold in folds:
        study.verify_worker_receipt(root, protocol, phase, fold, selected["selected_margin"] if phase == "confirm" else None)
        for margin in margins:
            for seed in seeds:
                identifier = study.run_id(fold, margin, phase, seed)
                directory = Path(root) / "runs" / identifier
                marker = directory / "completed.json"
                completed = read_config(marker)
                config = study.config_for(protocol, root, fold, margin, phase, seed)
                if (completed["protocol_sha256"] != json_digest(protocol) or completed["config_sha256"] != json_digest(config)
                        or completed["tool_sha256"] != protocol["tool_sha256"]
                        or completed["selection_sha256"] != (file_hash(Path(root) / "selection.json") if phase == "confirm" else None)
                        or phase == "screen" and file_hash(marker) != selected["completed_sha256"][identifier]):
                    raise ValueError("Diagnostic trial differs from its frozen selection and provenance")
                study.original.verify_leaf_files(directory, completed["files"])
                row = completed["record"]
                if (row["fold"], row["margin"], row["phase"], row["seed"], row["steps"]) != (fold, margin, phase, seed, study.original.BUDGETS[phase]):
                    raise ValueError("Diagnostic trial is not the registered fixed-budget condition")
                data = study.MarginData(config)  # Local table parsing only; no env() call.
                for name, steps in (("initial_model", 0), ("final_model", row["steps"])):
                    study.verify_model(directory / name, read_config(directory / name / "metadata.json"), config, protocol, data.contract, steps)
                audit = read_config(directory / "final_audit.json")
                if row["audit"] != audit or audit["actual_timesteps"] != row["steps"] or audit["experience"]["total_steps"] != row["steps"]:
                    raise ValueError("Saved optimizer/native counters differ from the actual trial")
                study.original.verify_timing_trace(directory / "training_trace", audit["training_trace_prefix"], steps=row["steps"], expected_contract_sha256=json_digest(data.contract))
                study.original.verify_rollouts(directory, audit["rollout_buffers"], row["steps"], audit["training_trace_prefix"])
                for mode in row["metrics"]:
                    for label in ("initialized", "trained"):
                        folder = directory / label / mode
                        result = read_config(folder / "result.json")
                        study.original._verify_result(folder, result, data.contract, mode, expected_dates=study.original.validation_dates(data))
                        if result["metrics"] != row["metrics"][mode][label] or result["benchmark_metrics"] != row["metrics"][mode]["benchmark"]:
                            raise ValueError("Diagnostic economic curves differ from their sealed row")
                        if label == "trained" and study.case_changes(folder, result) != row["changes"][mode]:
                            raise ValueError("Diagnostic behavior flags differ from native evidence")
                rows.append(row)
    return study.validate_records(rows, phase, selected["selected_margin"] if phase == "confirm" else None)


def analyze_trial(root, row):
    identifier = study.run_id(row["fold"], row["margin"], row["phase"], row["seed"])
    directory = Path(root) / "runs" / identifier
    frame = common.read_native(directory / "training_trace", row["audit"]["training_trace_prefix"])
    groups = [dict(run_id=identifier, scope="training", phase=int(phase), mode=mode, **margin_metrics(part, row["margin"]))
              for (phase, mode), part in frame.groupby(["phase", "mode"], sort=True)]
    episodes = read_config(directory / "episodes.json")
    for episode in episodes:
        episode["run_id"] = identifier
    updates = pd.read_csv(directory / "training_updates.csv")
    report = dict(run_id=identifier, margin=row["margin"], fold=row["fold"], phase=row["phase"], seed=row["seed"],
        completed_sha256=file_hash(directory / "completed.json"), training=margin_metrics(frame, row["margin"]),
        experience=row["audit"]["experience"], actual_optimizer_updates=row["audit"]["actual_optimizer_updates"],
        critic=common.critic_metrics(directory, row["audit"]["rollout_buffers"]),
        actor_tensor_delta_l2={key: value for key, value in row["audit"]["tensor_delta_l2"].items() if key.startswith("action_net.")},
        saved_approx_kl=common.stats(updates["train/approx_kl"].dropna()), evaluations={})
    for mode in row["metrics"]:
        folder = directory / "trained" / mode
        result = read_config(folder / "result.json")
        sample = common.read_native(folder / "trace", result["trace"])
        report["evaluations"][mode] = dict(**margin_metrics(sample, row["margin"]), metrics=result["metrics"],
            paired_net_roi_gap=result["metrics"]["cumulative_return"] - result["benchmark_metrics"]["cumulative_return"])
        groups.extend(dict(run_id=identifier, scope="deterministic_evaluation", phase=int(phase), mode=mode, **margin_metrics(part, row["margin"]))
                      for phase, part in sample.groupby("phase", sort=True))
    return report, groups, episodes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reports/runs/ppo_timing_margin_20261014")
    parser.add_argument("--stage", choices=("screen", "confirm", "all"), required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    protocol = read_config(root / "protocol.json")
    study.check_frozen(protocol)
    if not (root / "selection.json").exists():
        raise ValueError("Whole-stage diagnostics wait for all six sealed screens and root's fixed selection")
    selected = read_config(root / "selection.json")
    screen = read_stage(root, protocol, "screen", selected)
    recomputed = study.select_screen(screen)
    if selected["protocol_sha256"] != json_digest(protocol) or any(selected.get(key) != value for key, value in recomputed.items()):
        raise ValueError("Frozen selection no longer follows the registered margin rule")
    rows = screen if args.stage != "confirm" else []
    if args.stage in ("confirm", "all") and selected["admitted"]:
        rows += read_stage(root, protocol, "confirm", selected)
    elif args.stage == "confirm":
        raise ValueError("Failed screen authorizes no confirmation diagnostics")
    output = Path(args.output or root / "diagnostics" / ("margin_" + args.stage)).resolve()
    if not output.is_relative_to(root / "diagnostics") or output == root / "diagnostics" or output.exists():
        raise ValueError("Output must be a new diagnostics child, never a signed leaf or existing result")
    reports, groups, episodes = [], [], []
    for row in rows:
        report, grouped, sampled = analyze_trial(root, row)
        reports.append(report)
        groups.extend(grouped)
        episodes.extend(sampled)
    output.mkdir(parents=True)
    result = dict(schema_version=1, status="complete_for_sealed_stage", stage=args.stage,
        runs=len(rows), actual_training_steps=sum(row["steps"] for row in rows), actual_optimizer_updates=sum(row["actual_optimizer_updates"] for row in reports),
        actual_confirmation_runs=sum(row["phase"] == "confirm" for row in rows), screen_admitted=selected["admitted"],
        protocol_sha256=file_hash(root / "protocol.json"), selection_sha256=file_hash(root / "selection.json"), tool_sha256=file_hash(Path(__file__)),
        no_model_load=True, no_tensor_deserialization=True, no_forward=True, no_environment_steps=True, no_gradient_updates=True, no_trial_leaf_writes=True,
        trials=reports, interpretation=["Initialization logit gap, entropy and KL use each actual registered margin, not a hard-coded value of two.",
            "Saved training distributions describe successive old policies on changing sampled accounts, not the final policy on fixed states.",
            "Action-day conditional return is descriptive; continuing holdings create subsequent path effects.",
            "GAE targets are bootstrapped, and Critic diagnostics are pre-update; no fresh predictions were computed.",
            "This tool does not choose a margin, change admission or promote a model."])
    write_json(output / "diagnostics.json", result)
    pd.DataFrame(groups).to_csv(output / "phase_mode.csv", index=False)
    pd.DataFrame(episodes).drop(columns=["rewards", "raw_returns", "benchmark_returns", "initial_account"], errors="ignore").to_csv(output / "episode_coverage.csv", index=False)
    print(f'MARGIN_DIAGNOSTICS {args.stage} sealed_runs={len(rows)} RL_steps={result["actual_training_steps"]} Adam={result["actual_optimizer_updates"]} output={output}', flush=True)
    return result


if __name__ == "__main__":
    main()
