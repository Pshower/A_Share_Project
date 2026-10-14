"""Offline shared-path budget and same-budget Actor comparisons after sealing."""

import argparse
from copy import deepcopy
import io
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import file_hash, read_config, write_json
from tools import research_timing_phase_budget as study
from tools import analyze_timing as common
from tools.analyze_timing_credit import scale_metrics
from tools.analyze_timing_margin import margin_metrics
from tools.audit_timing_credit import estimator_arrays


def output_path(root, requested, stage):
    root = Path(root).resolve()
    output = Path(requested or root / "diagnostics" / ("phase_budget_" + stage)).resolve()
    if output == root / "diagnostics" or not output.is_relative_to(root / "diagnostics") or output.exists():
        raise ValueError("Output must be a new diagnostics child, never a signed leaf or existing result")
    return output


def unique_physical_rows(rows):
    physical = {}
    for row in rows:
        study.cell_label(row["architecture"], row["steps"])
        name = row["physical_run_id"]
        if name in physical:
            previous = physical[name]
            if (previous["architecture"], previous["fold"], previous["phase"], previous["seed"]) != (
                    row["architecture"], row["fold"], row["phase"], row["seed"]):
                raise ValueError("A physical path cannot combine distinct conditions")
        if name not in physical or row["steps"] > physical[name]["steps"]:
            physical[name] = row
    return physical


def physical_counters(rows):
    physical = unique_physical_rows(rows)
    initialized = sum(sum(row["evaluation_steps"]["initialized"].values()) for row in physical.values())
    trained = sum(sum(row["evaluation_steps"]["trained"].values()) for row in rows)
    return dict(physical_training_runs=len(physical), evaluation_snapshots=len(rows),
        actual_agent_rl_steps=sum(row["steps"] for row in physical.values()),
        separate_training_benchmark_steps=sum(row["audit"]["experience"]["benchmark_steps"] for row in physical.values()),
        actual_adam_updates=sum(row["audit"]["actual_optimizer_updates"] for row in physical.values()),
        snapshot_step_sum=sum(row["steps"] for row in rows),
        shared_screen_prefix_steps=sum(row["steps"] for row in rows if row["phase"] == "screen" and row["steps"] == 8192),
        original_unique_gae_buffers=sum(len(row["audit"]["rollout_buffers"]) for row in physical.values()),
        snapshot_gae_buffer_views=sum(len(row["audit"]["rollout_buffers"]) for row in rows),
        initialized_agent_paths=4 * len(physical), trained_agent_paths=4 * len(rows),
        evaluation_agent_steps=initialized + trained, evaluation_parallel_benchmark_steps=initialized + trained)


def paired_effects(rows):
    screen = [row for row in rows if row["phase"] == "screen"]
    study.validate_records(screen, "screen")
    lookup = {(row["fold"], row["architecture"], row["steps"]): row for row in screen}
    effects = []
    def compare(left, right, kind):
        for mode in study.MODES:
            a, b = left["metrics"][mode], right["metrics"][mode]
            if a["benchmark"] != b["benchmark"] or a["initialized"] != b["initialized"]:
                raise ValueError("Paired effects need identical unchanged baseline and initialization")
            effects.append(dict(comparison=kind, fold=left["fold"], seed=42, mode=mode,
                left_architecture=left["architecture"], left_steps=left["steps"], right_architecture=right["architecture"], right_steps=right["steps"],
                net_roi_difference=a["trained"]["cumulative_return"] - b["trained"]["cumulative_return"],
                baseline_gap_left=a["trained"]["cumulative_return"] - a["benchmark"]["cumulative_return"],
                baseline_gap_right=b["trained"]["cumulative_return"] - b["benchmark"]["cumulative_return"],
                own_initialization_gain_left=a["trained"]["cumulative_return"] - a["initialized"]["cumulative_return"],
                own_initialization_gain_right=b["trained"]["cumulative_return"] - b["initialized"]["cumulative_return"],
                drawdown_difference=abs(a["trained"]["max_drawdown"]) - abs(b["trained"]["max_drawdown"]),
                fee_difference=a["trained"]["total_fees"] - b["trained"]["total_fees"],
                slippage_difference=a["trained"]["total_slippage_cost"] - b["trained"]["total_slippage_cost"],
                left_changes=deepcopy(left["changes"][mode]), right_changes=deepcopy(right["changes"][mode])))
    for fold in study.original.SCREEN_FOLDS:
        for architecture in study.ARCHITECTURES:
            compare(lookup[fold, architecture, 32768], lookup[fold, architecture, 8192], "same_path_long_minus_short")
        for steps in study.SNAPSHOT_STEPS:
            compare(lookup[fold, "due_phase_residual", steps], lookup[fold, "linear", steps], "same_budget_due_minus_linear")
    return effects


def estimator_scales(directory, row):
    """Rebuild every actual GAE once per physical path, including true boundaries."""
    audit = row["audit"]
    buffers, sides, chunks = audit["rollout_buffers"], audit["credit_estimator_buffers"], audit["training_trace_prefix"]["chunks"]
    if len(buffers) != len(sides) or len(buffers) != len(chunks):
        raise ValueError("Physical estimator evidence omits an original buffer")
    updates = pd.read_csv(directory / audit["training_updates_path"], float_precision="round_trip")
    previous, scales = True, []
    for index, (buffer_record, side_record, chunk) in enumerate(zip(buffers, sides, chunks)):
        with np.load(directory / buffer_record["path"], allow_pickle=False) as saved:
            buffer = {key: saved[key] for key in saved.files}
        with np.load(directory / side_record["path"], allow_pickle=False) as saved:
            side = {key: saved[key] for key in saved.files}
        with np.load(directory / "training_trace" / chunk["path"], allow_pickle=False) as saved:
            native = {key: saved[key] for key in ("done", "action_learning_reward")}
        checked = estimator_arrays(buffer, side, native, 1., previous)
        previous = bool(side["last_dones"][0])
        for field, key in (("returns", "credit/returns_std"), ("advantages", "credit/advantages_std"), ("rewards", "credit/buffer_rewards_std")):
            if not np.isclose(float(np.std(buffer[field])), updates.iloc[index][key], atol=1e-7, rtol=1e-6):
                raise ValueError("Actual estimator-scale log differs from the stored array")
        scales.append(dict(physical_run_id=row["physical_run_id"], first_step=buffer_record["first_step"], last_step=buffer_record["last_step"],
            original_buffer_sha256=buffer_record["sha256"], actual_boundary_sha256=side_record["sha256"],
            returns=scale_metrics(buffer["returns"]), advantages=scale_metrics(buffer["advantages"]), old_values=scale_metrics(buffer["values"]),
            raw_rewards=scale_metrics(side["raw_rewards"]), corrected_rewards=scale_metrics(buffer["rewards"]),
            actual_tail_value=float(side["last_values"][0]), actual_tail_done=bool(side["last_dones"][0]), **checked))
    return scales


def snapshot_scale_view(row, physical, scales):
    actual = row["audit"]["rollout_buffers"]
    complete = physical["audit"]["rollout_buffers"]
    if actual != complete[:len(actual)] or len(actual) != row["steps"] // 512:
        raise ValueError("Short estimator view is not the same physical path prefix")
    return scales[:len(actual)]


def read_theta(folder):
    import torch
    with zipfile.ZipFile(Path(folder) / "model.zip") as zipped:
        theta = torch.load(io.BytesIO(zipped.read("policy.pth")), map_location="cpu", weights_only=True)
    return {key: value.detach().cpu().numpy().astype(np.float64) for key, value in theta.items()}


def theta_difference(left, right):
    """Compare the shared parameterization, keeping the extra due vector explicit."""
    def mapped(theta):
        return {key.replace("action_net.base.", "action_net."): value for key, value in theta.items() if key != "action_net.residual.weight"}
    a, b = mapped(left), mapped(right)
    if set(a) != set(b) or any(a[key].shape != b[key].shape for key in a):
        raise ValueError("Shared Actor/Critic parameter coordinates differ")
    return dict(shared_tensor_l2={key: float(np.linalg.norm(a[key] - b[key])) for key in a},
        left_due_vector_l2=float(np.linalg.norm(left["action_net.residual.weight"])) if "action_net.residual.weight" in left else 0.,
        right_due_vector_l2=float(np.linalg.norm(right["action_net.residual.weight"])) if "action_net.residual.weight" in right else 0.,
        due_vector_difference_l2=float(np.linalg.norm(left["action_net.residual.weight"] - right["action_net.residual.weight"]))
            if "action_net.residual.weight" in left and "action_net.residual.weight" in right else None,
        computation="CPU weights-only cold read; NumPy float64 norm; no network construction or forward")


def analyze_snapshot(root, row, scales):
    directory = root / "runs" / row["physical_run_id"]
    frame = common.read_native(directory / "training_trace", row["audit"]["training_trace_prefix"])
    identity = dict(physical_run_id=row["physical_run_id"], architecture=row["architecture"], snapshot_steps=row["steps"], fold=row["fold"], seed=row["seed"])
    groups = [dict(**identity, scope="training_prefix", phase=int(phase), mode=mode, **margin_metrics(part, 2.))
              for (phase, mode), part in frame.groupby(["phase", "mode"], sort=True)]
    result = dict(**identity, stage=row["phase"], training=margin_metrics(frame, 2.), experience=row["audit"]["experience"],
        actual_optimizer_updates_at_snapshot=row["audit"]["actual_optimizer_updates"], critic=common.critic_metrics(directory, row["audit"]["rollout_buffers"]),
        actual_boundary_gae=dict(buffers=len(scales), steps=sum(item["steps"] for item in scales),
            timeout_bootstraps=sum(item["timeout_bootstraps"] for item in scales), natural_terminals=sum(item["natural_terminals"] for item in scales),
            active_tails=sum(not item["actual_tail_done"] for item in scales), done_tails=sum(item["actual_tail_done"] for item in scales),
            maximum_advantage_error=max(item["maximum_advantage_error"] for item in scales), maximum_return_error=max(item["maximum_return_error"] for item in scales)),
        mean_rollout_scales={key: float(np.mean([item[key]["std"] for item in scales])) for key in ("returns", "advantages", "raw_rewards", "corrected_rewards")},
        validated_initial_to_snapshot_tensor_l2=row["audit"]["tensor_delta_l2"],
        theta_verification="Frozen cached_trial recomputes each norm from actual initial/snapshot float32 archives", evaluations={})
    for mode in study.MODES:
        folder = directory / "snapshots" / str(row["steps"]) / "trained" / mode
        saved = read_config(folder / "result.json")
        sample = common.read_native(folder / "trace", saved["trace"])
        result["evaluations"][mode] = dict(**margin_metrics(sample, 2.), metrics=saved["metrics"], baseline_metrics=saved["benchmark_metrics"],
            net_roi_gap=saved["metrics"]["cumulative_return"] - saved["benchmark_metrics"]["cumulative_return"], changes=row["changes"][mode])
        groups.extend(dict(**identity, scope="deterministic_evaluation", phase=int(phase), mode=mode, **margin_metrics(part, 2.))
                      for phase, part in sample.groupby("phase", sort=True))
    return result, groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reports/runs/ppo_timing_phase_budget_20261015")
    parser.add_argument("--stage", choices=("screen", "confirm", "all"), required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    protocol = read_config(root / "protocol.json")
    study.check_frozen(protocol)
    if not (root / "selection.json").exists():
        raise ValueError("Diagnostics wait for all four long paths, eight sealed snapshots and locked selection")
    selected = read_config(root / "selection.json")
    screen = study.load_stage(root, protocol, "screen")
    if not study.same_json(selected, study.selection_record(root, protocol, screen)):
        raise ValueError("Analysis selection differs from the registered complete matrix")
    confirmed = study.load_stage(root, protocol, "confirm", (selected["selected_architecture"], selected["selected_budget"])) if args.stage in ("confirm", "all") and selected["admitted"] else []
    if args.stage == "confirm" and not selected["admitted"]:
        raise ValueError("Failed screen authorizes zero confirmation analysis")
    rows = confirmed if args.stage == "confirm" else screen + confirmed
    output = output_path(root, args.output, args.stage)
    physical = unique_physical_rows(rows)
    scales = {name: estimator_scales(root / "runs" / name, row) for name, row in physical.items()}
    reports, groups = [], []
    for row in rows:
        view = snapshot_scale_view(row, physical[row["physical_run_id"]], scales[row["physical_run_id"]])
        report, grouped = analyze_snapshot(root, row, view)
        reports.append(report)
        groups.extend(grouped)
    effects = paired_effects(screen) if args.stage != "confirm" else []
    theta_pairs = []
    if args.stage != "confirm":
        theta = {(row["fold"], row["architecture"], row["steps"]): read_theta(row["model_path"]) for row in screen}
        for fold in study.original.SCREEN_FOLDS:
            for architecture in study.ARCHITECTURES:
                theta_pairs.append(dict(comparison="same_path_long_minus_short", fold=fold, architecture=architecture,
                    **theta_difference(theta[fold, architecture, 32768], theta[fold, architecture, 8192])))
            for steps in study.SNAPSHOT_STEPS:
                theta_pairs.append(dict(comparison="same_budget_due_minus_linear", fold=fold, snapshot_steps=steps,
                    **theta_difference(theta[fold, "due_phase_residual", steps], theta[fold, "linear", steps])))
    result = dict(schema_version=1, status="complete_for_sealed_stage", stage=args.stage, counters=physical_counters(rows),
        protocol_sha256=file_hash(root / "protocol.json"), selection_sha256=file_hash(root / "selection.json"), tool_sha256=file_hash(Path(__file__)),
        selected_cell=dict(architecture=selected["selected_architecture"], budget=selected["selected_budget"], admitted=selected["admitted"]),
        snapshots=reports, paired_effects=effects, paired_theta=theta_pairs,
        unique_physical_estimator_summaries={name: dict(buffers=len(items), timeout_bootstraps=sum(item["timeout_bootstraps"] for item in items),
            natural_terminals=sum(item["natural_terminals"] for item in items), active_tails=sum(not item["actual_tail_done"] for item in items),
            maximum_advantage_error=max(item["maximum_advantage_error"] for item in items), maximum_return_error=max(item["maximum_return_error"] for item in items)) for name, items in scales.items()},
        no_model_load=True, no_network_forwards=True, no_environment_steps=True, no_gradient_updates=True, no_trial_leaf_writes=True,
        tensor_read_scope="CPU weights_only archive reads in frozen cache and explicit parameter comparisons; no policy construction",
        interpretation=["The 8192 snapshot is an optimized prefix of the same 32768 path, not an independent run or additional RL experience.",
            "Budget effects are paired within physical paths; Actor effects compare the same registered fold, seed and budget.",
            "There is only one development seed and two folds; these descriptive effects do not establish statistical significance.",
            "Categorical KL is against calendar initialization, not PPO old-policy approximate KL and not a holdings-change KL.",
            "Critic error is pre-update versus captured bootstrapped targets; parameter growth and greedy changes are not profit improvements.",
            "Actual GAE is recomputed once per physical path; short buffer views are reused, not counted as new targets or training.",
            "Posthoc analysis does not change selection, admission, qualification or deployment."])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "diagnostics.json", result)
    pd.DataFrame(groups).to_csv(output / "phase_mode.csv", index=False)
    flat_scales = [dict(physical_run_id=name, **{key: value for key, value in item.items() if key != "physical_run_id"}) for name, items in scales.items() for item in items]
    pd.DataFrame(flat_scales).to_csv(output / "rollout_scales.csv", index=False)
    pd.DataFrame(effects).to_csv(output / "paired_effects.csv", index=False)
    print(f'PHASE_BUDGET_DIAGNOSTICS {args.stage} physical={len(physical)} snapshots={len(rows)} actual_RL={result["counters"]["actual_agent_rl_steps"]} output={output}', flush=True)
    return result


if __name__ == "__main__":
    main()
