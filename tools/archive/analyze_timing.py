"""Read sealed timing evidence only; never load/forward a model or replay an env."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import file_hash, read_config, write_json
from src.training.research_timing import check_frozen, _load_stage, select_screen


def stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(count=0, mean=None, minimum=None, maximum=None)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite sealed diagnostic values")
    return dict(count=len(values), mean=float(values.mean()), minimum=float(values.min()), maximum=float(values.max()))


def timing_metrics(frame):
    """Observed probability margins; training rows represent successive old policies."""
    if not len(frame):
        return dict(steps=0)
    raw_phase, raw_action = frame.phase.to_numpy(), frame.macro.to_numpy()
    p = frame.p_rebalance.to_numpy(dtype=float)
    if (not np.isin(raw_action, [0, 1]).all() or not np.isin(raw_phase, np.arange(20)).all()
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise ValueError("Invalid sealed categorical sample")
    phase, action = raw_phase.astype(int), raw_action.astype(int)
    calendar = phase == 0
    # Equal probabilities choose HOLD, matching Categorical argmax coordinate 0.
    greedy = p > .5
    bounded = np.clip(p, 1e-15, 1 - 1e-15)
    gap = np.log(bounded) - np.log1p(-bounded)
    initial_gap = np.where(calendar, 2., -2.)
    excess = np.log1p(frame.net_return.to_numpy()) - np.log1p(frame.benchmark_return.to_numpy())
    return dict(steps=len(frame), macro_rebalances=int(action.sum()), macro_holds=int((action == 0).sum()),
        calendar_steps=int(calendar.sum()), on_calendar_skips=int((calendar & (action == 0)).sum()),
        off_calendar_extra_rebalances=int((~calendar & (action == 1)).sum()),
        greedy_calendar_flips=int((greedy != calendar).sum()), greedy_calendar_flip_rate=float(np.mean(greedy != calendar)),
        near_decision_threshold=int((np.abs(p - .5) <= .05).sum()), saturated_probability_rows=int(((p == 0) | (p == 1)).sum()),
        rebalance_probability=stats(p), logit_gap=stats(gap), logit_gap_delta_from_calendar=stats(gap - initial_gap),
        conditional_daily_log_excess=stats(excess), agent_true_fill_stock_days=int(frame.fills.sum()),
        benchmark_true_fill_stock_days=int(frame.benchmark_fills.sum()),
        agent_fees=float(frame.fees.sum()), benchmark_fees=float(frame.benchmark_fees.sum()),
        agent_slippage=float(frame.slippage.sum()), benchmark_slippage=float(frame.benchmark_slippage.sum()),
        agent_gross_filled_notional=float(frame.gross.sum()), benchmark_gross_filled_notional=float(frame.benchmark_gross.sum()),
        agent_mean_cash_ratio=float(frame.cash_ratio.mean()), benchmark_mean_cash_ratio=float(frame.benchmark_cash_ratio.mean()))


def read_native(directory, manifest):
    frames = []
    for chunk, reference in zip(manifest["chunks"], manifest["benchmark_trace"]["chunks"]):
        with np.load(Path(directory) / chunk["path"], allow_pickle=False) as agent, np.load(Path(directory) / "benchmark" / reference["path"], allow_pickle=False) as benchmark:
            values = dict(phase=agent["action_phase"], macro=agent["action_macro_action"],
                p_rebalance=agent["action_macro_probabilities"][:, 1], mode=agent["initial_mode"],
                net_return=agent["action_raw_net_return"], benchmark_return=agent["action_benchmark_net_return"],
                learning_reward=agent["action_learning_reward"], decision_date=agent["decision_date"])
            for prefix, book in (("", agent), ("benchmark_", benchmark)):
                values[prefix + "fills"] = (book["execution_filled_shares"] != 0).sum(axis=1)
                values[prefix + "fees"] = book["execution_fee"].sum(axis=1)
                values[prefix + "slippage"] = book["execution_slippage_cost"].sum(axis=1)
                values[prefix + "gross"] = np.abs(book["execution_filled_notional"]).sum(axis=1)
                values[prefix + "cash_ratio"] = book["after_cash"] / book["after_equity"]
            frames.append(pd.DataFrame(values))
    return pd.concat(frames, ignore_index=True)


def critic_metrics(directory, buffers):
    values, targets, advantages = [], [], []
    for item in buffers:
        with np.load(Path(directory) / item["path"], allow_pickle=False) as buffer:
            values.extend(buffer["values"].reshape(-1).tolist())
            targets.extend(buffer["returns"].reshape(-1).tolist())
            advantages.extend(buffer["advantages"].reshape(-1).tolist())
    predicted, target = np.asarray(values), np.asarray(targets)
    variance = float(np.var(target))
    return dict(samples=len(target), old_value_mse=float(np.mean((target - predicted) ** 2)),
        old_value_explained_variance=float(1 - np.var(target - predicted) / variance) if variance > 1e-12 else None,
        targets=stats(target), old_values=stats(predicted), advantages=stats(advantages),
        semantics="Old values before PPO update versus stored bootstrapped GAE targets; not post-update Critic fit or realized future ROI.")


def analyze_trial(root, row):
    run_id = f'{row["phase"]}_{row["fold"]}_{row["method"]}_{row["seed"]}'
    directory = Path(root) / "runs" / run_id
    frame = read_native(directory / "training_trace", row["audit"]["training_trace_prefix"])
    group_rows = []
    for (phase, mode), group in frame.groupby(["phase", "mode"], sort=True):
        group_rows.append(dict(run_id=run_id, scope="training", phase=int(phase), mode=mode, **timing_metrics(group)))
    episodes = read_config(directory / "episodes.json")
    for episode in episodes:
        episode["run_id"] = run_id
    updates = pd.read_csv(directory / "training_updates.csv")
    optimizer = row["audit"]["actual_optimizer_updates"]
    nominal = row["steps"] // 512 * 5 * 4
    report = dict(run_id=run_id, phase=row["phase"], fold=row["fold"], method=row["method"], seed=row["seed"],
        sealed_completed_sha256=file_hash(directory / "completed.json"), training=timing_metrics(frame),
        experience=row["audit"]["experience"], actual_optimizer_updates=optimizer, nominal_optimizer_updates=nominal,
        fewer_than_nominal_updates=optimizer < nominal, saved_update_records=len(updates),
        actor_tensor_delta_l2={key: value for key, value in row["audit"]["tensor_delta_l2"].items() if key.startswith("action_net.")},
        critic=critic_metrics(directory, row["audit"]["rollout_buffers"]), evaluations={})
    if "train/approx_kl" in updates:
        report["saved_approx_kl"] = stats(updates["train/approx_kl"].dropna())
    for mode in row["metrics"]:
        result = read_config(directory / "trained" / mode / "result.json")
        sample = read_native(directory / "trained" / mode / "trace", result["trace"])
        report["evaluations"][mode] = dict(**timing_metrics(sample), metrics=result["metrics"],
            benchmark_metrics=result["benchmark_metrics"], paired_net_roi_gap=result["metrics"]["cumulative_return"] - result["benchmark_metrics"]["cumulative_return"])
        for phase, group in sample.groupby("phase", sort=True):
            group_rows.append(dict(run_id=run_id, scope="deterministic_evaluation", phase=int(phase), mode=mode, **timing_metrics(group)))
    return report, group_rows, episodes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reports/runs/ppo_timing_20261014")
    parser.add_argument("--stage", choices=("screen", "confirm", "all"), required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    protocol = read_config(root / "protocol.json")
    if protocol.get("study") != "timing_controller":
        raise ValueError("This tool accepts only the frozen timing study")
    check_frozen(protocol)
    # These helpers verify every leaf, macro/rollout/ledger contract and full
    # validation date array; they do not call model.load, forward or env.step.
    screen = _load_stage(root, protocol, "screen")
    records = screen if args.stage != "confirm" else []
    if args.stage in ("confirm", "all"):
        if not (root / "selection.json").exists():
            raise ValueError("Confirmation diagnostics require an already frozen selection")
        selection = read_config(root / "selection.json")
        expected = select_screen(screen)
        if any(selection.get(key) != value for key, value in expected.items()):
            raise ValueError("Screen selection no longer matches the frozen rule")
        records += _load_stage(root, protocol, "confirm", selection["selected_method"])
    output = Path(args.output or root / "diagnostics" / ("timing_" + args.stage)).resolve()
    if not output.is_relative_to(root / "diagnostics") or output == root / "diagnostics" or output.exists():
        raise ValueError("Diagnostic output must be a new child of root/diagnostics, never a signed trial leaf")
    reports, groups, episodes = [], [], []
    for row in records:
        report, grouped, sampled = analyze_trial(root, row)
        reports.append(report)
        groups.extend(grouped)
        episodes.extend(sampled)
    output.mkdir(parents=True)
    result = dict(schema_version=1, status="complete_for_sealed_stage", stage=args.stage, runs=len(records),
        protocol_sha256=file_hash(root / "protocol.json"), source_git=protocol["git"],
        actual_training_steps=sum(row["steps"] for row in records), actual_optimizer_updates=sum(r["actual_optimizer_updates"] for r in reports),
        no_model_load=True, no_forward=True, no_environment_steps=True, no_gradient_updates=True,
        no_trial_leaf_writes=True, tool_sha256=file_hash(Path(__file__)), trials=reports,
        interpretation=["Training probability rows mix successive old policies and different sampled accounts/dates, not the final policy on fixed states.",
            "Calendar logit gap is +2 at phase zero and -2 otherwise; initial preferred probability is sigmoid(2)=0.880797, not 0.982014.",
            "Conditional action-day log excess is descriptive; holdings persist, so it is not an isolated causal effect of that day's action.",
            "Training episodes reset accounts; do not compound all training rewards as one investable equity curve.",
            "This diagnostic does not select a method, alter a gate or promote a model."])
    write_json(output / "diagnostics.json", result)
    pd.DataFrame(groups).to_csv(output / "phase_mode.csv", index=False)
    pd.DataFrame(episodes).drop(columns=["rewards", "raw_returns", "benchmark_returns", "initial_account"], errors="ignore").to_csv(output / "episode_coverage.csv", index=False)
    print(f'DIAGNOSTICS {args.stage} sealed_runs={len(records)} RL_steps={result["actual_training_steps"]} Adam={result["actual_optimizer_updates"]} output={output}', flush=True)
    return result


if __name__ == "__main__":
    main()
