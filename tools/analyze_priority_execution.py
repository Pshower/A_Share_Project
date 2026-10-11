"""Read sealed 3N traces to describe request, priority projection and real fills."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tools.plot_priority_rank import (
    ROOT, STUDY_ROOT, ACCOUNTS, load_study, read_json, sha256, verify_leaf,
)


def correlation(moment):
    n = moment["n"]
    if n < 2:
        return None
    qvar = moment["qq"] - moment["q"] ** 2 / n
    pvar = moment["pp"] - moment["p"] ** 2 / n
    if qvar <= 1e-14 or pvar <= 1e-14:
        return None
    return float(np.clip((moment["qp"] - moment["q"] * moment["p"] / n) / math.sqrt(qvar * pvar), -1, 1))


def ordinal_percentiles(values):
    """Average tied ordinal ranks / request count, computed within each decision."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    end = np.cumsum(counts)
    ranks = (2 * end - counts + 1) / (2 * len(values))
    return ranks[inverse]


def moments(qty, priority):
    return dict(n=int(len(qty)), q=float(qty.sum()), p=float(priority.sum()),
                qq=float(np.dot(qty, qty)), pp=float(np.dot(priority, priority)), qp=float(np.dot(qty, priority)))


def numeric_stats(parts):
    values = np.concatenate(parts) if parts else np.empty(0)
    if not len(values):
        return dict(n=0, mean=None, std=None, p10=None, median=None, p90=None)
    return dict(n=int(len(values)), mean=float(values.mean()), std=float(values.std(ddof=1)) if len(values) > 1 else None,
                p10=float(np.quantile(values, .1)), median=float(np.median(values)), p90=float(np.quantile(values, .9)))


def ratio(a, b):
    return a / b if b else None


def analyze_trace(trace_directory, directory, marker, identity, inputs, expected_manifest=None, result=None):
    trace_directory, directory = Path(trace_directory), Path(directory)
    index_path = trace_directory / "index.json"
    verify_leaf(index_path, directory, marker, inputs)
    manifest = read_json(index_path, inputs)
    if expected_manifest is not None and manifest != expected_manifest:
        raise ValueError("Trace index does not match its sealed recorded prefix")
    if (manifest.get("schema_version") != 2 or manifest.get("action_layout") != "direction_N_quantity_N_priority_N"
            or manifest["identity"].get("actor_variant") != identity["actor"]):
        raise ValueError("Only original registered 3N priority traces may be analyzed")
    n = len(manifest["identity"]["codes"])
    counts = defaultdict(int)
    pooled = dict(n=0, q=0., p=0., qq=0., pp=0., qp=0.)
    centered = dict(n=0, q=0., p=0., qq=0., pp=0., qp=0.)
    rank_sums = defaultdict(float)
    qty_parts, priority_parts = [], []
    previous = 0
    for chunk in manifest["chunks"]:
        path = (trace_directory / chunk["path"]).resolve()
        if not path.is_relative_to(trace_directory.resolve()) or chunk["first_step"] != previous + 1:
            raise ValueError("Native trace chunk path or step continuity changed")
        verify_leaf(path, directory, marker, inputs)
        if sha256(path) != chunk["sha256"]:
            raise ValueError("Native trace chunk digest differs from its prefix")
        with np.load(path, allow_pickle=False) as values:
            raw = np.asarray(values["raw_packed_action"], dtype=np.float64)
            if raw.shape != (chunk["steps"], 3 * n) or not np.isfinite(raw).all():
                raise ValueError("Original action is not finite 3N")
            directions, qty, priority = np.split(raw, 3, axis=1)
            buy = directions == 1
            legal = values["obs_legal_directions"]
            if (not np.isin(directions, range(4)).all() or not np.array_equal(directions, values["action_directions"])
                    or not np.take_along_axis(legal, directions.astype(np.int64)[..., None], axis=2).all()
                    or np.any(priority[~buy] != 0)
                    or identity["actor"] == "coupled" and np.any(priority[buy] != qty[buy])):
                raise ValueError("Priority samples violate original state-known action support")
            if (not np.array_equal(qty, values["action_raw_quantity_latent"])
                    or not np.array_equal(priority, values["action_raw_priority_latent"])):
                raise ValueError("Original 3N coordinates disagree with explicit trace aliases")
            mapped = values["action_shares"] > 0
            filled = values["execution_filled_shares"] > 0
            top = values["action_top_n_filtered"].astype(bool)
            cash = values["action_cash_filtered"].astype(bool)
            if (np.any(mapped & ~buy) or np.any(filled & ~buy) or np.any(top & cash)
                    or np.any((top | cash) & ~buy)):
                raise ValueError("BUY projection/fill diagnostics have inconsistent supports")
            counts["steps"] += len(raw)
            counts["stock_steps"] += raw.shape[0] * n
            counts["raw_buy_requests"] += int(buy.sum())
            counts["mapped_buys"] += int(mapped.sum())
            counts["filled_buys"] += int(filled.sum())
            counts["top_n_filtered_buys"] += int(top.sum())
            counts["cash_filtered_buys"] += int(cash.sum())
            counts["mapped_buy_without_actual_fill"] += int((mapped & ~filled).sum())
            counts["raw_nonhold_requests"] += int((directions != 0).sum())
            counts["mapped_nonzero_orders"] += int((values["action_shares"] != 0).sum())
            counts["actual_nonzero_fills"] += int((values["execution_filled_shares"] != 0).sum())
            counts["forced_hold_stock_steps"] += int((legal.sum(axis=2) == 1).sum())
            counts["quantity_active_stock_steps"] += int(values["action_size_active"].sum())
            counts["priority_active_stock_steps"] += int(buy.sum())
            counts["actual_full_exits"] += int(((directions == 3) & (values["execution_filled_shares"] < 0)
                                                & (values["after_positions"] == 0)).sum())
            q, p = qty[buy], priority[buy]
            qty_parts.append(q)
            priority_parts.append(p)
            for key, value in moments(q, p).items():
                pooled[key] += value
            for row in range(len(raw)):
                chosen = buy[row]
                size = int(chosen.sum())
                if size < 2:
                    continue
                q, p = qty[row, chosen], priority[row, chosen]
                ranks = ordinal_percentiles(q)
                centered_values = moments(q - q.mean(), p - p.mean())
                for key, value in centered_values.items():
                    centered[key] += value
                counts["decisions_with_competing_buy_requests"] += 1
                for name, mask in (("requested", np.ones(size, dtype=bool)), ("mapped", mapped[row, chosen]),
                                   ("filled", filled[row, chosen]), ("top_filtered", top[row, chosen]),
                                   ("cash_filtered", cash[row, chosen])):
                    count = int(mask.sum())
                    counts[f"rank_{name}_n"] += count
                    rank_sums[f"rank_{name}_sum"] += float(ranks[mask].sum())
                    rank_sums[f"rank_{name}_baseline_sum"] += float(ranks.mean() * count)
            previous = chunk["last_step"]
    if previous != manifest["total_steps"] or counts["steps"] != manifest["total_steps"]:
        raise ValueError("Native diagnostic lacks a complete trajectory")
    selection = {}
    for name in ("requested", "mapped", "filled", "top_filtered", "cash_filtered"):
        count = counts[f"rank_{name}_n"]
        selection[name] = dict(n=count, mean_within_decision_quantity_percentile=ratio(rank_sums[f"rank_{name}_sum"], count),
            matched_requested_baseline=ratio(rank_sums[f"rank_{name}_baseline_sum"], count),
            selection_bias=ratio(rank_sums[f"rank_{name}_sum"] - rank_sums[f"rank_{name}_baseline_sum"], count))
    metrics = result["metrics"] if result is not None else None
    if result is not None:
        daily_path = trace_directory.parent / "daily.csv"
        verify_leaf(daily_path, directory, marker, inputs)
        daily = pd.read_csv(daily_path)
        observed = float(daily.equity.iloc[-1] / daily.equity.iloc[0] - 1)
        if not math.isclose(observed, metrics["cumulative_return"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("Native path curve does not reconcile with its sealed net return")
    return dict(identity=identity, metrics=metrics, counts=dict(counts), rank_sums=dict(rank_sums), pooled_moments=pooled,
        within_decision_centered_moments=centered, quantity_priority_pearson=correlation(pooled),
        within_decision_quantity_priority_correlation=correlation(centered),
        raw_buy_quantity_latent=numeric_stats(qty_parts), raw_buy_priority_latent=numeric_stats(priority_parts),
        quantity_selection_percentiles=selection,
        rates=dict(top_n_filter_per_raw_buy=ratio(counts["top_n_filtered_buys"], counts["raw_buy_requests"]),
                   cash_filter_per_raw_buy=ratio(counts["cash_filtered_buys"], counts["raw_buy_requests"]),
                   mapped_buy_per_raw_buy=ratio(counts["mapped_buys"], counts["raw_buy_requests"]),
                   actual_buy_fill_per_mapped_buy=ratio(counts["filled_buys"], counts["mapped_buys"])))


def aggregate(rows):
    counts, ranks = defaultdict(int), defaultdict(float)
    pooled = dict(n=0, q=0., p=0., qq=0., pp=0., qp=0.)
    centered = dict(n=0, q=0., p=0., qq=0., pp=0., qp=0.)
    for row in rows:
        for key, value in row["counts"].items():
            counts[key] += value
        for key, value in row["rank_sums"].items():
            ranks[key] += value
        for key in pooled:
            pooled[key] += row["pooled_moments"][key]
            centered[key] += row["within_decision_centered_moments"][key]
    selection = {}
    for name in ("requested", "mapped", "filled", "top_filtered", "cash_filtered"):
        n = counts[f"rank_{name}_n"]
        selection[name] = dict(n=n, mean_within_decision_quantity_percentile=ratio(ranks[f"rank_{name}_sum"], n),
            matched_requested_baseline=ratio(ranks[f"rank_{name}_baseline_sum"], n),
            selection_bias=ratio(ranks[f"rank_{name}_sum"] - ranks[f"rank_{name}_baseline_sum"], n))
    return dict(paths=len(rows), counts=dict(counts), quantity_priority_pearson=correlation(pooled),
        within_decision_quantity_priority_correlation=correlation(centered), quantity_selection_percentiles=selection,
        rates=dict(top_n_filter_per_raw_buy=ratio(counts["top_n_filtered_buys"], counts["raw_buy_requests"]),
                   cash_filter_per_raw_buy=ratio(counts["cash_filtered_buys"], counts["raw_buy_requests"]),
                   mapped_buy_per_raw_buy=ratio(counts["mapped_buys"], counts["raw_buy_requests"]),
                   actual_buy_fill_per_mapped_buy=ratio(counts["filled_buys"], counts["mapped_buys"])))


def grouped(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        group = tuple(row["identity"][key] for key in keys)
        groups[group].append(row)
    return [{"group": dict(zip(keys, group)), **aggregate(values)} for group, values in sorted(groups.items())]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(STUDY_ROOT))
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    _, records, summary, markers, inputs = load_study(root)
    trained, training, initialized, rules = [], [], [], []
    seen_training, seen_initial = set(), set()
    for record in records:
        directory = root / "runs" / f"{record['fold']}_{record['actor']}_{record['method']}_{record['seed']}"
        marker = markers[directory.name]
        identity = {key: record[key] for key in ("fold", "actor", "method", "seed", "budget")}
        for account in ACCOUNTS:
            path = directory / record["budget"] / account
            verify_leaf(path / "result.json", directory, marker, inputs)
            result = read_json(path / "result.json", inputs)
            if result["metrics"] != record["metrics"][account]["ppo"] or result["trace"] != record["metrics"][account]["trace"]:
                raise ValueError("Evaluation leaf differs from its sealed record")
            trained.append(analyze_trace(path / "trace", directory, marker, dict(identity, account=account), inputs,
                                         result["trace"], result))
        if record["budget"] == "long" and directory.name not in seen_training:
            seen_training.add(directory.name)
            training.append(analyze_trace(directory / "training_trace", directory, marker,
                dict(identity, account="mixed"), inputs, record["training_trace_prefix"]))
        control = (ROOT / record["initial_controls_path"]).resolve()
        wanted = root / "initial_controls" / f"{record['fold']}_{record['actor']}_{record['seed']}"
        if control != wanted:
            raise ValueError("Initialization control belongs to a different actor/fold/seed")
        if control in seen_initial:
            continue
        seen_initial.add(control)
        initial_marker = read_json(control / "completed.json", inputs)
        for account in ACCOUNTS:
            path = control / account
            verify_leaf(path / "result.json", control, initial_marker, inputs)
            result = read_json(path / "result.json", inputs)
            initialized.append(analyze_trace(path / "trace", control, initial_marker,
                dict(fold=record["fold"], actor=record["actor"], seed=record["seed"], account=account), inputs,
                result["trace"], result))
    for fold in ("2023", "2024"):
        directory = root / "rules" / fold
        marker = read_json(directory / "completed.json", inputs)
        for account in ACCOUNTS:
            for name in ("hold", "momentum_daily", "momentum_20day"):
                path = directory / account / name
                verify_leaf(path / "result.json", directory, marker, inputs)
                result = read_json(path / "result.json", inputs)
                rules.append(analyze_trace(path / "trace", directory, marker,
                    dict(fold=fold, actor="coupled", account=account, rule=name), inputs, result["trace"], result))
    if (len(training), len(trained), len(initialized), len(rules)) != (24, 192, 48, 24):
        raise ValueError("Priority execution diagnostics require the complete registered paths")
    output = dict(schema_version=1, study="priority_rank_execution", no_model_loading=True, no_inference=True,
        additional_environment_steps=0, additional_optimizer_steps=0, gradient_calls=0,
        script_sha256=sha256(__file__), helper_sha256=sha256(ROOT / "tools" / "plot_priority_rank.py"), inputs=inputs,
        aggregation="Exposure-weighted descriptive request/fill counts; paired-seed ROI effects remain in study_summary.json.",
        scope="Original 3N directions, quantities and priorities; training full long prefix once, fixed checkpoint validation and initialized controls.",
        denominators=dict(filters="Filtered BUY requests / all raw BUY requests; topN and cash flags disjoint.",
            fills="Positive actual fills / positive mapped BUY orders; not requested direction count.",
            correlations="Original raw BUY quantity/priorities, before clipping or any selection; null for constant coordinates.",
            quantity_percentile="Within-decision average-tied ordinal quantity rank / request count; only >=2 competing BUY requests.",
            selection_bias="Selected percentile minus that same decision's all-request mean, weighted by selected count.",
            path_quantiles="Exact quantiles within each native path; not a pooled independent-seed estimate."),
        caveats=["Observed selection correlations do not establish causal profitability.",
            "Teacher rules intentionally couple priority and quantity and are not neural model comparisons.",
            "Multiple checkpoint/account paths are not additional independent seeds.",
            "Gross five-day auxiliary labels exclude costs; observed net ROI still includes actual fees/slippage."],
        qualified_candidate=summary["qualified_candidate"],
        training_groups=grouped(training, ("fold", "actor", "method")),
        trained_groups=grouped(trained, ("fold", "actor", "method", "budget")),
        initialized_groups=grouped(initialized, ("fold", "actor")), rule_groups=grouped(rules, ("fold", "rule")),
        training_paths=training, trained_paths=trained, initialized_paths=initialized, rule_paths=rules)
    path = root / "execution_diagnostics.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(path), sha256=sha256(path), training_paths=24, trained_paths=192,
                         initialized_paths=48, rule_paths=24, model_loading=0, env_steps=0)))


if __name__ == "__main__":
    main()
