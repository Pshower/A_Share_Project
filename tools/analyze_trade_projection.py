"""Analyze sealed trade traces without loading models or stepping environments."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACCOUNTS = ("cash", "equal", "concentrated", "random")
DIRECTIONS = ("hold", "buy", "reduce", "exit")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path, inputs):
    inputs[str(path.relative_to(ROOT))] = sha256(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def marked_notional(shares, marks):
    active = shares != 0
    if np.any(active & (~np.isfinite(marks) | (marks <= 0))):
        raise ValueError("Nonzero trade has no finite positive decision mark")
    return np.abs(shares) * np.where(active, marks, 0.)


def analyze_path(path, identity, inputs):
    result = load_json(path / "result.json", inputs)
    index = load_json(path / "trace" / "index.json", inputs)
    counts, sums, reasons = Counter(), Counter(), Counter()
    for chunk in index["chunks"]:
        chunk_path = path / "trace" / chunk["path"]
        digest = sha256(chunk_path)
        inputs[str(chunk_path.relative_to(ROOT))] = digest
        if digest != chunk["sha256"]:
            raise ValueError("Sealed trace hash changed: " + str(chunk_path))
        with np.load(chunk_path, allow_pickle=False) as z:
            directions = z["action_directions"]
            legal = z["action_legal_directions"]
            held = z["decision_positions"] > 0
            valid = z["obs_valid_mask"].astype(bool)
            buy, active = directions == 1, z["action_size_active"]
            top, cash = z["action_top_n_filtered"], z["action_cash_filtered"]
            proposed, mapped = z["action_proposed_shares"], z["action_shares"]
            filled = z["execution_filled_shares"]
            if np.any(top & cash) or np.any((top | cash) & ~buy):
                raise ValueError("Unexpected overlapping/non-BUY projection filters")
            counts.update(steps=len(directions), stock_steps=directions.size,
                          valid_stock_steps=int(valid.sum()), held_stock_steps=int(held.sum()),
                          held_hold=int(((directions == 0) & held).sum()),
                          voluntary_hold=int(((directions == 0) & (legal.sum(-1) > 1)).sum()),
                          forced_hold=int(((directions == 0) & (legal.sum(-1) == 1)).sum()),
                          top_filtered=int(top.sum()), cash_filtered=int(cash.sum()),
                          requested_orders=int((directions != 0).sum()),
                          mapped_orders=int((mapped != 0).sum()),
                          filled_orders=int((filled != 0).sum()),
                          buy_fills=int((filled > 0).sum()), sell_fills=int((filled < 0).sum()),
                          broker_zero_fills=int(((mapped != 0) & (filled == 0)).sum()),
                          broker_quantity_changed=int(((mapped != 0) & (filled != mapped)).sum()),
                          size_active_stock_steps=int(active.sum()))
            for direction, name in enumerate(DIRECTIONS):
                counts[name] += int((directions == direction).sum())
            marks = z["decision_marks"]
            proposed_value = marked_notional(proposed, marks)
            mapped_value = marked_notional(mapped, marks)
            filled_value = marked_notional(filled, marks)
            sums.update(proposed_notional=float(proposed_value.sum()),
                        mapped_notional=float(mapped_value.sum()),
                        filled_decision_notional=float(filled_value.sum()),
                        buy_proposed_notional=float(proposed_value[buy].sum()),
                        buy_mapped_notional=float(mapped_value[buy].sum()),
                        buy_filled_decision_notional=float(filled_value[buy].sum()))
            mean = z["action_size_mean"]
            sigma = np.broadcast_to(z["action_size_sigma"][:, None], mean.shape)
            sums.update(size_abs_mean=float(np.abs(mean[active]).sum()),
                        size_abs_mean_over_sigma=float(np.abs(mean[active] / sigma[active]).sum()),
                        size_sigma=float(sigma[active].sum()),
                        size_midpoint_fraction=float((1 / (1 + np.exp(-mean[active]))).sum()))
            probabilities = z["action_direction_probabilities"]
            sorted_probability = np.sort(probabilities, axis=-1)
            margin = sorted_probability[..., -1] - sorted_probability[..., -2]
            choice = valid & (legal.sum(-1) > 1)
            counts["direction_choice_stock_steps"] += int(choice.sum())
            counts["direction_margin_below_0_01"] += int((margin[choice] < .01).sum())
            sums["direction_margin"] += float(margin[choice].sum())
            for value in z["execution_reasons_json"].flat:
                reasons.update(json.loads(str(value)))
    if counts["steps"] != index["total_steps"]:
        raise ValueError("Trace total does not match saved index")
    if counts["requested_orders"] - counts["top_filtered"] - counts["cash_filtered"] != counts["mapped_orders"]:
        raise ValueError("Request/projection count conservation failed")
    metrics = result["metrics"]
    daily_path = path / "daily.csv"
    inputs[str(daily_path.relative_to(ROOT))] = sha256(daily_path)
    with daily_path.open(encoding="utf-8", newline="") as stream:
        daily = list(csv.DictReader(stream))
    if len(daily) != counts["steps"] + 1:
        raise ValueError("Daily series does not match trace length")
    return dict(identity, counts=dict(counts), sums=dict(sums), broker_reason_counts=dict(reasons),
                metrics={key: float(metrics[key]) for key in (
                    "cumulative_return", "max_drawdown", "total_fees", "total_slippage_cost",
                    "mean_daily_turnover", "mean_cash_ratio")},
                mean_held_names=float(np.mean([int(row["holding_count"]) for row in daily[1:]])),
                daily_unfilled_order_events=sum(int(row["unfilled_orders"]) for row in daily[1:]))


def aggregate(rows):
    counts, sums, reasons = Counter(), Counter(), Counter()
    for row in rows:
        counts.update(row["counts"])
        sums.update(row["sums"])
        reasons.update(row["broker_reason_counts"])
    metrics = {key: float(np.mean([row["metrics"][key] for row in rows])) for key in rows[0]["metrics"]}
    buy, steps, active = counts["buy"], counts["steps"], counts["size_active_stock_steps"]
    result = dict(paths=len(rows), counts=dict(counts), sums=dict(sums),
                  broker_reason_counts=dict(reasons), mean_path_metrics=metrics,
                  mean_path_cost=metrics["total_fees"] + metrics["total_slippage_cost"],
                  mean_held_names=float(np.mean([row["mean_held_names"] for row in rows])),
                  daily_unfilled_order_events=sum(row["daily_unfilled_order_events"] for row in rows))
    result["ratios"] = dict(
        directions_per_step={name: ratio(counts[name], steps) for name in DIRECTIONS},
        filled_orders_per_step=ratio(counts["filled_orders"], steps),
        buy_fills_per_step=ratio(counts["buy_fills"], steps),
        sell_fills_per_step=ratio(counts["sell_fills"], steps),
        held_hold_fraction=ratio(counts["held_hold"], counts["held_stock_steps"]),
        top_filtered_per_buy=ratio(counts["top_filtered"], buy),
        cash_filtered_per_buy=ratio(counts["cash_filtered"], buy),
        mapped_orders_per_request=ratio(counts["mapped_orders"], counts["requested_orders"]),
        nonzero_fills_per_mapped_order=ratio(counts["filled_orders"], counts["mapped_orders"]),
        buy_mapped_proposed_decision_notional=ratio(sums["buy_mapped_notional"], sums["buy_proposed_notional"]),
        filled_mapped_decision_notional=ratio(sums["filled_decision_notional"], sums["mapped_notional"]),
        size_abs_mean=ratio(sums["size_abs_mean"], active),
        size_abs_mean_over_sigma=ratio(sums["size_abs_mean_over_sigma"], active),
        size_sigma=ratio(sums["size_sigma"], active),
        size_midpoint_fraction=ratio(sums["size_midpoint_fraction"], active),
        direction_top2_probability_margin=ratio(sums["direction_margin"], counts["direction_choice_stock_steps"]),
        direction_margin_below_0_01=ratio(counts["direction_margin_below_0_01"], counts["direction_choice_stock_steps"]),
    )
    return result


def grouped(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    return [dict(zip(fields, key), **aggregate(value)) for key, value in sorted(groups.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="reports/runs/ppo_attribution_20261007/summary.json")
    parser.add_argument("--output", default="reports/runs/ppo_market_summary_20261007_diagnostics/projection.json")
    args = parser.parse_args()
    summary_path, output = (ROOT / args.summary).resolve(), (ROOT / args.output).resolve()
    if not summary_path.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        raise ValueError("Inputs and outputs must remain inside this repository")
    if output.is_relative_to(summary_path.parent):
        raise ValueError("Diagnostics must not overwrite or extend the sealed input study")
    inputs = {}
    records = load_json(summary_path, inputs)
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty sealed snapshot record list")
    trained, initialized, training_audits, seen_initial = [], [], [], set()
    for record in records:
        model_path = (ROOT / record["model_path"]).resolve()
        if not model_path.is_relative_to(summary_path.parent):
            raise ValueError("Model path falls outside the sealed study")
        directory = model_path.parent
        identity = {key: record[key] for key in ("fold", "cell", "budget", "seed")}
        for account in ACCOUNTS:
            trained.append(analyze_path(directory / record["budget"] / account,
                                        dict(identity, account=account), inputs))
        audit_path = directory / (record["budget"] + "_audit.json")
        audit = load_json(audit_path, inputs)
        training_audits.append(dict(identity, actual_timesteps=audit["actual_timesteps"],
                                   training_diagnostics=audit["training_diagnostics"],
                                   mode_statistics=audit["mode_statistics"]))
        initial_path = (ROOT / record["initial_controls_path"]).resolve()
        if initial_path in seen_initial:
            continue
        if not initial_path.is_relative_to(summary_path.parent):
            raise ValueError("Initialization path falls outside the sealed study")
        seen_initial.add(initial_path)
        route = record["cell"].split("_")[0]
        for account in ACCOUNTS:
            initialized.append(analyze_path(initial_path / account,
                dict(fold=record["fold"], route=route, seed=record["seed"], account=account), inputs))
    result = dict(schema_version=1, scope="read_only_sealed_trade_projection_diagnostics",
        no_model_loading=True, no_inference=True, no_environment_steps=True,
        no_gradient_computation=True, extra_optimizer_steps=0, script_sha256=sha256(Path(__file__)),
        inputs=inputs,
        denominators={
            "directions_per_step": "all stock direction codes divided by decision steps (forced HOLD included)",
            "top_filtered_per_buy": "top-N filtered raw BUY requests divided by all raw BUY requests",
            "cash_filtered_per_buy": "cash filtered raw BUY requests divided by all raw BUY requests; disjoint from top-N",
            "mapped_orders_per_request": "nonzero mapped stock orders divided by raw non-HOLD direction requests",
            "nonzero_fills_per_mapped_order": "nonzero actual filled stock quantities divided by nonzero mapped orders",
            "size_statistics": "all raw size-active stock-steps; includes orders later filtered; sigma from saved trace",
            "direction_margin": "valid stock-steps with more than one legal direction; excludes forced HOLD",
            "notionals": "absolute quantities times decision-date marks, consistent price basis before/after projection",
            "held_hold_fraction": "HOLD choices on held stocks divided by all held stock-steps",
            "mean_path_metrics": "equal-weight path mean (four accounts within each seed; three seeds per fold/cell/budget)",
            "broker_reasons": "reason occurrences, not mutually exclusive order counts",
            "daily_unfilled_order_events": "daily broker unfilled counts, including partial fills; not just zero fills",
        },
        caveats=[
            "Descriptive diagnostics do not establish that any actor redesign or pooling change improves net ROI.",
            "Fees plus slippage are observed ledger costs, not a causal zero-cost counterfactual return.",
            "Initial positions exceeding top-N are grandfathered; held-name counts above K are not violations.",
            "HFQ research prices and uniform lots are not real execution quotes or board-specific share rules.",
            "Short and long training audit entries are prefix/final snapshots, not additional training runs.",
            "Initialized controls are deduplicated across critic flags; no new initialization trajectories are run.",
        ],
        trained_total=aggregate(trained),
        trained_groups=grouped(trained, ("fold", "cell", "budget")),
        trained_account_groups=grouped(trained, ("fold", "cell", "budget", "account")),
        initialized_groups=grouped(initialized, ("fold", "route")),
        trained_paths=trained, initialized_paths=initialized, training_audits=training_audits)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output.relative_to(ROOT)), sha256=sha256(output),
                         input_files=len(inputs), trained_paths=len(trained),
                         initialized_paths=len(initialized), trained_total=result["trained_total"]), indent=2))


if __name__ == "__main__":
    main()
