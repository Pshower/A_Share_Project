"""Cold native-ledger analysis of the sealed train-only paired-label pilot.

No model construction, environment replay, fitting or new market experience.
The results describe overlapping historical Q_rule labels, not predictability.
"""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import research_timing_counterfactual_pilot as driver


FILES = ("analysis.json", "pairs.csv", "strata.csv", "manifest.json")
OWN_FILES = ("tools/analyze_counterfactual_pilot.py", "tests/test_counterfactual_analysis.py")
MONEY_ATOL = 1e-6
REWARD_ATOL = 1e-10
ECONOMIC_FIELDS = ("date", "equity", "cash", "positions", "available_positions", "receivables",
                   "event_cash", "event_shares", "total_fees")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name):
    require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.number))
            and np.isfinite(value), "Invalid finite number: " + name)
    return float(value)


def close(actual, expected, name, atol=MONEY_ATOL):
    require(np.allclose(actual, expected, rtol=0., atol=atol), "Native arithmetic differs: " + name)


def vector(value, n, name, nonnegative=False):
    original = np.asarray(value)
    require(original.dtype.kind in "if", "Nonnumeric native vector: " + name)
    array = original.astype(float)
    require(array.shape == (n,) and np.isfinite(array).all(), "Invalid native vector: " + name)
    require(not nonnegative or (array >= 0).all(), "Negative native vector: " + name)
    return array


def _economic(ledger):
    return {name: ledger[name] for name in ECONOMIC_FIELDS}


def _native_book(branch, ledger_key, codes, broker):
    n, lookup = len(codes), {code: index for index, code in enumerate(codes)}
    start_nav = number(branch["initial_agent_nav" if ledger_key == "agent_ledger" else "initial_benchmark_nav"], "initial NAV")
    require(start_nav > 0, "Initial NAV must be positive")
    previous_nav, initial_fees, previous_ledger = start_nav, None, None
    logs, dates, economics = [], [], []
    totals = dict(fees=0., slippage=0., gross_notional=0., filled_orders=0, rejected_orders=0,
                  partial_orders=0, filled_stock_days=0, first_step_filled_orders=0, turnover=0.)
    for j, step in enumerate(branch["steps"]):
        ledger = step[ledger_key]
        decision, opening = ledger["decision_snapshot"], ledger["opening_snapshot"]
        require(decision["codes"] == opening["codes"] == codes, "Native stock coordinates differ")
        nav, cash = number(ledger["equity"], "NAV"), number(ledger["cash"], "cash")
        require(nav > 0 and cash >= 0, "Native NAV/cash is invalid")
        close(number(decision["equity"], "decision NAV"), previous_nav, "decision NAV continuity")
        positions = vector(decision["positions"], n, "decision positions", True)
        available = vector(decision["available_positions"], n, "decision available", True)
        require((available <= positions).all(), "Decision availability exceeds positions")
        held = positions > 0
        require(len(decision["marks"]) == n, "Decision mark coordinates differ")
        marks = np.asarray([number(decision["marks"][asset], "held decision mark") for asset in np.flatnonzero(held)])
        require((marks > 0).all(), "Held assets have invalid decision marks")
        close(previous_nav, number(decision["cash"], "decision cash")
              + number(decision["receivables"], "decision receivables") + np.dot(positions[held], marks),
              "decision valuation")
        opened = vector(opening["positions"], n, "opening positions", True)
        open_available = vector(opening["available_positions"], n, "opening available", True)
        require((open_available <= opened).all(), "Opening availability exceeds positions")
        close(opened, positions + vector(ledger["event_shares"], n, "event shares"), "share events")
        open_cash = number(opening["cash"], "opening cash")
        close(open_cash, number(decision["cash"], "decision cash") + number(ledger["event_cash"], "event cash"), "cash events")
        if previous_ledger is not None:
            require(decision["date"] == previous_ledger["date"], "Decision date continuity differs")
            close(positions, previous_ledger["positions"], "positions continuity")
            close(number(decision["cash"], "decision cash"), previous_ledger["cash"], "cash continuity")
            close(number(decision["receivables"], "decision receivables"), previous_ledger["receivables"], "receivable continuity")
        fees = slippage = gross = net = 0.
        fills, sales, filled_assets = np.zeros(n), np.zeros(n), set()
        for order in ledger["orders"]:
            code = order["code"]
            require(code in lookup, "Unknown filled stock code")
            asset, quantity = lookup[code], number(order["filled_shares"], "filled shares")
            fee, slip = number(order["fee"], "order fee"), number(order["slippage_cost"], "order slippage")
            require(quantity == math.floor(quantity) and fee >= 0 and slip >= -MONEY_ATOL, "Invalid lot or cost")
            fees += fee
            slippage += slip
            if quantity == 0:
                # A missing-open rejection stores a typed nonfinite price tag.
                close(fee, 0., "unfilled fee")
                close(slip, 0., "unfilled slippage")
                require(order["status"] == "rejected", "Zero fill must be rejected")
                totals["rejected_orders"] += 1
                continue
            price = number(order["price"], "filled execution price")
            raw = number(opening["prices"][asset], "filled raw open")
            require(price > 0 and raw > 0 and order["side"] == ("sell" if quantity < 0 else "buy"), "Invalid filled side/quote")
            amount = abs(quantity) * price
            expected_fee = max(broker["min_commission"], amount * broker["commission"]) + (amount * broker["sell_tax"] if quantity < 0 else 0.)
            close(fee, expected_fee, "broker commission/tax")
            close(slip, quantity * (price - raw), "side-specific native slippage")
            require(order["status"] in {"filled", "partial"}, "A nonzero fill has invalid status")
            fills[asset] += quantity
            if quantity < 0:
                sales[asset] += quantity
            filled_assets.add(asset)
            gross += amount
            net += quantity * price
            totals["filled_orders"] += 1
            totals["partial_orders"] += int(order["status"] == "partial")
            totals["first_step_filled_orders"] += int(j == 0)
        close(ledger["positions"], opened + fills, "filled positions")
        close(ledger["available_positions"], open_available + sales, "T+1 availability")
        close(cash, open_cash - net - fees, "cash including execution costs")
        cumulative = number(ledger["total_fees"], "cumulative fees")
        if initial_fees is None:
            initial_fees = cumulative - fees
            require(initial_fees >= -MONEY_ATOL, "Inferred initial fees are negative")
        close(cumulative, initial_fees + totals["fees"] + fees, "cumulative fee increments")
        totals["fees"] += fees
        totals["slippage"] += slippage
        totals["gross_notional"] += gross
        totals["turnover"] += gross / previous_nav
        totals["filled_stock_days"] += len(filled_assets)
        log_return = math.log(nav) - math.log(previous_nav)
        if "raw_net_return" in ledger:
            close(number(ledger["raw_net_return"], "simple net return"), nav / previous_nav - 1., "simple return", REWARD_ATOL)
        elif ledger_key == "benchmark_ledger":
            close(number(ledger["reward"], "benchmark simple return"), nav / previous_nav - 1., "benchmark return", REWARD_ATOL)
        logs.append(log_return)
        require(opening["date"][:10] == ledger["date"][:10], "Opening and closing dates differ")
        dates.append(ledger["date"])
        economics.append(_economic(ledger))
        previous_nav, previous_ledger = nav, ledger
    require(len(dates) == len(set(dates)) and dates == sorted(dates), "Native dates must advance monotonically")
    close(previous_nav, number(branch["final_agent_nav" if ledger_key == "agent_ledger" else "final_benchmark_nav"], "final NAV"), "terminal NAV")
    totals.update(initial_cumulative_fees=initial_fees, final_nav=previous_nav)
    return np.asarray(logs), totals, dates, economics


def audit_label(label, record, stock_codes, broker):
    """Recalculate one pair from native NAV and orders, without model or engine."""
    codes = list(stock_codes)
    require(len(codes) == len(set(codes)) and codes, "Stock coordinates must be unique")
    for key in ("commission", "min_commission", "sell_tax"):
        require(number(broker[key], key) >= 0, "Invalid registered broker cost")
    target, branches = label["labels"], label["labels"]["branches"]
    require(type(target["horizon"]) is int and target["horizon"] == 20
            and type(target["gamma"]) is float and target["gamma"] == .995, "Unknown estimator registration")
    require(type(record["phase"]) is int and 0 <= record["phase"] < 20, "Invalid phase")
    require(record["decision_date"] < record["endpoint"] <= record["fold"] + "-03-31", "Non-training label dates")
    results = {}
    for first, name in enumerate(("hold", "rebalance")):
        branch = branches[name]
        require(len(branch["steps"]) == 20 and type(branch["actual_agent_steps"]) is int
                and type(branch["actual_benchmark_steps"]) is int
                and branch["actual_agent_steps"] == branch["actual_benchmark_steps"] == 20,
                "Incomplete branch or actual counters")
        require(branch["steps"][0]["agent_ledger"]["decision_snapshot"]["date"][:10] == record["decision_date"],
                "Initial decision date differs")
        for j, step in enumerate(branch["steps"]):
            require(type(step["index"]) is int and step["index"] == j
                    and type(step["phase"]) is int and step["phase"] == (record["phase"] + j) % 20
                    and type(step["macro_action"]) is int and step["macro_action"] == (first if j == 0 else int(step["phase"] == 0))
                    and type(step["terminated"]) is bool and type(step["truncated"]) is bool
                    and (j == 19 or not step["terminated"] and not step["truncated"]), "Calendar or boundary differs")
        agent = _native_book(branch, "agent_ledger", codes, broker)
        shadow = _native_book(branch, "benchmark_ledger", codes, broker)
        require(agent[2] == shadow[2] and agent[2][-1][:10] == record["endpoint"]
                and agent[2][0][:10] > record["decision_date"], "Agent/shadow execution dates differ")
        relative = agent[0] - shadow[0]
        close(relative, [row["relative_log_reward"] for row in branch["steps"]], "relative log rewards", REWARD_ATOL)
        close([row["agent_ledger"]["learning_reward"] for row in branch["steps"]], relative, "learning rewards", REWARD_ATOL)
        results[name] = dict(relative=relative, agent=agent, shadow=shadow)
    hold, rebalance = results["hold"], results["rebalance"]
    require(driver.canonical([s["benchmark_ledger"] for s in branches["hold"]["steps"]])
            == driver.canonical([s["benchmark_ledger"] for s in branches["rebalance"]["steps"]]), "Full shadow paths differ")
    close(branches["hold"]["initial_agent_nav"], branches["rebalance"]["initial_agent_nav"], "paired initial NAV")
    close(hold["agent"][1]["initial_cumulative_fees"], rebalance["agent"][1]["initial_cumulative_fees"], "paired initial cumulative fees")
    delta_steps = rebalance["relative"] - hold["relative"]
    primary = float(np.dot(.995 ** np.arange(20), delta_steps))
    terminal = float(math.log(rebalance["agent"][1]["final_nav"]) - math.log(hold["agent"][1]["final_nav"]))
    for key, actual in (("discounted_relative_delta", primary), ("terminal_log_nav_ratio", terminal),
                        ("undiscounted_relative_delta", float(delta_steps.sum()))):
        close(number(target[key], key), actual, key, REWARD_ATOL)
    close(target["per_step_relative_reward_difference"], delta_steps, "saved reward difference", REWARD_ATOL)
    close(delta_steps.sum(), terminal, "telescoping terminal identity", REWARD_ATOL)
    calendar = rebalance if record["phase"] == 0 else hold
    require(driver.canonical(calendar["agent"][3]) == driver.canonical(calendar["shadow"][3]),
            "Calendar branch differs economically from the fixed collection benchmark")
    equivalent = driver.canonical(hold["agent"][3]) == driver.canonical(rebalance["agent"][3])
    if equivalent:
        close(primary, 0., "economically equivalent pair", REWARD_ATOL)
    left, right = hold["agent"][1], rebalance["agent"][1]
    deviation = -primary if record["phase"] == 0 else primary
    row = {key: record[key] for key in ("fold", "state_id", "episode", "offset", "decision_date", "endpoint", "phase",
                                       "initial_mode", "date_stratum", "phase_bin", "input_sha256", "full_state_sha256")}
    row.update(discounted_relative_delta=primary, terminal_log_nav_ratio=terminal,
        calendar_deviation_delta=deviation, oracle_calendar_advantage=max(deviation, 0.),
        signal="positive" if primary >= .0001 else "negative" if primary <= -.0001 else "neutral",
        first_rebalance_no_fill=right["first_step_filled_orders"] == 0,
        both_whole_horizons_no_fill=left["filled_orders"] == right["filled_orders"] == 0,
        economic_paths_equivalent=equivalent, neutral_label=abs(primary) < .0001)
    for key in ("fees", "slippage", "gross_notional", "turnover", "filled_orders", "filled_stock_days", "rejected_orders", "partial_orders"):
        row.update({"hold_" + key: left[key], "rebalance_" + key: right[key], key + "_delta": right[key] - left[key]})
    row["fee_delta_over_initial_nav"] = row["fees_delta"] / branches["hold"]["initial_agent_nav"]
    row["slippage_delta_over_initial_nav"] = row["slippage_delta"] / branches["hold"]["initial_agent_nav"]
    return row


def describe(rows):
    values = [row["discounted_relative_delta"] for row in rows]
    deviations = [row["calendar_deviation_delta"] for row in rows]
    windows = Counter((row["decision_date"], row["endpoint"]) for row in rows)
    return dict(states=len(rows), episodes=len({(row["fold"], row["episode"]) for row in rows}),
        mean=statistics.mean(values) if values else None, median=statistics.median(values) if values else None,
        quantiles={str(q): float(np.quantile(values, q)) if values else None for q in (0., .25, .5, .75, 1.)},
        positive=sum(value >= .0001 for value in values), negative=sum(value <= -.0001 for value in values),
        neutral=sum(abs(value) < .0001 for value in values),
        calendar_deviation_mean=statistics.mean(deviations) if rows else None,
        oracle_mean=statistics.mean(row["oracle_calendar_advantage"] for row in rows) if rows else None,
        oracle_signal_count=sum(value >= .0001 for value in deviations),
        first_rebalance_no_fill=sum(row["first_rebalance_no_fill"] for row in rows),
        both_whole_horizons_no_fill=sum(row["both_whole_horizons_no_fill"] for row in rows),
        economic_paths_equivalent=sum(row["economic_paths_equivalent"] for row in rows),
        mean_fee_delta=statistics.mean(row["fees_delta"] for row in rows) if rows else None,
        mean_slippage_delta=statistics.mean(row["slippage_delta"] for row in rows) if rows else None,
        unique_date_windows=len(windows), repeated_date_windows=sum(count - 1 for count in windows.values()),
        duplicated_inputs=sum(count - 1 for count in Counter(row["input_sha256"] for row in rows).values()),
        overlapping_interval_pairs=sum(a["decision_date"] <= b["endpoint"] and b["decision_date"] <= a["endpoint"]
            for i, a in enumerate(rows) for b in rows[i + 1:]))


def strata(rows):
    groups = defaultdict(list)
    for row in rows:
        for key in ("phase", "phase_bin", "initial_mode", "date_stratum", "offset"):
            groups[(row["fold"], key, str(row[key]))].append(row)
        groups[(row["fold"], "due_status", "phase0" if row["phase"] == 0 else "non0")].append(row)
    for fold in {row["fold"] for row in rows}:
        for value in ("phase0", "non0"):
            groups.setdefault((fold, "due_status", value), [])
    return [dict(fold=fold, group=key, value=value, **describe(group))
            for (fold, key, value), group in sorted(groups.items())]


def _leaf(root, fold, name):
    logical = PurePosixPath(name)
    require(logical.as_posix() == name and not logical.is_absolute() and ".." not in logical.parts
            and "\\" not in name and ":" not in name, "Noncanonical analysis input path")
    directory = root / "folds" / fold
    path = directory / name
    require(path.resolve().is_relative_to(directory.resolve()), "Analysis input escapes its sealed fold")
    return path


def _verify_hashes(files):
    for path, sha in files.items():
        require(driver.file_hash(path) == sha, "Analysis input changed: " + str(path))


def build_analysis(root):
    root = driver.output_root(root)
    protocol = driver.read_json(root / "protocol.json")
    driver.check_frozen(protocol, root)
    require({path.name for path in (root / "folds").iterdir()} == set(driver.FOLDS), "Only two sealed training folds may be analyzed")
    inputs, rows, counts = {root / "protocol.json": driver.file_hash(root / "protocol.json")}, [], Counter()
    for fold in driver.FOLDS:
        registered = driver.cached_fold(root, protocol, fold)
        require(registered is not None and len(registered) == 96, "Both 96-label folds must be sealed")
        directory = root / "folds" / fold
        completed = driver.read_json(directory / "completed.json")
        inputs[directory / "completed.json"] = driver.file_hash(directory / "completed.json")
        for name, sha in completed["files"].items():
            inputs[_leaf(root, fold, name)] = sha
        require(completed["counters"] == driver.expected_counts(), "Actual sealed counters differ")
        counts.update(completed["counters"])
        plan = driver.read_json(directory / "sampling_plan.json")
        contract = protocol["folds"][fold]["data_identity"]["contract"]
        for record in plan["states"]:
            label = json.loads(gzip.decompress((directory / "labels" / (record["state_id"] + ".json.gz")).read_bytes()))
            rows.append(audit_label(label, dict(record, fold=fold), contract["stock_codes"], contract["broker"]))
    require(len(rows) == 192 and counts["completed_pairs"] == 192
            and counts["collection_agent_steps"] + counts["branch_agent_steps"] == 19200
            and counts["collection_benchmark_steps"] + counts["branch_benchmark_steps"] == 19200,
            "Cold analysis experience scope differs")
    _verify_hashes(inputs)
    summary = dict(schema_version=1, status="passed_cold_native_analysis", counters=dict(counts),
        total_agent_steps=19200, total_benchmark_steps=19200, total_book_steps=38400,
        folds={fold: describe([row for row in rows if row["fold"] == fold]) for fold in driver.FOLDS},
        pooled_descriptive_only=describe(rows), new_environment_steps=0, new_nn_forwards=0,
        new_gradient_updates=0, model_fits=0, actionable=False, qualified_candidate=None,
        audit_scope="independent native NAV/reward/cost/share arithmetic; identity checks reused from frozen driver; no fresh quote replay",
        limitations=["192 overlapping historical states are not 192 independent market samples",
            "Oracle is conditional on known historical futures and fixed rule continuation, not predictability or an executable equity curve",
            "Labels use discounted log differences, not simple ROI percentage points",
            "HFQ/static cohort research proxy; no validation, protected-period or confirmation evaluation",
            "Outer-train scaler does not certify inner chronological causality; no strategy fitting performed",
            "Initial cumulative fees inferred from first native cumulative total minus first-day fees; final close NAV taken from saved native book"])
    return root, summary, rows, strata(rows), inputs


def _write_json(path, value):
    with path.open("xb") as stream:
        stream.write(driver.canonical(value) + b"\n")


def _write_csv(path, rows):
    keys = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: driver.canonical(value).decode() if isinstance(value, (dict, list)) else value for key, value in row.items()})


def run(root=driver.OUTPUT):
    root = driver.output_root(root)
    parent = root / "diagnostics"
    output = parent / "cold_analysis"
    require(parent.resolve() == parent and output.resolve() == output and not output.exists(),
            "Cold analysis destination is aliased or already exists")
    root, summary, rows, groups, inputs = build_analysis(root)
    for name in OWN_FILES:
        inputs[ROOT / name] = driver.file_hash(ROOT / name)
    _verify_hashes(inputs)
    parent.mkdir(exist_ok=True)
    require(parent.resolve() == parent, "Diagnostics directory became aliased")
    output.mkdir(exist_ok=False)
    _write_json(output / "analysis.json", summary)
    _write_csv(output / "pairs.csv", rows)
    _write_csv(output / "strata.csv", groups)
    _verify_hashes(inputs)
    manifest = dict(schema_version=1, status="complete", protocol_sha256=inputs[root / "protocol.json"],
        input_files_sha256={str(path): sha for path, sha in sorted(inputs.items(), key=lambda pair: str(pair[0]))},
        outputs_sha256={name: driver.file_hash(output / name) for name in FILES if name != "manifest.json"},
        native_label_count=192, no_new_environment_steps=True, no_new_neural_calls=True,
        no_trial_leaf_writes=True, no_fitting=True)
    _write_json(output / "manifest.json", manifest)
    _verify_hashes(inputs)
    for name, sha in manifest["outputs_sha256"].items():
        require(driver.file_hash(output / name) == sha, "Analysis output changed during completion")
    print("COUNTERFACTUAL_COLD_ANALYSIS labels=192 environment_steps=0 fitting=0 output=" + str(output), flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=driver.OUTPUT, help="Registered pilot root, not an arbitrary destination")
    args = parser.parse_args(argv)
    run(args.output)


if __name__ == "__main__":
    main()
