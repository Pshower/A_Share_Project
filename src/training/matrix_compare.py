"""Compare complete prior/random PPO matrices using registered local evidence only."""

import argparse
from copy import deepcopy
import csv
from datetime import date
import hashlib
import json
import math
from pathlib import Path

from .matrix_report import ACCOUNTS, BUDGETS, SIZES, _stats, summarize_matrix


TOLERANCE = 1e-10
MAX_DRAWDOWN = .10
MODES = ("prior", "random")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def _equal_values(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_equal_values(left[key], right[key]) for key in left)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=0, abs_tol=TOLERANCE)
    return left == right


def _record_key(record):
    return record["size"], record["budget"], record["seed"]


def _validate_protocols(protocols, summaries):
    for mode in MODES:
        protocol, summary = protocols[mode], summaries[mode]
        if protocol.get("initialization_mode", "prior") != mode:
            raise ValueError("Protocol initialization mode mismatch: " + mode)
        if protocol.get("test_evaluated") is not False:
            raise ValueError("Comparison requires development validation with test_evaluated=false")
        if protocol.get("engineering_smoke", False):
            raise ValueError("Engineering smoke results cannot enter the formal comparison")
        if sorted(protocol.get("seeds", [])) != summary["seeds"] or len(summary["seeds"]) != 3:
            raise ValueError("Comparison requires the same three registered training seeds")
        if set(protocol.get("evaluation_modes", [])) != set(ACCOUNTS):
            raise ValueError("Protocol must register all four evaluation account modes")
        for cell in summary["cells"].values():
            if cell["steps"] != protocol[cell["budget"] + "_steps"]:
                raise ValueError("Checkpoint steps differ from registered budget")
    prior, random = (protocols[mode] for mode in MODES)
    for key in ("seeds", "short_steps", "long_steps", "train_end", "val_end",
                "evaluation_seed", "evaluation_modes", "checkpoint_selection"):
        if prior.get(key) != random.get(key):
            raise ValueError("Protocol comparison mismatch: " + key)
    old_core, new_core = deepcopy(prior["core_contract"]), deepcopy(random["core_contract"])
    if not isinstance(old_core.pop("prior", None), dict) or new_core.pop("prior", None) is not None:
        raise ValueError("Prior contract must contain a prior; random contract must omit it")
    if old_core != new_core:
        raise ValueError("Core data, scaler, feature or execution contract differs beyond prior")
    old_base, new_base = deepcopy(prior["base_config"]), deepcopy(random["base_config"])
    if old_base.get("prior_feature_index") is None or "prior_feature_index" not in new_base or new_base["prior_feature_index"] is not None:
        raise ValueError("Random base_config must explicitly set prior_feature_index=null")
    for base, mode in [(old_base, "prior"), (new_base, "random")]:
        if base.get("initialization_mode", mode) != mode:
            raise ValueError("Base configuration initialization mode mismatch")
        for key in ("prior_feature_index", "output_root", "initialization_mode"):
            base.pop(key, None)
    if old_base != new_base:
        raise ValueError("Base hyperparameters or exploration noise differ beyond prior mode")
    for key, cell in summaries["prior"]["cells"].items():
        other = summaries["random"]["cells"][key]
        if any(cell[field] != other[field] for field in ("seeds", "steps", "hidden_sizes", "parameters")):
            raise ValueError("Matched matrix cells differ in seeds, budget or capacity: " + key)


def _representative(cell, records):
    target = cell["metrics"]["cumulative_return"]["mean"]
    selected = min(cell["per_seed"], key=lambda row: (abs(row["metrics"]["cumulative_return"] - target), row["seed"]))
    record = next(row for row in records if _record_key(row) == (cell["size"], cell["budget"], selected["seed"]))
    return dict(seed=selected["seed"], four_account_return=selected["metrics"]["cumulative_return"],
                model_path=record.get("model_path"), model_sha256=record.get("model_sha256"),
                rule="seed closest to cell mean return; median representative, not highest-return seed")


def summarize_comparison(prior_records, random_records, prior_protocol, random_protocol):
    records = dict(prior=prior_records, random=random_records)
    protocols = dict(prior=prior_protocol, random=random_protocol)
    summaries = {mode: summarize_matrix(records[mode]) for mode in MODES}
    _validate_protocols(protocols, summaries)
    indexed = {mode: {_record_key(row): row for row in records[mode]} for mode in MODES}
    baseline_reference = prior_records[0]["metrics"]
    for mode in MODES:
        for row in records[mode]:
            if row.get("initialization_mode", "prior") != mode or row.get("test_evaluated", False):
                raise ValueError("Record mode or evaluation split mismatch")
            for account in ACCOUNTS:
                if not _equal_values(row["metrics"][account]["baselines"], baseline_reference[account]["baselines"]):
                    raise ValueError("Baseline values differ across fixed evaluation controls")
    for key, prior in indexed["prior"].items():
        random = indexed["random"][key]
        for account in ACCOUNTS:
            if not _equal_values(prior["metrics"][account]["baselines"], random["metrics"][account]["baselines"]):
                raise ValueError(f"Baseline values differ at {key}/{account}")
    cells, effects = {}, {}
    for mode in MODES:
        for name, original in summaries[mode]["cells"].items():
            cell = deepcopy(original)
            rows = [indexed[mode][cell["size"], cell["budget"], seed] for seed in cell["seeds"]]
            drawdowns = [row["metrics"][account]["ppo"].get("max_drawdown") for row in rows for account in ACCOUNTS]
            eligible = all(value is not None and abs(value) <= MAX_DRAWDOWN for value in drawdowns)
            reasons = [] if eligible else ["missing_or_over_10_percent_account_drawdown"]
            cell.update(initialization_mode=mode, risk_eligible=eligible, risk_reasons=reasons,
                        beats_daily_momentum=cell["return_gaps"]["momentum_daily"]["mean"] > 0,
                        beats_20day_momentum=cell["return_gaps"]["momentum"]["mean"] > 0,
                        representative=_representative(cell, records[mode]))
            cost_ratios = []
            capital = protocols[mode]["core_contract"]["initial_capital"]
            for row in rows:
                costs = [scenario["ppo"].get("total_fees") for scenario in row["metrics"].values()]
                slippage = [scenario["ppo"].get("total_slippage_cost") for scenario in row["metrics"].values()]
                cost_ratios.append(sum(costs + slippage) / (len(ACCOUNTS) * capital)
                                   if all(value is not None for value in costs + slippage) else None)
            cell["cost_ratio"] = _stats(cost_ratios)
            for account in ACCOUNTS:
                scenarios = [row["metrics"][account] for row in rows]
                cell["per_account"][account]["return_gaps"] = {
                    baseline: _stats(item["ppo"]["cumulative_return"] -
                        (item["initialized"] if baseline == "initialized" else item["baselines"][baseline])["cumulative_return"]
                        for item in scenarios)
                    for baseline in ("initialized", "momentum_daily", "momentum")
                }
            cells[mode + "_" + name] = cell
    for size in SIZES:
        for budget in BUDGETS:
            name = size + "_" + budget
            paired = []
            for seed in summaries["prior"]["seeds"]:
                old, new = indexed["prior"][size, budget, seed], indexed["random"][size, budget, seed]
                accounts = {}
                for account in ACCOUNTS:
                    before, after = old["metrics"][account], new["metrics"][account]
                    accounts[account] = dict(return_gap=after["ppo"]["cumulative_return"] - before["ppo"]["cumulative_return"],
                        learning_increment_gap=(after["ppo"]["cumulative_return"] - after["initialized"]["cumulative_return"])
                            - (before["ppo"]["cumulative_return"] - before["initialized"]["cumulative_return"]))
                paired.append(dict(seed=seed, accounts=accounts,
                    return_gap=sum(item["return_gap"] for item in accounts.values()) / len(ACCOUNTS),
                    learning_increment_gap=sum(item["learning_increment_gap"] for item in accounts.values()) / len(ACCOUNTS)))
            effects[name] = dict(return_gap=_stats(row["return_gap"] for row in paired),
                learning_increment_gap=_stats(row["learning_increment_gap"] for row in paired), per_seed=paired,
                per_account={account: {metric: _stats(row["accounts"][account][metric] for row in paired)
                    for metric in ("return_gap", "learning_increment_gap")} for account in ACCOUNTS})
            cell = cells["random_" + name]
            positive = sum(row["return_gaps"]["initialized"] > 0 for row in cell["per_seed"])
            minimum = math.ceil(2 * len(cell["seeds"]) / 3)
            gates = dict(all_accounts_within_drawdown=cell["risk_eligible"],
                mean_return_above_matched_prior=effects[name]["return_gap"]["mean"] > 0,
                mean_learning_increment_positive=cell["return_gaps"]["initialized"]["mean"] > 0,
                at_least_two_thirds_seeds_learn=positive >= minimum)
            cell.update(random_gate=gates, positive_learning_seeds=positive, required_positive_learning_seeds=minimum,
                        qualified_random=all(gates.values()))
    def choose(names):
        if not names:
            return None
        name = min(names, key=lambda key: (-cells[key]["metrics"]["cumulative_return"]["mean"],
                    cells[key]["worst_account_drawdown_magnitude"]["max"],
                    sum(cells[key]["metrics"].get(metric, {}).get("mean") or 0.
                        for metric in ("total_fees", "total_slippage_cost")),
                    cells[key]["steps"], cells[key]["parameters"], key))
        cell = cells[name]
        return dict(cell=name, initialization_mode=cell["initialization_mode"],
                    mean_return=cell["metrics"]["cumulative_return"]["mean"],
                    std_return=cell["metrics"]["cumulative_return"]["std"], representative=cell["representative"],
                    beats_daily_momentum=cell["beats_daily_momentum"], beats_20day_momentum=cell["beats_20day_momentum"])
    changes = {name: dict(prior=prior_protocol.get("source_hashes", {}).get(name),
                          random=random_protocol.get("source_hashes", {}).get(name))
               for name in sorted(set(prior_protocol.get("source_hashes", {})) | set(random_protocol.get("source_hashes", {})))
               if prior_protocol.get("source_hashes", {}).get(name) != random_protocol.get("source_hashes", {}).get(name)}
    control_cell = summaries["prior"]["cells"]["small_short"]
    controls = {}
    for strategy in ("momentum_daily", "momentum"):
        metrics = control_cell["baselines"][strategy]
        drawdowns = [scenario["baselines"][strategy].get("max_drawdown")
                     for scenario in baseline_reference.values()]
        controls[strategy] = dict(metrics=metrics,
            worst_account_drawdown_magnitude=max(abs(value) for value in drawdowns)
                if all(value is not None for value in drawdowns) else None,
            cost_ratio=(metrics["total_fees"]["mean"] + metrics["total_slippage_cost"]["mean"])
                / prior_protocol["core_contract"]["initial_capital"]
                if all(metrics.get(name, {}).get("mean") is not None for name in ("total_fees", "total_slippage_cost")) else None)
    return dict(schema_version=1, seeds=summaries["prior"]["seeds"], account_scenarios=list(ACCOUNTS),
        matrices=summaries, cells=cells, controls=controls,
        paired_random_minus_prior=effects, source_hash_differences=changes,
        selection=dict(best_observed_ppo=choose([name for name, cell in cells.items() if cell["risk_eligible"]]),
            qualified_random_candidate=choose([name for name, cell in cells.items() if cell.get("qualified_random")]),
            rule="Risk-eligible three-seed four-account cell mean; exact ties use lower worst drawdown, costs, steps, parameters, then key",
            max_account_drawdown=MAX_DRAWDOWN, cost_gate=None,
            cost_policy="Fees and slippage are included in net returns; costs are disclosed, not deducted again"),
        evaluation=dict(train_end=prior_protocol["train_end"], val_end=prior_protocol["val_end"],
            validation_already_exposed=bool(prior_protocol.get("validation_already_exposed")
                or random_protocol.get("validation_already_exposed")), test_evaluated=False,
            test_protected_from_prior_exposure=None,
            verification_note="Both matrices exclude final-test evaluation; untouched-test provenance is not verified by this report.",
            note="Development window was already exposed. No final test is run here; test_evaluated=false does not prove the later window was never inspected."),
        actionable=False, global_optimality_claim=False,
        aggregation="Four-account means within training seed, then sample standard deviation across three seeds; random-prior differences are paired within seed/account")


def _safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Registered artifact escapes its root")
    return path


def _verify_files(folder, marker, verified):
    files = marker.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Missing registered file hashes: " + str(folder))
    for name, digest in files.items():
        artifact = _safe_path(folder, name)
        if not artifact.is_file() or _file_hash(artifact) != digest:
            raise ValueError("Registered artifact hash mismatch: " + str(artifact))
        verified.add(str(artifact))
    return files


def _curve(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    dates = [date.fromisoformat(row["date"][:10]).isoformat() for row in rows]
    values = [float(row["net_value"]) for row in rows]
    if len(dates) < 2 or dates != sorted(set(dates)) or not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("Invalid registered daily curve: " + str(path))
    return dates, values


def load_matrix(root, mode):
    root = Path(root).resolve()
    protocol = _read(root / "protocol.json")
    if mode not in MODES or protocol.get("initialization_mode", "prior") != mode:
        raise ValueError("Protocol initialization mode mismatch")
    if protocol.get("engineering_smoke", False):
        raise ValueError("Engineering smoke results cannot enter the formal comparison")
    if ((protocol.get("core_contract", {}).get("prior") is not None) != (mode == "prior")
            or (protocol.get("base_config", {}).get("prior_feature_index") is not None) != (mode == "prior")):
        raise ValueError("Protocol prior/index differs from initialization mode")
    prior = protocol["core_contract"].get("prior")
    if mode == "prior" and (not isinstance(prior, dict)
            or protocol["base_config"]["prior_feature_index"] != prior.get("feature_index")):
        raise ValueError("Protocol prior feature index differs from base configuration")
    if protocol.get("test_evaluated") is not False:
        raise ValueError("Formal comparison requires test_evaluated=false")
    if _read(root / "status.json").get("status") != "complete":
        raise ValueError("Matrix is not complete: " + str(root))
    raw = _read(root / "summary.json")
    records = raw["records"] if isinstance(raw, dict) else raw
    summary = summarize_matrix(records)
    if len(summary["seeds"]) != 3 or sorted(protocol.get("seeds", [])) != summary["seeds"]:
        raise ValueError("Formal comparison requires three registered seeds")
    verified, completed, controls, dates = set(), {}, {}, None
    for record in records:
        if record.get("initialization_mode", "prior") != mode or record.get("test_evaluated", False):
            raise ValueError("Record mode or evaluation split mismatch")
        if record["steps"] != protocol[record["budget"] + "_steps"]:
            raise ValueError("Checkpoint steps differ from registered budget")
        run_id = record.get("training_run_id")
        if not isinstance(run_id, str) or Path(run_id).name != run_id or run_id in {"", ".", ".."}:
            raise ValueError("Registered records require a safe training_run_id")
        directory = _safe_path(root, "runs/" + run_id)
        if run_id not in completed:
            marker = _read(directory / "completed.json")
            files = _verify_files(directory, marker, verified)
            required = {"config.json", "initial_model/model.zip", "initial_model/metadata.json"}
            required |= {f"{budget}_model/{name}" for budget in BUDGETS for name in ("model.zip", "metadata.json")}
            if not required <= set(files):
                raise ValueError("Model/configuration evidence is not registered")
            config = _read(directory / "config.json")
            if marker.get("protocol_sha256") != _json_digest(protocol) or marker.get("config_sha256") != _json_digest(config):
                raise ValueError("Completed protocol/configuration digest mismatch")
            expected = deepcopy(protocol["base_config"])
            if mode == "random":
                expected["prior_feature_index"] = None
            expected.update(seed=record["seed"], hidden_sizes=record["hidden_sizes"], device="cuda",
                            torch_threads=1, output_root=str(root / "runs"))
            if config != expected:
                raise ValueError("Run configuration differs from registered base configuration")
            initial_dir = directory / "initial_model"
            if not {"initial_model/model.zip", "initial_model/metadata.json"} <= set(files):
                raise ValueError("Initialization model is not registered")
            initial = _read(initial_dir / "metadata.json")
            initial_hash = _file_hash(initial_dir / "model.zip")
            contract = deepcopy(protocol["core_contract"])
            contract["hidden_sizes"] = record["hidden_sizes"]
            if (initial.get("num_timesteps") != 0 or initial.get("model_sha256") != initial_hash
                    or initial.get("contract") != contract or initial.get("config") != config):
                raise ValueError("Initialization model metadata differs from run identity")
            completed[run_id] = marker
        marker = completed[run_id]
        registered = next((row for row in marker["records"] if _record_key(row) == _record_key(record)), None)
        keys = ("size", "budget", "seed", "steps", "hidden_sizes", "parameters", "metrics", "model_sha256",
                "model_path", "training_run_id", "prefix_shared")
        if registered is None or any(record.get(key) != registered.get(key) for key in keys):
            raise ValueError("Summary differs from completed checkpoint evidence")
        budget = record["budget"]
        metadata = _read(directory / (budget + "_model") / "metadata.json")
        artifact = directory / (budget + "_model") / "model.zip"
        if (record.get("model_sha256") != _file_hash(artifact) or metadata.get("model_sha256") != record.get("model_sha256")
                or metadata.get("num_timesteps") != record["steps"]
                or metadata.get("contract") != _read(directory / "initial_model" / "metadata.json")["contract"]
                or metadata.get("config") != _read(directory / "config.json")):
            raise ValueError("Checkpoint model metadata/hash mismatch")
        audit_path = directory / (budget + "_audit.json")
        if audit_path.exists():
            audit = _read(audit_path)
            if (audit_path.name not in marker["files"] or audit.get("actual_timesteps") != record["steps"]
                    or not audit.get("all_parameters_finite") or audit.get("model_sha256") != record["model_sha256"]
                    or audit.get("parameters") != record["parameters"]):
                raise ValueError("Checkpoint training audit differs from summary")
        control_dir = root / "controls" / run_id if mode == "random" else root / "controls"
        cache_key = str(control_dir)
        if cache_key not in controls:
            control = _read(control_dir / "controls.json")
            files = _verify_files(control_dir, control, verified)
            if not {account + "/metrics.json" for account in ACCOUNTS} <= set(files):
                raise ValueError("Control account metrics are not registered")
            if control.get("zero_actor_prior") is not (mode == "prior"):
                raise ValueError("Control initialization mode mismatch")
            controls[cache_key] = control
        control = controls[cache_key]
        identity = control["identity"]
        expected_data = deepcopy(protocol["core_contract"])
        if mode == "random":
            expected_data["hidden_sizes"] = record["hidden_sizes"]
            initial_hash = _file_hash(directory / "initial_model" / "model.zip")
            if (identity.get("initialization_mode") != "random" or identity.get("training_seed") != record["seed"]
                    or identity.get("initial_model_sha256") != initial_hash):
                raise ValueError("Random control is not bound to its own seed initialization model")
        if (identity.get("data") != expected_data or identity.get("seed") != protocol["evaluation_seed"]
                or identity.get("modes") != protocol["evaluation_modes"]
                or identity.get("broker") != protocol["core_contract"]["broker"]
                or identity.get("initial_capital") != protocol["core_contract"]["initial_capital"]):
            raise ValueError("Control evaluation identity differs from protocol")
        for account in ACCOUNTS:
            scenario = record["metrics"][account]
            control_metrics = _read(control_dir / account / "metrics.json")
            if not _equal_values(control_metrics, control["metrics"][account]):
                raise ValueError("Control metrics differ from their registered file")
            if (not _equal_values(scenario["initialized"], control_metrics["ppo"])
                    or not _equal_values(scenario["baselines"], {key: value for key, value in control_metrics.items() if key != "ppo"})):
                raise ValueError("Summary initialization/baselines differ from own controls")
            validation = directory / (budget + "_validation") / account
            metrics_name = f"{budget}_validation/{account}/metrics.json"
            if metrics_name not in marker["files"] or not _equal_values(_read(validation / "metrics.json")["ppo"], scenario["ppo"]):
                raise ValueError("Summary PPO metrics differ from registered validation")
            for curve_path in [validation / "ppo" / "daily.csv", *[
                    control_dir / account / strategy / "daily.csv" for strategy in control_metrics]]:
                current, _ = _curve(curve_path)
                if str(curve_path) not in verified:
                    raise ValueError("Evaluation curve is not registered")
                if dates is None:
                    dates = current
                if current != dates:
                    raise ValueError("Evaluation dates differ within matrix")
    if dates[0] > protocol["train_end"] or dates[1] <= protocol["train_end"] or dates[-1] > protocol["val_end"]:
        raise ValueError("Evaluation curve escapes registered validation boundaries")
    return dict(root=root, protocol=protocol, records=records,
                evidence=dict(verified_file_count=len(verified), training_runs=len(completed),
                              initialization_controls=len(controls), dates=dates,
                              protocol_file_sha256=_file_hash(root / "protocol.json"),
                              summary_file_sha256=_file_hash(root / "summary.json")))


def _format(stat, scale=100):
    if stat["mean"] is None:
        return "undefined"
    deviation = "" if stat["std"] is None else f" +/- {stat['std'] * scale:.3f}"
    return f"{stat['mean'] * scale:.3f}{deviation}"


def _plot_curves(output, loaded, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    dates = loaded["prior"]["evidence"]["dates"]
    x = [date.fromisoformat(value) for value in dates]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (size, budget) in zip(axes.flat, [(size, budget) for budget in BUDGETS for size in SIZES]):
        name = size + "_" + budget
        for mode, color in [("prior", "#278d70"), ("random", "#a04b82")]:
            seed_curves = []
            for record in loaded[mode]["records"]:
                if (record["size"], record["budget"]) != (size, budget):
                    continue
                directory = loaded[mode]["root"] / "runs" / record["training_run_id"]
                curves = [_curve(directory / (budget + "_validation") / account / "ppo" / "daily.csv")[1]
                          for account in ACCOUNTS]
                seed_curves.append(np.mean(curves, axis=0))
            curves = np.array(seed_curves)
            average = curves.mean(axis=0)
            deviation = curves.std(axis=0, ddof=1)
            ax.plot(x, average, color=color, label=mode)
            ax.fill_between(x, average - deviation, average + deviation, color=color, alpha=.12)
        control = loaded["prior"]["root"] / "controls"
        for strategy, color in [("momentum_daily", "#2874a6"), ("momentum", "#bf8b2f")]:
            curves = [_curve(control / account / strategy / "daily.csv")[1] for account in ACCOUNTS]
            ax.plot(x, np.mean(curves, axis=0), color=color, linestyle="--", label=strategy)
        ax.set_title(name.replace("_", " / "))
        ax.set_ylabel("Net value, four-account mean")
        ax.grid(alpha=.2)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=8)
    fig.suptitle("Already-exposed development validation; shading is sample standard deviation across three seeds")
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)


def _report_lines(summary):
    lines = ["# Prior And Random PPO Matrix Comparison", "", summary["evaluation"]["note"], "",
        "Each cell averages four account scenarios within seed, then reports the mean and sample standard deviation across three seeds. "
        "Short/long checkpoints share training prefixes. Costs are already included in net returns.", "",
        "| Mode / cell | Steps | Parameters | Net return % +/- seed SD | Learning increment pp | Daily momentum gap pp | 20-day momentum gap pp |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    ordered = [(mode + "_" + size + "_" + budget, summary["cells"][mode + "_" + size + "_" + budget])
               for mode in MODES for budget in BUDGETS for size in SIZES]
    for name, cell in ordered:
        values = [_format(cell["metrics"]["cumulative_return"]), *[_format(cell["return_gaps"][key])
                  for key in ("initialized", "momentum_daily", "momentum")]]
        lines.append(f"| {name} | {cell['steps']:,} | {cell['parameters']:,} | " + " | ".join(values) + " |")
    lines += ["", "## Risk, Costs And Cash", "",
        "Eligibility requires every account in every seed to have drawdown magnitude <= 10%. No separate cost threshold was registered.", "",
        "| Mode / cell | Worst drawdown % | Fees +/- SD | Slippage +/- SD | Cost / capital % | Mean cash % | Risk eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    empty = _stats([])
    for name, cell in ordered:
        worst = cell["worst_account_drawdown_magnitude"]["max"]
        values = [f"{worst * 100:.3f}" if worst is not None else "undefined",
            _format(cell["metrics"].get("total_fees", empty), 1),
            _format(cell["metrics"].get("total_slippage_cost", empty), 1), _format(cell["cost_ratio"]),
            _format(cell["metrics"].get("mean_cash_ratio", empty)), str(cell["risk_eligible"])]
        lines.append("| " + name + " | " + " | ".join(values) + " |")
    lines += ["", "## Fixed Momentum Controls", "",
        "Controls use the same registered evaluation seed and account scenarios for both initialization modes.", "",
        "| Control | Net return % | Worst account drawdown % | Fees | Slippage | Cost / capital % |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, control in summary["controls"].items():
        worst = control["worst_account_drawdown_magnitude"]
        metrics = control["metrics"]
        values = [_format(metrics["cumulative_return"]), f"{worst * 100:.3f}" if worst is not None else "undefined",
            _format(metrics.get("total_fees", empty), 1), _format(metrics.get("total_slippage_cost", empty), 1),
            f"{control['cost_ratio'] * 100:.3f}" if control["cost_ratio"] is not None else "undefined"]
        lines.append("| " + name + " | " + " | ".join(values) + " |")
    lines += ["", "## Paired Random Minus Prior", "",
        "| Cell | Net return gap pp +/- SD | Learning increment gap pp +/- SD |",
        "| --- | ---: | ---: |"]
    for name, effect in summary["paired_random_minus_prior"].items():
        lines.append(f"| {name} | {_format(effect['return_gap'])} | {_format(effect['learning_increment_gap'])} |")
    lines += ["", "## Development Candidate", "",
        "The random candidate must pass the drawdown gate, beat the matched prior cell on mean net return, "
        "have positive mean improvement over its own initialization, and have positive learning increments in at least two of three seeds.", ""]
    for field in ("best_observed_ppo", "qualified_random_candidate"):
        result = summary["selection"][field]
        if result is None:
            lines.append(field + ": none.")
        else:
            representative = result["representative"]
            lines.append(f"{field}: **{result['cell']}**, mean return {result['mean_return'] * 100:.3f}%, "
                         f"representative seed {representative['seed']} (closest to cell mean, not best-return seed).")
            lines.append(f"Beats daily momentum: {result['beats_daily_momentum']}; beats 20-day momentum: {result['beats_20day_momentum']}.")
        lines.append("")
    lines += ["Best observed PPO is a descriptive cell comparison, not proof of an optimal strategy. "
        "Failing either momentum baseline prevents a claim of superiority to the strong controls. "
        "No global optimality claim or actionable model is produced; no final-test evaluation is performed.", "",
        "## Account Detail", "",
        "| Mode / cell | Account | Return % +/- SD | Own initialization gap pp | Daily gap pp | 20-day gap pp |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, cell in ordered:
        for account in ACCOUNTS:
            data = cell["per_account"][account]
            values = [_format(data["metrics"]["cumulative_return"]),
                      *[_format(data["return_gaps"][key]) for key in ("initialized", "momentum_daily", "momentum")]]
            lines.append(f"| {name} | {account} | " + " | ".join(values) + " |")
    lines += ["", "## Source And Evidence", "",
        f"Source hash changes recorded: {len(summary['source_hash_differences'])}. Different source versions are disclosed; "
        "this comparison does not establish an isolated causal effect of Critic capacity.", "",
        "![Development net-value curves](comparison.png)", "", "[Machine summary](combined_summary.json)"]
    return lines


def write_comparison(prior_root, random_root, output):
    loaded = dict(prior=load_matrix(prior_root, "prior"), random=load_matrix(random_root, "random"))
    reference = loaded["random"]["protocol"].get("reference_matrix")
    if reference is not None and (Path(reference["root"]).resolve() != loaded["prior"]["root"]
            or reference.get("protocol_sha256") != loaded["prior"]["evidence"]["protocol_file_sha256"]):
        raise ValueError("Random protocol is bound to a different reference matrix")
    if loaded["prior"]["evidence"]["dates"] != loaded["random"]["evidence"]["dates"]:
        raise ValueError("Evaluation dates differ between prior/random matrices")
    summary = summarize_comparison(loaded["prior"]["records"], loaded["random"]["records"],
                                   loaded["prior"]["protocol"], loaded["random"]["protocol"])
    summary["evidence"] = {mode: dict(root=str(data["root"]), **data["evidence"]) for mode, data in loaded.items()}
    output = Path(output).resolve()
    if output in [data["root"] for data in loaded.values()]:
        raise ValueError("Comparison output must be separate from source matrix roots")
    output.mkdir(parents=True, exist_ok=True)
    _plot_curves(output, loaded, summary)
    lines = _report_lines(summary)
    (output / "combined_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    candidate = summary["selection"]["qualified_random_candidate"]
    short = ["# Prior And Random Matrix Summary", "", summary["evaluation"]["note"], "",
             "Qualified random development candidate: " + (candidate["cell"] if candidate else "none") + ".", "",
             "Selection uses three-seed four-account cell means. Representative seed is closest to the cell mean, not the highest return. "
             "actionable=false; global_optimality_claim=false; test_evaluated=false.", "",
             "[Full report](report.md) | [Machine summary](combined_summary.json) | [Curves](comparison.png)"]
    (output / "summary.md").write_text("\n".join(short) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--random-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = write_comparison(args.prior_root, args.random_root, args.output)
    print(json.dumps(summary["selection"], allow_nan=False))
    return summary


if __name__ == "__main__":
    main()
