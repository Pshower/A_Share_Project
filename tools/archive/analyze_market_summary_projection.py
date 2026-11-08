"""Read sealed pooling traces; never load models or step an environment."""

import argparse
import json
import math
from pathlib import Path

from tools.analyze_trade_projection import ROOT, ACCOUNTS, aggregate, analyze_path, grouped, load_json, sha256


def verify_read_inputs(directory, files, inputs):
    directory = directory.resolve()
    for name, digest in inputs.items():
        path = (ROOT / name).resolve()
        if path.is_relative_to(directory):
            relative = path.relative_to(directory).as_posix()
            if files.get(relative) != digest:
                raise ValueError("A diagnostic input differs from its sealed registry: " + name)


def attention_bounds(query_norm):
    if not math.isfinite(query_norm) or query_norm < 0:
        raise ValueError("Query norm must be finite and nonnegative")
    log_bound = 2 * query_norm
    return dict(
        absolute_log_ratio_bound=log_bound,
        minimum_weight_over_uniform=math.exp(-log_bound),
        maximum_weight_over_uniform=math.exp(log_bound) if log_bound < 700 else None,
        maximum_relative_deviation=math.expm1(log_bound) if log_bound < 700 else None,
        interpretation="Worst-case bound from tanh encodings, not measured attention weights, entropy or an ROI bound.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reports/runs/ppo_market_summary_20261007")
    args = parser.parse_args()
    root = (ROOT / args.root).resolve()
    inputs = {}
    protocol = load_json(root / "protocol.json", inputs)
    records = load_json(root / "summary.json", inputs)
    expected = {(fold, arm, seed, budget, steps)
                for fold in protocol["folds"] for arm in protocol["arms"] for seed in protocol["seeds"]
                for budget, steps in protocol["budgets"].items()}
    keys = [(r["fold"], r["arm"], r["seed"], r["budget"], r["steps"]) for r in records]
    if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("Diagnostics require all 36 sealed snapshots, not partial results")
    trained, initialized, audits, controls, checked = [], [], [], set(), {}
    for record in records:
        directory = (ROOT / record["model_path"]).resolve().parent
        if not directory.is_relative_to(root / "runs"):
            raise ValueError("Model path does not belong to the registered study")
        if directory not in checked:
            checked[directory] = load_json(directory / "completed.json", inputs)
        marker = checked[directory]
        if record not in marker["records"]:
            raise ValueError("Global record differs from its sealed trial")
        identity = {key: record[key] for key in ("fold", "arm", "budget", "seed")}
        for account in ACCOUNTS:
            row = analyze_path(directory / record["budget"] / account, dict(identity, account=account), inputs)
            if any(value != record["metrics"][account]["ppo"][key] for key, value in row["metrics"].items()):
                raise ValueError("Trace metrics differ from the sealed snapshot record")
            trained.append(row)
        audit = load_json(directory / (record["budget"] + "_audit.json"), inputs)
        participation = audit["parameter_participation"]
        audits.append(dict(identity, participation=participation,
            attention_uniform_bounds=attention_bounds(participation["summary_query_l2"])
                if record["arm"] == "attention" else None,
            training_diagnostics=audit["training_diagnostics"]))
        # A completed marker is the registry itself, not one of its own leaves.
        leaf_inputs = {name: digest for name, digest in inputs.items()
                       if (ROOT / name).resolve() != directory / "completed.json"}
        verify_read_inputs(directory, marker["files"], leaf_inputs)
        initial = (ROOT / record["initial_controls_path"]).resolve()
        if not initial.is_relative_to(root / "initial_controls"):
            raise ValueError("Initialization control falls outside this study")
        if initial in controls:
            continue
        controls.add(initial)
        initial_marker = load_json(initial / "completed.json", inputs)
        for account in ACCOUNTS:
            initialized.append(analyze_path(initial / account,
                dict(fold=record["fold"], seed=record["seed"], account=account), inputs))
        leaf_inputs = {name: digest for name, digest in inputs.items()
                       if (ROOT / name).resolve() != initial / "completed.json"}
        verify_read_inputs(initial, initial_marker["files"], leaf_inputs)
    result = dict(
        no_model_loading=True, no_inference=True, additional_environment_steps=0,
        additional_optimizer_steps=0, script_sha256=sha256(Path(__file__)),
        helper_sha256=sha256(ROOT / "tools" / "analyze_trade_projection.py"), inputs=inputs,
        aggregation="Equal four-account paths within each fold/arm/budget; paired-seed ROI inference is in study_summary.json.",
        scope="Raw directions, size-active requests, top/cash projection and actual fills; observational, not causal.",
        denominators=dict(
            direction_rates="All requested stock codes divided by decision steps; forced HOLD included.",
            top_cash_filters="Filtered BUY requests divided by all raw BUY requests; filters are disjoint.",
            mapped_rate="Mapped nonzero orders divided by all raw non-HOLD requests.",
            fill_rate="Actual nonzero fills divided by mapped nonzero orders.",
            size_statistics="All raw size-active stock-steps, including subsequently filtered requests.",
            probability_margin="Valid stock-steps with more than one legal direction, excluding forced HOLD.",
            notionals="Absolute shares multiplied by the same decision-date mark before/after projection.",
        ),
        caveats=["Observed fees cannot be added back to ROI as a causal zero-turnover counterfactual.",
                 "Snapshots and account paths are not additional independent training seeds."],
        trained_total=aggregate(trained), trained_groups=grouped(trained, ("fold", "arm", "budget")),
        initialized_groups=grouped(initialized, ("fold",)),
        trained_paths=trained, initialized_paths=initialized, training_audits=audits,
    )
    output = root / "projection_diagnostics.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), sha256=sha256(output),
                         trained_paths=len(trained), initialized_paths=len(initialized))))


if __name__ == "__main__":
    main()
