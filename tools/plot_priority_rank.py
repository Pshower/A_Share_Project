"""Plot the complete registered priority/rank study without loading a model."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = ROOT / "reports" / "runs" / "ppo_priority_rank_20261011_16k"
FOLDS = ("2023", "2024")
ACTORS = ("coupled", "decoupled")
METHODS = ("ppo", "rank_aux")
SEEDS = (42, 123, 2026)
BUDGETS = dict(short=8192, long=16384)
ACCOUNTS = ("cash", "equal", "concentrated", "random")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path, inputs):
    path = Path(path).resolve()
    if not path.is_relative_to(STUDY_ROOT):
        raise ValueError("Study input escapes the registered 16k output")
    inputs[path.relative_to(ROOT).as_posix()] = sha256(path)
    return json.loads(path.read_text(encoding="utf-8"))


def verify_leaf(path, directory, marker, inputs):
    path, directory = Path(path).resolve(), Path(directory).resolve()
    if not path.is_relative_to(directory) or not directory.is_relative_to(STUDY_ROOT):
        raise ValueError("Input is not a leaf of the registered study")
    relative = path.relative_to(directory).as_posix()
    digest = sha256(path)
    if marker["files"].get(relative) != digest:
        raise ValueError("Input differs from its sealed registry: " + relative)
    inputs[path.relative_to(ROOT).as_posix()] = digest


def load_study(root):
    root = Path(root).resolve()
    if root != STUDY_ROOT.resolve():
        raise ValueError("This tool only accepts the registered 20261011 standard 16k study")
    inputs = {}
    protocol = read_json(root / "protocol.json", inputs)
    records = read_json(root / "summary.json", inputs)
    summary = read_json(root / "study_summary.json", inputs)
    status = read_json(root / "status.json", inputs)
    if (protocol.get("study") != "priority_rank_factorial" or protocol.get("profile") != "standard"
            or protocol.get("budgets") != BUDGETS or set(protocol["folds"]) != set(FOLDS)
            or protocol.get("actors") != list(ACTORS) or protocol.get("methods") != list(METHODS)
            or protocol.get("seeds") != list(SEEDS) or status.get("status") != "complete"):
        raise ValueError("Only a complete preregistered standard study can be plotted")
    expected = {(fold, actor, method, seed, budget, steps) for fold in FOLDS for actor in ACTORS
                for method in METHODS for seed in SEEDS for budget, steps in BUDGETS.items()}
    actual = [(r["fold"], r["actor"], r["method"], r["seed"], r["budget"], r["steps"]) for r in records]
    counts = dict(actual_training_steps=393216, trained_runs=24, snapshots=48, checkpoint_steps_sum=589824)
    if (len(actual) != 48 or len(set(actual)) != 48 or set(actual) != expected
            or any(summary.get(k) != v or status.get(k) != v for k, v in counts.items())
            or summary.get("test_evaluated") is not False or summary.get("actionable") is not False):
        raise ValueError("Partial or changed results cannot be marked complete")
    markers = {}
    for fold in FOLDS:
        receipt = read_json(root / f"worker_{fold}.json", inputs)
        ids = {f"{fold}_{actor}_{method}_{seed}" for actor in ACTORS for method in METHODS for seed in SEEDS}
        completed = receipt["completed_runs"]
        if (receipt.get("status") != "complete" or len(completed) != 12
                or {x["run_id"] for x in completed} != ids
                or receipt["protocol_sha256"] != sha256(root / "protocol.json")):
            raise ValueError("A fold worker is not fully sealed")
        for item in completed:
            directory = root / "runs" / item["run_id"]
            marker = read_json(directory / "completed.json", inputs)
            if sha256(directory / "completed.json") != item["completed_sha256"]:
                raise ValueError("Worker completion marker changed")
            wanted = [r for r in records if (r["fold"], r["actor"], r["method"], r["seed"]) ==
                      (fold, item["run_id"].split("_")[1], "rank_aux" if "_rank_aux_" in item["run_id"] else "ppo", int(item["run_id"].split("_")[-1]))]
            if marker["records"] != wanted:
                raise ValueError("Global records differ from their sealed trial")
            markers[directory.name] = marker
    return protocol, records, summary, markers, inputs


def parameter_sizes(contract):
    h, d = contract["hidden_sizes"]
    width = contract["lookback"] * len(contract["feature_cols"])
    context = 2 * d + 9
    return {"network." + name: size for name, size in {
        "market_encoder.0.weight": h * width, "market_encoder.0.bias": h,
        "market_encoder.2.weight": d * h, "market_encoder.2.bias": d,
        "account_encoder.0.weight": 11 * d, "account_encoder.0.bias": d,
        "stock_body.0.weight": d * (d + context), "stock_body.0.bias": d,
        "direction_head.weight": 4 * d, "direction_head.bias": 4,
        "size_head.weight": d, "size_head.bias": 1, "log_std": 1,
        "value_head.0.weight": d * context, "value_head.0.bias": d,
        "value_head.2.weight": d, "value_head.2.bias": 1,
        "priority_head.weight": d, "priority_head.bias": 1, "priority_log_std": 1,
        "signal_head.weight": d, "signal_head.bias": 1,
    }.items()}


def build_index(root, records, markers, inputs):
    index = []
    for record in records:
        directory = root / "runs" / f"{record['fold']}_{record['actor']}_{record['method']}_{record['seed']}"
        marker = markers[directory.name]
        path = (ROOT / record["model_path"]).resolve()
        if path != directory / (record["budget"] + "_model"):
            raise ValueError("Model does not belong to its registered trial checkpoint")
        for leaf in (path / "metadata.json", path / "model.zip", directory / (record["budget"] + "_audit.json")):
            verify_leaf(leaf, directory, marker, inputs)
        metadata = read_json(path / "metadata.json", inputs)
        audit = read_json(directory / (record["budget"] + "_audit.json"), inputs)
        sizes = parameter_sizes(metadata["contract"])
        if (sum(sizes.values()) != record["parameters"] or record["parameters"] != 125322
                or sha256(path / "model.zip") != record["model_sha256"]
                or metadata["model_sha256"] != record["model_sha256"] or metadata["num_timesteps"] != record["steps"]
                or audit["auxiliary_audit"] != record["auxiliary_audit"]
                or metadata["extra"]["auxiliary_audit"] != record["auxiliary_audit"]):
            raise ValueError("Model index identity or parameter/auxiliary evidence changed")
        inactive = set()
        if record["actor"] == "coupled":
            inactive.update(("network.priority_head.weight", "network.priority_head.bias", "network.priority_log_std"))
        if record["method"] == "ppo":
            inactive.update(("network.signal_head.weight", "network.signal_head.bias"))
        optimizer = audit["optimizer_steps_by_parameter"]
        if not set(optimizer) <= set(sizes):
            raise ValueError("Unknown optimizer tensor name cannot be assigned a parameter count")
        aux = record["auxiliary_audit"]
        roi = np.mean([record["metrics"][mode]["ppo"]["cumulative_return"] for mode in ACCOUNTS])
        init = np.mean([record["metrics"][mode]["initialized"]["cumulative_return"] for mode in ACCOUNTS])
        index.append(dict(fold=record["fold"], actor=record["actor"], method=record["method"], seed=record["seed"],
            budget=record["budget"], steps=record["steps"], parameters=record["parameters"],
            expected_objective_active_parameters=sum(n for name, n in sizes.items() if name not in inactive),
            observed_adam_state_tensors=len(optimizer), observed_adam_state_parameters=sum(sizes[name] for name in optimizer),
            inactive_names=";".join(sorted(inactive)), inactive_adam_states=";".join(sorted(set(optimizer) & inactive)),
            optimizer_steps_min=record["optimizer_steps_min"], optimizer_steps_max=record["optimizer_steps_max"],
            actual_joint_optimizer_updates=aux["actual_joint_optimizer_updates"], actual_aux_updates=aux["actual_aux_updates"],
            supervised_stock_date_exposures=aux["supervised_stock_date_exposures"], aux_pool_identity_sha256=aux["pool_identity_sha256"],
            four_account_mean_roi=roi, own_initialization_mean_roi=init, own_initialization_gain=roi - init,
            model_path=record["model_path"], model_sha256=record["model_sha256"], algorithm_class=metadata["algorithm_class"],
            git_commit=metadata["git"]["commit"], git_dirty=metadata["git"]["dirty"], source_count=len(metadata["source_hashes"])))
    return index


def plot(root, summary, inputs):
    fig, axes = plt.subplots(3, 2, figsize=(14, 11.5), layout="constrained")
    cells = [(a, m) for a in ACTORS for m in METHODS]
    labels = ["Coupled\nPPO", "Coupled\nRank aux", "Decoupled\nPPO", "Decoupled\nRank aux"]
    specs = [("actor_decoupled_minus_coupled", "actor_under_ppo", "actor_under_rank_aux"),
             ("method_rank_aux_minus_ppo", "method_under_coupled", "method_under_decoupled", "interaction")]
    for column, fold in enumerate(FOLDS):
        for budget, offset, color in (("short", -.18, "#39856e"), ("long", .18, "#387cba")):
            roi = [summary["fold_cells"][fold][f"{a}_{m}_{budget}"]["metrics"]["cumulative_return"] for a, m in cells]
            axes[0, column].bar(np.arange(4) + offset, [x["mean"] * 100 for x in roi],
                yerr=[x["std"] * 100 for x in roi], width=.34, color=color, capsize=3, label=f"{BUDGETS[budget]:,} steps")
            for row, names in enumerate(specs, start=1):
                effects = summary["factorial_effects"][fold][budget]["net_roi"]
                axes[row, column].bar(np.arange(len(names)) + offset, [effects[n]["mean"] * 100 for n in names],
                    yerr=[effects[n]["std"] * 100 for n in names], width=.34, color=color, capsize=3)
        marker_path = root / "rules" / fold / "completed.json"
        rules = read_json(marker_path, inputs)["metrics"]
        benchmark = np.mean([rules[mode]["momentum_20day"]["cumulative_return"] for mode in ACCOUNTS]) * 100
        axes[0, column].axhline(benchmark, linestyle="--", color="#555555", linewidth=1, label=f"20-day rule {benchmark:.2f}%")
        axes[0, column].set_title(f"{fold} Q2: historical development validation")
        axes[0, column].set_xticks(range(4), labels)
        axes[0, column].set_ylabel("Four-account mean net ROI (%)")
        axes[0, column].legend(fontsize=8)
        axes[1, column].set_xticks(range(3), ["Actor main", "Under PPO", "Under rank aux"])
        axes[1, column].set_ylabel("Decoupled minus coupled (pp)")
        axes[2, column].set_xticks(range(4), ["Method main", "Under coupled", "Under decoupled", "Interaction"])
        axes[2, column].set_ylabel("Rank aux minus PPO / interaction (pp)")
        for axis in axes[:, column]:
            axis.axhline(0, color="#333333", linewidth=.7)
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
    gate = "No configuration passed all gates" if summary["qualified_candidate"] is None else "Qualified research candidate only"
    fig.suptitle("Priority / rank auxiliary factorial: 24 runs, 48 fixed checkpoints\n"
        "Error bars: sample SD across three paired seeds; initial actor sampling supports differ\n"
        f"{gate}. HFQ proxy, not independent testing or automatic deployment.", fontsize=12)
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(STUDY_ROOT))
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    _, records, summary, markers, inputs = load_study(root)
    index = build_index(root, records, markers, inputs)
    plot(root, summary, inputs)
    pd.DataFrame(index).to_csv(root / "model_index.csv", index=False)
    result = dict(no_model_loading=True, no_inference=True, additional_environment_steps=0, additional_optimizer_steps=0,
        script_sha256=sha256(__file__), inputs=inputs, snapshots=48, aggregation=summary["aggregation"],
        qualified_candidate=deepcopy(summary["qualified_candidate"]),
        active_parameters_semantics="Expected contribution to registered joint objective; observed Adam-state coverage comes from signed audits, not tensor loading.",
        artifacts={name: sha256(root / name) for name in ("comparison.png", "model_index.csv")})
    (root / "plot_evidence.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(snapshots=48, output=str(root / "comparison.png"), model_loading=0, env_steps=0)))


if __name__ == "__main__":
    main()
