"""Plot sealed pooling results and index models, without new model evaluation."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.preprocess import ROOT, file_hash, read_config, write_json
from src.training.research_market_summary import ARMS, FOLDS, validate_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reports/runs/ppo_market_summary_20261007")
    args = parser.parse_args()
    root = (ROOT / args.root).resolve()
    summary = read_config(root / "study_summary.json")
    records = read_config(root / "summary.json")
    if any(summary[key] != value for key, value in validate_records(records).items()):
        raise ValueError("Summary counts differ from sealed snapshot records")
    inputs = {name: file_hash(root / name) for name in ("study_summary.json", "summary.json")}
    colors = ["#337cbd", "#36836b", "#ac4e60"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained", sharex=True)
    for column, fold in enumerate(FOLDS):
        for budget, offset, alpha, hatch in (("short", -.19, .55, "//"), ("long", .19, 1., None)):
            selected = [summary["fold_arms"][fold][f"{arm}_{budget}"] for arm in ARMS]
            for row, stats in enumerate((
                [value["metrics"]["cumulative_return"] for value in selected],
                [value["paired_roi_vs_mean"] for value in selected],
            )):
                axes[row, column].bar(
                    np.arange(3) + offset, [value["mean"] * 100 for value in stats],
                    yerr=[value["std"] * 100 for value in stats], width=.36,
                    color=colors, alpha=alpha, hatch=hatch, capsize=4,
                    label=f"{budget}: {selected[0]['steps']:,} steps",
                )
        rules_path = Path("rules") / fold / "completed.json"
        inputs[str(rules_path)] = file_hash(root / rules_path)
        rules = read_config(root / rules_path)["metrics"]
        for rule, label, color, style in (
            ("momentum_20day", "20-day momentum", "#222222", "--"),
            ("hold", "Hold initial shares", "#765b92", "-."),
        ):
            roi = np.mean([account[rule]["cumulative_return"] for account in rules.values()]) * 100
            axes[0, column].axhline(roi, color=color, linestyle=style, linewidth=1.1,
                                    label=f"{label}: {roi:.2f}%")
        axes[0, column].set_title(f"{fold} Q2 historical validation")
        axes[0, column].set_ylabel("Net ROI (%)")
        axes[0, column].legend(fontsize=8, loc="upper left", ncol=2, framealpha=1.)
        axes[1, column].set_ylabel("Paired ROI difference vs mean (pp)")
        axes[1, column].axhline(0, color="#222222", linewidth=.9)
        axes[1, column].set_xticks(range(3), ["Mean", "Attention", "Dispersion"])
        for axis in axes[:, column]:
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
    fig.suptitle(
        "Pure-market pooling: identical initial Actor and Critic functions\n"
        "Four-account mean per seed; error bars = sample SD across three paired seeds",
        fontsize=13,
    )
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)
    index = []
    for record in records:
        path = ROOT / record["model_path"]
        metadata = read_config(path / "metadata.json")
        actual = file_hash(path / "model.zip")
        if actual != record["model_sha256"] or actual != metadata["model_sha256"]:
            raise ValueError("Cannot index an altered model")
        index.append(dict(
            fold=record["fold"], arm=record["arm"], seed=record["seed"], budget=record["budget"],
            steps=record["steps"], parameters=record["parameters"], model_path=record["model_path"],
            model_sha256=actual, git_commit=metadata["git"]["commit"], git_dirty=metadata["git"]["dirty"],
            source_count=len(metadata["source_hashes"]),
        ))
    pd.DataFrame(index).to_csv(root / "model_index.csv", index=False)
    write_json(root / "plot_evidence.json", dict(
        no_inference=True, additional_environment_steps=0, additional_optimizer_steps=0,
        aggregation=summary["aggregation"], qualified_candidate=summary["qualified_candidate"],
        inputs=inputs, script_sha256=file_hash(__file__),
        artifacts={name: file_hash(root / name) for name in ("comparison.png", "model_index.csv")},
    ))


if __name__ == "__main__":
    main()
