"""Plot sealed research metrics; no model inference or environment steps."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.preprocess import ROOT, read_config, file_hash, write_json
from src.training.research_exploration import FORMAL, validate_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reports/runs/ppo_exploration_20261006")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    summary, records = read_config(root / "study_summary.json"), read_config(root / "summary.json")
    validate_records(records)
    rules = read_config(root / "rules" / "completed.json")["results"]
    cells = list(FORMAL)
    colors = ["#536672", "#337cbd", "#36836b", "#b27d37"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, layout="constrained")
    for budget, offset, alpha, hatch in (("short", -.19, .55, "//"), ("long", .19, 1., None)):
        values = [summary["cells"][f"{cell}_{budget}"] for cell in cells]
        roi = [value["metrics"]["cumulative_return"] for value in values]
        axes[0].bar(np.arange(4) + offset, [value["mean"] * 100 for value in roi], width=.36,
            yerr=[value["std"] * 100 for value in roi], color=colors, alpha=alpha, hatch=hatch,
            capsize=4, label=f"{budget}: {'2,048' if budget == 'short' else '8,192'} steps")
        axes[1].bar(np.arange(4) + offset, [value["cost"]["mean"] for value in values], width=.36,
                    color=colors, alpha=alpha, hatch=hatch)
    for rule, label, color, style in (("momentum_20day", "20-day momentum", "#ac4e60", "--"),
                                      ("momentum_daily", "daily momentum", "#777777", ":"),
                                      ("hold", "hold initial shares", "#222222", "-.")):
        value = np.mean([account[rule]["cumulative_return"] for account in rules.values()]) * 100
        axes[0].axhline(value, color=color, linestyle=style, linewidth=1.2, label=f"{label}: {value:.2f}%")
    axes[0].set_ylabel("Net ROI (%); error bars = seed SD")
    axes[0].legend(fontsize=9, loc="upper left", ncol=2, framealpha=1.)
    axes[1].set_ylabel("Mean fees + slippage (CNY)")
    axes[1].set_xticks(range(4), ["L1 mixed", "L5 mixed", "L1 split", "L1 quarter-balanced"])
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle("Retrospective 2025 Q2 research; not a blind test", fontsize=14)
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)
    index = []
    for record in records:
        path = ROOT / record["model_path"]
        metadata = read_config(path / "metadata.json")
        actual = file_hash(path / "model.zip")
        if actual != record["model_sha256"] or actual != metadata["model_sha256"]:
            raise ValueError("Plotting cannot index an altered model")
        index.append(dict(cell=record["cell"], seed=record["seed"], budget=record["budget"],
            steps=record["steps"], pilot=record["pilot"], parameters=record["parameters"],
            model_path=record["model_path"], model_sha256=actual,
            git_commit=metadata["git"]["commit"], git_dirty=metadata["git"]["dirty"],
            source_count=len(metadata["source_hashes"])))
    pd.DataFrame(index).to_csv(root / "model_index.csv", index=False)
    write_json(root / "plot_evidence.json", dict(no_inference=True, no_environment_steps=True,
        scope="Same 2025 fold, four account mean within seed, sample SD across three seeds; no pilot ranking.",
        inputs={name: file_hash(root / name) for name in ("study_summary.json", "summary.json")},
        script_sha256=file_hash(__file__),
        artifacts={name: file_hash(root / name) for name in ("comparison.png", "model_index.csv")}))


if __name__ == "__main__":
    main()
