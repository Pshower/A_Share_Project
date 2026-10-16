"""Plot sealed paired-label diagnostics without fitting or replay."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools import research_timing_counterfactual_pilot as driver


def build_figure(rows):
    by_fold = {fold: [row for row in rows if row["fold"] == fold] for fold in driver.FOLDS}
    driver.require(len(rows) == 192 and all(len(group) == 96 for group in by_fold.values()),
                   "Plots require the complete 192-state diagnostic")
    values = np.asarray([float(row["discounted_relative_delta"]) * 10000 for row in rows])
    driver.require(np.isfinite(values).all(), "Plot labels must be finite")
    costs_all = np.asarray([(float(row["fee_delta_over_initial_nav"])
                             + float(row["slippage_delta_over_initial_nav"])) * 10000 for row in rows])
    driver.require(np.isfinite(costs_all).all(), "Plot costs must be finite")
    extent = max(float(np.abs(values).max()), 1.)
    bins = np.linspace(-extent, extent, 22)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100, layout="constrained")
    for column, fold in enumerate(driver.FOLDS):
        group = by_fold[fold]
        delta = np.asarray([float(row["discounted_relative_delta"]) * 10000 for row in group])
        costs = np.asarray([(float(row["fee_delta_over_initial_nav"])
                             + float(row["slippage_delta_over_initial_nav"])) * 10000 for row in group])
        driver.require(np.isfinite(costs).all(), "Plot costs must be finite")
        hist, scatter = axes[:, column]
        hist.hist(delta, bins=bins, color="#29816e", edgecolor="white", linewidth=.5)
        hist.axvspan(-1, 1, color="#b9b9b9", alpha=.5, label="Fixed neutral band: +/-1 log-bp")
        hist.axvline(0, color="#333333", linewidth=.8)
        hist.set_title(f"{fold} training period: 96 overlapping states")
        hist.set_xlabel("Discounted reward difference R - H (log-bp)")
        hist.set_ylabel("States")
        hist.legend(fontsize=8, loc="upper left")
        for offset, color in ((0, "#29816e"), (60, "#b64b64")):
            mask = np.asarray([int(row["offset"]) == offset for row in group])
            scatter.scatter(costs[mask], delta[mask], s=24, alpha=.7, color=color,
                            label=f"Source offset {offset}", edgecolors="none")
        scatter.axhline(0, color="#333333", linewidth=.8)
        scatter.axvline(0, color="#777777", linewidth=.8)
        scatter.set_xlabel("Extra fees + slippage / initial NAV: R - H (bp)")
        scatter.set_ylabel("Discounted reward difference R - H (log-bp)")
        scatter.legend(fontsize=8)
        for axis in axes[:, column]:
            axis.grid(alpha=.2)
            axis.set_axisbelow(True)
    fig.suptitle("Train-only HOLD vs REBALANCE: historical outcomes, not a fitted strategy\n"
                 "20-step original M20 calendar, gamma=0.995; no achievable ROI claim", fontsize=12)
    return fig


def run(root=driver.OUTPUT):
    root = driver.output_root(root)
    cold = root / "diagnostics" / "cold_analysis"
    output = root / "diagnostics" / "figures"
    driver.require(output.resolve() == output and not output.exists(), "Plot destination is aliased or already exists")
    manifest_path = cold / "manifest.json"
    manifest_sha = driver.file_hash(manifest_path)
    manifest = driver.read_json(manifest_path)
    driver.require(manifest["status"] == "complete" and manifest["native_label_count"] == 192
                   and set(manifest["outputs_sha256"]) == {"analysis.json", "pairs.csv", "strata.csv"},
                   "Plot requires completed cold native analysis")
    inputs = {manifest_path: manifest_sha}
    for name, sha in manifest["outputs_sha256"].items():
        inputs[cold / name] = sha
    for name, sha in manifest["input_files_sha256"].items():
        path = Path(name).resolve()
        driver.require(path.is_relative_to(driver.ROOT.resolve()), "Plot input escapes the workspace")
        inputs[path] = sha
    for path, sha in inputs.items():
        driver.require(driver.file_hash(path) == sha, "Plot input identity changed")
    with (cold / "pairs.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    fig = build_figure(rows)
    try:
        for path, sha in inputs.items():
            driver.require(driver.file_hash(path) == sha, "Plot input changed during rendering")
        output.mkdir(exist_ok=False)
        path = output / "paired_outcomes.png"
        fig.savefig(path, dpi=160)
        for source, sha in inputs.items():
            driver.require(driver.file_hash(source) == sha, "Plot input changed during completion")
        evidence = dict(schema_version=1, status="complete", states=192, cold_manifest_sha256=manifest_sha,
            new_environment_steps=0, new_nn_forwards=0, new_optimizer_updates=0, model_fits=0,
            script_sha256=driver.file_hash(__file__), figure_sha256=driver.file_hash(path),
            interpretation="Descriptive log-reward differences, not strategy ROI or predictive performance")
        with (output / "manifest.json").open("xb") as stream:
            stream.write(driver.canonical(evidence) + b"\n")
    finally:
        plt.close(fig)
    print("COUNTERFACTUAL_PLOT states=192 environment_steps=0 fitting=0 output=" + str(output), flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=driver.OUTPUT)
    args = parser.parse_args(argv)
    run(args.output)


if __name__ == "__main__":
    main()
