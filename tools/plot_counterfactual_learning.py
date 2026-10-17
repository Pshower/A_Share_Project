"""Read-only, cold-audit-bound figures for completed paired/PPO experiments."""

import argparse
from pathlib import Path

import numpy as np

from tools import research_counterfactual_learning as runner


METHOD_LABELS = {"ppo_only": "PPO only", "value_warmup": "Ridge + PPO"}
COLORS = {"ppo_only": "#287c8e", "value_warmup": "#c47636"}
Y_LABELS = ("Net ROI (%)", "PPO - Start (pp)", "Worst account drawdown (%)")


def _ordered(rows):
    return sorted(rows, key=lambda row: (row["fold"], row["method"], row["seed"], row["steps"]))


def validated_rows(summary, phase, gate):
    runner.require(phase in ("screen", "confirm"), "Unknown figure phase")
    selection = runner.select_screen(summary["screen"])
    runner.require(summary["selection"] == selection, "Summary selection differs from actual records")
    if phase == "screen":
        rows, outcome = summary["screen"], selection
    else:
        runner.require(summary["status"] == "confirmation_complete" and selection["admitted"],
                       "Only complete authorized confirmation can be plotted")
        rows = summary["confirmation"]
        outcome = runner.confirmation_summary(rows, selection, gate)
        runner.require(summary["outcome"] == outcome and summary["qualified"] == outcome["qualified"]
                       and summary["candidate"] == outcome["candidate"], "Confirmation outcome differs")
    return [runner.validate_record(row, phase) for row in rows], outcome


def load_closed(root, phase):
    """Bind plotted rows to real closed runs and independently audited bytes."""
    root = Path(root).resolve()
    protocol = runner.read_json(root / "protocol.json")
    runner.check_frozen(protocol, root)
    summary = runner.read_json(root / "summary.json")
    rows, outcome = validated_rows(summary, phase, protocol["gate"])
    sealed = runner.load_rows(root, protocol, phase)
    runner.require(runner.canonical(_ordered(summary["screen" if phase == "screen" else "confirmation"]))
                   == runner.canonical(_ordered(sealed)), "Summary differs from sealed physical run records")
    directory = root / "diagnostics/cold_audit" / phase
    manifest_path, audit_path = directory / "manifest.json", directory / "audit.json"
    manifest, audit = runner.read_json(manifest_path), runner.read_json(audit_path)
    runner.require(manifest["status"] == "passed_cold_artifact_audit" and manifest["phase"] == phase
                   and manifest["protocol_sha256"] == runner.digest(protocol)
                   and manifest["outputs_sha256"] == {"audit.json": runner.file_hash(audit_path)},
                   "Cold audit manifest identity differs")
    for name, sha in manifest["inputs_sha256"].items():
        path = Path(name)
        runner.require(path.is_absolute() and path.resolve().is_relative_to(runner.ROOT.resolve())
                       and runner.file_hash(path) == sha, "A cold audit input changed")
    mandatory = [root / name for name in ("protocol.json", "selection.json", "selection_proof.json")]
    mandatory += [path / "completed.json" for path in sorted((root / "runs").glob(phase + "_*"))]
    runner.require(all(manifest["inputs_sha256"].get(str(path.resolve())) == runner.file_hash(path)
                       for path in mandatory), "Cold audit did not bind required selection or run markers")
    runner.require(audit["status"] == "verified_saved_evidence" and audit["phase"] == phase
                   and runner.canonical(_ordered(audit["records"])) == runner.canonical(_ordered(sealed))
                   and audit["outcome"] == outcome, "Cold audit records or outcome differ")
    frozen = dict(manifest["inputs_sha256"])
    frozen.update({str(path.resolve()): runner.file_hash(path) for path in mandatory + [manifest_path, audit_path]})
    return rows, dict(frozen_inputs_sha256=frozen, phase_record_digest=runner.digest(dict(records=_ordered(sealed), outcome=outcome)),
        observed_mutable_summary=dict(path=str(root / "summary.json"), sha256_at_read=runner.file_hash(root / "summary.json"),
            not_a_frozen_dependency=True))


def build_figure(rows, phase):
    """Bars use four fixed accounts; dots are accounts, never confidence limits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folds = runner.SCREEN if phase == "screen" else runner.CONFIRM
    rows = [runner.validate_record(row, phase) for row in rows]
    keys = ([(method, 42, steps) for method in runner.METHODS for steps in runner.SNAPSHOTS]
            if phase == "screen" else [(method, seed, rows[0]["steps"]) for method in runner.METHODS for seed in runner.SEEDS])
    expected = {(fold, method, seed, steps) for fold in folds for method, seed, steps in keys}
    actual = {(r["fold"], r["method"], r["seed"], r["steps"]) for r in rows}
    runner.require(len(rows) == len(actual) == len(expected) and actual == expected, "Figure rows are incomplete or duplicated")
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), squeeze=False)
    for column, fold in enumerate(folds):
        group = {(r["method"], r["seed"], r["steps"]): r for r in rows if r["fold"] == fold}
        baseline = group[keys[0]]["benchmark"]
        runner.require(all(row["benchmark"] == baseline for row in group.values()), "Accounts changed across compared models")
        x = np.arange(len(keys))
        labels = [f"{METHOD_LABELS[m]}\n{s // 1024}K" if phase == "screen" else f"{METHOD_LABELS[m]}\nseed {seed}"
                  for m, seed, s in keys]
        for index, key in enumerate(keys):
            row = group[key]
            roi = np.array([row["trained"][mode]["cumulative_return"] for mode in runner.MODES]) * 100
            gain = np.array([row["trained"][mode]["cumulative_return"] - row["auxiliary"][mode]["cumulative_return"]
                             for mode in runner.MODES]) * 100
            drawdown = np.abs([row["trained"][mode]["max_drawdown"] for mode in runner.MODES]) * 100
            for level, values in enumerate((roi, gain, drawdown)):
                axes[level, column].bar(index, values.max() if level == 2 else values.mean(), width=.66,
                    color=COLORS[key[0]], alpha=.85, zorder=2)
                axes[level, column].scatter(index + np.linspace(-.16, .16, len(values)), values,
                    color="#242424", s=13, alpha=.8, zorder=4)
        axes[0, column].set_title(f"{fold} Q2 / {'development seed 42' if phase == 'screen' else 'all three fixed seeds'}")
        axes[0, column].axhline(runner.account_mean(baseline) * 100, color="#202020", linestyle="--", linewidth=1.2, label="M20 mean")
        axes[1, column].axhline(0, color="#555555", linewidth=1)
        axes[2, column].axhline(max(abs(baseline[m]["max_drawdown"]) for m in runner.MODES) * 100,
                              color="#555555", linestyle="--", linewidth=1, label="M20 worst")
        for level in range(3):
            axis = axes[level, column]
            axis.set_xticks(x, labels, fontsize=8)
            axis.set_ylabel(Y_LABELS[level])
            axis.grid(axis="y", alpha=.22, zorder=0)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0, column].legend(fontsize=8)
        axes[2, column].legend(fontsize=8)
    figure.suptitle("PPO action-value learning / " + ("development" if phase == "screen" else f"confirmation, fixed {keys[0][2] // 1024}K budget"), fontsize=14)
    figure.text(.5, .015, "Bars: four-account mean (drawdown: worst). Dots: fixed accounts, not confidence intervals.\n"
                "PPO start = theta0 for PPO only; thetaaux for Ridge + PPO. Costs and slippage included.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .06, 1, .95))
    return figure


def write_figures(root, phase):
    root = Path(root).resolve()
    destination = root / "diagnostics" / ("figures_" + phase)
    runner.require(destination.resolve() == destination and destination.is_relative_to(root)
                   and not destination.exists(), "Never overwrite or redirect an existing figure directory")
    rows, proof = load_closed(root, phase)
    figure = build_figure(rows, phase)
    import matplotlib.pyplot as plt
    try:
        runner.require(destination.resolve() == destination, "Figure directory became aliased")
        destination.mkdir(parents=True)
        path = destination / "learning_comparison.png"
        figure.savefig(path, dpi=160, facecolor="white")
    finally:
        plt.close(figure)
    own = Path(__file__).resolve()
    test = own.parents[1] / "tests/test_plot_counterfactual_learning.py"
    runner.require(all(runner.file_hash(Path(name)) == sha for name, sha in proof["frozen_inputs_sha256"].items()),
                   "A frozen figure input changed while rendering")
    runner.write_json(destination / "manifest.json", dict(schema_version=1, phase=phase, **proof,
        source_test_sha256={str(item): runner.file_hash(item) for item in (own, test)},
        outputs_sha256={path.name: runner.file_hash(path)}, units=dict(roi="percent", learning_gain="percentage_points", drawdown="percent"),
        counters=dict(environment_steps=0, neural_forwards=0, model_fits=0, optimizer_updates=0), actionable=False))
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("screen", "confirm"))
    parser.add_argument("--root", default=str(runner.OUTPUT))
    args = parser.parse_args(argv)
    return write_figures(runner.output_root(args.root), args.phase)


if __name__ == "__main__":
    main()
