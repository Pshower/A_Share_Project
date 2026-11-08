"""Synthetic archives/canvas only; no real results, environments or model fit."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tools.archive import plot_counterfactual_learning as plot


GATE = dict(across_fold_mean_baseline_gap_min=.005, max_all_account_drawdown=.10,
            max_drawdown_excess_over_baseline=.02, each_fold_positive_gap_seeds=2)


def row(fold, method, steps, seed=42, phase="screen"):
    baseline = {mode: dict(cumulative_return=.03 + i * .001, max_drawdown=-.06,
        total_fees=100., total_slippage_cost=200.) for i, mode in enumerate(plot.runner.MODES)}
    auxiliary = deepcopy(baseline)
    if method == "value_warmup":
        for metric in auxiliary.values():
            metric["cumulative_return"] += .001
    trained = deepcopy(auxiliary)
    for i, metric in enumerate(trained.values()):
        metric["cumulative_return"] += .007 + i * .0002 + steps * 1e-9
        metric["max_drawdown"] = -.06 - i * .001
    result = dict(fold=fold, phase=phase, method=method, seed=seed, steps=steps,
        initialized=baseline, benchmark=baseline, auxiliary=auxiliary, trained=trained, actionable=False,
        changes={mode: dict(greedy_changed=True, economic_ledger_changed=True) for mode in baseline},
        changes_vs_aux={mode: dict(greedy_changed=True, economic_ledger_changed=True) for mode in baseline})
    for name, role in (("baseline_gap", "benchmark"), ("initial_gain", "initialized"), ("ppo_gain", "auxiliary")):
        result[name] = plot.runner.account_mean(trained) - plot.runner.account_mean(result[role])
    return result


def summary(phase="screen"):
    screen = [row(fold, method, steps) for fold in plot.runner.SCREEN
              for method in plot.runner.METHODS for steps in plot.runner.SNAPSHOTS]
    selection = plot.runner.select_screen(screen)
    result = dict(status=selection["status"], selection=selection, screen=screen, qualified=False, candidate=None)
    if phase == "confirm":
        confirm = [row(fold, method, selection["selected"]["steps"], seed, phase) for fold in plot.runner.CONFIRM
                   for method in plot.runner.METHODS for seed in plot.runner.SEEDS]
        outcome = plot.runner.confirmation_summary(confirm, selection, GATE)
        result.update(status="confirmation_complete", confirmation=confirm, outcome=outcome,
                      qualified=outcome["qualified"], candidate=outcome["candidate"])
    return result


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.setattr(plot.runner, "ROOT", tmp_path)
    monkeypatch.setattr(plot.runner, "check_frozen", lambda *_: None)

    def create(phase="screen"):
        root = tmp_path / phase
        root.mkdir()
        protocol = {"gate": GATE}
        data = summary(phase)
        for name, value in (("protocol.json", protocol), ("summary.json", data),
                            ("selection.json", data["selection"]), ("selection_proof.json", {})):
            plot.runner.write_json(root / name, value)
        records = data["screen" if phase == "screen" else "confirmation"]
        groups = {}
        for record in records:
            groups.setdefault((record["fold"], record["method"], record["seed"]), []).append(record)
        for (fold, method, seed), group in groups.items():
            directory = root / "runs" / f"{phase}_{fold}_{method}_{seed}"
            directory.mkdir(parents=True)
            plot.runner.seal(directory, protocol, records=group)
        directory = root / "diagnostics/cold_audit" / phase
        directory.mkdir(parents=True)
        _, outcome = plot.validated_rows(data, phase, GATE)
        audit = dict(status="verified_saved_evidence", phase=phase, records=records, outcome=outcome)
        plot.runner.write_json(directory / "audit.json", audit)
        paths = [root / name for name in ("protocol.json", "selection.json", "selection_proof.json")]
        paths += list((root / "runs").glob("*/completed.json"))
        plot.runner.write_json(directory / "manifest.json", dict(status="passed_cold_artifact_audit", phase=phase,
            protocol_sha256=plot.runner.digest(protocol),
            inputs_sha256={str(path.resolve()): plot.runner.file_hash(path) for path in paths},
            outputs_sha256={"audit.json": plot.runner.file_hash(directory / "audit.json")}))
        return root

    return create


@pytest.mark.parametrize("phase,bar_count", [("screen", 4), ("confirm", 6)])
def test_complete_rows_canvas_pixels_units_and_all_fixed_seeds(phase, bar_count):
    import matplotlib.pyplot as plt
    data = summary(phase)
    rows, _ = plot.validated_rows(data, phase, GATE)
    figure = plot.build_figure(rows, phase)
    try:
        figure.canvas.draw()
        pixels = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
        assert pixels.shape[0] > 500 and pixels.shape[1] > 700
        assert np.unique(pixels.reshape(-1, 3), axis=0).shape[0] > 100
        assert np.count_nonzero(np.any(pixels < 100, axis=2)) > 2000
        assert len(figure.axes) == 6
        assert [axis.get_ylabel() for axis in figure.axes[::2]] == list(plot.Y_LABELS)
        assert all(len(axis.patches) == bar_count for axis in figure.axes)
        assert all(len(axis.collections) == bar_count for axis in figure.axes)
        assert all(len(collection.get_offsets()) == 4 for axis in figure.axes for collection in axis.collections)
        if phase == "confirm":
            labels = [text.get_text() for text in figure.axes[0].get_xticklabels()]
            assert all(sum(f"seed {seed}" in value for value in labels) == 2 for seed in plot.runner.SEEDS)
        assert "not confidence intervals" in figure.texts[-1].get_text()
    finally:
        plt.close(figure)


@pytest.mark.parametrize("phase", ["screen", "confirm"])
def test_load_requires_real_closed_marker_and_cold_audit_binding(archive, phase):
    root = archive(phase)
    rows, proof = plot.load_closed(root, phase)
    assert len(rows) == (8 if phase == "screen" else 12)
    assert proof["observed_mutable_summary"]["not_a_frozen_dependency"] is True
    assert str(root / "summary.json") not in proof["frozen_inputs_sha256"]


def test_write_is_new_directory_only_and_registers_all_output_hashes(archive):
    root = archive()
    destination = plot.write_figures(root, "screen")
    manifest = plot.runner.read_json(destination / "manifest.json")
    assert manifest["outputs_sha256"] == {"learning_comparison.png": plot.runner.file_hash(destination / "learning_comparison.png")}
    assert manifest["units"]["learning_gain"] == "percentage_points"
    assert manifest["counters"] == dict(environment_steps=0, neural_forwards=0, model_fits=0, optimizer_updates=0)
    with pytest.raises(ValueError, match="overwrite"):
        plot.write_figures(root, "screen")


def test_figure_rejects_resolved_destination_alias_before_reading(archive, monkeypatch):
    from pathlib import Path
    root = archive()
    destination = root / "diagnostics/figures_screen"
    resolve = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda path, *a, **k: root.parent / "redirected"
                        if path == destination else resolve(path, *a, **k))
    with pytest.raises(ValueError, match="redirect"):
        plot.write_figures(root, "screen")
    assert not destination.exists()


def test_figure_input_change_during_render_never_gets_manifest(archive, monkeypatch):
    from matplotlib.figure import Figure
    root = archive()
    save = Figure.savefig

    def changed(figure, path, *args, **kwargs):
        save(figure, path, *args, **kwargs)
        target = root / "selection_proof.json"
        target.write_bytes(target.read_bytes() + b" ")

    monkeypatch.setattr(Figure, "savefig", changed)
    with pytest.raises(ValueError, match="changed while rendering"):
        plot.write_figures(root, "screen")
    assert not (root / "diagnostics/figures_screen/manifest.json").exists()


@pytest.mark.parametrize("damage", ["duplicate", "extra_budget", "scalar", "metrics", "partial"])
def test_invalid_summaries_rejected_before_drawing(archive, damage):
    phase = "confirm" if damage in {"extra_budget", "partial"} else "screen"
    root = archive(phase)
    value = plot.runner.read_json(root / "summary.json")
    key = "confirmation" if phase == "confirm" else "screen"
    if damage == "duplicate":
        value[key].append(deepcopy(value[key][0]))
    elif damage == "extra_budget":
        value[key][0]["steps"] = 8192
    elif damage == "scalar":
        value[key][0]["ppo_gain"] += .1
    elif damage == "metrics":
        value[key][0]["trained"][plot.runner.MODES[0]]["cumulative_return"] += .1
    else:
        value["status"] = "screen_admitted"
    plot.runner.write_json(root / "summary.json", value)
    with pytest.raises(ValueError):
        plot.load_closed(root, phase)
    assert not (root / "diagnostics" / ("figures_" + phase)).exists()


@pytest.mark.parametrize("damage", ["missing_marker", "unclosed_run", "audit_hash", "input_hash", "unbound_marker", "audit_rows"])
def test_unsealed_or_different_cold_artifacts_rejected(archive, damage):
    root = archive()
    directory = root / "diagnostics/cold_audit/screen"
    manifest = plot.runner.read_json(directory / "manifest.json")
    marker = next((root / "runs").glob("*/completed.json"))
    if damage == "missing_marker":
        marker.unlink()
    elif damage == "unclosed_run":
        (root / "runs/screen_extra").mkdir()
    elif damage == "audit_hash":
        manifest["outputs_sha256"]["audit.json"] = "0" * 64
    elif damage == "input_hash":
        manifest["inputs_sha256"][str(marker)] = "0" * 64
    elif damage == "unbound_marker":
        del manifest["inputs_sha256"][str(marker)]
    else:
        audit = plot.runner.read_json(directory / "audit.json")
        audit["records"][0]["trained"][plot.runner.MODES[0]]["cumulative_return"] += .01
        plot.runner.write_json(directory / "audit.json", audit)
        manifest["outputs_sha256"]["audit.json"] = plot.runner.file_hash(directory / "audit.json")
    plot.runner.write_json(directory / "manifest.json", manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        plot.load_closed(root, "screen")
