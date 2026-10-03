"""Paired arithmetic, selection and registered evidence checks; never train."""

from copy import deepcopy
import json

import pytest

from src.training import matrix_compare as compare
from src.training.matrix_report import ACCOUNTS


SEEDS = [42, 123, 2026]


def _metric(value, offset=0):
    return dict(cumulative_return=value + offset, sharpe=value * 10, max_drawdown=-.04,
                total_fees=100., total_slippage_cost=50., mean_cash_ratio=.5)


def _inputs():
    inputs = {}
    values = {
        "prior": {("small", "short"): [.11, .12, .13], ("small", "long"): [.13, .14, .15],
                  ("large", "short"): [.14, .15, .16], ("large", "long"): [.16, .17, .18]},
        "random": {("small", "short"): [.20, .22, .21], ("small", "long"): [.19, .21, .22],
                   ("large", "short"): [1., .10, .11], ("large", "long"): [.30, .31, .32]},
    }
    core = dict(dataset_version="fixture", manifest_sha256="same-manifest", artifacts={"scaler_params.json": "same-scaler"},
                feature_cols=["return"], stock_codes=["000001"], lookback=1, broker={"commission": .0003},
                initial_capital=1000000, price_basis="hfq_research", prior={"feature_index": 0, "strength": 4., "residual_scale": .25})
    base = dict(seed=42, hidden_sizes=[8, 4], device="cuda", torch_threads=1, output_root="runs", research_config="same.json",
                ppo={"n_steps": 512, "batch_size": 128, "n_epochs": 5, "learning_rate": .00001},
                prior_feature_index=0, prior_strength=4., residual_scale=.25, initial_log_std=-3.)
    protocol = dict(seeds=SEEDS, short_steps=1024, long_steps=2048, train_end="2026-03-31", val_end="2026-06-30",
                    evaluation_seed=20260927, evaluation_modes=list(ACCOUNTS), checkpoint_selection="fixed_budget_shared_prefix",
                    core_contract=core, base_config=base, source_hashes={"src/policy.py": "old"},
                    validation_already_exposed=True, test_evaluated=False, engineering_smoke=False)
    for mode in ["prior", "random"]:
        rows = []
        for (size, budget), returns in values[mode].items():
            for index, seed in enumerate(SEEDS):
                initialized = .1 if mode == "prior" else (.25 if size == "large" else .15)
                scenarios = {}
                for account, offset in zip(ACCOUNTS, [-.015, -.005, .005, .015]):
                    scenarios[account] = dict(ppo=_metric(returns[index], offset), initialized=_metric(initialized, offset),
                        baselines={"momentum_daily": _metric(.20, offset), "momentum": _metric(.35, offset), "cash": _metric(0., offset)})
                row = dict(size=size, budget=budget, seed=seed, steps=1024 if budget == "short" else 2048,
                    hidden_sizes=[8, 4] if size == "small" else [16, 8], parameters=123 if size == "small" else 456,
                    elapsed_seconds=10., training_diagnostics={}, experience={}, metrics=scenarios,
                    training_run_id=f"{size}_{seed}", prefix_shared=True, model_path=f"{size}_{seed}/{budget}_model",
                    model_sha256="fixture-model", test_evaluated=False)
                if mode == "random":
                    row["initialization_mode"] = mode
                rows.append(row)
        current = deepcopy(protocol)
        if mode == "random":
            current.update(initialization_mode="random", source_hashes={"src/policy.py": "new"})
            current["core_contract"].pop("prior")
            current["base_config"]["prior_feature_index"] = None
        inputs[mode] = dict(records=rows, protocol=current)
    return inputs


def _summarize(inputs):
    return compare.summarize_comparison(inputs["prior"]["records"], inputs["random"]["records"],
                                        inputs["prior"]["protocol"], inputs["random"]["protocol"])


def test_account_means_then_seed_uncertainty_and_paired_effects():
    result = _summarize(_inputs())
    cell = result["cells"]["random_large_long"]
    assert len(result["cells"]) == 8
    assert cell["metrics"]["cumulative_return"]["mean"] == pytest.approx(.31)
    assert cell["metrics"]["cumulative_return"]["std"] == pytest.approx(.01)
    assert cell["per_account"]["cash"]["metrics"]["cumulative_return"]["mean"] == pytest.approx(.295)
    assert cell["return_gaps"]["initialized"]["mean"] == pytest.approx(.06)
    effect = result["paired_random_minus_prior"]["large_long"]
    assert effect["return_gap"]["mean"] == pytest.approx(.14)
    assert effect["return_gap"]["std"] == pytest.approx(0.)
    assert effect["per_seed"][0]["accounts"]["cash"]["return_gap"] == pytest.approx(.14)
    assert cell["cost_ratio"]["mean"] == pytest.approx(.00015)
    assert result["controls"]["momentum"]["metrics"]["cumulative_return"]["mean"] == pytest.approx(.35)
    assert result["controls"]["momentum"]["worst_account_drawdown_magnitude"] == pytest.approx(.04)
    assert result["source_hash_differences"] == {"src/policy.py": {"prior": "old", "random": "new"}}


def test_candidate_requires_learning_in_two_seeds_and_uses_median_representative():
    result = _summarize(_inputs())
    selection = result["selection"]
    assert selection["best_observed_ppo"]["cell"] == "random_large_short"
    candidate = selection["qualified_random_candidate"]
    assert candidate["cell"] == "random_large_long"
    assert candidate["representative"]["seed"] == 123
    assert "not highest-return seed" in candidate["representative"]["rule"]
    assert candidate["beats_20day_momentum"] is False
    assert result["actionable"] is False and result["global_optimality_claim"] is False
    assert result["evaluation"]["test_evaluated"] is False
    assert result["evaluation"]["test_protected_from_prior_exposure"] is None


@pytest.mark.parametrize("gate", ["drawdown", "missing_drawdown", "prior_gap", "negative_mean_learning", "only_one_positive_seed"])
def test_each_registered_random_gate_can_disqualify_all_candidates(gate):
    inputs = _inputs()
    for record in inputs["random"]["records"]:
        prior = next(row for row in inputs["prior"]["records"] if compare._record_key(row) == compare._record_key(record))
        for account, scenario in record["metrics"].items():
            if gate == "drawdown":
                scenario["ppo"]["max_drawdown"] = -.10000001
            elif gate == "missing_drawdown":
                if account == "cash":
                    scenario["ppo"]["max_drawdown"] = None
            elif gate == "prior_gap":
                scenario["ppo"]["cumulative_return"] = prior["metrics"][account]["ppo"]["cumulative_return"]
            elif gate == "negative_mean_learning":
                scenario["initialized"]["cumulative_return"] = scenario["ppo"]["cumulative_return"] + .01
            else:
                scenario["initialized"]["cumulative_return"] = scenario["ppo"]["cumulative_return"] + (-1. if record["seed"] == 42 else .01)
    result = _summarize(inputs)
    assert result["selection"]["qualified_random_candidate"] is None


@pytest.mark.parametrize("change, message", [
    ("contract", "Core data"), ("noise", "hyperparameters"), ("random_index", "prior_feature_index"),
    ("seed", "registered training seeds"), ("capacity", "capacity"), ("dates", "train_end"),
    ("evaluation_seed", "evaluation_seed"), ("baseline", "Baseline values"), ("smoke", "smoke"),
    ("record_mode", "Record mode"), ("test", "test_evaluated"), ("missing_record", "same seeds"),
])
def test_invalid_or_incomparable_matrix_is_rejected(change, message):
    inputs = _inputs()
    protocol = inputs["random"]["protocol"]
    if change == "contract":
        protocol["core_contract"]["broker"]["commission"] = .001
    elif change == "noise":
        protocol["base_config"]["initial_log_std"] = 0.
    elif change == "random_index":
        protocol["base_config"]["prior_feature_index"] = 0
    elif change == "seed":
        protocol["seeds"] = [42, 123]
    elif change == "capacity":
        for row in inputs["random"]["records"]:
            if row["size"] == "large":
                row["parameters"] += 1
    elif change == "dates":
        protocol["train_end"] = "2026-01-01"
    elif change == "evaluation_seed":
        protocol["evaluation_seed"] = 1
    elif change == "baseline":
        inputs["random"]["records"][0]["metrics"]["cash"]["baselines"]["momentum"]["cumulative_return"] += 2e-10
    elif change == "smoke":
        protocol["engineering_smoke"] = True
    elif change == "record_mode":
        inputs["random"]["records"][0]["initialization_mode"] = "prior"
    elif change == "test":
        protocol["test_evaluated"] = True
    else:
        inputs["random"]["records"].pop()
    with pytest.raises(ValueError, match=message):
        _summarize(inputs)


def test_baseline_absolute_tolerance_and_exact_drawdown_boundary():
    inputs = _inputs()
    inputs["random"]["records"][0]["metrics"]["cash"]["baselines"]["momentum"]["cumulative_return"] += 5e-11
    for record in inputs["random"]["records"]:
        for scenario in record["metrics"].values():
            scenario["ppo"]["max_drawdown"] = -.10
    assert _summarize(inputs)["cells"]["random_large_long"]["risk_eligible"] is True


@pytest.mark.parametrize("preferred, drawdown, fees", [("random_large_long", -.03, 200.),
    ("random_large_long", -.04, 99.), ("prior_small_short", -.04, 100.)])
def test_exact_mean_ties_use_drawdown_then_costs_then_budget_and_capacity(preferred, drawdown, fees):
    inputs = _inputs()
    for current in inputs.values():
        for record in current["records"]:
            for scenario in record["metrics"].values():
                scenario["ppo"]["cumulative_return"] = .2
    for record in inputs["random"]["records"]:
        if (record["size"], record["budget"]) == ("large", "long"):
            for scenario in record["metrics"].values():
                scenario["ppo"].update(max_drawdown=drawdown, total_fees=fees)
    result = _summarize(inputs)
    assert result["selection"]["best_observed_ppo"]["cell"] == preferred


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_curve(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("date,net_value\n2026-03-31,1.0\n2026-04-01,1.1\n2026-06-30,1.2\n", encoding="utf-8")


def _write_evidence(root, mode, inputs):
    current = deepcopy(inputs[mode])
    protocol, rows = current["protocol"], current["records"]
    _write_json(root / "protocol.json", protocol)
    _write_json(root / "status.json", {"status": "complete"})
    for run_id in sorted({row["training_run_id"] for row in rows}):
        run = root / "runs" / run_id
        run_rows = [row for row in rows if row["training_run_id"] == run_id]
        first = run_rows[0]
        config = deepcopy(protocol["base_config"])
        config.update(seed=first["seed"], hidden_sizes=first["hidden_sizes"], device="cuda", torch_threads=1, output_root=str(root / "runs"))
        contract = dict(deepcopy(protocol["core_contract"]), hidden_sizes=first["hidden_sizes"])
        _write_json(run / "config.json", config)
        initial_zip = run / "initial_model" / "model.zip"
        initial_zip.parent.mkdir(parents=True, exist_ok=True)
        initial_zip.write_bytes((mode + run_id + "initial").encode())
        initial_hash = compare._file_hash(initial_zip)
        _write_json(initial_zip.with_name("metadata.json"), dict(num_timesteps=0, model_sha256=initial_hash, contract=contract, config=config))
        controls = root / "controls" / run_id if mode == "random" else root / "controls"
        if not (controls / "controls.json").exists():
            metrics = {account: dict(ppo=first["metrics"][account]["initialized"], **first["metrics"][account]["baselines"]) for account in ACCOUNTS}
            identity = dict(data=contract if mode == "random" else protocol["core_contract"], seed=protocol["evaluation_seed"],
                            modes=list(ACCOUNTS), broker=contract["broker"], initial_capital=contract["initial_capital"])
            if mode == "random":
                identity.update(initialization_mode="random", training_seed=first["seed"], initial_model_sha256=initial_hash)
            for account in ACCOUNTS:
                _write_json(controls / account / "metrics.json", metrics[account])
                for strategy in metrics[account]:
                    _write_curve(controls / account / strategy / "daily.csv")
            files = {path.relative_to(controls).as_posix(): compare._file_hash(path) for path in controls.rglob("*") if path.is_file()}
            _write_json(controls / "controls.json", dict(identity=identity, metrics=metrics, files=files, zero_actor_prior=mode == "prior"))
        for row in run_rows:
            budget = row["budget"]
            model = run / (budget + "_model") / "model.zip"
            model.parent.mkdir(parents=True)
            model.write_bytes((mode + run_id + budget).encode())
            row["model_sha256"] = compare._file_hash(model)
            _write_json(model.with_name("metadata.json"), dict(num_timesteps=row["steps"], model_sha256=row["model_sha256"], contract=contract, config=config))
            for account in ACCOUNTS:
                validation = run / (budget + "_validation") / account
                _write_json(validation / "metrics.json", {"ppo": row["metrics"][account]["ppo"]})
                _write_curve(validation / "ppo" / "daily.csv")
        files = {path.relative_to(run).as_posix(): compare._file_hash(path) for path in run.rglob("*") if path.is_file()}
        _write_json(run / "completed.json", dict(config_sha256=compare._json_digest(config), protocol_sha256=compare._json_digest(protocol), files=files, records=run_rows))
    _write_json(root / "summary.json", rows)
    return rows


def test_complete_registered_evidence_and_report_preserve_source_files(tmp_path, monkeypatch):
    inputs = _inputs()
    prior, random, output = (tmp_path / name for name in ["prior", "random", "comparison"])
    _write_evidence(prior, "prior", inputs)
    _write_evidence(random, "random", inputs)
    before = {str(path): path.read_bytes() for root in [prior, random] for path in root.rglob("*") if path.is_file()}
    monkeypatch.setattr(compare, "_plot_curves", lambda *args: None)
    result = compare.write_comparison(prior, random, output)
    assert result["evidence"]["prior"]["initialization_controls"] == 1
    assert result["evidence"]["random"]["initialization_controls"] == 6
    assert result["evidence"]["random"]["training_runs"] == 6
    assert json.loads((output / "combined_summary.json").read_text())["selection"] == result["selection"]
    report = (output / "report.md").read_text()
    assert "median" not in report or "not best-return seed" in report
    assert "No separate cost threshold" in report and "already exposed" in report
    assert "actionable=false" in (output / "summary.md").read_text()
    assert {str(path): path.read_bytes() for root in [prior, random] for path in root.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("change, message", [("hash", "hash mismatch"), ("control_seed", "own seed"),
    ("control_model", "own seed"), ("summary_init", "checkpoint evidence"), ("dates", "dates differ"),
    ("escaping_hash", "escapes"), ("incomplete", "not complete")])
def test_evidence_rejects_tampering_and_wrong_initialization_binding(tmp_path, change, message):
    inputs = _inputs()
    root = tmp_path / "random"
    _write_evidence(root, "random", inputs)
    controls_path = root / "controls" / "large_42" / "controls.json"
    if change == "hash":
        (root / "runs" / "large_42" / "initial_model" / "model.zip").write_bytes(b"changed")
    elif change in {"control_seed", "control_model"}:
        control = compare._read(controls_path)
        control["identity"]["training_seed" if change == "control_seed" else "initial_model_sha256"] = 999
        _write_json(controls_path, control)
    elif change == "summary_init":
        rows = compare._read(root / "summary.json")
        rows[0]["metrics"]["cash"]["initialized"]["cumulative_return"] += .1
        _write_json(root / "summary.json", rows)
    elif change == "dates":
        curve = root / "runs" / "large_42" / "long_validation" / "cash" / "ppo" / "daily.csv"
        curve.write_text("date,net_value\n2026-03-31,1.0\n2026-04-02,1.1\n2026-06-30,1.2\n")
        marker_path = root / "runs" / "large_42" / "completed.json"
        marker = compare._read(marker_path)
        marker["files"][curve.relative_to(marker_path.parent).as_posix()] = compare._file_hash(curve)
        _write_json(marker_path, marker)
    elif change == "escaping_hash":
        marker_path = root / "runs" / "large_42" / "completed.json"
        marker = compare._read(marker_path)
        marker["files"]["../../outside.json"] = "bad"
        _write_json(marker_path, marker)
    else:
        _write_json(root / "status.json", {"status": "training"})
    with pytest.raises(ValueError, match=message):
        compare.load_matrix(root, "random")


@pytest.mark.parametrize("change, message", [("smoke", "smoke"), ("mode", "mode mismatch"), ("prior", "prior/index")])
def test_standalone_loader_rejects_smoke_and_mode_prior_mismatch_before_artifact_loading(tmp_path, change, message):
    protocol = deepcopy(_inputs()["random"]["protocol"])
    if change == "smoke":
        protocol["engineering_smoke"] = True
    elif change == "mode":
        protocol["initialization_mode"] = "prior"
    else:
        protocol["core_contract"]["prior"] = {"feature_index": 0}
    _write_json(tmp_path / "protocol.json", protocol)
    with pytest.raises(ValueError, match=message):
        compare.load_matrix(tmp_path, "random")
