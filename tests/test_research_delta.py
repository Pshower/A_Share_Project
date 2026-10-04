"""Train-only pilot routing, fixed-budget arithmetic and cache checks; no training."""

from copy import deepcopy
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from src.training import research_delta as study


def _base():
    return dict(action_top_k=20, prior_feature_index=14, seed=1, hidden_sizes=[64, 32],
                ppo=dict(n_steps=512, batch_size=128, n_epochs=5, learning_rate=1e-5))


def _pilot(candidate=None, **changes):
    result = dict(candidate or study.PILOT_CANDIDATES[0], status="success", actual_steps=2048,
                  all_parameters_finite=True, optimizer_steps=80, gross_filled_notional=100.,
                  tail_mean_kl=.02, stock_sigma=999., validation_evaluated=False, test_evaluated=False)
    result.update(changes)
    return result


def test_config_is_delta_only_small_conditional_and_preserves_base(tmp_path):
    base = _base()
    original = deepcopy(base)
    candidate = study.PILOT_CANDIDATES[-1]
    config = study.config_for(base, tmp_path, 123, candidate)
    assert base == original
    assert config["action_mode"] == "stock_delta" and "action_top_k" not in config
    assert config["prior_feature_index"] is None
    assert config["hidden_sizes"] == [256, 128]
    assert config["max_delta_weight"] == .05 and config["actor_output_gain"] == .01
    assert config["top_n"] == 20 and config["top_n_choices"] == [10, 20, 40]
    assert config["seed"] == 123 and config["ppo"]["learning_rate"] == 1e-6
    assert config["initial_log_std"] == -1.5


@pytest.mark.parametrize("change", [{"status": "error"}, {"actual_steps": 512},
    {"all_parameters_finite": False}, {"optimizer_steps": 0}, {"gross_filled_notional": 0.},
    {"tail_mean_kl": .0450001}, {"tail_mean_kl": None}, {"tail_mean_kl": float("nan")}, {"tail_mean_kl": -.01}])
def test_pilot_eligibility_fails_closed(change):
    assert not study.pilot_eligible(_pilot(**change))
    assert study.choose_pilot([_pilot(**change)]) is None


def test_pilot_ranking_uses_actual_optimizer_then_kl_initial_sigma_lr_and_never_roi():
    reports = [_pilot(candidate, optimizer_steps=index + 1, validation_return=100 - index)
               for index, candidate in enumerate(study.PILOT_CANDIDATES)]
    assert study.choose_pilot(reports) == reports[-1]
    for report in reports:
        report.update(optimizer_steps=80, tail_mean_kl=.02)
    assert study.choose_pilot(reports)["name"] == study.PILOT_CANDIDATES[0]["name"]
    reports[-1]["tail_mean_kl"] = .01
    assert study.choose_pilot(reports) == reports[-1]
    assert study.pilot_eligible(_pilot(tail_mean_kl=.045))


def test_four_pilots_have_total_8192_steps_and_never_dispatch_formal_or_evaluation(tmp_path, monkeypatch):
    calls = []
    def fake(config, root, candidate, protocol):
        calls.append(config)
        path = root / "pilots" / candidate["name"]
        path.mkdir(parents=True)
        study.write_json(path / "completed.json", {"mock": True})
        return _pilot(candidate)
    monkeypatch.setattr(study, "run_pilot", fake)
    monkeypatch.setattr(study, "evaluate_agent", lambda *args, **kwargs: pytest.fail("Pilot evaluated validation"))
    monkeypatch.setattr(study, "run_formal", lambda *args: pytest.fail("Pilot automatically ran formal training"))
    result = study.run_pilots(dict(base_config=_base()), tmp_path)
    assert len(calls) == 4 and result["actual_steps"] == 8192
    assert {(config["ppo"]["learning_rate"], config["initial_log_std"]) for config in calls} == {
        (lr, std) for lr in [1e-5, 1e-6] for std in [-3., -1.5]}
    assert all(config["seed"] == 42 for config in calls)
    assert result["validation_evaluated"] is False and result["test_evaluated"] is False


def test_no_eligible_pilot_writes_failure_and_never_formal(tmp_path, monkeypatch):
    def fake(config, root, candidate, protocol):
        path = root / "pilots" / candidate["name"]
        path.mkdir(parents=True)
        study.write_json(path / "completed.json", {"mock": True})
        return _pilot(candidate, tail_mean_kl=1.)
    monkeypatch.setattr(study, "run_pilot", fake)
    with pytest.raises(RuntimeError, match="formal training is disabled"):
        study.run_pilots(dict(base_config=_base()), tmp_path)
    assert study.read_config(tmp_path / "status.json")["status"] == "pilot_failed_closed"


def test_direct_stock_sigma_uses_parameter_not_entropy(tmp_path, monkeypatch):
    data = SimpleNamespace(codes=["000001"], config={"validation_rollouts": 10000, "ppo": {"n_steps": 512}})
    callback = study.DeltaCallback(data, tmp_path, None, {},
                                   dict(pilot=2048), tmp_path / "status.json", "pilot")
    logger = MagicMock()
    logger.name_to_value = {"train/entropy_loss": 12345.}
    callback.model = SimpleNamespace(policy=SimpleNamespace(network=SimpleNamespace(log_std=torch.tensor([-1.5]))), logger=logger)
    monkeypatch.setattr(study.MatrixCallback, "record_update", lambda self: None)
    callback.record_update()
    value = next(call.args[1] for call in logger.record.call_args_list if call.args[0] == "train/stock_sigma")
    assert value == pytest.approx(math.exp(-1.5))


def _records():
    rows = []
    for budget, steps in study.BUDGETS.items():
        for index, seed in enumerate(study.SEEDS):
            value = .12 + index * .02 + (.06 if budget == "long" else 0.)
            scenarios = {}
            for account, offset in zip(study.MODES, [-.03, -.01, .01, .03]):
                def metric(value):
                    return dict(cumulative_return=value + offset, max_drawdown=-.04, total_fees=100., total_slippage_cost=50., mean_cash_ratio=.5)
                scenarios[account] = dict(ppo=metric(value), initialized=metric(.08),
                    baselines=dict(momentum=metric(.20), momentum_daily=metric(.17), cash=metric(0.)))
            rows.append(dict(size="small", budget=budget, seed=seed, steps=steps, hidden_sizes=[256, 128],
                parameters=159000, metrics=scenarios, auxiliary_metrics={str(k): {name: item["ppo"] for name, item in scenarios.items()} for k in [10, 40]},
                test_evaluated=False))
    references = {}
    for mode, adjustment in [("prior", -.02), ("random", -.05)]:
        previous = deepcopy(rows)
        for row in previous:
            for scenario in row["metrics"].values():
                scenario["ppo"]["cumulative_return"] += adjustment
        references[mode] = dict(records=previous)
    return rows, references


def test_two_budget_three_seed_arithmetic_and_legacy_pairing():
    rows, references = _records()
    summary = study.summarize_study(rows, references)
    short = summary["cells"]["short"]
    assert short["metrics"]["cumulative_return"]["mean"] == pytest.approx(.14)
    assert short["metrics"]["cumulative_return"]["std"] == pytest.approx(.02)
    assert short["return_gaps"]["initialized"]["mean"] == pytest.approx(.06)
    assert short["legacy_return_gaps"]["prior"]["mean"] == pytest.approx(.02)
    assert short["legacy_return_gaps"]["random"]["mean"] == pytest.approx(.05)
    assert short["per_account"]["cash"]["cumulative_return"]["mean"] == pytest.approx(.11)
    assert summary["paired_long_minus_short"]["mean"] == pytest.approx(.06)
    assert summary["actual_formal_steps"] == 196608
    assert summary["checkpoint_steps_sum"] == 221184
    assert summary["total_runs"] == 3 and summary["total_checkpoints"] == 6
    assert summary["auxiliary_used_for_selection"] is False and summary["actionable"] is False
    assert summary["qualified_candidate"]["budget"] == "long"
    assert summary["qualified_candidate"]["representative_seed"] == 123


@pytest.mark.parametrize("change, message", [("missing", "three seeds"), ("duplicate", "three seeds"),
    ("wrong_steps", "budget"), ("wrong_capacity", "capacity"), ("test", "split"), ("baseline", "baseline values")])
def test_summary_rejects_incomplete_or_incomparable_evidence(change, message):
    rows, references = _records()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif change == "wrong_steps":
        rows[0]["steps"] += 512
    elif change == "wrong_capacity":
        rows[0]["parameters"] += 1
    elif change == "test":
        rows[0]["test_evaluated"] = True
    else:
        rows[0]["metrics"]["cash"]["baselines"]["momentum"]["cumulative_return"] += .01
    with pytest.raises(ValueError, match=message):
        study.summarize_study(rows, references)


def test_registered_file_integrity_and_path_escape(tmp_path):
    artifact = tmp_path / "audit.json"
    artifact.write_text("{}")
    files = study._registered_files(tmp_path)
    study._verify_files(tmp_path, files)
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        study._verify_files(tmp_path, files)
    with pytest.raises(ValueError, match="artifact changed"):
        study._verify_files(tmp_path, {"../outside": "fake"})


@pytest.mark.parametrize("files", [None, [], {}, "not hashes"])
def test_registered_hash_registry_cannot_be_empty_or_unstructured(tmp_path, files):
    with pytest.raises(ValueError, match="nonempty dictionary"):
        study._verify_files(tmp_path, files)


def test_initialization_cache_requires_all_registered_account_artifacts(tmp_path):
    initial = tmp_path / "initial_model"
    initial.mkdir()
    (initial / "model.zip").write_bytes(b"initialized fixture")
    data = SimpleNamespace(contract={"action_mode": "stock_delta"}, config={"seed": 42})
    folder = tmp_path / "controls" / "small_42"
    metrics = {mode: {"ppo": {"cumulative_return": 0.}} for mode in study.MODES}
    for mode in study.MODES:
        account = folder / mode
        (account / "ppo").mkdir(parents=True)
        study.write_json(account / "metrics.json", metrics[mode])
        for name in ("daily.csv", "orders.csv", "positions.parquet"):
            (account / "ppo" / name).write_bytes(b"fixture artifact")
    identity = dict(contract=data.contract, training_seed=42, evaluation_seed=20260927,
                    evaluation_modes=list(study.MODES), top_n=20, initial_model_sha256=study.file_hash(initial / "model.zip"))
    files = study._registered_files(folder)
    files.pop("cash/ppo/orders.csv")
    study.write_json(folder / "controls.json", dict(identity=identity, metrics=metrics, files=files))
    with pytest.raises(ValueError, match="required registered"):
        study.evaluate_initial(data, initial, tmp_path, "small_42")


def test_unsafe_drawdowns_produce_explicit_null_candidate():
    rows, references = _records()
    for row in rows:
        row["metrics"]["cash"]["ppo"]["max_drawdown"] = -.11
    summary = study.summarize_study(rows, references)
    assert summary["qualified_candidate"] is None
    assert not any(cell["qualified"] for cell in summary["cells"].values())
def test_pilot_cache_binds_summary_to_actual_audit_and_hyperparameters(tmp_path):
    candidate = study.PILOT_CANDIDATES[0]
    directory = tmp_path / candidate["name"]
    directory.mkdir()
    config = study.config_for(_base(), tmp_path, candidate=candidate)
    protocol = {"registered": True}
    report = _pilot(candidate, stock_sigma=math.exp(-3.))
    model = directory / "pilot_model" / "model.zip"
    model.parent.mkdir()
    model.write_bytes(b"fixture policy, never trained")
    digest = study.file_hash(model)
    audit = dict(actual_timesteps=2048, all_parameters_finite=True, optimizer_steps_min=80,
        model_sha256=digest, training_diagnostics={"tail_mean/train/approx_kl": .02,
        "last/train/gross_filled_notional": 100., "last/train/stock_sigma": math.exp(-3.)})
    study.write_json(directory / "config.json", config)
    study.write_json(directory / "report.json", report)
    study.write_json(directory / "pilot_audit.json", audit)
    study.write_json(model.with_name("metadata.json"), dict(model_sha256=digest, num_timesteps=2048, config=config))
    def marker():
        study.write_json(directory / "completed.json", dict(config_sha256=study.json_digest(config),
            protocol_sha256=study.json_digest(protocol), report=report, files=study._registered_files(directory)))
    marker()
    assert study._load_pilot(directory, config, protocol) == report
    report["tail_mean_kl"] = .001
    study.write_json(directory / "report.json", report)
    marker()
    with pytest.raises(ValueError, match="training evidence"):
        study._load_pilot(directory, config, protocol)


def test_report_exports_simple_png_and_does_not_select_auxiliary_results(tmp_path):
    from PIL import Image
    rows, references = _records()
    for row in rows:
        for account in study.MODES:
            row["auxiliary_metrics"]["40"][account] = dict(row["auxiliary_metrics"]["40"][account], cumulative_return=1000.)
    for reference in references.values():
        reference["protocol"] = {"source_hashes": {}}
    protocol = dict(limitations=["HFQ research prices are not real fills."],
                    reference_contract_differences={}, source_hashes={})
    summary = study.write_report(tmp_path, rows, references, protocol)
    assert summary["cells"]["short"]["metrics"]["cumulative_return"]["mean"] == pytest.approx(.14)
    assert summary["auxiliary_used_for_selection"] is False
    assert study.read_config(tmp_path / "study_summary.json")["test_evaluated"] is False
    assert "not isolate the causal" in (tmp_path / "report.md").read_text()
    image = Image.open(tmp_path / "comparison.png").convert("RGB")
    assert image.size == (1350, 750)
    assert len(image.resize((90, 50)).getcolors(4500)) > 30


@pytest.mark.parametrize("stage", ["prepare", "pilot", "train"])
def test_cli_requires_explicit_stage_and_routes_once_without_automatic_training(tmp_path, monkeypatch, stage):
    protocol = {"mock": True}
    calls = []
    monkeypatch.setattr(study, "read_config", lambda path: _base())
    monkeypatch.setattr(study, "prepare_study", lambda *args: protocol)
    monkeypatch.setattr(study.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(study, "run_pilots", lambda *args: calls.append("pilot"))
    monkeypatch.setattr(study, "run_formal", lambda *args: calls.append("train"))
    study.main(["--stage", stage, "--config", "base.json", "--output", str(tmp_path),
                "--prior-root", "prior", "--random-root", "random"])
    assert calls == ([] if stage == "prepare" else [stage])
