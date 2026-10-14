"""Timing runner synthetic support, controls and fixed-selection arithmetic."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.agents.timing_agent import TimingAgent
from src.data.preprocess import write_json, read_config, file_hash
from src.envs.initial_account import MODES
from src.training import research_timing as study
from src.training.freeze import json_digest
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from tests.test_ppo_framework import local_data
from tests.test_trade_policy import trade_config


pytestmark = pytest.mark.usefixtures("no_parameter_updates")


@pytest.fixture(autouse=True)
def no_parameter_updates(request, monkeypatch):
    if request.node.name == "test_synthetic_final_train_cache_and_final_only_snapshot":
        return
    def forbidden(*args, **kwargs):
        raise AssertionError("Only the explicitly authorized eight-step synthetic integration may update parameters")
    monkeypatch.setattr(study.PPO, "learn", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def timing_config(local_data, method="log_net"):
    config = trade_config(local_data)
    config.update(action_mode="timing_control", top_n=20, top_n_choices=[], hidden_sizes=[64, 64],
        initial_portfolio="mixed", training_episode_steps=8, recent_episode_every=0, device="cpu", torch_threads=1,
        exploration=dict(architecture="split", common_lookback=5, sampling="recent", reward_basis="log_net", size_annealing=False),
        timing=dict(fold="2023", phase="screen", reward_basis=method, calendar_margin=2., period=20))
    return config


@pytest.mark.parametrize("phase,fold,seed", [("screen", "2023", 42), ("screen", "2024", 42),
    ("confirm", "2025", 123), ("confirm", "2026", 2026)])
@pytest.mark.parametrize("method", study.METHODS)
def test_config_registers_only_low_dim_timing_and_common_ppo(phase, fold, seed, method, tmp_path):
    base = {"ppo": {}}
    before = deepcopy(base)
    config = study.config_for(base, tmp_path, {"research_config": "immutable.json"}, fold, method, phase, seed)
    assert base == before
    assert config["action_mode"] == "timing_control"
    assert config["timing"]["reward_basis"] == method
    assert config["ppo"]["gamma"] == .995 and config["ppo"]["learning_rate"] == 3e-4
    assert config["ppo"]["n_steps"] == 512 and config["ppo"]["n_epochs"] == 5
    assert config["training_episode_steps"] == 240 and config["top_n_choices"] == []
    assert config["hidden_sizes"] == [64, 64] and config["device"] == "cpu"


@pytest.mark.parametrize("phase,fold,seed,method", [("screen", "2025", 42, "log_net"),
    ("screen", "2023", 123, "log_net"), ("confirm", "2024", 42, "log_net"),
    ("confirm", "2026", 42, "teacher"), ("pilot", "2023", 42, "log_net")])
def test_unregistered_conditions_fail_before_files(phase, fold, seed, method, tmp_path):
    with pytest.raises(ValueError, match="Unregistered"):
        study.config_for({}, tmp_path, {}, fold, method, phase, seed)


@pytest.mark.parametrize("method", study.METHODS)
def test_new_data_physically_crops_providers_and_keeps_original_artifacts(local_data, method):
    config = timing_config(local_data, method)
    data = study.TimingData(config)
    borrowed = deepcopy(config)
    borrowed["action_mode"] = "stock_trade"
    old = study.ExplorationData(borrowed)
    pd.testing.assert_frame_equal(data.features, old.features)
    pd.testing.assert_frame_equal(data.prices, old.prices)
    assert data.contract["artifacts"] == old.contract["artifacts"]
    assert data.contract["action_mode"] == "timing_control"
    assert len(data.contract["timing"]["feature_names"]) == 129
    env = data.env("train")
    try:
        vector, info = env.reset(seed=42)
        assert vector.shape == (129,) and env.action_space.n == 2
        assert env.env.unwrapped.market.panel.index.max() <= pd.Timestamp(data.manifest["train_end"])
        assert env.benchmark.unwrapped.market.panel.index.max() <= pd.Timestamp(data.manifest["train_end"])
        assert env.env.unwrapped.market.sampler is None
        assert env.observation_space.contains(vector)
    finally:
        env.close()


@pytest.mark.parametrize("method", study.METHODS)
@pytest.mark.parametrize("mode", MODES)
def test_calendar_init_dual_books_and_legacy_baseline_have_exact_ledgers(local_data, tmp_path, method, mode):
    data = study.TimingData(timing_config(local_data, method))
    data.settings["initial_portfolio"] = mode
    env = data.env("val")
    model = study.make_model(env, data.config)
    agent = TimingAgent(model, data.contract)
    try:
        result = study.evaluate_case(agent, data, tmp_path / "case", mode, "initialized")
        assert result["metrics"] == result["benchmark_metrics"]
        data.settings.update(seed=study.EVAL_SEED, initial_portfolio=mode, trade_top_n=20)
        daily, positions, _, _ = study.run_policy(data.market("val"), study.EconomicRule(data.stock_contract, "momentum_20", 20), data.settings)
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "case" / "positions.parquet"), positions)
        np.testing.assert_allclose(pd.read_csv(tmp_path / "case" / "daily.csv", float_precision="round_trip").equity, daily.equity, rtol=0., atol=0.)
        assert result["evaluation_steps"] == result["benchmark_steps"] == 30
    finally:
        env.close()


def metric(roi=.02, dd=-.03, cost=1.):
    return dict(cumulative_return=roi, max_drawdown=dd, total_fees=cost, total_slippage_cost=0.,
                annual_volatility=.1, sharpe=1., gross_turnover=1., mean_cash_ratio=.5)


def record(fold, method, phase, seed, gap=.01, dd=-.03, baseline_dd=-.03, cost=1.):
    baseline = metric(.02, baseline_dd, 1.)
    values = {mode: dict(trained=metric(.02 + gap, dd, cost), initialized=deepcopy(baseline), benchmark=deepcopy(baseline)) for mode in MODES}
    return dict(fold=fold, method=method, phase=phase, seed=seed, steps=study.BUDGETS[phase], metrics=values)


def screen():
    return [record(fold, method, "screen", 42) for fold in study.SCREEN_FOLDS for method in study.METHODS]


def confirmation(gap=.01):
    return [record(fold, "log_net", "confirm", seed, gap) for fold in study.CONFIRM_FOLDS for seed in study.SEEDS]


def test_screen_selection_uses_worst_fold_before_best_average():
    rows = screen()
    for row in rows:
        if row["method"] == "log_net":
            for mode in MODES:
                row["metrics"][mode]["trained"]["cumulative_return"] = .04 if row["fold"] == "2023" else .0
        else:
            for mode in MODES:
                row["metrics"][mode]["trained"]["cumulative_return"] = .025
    chosen = study.select_screen(rows)
    assert chosen["selected_method"] == "relative_log"
    assert chosen["actual_training_steps"] == 32768


def test_screen_exact_tie_uses_fixed_method_name():
    assert study.select_screen(screen())["selected_method"] == "log_net"


@pytest.mark.parametrize("bad", ["missing", "duplicate", "shorter", "nonfinite", "account", "initialization"])
def test_stage_validation_rejects_partial_or_changed_evidence(bad):
    rows = screen()
    if bad == "missing":
        rows.pop()
    elif bad == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif bad == "shorter":
        rows[0]["steps"] = 2048
    elif bad == "nonfinite":
        rows[0]["metrics"][MODES[0]]["trained"]["cumulative_return"] = np.nan
    elif bad == "account":
        rows[0]["metrics"].pop(MODES[0])
    else:
        rows[0]["metrics"][MODES[0]]["initialized"]["cumulative_return"] += .001
    with pytest.raises(ValueError):
        study.select_screen(rows)


def test_confirmation_gate_paired_seed_sd_and_budget():
    rows = confirmation()
    for row in rows:
        for mode in MODES:
            row["metrics"][mode]["trained"]["cumulative_return"] += {42: 0., 123: .01, 2026: -.005}[row["seed"]]
    summary = study.summarize_confirmation(rows, "log_net")
    assert summary["actual_training_steps"] == 98304 and summary["final_snapshots"] == 6
    assert summary["qualified"] and summary["qualified_candidate"]["representative_seed"] == 42
    np.testing.assert_allclose(summary["paired_across_fold_gap"]["mean"], np.mean([.01, .02, .005]))
    np.testing.assert_allclose(summary["paired_across_fold_gap"]["std"], np.std([.01, .02, .005], ddof=1))


@pytest.mark.parametrize("bad", ["gap_threshold", "negative_fold", "seed_count", "absolute_dd", "excess_dd"])
def test_confirmation_does_not_promote_failed_gate(bad):
    rows = confirmation(.004 if bad == "gap_threshold" else .01)
    for row in rows:
        for mode in MODES:
            if bad == "negative_fold" and row["fold"] == "2025":
                row["metrics"][mode]["trained"]["cumulative_return"] = .019
            if bad == "seed_count" and row["seed"] != 42:
                row["metrics"][mode]["trained"]["cumulative_return"] = .019
            if bad == "absolute_dd":
                row["metrics"][mode]["trained"]["max_drawdown"] = -.101
            if bad == "excess_dd":
                row["metrics"][mode]["trained"]["max_drawdown"] = -.051
    summary = study.summarize_confirmation(rows, "log_net")
    assert not summary["qualified"] and summary["qualified_candidate"] is None


def test_leaf_verification_rejects_unregistered_added_file(tmp_path):
    write_json(tmp_path / "result.json", {"ok": True})
    registered = study._leaf_files(tmp_path)
    study.verify_leaf_files(tmp_path, registered)
    write_json(tmp_path / "extra.json", {"ok": True})
    with pytest.raises(ValueError, match="leaf set"):
        study.verify_leaf_files(tmp_path, registered)


def test_report_and_confirm_cannot_create_missing_trials(tmp_path):
    write_json(tmp_path / "protocol.json", {})
    with pytest.raises((ValueError, KeyError)):
        study._load_stage(tmp_path, {}, "screen")
    assert not (tmp_path / "runs").exists()


def test_discrete_callback_captures_sampled_macro_and_standard_buffer_without_updates(local_data, tmp_path):
    config = timing_config(local_data, "relative_log")
    config["ppo"].update(n_steps=8, batch_size=4, n_epochs=1)
    data = study.TimingData(config)
    experience = study.TimingExperience(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    env.seed(42)
    model = study.make_model(env, config)
    initial = {name: value.detach().clone() for name, value in model.policy.state_dict().items()}
    callback = study.TimingCallback(data, tmp_path, experience, initial, 8, "synthetic")
    try:
        _, prepared = model._setup_learn(8, callback)
        assert model.collect_rollouts(model.env, prepared, model.rollout_buffer, n_rollout_steps=8)
        callback.trace.close()
        study.verify_rollouts(tmp_path, callback.rollouts, 8)
        study.verify_timing_trace(tmp_path / "training_trace", callback.trace.manifest(), steps=8)
        assert study._optimizer_counts(model) == {}
        assert experience.stats()["total_steps"] == experience.stats()["benchmark_steps"] == 8
        assert experience.stats()["completed_episodes"] == 1
        with np.load(tmp_path / callback.rollouts[0]["path"], allow_pickle=False) as buffer:
            with np.load(tmp_path / "training_trace" / callback.trace.chunks[0]["path"], allow_pickle=False) as trace:
                np.testing.assert_array_equal(trace["action_macro_action"], buffer["actions"].reshape(-1))
                np.testing.assert_allclose(trace["action_macro_log_prob"], buffer["log_probs"].reshape(-1), atol=0., rtol=0.)
                np.testing.assert_allclose(trace["action_macro_value"], buffer["values"].reshape(-1), atol=0., rtol=0.)
        for name, value in model.policy.state_dict().items():
            torch.testing.assert_close(value, initial[name], atol=0., rtol=0.)
    finally:
        callback.trace.close()
        env.close()


@pytest.mark.parametrize("bad", ["count", "hash", "action", "targets"])
def test_rollout_verifier_rejects_corrupt_categorical_evidence(tmp_path, bad):
    size = 8
    vector = np.zeros((size, 1, 129), np.float32)
    vector[:, :, 0] = 1
    arrays = dict(observations=vector, actions=np.zeros((size, 1, 1)), rewards=np.zeros((size, 1)),
                  values=np.zeros((size, 1)), log_probs=np.zeros((size, 1)), returns=np.zeros((size, 1)),
                  advantages=np.zeros((size, 1)), episode_starts=np.zeros((size, 1)))
    if bad == "action":
        arrays["actions"][0] = .5
    elif bad == "targets":
        arrays["returns"][0] = 1.
    path = tmp_path / "rollout.npz"
    np.savez_compressed(path, **arrays)
    signed = dict(path=path.name, sha256="0" * 64 if bad == "hash" else file_hash(path), first_step=1, last_step=8, steps=8)
    with pytest.raises(ValueError):
        study.verify_rollouts(tmp_path, [signed], 9 if bad == "count" else 8)


def test_synthetic_final_train_cache_and_final_only_snapshot(local_data, tmp_path, monkeypatch):
    """Eight artificial steps, two actual Adam updates; never real market data."""
    root = tmp_path / "study"
    root.mkdir()
    config = timing_config(local_data, "relative_log")
    config["ppo"].update(n_steps=8, batch_size=4, n_epochs=1)
    protocol = dict(source_hashes=study.source_hashes(), parameters=12805, feature_names=list(study.FEATURE_NAMES), git={"synthetic": True})
    write_json(root / "protocol.json", protocol)
    monkeypatch.setattr(study, "check_frozen", lambda value: None)
    monkeypatch.setattr(study, "trial_config", lambda *args: deepcopy(config))
    monkeypatch.setattr(study, "BUDGETS", {"screen": 8, "confirm": 16})
    monkeypatch.setattr(study, "ROOT", tmp_path)
    row = study.run_trial(root, protocol, "2023", "relative_log", "screen", 42)
    directory = root / "runs" / "screen_2023_relative_log_42"
    assert row["steps"] == row["audit"]["experience"]["total_steps"] == 8
    assert row["audit"]["actual_optimizer_updates"] > 0
    assert max(row["audit"]["tensor_delta_l2"].values()) > 0
    assert {path.name for path in directory.glob("*_model")} == {"initial_model", "final_model"}
    assert study.cached_trial(directory, config, protocol) == row
    # A resign of the leaf hashes must not make an impossible algorithm valid.
    metadata_path = directory / "final_model" / "metadata.json"
    metadata = read_config(metadata_path)
    metadata["algorithm_class"] = "fake.PPO"
    write_json(metadata_path, metadata)
    completed = read_config(directory / "completed.json")
    completed["files"] = study._leaf_files(directory)
    write_json(directory / "completed.json", completed)
    with pytest.raises(ValueError, match="model identity"):
        study.cached_trial(directory, config, protocol)


@pytest.mark.parametrize("phase,fold,method,expected", [("screen", "2023", None, 2), ("confirm", "2026", "log_net", 3)])
def test_worker_receipts_require_exact_registered_final_trials(tmp_path, phase, fold, method, expected):
    write_json(tmp_path / "protocol.json", {"registered": True})
    if phase == "confirm":
        write_json(tmp_path / "selection.json", {"selected_method": method})
    candidates, seeds = (study.METHODS, (42,)) if phase == "screen" else ((method,), study.SEEDS)
    completed = []
    for candidate in candidates:
        for seed in seeds:
            run_id = f'{phase}_{fold}_{candidate}_{seed}'
            path = tmp_path / "runs" / run_id / "completed.json"
            path.parent.mkdir(parents=True)
            write_json(path, {"synthetic_marker": True})
            completed.append(dict(run_id=run_id, completed_sha256=file_hash(path)))
    assert len(completed) == expected
    receipt = dict(status="complete", phase=phase, fold=fold, protocol_sha256=file_hash(tmp_path / "protocol.json"),
        selection_sha256=file_hash(tmp_path / "selection.json") if phase == "confirm" else None,
        completed_runs=completed, no_global_summary_writes=True)
    path = tmp_path / f'worker_{phase}_{fold}.json'
    write_json(path, receipt)
    assert study.verify_worker_receipt(tmp_path, {}, phase, fold, method) == receipt
    receipt["status"] = "running"
    write_json(path, receipt)
    with pytest.raises(ValueError, match="workers"):
        study.verify_worker_receipt(tmp_path, {}, phase, fold, method)


def test_prepare_creates_only_registration_and_never_constructs_a_model(tmp_path, monkeypatch):
    reference, folds_root, original = (tmp_path / name for name in ("reference", "folds", "original"))
    for directory in (reference, folds_root, original):
        directory.mkdir()
    write_json(reference / "protocol.json", {"ancestor": True})
    old = {"folds": {}}
    for fold in (*study.SCREEN_FOLDS, *study.CONFIRM_FOLDS):
        directory = folds_root / fold if fold != "2026" else original / "data"
        directory.mkdir(parents=True)
        manifest = dict(train_end=fold + "-03-31", val_end=fold + "-06-30",
                        end_date="2026-09-24" if fold == "2026" else "2026-06-30", protected_period_excluded=fold != "2026")
        write_json(directory / "manifest.json", manifest)
        for name in ("panel_data.parquet", "prices.parquet", "scaler_params.json"):
            (directory / name).write_bytes(b"synthetic registration fixture, not parquet")
        research_path = directory / "research.json" if fold != "2026" else original / "research.json"
        write_json(research_path, {"dataset": {"clean_dir": str(directory)}})
        if fold != "2026":
            old["folds"][fold] = dict(directory=str(directory), manifest_sha256=file_hash(directory / "manifest.json"),
                                      artifacts={name: file_hash(directory / name) for name in ("panel_data.parquet", "prices.parquet", "scaler_params.json", "research.json")})
    write_json(folds_root / "protocol.json", old)
    document = tmp_path / "protocol.md"
    document.write_text("synthetic preregistration", encoding="utf-8")
    monkeypatch.setattr(study, "DOCUMENT", document)
    monkeypatch.setattr(study, "_trusted_reference", lambda path: {"base_config": {"ppo": {}}, "source_hashes": {"old.py": "old"}})
    monkeypatch.setattr(study, "check_frozen", lambda protocol: None)
    fake = SimpleNamespace(contract={"data_source": "sina"}, codes=[str(index) for index in range(300)],
                           manifest={"feature_cols": list(study.FEATURE_NAMES[20:40])}, stock_contract={"synthetic": True})
    monkeypatch.setattr(study, "TimingData", lambda config: fake)
    monkeypatch.setattr(study, "make_model", lambda *args: pytest.fail("prepare must not construct a model"))
    root = tmp_path / "study"
    protocol = study.prepare(root, reference, folds_root, original / "research.json")
    assert protocol["planned_actual_training_steps"] == 131072 and protocol["planned_final_snapshots"] == 10
    assert protocol["parameters"] == 12805
    assert (root / "protocol.json").exists() and not (root / "runs").exists()
    assert study.prepare(root, reference, folds_root, original / "research.json") == protocol


@pytest.mark.parametrize("bad", [None, "obsolete_schema", "source_removed", "extra_source"])
def test_reference_uses_latest_signed_files_schema_and_exact_old_91_inventory(tmp_path, monkeypatch, bad):
    current = study.source_hashes()
    added = {str(study.Path(name)) for name in study.NEW_SOURCE_FILES}
    original = {name: digest for name, digest in current.items() if name not in added}
    assert len(original) == 91
    if bad == "source_removed":
        original.pop(next(iter(original)))
    protocol = dict(source_hashes=original, base_config={"ppo": {}})
    write_json(tmp_path / "protocol.json", protocol)
    audit = dict(status="passed", signed_files_sha256={"protocol.json": file_hash(tmp_path / "protocol.json")})
    if bad == "obsolete_schema":
        audit["anchors"] = audit.pop("signed_files_sha256")
    write_json(tmp_path / "final_audit.json", audit)
    monkeypatch.setattr(study, "REFERENCE_AUDIT_SHA256", file_hash(tmp_path / "final_audit.json"))
    if bad == "extra_source":
        monkeypatch.setattr(study, "source_hashes", lambda: dict(current, unregistered="fake"))
    if bad is None:
        assert study._trusted_reference(tmp_path) == protocol
    else:
        with pytest.raises(ValueError):
            study._trusted_reference(tmp_path)
