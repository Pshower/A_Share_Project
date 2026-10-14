"""Independent admission checks on synthetic files and an unoptimized rollout."""

from copy import deepcopy
import json
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.data.preprocess import DataPreprocessor, FEATURES, file_hash, read_config, write_json
from src.training import research_timing as study
from src.training.freeze import json_digest
from src.training.timing_env import FEATURE_NAMES, build_timing_contract
from tests.test_ppo_framework import local_data, config as framework_config
from tests.test_research_pipeline import fixture_config, raw_source
from tests.test_research_timing import timing_config, screen, record
from tests.test_trade_policy import trade_config


@pytest.fixture(autouse=True)
def no_updates(monkeypatch):
    def prohibited(*args, **kwargs):
        raise AssertionError("Independent review never updates neural parameters")
    monkeypatch.setattr(study.PPO, "learn", prohibited)
    monkeypatch.setattr(torch.Tensor, "backward", prohibited)
    monkeypatch.setattr(torch.optim.Adam, "step", prohibited)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_real_old_91_reference_reads_the_actual_signed_file_schema_without_prepare():
    reference = study._trusted_reference(study.DEFAULT_REFERENCE)
    assert len(reference["source_hashes"]) == 91
    assert file_hash(study.DEFAULT_REFERENCE / "final_audit.json") == study.REFERENCE_AUDIT_SHA256
    assert len(study.source_hashes()) == 96


def test_unregistered_additional_source_is_rejected_without_editing_any_real_source(monkeypatch):
    sources = study.source_hashes()
    monkeypatch.setattr(study, "source_hashes", lambda: dict(sources, **{"src/unregistered.py": "0" * 64}))
    with pytest.raises(ValueError):
        study._trusted_reference(study.DEFAULT_REFERENCE)


@pytest.mark.parametrize("phase", ["screen", "confirm"])
def test_report_stage_requires_complete_workers_and_never_starts_a_trial(tmp_path, monkeypatch, phase):
    write_json(tmp_path / "protocol.json", {"synthetic": True})
    if phase == "confirm":
        write_json(tmp_path / "selection.json", {"selected_method": "log_net"})
    monkeypatch.setattr(study, "run_trial", lambda *args, **kwargs: pytest.fail("A report must not train"))
    with pytest.raises(ValueError):
        study._load_stage(tmp_path, {}, phase, "log_net" if phase == "confirm" else None)
    assert not (tmp_path / "runs").exists()


def selection_fixture(tmp_path, monkeypatch):
    protocol = {"synthetic": True}
    write_json(tmp_path / "protocol.json", protocol)
    rows = screen()
    originals = {}
    for row in rows:
        name = f'{row["fold"]}_{row["method"]}'
        path = tmp_path / "runs" / f'screen_{name}_42' / "completed.json"
        path.parent.mkdir(parents=True)
        write_json(path, {"record": row, "protocol_sha256": json_digest(protocol)})
        originals[name] = file_hash(path)
    selection = dict(protocol_sha256=json_digest(protocol), **study.select_screen(rows),
                     screen_completed_sha256=originals)
    write_json(tmp_path / "selection.json", selection)
    monkeypatch.setattr(study, "verify_worker_receipt", lambda *args, **kwargs: {"status": "complete"})
    return protocol, selection, rows


def test_confirm_selection_accepts_only_registered_complete_screen_choice(tmp_path, monkeypatch):
    protocol, _, _ = selection_fixture(tmp_path, monkeypatch)
    digest = file_hash(tmp_path / "selection.json")
    study.validate_confirmation_selection(tmp_path, protocol, "log_net", digest)
    with pytest.raises(ValueError):
        study.validate_confirmation_selection(tmp_path, protocol, "relative_log", digest)


@pytest.mark.parametrize("bad", ["method", "selection_hash", "screen_hash", "choice", "partial"])
def test_confirmation_selection_rejects_resigned_or_partial_screen(tmp_path, monkeypatch, bad):
    protocol, selection, _ = selection_fixture(tmp_path, monkeypatch)
    method = "relative_log" if bad == "method" else "log_net"
    if bad == "screen_hash":
        selection["screen_completed_sha256"]["2023_log_net"] = "0" * 64
    elif bad == "choice":
        selection["selected_method"] = "relative_log"
        method = "relative_log"
    elif bad == "partial":
        selection["screen_completed_sha256"].pop("2024_relative_log")
    write_json(tmp_path / "selection.json", selection)
    digest = "0" * 64 if bad == "selection_hash" else file_hash(tmp_path / "selection.json")
    with pytest.raises(ValueError):
        study.validate_confirmation_selection(tmp_path, protocol, method, digest)


def test_wrong_confirm_method_fails_before_creating_directory_or_learning(tmp_path, monkeypatch):
    protocol, _, _ = selection_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(study, "check_frozen", lambda protocol: None)
    monkeypatch.setattr(study, "run_trial", lambda *args, **kwargs: pytest.fail("Wrong method may not train"))
    with pytest.raises(ValueError):
        study.run_worker(tmp_path, protocol, "confirm", "2025", "relative_log")
    assert not (tmp_path / "worker_confirm_2025.json").exists()


def test_2026_provider_physically_excludes_artificial_protected_rows(tmp_path):
    research = fixture_config(tmp_path)
    dates = pd.bdate_range("2025-10-01", periods=260)
    for code in ("000001", "600000", "000003"):
        raw_source(code, dates).to_csv(tmp_path / "raw" / f"{code}_data_fixture_hfq.csv", index=False)
    research["dataset"].update(start_date="2025-10-01", train_end="2026-03-31",
                               val_end="2026-06-30", end_date="2026-09-24")
    DataPreprocessor(**research["dataset"]).process_all_stocks_and_save_scaler()
    path = tmp_path / "research.json"
    write_json(path, research)
    base_config = framework_config()
    base_config["research_config"] = str(path)
    config = trade_config(SimpleNamespace(config=base_config))
    config.update(research_config=str(path), action_mode="timing_control", device="cpu", torch_threads=1,
        hidden_sizes=[64, 64], top_n=20, top_n_choices=[], initial_portfolio="mixed",
        training_episode_steps=8, recent_episode_every=0,
        exploration=dict(architecture="split", common_lookback=5, sampling="recent", reward_basis="log_net"),
        timing=dict(fold="2026", phase="confirm", reward_basis="relative_log", calendar_margin=2., period=20))
    data = study.TimingData(config)
    assert data.features.index.max() >= pd.Timestamp("2026-07-01")
    for split, end in (("train", "2026-03-31"), ("val", "2026-06-30")):
        market = data.market(split)
        assert market.panel.index.max() <= pd.Timestamp(end)
        assert market.dates.max() <= pd.Timestamp(end)
        assert market.prices.index.max() <= pd.Timestamp(end)
        assert not (market.panel.index >= pd.Timestamp("2026-07-01")).any()


def synthetic_rollout(local_data, tmp_path):
    config = timing_config(local_data, "relative_log")
    config["ppo"].update(n_steps=8, batch_size=4, n_epochs=1)
    data = study.TimingData(config)
    experience = study.TimingExperience(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    env.seed(42)
    model = study.make_model(env, config)
    initial = {name: value.detach().clone() for name, value in model.policy.state_dict().items()}
    callback = study.TimingCallback(data, tmp_path, experience, initial, 8, "independent_synthetic")
    try:
        _, initialized = model._setup_learn(8, callback)
        assert model.collect_rollouts(model.env, initialized, model.rollout_buffer, n_rollout_steps=8)
        callback.trace.close()
        manifest = callback.trace.manifest()
        study.verify_timing_trace(tmp_path / "training_trace", manifest, steps=8)
        study.verify_rollouts(tmp_path, callback.rollouts, 8, manifest)
        assert model.policy.parameter_count == 12805 and str(model.device) == "cpu"
        assert model.observation_space.shape == (129,) and model.action_space.n == 2
        assert study._optimizer_counts(model) == {}
        for name, value in model.policy.state_dict().items():
            torch.testing.assert_close(value, initial[name], atol=0., rtol=0.)
        assert experience.stats()["total_steps"] == experience.stats()["benchmark_steps"] == 8
        return callback.rollouts, manifest
    finally:
        callback.trace.close()
        env.close()


def test_actual_unoptimized_eight_step_rollout_keeps_macro_density_and_two_native_books(local_data, tmp_path):
    rollouts, manifest = synthetic_rollout(local_data, tmp_path)
    assert manifest["total_steps"] == manifest["benchmark_trace"]["total_steps"] == 8
    assert len(manifest["chunks"]) == len(manifest["benchmark_trace"]["chunks"]) == 1
    with np.load(tmp_path / rollouts[0]["path"], allow_pickle=False) as buffer:
        with np.load(tmp_path / "training_trace" / manifest["chunks"][0]["path"], allow_pickle=False) as native:
            assert native["raw_packed_action"].shape[1] == 2 * len(local_data.codes)
            assert buffer["actions"].shape == (8, 1, 1)
            np.testing.assert_array_equal(buffer["log_probs"].reshape(-1), native["action_macro_log_prob"])
            # Bootstrap-adjusted learner rewards are deliberately not the financial trace reward.
            assert rollouts[0]["reward_semantics"].startswith("PPO buffer reward")
            assert np.all(native["action_has_policy_probabilities"])


@pytest.mark.parametrize("field", ["actions", "log_probs", "values", "observations"])
def test_resigned_buffer_cannot_replace_the_actual_native_macro_sample(local_data, tmp_path, field):
    rollouts, manifest = synthetic_rollout(local_data, tmp_path)
    path = tmp_path / rollouts[0]["path"]
    with np.load(path, allow_pickle=False) as original:
        arrays = {name: original[name].copy() for name in original.files}
    if field == "actions":
        arrays[field][0, 0, 0] = 1 - arrays[field][0, 0, 0]
    elif field == "observations":
        arrays[field][0, 0, 20] += .1
    else:
        arrays[field][0, 0] += .1
        if field == "values":
            arrays["returns"] = arrays["advantages"] + arrays["values"]
    np.savez_compressed(path, **arrays)
    changed = deepcopy(rollouts)
    changed[0]["sha256"] = file_hash(path)
    with pytest.raises(ValueError):
        study.verify_rollouts(tmp_path, changed, 8, manifest)


def metadata_fixture(tmp_path):
    config = dict(device="cpu", timing=dict(fold="2023", phase="screen", reward_basis="log_net", calendar_margin=2., period=20))
    artifact = {name: str(index) * 64 for index, name in enumerate(("panel_data.parquet", "prices.parquet", "scaler_params.json"))}
    base = dict(stock_codes=["000001"], feature_cols=list(FEATURES), broker=dict(lot_size=100),
                price_basis="hfq_research", logit_bound=8., artifacts=artifact, manifest_sha256="4" * 64,
                action_mode="stock_trade", mapping_version="stock_trade_lots_hold_buy_reduce_exit_residual_cash_v4")
    contract = build_timing_contract(base, "log_net")
    contract["timing_research"] = deepcopy(config["timing"])
    record = dict(policy_class="src.agents.timing_policy.CalendarTimingPolicy", observation_shape=[129],
        observation_dtype="float32", feature_names=list(FEATURE_NAMES), action_names=["hold", "rebalance_m20"],
        net_arch={"pi": [], "vf": [64, 64]}, calendar_margin=2., phase_period=20, parameter_count=12805)
    archive = dict(timing_registered_algorithm="stable_baselines3.ppo.ppo.PPO", timing_registered_contract=contract,
        timing_feature_names=list(FEATURE_NAMES), timing_policy_record=record, num_timesteps=8192)
    directory = tmp_path / "model"
    directory.mkdir()
    with zipfile.ZipFile(directory / "model.zip", "w") as handle:
        handle.writestr("data", json.dumps(archive))
    protocol = dict(source_hashes={}, folds={"2023": dict(stock_contract_sha256=json_digest(base), files=artifact, manifest_sha256="4" * 64)})
    metadata = dict(config=config, contract=contract, num_timesteps=8192, actual_device="cpu",
        feature_names=list(FEATURE_NAMES), policy_record=record, source_hashes={}, model_sha256=file_hash(directory / "model.zip"),
        algorithm_class="stable_baselines3.ppo.ppo.PPO")
    return directory, metadata, config, protocol, contract


def test_static_snapshot_validation_accepts_exact_cpu_categorical_schema(tmp_path):
    directory, meta, config, protocol, contract = metadata_fixture(tmp_path)
    study._validate_snapshot_metadata(directory, meta, config, protocol, contract, 8192)


@pytest.mark.parametrize("bad", ["device", "parameters", "shape", "architecture", "names", "margin", "algorithm"])
def test_same_initial_and_final_metadata_does_not_validate_a_wrong_policy(tmp_path, bad):
    directory, meta, config, protocol, contract = metadata_fixture(tmp_path)
    if bad == "device":
        meta["actual_device"] = "cuda:0"
    elif bad == "names":
        meta["feature_names"][-1] = "wrong"
    elif bad == "algorithm":
        meta["algorithm_class"] = "fake.PPO"
    else:
        field, value = {"parameters": ("parameter_count", 12806), "shape": ("observation_shape", [109]),
                        "architecture": ("net_arch", {"pi": [64], "vf": [64, 64]}),
                        "margin": ("calendar_margin", 3.)}[bad]
        meta["policy_record"][field] = value
    with pytest.raises(ValueError):
        study._validate_snapshot_metadata(directory, meta, config, protocol, contract, 8192)


def test_validation_dates_use_actual_panel_rows_and_final_training_close_only():
    dates = pd.DatetimeIndex(["2026-03-27", "2026-03-31", "2026-04-01", "2026-04-07", "2026-06-30", "2026-07-01"])
    data = SimpleNamespace(features=pd.DataFrame(index=dates),
                           manifest=dict(train_end="2026-03-31", val_end="2026-06-30"))
    expected = dates[1:-1]
    pd.testing.assert_index_equal(study.validation_dates(data), expected)
    data.manifest["val_end"] = "2026-07-01"
    with pytest.raises(ValueError):
        study.validation_dates(data)


@pytest.mark.parametrize("bad", ["daily", "benchmark", "positions", "decision", "execution", "shorter"])
def test_complete_q2_date_arrays_cannot_be_shortened_or_shifted(tmp_path, bad):
    expected = pd.DatetimeIndex(["2026-03-31", "2026-04-01", "2026-04-07", "2026-06-30"])
    for name in ("daily.csv", "benchmark_daily.csv"):
        pd.DataFrame({"date": expected}).to_csv(tmp_path / name, index=False)
    positions = pd.DataFrame(0., index=expected, columns=["000001"])
    positions.to_parquet(tmp_path / "positions.parquet")
    trace = tmp_path / "trace"
    trace.mkdir()
    path = trace / "steps.npz"
    native = dict(decision_date=expected[:-1].to_numpy(), execution_date=expected[1:].to_numpy())
    np.savez_compressed(path, **native)
    result = dict(evaluation_steps=3, benchmark_steps=3, first_execution="2026-04-01", last_execution="2026-06-30",
                  trace=dict(chunks=[dict(path=path.name)]))
    pd.testing.assert_index_equal(study.verify_evaluation_dates(tmp_path, result, expected), expected)
    changed = expected.to_numpy(copy=True)
    changed[2] += np.timedelta64(1, "D")
    if bad in ("daily", "benchmark"):
        pd.DataFrame({"date": changed}).to_csv(tmp_path / ("daily.csv" if bad == "daily" else "benchmark_daily.csv"), index=False)
    elif bad == "positions":
        positions.index = changed
        positions.to_parquet(tmp_path / "positions.parquet")
    elif bad in ("decision", "execution"):
        name = "decision_date" if bad == "decision" else "execution_date"
        native[name][1] += np.timedelta64(1, "D")
        np.savez_compressed(path, **native)
    else:
        result.update(evaluation_steps=2, benchmark_steps=2)
    with pytest.raises(ValueError):
        study.verify_evaluation_dates(tmp_path, result, expected)
