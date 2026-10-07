"""Independent synthetic attribution checks; never learn or run an optimizer."""

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.ppo_agent import PPOAgent
from src.data.preprocess import file_hash, read_config, write_json
from src.training import research_attribution as runner
from src.training.freeze import json_digest
from src.training.matrix_experience import EpisodeAudit
from tests.test_ppo_framework import local_data, no_parameter_updates
from tests.test_trade_policy import trade_config


pytestmark = pytest.mark.usefixtures("no_parameter_updates")


def config(local_data, cell="pure_shared", fold="2023", seed=42):
    value = trade_config(local_data)
    value.update(seed=seed, torch_threads=1, actor_output_gain=.1,
        training_episode_steps=8, recent_episode_every=4, initial_portfolio="mixed",
        exploration=dict(architecture="split", common_lookback=5, sampling="recent",
                         reward_basis="simple_net", size_annealing=False),
        attribution=dict(runner.CELLS[cell], fold=fold, gradient_batch_size=128,
                         gradient_steps=[2048, 4096], gradient_phases=["pre_update", "post_update"]))
    value["ppo"].update(n_steps=128, batch_size=128, n_epochs=1)
    return value


def build(local_data, cell="pure_shared", fold="2023", seed=42):
    data = runner.AttributionData(config(local_data, cell, fold, seed))
    vector = data.vector_env()
    model = runner.make_model(vector, data.config)
    return data, vector, model


@pytest.mark.parametrize("cell", ["pure_shared", "fused_detached"])
def test_callback_pre_post_uses_same_original_targets_after_live_buffer_changes(
        local_data, tmp_path, monkeypatch, cell):
    data = runner.AttributionData(config(local_data, cell))
    experience = EpisodeAudit(data.env("train"))
    vector = DummyVecEnv([lambda: Monitor(experience)])
    callback = None
    try:
        model = runner.make_model(vector, data.config)
        model.set_logger(configure(str(tmp_path / "logs"), ["csv"]))
        initial = {name: value.clone() for name, value in model.policy.state_dict().items()}
        callback = runner.AttributionCallback(data, tmp_path, experience, initial,
            dict(short=128, long=256), tmp_path / "status.json", "synthetic_no_learning")
        # Snapshot persistence is tested separately; these counters are synthetic.
        monkeypatch.setattr(callback, "save_fixed_snapshot", lambda: None)
        model._setup_learn(total_timesteps=256, callback=callback)
        old_batches = []
        for epoch in (1, 2):
            assert model.collect_rollouts(vector, callback, model.rollout_buffer, n_rollout_steps=128)
            assert callback.gradient_records[-1]["phase"] == "pre_update"
            step = model.num_timesteps
            path = tmp_path / "gradient_batches" / f"batch_{step:06d}.npz"
            before = deepcopy(callback.gradient_batch)
            before_sha = file_hash(path)
            # SB3 resets/reuses its live buffer before the next rollout callback.
            model.rollout_buffer.returns.fill(999.)
            model.rollout_buffer.advantages.fill(-999.)
            model.rollout_buffer.log_probs.fill(-777.)
            model.rollout_buffer.observations["stock_features"].fill(555.)
            # This is a fabricated epoch counter, not a PPO/Adam update.
            model._n_updates = epoch
            callback.record_update()
            assert callback.gradient_batch is callback.gradient_step is None
            assert file_hash(path) == before_sha
            pair = callback.gradient_records[-2:]
            assert [item["phase"] for item in pair] == ["pre_update", "post_update"]
            assert pair[0]["batch_sha256"] == pair[1]["batch_sha256"] == before_sha
            assert pair[0]["batch_path"] == pair[1]["batch_path"]
            assert pair[0]["losses"] == pair[1]["losses"]
            with np.load(path, allow_pickle=False) as arrays:
                for name in ("actions", "old_log_prob", "advantages", "returns"):
                    np.testing.assert_array_equal(arrays[name], before[name])
                for name, values in before["observations"].items():
                    np.testing.assert_array_equal(arrays["obs_" + name], values)
            old_batches.append(before_sha)
        assert len(callback.gradient_records) == 4
        assert len(list((tmp_path / "gradient_batches").glob("*.npz"))) == 2
        assert len(set(old_batches)) == 2
        assert not model.policy.optimizer.state
        for name, value in model.policy.state_dict().items():
            torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    finally:
        if callback is not None:
            callback.trace.close()
        vector.close()


@pytest.mark.parametrize("wrong_cell", ["fused_shared", "pure_detached"])
def test_loader_rejects_policy_flags_hidden_behind_matching_legacy_contract(
        local_data, tmp_path, wrong_cell):
    data, vector, model = build(local_data, wrong_cell)
    try:
        expected_data = runner.AttributionData(config(local_data, "pure_shared"))
        # Legacy PPOAgent does not validate these new research-only flags.
        PPOAgent(model, expected_data.contract).save(tmp_path / "bad", data.config)
        with pytest.raises(ValueError, match="routing/gradient"):
            runner.load_agent(tmp_path / "bad", expected_data)
    finally:
        vector.close()


@pytest.mark.parametrize("wrong_cell", ["fused_shared", "pure_detached"])
def test_loader_rejects_saved_metadata_flags_even_when_policy_flags_are_correct(
        local_data, tmp_path, wrong_cell):
    data, vector, model = build(local_data, "pure_shared")
    try:
        wrong_data = runner.AttributionData(config(local_data, wrong_cell))
        # Full contract equality must reject this before checking policy flags.
        PPOAgent(model, wrong_data.contract).save(tmp_path / "bad_metadata", data.config)
        with pytest.raises(ValueError):
            runner.load_agent(tmp_path / "bad_metadata", data)
    finally:
        vector.close()


def test_only_critic_flag_can_be_removed_from_initial_control_cache_identity(
        local_data, tmp_path, monkeypatch):
    calls = []

    def synthetic_evaluation(agent, data, folder, label):
        folder = Path(folder)
        folder.mkdir(parents=True)
        metric = dict(cumulative_return=.01, max_drawdown=-.01)
        write_json(folder / "result.json", dict(metrics=metric, synthetic_only=True))
        calls.append(label)
        return dict(metrics=metric)

    monkeypatch.setattr(runner, "evaluate_case", synthetic_evaluation)
    protocol = dict(synthetic_only=True)
    vectors = []
    try:
        for cell, fold, seed, expected_calls in (
            ("pure_shared", "2023", 42, 4),
            ("pure_detached", "2023", 42, 4),
            ("fused_shared", "2023", 42, 8),
            ("pure_shared", "2024", 42, 12),
            ("pure_shared", "2023", 123, 16),
        ):
            data, vector, model = build(local_data, cell, fold, seed)
            vectors.append(vector)
            directory = tmp_path / "models" / f"{fold}_{cell}_{seed}"
            PPOAgent(model, data.contract).save(directory, data.config)
            _, control = runner.initial_controls(tmp_path, protocol, data, directory, fold, cell, seed)
            marker = read_config(control / "completed.json")
            canonical = deepcopy(data.contract)
            canonical["attribution"].pop("critic_features")
            assert marker["identity"]["prediction_contract"] == canonical
            assert len(calls) == expected_calls
            if cell == "pure_detached":
                shared = tmp_path / "models" / "2023_pure_shared_42"
                assert runner._policy_bytes(directory) == runner._policy_bytes(shared)
    finally:
        for vector in vectors:
            vector.close()


def test_prepare_rejects_modified_previously_registered_fold(tmp_path, monkeypatch):
    reference = tmp_path / "reference"
    reference.mkdir()
    folds = {}
    for fold in runner.FOLDS:
        directory = reference / "folds" / fold
        directory.mkdir(parents=True)
        for name in ("panel_data.parquet", "prices.parquet", "scaler_params.json", "research.json"):
            (directory / name).write_bytes(b"synthetic registered data")
        write_json(directory / "manifest.json", dict(end_date="2026-06-30", train_end=f"{fold}-03-31"))
        folds[fold] = dict(directory=str(directory), manifest_sha256=file_hash(directory / "manifest.json"),
            artifacts={name: file_hash(directory / name) for name in
                       ("panel_data.parquet", "prices.parquet", "scaler_params.json", "research.json")})
    write_json(reference / "protocol.json", dict(source_hashes={}, folds=folds, base_config={}))
    write_json(reference / "final_audit.json", dict(status="passed",
        anchors={"protocol.json": file_hash(reference / "protocol.json")}))
    monkeypatch.setattr(runner, "REFERENCE_AUDIT_SHA256", file_hash(reference / "final_audit.json"))
    # Alter data and its current manifest together; old registration must win.
    (reference / "folds" / "2023" / "scaler_params.json").write_bytes(b"changed after prior freeze")
    doc = tmp_path / "protocol.md"
    doc.write_text("synthetic preregistration", encoding="ascii")
    monkeypatch.setattr(runner, "DOCUMENT", doc)
    monkeypatch.setattr(runner, "source_hashes", lambda: {})
    monkeypatch.setattr(runner, "git_metadata", lambda: {})
    monkeypatch.setattr(runner, "config_for", lambda *args: {})
    env = SimpleNamespace(observation_space=SimpleNamespace(contains=lambda obs: True),
        unwrapped=SimpleNamespace(market=SimpleNamespace(stock_codes=["A"])),
        reset=lambda **kwargs: ({}, {}), close=lambda: None)
    monkeypatch.setattr(runner, "AttributionData", lambda cfg:
        SimpleNamespace(env=lambda split: env, codes=["A"]))
    with pytest.raises(ValueError):
        runner.prepare(tmp_path / "new", reference)


def gradient_manifest(tmp_path):
    """Build signed original-batch evidence without a model or optimizer."""
    folder = tmp_path / "gradient_batches"
    folder.mkdir()
    records = []
    for step in runner.BUDGETS.values():
        path = folder / f"batch_{step:06d}.npz"
        np.savez_compressed(path, obs_stock_features=np.zeros((128, 2, 1, 3), np.float32),
            actions=np.zeros((128, 4), np.float32), old_log_prob=np.zeros(128, np.float32),
            advantages=np.ones(128, np.float32), returns=np.zeros(128, np.float32))
        for phase in ("pre_update", "post_update"):
            records.append(dict(timesteps=step, phase=phase, batch_path=str(path.relative_to(tmp_path)),
                batch_sha256=file_hash(path), batch_size=128))
    return records


def test_gradient_manifest_accepts_exactly_two_signed_original_batches(tmp_path):
    records = gradient_manifest(tmp_path)
    runner.validate_gradient_manifest(tmp_path, records)


@pytest.mark.parametrize("change", ["missing_post", "duplicate_pre", "unregistered_step", "unknown_phase",
    "hash", "path", "batch_size", "batch_127", "missing_target", "missing_observation", "object_array"])
def test_gradient_manifest_rejects_incomplete_or_replaced_original_evidence(tmp_path, change):
    records = gradient_manifest(tmp_path)
    first = records[0]
    if change == "missing_post":
        records.pop(1)
    elif change == "duplicate_pre":
        records[1] = deepcopy(first)
    elif change == "unregistered_step":
        first["timesteps"] = 1234
    elif change == "unknown_phase":
        first["phase"] = "after_prediction"
    elif change == "hash":
        first["batch_sha256"] = "0" * 64
    elif change == "path":
        first["batch_path"] = "../unregistered.npz"
    elif change == "batch_size":
        first["batch_size"] = 127
    else:
        path = tmp_path / first["batch_path"]
        with np.load(path, allow_pickle=False) as batch:
            arrays = {name: value.copy() for name, value in batch.items()}
        if change == "batch_127":
            arrays["actions"] = arrays["actions"][:127]
        elif change == "missing_target":
            arrays.pop("returns")
        elif change == "missing_observation":
            arrays.pop("obs_stock_features")
        elif change == "object_array":
            arrays["actions"] = np.full((128, 4), object(), dtype=object)
        np.savez_compressed(path, **arrays)
        for row in records[:2]:
            row["batch_sha256"] = file_hash(path)
    with pytest.raises(ValueError):
        runner.validate_gradient_manifest(tmp_path, records)


def synthetic_records():
    return [dict(fold=fold, cell=cell, seed=seed, budget=budget, steps=steps)
        for fold in runner.FOLDS for cell in runner.CELLS for seed in runner.SEEDS
        for budget, steps in runner.BUDGETS.items()]


def test_record_matrix_counts_actual_runs_not_the_prefix_twice():
    assert runner.validate_records(synthetic_records()) == dict(actual_training_steps=98304,
        trained_runs=24, snapshots=48, checkpoint_steps_sum=147456)


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_fold", "wrong_cell", "wrong_seed",
    "wrong_budget", "wrong_steps", "extra"])
def test_record_matrix_rejects_missing_duplicate_and_unregistered_conditions(change):
    records = synthetic_records()
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records[-1] = deepcopy(records[0])
    elif change == "extra":
        records.append(deepcopy(records[0]))
    else:
        name = change.removeprefix("wrong_")
        records[0][name] = 9999 if name in {"seed", "steps"} else "unregistered"
    with pytest.raises(ValueError, match="48 snapshots"):
        runner.validate_records(records)


def cached_evidence(tmp_path, monkeypatch):
    """Exercise cache joins using signed arrays and fabricated, labeled records."""
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "_policy_bytes", lambda directory: b"synthetic initial tensors")
    monkeypatch.setattr(runner, "_verify_files", lambda directory, files: None)
    monkeypatch.setattr(runner, "verify_trace_prefix", lambda directory, prefix: None)
    directory = tmp_path / "runs" / "2023_pure_shared_42"
    directory.mkdir(parents=True)
    gradients = gradient_manifest(directory)
    write_json(directory / "gradient_diagnostics.json", dict(records=gradients))
    cfg = dict(seed=42, attribution=dict(runner.CELLS["pure_shared"], fold="2023"))
    protocol = dict(synthetic_only=True)
    canonical = dict(attribution=dict(cfg["attribution"]))
    canonical["attribution"].pop("critic_features")
    control = tmp_path / "initial_controls" / "2023_pure_42"
    control.mkdir(parents=True)
    write_json(control / "completed.json", dict(files={}, identity=dict(
        protocol_sha256=json_digest(protocol), prediction_contract=canonical,
        initial_policy_pth_sha256=hashlib.sha256(b"synthetic initial tensors").hexdigest())))
    (directory / "initial_model").mkdir()
    write_json(directory / "initial_model" / "metadata.json", dict(contract=dict(attribution=cfg["attribution"])))
    records = []
    for budget, steps in runner.BUDGETS.items():
        metric = dict(cumulative_return=.01, max_drawdown=-.01)
        evidence = dict(actual_timesteps=steps, parameters=125063, model_sha256=f"synthetic-{budget}",
            optimizer_steps_min=0, optimizer_steps_max=0, experience=dict(synthetic_only=True),
            training_diagnostics=dict(synthetic_only=True))
        write_json(directory / (budget + "_audit.json"), evidence)
        (directory / (budget + "_model")).mkdir()
        write_json(directory / (budget + "_model") / "metadata.json", dict(num_timesteps=steps,
            model_sha256=evidence["model_sha256"]))
        metrics = {}
        for mode in runner.MODES:
            result = dict(metrics=metric, trace=dict(synthetic_only=True), mode_statistics=dict(synthetic_only=True))
            (directory / budget / mode).mkdir(parents=True)
            (control / mode).mkdir(exist_ok=True)
            write_json(directory / budget / mode / "result.json", result)
            write_json(control / mode / "result.json", dict(metrics=metric))
            metrics[mode] = dict(ppo=metric, initialized=metric, trace=result["trace"],
                mode_statistics=result["mode_statistics"])
        records.append(dict(fold="2023", cell="pure_shared", seed=42, budget=budget, steps=steps,
            parameters=evidence["parameters"], model_sha256=evidence["model_sha256"],
            optimizer_steps_min=0, optimizer_steps_max=0, experience=evidence["experience"],
            training_diagnostics=evidence["training_diagnostics"], training_trace_prefix={},
            initial_controls_path=str(control.relative_to(tmp_path)),
            gradient_records=[r for r in gradients if r["timesteps"] == steps], metrics=metrics))
    marker = dict(config_sha256=json_digest(cfg), protocol_sha256=json_digest(protocol),
        records=records, files={})
    write_json(directory / "completed.json", marker)
    return directory, cfg, protocol, marker, control


def test_cached_trial_accepts_consistent_synthetic_evidence(tmp_path, monkeypatch):
    directory, cfg, protocol, marker, _ = cached_evidence(tmp_path, monkeypatch)
    assert runner.cached_trial(directory, cfg, protocol) == marker["records"]


@pytest.mark.parametrize("change", ["config", "protocol", "missing_checkpoint", "condition", "snapshot",
    "optimizer_evidence", "control_path", "control_contract", "control_theta", "gradient_record", "result"])
def test_cached_trial_rejects_cross_condition_or_modified_evidence(tmp_path, monkeypatch, change):
    directory, cfg, protocol, marker, control = cached_evidence(tmp_path, monkeypatch)
    if change in {"config", "protocol"}:
        marker[change + "_sha256"] = "0" * 64
    elif change == "missing_checkpoint":
        marker["records"].pop()
    elif change == "condition":
        marker["records"][0]["cell"] = "pure_detached"
    elif change == "snapshot":
        marker["records"][0]["model_sha256"] = "different"
    elif change == "optimizer_evidence":
        marker["records"][0]["optimizer_steps_max"] = 1
    elif change == "control_path":
        marker["records"][0]["initial_controls_path"] = "initial_controls/2024_pure_42"
    elif change in {"control_contract", "control_theta"}:
        path = control / "completed.json"
        saved = read_config(path)
        if change == "control_contract":
            saved["identity"]["prediction_contract"]["attribution"]["market_context"] = "fused"
        else:
            saved["identity"]["initial_policy_pth_sha256"] = "different"
        write_json(path, saved)
    elif change == "gradient_record":
        marker["records"][0]["gradient_records"][0]["batch_sha256"] = "different"
    elif change == "result":
        marker["records"][0]["metrics"][runner.MODES[0]]["ppo"]["cumulative_return"] = 999.
    write_json(directory / "completed.json", marker)
    with pytest.raises(ValueError):
        runner.cached_trial(directory, cfg, protocol)
