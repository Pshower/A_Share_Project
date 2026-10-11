"""Independent signed-file checks, without model updates or financial steps."""

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import file_hash, read_config, write_json
from src.training import research_priority_rank as runner
from src.training.freeze import json_digest
from src.training.priority_aux import AUX_SEED_SALT, TrainRankPool
from src.training.priority_trace import verify_priority_trace
from tests.test_ppo_framework import no_parameter_updates
from tests.test_priority_aux import frames


pytestmark = pytest.mark.usefixtures("no_parameter_updates")


def synthetic_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source = tmp_path / "src" / "frozen.py"
    source.parent.mkdir()
    source.write_bytes(b"synthetic frozen source")
    monkeypatch.setattr(runner, "source_hashes", lambda: {"src/frozen.py": file_hash(source)})
    document = tmp_path / "registered.md"
    document.write_bytes(b"synthetic preregistration")
    monkeypatch.setattr(runner, "DOCUMENT", document)
    monkeypatch.setattr(runner, "git_metadata", lambda: dict(commit="synthetic", dirty=False))
    reference = tmp_path / "reference"
    folds, tables = {}, {}
    for fold in runner.FOLDS:
        directory = reference / "folds" / fold
        directory.mkdir(parents=True)
        for name in ("panel_data.parquet", "prices.parquet", "scaler_params.json", "research.json"):
            (directory / name).write_bytes(b"synthetic signed input, not historical prices")
        manifest = dict(protected_period_excluded=True, end_date=f"{fold}-06-30", train_end=f"{fold}-03-31")
        write_json(directory / "manifest.json", manifest)
        folds[fold] = dict(directory=str(directory), manifest_sha256=file_hash(directory / "manifest.json"),
            files={name: file_hash(directory / name) for name in
                   ("panel_data.parquet", "prices.parquet", "scaler_params.json", "research.json")})
        panel, prices, codes, columns = frames()
        panel.index = pd.bdate_range(f"{fold}-01-02", periods=len(panel))
        prices.index = panel.index
        tables[fold] = panel, prices, codes, columns
    write_json(reference / "protocol.json", dict(source_hashes={"src/frozen.py": file_hash(source)},
                                                folds=folds, base_config=dict(synthetic_only=True)))
    write_json(reference / "final_audit.json", dict(status="passed",
        anchors={"protocol.json": file_hash(reference / "protocol.json")}))
    monkeypatch.setattr(runner, "REFERENCE_AUDIT_SHA256", file_hash(reference / "final_audit.json"))

    def config(base, reference_root, root, fold, actor, method, profile="standard", pool=None):
        return dict(seed=42, lookback=1, synthetic_only=True, priority_rank=dict(fold=fold,
            actor_variant=actor, training_method=method, market_pooling="pure_mean", profile=profile,
            auxiliary_coefficient=.05 if method == "rank_aux" else 0., auxiliary_batch_size=1024,
            auxiliary_horizon=5, auxiliary_huber_delta=.1, pool_identity=deepcopy(pool)))

    monkeypatch.setattr(runner, "config_for", config)

    def data(config):
        panel, prices, codes, columns = tables[config["priority_rank"]["fold"]]
        return SimpleNamespace(features=panel, prices=prices, codes=list(codes), config=deepcopy(config),
            manifest=dict(train_end=f"{config['priority_rank']['fold']}-03-31", feature_cols=list(columns)))

    monkeypatch.setattr(runner, "PriorityData", data)
    return reference, tables


def test_prepare_positive_uses_real_train_rank_pool_and_reuses_frozen_registration(tmp_path, monkeypatch):
    reference, _ = synthetic_reference(tmp_path, monkeypatch)
    root = tmp_path / "study"
    protocol = runner.prepare(root, reference)
    assert protocol["planned_actual_steps"] == 393216
    assert protocol["planned_trials"] == 24 and protocol["planned_snapshots"] == 48
    assert protocol["planned_initialized_paths"] == 48
    assert protocol["parameters"] == 125322
    assert protocol["learning_rate"] == 3e-5
    for fold, spec in protocol["folds"].items():
        registered = spec["label_pool"]
        pool = TrainRankPool.load(registered["directory"], registered["manifest_sha256"])
        assert type(pool) is TrainRankPool
        assert pool.identity_sha256 == registered["identity_sha256"]
        assert pool.train_end == f"{fold}-03-31"
        assert set(pool.input_hashes) == set(spec["files"]) | {"truncated_features", "truncated_open_close"}
    runner.check_frozen(protocol)
    assert runner.prepare(root, reference) == protocol


@pytest.mark.parametrize("change", ["audit", "protocol", "source", "manifest", "prices"])
def test_prepare_rejects_changed_trusted_inputs_before_label_creation(tmp_path, monkeypatch, change):
    reference, _ = synthetic_reference(tmp_path, monkeypatch)
    paths = dict(audit=reference / "final_audit.json", protocol=reference / "protocol.json",
        source=tmp_path / "src" / "frozen.py", manifest=reference / "folds" / "2023" / "manifest.json",
        prices=reference / "folds" / "2023" / "prices.parquet")
    paths[change].write_bytes(b"changed signed input")
    with pytest.raises(ValueError):
        runner.prepare(tmp_path / "study", reference)
    assert not (tmp_path / "study").exists()


@pytest.mark.parametrize("change", ["features.npy", "targets.npy", "manifest.json", "document", "source"])
def test_registered_freeze_rejects_label_arrays_or_contract_hash_changes(tmp_path, monkeypatch, change):
    reference, _ = synthetic_reference(tmp_path, monkeypatch)
    protocol = runner.prepare(tmp_path / "study", reference)
    path = (runner.DOCUMENT if change == "document" else tmp_path / "src" / "frozen.py" if change == "source"
            else Path(protocol["folds"]["2023"]["label_pool"]["directory"]) / change)
    path.write_bytes(b"changed registered bytes")
    with pytest.raises(ValueError):
        runner.check_frozen(protocol)


@pytest.mark.parametrize("change", ["stock_order", "feature_order", "boundary", "horizon", "support", "lookback", "input_hash", "physical_features"])
def test_label_pool_validation_rejects_same_shape_wrong_training_contract(tmp_path, monkeypatch, change):
    reference, _ = synthetic_reference(tmp_path, monkeypatch)
    protocol = runner.prepare(tmp_path / "study", reference)
    spec = protocol["folds"]["2023"]
    pool = TrainRankPool.load(spec["label_pool"]["directory"])
    data = runner.PriorityData(runner.trial_config(protocol, tmp_path / "study", "2023", "coupled", "ppo", 42))
    if change == "stock_order":
        data.codes.reverse()
    elif change == "feature_order":
        data.manifest["feature_cols"].reverse()
    elif change == "boundary":
        data.manifest["train_end"] = "2023-03-30"
    elif change == "horizon":
        object.__setattr__(pool, "horizon", 1)
    elif change == "support":
        object.__setattr__(pool, "common_lookback", 1)
    elif change == "lookback":
        data.config["lookback"] = 5
    elif change == "input_hash":
        pool.input_hashes["prices.parquet"] = "0" * 64
    else:
        data.features.iloc[10, 0] += .1
    with pytest.raises(ValueError):
        runner.validate_pool(pool, data, spec["files"])


def trace_files(directory, variant="coupled", chunk_steps=8192):
    directory.mkdir(parents=True)
    manifests = []
    for chunk in range(2):
        raw = np.zeros((chunk_steps, 6), dtype=np.float32)
        raw[:, 0], raw[:, 2], raw[:, 4] = 1, 12, 12 if variant == "coupled" else -15
        normal_constant = -np.log(.5 * np.sqrt(2 * np.pi))
        original_log_prob = -np.log(2) + normal_constant - .5 * (12 / .5) ** 2
        if variant == "decoupled":
            original_log_prob += normal_constant - .5 * (-15 / .5) ** 2
        arrays = dict(raw_packed_action=raw, action_directions=raw[:, :2].copy(),
            action_raw_quantity_latent=raw[:, 2:4].copy(), action_raw_priority_latent=raw[:, 4:].copy(),
            action_priority_active=raw[:, :2] == 1, action_size_latent=np.clip(raw[:, 2:4], -8, 8),
            action_priority_latent=np.clip(raw[:, 4:], -8, 8),
            obs_legal_directions=np.tile([[1, 1, 0, 0], [1, 0, 0, 0]], (chunk_steps, 1, 1)),
            obs_trade_lot_bounds=np.tile([[1, 20, 0, 0], [0, 0, 0, 0]], (chunk_steps, 1, 1)),
            log_prob=np.full(chunk_steps, original_log_prob, dtype=np.float32), value=np.zeros(chunk_steps),
            reward=np.zeros(chunk_steps), done=np.zeros(chunk_steps, dtype=bool))
        path = directory / f"chunk{chunk + 1}.npz"
        np.savez_compressed(path, **arrays)
        manifests.append(dict(path=path.name, first_step=chunk * chunk_steps + 1,
            last_step=(chunk + 1) * chunk_steps, steps=chunk_steps, sha256=file_hash(path)))
    first, rest = b"synthetic sparse order prefix\n", b"synthetic sparse order tail\n"
    (directory / "orders.jsonl").write_bytes(first + rest)
    result = []
    for count in (1, 2):
        raw = first if count == 1 else first + rest
        result.append(dict(schema_version=2, action_layout="direction_N_quantity_N_priority_N",
            identity=dict(codes=["A", "B"], actor_variant=variant), total_steps=count * chunk_steps,
            chunks=deepcopy(manifests[:count]), sparse_orders_path="orders.jsonl",
            sparse_orders_prefix_bytes=len(raw), sparse_orders_prefix_sha256=hashlib.sha256(raw).hexdigest()))
    return result


@pytest.mark.parametrize("variant", ["coupled", "decoupled"])
def test_signed_trace_keeps_unbounded_three_n_samples_and_prefixes(variant, tmp_path):
    prefixes = trace_files(tmp_path / "trace", variant, chunk_steps=3)
    for prefix in prefixes:
        assert verify_priority_trace(tmp_path / "trace", prefix, actor_variant=variant) == prefix
    with np.load(tmp_path / "trace" / "chunk1.npz", allow_pickle=False) as values:
        assert values["raw_packed_action"].shape == (3, 6)
        assert values["raw_packed_action"][0, 2] == 12
        assert values["action_size_latent"][0, 0] == 8
        assert np.isfinite(values["log_prob"]).all()
        original = -np.log(2) - np.log(.5 * np.sqrt(2 * np.pi)) - .5 * (12 / .5) ** 2
        clipped = -np.log(2) - np.log(.5 * np.sqrt(2 * np.pi)) - .5 * (8 / .5) ** 2
        if variant == "decoupled":
            original += -np.log(.5 * np.sqrt(2 * np.pi)) - .5 * (-15 / .5) ** 2
            clipped += -np.log(.5 * np.sqrt(2 * np.pi)) - .5 * (-8 / .5) ** 2
        assert values["log_prob"][0] == pytest.approx(original, rel=1e-7)
        assert abs(values["log_prob"][0] - clipped) > 100


@pytest.mark.parametrize("change", ["missing_coordinate", "nan", "inactive_priority", "copy_atom",
    "fractional_direction", "direction_mismatch", "illegal_direction", "coordinate_mismatch", "step_prefix", "variant"])
def test_resigned_trace_rejects_original_probability_support_or_prefix_changes(tmp_path, change):
    directory = tmp_path / "trace"
    prefix = trace_files(directory, chunk_steps=3)[0]
    if change == "step_prefix":
        prefix["chunks"][0]["first_step"] = 2
    elif change == "variant":
        prefix["identity"]["actor_variant"] = "unknown"
    else:
        path = directory / "chunk1.npz"
        with np.load(path, allow_pickle=False) as loaded:
            values = {key: loaded[key].copy() for key in loaded.files}
        if change == "missing_coordinate":
            values["raw_packed_action"] = values["raw_packed_action"][:, :4]
        elif change == "nan":
            values["log_prob"][0] = np.nan
        elif change == "inactive_priority":
            values["raw_packed_action"][0, 5] = .1
        elif change == "copy_atom":
            values["raw_packed_action"][0, 4] = 11
        elif change == "fractional_direction":
            values["raw_packed_action"][0, 0] = 1.5
        elif change == "direction_mismatch":
            values["action_directions"][0, 0] = 0
        elif change == "illegal_direction":
            values["obs_legal_directions"][0, 0, 1] = 0
        else:
            values["action_raw_quantity_latent"][0, 0] = 1
        np.savez_compressed(path, **values)
        prefix["chunks"][0]["sha256"] = file_hash(path)
    with pytest.raises(ValueError):
        verify_priority_trace(directory, prefix)


def fake_model(directory, config, contract, sources, steps, auxiliary=None):
    directory.mkdir(parents=True)
    with zipfile.ZipFile(directory / "model.zip", "w") as archive:
        archive.writestr("policy.pth", b"synthetic parameter tensors, never loaded")
    metadata = dict(config=deepcopy(config), contract=deepcopy(contract), num_timesteps=steps,
        source_hashes=deepcopy(sources), model_sha256=file_hash(directory / "model.zip"),
        algorithm_class="src.training.priority_aux.RankAuxPPO", extra=dict(auxiliary_audit=deepcopy(auxiliary)),
        priority_policy=dict(actor_variant=config["priority_rank"]["actor_variant"],
            network_actor_variant=config["priority_rank"]["actor_variant"],
            distribution_actor_variant=config["priority_rank"]["actor_variant"],
            priority_log_std=-.7, size_log_std=-.7))
    write_json(directory / "metadata.json", metadata)
    return metadata


def signed_cache(tmp_path, monkeypatch, method="rank_aux"):
    reference, _ = synthetic_reference(tmp_path, monkeypatch)
    root = tmp_path / "study"
    protocol = runner.prepare(root, reference)
    config = runner.trial_config(protocol, root, "2023", "coupled", method, 42)
    directory = root / "runs" / f"2023_coupled_{method}_42"
    directory.mkdir(parents=True)
    contract = dict(priority_rank=deepcopy(config["priority_rank"]), stock_codes=["A", "B"],
                    trade=dict(actor_variant="coupled"), broker=dict(commission=.0003),
                    artifacts=dict(scaler="synthetic"))
    write_json(directory / "config.json", config)
    initial = fake_model(directory / "initial_model", config, contract, protocol["source_hashes"], 0)
    control = root / "initial_controls" / "2023_coupled_42"
    metric = dict(cumulative_return=.01, max_drawdown=-.01, total_fees=1., total_slippage_cost=1.)
    for mode in runner.MODES:
        (control / mode).mkdir(parents=True)
        initial_trace = trace_files(control / mode / "trace", chunk_steps=3)[-1]
        initial_trace["identity"].update(contract_sha256=json_digest(contract), seed=runner.EVAL_SEED,
                                         initial_mode=mode, top_n=20)
        write_json(control / mode / "result.json", dict(metrics=metric, trace=initial_trace,
            contract=deepcopy(contract), mode_statistics=dict(economic_violations=0),
            initial_mode=mode, top_n=20, test_evaluated=False))
    write_json(control / "completed.json", dict(files=runner._registered_files(control),
        metrics={mode: metric for mode in runner.MODES}, identity=dict(protocol_sha256=json_digest(protocol),
            prediction_contract=runner._canonical_contract(contract),
            initial_policy_pth_sha256=hashlib.sha256(runner._policy_bytes(directory / "initial_model")).hexdigest(),
            evaluation_seed=runner.EVAL_SEED, modes=list(runner.MODES), top_n=20)))
    prefixes = trace_files(directory / "training_trace")
    rows = []
    for (budget, steps), prefix in zip(protocol["budgets"].items(), prefixes):
        prefix["identity"]["contract_sha256"] = json_digest(contract)
        updates = steps // 512 * 20
        auxiliary = dict(aux_coef=.05 if method == "rank_aux" else 0., batch_size=1024, huber_delta=.1,
            seed=42, seed_salt=AUX_SEED_SALT,
            pool_identity_sha256=protocol["folds"]["2023"]["label_pool"]["identity_sha256"],
            actual_joint_optimizer_updates=updates, actual_aux_updates=updates if method == "rank_aux" else 0,
            supervised_stock_date_exposures=updates * 1024 if method == "rank_aux" else 0,
            extra_optimizer_steps=0, auxiliary_rl_steps=0, supervision_in_observations=False)
        metadata = fake_model(directory / (budget + "_model"), config, contract, protocol["source_hashes"], steps, auxiliary)
        audit = dict(actual_timesteps=steps, parameters=protocol["parameters"], model_sha256=metadata["model_sha256"],
            optimizer_steps_min=updates, optimizer_steps_max=updates,
            optimizer_steps_by_parameter={"network.market_encoder.0.weight": updates},
            experience=dict(total_steps=steps), training_diagnostics=dict(synthetic_only=True),
            auxiliary_audit=auxiliary, training_trace_prefix=prefix, tensor_delta_l2=dict(synthetic=.1),
            mode_statistics=dict(economic_violations=0))
        write_json(directory / (budget + "_audit.json"), audit)
        metrics = {}
        for mode in runner.MODES:
            folder = directory / budget / mode
            folder.mkdir(parents=True)
            result_trace = trace_files(folder / "trace", chunk_steps=3)[-1]
            result_trace["identity"].update(contract_sha256=json_digest(contract), seed=runner.EVAL_SEED,
                                             initial_mode=mode, top_n=20)
            result = dict(metrics=metric, trace=result_trace, contract=deepcopy(contract),
                          mode_statistics=dict(economic_violations=0), initial_mode=mode,
                          top_n=20, test_evaluated=False)
            write_json(folder / "result.json", result)
            metrics[mode] = dict(ppo=metric, initialized=metric, trace=result["trace"], mode_statistics=result["mode_statistics"])
        row = dict(fold="2023", actor="coupled", method=method, seed=42, budget=budget, steps=steps,
            model_path=str((directory / (budget + "_model")).relative_to(tmp_path)),
            model_sha256=metadata["model_sha256"], initial_controls_path=str(control.relative_to(tmp_path)), metrics=metrics,
            training_mode_statistics=audit["mode_statistics"])
        for field in ("parameters", "optimizer_steps_min", "optimizer_steps_max", "optimizer_steps_by_parameter",
                      "experience", "training_diagnostics", "auxiliary_audit", "training_trace_prefix", "tensor_delta_l2"):
            row[field] = deepcopy(audit[field])
        rows.append(row)
    marker = dict(config_sha256=json_digest(config), protocol_sha256=json_digest(protocol), records=rows,
                  files=runner._registered_files(directory))
    write_json(directory / "completed.json", marker)
    return directory, config, protocol, marker, control


@pytest.mark.parametrize("method", ["ppo", "rank_aux"])
def test_strict_cached_trial_accepts_signed_native_method_and_full_prefix_without_loading_models(tmp_path, monkeypatch, method):
    directory, config, protocol, marker, _ = signed_cache(tmp_path, monkeypatch, method)
    assert runner.cached_trial(directory, config, protocol) == marker["records"]


@pytest.mark.parametrize("change", ["short_model/model.zip", "config.json", "short_audit.json", "initial_model/model.zip",
    "training_trace/chunk1.npz", "training_trace/orders.jsonl", "short/cash/result.json"])
def test_cache_rejects_unsigned_native_bytes(tmp_path, monkeypatch, change):
    directory, config, protocol, _, _ = signed_cache(tmp_path, monkeypatch)
    (directory / change).write_bytes(b"unsigned bytes")
    with pytest.raises(ValueError):
        runner.cached_trial(directory, config, protocol)


@pytest.mark.parametrize("change", ["row_method", "row_seed", "missing_snapshot", "duplicate_snapshot", "aux_coef",
    "aux_pool", "aux_updates", "aux_exposures", "snapshot_algorithm", "snapshot_actor", "snapshot_auxiliary",
    "snapshot_source", "snapshot_contract", "snapshot_steps", "initial_algorithm", "initial_actor",
    "control_theta", "control_metrics", "prefix_join"])
def test_cache_rejects_resigned_but_cross_method_or_inconsistent_evidence(tmp_path, monkeypatch, change):
    directory, config, protocol, marker, control = signed_cache(tmp_path, monkeypatch)
    row = marker["records"][0]
    audit_path, metadata_path = directory / "short_audit.json", directory / "short_model" / "metadata.json"
    audit, metadata = read_config(audit_path), read_config(metadata_path)
    if change == "row_method":
        row["method"] = "ppo"
    elif change == "row_seed":
        row["seed"] = 123
    elif change == "missing_snapshot":
        marker["records"].pop()
    elif change == "duplicate_snapshot":
        marker["records"][1] = deepcopy(row)
    elif change.startswith("aux_"):
        key = {"aux_coef": "aux_coef", "aux_pool": "pool_identity_sha256", "aux_updates": "actual_aux_updates",
               "aux_exposures": "supervised_stock_date_exposures"}[change]
        value = "0" * 64 if change == "aux_pool" else 0
        row["auxiliary_audit"][key] = audit["auxiliary_audit"][key] = value
        metadata["extra"]["auxiliary_audit"][key] = value
    elif change.startswith("snapshot_"):
        if change == "snapshot_algorithm":
            metadata["algorithm_class"] = "stable_baselines3.ppo.ppo.PPO"
        elif change == "snapshot_actor":
            metadata["priority_policy"]["network_actor_variant"] = "decoupled"
        elif change == "snapshot_auxiliary":
            metadata["extra"]["auxiliary_audit"]["actual_aux_updates"] = 0
        elif change == "snapshot_source":
            metadata["source_hashes"] = {}
        elif change == "snapshot_contract":
            metadata["contract"]["priority_rank"]["training_method"] = "ppo"
        else:
            metadata["num_timesteps"] = 0
    elif change.startswith("initial_"):
        path = directory / "initial_model" / "metadata.json"
        value = read_config(path)
        if change == "initial_algorithm":
            value["algorithm_class"] = "stable_baselines3.ppo.ppo.PPO"
        else:
            value["priority_policy"]["distribution_actor_variant"] = "decoupled"
        write_json(path, value)
    elif change.startswith("control_"):
        path = control / "completed.json"
        value = read_config(path)
        if change == "control_theta":
            value["identity"]["initial_policy_pth_sha256"] = "0" * 64
        else:
            value["metrics"][runner.MODES[0]]["cumulative_return"] = .5
        write_json(path, value)
    else:
        row["training_trace_prefix"]["chunks"][0]["first_step"] = 2
        audit["training_trace_prefix"] = deepcopy(row["training_trace_prefix"])
    write_json(audit_path, audit)
    write_json(metadata_path, metadata)
    marker["files"] = runner._registered_files(directory)
    write_json(directory / "completed.json", marker)
    with pytest.raises(ValueError):
        runner.cached_trial(directory, config, protocol)


def case_identity():
    contract = dict(stock_codes=["A", "B"], trade=dict(actor_variant="coupled"),
        priority_rank=dict(training_method="rank_aux", auxiliary_coefficient=.05,
                           actor_variant="coupled", fold="2023", pool_identity="same"))
    result = dict(contract=deepcopy(contract), trace=dict(identity=dict(
        codes=["A", "B"], actor_variant="coupled", contract_sha256=json_digest(contract),
        seed=runner.EVAL_SEED, initial_mode="cash", top_n=20)), initial_mode="cash", top_n=20,
        test_evaluated=False)
    return contract, result


def test_initialized_case_reuse_removes_only_method_and_coefficient():
    contract, result = case_identity()
    result["contract"]["priority_rank"].update(training_method="ppo", auxiliary_coefficient=0.)
    result["trace"]["identity"]["contract_sha256"] = json_digest(result["contract"])
    assert runner.validate_case_identity(result, contract, "cash", initialized=True) is result
    with pytest.raises(ValueError):
        runner.validate_case_identity(result, contract, "cash", initialized=False)


@pytest.mark.parametrize("change", ["contract", "trace_hash", "code_order", "actor", "seed", "mode", "top_n", "protected"])
def test_case_identity_rejects_cross_pool_or_changed_account_even_when_metrics_match(change):
    contract, result = case_identity()
    identity = result["trace"]["identity"]
    if change == "contract":
        result["contract"]["priority_rank"]["pool_identity"] = "other"
        identity["contract_sha256"] = json_digest(result["contract"])
    elif change == "trace_hash":
        identity["contract_sha256"] = "0" * 64
    elif change == "code_order":
        identity["codes"].reverse()
    elif change == "actor":
        identity["actor_variant"] = "decoupled"
    elif change == "seed":
        identity["seed"] += 1
    elif change == "mode":
        result["initial_mode"] = identity["initial_mode"] = "random"
    elif change == "top_n":
        result["top_n"] = identity["top_n"] = 10
    else:
        result["test_evaluated"] = True
    with pytest.raises(ValueError):
        runner.validate_case_identity(result, contract, "cash", initialized=True)


def worker_receipt(root, fold="2023"):
    rows = []
    for actor in runner.ACTORS:
        for method in runner.METHODS:
            for seed in runner.SEEDS:
                name = f"{fold}_{actor}_{method}_{seed}"
                directory = root / "runs" / name
                directory.mkdir(parents=True)
                write_json(directory / "completed.json", dict(synthetic_only=True))
                rows.append(dict(run_id=name, completed_sha256=file_hash(directory / "completed.json")))
    value = dict(status="complete", fold=fold, completed_runs=rows,
        protocol_sha256=file_hash(root / "protocol.json"), no_global_summary_writes=True)
    write_json(root / f"worker_{fold}.json", value)
    return value


def test_worker_receipt_accepts_exactly_twelve_signed_trial_markers(tmp_path):
    write_json(tmp_path / "protocol.json", dict(synthetic_only=True))
    expected = worker_receipt(tmp_path)
    assert runner.verify_worker_receipt(tmp_path, {}, "2023") == expected


@pytest.mark.parametrize("change", ["running", "missing_trial", "duplicate_trial", "wrong_fold", "wrong_protocol", "unsigned_marker"])
def test_report_rejects_incomplete_worker_without_training_or_evaluation(tmp_path, monkeypatch, change):
    write_json(tmp_path / "protocol.json", dict(synthetic_only=True))
    receipt = worker_receipt(tmp_path)
    if change == "running":
        receipt["status"] = "running"
    elif change == "missing_trial":
        receipt["completed_runs"].pop()
    elif change == "duplicate_trial":
        receipt["completed_runs"][-1] = deepcopy(receipt["completed_runs"][0])
    elif change == "wrong_fold":
        receipt["fold"] = "2024"
    elif change == "wrong_protocol":
        receipt["protocol_sha256"] = "0" * 64
    else:
        (tmp_path / "runs" / receipt["completed_runs"][0]["run_id"] / "completed.json").write_bytes(b"unsigned")
    write_json(tmp_path / "worker_2023.json", receipt)
    monkeypatch.setattr(runner, "prepare", lambda *args: {})
    monkeypatch.setattr(runner, "check_frozen", lambda *args: None)
    for name in ("run_trial", "evaluate_case", "rule_controls", "write_report"):
        monkeypatch.setattr(runner, name, lambda *args, **kwargs: pytest.fail("Report must not run financial work before worker barrier"))
    with pytest.raises(ValueError):
        runner.main(["--stage", "report", "--output", str(tmp_path)])
