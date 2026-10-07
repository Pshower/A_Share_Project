"""Independent signed-file runner checks; no optimizer or financial execution."""

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch

from src.agents.ppo_agent import PPOAgent
from src.data.preprocess import file_hash, read_config, write_json
from src.training import research_market_summary as runner
from src.training.freeze import json_digest
from tests.test_ppo_framework import local_data, no_parameter_updates
from tests.test_trade_policy import trade_config


pytestmark = pytest.mark.usefixtures("no_parameter_updates")


def synthetic_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source = tmp_path / "src" / "frozen.py"
    source.parent.mkdir()
    source.write_bytes(b"synthetic original source")
    monkeypatch.setattr(runner, "source_hashes", lambda: {"src/frozen.py": file_hash(source)})
    doc = tmp_path / "protocol.md"
    doc.write_bytes(b"synthetic preregistration")
    monkeypatch.setattr(runner, "DOCUMENT", doc)
    monkeypatch.setattr(runner, "git_metadata", lambda: dict(commit="synthetic", dirty=False))
    reference = tmp_path / "reference"
    folds = {}
    for fold in runner.FOLDS:
        directory = reference / "folds" / fold
        directory.mkdir(parents=True)
        for name in ("panel_data.parquet", "prices.parquet", "scaler_params.json", "research.json"):
            (directory / name).write_bytes(b"synthetic original data")
        write_json(directory / "manifest.json", dict(protected_period_excluded=True,
            end_date=f"{fold}-06-30", train_end=f"{fold}-03-31"))
        folds[fold] = dict(directory=str(directory), manifest_sha256=file_hash(directory / "manifest.json"),
            files={name: file_hash(directory / name) for name in
                ("panel_data.parquet", "prices.parquet", "scaler_params.json", "research.json")})
    write_json(reference / "protocol.json", dict(source_hashes={"src/frozen.py": file_hash(source)},
        folds=folds, base_config=dict(synthetic_only=True)))
    write_json(reference / "final_audit.json", dict(status="passed",
        anchors={"protocol.json": file_hash(reference / "protocol.json")}))
    monkeypatch.setattr(runner, "REFERENCE_AUDIT_SHA256", file_hash(reference / "final_audit.json"))
    monkeypatch.setattr(runner, "config_for", lambda *args: dict(synthetic_only=True))
    calls = []
    env = SimpleNamespace(observation_space=SimpleNamespace(contains=lambda obs: True),
        unwrapped=SimpleNamespace(market=SimpleNamespace(stock_codes=["A"])),
        reset=lambda **kwargs: (calls.append("reset") or {}, {}), close=lambda: calls.append("close"))
    monkeypatch.setattr(runner, "SummaryData", lambda cfg: SimpleNamespace(env=lambda split: env, codes=["A"]))
    return reference, calls


def test_prepare_registers_only_trusted_synthetic_folds_and_reuses_without_reset(tmp_path, monkeypatch):
    reference, calls = synthetic_reference(tmp_path, monkeypatch)
    root = tmp_path / "new_study"
    protocol = runner.prepare(root, reference)
    assert protocol["planned_actual_steps"] == 147456
    assert protocol["planned_trials"] == 18 and protocol["planned_snapshots"] == 36
    assert protocol["planned_initialized_paths"] == 24
    assert protocol["parameters"] == 141575
    assert protocol["gradient_diagnostics"] is False
    assert protocol["test_evaluated"] is protocol["actionable"] is False
    assert calls == ["reset", "close"]
    runner.check_frozen(protocol)
    assert runner.prepare(root, reference) == protocol
    assert calls == ["reset", "close"]


@pytest.mark.parametrize("change", ["trusted_audit", "reference_protocol", "source", "manifest", "data",
    "protected_period", "train_boundary"])
def test_prepare_rejects_changed_reference_before_environment_reset(tmp_path, monkeypatch, change):
    reference, calls = synthetic_reference(tmp_path, monkeypatch)
    if change == "trusted_audit":
        (reference / "final_audit.json").write_bytes(b"altered prior audit")
    elif change == "reference_protocol":
        value = read_config(reference / "protocol.json")
        value["base_config"]["unexpected"] = True
        write_json(reference / "protocol.json", value)
    elif change == "source":
        (tmp_path / "src" / "frozen.py").write_bytes(b"altered original source")
    elif change == "data":
        (reference / "folds" / "2023" / "prices.parquet").write_bytes(b"altered original data")
    else:
        path = reference / "folds" / "2023" / "manifest.json"
        value = read_config(path)
        value["end_date"] = "2026-07-01" if change == "protected_period" else value["end_date"]
        value["train_end"] = "2023-04-01" if change == "train_boundary" else value["train_end"]
        value["unexpected"] = True
        write_json(path, value)
    with pytest.raises(ValueError):
        runner.prepare(tmp_path / "new_study", reference)
    assert not calls


@pytest.mark.parametrize("change", ["source", "document", "fold_data", "reference_protocol", "reference_audit"])
def test_registered_freeze_rejects_changed_implementation_or_data(tmp_path, monkeypatch, change):
    reference, _ = synthetic_reference(tmp_path, monkeypatch)
    protocol = runner.prepare(tmp_path / "new_study", reference)
    paths = dict(source=tmp_path / "src" / "frozen.py", document=runner.DOCUMENT,
        fold_data=reference / "folds" / "2024" / "scaler_params.json",
        reference_protocol=reference / "protocol.json", reference_audit=reference / "final_audit.json")
    paths[change].write_bytes(b"modified after registration")
    with pytest.raises(ValueError):
        runner.check_frozen(protocol)


def _model(folder, contract, cfg, sources, steps, theta=b"synthetic initial policy tensors"):
    folder.mkdir(parents=True)
    with zipfile.ZipFile(folder / "model.zip", "w") as archive:
        archive.writestr("policy.pth", theta)
    metadata = dict(contract=deepcopy(contract), config=deepcopy(cfg), source_hashes=deepcopy(sources),
        num_timesteps=steps, model_sha256=file_hash(folder / "model.zip"))
    write_json(folder / "metadata.json", metadata)
    return metadata


def signed_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    directory = tmp_path / "runs" / "2023_mean_42"
    directory.mkdir(parents=True)
    cfg = dict(seed=42, market_summary=dict(pooling="mean", fold="2023", critic_features="shared",
        zero_query_initialization=True, zero_variance_projection_initialization=True), synthetic_only=True)
    protocol = dict(synthetic_only=True, source_hashes={"src/frozen.py": "synthetic-source-hash"})
    contract = dict(market_summary=deepcopy(cfg["market_summary"]), broker=dict(commission=.0003),
        artifacts={"scaler_params.json": "synthetic-scaler"})
    write_json(directory / "config.json", cfg)
    initial_metadata = _model(directory / "initial_model", contract, cfg, protocol["source_hashes"], 0)
    proof = runner.neutral_initialization(SimpleNamespace(network=SimpleNamespace(
        summary_query=torch.zeros(2), variance_projection=SimpleNamespace(weight=torch.zeros(2, 2)))))
    initial_digest = hashlib.sha256(runner._policy_bytes(directory / "initial_model")).hexdigest()
    write_json(directory / "initial_identity.json", dict(policy_pth_sha256=initial_digest,
        neutral_pooling=proof, matched_existing_arms=[], scope="synthetic-only"))
    control = tmp_path / "initial_controls" / "2023_42"
    control.mkdir(parents=True)
    metric = dict(cumulative_return=.01, max_drawdown=-.01)
    for mode in runner.MODES:
        (control / mode).mkdir()
        write_json(control / mode / "result.json", dict(metrics=metric, synthetic_only=True))
    canonical = deepcopy(contract)
    canonical["market_summary"].pop("pooling")
    write_json(control / "completed.json", dict(files=runner._registered_files(control),
        metrics={mode: metric for mode in runner.MODES}, identity=dict(protocol_sha256=json_digest(protocol),
            prediction_contract=canonical, initial_policy_pth_sha256=initial_digest,
            evaluation_seed=runner.EVAL_SEED, modes=list(runner.MODES), top_n=20, neutral_pooling=proof)))
    trace = directory / "training_trace"
    trace.mkdir()
    (trace / "chunk1.npz").write_bytes(b"synthetic trace chunk one, no observations executed")
    (trace / "chunk2.npz").write_bytes(b"synthetic trace chunk two, no observations executed")
    first_orders, rest_orders = b"synthetic orders first\n", b"synthetic orders second\n"
    (trace / "orders.jsonl").write_bytes(first_orders + rest_orders)
    rows = []
    for budget, steps in runner.BUDGETS.items():
        chunks = [dict(path="chunk1.npz", first_step=1, last_step=4096, steps=4096,
            sha256=file_hash(trace / "chunk1.npz"))]
        if budget == "long":
            chunks.append(dict(path="chunk2.npz", first_step=4097, last_step=8192, steps=4096,
                sha256=file_hash(trace / "chunk2.npz")))
        raw = first_orders if budget == "short" else first_orders + rest_orders
        prefix = dict(chunks=chunks, total_steps=steps, sparse_orders_path="orders.jsonl",
            sparse_orders_prefix_bytes=len(raw), sparse_orders_prefix_sha256=hashlib.sha256(raw).hexdigest())
        metadata = _model(directory / (budget + "_model"), contract, cfg, protocol["source_hashes"], steps,
                          theta=("synthetic trained " + budget).encode("ascii"))
        audit = dict(actual_timesteps=steps, parameters=runner.PARAMETERS, model_sha256=metadata["model_sha256"],
            optimizer_steps_min=0, optimizer_steps_max=0, experience=dict(synthetic_only=True),
            training_diagnostics=dict(synthetic_only=True), mode_statistics=dict(synthetic_only=True),
            parameter_participation=dict(synthetic_only=True), training_trace_prefix=prefix, tensor_delta_l2=.1)
        write_json(directory / (budget + "_audit.json"), audit)
        metrics = {}
        for mode in runner.MODES:
            result = dict(metrics=metric, trace=dict(synthetic_only=True), mode_statistics=dict(synthetic_only=True))
            (directory / budget / mode).mkdir(parents=True)
            write_json(directory / budget / mode / "result.json", result)
            metrics[mode] = dict(ppo=metric, initialized=metric, trace=result["trace"], mode_statistics=result["mode_statistics"])
        rows.append(dict(fold="2023", arm="mean", seed=42, budget=budget, steps=steps,
            parameters=audit["parameters"], model_sha256=audit["model_sha256"],
            model_path=str((directory / (budget + "_model")).relative_to(tmp_path)),
            optimizer_steps_min=0, optimizer_steps_max=0, experience=audit["experience"],
            training_diagnostics=audit["training_diagnostics"], training_mode_statistics=audit["mode_statistics"],
            parameter_participation=audit["parameter_participation"], training_trace_prefix=deepcopy(prefix),
            tensor_delta_l2=.1, initial_controls_path=str(control.relative_to(tmp_path)), metrics=deepcopy(metrics)))
    marker = dict(config_sha256=json_digest(cfg), protocol_sha256=json_digest(protocol), records=rows,
        files=runner._registered_files(directory))
    write_json(directory / "completed.json", marker)
    return directory, cfg, protocol, marker, control, initial_metadata


def test_cached_trial_accepts_exact_signed_synthetic_models_contracts_and_prefixes(tmp_path, monkeypatch):
    directory, cfg, protocol, marker, _, _ = signed_cache(tmp_path, monkeypatch)
    assert runner.cached_trial(directory, cfg, protocol) == marker["records"]


@pytest.mark.parametrize("change", ["model", "native_config", "initial_model", "trace_chunk", "sparse_orders", "evaluation", "control_result"])
def test_cached_trial_rejects_unsigned_artifact_changes(tmp_path, monkeypatch, change):
    directory, cfg, protocol, _, control, _ = signed_cache(tmp_path, monkeypatch)
    paths = dict(model=directory / "short_model" / "model.zip", native_config=directory / "config.json",
        initial_model=directory / "initial_model" / "model.zip", trace_chunk=directory / "training_trace" / "chunk1.npz",
        sparse_orders=directory / "training_trace" / "orders.jsonl", evaluation=directory / "short" / runner.MODES[0] / "result.json",
        control_result=control / runner.MODES[0] / "result.json")
    paths[change].write_bytes(b"unsigned change")
    with pytest.raises(ValueError):
        runner.cached_trial(directory, cfg, protocol)


@pytest.mark.parametrize("change", ["config_digest", "protocol_digest", "missing_checkpoint", "duplicate_checkpoint",
    "arm", "seed", "model_sha", "optimizer", "tensor_delta", "prefix_join", "prefix_steps", "control_path",
    "native_config", "snapshot_contract", "snapshot_config", "snapshot_source", "snapshot_steps", "actual_model_sha",
    "initial_contract", "initial_config", "initial_source", "initial_steps", "control_contract", "control_theta",
    "control_protocol", "control_seed", "control_neutral", "control_metrics", "result_metric"])
def test_cached_trial_rejects_resigned_but_inconsistent_evidence(tmp_path, monkeypatch, change):
    directory, cfg, protocol, marker, control, _ = signed_cache(tmp_path, monkeypatch)
    row = marker["records"][0]
    if change in {"config_digest", "protocol_digest"}:
        marker[change.replace("_digest", "_sha256")] = "0" * 64
    elif change == "missing_checkpoint":
        marker["records"].pop()
    elif change == "duplicate_checkpoint":
        marker["records"][1] = deepcopy(row)
    elif change == "arm":
        row["arm"] = "attention"
    elif change == "seed":
        row["seed"] = 123
    elif change == "model_sha":
        row["model_sha256"] = "0" * 64
    elif change == "optimizer":
        row["optimizer_steps_max"] = 1
    elif change == "tensor_delta":
        row["tensor_delta_l2"] = .2
    elif change == "prefix_join":
        row["training_trace_prefix"]["sparse_orders_prefix_bytes"] = 0
        row["training_trace_prefix"]["sparse_orders_prefix_sha256"] = hashlib.sha256(b"").hexdigest()
    elif change == "prefix_steps":
        row = marker["records"][1]
        row["training_trace_prefix"] = deepcopy(marker["records"][0]["training_trace_prefix"])
        audit = read_config(directory / "long_audit.json")
        audit["training_trace_prefix"] = deepcopy(row["training_trace_prefix"])
        write_json(directory / "long_audit.json", audit)
    elif change == "control_path":
        row["initial_controls_path"] = "initial_controls/2024_42"
    elif change == "native_config":
        value = deepcopy(cfg)
        value["seed"] = 123
        write_json(directory / "config.json", value)
    elif change.startswith("snapshot_") or change == "actual_model_sha":
        path = directory / "short_model" / "metadata.json"
        value = read_config(path)
        if change == "snapshot_contract":
            value["contract"]["broker"]["commission"] = .01
        elif change == "snapshot_config":
            value["config"]["seed"] = 123
        elif change == "snapshot_source":
            value["source_hashes"]["src/frozen.py"] = "different"
        elif change == "snapshot_steps":
            value["num_timesteps"] += 1
        else:
            (directory / "short_model" / "model.zip").write_bytes(b"modified model with stale metadata hash")
        write_json(path, value)
    elif change.startswith("initial_"):
        path = directory / "initial_model" / "metadata.json"
        value = read_config(path)
        if change == "initial_contract":
            value["contract"]["market_summary"]["pooling"] = "attention"
        elif change == "initial_config":
            value["config"]["seed"] = 123
        elif change == "initial_source":
            value["source_hashes"]["src/frozen.py"] = "different"
        else:
            value["num_timesteps"] = 1
        write_json(path, value)
    elif change.startswith("control_"):
        path = control / "completed.json"
        value = read_config(path)
        if change == "control_contract":
            value["identity"]["prediction_contract"]["broker"]["commission"] = .01
        elif change == "control_theta":
            value["identity"]["initial_policy_pth_sha256"] = "different"
        elif change == "control_protocol":
            value["identity"]["protocol_sha256"] = "different"
        elif change == "control_seed":
            value["identity"]["evaluation_seed"] += 1
        elif change == "control_neutral":
            value["identity"]["neutral_pooling"]["query_exact_zero"] = False
        else:
            value["metrics"][runner.MODES[0]]["cumulative_return"] = 999.
        write_json(path, value)
    elif change == "result_metric":
        row["metrics"][runner.MODES[0]]["ppo"]["cumulative_return"] = 999.
    marker["files"] = runner._registered_files(directory)
    write_json(directory / "completed.json", marker)
    with pytest.raises(ValueError):
        runner.cached_trial(directory, cfg, protocol)


def summary_config(local_data, arm="mean", fold="2023", seed=42):
    value = trade_config(local_data)
    value.update(seed=seed, torch_threads=1, actor_output_gain=.1,
        training_episode_steps=8, recent_episode_every=4, initial_portfolio="mixed",
        exploration=dict(architecture="split", common_lookback=5, sampling="recent",
            reward_basis="simple_net", size_annealing=False),
        market_summary=dict(pooling=arm, fold=fold, critic_features="shared",
            zero_query_initialization=True, zero_variance_projection_initialization=True))
    return value


@pytest.mark.parametrize("wrong_arm", ["attention", "dispersion"])
def test_loader_rejects_policy_pooling_hidden_behind_matching_metadata(local_data, tmp_path, wrong_arm):
    data = runner.SummaryData(summary_config(local_data, wrong_arm))
    expected = runner.SummaryData(summary_config(local_data, "mean"))
    env = data.vector_env()
    try:
        model = runner.make_model(env, data.config)
        PPOAgent(model, expected.contract).save(tmp_path / "wrong_policy", data.config)
        with pytest.raises(ValueError, match="policy flags"):
            runner.load_agent(tmp_path / "wrong_policy", expected)
    finally:
        env.close()


@pytest.mark.parametrize("wrong_arm", ["attention", "dispersion"])
def test_loader_rejects_metadata_pooling_hidden_behind_correct_policy(local_data, tmp_path, wrong_arm):
    expected = runner.SummaryData(summary_config(local_data, "mean"))
    data = runner.SummaryData(summary_config(local_data, wrong_arm))
    env = expected.vector_env()
    try:
        model = runner.make_model(env, expected.config)
        PPOAgent(model, data.contract).save(tmp_path / "wrong_contract", expected.config)
        with pytest.raises(ValueError, match="contract"):
            runner.load_agent(tmp_path / "wrong_contract", expected)
    finally:
        env.close()


def test_initial_controls_reuse_all_neutral_pooling_arms_but_keep_fold_and_seed(local_data, tmp_path, monkeypatch):
    calls, environments = [], []

    def evaluation(agent, data, folder, label):
        folder = Path(folder)
        folder.mkdir(parents=True)
        metrics = dict(cumulative_return=.01, max_drawdown=-.01)
        write_json(folder / "result.json", dict(metrics=metrics, synthetic_only=True))
        calls.append(label)
        return dict(metrics=metrics)

    monkeypatch.setattr(runner, "evaluate_case", evaluation)
    protocol = dict(synthetic_only=True)
    try:
        for arm, fold, seed, wanted in (("mean", "2023", 42, 4), ("attention", "2023", 42, 4),
            ("dispersion", "2023", 42, 4), ("mean", "2024", 42, 8), ("mean", "2023", 123, 12)):
            data = runner.SummaryData(summary_config(local_data, arm, fold, seed))
            env = data.vector_env()
            environments.append(env)
            model = runner.make_model(env, data.config)
            path = tmp_path / "models" / f"{fold}_{arm}_{seed}"
            PPOAgent(model, data.contract).save(path, data.config)
            _, control = runner.initial_controls(tmp_path, protocol, data, path, fold, seed)
            assert len(calls) == wanted
            identity = read_config(control / "completed.json")["identity"]
            canonical = deepcopy(data.contract)
            canonical["market_summary"].pop("pooling")
            assert identity["prediction_contract"] == canonical
            assert identity["neutral_pooling"]["query_exact_zero"] is True
            assert identity["neutral_pooling"]["variance_projection_exact_zero"] is True
            assert model.num_timesteps == 0 and not model.policy.optimizer.state
    finally:
        for env in environments:
            env.close()
