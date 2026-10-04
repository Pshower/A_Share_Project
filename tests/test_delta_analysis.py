"""External posthoc-analysis checks use synthetic data and never train a policy."""

from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.agents.delta_mapping import map_delta_action
from src.backtest.run import run_policy
from src.data.preprocess import file_hash, read_config, write_json
from src.envs.initial_account import MODES
from src.training.freeze import json_digest
from tools import analyze_delta_actor as analysis
from tests.test_trading_env import market


@pytest.fixture
def envelope(tmp_path, monkeypatch):
    frozen = {f"src/module_{i}.py": hashlib.sha256(str(i).encode()).hexdigest() for i in range(70)}
    protocol = dict(study_mode="delta", action_mode="stock_delta", seeds=[42, 123, 2026],
                    main_top_n=20, evaluation_seed=20260927, evaluation_modes=list(MODES),
                    short_steps=8192, long_steps=65536, source_hashes=frozen, test_evaluated=False)
    status = dict(status="complete", checkpoints=6, total_trained_steps=196608, test_evaluated=False)
    summary = dict(seeds=[42, 123, 2026], main_top_n=20, total_runs=3, total_checkpoints=6,
                   actual_formal_steps=196608, protocol_sha256=json_digest(protocol), test_evaluated=False)
    records = [dict(seed=seed, budget=budget, steps=steps, initialization_mode="delta", test_evaluated=False)
               for seed in [42, 123, 2026] for budget, steps in analysis.BUDGETS.items()]
    for name, value in [("protocol", protocol), ("status", status), ("study_summary", summary), ("summary", records)]:
        write_json(tmp_path / (name + ".json"), value)
    monkeypatch.setattr(analysis, "source_hashes", lambda: deepcopy(frozen))
    return SimpleNamespace(root=tmp_path, protocol=protocol, sources=frozen, records=records)


def test_completed_envelope_preserves_all_six_identity_records(envelope):
    study, protocol, summary, records = analysis.read_envelope(envelope.root)
    assert study == envelope.root.resolve()
    assert protocol == envelope.protocol and records == envelope.records
    assert summary["main_top_n"] == 20


@pytest.mark.parametrize("damage", ["status", "missing_summary", "missing_seed", "test", "source", "count"])
def test_analysis_rejects_unfinished_or_changed_study_identity_before_output(envelope, damage):
    root = envelope.root
    if damage == "status":
        value = read_config(root / "status.json")
        value["status"] = "training"
        write_json(root / "status.json", value)
    elif damage == "missing_summary":
        (root / "study_summary.json").rename(root / "unpublished_summary.json")
    elif damage == "missing_seed":
        write_json(root / "summary.json", envelope.records[:-2])
    elif damage == "test":
        value = read_config(root / "study_summary.json")
        value["test_evaluated"] = True
        write_json(root / "study_summary.json", value)
    else:
        value = deepcopy(envelope.protocol)
        if damage == "source":
            value["source_hashes"]["src/module_0.py"] = "changed"
        else:
            value["source_hashes"].pop("src/module_0.py")
        write_json(root / "protocol.json", value)
        summary = read_config(root / "study_summary.json")
        summary["protocol_sha256"] = json_digest(value)
        write_json(root / "study_summary.json", summary)
    with pytest.raises(ValueError):
        analysis.read_envelope(root)
    assert not (root / "analysis").exists()


@pytest.mark.parametrize("missing", ["config", "contract", "source_hashes", "num_timesteps", "model_sha256"])
def test_model_identity_must_not_be_missing(tmp_path, missing):
    (tmp_path / "model.zip").write_bytes(b"synthetic snapshot")
    config, contract, frozen = dict(seed=42), dict(action_mode="stock_delta"), {"src/a.py": "identity"}
    metadata = dict(config=config, contract=contract, source_hashes=frozen, num_timesteps=8192,
                    model_sha256=file_hash(tmp_path / "model.zip"))
    metadata.pop(missing)
    write_json(tmp_path / "metadata.json", metadata)
    with pytest.raises(ValueError, match="lacks a registered identity"):
        analysis.verify_snapshot(tmp_path, config, contract, frozen, 8192)


def test_model_and_registered_control_file_hashes_are_verified(tmp_path):
    (tmp_path / "model.zip").write_bytes(b"synthetic snapshot")
    config, contract, frozen = dict(seed=42), dict(action_mode="stock_delta"), {"src/a.py": "identity"}
    digest = file_hash(tmp_path / "model.zip")
    metadata = dict(config=config, contract=contract, source_hashes=frozen, num_timesteps=8192, model_sha256=digest)
    write_json(tmp_path / "metadata.json", metadata)
    assert analysis.verify_snapshot(tmp_path, config, contract, frozen, 8192, digest) == metadata
    registered = {}
    analysis.register_files(tmp_path, {"model.zip": digest}, registered)
    assert registered == {str((tmp_path / "model.zip").resolve()): digest}
    (tmp_path / "model.zip").write_bytes(b"changed snapshot")
    with pytest.raises(ValueError, match="SHA identity changed"):
        analysis.verify_snapshot(tmp_path, config, contract, frozen, 8192, digest)
    with pytest.raises(ValueError, match="artifact changed"):
        analysis.register_files(tmp_path, {"model.zip": digest}, {})


def test_reference_identity_must_be_registered():
    with pytest.raises(ValueError, match="prior/random evidence is missing"):
        analysis.verify_references({}, {})


def test_reference_manifests_are_verified_without_running_legacy_models(tmp_path, monkeypatch):
    references = {}
    evidence = dict(verified_file_count=1)
    for mode in ("prior", "random"):
        directory = tmp_path / mode
        run = directory / "runs" / "small_42"
        run.mkdir(parents=True)
        artifact = run / "artifact.bin"
        artifact.write_bytes(mode.encode())
        write_json(run / "completed.json", dict(files={"artifact.bin": file_hash(artifact)}))
        for name in ("protocol", "status", "summary"):
            write_json(directory / (name + ".json"), {})
        references[mode] = dict(root=str(directory), evidence=evidence)
    monkeypatch.setattr(analysis, "load_matrix", lambda root, mode: dict(evidence=evidence, protocol=dict(core_contract={})))
    protocol = dict(references=references, core_contract={})
    manifest = {}
    analysis.verify_references(protocol, manifest)
    assert len(manifest) == 10
    artifact = tmp_path / "prior" / "runs" / "small_42" / "artifact.bin"
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Registered artifact changed"):
        analysis.verify_references(protocol, {})


def runtime_fixture():
    obs = dict(stock_features=np.zeros((2, 1, 1), dtype=np.float32), portfolio=np.array([.1, .1, .8], dtype=np.float32),
               receivable_ratio=np.array([0.], dtype=np.float32), valid_mask=np.ones(2, dtype=np.int8),
               buy_mask=np.ones(2, dtype=np.int8), sell_mask=np.ones(2, dtype=np.int8),
               execution_features=np.array([[.01, .1, 0], [.01, .1, 0]], dtype=np.float32),
               allocation_context=np.array([10., .5, 0., 1., 0., 9.], dtype=np.float32),
               top_n=np.array([20.], dtype=np.float32))
    mapped = map_delta_action([.001, -.002], obs, .5, .05)
    mapped["future_numeric_diagnostic"] = np.array([7., 8.], dtype=np.float32)
    recorder = SimpleNamespace(observations=[obs, deepcopy(obs)], actions=[mapped, deepcopy(mapped)])
    return recorder


def test_npz_contains_every_actual_runtime_key_without_pickle(tmp_path):
    recorder = runtime_fixture()
    arrays, schema = analysis.runtime_arrays(recorder)
    path = tmp_path / "actions.npz"
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as loaded:
        assert {item["name"] for item in schema["action"]} == set(recorder.actions[0])
        for name in recorder.actions[0]:
            np.testing.assert_array_equal(loaded["action__" + name], np.stack([row[name] for row in recorder.actions]))
        assert loaded["action__freeze_mask"].dtype == bool
        assert loaded["observation__valid_mask"].shape == (2, 2)
        np.testing.assert_array_equal(loaded["action__future_numeric_diagnostic"], [[7, 8], [7, 8]])


def test_descriptive_statistics_measure_fills_lots_and_minimum_fees():
    arrays, _ = analysis.runtime_arrays(runtime_fixture())
    daily = pd.DataFrame(dict(date=pd.date_range("2024-01-02", periods=3, freq="B"),
                              cash=[10000., 8999.995, 8999.995], holding_count=[2, 2, 2]))
    orders = pd.DataFrame([dict(date=daily.date.iloc[1], side="sell", requested_shares=-1., filled_shares=-1., price=10., fee=5.005,
                                slippage_cost=.01, status="filled"),
                           dict(date=daily.date.iloc[1], side="buy", requested_shares=100., filled_shares=100., price=10., fee=5.,
                                slippage_cost=1., status="filled")])
    stats, thresholds = analysis.describe_actions(arrays, daily, orders,
                                                  dict(sell_tax=.0005, min_commission=5.), .05)
    assert stats["filled_buy_orders"] == stats["filled_sell_orders"] == 1
    assert stats["one_share_sells"] == 1
    assert stats["minimum_commission_orders"] == 2
    assert stats["minimum_commission_sells"] == 1
    assert stats["filled_buy_notional"] == 1000 and stats["filled_sell_notional"] == 10
    assert stats["requested_notional_at_logged_price"] == 1010
    assert stats["fees"] == pytest.approx(10.005) and stats["total_cost"] == pytest.approx(11.015)
    assert stats["no_order_days"] == 1 and stats["zero_delta_fraction"] == .5
    assert stats["positive_eligible_buy_latents_below_lot"] == 2
    np.testing.assert_allclose(thresholds, np.arctanh(.2), rtol=1e-6)


def test_model_rows_require_all_four_accounts_for_each_of_six_snapshots():
    statistics = dict(filled_buy_orders=1, filled_sell_orders=2, one_share_sells=1,
                      fees=5., slippage=1., total_cost=6., posthoc_abs_eligible_latent_median=.001)
    rows = [dict(seed=seed, budget=budget, account=mode, **statistics)
            for seed in analysis.SEEDS for budget in analysis.BUDGETS for mode in MODES]
    grouped = analysis.model_rows(rows)
    assert len(grouped) == 6 and all(row["accounts"] == 4 for row in grouped)
    assert all(row["filled_buys"] == 4 and row["mean_account_cost"] == 6. for row in grouped)
    with pytest.raises(ValueError, match="lacks four accounts"):
        analysis.model_rows(rows[:-1])


def test_actual_zero_delta_replay_matches_all_sealed_tables_including_empty_orders(tmp_path):
    contract = dict(action_mode="stock_delta", logit_bound=8., delta=dict(max_delta_weight=.05, top_n=20))

    class Agent:
        def __init__(self):
            self.contract = contract

        def inspect_action(self, obs, deterministic=True):
            assert deterministic
            return map_delta_action(np.zeros(2), obs, .6, .05)

    recorder = analysis.ReplayRecorder(Agent())
    settings = dict(initial_capital=10000., seed=20260927, initial_portfolio="equal",
                    delta_top_n=20, broker=dict(max_weight=.6, commission=0.))
    daily, positions, orders, _ = run_policy(market(), recorder, settings)
    saved = tmp_path / "sealed" / "ppo"
    saved.mkdir(parents=True)
    daily.to_csv(saved / "daily.csv", index=False)
    positions.to_parquet(saved / "positions.parquet")
    orders.to_csv(saved / "orders.csv", index=False)
    analysis.compare_replay(daily, positions, orders, saved.parent)
    assert len(recorder.actions) == 4 and orders.empty
    altered = daily.copy()
    altered.loc[1, "cash"] += 1.
    with pytest.raises(ValueError, match="differs from sealed daily"):
        analysis.compare_replay(altered, positions, orders, saved.parent)


def test_analysis_refuses_output_overwrite_before_loading_any_model(tmp_path, monkeypatch):
    (tmp_path / "analysis" / "main_k20").mkdir(parents=True)
    monkeypatch.setattr(analysis, "verify_study", lambda path: pytest.fail("Existing output must fail before loading"))
    with pytest.raises(FileExistsError):
        analysis.analyze(tmp_path)


def test_cuda_option_is_explicit_and_passed_to_inference_analysis_only(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(analysis, "analyze", lambda study, device: calls.append((study, device)) or tmp_path)
    assert analysis.main(["--study", str(tmp_path), "--device", "cuda"]) == tmp_path
    assert calls == [(tmp_path, "cuda")]


def test_replay_mismatch_never_publishes_an_aggregate(tmp_path, monkeypatch):
    obs = runtime_fixture().observations[0]

    class Agent:
        contract = dict(action_mode="stock_delta")
        model = SimpleNamespace(num_timesteps=8192)

        def inspect_action(self, observation, deterministic=True):
            return map_delta_action(np.zeros(2), observation, .5, .05)

    data = SimpleNamespace(market=lambda split: "val-only", contract=Agent.contract)
    case = dict(data=data, model_directory=tmp_path, metadata=dict(num_timesteps=8192), settings={}, saved=tmp_path)
    protocol = dict(train_end="2024-01-02", val_end="2024-01-03")
    monkeypatch.setattr(analysis, "verify_study", lambda study: (tmp_path, protocol, [case], {}))
    monkeypatch.setattr(analysis.PPOAgent, "load", lambda *args: Agent())
    monkeypatch.setattr(analysis, "policy_digest", lambda agent: "unchanged")

    def fake_replay(market_value, recorder, settings):
        assert market_value == "val-only"
        recorder.act(obs)
        return pd.DataFrame(dict(date=pd.to_datetime(["2024-01-02", "2024-01-03"]))), None, None, []

    def mismatch(*args):
        raise ValueError("CPU replay differs from sealed orders")

    monkeypatch.setattr(analysis, "run_policy", fake_replay)
    monkeypatch.setattr(analysis, "compare_replay", mismatch)
    with pytest.raises(ValueError, match="differs from sealed orders"):
        analysis.analyze(tmp_path)
    assert not (tmp_path / "analysis" / "main_k20").exists()
    assert not list((tmp_path / "analysis").rglob("summary.json"))
