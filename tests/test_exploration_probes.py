"""No learning or real-data performance selection in deployment probe tests."""

from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest
import torch

from src.agents.ppo_agent import PPOAgent, make_model
from src.data.preprocess import file_hash
from src.training.env_factory import ResearchData
from src.training import exploration_probes as probes
from tests.test_ppo_framework import local_data, no_parameter_updates
from tests.test_trade_policy import trade_config


pytestmark = pytest.mark.usefixtures("no_parameter_updates")


def synthetic_agent(local_data):
    data = ResearchData(trade_config(local_data))
    env = data.vector_env()
    model = make_model(env, data.config)
    agent = PPOAgent(model, data.contract)
    obs = {key: value[0] for key, value in env.reset().items()}
    return agent, data, env, obs


def rng_snapshot():
    return torch.random.get_rng_state().clone(), deepcopy(np.random.get_state()), random.getstate()


def assert_rng_unchanged(before):
    assert torch.equal(before[0], torch.random.get_rng_state())
    after = np.random.get_state()
    assert before[1][0] == after[0]
    np.testing.assert_array_equal(before[1][1], after[1])
    assert before[1][2:] == after[2:]
    assert before[2] == random.getstate()


def test_same_state_samples_keep_exact_support_parameters_rng_and_observation(local_data):
    agent, data, env, obs = synthetic_agent(local_data)
    try:
        obs["valid_mask"][:] = 1
        obs["execution_features"][:, 2] = 0
        obs["legal_directions"][:] = [[1, 1, 1, 1], [1, 0, 0, 0]]
        obs["trade_lot_bounds"][:] = [[1, 4, 1, 3], [0, 0, 0, 0]]
        untouched = deepcopy(obs)
        parameters = probes.parameter_digest(agent.model.policy)
        rng = rng_snapshot()
        summary, arrays = probes.sample_same_state(agent, obs, samples=64, seed=101)
        assert_rng_unchanged(rng)
        assert probes.parameter_digest(agent.model.policy) == parameters
        for key in untouched:
            np.testing.assert_array_equal(obs[key], untouched[key])
        assert summary["samples"] == 64 and summary["environment_steps"] == 0
        assert summary["actual_fills_evaluated"] is False
        assert arrays["raw_packed_action"].shape == (64, 4)
        assert arrays["proposed_shares"].shape == (64, 2)
        assert arrays["projected_shares"].shape == (64, 2)
        assert np.isfinite(arrays["log_prob"]).all()
        assert (arrays["raw_packed_action"][:, 1] == 0).all()
        assert (arrays["raw_packed_action"][:, 3] == 0).all()
        summary2, arrays2 = probes.sample_same_state(agent, obs, samples=64, seed=101)
        assert summary == summary2
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], arrays2[key])
        assert agent.model.num_timesteps == 0
    finally:
        env.close()


@pytest.mark.parametrize("samples,seed", [(0, 1), (1, -1), (1.5, 1), (1, .5)])
def test_same_state_invalid_sample_parameters_rejected(samples, seed):
    with pytest.raises(ValueError, match="count/seed"):
        probes.sample_same_state(None, None, samples=samples, seed=seed)


def test_real_shape_transfer_retains_shared_tensors_without_optimizer_steps(local_data):
    agent, original, env, obs = synthetic_agent(local_data)
    vector = None
    try:
        config = deepcopy(original.config)
        config["stock_codes"] = original.codes[:1]
        data = ResearchData(config)
        data.settings["initial_capital"] = 250000.
        data.contract["initial_capital"] = 250000.
        expected = probes.parameter_digest(agent.model.policy)
        transferred, vector = probes._transferred_model(agent, data)
        assert transferred.model.action_space.shape == (2,)
        assert transferred.model.observation_space["stock_features"].shape[0] == 1
        assert transferred.contract["initial_capital"] == 250000.
        assert probes.parameter_digest(transferred.model.policy) == expected
        new_obs = {key: value[0] for key, value in vector.reset().items()}
        transferred.validate_observation(new_obs)
        result, _ = probes.sample_same_state(transferred, new_obs, samples=2)
        assert result["parameters_unchanged"]
        assert transferred.model.num_timesteps == 0
        assert probes.parameter_digest(agent.model.policy) == expected
    finally:
        env.close()
        if vector is not None:
            vector.close()


def test_feasibility_counts_overlap_and_never_calls_next_day_execution(local_data):
    agent, data, env, obs = synthetic_agent(local_data)
    try:
        account = env.envs[0].unwrapped._info()
        before = account["equity"]
        obs["valid_mask"][1] = 0
        obs["execution_features"][1, 2] = 1
        obs["legal_directions"][1] = [1, 0, 0, 0]
        obs["trade_lot_bounds"][1] = [2, 0, 0, 0]
        contract = deepcopy(data.contract)
        contract["broker"]["min_commission"] = 5.
        result = probes.feasibility_summary(obs, account, contract)
        assert result["blockers_overlap"]
        assert result["blockers"]["invalid"] == 1
        assert result["blockers"]["frozen"] == 1
        assert result["blockers"]["economic_lot_interval_empty"] >= 1
        assert result["minimum_trade_notional"] == 5000.
        assert result["mask_frozen_only_hold"]
        assert env.envs[0].unwrapped._info()["equity"] == before
        obs["legal_directions"][1, 1] = 1
        with pytest.raises(AssertionError, match="non-HOLD"):
            probes.feasibility_summary(obs, account, contract)
    finally:
        env.close()


def test_one_complete_synthetic_deployment_retains_native_original_probability(local_data, tmp_path, monkeypatch):
    agent, data, env, _ = synthetic_agent(local_data)
    try:
        monkeypatch.setattr(probes, "_deployment_market", lambda value: value.market("train"))
        identity = {"parameter_sha256": probes.parameter_digest(agent.model.policy), "model_seed": 42,
                    "budget": "short", "steps": 8192, "model_sha256": "synthetic",
                    "contract_sha256": probes.json_digest(data.contract)}
        rng = rng_snapshot()
        result = probes._one_deployment(agent, data, identity, tmp_path / "case", "cash", 101)
        assert_rng_unchanged(rng)
        assert result["deterministic"] is False and result["sampling_seed"] == 101
        assert result["environment_steps"] > 0
        assert result["mode_statistics"]["economic_violations"] == 0
        folder = tmp_path / "case" / "native_trace"
        manifest = json.loads((folder / "index.json").read_text())
        assert manifest["total_steps"] == result["environment_steps"]
        path = folder / manifest["chunks"][0]["path"]
        with np.load(path, allow_pickle=False) as trace:
            assert np.isfinite(trace["log_prob"]).all()
            assert trace["done"][-1]
            assert not trace["done"][:-1].any()
            assert trace["raw_packed_action"].shape[1] == 4
            assert "decision_positions" in trace and "opening_positions" in trace
        assert (tmp_path / "case" / "daily.csv").is_file()
        assert (tmp_path / "case" / "positions.parquet").is_file()
        assert (tmp_path / "case" / "orders.csv").is_file()
        assert (tmp_path / "case" / "metrics.json").is_file()
        assert probes.parameter_digest(agent.model.policy) == identity["parameter_sha256"]
        assert agent.model.num_timesteps == 0
    finally:
        env.close()


def test_deployment_orchestration_keeps_every_registered_sampling_path(local_data, tmp_path, monkeypatch):
    agent, data, env, _ = synthetic_agent(local_data)
    try:
        identity = {"parameter_sha256": probes.parameter_digest(agent.model.policy), "model_seed": 42,
                    "budget": "short", "steps": 8192, "model_sha256": "synthetic"}
        monkeypatch.setattr(probes, "_frozen_models", lambda root: iter([(agent, data, identity)]))
        monkeypatch.setattr(probes, "_deployment_market", lambda value: value.market("train"))
        monkeypatch.setattr(probes, "SAMPLING_SEEDS", (101, 202))
        monkeypatch.setattr(probes, "PROBE_INDICES", (-3, -2))
        rng = rng_snapshot()
        result = probes.run_deployment_probes("unused", tmp_path)
        assert_rng_unchanged(rng)
        assert result["path_case_count"] == 4 * 3
        assert result["same_state_case_count"] == 4
        assert result["training_steps"] == result["training_updates"] == 0
        for mode in probes.MODES:
            rows = [row for row in result["cases"] if row["mode"] == mode]
            assert len(rows) == 3
            assert {row["sampling_seed"] for row in rows} == {None, 101, 202}
        for row in result["same_state_cases"]:
            assert row["environment_steps"] == 0 and row["samples"] == 64
            with np.load(row["path"], allow_pickle=False) as arrays:
                assert arrays["raw_packed_action"].shape[0] == 64
        assert (tmp_path / "deployment" / "summary.json").is_file()
    finally:
        env.close()


def test_feasibility_orchestration_accounts_and_different_shapes_are_observation_only(local_data, tmp_path, monkeypatch):
    agent, data, env, _ = synthetic_agent(local_data)
    try:
        identity = {"parameter_sha256": probes.parameter_digest(agent.model.policy), "model_seed": 42,
                    "budget": "short", "steps": 8192, "model_sha256": "synthetic"}
        monkeypatch.setattr(probes, "_frozen_models", lambda root: iter([(agent, data, identity)]))
        monkeypatch.setattr(probes, "POOL_SIZES", (1, 2))
        monkeypatch.setattr(probes, "CAPITALS", (250000., 1000000.))
        monkeypatch.setattr(probes, "PROBE_INDICES", (-3, -2))
        rng = rng_snapshot()
        result = probes.run_feasibility_probes("unused", tmp_path)
        assert_rng_unchanged(rng)
        assert result["case_count"] == 16
        assert result["training_steps"] == result["training_updates"] == result["environment_steps"] == 0
        for row in result["cases"]:
            assert row["equity_unchanged"] and row["parameters_unchanged"]
            assert row["stock_codes"] == data.codes[:row["n_stocks"]]
            assert row["contract"]["initial_capital"] == row["capital"]
            assert row["parameters_sha256"] == identity["parameter_sha256"]
            with np.load(row["path"], allow_pickle=False) as arrays:
                assert arrays["deterministic_raw"].shape == (2 * row["n_stocks"],)
                assert arrays["account_positions"].shape == (row["n_stocks"],)
        assert (tmp_path / "feasibility" / "summary.json").is_file()
        assert agent.model.num_timesteps == 0
    finally:
        env.close()


def test_frozen_original_sources_and_model_zip_are_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(probes, "ROOT", tmp_path)
    sources = {}
    for index in range(73):
        relative = f"src/module_{index}.py"
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text("pass\n", encoding="ascii")
        sources[relative] = file_hash(path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.zip").write_bytes(b"synthetic model identity")
    metadata = dict(model_sha256=file_hash(model / "model.zip"), source_hashes=sources, contract={})
    (model / "metadata.json").write_text(json.dumps(metadata), encoding="ascii")
    result, identity = probes.verify_old_identity(model)
    assert result == metadata and identity["original_source_count"] == 73
    (tmp_path / "src/module_0.py").write_text("changed\n", encoding="ascii")
    with pytest.raises(ValueError, match="source identity changed"):
        probes.verify_old_identity(model)
    (model / "model.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="ZIP identity"):
        probes.verify_old_identity(model)


def test_reward_identities_distinguish_sum_compounding_and_discounted_logs():
    daily = pd.DataFrame(dict(equity=[100., 110., 99., 118.8], daily_return=[0., .1, -.1, .2]))
    result = probes.reward_math_identity(daily, gamma=.99)
    assert result["sum_simple_rewards"] == pytest.approx(.2)
    assert result["compound_return"] == pytest.approx(.188)
    assert result["simple_sum_minus_compound"] == pytest.approx(.012)
    assert result["sum_log_rewards"] == pytest.approx(np.log(1.188))
    assert abs(result["log_identity_error"]) < 1e-12
    assert abs(result["discounted_log_minus_terminal"]) > 1e-6
    undiscounted = probes.reward_math_identity(daily, gamma=1)
    assert abs(undiscounted["discounted_log_minus_terminal"]) < 1e-12


@pytest.mark.parametrize("daily,gamma", [
    (pd.DataFrame(dict(equity=[100.], daily_return=[0.])), .99),
    (pd.DataFrame(dict(equity=[100., 0.], daily_return=[0., -1.])), .99),
    (pd.DataFrame(dict(equity=[100., 110.], daily_return=[0., .1])), -1),
    (pd.DataFrame(dict(equity=[100., 110.], daily_return=[0., .1])), 2)])
def test_reward_invalid_equity_or_gamma_is_rejected(daily, gamma):
    with pytest.raises(ValueError):
        probes.reward_math_identity(daily, gamma)


def test_reward_wrong_historical_rewards_fail_instead_of_silent_recalculation():
    daily = pd.DataFrame(dict(equity=[100., 110.], daily_return=[0., .2]))
    with pytest.raises(AssertionError):
        probes.reward_math_identity(daily)


@pytest.mark.parametrize("stage", ["deployment", "feasibility", "reward"])
def test_cli_dispatch_only_requested_stage(stage, monkeypatch, capsys):
    called = []
    for name in ("run_deployment_probes", "run_feasibility_probes", "run_reward_math"):
        monkeypatch.setattr(probes, name, lambda old, output, name=name:
            called.append((name, old, output)) or {"stage": stage, "status": "complete"})
    probes.main(["--stage", stage, "--old-root", "old", "--output-root", "output"])
    assert len(called) == 1 and called[0][1:] == (Path("old"), Path("output"))
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
