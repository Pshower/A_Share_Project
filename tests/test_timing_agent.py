"""Synthetic categorical timing artifacts; no financial inputs or backtests."""

from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from src.agents.timing_agent import InferenceOnlyTimingPPO, TimingAgent
from src.agents.timing_policy import CalendarTimingPolicy
from src.data.preprocess import FEATURES, write_json
from src.envs.broker import BrokerConfig
from src.training.timing_env import FEATURE_NAMES, build_timing_contract
from tests.test_timing_policy import SyntheticTimingEnv


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def model_and_contract(reward="log_net", *, policy=CalendarTimingPolicy, dimension=None):
    env = SyntheticTimingEnv(len(FEATURE_NAMES) if dimension is None else dimension)
    model = PPO(policy, env, seed=42, device="cpu", n_steps=8, batch_size=4, n_epochs=2)
    base = dict(action_mode="stock_trade", stock_codes=["000001", "600000"], feature_cols=list(FEATURES),
        lookback=1, price_basis="hfq_research", broker=asdict(BrokerConfig()),
        data_source=dict(provider="synthetic_only"), artifacts={"scaler_params.json": "0" * 64,
            "panel_data.parquet": "1" * 64, "prices.parquet": "2" * 64})
    return model, build_timing_contract(base, reward), env


def test_inspector_returns_only_the_sampled_macro_and_its_true_log_probability():
    model, contract, env = model_and_contract()
    try:
        agent = TimingAgent(model, contract)
        observation, _ = env.reset()
        result = agent.inspect_action(observation)
        assert result["macro_action"] == 1 and result["action_name"] == "rebalance_m20"
        assert result["action_probabilities"].shape == (2,)
        assert result["raw_log_prob"] == pytest.approx(np.log(result["action_probabilities"][1]))
        assert agent.act(observation) == 1
        assert "shares" not in result and "stock_actions" not in result
        tensor, _ = model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            value, probability, _ = model.policy.evaluate_actions(tensor, torch.tensor([1]))
        assert result["value"] == pytest.approx(value.item())
        assert result["raw_log_prob"] == pytest.approx(probability.item())
        with pytest.raises(ValueError, match="migration"):
            agent.transfer_to(env, {}, contract)
    finally:
        env.close()


@pytest.mark.parametrize("reward", ["log_net", "relative_log"])
def test_round_trip_restores_learned_weights_and_default_loader_cannot_train(tmp_path, reward):
    model, contract, env = model_and_contract(reward)
    try:
        model.learn(16)
        agent = TimingAgent(model, contract)
        observation, _ = env.reset()
        expected = agent.inspect_action(observation)
        folder = tmp_path / reward
        agent.save(folder, dict(timing=dict(reward_basis=reward)))
        loaded = TimingAgent.load(folder, contract)
        assert type(loaded.model) is InferenceOnlyTimingPPO
        assert loaded.feature_names == tuple(FEATURE_NAMES)
        actual = loaded.inspect_action(observation)
        assert actual["macro_action"] == expected["macro_action"]
        np.testing.assert_array_equal(actual["action_probabilities"], expected["action_probabilities"])
        assert actual["value"] == expected["value"]
        for name, parameter in model.policy.state_dict().items():
            torch.testing.assert_close(parameter, loaded.model.policy.state_dict()[name], rtol=0, atol=0)
        with pytest.raises(ValueError, match="Inference-only"):
            loaded.model.learn(8)
        with pytest.raises(ValueError, match="Inference-only"):
            loaded.model.train()
        assert loaded.model.num_timesteps == 16
        trainable = TimingAgent.load(folder, contract, allow_training=True)
        assert type(trainable.model) is PPO and not trainable.provenance["inference_only"]
        assert trainable.model.num_timesteps == 16
        assert {int(state["step"].item()) for state in trainable.model.policy.optimizer.state.values()} == {8}
        with pytest.raises(ValueError, match="standalone"):
            loaded.save(tmp_path / "bad_save", {})
    finally:
        env.close()


@pytest.mark.parametrize("change", ["mode", "mapping", "names", "period", "actions", "margin", "reward", "decoder", "benchmark"])
def test_contract_semantic_drift_is_rejected(change):
    model, contract, env = model_and_contract()
    try:
        wrong = deepcopy(contract)
        if change == "mode":
            wrong["action_mode"] = "stock_trade"
        elif change == "mapping":
            wrong["mapping_version"] = "old"
        elif change == "names":
            wrong["timing"]["feature_names"][-1] = "future_return"
        elif change == "period":
            wrong["timing"]["phase_period"] = 21
        elif change == "actions":
            wrong["timing"]["action_names"] = ["hold", "buy"]
        elif change == "margin":
            wrong["timing"]["calendar_margin"] = 1.
        elif change == "reward":
            wrong["timing"]["reward_basis"] = "profit"
        elif change == "decoder":
            wrong["timing"]["decoder_version"] = "target_weights"
        else:
            wrong["timing"]["benchmark_interval"] = 10
        with pytest.raises(ValueError):
            TimingAgent(model, wrong)
    finally:
        env.close()


def test_wrong_policy_or_vector_space_is_not_accepted():
    old, contract, old_env = model_and_contract(policy="MlpPolicy")
    short, _, short_env = model_and_contract(dimension=len(FEATURE_NAMES) - 1)
    correct, _, env = model_and_contract()
    try:
        for model in [old, short]:
            with pytest.raises(ValueError, match="policy spaces"):
                TimingAgent(model, contract)
        wrong = SimpleNamespace(policy=correct.policy, observation_space=correct.observation_space,
                                action_space=gym.spaces.Discrete(3))
        with pytest.raises(ValueError, match="policy spaces"):
            TimingAgent(wrong, contract)
    finally:
        old_env.close()
        short_env.close()
        env.close()


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "phase_zero", "phase_double"])
def test_bad_observations_are_rejected(kind):
    model, contract, env = model_and_contract()
    try:
        observation, _ = env.reset()
        if kind == "shape":
            observation = observation[:-1]
        elif kind == "dtype":
            observation = observation.astype(np.float64)
        elif kind == "nan":
            observation[-1] = np.nan
        elif kind == "phase_zero":
            observation[:20] = 0
        else:
            observation[1] = 1
        with pytest.raises(ValueError):
            TimingAgent(model, contract).inspect_action(observation)
    finally:
        env.close()


@pytest.mark.parametrize("field", ["algorithm", "names", "policy_record", "data_contract"])
def test_loader_rejects_metadata_or_archive_identity_tampering(tmp_path, field):
    model, contract, env = model_and_contract()
    try:
        folder = tmp_path / "model"
        TimingAgent(model, contract).save(folder, {})
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        expected = contract
        if field == "algorithm":
            metadata["algorithm_class"] = "src.training.priority_aux.RankAuxPPO"
        elif field == "names":
            metadata["feature_names"][-1] = "future_return"
        elif field == "policy_record":
            metadata["policy_record"]["calendar_margin"] = 1.
        else:
            metadata["contract"]["artifacts"]["prices.parquet"] = "3" * 64
            expected = deepcopy(metadata["contract"])
        write_json(folder / "metadata.json", metadata)
        with pytest.raises(ValueError):
            TimingAgent.load(folder, expected)
    finally:
        env.close()


def test_loader_rejects_wrong_expected_pool_and_artifact_hash(tmp_path):
    model, contract, env = model_and_contract()
    try:
        folder = tmp_path / "model"
        TimingAgent(model, contract).save(folder, {})
        wrong = deepcopy(contract)
        wrong["stock_codes"] = list(reversed(wrong["stock_codes"]))
        with pytest.raises(ValueError, match="contract mismatch"):
            TimingAgent.load(folder, wrong)
        path = folder / "model.zip"
        path.write_bytes(path.read_bytes() + b"corrupt")
        with pytest.raises(ValueError, match="hash mismatch"):
            TimingAgent.load(folder, contract)
    finally:
        env.close()


def test_nonfinite_parameters_are_not_saved(tmp_path):
    model, contract, env = model_and_contract()
    try:
        with torch.no_grad():
            model.policy.value_net.weight.fill_(float("nan"))
        with pytest.raises(FloatingPointError, match="parameters"):
            TimingAgent(model, contract).save(tmp_path / "not_saved", {})
        assert not (tmp_path / "not_saved").exists()
    finally:
        env.close()
