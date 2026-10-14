"""Known-policy archive/schema checks using synthetic non-financial PPO."""

from copy import deepcopy
from dataclasses import asdict
import json
import zipfile

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from src.agents.timing_agent import TimingAgent, InferenceOnlyTimingPPO
from src.agents.timing_policy import CalendarTimingPolicy
from src.data.preprocess import FEATURES, file_hash, write_json
from src.envs.broker import BrokerConfig
from src.training.timing_env import build_timing_contract
from tests.test_timing_policy import SyntheticTimingEnv
from tools.timing_phase_agent import PhaseTimingAgent, phase_policy_contract, same_json
from tools.timing_phase_policy import DuePhaseTimingPolicy


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(architecture="due_phase_residual"):
    env = SyntheticTimingEnv()
    kind = CalendarTimingPolicy if architecture == "linear" else DuePhaseTimingPolicy
    model = PPO(kind, env, seed=42, device="cpu", n_steps=8, batch_size=4, n_epochs=2,
                gamma=.995, gae_lambda=1.)
    base = dict(action_mode="stock_trade", stock_codes=["000001", "600000"], feature_cols=list(FEATURES),
        lookback=1, price_basis="hfq_research", broker=asdict(BrokerConfig()), data_source=dict(provider="synthetic_only"),
        artifacts={"scaler_params.json": "0" * 64, "panel_data.parquet": "1" * 64, "prices.parquet": "2" * 64})
    contract = phase_policy_contract(build_timing_contract(base, "relative_log"), architecture)
    return model, contract, env


@pytest.mark.parametrize("architecture", ["linear", "due_phase_residual"])
def test_known_policy_round_trip_preserves_tensors_macro_probability_value_and_optimizer(tmp_path, architecture):
    model, contract, env = fixture(architecture)
    try:
        model.learn(16)
        agent = PhaseTimingAgent(model, contract)
        vector, _ = env.reset()
        before = agent.inspect_action(vector)
        folder = tmp_path / architecture
        agent.save(folder, dict(synthetic_only=True))
        loaded = PhaseTimingAgent.load(folder, contract)
        assert type(loaded.model) is InferenceOnlyTimingPPO
        after = loaded.inspect_action(vector)
        for key in ("macro_action", "phase", "raw_log_prob", "value"):
            assert before[key] == after[key]
        np.testing.assert_array_equal(before["action_probabilities"], after["action_probabilities"])
        assert all(torch.equal(value, loaded.model.policy.state_dict()[key]) for key, value in model.policy.state_dict().items())
        assert len(loaded.model.policy.optimizer.state) == (8 if architecture == "linear" else 9)
        with pytest.raises(ValueError, match="Inference-only"):
            loaded.model.learn(8)
        resumed = PhaseTimingAgent.load(folder, contract, allow_training=True)
        assert type(resumed.model) is PPO
        resumed.model.set_env(env)
        resumed.model.learn(8, reset_num_timesteps=False)
        assert resumed.model.num_timesteps == 24
    finally:
        env.close()


def test_old_exact_class_guard_is_not_bypassed():
    model, contract, env = fixture()
    try:
        with pytest.raises(ValueError, match="policy spaces"):
            TimingAgent(model, contract)
        assert PhaseTimingAgent(model, contract).act(env.reset()[0]) == 1
    finally:
        env.close()


@pytest.mark.parametrize("change", ["policy", "source", "tool", "contract", "model_hash", "algorithm", "typed_due_phase", "parameter_count"])
def test_metadata_mismatch_rejected_before_ppo_deserialization(tmp_path, monkeypatch, change):
    model, contract, env = fixture()
    try:
        folder = tmp_path / "artifact"
        PhaseTimingAgent(model, contract).save(folder, {})
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if change == "policy":
            metadata["policy_record"]["policy_class"] = "untrusted.module.CustomPolicy"
        elif change == "source":
            metadata["code_identity"]["source_hashes"]["src/agents/timing_policy.py"] = "0" * 64
        elif change == "tool":
            metadata["code_identity"]["tool_hashes"]["tools/timing_phase_policy.py"] = "0" * 64
        elif change == "contract":
            metadata["contract"]["stock_codes"].reverse()
        elif change == "algorithm":
            metadata["algorithm_class"] = "untrusted.module.Algorithm"
        elif change == "typed_due_phase":
            metadata["policy_record"]["due_phase"] = False
        elif change == "parameter_count":
            metadata["policy_record"]["parameter_count"] = 12805
        else:
            metadata["model_sha256"] = "0" * 64
        write_json(folder / "metadata.json", metadata)
        monkeypatch.setattr(InferenceOnlyTimingPPO, "load", lambda *_args, **_kwargs: pytest.fail("Must reject before deserialization"))
        with pytest.raises(ValueError, match="artifact"):
            PhaseTimingAgent.load(folder, contract)
    finally:
        env.close()


def test_resigned_archive_policy_contract_change_is_rejected_before_loader(tmp_path, monkeypatch):
    model, contract, env = fixture()
    try:
        folder = tmp_path / "artifact"
        PhaseTimingAgent(model, contract).save(folder, {})
        path = folder / "model.zip"
        with zipfile.ZipFile(path, "r") as archive:
            content = {name: archive.read(name) for name in archive.namelist()}
        data = json.loads(content["data"])
        data["phase_registered_contract"]["phase_policy"]["architecture"] = "linear"
        content["data"] = json.dumps(data).encode()
        with zipfile.ZipFile(path, "w") as archive:
            for name, value in content.items():
                archive.writestr(name, value)
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        metadata["model_sha256"] = file_hash(path)
        write_json(folder / "metadata.json", metadata)
        monkeypatch.setattr(InferenceOnlyTimingPPO, "load", lambda *_args, **_kwargs: pytest.fail("Must reject before deserialization"))
        with pytest.raises(ValueError, match="archive"):
            PhaseTimingAgent.load(folder, contract)
    finally:
        env.close()


@pytest.mark.parametrize("change", ["feature_order", "residual_bias", "parameter_count", "class", "missing_optimizer_tensor", "typed_due_phase"])
def test_contract_or_trainable_schema_mismatch_cannot_construct_agent(change):
    model, contract, env = fixture()
    try:
        if change == "feature_order":
            contract["timing"]["feature_names"][20:22] = list(reversed(contract["timing"]["feature_names"][20:22]))
        elif change == "residual_bias":
            contract["phase_policy"]["residual_bias"] = True
        elif change == "parameter_count":
            contract["phase_policy"]["parameter_count"] = 12805
        elif change == "class":
            contract["phase_policy"]["policy_class"] = "unknown.Policy"
        elif change == "typed_due_phase":
            contract["phase_policy"]["due_phase"] = False
        else:
            model.policy.optimizer.param_groups[0]["params"] = list(model.policy.parameters())[:-1]
        with pytest.raises(ValueError):
            PhaseTimingAgent(model, contract)
    finally:
        env.close()


def test_mutated_optimizer_is_rejected_before_creating_save_directory(tmp_path):
    model, contract, env = fixture()
    try:
        agent = PhaseTimingAgent(model, contract)
        model.policy.optimizer.param_groups[0]["params"] = list(model.policy.parameters())[:-1]
        with pytest.raises(ValueError, match="optimizer"):
            agent.save(tmp_path / "artifact", {})
        assert not (tmp_path / "artifact").exists()
    finally:
        env.close()


@pytest.mark.parametrize("value", [False, np.bool_(False)])
def test_nested_record_comparison_does_not_coerce_false_to_phase_zero(value):
    assert not same_json({"phase_policy": {"due_phase": value}}, {"phase_policy": {"due_phase": 0}})


def test_unregistered_subclass_and_nonboolean_resume_are_rejected(tmp_path):
    class Unregistered(DuePhaseTimingPolicy):
        pass
    model, contract, env = fixture()
    try:
        unknown = PPO(Unregistered, env, seed=42, device="cpu", n_steps=8, batch_size=4, n_epochs=2)
        with pytest.raises(ValueError, match="policy class"):
            PhaseTimingAgent(unknown, contract)
        with pytest.raises(ValueError, match="boolean"):
            PhaseTimingAgent.load(tmp_path / "absent", contract, allow_training=1)
        with pytest.raises(ValueError, match="Unregistered"):
            phase_policy_contract(contract, "nonlinear")
        with pytest.raises(ValueError, match="migration"):
            PhaseTimingAgent(model, contract).transfer_to(env, {}, contract)
    finally:
        env.close()
