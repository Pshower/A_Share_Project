"""Deterministic inference, strict artifact contracts and explicit pool transfer."""

from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from src.backtest.run import git_metadata
from src.data.preprocess import ROOT, file_hash, write_json
from src.runtime.device import resolve_device
from .action_mapping import map_action, MAPPING_VERSION, mapping_version
from .ppo_policy import SharedStockPolicy
from .delta_policy import SharedDeltaPolicy
from .delta_mapping import DELTA_MAPPING_VERSION, DeltaOrder, map_delta_action


def make_model(env, config):
    torch.set_num_threads(config["torch_threads"])
    options = deepcopy(config["ppo"])
    if config.get("learning_rate_schedule", "constant") == "linear":
        initial, factor = options["learning_rate"], config.get("learning_rate_end_factor", 0.1)
        options["learning_rate"] = lambda remaining: initial * (factor + (1 - factor) * min(1.0, max(0.0, remaining)))
    kwargs = dict(hidden_sizes=tuple(config["hidden_sizes"]),
                   initial_log_std=config.get("initial_log_std", 0.0),
                   prior_feature_index=config.get("prior_feature_index"),
                   prior_strength=config.get("prior_strength", 4.0), residual_scale=config.get("residual_scale", 0.25))
    policy = SharedStockPolicy
    if config.get("action_mode") == "stock_delta":
        policy = SharedDeltaPolicy
        kwargs = dict(hidden_sizes=tuple(config["hidden_sizes"]), initial_log_std=config.get("initial_log_std", -1.5),
                      actor_output_gain=config.get("actor_output_gain", .01))
    return PPO(policy, env, seed=config["seed"], device=resolve_device(config["device"]), policy_kwargs=kwargs,
               verbose=0, **options)


class PPOAgent:
    def __init__(self, model, contract, provenance=None):
        self.model = model
        self.contract = deepcopy(contract)
        self.provenance = deepcopy(provenance or {})
        n = len(contract["stock_codes"])
        delta = contract.get("action_mode") == "stock_delta"
        expected_mapping = DELTA_MAPPING_VERSION if delta else mapping_version(contract.get("action_top_k"))
        if (contract["mapping_version"] != expected_mapping
                or model.action_space.shape != ((n,) if delta else (n + 1,))
                or model.observation_space["stock_features"].shape != (n, contract["lookback"], len(contract["feature_cols"]))
                or not np.all(model.action_space.high == contract["logit_bound"])
                or not np.all(model.action_space.low == -contract["logit_bound"])):
            raise ValueError("Model spaces do not match the artifact contract")
        if delta:
            if not isinstance(model.policy, SharedDeltaPolicy) or "delta" not in contract or contract.get("prior") is not None:
                raise ValueError("Delta policy does not match its action contract")
            if (model.observation_space["execution_features"].shape != (n, 3)
                    or model.observation_space["allocation_context"].shape != (6,)
                    or model.policy.actor_output_gain != contract["delta"]["actor_output_gain"]):
                raise ValueError("Delta execution state or initialization differs from contract")
            return
        network = model.policy.network
        expected_prior = contract.get("prior")
        actual_prior = (dict(feature_index=network.prior_feature_index, strength=network.prior_strength,
                             residual_scale=network.residual_scale) if network.prior_feature_index is not None else None)
        if actual_prior != expected_prior:
            raise ValueError("Policy prior differs from model contract")

    def reset(self):
        pass

    def validate_observation(self, observation):
        if not self.model.observation_space.contains(observation):
            raise ValueError("Observation shape, dtype or bounds do not match the model")
        if any(not np.isfinite(v).all() for v in observation.values()):
            raise ValueError("Observation must contain only finite values")

    def inspect_action(self, observation, deterministic=True):
        self.validate_observation(observation)
        self.model.policy.set_training_mode(False)
        tensor, _ = self.model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            latent = self.model.policy.get_distribution(tensor).get_actions(deterministic).cpu().numpy()[0]
        if self.contract.get("action_mode") == "stock_delta":
            return map_delta_action(latent, observation, self.contract["broker"]["max_weight"],
                                    self.contract["delta"]["max_delta_weight"], self.contract["logit_bound"])
        return map_action(latent, observation, self.contract["broker"]["max_weight"], self.contract["logit_bound"],
                          self.contract.get("action_top_k"))

    def act(self, observation, step=None, deterministic=True):
        result = self.inspect_action(observation, deterministic)
        if self.contract.get("action_mode") == "stock_delta":
            return DeltaOrder(result["delta_weights"], result["freeze_mask"], result["exit_mask"],
                              result["top_n"], result)
        return result["weights"]

    def save(self, directory, config, extra=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.model.save(directory / "model.zip")
        metadata = dict(contract=self.contract, config=config, provenance=self.provenance,
                        num_timesteps=self.model.num_timesteps, git=git_metadata(),
                        actual_device=str(self.model.device), cuda_build=torch.version.cuda,
                        model_sha256=file_hash(directory / "model.zip"),
                        packages={name: version(name) for name in ["torch", "stable-baselines3", "gymnasium", "numpy", "pandas", "pyarrow"]},
                        source_hashes={str(p.relative_to(ROOT)): file_hash(p) for p in (ROOT / "src").rglob("*.py")},
                        extra=extra or {})
        write_json(directory / "metadata.json", metadata)

    @classmethod
    def load(cls, directory, expected_contract, device="cpu"):
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["contract"] != expected_contract:
            raise ValueError("Artifact contract mismatch; pool changes require explicit transfer")
        if file_hash(directory / "model.zip") != metadata["model_sha256"]:
            raise ValueError("Model artifact hash mismatch")
        model = PPO.load(directory / "model.zip", device=resolve_device(device))
        provenance = deepcopy(metadata.get("provenance", {}))
        provenance.update(loaded_from=str(directory.resolve()), loaded_model_sha256=metadata["model_sha256"])
        return cls(model, metadata["contract"], provenance)

    def transfer_to(self, env, config, new_contract):
        # Only the pool and data identity may change; scaling and execution stay fixed.
        if self.contract.get("data_source") != new_contract.get("data_source"):
            raise ValueError("Pool transfer cannot change the training data source")
        for key in ["action_top_k", "prior", "action_mode", "delta"]:
            if self.contract.get(key) != new_contract.get(key):
                raise ValueError(f"Incompatible transfer contract: {key}")
        for key in ["feature_cols", "lookback", "price_basis", "broker", "initial_capital",
                    "mapping_version", "logit_bound", "hidden_sizes"]:
            if self.contract[key] != new_contract[key]:
                raise ValueError(f"Incompatible transfer contract: {key}")
        if self.contract["artifacts"]["scaler_params.json"] != new_contract["artifacts"]["scaler_params.json"]:
            raise ValueError("Transfer must reuse the training scaler")
        model = make_model(env, config)
        model.policy.load_state_dict(self.model.policy.state_dict(), strict=True)
        return PPOAgent(model, new_contract, dict(transfer_from=self.contract,
                        source_num_timesteps=self.model.num_timesteps,
                        optimizer_reset=True, new_rollout=True))
