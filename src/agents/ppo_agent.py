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
from .action_mapping import map_action, MAPPING_VERSION
from .ppo_policy import SharedStockPolicy


def make_model(env, config):
    torch.set_num_threads(config["torch_threads"])
    return PPO(SharedStockPolicy, env, seed=config["seed"], device=config["device"],
               policy_kwargs=dict(hidden_sizes=tuple(config["hidden_sizes"])),
               verbose=0, **config["ppo"])


class PPOAgent:
    def __init__(self, model, contract, provenance=None):
        self.model = model
        self.contract = deepcopy(contract)
        self.provenance = deepcopy(provenance or {})
        n = len(contract["stock_codes"])
        if (contract["mapping_version"] != MAPPING_VERSION
                or model.action_space.shape != (n + 1,)
                or model.observation_space["stock_features"].shape != (n, contract["lookback"], len(contract["feature_cols"]))
                or not np.all(model.action_space.high == contract["logit_bound"])
                or not np.all(model.action_space.low == -contract["logit_bound"])):
            raise ValueError("Model spaces do not match the artifact contract")

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
        return map_action(latent, observation, self.contract["broker"]["max_weight"], self.contract["logit_bound"])

    def act(self, observation, step=None, deterministic=True):
        return self.inspect_action(observation, deterministic)["weights"]

    def save(self, directory, config, extra=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.model.save(directory / "model.zip")
        metadata = dict(contract=self.contract, config=config, provenance=self.provenance,
                        num_timesteps=self.model.num_timesteps, git=git_metadata(),
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
        model = PPO.load(directory / "model.zip", device=device)
        provenance = deepcopy(metadata.get("provenance", {}))
        provenance.update(loaded_from=str(directory.resolve()), loaded_model_sha256=metadata["model_sha256"])
        return cls(model, metadata["contract"], provenance)

    def transfer_to(self, env, config, new_contract):
        # Only the pool and data identity may change; scaling and execution stay fixed.
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
