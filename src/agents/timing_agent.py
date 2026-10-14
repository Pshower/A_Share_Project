"""Strict standalone PPO artifacts for a categorical timing controller."""

from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.torch_layers import FlattenExtractor

from src.backtest.run import git_metadata
from src.data.preprocess import ROOT, file_hash, write_json
from src.runtime.device import resolve_device
from .timing_policy import CalendarTimingPolicy, PHASE_PERIOD, TIMING_ACTION_NAMES, TIMING_NET_ARCH


class InferenceOnlyTimingPPO(PPO):
    """An ordinary PPO archive loaded for inference cannot implicitly resume."""

    def learn(self, *args, **kwargs):
        raise ValueError("Inference-only timing model; explicitly allow training when loading")

    def train(self):
        raise ValueError("Inference-only timing model cannot train")


def _algorithm_name(algorithm):
    return f"{algorithm.__module__}.{algorithm.__qualname__}"


def _feature_schema(contract):
    from src.training.timing_env import FEATURE_NAMES, TIMING_MAPPING_VERSION, build_timing_contract

    spec = contract.get("timing")
    margin = spec.get("calendar_margin") if isinstance(spec, dict) else None
    if (contract.get("action_mode") != "timing_control"
            or contract.get("mapping_version") != TIMING_MAPPING_VERSION
            or not isinstance(spec, dict)
            or spec.get("feature_names") != list(FEATURE_NAMES)
            or spec.get("phase_period") != PHASE_PERIOD
            or spec.get("action_names") != list(TIMING_ACTION_NAMES)
            or isinstance(margin, bool)
            or not isinstance(margin, (int, float, np.integer, np.floating))
            or not np.isfinite(margin) or margin <= 0):
        raise ValueError("Timing action, feature or data contract mismatch")
    canonical = build_timing_contract(contract, spec.get("reward_basis"), spec.get("calendar_margin"))
    if canonical["timing"] != spec:
        raise ValueError("Timing observation, decoder or benchmark contract mismatch")
    for key in ("stock_codes", "feature_cols", "broker", "price_basis", "artifacts"):
        if key not in contract:
            raise ValueError(f"Timing data contract is missing {key}")
    return tuple(FEATURE_NAMES)


def _policy_record(model, feature_names):
    policy = model.policy
    return dict(policy_class=f"{type(policy).__module__}.{type(policy).__qualname__}",
        observation_shape=list(model.observation_space.shape),
        observation_dtype=str(model.observation_space.dtype),
        feature_names=list(feature_names), action_names=list(TIMING_ACTION_NAMES),
        net_arch=deepcopy(policy.net_arch), calendar_margin=policy.calendar_margin,
        phase_period=PHASE_PERIOD, parameter_count=policy.parameter_count)


class TimingAgent:
    """Predict macro codes only; the environment owns all derived stock orders."""

    def __init__(self, model, contract, provenance=None):
        self.model, self.contract = model, deepcopy(contract)
        self.provenance = deepcopy(provenance or {})
        self.feature_names = _feature_schema(self.contract)
        dimension = len(self.feature_names)
        policy = model.policy
        if (type(policy) is not CalendarTimingPolicy
                or not isinstance(model.action_space, gym.spaces.Discrete)
                or model.action_space.n != 2 or model.action_space.start != 0
                or not isinstance(model.observation_space, gym.spaces.Box)
                or model.observation_space.shape != (dimension,)
                or model.observation_space.dtype != np.dtype("float32")
                or policy.net_arch != TIMING_NET_ARCH
                or policy.calendar_margin != self.contract["timing"]["calendar_margin"]
                or not isinstance(policy.action_dist, CategoricalDistribution)
                or type(policy.features_extractor) is not FlattenExtractor
                or policy.features_dim != dimension
                or policy.action_net.in_features != dimension
                or policy.action_net.out_features != 2
                or policy.mlp_extractor.latent_dim_pi != dimension
                or policy.mlp_extractor.latent_dim_vf != 64):
            raise ValueError("Timing policy spaces, initialization or feature contract mismatch")

    def reset(self):
        pass

    def validate_observation(self, observation):
        if (not isinstance(observation, np.ndarray)
                or not self.model.observation_space.contains(observation)
                or not np.isfinite(observation).all()):
            raise ValueError("Timing observation shape, dtype or bounds do not match the contract")
        phase = observation[:PHASE_PERIOD]
        if not np.isin(phase, [0, 1]).all() or phase.sum() != 1:
            raise ValueError("Timing observation phase must be an exact one-hot")

    def inspect_action(self, observation, deterministic=True):
        self.validate_observation(observation)
        self.model.policy.set_training_mode(False)
        tensor, _ = self.model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            action, value, log_probability = self.model.policy(tensor, deterministic=deterministic)
            probabilities = self.model.policy.action_dist.distribution.probs.detach().cpu().numpy()[0]
        code = int(action.cpu().item())
        if (not np.isfinite(probabilities).all() or not torch.isfinite(value).all()
                or not torch.isfinite(log_probability).all()):
            raise FloatingPointError("Timing inference must produce finite probabilities and value")
        return dict(macro_action=code, action_name=TIMING_ACTION_NAMES[code],
            action_probabilities=probabilities, raw_log_prob=float(log_probability.cpu().item()),
            value=float(value.cpu().item()), phase=int(np.argmax(observation[:PHASE_PERIOD])))

    def act(self, observation, step=None, deterministic=True):
        return self.inspect_action(observation, deterministic)["macro_action"]

    def save(self, directory, config, extra=None):
        if type(self.model) is not PPO:
            raise ValueError("Timing artifacts require the registered standalone PPO algorithm")
        if any(not torch.isfinite(parameter).all() for parameter in self.model.policy.parameters()):
            raise FloatingPointError("Timing parameters must be finite before saving")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        record = _policy_record(self.model, self.feature_names)
        self.model.timing_registered_algorithm = _algorithm_name(PPO)
        self.model.timing_registered_contract = deepcopy(self.contract)
        self.model.timing_feature_names = list(self.feature_names)
        self.model.timing_policy_record = deepcopy(record)
        self.model.save(directory / "model.zip")
        metadata = dict(contract=self.contract, config=deepcopy(config), provenance=self.provenance,
            num_timesteps=self.model.num_timesteps, git=git_metadata(),
            actual_device=str(self.model.device), cuda_build=torch.version.cuda,
            algorithm_class=_algorithm_name(PPO), feature_names=list(self.feature_names),
            policy_record=record, model_sha256=file_hash(directory / "model.zip"),
            packages={name: version(name) for name in ("torch", "stable-baselines3", "gymnasium", "numpy", "pandas", "pyarrow")},
            source_hashes={str(path.relative_to(ROOT)): file_hash(path) for path in (ROOT / "src").rglob("*.py")},
            extra=deepcopy(extra or {}))
        write_json(directory / "metadata.json", metadata)

    @classmethod
    def load(cls, directory, expected_contract, device="cpu", *, allow_training=False):
        if type(allow_training) is not bool:
            raise ValueError("allow_training must be an explicit boolean")
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("contract") != expected_contract:
            raise ValueError("Timing artifact contract mismatch; data or pool migration must be explicit")
        names = _feature_schema(expected_contract)
        if (metadata.get("algorithm_class") != _algorithm_name(PPO)
                or metadata.get("feature_names") != list(names)):
            raise ValueError("Timing artifact algorithm or feature schema mismatch")
        if file_hash(directory / "model.zip") != metadata.get("model_sha256"):
            raise ValueError("Timing model artifact hash mismatch")
        loader = PPO if allow_training else InferenceOnlyTimingPPO
        model = loader.load(directory / "model.zip", device=resolve_device(device))
        if (getattr(model, "timing_registered_algorithm", None) != _algorithm_name(PPO)
                or getattr(model, "timing_registered_contract", None) != expected_contract
                or getattr(model, "timing_feature_names", None) != list(names)):
            raise ValueError("Timing archive algorithm, data contract or feature schema mismatch")
        agent = cls(model, metadata["contract"], metadata.get("provenance"))
        actual = _policy_record(model, names)
        if metadata.get("policy_record") != actual or getattr(model, "timing_policy_record", None) != actual:
            raise ValueError("Timing saved policy class or initialization record mismatch")
        agent.provenance.update(loaded_from=str(directory.resolve()),
            loaded_model_sha256=metadata["model_sha256"], inference_only=not allow_training)
        return agent

    def transfer_to(self, *args, **kwargs):
        raise ValueError("Timing research artifacts require explicit data and pool migration")
