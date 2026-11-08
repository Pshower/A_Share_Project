"""Explicit two-policy artifact whitelist for tools-only timing research."""

from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path
import zipfile

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.torch_layers import FlattenExtractor

from src.agents.timing_agent import TimingAgent, InferenceOnlyTimingPPO, _feature_schema, _algorithm_name
from src.agents.timing_policy import CalendarTimingPolicy, PHASE_PERIOD, TIMING_ACTION_NAMES, TIMING_NET_ARCH
from src.backtest.run import git_metadata
from src.data.preprocess import ROOT, file_hash, write_json
from src.runtime.device import resolve_device
from tools.archive.timing_phase_policy import DuePhaseActionHead, DuePhaseTimingPolicy, CONTEXT_DIMENSION, PHASE_ARCHITECTURES

POLICIES = {"linear": CalendarTimingPolicy, "due_phase_residual": DuePhaseTimingPolicy}
TOOL_FILES = ("tools/archive/timing_phase_policy.py", "tools/archive/timing_phase_agent.py")


def same_json(left, right):
    """Compare serializable records without Python's False == 0 coercion."""
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def code_identity():
    """Pin every source file actually present, so drift is a per-file hash mismatch.

    This used to reject any tree whose ``src/**/*.py`` count was not exactly 96, the
    2026-10-15 inventory of the phase-budget study. That constant is not a property
    this identity can enforce: adding an unrelated collector or web module changes the
    count without changing the timing code path, and it made the sealed study's unit
    tests unrunnable on any later checkout. Drift is enforced where it is measurable
    instead -- recorded hashes are compared with a fresh recomputation by
    ``validate_contract``, ``PhaseTimingAgent.load`` and the study runners, and each
    runner additionally requires its protocol's recorded ``source_hashes`` to equal
    this mapping, so a drifted checkout cannot be re-run under an old signature.
    """
    sources = {path.relative_to(ROOT).as_posix(): file_hash(path) for path in (ROOT / "src").rglob("*.py")}
    if not sources:
        raise ValueError("Timing phase research requires a non-empty source inventory")
    return dict(source_hashes=sources, tool_hashes={name: file_hash(ROOT / name) for name in TOOL_FILES})


def phase_policy_contract(base_contract, architecture):
    names = _feature_schema(base_contract)
    if architecture not in PHASE_ARCHITECTURES or base_contract["timing"]["calendar_margin"] != 2.:
        raise ValueError("Unregistered phase architecture or calendar margin")
    result = deepcopy(base_contract)
    residual = architecture == "due_phase_residual"
    result["phase_policy"] = dict(schema_version="timing_phase_v1", architecture=architecture,
        policy_class=_algorithm_name(POLICIES[architecture]), observation_dimension=len(names), phase_period=PHASE_PERIOD,
        context_offset=PHASE_PERIOD, context_dimension=CONTEXT_DIMENSION, due_phase=0, residual_action=1,
        residual_bias=False, initialization="zero_due_vector_preserving_baseline_v1" if residual else "calendar_baseline_v1",
        parameter_count=12914 if residual else 12805, optimizer_tensor_count=9 if residual else 8, **code_identity())
    return result


def validate_contract(contract):
    spec = contract.get("phase_policy")
    if not isinstance(spec, dict) or spec.get("architecture") not in POLICIES:
        raise ValueError("Missing registered phase policy contract")
    expected = phase_policy_contract(contract, spec["architecture"])
    if not same_json(spec, expected["phase_policy"]):
        raise ValueError("Phase policy schema, source inventory or tool hash mismatch")
    return _feature_schema(contract), spec


def expected_policy_record(names, spec):
    return dict(policy_class=spec["policy_class"], architecture=spec["architecture"],
        feature_names=list(names), observation_shape=[len(names)], observation_dtype="float32", action_names=list(TIMING_ACTION_NAMES),
        phase_period=PHASE_PERIOD, net_arch=deepcopy(TIMING_NET_ARCH), calendar_margin=2.,
        parameter_count=spec["parameter_count"], optimizer_tensor_count=spec["optimizer_tensor_count"],
        context_offset=PHASE_PERIOD, context_dimension=CONTEXT_DIMENSION, due_phase=0, residual_action=1,
        residual_bias=False, initialization=spec["initialization"])


def policy_record(model, names, spec):
    result = expected_policy_record(names, spec)
    result.update(policy_class=_algorithm_name(type(model.policy)), observation_shape=list(model.observation_space.shape),
        observation_dtype=str(model.observation_space.dtype), net_arch=deepcopy(model.policy.net_arch),
        calendar_margin=model.policy.calendar_margin, parameter_count=model.policy.parameter_count,
        optimizer_tensor_count=len(list(model.policy.parameters())))
    return result


class PhaseTimingAgent(TimingAgent):
    """Reuse macro-only inference, not the old agent's exact-class constructor."""

    def __init__(self, model, contract, provenance=None):
        self.model, self.contract = model, deepcopy(contract)
        self.provenance = deepcopy(provenance or {})
        self.feature_names, spec = validate_contract(self.contract)
        policy = model.policy
        dimension = len(self.feature_names)
        if (type(policy) is not POLICIES[spec["architecture"]]
                or not isinstance(model.action_space, gym.spaces.Discrete)
                or model.action_space.n != 2 or model.action_space.start != 0
                or not isinstance(model.observation_space, gym.spaces.Box)
                or model.observation_space.shape != (dimension,) or model.observation_space.dtype != np.dtype("float32")
                or not np.all(np.isneginf(model.observation_space.low)) or not np.all(np.isposinf(model.observation_space.high))
                or policy.net_arch != TIMING_NET_ARCH or policy.calendar_margin != 2.
                or not isinstance(policy.action_dist, CategoricalDistribution)
                or type(policy.features_extractor) is not FlattenExtractor or policy.features_dim != dimension
                or policy.activation_fn is not nn.Tanh or policy.optimizer_class is not torch.optim.Adam
                or policy.parameter_count != spec["parameter_count"] or not policy.ortho_init
                or policy.mlp_extractor.latent_dim_pi != dimension or policy.mlp_extractor.latent_dim_vf != 64):
            raise ValueError("Phase policy class, spaces, architecture or parameter contract mismatch")
        base = policy.action_net.base if type(policy) is DuePhaseTimingPolicy else policy.action_net
        if type(base) is not nn.Linear or (base.in_features, base.out_features) != (dimension, 2):
            raise ValueError("Phase policy lost its shared baseline linear Actor")
        if type(policy) is DuePhaseTimingPolicy:
            head = policy.action_net
            if (type(head) is not DuePhaseActionHead or type(head.residual) is not nn.Linear
                    or (head.residual.in_features, head.residual.out_features) != (CONTEXT_DIMENSION, 1)
                    or head.residual.bias is not None):
                raise ValueError("Phase policy residual definition mismatch")
        trainable = list(policy.parameters())
        optimized = [parameter for group in policy.optimizer.param_groups for parameter in group["params"]]
        if (len(trainable) != spec["optimizer_tensor_count"] or not all(parameter.requires_grad for parameter in trainable)
                or len(optimized) != len(trainable) or {id(p) for p in optimized} != {id(p) for p in trainable}):
            raise ValueError("Phase optimizer must register every trainable tensor exactly once")
        if (set(model.policy_kwargs) - {"calendar_margin", "net_arch"}
                or model.policy_kwargs.get("calendar_margin", 2.) != 2.
                or model.policy_kwargs.get("net_arch", TIMING_NET_ARCH) != TIMING_NET_ARCH):
            raise ValueError("Phase artifacts require only the registered safe policy constructor options")

    def save(self, directory, config, extra=None):
        if type(self.model) is not PPO:
            raise ValueError("Phase artifacts require the known standalone PPO algorithm")
        if any(not torch.isfinite(parameter).all() for parameter in self.model.policy.parameters()):
            raise FloatingPointError("Phase policy parameters must be finite")
        PhaseTimingAgent(self.model, self.contract, self.provenance)
        names, spec = validate_contract(self.contract)
        record = policy_record(self.model, names, spec)
        self.model.phase_registered_algorithm = _algorithm_name(PPO)
        self.model.phase_registered_contract = deepcopy(self.contract)
        self.model.phase_policy_record = deepcopy(record)
        self.model.phase_code_identity = code_identity()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.model.save(directory / "model.zip")
        metadata = dict(contract=self.contract, config=deepcopy(config), provenance=deepcopy(self.provenance),
            algorithm_class=_algorithm_name(PPO), num_timesteps=self.model.num_timesteps, policy_record=record,
            code_identity=code_identity(), feature_names=list(names), model_sha256=file_hash(directory / "model.zip"),
            git=git_metadata(), actual_device=str(self.model.device), extra=deepcopy(extra or {}),
            packages={name: version(name) for name in ("torch", "stable-baselines3", "gymnasium", "numpy")})
        write_json(directory / "metadata.json", metadata)

    @classmethod
    def load(cls, directory, expected_contract, device="cpu", *, allow_training=False):
        if type(allow_training) is not bool:
            raise ValueError("allow_training must be an explicit boolean")
        names, spec = validate_contract(expected_contract)
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if (not same_json(metadata.get("contract"), expected_contract) or metadata.get("algorithm_class") != _algorithm_name(PPO)
                or metadata.get("feature_names") != list(names) or metadata.get("code_identity") != code_identity()
                or metadata.get("model_sha256") != file_hash(directory / "model.zip")
                or not same_json(metadata.get("policy_record"), expected_policy_record(names, spec))):
            raise ValueError("Phase artifact hash, known algorithm, policy, source or contract mismatch")
        with zipfile.ZipFile(directory / "model.zip", "r") as zipped:
            if zipped.testzip() is not None:
                raise ValueError("Phase archive CRC mismatch")
            archived = json.loads(zipped.read("data"))
        if (archived.get("phase_registered_algorithm") != _algorithm_name(PPO)
                or not same_json(archived.get("phase_registered_contract"), expected_contract)
                or not same_json(archived.get("phase_policy_record"), metadata["policy_record"])
                or archived.get("phase_code_identity") != code_identity()):
            raise ValueError("Phase archive algorithm, policy, source or contract mismatch")
        loader = PPO if allow_training else InferenceOnlyTimingPPO
        model = loader.load(directory / "model.zip", device=resolve_device(device), custom_objects={
            "policy_class": POLICIES[spec["architecture"]],
            "policy_kwargs": dict(calendar_margin=2., net_arch=deepcopy(TIMING_NET_ARCH)),
            "observation_space": gym.spaces.Box(-np.inf, np.inf, (len(names),), np.float32),
            "action_space": gym.spaces.Discrete(2)})
        agent = cls(model, expected_contract, metadata.get("provenance"))
        actual = policy_record(model, names, spec)
        if actual != metadata["policy_record"] or model.num_timesteps != metadata["num_timesteps"]:
            raise ValueError("Loaded phase policy record or training clock mismatch")
        agent.provenance.update(loaded_from=str(directory.resolve()), loaded_model_sha256=metadata["model_sha256"],
                                inference_only=not allow_training)
        return agent
