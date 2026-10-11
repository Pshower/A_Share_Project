"""Strict research artifact contract and inference for packed priority requests."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from src.data.preprocess import file_hash, write_json
from src.runtime.device import resolve_device
from .delta_mapping import EXECUTION_FIELDS
from .ppo_agent import PPOAgent
from .priority_mapping import (PRIORITY_ACTION_LAYOUT, PRIORITY_MAPPING_VERSION,
                               PRIORITY_SEMANTICS, map_priority_trade_action)
from .priority_policy import PriorityTradePolicy, StockPriorityTradeDistribution, validate_actor_variant
from .trade_mapping import DIRECTION_NAMES, LOT_BOUND_FIELDS, SHARE_FIELDS, TRADE_ALLOCATION_FIELDS, TradeOrder


def priority_contract(base_contract, actor_variant):
    """Register only the new action semantics; data, scaler and costs remain fixed."""
    validate_actor_variant(actor_variant)
    if base_contract.get("action_mode") != "stock_trade":
        raise ValueError("Priority conversion requires a stock_trade base contract")
    result = deepcopy(base_contract)
    result.update(action_mode="stock_priority", mapping_version=PRIORITY_MAPPING_VERSION)
    result["trade"].update(action_layout=PRIORITY_ACTION_LAYOUT,
        actor_variant=actor_variant, priority_semantics=PRIORITY_SEMANTICS)
    return result


class InferenceOnlyPriorityPPO(PPO):
    """Default artifact loading cannot silently resume under a different method."""

    def learn(self, *args, **kwargs):
        raise ValueError("Inference-only priority artifact; explicitly load the registered algorithm to train")

    def train(self):
        raise ValueError("Inference-only priority artifact cannot train")


def _algorithm_name(algorithm):
    return f"{algorithm.__module__}.{algorithm.__qualname__}"


def _policy_record(model):
    return dict(actor_variant=model.policy.actor_variant,
        network_actor_variant=model.policy.network.actor_variant,
        distribution_actor_variant=model.policy.action_dist.actor_variant,
        priority_log_std=float(model.policy.network.priority_log_std.detach().cpu().item()),
        size_log_std=float(model.policy.network.log_std.detach().cpu().item()))


def _validate_training_method(model, contract, config):
    spec = contract.get("priority_rank")
    if spec != config.get("priority_rank"):
        raise ValueError("Priority training method contract/config mismatch")
    if spec is None:
        if getattr(model, "aux_coef", 0.) > 0:
            raise ValueError("Priority auxiliary training requires a registered method contract")
        return
    method = spec.get("training_method")
    coefficient = 0. if method == "ppo" else .05
    if (method not in {"ppo", "rank_aux"}
            or spec.get("actor_variant") != model.policy.actor_variant
            or spec.get("auxiliary_coefficient") != coefficient
            or getattr(model, "aux_coef", None) != coefficient
            or getattr(model, "aux_batch_size", None) != spec.get("auxiliary_batch_size")
            or getattr(model, "aux_huber_delta", None) != spec.get("auxiliary_huber_delta")):
        raise ValueError("Priority saved algorithm training method mismatch")
    expected_identity = spec.get("pool_identity", {}).get("identity_sha256")
    actual_identity = getattr(model, "aux_pool_identity", None)
    if (method == "rank_aux" and (not isinstance(expected_identity, str)
                                  or len(expected_identity) != 64 or actual_identity != expected_identity)
            or actual_identity is not None and actual_identity != expected_identity):
        raise ValueError("Priority saved auxiliary pool identity mismatch")


class PriorityAgent(PPOAgent):
    def __init__(self, model, contract, provenance=None):
        self.model, self.contract = model, deepcopy(contract)
        self.provenance = deepcopy(provenance or {})
        n = len(contract["stock_codes"])
        trade = contract.get("trade", {})
        validate_actor_variant(trade.get("actor_variant"))
        semantics = dict(action_layout=PRIORITY_ACTION_LAYOUT, priority_semantics=PRIORITY_SEMANTICS,
            quantity_units="integer_lots_buy_reduce_full_book_shares_exit",
            share_state_units="shares_divided_by_lot_size_including_odd_fraction",
            conditional_size="BUY_always_REDUCE_varying_bounds_HOLD_EXIT_zero_atom",
            reduce_semantics="whole_lots_preserve_at_least_one_share",
            exit_semantics="known_full_available_actual_full_exit_only_waiver_and_slot_release")
        if (any(trade.get(key) != value for key, value in semantics.items())
                or trade.get("execution_fields") != list(EXECUTION_FIELDS)
                or trade.get("allocation_fields") != list(TRADE_ALLOCATION_FIELDS)
                or trade.get("share_fields") != list(SHARE_FIELDS)
                or trade.get("lot_bound_fields") != list(LOT_BOUND_FIELDS)
                or trade.get("direction_names") != list(DIRECTION_NAMES)
                or not np.isfinite([trade.get("max_trade_weight", float("nan")),
                                    trade.get("max_commission_ratio", float("nan"))]).all()
                or not 0 < trade.get("max_trade_weight", 0) <= 1
                or not contract["broker"]["commission"] <= trade.get("max_commission_ratio", 0) <= 1
                or trade.get("max_commission_ratio", 0) <= 0):
            raise ValueError("Priority trade units, conditional support or economic contract mismatch")
        expected = {"execution_features": (n, 3), "allocation_context": (7,), "share_state": (n, 2),
                    "trade_lot_bounds": (n, 4), "legal_directions": (n, 4)}
        if (contract.get("action_mode") != "stock_priority"
                or contract.get("mapping_version") != PRIORITY_MAPPING_VERSION
                or contract.get("prior") is not None or type(model.policy) is not PriorityTradePolicy
                or model.policy.actor_variant != trade["actor_variant"]
                or model.policy.network.actor_variant != trade["actor_variant"]
                or type(model.policy.action_dist) is not StockPriorityTradeDistribution
                or model.policy.action_dist.actor_variant != trade["actor_variant"]
                or model.action_space.shape != (3 * n,)
                or model.observation_space["stock_features"].shape != (n, contract["lookback"], len(contract["feature_cols"]))
                or any(model.observation_space[name].shape != shape for name, shape in expected.items())
                or not np.all(model.action_space.low[:n] == 0) or not np.all(model.action_space.high[:n] == 3)
                or not np.all(model.action_space.low[n:] == -contract["logit_bound"])
                or not np.all(model.action_space.high[n:] == contract["logit_bound"])
                or model.observation_space["share_state"].dtype != np.dtype("float64")
                or model.observation_space["trade_lot_bounds"].dtype != np.dtype("int64")
                or tuple(model.policy.hidden_sizes) != tuple(contract["hidden_sizes"])
                or model.policy.actor_output_gain != trade["actor_output_gain"]):
            raise ValueError("Priority policy spaces, initialization or execution contract mismatch")

    def save(self, directory, config, extra=None):
        from src.training.priority_aux import RankAuxPPO

        if type(self.model) not in {PPO, RankAuxPPO}:
            raise ValueError("Priority artifact requires an explicitly registered training algorithm")
        _validate_training_method(self.model, self.contract, config)
        if self.contract.get("priority_rank") is not None and type(self.model) is not RankAuxPPO:
            raise ValueError("Priority research training methods require RankAuxPPO")
        self.model.priority_registered_algorithm = _algorithm_name(type(self.model))
        self.model.priority_training_method_record = deepcopy(self.contract.get("priority_rank"))
        super().save(directory, config, extra)
        path = Path(directory) / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata.update(algorithm_class=_algorithm_name(type(self.model)),
                        priority_policy=_policy_record(self.model))
        write_json(path, metadata)

    @classmethod
    def load(cls, directory, expected_contract, device="cpu", *, algo_cls=None):
        from src.training.priority_aux import RankAuxPPO

        if algo_cls is not None and algo_cls not in {PPO, RankAuxPPO}:
            raise ValueError("Priority loader accepts only the registered PPO/RankAuxPPO algorithms")
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["contract"] != expected_contract:
            raise ValueError("Artifact contract mismatch; pool changes require explicit transfer")
        if file_hash(directory / "model.zip") != metadata["model_sha256"]:
            raise ValueError("Model artifact hash mismatch")
        declared = metadata.get("algorithm_class")
        allowed = {_algorithm_name(PPO), _algorithm_name(RankAuxPPO)}
        if declared not in allowed or (algo_cls is not None and declared != _algorithm_name(algo_cls)):
            raise ValueError("Priority artifact algorithm class mismatch")
        if (expected_contract.get("priority_rank") is not None
                and declared != _algorithm_name(RankAuxPPO)):
            raise ValueError("Priority research artifact cannot resume as ordinary PPO")
        loader = algo_cls or InferenceOnlyPriorityPPO
        model = loader.load(directory / "model.zip", device=resolve_device(device))
        if (getattr(model, "priority_registered_algorithm", None) != declared
                or getattr(model, "priority_training_method_record", None)
                != expected_contract.get("priority_rank")):
            raise ValueError("Priority archive algorithm/method declaration mismatch")
        if type(model.policy) is not PriorityTradePolicy:
            raise ValueError("Priority artifact policy class mismatch")
        if metadata.get("priority_policy") != _policy_record(model):
            raise ValueError("Priority saved policy parameters or actor variant mismatch")
        _validate_training_method(model, expected_contract, metadata["config"])
        provenance = deepcopy(metadata.get("provenance", {}))
        provenance.update(loaded_from=str(directory.resolve()),
            loaded_model_sha256=metadata["model_sha256"], inference_only=algo_cls is None,
            registered_algorithm_class=declared)
        return cls(model, metadata["contract"], provenance)

    def inspect_action(self, observation, deterministic=True):
        self.validate_observation(observation)
        self.model.policy.set_training_mode(False)
        tensor, _ = self.model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution, value = self.model.policy._distribution_value(tensor)
            packed = distribution.get_actions(deterministic)
            latent = packed.cpu().numpy()[0]
            log_probability = float(distribution.log_prob(packed).cpu().item())
        result = map_priority_trade_action(latent, observation, self.contract["logit_bound"],
            self.contract["broker"]["lot_size"], self.contract["stock_codes"],
            self.model.policy.actor_variant)
        priority_mean = (distribution.size_distribution.loc if self.model.policy.actor_variant == "coupled"
                         else distribution.priority_distribution.loc)
        result.update(direction_probabilities=distribution.direction_distribution.probs.detach().cpu().numpy()[0],
            size_mean=distribution.size_distribution.loc.detach().cpu().numpy()[0],
            size_sigma=float(self.model.policy.network.log_std.detach().exp().cpu().item()),
            priority_mean=priority_mean.detach().cpu().numpy()[0],
            priority_log_std=float(self.model.policy.network.priority_log_std.detach().cpu().item()),
            priority_sigma=(0. if self.model.policy.actor_variant == "coupled"
                            else float(self.model.policy.network.priority_log_std.detach().exp().cpu().item())),
            priority_support=("quantity_copy_atom" if self.model.policy.actor_variant == "coupled" else "independent_normal"),
            category_entropy=float(distribution.categorical_entropy().detach().cpu().item()),
            conditional_size_entropy=float(distribution.conditional_size_entropy().detach().cpu().item()),
            conditional_priority_entropy=float(distribution.conditional_priority_entropy().detach().cpu().item()),
            raw_log_prob=log_probability, value=float(value.cpu().item()))
        return result

    def act(self, observation, step=None, deterministic=True):
        result = self.inspect_action(observation, deterministic)
        return TradeOrder(result["shares"], result["freeze_mask"], result["exit_mask"],
                          result["priority"], result["top_n"], result)

    def transfer_to(self, env, config, new_contract):
        raise ValueError("Priority research artifacts require explicit validated pool migration")
