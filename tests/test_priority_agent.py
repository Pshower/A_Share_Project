"""Synthetic SB3 adapter and strict artifact contracts; no financial experiments."""

from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.agents.delta_mapping import EXECUTION_FIELDS
from src.agents.priority_agent import InferenceOnlyPriorityPPO, PriorityAgent, priority_contract
from src.agents.priority_mapping import PriorityTradeActionWrapper
from src.agents.priority_policy import PriorityTradePolicy
from src.agents.trade_mapping import BUY, DIRECTION_NAMES, LOT_BOUND_FIELDS, SHARE_FIELDS, TRADE_ALLOCATION_FIELDS, TradeOrder
from src.envs.broker import BrokerConfig
from src.envs.trading import TradingEnv
from src.data.preprocess import write_json
from src.training.priority_aux import RankAuxPPO, build_rank_pool
from tests.test_trading_env import market


def model_and_contract(variant="coupled", bound=8., algorithm=PPO, algorithm_kwargs=None):
    torch.set_num_threads(1)
    env = PriorityTradeActionWrapper(TradingEnv(market(), initial_capital=200000,
        broker_config=BrokerConfig(min_commission=5, max_weight=.5)), top_n=1,
        actor_variant=variant, bound=bound)
    model = algorithm(PriorityTradePolicy, env, seed=42, device="cpu", n_steps=8, batch_size=4,
                policy_kwargs=dict(hidden_sizes=(16, 8), actor_variant=variant), **(algorithm_kwargs or {}))
    base = dict(action_mode="stock_trade", stock_codes=list(env.unwrapped.market.stock_codes),
        feature_cols=list(env.unwrapped.market.feature_cols), lookback=1, hidden_sizes=[16, 8], prior=None,
        broker=asdict(env.unwrapped.broker.config), logit_bound=bound,
        trade=dict(quantity_units="integer_lots_buy_reduce_full_book_shares_exit",
            share_state_units="shares_divided_by_lot_size_including_odd_fraction",
            conditional_size="BUY_always_REDUCE_varying_bounds_HOLD_EXIT_zero_atom",
            reduce_semantics="whole_lots_preserve_at_least_one_share",
            exit_semantics="known_full_available_actual_full_exit_only_waiver_and_slot_release",
            execution_fields=list(EXECUTION_FIELDS), allocation_fields=list(TRADE_ALLOCATION_FIELDS),
            share_fields=list(SHARE_FIELDS), lot_bound_fields=list(LOT_BOUND_FIELDS),
            direction_names=list(DIRECTION_NAMES), actor_output_gain=.1,
            max_trade_weight=.05, max_commission_ratio=.001))
    return model, priority_contract(base, variant), env


@pytest.mark.parametrize("variant", ["coupled", "decoupled"])
def test_agent_inspector_exact_raw_probability_value_and_typed_order(variant):
    model, contract, env = model_and_contract(variant)
    try:
        obs, _ = env.reset(seed=42)
        agent = PriorityAgent(model, contract)
        result = agent.inspect_action(obs)
        assert result["latent"].shape == (6,) and result["raw_packed_action"].shape == (6,)
        assert result["priority_mean"].shape == (2,)
        tensor, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            values, probability, entropy = model.policy.evaluate_actions(tensor,
                torch.tensor(result["raw_packed_action"]).float().unsqueeze(0))
        assert result["raw_log_prob"] == pytest.approx(probability.item())
        assert result["value"] == pytest.approx(values.item())
        assert torch.isfinite(entropy).all()
        assert isinstance(agent.act(obs), TradeOrder)
        assert result["priority_sigma"] == (0. if variant == "coupled" else pytest.approx(np.exp(-.7)))
        with pytest.raises(ValueError, match="migration"):
            agent.transfer_to(env, {}, contract)
    finally:
        env.close()


@pytest.mark.parametrize("variant", ["coupled", "decoupled"])
def test_artifact_round_trip_restores_actor_class_and_variant(tmp_path, variant):
    model, contract, env = model_and_contract(variant)
    try:
        agent = PriorityAgent(model, contract)
        obs, _ = env.reset(seed=42)
        before = agent.inspect_action(obs)
        folder = tmp_path / variant
        agent.save(folder, dict(actor_variant=variant))
        restored = PriorityAgent.load(folder, contract)
        assert type(restored.model.policy) is PriorityTradePolicy
        assert restored.model.policy.actor_variant == variant
        np.testing.assert_array_equal(before["latent"], restored.inspect_action(obs)["latent"])
        assert restored.model.num_timesteps == 0
        wrong = deepcopy(contract)
        wrong["trade"]["actor_variant"] = "decoupled" if variant == "coupled" else "coupled"
        with pytest.raises(ValueError, match="Artifact contract mismatch"):
            PriorityAgent.load(folder, wrong)
    finally:
        env.close()


@pytest.mark.parametrize("field", ["mode", "mapping", "layout", "semantics", "variant", "gain", "units", "commission", "fields"])
def test_agent_contract_rejects_semantic_or_actor_identity_drift(field):
    model, contract, env = model_and_contract()
    try:
        wrong = deepcopy(contract)
        if field == "mode":
            wrong["action_mode"] = "stock_trade"
        elif field == "mapping":
            wrong["mapping_version"] = "old"
        elif field == "variant":
            wrong["trade"]["actor_variant"] = "decoupled"
        elif field == "gain":
            wrong["trade"]["actor_output_gain"] = .01
        elif field == "commission":
            wrong["trade"]["max_commission_ratio"] = 0.
        elif field == "fields":
            wrong["trade"]["execution_fields"] = list(reversed(EXECUTION_FIELDS))
        else:
            key = {"layout": "action_layout", "semantics": "priority_semantics", "units": "quantity_units"}[field]
            wrong["trade"][key] = "wrong"
        with pytest.raises(ValueError, match="contract mismatch"):
            PriorityAgent(model, wrong)
    finally:
        env.close()


def test_initial_deterministic_functions_and_mappings_match_between_variants():
    coupled, c_contract, c_env = model_and_contract("coupled")
    decoupled, d_contract, d_env = model_and_contract("decoupled")
    try:
        for name, value in coupled.policy.state_dict().items():
            torch.testing.assert_close(value, decoupled.policy.state_dict()[name], rtol=0, atol=0)
        observation, _ = c_env.reset(seed=42)
        first = PriorityAgent(coupled, c_contract).inspect_action(observation)
        second = PriorityAgent(decoupled, d_contract).inspect_action(observation)
        for name in ["latent", "shares", "priority", "weights"]:
            np.testing.assert_array_equal(first[name], second[name])
        assert first["value"] == second["value"]
        assert first["raw_log_prob"] != second["raw_log_prob"] or not (first["directions"] == BUY).any()
    finally:
        c_env.close()
        d_env.close()


class Continue(BaseCallback):
    def _on_step(self):
        return True


@pytest.mark.parametrize("variant", ["coupled", "decoupled"])
def test_synthetic_rollout_keeps_unclipped_actions_and_exact_log_probabilities(variant):
    model, _, env = model_and_contract(variant, bound=.05)
    try:
        before = {name: value.clone() for name, value in model.policy.state_dict().items()}
        _, callback = model._setup_learn(8, Continue())
        callback.on_training_start({}, {})
        assert model.collect_rollouts(model.get_env(), callback, model.rollout_buffer, 8)
        assert (np.abs(model.rollout_buffer.actions[..., 2:]) > .05).any()
        with torch.no_grad():
            for batch in model.rollout_buffer.get(batch_size=4):
                _, log_probability, _ = model.policy.evaluate_actions(batch.observations, batch.actions)
                torch.testing.assert_close(log_probability, batch.old_log_prob, rtol=1e-5, atol=1e-6)
        for name, value in model.policy.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        assert not model.policy.optimizer.state
    finally:
        env.close()


def test_default_load_is_inference_only_and_explicit_generic_ppo_load_is_allowed(tmp_path):
    model, contract, env = model_and_contract()
    try:
        PriorityAgent(model, contract).save(tmp_path / "generic", {})
        default = PriorityAgent.load(tmp_path / "generic", contract)
        assert type(default.model) is InferenceOnlyPriorityPPO
        with pytest.raises(ValueError, match="Inference-only"):
            default.model.learn(8)
        with pytest.raises(ValueError, match="Inference-only"):
            default.model.train()
        assert default.model.num_timesteps == 0
        explicit = PriorityAgent.load(tmp_path / "generic", contract, algo_cls=PPO)
        assert type(explicit.model) is PPO
        assert not explicit.provenance["inference_only"]
        with pytest.raises(ValueError, match="algorithm class mismatch"):
            PriorityAgent.load(tmp_path / "generic", contract, algo_cls=RankAuxPPO)
    finally:
        env.close()


@pytest.mark.parametrize("field", ["actor_variant", "network_actor_variant", "distribution_actor_variant", "priority_log_std", "size_log_std"])
def test_loader_rejects_tampered_saved_policy_parameters(tmp_path, field):
    model, contract, env = model_and_contract()
    try:
        folder = tmp_path / "model"
        PriorityAgent(model, contract).save(folder, {})
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        metadata["priority_policy"][field] = ("decoupled" if "variant" in field else -.4)
        write_json(folder / "metadata.json", metadata)
        with pytest.raises(ValueError, match="parameters or actor variant mismatch"):
            PriorityAgent.load(folder, contract)
    finally:
        env.close()


def synthetic_auxiliary_pool():
    dates = pd.bdate_range("2023-01-02", periods=24)
    codes, features = ("000001", "600000"), ("signal",)
    panel = pd.DataFrame(np.arange(48).reshape(24, 2) / 100, index=dates,
        columns=pd.MultiIndex.from_product([codes, features]))
    prices = pd.DataFrame(index=dates, columns=pd.MultiIndex.from_product([codes, ("open", "close")]), dtype=float)
    for index, code in enumerate(codes):
        prices[(code, "open")] = 10 + np.arange(24) * (.1 + index * .02)
        prices[(code, "close")] = prices[(code, "open")] * (1 + index * .002)
    return build_rank_pool(panel, prices, codes, features, dates[19])


@pytest.mark.parametrize("method", ["ppo", "rank_aux"])
def test_registered_algorithm_roundtrip_preserves_method_and_guard_before_any_rollout(tmp_path, method):
    pool = synthetic_auxiliary_pool()
    coefficient = 0. if method == "ppo" else .05
    model, contract, env = model_and_contract(algorithm=RankAuxPPO,
        algorithm_kwargs=dict(aux_pool=pool if coefficient else None, aux_coef=coefficient,
                              aux_batch_size=16, aux_huber_delta=.1, aux_seed=42))
    spec = dict(actor_variant="coupled", training_method=method,
        auxiliary_coefficient=coefficient, auxiliary_batch_size=16, auxiliary_huber_delta=.1,
        pool_identity=dict(identity_sha256=pool.identity_sha256))
    contract["priority_rank"] = deepcopy(spec)
    config = dict(priority_rank=deepcopy(spec))
    try:
        folder = tmp_path / method
        PriorityAgent(model, contract).save(folder, config)
        with pytest.raises(ValueError, match="algorithm class mismatch"):
            PriorityAgent.load(folder, contract, algo_cls=PPO)
        inference = PriorityAgent.load(folder, contract)
        assert type(inference.model) is InferenceOnlyPriorityPPO
        with pytest.raises(ValueError, match="Inference-only"):
            inference.model.learn(8)
        restored = PriorityAgent.load(folder, contract, algo_cls=RankAuxPPO)
        assert type(restored.model) is RankAuxPPO
        assert restored.model.aux_coef == coefficient and restored.model.aux_pool is None
        obs, _ = env.reset(seed=42)
        restored.inspect_action(obs)
        if coefficient:
            with pytest.raises(ValueError, match="without its train-only label pool"):
                restored.model.learn(8)
            assert restored.model.num_timesteps == 0
            restored.model.attach_aux_pool(pool)
            assert restored.model.aux_pool_identity == pool.identity_sha256
            assert restored.model.aux_rng_state == model.aux_rng_state
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        metadata["config"]["priority_rank"]["training_method"] = "rank_aux" if method == "ppo" else "ppo"
        write_json(folder / "metadata.json", metadata)
        with pytest.raises(ValueError, match="contract/config mismatch"):
            PriorityAgent.load(folder, contract, algo_cls=RankAuxPPO)
    finally:
        env.close()


@pytest.mark.parametrize("field", ["network", "distribution"])
def test_agent_rejects_actual_internal_actor_variant_drift(field):
    model, contract, env = model_and_contract()
    try:
        component = model.policy.network if field == "network" else model.policy.action_dist
        component.actor_variant = "decoupled"
        with pytest.raises(ValueError, match="contract mismatch"):
            PriorityAgent(model, contract)
    finally:
        env.close()


def test_tampered_metadata_cannot_change_original_archive_algorithm(tmp_path):
    model, contract, env = model_and_contract()
    try:
        folder = tmp_path / "generic"
        PriorityAgent(model, contract).save(folder, {})
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        metadata["algorithm_class"] = "src.training.priority_aux.RankAuxPPO"
        write_json(folder / "metadata.json", metadata)
        with pytest.raises(ValueError, match="archive algorithm/method"):
            PriorityAgent.load(folder, contract, algo_cls=RankAuxPPO)
    finally:
        env.close()


def test_auxiliary_model_without_registered_method_cannot_be_saved(tmp_path):
    pool = synthetic_auxiliary_pool()
    model, contract, env = model_and_contract(algorithm=RankAuxPPO,
        algorithm_kwargs=dict(aux_pool=pool, aux_coef=.05))
    try:
        with pytest.raises(ValueError, match="registered method contract"):
            PriorityAgent(model, contract).save(tmp_path / "not_saved", {})
        assert not (tmp_path / "not_saved").exists()
    finally:
        env.close()
