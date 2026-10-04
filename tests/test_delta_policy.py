"""CPU delta-policy checks never learn, backpropagate or step an optimizer."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from torch.distributions import Normal

from src.agents.delta_mapping import (
    DELTA_MAPPING_VERSION,
    DeltaActionWrapper,
    DeltaOrder,
    active_delta_coordinates,
    enrich_delta_observation,
    map_delta_action,
)
from src.agents.delta_policy import SharedDeltaNetwork, StateMaskedGaussian
from src.agents.ppo_agent import PPOAgent, make_model
from src.backtest.metrics import performance
from src.backtest.run import run_policy
from src.envs.broker import BrokerConfig, BrokerSimulator
from src.envs.portfolio import Portfolio
from src.envs.trading import TradingEnv
from src.training.env_factory import ResearchData, validate_config
from tests.test_ppo_framework import local_data, no_parameter_updates
from tests.test_trading_env import market, panels


def observation(weights, *, cash=None, available=None, frozen=None, buy=None, sell=None,
                valid=None, lot=.001, top_n=2, cap=.5):
    weights = np.asarray(weights, dtype=np.float32)
    n = len(weights)
    values = lambda value, default: np.full(n, default) if value is None else np.asarray(value)
    available_weights = weights if available is None else np.asarray(available)
    frozen_flags = values(frozen, 0)
    held_count = np.count_nonzero(weights > 0)
    locked_count = np.count_nonzero((weights > 0) & ((frozen_flags != 0) | (available_weights < weights - 1e-7)))
    return dict(
        stock_features=np.arange(n * 3, dtype=np.float32).reshape(n, 1, 3) / 10,
        portfolio=np.append(weights, 1 - weights.sum() if cash is None else cash).astype(np.float32),
        receivable_ratio=np.array([0.], dtype=np.float32),
        valid_mask=values(valid, 1).astype(np.int8),
        buy_mask=values(buy, 1).astype(np.int8),
        sell_mask=values(sell, 1).astype(np.int8),
        execution_features=np.column_stack([np.full(n, lot), available_weights, frozen_flags]).astype(np.float32),
        allocation_context=np.array([top_n / n, cap, 0., held_count / n,
                                     locked_count / n, max(0, top_n - held_count) / n], dtype=np.float32),
        top_n=np.array([top_n], dtype=np.float32),
    )


def tensors(obs):
    return {name: torch.as_tensor(value).unsqueeze(0) for name, value in obs.items()}


def delta_config(data):
    config = deepcopy(data.config)
    config.update(action_mode="stock_delta", prior_feature_index=None, action_top_k=None,
                  max_delta_weight=.05, actor_output_gain=.01, initial_log_std=-1.5,
                  top_n=2, top_n_choices=[1, 2], device="cpu")
    return config


def test_zero_mapping_keeps_positions_and_existing_cash_including_frozen_over_cap():
    obs = observation([.11, .23, .02], cash=.47, frozen=[0, 1, 0], top_n=1)
    result = map_delta_action(np.zeros(3), obs, .05, .05)
    np.testing.assert_array_equal(result["delta_weights"], np.zeros(3))
    np.testing.assert_array_equal(result["weights"], obs["portfolio"][:-1])
    assert result["cash_weight"] == float(obs["portfolio"][-1])
    assert result["cash_weight"] != pytest.approx(1 - result["weights"].sum())
    assert result["top_n_excess"] == 2


def test_zero_wrapper_really_holds_shares_when_prices_drift():
    features, prices = panels()
    prices.loc[prices.index[1], ("000001", "open")] = 12.
    prices.loc[prices.index[1], ("000001", "close")] = 14.
    env = DeltaActionWrapper(TradingEnv(market(features, prices), initial_capital=10000,
        broker_config=BrokerConfig(max_weight=.05, lot_size=100)), top_n=1)
    try:
        env.reset(seed=42)
        env.unwrapped.portfolio.fill(0, 200, 10., 0.)
        env._observation = env.enrich(env.unwrapped._observation())
        before = env.unwrapped.portfolio.positions.copy()
        cash = env.unwrapped.portfolio.cash
        _, _, _, _, info = env.step(np.zeros(2))
        np.testing.assert_array_equal(info["positions"], before)
        assert info["cash"] == cash
        assert info["equity"] == 10800.
        assert info["orders"] == [] and info["total_fees"] == 0
    finally:
        env.close()


def test_mapping_freezes_whole_stock_and_bounds_sales_by_partial_available_weight():
    obs = observation([.2, .2], available=[.03, .2], frozen=[0, 1])
    result = map_delta_action([-8, -8], obs, .5, .1)
    assert result["delta_weights"][0] == pytest.approx(-.03)
    assert result["weights"][0] == pytest.approx(.17)
    assert result["delta_weights"][1] == 0
    np.testing.assert_allclose(result["locked_weights"], [.17, 0.])
    assert active_delta_coordinates(obs).tolist() == [True, False]


def test_buy_blocked_sell_allowed_coordinate_can_hold_or_reduce_but_not_increase():
    obs = observation([.2], available=[.2], buy=[0], sell=[1], top_n=1)
    assert active_delta_coordinates(obs).tolist() == [True]
    assert map_delta_action([8], obs, .5, .05)["delta_weights"][0] == 0
    assert map_delta_action([0], obs, .5, .05)["weights"][0] == obs["portfolio"][0]
    sold = map_delta_action([-1], obs, .5, .05)
    assert -.05 < sold["delta_weights"][0] < 0
    assert sold["cash_weight"] > obs["portfolio"][-1]


def test_mapping_respects_new_name_slots_cash_budget_and_sub_lot_rejection():
    obs = observation([0., 0., 0.], cash=.015, lot=.005, top_n=2)
    mapped = map_delta_action([8, 7, 6], obs, .5, .05)
    assert np.count_nonzero(mapped["delta_weights"] > 0) == 2
    assert mapped["delta_weights"][2] == 0
    assert mapped["delta_weights"].sum() <= float(obs["portfolio"][-1]) + 1e-8
    assert mapped["cash_weight"] >= 0
    below_lot = observation([0., 0.], lot=.02, top_n=2)
    np.testing.assert_array_equal(map_delta_action([.1, .1], below_lot, .5, .05)["delta_weights"], [0., 0.])


def test_partial_lock_does_not_free_a_slot_and_failed_planned_exit_cannot_fund_new_name():
    partial = observation([.04, 0.], available=[.01, 0.], top_n=1, cap=.2)
    result = map_delta_action([-8, 8], partial, .2, .05)
    assert result["delta_weights"][0] == pytest.approx(-.01)
    assert result["delta_weights"][1] == 0
    full = observation([.04, 0.], available=[.04, 0.], top_n=1, cap=.2)
    planned = map_delta_action([-8, 8], full, .2, .05)
    assert planned["exit_mask"][0] and planned["delta_weights"][1] > 0
    portfolio = Portfolio(10000, ["000001", "600000"])
    portfolio.start_day("2024-01-02")
    portfolio.fill(0, 40, 10., 0.)
    portfolio.start_day("2024-01-03")
    broker = BrokerSimulator(BrokerConfig(max_weight=.2, lot_size=1))
    orders = broker.trade_deltas(portfolio, planned["delta_weights"], [10, 10], [10, 10],
        "2024-01-04", sell_mask=[0, 1], freeze_mask=planned["freeze_mask"],
        exit_mask=planned["exit_mask"], top_n=planned["top_n"])
    np.testing.assert_array_equal(portfolio.positions, [40., 0.])
    assert portfolio.cash == 9600.
    assert any("top_n_limit" in order["reasons"] for order in orders)
    assert all(order["filled_shares"] == 0 for order in orders)


@pytest.mark.parametrize("raw", [[0], [0, 0, 0], [np.nan, 0], [np.inf, 0]])
def test_mapping_rejects_wrong_shape_or_nonfinite_latents(raw):
    with pytest.raises(ValueError, match="finite N-stock"):
        map_delta_action(raw, observation([0., 0.]), .5, .05)


def test_state_masked_gaussian_matches_active_normal_density_and_omits_inactive():
    means = torch.tensor([[.2, 9., -.3]])
    log_std = torch.tensor([-.7, .4, -1.2])
    active = torch.tensor([[1, 0, 1]])
    distribution = StateMaskedGaussian(3, active).proba_distribution(means, log_std)
    actions = torch.tensor([[.5, 0., -.1]])
    expected = Normal(means[:, [0, 2]], log_std[[0, 2]].exp())
    torch.testing.assert_close(distribution.log_prob(actions), expected.log_prob(actions[:, [0, 2]]).sum(1))
    torch.testing.assert_close(distribution.entropy(), expected.entropy().sum(1))
    assert distribution.mode()[0, 1] == distribution.sample()[0, 1] == 0
    illegal = actions.clone()
    illegal[0, 1] = .001
    assert torch.isneginf(distribution.log_prob(illegal)).all()


def test_action_dependent_topn_does_not_remove_active_density_coordinates():
    obs = observation([0., 0., 0.], top_n=1)
    mapped = map_delta_action([8, 7, 6], obs, .5, .05)
    assert np.count_nonzero(mapped["delta_weights"]) == 1
    active = torch.as_tensor(active_delta_coordinates(obs)).unsqueeze(0)
    assert active.all()
    means, actions = torch.zeros((1, 3)), torch.tensor([[8., 7., 6.]])
    distribution = StateMaskedGaussian(3, active).proba_distribution(means, torch.zeros(3))
    expected = Normal(means, torch.ones_like(means)).log_prob(actions).sum(1)
    torch.testing.assert_close(distribution.log_prob(actions), expected)


def test_delta_network_has_stock_only_small_nonzero_actor_and_uses_topn_condition():
    torch.manual_seed(42)
    network = SharedDeltaNetwork(1, 3, (16, 8), actor_output_gain=.01)
    obs = tensors(observation([.2, .1, 0.], top_n=1))
    with torch.no_grad():
        means, values, log_std = network(obs)
        changed = {key: value.clone() for key, value in obs.items()}
        changed["top_n"][:] = 2
        changed["allocation_context"][:, 0] = 2 / 3
        other = network(changed)[0]
    assert means.shape == (1, 3) and values.shape == (1, 1) and log_std.shape == (3,)
    assert not hasattr(network, "cash_head")
    assert torch.linalg.vector_norm(network.stock_head[-1].weight).item() == pytest.approx(.01)
    assert torch.count_nonzero(means) > 0 and means.abs().max() < .04
    assert not torch.equal(means, other)


def test_delta_network_is_stock_permutation_equivariant_and_no_valid_states_are_finite():
    torch.manual_seed(123)
    network = SharedDeltaNetwork(1, 3, (16, 8))
    obs = tensors(observation([.2, .1, 0.], top_n=2))
    order = [2, 0, 1]
    moved = {name: value.clone() for name, value in obs.items()}
    for name in ["stock_features", "valid_mask", "buy_mask", "sell_mask", "execution_features"]:
        moved[name] = obs[name][:, order]
    moved["portfolio"] = torch.cat([obs["portfolio"][:, order], obs["portfolio"][:, -1:]], 1)
    with torch.no_grad():
        means, value, _ = network(obs)
        changed, changed_value, _ = network(moved)
        torch.testing.assert_close(changed, means[:, order], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(value, changed_value, rtol=1e-5, atol=1e-6)
        invalid = deepcopy(obs)
        invalid["valid_mask"].zero_()
        assert all(torch.isfinite(part).all() for part in network(invalid))
    empty = StateMaskedGaussian(3, torch.zeros((1, 3))).proba_distribution(torch.ones((1, 3)), torch.zeros(3))
    torch.testing.assert_close(empty.mode(), torch.zeros((1, 3)))
    torch.testing.assert_close(empty.log_prob(empty.sample()), torch.zeros(1))
    torch.testing.assert_close(empty.entropy(), torch.zeros(1))


def test_delta_cpu_factory_prediction_execution_and_artifact_roundtrip(local_data, tmp_path):
    config = delta_config(local_data)
    data = ResearchData(config)
    vector = data.vector_env()
    raw_env = data.env("train")
    try:
        agent = PPOAgent(make_model(vector, config), data.contract)
        # A controlled positive output exercises real fills without optimization.
        with torch.no_grad():
            agent.model.policy.network.stock_head[-1].bias.fill_(1.)
        observation_value, _ = raw_env.reset(seed=42)
        assert observation_value["allocation_context"].shape == (6,)
        assert observation_value["execution_features"].shape == (len(data.codes), 3)
        assert agent.model.observation_space["allocation_context"].shape == (6,)
        before = {name: value.detach().clone() for name, value in agent.model.policy.state_dict().items()}
        assert data.contract["action_mode"] == "stock_delta"
        assert data.contract["mapping_version"] == DELTA_MAPPING_VERSION
        assert agent.model.action_space.shape == (len(data.codes),)
        latent, _ = agent.model.predict(observation_value, deterministic=True)
        inspected = agent.inspect_action(observation_value)
        np.testing.assert_array_equal(latent, inspected["latent"])
        tensor, _ = agent.model.policy.obs_to_tensor(observation_value)
        with torch.no_grad():
            density = agent.model.policy.get_distribution(tensor)
            assert density.mode().shape == (1, len(data.codes))
            assert torch.isfinite(density.log_prob(density.mode())).all()
        order = agent.act(observation_value)
        assert isinstance(order, DeltaOrder)
        np.testing.assert_array_equal(order.deltas, inspected["delta_weights"])
        after, _, _, _, info = raw_env.step_order(order)
        agent.validate_observation(after)
        np.testing.assert_array_equal(info["action_diagnostics"]["delta_weights"], inspected["delta_weights"])
        agent.save(tmp_path / "delta", config)
        loaded = PPOAgent.load(tmp_path / "delta", data.contract, "cpu")
        np.testing.assert_array_equal(agent.inspect_action(observation_value)["latent"], loaded.inspect_action(observation_value)["latent"])
        np.testing.assert_array_equal(agent.act(observation_value).deltas, loaded.act(observation_value).deltas)
        assert loaded.model.policy.actor_output_gain == .01
        assert loaded.model.num_timesteps == agent.model.num_timesteps == 0
        assert not agent.model.policy.optimizer.state and not loaded.model.policy.optimizer.state
        original = run_policy(data.market("val"), agent, data.settings)
        restored = run_policy(data.market("val"), loaded, data.settings)
        for first, second in zip(original[:3], restored[:3]):
            pd.testing.assert_frame_equal(first, second, check_exact=True)
        daily, holdings, orders = original[:3]
        assert len(daily) == len(holdings) == 31
        assert str(daily.date.iloc[0].date()) == data.manifest["train_end"]
        assert str(daily.date.iloc[-1].date()) == data.manifest["val_end"]
        assert (daily.cash >= 0).all() and (holdings >= 0).all().all()
        assert (orders.filled_shares > 0).any()
        assert performance(daily)["trading_steps"] == 30
        for name, value in before.items():
            torch.testing.assert_close(value, agent.model.policy.state_dict()[name], rtol=0, atol=0)
    finally:
        raw_env.close()
        vector.close()


def test_delta_factory_fixed_inference_k_and_independent_episode_choice_rng(local_data):
    baseline_config = deepcopy(local_data.config)
    baseline_config.update(initial_portfolio="mixed", training_episode_steps=8, recent_episode_every=4)
    baseline_data = ResearchData(baseline_config)
    data = ResearchData(delta_config(baseline_data))
    left, right = data.env("train"), data.env("train")
    baseline = baseline_data.env("train")
    try:
        selected = []
        for episode in range(12):
            seed = 42 if episode == 0 else None
            first, first_info = left.reset(seed=seed)
            second, _ = right.reset(seed=seed)
            _, old_info = baseline.reset(seed=seed)
            assert first["top_n"][0] == second["top_n"][0]
            assert first_info["date"] == old_info["date"]
            assert first_info["cash"] == old_info["cash"]
            np.testing.assert_array_equal(first_info["positions"], old_info["positions"])
            assert left.unwrapped.np_random.bit_generator.state == baseline.unwrapped.np_random.bit_generator.state
            count = np.count_nonzero(first_info["positions"] > 0)
            assert first["allocation_context"].shape == (6,)
            assert first["allocation_context"][3] == pytest.approx(count / len(data.codes))
            assert first["allocation_context"][5] == pytest.approx(max(0, first["top_n"][0] - count) / len(data.codes))
            selected.append(int(first["top_n"][0]))
        assert set(selected) == {1, 2}
        for top in [10, 20]:
            data.settings["delta_top_n"] = top
            fixed = data.env("val")
            try:
                obs, _ = fixed.reset(seed=42)
                assert obs["top_n"][0] == top
                assert obs["allocation_context"][0] == pytest.approx(top / len(data.codes))
                assert not fixed.top_n_choices
            finally:
                fixed.close()
    finally:
        left.close()
        right.close()
        baseline.close()


def test_wrapper_context_counts_partial_locks_and_projects_only_known_next_day_release():
    config = BrokerConfig(max_weight=.5, min_commission=5., lot_size=100)
    env = DeltaActionWrapper(TradingEnv(market(), initial_capital=10000, broker_config=config), top_n=1)
    try:
        env.reset(seed=42)
        portfolio = env.unwrapped.portfolio
        portfolio.fill(0, 100, 10., 0.)
        portfolio.fill(1, 100, 10., 0.)
        portfolio.apply_corporate_action(0, 0., 2., "2024-01-02", "2024-01-08")
        original = env.unwrapped._observation()
        projected = env.enrich(original)
        current = enrich_delta_observation(original, portfolio, [10., 10.], config, 1)
        np.testing.assert_allclose(projected["allocation_context"], [.5, .5, 5 / 11000, 1., .5, 0.])
        assert current["allocation_context"][4] == 1.
        np.testing.assert_allclose(projected["execution_features"][:, 1], [1000 / 11000, 1000 / 11000])
        np.testing.assert_array_equal(projected["execution_features"][:, 2], [0., 0.])
        np.testing.assert_array_equal(current["execution_features"][:, 2], [1., 1.])
        assert projected["allocation_context"].shape == (6,)
    finally:
        env.close()


def test_delta_contract_rejects_legacy_transfer_and_wrong_actor_gain(local_data):
    config = delta_config(local_data)
    data = ResearchData(config)
    delta_env, legacy_env = data.vector_env(), local_data.vector_env()
    try:
        delta = PPOAgent(make_model(delta_env, config), data.contract)
        legacy = PPOAgent(make_model(legacy_env, local_data.config), local_data.contract)
        with pytest.raises(ValueError, match="action_mode"):
            delta.transfer_to(legacy_env, local_data.config, local_data.contract)
        with pytest.raises(ValueError, match="action_mode"):
            legacy.transfer_to(delta_env, config, data.contract)
        wrong = deepcopy(data.contract)
        wrong.pop("action_mode")
        with pytest.raises(ValueError, match="spaces|contract"):
            PPOAgent(delta.model, wrong)
        wrong = deepcopy(data.contract)
        wrong["delta"]["actor_output_gain"] = .02
        with pytest.raises(ValueError, match="initialization"):
            PPOAgent(delta.model, wrong)
    finally:
        delta_env.close()
        legacy_env.close()


@pytest.mark.parametrize("patch", [{"prior_feature_index": 14}, {"action_top_k": 20},
    {"actor_output_gain": 0.}, {"max_delta_weight": float("nan")}, {"top_n_choices": [1, 1]}])
def test_delta_config_rejects_incompatible_or_invalid_options(local_data, patch):
    config = delta_config(local_data)
    config.update(patch)
    with pytest.raises(ValueError):
        validate_config(config)
