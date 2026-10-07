"""Synthetic forward/gradient attribution checks; no PPO or optimizer updates."""

from copy import deepcopy
import io
import math

import pytest
import torch
from stable_baselines3 import PPO

from src.agents.ppo_agent import PPOAgent
from src.agents.attribution_policy import AttributionTradeNetwork, AttributionTradePolicy
from src.agents.exploration_policy import SplitTradeNetwork
from src.agents.trade_mapping import BUY, HOLD, REDUCE
from src.agents.trade_policy import SharedTradePolicy
from src.training.env_factory import ResearchData
from tests.test_exploration_policy import observation, spaces
from tests.test_ppo_framework import local_data, no_parameter_updates
from tests.test_trade_policy import trade_config


pytestmark = pytest.mark.usefixtures("no_parameter_updates")
ROUTES = ("pure", "fused")
CRITICS = ("shared", "detached")


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def policy(obs, market_context="pure", critic_features="shared"):
    return AttributionTradePolicy(*spaces(obs), lambda _: 1e-5,
        market_context=market_context, critic_features=critic_features)


def serialized_state(module):
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return buffer.getvalue()


def initialized(route, critic, **kwargs):
    torch.manual_seed(42)
    return AttributionTradeNetwork(1, 20, market_context=route, critic_features=critic, **kwargs)


def test_four_cells_have_exactly_identical_parameter_names_bytes_and_count():
    networks = [initialized(route, critic) for route in ROUTES for critic in CRITICS]
    reference = networks[0]
    for network in networks:
        assert network.parameter_count == 125063
        assert list(network.state_dict()) == list(reference.state_dict())
        assert serialized_state(network) == serialized_state(reference)
        for name, tensor in network.state_dict().items():
            torch.testing.assert_close(tensor, reference.state_dict()[name], rtol=0, atol=0)
    torch.manual_seed(42)
    existing_split = SplitTradeNetwork(1, 20)
    assert serialized_state(existing_split) == serialized_state(reference)
    for actual, expected in zip(reference(observation()), existing_split(observation())):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_policy_four_cells_have_same_initial_tensors_without_extra_parameters():
    obs = observation()
    instances = []
    for route in ROUTES:
        for critic in CRITICS:
            torch.manual_seed(42)
            instances.append(policy(obs, route, critic))
    assert issubclass(AttributionTradePolicy, SharedTradePolicy)
    for instance in instances:
        assert instance.parameter_count == 125063
        assert serialized_state(instance) == serialized_state(instances[0])
        assert instance.architecture == "split"
        assert instance.actor_output_gain == .1 and instance.initial_log_std == -.7


@pytest.mark.parametrize("route", ROUTES)
def test_critic_axis_does_not_change_initial_forward_or_distribution(route):
    obs = observation()
    networks = [initialized(route, critic) for critic in CRITICS]
    for actual, expected in zip(networks[0](obs), networks[1](obs)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.manual_seed(42)
    shared = policy(obs, route, "shared")
    torch.manual_seed(42)
    detached = policy(obs, route, "detached")
    with torch.no_grad():
        for actual, expected in zip(shared(obs, deterministic=True), detached(obs, deterministic=True)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_market_route_changes_only_market_part_of_context_not_individual_or_held_encoding():
    obs = observation()
    pure, fused = initialized("pure", "shared"), initialized("fused", "shared")
    first, second = pure.encode_components(obs), fused.encode_components(obs)
    for name in ("market_encoded", "account_encoded", "encoded", "held"):
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
    assert not torch.equal(first["market"], second["market"])
    torch.testing.assert_close(first["context"][:, 128:], second["context"][:, 128:], rtol=0, atol=0)
    torch.testing.assert_close(second["market"], second["encoded"].mean(1))
    assert not torch.equal(pure(obs)[0], fused(obs)[0])
    assert not torch.equal(pure(obs)[2], fused(obs)[2])


@pytest.mark.parametrize("route", ROUTES)
def test_fixed_valid_market_summary_is_account_invariant_only_in_pure_route(route):
    obs = observation()
    changed = deepcopy(obs)
    changed["portfolio"][:, 0] += .03
    changed["portfolio"][:, 1] -= .03
    changed["buy_mask"][:, 0] = 0
    changed["execution_features"][:, 0, 1] = .02
    changed["trade_lot_bounds"][:, 0, 1] = 30
    network = initialized(route, "shared")
    first, second = network.encode_components(obs), network.encode_components(changed)
    torch.testing.assert_close(obs["valid_mask"], changed["valid_mask"], rtol=0, atol=0)
    torch.testing.assert_close(first["market_encoded"], second["market_encoded"], rtol=0, atol=0)
    assert not torch.equal(first["held"], second["held"])
    if route == "pure":
        torch.testing.assert_close(first["market"], second["market"], rtol=0, atol=0)
    else:
        assert not torch.equal(first["market"], second["market"])
    assert not torch.equal(network(obs)[0], network(changed)[0])


@pytest.mark.parametrize("route", ROUTES)
def test_held_invalid_stock_is_preserved_but_never_enters_valid_market_mean(route):
    obs = observation()
    obs["valid_mask"][:, -1] = 0
    obs["stock_features"][:, -1] = 0
    obs["execution_features"][:, -1, 2] = 1
    obs["legal_directions"][:, -1] = torch.tensor([1, 0, 0, 0], dtype=torch.int8)
    network = initialized(route, "shared")
    components = network.encode_components(obs)
    encoded_name = "market_encoded" if route == "pure" else "encoded"
    torch.testing.assert_close(components["market"], components[encoded_name][:, :-1].mean(1))
    torch.testing.assert_close(components["held"],
        (components["encoded"] * obs["portfolio"][:, :-1].unsqueeze(-1)).sum(1))
    excluding_invalid = (components["encoded"][:, :-1]
        * obs["portfolio"][:, :-2].unsqueeze(-1)).sum(1)
    assert not torch.equal(components["held"], excluding_invalid)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("critic", CRITICS)
def test_permutation_equivariance_and_stock_count_independence(route, critic):
    obs = observation()
    network = initialized(route, critic)
    before = network(obs)
    permutation = [3, 1, 0, 2]
    other = deepcopy(obs)
    for key in ("stock_features", "valid_mask", "buy_mask", "sell_mask", "execution_features",
                "share_state", "trade_lot_bounds", "legal_directions"):
        other[key] = obs[key][:, permutation]
    other["portfolio"] = obs["portfolio"][:, permutation + [4]]
    after = network(other)
    torch.testing.assert_close(after[0], before[0][:, permutation], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(after[1], before[1][:, permutation], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(after[2], before[2], atol=1e-6, rtol=1e-5)
    for n in (1, 7):
        result = network(observation(n=n))
        assert result[0].shape == (1, n, 4) and result[1].shape == (1, n)
        assert result[2].shape == (1, 1)
        assert all(torch.isfinite(value).all() for value in result)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("critic", CRITICS)
def test_no_valid_stocks_produces_finite_hold_only_policy(route, critic):
    obs = observation()
    obs["valid_mask"][:] = 0
    obs["execution_features"][..., 2] = 1
    obs["legal_directions"][:] = torch.tensor([1, 0, 0, 0], dtype=torch.int8)
    instance = policy(obs, route, critic)
    with torch.no_grad():
        components = instance.network.encode_components(obs)
        assert torch.count_nonzero(components["market"]) == 0
        assert torch.count_nonzero(components["held"]) != 0
        action, value, log_prob = instance(obs)
        assert (action == 0).all() and (log_prob == 0).all()
        assert torch.isfinite(value).all()
        assert (instance.get_distribution(obs).entropy() == 0).all()


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("critic", CRITICS)
def test_forward_evaluate_likelihood_and_constructor_save_load(route, critic, tmp_path):
    obs = observation()
    obs["legal_directions"][:, 1] = torch.tensor([1, 0, 1, 1], dtype=torch.int8)
    obs["trade_lot_bounds"][:, 1] = torch.tensor([0, 0, 2, 2])
    instance = policy(obs, route, critic)
    before = serialized_state(instance)
    with torch.no_grad():
        actions, value, original = instance(obs)
        new_value, actual, entropy = instance.evaluate_actions(obs, actions)
        torch.testing.assert_close(value, new_value, rtol=0, atol=0)
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
        assert torch.isfinite(entropy).all()
        raw = torch.tensor([[BUY, REDUCE, HOLD, HOLD, 12., 0., 0., 0.]])
        expected = instance.get_distribution(obs).log_prob(raw)
        torch.testing.assert_close(instance.evaluate_actions(obs, raw)[1], expected, rtol=0, atol=0)
        assert torch.isfinite(expected).all()
        raw[:, 5] = .1
        assert torch.isneginf(instance.evaluate_actions(obs, raw)[1]).all()
        original_forward = instance(obs, deterministic=True)
    destination = tmp_path / "attribution.pt"
    instance.save(destination)
    restored = AttributionTradePolicy.load(destination, device="cpu")
    assert restored.market_context == route and restored.critic_features == critic
    assert restored.architecture == "split"
    assert restored.network.market_context == route and restored.network.critic_features == critic
    assert restored.parameter_count == 125063
    assert restored.actor_output_gain == .1 and restored.initial_log_std == -.7
    assert serialized_state(instance) == before
    assert serialized_state(restored) == before
    with torch.no_grad():
        for actual, expected in zip(restored(obs, deterministic=True), original_forward):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("critic", CRITICS)
def test_ppo_agent_accepts_subclass_and_preserves_flags_in_synthetic_artifact(route, critic, local_data, tmp_path):
    config = trade_config(local_data)
    config["actor_output_gain"] = .1
    data = ResearchData(config)
    env = data.vector_env()
    try:
        model = PPO(AttributionTradePolicy, env, device="cpu", seed=42, verbose=0,
            policy_kwargs=dict(hidden_sizes=tuple(config["hidden_sizes"]), actor_output_gain=.1,
                initial_log_std=-.7, market_context=route, critic_features=critic), **config["ppo"])
        agent = PPOAgent(model, data.contract)
        obs = {name: value[0] for name, value in env.reset().items()}
        before = serialized_state(model.policy)
        inspected = agent.inspect_action(obs)
        destination = tmp_path / "synthetic_attribution"
        agent.save(destination, config)
        restored = PPOAgent.load(destination, data.contract, device="cpu")
        assert restored.model.policy.market_context == route
        assert restored.model.policy.critic_features == critic
        assert restored.model.policy.architecture == "split"
        assert serialized_state(model.policy) == before
        assert serialized_state(restored.model.policy) == before
        actual = restored.inspect_action(obs)
        torch.testing.assert_close(torch.as_tensor(actual["latent"]), torch.as_tensor(inspected["latent"]),
                                   rtol=0, atol=0)
        assert model.num_timesteps == restored.model.num_timesteps == 0
    finally:
        env.close()


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("critic", CRITICS)
def test_value_gradient_detaches_encoders_only_when_requested(route, critic):
    network = initialized(route, critic)
    before = serialized_state(network)
    named = dict(network.named_parameters())
    loss = network(observation())[2].square().mean()
    gradients = torch.autograd.grad(loss, list(named.values()), allow_unused=True)
    gradients = dict(zip(named, gradients))
    for prefix in ("market_encoder.", "account_encoder."):
        selected = [gradient for name, gradient in gradients.items() if name.startswith(prefix)]
        if critic == "detached":
            assert all(gradient is None for gradient in selected)
        else:
            assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in selected)
    value_gradients = [gradient for name, gradient in gradients.items() if name.startswith("value_head.")]
    assert all(gradient is not None for gradient in value_gradients)
    assert any(torch.count_nonzero(gradient) for gradient in value_gradients)
    assert all(gradient is None for name, gradient in gradients.items()
               if name.startswith(("direction_head.", "size_head.", "stock_body.")))
    assert all(parameter.grad is None for parameter in network.parameters())
    assert serialized_state(network) == before


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("critic", CRITICS)
def test_actor_likelihood_gradient_reaches_both_encoders_even_with_detached_critic(route, critic):
    instance = policy(observation(), route, critic)
    before = serialized_state(instance)
    action = torch.tensor([[BUY, HOLD, REDUCE, HOLD, .4, 0., -.3, 0.]])
    named = dict(instance.network.named_parameters())
    loss = -instance.evaluate_actions(observation(), action)[1].mean()
    gradients = dict(zip(named, torch.autograd.grad(loss, list(named.values()), allow_unused=True)))
    for prefix in ("market_encoder.", "account_encoder.", "direction_head.", "size_head.", "stock_body."):
        selected = [gradient for name, gradient in gradients.items() if name.startswith(prefix)]
        assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in selected)
    assert all(gradient is None for name, gradient in gradients.items() if name.startswith("value_head."))
    assert all(parameter.grad is None for parameter in instance.parameters())
    assert serialized_state(instance) == before


@pytest.mark.parametrize("flags", [dict(market_context="other"), dict(critic_features="other"),
                                  dict(market_context=None), dict(critic_features=None)])
def test_invalid_attribution_flags_are_rejected(flags):
    with pytest.raises(ValueError):
        AttributionTradeNetwork(1, 20, **flags)
    with pytest.raises(ValueError):
        AttributionTradePolicy(*spaces(observation()), lambda _: 1e-5, **flags)


def test_non_split_architecture_is_rejected():
    with pytest.raises(ValueError, match="architecture='split'"):
        AttributionTradePolicy(*spaces(observation()), lambda _: 1e-5, architecture="baseline")


def test_actor_output_gain_and_std_follow_existing_split_initialization():
    network = initialized("pure", "shared")
    for head in (network.direction_head, network.size_head):
        torch.testing.assert_close(head.weight @ head.weight.T,
            torch.eye(head.out_features) * .1 ** 2, atol=1e-8, rtol=1e-5)
        assert torch.count_nonzero(head.bias) == 0
    assert network.log_std.exp().item() == pytest.approx(math.exp(-.7))
