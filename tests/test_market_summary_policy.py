"""Synthetic pooling controls: no data downloads or optimizer updates."""

from copy import deepcopy
import io

import pytest
import torch
from stable_baselines3 import PPO

from src.agents.exploration_policy import SplitTradeNetwork
from src.agents.market_summary_policy import (
    MarketSummaryNetwork, MarketSummaryPolicy, POOLING_MODES,
)
from src.agents.ppo_agent import PPOAgent
from src.agents.trade_mapping import BUY, HOLD, REDUCE
from src.training.env_factory import ResearchData
from tests.test_exploration_policy import observation, spaces
from tests.test_ppo_framework import local_data, no_parameter_updates
from tests.test_trade_policy import trade_config


pytestmark = pytest.mark.usefixtures("no_parameter_updates")


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def serialized_state(module):
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return buffer.getvalue()


def initialized(mode, **kwargs):
    torch.manual_seed(42)
    return MarketSummaryNetwork(1, 20, pooling=mode, **kwargs)


def policy(obs, mode):
    return MarketSummaryPolicy(*spaces(obs), lambda _: 1e-5, pooling=mode)


def activate_summary(network):
    with torch.no_grad():
        network.summary_query.copy_(torch.linspace(-2., 2., network.summary_query.numel()))
        network.variance_projection.weight.copy_(
            torch.eye(network.summary_query.numel()) * .7)


def test_all_arms_have_identical_parameter_bytes_and_expected_count():
    networks = [initialized(mode) for mode in POOLING_MODES]
    for network in networks:
        assert network.parameter_count == 141575
        assert list(network.state_dict()) == list(networks[0].state_dict())
        assert serialized_state(network) == serialized_state(networks[0])
        assert torch.count_nonzero(network.summary_query) == 0
        assert torch.count_nonzero(network.variance_projection.weight) == 0
        assert network.variance_projection.bias is None
    torch.manual_seed(42)
    split = SplitTradeNetwork(1, 20)
    for name, value in split.state_dict().items():
        torch.testing.assert_close(networks[0].state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("n", [1, 3, 4, 7, 19, 298, 300])
@pytest.mark.parametrize("valid_count", [0, 1, 3, None])
def test_all_arms_have_exactly_identical_initial_forward_for_masked_pool_sizes(n, valid_count):
    obs = observation(n=n)
    obs["valid_mask"][:, n if valid_count is None else min(valid_count, n):] = 0
    torch.manual_seed(42)
    reference = SplitTradeNetwork(1, 20)(obs)
    for mode in POOLING_MODES:
        network = initialized(mode)
        for actual, expected in zip(network(obs), reference):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        components = network.encode_components(obs)
        torch.testing.assert_close(components["market"], components["market_mean"], rtol=0, atol=0)


def test_initial_forward_is_identical_for_mixed_batch_masks_and_longer_lookback():
    obs = observation(n=19, lookback=5)
    obs = {key: value.expand(4, *value.shape[1:]).clone() for key, value in obs.items()}
    for row, valid_count in enumerate((0, 1, 7, 19)):
        obs["valid_mask"][row, valid_count:] = 0
    torch.manual_seed(42)
    reference = SplitTradeNetwork(5, 20)(obs)
    for mode in POOLING_MODES:
        torch.manual_seed(42)
        network = MarketSummaryNetwork(5, 20, pooling=mode)
        for actual, expected in zip(network(obs), reference):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("branch", ["actor", "critic"])
def test_shared_parameter_initial_gradients_are_exactly_identical_across_arms(branch):
    all_gradients = []
    for mode in POOLING_MODES:
        network = initialized(mode)
        parameters = {name: parameter for name, parameter in network.named_parameters()
                      if name not in ("summary_query", "variance_projection.weight")}
        outputs = network(observation())
        loss = outputs[0].square().mean() + outputs[1].square().mean() if branch == "actor" else outputs[2].square().mean()
        all_gradients.append(torch.autograd.grad(loss, list(parameters.values()), allow_unused=True))
    for gradients in all_gradients[1:]:
        for actual, expected in zip(gradients, all_gradients[0]):
            assert (actual is None) == (expected is None)
            if actual is not None:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_all_policy_initial_distributions_and_sampled_actions_are_identical():
    obs = observation()
    instances = []
    for mode in POOLING_MODES:
        torch.manual_seed(42)
        instances.append(policy(obs, mode))
    reference = serialized_state(instances[0])
    outputs = []
    for instance in instances:
        assert serialized_state(instance) == reference
        torch.manual_seed(123)
        with torch.no_grad():
            outputs.append(instance(obs))
    for output in outputs[1:]:
        for actual, expected in zip(output, outputs[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", POOLING_MODES)
def test_market_summary_never_depends_on_account_state_even_after_residual_activation(mode):
    obs, changed = observation(), observation()
    changed["portfolio"][:, 0] += .03
    changed["portfolio"][:, 1] -= .03
    changed["buy_mask"][:, 0] = 0
    changed["sell_mask"][:, 1] = 0
    changed["execution_features"][:, 0] = torch.tensor([.2, .5, 1.])
    changed["trade_lot_bounds"][:, 0] = 0
    changed["share_state"][:, 1] = 0
    changed["legal_directions"][:, 0] = torch.tensor([1, 0, 0, 0], dtype=torch.int8)
    changed["allocation_context"][:, 0] = .7
    changed["receivable_ratio"][:] = .1
    network = initialized(mode)
    activate_summary(network)
    first, second = network.encode_components(obs), network.encode_components(changed)
    for key in ("market_encoded", "market", "market_mean", "market_variance", "market_attention_weights"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert not torch.equal(first["held"], second["held"])
    assert not torch.equal(first["context"], second["context"])


@pytest.mark.parametrize("mode", POOLING_MODES)
def test_masked_moments_attention_and_held_invalid_bookkeeping_are_correct(mode):
    obs = observation()
    obs["valid_mask"][:, -1] = 0
    network = initialized(mode)
    activate_summary(network)
    result = network.encode_components(obs)
    encodings = result["market_encoded"][:, :-1]
    expected_mean = encodings.mean(1)
    expected_var = (encodings - expected_mean.unsqueeze(1)).square().mean(1)
    torch.testing.assert_close(result["market_mean"], expected_mean)
    torch.testing.assert_close(result["market_variance"], expected_var)
    torch.testing.assert_close(result["held"],
        (result["encoded"] * obs["portfolio"][:, :-1].unsqueeze(-1)).sum(1))
    assert result["market_attention_weights"][0, -1].item() == 0
    if mode == "attention":
        logits = (encodings * network.summary_query).sum(-1) / network.summary_query.numel() ** .5
        expected_weights = torch.softmax(logits, 1)
        torch.testing.assert_close(result["market_attention_weights"][:, :-1], expected_weights)
        torch.testing.assert_close(result["market"], (encodings * expected_weights.unsqueeze(-1)).sum(1))
    elif mode == "dispersion":
        torch.testing.assert_close(result["market"], expected_mean + network.variance_projection(expected_var))
    else:
        torch.testing.assert_close(result["market"], expected_mean, rtol=0, atol=0)


@pytest.mark.parametrize("mode", POOLING_MODES)
def test_invalid_features_do_not_change_summary_but_held_invalid_stock_is_preserved(mode):
    obs = observation()
    obs["valid_mask"][:, -1] = 0
    changed = deepcopy(obs)
    changed["stock_features"][:, -1] += 100
    network = initialized(mode)
    activate_summary(network)
    first, second = network.encode_components(obs), network.encode_components(changed)
    for key in ("market", "market_mean", "market_variance", "market_attention_weights"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert not torch.equal(first["held"], second["held"])


@pytest.mark.parametrize("mode", POOLING_MODES)
def test_permutation_equivariance_pool_expansion_and_invalid_padding(mode):
    obs = observation()
    network = initialized(mode)
    activate_summary(network)
    before = network(obs)
    permutation = [3, 1, 0, 2]
    other = deepcopy(obs)
    stock_keys = ("stock_features", "valid_mask", "buy_mask", "sell_mask", "execution_features",
                  "share_state", "trade_lot_bounds", "legal_directions")
    for key in stock_keys:
        other[key] = obs[key][:, permutation]
    other["portfolio"] = obs["portfolio"][:, permutation + [4]]
    after = network(other)
    torch.testing.assert_close(after[0], before[0][:, permutation], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(after[1], before[1][:, permutation], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(after[2], before[2], rtol=1e-5, atol=1e-6)
    padded = deepcopy(obs)
    for key in stock_keys:
        padded[key] = torch.cat([obs[key], torch.zeros_like(obs[key][:, :1])], dim=1)
    padded["portfolio"] = torch.cat([obs["portfolio"][:, :-1], torch.zeros(1, 1), obs["portfolio"][:, -1:]], 1)
    padding_result = network(padded)
    torch.testing.assert_close(padding_result[0][:, :4], before[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(padding_result[1][:, :4], before[1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(padding_result[2], before[2], rtol=1e-5, atol=1e-6)
    for n in (1, 7, 300):
        result = network(observation(n=n))
        assert result[0].shape == (1, n, 4) and result[1].shape == (1, n)
        assert result[2].shape == (1, 1)
        assert all(torch.isfinite(value).all() for value in result)


@pytest.mark.parametrize("mode", POOLING_MODES)
def test_no_valid_stocks_is_finite_and_hold_only_after_activation(mode):
    obs = observation()
    obs["valid_mask"][:] = 0
    obs["execution_features"][..., 2] = 1
    obs["legal_directions"][:] = torch.tensor([1, 0, 0, 0], dtype=torch.int8)
    instance = policy(obs, mode)
    activate_summary(instance.network)
    with torch.no_grad():
        components = instance.network.encode_components(obs)
        for key in ("market", "market_mean", "market_variance", "market_attention_weights"):
            assert torch.count_nonzero(components[key]) == 0
        assert torch.count_nonzero(components["held"]) != 0
        actions, value, log_prob = instance(obs)
        assert (actions == HOLD).all() and (log_prob == 0).all()
        assert torch.isfinite(value).all()
        assert (instance.get_distribution(obs).entropy() == 0).all()


@pytest.mark.parametrize("mode", POOLING_MODES)
def test_single_valid_stock_has_zero_residual_even_with_nonzero_parameters(mode):
    obs = observation()
    obs["valid_mask"][:, 1:] = 0
    network = initialized(mode)
    activate_summary(network)
    components = network.encode_components(obs)
    assert torch.count_nonzero(components["market_variance"]) == 0
    torch.testing.assert_close(components["market"], components["market_encoded"][:, 0], rtol=0, atol=0)


@pytest.mark.parametrize("mode", POOLING_MODES)
def test_only_the_selected_residual_receives_initial_actor_and_value_gradients(mode):
    network = initialized(mode)
    before = serialized_state(network)
    named = dict(network.named_parameters())
    for branch in ("actor", "critic"):
        outputs = network(observation())
        loss = outputs[0].square().mean() + outputs[1].square().mean() if branch == "actor" else outputs[2].square().mean()
        gradients = dict(zip(named, torch.autograd.grad(loss, list(named.values()), allow_unused=True)))
        for parameter_name, selected_mode in (("summary_query", "attention"),
                                              ("variance_projection.weight", "dispersion")):
            gradient = gradients[parameter_name]
            if mode == selected_mode:
                assert gradient is not None and torch.count_nonzero(gradient)
            else:
                assert gradient is None
        assert all(parameter.grad is None for parameter in network.parameters())
    assert serialized_state(network) == before


@pytest.mark.parametrize("mode", POOLING_MODES)
def test_forward_evaluate_likelihood_and_constructor_save_load(mode, tmp_path):
    obs = observation()
    obs["legal_directions"][:, 1] = torch.tensor([1, 0, 1, 1], dtype=torch.int8)
    obs["trade_lot_bounds"][:, 1] = torch.tensor([0, 0, 2, 2])
    instance = policy(obs, mode)
    activate_summary(instance.network)
    before = serialized_state(instance)
    with torch.no_grad():
        actions, value, original = instance(obs)
        actual_value, actual, entropy = instance.evaluate_actions(obs, actions)
        torch.testing.assert_close(value, actual_value, rtol=0, atol=0)
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
        assert torch.isfinite(entropy).all()
        raw = torch.tensor([[BUY, REDUCE, HOLD, HOLD, 12., 0., 0., 0.]])
        expected = instance.get_distribution(obs).log_prob(raw)
        torch.testing.assert_close(instance.evaluate_actions(obs, raw)[1], expected, rtol=0, atol=0)
        assert torch.isfinite(expected).all()
        raw[:, 5] = .1
        assert torch.isneginf(instance.evaluate_actions(obs, raw)[1]).all()
        deterministic = instance(obs, deterministic=True)
    destination = tmp_path / "market_summary.pt"
    instance.save(destination)
    restored = MarketSummaryPolicy.load(destination, device="cpu")
    assert restored.pooling == restored.network.pooling == mode
    assert restored.architecture == "split" and restored.parameter_count == 141575
    assert restored.actor_output_gain == .1 and restored.initial_log_std == -.7
    assert serialized_state(restored) == serialized_state(instance) == before
    with torch.no_grad():
        for actual, expected in zip(restored(obs, deterministic=True), deterministic):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", POOLING_MODES)
def test_ppo_agent_preserves_summary_flag_in_synthetic_artifact(mode, local_data, tmp_path):
    config = trade_config(local_data)
    config["actor_output_gain"] = .1
    data = ResearchData(config)
    env = data.vector_env()
    try:
        model = PPO(MarketSummaryPolicy, env, device="cpu", seed=42, verbose=0,
            policy_kwargs=dict(hidden_sizes=tuple(config["hidden_sizes"]), actor_output_gain=.1,
                initial_log_std=-.7, pooling=mode), **config["ppo"])
        agent = PPOAgent(model, data.contract)
        obs = {name: value[0] for name, value in env.reset().items()}
        before = serialized_state(model.policy)
        original = agent.inspect_action(obs)
        destination = tmp_path / "synthetic_market_summary"
        agent.save(destination, config)
        restored = PPOAgent.load(destination, data.contract, device="cpu")
        assert restored.model.policy.pooling == mode
        assert restored.model.policy.architecture == "split"
        assert serialized_state(restored.model.policy) == serialized_state(model.policy) == before
        actual = restored.inspect_action(obs)
        torch.testing.assert_close(torch.as_tensor(actual["latent"]), torch.as_tensor(original["latent"]), rtol=0, atol=0)
        assert restored.model.num_timesteps == model.num_timesteps == 0
    finally:
        env.close()


@pytest.mark.parametrize("mode", [None, "other", 1])
def test_invalid_pooling_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="pooling"):
        MarketSummaryNetwork(1, 20, pooling=mode)
    with pytest.raises(ValueError, match="pooling"):
        policy(observation(), mode)


def test_non_split_architecture_is_rejected():
    with pytest.raises(ValueError, match="architecture='split'"):
        MarketSummaryPolicy(*spaces(observation()), lambda _: 1e-5, architecture="baseline")
