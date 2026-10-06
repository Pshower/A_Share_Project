"""Research architecture checks: synthetic CPU forwards, no optimizer updates."""

from copy import deepcopy
import math

import gymnasium as gym
import numpy as np
import pytest
import torch

from src.agents.exploration_policy import ExplorationTradePolicy, SplitTradeNetwork
from src.agents.trade_mapping import HOLD, BUY, REDUCE
from src.agents.trade_policy import SharedTradeNetwork, SharedTradePolicy
from tests.test_ppo_framework import no_parameter_updates


pytestmark = pytest.mark.usefixtures("no_parameter_updates")


def observation(n=4, lookback=1, features=20):
    generator = torch.Generator().manual_seed(13)
    weights = torch.linspace(.02, .12, n).unsqueeze(0)
    return dict(stock_features=torch.randn(1, n, lookback, features, generator=generator),
        portfolio=torch.cat([weights, 1 - weights.sum(1, keepdim=True)], dim=1),
        receivable_ratio=torch.zeros(1, 1), valid_mask=torch.ones(1, n),
        buy_mask=torch.ones(1, n), sell_mask=torch.ones(1, n),
        execution_features=torch.tensor([.001, .1, 0.]).expand(1, n, -1).clone(),
        allocation_context=torch.tensor([[1., .5, .000005, .5, 0., .5, .005]]),
        share_state=torch.ones(1, n, 2, dtype=torch.float64),
        trade_lot_bounds=torch.tensor([5, 20, 5, 10], dtype=torch.int64).expand(1, n, -1).clone(),
        legal_directions=torch.ones(1, n, 4, dtype=torch.int8))


def spaces(obs):
    values = {key: value[0].numpy() for key, value in obs.items()}
    space = gym.spaces.Dict({key: gym.spaces.Box(
        low=0 if np.issubdtype(value.dtype, np.integer) else -np.inf,
        high=np.iinfo(value.dtype).max if np.issubdtype(value.dtype, np.integer) else np.inf,
        shape=value.shape, dtype=value.dtype) for key, value in values.items()})
    n = values["stock_features"].shape[0]
    return space, gym.spaces.Box(-8., 8., (2 * n,), dtype=np.float32)


def policy(obs, architecture, **kwargs):
    observation_space, action_space = spaces(obs)
    return ExplorationTradePolicy(observation_space, action_space, lambda _: 1e-5,
        architecture=architecture, actor_output_gain=.1, initial_log_std=-.7, **kwargs)


@pytest.mark.parametrize("lookback", [1, 5])
def test_baseline_matches_production_network_and_parameter_count(lookback):
    obs = observation(lookback=lookback)
    torch.manual_seed(42)
    reference = SharedTradeNetwork(lookback, 20, actor_output_gain=.1)
    torch.manual_seed(42)
    actual = policy(obs, "baseline")
    assert actual.parameter_count == 126343 + (lookback - 1) * 20 * 256
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(actual.network.state_dict()[name], value, rtol=0, atol=0)
    for output, expected in zip(actual.network(obs), reference(obs)):
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.manual_seed(42)
    production = SharedTradePolicy(*spaces(obs), lambda _: 1e-5, actor_output_gain=.1,
                                   initial_log_std=-.7)
    for output, expected in zip(actual(obs, deterministic=True), production(obs, deterministic=True)):
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("lookback", [1, 5])
def test_split_parameter_count_gain_and_no_buy_or_hold_bias(lookback):
    torch.manual_seed(42)
    network = SplitTradeNetwork(lookback, 20)
    assert network.parameter_count == 125063 + (lookback - 1) * 20 * 256
    for head in (network.direction_head, network.size_head):
        torch.testing.assert_close(head.weight @ head.weight.T,
            torch.eye(head.out_features) * .1 ** 2, atol=1e-8, rtol=1e-5)
        assert torch.count_nonzero(head.bias) == 0
    assert network.log_std.item() == pytest.approx(-.7)
    assert network.log_std.exp().item() == pytest.approx(math.exp(-.7))


def test_split_market_encoding_excludes_holdings_and_trade_support_but_heads_see_account():
    obs = observation()
    network = SplitTradeNetwork(1, 20)
    before = network.encode_components(obs)
    changed = deepcopy(obs)
    changed["portfolio"][:, :2] = changed["portfolio"][:, [1, 0]]
    changed["buy_mask"][:, 0] = 0
    changed["sell_mask"][:, 1] = 0
    changed["trade_lot_bounds"][:, 0] = 0
    changed["execution_features"][:, 0, 2] = 1
    after = network.encode_components(changed)
    for key in ("market_encoded", "market"):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    for key in ("account_encoded", "encoded", "held", "context"):
        assert not torch.equal(before[key], after[key])
    original_logits, original_size, original_value, _ = network(obs)
    other_logits, other_size, other_value, _ = network(changed)
    assert not torch.equal(original_logits, other_logits)
    assert not torch.equal(original_size, other_size)
    assert not torch.equal(original_value, other_value)


def test_split_invalid_market_is_excluded_but_held_invalid_asset_remains_in_context():
    obs = observation()
    obs["valid_mask"][:, -1] = 0
    network = SplitTradeNetwork(1, 20)
    result = network.encode_components(obs)
    torch.testing.assert_close(result["market"], result["market_encoded"][:, :-1].mean(1))
    torch.testing.assert_close(result["held"],
        (result["encoded"] * obs["portfolio"][:, :-1].unsqueeze(-1)).sum(1))
    without_invalid_holding = (result["encoded"][:, :-1]
        * obs["portfolio"][:, :-2].unsqueeze(-1)).sum(1)
    assert not torch.equal(result["held"], without_invalid_holding)


def test_lookback_flattens_only_each_stocks_day_feature_axes():
    obs = observation(n=3, lookback=5, features=2)
    network = SplitTradeNetwork(5, 2, (16, 8))
    captured = []
    hook = network.market_encoder[0].register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone()))
    try:
        network(obs)
    finally:
        hook.remove()
    assert captured[0].shape == (1, 3, 10)
    torch.testing.assert_close(captured[0], obs["stock_features"].reshape(1, 3, 10))


@pytest.mark.parametrize("architecture", ["baseline", "split"])
@pytest.mark.parametrize("lookback", [1, 5])
def test_permutation_equivariance_and_different_stock_counts(architecture, lookback):
    obs = observation(lookback=lookback)
    network = policy(obs, architecture).network
    logits, size, value, _ = network(obs)
    permutation = [3, 1, 0, 2]
    permuted = deepcopy(obs)
    for key in ("stock_features", "valid_mask", "buy_mask", "sell_mask", "execution_features",
                "share_state", "trade_lot_bounds", "legal_directions"):
        permuted[key] = obs[key][:, permutation]
    permuted["portfolio"] = obs["portfolio"][:, permutation + [4]]
    other_logits, other_size, other_value, _ = network(permuted)
    torch.testing.assert_close(other_logits, logits[:, permutation], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(other_size, size[:, permutation], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(other_value, value, rtol=1e-5, atol=1e-6)
    for n in (1, 7):
        output = network(observation(n=n, lookback=lookback))
        assert output[0].shape == (1, n, 4) and output[1].shape == (1, n)
        assert output[2].shape == (1, 1)
        assert all(torch.isfinite(item).all() for item in output)


@pytest.mark.parametrize("architecture", ["baseline", "split"])
def test_no_valid_stocks_and_frozen_support_produces_only_hold(architecture):
    obs = observation()
    obs["valid_mask"][:] = 0
    obs["execution_features"][..., 2] = 1
    obs["legal_directions"][:] = torch.tensor([1, 0, 0, 0], dtype=torch.int8)
    result = policy(obs, architecture)
    with torch.no_grad():
        action, value, log_prob = result(obs)
        assert (action == HOLD).all() and (log_prob == 0).all()
        assert torch.isfinite(value).all()
        assert (result.get_distribution(obs).entropy() == 0).all()
        if architecture == "split":
            components = result.network.encode_components(obs)
            assert torch.count_nonzero(components["market"]) == 0
            assert torch.count_nonzero(components["held"]) != 0
        obs["legal_directions"][:, 0, BUY] = 1
        with pytest.raises(ValueError, match="only allow HOLD"):
            result(obs)


@pytest.mark.parametrize("architecture", ["baseline", "split"])
@pytest.mark.parametrize("lookback", [1, 5])
def test_forward_evaluate_raw_likelihood_and_constructor_save_load(architecture, lookback, tmp_path):
    obs = observation(lookback=lookback)
    obs["legal_directions"][:, 1] = torch.tensor([1, 0, 1, 1], dtype=torch.int8)
    obs["trade_lot_bounds"][:, 1] = torch.tensor([0, 0, 2, 2])
    result = policy(obs, architecture)
    original_parameters = {key: value.clone() for key, value in result.state_dict().items()}
    with torch.no_grad():
        actions, value, old_log_prob = result(obs)
        other_value, log_prob, entropy = result.evaluate_actions(obs, actions)
        torch.testing.assert_close(old_log_prob, log_prob, rtol=0, atol=0)
        torch.testing.assert_close(value, other_value, rtol=0, atol=0)
        assert torch.isfinite(entropy).all()
        raw = torch.tensor([[BUY, REDUCE, HOLD, HOLD, 12., 0., 0., 0.]])
        expected = result.get_distribution(obs).log_prob(raw)
        torch.testing.assert_close(result.evaluate_actions(obs, raw)[1], expected, rtol=0, atol=0)
        assert torch.isfinite(expected).all()
        raw[:, 5] = .1
        assert torch.isneginf(result.evaluate_actions(obs, raw)[1]).all()
        original = result(obs, deterministic=True)
    destination = tmp_path / "policy.pt"
    result.save(destination)
    loaded = ExplorationTradePolicy.load(destination, device="cpu")
    assert loaded.architecture == architecture
    assert loaded.hidden_sizes == (256, 128)
    assert loaded.initial_log_std == -.7 and loaded.actor_output_gain == .1
    assert loaded.parameter_count == result.parameter_count
    with torch.no_grad():
        for actual, expected in zip(loaded(obs, deterministic=True), original):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, tensor in result.state_dict().items():
        torch.testing.assert_close(tensor, original_parameters[name], rtol=0, atol=0)


@pytest.mark.parametrize("architecture", ["other", None, 1])
def test_unknown_architecture_is_rejected(architecture):
    with pytest.raises(ValueError, match="architecture"):
        policy(observation(), architecture)


@pytest.mark.parametrize("arguments", [dict(lookback=0), dict(features=0),
    dict(hidden_sizes=(16,)), dict(hidden_sizes=(16, 0)), dict(actor_output_gain=0)])
def test_invalid_split_dimensions_are_rejected(arguments):
    kwargs = dict(lookback=1, features=20)
    kwargs.update(arguments)
    with pytest.raises(ValueError, match="Invalid split"):
        SplitTradeNetwork(**kwargs)
