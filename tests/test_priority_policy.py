"""Synthetic probability, atom support, initialization and gradient checks."""

import math

import pytest
import torch
from torch.distributions import Normal

from src.agents.exploration_policy import SplitTradeNetwork
from src.agents.priority_policy import PriorityTradeNetwork, StockPriorityTradeDistribution
from src.agents.trade_mapping import BUY, EXIT, HOLD, REDUCE
from src.agents.trade_policy import StockTradeDistribution
from tests.test_trade_policy import distribution as old_distribution, network_observation


def distribution(variant="coupled", batch=1):
    old, logits, mean, legal, bounds = old_distribution(batch)
    result = StockPriorityTradeDistribution(3, variant).proba_distribution(
        logits, mean, torch.tensor([-.7]), mean + .1, torch.tensor([-.4]), legal, bounds)
    return result, old


def test_coupled_probability_and_entropy_exactly_match_legacy_density():
    result, old = distribution()
    raw = torch.tensor([[BUY, REDUCE, HOLD, .4, 0., 0., .4, 0., 0.]])
    torch.testing.assert_close(result.log_prob(raw), old.log_prob(raw[:, :6]), rtol=0, atol=0)
    torch.testing.assert_close(result.entropy(), old.entropy(), rtol=0, atol=0)
    assert result.conditional_priority_entropy().item() == 0


def test_decoupled_probability_adds_only_buy_priority_density():
    result, old = distribution("decoupled")
    raw = torch.tensor([[BUY, REDUCE, HOLD, .4, 0., 0., -.8, 0., 0.]])
    expected = old.log_prob(raw[:, :6]) + Normal(.35, math.exp(-.4)).log_prob(raw[:, 6])
    torch.testing.assert_close(result.log_prob(raw), expected)
    expected_entropy = old.entropy() + result.direction_distribution.probs[:, 0, BUY] * Normal(0., math.exp(-.4)).entropy()
    torch.testing.assert_close(result.entropy(), expected_entropy)


@pytest.mark.parametrize("variant", ["coupled", "decoupled"])
def test_samples_obey_atoms_and_have_finite_joint_probabilities(variant):
    torch.manual_seed(42)
    result, _ = distribution(variant, 512)
    raw = result.sample()
    codes, sizes, priorities = raw.split(3, -1)
    assert (priorities[codes != BUY] == 0).all()
    assert (sizes[:, 2] == 0).all() and (priorities[:, 2] == 0).all()
    if variant == "coupled":
        torch.testing.assert_close(priorities[codes == BUY], sizes[codes == BUY], rtol=0, atol=0)
    else:
        assert (priorities[codes == BUY] != sizes[codes == BUY]).all()
    assert torch.isfinite(result.log_prob(raw)).all()


@pytest.mark.parametrize("variant", ["coupled", "decoupled"])
@pytest.mark.parametrize("corruption", ["fractional", "illegal", "inactive_size", "inactive_priority", "nan", "infinite"])
def test_invalid_requests_fail_closed(variant, corruption):
    result, _ = distribution(variant)
    raw = torch.tensor([[BUY, REDUCE, HOLD, .4, 0., 0., .4, 0., 0.]])
    if corruption == "fractional":
        raw[0, 0] = 1.5
    elif corruption == "illegal":
        raw[0, 2] = EXIT
    elif corruption == "inactive_size":
        raw[0, 4] = .1
    elif corruption == "inactive_priority":
        raw[0, 7] = .1
    else:
        raw[0, 6] = float("nan") if corruption == "nan" else float("inf")
    assert torch.isneginf(result.log_prob(raw)).all()


def test_coupled_priority_copy_atom_rejects_even_finite_nearby_value():
    result, _ = distribution()
    raw = torch.tensor([[BUY, REDUCE, HOLD, .4, 0., 0., .400001, 0., 0.]])
    assert torch.isneginf(result.log_prob(raw)).all()


@pytest.mark.parametrize("variant", ["coupled", "decoupled"])
def test_original_unbounded_latents_retain_normal_log_probability(variant):
    result, _ = distribution(variant)
    raw = torch.tensor([[BUY, REDUCE, HOLD, 12., 0., 0., 12. if variant == "coupled" else -15., 0., 0.]])
    assert torch.isfinite(result.log_prob(raw)).all()
    clipped = raw.clone()
    clipped[:, 3:] = clipped[:, 3:].clamp(-8, 8)
    assert not torch.equal(result.log_prob(raw), result.log_prob(clipped))


@pytest.mark.parametrize("variant", ["coupled", "decoupled"])
def test_inactive_priority_head_has_no_actor_gradients(variant):
    logits = torch.tensor([[[1., 2., 3., 4.]]], requires_grad=True)
    mean = torch.tensor([[.1]], requires_grad=True)
    priority_mean = torch.tensor([[.3]], requires_grad=True)
    priority_log_std = torch.tensor([-.7], requires_grad=True)
    result = StockPriorityTradeDistribution(1, variant).proba_distribution(logits, mean,
        torch.tensor([-.7]), priority_mean, priority_log_std,
        torch.tensor([[[1, 1, 0, 0]]]), torch.tensor([[[1, 3, 0, 0]]]))
    raw = torch.tensor([[HOLD, 0., 0.]])
    gradients = torch.autograd.grad(result.log_prob(raw).sum(),
        [mean, priority_mean, priority_log_std], allow_unused=True)
    assert all(value is None or torch.count_nonzero(value) == 0 for value in gradients)


def test_decoupled_priority_receives_gradient_but_coupled_head_is_unused():
    for variant in ["coupled", "decoupled"]:
        mean = torch.tensor([[.1]], requires_grad=True)
        priority_mean = torch.tensor([[.3]], requires_grad=True)
        result = StockPriorityTradeDistribution(1, variant).proba_distribution(
            torch.zeros(1, 1, 4), mean, torch.tensor([-.7]), priority_mean,
            torch.tensor([-.7]), torch.tensor([[[1, 1, 0, 0]]]), torch.tensor([[[1, 3, 0, 0]]]))
        raw = torch.tensor([[BUY, .8, .8 if variant == "coupled" else -.8]])
        grad = torch.autograd.grad(result.log_prob(raw).sum(), priority_mean, allow_unused=True)[0]
        assert grad is None if variant == "coupled" else grad is not None and grad.abs().item() > 0


def test_network_common_initial_parameters_and_functions_match_split():
    obs = network_observation()
    torch.manual_seed(42)
    old = SplitTradeNetwork(1, 20)
    torch.manual_seed(42)
    new = PriorityTradeNetwork(1, 20)
    assert new.parameter_count == 125322
    for name, parameter in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[name], parameter, rtol=0, atol=0)
    old_outputs, new_outputs = old(obs), new(obs)
    for before, after in zip(old_outputs, new_outputs[:4]):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    torch.testing.assert_close(new_outputs[1], new_outputs[4], rtol=0, atol=0)
    torch.testing.assert_close(new_outputs[3], new_outputs[5], rtol=0, atol=0)
    assert torch.count_nonzero(new.signal_head.weight) == 0 and torch.count_nonzero(new.signal_head.bias) == 0


def test_network_permutation_and_pure_market_account_separation():
    obs = network_observation()
    network = PriorityTradeNetwork(1, 20)
    original = network(obs)
    perm = [2, 0, 1]
    other = {name: value.clone() for name, value in obs.items()}
    for name in ["stock_features", "valid_mask", "buy_mask", "sell_mask", "execution_features", "trade_lot_bounds"]:
        other[name] = other[name][:, perm]
    other["portfolio"] = other["portfolio"][:, [2, 0, 1, 3]]
    reordered = network(other)
    for index in [0, 1, 4]:
        torch.testing.assert_close(reordered[index], original[index][:, perm], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(reordered[2], original[2], atol=1e-6, rtol=1e-5)
    changed = {name: value.clone() for name, value in obs.items()}
    changed["portfolio"][:, [0, 1]] = changed["portfolio"][:, [1, 0]]
    torch.testing.assert_close(network.encode_components(changed)["market"], network.encode_components(obs)["market"], rtol=0, atol=0)


@pytest.mark.parametrize("variant", [None, "sparse", ""])
def test_unknown_actor_variant_is_rejected(variant):
    with pytest.raises(ValueError, match="actor_variant"):
        StockPriorityTradeDistribution(1, variant)


@pytest.mark.parametrize("field", ["shape", "nan_mean", "nan_std", "overflow_std"])
def test_bad_priority_parameters_fail_closed(field):
    _, logits, mean, legal, bounds = old_distribution()
    priority = mean.clone()
    scale = torch.tensor([-.7])
    if field == "shape":
        priority = priority[:, :1]
    elif field == "nan_mean":
        priority[0, 0] = float("nan")
    else:
        scale[:] = float("nan") if field == "nan_std" else 1000
    with pytest.raises((ValueError, FloatingPointError)):
        StockPriorityTradeDistribution(3).proba_distribution(logits, mean, torch.tensor([-.7]), priority, scale, legal, bounds)


@pytest.mark.parametrize("n", [1, 7, 300])
@pytest.mark.parametrize("variant", ["coupled", "decoupled"])
def test_variable_pool_distribution_shapes_and_permutation(n, variant):
    torch.manual_seed(12)
    logits, mean, priority = torch.randn(2, n, 4), torch.randn(2, n), torch.randn(2, n)
    legal = torch.ones(2, n, 4, dtype=torch.int8)
    bounds = torch.tensor([1, 10, 1, 10]).expand(2, n, -1)
    legal[:, ::3] = torch.tensor([1, 0, 0, 0], dtype=torch.int8)
    dist = StockPriorityTradeDistribution(n, variant).proba_distribution(
        logits, mean, torch.tensor([-.7]), priority, torch.tensor([-.7]), legal, bounds)
    raw = dist.sample()
    assert raw.shape == (2, 3 * n) and torch.isfinite(dist.log_prob(raw)).all()
    permutation = torch.randperm(n)
    other = StockPriorityTradeDistribution(n, variant).proba_distribution(
        logits[:, permutation], mean[:, permutation], torch.tensor([-.7]), priority[:, permutation],
        torch.tensor([-.7]), legal[:, permutation], bounds[:, permutation])
    reordered = torch.cat([field[:, permutation] for field in raw.split(n, -1)], -1)
    torch.testing.assert_close(other.log_prob(reordered), dist.log_prob(raw), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(other.entropy(), dist.entropy(), rtol=1e-5, atol=1e-5)


def test_generic_distribution_head_factory_is_3n_compatible_and_copied():
    heads = StockPriorityTradeDistribution(7).proba_distribution_net(5)
    direction, quantity, log_std, priority, priority_std = heads
    assert direction(torch.ones(2, 5)).shape == (2, 28)
    torch.testing.assert_close(quantity(torch.ones(2, 5)), priority(torch.ones(2, 5)), rtol=0, atol=0)
    torch.testing.assert_close(log_std, priority_std, rtol=0, atol=0)
    assert quantity.weight is not priority.weight and log_std is not priority_std
