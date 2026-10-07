"""Synthetic autograd only: diagnostics never learn, step Adam, or accumulate .grad."""

from copy import deepcopy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.agents.exploration_policy import SplitTradeNetwork
from src.agents.trade_policy import SharedTradePolicy, StockTradeDistribution
from src.training.attribution_gradients import gradient_diagnostics
from tests.test_exploration_policy import observation, spaces
from tests.test_ppo_framework import no_parameter_updates


pytestmark = pytest.mark.usefixtures("no_parameter_updates")


class AttributionNetwork(SplitTradeNetwork):
    def forward(self, obs):
        components = self.encode_components(obs)
        encoded, context = components["encoded"], components["context"]
        if self.mixed_market:
            valid = obs["valid_mask"].float()
            mixed = (encoded * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
            context = torch.cat([mixed, context[:, encoded.shape[-1]:]], dim=1)
        stock = self.stock_body(torch.cat([encoded, context.unsqueeze(1).expand(-1, encoded.shape[1], -1)], dim=-1))
        value_context = context.detach() if self.detach_critic else context
        return self.direction_head(stock), self.size_head(stock).squeeze(-1), self.value_head(value_context), self.log_std


class AttributionPolicy(SharedTradePolicy):
    def __init__(self, *args, mixed_market=False, detach_critic=False, **kwargs):
        self.mixed_market, self.detach_critic = mixed_market, detach_critic
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        n, lookback, features = self.observation_space["stock_features"].shape
        self.network = AttributionNetwork(lookback, features, self.hidden_sizes,
                                         self.initial_log_std, self.actor_output_gain)
        self.network.mixed_market, self.network.detach_critic = self.mixed_market, self.detach_critic
        self.action_dist = StockTradeDistribution(n)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


def synthetic(mixed=False, detached=False, batch_size=8):
    torch.manual_seed(42)
    base = observation(n=4)
    policy = AttributionPolicy(*spaces(base), lambda _: 1e-5, hidden_sizes=(16, 8),
        actor_output_gain=.1, mixed_market=mixed, detach_critic=detached)
    obs = {key: value.repeat(batch_size, *([1] * (value.ndim - 1))) for key, value in base.items()}
    obs["stock_features"] += torch.linspace(-.2, .2, batch_size)[:, None, None, None]
    directions = np.asarray([np.roll([1, 2, 0, 3], row % 4) for row in range(batch_size)], dtype=np.float32)
    sizes = np.linspace(-.8, .7, batch_size * 4).reshape(batch_size, 4).astype(np.float32)
    sizes[(directions == 0) | (directions == 3)] = 0
    actions = np.concatenate([directions, sizes], axis=1)
    with torch.no_grad():
        _, log_prob, _ = policy.evaluate_actions(obs, torch.from_numpy(actions))
    batch = dict(observations={key: value.numpy().copy() for key, value in obs.items()}, actions=actions,
        old_log_prob=log_prob.numpy().copy(), advantages=np.linspace(-2, 3, batch_size).astype(np.float32),
        returns=np.linspace(-.1, .4, batch_size).astype(np.float32))
    model = SimpleNamespace(policy=policy, normalize_advantage=True, clip_range_vf=None, ent_coef=0.0)
    return model, batch


def nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for first, second in zip(a, b):
            nested_equal(first, second)
    else:
        assert a == b


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("detached", [False, True])
def test_four_cells_share_named_parameter_sets_and_detached_encoder_credit(mixed, detached):
    model, batch = synthetic(mixed, detached)
    result = gradient_diagnostics(model, batch)
    reference, _ = synthetic()
    assert result["parameter_count"] == sum(p.numel() for p in reference.policy.parameters())
    assert dict(model.policy.named_parameters()).keys() == dict(reference.policy.named_parameters()).keys()
    for name, parameter in model.policy.named_parameters():
        assert torch.equal(parameter, dict(reference.policy.named_parameters())[name])
    assert result["groups"]["combined_encoder"]["parameter_count"] == sum(
        result["groups"][key]["parameter_count"] for key in ["market_encoder", "account_encoder"])
    assert result["groups"]["combined_encoder"]["norm_actor"] > 0
    for key in ["market_encoder", "account_encoder", "combined_encoder"]:
        group = result["groups"][key]
        if detached:
            assert group["norm_critic"] == 0 and group["norm_critic_raw"] == 0
            assert group["cosine"] is None and group["dot"] == 0
            assert group["combinednorm"] == pytest.approx(group["norm_actor"])
        else:
            assert group["norm_critic"] > 0 and -1 <= group["cosine"] <= 1
        assert group["combinednorm"] ** 2 == pytest.approx(
            group["norm_actor"] ** 2 + group["norm_critic"] ** 2 + 2 * group["dot"], abs=1e-12)
    assert result["value_head_critic_gradnorm"] > 0
    assert result["actor_head_actor_gradnorm"] > 0
    assert result["actor_head_critic_gradnorm"] == 0
    assert result["groups"]["value_head"]["norm_actor"] == 0
    json.dumps(result, allow_nan=False)


def test_losses_match_installed_sb3_formula_with_sample_std_and_clipping():
    model, batch = synthetic()
    batch["old_log_prob"] -= np.log(np.asarray([2., .5] * 4, dtype=np.float32))
    obs = {key: torch.from_numpy(value) for key, value in batch["observations"].items()}
    with torch.no_grad():
        values, log_prob, entropy = model.policy.evaluate_actions(obs, torch.from_numpy(batch["actions"]))
        advantages = torch.from_numpy(batch["advantages"])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ratio = torch.exp(log_prob - torch.from_numpy(batch["old_log_prob"]))
        actor = -torch.minimum(advantages * ratio, advantages * ratio.clamp(.8, 1.2)).mean()
        critic = torch.nn.functional.mse_loss(values.flatten(), torch.from_numpy(batch["returns"]))
    result = gradient_diagnostics(model, batch)
    assert result["losses"]["actor"] == pytest.approx(float(actor))
    assert result["losses"]["critic_raw"] == pytest.approx(float(critic))
    assert result["losses"]["critic_scaled"] == pytest.approx(.5 * float(critic))
    assert result["losses"]["entropy_mean"] == pytest.approx(float(entropy.mean()))
    assert result["losses"]["advantage_std_ddof"] == 1
    assert result["losses"]["clip_fraction"] == pytest.approx(1)


def test_weighted_and_raw_critic_norms_are_distinct_and_vf_zero_is_finite():
    model, batch = synthetic()
    half, full, zero = (gradient_diagnostics(model, batch, vf_coef=scale) for scale in [.5, 1., 0.])
    for name in half["groups"]:
        a, b, c = half["groups"][name], full["groups"][name], zero["groups"][name]
        assert a["norm_actor"] == pytest.approx(b["norm_actor"])
        assert a["norm_critic"] == pytest.approx(.5 * b["norm_critic"])
        assert a["norm_critic_raw"] == pytest.approx(b["norm_critic_raw"])
        assert c["norm_critic"] == 0 and c["cosine"] is None
        assert c["norm_critic_raw"] == pytest.approx(b["norm_critic_raw"])


def test_reported_vectors_match_independently_computed_full_ppo_gradients():
    model, batch = synthetic(mixed=True)
    obs = {key: torch.from_numpy(value) for key, value in batch["observations"].items()}
    values, log_prob, _ = model.policy.evaluate_actions(obs, torch.from_numpy(batch["actions"]))
    advantages = torch.from_numpy(batch["advantages"])
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    ratio = torch.exp(log_prob - torch.from_numpy(batch["old_log_prob"]))
    actor = -torch.minimum(advantages * ratio, advantages * ratio.clamp(.8, 1.2)).mean()
    critic = .5 * torch.nn.functional.mse_loss(values.flatten(), torch.from_numpy(batch["returns"]))
    named = list(model.policy.named_parameters())
    actor_grads = torch.autograd.grad(actor, [p for _, p in named], retain_graph=True, allow_unused=True)
    critic_grads = torch.autograd.grad(critic, [p for _, p in named], allow_unused=True)
    result = gradient_diagnostics(model, batch)
    for group in result["groups"].values():
        indices = [i for i, (name, _) in enumerate(named) if name in group["parameter_names"]]
        expected = []
        for gradients in [actor_grads, critic_grads]:
            expected.append(torch.cat([torch.zeros_like(named[i][1]).flatten().double() if gradients[i] is None
                                       else gradients[i].flatten().double() for i in indices]))
        a, c = expected
        assert group["norm_actor"] == pytest.approx(float(a.norm()), rel=1e-7)
        assert group["norm_critic"] == pytest.approx(float(c.norm()), rel=1e-7)
        assert group["dot"] == pytest.approx(float(a @ c), rel=1e-7, abs=1e-12)
        assert group["combinednorm"] == pytest.approx(float((a + c).norm()), rel=1e-7)


def test_policy_parameters_existing_grad_references_modes_cache_and_optimizer_are_preserved():
    model, batch = synthetic()
    for index, parameter in enumerate(model.policy.parameters()):
        if index % 2:
            parameter.grad = torch.full_like(parameter, .123)
    model.policy.eval()
    model.policy.network.account_encoder.train(True)
    modes = [module.training for module in model.policy.modules()]
    before = deepcopy(model.policy.state_dict())
    gradients = [(p.grad, None if p.grad is None else p.grad.clone()) for p in model.policy.parameters()]
    optimizer = deepcopy(model.policy.optimizer.state_dict())
    distribution = model.policy.action_dist.__dict__.copy()
    inputs = deepcopy(batch)
    result = gradient_diagnostics(model, batch)
    nested_equal(before, model.policy.state_dict())
    for parameter, (reference, value) in zip(model.policy.parameters(), gradients):
        assert parameter.grad is reference
        if value is not None:
            assert torch.equal(parameter.grad, value)
    assert modes == [module.training for module in model.policy.modules()]
    assert distribution.keys() == model.policy.action_dist.__dict__.keys()
    for key in distribution:
        assert distribution[key] is model.policy.action_dist.__dict__[key]
    nested_equal(optimizer, model.policy.optimizer.state_dict())
    nested_equal(inputs, batch)
    assert all(result["state_preservation"][key] for key in ["parameters_unchanged", "gradients_unchanged", "rng_unchanged"])


def test_python_numpy_torch_rngs_are_restored_even_if_evaluator_consumes_randomness(monkeypatch):
    model, batch = synthetic()
    original = model.policy.evaluate_actions

    def consumes_randomness(*args):
        random.random()
        np.random.random()
        torch.rand(3)
        return original(*args)

    monkeypatch.setattr(model.policy, "evaluate_actions", consumes_randomness)
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    gradient_diagnostics(model, batch)
    nested_equal(states, (random.getstate(), np.random.get_state(), torch.get_rng_state()))


def test_cpu_diagnostics_never_query_or_initialize_cuda(monkeypatch):
    model, batch = synthetic()

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU diagnostics must not initialize or inspect CUDA RNGs")

    for name in ["device_count", "current_device", "get_rng_state", "get_rng_state_all", "_lazy_init"]:
        monkeypatch.setattr(torch.cuda, name, forbidden)
    gradient_diagnostics(model, batch)


def test_exact_int64_bounds_float64_share_state_and_targets_remain_separate(monkeypatch):
    model, batch = synthetic(batch_size=128)
    bounds = batch["observations"]["trade_lot_bounds"]
    bounds[..., 0], bounds[..., 1] = 2 ** 24 + 1, 2 ** 24 + 17
    batch["observations"]["share_state"][..., 0] = 2 ** 24 + .25
    batch["returns"] = batch["returns"].astype(np.float64)
    captured = {}
    original = model.policy.evaluate_actions

    def capture(obs, actions):
        captured.update(bounds=obs["trade_lot_bounds"].clone(), shares=obs["share_state"].clone())
        return original(obs, actions)

    monkeypatch.setattr(model.policy, "evaluate_actions", capture)
    result = gradient_diagnostics(model, batch)
    assert result["batch_size"] == 128
    assert captured["bounds"].dtype == torch.int64 and captured["bounds"][0, 0, 0] == 2 ** 24 + 1
    assert captured["shares"].dtype == torch.float64 and captured["shares"][0, 0, 0] == 2 ** 24 + .25
    assert result["observation_dtypes"]["legal_directions"] == "torch.int8"
    assert result["effective_target_dtypes"]["returns"] == "torch.float32"
    assert result["targets_detached"]


def test_single_sample_and_constant_advantages_do_not_make_nan():
    for size in [1, 8]:
        model, batch = synthetic(batch_size=size)
        batch["advantages"][:] = 0
        result = gradient_diagnostics(model, batch)
        assert result["groups"]["all_parameters"]["norm_actor"] == 0
        assert result["groups"]["all_parameters"]["cosine"] is None
        assert result["losses"]["normalize_advantage"] == (size > 1)
        json.dumps(result, allow_nan=False)


def test_outer_no_grad_or_inference_mode_is_preserved():
    model, batch = synthetic()
    for context in [torch.no_grad(), torch.inference_mode()]:
        with context:
            inference, enabled = torch.is_inference_mode_enabled(), torch.is_grad_enabled()
            result = gradient_diagnostics(model, batch)
            assert result["groups"]["combined_encoder"]["norm_actor"] > 0
            assert torch.is_inference_mode_enabled() == inference and torch.is_grad_enabled() == enabled


def test_raw_unclipped_size_is_used_and_illegal_action_failures_restore_state():
    model, batch = synthetic()
    batch["actions"][0, 4] = 12.0
    obs = {key: torch.from_numpy(value) for key, value in batch["observations"].items()}
    with torch.no_grad():
        batch["old_log_prob"] = model.policy.evaluate_actions(obs, torch.from_numpy(batch["actions"]))[1].numpy()
    result = gradient_diagnostics(model, batch)
    assert result["losses"]["approximate_kl"] == pytest.approx(0, abs=1e-7)
    model.policy.eval()
    state = deepcopy(model.policy.state_dict())
    batch["actions"][0, 0] = .5
    with pytest.raises(FloatingPointError, match="joint log_prob"):
        gradient_diagnostics(model, batch)
    assert not model.policy.training
    nested_equal(state, model.policy.state_dict())


@pytest.mark.parametrize("change", ["returns_shape", "actions_shape", "observation_batch", "nan", "missing", "value_clip"])
def test_invalid_inputs_fail_without_parameter_changes(change):
    model, batch = synthetic()
    before = deepcopy(model.policy.state_dict())
    if change == "returns_shape":
        batch["returns"] = np.zeros((8, 2))
    elif change == "actions_shape":
        batch["actions"] = np.zeros((8, 4), dtype=np.float32)
    elif change == "observation_batch":
        batch["observations"]["portfolio"] = batch["observations"]["portfolio"][:1]
    elif change == "nan":
        batch["returns"][0] = np.nan
    elif change == "missing":
        del batch["old_log_prob"]
    else:
        model.clip_range_vf = .1
    with pytest.raises((ValueError, FloatingPointError)):
        gradient_diagnostics(model, batch)
    nested_equal(before, model.policy.state_dict())


def test_dropout_is_rejected_and_negative_continuous_entropy_is_not_an_error():
    model, batch = synthetic()
    with torch.no_grad():
        model.policy.network.log_std.fill_(-5)
        obs = {key: torch.from_numpy(value) for key, value in batch["observations"].items()}
        batch["old_log_prob"] = model.policy.evaluate_actions(obs, torch.from_numpy(batch["actions"]))[1].numpy()
    result = gradient_diagnostics(model, batch)
    assert result["losses"]["entropy_mean"] < 0
    assert result["losses"]["conditional_size_entropy_mean"] < 0
    model.policy.network.add_module("forbidden_dropout", nn.Dropout(.1))
    with pytest.raises(ValueError, match="Dropout"):
        gradient_diagnostics(model, batch)
