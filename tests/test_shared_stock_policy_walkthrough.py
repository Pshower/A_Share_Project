"""Explain the SharedStockPolicy actor/critic adapter with synthetic observations.

Run from the repository root:
    python -m tests.test_shared_stock_policy_walkthrough
    python -m pytest tests/test_shared_stock_policy_walkthrough.py -q

Uses real policy/distribution APIs. No environment rollout or optimizer step.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import torch

from src.agents.action_mapping import map_action
from src.agents.ppo_policy import SharedStockPolicy
from tests.test_shared_stock_network_walkthrough import FEATURES, HIDDEN_SIZES, make_observation, show


def make_policy(with_prior=False, bound=8.0):
    obs = make_observation()
    n, lookback, features = obs["stock_features"].shape[1:]
    observation_space = gym.spaces.Dict({
        "stock_features": gym.spaces.Box(-np.inf, np.inf, (n, lookback, features), np.float32),
        "portfolio": gym.spaces.Box(0, 1, (n + 1,), np.float32),
        "valid_mask": gym.spaces.MultiBinary(n),
        "buy_mask": gym.spaces.MultiBinary(n),
        "sell_mask": gym.spaces.MultiBinary(n),
        "receivable_ratio": gym.spaces.Box(0, 1, (1,), np.float32),
    })
    action_space = gym.spaces.Box(-bound, bound, (n + 1,), np.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        policy = SharedStockPolicy(
            observation_space,
            action_space,
            lr_schedule=lambda _: 3e-4,
            hidden_sizes=HIDDEN_SIZES,
            initial_log_std=-3.0,
            prior_feature_index=FEATURES.index("momentum_20") if with_prior else None,
            prior_strength=4.0,
            residual_scale=0.25,
        )
    policy.set_training_mode(False)
    return policy


def sample_action(policy, obs):
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(123)
        return policy(obs, deterministic=False)


def unbatched_numpy(obs):
    return {name: tensor[0].detach().cpu().numpy() for name, tensor in obs.items()}


def test_optimizer_contains_actor_critic_encoder_and_exploration_parameters():
    policy = make_policy()
    registered = {id(p) for p in policy.parameters()}
    optimized = {id(p) for group in policy.optimizer.param_groups for p in group["params"]}
    assert registered == optimized
    assert sum(p.numel() for p in policy.parameters()) == 289
    assert policy.optimizer.param_groups[0]["lr"] == 3e-4
    assert len(policy.optimizer.state) == 0


def test_deterministic_forward_returns_mean_value_and_joint_log_density():
    policy, obs = make_policy(), make_observation()
    with torch.no_grad():
        mean, expected_value, log_std = policy.network(obs)
        action, value, log_prob = policy(obs, deterministic=True)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        expected_log_prob = distribution.log_prob(action).sum(dim=-1)
        predicted_value = policy.predict_values(obs)
    torch.testing.assert_close(action, mean)
    torch.testing.assert_close(value, expected_value)
    torch.testing.assert_close(value, predicted_value)
    torch.testing.assert_close(log_prob, expected_log_prob)
    assert action.shape == (1, 5) and value.shape == (1, 1) and log_prob.shape == (1,)


def test_sampled_action_revaluation_agrees_when_policy_is_unchanged():
    policy, obs = make_policy(), make_observation()
    action, value, old_log_prob = sample_action(policy, obs)
    repeated = sample_action(policy, obs)[0]
    with torch.no_grad():
        mean, _, _ = policy.network(obs)
        reevaluated_value, log_prob, entropy = policy.evaluate_actions(obs, action)
    assert not torch.allclose(action, mean)
    torch.testing.assert_close(action, repeated)
    torch.testing.assert_close(value, reevaluated_value)
    torch.testing.assert_close(log_prob, old_log_prob)
    torch.testing.assert_close((log_prob - old_log_prob).exp(), torch.ones(1))
    assert entropy.shape == (1,) and torch.isfinite(entropy).all()


def test_actor_bias_perturbation_changes_probability_ratio_but_not_critic():
    policy, obs = make_policy(), make_observation()
    action, value, old_log_prob = sample_action(policy, obs)
    before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    bias = policy.network.stock_head[-1].bias
    try:
        with torch.no_grad():
            bias.add_(0.01)
            changed_value, new_log_prob, _ = policy.evaluate_actions(obs, action)
        torch.testing.assert_close(changed_value, value)
        assert not torch.allclose((new_log_prob - old_log_prob).exp(), torch.ones(1))
    finally:
        with torch.no_grad():
            bias.copy_(before["network.stock_head.2.bias"])
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter, before[name])
    assert len(policy.optimizer.state) == 0


def test_action_mapping_enforces_eligibility_and_caps_after_policy_forward():
    policy, obs = make_policy(with_prior=True), make_observation()
    with torch.no_grad():
        action, _, _ = policy(obs, deterministic=True)
    mapped = map_action(action[0].numpy(), unbatched_numpy(obs), max_weight=0.4, top_k=2)
    torch.testing.assert_close(action, torch.tensor([[-4., 0., 4., 0., 0.]]))
    assert mapped["weights"].shape == (4,)
    assert mapped["weights"][1] == mapped["weights"][3] == 0
    assert 0 < mapped["weights"][0] < mapped["weights"][2] <= 0.4
    assert np.isclose(mapped["weights"].astype(float).sum() + mapped["cash_weight"], 1)


def test_public_predict_clips_to_box_while_forward_keeps_raw_latent_scores():
    policy, obs = make_policy(bound=0.05), make_observation()
    with torch.no_grad():
        raw_action, _, _ = policy(obs, deterministic=True)
    bounded_action, _ = policy.predict(unbatched_numpy(obs), deterministic=True)
    assert (raw_action.abs() > 0.05).any()
    np.testing.assert_allclose(bounded_action, np.clip(raw_action[0].numpy(), -0.05, 0.05))


def test_numpy_conversion_and_batch_forward_keep_independent_observations():
    policy, obs = make_policy(), make_observation()
    converted, vectorized = policy.obs_to_tensor(unbatched_numpy(obs))
    assert not vectorized
    for name in obs:
        torch.testing.assert_close(converted[name], obs[name])
    batched = {name: tensor.repeat(2, *([1] * (tensor.ndim - 1))) for name, tensor in obs.items()}
    batched["portfolio"][1, :3] = torch.tensor([0.35, 0., 0.15])
    with torch.no_grad():
        actions, values, log_probs = policy(batched, deterministic=True)
        for i in range(2):
            single = {name: tensor[i:i + 1] for name, tensor in batched.items()}
            one = policy(single, deterministic=True)
            for together, individual in zip((actions, values, log_probs), one):
                torch.testing.assert_close(together[i:i + 1], individual)
    assert actions.shape == (2, 5) and values.shape == (2, 1) and log_probs.shape == (2,)


def test_policy_save_load_preserves_network_weights_and_prior_options():
    policy, obs = make_policy(with_prior=True), make_observation()
    with TemporaryDirectory(prefix="stock_policy_walkthrough_") as folder:
        path = Path(folder) / "policy.pt"
        policy.save(path)
        restored = SharedStockPolicy.load(path, device="cpu")
        with torch.no_grad():
            original = policy(obs, deterministic=True)
            reloaded = restored(obs, deterministic=True)
        for left, right in zip(original, reloaded):
            torch.testing.assert_close(left, right)
        assert restored.hidden_sizes == policy.hidden_sizes
        assert restored.prior_options == policy.prior_options


def main():
    policy, obs = make_policy(), make_observation()
    print("SharedStockPolicy walkthrough: synthetic observations, randomly initialized toy policy")
    print("CPU only; no PPO rollout, backward call or optimizer step.")
    print("\n1. _build(): connect observation/action spaces, network, Gaussian and optimizer")
    print("stock_features space:", policy.observation_space["stock_features"].shape)
    print("latent action space:", policy.action_space.shape, "[A, B, C, D, cash]")
    print("optimizer:", type(policy.optimizer).__name__, "lr=", policy.optimizer.param_groups[0]["lr"])
    print("parameters:", sum(p.numel() for p in policy.parameters()))
    converted, vectorized = policy.obs_to_tensor(unbatched_numpy(obs))
    print("Unbatched NumPy -> batched tensors; vectorized input:", vectorized)
    print("stock_features tensor:", tuple(converted["stock_features"].shape))

    print("\n2. get_distribution(): network output -> independent Gaussian coordinates")
    with torch.no_grad():
        mean, value, log_std = policy.network(obs)
        distribution = policy.get_distribution(obs)
    show("Gaussian mean", distribution.distribution.mean)
    show("Gaussian std = exp(log_std)", distribution.distribution.stddev)
    show("critic value", value)

    print("\n3. forward(deterministic=True): action equals the Gaussian mean")
    with torch.no_grad():
        deterministic, _, log_prob_at_mean = policy(obs, deterministic=True)
    show("deterministic latent action", deterministic)
    show("joint log density at the mean", log_prob_at_mean)
    print("log_prob is a continuous probability density in log space; it may be positive.")

    print("\n4. forward(deterministic=False): sample around the mean for exploration")
    action, sampled_value, old_log_prob = sample_action(policy, obs)
    show("sampled latent action", action)
    show("sample minus mean", action - mean)
    show("stored old_log_prob", old_log_prob)

    print("\n5. evaluate_actions(): reevaluate the SAME stored latent action")
    with torch.no_grad():
        reevaluated_value, new_log_prob, entropy = policy.evaluate_actions(obs, action)
        per_coordinate = policy.get_distribution(obs).distribution.log_prob(action)
    show("per-coordinate log densities", per_coordinate)
    show("new_log_prob: sum over all five coordinates", new_log_prob)
    show("PPO ratio = exp(new_log_prob - old_log_prob)", (new_log_prob - old_log_prob).exp())
    show("entropy: sum of Gaussian differential entropies", entropy)
    print("The ratio is 1 before parameter changes. Narrow continuous Gaussians can have negative entropy.")
    print("predict_values() and evaluate_actions() use the same critic.")
    torch.testing.assert_close(sampled_value, reevaluated_value)

    print("\n6. Controlled actor perturbation: stock-head bias +0.01, then restore")
    bias = policy.network.stock_head[-1].bias
    original_bias = bias.detach().clone()
    try:
        with torch.no_grad():
            bias.add_(0.01)
            changed_value, changed_log_prob, _ = policy.evaluate_actions(obs, action)
        show("changed PPO probability ratio", (changed_log_prob - old_log_prob).exp())
        print("Critic value unchanged:", torch.allclose(changed_value, sampled_value))
    finally:
        with torch.no_grad():
            bias.copy_(original_bias)
    print("Real PPO uses this ratio with advantages, value loss and entropy loss to update parameters.")

    print("\n7. Latent actions -> target weights through the separate map_action()")
    with torch.no_grad():
        prior_action, _, _ = make_policy(with_prior=True)(obs, deterministic=True)
    mapped = map_action(prior_action[0].numpy(), unbatched_numpy(obs), max_weight=0.4, top_k=2)
    show("prior latent action", prior_action)
    print("eligible A/B/C/D:", mapped["eligible"])
    print("target weights A/B/C/D:", mapped["weights"].round(6))
    print("target cash:", round(mapped["cash_weight"], 6))
    print("PPO log probabilities refer to latent actions, not mapped portfolio weights.")
    print("Live account/execution checks, including invalid held D, are outside this policy demo.")

    print("\n8. public predict() also clips latent actions to the declared Box bounds")
    tight_policy = make_policy(bound=0.05)
    with torch.no_grad():
        raw, _, _ = tight_policy(obs, deterministic=True)
    bounded, _ = tight_policy.predict(unbatched_numpy(obs), deterministic=True)
    show("raw forward action with bound=0.05", raw)
    print("public predict action:", bounded.round(4))
    print("SB3 rollouts store raw actions/log densities and send clipped actions to the wrapper.")

    print("\n9. Verify interfaces and distribution behavior")
    checks = (
        test_optimizer_contains_actor_critic_encoder_and_exploration_parameters,
        test_deterministic_forward_returns_mean_value_and_joint_log_density,
        test_sampled_action_revaluation_agrees_when_policy_is_unchanged,
        test_actor_bias_perturbation_changes_probability_ratio_but_not_critic,
        test_action_mapping_enforces_eligibility_and_caps_after_policy_forward,
        test_public_predict_clips_to_box_while_forward_keeps_raw_latent_scores,
        test_numpy_conversion_and_batch_forward_keep_independent_observations,
        test_policy_save_load_preserves_network_weights_and_prior_options,
    )
    for check in checks:
        check()
        print("PASS " + check.__name__)
    print(f"Completed: {len(checks)} checks passed; optimizer state entries={len(policy.optimizer.state)}.")


if __name__ == "__main__":
    main()
