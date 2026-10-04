"""Conditional stock-increment PPO actor with no independent cash action."""

import torch
from torch import nn
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class StateMaskedGaussian(DiagGaussianDistribution):
    """Inactive coordinates are deterministic zero, not top-k-selected coordinates."""
    def __init__(self, n, active):
        super().__init__(n)
        self.active = active.float()

    def sample(self):
        return super().sample() * self.active

    def mode(self):
        return super().mode() * self.active

    def log_prob(self, actions):
        result = (self.distribution.log_prob(actions) * self.active).sum(dim=1)
        invalid = ((actions != 0) & (self.active == 0)).any(dim=1)
        return torch.where(invalid, torch.full_like(result, -torch.inf), result)

    def entropy(self):
        return (self.distribution.entropy() * self.active).sum(dim=1)


class SharedDeltaNetwork(nn.Module):
    def __init__(self, lookback, features, hidden_sizes=(256, 128), initial_log_std=-1.5, actor_output_gain=.01):
        super().__init__()
        h, d = hidden_sizes
        self.encoder = nn.Sequential(nn.Linear(lookback * features + 7, h), nn.Tanh(), nn.Linear(h, d), nn.Tanh())
        context = 2 * d + 8
        self.stock_head = nn.Sequential(nn.Linear(d + context, d), nn.Tanh(), nn.Linear(d, 1))
        self.value_head = nn.Sequential(nn.Linear(context, d), nn.Tanh(), nn.Linear(d, 1))
        self.log_std = nn.Parameter(torch.tensor([float(initial_log_std)]))
        nn.init.orthogonal_(self.stock_head[-1].weight, gain=actor_output_gain)
        nn.init.zeros_(self.stock_head[-1].bias)

    def forward(self, obs):
        weights, valid = obs["portfolio"][:, :-1].float(), obs["valid_mask"].float()
        flags = torch.stack([weights, valid, obs["buy_mask"].float(), obs["sell_mask"].float()], dim=-1)
        encoded = self.encoder(torch.cat([obs["stock_features"].float().flatten(start_dim=2),
                                         flags, obs["execution_features"].float()], dim=-1))
        market = (encoded * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        held = (encoded * weights.unsqueeze(-1)).sum(1)
        context = torch.cat([market, held, obs["portfolio"][:, -1:].float(),
                             obs["receivable_ratio"].float(), obs["allocation_context"].float()], dim=1)
        mean = self.stock_head(torch.cat([encoded, context.unsqueeze(1).expand(-1, encoded.shape[1], -1)], dim=-1)).squeeze(-1)
        return mean, self.value_head(context), self.log_std.expand(encoded.shape[1])


class SharedDeltaPolicy(ActorCriticPolicy):
    def __init__(self, *args, hidden_sizes=(256, 128), initial_log_std=-1.5, actor_output_gain=.01, **kwargs):
        self.hidden_sizes, self.initial_log_std = tuple(hidden_sizes), float(initial_log_std)
        self.actor_output_gain = float(actor_output_gain)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        n, lookback, features = self.observation_space["stock_features"].shape
        if self.action_space.shape != (n,) or self.use_sde:
            raise ValueError("Delta policy requires N stock actions without SDE")
        self.network = SharedDeltaNetwork(lookback, features, self.hidden_sizes,
                                         self.initial_log_std, self.actor_output_gain)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _distribution_value(self, obs):
        current, auxiliary = obs["portfolio"][:, :-1], obs["execution_features"]
        buy = (obs["buy_mask"].bool() & (obs["allocation_context"][:, 1:2] - current >= auxiliary[:, :, 0] - 1e-8)
               & (auxiliary[:, :, 0] > 0) & (obs["top_n"] > 0))
        sell = obs["sell_mask"].bool() & (auxiliary[:, :, 1] > 0)
        active = obs["valid_mask"].bool() & (auxiliary[:, :, 2] == 0) & (buy | sell)
        mean, value, std = self.network(obs)
        distribution = StateMaskedGaussian(mean.shape[1], active).proba_distribution(mean, std)
        return distribution, value

    def forward(self, obs, deterministic=False):
        distribution, value = self._distribution_value(obs)
        action = distribution.get_actions(deterministic=deterministic)
        return action, value, distribution.log_prob(action)

    def evaluate_actions(self, obs, actions):
        distribution, value = self._distribution_value(obs)
        return value, distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs):
        return self._distribution_value(obs)[0]

    def predict_values(self, obs):
        return self.network(obs)[1]

    def _get_constructor_parameters(self):
        result = super()._get_constructor_parameters()
        result.update(hidden_sizes=self.hidden_sizes, initial_log_std=self.initial_log_std,
                      actor_output_gain=self.actor_output_gain)
        return result
