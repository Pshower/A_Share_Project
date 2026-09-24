"""Size-independent stock encoder and heads, integrated with SB3 PPO."""

import torch
from torch import nn
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class SharedStockNetwork(nn.Module):
    def __init__(self, lookback, n_features, hidden_sizes=(128, 64)):
        super().__init__()
        if len(hidden_sizes) != 2 or any(not isinstance(v, int) or v < 1 for v in hidden_sizes):
            raise ValueError("hidden_sizes must contain two positive integers")
        h, d = hidden_sizes
        self.encoder = nn.Sequential(nn.Linear(lookback * n_features + 4, h), nn.Tanh(),
                                     nn.Linear(h, d), nn.Tanh())
        # Valid-market mean and holdings-weighted sum preserve frozen holdings.
        context_size = d * 2 + 2
        self.stock_head = nn.Sequential(nn.Linear(d + context_size, d), nn.Tanh(), nn.Linear(d, 1))
        self.cash_head = nn.Sequential(nn.Linear(context_size, d), nn.Tanh(), nn.Linear(d, 1))
        self.value_head = nn.Sequential(nn.Linear(context_size, d), nn.Tanh(), nn.Linear(d, 1))
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, obs):
        features = obs["stock_features"].float().flatten(start_dim=2)
        weights = obs["portfolio"][:, :-1].float()
        valid = obs["valid_mask"].float()
        flags = torch.stack([weights, valid, obs["buy_mask"].float(), obs["sell_mask"].float()], dim=-1)
        encoded = self.encoder(torch.cat([features, flags], dim=-1))
        market = (encoded * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        held = (encoded * weights.unsqueeze(-1)).sum(1)
        context = torch.cat([market, held, obs["portfolio"][:, -1:].float(),
                             obs["receivable_ratio"].float()], dim=-1)
        stock = self.stock_head(torch.cat([encoded, context.unsqueeze(1).expand(-1, encoded.shape[1], -1)], dim=-1)).squeeze(-1)
        mean = torch.cat([stock, self.cash_head(context)], dim=1)
        log_std = torch.cat([self.log_std[:1].expand(encoded.shape[1]), self.log_std[1:]])
        return mean, self.value_head(context), log_std


class SharedStockPolicy(ActorCriticPolicy):
    def __init__(self, *args, hidden_sizes=(128, 64), **kwargs):
        self.hidden_sizes = tuple(hidden_sizes)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        n, lookback, features = self.observation_space["stock_features"].shape
        if self.action_space.shape != (n + 1,) or self.use_sde:
            raise ValueError("SharedStockPolicy requires N+1 Gaussian actions without SDE")
        self.network = SharedStockNetwork(lookback, features, self.hidden_sizes)
        self.action_dist = DiagGaussianDistribution(n + 1)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _distribution_value(self, obs):
        mean, value, log_std = self.network(obs)
        return self.action_dist.proba_distribution(mean, log_std), value

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
        data = super()._get_constructor_parameters()
        data["hidden_sizes"] = self.hidden_sizes
        return data
