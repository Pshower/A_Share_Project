"""State-masked categorical trade directions with conditional Normal lot sizing."""

import torch
from torch import nn
from torch.distributions import Categorical, Normal
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy

from .trade_mapping import HOLD, BUY, REDUCE, EXIT


class StockTradeDistribution(Distribution):
    """Pack N direction codes followed by N latent sizes; inactive sizes are atoms at zero."""

    def __init__(self, n_stocks):
        super().__init__()
        if type(n_stocks) is not int or n_stocks < 1:
            raise ValueError("A trade distribution requires a positive stock count")
        self.n_stocks = n_stocks

    def proba_distribution_net(self, latent_dim, log_std_init=-.7):
        return (nn.Linear(latent_dim, self.n_stocks * 4), nn.Linear(latent_dim, self.n_stocks),
                nn.Parameter(torch.full((1,), float(log_std_init))))

    def proba_distribution(self, logits, mean, log_std, legal_directions, lot_bounds):
        expected = (*mean.shape, 4)
        if mean.ndim != 2 or mean.shape[1] != self.n_stocks or logits.shape != expected:
            raise ValueError("Trade logits/means must have shapes [B,N,4]/[B,N]")
        if legal_directions.shape != expected or lot_bounds.shape != expected:
            raise ValueError("Trade support fields must have shape [B,N,4]")
        if (not torch.isfinite(logits).all() or not torch.isfinite(mean).all()
                or not torch.isfinite(log_std).all() or not torch.isfinite(legal_directions).all()
                or not torch.isfinite(lot_bounds).all()):
            raise FloatingPointError("Trade distribution parameters/support must be finite")
        if not ((legal_directions == 0) | (legal_directions == 1)).all():
            raise ValueError("Legal directions must be binary")
        if (lot_bounds < 0).any() or not (lot_bounds == lot_bounds.round()).all():
            raise ValueError("Lot bounds must be nonnegative integer counts")
        legal = legal_directions.bool()
        if not legal[..., HOLD].all():
            raise ValueError("HOLD must always be legal")
        for direction, lo, hi in [(BUY, 0, 1), (REDUCE, 2, 3)]:
            invalid = legal[..., direction] & ((lot_bounds[..., lo] < 1) | (lot_bounds[..., hi] < lot_bounds[..., lo]))
            if invalid.any():
                raise ValueError("Legal quantity directions require nonempty positive lot bounds")
        sigma = log_std.exp().expand_as(mean)
        if not torch.isfinite(sigma).all() or (sigma <= 0).any():
            raise FloatingPointError("Size standard deviation must be finite and positive")
        self.direction_distribution = Categorical(logits=logits.masked_fill(~legal, -torch.inf))
        self.size_distribution = Normal(mean, sigma)
        self.legal_directions = legal
        self.size_supports = torch.zeros_like(legal)
        self.size_supports[..., BUY] = legal[..., BUY]
        self.size_supports[..., REDUCE] = legal[..., REDUCE] & (lot_bounds[..., 3] > lot_bounds[..., 2])
        return self

    def _size_active(self, directions):
        return self.size_supports.gather(-1, directions.long().unsqueeze(-1)).squeeze(-1)

    def sample(self):
        directions = self.direction_distribution.sample()
        sizes = torch.where(self._size_active(directions), self.size_distribution.sample(),
                            torch.zeros_like(self.size_distribution.loc))
        return torch.cat([directions.to(sizes.dtype), sizes], dim=-1)

    def mode(self):
        directions = self.direction_distribution.probs.argmax(dim=-1)
        sizes = torch.where(self._size_active(directions), self.size_distribution.loc,
                            torch.zeros_like(self.size_distribution.loc))
        return torch.cat([directions.to(sizes.dtype), sizes], dim=-1)

    def log_prob(self, actions):
        if actions.shape != (self.size_distribution.loc.shape[0], 2 * self.n_stocks):
            raise ValueError("Trade actions must have shape [B,2N]")
        directions, sizes = actions.split(self.n_stocks, dim=-1)
        finite = torch.isfinite(directions) & torch.isfinite(sizes)
        valid_codes = finite & (directions == directions.round()) & (directions >= HOLD) & (directions <= EXIT)
        safe_codes = torch.where(torch.isfinite(directions), directions, torch.zeros_like(directions)).clamp(HOLD, EXIT).long()
        active = self._size_active(safe_codes)
        canonical = active | (sizes == 0)
        safe_sizes = torch.where(torch.isfinite(sizes), sizes, torch.zeros_like(sizes))
        # The probability belongs to the sampled request, even when execution later clips it.
        direction_log_prob = self.direction_distribution.log_prob(safe_codes)
        size_log_prob = torch.where(active, self.size_distribution.log_prob(safe_sizes), torch.zeros_like(safe_sizes))
        result = (direction_log_prob + size_log_prob).sum(dim=-1)
        invalid = ~(valid_codes & canonical).all(dim=-1)
        return torch.where(invalid, torch.full_like(result, -torch.inf), result)

    def categorical_entropy(self):
        return self.direction_distribution.entropy().sum(dim=-1)

    def conditional_size_entropy(self):
        probability = (self.direction_distribution.probs * self.size_supports).sum(dim=-1)
        return (probability * self.size_distribution.entropy()).sum(dim=-1)

    def entropy(self):
        return self.categorical_entropy() + self.conditional_size_entropy()

    def actions_from_params(self, logits, mean, log_std, legal_directions, lot_bounds, deterministic=False):
        self.proba_distribution(logits, mean, log_std, legal_directions, lot_bounds)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, logits, mean, log_std, legal_directions, lot_bounds):
        actions = self.actions_from_params(logits, mean, log_std, legal_directions, lot_bounds)
        return actions, self.log_prob(actions)


class SharedTradeNetwork(nn.Module):
    """Shared stock encoder, direction/size heads and an unchanged net-return value target."""

    def __init__(self, lookback, features, hidden_sizes=(256, 128), initial_log_std=-.7, actor_output_gain=.01):
        super().__init__()
        if (len(hidden_sizes) != 2 or any(type(value) is not int or value < 1 for value in hidden_sizes)
                or not 0 < actor_output_gain <= 1):
            raise ValueError("Invalid trade network widths or actor output gain")
        h, d = hidden_sizes
        self.encoder = nn.Sequential(nn.Linear(lookback * features + 11, h), nn.Tanh(), nn.Linear(h, d), nn.Tanh())
        context = 2 * d + 9
        self.stock_body = nn.Sequential(nn.Linear(d + context, d), nn.Tanh())
        self.direction_head = nn.Linear(d, 4)
        self.size_head = nn.Linear(d, 1)
        self.value_head = nn.Sequential(nn.Linear(context, d), nn.Tanh(), nn.Linear(d, 1))
        self.log_std = nn.Parameter(torch.tensor([float(initial_log_std)]))
        for head in (self.direction_head, self.size_head):
            nn.init.orthogonal_(head.weight, gain=actor_output_gain)
            nn.init.zeros_(head.bias)

    def forward(self, obs):
        weights, valid = obs["portfolio"][:, :-1].float(), obs["valid_mask"].float()
        flags = torch.stack([weights, valid, obs["buy_mask"].float(), obs["sell_mask"].float()], dim=-1)
        execution = obs["execution_features"].float()
        notionals = obs["trade_lot_bounds"].float() * execution[..., :1]
        encoded = self.encoder(torch.cat([obs["stock_features"].float().flatten(start_dim=2), flags, execution, notionals], dim=-1))
        market = (encoded * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        held = (encoded * weights.unsqueeze(-1)).sum(1)
        context = torch.cat([market, held, obs["portfolio"][:, -1:].float(),
                             obs["receivable_ratio"].float(), obs["allocation_context"].float()], dim=1)
        stock = self.stock_body(torch.cat([encoded, context.unsqueeze(1).expand(-1, encoded.shape[1], -1)], dim=-1))
        return self.direction_head(stock), self.size_head(stock).squeeze(-1), self.value_head(context), self.log_std


class SharedTradePolicy(ActorCriticPolicy):
    def __init__(self, *args, hidden_sizes=(256, 128), initial_log_std=-.7, actor_output_gain=.01, **kwargs):
        self.hidden_sizes = tuple(hidden_sizes)
        self.initial_log_std, self.actor_output_gain = float(initial_log_std), float(actor_output_gain)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        n, lookback, features = self.observation_space["stock_features"].shape
        expected = {"execution_features": (n, 3), "allocation_context": (7,), "share_state": (n, 2),
                    "trade_lot_bounds": (n, 4), "legal_directions": (n, 4)}
        if self.action_space.shape != (2 * n,) or self.use_sde:
            raise ValueError("Trade policy requires packed 2N actions without SDE")
        if any(self.observation_space[name].shape != shape for name, shape in expected.items()):
            raise ValueError("Trade policy observation support schema mismatch")
        self.network = SharedTradeNetwork(lookback, features, self.hidden_sizes,
                                         self.initial_log_std, self.actor_output_gain)
        self.action_dist = StockTradeDistribution(n)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _distribution_value(self, obs):
        frozen = (obs["valid_mask"] == 0) | (obs["execution_features"][..., 2] != 0)
        if (obs["legal_directions"][..., 1:].bool() & frozen.unsqueeze(-1)).any():
            raise ValueError("Invalid/frozen stocks must only allow HOLD")
        logits, mean, value, log_std = self.network(obs)
        distribution = self.action_dist.proba_distribution(logits, mean, log_std,
            obs["legal_directions"], obs["trade_lot_bounds"])
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
        return self.network(obs)[2]

    def _get_constructor_parameters(self):
        result = super()._get_constructor_parameters()
        result.update(hidden_sizes=self.hidden_sizes, initial_log_std=self.initial_log_std,
                      actor_output_gain=self.actor_output_gain)
        return result
