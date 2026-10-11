"""Research-only quantity/priority separation with exact hybrid likelihoods."""

from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Normal

from .exploration_policy import SplitTradeNetwork
from .trade_mapping import BUY
from .trade_policy import SharedTradePolicy, StockTradeDistribution


ACTOR_VARIANTS = ("coupled", "decoupled")


def validate_actor_variant(actor_variant):
    if actor_variant not in ACTOR_VARIANTS:
        raise ValueError("actor_variant must be 'coupled' or 'decoupled'")


class StockPriorityTradeDistribution(StockTradeDistribution):
    """Directions, quantities, then BUY priorities; deterministic copies are atoms.

    The coupled arm has the legacy request probability: repeating a sampled
    quantity does not introduce another Normal density. The decoupled arm adds
    one independent priority density per BUY, including fixed-quantity BUYs.
    """

    def __init__(self, n_stocks, actor_variant="coupled"):
        super().__init__(n_stocks)
        validate_actor_variant(actor_variant)
        self.actor_variant = actor_variant

    def proba_distribution_net(self, latent_dim, log_std_init=-.7):
        direction, quantity, quantity_log_std = super().proba_distribution_net(latent_dim, log_std_init)
        return (direction, quantity, quantity_log_std, deepcopy(quantity),
                nn.Parameter(quantity_log_std.detach().clone()))

    def proba_distribution(self, logits, mean, log_std, priority_mean,
                           priority_log_std, legal_directions, lot_bounds):
        super().proba_distribution(logits, mean, log_std, legal_directions, lot_bounds)
        if priority_mean.shape != mean.shape:
            raise ValueError("Priority means must have shape [B,N]")
        if (not torch.isfinite(priority_mean).all()
                or not torch.isfinite(priority_log_std).all()):
            raise FloatingPointError("Priority parameters must be finite")
        sigma = priority_log_std.exp().expand_as(mean)
        if not torch.isfinite(sigma).all() or (sigma <= 0).any():
            raise FloatingPointError("Priority standard deviation must be positive and finite")
        self.priority_distribution = Normal(priority_mean, sigma)
        return self

    def _pack(self, quantities, deterministic):
        directions = quantities[:, :self.n_stocks]
        sizes = quantities[:, self.n_stocks:]
        if self.actor_variant == "coupled":
            priorities = sizes
        else:
            priorities = (self.priority_distribution.loc if deterministic
                          else self.priority_distribution.sample())
        priorities = torch.where(directions == BUY, priorities, torch.zeros_like(sizes))
        return torch.cat([quantities, priorities], dim=-1)

    def sample(self):
        return self._pack(super().sample(), False)

    def mode(self):
        return self._pack(super().mode(), True)

    def log_prob(self, actions):
        if actions.shape != (self.size_distribution.loc.shape[0], 3 * self.n_stocks):
            raise ValueError("Priority actions must have shape [B,3N]")
        quantities, priorities = actions[:, :2 * self.n_stocks], actions[:, 2 * self.n_stocks:]
        directions, sizes = quantities.split(self.n_stocks, dim=-1)
        probability = super().log_prob(quantities)
        buys = directions == BUY
        finite = torch.isfinite(priorities)
        if self.actor_variant == "coupled":
            canonical = priorities == torch.where(buys, sizes, torch.zeros_like(sizes))
        else:
            canonical = buys | (priorities == 0)
            safe = torch.where(finite, priorities, torch.zeros_like(priorities))
            probability = probability + torch.where(buys,
                self.priority_distribution.log_prob(safe), torch.zeros_like(safe)).sum(-1)
        return torch.where((finite & canonical).all(-1), probability,
                           torch.full_like(probability, -torch.inf))

    def conditional_priority_entropy(self):
        if self.actor_variant == "coupled":
            return torch.zeros_like(self.categorical_entropy())
        return (self.direction_distribution.probs[..., BUY]
                * self.priority_distribution.entropy()).sum(-1)

    def entropy(self):
        return super().entropy() + self.conditional_priority_entropy()

    def actions_from_params(self, logits, mean, log_std, priority_mean,
                            priority_log_std, legal_directions, lot_bounds,
                            deterministic=False):
        self.proba_distribution(logits, mean, log_std, priority_mean,
            priority_log_std, legal_directions, lot_bounds)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, logits, mean, log_std, priority_mean,
                             priority_log_std, legal_directions, lot_bounds):
        actions = self.actions_from_params(logits, mean, log_std, priority_mean,
            priority_log_std, legal_directions, lot_bounds)
        return actions, self.log_prob(actions)


class PriorityTradeNetwork(SplitTradeNetwork):
    """Pure-mean market context; every experimental arm owns the same heads."""

    def __init__(self, lookback, features, hidden_sizes=(256, 128),
                 initial_log_std=-.7, actor_output_gain=.1, *, actor_variant="coupled"):
        validate_actor_variant(actor_variant)
        super().__init__(lookback, features, hidden_sizes, initial_log_std, actor_output_gain)
        self.actor_variant = actor_variant
        self.priority_head = deepcopy(self.size_head)
        self.priority_log_std = nn.Parameter(self.log_std.detach().clone())
        self.signal_head = nn.Linear(hidden_sizes[-1], 1)
        nn.init.zeros_(self.signal_head.weight)
        nn.init.zeros_(self.signal_head.bias)

    def forward(self, obs):
        components = self.encode_components(obs)
        encoded, context = components["encoded"], components["context"]
        stock = self.stock_body(torch.cat([encoded, context.unsqueeze(1).expand(
            -1, encoded.shape[1], -1)], dim=-1))
        return (self.direction_head(stock), self.size_head(stock).squeeze(-1),
                self.value_head(context), self.log_std,
                self.priority_head(stock).squeeze(-1), self.priority_log_std)


class PriorityTradePolicy(SharedTradePolicy):
    """SB3 ActorCriticPolicy adapter retaining the sampled 3N request likelihood."""

    def __init__(self, *args, actor_variant="coupled", **kwargs):
        validate_actor_variant(actor_variant)
        self.actor_variant = actor_variant
        kwargs.setdefault("actor_output_gain", .1)
        super().__init__(*args, **kwargs)

    @property
    def parameter_count(self):
        return self.network.parameter_count

    def _build(self, lr_schedule):
        n, lookback, features = self.observation_space["stock_features"].shape
        expected = {"execution_features": (n, 3), "allocation_context": (7,),
                    "share_state": (n, 2), "trade_lot_bounds": (n, 4),
                    "legal_directions": (n, 4)}
        if self.action_space.shape != (3 * n,) or self.use_sde:
            raise ValueError("Priority policy requires packed 3N actions without SDE")
        if any(self.observation_space[name].shape != shape for name, shape in expected.items()):
            raise ValueError("Priority observation support schema mismatch")
        self.network = PriorityTradeNetwork(lookback, features, self.hidden_sizes,
            self.initial_log_std, self.actor_output_gain, actor_variant=self.actor_variant)
        self.action_dist = StockPriorityTradeDistribution(n, self.actor_variant)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                               **self.optimizer_kwargs)

    def _distribution_value(self, obs):
        frozen = (obs["valid_mask"] == 0) | (obs["execution_features"][..., 2] != 0)
        if (obs["legal_directions"][..., 1:].bool() & frozen.unsqueeze(-1)).any():
            raise ValueError("Invalid/frozen stocks must only allow HOLD")
        logits, mean, value, log_std, priority_mean, priority_log_std = self.network(obs)
        distribution = self.action_dist.proba_distribution(logits, mean, log_std,
            priority_mean, priority_log_std, obs["legal_directions"], obs["trade_lot_bounds"])
        return distribution, value

    def _get_constructor_parameters(self):
        result = super()._get_constructor_parameters()
        result.update(actor_variant=self.actor_variant)
        return result
