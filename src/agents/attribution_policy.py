"""Same-parameter attribution of market routing and Critic encoder gradients."""

import torch

from .exploration_policy import ExplorationTradePolicy, SplitTradeNetwork
from .trade_policy import StockTradeDistribution


def _validate_flags(market_context, critic_features):
    if market_context not in ("pure", "fused"):
        raise ValueError("market_context must be 'pure' or 'fused'")
    if critic_features not in ("shared", "detached"):
        raise ValueError("critic_features must be 'shared' or 'detached'")


class AttributionTradeNetwork(SplitTradeNetwork):
    """Keep every Split parameter while varying two forward/gradient routes.

    The Critic flag changes gradients only, not its initial predictions. The
    market flag changes the function even with identical initial parameters.
    """

    def __init__(self, lookback, features, hidden_sizes=(256, 128), initial_log_std=-.7,
                 actor_output_gain=.1, *, market_context="pure", critic_features="shared"):
        _validate_flags(market_context, critic_features)
        super().__init__(lookback, features, hidden_sizes, initial_log_std, actor_output_gain)
        self.market_context, self.critic_features = market_context, critic_features

    def encode_components(self, obs):
        components = super().encode_components(obs)
        if self.market_context == "fused":
            valid = obs["valid_mask"].float()
            market = ((components["encoded"] * valid.unsqueeze(-1)).sum(1)
                      / valid.sum(1, keepdim=True).clamp_min(1))
            components["market"] = market
            components["context"] = torch.cat([market, components["held"],
                obs["portfolio"][:, -1:].float(), obs["receivable_ratio"].float(),
                obs["allocation_context"].float()], dim=1)
        return components

    def forward(self, obs):
        components = self.encode_components(obs)
        encoded, context = components["encoded"], components["context"]
        stock = self.stock_body(torch.cat([encoded, context.unsqueeze(1).expand(
            -1, encoded.shape[1], -1)], dim=-1))
        # Detach only the value branch; the Actor still learns through both encoders.
        value_context = context.detach() if self.critic_features == "detached" else context
        return (self.direction_head(stock), self.size_head(stock).squeeze(-1),
                self.value_head(value_context), self.log_std)


class AttributionTradePolicy(ExplorationTradePolicy):
    """Research-only 2x2 routes with identical masked action likelihoods."""

    def __init__(self, *args, market_context="pure", critic_features="shared",
                 architecture="split", **kwargs):
        _validate_flags(market_context, critic_features)
        if architecture != "split":
            raise ValueError("Attribution policy requires architecture='split'")
        self.market_context, self.critic_features = market_context, critic_features
        kwargs.setdefault("actor_output_gain", .1)
        super().__init__(*args, architecture=architecture, **kwargs)

    def _build(self, lr_schedule):
        n, lookback, features = self.observation_space["stock_features"].shape
        expected = {"execution_features": (n, 3), "allocation_context": (7,),
                    "share_state": (n, 2), "trade_lot_bounds": (n, 4),
                    "legal_directions": (n, 4)}
        if self.action_space.shape != (2 * n,) or self.use_sde:
            raise ValueError("Trade policy requires packed 2N actions without SDE")
        if any(self.observation_space[name].shape != shape for name, shape in expected.items()):
            raise ValueError("Trade policy observation support schema mismatch")
        self.network = AttributionTradeNetwork(lookback, features, self.hidden_sizes,
            self.initial_log_std, self.actor_output_gain, market_context=self.market_context,
            critic_features=self.critic_features)
        self.action_dist = StockTradeDistribution(n)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        result = super()._get_constructor_parameters()
        result.update(market_context=self.market_context, critic_features=self.critic_features)
        return result
