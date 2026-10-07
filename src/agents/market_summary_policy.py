"""Pure-market pooling ablations with identical parameters and initial functions."""

import math

import torch
from torch import nn

from .exploration_policy import ExplorationTradePolicy, SplitTradeNetwork
from .trade_policy import StockTradeDistribution


POOLING_MODES = ("mean", "attention", "dispersion")


def _validate_pooling(pooling):
    if pooling not in POOLING_MODES:
        raise ValueError("pooling must be 'mean', 'attention', or 'dispersion'")


class MarketSummaryNetwork(SplitTradeNetwork):
    """Compare first moments, learnable stock emphasis, and latent dispersion.

    Every arm owns the same zero-initialized query and variance projection.
    At initialization both residuals are zero, preserving the Split network's
    initial Actor and Critic functions, not merely its parameter tensors.
    Unused parameters intentionally receive no gradients in the mean arm.
    """

    def __init__(self, lookback, features, hidden_sizes=(256, 128), initial_log_std=-.7,
                 actor_output_gain=.1, *, pooling="mean"):
        _validate_pooling(pooling)
        super().__init__(lookback, features, hidden_sizes, initial_log_std, actor_output_gain)
        self.pooling = pooling
        dimension = hidden_sizes[-1]
        self.summary_query = nn.Parameter(torch.zeros(dimension))
        self.variance_projection = nn.Linear(dimension, dimension, bias=False)
        nn.init.zeros_(self.variance_projection.weight)

    def encode_components(self, obs):
        components = super().encode_components(obs)
        encoded, mean = components["market_encoded"], components["market"]
        valid = obs["valid_mask"].float()
        count = valid.sum(1, keepdim=True).clamp_min(1)
        uniform = valid / count
        variance = ((encoded - mean.unsqueeze(1)).square()
                    * valid.unsqueeze(-1)).sum(1) / count
        attention_weights = uniform
        market = mean
        if self.pooling == "attention":
            logits = (encoded * self.summary_query).sum(-1) / math.sqrt(encoded.shape[-1])
            logits = logits.masked_fill(valid == 0, -torch.inf)
            maximum = torch.where(valid.any(1, keepdim=True), logits.max(1, keepdim=True).values,
                                  torch.zeros_like(count))
            exponent = torch.exp(logits - maximum) * valid
            # Zero logits use exactly the same division as uniform pooling;
            # all-invalid pools produce zero weights without a 0/0 softmax.
            attention_weights = exponent / exponent.sum(1, keepdim=True).clamp_min(1)
            # The residual is exactly zero for a zero query, unlike replacing
            # the existing reduction with a numerically different weighted sum.
            market = mean + ((attention_weights - uniform).unsqueeze(-1) * encoded).sum(1)
        elif self.pooling == "dispersion":
            market = mean + self.variance_projection(variance)
        components.update(market=market, market_mean=mean, market_variance=variance,
                          market_attention_weights=attention_weights)
        components["context"] = torch.cat([market, components["held"],
            obs["portfolio"][:, -1:].float(), obs["receivable_ratio"].float(),
            obs["allocation_context"].float()], dim=1)
        return components


class MarketSummaryPolicy(ExplorationTradePolicy):
    """Research-only summary pooling; trade likelihoods and execution stay fixed."""

    def __init__(self, *args, pooling="mean", architecture="split", **kwargs):
        _validate_pooling(pooling)
        if architecture != "split":
            raise ValueError("Market summary policy requires architecture='split'")
        self.pooling = pooling
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
        self.network = MarketSummaryNetwork(lookback, features, self.hidden_sizes,
            self.initial_log_std, self.actor_output_gain, pooling=self.pooling)
        self.action_dist = StockTradeDistribution(n)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        result = super()._get_constructor_parameters()
        result.update(pooling=self.pooling)
        return result
