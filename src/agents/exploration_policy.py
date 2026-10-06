"""Research-only trade policies with unchanged action and reward semantics."""

import torch
from torch import nn

from .trade_policy import SharedTradeNetwork, SharedTradePolicy, StockTradeDistribution


class SplitTradeNetwork(nn.Module):
    """Separate pure market signals from account-dependent stock representations.

    All stocks share the two encoders and heads. The market context cannot depend
    on holdings, while the held context includes invalid stocks that remain held.
    A longer lookback changes only the per-stock flattened market input width.
    """

    def __init__(self, lookback, features, hidden_sizes=(256, 128), initial_log_std=-.7,
                 actor_output_gain=.1):
        super().__init__()
        if (type(lookback) is not int or lookback < 1 or type(features) is not int or features < 1
                or len(hidden_sizes) != 2
                or any(type(value) is not int or value < 1 for value in hidden_sizes)
                or not 0 < actor_output_gain <= 1):
            raise ValueError("Invalid split network dimensions or actor output gain")
        h, d = hidden_sizes
        self.market_encoder = nn.Sequential(nn.Linear(lookback * features, h), nn.Tanh(),
                                            nn.Linear(h, d), nn.Tanh())
        self.account_encoder = nn.Sequential(nn.Linear(11, d), nn.Tanh())
        context = 2 * d + 9
        self.stock_body = nn.Sequential(nn.Linear(d + context, d), nn.Tanh())
        self.direction_head = nn.Linear(d, 4)
        self.size_head = nn.Linear(d, 1)
        self.value_head = nn.Sequential(nn.Linear(context, d), nn.Tanh(), nn.Linear(d, 1))
        self.log_std = nn.Parameter(torch.tensor([float(initial_log_std)]))
        for head in (self.direction_head, self.size_head):
            nn.init.orthogonal_(head.weight, gain=actor_output_gain)
            nn.init.zeros_(head.bias)

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def encode_components(self, obs):
        """Return pure-market and account-aware tensors without changing state."""
        weights, valid = obs["portfolio"][:, :-1].float(), obs["valid_mask"].float()
        flags = torch.stack([weights, valid, obs["buy_mask"].float(),
                             obs["sell_mask"].float()], dim=-1)
        execution = obs["execution_features"].float()
        notionals = obs["trade_lot_bounds"].float() * execution[..., :1]
        market_encoded = self.market_encoder(obs["stock_features"].float().flatten(start_dim=2))
        account_encoded = self.account_encoder(torch.cat([flags, execution, notionals], dim=-1))
        encoded = torch.tanh(market_encoded + account_encoded)
        market = ((market_encoded * valid.unsqueeze(-1)).sum(1)
                  / valid.sum(1, keepdim=True).clamp_min(1))
        held = (encoded * weights.unsqueeze(-1)).sum(1)
        context = torch.cat([market, held, obs["portfolio"][:, -1:].float(),
                             obs["receivable_ratio"].float(), obs["allocation_context"].float()], dim=1)
        return {"market_encoded": market_encoded, "account_encoded": account_encoded,
                "encoded": encoded, "market": market, "held": held, "context": context}

    def forward(self, obs):
        components = self.encode_components(obs)
        encoded, context = components["encoded"], components["context"]
        stock = self.stock_body(torch.cat([encoded, context.unsqueeze(1).expand(
            -1, encoded.shape[1], -1)], dim=-1))
        return self.direction_head(stock), self.size_head(stock).squeeze(-1), self.value_head(context), self.log_std


class ExplorationTradePolicy(SharedTradePolicy):
    """Isolated baseline/split ablation using the production hybrid distribution."""

    def __init__(self, *args, architecture="baseline", **kwargs):
        if architecture not in {"baseline", "split"}:
            raise ValueError("architecture must be 'baseline' or 'split'")
        self.architecture = architecture
        super().__init__(*args, **kwargs)

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.network.parameters())

    def _build(self, lr_schedule):
        n, lookback, features = self.observation_space["stock_features"].shape
        expected = {"execution_features": (n, 3), "allocation_context": (7,),
                    "share_state": (n, 2), "trade_lot_bounds": (n, 4),
                    "legal_directions": (n, 4)}
        if self.action_space.shape != (2 * n,) or self.use_sde:
            raise ValueError("Trade policy requires packed 2N actions without SDE")
        if any(self.observation_space[name].shape != shape for name, shape in expected.items()):
            raise ValueError("Trade policy observation support schema mismatch")
        network_class = SharedTradeNetwork if self.architecture == "baseline" else SplitTradeNetwork
        self.network = network_class(lookback, features, self.hidden_sizes,
                                     self.initial_log_std, self.actor_output_gain)
        self.action_dist = StockTradeDistribution(n)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        result = super()._get_constructor_parameters()
        result.update(architecture=self.architecture)
        return result
