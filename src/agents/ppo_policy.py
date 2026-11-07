"""Size-independent stock encoder and heads, integrated with SB3 PPO."""

import torch
from torch import nn
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class SharedStockNetwork(nn.Module):
    """Encode each stock with shared weights, then score it in portfolio context.

    ``lookback`` (L) and ``n_features`` (F) set the per-stock history size.
    ``hidden_sizes`` is (h, d): encoder hidden width and stock embedding width.
    The number of stocks (N) is taken from the observation, not the constructor.
    ``initial_log_std`` initializes two learnable exploration scales: one shared
    by all stocks and one for cash. An optional ``prior_feature_index`` selects
    the latest feature used for ranking; ``prior_strength`` scales that ranking
    and ``residual_scale`` bounds the actor's learned adjustment to the prior.

    Outputs parameterize allocation actions. ``map_action`` constructs target
    weights, and the broker determines the holdings reached through execution.
    """

    def __init__(self, lookback, n_features, hidden_sizes=(128, 64), initial_log_std=0.0,
                 prior_feature_index=None, prior_strength=4.0, residual_scale=0.25,
                 full_invest=False):
        super().__init__()
        if len(hidden_sizes) != 2 or any(not isinstance(v, int) or v < 1 for v in hidden_sizes):
            raise ValueError("hidden_sizes must contain two positive integers")
        h, d = hidden_sizes
        self.prior_feature_index = prior_feature_index
        self.prior_strength, self.residual_scale = prior_strength, residual_scale
        self.full_invest = bool(full_invest)
        if prior_feature_index is not None and (type(prior_feature_index) is not int or not 0 <= prior_feature_index < n_features):
            raise ValueError("Invalid prior feature index")
        # Each stock supplies L*F features plus [weight, valid, buy, sell].
        # Linear operates on the last axis, sharing this encoder across stocks.
        self.encoder = nn.Sequential(nn.Linear(lookback * n_features + 4, h), nn.Tanh(),
                                     nn.Linear(h, d), nn.Tanh())
        # Context combines two d-dimensional summaries, cash and receivables.
        # Actor heads score stock/cash actions; the critic estimates future reward.
        context_size = d * 2 + 2
        self.stock_head = nn.Sequential(nn.Linear(d + context_size, d), nn.Tanh(), nn.Linear(d, 1))
        # A fully invested book has no cash action, so the head would only collect dead
        # parameters. It is built only when the mapping can actually use it.
        if not self.full_invest:
            self.cash_head = nn.Sequential(nn.Linear(context_size, d), nn.Tanh(), nn.Linear(d, 1))
        self.value_head = nn.Sequential(nn.Linear(context_size, d), nn.Tanh(), nn.Linear(d, 1))
        self.log_std = nn.Parameter(torch.full((2,), float(initial_log_std)))
        if prior_feature_index is not None:
            # A fresh actor starts at the pure prior; training learns the residual.
            # The critic retains its normal initialization.
            for head in [self.stock_head] + ([] if self.full_invest else [self.cash_head]):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def forward(self, obs):
        """Process a batch of observations; B denotes batch size.

        Inputs: stock_features[B,N,L,F], portfolio[B,N+1] (cash last),
        valid_mask/buy_mask/sell_mask[B,N], receivable_ratio[B,1]. Upstream
        observation construction supplies finite feature values and masks.
        Returns: mean[B,N(+1)], value[B,1], log_std[N(+1)]. The Gaussian distribution
        broadcasts log_std across the batch and uses exp(log_std) as its scale.
        """
        # Flatten only time and feature axes: [B,N,L,F] -> [B,N,L*F].
        features = obs["stock_features"].float().flatten(start_dim=2)
        weights = obs["portfolio"][:, :-1].float()
        valid = obs["valid_mask"].float()
        flags = torch.stack([weights, valid, obs["buy_mask"].float(), obs["sell_mask"].float()], dim=-1)
        encoded = self.encoder(torch.cat([features, flags], dim=-1))
        # encoded[B,N,d] includes holdings and trading flags as well as features.
        # Average valid stocks into market[B,d]; an empty valid pool yields zero.
        market = (encoded * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        # held[B,d] retains positions in invalid stocks. Its magnitude also
        # reflects total invested weight; it is a weighted sum, not an average.
        held = (encoded * weights.unsqueeze(-1)).sum(1)
        context = torch.cat([market, held, obs["portfolio"][:, -1:].float(),
                             obs["receivable_ratio"].float()], dim=-1)
        # context[B,2*d+2] is broadcast to every stock. Stock-head input is
        # [B,N,3*d+2], combining its own embedding with the same portfolio state.
        stock = self.stock_head(torch.cat([encoded, context.unsqueeze(1).expand(-1, encoded.shape[1], -1)], dim=-1)).squeeze(-1)
        cash = None if self.full_invest else self.cash_head(context)
        if self.prior_feature_index is not None:
            # Rank only valid, buyable stocks by their latest selected feature.
            # In R1 the selected feature is momentum_20. Infinity moves excluded
            # stocks behind eligible candidates; equal signals get equal midranks.
            eligible = obs["valid_mask"].bool() & obs["buy_mask"].bool()
            signal = obs["stock_features"][:, :, -1, self.prior_feature_index].contiguous()
            sorted_signal = signal.masked_fill(~eligible, float("inf")).sort(dim=1).values.contiguous()
            left = torch.searchsorted(sorted_signal, signal, right=False)
            right = torch.searchsorted(sorted_signal, signal, right=True)
            midrank = (left + right - 1).float() / 2
            count = eligible.sum(1, keepdim=True)
            # Scale ranks to [-1,1]; a lone candidate gets +1. Excluded stocks
            # receive zero prior, while final eligibility is enforced by map_action.
            rank = torch.where(count > 1, 2 * midrank / (count - 1).clamp_min(1) - 1, torch.ones_like(signal))
            prior = torch.where(eligible, rank * self.prior_strength, torch.zeros_like(signal))
            # R1 uses 4*rank + 0.25*tanh(stock), and 0.25*tanh(cash).
            stock = prior + self.residual_scale * stock.tanh()
            if cash is not None:
                cash = self.residual_scale * cash.tanh()
        # Mean contains latent stock scores plus cash, before weight mapping.
        # The critic estimates discounted future rewards for PPO's advantage target.
        if self.full_invest:
            log_std = self.log_std[:1].expand(encoded.shape[1])
            return stock, self.value_head(context), log_std
        mean = torch.cat([stock, cash], dim=1)
        log_std = torch.cat([self.log_std[:1].expand(encoded.shape[1]), self.log_std[1:]])
        return mean, self.value_head(context), log_std


class SharedStockPolicy(ActorCriticPolicy):
    """Connect shared stock representations to SB3's Gaussian actor and critic."""

    def __init__(self, *args, hidden_sizes=(128, 64), initial_log_std=0.0, prior_feature_index=None,
                 prior_strength=4.0, residual_scale=0.25, full_invest=False, **kwargs):
        self.hidden_sizes = tuple(hidden_sizes)
        self.full_invest = bool(full_invest)
        self.prior_options = dict(initial_log_std=initial_log_std, prior_feature_index=prior_feature_index,
                                  prior_strength=prior_strength, residual_scale=residual_scale,
                                  full_invest=self.full_invest)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        n, lookback, features = self.observation_space["stock_features"].shape
        size = n if self.full_invest else n + 1
        if self.action_space.shape != (size,) or self.use_sde:
            raise ValueError("SharedStockPolicy action shape does not match the cash/full-invest mapping")
        self.network = SharedStockNetwork(lookback, features, self.hidden_sizes, **self.prior_options)
        self.action_dist = DiagGaussianDistribution(size)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _distribution_value(self, obs):
        mean, value, log_std = self.network(obs)
        # One Gaussian coordinate per stock and one for cash; these are latent
        # actions. Masks, softmax and caps are applied by the environment wrapper.
        return self.action_dist.proba_distribution(mean, log_std), value

    def forward(self, obs, deterministic=False):
        distribution, value = self._distribution_value(obs)
        # Training samples for exploration; deterministic inference uses the mean.
        action = distribution.get_actions(deterministic=deterministic)
        return action, value, distribution.log_prob(action)

    def evaluate_actions(self, obs, actions):
        # PPO reevaluates stored latent actions for probability ratios and loss.
        distribution, value = self._distribution_value(obs)
        return value, distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs):
        return self._distribution_value(obs)[0]

    def predict_values(self, obs):
        return self.network(obs)[1]

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["hidden_sizes"] = self.hidden_sizes
        data.update(self.prior_options)
        return data
