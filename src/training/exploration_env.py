"""Research-only common observation support, independent date sampling and rewards."""

from copy import deepcopy

import numpy as np
import pandas as pd

from src.agents.trade_mapping import TradeActionWrapper, map_trade_action, TradeOrder
from src.envs.broker import BrokerConfig
from src.envs.market import MarketDataProvider
from src.envs.trading import TradingEnv
from .env_factory import ResearchData


START_SEED_SALT = 0x510A
MODE_SEED_SALT = 0xA110
ACCOUNT_SEED_SALT = 0xACCD


class DateSampler:
    def __init__(self, dates, episode_steps, mode="recent", recent_every=4):
        if mode not in {"recent", "quarter_balanced"} or episode_steps < 2:
            raise ValueError("Invalid research date sampler")
        self.dates = pd.DatetimeIndex(dates)
        self.high = len(dates) - episode_steps
        if self.high < 1:
            raise ValueError("A full training episode must fit within the training boundary")
        self.mode, self.recent_every = mode, recent_every
        quarters = self.dates[:self.high].to_period("Q").astype(str)
        self.buckets = {key: np.flatnonzero(quarters == key) for key in sorted(set(quarters))}
        self.seed(0)

    def seed(self, seed):
        self.rng = np.random.default_rng(int(seed) ^ START_SEED_SALT)
        self.count, self.remaining = 0, []

    def next_index(self):
        if self.mode == "recent":
            index = self.high - 1 if self.recent_every and self.count % self.recent_every == 0 else int(self.rng.integers(self.high))
        else:
            if not self.remaining:
                self.remaining = list(self.rng.permutation(list(self.buckets)))
            choices = self.buckets[self.remaining.pop()]
            index = int(choices[self.rng.integers(len(choices))])
        self.count += 1
        return index


class ExplorationMarket(MarketDataProvider):
    def __init__(self, *args, common_lookback=5, sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        if type(common_lookback) is not int or common_lookback < self.lookback:
            raise ValueError("Common support window cannot be shorter than model lookback")
        self.common_lookback, self.sampler = common_lookback, sampler
        finite = np.isfinite(self._features).all(axis=2)
        self.common_valid = pd.DataFrame(finite).rolling(common_lookback).sum().eq(common_lookback).to_numpy()

    def observation(self, idx=None):
        index = self.current_index if idx is None else idx
        values, valid, buy, sell = super().observation(index)
        valid &= self.common_valid[self._indices[index]]
        return values, valid, buy & valid, sell & valid

    def reset(self):
        self.current_index = self.sampler.next_index() if self.sampler is not None else 0
        return self.get_state()


class ExplorationEnv(TradingEnv):
    def __init__(self, *args, reward_basis="simple_net", **kwargs):
        if reward_basis not in {"simple_net", "log_net"}:
            raise ValueError("Unknown research reward basis")
        self.reward_basis = reward_basis
        super().__init__(*args, random_start=False, recent_episode_every=0, **kwargs)
        self.account_seed, self.account_episode, self.mode_bag = 0, 0, []
        self.mode_rng = np.random.default_rng(MODE_SEED_SALT)

    def reset(self, *, seed=None, options=None):
        if self.market.sampler is None:
            return super().reset(seed=seed, options=options)
        if seed is not None:
            self.market.sampler.seed(seed)
            self.account_seed, self.account_episode, self.mode_bag = int(seed), 0, []
            self.mode_rng = np.random.default_rng(int(seed) ^ MODE_SEED_SALT)
            self._episode_count = 0
        requested = self.initial_portfolio
        if requested == "mixed":
            from src.envs.initial_account import MODES
            if not self.mode_bag:
                self.mode_bag = list(self.mode_rng.permutation(MODES))
            self.initial_portfolio = self.mode_bag.pop()
        # Different eligible pools may consume different draws within an account.
        # A fresh episode RNG prevents that from changing later account modes.
        account_seed = int(np.random.SeedSequence([self.account_seed, ACCOUNT_SEED_SALT, self.account_episode]).generate_state(1)[0])
        self._np_random, self._np_random_seed = np.random.default_rng(account_seed), account_seed
        self.account_episode += 1
        try:
            return super().reset(seed=None, options=options)
        finally:
            self.initial_portfolio = requested

    def _advance(self, *args, **kwargs):
        obs, raw, terminated, truncated, info = super()._advance(*args, **kwargs)
        if raw <= -1 or not np.isfinite(raw):
            raise FloatingPointError("Net equity must be positive for research log reward")
        reward = float(np.log1p(raw)) if self.reward_basis == "log_net" else float(raw)
        info.update(raw_net_return=float(raw), learning_reward=reward, reward_basis=self.reward_basis, reward=reward)
        return obs, reward, terminated, truncated, info


class ExplorationData(ResearchData):
    def __init__(self, config):
        super().__init__(config)
        spec = config.get("exploration", {})
        self.architecture = spec.get("architecture", "baseline")
        self.common_lookback = spec.get("common_lookback", 5)
        self.sampling = spec.get("sampling", "recent")
        self.reward_basis = spec.get("reward_basis", "simple_net")
        if (self.architecture not in {"baseline", "split"} or self.sampling not in {"recent", "quarter_balanced"}
                or self.reward_basis not in {"simple_net", "log_net"}
                or type(self.common_lookback) is not int or self.common_lookback < config["lookback"]):
            raise ValueError("Invalid isolated exploration specification")
        self.contract["exploration"] = dict(architecture=self.architecture, common_lookback=self.common_lookback,
            sampling=self.sampling, date_sampler_seed_salt=START_SEED_SALT, reward_basis=self.reward_basis,
            account_mode_seed_salt=MODE_SEED_SALT, account_episode_seed_salt=ACCOUNT_SEED_SALT,
            size_annealing=bool(spec.get("size_annealing", False)), research_only=True)

    def market(self, split, *, allow_test=False):
        if split not in {"train", "val"} or allow_test:
            raise ValueError("Exploration never opens the reserved test split")
        end = pd.Timestamp(self.manifest["train_end" if split == "train" else "val_end"])
        columns = pd.MultiIndex.from_product([self.codes, self.manifest["feature_cols"]], names=self.features.columns.names)
        features = self.features.loc[:end].reindex(columns=columns).copy()
        prices = self.prices.loc[:end, pd.IndexSlice[self.codes, :]].copy()
        if split == "val":
            start = features.index[features.index <= pd.Timestamp(self.manifest["train_end"])][-1]
        else:
            finite = np.isfinite(features.to_numpy().reshape(len(features), len(self.codes), -1)).all(axis=2)
            eligible = pd.DataFrame(finite).rolling(self.common_lookback).sum().eq(self.common_lookback).any(axis=1)
            valid = np.flatnonzero(eligible.to_numpy())
            if not len(valid):
                raise ValueError("No common-window training data")
            start = features.index[valid[0]]
        market = ExplorationMarket(features, self.manifest["feature_cols"], price_path=prices,
            price_basis=self.manifest["price_basis"], start_date=start, end_date=end,
            lookback=self.config["lookback"], common_lookback=self.common_lookback)
        if split == "train":
            market.sampler = DateSampler(market.dates, self.config["training_episode_steps"], self.sampling,
                                        self.config.get("recent_episode_every", 4))
        return market

    def env(self, split):
        training = split == "train"
        env = ExplorationEnv(self.market(split), self.settings["initial_capital"], BrokerConfig(**self.settings["broker"]),
            initial_portfolio=self.config.get("initial_portfolio", "cash") if training else self.settings.get("initial_portfolio", "cash"),
            max_steps=self.config.get("training_episode_steps") if training else None,
            reward_basis=self.reward_basis if training else "simple_net")
        return TradeActionWrapper(env, self.config["logit_bound"], self.config.get("max_trade_weight", .05),
            self.config.get("max_commission_ratio", .001), self.settings.get("trade_top_n", self.config.get("top_n", 20)),
            self.config.get("top_n_choices") if training else None)


class EconomicRule:
    """Known-feature ranking rules sharing the neural actor's mapper and broker."""

    def __init__(self, contract, kind="hold", interval=1):
        if kind not in {"hold", "momentum_20"} or type(interval) is not int or interval < 1:
            raise ValueError("Unsupported fixed economic baseline")
        self.contract, self.kind, self.interval = deepcopy(contract), kind, interval
        self.feature_index = contract["feature_cols"].index("momentum_20")

    def act(self, obs, step=None):
        n = len(obs["portfolio"]) - 1
        raw = np.zeros(2 * n, dtype=np.float64)
        if self.kind != "hold" and (step or 0) % self.interval == 0:
            legal = obs["legal_directions"].astype(bool)
            scores = obs["stock_features"][:, -1, self.feature_index]
            valid = obs["valid_mask"].astype(bool)
            indices = np.flatnonzero(valid)
            ranked = sorted(indices, key=lambda i: (-float(scores[i]), self.contract["stock_codes"][i]))
            chosen = ranked[:int(obs["top_n"][0])]
            selected = np.zeros(n, dtype=bool)
            selected[chosen] = True
            raw[:n][~selected & legal[:, 3]] = 3
            buy = np.array([i for i in chosen if legal[i, 1]], dtype=int)
            raw[buy] = 1
            # Rank-dependent size is explicitly part of this rule's mechanism.
            raw[n + buy] = np.linspace(4., -4., len(buy)) if len(buy) else []
        result = map_trade_action(raw, obs, self.contract["logit_bound"], self.contract["broker"]["lot_size"],
                                  self.contract["stock_codes"])
        return TradeOrder(result["shares"], result["freeze_mask"], result["exit_mask"], result["priority"], result["top_n"], result)
