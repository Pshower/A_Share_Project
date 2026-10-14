"""Causal macro timing observations and independently executed M20 benchmark."""

from copy import deepcopy
from itertools import product

import gymnasium as gym
import numpy as np
import pandas as pd

from src.agents.trade_mapping import TradeActionWrapper, TRADE_MAPPING_VERSION
from src.data.preprocess import FEATURES
from src.envs.initial_account import MODES, initialize_account
from src.envs.trading import TradingEnv
from .exploration_env import EconomicRule


PHASE_PERIOD = 20
TIMING_MAPPING_VERSION = "timing_hold_rebalance_m20_two_books_v1"
TIMING_OBSERVATION_VERSION = "causal_phase_candidate_two_account_summary_v1"
ACTION_NAMES = ("hold", "rebalance_m20")
DATE_STRATA = ("early", "middle", "latest")
SCHEDULE_SEED_SALT = 0x714A
DATE_SEED_SALT = 0x7DA7
PHASE_SEED_SALT = 0x7FA5
ACCOUNT_SEED_SALT = 0x7ACC
ACCOUNT_FIELDS = (
    "cash_ratio", "receivable_ratio", "stock_exposure", "held_fraction",
    "locked_fraction", "new_slot_slack", "frozen_weight", "sellable_weight",
    "weight_hhi", "held_valid_weight", "valid_fraction", "buy_legal_fraction",
    "reduce_legal_fraction", "exit_legal_fraction",
)
FEATURE_NAMES = (
    tuple(f"phase_{phase:02d}" for phase in range(PHASE_PERIOD))
    + tuple(f"market_mean_{name}" for name in FEATURES)
    + tuple(f"agent_held_{name}" for name in FEATURES)
    + tuple(f"benchmark_held_{name}" for name in FEATURES)
    + tuple(f"candidate_mean_{name}" for name in FEATURES)
    + tuple(f"agent_{name}" for name in ACCOUNT_FIELDS)
    + tuple(f"benchmark_{name}" for name in ACCOUNT_FIELDS)
    + ("log_agent_benchmark_equity_ratio",)
)


def build_timing_contract(base_contract, reward_basis, calendar_margin=2.):
    if reward_basis not in {"log_net", "relative_log"}:
        raise ValueError("Unknown timing reward basis")
    if not np.isfinite(calendar_margin) or calendar_margin <= 0:
        raise ValueError("Calendar margin must be positive and finite")
    contract = deepcopy(base_contract)
    if tuple(contract.get("feature_cols", ())) != tuple(FEATURES):
        raise ValueError("Timing observations require the registered twenty features")
    codes = contract.get("stock_codes", ())
    if not codes or len(codes) != len(set(codes)) or not all(isinstance(code, str) for code in codes):
        raise ValueError("Timing contract needs a unique ordered stock pool")
    contract.update(action_mode="timing_control", mapping_version=TIMING_MAPPING_VERSION)
    contract["timing"] = dict(feature_names=list(FEATURE_NAMES),
        observation_version=TIMING_OBSERVATION_VERSION, phase_period=PHASE_PERIOD,
        action_names=list(ACTION_NAMES), calendar_margin=float(calendar_margin),
        reward_basis=reward_basis, decoder_version=TRADE_MAPPING_VERSION,
        benchmark_interval=PHASE_PERIOD, benchmark_independent_book=True,
        candidate_selection="valid momentum_20 descending, stock-code ascending ties, top_n",
        research_only=True)
    return contract


def _state_summary(observation):
    features = np.asarray(observation["stock_features"], dtype=np.float64)
    portfolio = np.asarray(observation["portfolio"], dtype=np.float64)
    valid_values = np.asarray(observation["valid_mask"])
    legal_values = np.asarray(observation["legal_directions"])
    execution = np.asarray(observation["execution_features"], dtype=np.float64)
    allocation = np.asarray(observation["allocation_context"], dtype=np.float64)
    share_state = np.asarray(observation["share_state"], dtype=np.float64)
    receivables = np.asarray(observation["receivable_ratio"], dtype=np.float64)
    n = len(portfolio) - 1
    if (n < 1 or features.ndim != 3 or features.shape[0] != n
            or features.shape[1] < 1 or features.shape[2] != len(FEATURES)
            or valid_values.shape != (n,) or legal_values.shape != (n, 4)
            or execution.shape != (n, 3) or allocation.shape != (7,)
            or share_state.shape != (n, 2) or receivables.shape != (1,)
            or not np.isin(valid_values, (0, 1)).all()
            or not np.isin(legal_values, (0, 1)).all()
            or not np.isin(execution[:, 2], (0, 1)).all()
            or not np.isfinite(features).all() or not np.isfinite(portfolio).all()
            or not np.isfinite(execution).all() or not np.isfinite(allocation).all()
            or not np.isfinite(share_state).all() or not np.isfinite(receivables).all()
            or (portfolio < 0).any() or (execution < 0).any()
            or (share_state < 0).any() or (allocation < 0).any()
            or (receivables < 0).any()):
        raise ValueError("Malformed full stock observation")
    valid, legal = valid_values.astype(bool), legal_values.astype(bool)
    latest = features[:, -1, :]
    weights = portfolio[:-1]
    held = share_state[:, 0] > 0
    locked = held & ((share_state[:, 1] < share_state[:, 0]) | execution[:, 2].astype(bool))
    market_mean = latest[valid].mean(axis=0) if valid.any() else np.zeros(len(FEATURES))
    held_features = (latest * (weights * valid)[:, None]).sum(axis=0)
    account = np.array([
        portfolio[-1], receivables[0], weights.sum(), held.mean(), locked.mean(),
        allocation[5], weights[execution[:, 2].astype(bool)].sum(), execution[:, 1].sum(),
        np.square(weights).sum(), weights[valid].sum(), valid.mean(),
        legal[:, 1].mean(), legal[:, 2].mean(), legal[:, 3].mean(),
    ], dtype=np.float64)
    return latest, valid, market_mean, held_features, account


def timing_observation(stock_observation, benchmark_observation, phase,
                       agent_equity, benchmark_equity, stock_codes):
    """Pure snapshot transform shared by rollout and explicit two-book inference."""
    if type(phase) is not int or not 0 <= phase < PHASE_PERIOD:
        raise ValueError("Calendar phase must be an integer in 0..19")
    if not np.isfinite([agent_equity, benchmark_equity]).all() or min(agent_equity, benchmark_equity) <= 0:
        raise ValueError("Both account equities must be positive and finite")
    latest, valid, market_mean, agent_held, agent_account = _state_summary(stock_observation)
    _, _, _, benchmark_held, benchmark_account = _state_summary(benchmark_observation)
    if (len(stock_codes) != len(valid) or len(set(stock_codes)) != len(stock_codes)
            or len(benchmark_observation["portfolio"]) != len(stock_observation["portfolio"])):
        raise ValueError("Timing summary stock coordinates differ")
    if (not np.array_equal(stock_observation["stock_features"], benchmark_observation["stock_features"])
            or not np.array_equal(stock_observation["valid_mask"], benchmark_observation["valid_mask"])):
        raise ValueError("Benchmark summary must use the same current market snapshot")
    top_values = np.asarray(stock_observation["top_n"], dtype=float)
    if top_values.shape != (1,) or not np.isfinite(top_values).all() or not top_values[0].is_integer() or not 0 <= top_values[0] <= 500:
        raise ValueError("Invalid current top_n")
    momentum = FEATURES.index("momentum_20")
    ranked = sorted(np.flatnonzero(valid), key=lambda i: (-float(latest[i, momentum]), stock_codes[i]))
    chosen = ranked[:int(top_values[0])]
    candidates = latest[chosen].mean(axis=0) if chosen else np.zeros(len(FEATURES))
    onehot = np.eye(PHASE_PERIOD, dtype=np.float64)[phase]
    vector = np.concatenate([onehot, market_mean, agent_held, benchmark_held, candidates,
                             agent_account, benchmark_account,
                             [np.log(agent_equity) - np.log(benchmark_equity)]])
    with np.errstate(over="ignore"):
        vector = vector.astype(np.float32)
    if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("Timing features must be finite and match the registered names")
    return vector


class StratifiedTimingSampler:
    """Each phase block covers all twelve account/date-stratum combinations."""

    def __init__(self, dates, episode_steps=240):
        self.dates = pd.DatetimeIndex(dates)
        self.episode_steps = episode_steps
        if type(episode_steps) is not int or episode_steps < 1:
            raise ValueError("Episode steps must be a positive integer")
        self.maximum_start = len(self.dates) - episode_steps - 1
        if self.maximum_start < 2 or self.dates.has_duplicates or not self.dates.is_monotonic_increasing:
            raise ValueError("Three start strata require enough unique ordered training dates")
        early, middle = np.array_split(np.arange(self.maximum_start), 2)
        self.starts = {"early": early, "middle": middle, "latest": np.array([self.maximum_start])}
        self.seed(0)

    def seed(self, seed):
        self.seed_value = int(seed)
        self.schedule_rng = np.random.default_rng(self.seed_value ^ SCHEDULE_SEED_SALT)
        self.date_rng = np.random.default_rng(self.seed_value ^ DATE_SEED_SALT)
        self.phase_rng = np.random.default_rng(self.seed_value ^ PHASE_SEED_SALT)
        self.phase_bag, self.block = [], []
        self.episode_count, self.history = 0, []

    def next(self):
        if not self.block:
            if not self.phase_bag:
                self.phase_bag = list(self.schedule_rng.permutation(4))
            phase_bin = int(self.phase_bag.pop())
            combinations = list(product(MODES, DATE_STRATA))
            self.block = [(str(combinations[int(index)][0]), combinations[int(index)][1], phase_bin)
                          for index in self.schedule_rng.permutation(len(combinations))]
        mode, stratum, phase_bin = self.block.pop()
        starts = self.starts[stratum]
        index = int(starts[int(self.date_rng.integers(len(starts)))])
        phase = int(5 * phase_bin + self.phase_rng.integers(5))
        result = dict(episode=self.episode_count, initial_mode=mode, date_stratum=stratum,
                      phase_bin=phase_bin, phase=phase, start_index=index,
                      start_date=str(self.dates[index].date()))
        self.episode_count += 1
        self.history.append(deepcopy(result))
        return result

    def audit(self):
        return dict(seed=self.seed_value, schedule_seed_salt=SCHEDULE_SEED_SALT,
                    date_seed_salt=DATE_SEED_SALT, phase_seed_salt=PHASE_SEED_SALT,
                    episode_count=self.episode_count, history=deepcopy(self.history),
                    rng_states={"schedule": deepcopy(self.schedule_rng.bit_generator.state),
                                "date": deepcopy(self.date_rng.bit_generator.state),
                                "phase": deepcopy(self.phase_rng.bit_generator.state)})


class TimingEnv(gym.Wrapper):
    """HOLD/REBALANCE macro actions with a separate, always-M20 reference account."""

    def __init__(self, env, contract, *, benchmark=None, training=False,
                 reward_basis="log_net", episode_steps=240, initial_mode=None):
        if not isinstance(env, TradeActionWrapper) or not isinstance(env.env, TradingEnv):
            raise TypeError("TimingEnv requires a full TradeActionWrapper over TradingEnv")
        if reward_basis not in {"log_net", "relative_log"}:
            raise ValueError("Unknown timing reward basis")
        if not isinstance(training, bool) or type(episode_steps) is not int or episode_steps < 1:
            raise ValueError("Invalid timing episode configuration")
        if env.top_n != 20 or env.top_n_choices or env.env.random_start or getattr(env.env.market, "sampler", None) is not None:
            raise ValueError("Timing requires fixed K20 and exclusively owned start sampling")
        if training and env.env.max_steps != episode_steps:
            raise ValueError("Training stock engine must truncate at the registered episode length")
        if tuple(contract.get("feature_cols", ())) != tuple(FEATURES) or list(contract.get("stock_codes", ())) != env.env.market.stock_codes:
            raise ValueError("Timing contract differs from the stock engine coordinates")
        if (contract.get("price_basis", env.env.market.price_basis) != env.env.market.price_basis
                or contract.get("logit_bound") != env.bound
                or not isinstance(contract.get("broker"), dict)
                or contract["broker"].get("lot_size") != env.env.broker.config.lot_size
                or any(name not in env.env.broker.config.__dataclass_fields__
                       or value != getattr(env.env.broker.config, name)
                       for name, value in contract["broker"].items())
                or any(name in contract and contract[name] != getattr(env, name)
                       for name in ("max_trade_weight", "max_commission_ratio"))
                or ("top_n" in contract and contract["top_n"] != env.top_n)):
            raise ValueError("Timing contract price basis, mapper or broker differs")
        super().__init__(env)
        self.benchmark = deepcopy(env) if benchmark is None else benchmark
        if not isinstance(self.benchmark, TradeActionWrapper) or not isinstance(self.benchmark.env, TradingEnv):
            raise TypeError("Benchmark needs its own full stock engine")
        reference = self.benchmark.env
        if (self.benchmark is env or reference is env.env or reference.portfolio is env.env.portfolio
                or reference.broker is env.env.broker or reference.market is env.env.market
                or self.benchmark.top_n != 20 or self.benchmark.top_n_choices
                or reference.random_start or getattr(reference.market, "sampler", None) is not None
                or reference.max_steps != env.env.max_steps
                or reference.broker.config != env.env.broker.config
                or not reference.market.dates.equals(env.env.market.dates)
                or reference.market.stock_codes != env.env.market.stock_codes
                or reference.market.feature_cols != env.env.market.feature_cols):
            raise ValueError("Benchmark must be an independent, identically configured engine")
        market, reference_market = env.env.market, reference.market
        if (reference_market.price_basis != market.price_basis or reference_market.lookback != market.lookback
                or not np.array_equal(reference_market._indices, market._indices)
                or not np.array_equal(reference_market._features, market._features, equal_nan=True)
                or not np.array_equal(reference_market._valuation, market._valuation, equal_nan=True)
                or not reference_market.prices.equals(market.prices)
                or set(reference_market._fields) != set(market._fields)
                or any(not np.array_equal(reference_market._fields[name], values, equal_nan=True)
                       for name, values in market._fields.items())
                or not reference_market.corporate_actions.equals(market.corporate_actions)
                or getattr(reference_market, "common_lookback", None) != getattr(market, "common_lookback", None)):
            raise ValueError("Benchmark market quotes, features or supplied events differ")
        self.contract = build_timing_contract(contract, reward_basis,
            calendar_margin=contract.get("timing", {}).get("calendar_margin", 2.))
        self.training, self.reward_basis, self.episode_steps = training, reward_basis, episode_steps
        mode = env.env.initial_portfolio if initial_mode is None else initial_mode
        if not training and mode not in MODES:
            raise ValueError("Evaluation needs an explicit settled initial account mode")
        self.initial_mode = mode
        self.sampler = StratifiedTimingSampler(env.env.market.dates, episode_steps) if training else None
        self.feature_names = FEATURE_NAMES
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (len(FEATURE_NAMES),), np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.rebalance_rule = EconomicRule(contract, "momentum_20", interval=1)
        self.hold_rule = EconomicRule(contract, "hold")
        self.benchmark_rule = EconomicRule(contract, "momentum_20", interval=PHASE_PERIOD)
        self.phase, self.episode_number, self.seed_value = 0, 0, 0
        self.total_agent_steps, self.total_benchmark_steps = 0, 0
        self.stock_observation = self.benchmark_stock_observation = None
        self.last_stock_obs = self.last_benchmark_stock_obs = None
        self.last_order = self.last_benchmark_order = None
        self.last_timing_observation = self.last_macro_action = None
        self._agent_equity = self._benchmark_equity = None
        self._done = True

    @staticmethod
    def _refresh(wrapper):
        base = wrapper.env
        wrapper._observation = wrapper.enrich(base._observation())
        info = base._info()
        info["top_n"] = wrapper.top_n
        base.history = [info]
        return wrapper._observation, info

    def reset(self, *, seed=None, options=None):
        options = {} if options is None else dict(options)
        if set(options) - {"initial_mode"} or (self.training and options):
            raise ValueError("Training schedule cannot be overridden; evaluation accepts initial_mode only")
        if seed is not None:
            self.seed_value, self.episode_number = int(seed), 0
            self.action_space.seed(int(seed) ^ 0x7AC7)
            if self.sampler is not None:
                self.sampler.seed(seed)
        if self.sampler is not None:
            schedule = self.sampler.next()
        else:
            mode = options.get("initial_mode", self.initial_mode)
            if mode not in MODES:
                raise ValueError("Invalid evaluation initial account mode")
            schedule = dict(episode=self.episode_number, initial_mode=mode, date_stratum="evaluation",
                            phase_bin=0, phase=0, start_index=0,
                            start_date=str(self.env.env.market.dates[0].date()))
        account_seed = (int(np.random.SeedSequence([self.seed_value, ACCOUNT_SEED_SALT, self.episode_number]).generate_state(1)[0])
                        if self.training else self.seed_value)
        # Complete the reset at the selected date before the only account initializer.
        # No intermediate account or date-zero observation escapes this wrapper.
        for wrapper in (self.env, self.benchmark):
            base = wrapper.env
            base.market.current_index = schedule["start_index"]
            base.portfolio.reset()
            base.portfolio.start_day(base.market.current_date)
            base.broker.reset()
            base._steps, base._done = 0, False
        base = self.env.env
        base._np_random, base._np_random_seed = np.random.default_rng(account_seed), account_seed
        base.initial_account = initialize_account(base.portfolio,
            base.market.prices.iloc[base.market.current_index].to_numpy(), base.market.observation()[1],
            base.broker.config, schedule["initial_mode"], base.np_random)
        self.benchmark.env.portfolio = deepcopy(base.portfolio)
        self.benchmark.env.initial_account = deepcopy(base.initial_account)
        self.stock_observation, info = self._refresh(self.env)
        self.benchmark_stock_observation, benchmark_info = self._refresh(self.benchmark)
        self.phase, self.episode_number = schedule["phase"], self.episode_number + 1
        self._agent_equity, self._benchmark_equity = float(info["equity"]), float(benchmark_info["equity"])
        self.last_stock_obs = self.last_benchmark_stock_obs = None
        self.last_order = self.last_benchmark_order = None
        self.last_timing_observation = self.last_macro_action = None
        self._done = False
        observation = self.observe()
        info["timing"] = dict(observation=observation.copy(), phase=self.phase, phase_before=self.phase,
            phase_after=self.phase, macro_action=None, schedule=deepcopy(schedule), account_seed=account_seed,
            benchmark_ledger=deepcopy(benchmark_info), benchmark_step_count=0,
            total_agent_steps=self.total_agent_steps, total_benchmark_steps=self.total_benchmark_steps,
            reward_basis=self.reward_basis)
        return observation, info

    def observe(self):
        if self.stock_observation is None or self.benchmark_stock_observation is None:
            raise RuntimeError("Call reset before observing the timing state")
        return timing_observation(self.stock_observation, self.benchmark_stock_observation, self.phase,
                                  self._agent_equity, self._benchmark_equity, self.contract["stock_codes"])

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset before stepping or after episode end")
        value = np.asarray(action)
        if value.shape != () or value.dtype.kind not in "iu" or int(value) not in (0, 1):
            raise ValueError("Timing action must be one integer HOLD/REBALANCE code")
        macro = int(value)
        phase = self.phase
        self.last_stock_obs = deepcopy(self.stock_observation)
        self.last_benchmark_stock_obs = deepcopy(self.benchmark_stock_observation)
        self.last_timing_observation = self.observe().copy()
        self.last_macro_action = macro
        self.last_order = (self.rebalance_rule if macro else self.hold_rule).act(self.stock_observation)
        self.last_benchmark_order = self.benchmark_rule.act(self.benchmark_stock_observation, step=phase)
        stock, original_reward, terminated, truncated, info = self.env.step_order(self.last_order)
        reference, _, benchmark_terminated, benchmark_truncated, benchmark_info = self.benchmark.step_order(self.last_benchmark_order)
        self.total_agent_steps += 1
        self.total_benchmark_steps += 1
        if (info["date"] != benchmark_info["date"] or terminated != benchmark_terminated or truncated != benchmark_truncated):
            raise RuntimeError("Agent and benchmark execution clocks or episode boundaries diverged")
        agent_equity, benchmark_equity = float(info["equity"]), float(benchmark_info["equity"])
        if not np.isfinite([agent_equity, benchmark_equity]).all() or min(agent_equity, benchmark_equity) <= 0:
            raise FloatingPointError("Timing log rewards need positive finite account equity")
        agent_return = agent_equity / self._agent_equity - 1
        benchmark_return = benchmark_equity / self._benchmark_equity - 1
        agent_log, benchmark_log = np.log1p(agent_return), np.log1p(benchmark_return)
        learning_reward = float(agent_log - benchmark_log if self.reward_basis == "relative_log" else agent_log)
        self.stock_observation, self.benchmark_stock_observation = stock, reference
        self._agent_equity, self._benchmark_equity = agent_equity, benchmark_equity
        self.phase = (phase + 1) % PHASE_PERIOD
        self._done = bool(terminated or truncated)
        info["timing_stock_observation"] = self.last_stock_obs
        info["timing_benchmark_stock_observation"] = self.last_benchmark_stock_obs
        info["timing"] = dict(observation=self.last_timing_observation.copy(), macro_action=macro,
            phase=phase, phase_before=phase, phase_after=self.phase,
            raw_net_return=float(agent_return), benchmark_net_return=float(benchmark_return),
            agent_log_return=float(agent_log), benchmark_log_return=float(benchmark_log),
            learning_reward=learning_reward, stock_engine_reward=float(original_reward), reward_basis=self.reward_basis,
            benchmark_step_count=self.benchmark.env._steps,
            total_agent_steps=self.total_agent_steps, total_benchmark_steps=self.total_benchmark_steps,
            benchmark_ledger=deepcopy(benchmark_info),
            derived_stock_raw_action=self.last_order.diagnostics["latent"].copy(),
            benchmark_raw_action=self.last_benchmark_order.diagnostics["latent"].copy())
        info.update(raw_net_return=float(agent_return), learning_reward=learning_reward,
                    reward_basis=self.reward_basis, reward=learning_reward)
        return self.observe(), learning_reward, terminated, truncated, info

    def schedule_audit(self):
        return dict(schedule=self.sampler.audit() if self.sampler is not None else None,
                    account_seed_salt=ACCOUNT_SEED_SALT, episode_count=self.episode_number,
                    agent_environment_steps=self.total_agent_steps,
                    benchmark_extra_environment_steps=self.total_benchmark_steps)

    def close(self):
        try:
            self.env.close()
        finally:
            self.benchmark.close()
