"""Synthetic two-book timing checks; no downloads, neural calls or training."""

from collections import Counter
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.agents.trade_mapping import TradeActionWrapper
from src.data.preprocess import FEATURES
from src.envs.broker import BrokerConfig
from src.envs.trading import TradingEnv
from src.training.exploration_env import EconomicRule, ExplorationMarket
from src.training.timing_env import (
    ACCOUNT_FIELDS, DATE_STRATA, FEATURE_NAMES, MODES, PHASE_PERIOD,
    StratifiedTimingSampler, TimingEnv, build_timing_contract, timing_observation,
)


def stock_engine(*, days=58, stocks=24, training=False, mode="cash", price_basis="hfq_research"):
    dates = pd.bdate_range("2020-01-01", periods=days + 4)
    codes = [f"{index:06d}" for index in range(stocks)]
    time, asset, feature = np.ogrid[:len(dates), :stocks, :len(FEATURES)]
    values = np.sin(time / 9 + asset / 4) + feature / 100 + asset / 20
    panel = pd.DataFrame(values.reshape(len(dates), -1), index=dates,
                         columns=pd.MultiIndex.from_product([codes, FEATURES]))
    quote = pd.DataFrame(index=dates, columns=pd.MultiIndex.from_product([codes, ("open", "close", "volume")]), dtype=float)
    for index, code in enumerate(codes):
        close = 10 + index / 5 + np.arange(len(dates)) / 100 + np.sin(np.arange(len(dates)) / 7 + index) / 5
        quote[code, "close"] = close
        quote[code, "open"] = close + .03
        quote[code, "volume"] = 1e8
    market = ExplorationMarket(panel, FEATURES, price_path=quote, price_basis=price_basis,
                               start_date=dates[4], common_lookback=5)
    env = TradingEnv(market, 1e6, BrokerConfig(), max_steps=240 if training else None,
                     initial_portfolio="mixed" if training else mode)
    wrapper = TradeActionWrapper(env, top_n=20)
    contract = dict(stock_codes=codes, feature_cols=list(FEATURES), logit_bound=8.,
                    broker=dict(lot_size=100), price_basis=price_basis)
    return wrapper, contract


def timing_env(*, basis="log_net", training=False, mode="cash", days=58, stocks=24):
    wrapper, contract = stock_engine(days=days, stocks=stocks, training=training, mode=mode)
    return TimingEnv(wrapper, contract, training=training, reward_basis=basis)


def equal_values(left, right):
    if isinstance(left, dict):
        assert set(left) == set(right)
        for name in left:
            equal_values(left[name], right[name])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for old, new in zip(left, right):
            equal_values(old, new)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


def ledger_equal(left, right):
    for name in ("date", "equity", "cash", "positions", "available_positions", "receivables",
                 "event_cash", "event_shares", "total_fees", "orders", "initial_account"):
        equal_values(left[name], right[name])


def test_feature_names_and_contract_are_stable_unique_and_do_not_mutate_base():
    _, base = stock_engine()
    original = deepcopy(base)
    contract = build_timing_contract(base, "relative_log")
    assert base == original
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 129
    assert FEATURE_NAMES[:20] == tuple(f"phase_{index:02d}" for index in range(20))
    assert len(ACCOUNT_FIELDS) == 14
    assert contract["timing"]["feature_names"] == list(FEATURE_NAMES)
    assert contract["timing"]["decoder_version"] == original.get("mapping_version", "stock_trade_lots_hold_buy_reduce_exit_residual_cash_v4")
    assert contract["timing"]["action_names"] == ["hold", "rebalance_m20"]
    assert contract["action_mode"] == "timing_control"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("basis", ["log_net", "relative_log"])
def test_calendar_macro_matches_the_unmodified_legacy_m20_path_exactly(mode, basis):
    macro = timing_env(mode=mode, basis=basis, days=43)
    legacy, contract = stock_engine(mode=mode, days=43)
    rule = EconomicRule(contract, "momentum_20", interval=20)
    vector, initial = macro.reset(seed=20260927)
    old_obs, old_initial = legacy.reset(seed=20260927)
    ledger_equal(initial, old_initial)
    assert macro.observation_space.contains(vector)
    assert macro.total_agent_steps == macro.total_benchmark_steps == 0
    steps = 0
    while True:
        assert np.argmax(vector[:20]) == steps % 20
        order = rule.act(old_obs, step=steps)
        vector, reward, done, truncated, info = macro.step(int(steps % 20 == 0))
        old_obs, _, old_done, old_truncated, old_info = legacy.step_order(order)
        ledger_equal(info, old_info)
        ledger_equal(info, info["timing"]["benchmark_ledger"])
        equal_values(info["action_diagnostics"], old_info["action_diagnostics"])
        np.testing.assert_array_equal(info["timing"]["derived_stock_raw_action"], order.diagnostics["latent"])
        assert reward == (0. if basis == "relative_log" else info["timing"]["agent_log_return"])
        assert (done, truncated) == (old_done, old_truncated)
        steps += 1
        assert macro.total_agent_steps == macro.total_benchmark_steps == steps
        if done or truncated:
            break
    assert steps == 42


def test_relative_reward_uses_two_independent_cost_ledgers_not_agent_resets():
    env = timing_env(basis="relative_log", days=8)
    _, initial = env.reset(seed=13)
    _, reward, _, _, info = env.step(0)
    benchmark = info["timing"]["benchmark_ledger"]
    assert info["orders"] == [] and info["total_fees"] == 0
    assert benchmark["orders"] and benchmark["total_fees"] > 0
    assert info["equity"] != benchmark["equity"]
    assert reward == pytest.approx(np.log(info["equity"] / initial["equity"]) - np.log(benchmark["equity"] / initial["equity"]))
    benchmark_positions = benchmark["positions"].copy()
    _, _, _, _, following = env.step(0)
    np.testing.assert_array_equal(following["timing"]["benchmark_ledger"]["positions"], benchmark_positions)
    assert following["timing"]["benchmark_ledger"]["total_fees"] == benchmark["total_fees"]
    assert env.env.env.portfolio is not env.benchmark.env.portfolio
    assert env.env.env.broker is not env.benchmark.env.broker


def test_reward_methods_have_identical_causal_observations_schedule_and_books():
    one = timing_env(basis="log_net", training=True, days=505)
    two = timing_env(basis="relative_log", training=True, days=505)
    for episode in range(16):
        a, ai = one.reset(seed=42 if not episode else None)
        b, bi = two.reset(seed=42 if not episode else None)
        np.testing.assert_array_equal(a, b)
        equal_values(ai["timing"]["schedule"], bi["timing"]["schedule"])
        ledger_equal(ai, bi)
        for macro in (0, 1, 0):
            a, _, _, _, ai = one.step(macro)
            b, _, _, _, bi = two.step(macro)
            np.testing.assert_array_equal(a, b)
            ledger_equal(ai, bi)
            ledger_equal(ai["timing"]["benchmark_ledger"], bi["timing"]["benchmark_ledger"])


def test_phase_blocks_cover_each_account_and_start_stratum_before_full_cycle():
    sampler = StratifiedTimingSampler(pd.bdate_range("2017-01-01", periods=600))
    sampler.seed(42)
    records = [sampler.next() for _ in range(48)]
    combinations = {(mode, stratum) for mode in MODES for stratum in DATE_STRATA}
    for offset in range(0, 48, 12):
        block = records[offset:offset + 12]
        assert {(row["initial_mode"], row["date_stratum"]) for row in block} == combinations
        assert len({row["phase_bin"] for row in block}) == 1
    assert len({(row["initial_mode"], row["date_stratum"], row["phase_bin"]) for row in records}) == 48
    assert min(Counter((row["initial_mode"], row["date_stratum"]) for row in records[:34]).values()) >= 2
    assert all(row["phase"] // 5 == row["phase_bin"] for row in records)
    assert all(row["start_index"] + 240 < len(sampler.dates) for row in records)
    assert all(row["start_index"] == sampler.maximum_start for row in records if row["date_stratum"] == "latest")
    sampler.seed(42)
    assert records == [sampler.next() for _ in range(48)]


def test_date_and_phase_rng_are_independent_of_global_randomness_and_account_draws():
    one = timing_env(training=True, days=505, stocks=24)
    two = timing_env(training=True, days=505, stocks=3)
    first = one.reset(seed=123)[1]
    second = two.reset(seed=123)[1]
    equal_values(first["timing"]["schedule"], second["timing"]["schedule"])
    for _ in range(15):
        np.random.normal(size=300)
        a, b = one.reset()[1], two.reset()[1]
        equal_values(a["timing"]["schedule"], b["timing"]["schedule"])
    assert one.schedule_audit()["benchmark_extra_environment_steps"] == 0


def test_training_reset_initializes_exactly_once_at_the_selected_date(monkeypatch):
    import src.training.timing_env as module
    original = module.initialize_account
    calls = []

    def record(portfolio, marks, valid, broker, mode, rng):
        calls.append(portfolio.current_date)
        return original(portfolio, marks, valid, broker, mode, rng)

    monkeypatch.setattr(module, "initialize_account", record)
    env = timing_env(training=True, days=505)
    for episode in range(13):
        vector, info = env.reset(seed=42 if not episode else None)
        schedule = info["timing"]["schedule"]
        assert calls[-1] == info["date"] == env.env.env.market.dates[schedule["start_index"]]
        assert len(calls) == episode + 1
        assert env.env.env._steps == env.benchmark.env._steps == 0
        assert not env.env.env._done and not env.benchmark.env._done
        assert len(env.env.env.history) == len(env.benchmark.env.history) == 1
        assert np.argmax(vector[:20]) == schedule["phase"]
        equal_values(env.env._observation, env.stock_observation)
        ledger_equal(info, info["timing"]["benchmark_ledger"])


def test_reset_deepcopies_locks_receivables_cost_basis_and_event_fields(monkeypatch):
    import src.training.timing_env as module
    original = module.initialize_account

    def initialize_events(portfolio, *args):
        result = original(portfolio, *args)
        asset = int(np.flatnonzero(portfolio.positions)[0])
        portfolio.apply_corporate_action(asset, .1, 2., portfolio.current_date + pd.Timedelta(days=4),
                                         portfolio.current_date + pd.Timedelta(days=7))
        return result

    monkeypatch.setattr(module, "initialize_account", initialize_events)
    env = timing_env(mode="equal")
    env.reset(seed=42)
    agent, benchmark = env.env.env.portfolio, env.benchmark.env.portfolio
    assert agent.receivables and agent.share_locks
    equal_values(vars(agent), vars(benchmark))
    assert agent.receivables is not benchmark.receivables and agent.share_locks is not benchmark.share_locks
    assert not np.shares_memory(agent.positions, benchmark.positions)
    assert not np.shares_memory(agent.event_shares, benchmark.event_shares)


def test_candidate_summary_uses_valid_momentum_and_code_ties_not_buy_mask():
    env = timing_env()
    env.reset(seed=42)
    stock = deepcopy(env.stock_observation)
    momentum = FEATURES.index("momentum_20")
    stock["stock_features"][:, -1, :] = np.arange(24)[:, None]
    stock["stock_features"][:, -1, momentum] = 0.
    stock["buy_mask"][:] = 0
    benchmark = deepcopy(stock)
    vector = timing_observation(stock, benchmark, 0, 1e6, 1e6, env.contract["stock_codes"])
    offset = FEATURE_NAMES.index("candidate_mean_return")
    np.testing.assert_array_equal(vector[offset:offset + 20], np.r_[np.repeat(9.5, momentum), 0., np.repeat(9.5, 19 - momentum)])
    stock["valid_mask"][:5] = 0
    benchmark["valid_mask"][:5] = 0
    vector = timing_observation(stock, benchmark, 0, 1e6, 1e6, env.contract["stock_codes"])
    assert vector[offset] == 14.


def test_future_feature_quote_changes_do_not_change_the_current_timing_vector():
    env = timing_env(mode="random")
    before, _ = env.reset(seed=13)
    for wrapper in (env.env, env.benchmark):
        market = wrapper.env.market
        following = market._indices[market.current_index] + 1
        market._features[following:] *= 1000
        market._fields["open"][following:] *= 100
        market._fields["close"][following:] *= 100
    stock = env.env.enrich(env.env.env._observation())
    benchmark = env.benchmark.enrich(env.benchmark.env._observation())
    after = timing_observation(stock, benchmark, env.phase, env._agent_equity, env._benchmark_equity,
                               env.contract["stock_codes"])
    np.testing.assert_array_equal(before, after)
    assert env.total_agent_steps == env.total_benchmark_steps == 0


def test_no_valid_stocks_have_finite_zero_market_candidate_and_held_summaries():
    env = timing_env()
    env.reset(seed=42)
    stock = deepcopy(env.stock_observation)
    stock["valid_mask"][:] = 0
    vector = timing_observation(stock, stock, 3, 1e6, 1e6, env.contract["stock_codes"])
    assert np.isfinite(vector).all()
    np.testing.assert_array_equal(vector[20:100], np.zeros(80))


def test_240_step_truncation_preserves_native_termination_and_counts_benchmark_steps():
    env = timing_env(training=True, days=505, stocks=3)
    env.reset(seed=42)
    for index in range(240):
        _, _, done, truncated, info = env.step(0)
        assert not done
        assert truncated == (index == 239)
        assert info["timing"]["benchmark_step_count"] == index + 1
    assert env.total_agent_steps == env.total_benchmark_steps == 240
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset()
    assert env.total_agent_steps == env.total_benchmark_steps == 240


@pytest.mark.parametrize("action", [-1, 2, .5, 1., True, [0], np.nan])
def test_invalid_macro_actions_are_rejected_without_advancing_either_book(action):
    env = timing_env()
    env.reset(seed=42)
    with pytest.raises(ValueError):
        env.step(action)
    assert env.total_agent_steps == env.total_benchmark_steps == 0


def test_unknown_configuration_or_unowned_sampler_and_shared_benchmark_are_rejected():
    env, contract = stock_engine()
    with pytest.raises(ValueError):
        TimingEnv(env, contract, benchmark=env)
    env.env.market.sampler = object()
    with pytest.raises(ValueError):
        TimingEnv(env, contract)
    with pytest.raises(ValueError):
        build_timing_contract(contract, "simple_net")
    with pytest.raises(ValueError):
        StratifiedTimingSampler(pd.bdate_range("2020-01-01", periods=242))


def test_different_benchmark_market_and_current_summary_are_rejected():
    env, contract = stock_engine()
    benchmark = deepcopy(env)
    benchmark.env.market._fields["open"][5, 0] += 1
    with pytest.raises(ValueError):
        TimingEnv(env, contract, benchmark=benchmark)
    macro = timing_env()
    macro.reset(seed=42)
    wrong = deepcopy(macro.benchmark_stock_observation)
    wrong["stock_features"][0, 0, 0] += 1
    with pytest.raises(ValueError):
        timing_observation(macro.stock_observation, wrong, 0, 1e6, 1e6, macro.contract["stock_codes"])


def test_training_schedule_overrides_are_rejected_and_eval_mode_is_explicit():
    training = timing_env(training=True, days=505)
    with pytest.raises(ValueError):
        training.reset(seed=42, options={"initial_mode": "cash"})
    evaluation = timing_env()
    _, info = evaluation.reset(seed=42, options={"initial_mode": "random"})
    assert info["initial_account"]["mode"] == "random" and info["timing"]["phase"] == 0
    with pytest.raises(ValueError):
        evaluation.reset(options={"start_index": 1})


@pytest.mark.parametrize("name,value", [("price_basis", "unadjusted"), ("logit_bound", 3.),
                                      ("top_n", 10), ("max_trade_weight", .1),
                                      ("max_commission_ratio", .01)])
def test_explicit_contract_must_match_the_stock_executor(name, value):
    env, contract = stock_engine()
    contract[name] = value
    with pytest.raises(ValueError):
        TimingEnv(env, contract)


def test_contract_calendar_margin_is_preserved_without_a_hidden_default_override():
    env, contract = stock_engine()
    contract = build_timing_contract(contract, "log_net", calendar_margin=3.)
    timing = TimingEnv(env, contract)
    assert timing.contract["timing"]["calendar_margin"] == 3.


@pytest.mark.parametrize("cache", ["prices", "valuation"])
def test_benchmark_derived_valuation_caches_cannot_differ_from_the_agent(cache):
    env, contract = stock_engine()
    reference = deepcopy(env)
    if cache == "prices":
        reference.env.market.prices.iloc[5, 0] += 1.
    else:
        reference.env.market._valuation[5, 0] += 1.
    with pytest.raises(ValueError):
        TimingEnv(env, contract, benchmark=reference)
