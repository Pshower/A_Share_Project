"""Versioned, typed-state paired labels with conservative fast isolation.

The production horizon is twenty trading transitions. Smaller explicit horizons
support synthetic tests; this module has no dataset reader, model or CLI runner.
"""

from copy import deepcopy
from datetime import date, datetime
import hashlib
import json
import math
import random
import struct
import sys

import numpy as np
import pandas as pd

from src.agents.trade_mapping import enrich_trade_observation
from src.training.timing_env import FEATURE_NAMES, PHASE_PERIOD, TimingEnv
from tools import counterfactual_storage_check as storage


LABEL_VERSION = "paired_m20_discounted_relative_log_v2"
STATE_CHECK_VERSION = "supported_schema_exact_phase_byte_union_v2"
EXACT_SAMPLER_CLASS = "tools.counterfactual_exact_schedule.ExactPhaseSampler"
GAMMA = .995
HORIZON = 20


class CounterfactualLabelError(RuntimeError):
    """No label was emitted; actual attempts and book steps remain auditable."""

    def __init__(self, message, *, counters, evidence):
        super().__init__(message)
        self.counters = dict(counters)
        self.evidence = evidence


def _require(condition, message, error=ValueError):
    if not condition:
        raise error(message)


def _scalar(value):
    if isinstance(value, np.generic):
        if isinstance(value, np.floating) and not np.isfinite(value):
            return {"nonfinite_numpy": value.dtype.str, "bits": value.tobytes().hex()}
        return _scalar(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_python": ">f8", "bits": struct.pack(">d", value).hex()}
    return value


def _execution_object(value):
    name = type(value).__module__ + "." + type(value).__qualname__
    if not hasattr(value, "__dict__"):
        return False
    if type(value).__module__.startswith(("src.", "gymnasium.")):
        return True
    if name == EXACT_SAMPLER_CLASS:
        from tools.counterfactual_exact_schedule import ExactPhaseSampler
        return type(value) is ExactPhaseSampler
    return False


def _immutable_atom(value):
    return (value is None or type(value) in (bool, int, float, str, bytes)
            or isinstance(value, (pd.Timestamp, datetime, date, pd.Timedelta, type))
            or isinstance(value, np.dtype) and _supported_dtype(value)
            or isinstance(value, np.generic) and value.dtype.fields is None
            and value.dtype.kind in "biufmMUS" and _supported_dtype(value.dtype))


def _supported_dtype(dtype):
    return (dtype.fields is None and dtype.subdtype is None and dtype.metadata is None and dtype.kind in "biufmMUSO"
            and not (dtype.kind in "biufmM" and dtype.itemsize > 8))


def assert_supported_state(value, seen=None):
    """Fail closed where the typed codec or storage walk lacks full semantics."""
    seen = set() if seen is None else seen
    if _immutable_atom(value):
        return
    identifier = id(value)
    _require(identifier not in seen, "Cyclic execution state is not supported", TypeError)
    seen.add(identifier)
    try:
        if isinstance(value, np.generic):
            raise TypeError("Unsupported NumPy scalar dtype: " + str(value.dtype))
        if isinstance(value, np.ndarray):
            _require(type(value) is np.ndarray, "ndarray subclasses are unsupported", TypeError)
            _require(_supported_dtype(value.dtype), "Complex or structured ndarray dtype is unsupported", TypeError)
            if value.dtype.hasobject:
                for item in value.flat:
                    assert_supported_state(item, seen)
        elif isinstance(value, pd.DataFrame):
            _require(type(value) is pd.DataFrame, "DataFrame subclasses are unsupported", TypeError)
            for block in value._mgr.blocks:
                _require(type(block.values) is np.ndarray,
                         "Pandas extension-array backing storage is unsupported", TypeError)
                assert_supported_state(block.values, seen)
            assert_supported_state(value.index, seen)
            assert_supported_state(value.columns, seen)
            assert_supported_state(value.attrs, seen)
        elif isinstance(value, pd.MultiIndex):
            assert_supported_state(list(value.names), seen)
            for level in value.levels:
                assert_supported_state(level, seen)
            for code in value.codes:
                assert_supported_state(code, seen)
        elif isinstance(value, pd.Index):
            _require(type(value) in (pd.Index, pd.RangeIndex, pd.DatetimeIndex, pd.TimedeltaIndex),
                     "Unsupported Pandas Index type", TypeError)
            _require(value.dtype.kind in "biufmMUSO" and not isinstance(value.dtype, pd.api.extensions.ExtensionDtype),
                     "Pandas extension Index dtype is unsupported", TypeError)
            _require(_immutable_atom(value.name), "Mutable Index name is unsupported", TypeError)
            array = value.to_numpy()
            _require(type(array) is np.ndarray and array.dtype.fields is None,
                     "Unsupported Index storage", TypeError)
            if array.dtype.hasobject:
                for item in array.flat:
                    _require(_immutable_key(item), "Mutable object Index elements are unsupported", TypeError)
        elif isinstance(value, np.random.Generator):
            assert_supported_state(value.bit_generator.state, seen)
        elif isinstance(value, np.random.RandomState):
            assert_supported_state(value.get_state(), seen)
        elif isinstance(value, dict):
            _require(type(value) is dict, "Custom dict subclasses are unsupported", TypeError)
            for key, item in value.items():
                _require(_immutable_key(key), "Mutable or custom dictionary keys are unsupported", TypeError)
                assert_supported_state(item, seen)
        elif isinstance(value, (tuple, list, set, frozenset)):
            _require(type(value) in (tuple, list, set, frozenset),
                     "Custom container subclasses are unsupported", TypeError)
            for item in value:
                if isinstance(value, (set, frozenset)):
                    _require(_immutable_key(item), "Mutable set/frozenset elements are unsupported", TypeError)
                assert_supported_state(item, seen)
        elif _execution_object(value):
            assert_supported_state(vars(value), seen)
        else:
            raise TypeError("Unsupported state type: " + type(value).__module__ + "." + type(value).__qualname__)
    finally:
        seen.remove(identifier)


def _immutable_key(value):
    return _immutable_atom(value) or type(value) is tuple and all(_immutable_key(item) for item in value)


def _encode(value, memo=None):
    """Value-complete typed state; aliases are checked separately by _clone."""
    memo = set() if memo is None else memo
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return [type(value).__module__ + "." + type(value).__qualname__, value.dtype.str, value.tobytes().hex()]
    if isinstance(value, np.bytes_):
        return ["numpy.bytes_", value.tobytes().hex()]
    if value is None or type(value) in (bool, int, str, float) or isinstance(value, np.generic):
        return [type(value).__module__ + "." + type(value).__qualname__, _scalar(value)]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return [type(value).__qualname__, value.isoformat()]
    if isinstance(value, pd.Timedelta):
        return ["Timedelta", value.value]
    if isinstance(value, np.dtype):
        return ["dtype", value.str]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, type):
        return ["type", value.__module__ + "." + value.__qualname__]
    identifier = id(value)
    _require(identifier not in memo, "Cyclic execution state is not supported")
    memo.add(identifier)
    try:
        return _encode_container(value, memo)
    finally:
        memo.remove(identifier)


def _encode_container(value, memo):
    if isinstance(value, np.ndarray):
        content = (_encode(value.tolist(), memo) if value.dtype.hasobject else
                   hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
        return ["array", value.dtype.str, list(value.shape), content]
    if isinstance(value, pd.DataFrame):
        dtypes = list(value.dtypes)
        homogeneous = bool(dtypes) and all(isinstance(dtype, np.dtype) and not dtype.hasobject
                                           and dtype == dtypes[0] for dtype in dtypes)
        if homogeneous:
            array = value.to_numpy(copy=False)
            _require(array.dtype == dtypes[0], "Homogeneous DataFrame changed dtype during encoding")
            data = ["homogeneous_numpy", _encode(array)]
        else:
            data = ["typed_columns", [_encode(value.iloc[:, column].to_numpy())
                                     for column in range(len(value.columns))]]
        return ["DataFrame", _encode(value.index, memo), _encode(value.columns, memo),
                [str(dtype) for dtype in dtypes], data, _encode(value.attrs, memo)]
    if isinstance(value, pd.MultiIndex):
        return ["MultiIndex", _encode(list(value.names), memo),
                [_encode(level, memo) for level in value.levels],
                [_encode(code, memo) for code in value.codes]]
    if isinstance(value, pd.Index):
        return [type(value).__qualname__, str(value.dtype), _encode(value.name, memo),
                _encode(value.to_numpy())]
    if isinstance(value, np.random.Generator):
        return ["Generator", type(value.bit_generator).__name__, _encode(value.bit_generator.state, memo)]
    if isinstance(value, np.random.RandomState):
        return ["RandomState", _encode(value.get_state(), memo)]
    if isinstance(value, dict):
        return ["dict", [[_encode(key, memo), _encode(item, memo)]
                         for key, item in sorted(value.items(), key=lambda pair: (type(pair[0]).__name__, str(pair[0])))]]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [_encode(item, memo) for item in value]]
    if isinstance(value, (set, frozenset)):
        return [type(value).__name__, sorted((_encode(item) for item in value), key=lambda item: json.dumps(item))]
    module = type(value).__module__
    if _execution_object(value):
        return [module + "." + type(value).__qualname__, _encode(vars(value), memo)]
    raise TypeError("Unsupported state type: " + module + "." + type(value).__qualname__)


def _digest(value):
    raw = json.dumps(_encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def snapshot_state_fingerprint(env):
    """Hash all instance execution values, market arrays and local RNG states."""
    _require(isinstance(env, TimingEnv), "A full TimingEnv is required")
    assert_supported_state(env)
    return _digest(env)


def _global_rng_state():
    result = dict(python=random.getstate(), numpy=np.random.get_state())
    torch = sys.modules.get("torch")
    if torch is not None:
        result["torch_cpu"] = torch.get_rng_state().cpu().numpy().copy()
        if torch.cuda.is_initialized():
            result["torch_already_initialized_cuda"] = [state.cpu().numpy().copy()
                for state in torch.cuda.get_rng_state_all()]
    return result


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        return _scalar(value)
    raise TypeError("Unsupported label payload type: " + type(value).__qualname__)


def _objects(value, seen=None, mutable=None, arrays=None):
    seen = set() if seen is None else seen
    mutable = set() if mutable is None else mutable
    arrays = [] if arrays is None else arrays
    if id(value) in seen:
        return mutable, arrays
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        mutable.add(id(value))
        arrays.append(value)
        if value.dtype.hasobject:
            for item in value.flat:
                _objects(item, seen, mutable, arrays)
    elif isinstance(value, pd.DataFrame):
        mutable.add(id(value))
        for block in value._mgr.blocks:
            _objects(np.asarray(block.values), seen, mutable, arrays)
        _objects(value.attrs, seen, mutable, arrays)
    elif isinstance(value, pd.Index):
        # Index values are immutable by Pandas API; cloned indexes are detached too.
        pass
    elif isinstance(value, (np.random.Generator, np.random.RandomState)):
        mutable.add(id(value))
        if isinstance(value, np.random.Generator):
            mutable.add(id(value.bit_generator))
    elif isinstance(value, (dict, list, set)):
        mutable.add(id(value))
        children = value.values() if isinstance(value, dict) else value
        for item in children:
            _objects(item, seen, mutable, arrays)
    elif isinstance(value, tuple):
        for item in value:
            _objects(item, seen, mutable, arrays)
    elif _execution_object(value):
        mutable.add(id(value))
        _objects(vars(value), seen, mutable, arrays)
    return mutable, arrays


def _independent(left, right):
    left_ids, left_arrays = _objects(left)
    right_ids, right_arrays = _objects(right)
    _require(not left_ids.intersection(right_ids), "Execution books share mutable objects", RuntimeError)
    if not storage.array_bounds_prove_disjoint(left_arrays, right_arrays):
        _require(not any(np.shares_memory(old, new) for old in left_arrays for new in right_arrays),
                 "Execution books share mutable array storage", RuntimeError)


def _clone(env, fingerprint):
    clone = deepcopy(env)
    _independent(env, clone)
    # Pandas deep copies deliberately share immutable Index storage. Detach it
    # explicitly as well, without introducing a shared-market optimization.
    for wrapper in (clone.env, clone.benchmark):
        market = wrapper.env.market
        for name, value in vars(market).items():
            if isinstance(value, pd.DataFrame):
                value.index = value.index.copy(deep=True)
                value.columns = value.columns.copy(deep=True)
            elif isinstance(value, pd.Index):
                setattr(market, name, value.copy(deep=True))
    _require(snapshot_state_fingerprint(clone) == fingerprint,
             "Deep copy changed execution state or graph aliases", RuntimeError)
    return clone


def _validate(env, horizon, train_end, gamma):
    _require(type(horizon) is int and horizon > 0, "horizon must be a positive integer")
    _require(not isinstance(gamma, (bool, np.bool_)) and isinstance(gamma, (float, np.floating))
             and np.isfinite(gamma) and float(gamma) == GAMMA, "Only registered gamma=.995 is supported")
    _require(isinstance(env, TimingEnv) and env.training is True, "Only a training TimingEnv is allowed")
    cutoff = pd.Timestamp(train_end)
    _require(not pd.isna(cutoff) and cutoff.tz is None and cutoff == cutoff.normalize(),
             "train_end must be a timezone-naive trading-date cutoff")
    _require(env.stock_observation is not None and env.benchmark_stock_observation is not None,
             "Reset the source environment before labeling")
    _require(env._done is False and type(env.phase) is int and 0 <= env.phase < PHASE_PERIOD,
             "Source timing state is done or its phase is invalid")
    _independent(env.env, env.benchmark)
    for wrapper, stored, equity in ((env.env, env.stock_observation, env._agent_equity),
                                    (env.benchmark, env.benchmark_stock_observation, env._benchmark_equity)):
        base, market = wrapper.env, wrapper.env.market
        _require(base._done is False and type(base._steps) is int and base._steps >= 0,
                 "Source stock engine is done or has an invalid clock")
        _require(type(market.current_index) is int and 0 <= market.current_index < market.n_dates,
                 "Source market cursor is invalid")
        _require(market.current_index + horizon < market.n_dates, "Insufficient market transitions for the full horizon")
        _require(base.max_steps is not None and base._steps + horizon <= base.max_steps,
                 "The full horizon would cross the original TimeLimit")
        _require(market.dates.max() <= cutoff and market.panel.index.max() <= cutoff,
                 "Provider is not physically restricted to train_end")
        _require(market.dates[market.current_index + horizon] <= cutoff, "Label endpoint exceeds train_end")
        _require(market.corporate_actions.empty or market.corporate_actions.date.max() <= cutoff,
                 "Provider contains corporate-action ex-dates after train_end")
        portfolio = base.portfolio
        _require(portfolio.current_date == market.current_date, "Portfolio and market dates differ")
        for name in ("positions", "available_positions", "cost_basis", "event_shares"):
            values = getattr(portfolio, name)
            _require(isinstance(values, np.ndarray) and values.shape == (market.n_stocks,)
                     and np.isfinite(values).all() and (values >= 0).all(), "Malformed portfolio " + name)
        _require((portfolio.available_positions <= portfolio.positions).all()
                 and np.isfinite([portfolio.cash, portfolio.total_fees, portfolio.event_cash]).all()
                 and min(portfolio.cash, portfolio.total_fees, portfolio.event_cash) >= 0,
                 "Malformed portfolio cash, fees or availability")
        for due, amount in portfolio.receivables:
            _require(not pd.isna(pd.Timestamp(due)) and pd.Timestamp(due) > portfolio.current_date
                     and np.isfinite(amount) and amount > 0, "Malformed outstanding receivable")
        locked = np.zeros(market.n_stocks)
        for due, asset, quantity in portfolio.share_locks:
            _require(not pd.isna(pd.Timestamp(due)) and pd.Timestamp(due) > portfolio.current_date
                     and type(asset) is int and 0 <= asset < market.n_stocks
                     and np.isfinite(quantity) and quantity > 0 and quantity == np.floor(quantity),
                     "Malformed outstanding share lock")
            locked[asset] += quantity
        _require((portfolio.available_positions <= portfolio.positions - locked + 1e-8).all(),
                 "Share locks contradict available shares")
        broker = base.broker
        _require((broker._date is None) == (broker._used_volume is None), "Malformed broker session state")
        if broker._date is not None:
            _require(pd.Timestamp(broker._date) <= market.current_date
                     and isinstance(broker._used_volume, np.ndarray)
                     and broker._used_volume.shape == (market.n_stocks,)
                     and np.isfinite(broker._used_volume).all() and (broker._used_volume >= 0).all(),
                     "Malformed broker volume usage")
        _require(np.isfinite(equity) and equity > 0 and base._info()["equity"] == equity,
                 "Source NAV is nonpositive, nonfinite or stale")
        actual = enrich_trade_observation(base._observation(), portfolio,
            market.prices.iloc[market.current_index].to_numpy(), base.broker.config, wrapper.top_n,
            wrapper.max_trade_weight, wrapper.max_commission_ratio, market.dates[market.current_index + 1])
        _require(_digest(actual) == _digest(stored) == _digest(wrapper._observation),
                 "Source stock observation is stale or corrupted")
        _require(np.array_equal(wrapper.action_space.legal, actual["legal_directions"])
                 and np.array_equal(wrapper.action_space.bounds, actual["trade_lot_bounds"]),
                 "Source legal-action cache is stale")
    _require(env.env.env._steps == env.benchmark.env._steps
             and _digest(env.env.env.market) == _digest(env.benchmark.env.market),
             "Source agent and benchmark clocks or market values differ")
    observation = env.observe()
    _require(observation.shape == (len(FEATURE_NAMES),) and np.isfinite(observation).all(),
             "Current timing observation must be finite and have 129 features")
    return cutoff, observation


def _branch(env, fingerprint, first_action, horizon, counters, evidence):
    branch = _clone(env, fingerprint)
    starts = (branch.env.env._steps, branch.benchmark.env._steps)
    try:
        return _run_branch(branch, first_action, horizon, counters)
    finally:
        final = (branch.env.env._steps, branch.benchmark.env._steps)
        deltas = tuple(after - before for before, after in zip(starts, final))
        _require(all(type(value) is int and value >= 0 for value in deltas),
                 "Branch base clocks moved backwards or lost their integer type", RuntimeError)
        counters["branch_agent_steps"] += deltas[0]
        counters["branch_benchmark_steps"] += deltas[1]
        evidence.append(dict(first_action=first_action, initial_base_steps=list(starts),
            final_base_steps=list(final), completed_agent_steps=deltas[0], completed_benchmark_steps=deltas[1]))


def _run_branch(branch, first_action, horizon, counters):
    initial_equity = float(branch._agent_equity)
    initial_benchmark = float(branch._benchmark_equity)
    starts = (branch.total_agent_steps, branch.total_benchmark_steps,
              branch.env.env._steps, branch.benchmark.env._steps,
              branch.env.env.market.current_index, branch.benchmark.env.market.current_index)
    steps, relatives, benchmark_steps = [], [], []
    for index in range(horizon):
        phase = branch.phase
        action = first_action if index == 0 else int(phase == 0)
        counters["macro_attempts"] += 1
        _, _, terminated, truncated, info = branch.step(action)
        timing = info["timing"]
        clocks = (branch.total_agent_steps, branch.total_benchmark_steps,
                  branch.env.env._steps, branch.benchmark.env._steps,
                  branch.env.env.market.current_index, branch.benchmark.env.market.current_index)
        _require(clocks == tuple(value + index + 1 for value in starts),
                 "Actual branch or benchmark transition counters diverged", RuntimeError)
        _require(timing["phase_before"] == phase and branch.phase == (phase + 1) % PHASE_PERIOD,
                 "Continuation changed the original calendar phase", RuntimeError)
        _require(not (terminated or truncated) or index == horizon - 1,
                 "Branch ended before its full horizon; resetting is forbidden", RuntimeError)
        _require(not truncated or index == horizon - 1,
                 "Branch crossed its TimeLimit", RuntimeError)
        relative = float(timing["agent_log_return"] - timing["benchmark_log_return"])
        _require(np.isfinite(relative) and np.isfinite(info["equity"]) and info["equity"] > 0,
                 "Branch produced invalid rewards or NAV", RuntimeError)
        relatives.append(relative)
        benchmark_steps.append(_digest(timing["benchmark_ledger"]))
        steps.append(dict(index=index, phase=phase, macro_action=action,
            terminated=bool(terminated), truncated=bool(truncated), relative_log_reward=relative,
            agent_ledger=_json_safe({key: value for key, value in info.items()
                if key not in {"timing", "timing_stock_observation", "timing_benchmark_stock_observation"}}),
            benchmark_ledger=_json_safe(timing["benchmark_ledger"])))
    return dict(initial_agent_nav=initial_equity, initial_benchmark_nav=initial_benchmark,
        final_agent_nav=float(branch._agent_equity), final_benchmark_nav=float(branch._benchmark_equity),
        actual_agent_steps=branch.total_agent_steps - starts[0],
        actual_benchmark_steps=branch.total_benchmark_steps - starts[1],
        endpoint=str(branch.env.env.market.current_date.date()), steps=steps), np.asarray(relatives), benchmark_steps


def paired_rule_label(env, *, horizon=HORIZON, train_end, gamma=GAMMA):
    """Fork actual train state twice; return causal input and separate future labels.

    Both branches use the fixed original M20 calendar after their first action.
    This is a historical finite-horizon Q_rule difference, not Q_pi, GAE or a
    likelihood observation. No model, optimizer, reset or dataset read is used.
    """
    counters = dict(branch_agent_steps=0, branch_benchmark_steps=0, macro_attempts=0,
        collection_steps=0, rl_steps=0, nn_forwards=0, optimizer_updates=0)
    evidence = dict(branches=[], label_emitted=False,
        book_steps_semantics="actual base._steps deltas; macro attempts include potentially partial failed transitions",
        partial_branch_effects_are_not_rolled_back=True)
    before = after = None
    global_state = _global_rng_state()
    global_before = _digest(global_state)
    evidence["global_rng_scope"] = list(global_state)
    failure = None
    try:
        before = snapshot_state_fingerprint(env)
        cutoff, observation = _validate(env, horizon, train_end, gamma)
        hold, hold_rewards, hold_benchmark = _branch(env, before, 0, horizon, counters, evidence["branches"])
        rebalance, rebalance_rewards, rebalance_benchmark = _branch(env, before, 1, horizon, counters, evidence["branches"])
        _require(hold_benchmark == rebalance_benchmark
                 and hold["initial_benchmark_nav"] == rebalance["initial_benchmark_nav"]
                 and hold["final_benchmark_nav"] == rebalance["final_benchmark_nav"],
                 "Independent shadow benchmark paths differ between branches", RuntimeError)
        _require(hold["initial_agent_nav"] == rebalance["initial_agent_nav"]
                 and hold["endpoint"] == rebalance["endpoint"],
                 "Paired branches do not share initial NAV and endpoint", RuntimeError)
        difference = rebalance_rewards - hold_rewards
        primary = float(np.dot(np.power(float(gamma), np.arange(horizon)), difference))
        terminal = float(np.log(rebalance["final_agent_nav"]) - np.log(hold["final_agent_nav"]))
        _require(np.isclose(difference.sum(), terminal, rtol=0., atol=5e-13),
                 "Undiscounted relative rewards do not telescope to paired terminal NAV", RuntimeError)
        _require(counters["branch_agent_steps"] == counters["branch_benchmark_steps"] == 2 * horizon
                 and counters["macro_attempts"] == 2 * horizon,
                 "Actual branch experience differs from its complete paired horizon", RuntimeError)
        result = dict(schema_version=1, label_version=LABEL_VERSION,
            input=dict(observation=observation.tolist(), feature_names=list(FEATURE_NAMES),
                decision_date=str(env.env.env.market.current_date.date()), phase=env.phase),
            labels=dict(horizon=horizon, gamma=float(gamma), train_end=str(cutoff.date()),
                discounted_relative_delta=primary, terminal_log_nav_ratio=terminal,
                undiscounted_relative_delta=float(difference.sum()),
                per_step_relative_reward_difference=difference.tolist(), branches=dict(hold=hold, rebalance=rebalance),
                semantics="finite historical Q_rule with fixed M20 continuation; not Q_pi, PPO GAE or new market data"),
            provenance=dict(source_before_sha256=before, contract_sha256=_digest(env.contract),
                state_check_version=STATE_CHECK_VERSION, storage_check_version=storage.STORAGE_CHECK_VERSION,
                market_sha256=_digest(env.env.env.market), price_basis=env.env.env.market.price_basis,
                benchmark_paths_equal=True, clone_initial_state_equal=True, clone_storage_independent=True,
                source_unchanged=True, global_rng_unchanged=True, global_rng_scope=list(global_state),
                current_close_observation_verified=True, pipeline_feature_causality_not_certified=True,
                research_only=True, actionable=False), counters=dict(counters))
        json.dumps(result, allow_nan=False)
    except Exception as error:
        failure = error
    finally:
        try:
            if before is not None:
                after = snapshot_state_fingerprint(env)
                _require(before == after, "Source execution state changed during paired labeling", RuntimeError)
            _require(global_before == _digest(_global_rng_state()),
                     "Global RNG changed during paired labeling", RuntimeError)
        except Exception as guard_error:
            evidence["preceding_error"] = None if failure is None else str(failure)
            failure = guard_error
    if failure is not None:
        evidence.update(source_before_sha256=before, source_after_sha256=after,
            source_unchanged=None if before is None else before == after,
            global_rng_unchanged=global_before == _digest(_global_rng_state()),
            reason_type=type(failure).__name__, partial_branch_effects_possible=counters["macro_attempts"] > 0)
        raise CounterfactualLabelError(str(failure), counters=counters, evidence=evidence) from failure
    result["provenance"]["source_after_sha256"] = after
    return result
