"""Pure synthetic full-engine counterfactual tests; no model or real dataset."""

from copy import deepcopy
from dataclasses import asdict
import json
import random

import numpy as np
import pandas as pd
import pytest

from src.agents.trade_mapping import TradeActionWrapper
from src.data.preprocess import FEATURES
from src.envs.broker import BrokerConfig
from src.envs.trading import TradingEnv
from src.training.exploration_env import ExplorationMarket
from src.training.timing_env import TimingEnv
from tools.archive import timing_counterfactual as tool


def make_env(*, days=40, episode_steps=12, mode="cash", phase=0,
             price_basis="hfq_research", fees=True, flat=False):
    dates = pd.bdate_range("2020-01-01", periods=days + 4)
    codes = [f"{index:06d}" for index in range(24)]
    time, asset, feature = np.ogrid[:len(dates), :len(codes), :len(FEATURES)]
    values = .01 * time + asset / 20 + feature / 100
    panel = pd.DataFrame(values.reshape(len(dates), -1), index=dates,
        columns=pd.MultiIndex.from_product([codes, FEATURES]))
    quotes = pd.DataFrame(index=dates,
        columns=pd.MultiIndex.from_product([codes, ("open", "close", "volume")]), dtype=float)
    for index, code in enumerate(codes):
        close = np.full(len(dates), 10 + index / 10) if flat else 10 + index / 10 + np.arange(len(dates)) / 100
        quotes[code, "open"] = close if flat else close + .02
        quotes[code, "close"] = close
        quotes[code, "volume"] = 1e8
    market = ExplorationMarket(panel, FEATURES, price_path=quotes, price_basis=price_basis,
        start_date=dates[4], common_lookback=5)
    broker = BrokerConfig(commission=.0003 if fees else 0., min_commission=5. if fees else 0.,
        sell_tax=.001 if fees else 0., slippage=.001 if fees else 0.)
    stock = TradeActionWrapper(TradingEnv(market, broker_config=broker, max_steps=episode_steps,
        initial_portfolio="mixed"), top_n=20)
    contract = dict(stock_codes=codes, feature_cols=list(FEATURES), logit_bound=8.,
        broker=asdict(broker), price_basis=price_basis)
    env = TimingEnv(stock, contract, training=True, reward_basis="relative_log", episode_steps=episode_steps)
    env.reset(seed=42)
    env.sampler.block = [(mode, "early", phase // 5)]
    env.reset()
    env.phase = phase
    return env


def label(env, horizon=3, **kwargs):
    return tool.paired_rule_label(env, horizon=horizon,
        train_end=env.env.env.market.panel.index[-1], **kwargs)


def test_discounted_target_uses_actual_paired_rewards_not_terminal_or_gae():
    env = make_env()
    result = label(env, horizon=5)
    targets = result["labels"]
    differences = np.asarray(targets["per_step_relative_reward_difference"])
    expected = sum(.995 ** index * value for index, value in enumerate(differences))
    assert targets["discounted_relative_delta"] == pytest.approx(expected, abs=1e-16)
    hold, rebalance = (targets["branches"][key] for key in ("hold", "rebalance"))
    terminal = np.log(rebalance["final_agent_nav"] / hold["final_agent_nav"])
    assert targets["terminal_log_nav_ratio"] == pytest.approx(terminal, abs=1e-14)
    assert differences.sum() == pytest.approx(terminal, abs=1e-14)
    assert abs(expected - terminal) > 1e-8
    assert result["counters"] == dict(branch_agent_steps=10, branch_benchmark_steps=10, macro_attempts=10,
        collection_steps=0, rl_steps=0, nn_forwards=0, optimizer_updates=0)
    assert len(result["input"]["observation"]) == 129
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("phase,expected", [(0, [0, 0, 0]), (19, [0, 1, 0]), (18, [0, 0, 1])])
def test_first_action_does_not_restart_the_original_calendar(phase, expected):
    result = label(make_env(phase=phase))
    branches = result["labels"]["branches"]
    assert [row["macro_action"] for row in branches["hold"]["steps"]] == expected
    assert [row["macro_action"] for row in branches["rebalance"]["steps"]] == [1, *expected[1:]]
    assert [row["phase"] for row in branches["hold"]["steps"]] == [(phase + index) % 20 for index in range(3)]
    assert [row["benchmark_ledger"] for row in branches["hold"]["steps"]] == [
        row["benchmark_ledger"] for row in branches["rebalance"]["steps"]]


@pytest.mark.parametrize("mode", ["cash", "equal", "concentrated", "random"])
def test_source_and_global_rng_are_unchanged_and_labeling_is_deterministic(mode):
    env = make_env(mode=mode)
    state = tool.snapshot_state_fingerprint(env)
    python_rng, numpy_rng = random.getstate(), deepcopy(np.random.get_state())
    first, second = label(env), label(env)
    assert first == second
    assert state == tool.snapshot_state_fingerprint(env) == first["provenance"]["source_after_sha256"]
    assert random.getstate() == python_rng
    assert np.random.get_state()[0] == numpy_rng[0]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
    assert np.random.get_state()[2:] == numpy_rng[2:]


@pytest.mark.parametrize("fees", [False, True])
def test_flat_prices_capture_real_execution_costs(fees):
    targets = label(make_env(flat=True, fees=fees))["labels"]
    if fees:
        assert targets["discounted_relative_delta"] < 0
        assert targets["branches"]["rebalance"]["steps"][0]["agent_ledger"]["total_fees"] > 0
    else:
        assert targets["discounted_relative_delta"] == 0
        assert targets["terminal_log_nav_ratio"] == 0


def test_no_fill_due_to_missing_next_open_is_economically_equal_and_json_safe():
    env = make_env(phase=1)
    for wrapper in (env.env, env.benchmark):
        market = wrapper.env.market
        market._fields["open"][market._indices[market.current_index + 1]] = np.nan
    targets = label(env)["labels"]
    assert targets["discounted_relative_delta"] == 0
    assert targets["terminal_log_nav_ratio"] == 0
    orders = targets["branches"]["rebalance"]["steps"][0]["agent_ledger"]["orders"]
    assert orders and all(order["filled_shares"] == 0 for order in orders)
    assert any("nonfinite_python" in order["price"] for order in orders)


def test_dividends_locked_bonus_shares_and_t_plus_one_survive_cloning():
    env = make_env(mode="equal", price_basis="unadjusted", phase=19)
    source = env.env.env
    asset = int(np.flatnonzero(source.portfolio.positions)[0])
    current = source.market.current_date
    for wrapper in (env.env, env.benchmark):
        portfolio = wrapper.env.portfolio
        portfolio.apply_corporate_action(asset, .1, 2., current + pd.Timedelta(days=3),
            current + pd.Timedelta(days=6))
        wrapper._observation = wrapper.enrich(wrapper.env._observation())
    env.stock_observation, env.benchmark_stock_observation = env.env._observation, env.benchmark._observation
    env._agent_equity = env.env.env._info()["equity"]
    env._benchmark_equity = env.benchmark.env._info()["equity"]
    before = tool.snapshot_state_fingerprint(env)
    assert source.portfolio.receivables and source.portfolio.share_locks
    result = label(env, horizon=5)
    assert before == tool.snapshot_state_fingerprint(env)
    first = result["labels"]["branches"]["rebalance"]["steps"][0]["agent_ledger"]
    assert first["available_positions"][asset] < first["positions"][asset]
    assert any(row["agent_ledger"]["event_cash"] > 0 for row in result["labels"]["branches"]["hold"]["steps"])


@pytest.mark.parametrize("mutation", [
    lambda env: setattr(env, "_done", True),
    lambda env: setattr(env.env.env, "_done", True),
    lambda env: setattr(env, "stock_observation", None),
    lambda env: setattr(env, "training", False),
    lambda env: setattr(env, "phase", True),
    lambda env: setattr(env, "_agent_equity", 0.),
    lambda env: setattr(env, "_agent_equity", float("nan")),
    lambda env: env.env.env.portfolio.positions.__setitem__(0, np.nan),
    lambda env: env.stock_observation["stock_features"].__setitem__((0, 0, 0), 123.),
    lambda env: env.env.action_space.bounds.__setitem__((0, 0), 999),
])
def test_rejects_unreset_done_invalid_or_stale_source_before_branching(monkeypatch, mutation):
    env = make_env()
    mutation(env)
    monkeypatch.setattr(tool, "_branch", lambda *args: pytest.fail("invalid source reached branch"))
    with pytest.raises(tool.CounterfactualLabelError):
        label(env)


@pytest.mark.parametrize("horizon", [True, np.int64(3), 0, -1, 3.0])
def test_rejects_unregistered_horizon_types(horizon):
    with pytest.raises(tool.CounterfactualLabelError, match="horizon"):
        label(make_env(), horizon=horizon)


@pytest.mark.parametrize("gamma", [True, np.bool_(True), 1., .95, float("nan")])
def test_rejects_wrong_gamma(gamma):
    with pytest.raises(tool.CounterfactualLabelError, match="gamma"):
        label(make_env(), gamma=gamma)


def test_rejects_uncropped_provider_even_when_the_endpoint_would_be_train_only():
    env = make_env()
    cutoff = env.env.env.market.dates[env.env.env.market.current_index + 5]
    with pytest.raises(tool.CounterfactualLabelError, match="physically"):
        tool.paired_rule_label(env, horizon=3, train_end=cutoff)


def test_rejects_time_limit_crossing_and_preserves_state():
    env = make_env(episode_steps=4)
    env.step(0)
    before = tool.snapshot_state_fingerprint(env)
    with pytest.raises(tool.CounterfactualLabelError, match="TimeLimit"):
        label(env, horizon=5)
    assert before == tool.snapshot_state_fingerprint(env)


def test_rejects_early_done_without_resetting_or_touching_the_source(monkeypatch):
    env = make_env()
    before = tool.snapshot_state_fingerprint(env)
    original = TimingEnv.step

    def early(self, action):
        observation, reward, _, truncated, info = original(self, action)
        return observation, reward, True, truncated, info

    monkeypatch.setattr(TimingEnv, "step", early)
    with pytest.raises(tool.CounterfactualLabelError, match="before its full horizon") as caught:
        label(env)
    assert caught.value.counters["branch_agent_steps"] == caught.value.counters["branch_benchmark_steps"] == 1
    assert caught.value.counters["macro_attempts"] == 1
    assert caught.value.evidence["label_emitted"] is False
    assert before == tool.snapshot_state_fingerprint(env)


def test_clone_rejects_deliberately_shared_mutable_storage(monkeypatch):
    env = make_env()
    monkeypatch.setattr(tool, "deepcopy", lambda source: source)
    with pytest.raises(tool.CounterfactualLabelError, match="share mutable") as caught:
        label(env)
    assert caught.value.counters["macro_attempts"] == caught.value.counters["branch_agent_steps"] == 0


def test_default_production_horizon_is_twenty_real_transitions():
    env = make_env(days=70, episode_steps=28, phase=19)
    result = tool.paired_rule_label(env, train_end=env.env.env.market.panel.index[-1])
    assert result["labels"]["horizon"] == 20
    assert result["counters"]["branch_agent_steps"] == result["counters"]["branch_benchmark_steps"] == 40
    assert [row["macro_action"] for row in result["labels"]["branches"]["hold"]["steps"]] == [0, 1] + [0] * 18


def test_full_horizon_may_end_exactly_at_time_limit_without_bootstrap_or_reset():
    result = label(make_env(episode_steps=3))
    for branch in result["labels"]["branches"].values():
        assert branch["steps"][-1]["truncated"] is True
        assert not any(row["truncated"] for row in branch["steps"][:-1])
        assert branch["actual_agent_steps"] == 3


def test_full_horizon_may_end_at_last_train_day_with_natural_termination():
    env = make_env(days=8, episode_steps=5)
    env.sampler.block = [("cash", "latest", 0)]
    env.reset()
    result = label(env, horizon=5)
    for branch in result["labels"]["branches"].values():
        assert branch["steps"][-1]["terminated"] is True
        assert branch["steps"][-1]["truncated"] is False
        assert branch["endpoint"] == str(env.env.env.market.panel.index[-1].date())


@pytest.mark.parametrize("mutation", [
    lambda env: env.env.env.portfolio.cost_basis.__setitem__(0, 1.),
    lambda env: env.env.env.portfolio.receivables.append((env.env.env.market.current_date + pd.Timedelta(days=7), 1.)),
    lambda env: env.env.env.portfolio.share_locks.append((env.env.env.market.current_date + pd.Timedelta(days=7), 0, 1.)),
    lambda env: env.env.env.broker.orders.append({"price": float("nan"), "filled_shares": 0.}),
    lambda env: setattr(env.env.env.broker, "_used_volume", np.zeros(24)),
    lambda env: env.env.env.history.append({"audit_event": 1}),
    lambda env: env.sampler.date_rng.random(),
    lambda env: env.env._top_rng.random(),
    lambda env: env.env.env.market._fields["open"].__setitem__((10, 0), 55.),
    lambda env: setattr(env, "total_agent_steps", 1),
])
def test_fingerprint_covers_execution_state_beyond_observation(mutation):
    env = make_env()
    before = tool.snapshot_state_fingerprint(env)
    mutation(env)
    assert tool.snapshot_state_fingerprint(env) != before


@pytest.mark.parametrize("kind", ["receivable", "lock", "volume"])
def test_malformed_hidden_execution_state_is_rejected_before_branch(kind):
    env = make_env()
    if kind == "receivable":
        env.env.env.portfolio.receivables.append((env.env.env.market.current_date + pd.Timedelta(days=3), -1.))
    elif kind == "lock":
        env.env.env.portfolio.share_locks.append((env.env.env.market.current_date + pd.Timedelta(days=3), 0, 100.))
    else:
        env.env.env.broker._used_volume = np.zeros(24)
    with pytest.raises(tool.CounterfactualLabelError, match="receivable|lock|broker"):
        label(env)


def test_independently_replayed_environments_have_the_same_value_fingerprint():
    one, two = make_env(), make_env()
    for action in (0, 1, 0):
        one.step(action)
        two.step(action)
    assert tool.snapshot_state_fingerprint(one) == tool.snapshot_state_fingerprint(two)
    assert one.env.env.portfolio is not two.env.env.portfolio


def test_failure_after_agent_before_benchmark_preserves_actual_asymmetric_cost(monkeypatch):
    env = make_env()
    env.benchmark._test_benchmark = True
    before = tool.snapshot_state_fingerprint(env)
    original = TradeActionWrapper.step_order

    def fail_benchmark(self, order):
        if getattr(self, "_test_benchmark", False):
            raise RuntimeError("synthetic benchmark execution failure")
        return original(self, order)

    monkeypatch.setattr(TradeActionWrapper, "step_order", fail_benchmark)
    with pytest.raises(tool.CounterfactualLabelError, match="benchmark execution failure") as caught:
        label(env)
    assert caught.value.counters["branch_agent_steps"] == 1
    assert caught.value.counters["branch_benchmark_steps"] == 0
    assert caught.value.counters["macro_attempts"] == 1
    assert caught.value.evidence["source_unchanged"] is True
    assert caught.value.evidence["partial_branch_effects_possible"] is True
    assert before == tool.snapshot_state_fingerprint(env)


def test_torch_cpu_rng_perturbation_is_rejected_without_silent_restore(monkeypatch):
    import torch

    env = make_env()
    state = torch.get_rng_state().clone()
    original = TimingEnv.step

    def perturb(self, action):
        torch.rand(1)
        return original(self, action)

    monkeypatch.setattr(TimingEnv, "step", perturb)
    try:
        with pytest.raises(tool.CounterfactualLabelError, match="Global RNG changed") as caught:
            label(env)
        assert caught.value.counters["branch_agent_steps"] == caught.value.counters["branch_benchmark_steps"] == 6
        assert "torch_cpu" in caught.value.evidence["global_rng_scope"]
        assert caught.value.evidence["global_rng_unchanged"] is False
    finally:
        torch.set_rng_state(state)


def test_causal_scope_does_not_certify_arbitrary_feature_generation_or_real_execution():
    provenance = label(make_env())["provenance"]
    assert provenance["current_close_observation_verified"] is True
    assert provenance["pipeline_feature_causality_not_certified"] is True
    assert provenance["research_only"] is True and provenance["actionable"] is False
    assert "feature_inputs_causal" not in provenance


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64, np.bool_])
def test_homogeneous_numpy_frames_use_one_exact_dtype_preserving_block(dtype):
    frame = pd.DataFrame(np.asarray([[0, 1], [1, 0]], dtype=dtype))
    encoded = tool._encode(frame)
    assert encoded[4][0] == "homogeneous_numpy"
    assert encoded[4][1][1] == np.dtype(dtype).str
    assert tool._digest(frame) == tool._digest(deepcopy(frame))
    changed = frame.copy()
    changed.iloc[0, 0] = 1
    assert tool._digest(frame) != tool._digest(changed)


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"integer": np.asarray([2 ** 53 + 1], dtype=np.int64), "float": [1.]}),
    pd.DataFrame({"one": np.asarray([1], dtype=np.int32), "two": np.asarray([1], dtype=np.int64)}),
    pd.DataFrame({"integer": pd.Series([1], dtype="Int64"), "other": pd.Series([1], dtype="Int64")}),
    pd.DataFrame({"object": ["one"], "other": ["two"]}),
])
def test_mixed_extension_and_object_frames_retain_typed_column_fallback(frame):
    assert tool._encode(frame)[4][0] == "typed_columns"
    assert tool._digest(frame) == tool._digest(deepcopy(frame))


def test_large_integer_mutations_cannot_be_lost_to_float_upcasting():
    for mixed in (False, True):
        frame = pd.DataFrame({"integer": np.asarray([2 ** 53 + 1], dtype=np.int64)})
        if mixed:
            frame["float"] = 1.
        before = tool._digest(frame)
        frame.loc[0, "integer"] = 2 ** 53 + 2
        assert tool._digest(frame) != before


@pytest.mark.parametrize("mutation", [
    lambda frame: frame.__setitem__("a", frame["a"].astype(np.float32)),
    lambda frame: frame.__setitem__("a", frame["a"] + 1),
    lambda frame: setattr(frame, "columns", ["changed", "b"]),
    lambda frame: setattr(frame, "index", pd.Index(["changed", "row2"])),
    lambda frame: frame.attrs.update({"identity": "changed"}),
    lambda frame: frame["a"].to_numpy().view(np.uint64).__setitem__(0, frame["a"].to_numpy().view(np.uint64)[0] ^ np.uint64(1)),
])
def test_frame_fastpath_preserves_dtype_bits_coordinates_and_attrs(mutation):
    frame = pd.DataFrame([[1., 2.], [3., 4.]], index=["row1", "row2"], columns=["a", "b"])
    before = tool._digest(frame)
    mutation(frame)
    assert tool._digest(frame) != before


def test_pandas_object_cell_shared_by_deepcopy_is_rejected_before_any_transition():
    env = make_env()
    env.env.env.market.extra_frame = pd.DataFrame({"meta": [{"list": [1]}]})
    env.benchmark.env.market.extra_frame = deepcopy(env.env.env.market.extra_frame)
    # Ensure the source books themselves are independent, so the failure below
    # specifically checks Pandas deep-copy semantics in the counterfactual fork.
    env.benchmark.env.market.extra_frame.iat[0, 0] = {"list": [1]}
    before = tool.snapshot_state_fingerprint(env)
    with pytest.raises(tool.CounterfactualLabelError, match="share mutable") as caught:
        label(env)
    assert caught.value.counters["macro_attempts"] == 0
    assert caught.value.counters["branch_agent_steps"] == caught.value.counters["branch_benchmark_steps"] == 0
    assert env.env.env.market.extra_frame.iloc[0, 0] == {"list": [1]}
    assert tool.snapshot_state_fingerprint(env) == before


def test_numpy_object_array_nested_mutable_cells_are_fully_isolated_by_deepcopy():
    env = make_env()
    values = np.empty(1, dtype=object)
    values[0] = {"nested": [1, {"values": np.asarray([2.])}]}
    for wrapper in (env.env, env.benchmark):
        wrapper.env.market.extra_objects = deepcopy(values)
    result = label(env)
    assert result["provenance"]["clone_storage_independent"] is True
    assert result["counters"]["branch_agent_steps"] == 6


def test_numpy_object_array_distinct_storage_cannot_hide_a_shared_nested_mutable():
    one, two = np.empty(1, dtype=object), np.empty(1, dtype=object)
    one[0] = {"nested": [1]}
    two[0] = one[0]
    assert not np.shares_memory(one, two)
    with pytest.raises(RuntimeError, match="share mutable"):
        tool._independent(one, two)


def test_dataframe_attrs_nested_mutable_is_also_checked_for_shared_storage():
    one = pd.DataFrame({"float": [1.]})
    one.attrs["nested"] = {"array": np.asarray([1.])}
    two = deepcopy(one)
    two.attrs["nested"] = one.attrs["nested"]
    with pytest.raises(RuntimeError, match="share mutable"):
        tool._independent(one, two)
