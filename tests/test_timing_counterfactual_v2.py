"""Synthetic v2 equivalence and fail-closed isolation; no data or NN fitting."""

from copy import deepcopy
from datetime import date
import hashlib
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest

from src.agents.trade_mapping import TradeActionWrapper
from tools import timing_counterfactual as original
from tools import timing_counterfactual_v2 as tool
from tests.test_timing_counterfactual import make_env


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SHA = "97f7c7654150fd23dda728b4ad6e683b42bd88464fe2ec1fbda3c38c895ced47"


def label(env, horizon=3):
    return tool.paired_rule_label(env, horizon=horizon, train_end=env.env.env.market.panel.index[-1])


def normalized(result):
    result = deepcopy(result)
    result.pop("label_version")
    for key in ("state_check_version", "storage_check_version"):
        result["provenance"].pop(key, None)
    return result


def test_frozen_original_helper_is_unchanged():
    assert hashlib.sha256((ROOT / "tools/timing_counterfactual.py").read_bytes()).hexdigest() == ORIGINAL_SHA


@pytest.mark.parametrize("mode", ["cash", "equal", "concentrated", "random"])
@pytest.mark.parametrize("phase", [0, 1, 18, 19])
def test_normal_full_state_labels_are_exactly_equal_to_frozen_helper(mode, phase):
    env = make_env(mode=mode, phase=phase)
    assert tool.snapshot_state_fingerprint(env) == original.snapshot_state_fingerprint(env)
    expected = original.paired_rule_label(env, horizon=3, train_end=env.env.env.market.panel.index[-1])
    actual = label(env)
    assert normalized(actual) == normalized(expected)
    assert actual["provenance"]["state_check_version"] == tool.STATE_CHECK_VERSION


@pytest.mark.parametrize("fees", [False, True])
def test_zero_and_negative_economic_targets_are_retained(fees):
    result = label(make_env(flat=True, fees=fees))
    assert result["labels"]["discounted_relative_delta"] < 0 if fees else result["labels"]["discounted_relative_delta"] == 0


def test_noncalendar_behavior_and_its_local_rng_are_valid_and_fully_fingerprinted():
    env = make_env(mode="random", phase=3, episode_steps=20)
    env.counterfactual_source_rng = np.random.default_rng(8)
    env.counterfactual_source_behavior = "calendar_initialized_bernoulli_v1"
    before_rng = deepcopy(env.counterfactual_source_rng.bit_generator.state)
    fingerprint = tool.snapshot_state_fingerprint(env)
    env.counterfactual_source_behavior = "other_behavior"
    assert tool.snapshot_state_fingerprint(env) != fingerprint
    env.counterfactual_source_behavior = "calendar_initialized_bernoulli_v1"
    env.counterfactual_source_rng.random()
    assert tool.snapshot_state_fingerprint(env) != fingerprint
    env.counterfactual_source_rng.bit_generator.state = before_rng
    assert tool.snapshot_state_fingerprint(env) == fingerprint
    env.step(1)
    assert env._agent_equity != env._benchmark_equity
    before = tool.snapshot_state_fingerprint(env)
    result = label(env, 5)
    assert tool.snapshot_state_fingerprint(env) == before
    assert result["provenance"]["benchmark_paths_equal"]
    assert env.counterfactual_source_rng.bit_generator.state == before_rng


def test_t1_locks_receivables_and_true_account_values_survive_both_clones():
    env = make_env(mode="equal", price_basis="unadjusted", phase=19)
    base = env.env.env
    asset = int(np.flatnonzero(base.portfolio.positions)[0])
    current = base.market.current_date
    for wrapper in (env.env, env.benchmark):
        wrapper.env.portfolio.apply_corporate_action(asset, .1, 2., current + pd.Timedelta(days=3), current + pd.Timedelta(days=6))
        wrapper._observation = wrapper.enrich(wrapper.env._observation())
    env.stock_observation, env.benchmark_stock_observation = env.env._observation, env.benchmark._observation
    env._agent_equity, env._benchmark_equity = base._info()["equity"], env.benchmark.env._info()["equity"]
    before = tool.snapshot_state_fingerprint(env)
    expected = original.paired_rule_label(env, horizon=5, train_end=base.market.panel.index[-1])
    actual = label(env, 5)
    assert normalized(actual) == normalized(expected)
    assert tool.snapshot_state_fingerprint(env) == before


def test_fast_disjoint_storage_does_not_call_pairwise_exact_checker(monkeypatch):
    env = make_env()
    def forbidden(*args, **kwargs):
        pytest.fail("Deep-copied full books should have disjoint byte-range unions")
    monkeypatch.setattr(np, "shares_memory", forbidden)
    result = label(env)
    assert result["provenance"]["clone_storage_independent"]


def test_exact_registered_sampler_is_fully_hashed_and_deep_copied():
    from tools.counterfactual_exact_schedule import ExactPhaseSampler
    env = make_env(days=120, episode_steps=80)
    env.sampler = ExactPhaseSampler(env.env.env.market.dates, episode_steps=80)
    env.reset(seed=42)
    before = tool.snapshot_state_fingerprint(env)
    clone = tool._clone(env, before)
    assert clone.sampler is not env.sampler
    assert clone.sampler.history is not env.sampler.history
    assert clone.sampler.remaining is not env.sampler.remaining
    assert clone.sampler.schedule_rng is not env.sampler.schedule_rng
    assert clone.sampler.date_rng is not env.sampler.date_rng
    assert tool.snapshot_state_fingerprint(clone) == before
    clone.sampler.remaining.pop()
    assert tool.snapshot_state_fingerprint(clone) != before
    assert tool.snapshot_state_fingerprint(env) == before
    assert label(env)["provenance"]["source_after_sha256"] == before


def test_unregistered_class_cannot_spoof_exact_sampler_class_path():
    Fake = type("ExactPhaseSampler", (), {"__module__": "tools.counterfactual_exact_schedule"})
    fake = Fake()
    fake.mutable = []
    with pytest.raises(TypeError, match="Unsupported state type"):
        tool.assert_supported_state(fake)


def test_overlapping_bounds_use_exact_fallback_and_accept_disjoint_strided_holes(monkeypatch):
    array = np.arange(30)
    calls = []
    exact = np.shares_memory
    def checked(left, right):
        calls.append((left, right))
        return exact(left, right)
    monkeypatch.setattr(np, "shares_memory", checked)
    tool._independent(array[::2], array[1::2])
    assert calls
    with pytest.raises(RuntimeError, match="array storage"):
        tool._independent(array[:10], array[4:])


@pytest.mark.parametrize("phase", [True, -1, 20, np.int64(0)])
def test_invalid_phases_have_zero_branch_cost(phase):
    env = make_env()
    env.phase = phase
    with pytest.raises(tool.CounterfactualLabelError) as caught:
        label(env)
    assert caught.value.counters["macro_attempts"] == caught.value.counters["branch_agent_steps"] == 0


def test_actual_asymmetric_failure_cost_is_not_hidden(monkeypatch):
    env = make_env()
    env.benchmark._test_benchmark = True
    before = tool.snapshot_state_fingerprint(env)
    step = TradeActionWrapper.step_order
    def fail(self, order):
        if getattr(self, "_test_benchmark", False):
            raise RuntimeError("synthetic benchmark failure")
        return step(self, order)
    monkeypatch.setattr(TradeActionWrapper, "step_order", fail)
    with pytest.raises(tool.CounterfactualLabelError, match="benchmark failure") as caught:
        label(env)
    assert caught.value.counters["branch_agent_steps"] == 1
    assert caught.value.counters["branch_benchmark_steps"] == 0
    assert caught.value.counters["macro_attempts"] == 1
    assert caught.value.evidence["source_unchanged"] is True
    assert caught.value.evidence["partial_branch_effects_possible"] is True
    assert tool.snapshot_state_fingerprint(env) == before


def test_full_registered_h20_uses_both_unchanged_calendar_continuations():
    env = make_env(days=70, episode_steps=28, phase=19)
    result = label(env, 20)
    assert result["counters"]["branch_agent_steps"] == result["counters"]["branch_benchmark_steps"] == 40
    branches = result["labels"]["branches"]
    assert [row["macro_action"] for row in branches["hold"]["steps"]] == [0, 1] + [0] * 18
    assert [row["benchmark_ledger"] for row in branches["hold"]["steps"]] == [row["benchmark_ledger"] for row in branches["rebalance"]["steps"]]


def test_global_rng_side_effect_is_rejected_without_silent_rollback(monkeypatch):
    env = make_env()
    state = random.getstate()
    run = tool._run_branch
    def perturb(*args, **kwargs):
        random.random()
        return run(*args, **kwargs)
    monkeypatch.setattr(tool, "_run_branch", perturb)
    try:
        with pytest.raises(tool.CounterfactualLabelError, match="Global RNG changed") as caught:
            label(env)
        assert caught.value.counters["branch_agent_steps"] == caught.value.counters["branch_benchmark_steps"] == 6
        assert caught.value.evidence["global_rng_unchanged"] is False
        assert random.getstate() != state
    finally:
        random.setstate(state)


def test_source_postguard_detects_mutation_instead_of_emitting_label(monkeypatch):
    env = make_env()
    env.source_behavior = "before"
    run = tool._run_branch
    def perturb(*args, **kwargs):
        env.source_behavior = "corrupted"
        return run(*args, **kwargs)
    monkeypatch.setattr(tool, "_run_branch", perturb)
    with pytest.raises(tool.CounterfactualLabelError, match="Source execution state changed") as caught:
        label(env)
    # The second clone fails its original-fingerprint guard before stepping.
    assert caught.value.counters["branch_agent_steps"] == caught.value.counters["branch_benchmark_steps"] == 3
    assert caught.value.evidence["source_unchanged"] is False
    assert caught.value.evidence["label_emitted"] is False


class MutableHashable:
    def __init__(self):
        self.payload = []


def unsupported(kind):
    if kind == "structured_object_array":
        value = np.empty(1, dtype=[("cell", object)])
        value["cell"][0] = {"values": []}
        return value
    if kind == "structured_scalar":
        return np.zeros(1, dtype=[("field", "i8")])[0]
    if kind == "structured_numeric_array":
        return np.zeros(1, dtype=[("field", "i8")])
    if kind == "complex_array":
        return np.ones(2, dtype=np.complex128)
    if kind == "complex_scalar":
        return np.complex128(1 + 2j)
    if kind == "metadata_dtype":
        return np.dtype("f8", metadata={"mutable": []})
    if kind == "metadata_array":
        return np.ones(2, dtype=np.dtype("f8", metadata={"mutable": []}))
    if kind == "subclass":
        class Subclass(np.ndarray):
            pass
        return np.arange(4).view(Subclass)
    if kind == "extension_frame":
        return pd.DataFrame({"x": pd.Series([1, None], dtype="Int64")})
    if kind == "categorical_index":
        return pd.CategoricalIndex(["a", "b"])
    if kind == "mutable_index":
        return pd.Index([MutableHashable()], dtype=object)
    if kind == "mutable_frozenset":
        return frozenset([MutableHashable()])
    if kind == "mutable_dict_key":
        return {MutableHashable(): "value"}
    if kind == "unsupported_class":
        return MutableHashable()
    if kind == "cycle":
        value = []
        value.append(value)
        return value
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["structured_object_array", "structured_scalar", "structured_numeric_array", "complex_array", "complex_scalar",
    "metadata_dtype", "metadata_array", "subclass", "extension_frame", "categorical_index", "mutable_index", "mutable_frozenset", "mutable_dict_key",
    "unsupported_class", "cycle"])
def test_unsupported_state_is_rejected_before_any_actual_step(monkeypatch, kind):
    env = make_env()
    env.unsupported = unsupported(kind)
    monkeypatch.setattr(tool, "_branch", lambda *args: pytest.fail("Unsupported source reached branch execution"))
    with pytest.raises(tool.CounterfactualLabelError) as caught:
        label(env)
    assert caught.value.counters["branch_agent_steps"] == caught.value.counters["branch_benchmark_steps"] == 0
    assert caught.value.counters["macro_attempts"] == 0


@pytest.mark.parametrize("value", [np.float32(.5), np.int64(3), np.str_("x"), np.bytes_(b"x"),
    np.datetime64("2020-01-01"), np.timedelta64(3, "D"), date(2020, 1, 1),
    frozenset(["a", ("b", 2)]), pd.date_range("2020-01-01", periods=3),
    pd.MultiIndex.from_product([["A", "B"], ["x", "y"]])])
def test_supported_plain_scalar_date_and_index_values_are_deterministic(value):
    tool.assert_supported_state(value)
    assert tool._digest(value) == tool._digest(deepcopy(value))


def test_plain_object_cells_must_be_truly_independent_despite_disjoint_array_storage():
    env = make_env()
    for wrapper in (env.env, env.benchmark):
        wrapper.env.market.extra = pd.DataFrame({"cell": [{"values": [1]}]})
    before = tool.snapshot_state_fingerprint(env)
    with pytest.raises(tool.CounterfactualLabelError, match="mutable objects") as caught:
        label(env)
    assert caught.value.counters["macro_attempts"] == 0
    assert tool.snapshot_state_fingerprint(env) == before
