"""Independent paired-label invariants on synthetic complete economic engines."""

from copy import deepcopy
import json
import random

import numpy as np
import pandas as pd
import pytest
import torch

from src.agents.trade_mapping import TradeActionWrapper
from src.training.timing_env import TimingEnv
from tests.archive.test_timing_counterfactual import make_env
from tools.archive import timing_counterfactual as tool


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def label(env, horizon=3, **kwargs):
    return tool.paired_rule_label(env, horizon=horizon,
        train_end=env.env.env.market.panel.index[-1], **kwargs)


def manual_branch(source, first, horizon):
    branch = deepcopy(source)
    rewards, phases, actions = [], [], []
    for index in range(horizon):
        phase = branch.phase
        action = first if index == 0 else int(phase == 0)
        _, _, ended, truncated, info = branch.step(action)
        assert not (ended or truncated) or index == horizon - 1
        phases.append(phase)
        actions.append(action)
        rewards.append(info["timing"]["agent_log_return"] - info["timing"]["benchmark_log_return"])
    return np.array(rewards), branch, phases, actions


def test_fingerprint_is_value_stable_across_independent_factories_and_deep_copy():
    first, second = make_env(mode="random"), make_env(mode="random")
    expected = tool.snapshot_state_fingerprint(first)
    assert expected == tool.snapshot_state_fingerprint(second)
    assert expected == tool.snapshot_state_fingerprint(deepcopy(first))


@pytest.mark.parametrize("change", ["cash", "available", "cost_basis", "receivable", "share_lock", "broker_usage", "date", "local_rng"])
def test_execution_state_fingerprint_includes_account_events_broker_and_local_rng(change):
    env = make_env(mode="equal")
    base, portfolio = env.env.env, env.env.env.portfolio
    before = tool.snapshot_state_fingerprint(env)
    if change == "cash":
        portfolio.cash += 1
    elif change == "available":
        portfolio.available_positions[0] += 1
    elif change == "cost_basis":
        portfolio.cost_basis[0] += .01
    elif change == "receivable":
        portfolio.receivables.append((portfolio.current_date + pd.Timedelta(days=3), 1.))
    elif change == "share_lock":
        portfolio.share_locks.append((portfolio.current_date + pd.Timedelta(days=3), 0, 100))
    elif change == "broker_usage":
        base.broker._used_volume = np.ones(base.market.n_stocks)
    elif change == "date":
        portfolio.current_date += pd.Timedelta(days=1)
    else:
        base.np_random.integers(100)
    assert before != tool.snapshot_state_fingerprint(env)


def test_full_twenty_day_pair_has_fixed_calendar_and_separate_future_targets():
    env = make_env(days=90, episode_steps=30, phase=7)
    source = tool.snapshot_state_fingerprint(env)
    result = tool.paired_rule_label(env, train_end=env.env.env.market.panel.index[-1])
    assert set(result["input"]) == {"observation", "feature_names", "decision_date", "phase"}
    assert len(result["input"]["observation"]) == 129
    assert result["labels"]["horizon"] == 20 and result["labels"]["gamma"] == .995
    for name, first in (("hold", 0), ("rebalance", 1)):
        steps = result["labels"]["branches"][name]["steps"]
        assert len(steps) == 20
        assert [step["phase"] for step in steps] == [(7 + index) % 20 for index in range(20)]
        assert [step["macro_action"] for step in steps] == [first] + [int((7 + index) % 20 == 0) for index in range(1, 20)]
    assert result["counters"] == dict(branch_agent_steps=40, branch_benchmark_steps=40,
        macro_attempts=40, collection_steps=0, rl_steps=0, nn_forwards=0, optimizer_updates=0)
    assert result["provenance"]["source_before_sha256"] == result["provenance"]["source_after_sha256"] == source
    assert result["provenance"]["research_only"] is True and result["provenance"]["actionable"] is False
    assert result["provenance"]["pipeline_feature_causality_not_certified"] is True
    json.dumps(result, allow_nan=False)


def test_warm_source_keeps_different_agent_reference_books_and_counts_only_new_branch_steps():
    env = make_env(days=100, episode_steps=40)
    for _ in range(7):
        env.step(0)
    assert env.total_agent_steps == env.total_benchmark_steps == 7
    assert env._agent_equity != env._benchmark_equity
    before = tool.snapshot_state_fingerprint(env)
    # Evaluate the manual branches in the reverse order of the production helper.
    rebalanced, r_env, _phases, _actions = manual_branch(env, 1, 5)
    held, h_env, _phases, _actions = manual_branch(env, 0, 5)
    result = label(env, horizon=5)
    expected = float(np.dot(.995 ** np.arange(5), rebalanced - held))
    assert result["labels"]["discounted_relative_delta"] == pytest.approx(expected, abs=1e-16)
    assert result["labels"]["terminal_log_nav_ratio"] == pytest.approx(np.log(r_env._agent_equity / h_env._agent_equity), abs=1e-14)
    assert result["counters"]["branch_agent_steps"] == result["counters"]["branch_benchmark_steps"] == 10
    assert env.total_agent_steps == env.total_benchmark_steps == 7
    assert tool.snapshot_state_fingerprint(env) == before


def test_future_open_changes_target_but_not_current_close_input_or_cost_features():
    ordinary = make_env(phase=7)
    changed = make_env(phase=7)
    before = ordinary.observe().copy()
    for wrapper in (changed.env, changed.benchmark):
        market = wrapper.env.market
        market._fields["open"][market._indices[market.current_index + 1]] *= 1.1
    np.testing.assert_array_equal(ordinary.observe(), changed.observe())
    first, second = label(ordinary), label(changed)
    np.testing.assert_array_equal(first["input"]["observation"], before)
    np.testing.assert_array_equal(first["input"]["observation"], second["input"]["observation"])
    assert first["labels"]["discounted_relative_delta"] != second["labels"]["discounted_relative_delta"]


def test_terminal_ratio_is_not_substituted_for_discounted_teacher_difference():
    result = label(make_env(), horizon=5)["labels"]
    difference = np.array(result["per_step_relative_reward_difference"])
    assert result["discounted_relative_delta"] == pytest.approx(sum(.995 ** k * x for k, x in enumerate(difference)), abs=1e-16)
    assert result["terminal_log_nav_ratio"] == pytest.approx(difference.sum(), abs=1e-13)
    assert abs(result["discounted_relative_delta"] - result["terminal_log_nav_ratio"]) > 1e-8


def test_zero_economic_difference_is_not_relabelled_as_a_positive_rebalance():
    result = label(make_env(flat=True, fees=False, phase=7))["labels"]
    assert result["discounted_relative_delta"] == result["terminal_log_nav_ratio"] == 0.
    assert result["branches"]["hold"]["steps"][0]["macro_action"] == 0
    assert result["branches"]["rebalance"]["steps"][0]["macro_action"] == 1


def test_hidden_future_provider_rows_are_rejected_before_any_simulation():
    env = make_env()
    cutoff = env.env.env.market.dates[env.env.env.market.current_index + 8]
    for wrapper in (env.env, env.benchmark):
        market = wrapper.env.market
        selected = market.dates <= cutoff
        market.dates, market._indices = market.dates[selected], market._indices[selected]
        market.n_dates = len(market.dates)
        market.prices = market.prices.loc[:cutoff]
    with pytest.raises(tool.CounterfactualLabelError, match="physically") as error:
        tool.paired_rule_label(env, horizon=3, train_end=cutoff)
    assert error.value.counters["branch_agent_steps"] == error.value.counters["branch_benchmark_steps"] == 0
    assert error.value.counters["macro_attempts"] == 0
    assert error.value.evidence["label_emitted"] is False


def test_time_limit_refusal_does_not_reset_or_hide_any_steps():
    env = make_env(episode_steps=4)
    env.step(0)
    before = tool.snapshot_state_fingerprint(env)
    with pytest.raises(tool.CounterfactualLabelError, match="TimeLimit") as error:
        label(env, horizon=5)
    assert error.value.counters["macro_attempts"] == 0
    assert error.value.counters["branch_agent_steps"] == error.value.counters["branch_benchmark_steps"] == 0
    assert tool.snapshot_state_fingerprint(env) == before


@pytest.mark.parametrize("rng_kind", ["generator", "random_state"])
def test_clones_cannot_share_supported_mutable_rng_objects(monkeypatch, rng_kind):
    env = make_env()
    if rng_kind == "random_state":
        env.env.env._np_random = np.random.RandomState(42)
    genuine = deepcopy

    def alias_rng(source):
        clone = genuine(source)
        clone.env.env._np_random = source.env.env._np_random
        return clone

    monkeypatch.setattr(tool, "deepcopy", alias_rng)
    with pytest.raises(tool.CounterfactualLabelError, match="share.*mutable") as error:
        label(env)
    assert error.value.counters["macro_attempts"] == 0
    assert error.value.evidence["label_emitted"] is False


def test_benchmark_ledger_cannot_differ_even_when_its_nav_is_unchanged(monkeypatch):
    env = make_env(phase=7)
    genuine = TimingEnv.step

    def corrupt(self, action):
        result = genuine(self, action)
        if self.env.env._steps == 1 and action == 1:
            result[4]["timing"]["benchmark_ledger"]["total_fees"] += .01
        return result

    monkeypatch.setattr(TimingEnv, "step", corrupt)
    with pytest.raises(tool.CounterfactualLabelError, match="shadow benchmark") as error:
        label(env)
    assert error.value.counters["branch_agent_steps"] == error.value.counters["branch_benchmark_steps"] == 6
    assert error.value.counters["macro_attempts"] == 6
    assert error.value.evidence["label_emitted"] is False


def test_asymmetric_failure_counts_advanced_agent_without_a_completed_macro(monkeypatch):
    env = make_env()
    env.benchmark.counterfactual_review_fault = True
    before = tool.snapshot_state_fingerprint(env)
    genuine = TradeActionWrapper.step_order

    def fail_benchmark(self, order):
        if vars(self).get("counterfactual_review_fault", False):
            raise RuntimeError("Synthetic benchmark failure before advancement")
        return genuine(self, order)

    monkeypatch.setattr(TradeActionWrapper, "step_order", fail_benchmark)
    with pytest.raises(tool.CounterfactualLabelError, match="Synthetic benchmark failure") as error:
        label(env)
    assert error.value.counters["branch_agent_steps"] == 1
    assert error.value.counters["branch_benchmark_steps"] == 0
    assert error.value.counters["macro_attempts"] == 1
    assert error.value.counters["rl_steps"] == 0
    assert error.value.evidence["label_emitted"] is False
    assert tool.snapshot_state_fingerprint(env) == before


def test_failure_before_base_clock_commit_does_not_claim_a_completed_step(monkeypatch):
    env = make_env()
    genuine = TradeActionWrapper.step_order

    def partial(self, order):
        if not vars(self).get("counterfactual_review_advanced", False):
            self.env.portfolio.cash -= 1.
            raise RuntimeError("Synthetic partial event before base step commit")
        return genuine(self, order)

    monkeypatch.setattr(TradeActionWrapper, "step_order", partial)
    with pytest.raises(tool.CounterfactualLabelError, match="partial event") as error:
        label(env)
    assert error.value.counters["branch_agent_steps"] == error.value.counters["branch_benchmark_steps"] == 0
    assert error.value.counters["macro_attempts"] == 1
    assert error.value.evidence["partial_branch_effects_possible"] is True
    assert error.value.evidence["partial_branch_effects_are_not_rolled_back"] is True


@pytest.mark.parametrize("kind", ["numpy", "python", "torch"])
def test_unexpected_global_rng_use_aborts_without_emitting_a_label(monkeypatch, kind):
    env = make_env()
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    genuine = TimingEnv.step

    def random_draw(self, action):
        if kind == "numpy":
            np.random.random()
        elif kind == "python":
            random.random()
        else:
            torch.rand(1)
        return genuine(self, action)

    monkeypatch.setattr(TimingEnv, "step", random_draw)
    try:
        with pytest.raises(tool.CounterfactualLabelError, match="Global RNG changed") as error:
            label(env)
        assert error.value.counters["branch_agent_steps"] == error.value.counters["branch_benchmark_steps"] == 6
        assert error.value.counters["macro_attempts"] == 6
        assert error.value.evidence["global_rng_unchanged"] is False
        assert error.value.evidence["label_emitted"] is False
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])


def test_cuda_rng_is_not_initialized_or_queried_for_a_cpu_label(monkeypatch):
    env = make_env()
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("No CUDA RNG read before initialization"))
    result = label(env)
    assert "torch_cpu" in result["provenance"]["global_rng_scope"]
    assert "torch_already_initialized_cuda" not in result["provenance"]["global_rng_scope"]
