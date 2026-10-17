"""Pure synthetic schedule coverage; no environment or model is executed."""

from collections import Counter
from copy import deepcopy
from itertools import product
import random

import numpy as np
import pandas as pd
import pytest

from src.envs.initial_account import MODES
from tools.counterfactual_exact_schedule import DATE_STRATA, ExactPhaseSampler


def dates(count=360):
    return pd.bdate_range("2018-01-01", periods=count)


def test_one_cycle_exactly_covers_all_phase_account_date_combinations():
    sampler = ExactPhaseSampler(dates())
    sampler.seed(42)
    rows = [sampler.next() for _ in range(240)]
    combinations = Counter((row["initial_mode"], row["date_stratum"], row["phase"]) for row in rows)
    assert set(combinations) == set(product(MODES, DATE_STRATA, range(20)))
    assert set(combinations.values()) == {1}
    assert Counter(row["phase"] for row in rows) == {phase: 12 for phase in range(20)}
    assert [row["episode"] for row in rows] == list(range(240))
    assert all(row["phase_bin"] == row["phase"] // 5 for row in rows)


def test_three_thirds_partition_all_feasible_starts_including_latest_region():
    sampler = ExactPhaseSampler(dates(100), episode_steps=80)
    np.testing.assert_array_equal(np.concatenate(list(sampler.starts.values())), np.arange(20))
    assert sampler.maximum_start == 19
    assert len(sampler.starts["latest"]) > 1
    assert sampler.starts["latest"][-1] == sampler.maximum_start
    rows = [sampler.next() for _ in range(480)]
    for row in rows:
        assert row["start_index"] in sampler.starts[row["date_stratum"]]
        assert row["start_index"] + 80 < len(sampler.dates)
        assert row["start_date"] == str(sampler.dates[row["start_index"]].date())
    assert len({row["start_index"] for row in rows if row["date_stratum"] == "latest"}) > 1


def test_seed_replays_audit_and_all_next_outputs():
    first, second = ExactPhaseSampler(dates()), ExactPhaseSampler(dates())
    first.seed(123)
    second.seed(123)
    assert [first.next() for _ in range(270)] == [second.next() for _ in range(270)]
    assert first.audit() == second.audit()
    first.seed(123)
    second.seed(123)
    assert first.next() == second.next()
    assert first.episode_count == 1


def test_schedule_rng_is_independent_of_date_draw_count_and_stratum_sizes():
    first, second = ExactPhaseSampler(dates(360)), ExactPhaseSampler(dates(500))
    first.seed(2026)
    second.seed(2026)
    for _ in range(480):
        left, right = first.next(), second.next()
        assert (left["initial_mode"], left["date_stratum"], left["phase"]) == (
            right["initial_mode"], right["date_stratum"], right["phase"])


def test_audit_and_next_results_do_not_alias_sampler_state():
    sampler = ExactPhaseSampler(dates())
    row = sampler.next()
    original = deepcopy(sampler.history)
    row["phase"] = 99
    audit = sampler.audit()
    audit["history"][0]["phase"] = 98
    audit["date_strata"]["early"].clear()
    audit["rng_states"]["date"]["state"]["state"] = 0
    assert sampler.history == original
    assert len(sampler.starts["early"]) > 0
    assert sampler.audit()["rng_states"]["date"]["state"]["state"] != 0


def test_sampler_preserves_python_and_numpy_global_rng():
    py, np_state = random.getstate(), np.random.get_state()
    sampler = ExactPhaseSampler(dates())
    sampler.seed(42)
    for _ in range(250):
        sampler.next()
    assert random.getstate() == py
    current = np.random.get_state()
    assert current[0] == np_state[0] and current[2:] == np_state[2:]
    np.testing.assert_array_equal(current[1], np_state[1])


@pytest.mark.parametrize("count,steps", [(82, 80), (90, 0), (90, -1), (90, True), (90, 80.0)])
def test_invalid_length_or_episode_steps_rejected(count, steps):
    with pytest.raises(ValueError):
        ExactPhaseSampler(dates(count), steps)


@pytest.mark.parametrize("values", [
    ["2020-01-02", "2020-01-01"], ["2020-01-01"] * 90,
    [pd.NaT] + list(dates(90)), pd.date_range("2020-01-01", periods=90, tz="UTC"),
])
def test_invalid_date_coordinates_rejected(values):
    with pytest.raises(ValueError):
        ExactPhaseSampler(values)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "42"])
def test_non_integer_or_negative_seed_rejected(seed):
    with pytest.raises(ValueError):
        ExactPhaseSampler(dates()).seed(seed)


def test_minimum_three_feasible_starts_are_supported():
    sampler = ExactPhaseSampler(dates(83))
    assert {key: value.tolist() for key, value in sampler.starts.items()} == {
        "early": [0], "middle": [1], "latest": [2]}
    assert sampler.audit()["cycle_episodes"] == 240
