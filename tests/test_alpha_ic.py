"""Unit tests for the IC measurement machinery.

The point of these tests is that the *judge* is correct before any factor is judged: a factor
that is the answer must score 1.0, a factor that is noise must score about zero with a sane
standard error, and a factor built only from a return window that has already closed must
collapse.  Two of those three are the only defence against a look-ahead bug.
"""

import math

import numpy as np
import pandas as pd
import pytest

from tools import alpha_factors as factors
from tools import alpha_ic as measure


def panel(rows=300, columns=40, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    values = rng.standard_normal((rows, columns))
    return pd.DataFrame(values, index=dates, columns=[f"{index:06d}" for index in range(columns)])


def test_rank_ic_is_one_when_the_factor_is_the_answer():
    labels = panel()
    dates = labels.index
    series = measure.rank_ic(labels.to_numpy(), labels.to_numpy(), dates)
    assert series.notna().all()
    np.testing.assert_allclose(series.to_numpy(), 1.0, atol=1e-12)


def test_rank_ic_is_minus_one_for_the_reversed_order():
    labels = panel()
    series = measure.rank_ic(-labels.to_numpy(), labels.to_numpy(), labels.index)
    np.testing.assert_allclose(series.to_numpy(), -1.0, atol=1e-12)


def test_rank_ic_needs_enough_names_per_date():
    labels = panel(rows=5, columns=10)
    series = measure.rank_ic(labels.to_numpy(), labels.to_numpy(), labels.index, min_names=30)
    assert series.isna().all()


def test_rank_ic_ignores_pairs_missing_on_either_side():
    labels = panel(rows=4, columns=60)
    values = labels.to_numpy().copy()
    values[:, :20] = np.nan
    series = measure.rank_ic(values, labels.to_numpy(), labels.index, min_names=30)
    np.testing.assert_allclose(series.to_numpy(), 1.0, atol=1e-12)


def test_nw_se_matches_the_naive_error_on_white_noise():
    rng = np.random.default_rng(5)
    values = rng.standard_normal(4000)
    naive = values.std(ddof=1) / math.sqrt(len(values))
    assert measure.nw_se(values, 20) == pytest.approx(naive, rel=0.10)


def test_nw_se_inflates_by_the_bartlett_factor_under_positive_autocorrelation():
    """An AR(1) with rho=0.5 inflates the variance of the mean by (1+r)/(1-r) = 3."""
    rng = np.random.default_rng(9)
    rho, size = 0.5, 8000
    shocks = rng.standard_normal(size)
    values = np.empty(size)
    values[0] = shocks[0]
    for index in range(1, size):
        values[index] = rho * values[index - 1] + shocks[index]
    naive = values.std(ddof=1) / math.sqrt(size)
    assert measure.nw_se(values, 20) == pytest.approx(naive * math.sqrt(3.0), rel=0.18)


def test_nw_se_is_never_negative_under_negative_autocorrelation():
    values = np.array([1.0, -1.0] * 500)
    assert measure.nw_se(values, 50) >= 0.0
    assert math.isfinite(measure.nw_se(values, 50))


def test_perfect_injection_scores_exactly_one():
    labels = panel()
    injected = measure._inject("perfect", labels, 20, labels.shape)
    series = measure.rank_ic(injected, labels.to_numpy(), labels.index)
    np.testing.assert_allclose(series.dropna().to_numpy(), 1.0, atol=1e-12)


def test_noise_injection_stays_within_three_standard_errors_of_zero():
    labels = panel(rows=800)
    injected = measure._inject("noise", labels, 20, labels.shape)
    series = measure.rank_ic(injected, labels.to_numpy(), labels.index)
    stats = measure.ic_stats(series, 20, 0.0048)
    assert abs(stats["mean_ic"]) < 3.0 * stats["lag_h"]["se"]


def test_placebo_injection_from_a_closed_return_window_collapses():
    """A factor describing returns that ended before the decision date carries no signal."""
    labels = panel(rows=600)
    injected = measure._inject("placebo", labels, 20, labels.shape)
    finite = np.isfinite(injected).all(axis=1)
    series = measure.rank_ic(injected[finite], labels.to_numpy()[finite], labels.index[finite])
    stats = measure.ic_stats(series, 20, 0.0048)
    assert abs(stats["mean_ic"]) < 4.0 * stats["lag_h"]["se"]
    assert abs(stats["mean_ic"]) < 0.05


def test_effective_dates_shrink_when_the_ic_series_persists():
    rng = np.random.default_rng(13)
    white = rng.standard_normal(2000)
    persistent = np.empty(2000)
    persistent[0] = rng.standard_normal()
    for index in range(1, 2000):
        persistent[index] = 0.9 * persistent[index - 1] + rng.standard_normal()
    a = measure.ic_stats(white, 20, 0.0048)
    b = measure.ic_stats(persistent, 20, 0.0048)
    assert a["effective_dates"] > 0.6 * a["dates"]
    assert b["effective_dates"] < 0.2 * b["dates"]


def test_guard_refuses_the_reserve_and_the_protected_sample():
    for low, high in measure.UNLICENSED.values():
        with pytest.raises(ValueError):
            measure.guard(low, high)
        with pytest.raises(ValueError):
            measure.guard(pd.Timestamp(low) - pd.Timedelta(days=10), pd.Timestamp(low) + pd.Timedelta(days=10))


def test_guard_accepts_a_window_that_stops_before_the_reserve():
    measure.guard("2019-01-01", "2024-12-31")


def test_resolve_window_accepts_only_spent_samples():
    assert measure.resolve_window("V") == ("2019-01-01", "2024-12-31")
    for name in measure.UNLICENSED:
        with pytest.raises(ValueError):
            measure.resolve_window(name)


def test_verdict_requires_magnitude_significance_and_direction():
    good = dict(mean_ic=0.010, threshold=0.0048, t_nw_h=3.0)
    assert measure.verdict(good)["pass_"]
    assert not measure.verdict(dict(good, mean_ic=0.0040))["pass_"]   # too small
    assert not measure.verdict(dict(good, t_nw_h=1.5))["pass_"]       # not significant
    assert not measure.verdict(dict(good, mean_ic=-0.010, t_nw_h=-3.0))["pass_"]
    assert not measure.verdict(dict(mean_ic=None, threshold=0.0048, t_nw_h=None))["pass_"]


def test_the_threshold_is_the_registered_economic_ic():
    block = measure._budget_block(measure.BUDGET / "budget_000300_2005_2026.json")
    assert 0.012 <= block["required_alpha"] <= 0.016
    assert 0.40 <= block["sigma_cs"] <= 0.55
    assert 0.0040 <= measure.required_ic(block, 20) <= 0.0056
    # A longer holding period needs a larger IC, by the square root of the horizon.
    assert measure.required_ic(block, 60) > measure.required_ic(block, 20)


def test_family_wise_error_is_reported_for_both_conventions():
    block = measure.family_false_positive()
    assert block["families"] == 5
    assert 0.20 <= block["registered"] <= 0.25
    assert block["implied_by_t2"] < block["registered"]


def test_a_control_is_judged_against_its_known_answer():
    perfect = measure.control_match("perfect", dict(mean_ic=1.0, lag_h=dict(se=0.0)))
    assert perfect["matches"]
    assert not measure.control_match("perfect", dict(mean_ic=0.97, lag_h=dict(se=0.0)))["matches"]
    placebo = measure.control_match("placebo", dict(mean_ic=0.02, lag_h=dict(se=0.01)))
    assert placebo["matches"]
    assert not measure.control_match("placebo", dict(mean_ic=0.20, lag_h=dict(se=0.01)))["matches"]


def test_power_floor_reports_how_small_an_ic_the_reserve_could_resolve():
    block = {
        "h20": dict(rows=1000, required_ic=dict(primary=0.0048), signals={
            "a": dict(family="growth", mean_ic=0.01, ic_std=0.14, effective_dates=100.0),
            "b": dict(family="quality", mean_ic=0.01, ic_std=0.14, effective_dates=100.0),
            "control": dict(family="control", mean_ic=1.0, ic_std=0.0, effective_dates=0.0),
        }),
    }
    power = measure.power_floor(block, reserve_days=359)["h20"]
    # One independent observation per ten decision dates, so R holds about 36 of them.
    assert power["reserve_effective_rows"] == pytest.approx(35.9)
    assert power["reserve_se"] == pytest.approx(0.14 / math.sqrt(35.9))
    # The economically required IC is well inside one standard error: the test cannot resolve it.
    assert power["required_over_se"] < 1.0
    assert power["detectable_2sigma"] > 3.0 * power["required_ic"]
    assert power["years_needed_for_t2"] > 20.0
