"""Regression tests for the equal-weight index cross-check.

The cross-check exists because a panel-built equal-weight premium agreed with itself
across every size band and every sub-period and was still wrong by an order of
magnitude.  These tests pin the two things that make it work:

* the arithmetic identity it rests on, that an equal-weighted book compounds the
  cross-sectional mean of daily returns; and
* the verdict logic that refuses a panel premium the index pair does not reproduce.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import capm_backtest as capm
from tools import equal_weight_crosscheck as check


def _volatile_panel(names=60, days=500, daily_vol=0.025, drift=0.0, seed=7):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=days)
    returns = rng.normal(drift, daily_vol, (days, names))
    returns[0] = 0.0
    columns = [f"S{n:03d}" for n in range(names)]
    prices = pd.DataFrame(100 * np.exp(np.cumsum(np.log1p(returns), axis=0)), index=index, columns=columns)
    return prices


def test_daily_rebalanced_equal_weight_is_exactly_the_cross_sectional_average():
    # With step=1 the book is reset every session, so its return is the cross-sectional
    # mean by construction.  That identity is what the diagnostic rests on: at the
    # monthly step of 20 the two differ by the drift between rebalances (15.36% against
    # 14.97% on the real band), which is why the check compares the daily object.
    prices = _volatile_panel()
    # daily_frame leaves the first row NaN, which would leave the book without weights
    # for that session; dropping it makes the identity exact rather than approximate.
    returns = capm.daily_frame(prices).dropna(how="all")
    equal, _ = capm.equal_weight_portfolio(returns, step=1)
    average = returns.mean(axis=1)
    assert np.allclose(equal.to_numpy(), average.to_numpy(), atol=1e-12)


def test_mechanism_shows_the_variance_drag_between_names_and_the_book():
    prices = _volatile_panel(daily_vol=0.025)
    detail = check.mechanism(prices, start="2020-01-02", end="2024-12-31")
    # High single-name volatility: the arithmetic average must beat the median name
    # by roughly half that variance, which is exactly the premium the panel reports.
    assert detail["median_single_vol"] > 0.30
    assert detail["arithmetic_average"] > detail["median_single_name"] + 0.04
    drag = 0.5 * detail["median_single_vol"] ** 2
    gap = detail["arithmetic_average"] - detail["median_single_name"]
    assert gap == pytest.approx(drag, rel=0.35, abs=0.05)


def test_verdict_rejects_a_panel_premium_the_index_does_not_reproduce():
    decision = check.verdict(0.0792, 0.0059, tolerance=0.01)
    assert decision["status"] == "unsupported"
    assert decision["difference"] == pytest.approx(0.0733, abs=1e-4)


def test_verdict_accepts_a_panel_premium_the_index_does_reproduce():
    decision = check.verdict(0.0080, 0.0059, tolerance=0.01)
    assert decision["status"] == "supported"


def test_verdict_refuses_to_judge_when_one_side_is_missing():
    assert check.verdict(0.05, None)["status"] == "unknown"
    assert check.verdict(None, 0.01)["status"] == "unknown"


def test_index_pair_premium_is_small_and_tracks_the_cap_weighted_sibling():
    detail = check.index_premium("000905", "000982", "2019-01-02", "2024-12-31")
    assert detail["status"] == "ready"
    # The published pair is the reference this whole correction rests on: same 500
    # constituents, different weighting, so the premium must be small.
    assert abs(detail["premium"]) < 0.03
    assert detail["correlation"] > 0.98

def test_point_in_time_selection_removes_the_lookahead_premium(tmp_path=None):
    # The static band is chosen by median capitalisation over the whole sample, which is
    # a look-ahead: winners that were small in 2019 keep a full equal weight on day one.
    # Re-choosing the band each rebalance from the lagged cap must shrink the premium
    # towards the published index pair. This pins the finding so it cannot silently
    # come back.
    static = check.panel_premium(check.DEFAULT_PANEL, 300, 800, "2019-01-02", "2024-12-31")
    pit = check.dynamic_band_premium(check.DEFAULT_PANEL, 300, 800, "2019-01-02", "2024-12-31")
    index = check.index_premium("000905", "000982", "2019-01-02", "2024-12-31")
    assert static["premium"] > 0.05
    assert abs(pit["premium"]) < 0.02
    assert abs(pit["premium"] - index["premium"]) < 0.02


def test_point_in_time_premium_is_judged_supported_while_static_is_not():
    static = 0.0792
    pit = 0.0037
    index = 0.0059
    assert check.verdict(static, index)["status"] == "unsupported"
    assert check.verdict(pit, index)["status"] == "supported"
