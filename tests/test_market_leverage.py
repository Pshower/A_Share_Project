"""The gate's machinery, not its conclusion, is what these tests pin.

The pre-registration is only worth the code that executes it, so the tests check the four things
that could quietly change the answer: the basket's three conditions, the arithmetic of the signal
and its targets, the Newey-West slope, and the rule that a failed guard voids the round before any
bar is read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import market_leverage as gate

DAYS = pd.to_datetime(pd.bdate_range("2020-01-01", periods=40))


def _margin(days, codes, balance=1e8):
    rows = []
    for code in codes:
        for day in days:
            rows.append(dict(code=code, date=day, balance=balance, short_balance=balance / 10))
    return pd.DataFrame(rows)


def test_the_signal_is_the_log_ratio_over_twenty_sessions():
    level = pd.Series(np.exp(np.linspace(0, 1, 40)), index=DAYS)
    frame = pd.DataFrame(dict(lev=level, codes=np.full(40, 100.0)))
    built = gate.build(frame, pd.Series(np.exp(np.linspace(0, 0.5, 40)), index=DAYS))
    expected = np.log(level.iloc[20] / level.iloc[0])
    assert built["r20"].iloc[20] == pytest.approx(expected)


def test_the_forward_drawdown_is_positive_when_the_market_falls():
    close = pd.Series(np.r_[np.full(10, 100.0), [90.0], np.full(29, 100.0)][:40], index=DAYS)
    target = gate.targets(close)
    # The dip sits at index 10, so it lands inside the window that starts at index 0 ... 9.
    assert target["dd"].iloc[0] == pytest.approx(1 - 90.0 / 100.0)
    assert target["fwd"].iloc[-1] != target["fwd"].iloc[-1]  # NaN: no forward window left


def test_the_basket_needs_days_of_both_sources_and_a_positive_balance():
    days = pd.to_datetime(pd.bdate_range("2020-01-01", periods=10))
    margin = _margin(days, ["aaaa", "bbbb"])
    margin.loc[margin["code"] == "bbbb", "balance"] = 0.0     # never actually financed
    cap = pd.DataFrame(100.0, index=days, columns=["aaaa", "bbbb", "cccc"])
    cap["cccc"] = np.nan                                       # no price
    basket = gate.select_basket(margin, cap, min_days=10)
    assert basket == ["aaaa"]


def test_leverage_divides_by_the_same_basket_it_sums():
    days = pd.to_datetime(pd.bdate_range("2020-01-01", periods=3))
    margin = _margin(days, ["aaaa", "bbbb"])
    cap = pd.DataFrame({"aaaa": [100.0, 200.0, 200.0], "bbbb": [300.0, 300.0, 300.0],
                        "cccc": [1e12, 1e12, 1e12]}, index=days)
    frame = gate.leverage(margin, cap, ["aaaa", "bbbb"])
    # 2e8 of balance over 400 of cap on day 1, and 500 on day 2 -- uniform growth, so lev falls.
    assert frame["lev"].iloc[0] == pytest.approx(100 * 2e8 / 400)
    assert frame["lev"].iloc[1] == pytest.approx(100 * 2e8 / 500)
    assert "cccc" not in frame  # a code outside the basket cannot enter the denominator


def test_the_newey_west_slope_recovers_a_known_coefficient():
    rng = np.random.default_rng(7)
    x = pd.Series(rng.normal(0, 1, 400))
    y = -0.02 * x + pd.Series(rng.normal(0, 0.03, 400))
    row = gate.nw_slope(x, y, lag=20)
    assert row["slope"] == pytest.approx(-0.02, abs=0.004)
    assert row["t"] < -2.0
    assert row["n"] == 400


def test_a_flat_predictor_returns_nan_instead_of_dividing_by_zero():
    x = pd.Series(np.ones(30))
    y = pd.Series(np.arange(30.0))
    assert not np.isfinite(gate.nw_slope(x, y)["slope"])


def test_coverage_contamination_is_caught_by_the_first_guard():
    """If the signal moves with the number of reporting codes, it is archive growth, not leverage."""
    n = 200
    driver = np.linspace(0.0, 1.0, n)          # the archive grows from 819 to 3,718 codes
    frame = pd.DataFrame(dict(r20=0.30 * driver + 0.01, dn20=2899.0 * driver))
    cap = pd.DataFrame(1.0, index=pd.bdate_range("2020-01-01", periods=n), columns=["a", "b"])
    found = gate.guards(frame, cap, ["a"])
    assert found["contaminated"] is True
    assert abs(found["coverage_corr"]) > 0.99


def test_a_stable_archive_is_not_contaminated():
    n = 120
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(dict(r20=rng.normal(0, 0.1, n), dn20=rng.normal(0, 5, n)))
    cap = pd.DataFrame(1.0, index=pd.bdate_range("2020-01-01", periods=n), columns=["a", "b"])
    found = gate.guards(frame, cap, ["a"])
    assert found["contaminated"] is False
    assert found["cap_share"] == pytest.approx(0.5)


def test_a_thin_basket_is_caught_by_the_second_guard():
    frame = pd.DataFrame(dict(r20=np.linspace(-.1, .1, 30), dn20=np.zeros(30)),
                         index=pd.bdate_range("2020-01-01", periods=30))
    cap = pd.DataFrame({"a": np.full(30, 1.0), "b": np.full(30, 99.0)},
                       index=pd.bdate_range("2020-01-01", periods=30))
    found = gate.guards(frame, cap, ["a"])
    assert found["thinly_representative"] is True
    assert found["cap_share"] == pytest.approx(0.01)


def test_a_void_round_never_reports_pass_or_fail():
    guard = dict(contaminated=True, coverage_corr=0.9, coverage_limit=0.3,
                 cap_share=0.8, cap_share_limit=0.3, thinly_representative=False)
    tests = dict(bars=dict(T1=True, T2=True, T3=True, T4=True))
    assert gate.verdict(guard, tests)["status"] == "void"


def test_one_failed_bar_fails_the_whole_round():
    guard = dict(contaminated=False, coverage_corr=0.0, coverage_limit=0.3,
                 cap_share=0.8, cap_share_limit=0.3, thinly_representative=False)
    tests = dict(bars=dict(T1=True, T2=False, T3=True, T4=True))
    row = gate.verdict(guard, tests)
    assert row["status"] == "failed" and "T2" in row["reason"]


def test_all_four_bars_pass_together():
    guard = dict(contaminated=False, coverage_corr=0.0, coverage_limit=0.3,
                 cap_share=0.8, cap_share_limit=0.3, thinly_representative=False)
    tests = dict(bars=dict(T1=True, T2=True, T3=True, T4=True))
    assert gate.verdict(guard, tests)["status"] == "passed"
