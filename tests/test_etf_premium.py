"""The premium series has two traps: a split the price history never adjusted, and a NAV lag that
looks like a premium.  Both must be handled before any number from it is quoted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import etf_premium


def frame(premiums, start="2020-01-01"):
    index = pd.bdate_range(start, periods=len(premiums))
    return pd.DataFrame({"premium": premiums}, index=index)


def test_the_split_detector_finds_the_session_where_the_two_series_disagree():
    index = pd.bdate_range("2022-03-01", periods=30)
    close = pd.Series(np.r_[np.full(20, 2.7), np.full(10, 1.39)], index=index)
    nav = pd.Series(np.r_[np.full(19, 2.75), np.full(11, 1.39)], index=index)
    days = etf_premium.split_days(close, nav)
    assert index[20] in days or index[19] in days
    assert index[0] not in days


def test_the_split_detector_is_quiet_on_a_normal_series():
    index = pd.bdate_range("2020-01-01", periods=30)
    close = pd.Series(np.linspace(2.0, 2.2, 30), index=index)
    nav = pd.Series(np.linspace(2.0, 2.2, 30), index=index)
    assert etf_premium.split_days(close, nav) == []


def test_premium_is_price_over_nav_minus_one():
    index = pd.bdate_range("2020-01-01", periods=3)
    close = pd.Series([1.10, 1.20, 1.00], index=index)
    nav = pd.Series([1.00, 1.00, 1.00], index=index)
    joined = etf_premium.premium(close, nav, drop_splits=False)
    assert joined["premium"].tolist() == pytest.approx([0.10, 0.20, 0.0])


def test_distribution_reports_the_share_above_each_level():
    stats = etf_premium.distribution(frame([0.01, 0.03, 0.09]).iloc[:0]["premium"]) if False else None
    stats = etf_premium.distribution(pd.Series([0.01, 0.03, 0.09]))
    assert stats["days"] == 3
    assert stats["share_above"]["1%"] == pytest.approx(2 / 3)
    assert stats["share_above"]["5%"] == pytest.approx(1 / 3)


def test_entry_cost_is_what_you_pay_less_what_you_get_back():
    joined = frame([0.06, 0.06, 0.02, 0.02])              # buy at 6%, sell two sessions later at 2%
    cost = etf_premium.entry_cost(joined, threshold=0.99, hold_sessions=2)["buy_always"]
    assert cost["mean"] == pytest.approx(0.04)
    assert cost["share_positive"] == pytest.approx(1.0)


def test_a_gate_restricts_which_days_count():
    joined = frame([0.01, 0.05, 0.01, 0.05])
    gated = etf_premium.entry_cost(joined, threshold=0.02, hold_sessions=1)
    assert gated["below_threshold"]["pairs_allowed"] == 2
    assert gated["below_threshold"]["share"] == pytest.approx(2 / 3)   # the last day has no successor


def test_entry_cost_returns_nothing_when_the_window_is_too_short():
    assert etf_premium.entry_cost(frame([0.01, 0.02]), threshold=0.03, hold_sessions=5) == {}
