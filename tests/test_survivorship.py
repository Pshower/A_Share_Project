"""Survivorship: the tool must be able to say "this panel cannot answer that".

The panel holds only names still listed when it was built, so the honest output is a count of early
stops (almost none) plus a bound from the exchange delisting counts -- not a measured correction.
These tests pin the three pieces of that arithmetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import survivorship


def frame(columns):
    index = pd.bdate_range("2020-01-01", periods=len(next(iter(columns.values()))))
    return pd.DataFrame(columns, index=index)


def test_aliveness_notices_who_stops_and_who_arrives():
    days = pd.bdate_range("2020-01-01", periods=60)
    close = pd.DataFrame({
        "alive": np.linspace(10, 11, 60),
        "dies": np.r_[np.linspace(10, 9, 20), [np.nan] * 40],
        "late": np.r_[[np.nan] * 40, np.linspace(10, 12, 20)],
    }, index=days)
    alive = survivorship.aliveness(close)
    assert alive["codes"] == 3
    assert alive["stopped_early"] == 1
    assert alive["listed_during_window"] == 1
    assert alive["alive_at_end"] == 2


def test_buy_and_hold_is_the_mean_of_multiples_not_of_returns():
    days = pd.bdate_range("2020-01-01", periods=2)
    close = pd.DataFrame({"a": [10.0, 20.0], "b": [10.0, 10.0]}, index=days)
    series = survivorship.equal_weight(close, days[0], days[1], cadence="buy_hold")
    assert series.iloc[-1] == pytest.approx(1.5)          # (2.0 + 1.0) / 2


def test_daily_rebalancing_earns_more_than_buy_and_hold_when_names_move_together():
    """Two +100% moves on different days: rebalancing captures both, holding splits them."""
    days = pd.bdate_range("2020-01-01", periods=3)
    close = pd.DataFrame({"a": [10.0, 20.0, 20.0], "b": [10.0, 10.0, 20.0]}, index=days)
    hold = survivorship.equal_weight(close, days[0], days[-1], cadence="buy_hold")
    daily = survivorship.equal_weight(close, days[0], days[-1], cadence="daily")
    assert hold.iloc[-1] == pytest.approx(2.0)       # (2x + 2x) / 2
    assert daily.iloc[-1] == pytest.approx(2.25)     # +50% twice


def test_the_bound_is_zero_when_nothing_delisted_and_grows_with_the_count():
    assert survivorship.bias_bound(5000, 0, 3.0, 0.0, 6.0) == pytest.approx(0.0)
    small = survivorship.bias_bound(5000, 100, 3.0, 0.0, 6.0)
    large = survivorship.bias_bound(5000, 500, 3.0, 0.0, 6.0)
    assert large < small < 0


def test_a_kinder_recovery_narrows_the_bound():
    harsh = survivorship.bias_bound(5000, 213, 3.0, 0.0, 6.0)
    kind = survivorship.bias_bound(5000, 213, 3.0, 0.8, 6.0)
    assert kind > harsh


def test_annualised_compounds_over_242_session_years():
    index = pd.bdate_range("2020-01-01", periods=242)
    series = pd.Series(np.linspace(1.0, 1.2, 242), index=index)
    assert survivorship.annualised(series) == pytest.approx(0.2, rel=1e-6)
