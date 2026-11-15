"""Carrier arithmetic: unit-conversion repair, dividend reinvestment, and the comparison itself.

The repair test is a regression.  Reading a 3.5x 份额折算 as performance put 510500 at +20pp a
year; reading it as a 50% loss put 513500 at -7pp.  Both were wrong by the size of the event.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import fund_vehicles as vehicles


def series(values, start="2015-01-05"):
    index = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=index, dtype=float)


def test_a_two_for_one_split_is_repaired_and_recorded():
    prices = series([2.0, 2.0, 1.0, 1.0])
    repaired, events = vehicles.repair(prices, {})
    assert repaired.iloc[1] == pytest.approx(1.0)          # earlier prices halved
    assert len(events) == 1 and events[0]["kind"] == "split"


def test_an_upward_unit_conversion_is_repaired_in_the_same_direction():
    prices = series([1.0, 1.0, 3.5, 3.5])
    repaired, events = vehicles.repair(prices, {})
    assert repaired.iloc[1] == pytest.approx(3.5)          # earlier prices scaled up
    assert len(events) == 1 and events[0]["kind"] == "reverse_split"


def test_a_large_move_on_a_dividend_date_is_left_alone():
    prices = series([2.0, 1.0, 1.0])
    stamps = sorted(prices.index)
    repaired, events = vehicles.repair(prices, {stamps[1]: 1.0})
    assert len(events) == 0
    assert repaired.iloc[0] == pytest.approx(2.0)


def test_a_real_crash_that_is_not_a_round_ratio_is_left_alone():
    prices = series([1.0, 1.0, 0.62, 0.62])
    repaired, events = vehicles.repair(prices, {})
    assert len(events) == 0
    assert repaired.iloc[1] == pytest.approx(1.0)


def test_total_return_reinvests_the_distribution():
    prices = series([1.0, 1.0, 0.9, 0.9])
    stamps = sorted(prices.index)
    index = vehicles.total_return(prices, {stamps[2]: 0.1})
    assert index.iloc[-1] == pytest.approx(1.0)            # 0.9 price + 0.1 cash, no loss
    assert index.iloc[2] == pytest.approx(1.0)


def test_compare_finds_the_shortfall_of_a_lagging_replica(tmp_path):
    index = pd.bdate_range("2015-01-05", periods=1040)
    days = (index - index[0]).days.to_numpy()
    levels = pd.DataFrame({"bench": (1.10 ** (days / 365.25))}, index=index)
    (tmp_path / "fund").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": index.strftime("%Y-%m-%d"),
                  "close": 100.0 * (1.09 ** (days / 365.25))}).to_csv(
        tmp_path / "fund" / "hist_999999.csv", index=False)
    pd.DataFrame(columns=["date", "per_unit"]).to_csv(tmp_path / "fund" / "dist_999999.csv",
                                                      index=False)
    row = vehicles.compare("999999", "bench", levels, online=tmp_path)
    assert row is not None
    assert row["difference_pp"] == pytest.approx(-1.0, abs=0.1)


def test_compare_refuses_a_history_shorter_than_the_minimum(tmp_path):
    index = pd.bdate_range("2024-01-02", periods=40)
    levels = pd.DataFrame({"bench": np.linspace(1.0, 1.1, len(index))}, index=index)
    prices = pd.DataFrame({"date": index.strftime("%Y-%m-%d"), "close": np.linspace(1.0, 1.1, 40)})
    (tmp_path / "fund").mkdir(parents=True, exist_ok=True)
    prices.to_csv(tmp_path / "fund" / "hist_999999.csv", index=False)
    pd.DataFrame(columns=["date", "per_unit"]).to_csv(tmp_path / "fund" / "dist_999999.csv",
                                                      index=False)
    assert vehicles.compare("999999", "bench", levels, online=tmp_path) is None
