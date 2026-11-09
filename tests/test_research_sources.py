"""Tests for the two remaining tools whose output is a published number.

``dividend_yield`` produced the 2.58 / 1.83 / 1.68 dividend rates that moved the headline
return from 5.31 to 7.49 percent, and ``preannounce_ic`` produced the 0.0216 that closed the
earnings-forecast source.  Both were computed once and read; neither was checked.

The risk in each is different.  The dividend rate can be wrong in a way that looks plausible,
because every wrong method still yields a number near two percent; the preannouncement signal
can be wrong by looking at a filing that has not happened yet, which inflates the IC without
raising anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import dividend_yield, preannounce_ic


class TestDividendYield:
    def _cache(self, tmp_path, code, rows):
        frame = pd.DataFrame(rows)
        frame.to_csv(tmp_path / f"{code}.csv", index=False)
        return tmp_path

    def test_cash_per_ten_shares_is_converted_to_per_share(self, tmp_path):
        rows = [{"派息": 10.0, "除权除息日": "2020-06-01", "进度": "实施"}]
        cache = self._cache(tmp_path, "000001", rows)
        table = dividend_yield.dividends("000001", cache=cache)
        assert table["cash"].iloc[0] == pytest.approx(1.0)

    def test_unimplemented_plans_are_dropped(self, tmp_path):
        rows = [{"派息": 10.0, "除权除息日": "2020-06-01", "进度": "实施"},
                {"派息": 20.0, "除权除息日": "2021-06-01", "进度": "董事会预案"}]
        cache = self._cache(tmp_path, "000001", rows)
        table = dividend_yield.dividends("000001", cache=cache)
        assert len(table) == 1

    def test_a_missing_field_returns_nothing_rather_than_guessing(self, tmp_path):
        rows = [{"公告日期": "2020-06-01", "进度": "实施"}]
        cache = self._cache(tmp_path, "000001", rows)
        assert dividend_yield.dividends("000001", cache=cache) is None

    def test_the_yield_is_cash_over_the_ex_date_price(self, tmp_path):
        # one event of one yuan against a ten yuan price is ten percent in that year
        rows = [{"派息": 10.0, "除权除息日": "2020-06-01", "进度": "实施"}]
        cache = self._cache(tmp_path, "000001", rows)
        series = pd.Series([10.0] * 5, index=pd.to_datetime(
            ["2020-05-28", "2020-05-29", "2020-06-01", "2020-06-02", "2020-06-03"]))
        event = dividend_yield.dividends("000001", cache=cache).iloc[0]
        assert event["cash"] / series.asof(event["ex_date"]) == pytest.approx(0.10)


class TestPreannounceSignal:
    def _events(self):
        return pd.DataFrame({
            "code": ["000001", "000001"],
            "filed": pd.to_datetime(["2020-04-28", "2020-08-28"]),
            "change": [10.0, -5.0]})

    def test_the_latest_filing_within_the_window_is_used(self):
        index = pd.date_range("2020-09-01", periods=5, freq="B")
        matrix = preannounce_ic.signal_matrix(self._events(), index, ["000001"], recent=60)
        assert matrix["000001"].iloc[0] == pytest.approx(-5.0)

    def test_a_filing_older_than_the_window_is_ignored(self):
        index = pd.date_range("2021-03-01", periods=5, freq="B")
        matrix = preannounce_ic.signal_matrix(self._events(), index, ["000001"], recent=60)
        assert matrix["000001"].isna().all()

    def test_a_filing_is_never_visible_before_it_is_filed(self):
        index = pd.date_range("2020-04-01", periods=20, freq="B")
        matrix = preannounce_ic.signal_matrix(self._events(), index, ["000001"], recent=60)
        before = matrix.loc[:pd.Timestamp("2020-04-27"), "000001"]
        assert before.isna().all(), "a forecast filed on the 28th must not score the 27th"
        assert matrix.loc[pd.Timestamp("2020-04-28"):, "000001"].notna().any()

    def test_codes_outside_the_pool_are_dropped(self):
        index = pd.date_range("2020-09-01", periods=3, freq="B")
        matrix = preannounce_ic.signal_matrix(self._events(), index, ["999999"], recent=60)
        assert matrix.shape == (3, 1)
        assert matrix["999999"].isna().all()

    def test_the_reporting_period_list_covers_the_intended_span(self):
        periods = list(preannounce_ic.periods())
        assert periods[0] == "20181231"
        assert periods[-1] == "20240930"
        # four report periods a year across six years, inclusive of the ends
        assert len(periods) == 24
