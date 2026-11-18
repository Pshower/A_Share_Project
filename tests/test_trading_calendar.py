"""A calendar that answers "holiday" for dates it has never seen is worse than no calendar.

Both daily jobs read False as "do nothing today", so a cache that ends in December would silence
them from January onward without a single error.  These tests pin the intended loudness.
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd
import pytest

from tools import trading_calendar


def calendar():
    return pd.DatetimeIndex(pd.to_datetime(["2026-11-12", "2026-11-13", "2026-11-16"]))


def test_a_session_inside_the_calendar_is_true():
    assert trading_calendar.is_trading_day("2026-11-16", calendar=calendar()) is True


def test_a_weekend_inside_the_calendar_is_false():
    assert trading_calendar.is_trading_day("2026-11-14", calendar=calendar()) is False


def test_a_date_past_the_calendar_raises_rather_than_answering_false():
    with pytest.raises(ValueError, match="calendar ends"):
        trading_calendar.is_trading_day("2027-01-04", calendar=calendar())


def test_the_horizon_asks_for_a_week_of_slack_not_a_year():
    """The exchange publishes to the next year end, so "a year ahead" could never be satisfied
    and every call downloaded the file again."""
    assert trading_calendar.COVER <= 30


def write_cache(root, last, age_days=0):
    root.mkdir(parents=True, exist_ok=True)
    index = pd.bdate_range(end=last, periods=40)
    path = root / "trade_dates.csv"
    path.write_text(
        pd.DataFrame({"trade_date": index.strftime("%Y-%m-%d")}).to_csv(index=False),
        encoding="utf-8")
    if age_days:
        stamp = time.time() - age_days * 86400
        os.utime(path, (stamp, stamp))
    return path


class FakeAkshare:
    calls = 0

    @staticmethod
    def tool_trade_date_hist_sina():
        FakeAkshare.calls += 1
        return pd.DataFrame({"trade_date": pd.bdate_range("2024-01-01", "2027-12-31").strftime("%Y-%m-%d")})


class BrokenAkshare:
    @staticmethod
    def tool_trade_date_hist_sina():
        raise OSError("feed down")


def test_a_cache_with_a_week_of_slack_is_not_refetched(tmp_path, monkeypatch):
    today = pd.Timestamp.now().normalize()
    write_cache(tmp_path, today + pd.Timedelta(days=30))
    FakeAkshare.calls = 0
    monkeypatch.setitem(sys.modules, "akshare", FakeAkshare)
    dates = trading_calendar.load(cache=tmp_path)
    assert FakeAkshare.calls == 0
    assert dates.max() > today


def test_a_cache_that_runs_out_is_refetched(tmp_path, monkeypatch):
    today = pd.Timestamp.now().normalize()
    write_cache(tmp_path, today, age_days=1)          # yesterday's fetch: due again
    FakeAkshare.calls = 0
    monkeypatch.setitem(sys.modules, "akshare", FakeAkshare)
    dates = trading_calendar.load(cache=tmp_path)
    assert FakeAkshare.calls == 1
    assert dates.max() > today + pd.Timedelta(days=trading_calendar.COVER)


def test_a_cache_refreshed_today_is_not_refetched_again(tmp_path, monkeypatch):
    """At the end of December the week of slack can never be met; one attempt per day is enough."""
    today = pd.Timestamp.now().normalize()
    write_cache(tmp_path, today)                       # written moments ago
    FakeAkshare.calls = 0
    monkeypatch.setitem(sys.modules, "akshare", FakeAkshare)
    dates = trading_calendar.load(cache=tmp_path)
    assert FakeAkshare.calls == 0
    assert dates.max() >= today


def test_a_failed_refresh_still_answers_from_a_cache_that_covers_today(tmp_path, monkeypatch):
    today = pd.Timestamp.now().normalize()
    write_cache(tmp_path, today)
    monkeypatch.setitem(sys.modules, "akshare", BrokenAkshare)
    dates = trading_calendar.load(cache=tmp_path)
    assert dates.max() >= today
