"""A calendar that answers "holiday" for dates it has never seen is worse than no calendar.

Both daily jobs read False as "do nothing today", so a cache that ends in December would silence
them from January onward without a single error.  These tests pin the intended loudness.
"""
from __future__ import annotations

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


def test_the_horizon_is_a_year_so_the_cache_refreshes_before_it_runs_out():
    assert trading_calendar.HORIZON >= 300
