"""The repair rules behind the plan backtest must be pinned, or every number above them is fiction.

A split read as a loss inflated the worst drawdown from -8.6% to -16% in the first version of this
work.  These tests fix the three cases: a split that must be rescaled, a payout the interface
lost, and a payout it recorded, which must not be added twice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import plan_backtest as pb


def series(values, start="2024-01-01"):
    index = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=index, dtype=float)


def test_a_halving_with_no_payout_is_rescaled_as_a_split():
    prices = series([2.0, 2.0, 1.0, 1.02])          # 1:2 split on day 2
    index, events = pb.total_return(prices, {})
    assert events and events[0]["kind"] == "split"
    assert index.pct_change().abs().max() < 0.05      # the fake -50% is gone
    assert index.iloc[-1] > index.iloc[0]             # the real +1% survives


def test_a_large_drop_with_no_payout_is_treated_as_a_lost_distribution():
    prices = series([10.0, 10.0, 8.8, 8.8])           # -12% with nothing recorded
    index, events = pb.total_return(prices, {})
    assert any(event["kind"] == "payout" and event["amount"] == pytest.approx(1.2)
               for event in events)
    assert index.iloc[2] == pytest.approx(index.iloc[1])   # flat, not a 12% loss


def test_a_recorded_payout_is_used_and_not_double_counted():
    prices = series([10.0, 10.0, 8.8])
    paid = {prices.index[2]: 1.2}
    index, events = pb.total_return(prices, paid)
    assert events == []                                   # nothing to infer
    assert index.iloc[2] == pytest.approx(index.iloc[1])


def test_a_small_move_is_left_alone():
    prices = series([10.0, 10.1, 9.9])
    index, events = pb.total_return(prices, {})
    assert events == []
    assert index.iloc[2] == pytest.approx((10.1 / 10.0) * (9.9 / 10.1))


def test_month_starts_are_timestamps():
    index = pd.bdate_range("2024-01-01", periods=60)
    starts = pb.month_starts(index)
    assert all(isinstance(stamp, pd.Timestamp) for stamp in starts)
    assert starts[0] == index[0]
    assert len(starts) in (3, 4)


def test_summarize_reports_the_share_inside_the_cap():
    rows = [dict(start="2024-01-01", annual=0.02, drawdown=-0.05),
            dict(start="2024-02-01", annual=0.06, drawdown=-0.12),
            dict(start="2024-03-01", annual=-0.01, drawdown=-0.09)]
    summary = pb.summarize(rows)
    assert summary["runs"] == 3
    assert summary["share_under_cap"] == pytest.approx(2 / 3)
    assert summary["drawdown_worst"] == pytest.approx(-0.12)
