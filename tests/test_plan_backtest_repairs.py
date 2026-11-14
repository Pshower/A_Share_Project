"""The repair rules behind the plan backtest must be pinned, or every number above them is fiction.

A split read as a loss inflated the worst drawdown from -8.6% to -16% in the first version of this
work.  The first fix over-corrected in the other direction: any drop of 10% or more with nothing
recorded became a "lost distribution", which erased the April 2025 tariff crash and lifted the
published median from 4.4% to 5.3%.  So the tests now pin a third rule: only a gap that lands on a
simple split ratio is a split, only a *small* gap may be guessed to be a payout, and everything else
is a real fall that stays in the series.
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


def test_a_large_drop_that_is_not_a_split_ratio_stays_a_loss():
    """-12% is a crash, not a split and not a lost distribution."""
    prices = series([10.0, 10.0, 8.8, 8.8])
    index, events = pb.total_return(prices, {})
    assert events == [] or all(event["kind"] != "payout" for event in events)
    assert index.iloc[2] / index.iloc[1] - 1 == pytest.approx(-0.12)


def test_a_gap_off_the_simple_ratios_is_reported_as_an_unrepaired_fall():
    """-48% is close to a 1:2 split but not close enough; it must be visible, not silently fixed."""
    prices = series([10.0, 10.0, 5.2, 5.2])
    _, _, events = pb.price_and_payouts(prices, {})
    assert any(event["kind"] == "unrepaired_fall" for event in events)


def test_an_unrecorded_distribution_is_not_invented():
    """513500 has no distributions at all; a -4% day is a -4% day, not a payout nobody published."""
    prices = series([10.0, 10.0, 9.6])
    index, events = pb.total_return(prices, {})
    assert events == []
    assert index.iloc[2] / index.iloc[1] - 1 == pytest.approx(-0.04)


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


def test_price_and_payouts_reports_every_repair():
    prices = series([2.0, 2.0, 1.0, 1.02])
    repaired, payouts, events = pb.price_and_payouts(prices, {})
    assert payouts == {}
    assert [event["kind"] for event in events] == ["split"]
    assert repaired.pct_change().abs().max() < 0.05


def test_price_and_payouts_keeps_a_crash_as_a_crash():
    prices = series([10.0, 10.0, 8.8])
    repaired, payouts, events = pb.price_and_payouts(prices, {})
    assert payouts == {}
    assert repaired.iloc[2] == pytest.approx(8.8)         # nothing invented
    assert events == [] or all(event["kind"] != "payout" for event in events)


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
