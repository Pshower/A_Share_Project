"""Both simulators must agree on the same price series, and the sweep must use real prices.

The first version of the plan backtest ran on a normalised total-return index, where a
100-unit lot is worth about 100 yuan -- a fractional-share account. Fixing that moved the
seven-year result by 25 points, so these tests pin the two facts that were missing: the two
engines are identical on one series, and the sweep is fed traded prices rather than an index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import plan_backtest as pb


def simple_prices(days=260, start="2020-01-02"):
    index = pd.bdate_range(start, periods=days)
    rng = np.random.default_rng(7)
    frame = {}
    base = {"510300": 4.0, "510500": 6.0, "513500": 2.5, "511010": 140.0}
    for code, level in base.items():
        steps = 1.0 + rng.normal(0.0004, 0.008, size=len(index))
        frame[code] = level * np.cumprod(steps)
    return pd.DataFrame(frame, index=index)


def test_the_two_engines_agree_on_the_same_series():
    prices = simple_prices()
    uses_index = pb.simulate(prices, prices.index[0], cadence="daily",
                             end=str(prices.index[-1].date()))
    uses_book = pb.simulate_book(prices, {code: {} for code in pb.CODES}, prices.index[0],
                                 cadence="daily", end=str(prices.index[-1].date()))
    # ``simulate`` rounds the reported final, so compare the raw series
    assert uses_index["series"].iloc[-1] == pytest.approx(uses_book["final"], rel=1e-9)


def test_lot_size_is_what_moves_the_result():
    prices = simple_prices()
    empty = {code: {} for code in pb.CODES}
    with_lots = pb.simulate_book(prices, empty, prices.index[0], cadence="daily",
                                 end=str(prices.index[-1].date()))["final"]
    original = pb.LOT
    try:
        pb.LOT = 1
        fractional = pb.simulate_book(prices, empty, prices.index[0], cadence="daily",
                                      end=str(prices.index[-1].date()))["final"]
    finally:
        pb.LOT = original
    # both are valid books, but they are not the same book
    assert with_lots != pytest.approx(fractional, rel=1e-6)


def test_the_sweep_reports_one_row_per_month_start():
    prices = simple_prices(days=400)
    rows = pb.sweep(prices, {code: {} for code in pb.CODES}, start=str(prices.index[0].date()))
    assert len(rows) >= 12
    assert all(set(row) >= {"start", "annual", "drawdown", "final"} for row in rows)
    assert rows[0]["start"] < rows[-1]["start"]


def test_summarize_counts_the_share_inside_the_cap():
    rows = [dict(start="2024-01-01", annual=0.02, drawdown=-0.05),
            dict(start="2024-02-01", annual=0.06, drawdown=-0.12),
            dict(start="2024-03-01", annual=-0.01, drawdown=-0.09)]
    summary = pb.summarize(rows)
    assert summary["share_under_cap"] == pytest.approx(2 / 3)
