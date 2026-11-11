"""The monthly loop: mark, check the band, and write the statement.

The plan is a monthly decision, so the tool has to answer three questions in one pass -- what
is the account worth, is the exposure outside its band, and what does the month look like. The
tests pin the arithmetic that decides whether a trade happens at all.
"""
from __future__ import annotations

import pytest

from tools import account_ledger as ledger
from tools import plan_orders


def test_drift_is_current_minus_target():
    nav = 1_000_000.0
    positions = {"510300": 25_000}                      # 100k at 4.0 -> 10% of NAV
    prices = {"510300": 4.0, "510500": 1.0, "513500": 1.0, "511010": 1.0}
    current = {code: positions.get(code, 0) * prices[code] / nav for code in ledger.LEGS}
    targets = plan_orders.target_weights(nav, nav)["weights"]
    drift = {code: current[code] - targets[code] for code in ledger.LEGS}
    assert drift["510300"] == pytest.approx(0.10 - 0.15)
    assert drift["511010"] == pytest.approx(-0.40)


def test_the_band_only_fires_when_the_risky_sleeve_moves_enough():
    # target risky 0.60; held 0.63 is inside the 5% band, 0.50 is not
    inside = plan_orders.target_weights(1_000_000.0, 1_000_000.0, held_risky=0.63)
    outside = plan_orders.target_weights(1_000_000.0, 1_000_000.0, held_risky=0.50)
    assert inside["traded"] is False and outside["traded"] is True


def test_report_month_picks_only_that_month():
    snapshots = [dict(date="2026-10-31", nav=3_000_000.0, cash=0.0, prices={}, positions={}),
                 dict(date="2026-11-02", nav=3_000_000.0, cash=0.0, prices={}, positions={}),
                 dict(date="2026-11-30", nav=3_030_000.0, cash=0.0, prices={}, positions={}),
                 dict(date="2026-12-01", nav=3_100_000.0, cash=0.0, prices={}, positions={})]
    report = ledger.monthly_report("2026-11", snapshots=snapshots, trades=[],
                                   cfg=dict(capital=3_000_000.0))
    assert report["days"] == 2
    assert report["nav_start"] == pytest.approx(3_000_000.0)
    assert report["nav_end"] == pytest.approx(3_030_000.0)


def test_drawdown_uses_the_peak_inside_the_account():
    snapshots = [dict(date="2026-11-02", nav=3_200_000.0, cash=0.0, prices={}, positions={}),
                 dict(date="2026-11-30", nav=3_040_000.0, cash=0.0, prices={}, positions={})]
    report = ledger.monthly_report("2026-11", snapshots=snapshots, trades=[],
                                   cfg=dict(capital=3_000_000.0))
    assert report["peak"] == pytest.approx(3_200_000.0)
    assert report["drawdown"] == pytest.approx(3_040_000.0 / 3_200_000.0 - 1.0)
