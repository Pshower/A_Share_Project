"""The return record is only trustworthy if a day can be marked twice without doubling it.

The daily job may run again after a late fill, and a duplicated NAV would silently double a day
in every return and every drawdown computed from it.  These tests pin the idempotency and the
curve arithmetic.
"""
from __future__ import annotations

import pytest

from tools import account_ledger as ledger

CFG = dict(capital=1_000_000.0, start_date="2026-11-11")


def test_marking_the_same_day_twice_keeps_one_point(tmp_path):
    path = tmp_path / "snapshots.jsonl"
    cfg = dict(CFG)
    first = ledger.mark("2026-11-11", prices={"511010": 141.34, "510300": 4.15,
                                              "510500": 6.748, "513500": 2.662},
                        path=path, cfg=cfg)
    second = ledger.mark("2026-11-11", prices={"511010": 141.34, "510300": 4.15,
                                               "510500": 6.748, "513500": 2.662},
                         path=path, cfg=cfg)
    rows = ledger._read(path)
    assert len(rows) == 1 and first["nav"] == second["nav"]


def test_the_curve_reports_return_peak_and_worst_drawdown():
    snapshots = [dict(date="2026-11-11", nav=1_000_000.0),
                 dict(date="2026-11-12", nav=1_050_000.0),
                 dict(date="2026-11-13", nav=1_030_000.0)]
    row = ledger.curve(snapshots=snapshots, cfg=CFG)
    assert row["days"] == 3
    assert row["total_return"] == pytest.approx(0.03)
    assert row["peak"] == pytest.approx(1_050_000.0)
    assert row["drawdown"] == pytest.approx(1_030_000 / 1_050_000 - 1)
    assert row["worst_drawdown"] == pytest.approx(1_030_000 / 1_050_000 - 1)


def test_the_curve_is_sorted_even_if_the_file_is_not():
    snapshots = [dict(date="2026-11-13", nav=1_030_000.0),
                 dict(date="2026-11-11", nav=1_000_000.0),
                 dict(date="2026-11-12", nav=1_050_000.0)]
    row = ledger.curve(snapshots=snapshots, cfg=CFG)
    assert row["first"] == "2026-11-11" and row["last"] == "2026-11-13"
    assert row["nav"] == pytest.approx(1_030_000.0)


def test_an_empty_account_has_no_curve():
    assert ledger.curve(snapshots=[], cfg=CFG)["status"] == "no_snapshots"
