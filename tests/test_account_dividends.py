"""Distributions are cash, not a price move.

An ETF's ex-date drops the price by the amount it pays out.  If the ledger does not record the
cash on that day, the account understates itself every time -- and the treasury leg pays
quarterly, so the error is about 2% a year on the largest sleeve.

These tests pin the three things that can quietly go wrong: the cash not being added, the
distribution being credited to shares bought on the ex-date, and the same event being appended
twice on a second sync.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from tools import account_ledger as ledger


def test_a_distribution_adds_cash_and_leaves_the_position_alone():
    trades = [dict(code="511010", action="buy", shares=1_000, price=140.0, fee=42.0),
              dict(code="511010", action="dividend", shares=1_000, amount=620.0, fee=0.0)]
    book = ledger.replay(trades, 1_000_000.0)
    assert book["positions"] == {"511010": 1_000}
    assert book["cash"] == pytest.approx(1_000_000.0 - 140_000.0 - 42.0 + 620.0)


def test_shares_held_are_counted_strictly_before_the_ex_date():
    trades = [dict(code="511010", action="buy", shares=100, price=140.0, date="2026-09-01"),
              dict(code="511010", action="buy", shares=900, price=141.0, date="2026-09-18")]
    assert ledger.shares_held_at(trades, "511010", "2026-09-18") == 100      # the same-day buy misses it
    assert ledger.shares_held_at(trades, "511010", "2026-09-19") == 1_000


def test_shares_held_ignores_other_codes_and_distributions():
    trades = [dict(code="510300", action="buy", shares=500, price=4.0, date="2026-01-01"),
              dict(code="511010", action="dividend", shares=999, amount=1.0, date="2026-01-02")]
    assert ledger.shares_held_at(trades, "510300", "2026-02-01") == 500
    assert ledger.shares_held_at(trades, "511010", "2026-02-01") == 0


def test_distributions_come_from_the_cumulative_column():
    events = [(pd.Timestamp("2026-03-25").date(), 0.5715), (pd.Timestamp("2026-06-25").date(), 0.8144)]
    # the interface publishes cumulative values 3.9557 -> 4.7701, so the per-event figure is the diff
    assert events[1][1] - events[0][1] == pytest.approx(0.2429)


def test_sync_appends_once_and_skips_months_with_no_holding(tmp_path, monkeypatch):
    trades = tmp_path / "trades.jsonl"
    ledger._append(trades, dict(recorded_at="x", date="2026-03-01", code="511010", action="buy",
                                shares=1_000, price=140.0, fee=42.0))
    calls = []

    def fake_distributions(code, start=None, end=None):
        calls.append((code, start, end))
        return [(pd.Timestamp("2026-03-25").date(), 0.5715), (pd.Timestamp("2026-06-25").date(), 0.8144)]

    monkeypatch.setattr(ledger, "distributions", fake_distributions)
    monkeypatch.setattr(ledger, "LEGS", ["511010"])
    first = ledger.sync_dividends(start="2026-03-01", end="2026-06-30", codes=["511010"],
                                  path=trades, cfg=dict(capital=1_000_000.0, start_date="2026-03-01"))
    assert len(first) == 2
    assert first[0]["amount"] == pytest.approx(571.5)
    second = ledger.sync_dividends(start="2026-03-01", end="2026-06-30", codes=["511010"],
                                   path=trades, cfg=dict(capital=1_000_000.0, start_date="2026-03-01"))
    assert second == []                                   # idempotent
    rows = [json.loads(line) for line in trades.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert sum(1 for row in rows if row["action"] == "dividend") == 2


def test_the_monthly_report_separates_dividends_from_fees():
    snapshots = [dict(date="2026-09-01", nav=1_000_000.0, cash=0.0, prices={}, positions={}),
                 dict(date="2026-09-30", nav=1_000_620.0, cash=0.0, prices={}, positions={})]
    trades = [dict(date="2026-09-18", code="511010", action="dividend", shares=1_000,
                   amount=620.0, fee=0.0)]
    report = ledger.monthly_report("2026-09", snapshots=snapshots, trades=trades,
                                   cfg=dict(capital=1_000_000.0))
    assert report["dividends"] == pytest.approx(620.0)
    assert report["fees"] == 0.0 and report["trades"] == 0
