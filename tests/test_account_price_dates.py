"""A mark dated today against yesterday's quotes is not a flat day, it is a missing day.

The daily job runs at 20:15 and the Sina fund history sometimes still ends on the previous
session then.  Before this guard, ``mark --date today`` wrote today's date onto yesterday's
close and the curve read it as a genuine 0.00% day -- the account looked flat when it had
simply not been priced.  These tests pin the refusal, the provisional override and the
safe-leg quote check that let a stale 511010 through the gate.
"""
from __future__ import annotations

import json

import pytest

from tools import account_ledger as ledger

CFG = dict(capital=1_000_000.0, start_date="2026-11-11")
PRICES = {"511010": 141.363, "510300": 4.114, "510500": 6.628, "513500": 2.649}


@pytest.fixture
def held(monkeypatch, tmp_path):
    """A book holding one risky leg and the safe leg, so a mark has prices to date."""
    trades = tmp_path / "trades.jsonl"
    rows = [dict(date="2026-11-11", code="510300", action="buy", shares=100, price=4.0,
                 fee=0.0, source="manual"),
            dict(date="2026-11-11", code="511010", action="buy", shares=100, price=140.0,
                 fee=0.0, source="manual")]
    trades.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(ledger, "TRADES", trades)
    return tmp_path / "snapshots.jsonl"


def test_a_stale_quote_is_refused_and_writes_nothing(held):
    dates = {code: "2026-11-12" for code in PRICES}
    with pytest.raises(ValueError, match="not from 2026-11-13"):
        ledger.mark("2026-11-13", prices=dict(PRICES), price_dates=dates, path=held, cfg=CFG)
    assert not held.exists()


def test_the_refusal_names_every_stale_leg(held):
    dates = dict({code: "2026-11-13" for code in PRICES}, **{"511010": "2026-11-12"})
    with pytest.raises(ValueError) as caught:
        ledger.mark("2026-11-13", prices=dict(PRICES), price_dates=dates, path=held, cfg=CFG)
    assert "511010" in str(caught.value) and "510300" not in str(caught.value)


def test_a_provisional_mark_is_allowed_but_says_so(held):
    dates = {code: "2026-11-12" for code in PRICES}
    row = ledger.mark("2026-11-13", prices=dict(PRICES), price_dates=dates, allow_stale=True,
                      path=held, cfg=CFG)
    assert row["provisional"] is True
    assert set(row["price_dates"].values()) == {"2026-11-12"}
    assert ledger._read(held)[0]["provisional"] is True


def test_a_current_quote_is_stamped_with_its_date_and_not_flagged(held):
    dates = {code: "2026-11-13" for code in PRICES}
    row = ledger.mark("2026-11-13", prices=dict(PRICES), price_dates=dates, path=held, cfg=CFG)
    assert "provisional" not in row
    # Only legs actually held are stamped: a leg the book has sold has no quote in the NAV.
    assert row["price_dates"] == {"510300": "2026-11-13", "511010": "2026-11-13"}


def test_a_cash_only_book_needs_no_quote_date(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, "TRADES", tmp_path / "trades.jsonl")
    path = tmp_path / "snapshots.jsonl"
    dates = {code: "2026-11-12" for code in PRICES}
    row = ledger.mark("2026-11-13", prices=dict(PRICES), price_dates=dates, path=path, cfg=CFG)
    assert row["nav"] == pytest.approx(1_000_000.0)
    assert row["prices"] == {} and row["price_dates"] == {}


def test_a_stale_safe_leg_blocks_the_gate():
    """511010 is 40% of the book and used to be exempt because the loop walked the risky sleeve."""
    orders = [dict(code="511010", delta=2800, weight=0.40)]
    dates = {code: "2026-11-13" for code in PRICES}
    dates["511010"] = "2026-11-12"
    state = dict(date="2026-11-13", nav=1_000_000.0, peak=1_000_000.0, cash=1_000_000.0)
    problems = ledger.gate(orders, dict(PRICES), dates, state)
    assert any("511010" in problem for problem in problems)


def test_the_gate_passes_when_every_leg_is_from_today():
    orders = [dict(code="510300", delta=36000, weight=0.15)]
    dates = {code: "2026-11-13" for code in PRICES}
    state = dict(date="2026-11-13", nav=1_000_000.0, peak=1_000_000.0, cash=1_000_000.0)
    assert ledger.gate(orders, dict(PRICES), dates, state) == []
