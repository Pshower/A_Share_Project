"""Booking the plan must be one command, and it must never claim to be a real fill.

The failure this guards against is a paper account that looks real: if ``book`` produced rows
indistinguishable from reported fills, every later return would silently mix a model execution
with actual ones.
"""
from __future__ import annotations

import pytest

from tools import account_ledger as ledger

CFG = dict(capital=1_000_000.0, start_date="2026-11-11")
PRICES = {"510300": 4.0, "510500": 6.0, "513500": 2.5, "511010": 140.0}


def test_booking_buys_the_four_legs_and_labels_them_modelled(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, "market_price", lambda code, stamp, field="open": PRICES[code])
    trades = tmp_path / "trades.jsonl"
    result = ledger.book("2026-11-12", path=trades, cfg=CFG)
    assert result["booked"] == 4
    assert all(row["source"] == "model_open" for row in result["trades"])
    book = ledger.replay(ledger._read(trades), CFG["capital"])
    assert set(book["positions"]) == set(PRICES)
    assert book["cash"] > 0


def test_booking_twice_does_not_double_the_position(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, "market_price", lambda code, stamp, field="open": PRICES[code])
    trades = tmp_path / "trades.jsonl"
    ledger.book("2026-11-12", path=trades, cfg=CFG)
    first = ledger.replay(ledger._read(trades), CFG["capital"])["positions"]
    second_result = ledger.book("2026-11-12", path=trades, cfg=CFG)
    second = ledger.replay(ledger._read(trades), CFG["capital"])["positions"]
    # the book is already at target, so the second run trades nothing
    assert second_result["booked"] == 0
    assert second == first


def test_sells_are_booked_before_buys(monkeypatch, tmp_path):
    prices = dict(PRICES)
    monkeypatch.setattr(ledger, "market_price", lambda code, stamp, field="open": prices[code])
    trades = tmp_path / "trades.jsonl"
    ledger._append(trades, dict(recorded_at="x", date="2026-11-12", code="511010", action="buy",
                                shares=5_000, price=140.0, fee=210.0, source="manual"))
    result = ledger.book("2026-11-12", path=trades, cfg=CFG)
    actions = [row["action"] for row in result["trades"]]
    assert actions and actions[0] == "sell"
