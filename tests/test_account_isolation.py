"""A paper book and a real book must not share a single file.

The failure this guards is the one that makes an account tool dangerous: a modelled fill
landing in the real ledger and reporting a position nobody holds.
"""
from __future__ import annotations

import pytest

from tools import account_ledger as ledger


def test_use_switches_every_path(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "ACCOUNT_DIR", tmp_path)
    monkeypatch.setattr(ledger, "ACCOUNTS", ("live", "paper"))
    try:
        ledger.use("paper")
        assert ledger.TRADES == tmp_path / "paper" / "trades.jsonl"
        assert ledger.SNAPSHOTS == tmp_path / "paper" / "snapshots.jsonl"
        ledger.use("live")
        assert ledger.CONFIG == tmp_path / "live" / "account.json"
    finally:
        ledger.use("live")


def test_an_unknown_account_is_rejected():
    with pytest.raises(ValueError, match="Unknown account"):
        ledger.use("shadow")


def test_booking_the_paper_book_leaves_the_live_book_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "ACCOUNT_DIR", tmp_path)
    monkeypatch.setattr(ledger, "ACCOUNTS", ("live", "paper"))
    prices = {"510300": 4.0, "510500": 6.0, "513500": 2.5, "511010": 140.0}
    monkeypatch.setattr(ledger, "market_price", lambda code, stamp, field="open": prices[code])
    try:
        ledger.use("live")
        ledger.init(1_000_000, "2026-11-11")
        ledger.use("paper")
        ledger.init(1_000_000, "2026-11-11")
        ledger.book("2026-11-12", cfg=dict(capital=1_000_000.0, start_date="2026-11-11"))
        paper = ledger.replay(ledger._read(ledger.TRADES), 1_000_000.0)
        ledger.use("live")
        live = ledger.replay(ledger._read(ledger.TRADES), 1_000_000.0)
        assert paper["positions"] and not live["positions"]
        assert live["cash"] == pytest.approx(1_000_000.0)
    finally:
        ledger.use("live")
