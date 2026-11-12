"""A fill can be booked without typing a price, and the log says which kind it is.

The plan executes at the next open, so the next open is the honest stand-in for a fill that
already happened. These tests pin that the stand-in is used, that a real fill overrides it, and
that the row records which one it was -- a modelled fill and a reported fill must never be
confused in a return series.
"""
from __future__ import annotations

import pytest

from tools import account_ledger as ledger

CFG = dict(capital=1_000_000.0)


def test_a_missing_price_is_taken_from_the_market(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, "market_price", lambda code, stamp, field="open": 4.15)
    row = ledger.record_trade("2026-11-12", "510300", "buy", 1_000, path=tmp_path / "t.jsonl", cfg=CFG)
    assert row["price"] == pytest.approx(4.15)
    assert row["source"] == "model_open"


def test_the_field_choice_is_visible_in_the_row(monkeypatch, tmp_path):
    calls = []

    def fake(code, stamp, field="open"):
        calls.append(field)
        return 4.10 if field == "open" else 4.20

    monkeypatch.setattr(ledger, "market_price", fake)
    row = ledger.record_trade("2026-11-12", "510300", "buy", 100, path=tmp_path / "t.jsonl",
                              cfg=CFG, field="close")
    assert calls == ["close"] and row["source"] == "model_close" and row["price"] == pytest.approx(4.20)


def test_a_reported_fill_wins_and_is_labelled(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, "market_price", lambda *a, **k: 4.15)
    row = ledger.record_trade("2026-11-12", "510300", "buy", 100, price=4.18,
                              path=tmp_path / "t.jsonl", cfg=CFG)
    assert row["price"] == pytest.approx(4.18) and row["source"] == "manual"


def test_a_missing_bar_is_an_error_not_a_guess(monkeypatch, tmp_path):
    def boom(code, stamp, field="open"):
        raise ValueError("No bar for 510300 on 2026-11-15")

    monkeypatch.setattr(ledger, "market_price", boom)
    with pytest.raises(ValueError, match="No bar"):
        ledger.record_trade("2026-11-15", "510300", "buy", 100, path=tmp_path / "t.jsonl", cfg=CFG)


def test_modelled_fills_still_respect_cash():
    with pytest.raises(ValueError, match="cash"):
        ledger.record_trade("2026-11-12", "510300", "buy", 1_000_000, price=4.15,
                            path=None, cfg=dict(capital=1_000.0))
