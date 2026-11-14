"""The exported backtest book must reproduce the sweep exactly, or it is just a picture.

The whole point of writing a simulation into the ledger is that the same rule can be read back
through curve and report -- but only if the numbers are the ones the sweep published.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from tools import account_ledger, plan_backtest


def prices(days=320):
    index = pd.bdate_range("2020-01-02", periods=days)
    rng = pd.Series(range(days), index=index, dtype=float)
    frame = {}
    for code, level in (("510300", 4.0), ("510500", 6.0), ("513500", 2.5), ("511010", 140.0)):
        frame[code] = level * (1.0 + (rng - days / 2) / (days * 4))
    return pd.DataFrame(frame, index=index)


def test_the_book_written_matches_the_simulation(tmp_path, monkeypatch):
    frame = prices()
    payouts = {code: {} for code in plan_backtest.CODES}
    monkeypatch.setattr(account_ledger, "ACCOUNT_DIR", tmp_path)
    monkeypatch.setattr(plan_backtest, "raw_series",
                        lambda code, start, end=None: (frame[code].loc[start:], {}))
    result = plan_backtest.export_account(str(frame.index[0].date()), name="backtest",
                                          cadence="daily")
    simulated = plan_backtest.simulate_book(frame, payouts, frame.index[0], cadence="daily")
    assert result["final"] == pytest.approx(simulated["final"], rel=1e-9)
    rows = [json.loads(line) for line in
            (tmp_path / "backtest" / "snapshots.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == result["days"]
    assert rows[-1]["nav"] == pytest.approx(result["final"], rel=1e-6)
    fills = [json.loads(line) for line in
             (tmp_path / "backtest" / "trades.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(fills) == result["fills"]
    assert all(row["source"] == "backtest" for row in fills)


def test_the_exported_log_rebuilds_its_own_cash(tmp_path, monkeypatch):
    """The ledger's premise is that the trade log is the only source of truth.

    The first export broke it: dividends were credited to the snapshots but never written as
    fills, and fills were priced before slippage, so replaying the log came out 69,742 and then
    28,220 away from the book's own cash.  Both showed up as a negative cash balance the moment
    anything marked the book with a later price.
    """
    frame = prices()
    payout_day = frame.index[120]
    payouts = {code: {} for code in plan_backtest.CODES}
    payouts["510300"] = {payout_day: 0.25}
    monkeypatch.setattr(account_ledger, "ACCOUNT_DIR", tmp_path)
    monkeypatch.setattr(plan_backtest, "raw_series",
                        lambda code, start, end=None: (
                            frame[code].loc[start:],
                            {payout_day: 0.25} if code == "510300" else {}))
    result = plan_backtest.export_account(str(frame.index[0].date()), name="backtest",
                                          cadence="daily")
    rows = [json.loads(line) for line in
            (tmp_path / "backtest" / "trades.jsonl").read_text(encoding="utf-8").splitlines() if line]
    dividends = [row for row in rows if row["action"] == "dividend"]
    assert dividends and dividends[0]["code"] == "510300"
    book = account_ledger.replay(rows, 1_000_000.0)
    snaps = [json.loads(line) for line in
             (tmp_path / "backtest" / "snapshots.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert book["positions"] == snaps[-1]["positions"]
    # A tenth of a yuan is the fee rounding; the two bugs above were four and five figures.
    assert book["cash"] == pytest.approx(snaps[-1]["cash"], abs=0.15)
    assert result["final"] == pytest.approx(snaps[-1]["nav"], abs=0.01)   # the row is rounded to cents


def test_a_sealed_export_refuses_a_later_mark(tmp_path, monkeypatch):
    sealed = tmp_path / "snapshots.jsonl"
    monkeypatch.setattr(account_ledger, "TRADES", tmp_path / "trades.jsonl")
    with pytest.raises(ValueError, match="closed book"):
        account_ledger.mark("2026-11-14", prices={"511010": 141.0}, path=sealed,
                            cfg=dict(capital=1e6, start_date="2019-01-02", sealed=True,
                                     source="backtest"))
    assert not sealed.exists()
