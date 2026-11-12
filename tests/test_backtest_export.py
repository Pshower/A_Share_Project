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
