"""The restatement must refund only tax that was actually recorded."""

from __future__ import annotations

import json

from tools import restate_book


def make_book(tmp_path, rows, snapshots=None):
    folder = tmp_path / "reports" / "account" / "book"
    folder.mkdir(parents=True)
    (folder / "trades.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    snaps = snapshots or []
    (folder / "snapshots.jsonl").write_text(
        "\n".join(json.dumps(row) for row in snaps) + "\n", encoding="utf-8")
    (folder / "account.json").write_text(json.dumps({"capital": 1000000, "cash": 100.0}),
                                         encoding="utf-8")
    return folder


def test_a_commission_only_fee_is_not_refunded(tmp_path, monkeypatch):
    folder = make_book(tmp_path, [
        dict(action="sell", code="510300", date="2024-01-02", shares=10000, price=4.0, fee=12.0),
    ], [dict(date="2024-01-02", nav=1000000.0, cash=1000.0, positions={}, prices={})])
    monkeypatch.setattr(restate_book, "ACCOUNT", tmp_path / "reports" / "account")
    record = restate_book.restate("book", apply_changes=True, log=lambda *_: None)
    assert record["legs"] == 0 and record["removed"] == 0.0
    assert not (folder / "restatement_20261115.json").exists()


def test_a_recorded_tax_is_refunded_once(tmp_path, monkeypatch):
    rows = [dict(action="sell", code="510300", date="2024-01-02", shares=10000, price=4.0,
                 fee=32.0)]                                   # 12 commission + 20 tax
    snaps = [dict(date="2024-01-02", nav=1000000.0, cash=1000.0, positions={}, prices={}),
             dict(date="2024-01-03", nav=1000100.0, cash=1100.0, positions={}, prices={})]
    folder = make_book(tmp_path, rows, snaps)
    monkeypatch.setattr(restate_book, "ACCOUNT", tmp_path / "reports" / "account")
    record = restate_book.restate("book", apply_changes=True, log=lambda *_: None)
    assert record["legs"] == 1
    assert record["removed"] == 20.0
    restated = [json.loads(line) for line in
                (folder / "snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
    assert restated[-1]["nav"] == 1000120.0
    again = restate_book.restate("book", apply_changes=True, log=lambda *_: None)
    assert again["legs"] == 1                      # the record is idempotent, not re-applied
