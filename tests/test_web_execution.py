"""The execution books and the shadow recorder agree with the CLI or they are not worth showing.

These tests run against the real books in ``reports/account``: the page's curve must equal
``tools.account_ledger.curve`` row for row, its holdings must equal the ledger's replay, and a
book that was never opened must not be invented on the way to the screen.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.web import execution
from tools import account_ledger as ledger

ROOT = Path(__file__).resolve().parents[1]


def test_the_books_list_matches_the_ledger_curve_of_each_book():
    rows = {row["name"]: row for row in execution.books(ROOT)}
    for name, row in rows.items():
        config = ROOT / "reports" / "account" / name / "account.json"
        if not config.exists():
            assert row["opened"] is False
            continue
        cfg = ledger.config(config)
        snapshots = execution._read(config.parent / "snapshots.jsonl")
        expected = ledger.curve(snapshots=snapshots, cfg=cfg)
        assert row["nav"] == expected["nav"]
        assert row["worst_drawdown"] == expected["worst_drawdown"]
        assert row["last"] == expected["last"]


def test_the_backtest_book_reports_its_full_curve():
    row = execution.book(ROOT, "backtest")
    assert row["opened"] is True
    assert len(row["dates"]) == len(row["navs"]) > 1_000
    assert row["curve"]["worst_drawdown"] < 0


def test_holdings_come_from_the_trade_log_not_from_the_snapshot():
    row = execution.book(ROOT, "paper")
    assert row["opened"] is True
    replayed = ledger.replay(execution._read(ROOT / "reports/account/paper/trades.jsonl"),
                             row["capital"])
    assert {item["code"]: item["shares"] for item in row["positions"]} == replayed["positions"]
    assert row["cash"] == pytest.approx(replayed["cash"])


def test_an_unopened_book_is_reported_as_unopened(tmp_path):
    (tmp_path / "reports").mkdir()
    assert execution.books(tmp_path) == [dict(name=name, opened=False) for name in execution.BOOKS]
    assert execution.book(tmp_path, "live") == dict(name="live", opened=False)


def test_an_unknown_book_is_refused():
    with pytest.raises(ValueError, match="unknown book"):
        execution.book(ROOT, "savings")


def test_the_shadow_view_counts_what_the_recorder_wrote():
    row = execution.shadow(ROOT)
    rows = execution._read(ROOT / execution.SHADOW)
    assert row["rows"] == len(rows)
    assert row["decided"] + row["stale"] == len(rows)
    assert len(row["recent"]) == min(8, len(rows))
    assert row["recent"][-1]["panel_date"] == rows[-1]["panel_date"]


def test_a_cash_only_book_has_no_prices_to_show(tmp_path):
    base = tmp_path / "reports" / "account" / "live"
    base.mkdir(parents=True)
    (base / "account.json").write_text(json.dumps(dict(capital=1e6, start_date="2026-11-11")))
    (base / "snapshots.jsonl").write_text(
        json.dumps(dict(date="2026-11-11", nav=1e6, cash=1e6, positions={}, prices={})) + "\n")
    row = execution.book(tmp_path, "live")
    assert row["navs"] == [1e6] and row["positions"] == [] and row["trades"]["count"] == 0
