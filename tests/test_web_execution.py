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


def test_each_decision_day_carries_its_thirty_names():
    row = execution.shadow(ROOT)
    assert row["days"], "the recorder has decisions but the view exposed none"
    for day in row["days"]:
        assert len(day["names"]) == 30
        assert all(item["code"] and "score" in item for item in day["names"])
    latest = row["days"][-1]
    assert latest["panel_date"] == "2026-11-13"
    assert [item["change"] for item in latest["names"]].count("in") == len(latest["entries"])


def test_names_are_joined_from_the_local_snapshot():
    lookup = execution.names(ROOT)
    if not lookup:
        pytest.skip("no local code-name snapshot")
    assert lookup["601633"] == "\u957f\u57ce\u6c7d\u8f66"
    assert lookup["600905"] == "\u4e09\u5ce1\u80fd\u6e90"


def test_a_fresh_book_reports_no_entries_and_no_overlap():
    """The 2024-12-31 seed and the 11-10 restart are books being opened, not rotated."""
    for day in execution.shadow(ROOT)["days"]:
        if not day["held_from"]:
            assert day["entries"] == [] and day["exits"] == [] and day["overlap"] is None


def test_consecutive_days_agree_with_their_own_overlap():
    days = [day for day in execution.shadow(ROOT)["days"] if day["held_from"]]
    assert days, "expected at least one rotated day"
    for day in days:
        assert len(day["entries"]) == 30 - day["overlap"]
        assert len(day["exits"]) == len(day["entries"])
        entry_codes = {item["code"] for item in day["entries"]}
        exit_codes = {item["code"] for item in day["exits"]}
        assert not entry_codes & exit_codes


def test_each_days_names_are_ordered_so_a_topk_slice_is_meaningful():
    """The page shows the first K entries, so the order is part of the contract."""
    for day in execution.shadow(ROOT)["days"]:
        scores = [item["score"] for item in day["names"] if item["score"] is not None]
        assert scores == sorted(scores), day["panel_date"]
        assert len(set(item["code"] for item in day["names"])) == len(day["names"])


def test_the_api_chains_the_benchmark_columns_too(tmp_path):
    """The page draws three lines; the API must carry three chained levels."""
    row = dict(status="decided", panel_date="2026-11-13", held_from="2026-11-10",
               book=["000001"], score={"000001": 0.2},
               interval=dict(**{"return": 0.02}),
               benchmark=dict(**{"from": "2026-11-10", "to": "2026-11-13",
                                 "pool_equal": 0.01, "hs300": -0.01}))
    log = tmp_path / "reports" / "shadow" / "e1_shadow.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in (row,)) + "\n",
                   encoding="utf-8")
    (log.parent / "e1_state.json").write_text("{}", encoding="utf-8")
    data = execution.shadow(tmp_path)
    assert data["chained_nav"] == pytest.approx(1.02)
    assert data["chained_pool"] == pytest.approx(1.01)
    assert data["chained_hs300"] == pytest.approx(0.99)
    day = data["days"][0]
    assert day["int_return"] == pytest.approx(0.02)
    assert day["pool_return"] == pytest.approx(0.01)
