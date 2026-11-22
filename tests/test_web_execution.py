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
    # The latest decision is read from the ledger, not hardcoded: the recorder writes a new row
    # every trading day, and a frozen date here fails the suite the first time the job runs.
    rows = [json.loads(line) for line in
            (ROOT / "reports" / "shadow" / "e1_shadow.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    decided = [row for row in rows if row.get("status") == "decided" and row.get("book")]
    row = execution.shadow(ROOT)
    assert row["days"], "the recorder has decisions but the view exposed none"
    for day in row["days"]:
        assert len(day["names"]) == 30
        assert all(item["code"] and "score" in item for item in day["names"])
    latest = row["days"][-1]
    assert latest["panel_date"] == str(decided[-1]["panel_date"])
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


def test_a_stale_row_between_decisions_does_not_shift_the_chain(tmp_path):
    """The chain has one entry per row; indexing it by decided-count made every later day lag."""
    rows = [
        dict(status="decided", panel_date="2026-11-10", held_from=None, book=["000001"],
             score={"000001": 0.2}, interval={"from": "2026-11-03", "return": 0.01},
             benchmark={"pool_equal": 0.005, "hs300": 0.0}),
        dict(status="stale", panel_date="2026-11-11", reason="panel unchanged"),
        dict(status="stale", panel_date="2026-11-11", reason="panel unchanged"),
        dict(status="decided", panel_date="2026-11-12", held_from="2026-11-10", book=["000001"],
             score={"000001": 0.2}, interval={"from": "2026-11-10", "return": 0.02},
             benchmark={"pool_equal": 0.01, "hs300": -0.01}),
    ]
    log = tmp_path / "reports" / "shadow" / "e1_shadow.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (log.parent / "e1_state.json").write_text("{}", encoding="utf-8")
    data = execution.shadow(tmp_path)
    assert data["chained_nav"] == pytest.approx(1.01 * 1.02)
    assert data["days"][-1]["chained_nav"] == pytest.approx(data["chained_nav"])
    assert data["days"][-1]["pool_return"] == pytest.approx(0.01)


def test_the_variant_view_reads_every_letter_the_recorder_wrote():
    rows = execution.variant_books(ROOT)
    assert {row["letter"] for row in rows} == {"A", "B", "C", "D"}
    for row in rows:
        path = ROOT / execution.VARIANTS / ("%s_shadow.jsonl" % row["letter"])
        if not path.exists():
            assert row["opened"] is False
            continue
        log = execution._read(path)
        decided = [record for record in log if record.get("status") == "decided"]
        assert row["opened"] is True
        assert row["rows"] == len(log)
        assert row["decided"] == len(decided)
        if decided:
            assert row["latest"]["panel_date"] == decided[-1]["panel_date"]


def test_the_variant_chain_is_compounded_from_the_ledgers_own_intervals():
    for row in execution.variant_books(ROOT):
        if not row.get("opened"):
            continue
        log = execution._read(ROOT / execution.VARIANTS / ("%s_shadow.jsonl" % row["letter"]))
        level = 1.0
        for record in log:
            change = (record.get("interval") or {}).get("return")
            if change is not None:
                level *= 1.0 + float(change)
        assert row["chained_nav"] == pytest.approx(level, abs=1e-6)


def test_the_alloc_view_reads_the_three_books_and_their_latest_rows():
    rows = execution.alloc_books(ROOT)
    assert {row["arm"] for row in rows} == {"fixed6040", "minvar", "cvar"}
    for row in rows:
        path = ROOT / execution.ALLOC / ("%s_alloc.jsonl" % row["arm"])
        if not path.exists():
            assert row["opened"] is False
            continue
        log = execution._read(path)
        decisions = [record for record in log if record.get("kind") == "decision"]
        assert row["opened"] is True
        assert row["decisions"] == len(decisions)
        assert row["latest"]["price_date"] == decisions[-1]["price_date"]
        assert row["latest"]["weights"] == decisions[-1]["weights"]
        assert row["judged_from"] == "2026-11-20"


def test_the_shadow_views_do_not_invent_unopened_books(tmp_path):
    (tmp_path / "reports").mkdir()
    assert all(row["opened"] is False for row in execution.variant_books(tmp_path))
    assert all(row["opened"] is False for row in execution.alloc_books(tmp_path))
