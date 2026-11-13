"""A NAV is worth keeping only if it can be rebuilt from the closes of its own date.

The daily job used to price a day off whatever the fund endpoint ended on, so a row's date and
its prices could disagree.  These tests pin the audit that finds such a row, the repair that
rebuilds it from the real close, and the fill check -- plus the stamping of rows that were
written before ``price_dates`` existed but whose prices were right all along.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from tools import account_ledger as ledger
from tools import audit_marks

DAYS = pd.to_datetime(["2026-11-12", "2026-11-13"])


def _frames():
    """Two sessions: the 11-13 row that a stale job would have priced off 11-12."""
    bars = {
        "510300": [[4.145, 4.114], [4.124, 4.083]],
        "511010": [[141.34, 141.363], [141.37, 141.339]],
    }
    return {code: pd.DataFrame(rows, index=DAYS, columns=["open", "close"])
            for code, rows in bars.items()}


def _row(day, prices, cash=4520.74):
    return dict(date=day, cash=cash, nav=0.0, positions={"510300": 100, "511010": 10},
                prices=prices)


def test_a_row_priced_off_the_previous_session_is_named():
    frames = _frames()
    row = _row("2026-11-13", {"510300": 4.114, "511010": 141.363})
    findings = audit_marks.audit_snapshots([row], audit_marks._wide(frames, "close"))
    assert {item["code"] for item in findings} == {"510300", "511010"}
    assert all(item["origin"] == "2026-11-12" for item in findings)
    assert all(item["issue"] == "priced off another session" for item in findings)


def test_a_row_on_its_own_closes_is_not_a_finding():
    frames = _frames()
    row = _row("2026-11-13", {"510300": 4.083, "511010": 141.339})
    assert audit_marks.audit_snapshots([row], audit_marks._wide(frames, "close")) == []


def test_repair_rebuilds_the_nav_from_the_real_closes():
    frames = _frames()
    row = _row("2026-11-13", {"510300": 4.114, "511010": 141.363})
    fixed, changed, stamped = audit_marks.repair_snapshots([row], audit_marks._wide(frames, "close"))
    expected = round(4520.74 + 100 * 4.083 + 10 * 141.339, 10)
    assert fixed[0]["nav"] == pytest.approx(expected)
    assert fixed[0]["prices"] == {"510300": 4.083, "511010": 141.339}
    assert fixed[0]["price_dates"] == {"510300": "2026-11-13", "511010": "2026-11-13"}
    assert changed[0]["delta"] == pytest.approx(fixed[0]["nav"] - changed[0]["nav_before"])
    assert stamped == []


def test_a_correct_row_keeps_its_nav_but_gains_its_price_dates():
    frames = _frames()
    row = _row("2026-11-13", {"510300": 4.083, "511010": 141.339})
    row["nav"] = ledger.nav(row["positions"], row["cash"], row["prices"])
    before = row["nav"]
    fixed, changed, stamped = audit_marks.repair_snapshots([row], audit_marks._wide(frames, "close"))
    assert changed == [] and stamped == ["2026-11-13"]
    assert fixed[0]["nav"] == before
    assert fixed[0]["price_dates"] == {"510300": "2026-11-13", "511010": "2026-11-13"}


def test_a_manual_fill_is_listed_as_unverifiable_not_failed():
    frames = _frames()
    opens, closes = audit_marks._wide(frames, "open"), audit_marks._wide(frames, "close")
    found = audit_marks.audit_fills([dict(date="2026-11-13", code="510300", price=4.0,
                                          action="buy", source="manual")], opens, closes)
    assert found[0]["issue"].startswith("manual fill")


def test_a_model_fill_must_match_its_bar():
    frames = _frames()
    opens, closes = audit_marks._wide(frames, "open"), audit_marks._wide(frames, "close")
    good = audit_marks.audit_fills([dict(date="2026-11-13", code="510300", price=4.124,
                                         action="buy", source="model_open")], opens, closes)
    bad = audit_marks.audit_fills([dict(date="2026-11-13", code="510300", price=4.114,
                                        action="buy", source="model_open")], opens, closes)
    assert good == []
    assert bad[0]["truth"] == pytest.approx(4.124) and "open" in bad[0]["issue"]


def test_the_simulated_book_refuses_to_be_audited():
    with pytest.raises(ValueError, match="own simulator"):
        audit_marks.run("backtest")
