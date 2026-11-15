"""The scorecard must not turn "never tested" into a middle score, and must not read
"answered but empty" as "unavailable".  Those two errors would silently reorder the ranking.
"""
from __future__ import annotations

import pytest

from tools import data_scorecard as card


def probe(status="ok", last="2026-11-13", **extra):
    row = dict(status=status, last_date=last)
    row.update(extra)
    return row


def test_an_untested_family_gets_no_score_rather_than_a_middle_one():
    entry = dict(grade="U", sources=["x"], evidence="never tested", tradable="未知",
                 tradable_grade="未知", cited="—")
    row = card.score_family("未测家族", entry, {"x": probe()}, "2026-11-15")
    assert row["effectiveness"] is None
    assert row["tradable_score"] is None


def test_a_tested_but_failed_family_still_scores_below_an_untested_one_in_rank():
    tested = dict(grade="C", sources=["x"], evidence="zero", tradable="否",
                  tradable_grade="否", cited="doc")
    row = card.score_family("已测无效", tested, {"x": probe()}, "2026-11-15")
    assert row["effectiveness"] is not None and row["effectiveness"] < 4


def test_freshness_drives_the_availability_score():
    assert card.availability(probe(last="2026-11-13"), "2026-11-15") == 10.0
    assert card.availability(probe(last="2026-10-01"), "2026-11-15") == 7.0
    assert card.availability(probe(last="2025-09-08"), "2026-11-15") == 1.0   # a year stale is dead\n    assert card.availability(probe(last="2026-03-01"), "2026-11-15") == 3.0   # 259 days: stale, not dead


def test_an_answered_but_empty_source_is_reachable_not_missing():
    """The unlock query returned no rows because that stock had none, not because the feed is down."""
    assert card.availability(probe(status="empty", last=None), "2026-11-15") == 6.0


def test_a_refused_connection_scores_zero_availability():
    assert card.availability(probe(status="blocked", last=None), "2026-11-15") == 0.0
    assert card.availability(probe(status="timeout", last=None), "2026-11-15") == 0.0


def test_the_weights_are_a_distribution():
    assert sum(card.WEIGHTS.values()) == pytest.approx(1.0)


def test_every_family_declares_a_grade_the_scorer_knows():
    for name, entry in card.EVIDENCE.items():
        assert entry["grade"] in card.GRADE_SCORE or entry["grade"] == "U", name
        assert entry.get("tradable_grade") in card.TRADABLE_SCORE, name
