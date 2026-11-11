"""The PIT anchor for the performance reports is the statutory deadline.

``最新公告日期`` looks like the announcement date and is not: for the 2024 Q1 period it
puts 5,440 of 5,889 rows in April 2025.  These tests pin the anchor that is used instead,
and the as-of behaviour that keeps a stale report from living forever.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import perf_ic


def test_deadlines_follow_the_statutory_calendar():
    assert perf_ic.deadline_for("2024-03-31") == pd.Timestamp("2024-04-30")
    assert perf_ic.deadline_for("2024-06-30") == pd.Timestamp("2024-08-31")
    assert perf_ic.deadline_for("2024-09-30") == pd.Timestamp("2024-10-31")
    # The annual report for 2024 is due in April 2025, not December 2024.
    assert perf_ic.deadline_for("2024-12-31") == pd.Timestamp("2025-04-30")


def test_asof_matrix_uses_the_deadline_not_the_period_end():
    index = pd.DatetimeIndex(["2024-04-01", "2024-05-02"])
    events = pd.DataFrame({"code": ["600000"], "period": ["2024-03-31"],
                           "value": [15.0], "available": [pd.Timestamp("2024-04-30")]})
    matrix = perf_ic.asof_matrix(events, index, ["600000"], max_age=1000)
    assert np.isnan(matrix.loc["2024-04-01", "600000"])
    assert matrix.loc["2024-05-02", "600000"] == pytest.approx(15.0)


def test_asof_matrix_expires_a_report_older_than_the_max_age():
    index = pd.DatetimeIndex(["2024-05-02", "2026-05-02"])
    events = pd.DataFrame({"code": ["600000"], "period": ["2024-03-31"],
                           "value": [15.0], "available": [pd.Timestamp("2024-04-30")]})
    matrix = perf_ic.asof_matrix(events, index, ["600000"], max_age=100)
    assert matrix.loc["2024-05-02", "600000"] == pytest.approx(15.0)
    assert np.isnan(matrix.loc["2026-05-02", "600000"])
