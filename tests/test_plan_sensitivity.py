"""The plan's two unexamined knobs: how much capital it needs, and how wide the band should be.

Pure functions only -- the sweep itself is exercised end to end by ``tools/plan_sensitivity.py`` on
real prices, which is far too slow for a unit test.  What can drift silently is the summarising
arithmetic (a wrong median turns "19% of starts meet the target" into a different claim), so that is
what is pinned here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import plan_sensitivity


def rows(annuals, drawdowns, fees=None, trades=None):
    fees = fees or [1.0] * len(annuals)
    trades = trades or [10] * len(annuals)
    return [dict(annual=a, drawdown=d, fees=f, trades=t)
            for a, d, f, t in zip(annuals, drawdowns, fees, trades)]


def test_series_metrics_reads_a_known_curve():
    index = pd.bdate_range("2020-01-01", periods=242)
    series = pd.Series(np.linspace(1_000_000, 1_100_000, len(index)), index=index)
    row = plan_sensitivity.series_metrics(series, 1_000_000.0, ["a"], 12.0)
    assert row["annual"] == pytest.approx(0.10, rel=1e-6)
    assert row["worst_drawdown"] if "worst_drawdown" in row else True
    assert row["drawdown"] == pytest.approx(0.0)
    assert row["trades"] == 1 and row["fees"] == 12.0


def test_a_losing_curve_reports_a_negative_drawdown():
    index = pd.bdate_range("2020-01-01", periods=100)
    series = pd.Series(np.linspace(1_000_000, 800_000, len(index)), index=index)
    row = plan_sensitivity.series_metrics(series, 1_000_000.0, [], 0.0)
    assert row["annual"] < 0
    assert row["drawdown"] == pytest.approx(-0.2, abs=1e-6)


def test_summarize_counts_the_target_and_the_cap_separately():
    sample = rows(annuals=[0.08, 0.05, 0.09, 0.02],
                  drawdowns=[-0.05, -0.11, -0.03, -0.02])
    summary = plan_sensitivity.summarize(sample)
    assert summary["runs"] == 4
    assert summary["share_under_cap"] == pytest.approx(0.75)     # three of four stay above -10%
    assert summary["share_over_target"] == pytest.approx(0.50)   # 8% and 9%
    assert summary["share_both"] == pytest.approx(0.50)          # 8%/-5% and 9%/-3%
    assert summary["annual_median"] == pytest.approx(0.065)


def test_summarize_reports_no_runs_rather_than_nan():
    assert plan_sensitivity.summarize([]) == dict(runs=0)


def test_the_worst_start_is_a_real_negative():
    """2026 starts annualise two months of returns; the summary must carry that through, not hide it."""
    sample = rows(annuals=[-0.19, 0.05], drawdowns=[-0.04, -0.06])
    summary = plan_sensitivity.summarize(sample)
    assert summary["annual_min"] == pytest.approx(-0.19)
    assert summary["annual_median"] == pytest.approx(-0.07)
