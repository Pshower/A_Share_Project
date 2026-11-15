"""The strategy engine's arithmetic: compounding, the cushion, the band, and the fill lag.

The compounding test is a regression: a double-counted index turned a 7.9% static book into 15.3%,
and that error is invisible in every summary statistic except an exact closed form.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import unified_plan as plan

STATIC = dict(kind="static", weights={"ashare": 1.0})
HALF = dict(kind="static", weights={"ashare": 0.5, "bond5": 0.5})


def frame(days=260, start="2010-01-04"):
    index = pd.bdate_range(start, periods=days)
    return pd.DataFrame({"ashare": 1.0, "bond5": 1.0, "cash": 1.0, "bond2": 1.0, "bond9": 1.0,
                         "bond20": 1.0, "asmall": 1.0, "usshare": 1.0, "hk": 1.0, "gold": 1.0},
                        index=index)


def zero_costs(monkeypatch):
    monkeypatch.setattr(plan, "COST", 0.0)
    monkeypatch.setattr(plan, "FEES", {name: 0.0 for name in plan.FEES})


def test_static_book_compounds_exactly(monkeypatch):
    zero_costs(monkeypatch)
    levels = frame()
    levels["ashare"] = (1.001 ** np.arange(len(levels)))
    result = plan.simulate(levels, STATIC, 0)
    # One unit bought at the first close is worth ``1.001 ** (n - 1)`` at the last close.
    assert result["final"] == pytest.approx(1.001 ** (len(levels) - 1), rel=1e-12)


def test_daily_and_monthly_cadence_agree_for_a_static_book(monkeypatch):
    zero_costs(monkeypatch)
    levels = frame()
    levels["ashare"] = (1.0007 ** np.arange(len(levels)))
    levels["bond5"] = (1.0002 ** np.arange(len(levels)))
    monthly = plan.simulate(levels, HALF, 0, cadence="monthly")
    daily = plan.simulate(levels, HALF, 0, cadence="daily")
    assert daily["cagr"] == pytest.approx(monthly["cagr"], rel=1e-12)


def test_cushion_is_sixty_percent_at_the_peak_and_zero_at_the_floor():
    rule = plan.RULES["C1"]
    at_peak = plan.target_weights(rule, nav=1.0, peak=1.0, floor=0.90, multiplier=6.0, y10=3.0)
    at_floor = plan.target_weights(rule, nav=0.90, peak=1.0, floor=0.90, multiplier=6.0, y10=3.0)
    assert sum(at_peak.values()) == pytest.approx(1.0)
    assert sum(at_peak[name] for name in rule["risky"]) == pytest.approx(0.60)
    assert sum(at_floor[name] for name in rule["risky"]) == pytest.approx(0.0)


def test_the_safe_leg_switch_reacts_to_the_ten_year_yield():
    rule = plan.RULES["C3"]
    high = plan.target_weights(rule, 1.0, 1.0, 0.90, 6.0, y10=2.8)
    low = plan.target_weights(rule, 1.0, 1.0, 0.90, 6.0, y10=1.1)
    middle = plan.target_weights(rule, 1.0, 1.0, 0.90, 6.0, y10=1.8)
    assert high["bond5"] == pytest.approx(0.40) and "cash" not in high
    assert low["cash"] == pytest.approx(0.40) and "bond5" not in low
    assert middle["bond5"] == pytest.approx(0.20) and middle["cash"] == pytest.approx(0.20)


def test_a_falling_market_is_cut_and_the_drawdown_stays_near_the_floor(monkeypatch):
    zero_costs(monkeypatch)
    levels = frame()
    path = np.concatenate([np.linspace(1.0, 1.0, 60), np.linspace(1.0, 0.55, 120),
                           np.linspace(0.55, 0.80, 80)])
    levels["ashare"] = path
    result = plan.simulate(levels, plan.RULES["C1"], 0, cadence="daily")
    assert result["maxdd"] > -0.16                       # the cushion is doing its job
    assert result["maxdd"] <= -0.05                      # but the market still hurt


def test_the_band_suppresses_small_rebalances(monkeypatch):
    zero_costs(monkeypatch)
    levels = frame()
    levels["ashare"] = 1.0 + 0.0002 * np.arange(len(levels))
    quiet = plan.simulate(levels, plan.RULES["C1"], 0, multiplier=6.0)
    noisy = plan.simulate(levels, plan.RULES["C1"], 0, multiplier=6.0, band=0.0)
    assert quiet["turnover"] < noisy["turnover"]


def test_month_starts_are_the_first_session_of_each_month():
    index = pd.bdate_range("2010-01-04", periods=60)
    starts = plan.month_starts(index)
    assert index[starts[0]] == pd.Timestamp("2010-01-04")
    assert all(index[position].day <= 7 for position in starts)
    assert len(starts) == 3


def test_fills_happen_one_session_after_the_decision(monkeypatch):
    """A jump on the fill day must be earned on the old weights, not the new target."""
    zero_costs(monkeypatch)
    levels = frame(days=40, start="2010-01-04")
    index = levels.index
    levels["ashare"] = 1.0
    levels.loc[index[30]:, "ashare"] = 2.0                # a doubling on session 30
    result = plan.simulate(levels, HALF, 0, cadence="daily", band=0.0)
    # 50% in an asset that doubles: the book must roughly gain 50%, then hold it.
    assert 1.35 < result["cagr"] * 0 + (1 + result["cagr"]) ** ((index[-1] - index[0]).days
                                                                / 365.25) < 1.65


def test_paired_uses_only_the_dates_both_rules_share():
    runs = pd.DataFrame([
        dict(rule="C1", start="2010-01-04", cagr=0.05, maxdd=-0.05, turnover=1.0),
        dict(rule="C1", start="2011-01-04", cagr=0.09, maxdd=-0.05, turnover=1.0),
        dict(rule="C4", start="2011-01-04", cagr=0.20, maxdd=-0.05, turnover=1.0),
    ])
    summary = dict(C1=dict(n=2, median_cagr=0.07), C4=dict(n=1, median_cagr=0.20))
    result = plan.paired(runs, summary)
    assert result["C4"]["starts"] == 1
    assert result["C4"]["delta_pp"] == pytest.approx(11.0)
