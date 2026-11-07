"""Unit checks for the stage-2 volatility-scaling evaluation."""

import numpy as np
import pandas as pd
import pytest

from tools import exposure_stage2 as stage2


def test_exposure_rule_caps_at_one_and_stops_at_the_floor():
    vol = pd.Series([0.10, 0.209, 0.40, 3.00], index=pd.RangeIndex(4))
    target = stage2.exposure_rule(vol, 0.209, floor=0.20)
    assert target.iloc[0] == 1.0                    # below target vol -> capped
    assert abs(target.iloc[1] - 1.0) < 1e-12        # exactly at target
    assert abs(target.iloc[2] - 0.5225) < 1e-12     # scaled down
    assert target.iloc[3] == 0.20                   # floored


def test_exposure_rule_keeps_missing_volatility_missing():
    vol = pd.Series([np.nan, 0.209], index=pd.RangeIndex(2))
    target = stage2.exposure_rule(vol, 0.209)
    assert pd.isna(target.iloc[0])


def test_simulate_with_full_exposure_and_no_cost_tracks_the_index():
    returns = pd.Series([0.01, -0.02, 0.03, 0.00])
    targets = pd.Series([1.0, 1.0, 1.0, 1.0])
    result = stage2.simulate(returns, targets, buy_rate=0.0, sell_rate=0.0)
    expected = float(np.prod(1 + returns.to_numpy()))
    assert abs(result["final"] - expected) < 1e-12
    # The first row pays for the initial build, so traded notional is the weight path.
    assert result["traded"] >= 1.0


def test_simulate_charges_more_to_sell_than_to_buy():
    returns = pd.Series([0.0, 0.0, 0.0])
    up = stage2.simulate(returns, pd.Series([0.0, 0.5, 0.5]))
    down = stage2.simulate(returns, pd.Series([0.5, 0.5, 0.0]))
    assert down["cost"] > up["cost"] > 0


def test_simulate_costs_nothing_when_the_target_never_moves():
    returns = pd.Series([0.01, -0.01, 0.02])
    held = stage2.simulate(returns, pd.Series([0.0, 0.0, 0.0]))
    assert held["cost"] == 0.0 and held["final"] == 1.0


def test_summarize_reports_drawdown_and_calmar_on_a_known_path():
    nav = pd.Series([1.0, 1.2, 0.9, 1.1])
    summary = stage2.summarize(nav)
    assert summary["total"] == 1.1
    assert abs(summary["max_drawdown"] - (0.9 / 1.2 - 1)) < 1e-12
    assert summary["calmar"] > 0


def test_verdict_requires_both_calmar_and_a_return_floor():
    def payload(rule_calmar, control_calmar, rule_return, control_return, rule_dd=-0.10):
        return dict(index=dict(
            rule=dict(summary=dict(calmar=rule_calmar, annual_return=rule_return, max_drawdown=rule_dd)),
            control=dict(summary=dict(calmar=control_calmar, annual_return=control_return))))

    assert stage2.verdict(payload(0.12, 0.10, 0.05, 0.05))["passed"] is True
    assert stage2.verdict(payload(0.08, 0.10, 0.05, 0.05))["passed"] is False   # Calmar worse
    assert stage2.verdict(payload(0.12, 0.10, 0.00, 0.05))["passed"] is False   # return 5pp behind
