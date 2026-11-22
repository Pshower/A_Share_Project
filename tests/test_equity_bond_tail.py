"""The tail re-check must obey its own pre-registration before any number leaves the tool.

Three behaviours carry the verdicts, so they are pinned here: the one manual payout repair may
only ever *add* a missing row and never override the published table; both optimised arms must
pick their grid point by the frozen objective on the frozen window; and the three verdict rules
must be mechanical -- the same rows always produce the same pass/fail, so a table in a report
cannot quietly reorder the comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import equity_bond_tail as ebt


def frame_from(returns: dict, start="2020-01-01") -> pd.DataFrame:
    """A price frame from daily returns, all legs the same length."""
    index = pd.bdate_range(start, periods=len(next(iter(returns.values()))))
    prices = {}
    for code, row in returns.items():
        path = np.cumprod(1.0 + np.asarray(row, dtype=float))
        prices[code] = pd.Series(path, index=index)
    return pd.DataFrame(prices)


def rng_window(days=ebt.ESTIMATE_WINDOW, seed=7):
    """A (days, 4) return window with three restless legs and one calm bond leg."""
    generator = np.random.default_rng(seed)
    restless = generator.normal(0.0, [0.012, 0.020, 0.016], size=(days, 3))
    calm = generator.normal(0.0002, 0.0004, size=(days, 1))
    return np.concatenate([restless, calm], axis=1)


def write_dist(root, code, rows):
    target = root / "fund"
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["date", "per_unit"]).to_csv(
        target / f"dist_{code}.csv", index=False, encoding="utf-8-sig")


def test_payout_table_adds_the_manual_repair_only_where_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ebt, "ONLINE", tmp_path)
    write_dist(tmp_path, "510500", [("2024-05-17", 0.087)])

    without = ebt.payout_table("510500", include_manual=False)
    assert pd.Timestamp("2022-08-29") not in without
    assert without[pd.Timestamp("2024-05-17")] == pytest.approx(0.087)

    with_manual = ebt.payout_table("510500", include_manual=True)
    assert with_manual[pd.Timestamp("2022-08-29")] == pytest.approx(0.92)

    # A cache that someday carries the real row must not be overridden by the manual one.
    write_dist(tmp_path, "510500", [("2022-08-29", 0.90), ("2024-05-17", 0.087)])
    kept = ebt.payout_table("510500", include_manual=True)
    assert kept[pd.Timestamp("2022-08-29")] == pytest.approx(0.90)


def test_other_legs_gain_no_manual_payout(tmp_path, monkeypatch):
    monkeypatch.setattr(ebt, "ONLINE", tmp_path)
    write_dist(tmp_path, "510300", [("2024-01-18", 0.069)])
    paid = ebt.payout_table("510300", include_manual=True)
    assert len(paid) == 1                       # the repair belongs to 510500 only


def test_simplex_grid_is_complete_and_long_only():
    grid = ebt.simplex_grid()
    assert len(grid) == 286
    assert np.allclose(grid.sum(axis=1), 1.0)
    assert grid.min() == pytest.approx(0.0)
    assert grid.max() == pytest.approx(1.0)
    assert any(np.allclose(point, [1.0, 0.0, 0.0, 0.0]) for point in grid)


def test_invvol_weights_the_calm_leg_most():
    window = rng_window()
    target = ebt.dynamic_target("invvol", window)
    assert target[3] == pytest.approx(target.max())
    expected = 1.0 / window.std(axis=0, ddof=1)
    assert np.allclose(target, expected / expected.sum())


def test_minvar_on_the_grid_puts_everything_on_the_calm_leg():
    window = rng_window()
    target = ebt.dynamic_target("minvar", window)
    assert target[3] == pytest.approx(1.0)


def test_cvar_target_drops_the_tail_heavy_leg():
    generator = np.random.default_rng(11)
    window = rng_window(seed=13)
    crashes = generator.binomial(1, 0.05, size=ebt.ESTIMATE_WINDOW) == 1
    window[crashes, 0] = -0.08                     # leg 0 eats rare -8% days
    target = ebt.dynamic_target("cvar", window)
    assert target[0] == pytest.approx(0.0)


def test_establishment_cost_is_booked_on_day_zero_and_drift_needs_no_repair():
    flat = frame_from({"510300": [0.0] * 40, "510500": [0.0] * 40,
                       "513500": [0.0] * 40, "511010": [0.0] * 40})
    result = ebt.run_arm("eqw", flat, flat.index[0])
    assert result["series"].iloc[0] == pytest.approx(1.0 - ebt.COST_PER_SIDE)
    assert result["turnover"] == pytest.approx(1.0)      # nothing to repair when prices sit still
    assert result["series"].iloc[-1] == pytest.approx(1.0 - ebt.COST_PER_SIDE)


def test_costs_follow_the_turnover_exactly():
    generator = np.random.default_rng(3)
    row = generator.normal(0.0005, 0.01, size=40)
    live = frame_from({"510300": row, "510500": row * 0.5, "513500": row * 1.5,
                       "511010": row * 0.02})
    result = ebt.run_arm("eqw", live, live.index[0])
    assert result["cost"] == pytest.approx(ebt.COST_PER_SIDE * result["turnover"])


def test_metrics_match_hand_math():
    series = pd.Series([1.0, 1.01, 0.99, 1.02])
    got = ebt.metrics(series, turnover=1.0, cost=ebt.COST_PER_SIDE)
    assert got["final"] == pytest.approx(1.02)
    assert got["drawdown"] == pytest.approx(0.99 / 1.01 - 1.0)
    assert got["cvar95"] == pytest.approx(min(series.pct_change().dropna()))
    assert got["steps"] == 3
    assert got["cagr"] == pytest.approx(1.02 ** (ebt.ANNUAL / 3) - 1.0)


LOW_STARTS = ["2018-01-0%d" % i for i in range(2, 11)] + ["2018-05-04"]


def rows_for(starts, cvar_by_start, sharpe_by_start, cagr_by_start=None):
    """Fabricated sweep rows: one per start, the two optimised arms steered per hypothesis."""
    cagr_by_start = cagr_by_start or {start: 0.05 for start in starts}
    rows = []
    for start in starts:
        for arm in ("eqw", "fixed6040", "invvol", "minvar", "cvar"):
            steer = arm in ("minvar", "cvar")
            rows.append(dict(
                arm=arm, start=start,
                cagr=cagr_by_start[start] if arm == "fixed6040" else 0.04,
                sharpe=sharpe_by_start[start] if steer else 0.50,
                cvar95=cvar_by_start[start] if steer else -0.030))
    return rows


def buckets_low_only(starts):
    return {start: dict(y10=1.5, bucket="低 carry") for start in starts}


def test_verdict_h1_passes_when_both_arms_cut_the_tail_consistently():
    starts = LOW_STARTS
    rows = rows_for(starts, {start: -0.020 for start in starts}, {start: 0.50 for start in starts})
    verdict = ebt.verdicts(rows, buckets_low_only(starts))
    assert verdict["H1"]["minvar_share"] == pytest.approx(1.0)
    assert verdict["H1"]["cvar_share"] == pytest.approx(1.0)
    assert verdict["H1"]["pass"] is True


def test_verdict_h1_fails_when_one_arm_misses_the_share_bar():
    starts = LOW_STARTS
    cvar = {start: (-0.020 if i <= 7 else -0.035) for i, start in enumerate(starts, start=1)}
    rows = rows_for(starts, cvar, {start: 0.50 for start in starts})   # 7/10, exactly at the bar
    verdict = ebt.verdicts(rows, buckets_low_only(starts))
    assert verdict["H1"]["minvar_share"] == pytest.approx(0.70)
    assert verdict["H1"]["pass"] is True                      # the frozen bar is inclusive
    cvar_miss = {start: (-0.020 if i <= 6 else -0.035) for i, start in enumerate(starts, start=1)}
    rows = rows_for(starts, cvar_miss, {start: 0.50 for start in starts})
    assert ebt.verdicts(rows, buckets_low_only(starts))["H1"]["pass"] is False


def test_verdict_h2_is_overturned_only_by_a_material_sharpe_edge():
    starts = LOW_STARTS[:9]
    tight = {start: 0.53 for start in starts}                 # edge 0.03 <= 0.05: no win
    assert ebt.verdicts(rows_for(starts, {start: -0.02 for start in starts}, tight),
                        buckets_low_only(starts))["H2"]["pass"] is True
    loud = {start: (0.80 if i <= 6 else 0.50) for i, start in enumerate(starts, start=1)}
    verdict = ebt.verdicts(rows_for(starts, {start: -0.02 for start in starts}, loud),
                           buckets_low_only(starts))
    assert verdict["H2"]["arms"]["minvar"]["wins"] is True
    assert verdict["H2"]["pass"] is False


def test_verdict_h3_checks_the_low_carry_bucket_only():
    starts = LOW_STARTS
    buckets = buckets_low_only(starts)
    buckets["2018-05-04"] = dict(y10=2.5, bucket="中 carry")      # one start leaves the subset
    calm = {start: 0.05 for start in starts}
    rows = rows_for(starts, {start: -0.02 for start in starts},
                    {start: 0.50 for start in starts}, cagr_by_start=calm)
    verdict = ebt.verdicts(rows, buckets)
    assert verdict["H3"]["low_starts"] == 9
    assert verdict["H3"]["pass"] is True
    noisy = dict(calm)                       # flip five of nine: the median itself must move
    for start in starts[:5]:
        noisy[start] = 0.03
    rows = rows_for(starts, {start: -0.02 for start in starts},
                    {start: 0.50 for start in starts}, cagr_by_start=noisy)
    assert ebt.verdicts(rows, buckets)["H3"]["pass"] is False


def test_feasible_starts_own_a_full_estimation_window():
    index = pd.bdate_range("2020-01-01", periods=700)
    frame = pd.DataFrame(1.0, index=index, columns=ebt.CODES)
    starts = ebt.feasible_starts(frame, floor="2020-01-01")
    assert starts                                # 700 bdays: about two years are eligible
    assert all(frame.index.get_loc(stamp) >= ebt.ESTIMATE_WINDOW - 1 for stamp in starts)
    early = [stamp for stamp in ebt.pb.month_starts(index) if stamp < starts[0]]
    assert early and all(frame.index.get_loc(stamp) < ebt.ESTIMATE_WINDOW - 1 for stamp in early)


def test_sweep_pairs_every_start_across_all_arms():
    generator = np.random.default_rng(5)
    index = pd.bdate_range("2020-01-01", periods=800)
    frame = pd.DataFrame(
        np.cumprod(1.0 + generator.normal(0.0003, 0.01, size=(800, 4)), axis=0),
        index=index, columns=ebt.CODES)
    starts = ebt.feasible_starts(frame, floor="2021-01-01")
    assert starts                                  # 800 bdays: warmup and runway both fit
    rows = ebt.sweep(frame, starts)
    arms = {row["arm"] for row in rows}
    assert arms == set(ebt.ARMS)
    per_start = {}
    for row in rows:
        per_start.setdefault(row["start"], set()).add(row["arm"])
    assert per_start and all(set(arms) == covered for covered in per_start.values())
