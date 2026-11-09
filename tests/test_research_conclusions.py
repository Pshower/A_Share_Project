"""Tests for the arithmetic behind every number the research documents quote.

The conclusions of the 2026-11 series all come from a handful of functions: a rank IC and
its Newey-West t, a feature-selection rule, a paired difference, a forward event window, a
cross-sectional correlation, and two append-only stores.  None of them was tested, and each
of them decides whether a document says "clears the bar" or "does not".

Every test here builds a series whose answer is known by construction, so a silent change in
the statistic fails a test rather than quietly changing a published figure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import alpha_ic, archive_fund_flow, feature_backtest, fund_flow_collinearity
from tools import shadow_run, supply_overlay, unlock_ic


def dates(n=40, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B")


def ranks(values):
    return pd.Series(values).rank().to_numpy(dtype=float)


class TestRankIC:
    def test_a_perfect_ordering_scores_one(self):
        index = dates(5)
        rng = np.random.default_rng(0)
        values = rng.normal(size=(5, 60))
        target = values * 3.0
        series = alpha_ic.rank_ic(values, target, pd.DatetimeIndex(index))
        assert series.dropna().mean() == pytest.approx(1.0, abs=1e-6)

    def test_a_reversed_ordering_scores_minus_one(self):
        rng = np.random.default_rng(1)
        values = rng.normal(size=(4, 60))
        target = -values
        series = alpha_ic.rank_ic(values, target, pd.DatetimeIndex(dates(4)))
        assert series.dropna().mean() == pytest.approx(-1.0, abs=1e-6)

    def test_a_thin_cross_section_is_skipped(self):
        rng = np.random.default_rng(2)
        values = rng.normal(size=(3, 10))
        series = alpha_ic.rank_ic(values, values, pd.DatetimeIndex(dates(3)))
        assert series.isna().all()

    def test_the_t_is_positive_for_a_real_signal_and_finite(self):
        rng = np.random.default_rng(3)
        values = rng.normal(size=(24, 200))
        target = values + rng.normal(scale=4.0, size=(24, 200))
        clean = alpha_ic.rank_ic(values, target, pd.DatetimeIndex(dates(24))).dropna()
        stats = alpha_ic.ic_stats(clean, 20, 0.031)
        assert stats["mean_ic"] > 0
        assert np.isfinite(stats["t_nw_h"])
        assert stats["t_nw_h"] > 2


class TestFeatureSelection:
    def test_signs_follow_the_measured_direction(self):
        table = {"cheap": dict(mean_ic=0.05, t=4.0), "costly": dict(mean_ic=-0.06, t=5.0),
                 "noise": dict(mean_ic=0.002, t=0.3)}
        chosen = feature_backtest.select_features(table)
        assert chosen == {"cheap": 1.0, "costly": -1.0}

    def test_a_large_ic_without_significance_is_rejected(self):
        chosen = feature_backtest.select_features({"lucky": dict(mean_ic=0.09, t=1.1)})
        assert chosen == {}

    def test_the_thresholds_are_respected(self):
        table = {"marginal": dict(mean_ic=0.012, t=2.1)}
        assert feature_backtest.select_features(table, min_ic=0.01, min_t=2.0) == {"marginal": 1.0}
        assert feature_backtest.select_features(table, min_ic=0.02, min_t=2.0) == {}


class TestPairedDifference:
    def test_a_constant_edge_produces_a_large_positive_t(self):
        index = dates(240)
        base = pd.Series(np.random.default_rng(0).normal(0, 0.01, 240), index=index)
        control = base
        treated = base + 0.001
        result = supply_overlay.paired(control, treated, block=20)
        assert result["annual_gap"] > 0
        assert result["block_t"] > 3

    def test_no_difference_produces_no_edge(self):
        index = dates(240)
        series = pd.Series(np.random.default_rng(1).normal(0, 0.01, 240), index=index)
        result = supply_overlay.paired(series, series, block=20)
        assert result["annual_gap"] == pytest.approx(0.0, abs=1e-12)
        assert np.isnan(result["block_t"]) or abs(result["block_t"]) < 1e-6

    def test_a_negative_edge_is_reported_as_negative(self):
        index = dates(240)
        base = pd.Series(np.random.default_rng(2).normal(0, 0.01, 240), index=index)
        result = supply_overlay.paired(base, base - 0.001, block=20)
        assert result["annual_gap"] < 0


class TestEventWindow:
    def test_an_unlock_is_seen_only_before_it_happens(self):
        index = dates(30)
        events = pd.DataFrame({"date": [index[10]], "code": ["000001"], "ratio": [0.05]})
        matrix = unlock_ic.forward_matrix(events, index, ["000001"], horizon=5)
        hit = matrix["000001"]
        assert hit.loc[index[10]] == pytest.approx(0.0)
        assert (hit.loc[index[5]:index[9]] > 0).all()

    def test_the_window_counts_sessions_not_calendar_days(self):
        # an event at session 35 belongs to sessions 30 through 34 and to nothing else
        index = dates(40)
        events = pd.DataFrame({"date": [index[35]], "code": ["000001"], "ratio": [0.05]})
        matrix = unlock_ic.forward_matrix(events, index, ["000001"], horizon=5)
        marked = matrix["000001"][matrix["000001"] > 0].index
        assert list(marked) == list(index[30:35])
        assert matrix["000001"].loc[index[25]] == pytest.approx(0.0)
        assert matrix["000001"].loc[index[35]:].abs().sum() == pytest.approx(0.0)

    def test_a_code_outside_the_pool_is_ignored(self):
        index = dates(30)
        events = pd.DataFrame({"date": [index[10]], "code": ["999999"], "ratio": [0.05]})
        matrix = unlock_ic.forward_matrix(events, index, ["000001"], horizon=5)
        assert matrix["000001"].abs().sum() == pytest.approx(0.0)


class TestFlowCollinearity:
    def test_identical_series_correlate_at_one(self):
        index = dates(30)
        values = np.random.default_rng(0).normal(size=(30, 80))
        flow = pd.DataFrame(values, index=index)
        result = fund_flow_collinearity.correlation(flow, pd.DataFrame(values, index=index))
        assert result["pearson"].mean() == pytest.approx(1.0, abs=1e-9)

    def test_the_summary_reports_the_threshold_breach(self):
        table = pd.DataFrame({"date": dates(20), "names": [80] * 20,
                              "pearson": [0.7] * 20, "rank": [0.7] * 20})
        summary = fund_flow_collinearity.summarise(table)
        assert summary["rank"]["share_above"] == pytest.approx(1.0)
        assert summary["rank"]["mean"] == pytest.approx(0.7)

    def test_a_low_correlation_marks_nothing_as_redundant(self):
        table = pd.DataFrame({"date": dates(20), "names": [80] * 20,
                              "pearson": [0.2] * 20, "rank": [0.2] * 20})
        summary = fund_flow_collinearity.summarise(table)
        assert summary["rank"]["share_above"] == pytest.approx(0.0)


class TestShadowLedger:
    def test_the_ledger_is_append_only(self, tmp_path, monkeypatch):
        log = tmp_path / "shadow.jsonl"
        monkeypatch.setattr(shadow_run, "LOG", log)
        shadow_run.append({"status": "decided"})
        shadow_run.append({"status": "stale"})
        lines = log.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert "decided" in lines[0] and "stale" in lines[1]

    def test_state_round_trips(self, tmp_path, monkeypatch):
        state = tmp_path / "state.json"
        monkeypatch.setattr(shadow_run, "STATE", state)
        shadow_run.save_state(dict(holdings=["000001"], runs=1))
        assert shadow_run.load_state()["holdings"] == ["000001"]
        assert shadow_run.load_state()["runs"] == 1

    def test_a_missing_state_file_starts_fresh(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shadow_run, "STATE", tmp_path / "absent.json")
        fresh = shadow_run.load_state()
        assert fresh["status"] == "new"
        assert fresh["holdings"] == []


class TestFundFlowArchive:
    def test_a_repeated_pull_does_not_duplicate_rows(self, tmp_path, monkeypatch):
        index = dates(5)
        pulled = pd.DataFrame({"date": index, "code": "000001", "close": 10.0,
                               "pct_change": 0.0, "main_net": 1.0, "main_pct": 0.5})
        calls = []

        def fake_fetch(code, pause=0.0):
            calls.append(code)
            return pulled.copy()

        monkeypatch.setattr(archive_fund_flow, "fetch", fake_fetch)
        first = archive_fund_flow.run(codes=["000001"], store=tmp_path)
        second = archive_fund_flow.run(codes=["000001"], store=tmp_path)
        assert first["rows_total"] == 5
        # the second pass sees the code already at the latest date and skips it entirely
        assert second["rows_total"] == 5
        assert second["skipped"] == 1
        assert len(calls) == 1

    def test_a_failing_code_is_counted_not_raised(self, tmp_path, monkeypatch):
        def broken(code, pause=0.0):
            raise RuntimeError("provider down")

        monkeypatch.setattr(archive_fund_flow, "fetch", broken)
        summary = archive_fund_flow.run(codes=["000001"], store=tmp_path)
        assert summary["failed"] == 1
        assert summary["rows_total"] == 0
