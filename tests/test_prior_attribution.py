"""Pure-function tests for the R1 prior attribution."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tools import prior_attribution as pa


def test_prior_rank_spans_minus_one_to_one_and_keeps_order():
    rank = pa.prior_rank([1.0, 2.0, 3.0], [True, True, True])
    assert rank.tolist() == [-1.0, 0.0, 1.0]


def test_prior_rank_shares_a_midrank_between_ties():
    rank = pa.prior_rank([1.0, 1.0, 2.0], [True, True, True])
    assert rank[0] == pytest.approx(rank[1])
    assert rank[0] == pytest.approx(-0.5)
    assert rank[2] == pytest.approx(1.0)


def test_prior_rank_ignores_ineligible_rows_and_keeps_them_at_zero():
    rank = pa.prior_rank([5.0, 1.0, 2.0, 9.0], [True, False, True, True])
    # Eligible values are 5.0, 2.0 and 9.0, so they take -1, 0 and +1 in that order.
    assert rank[2] == pytest.approx(-1.0)
    assert rank[0] == pytest.approx(0.0)
    assert rank[3] == pytest.approx(1.0)
    assert rank[1] == 0.0


def test_prior_rank_gives_a_lone_candidate_plus_one():
    rank = pa.prior_rank([4.0, 1.0], [True, False])
    assert rank[0] == pytest.approx(1.0)
    assert rank[1] == 0.0


def test_prior_rank_rejects_a_shape_mismatch():
    with pytest.raises(ValueError):
        pa.prior_rank([1.0, 2.0], [True])


def test_displacement_bound_is_one_rank_step_per_residual_quantum():
    # One rank step costs 2 * strength / (count - 1) because ranks span [-1, 1].
    step = 2 * 4.0 / (300 - 1)
    assert pa.displacement_bound(300, 4.0, 0.25) == pytest.approx(0.5 / step)
    # The share of the rank axis is invariant to pool size.
    assert pa.displacement_bound(300, 4.0, 0.25) / (300 - 1) == pytest.approx(0.25 / 4.0)
    assert pa.displacement_bound(19, 4.0, 0.25) / (19 - 1) == pytest.approx(0.25 / 4.0)
    assert np.isnan(pa.displacement_bound(1, 4.0, 0.25))


def test_spearman_recovers_both_signs_and_rejects_a_constant_side():
    x = np.arange(6.0)
    assert pa.spearman(x, x) == pytest.approx(1.0)
    assert pa.spearman(x, -x) == pytest.approx(-1.0)
    assert np.isnan(pa.spearman(x, np.ones(6)))


def test_spearman_ignores_monotone_rescaling():
    x = np.arange(5.0)
    assert pa.spearman(x, np.exp(x)) == pytest.approx(1.0)


def test_fixed_observation_holds_an_equal_weight_book_and_no_cash():
    features = np.zeros((4, 1, 2), dtype=np.float32)
    observation = pa.fixed_observation(features, np.array([True, True, False, True]),
                                       np.array([True, True, False, True]),
                                       np.array([True, True, False, True]))
    weights = observation["portfolio"]
    assert weights.shape == (5,)
    assert weights[:4].tolist() == pytest.approx([1 / 3, 1 / 3, 0.0, 1 / 3])
    assert weights[4] == 0.0
    assert observation["valid_mask"].dtype == np.int8
    assert observation["receivable_ratio"].tolist() == [0.0]


def test_fixed_observation_without_valid_names_leaves_the_book_empty():
    features = np.zeros((2, 1, 1), dtype=np.float32)
    observation = pa.fixed_observation(features, np.array([False, False]),
                                       np.array([False, False]), np.array([False, False]))
    assert observation["portfolio"].tolist() == [0.0, 0.0, 0.0]


def test_probe_dates_samples_by_stride_within_the_calendar():
    dates = pd.bdate_range("2024-01-01", periods=20)
    market = SimpleNamespace(dates=dates)
    assert pa.probe_dates(market, end="2024-12-31", stride=2).tolist() == list(range(0, 20, 2))
    assert pa.probe_dates(market, end="2024-01-05", stride=3).tolist() == [0, 3]


def test_probe_dates_never_returns_a_row_past_the_end():
    dates = pd.bdate_range("2024-01-01", periods=20)
    market = SimpleNamespace(dates=dates)
    limit = dates[9]
    picked = pa.probe_dates(market, end=str(limit.date()), stride=2)
    assert picked.tolist() == [0, 2, 4, 6, 8]
    assert all(dates[i] <= limit for i in picked)


def test_probe_dates_rejects_a_window_without_any_date():
    dates = pd.bdate_range("2024-01-01", periods=5)
    with pytest.raises(ValueError):
        pa.probe_dates(SimpleNamespace(dates=dates), end="2023-01-01", stride=1)


def test_summarize_reports_the_design_bound_next_to_the_measured_residual():
    rows = [dict(date="2024-01-02", eligible=257, spearman_signal=1.0, spearman_prior=1.0,
                 residual_signal=0.25, residual_max=0.002, top_overlap=20)]
    record = pa.summarize("trained", rows, 4.0, 0.25, 257)
    assert record["residual_share_of_prior"] == pytest.approx(0.0005)
    assert record["displacement_bound"] == pytest.approx(16.0)
    assert record["displacement_share"] == pytest.approx(0.0625)
    assert record["top_overlap"] == 20.0
