"""Tests for the price-feature re-measurement: descriptive statistics, no direction claim."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import price_feature_ic as tool


def test_windows_never_leave_the_clean_dataset():
    """The dataset ends 2024-12-31, so every declared window must end by then."""
    for low, high in tool.WINDOWS.values():
        assert pd.Timestamp(high) <= pd.Timestamp("2024-12-31")
        assert pd.Timestamp(low) <= pd.Timestamp(high)


def test_feature_matrices_align_to_the_panel_grid():
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    columns = pd.MultiIndex.from_product([["000001", "600000"], ["momentum_20", "rsi_14"]])
    panel = pd.DataFrame(np.arange(12, dtype=float).reshape(3, 4), index=dates, columns=columns)
    matrices = tool.feature_matrices(panel, ["000001", "600000"], ["momentum_20"])
    assert matrices["momentum_20"].shape == (3, 2)
    np.testing.assert_allclose(matrices["momentum_20"][:, 0], panel[("000001", "momentum_20")])


def test_assessment_is_two_sided_and_ignores_the_sign():
    threshold = 0.0048
    for value, t in [(-0.03, -2.5), (0.03, 2.5)]:
        mark = tool.assess(dict(mean_ic=value, t_nw_h=t), threshold)
        assert mark["magnitude"] and mark["significant"]
    small = tool.assess(dict(mean_ic=0.001, t_nw_h=2.5), threshold)
    assert not small["magnitude"]
    weak = tool.assess(dict(mean_ic=-0.03, t_nw_h=-1.9), threshold)
    assert weak["magnitude"] and not weak["significant"]


def test_assessment_handles_a_missing_statistic():
    mark = tool.assess(dict(mean_ic=None, t_nw_h=None), 0.0048)
    assert mark == dict(magnitude=False, significant=False)


def test_the_report_states_that_no_direction_was_registered():
    payload = dict(window="V", start="2019-01-01", end="2024-12-31",
                   dataset="data/clean/research_v3", version="research_v3", codes=2,
                   features=["momentum_20"], rows=3,
                   label_definition="close[t+h] / open[t+1] - 1, complete positive path",
                   warning="w", primary_alpha=0.0141,
                   horizons={"h20": dict(rows=10, first="2019-01-02", last="2024-12-01",
                                         required_ic=0.0048, required_alpha=0.0141,
                                         magnitude_count=1, significant_count=0,
                                         lag5_significant_count=4,
                                         ranking=["momentum_20"],
                                         signals={"momentum_20": dict(
                                             mean_ic=-0.02, t_nw_h=-1.5, effective_dates=120.0,
                                             assessment=dict(magnitude=True, significant=False),
                                             lag5=dict(t=-2.4))})})
    text = tool._markdown(payload)
    # The lag-5 counterfactual must be visible, not buried.
    assert "若仍用 lag-5" in text and "4" in text
    assert "0/1" in text
