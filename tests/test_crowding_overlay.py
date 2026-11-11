"""The overlay is only honest if the blanking and the pairing are exactly as registered.

The failure mode these tests guard is a silent one: blanking the wrong cells would still
produce a plausible-looking paired difference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import crowding_overlay


def test_crowded_mask_selects_the_top_decile_of_growth():
    dates = pd.bdate_range("2024-01-01", periods=3)
    events = pd.DataFrame({"code": ["000001", "000002"],
                           "date": [dates[0], dates[0]],
                           "balance": [100.0, 10.0]})
    mask = crowding_overlay.crowded_mask(events, dates, ["000001", "000002"], threshold=0.5)
    # growth needs a lagged balance, so day 0 has nothing to rank
    assert not mask.loc[dates[0]].any()


def test_blank_crowded_only_clears_the_flagged_stock_day():
    dates = pd.bdate_range("2024-01-01", periods=3)
    codes = ["000001", "000002"]
    columns = pd.MultiIndex.from_tuples([(code, "f1") for code in codes])
    panel = pd.DataFrame(1.0, index=dates, columns=columns)
    mask = pd.DataFrame(False, index=dates, columns=codes)
    mask.loc[dates[1], "000002"] = True
    out = crowding_overlay.blank_crowded(panel, mask)
    assert np.isnan(out.loc[dates[1], ("000002", "f1")])
    assert out.loc[dates[1], ("000001", "f1")] == pytest.approx(1.0)
    assert out.loc[dates[0], ("000002", "f1")] == pytest.approx(1.0)
    # the original object is untouched
    assert panel.loc[dates[1], ("000002", "f1")] == pytest.approx(1.0)


def test_block_pairs_are_non_overlapping_and_paired():
    index = pd.bdate_range("2024-01-01", periods=6)
    left = pd.Series(0.01, index=index)
    right = pd.Series(0.00, index=index)
    frame = crowding_overlay.block_pairs(left, right, horizon=2)
    assert len(frame) == 3
    assert frame["diff"].iloc[0] == pytest.approx((1.01 ** 2 - 1) - 0.0)


def test_paired_reports_a_t_and_refuses_a_thin_sample():
    thin = pd.DataFrame({"diff": [0.01, 0.02]})
    assert crowding_overlay.paired(thin)["status"] == "insufficient"
    wide = pd.DataFrame({"diff": np.full(20, 0.01), "base": 0.0, "overlay": 0.01})
    stats = crowding_overlay.paired(wide)
    assert stats["status"] == "ready" and stats["t"] > 5


def test_pass_requires_positive_mean_and_the_higher_bar():
    # the same shape the registered rule uses
    frame = pd.DataFrame({"diff": np.r_[np.full(30, 0.001), np.full(67, -0.0005)],
                          "base": 0.0, "overlay": 0.0})
    stats = crowding_overlay.paired(frame)
    stats["passed"] = bool(stats["mean_block"] > 0 and stats["t"] >= 2.5)
    assert stats["passed"] is False
