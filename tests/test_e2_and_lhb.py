"""Guards for the E2 pool definition and the 龙虎榜 archive.

Both tools decide who is in the cross-section and what a signal means, so the failure modes
worth a test are the ones that change membership or shift a signal by a day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import archive_lhb, e2_dynamic_pool, e2_run, lhb_ic


def test_membership_ranks_on_the_previous_session():
    dates = pd.bdate_range("2024-01-01", periods=3)
    cap = pd.DataFrame({"A": [100.0, 100.0, 1.0], "B": [10.0, 10.0, 10.0]}, index=dates)
    member = e2_dynamic_pool.membership(cap, top=1)
    # Day 0 has no previous session: nobody is a member.
    assert not member.iloc[0].any()
    assert member["A"].tolist() == [False, True, True]      # ranked on the day before
    assert member["B"].tolist() == [False, False, False]


def test_membership_treats_a_missing_lagged_cap_as_absent():
    dates = pd.bdate_range("2024-01-01", periods=3)
    cap = pd.DataFrame({"A": [np.nan, 100.0, 100.0], "B": [5.0, 5.0, 5.0]}, index=dates)
    member = e2_dynamic_pool.membership(cap, top=1)
    # A has no lagged cap on day 1, so it cannot outrank B even though it is the larger
    # name on that day; from day 2 the lag is available and A wins.
    assert member["A"].tolist() == [False, False, True]
    assert member["B"].tolist() == [False, True, False]


def test_lhb_normalize_maps_the_columns_akshare_returns():
    raw = pd.DataFrame({"代码": ["600000", "1"], "上榜日": ["2024-01-02", "2024-01-03"],
                        "龙虎榜净买额": ["1.5", "2"], "龙虎榜成交额": ["10", "20"],
                        "上榜原因": ["a", "b"]})
    data = archive_lhb.normalize(raw)
    assert list(data["code"]) == ["600000", "000001"]
    assert data["date"].iloc[0] == pd.Timestamp("2024-01-02").date()
    assert data["net_buy"].iloc[0] == pytest.approx(1.5)


def test_lhb_normalize_rejects_an_unexpected_schema():
    with pytest.raises(ValueError, match="Unexpected LHB schema"):
        archive_lhb.normalize(pd.DataFrame({"代码": ["600000"]}))


def test_signal_matrix_is_lagged_one_session():
    index = pd.bdate_range("2024-01-01", periods=4)
    events = pd.DataFrame({"code": ["600000"], "date": [index[1]],
                           "net_buy": [5.0], "amount": [10.0]})
    matrix = lhb_ic.signal_matrix(events, index, ["600000"])
    # The list is published after the close, so the decision on day 1 cannot use it.
    assert np.isnan(matrix.loc[index[1], "600000"])
    assert matrix.loc[index[2], "600000"] == pytest.approx(0.5)


def test_rank_book_keeps_a_held_name_inside_the_buffer():
    dates = pd.bdate_range("2024-01-01", periods=1)
    codes = [f"{i:06d}" for i in range(1, 7)]
    columns = pd.MultiIndex.from_product([codes, ["f1"]])
    panel = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], index=dates, columns=columns)
    book = e2_run.rank_book(panel, ["f1"], dates[0], codes, ["000006"], top_k=2, buffer=6)
    # The cheapest name wins the reversal slot and the held name stays because it is still
    # inside the sixty-name buffer; the held name keeps its own position in the book.
    assert book == ["000006", "000001"]
    without = e2_run.rank_book(panel, ["f1"], dates[0], codes, [], top_k=2, buffer=6)
    assert without == ["000001", "000002"]


def test_forward_blocks_uses_the_registered_horizon():
    dates = pd.bdate_range("2024-01-01", periods=41)
    prices = pd.DataFrame({"A": np.arange(41, dtype=float) + 1.0}, index=dates)
    block, decisions = e2_run.forward_blocks(prices, dates, horizon=20)
    assert decisions[0] == dates[0] and decisions[1] == dates[20]
    assert block.loc[dates[0], "A"] == pytest.approx(21.0 / 1.0 - 1.0)
