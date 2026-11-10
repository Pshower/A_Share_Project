"""Guards for the two newest source archives: margin balances and shareholder counts.

Both are about *when* a number becomes usable.  The margin table is a daily snapshot, so a
day that predates the first capture must not silently look like zero leverage; the
shareholder count is announced weeks after it is measured, so a value must not appear
before its announcement and must not live forever after it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import archive_margin, gdhs_ic


def test_margin_normalize_reads_the_sse_layout():
    raw = pd.DataFrame({"标的证券代码": ["600000", "1"], "信用交易日期": ["20240102", "20240102"],
                        "融资余额": ["100", "200"], "融资买入额": ["10", "20"],
                        "融券余量": ["5", "6"]})
    data = archive_margin.normalize(raw, pd.Timestamp("2024-01-02").date())
    assert list(data["code"]) == ["600000", "000001"]
    assert data["balance"].iloc[0] == pytest.approx(100.0)
    assert data["short_balance"].iloc[1] == pytest.approx(6.0)
    assert set(data["date"]) == {pd.Timestamp("2024-01-02").date()}


def test_margin_normalize_reads_the_szse_layout_and_fills_the_date():
    raw = pd.DataFrame({"证券代码": ["000001"], "融资余额": ["5129"], "融资买入额": ["187"],
                        "融券余量": ["10281"]})
    data = archive_margin.normalize(raw, pd.Timestamp("2024-01-02").date())
    assert data.loc[0, "code"] == "000001"
    assert data.loc[0, "balance"] == pytest.approx(5129.0)
    assert "repay" not in data or np.isnan(data.loc[0, "repay"])


def test_margin_normalize_rejects_a_table_without_codes():
    with pytest.raises(ValueError, match="No stock-code column"):
        archive_margin.normalize(pd.DataFrame({"融资余额": ["1"]}), pd.Timestamp("2024-01-02").date())


def test_asof_matrix_uses_the_announcement_date_not_the_cutoff():
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-31", "2024-06-30"])
    events = pd.DataFrame({"code": ["600000"], "announced": [pd.Timestamp("2024-01-15")],
                           "change_pct": [-5.0], "cutoff": ["2023-12-31"]})
    matrix = gdhs_ic.asof_matrix(events, index, ["600000"], max_age=1000)
    # Measured at the quarter end, published in the middle of January: unusable on 01-02.
    assert np.isnan(matrix.loc["2024-01-02", "600000"])
    assert matrix.loc["2024-01-31", "600000"] == pytest.approx(-5.0)
    assert matrix.loc["2024-06-30", "600000"] == pytest.approx(-5.0)


def test_asof_matrix_expires_a_stale_value():
    index = pd.DatetimeIndex(["2024-01-31", "2024-12-31"])
    events = pd.DataFrame({"code": ["600000"], "announced": [pd.Timestamp("2024-01-15")],
                           "change_pct": [-2.0], "cutoff": ["2023-12-31"]})
    matrix = gdhs_ic.asof_matrix(events, index, ["600000"], max_age=100)
    assert matrix.loc["2024-01-31", "600000"] == pytest.approx(-2.0)
    assert np.isnan(matrix.loc["2024-12-31", "600000"])


def test_asof_matrix_keeps_the_latest_announcement():
    index = pd.DatetimeIndex(["2024-09-30"])
    events = pd.DataFrame({"code": ["600000", "600000"],
                           "announced": [pd.Timestamp("2024-04-20"), pd.Timestamp("2024-08-20")],
                           "change_pct": [-9.0, 3.0], "cutoff": ["2024-03-31", "2024-06-30"]})
    matrix = gdhs_ic.asof_matrix(events, index, ["600000"], max_age=1000)
    assert matrix.loc["2024-09-30", "600000"] == pytest.approx(3.0)
