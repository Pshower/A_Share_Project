"""The catalogue tool must not lie about what answered.

Two mistakes were made while building it, and both are pinned here: parsing every column as a date
(so a code or a price read as 1970) and treating a refused connection the same as a broken
interface.  A survey whose rows cannot be trusted is worse than no survey.
"""
from __future__ import annotations

import pandas as pd
import pytest

from tools import data_catalog


def frame(columns):
    return pd.DataFrame(columns)


def test_a_chinese_month_column_is_read_as_a_date():
    series = pd.Series(["2026年10月份", "2026年09月份", "2026年08月份"])
    parsed = data_catalog.to_dates(series)
    assert str(parsed.max().date()) == "2026-10-01"


def test_a_plain_iso_column_is_read_as_a_date():
    parsed = data_catalog.to_dates(pd.Series(["2026-11-13", "2026-11-12"]))
    assert str(parsed.max().date()) == "2026-11-13"


def test_codes_and_prices_are_not_dates():
    frame = pd.DataFrame({"代码": ["600519", "000001"], "最新价": [1700.0, 11.2]})
    rows, columns, last, names = data_catalog.shape(frame)
    assert rows == 2 and columns == 2
    assert last is None                      # neither column looks like a date
    assert "代码" in names


def test_only_date_like_columns_are_parsed():
    frame = pd.DataFrame({"日期": ["2026-11-13", "2026-11-12"], "成交额": [1e9, 2e9]})
    _, _, last, _ = data_catalog.shape(frame)
    assert last == "2026-11-13"


def test_a_blocked_connection_is_classified_apart_from_a_broken_interface():
    assert data_catalog.classify(ConnectionError("Connection aborted"))[0] == "blocked"
    assert data_catalog.classify(AttributeError("no attribute"))[0] == "error"
    assert data_catalog.classify(ValueError("bad shape"))[0] == "error"


def test_the_summary_counts_every_status():
    rows = [
        dict(category="宏观", status="ok"),
        dict(category="宏观", status="blocked"),
        dict(category="技术", status="empty"),
        dict(category="技术", status="timeout"),
        dict(category="技术", status="error"),
    ]
    summary = data_catalog.summarise(rows)
    assert summary["宏观"] == dict(total=2, ok=1, empty=0, blocked=1, failed=0)
    assert summary["技术"] == dict(total=3, ok=0, empty=1, blocked=0, failed=2)


def test_every_catalogue_entry_declares_one_of_the_three_categories():
    categories = {entry[1] for entry in data_catalog.CATALOG}
    assert categories == {"宏观", "技术", "基本面"}


def test_the_catalogue_has_no_duplicate_ids():
    ids = [entry[0] for entry in data_catalog.CATALOG]
    assert len(ids) == len(set(ids))
