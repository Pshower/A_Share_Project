"""Tests for the executable benchmark and the selection diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import selection_diagnostics as diag


def test_commission_has_a_five_yuan_floor():
    assert diag.buy_cost(1_000) == pytest.approx(5.0 + 1.0)
    assert diag.buy_cost(1_000_000) == pytest.approx(300.0 + 1_000.0)
    assert diag.sell_cost(1_000) == pytest.approx(5.0 + 0.5 + 1.0)


def test_affordable_shares_uses_whole_lots_and_pays_the_commission():
    # 10,000 yuan would buy ten lots at 10 yuan, but the commission pushes it over the
    # budget, so the floor lands on nine.
    assert diag.affordable_shares(10_000.0, 10.0) == 900
    assert diag.affordable_shares(10_000.0, 10.0) % 100 == 0
    assert diag.affordable_shares(1_000.0, 10.0) == 0
    assert diag.affordable_shares(100.0, 10.0) == 0
    assert diag.affordable_shares(1e6, np.nan) == 0


def test_simulate_holds_whole_lots_and_reports_the_cash_it_cannot_deploy():
    index = pd.bdate_range("2020-01-01", periods=40)
    prices = pd.DataFrame({"A": np.linspace(10, 12, 40), "B": np.linspace(1000, 1100, 40)}, index=index)
    books = {index[0]: ["A", "B"]}
    run = diag.simulate(books, prices, capital=100_000.0)
    assert run["cash_share"] > 0.4  # B costs 100,000 a lot, so most of the money parks
    assert run["equity"].iloc[-1] > 0


def test_cap_percentile_reads_zero_for_the_smallest_name_and_one_for_the_largest():
    index = pd.bdate_range("2020-01-01", periods=3)
    # The percentile needs a pool of at least ten names to mean anything.
    sizes = {f"S{n}": 10.0 * (n + 1) for n in range(12)}
    caps = pd.DataFrame({code: [size] * 3 for code, size in sizes.items()}, index=index)
    pool = list(sizes)
    small = diag.cap_percentile({index[0]: ["S0"]}, caps, pool)
    large = diag.cap_percentile({index[0]: ["S11"]}, caps, pool)
    assert small["mean"] == pytest.approx(0.0)
    assert large["mean"] == pytest.approx(1.0)


def test_random_books_draws_the_requested_size_without_replacement():
    pool = [f"S{index}" for index in range(50)]
    dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")]
    books = diag.random_books(pool, 10, dates, seed=3, draws=5)
    assert len(books) == 5
    for book in books:
        for date in dates:
            assert len(book[date]) == 10
            assert len(set(book[date])) == 10


def test_buy_and_sell_costs_are_asymmetric_on_the_sell_side():
    # Stamp duty is charged on the way out only.
    assert diag.sell_cost(1_000_000) > diag.buy_cost(1_000_000)
