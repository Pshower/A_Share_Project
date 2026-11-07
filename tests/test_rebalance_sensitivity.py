"""Unit checks for the rebalance-date sensitivity runner."""

import pandas as pd
import pytest

from tools import rebalance_sensitivity as sensitivity


class FakeMarket:
    """The slice of MarketDataProvider the shim touches."""

    def __init__(self, dates=10):
        self.n_dates = dates
        self.dates = pd.bdate_range("2024-01-01", periods=dates)
        self.current_index = 0
        self.stock_codes = ["000001", "600000"]

    def reset(self):
        self.current_index = 0


def test_shifted_market_moves_the_opening_date_and_delegates_the_rest():
    market = FakeMarket()
    view = sensitivity.ShiftedMarket(market, 3)
    assert view.stock_codes == ["000001", "600000"]
    assert view.n_dates == market.n_dates
    assert str(view.window[0].date()) == str(market.dates[3].date())
    assert len(view.window) == market.n_dates - 3
    # The provider resets to index 0; the shim must put it back on the shifted day.
    view.reset()
    assert market.current_index == 3


def test_shifted_market_refuses_a_shift_that_leaves_nothing_to_trade():
    for offset in (-1, 9, 10):
        with pytest.raises(ValueError, match="no room to trade"):
            sensitivity.ShiftedMarket(FakeMarket(), offset)


def test_markdown_lists_every_shift_with_its_excess():
    payload = dict(fold="2023", version="v1", draws=100, rows=[
        dict(shift=0, start="2023-03-31", steps=59, momentum=0.065411,
             random_median=0.009842, excess=0.055568, percentile=0.94),
        dict(shift=1, start="2023-04-03", steps=58, momentum=-0.015441,
             random_median=0.001010, excess=-0.016451, percentile=0.33),
    ])
    text = sensitivity._markdown(payload)
    assert payload["version"] in text
    assert "2023-03-31" in text and "2023-04-03" in text
    assert "+5.5568%" in text and "-1.6451%" in text
    assert text.count("\n| ") == 2 + 2  # header, separator and two data rows
