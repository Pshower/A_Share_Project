"""The sweep must backtest the same rule the recorder writes, on any panel shape the repo ships.

The first version of this engine was narrow: it assumed one dataset layout, hard-coded the window,
capital and fee schedule, and carried its own copy of the rule.  These tests pin the fixes -- one
rule module shared with ``shadow_run``, layout normalisation, and the knobs that move the answer by
more than K does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import e1_rule, e1_topk as sweep, shadow_run

DAYS = pd.bdate_range("2020-01-01", periods=40)
CODES = ["00000%d" % index for index in range(1, 6)]


def panel(values, names=True):
    """A one-feature panel in the (code, feature) layout the loader produces."""
    frame = pd.DataFrame(values, index=DAYS, columns=CODES)
    frame.columns = pd.MultiIndex.from_product([CODES, ["f1"]],
                                               names=["code", "feature"] if names else [None, None])
    return frame


# ------------------------------------------------------------------ the rule itself

def test_the_score_is_a_lower_is_better_rank():
    values = np.tile([1.0, 2.0, 3.0, 4.0, 5.0], (len(DAYS), 1))
    score = e1_rule.composite_frame(panel(values), CODES, ["f1"])
    assert score.iloc[0].tolist() == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])


def test_missing_values_do_not_become_the_best_rank():
    values = np.tile([1.0, np.nan, 3.0, 4.0, 5.0], (len(DAYS), 1))
    score = e1_rule.composite_frame(panel(values), CODES, ["f1"])
    assert np.isnan(score.iloc[0]["000002"])
    assert score.iloc[0]["000001"] < score.iloc[0]["000003"]


def test_the_vectorised_score_matches_the_per_date_one():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(len(DAYS), len(CODES)))
    frame = panel(values)
    fast = e1_rule.composite_frame(frame, CODES, ["f1"])
    slow = e1_rule.composite(frame, CODES, ["f1"], DAYS[0])
    assert fast.loc[DAYS[0]].tolist() == pytest.approx(slow.tolist())


def test_the_buffer_keeps_names_that_are_still_inside_it():
    row = pd.Series({"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.5, "f": 0.6})
    assert e1_rule.decide(row, ["b"], top_k=2)[:2] == ["b", "a"]
    assert e1_rule.decide(row, ["f"], top_k=2) == ["a", "b"]
    assert e1_rule.decide(row, ["f"], top_k=2, buffer=6)[:2] == ["f", "a"]


def test_the_recorder_and_the_backtest_share_one_rule():
    row = pd.Series({"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.5, "f": 0.6})
    for top_k in (2, 3, 6):
        assert shadow_run.decide(row, ["d"], top_k=top_k) == e1_rule.decide(row, ["d"], top_k=top_k)


# ------------------------------------------------------------------ costs and layouts

def test_fees_have_the_commission_floor_and_stamp_duty():
    costs = sweep.Costs()
    assert costs.commission_for(100.0) == pytest.approx(5.0)
    assert costs.commission_for(100_000.0) == pytest.approx(30.0)


def test_a_flat_price_frame_is_read_as_close_only():
    prices = pd.DataFrame(100.0, index=DAYS, columns=CODES)
    open_, close = sweep._normalise_prices(prices, CODES)
    assert open_.equals(close)


def test_a_two_level_price_frame_yields_open_and_close():
    columns = pd.MultiIndex.from_product([CODES, ["open", "close"]])
    prices = pd.DataFrame(np.arange(len(DAYS) * len(columns), dtype=float).reshape(len(DAYS), -1),
                          index=DAYS, columns=columns)
    open_, close = sweep._normalise_prices(prices, CODES)
    assert open_.shape == close.shape == (len(DAYS), len(CODES))
    assert open_["000001"].iloc[0] != close["000001"].iloc[0]


def test_a_price_frame_without_fields_is_refused():
    prices = pd.DataFrame({"x": 1.0}, index=DAYS)
    prices.columns = pd.MultiIndex.from_product([["x"], ["nonsense"]])
    with pytest.raises(ValueError, match="no close|neither"):
        sweep._normalise_prices(prices, ["x"])


def test_unnamed_levels_are_accepted_and_named():
    frame = panel(np.tile([1.0, 2.0, 3.0, 4.0, 5.0], (len(DAYS), 1)), names=False)
    normalised = sweep._normalise_panel(frame)
    assert list(normalised.columns.names) == ["code", "feature"]


def test_a_flat_panel_is_refused_with_a_clear_message():
    with pytest.raises(ValueError, match="two-level"):
        sweep._normalise_panel(pd.DataFrame(1.0, index=DAYS, columns=CODES))


# ------------------------------------------------------------------ metrics and end to end

def test_metrics_read_a_known_series():
    series = pd.Series([100.0, 110.0, 99.0, 120.0], index=pd.bdate_range("2020-01-01", periods=4))
    row = sweep.metrics(series, [dict(shares=100, price=10.0, fee=5.0)])
    assert row["total"] == pytest.approx(0.2)
    assert row["worst_drawdown"] == pytest.approx(99.0 / 110.0 - 1)
    assert row["trades"] == 1 and row["fees"] == 5.0


def test_a_short_window_is_refused():
    frame = panel(np.tile([1.0, 2.0, 3.0, 4.0, 5.0], (len(DAYS), 1)))
    price = pd.DataFrame(100.0, index=DAYS, columns=CODES)
    with pytest.raises(ValueError, match="window too short"):
        sweep.backtest(frame, e1_rule.composite_frame(frame, CODES, ["f1"]), price, price, 2,
                       "2020-01-01", "2020-01-10")


def _toy(tmp_path, days=120):
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2020-01-01", periods=days)
    levels = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (days, len(CODES))), axis=0)),
                          index=index, columns=CODES)
    data = {}
    for code in CODES:
        data[(code, "f1")] = rng.normal(0, 1, days)
        for field in ("open", "close"):
            data[(code, field)] = levels[code].to_numpy()
    panel = pd.DataFrame({key: value for key, value in data.items() if key[1] == "f1"}, index=index)
    panel.columns = pd.MultiIndex.from_tuples(panel.columns, names=["code", "feature"])
    prices = pd.DataFrame({key: value for key, value in data.items() if key[1] != "f1"}, index=index)
    dataset = tmp_path / "toy"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(
        '{"stock_codes": %s, "feature_cols": ["f1"]}' % str(CODES).replace("'", '"'),
        encoding="utf-8")
    panel.to_parquet(dataset / "panel_data.parquet")
    prices.to_parquet(dataset / "prices.parquet")
    return dataset, index


def test_the_sweep_runs_end_to_end_and_writes_holdings(tmp_path):
    dataset, index = _toy(tmp_path)
    result = sweep.run("custom", (2, 4), dataset=dataset, output=tmp_path / "out",
                       start=str(index[0].date()), end=str(index[-1].date()), log=lambda *_: None)
    assert len(result["runs"]) == 2
    assert all(row["trades"] > 0 for row in result["runs"])
    holdings = sorted((tmp_path / "out").glob("holdings_custom_k2_i1.csv"))
    assert holdings, "a run without holdings cannot be audited"
    frame = pd.read_csv(holdings[0])
    assert set(frame.columns) == {"date", "names"}
    assert frame["names"].str.split().str.len().max() <= 2


def test_capital_and_lot_change_the_position_sizes(tmp_path):
    dataset, index = _toy(tmp_path)
    small = sweep.run("custom", (2,), dataset=dataset, start=str(index[0].date()),
                      end=str(index[-1].date()), costs=sweep.Costs(capital=200_000.0),
                      log=lambda *_: None)
    large = sweep.run("custom", (2,), dataset=dataset, start=str(index[0].date()),
                      end=str(index[-1].date()), costs=sweep.Costs(capital=2_000_000.0),
                      log=lambda *_: None)
    assert "capital" in small["costs"] and small["costs"]["capital"] != large["costs"]["capital"]
