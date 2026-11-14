"""The top-K sweep must be arithmetic, not a second strategy: the rule and costs are frozen.

These tests pin the four things that decide the answer -- how a score is formed, how the book is
kept, how lots and fees are charged, and how the metrics are computed -- on synthetic panels.  The
real-data sweep lives in ``tools/e1_topk.py`` and its numbers are reported, never tuned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import e1_topk as sweep

DAYS = pd.to_datetime(pd.bdate_range("2020-01-01", periods=40))
CODES = ["00000%d" % index for index in range(1, 6)]


def panel(values):
    """A one-feature panel in the (code, feature) layout the loader produces."""
    frame = pd.DataFrame(values, index=DAYS, columns=CODES)
    frame.columns = pd.MultiIndex.from_product([CODES, ["f1"]])
    return frame


def test_the_score_is_a_lower_is_better_rank():
    values = np.tile([1.0, 2.0, 3.0, 4.0, 5.0], (len(DAYS), 1))
    score = sweep.scores(panel(values), CODES, ["f1"])
    assert score.iloc[0].tolist() == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])


def test_missing_values_do_not_become_the_best_rank():
    values = np.tile([1.0, np.nan, 3.0, 4.0, 5.0], (len(DAYS), 1))
    score = sweep.scores(panel(values), CODES, ["f1"])
    assert np.isnan(score.iloc[0]["000002"])
    assert score.iloc[0]["000001"] < score.iloc[0]["000003"]


def test_the_buffer_keeps_names_that_are_still_inside_it():
    row = pd.Series({"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.5, "f": 0.6})
    # top_k 2, buffer 4: "b" is rank 1, inside the buffer, so it stays ahead of a newcomer.
    book = sweep.decide(row, ["b"], top_k=2)
    assert book[:2] == ["b", "a"]


def test_a_name_outside_the_buffer_is_replaced():
    row = pd.Series({"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.5, "f": 0.6})
    book = sweep.decide(row, ["f"], top_k=2)
    assert book == ["a", "b"]


def test_fees_have_the_commission_floor_and_stamp_duty():
    assert sweep.fee_for(100.0) == pytest.approx(5.0)
    assert sweep.fee_for(100_000.0) == pytest.approx(100_000 * 0.0003)


def test_metrics_read_a_known_series():
    series = pd.Series([100.0, 110.0, 99.0, 120.0], index=pd.bdate_range("2020-01-01", periods=4))
    row = sweep.metrics(series, [dict(shares=100, price=10.0, fee=5.0)])
    assert row["total"] == pytest.approx(0.2)
    assert row["worst_drawdown"] == pytest.approx(99.0 / 110.0 - 1)
    assert row["trades"] == 1 and row["fees"] == 5.0


def test_a_short_window_is_refused():
    frame = panel(np.tile([1.0, 2.0, 3.0, 4.0, 5.0], (len(DAYS), 1)))
    price = pd.DataFrame(100.0, index=DAYS, columns=CODES)
    score = sweep.scores(frame, CODES, ["f1"])
    with pytest.raises(ValueError, match="window too short"):
        sweep.backtest(frame, score, price, price, 2, "2020-01-01", "2020-01-10")


def test_the_sweep_runs_end_to_end_on_a_synthetic_dataset(tmp_path, monkeypatch):
    days = pd.bdate_range("2020-01-01", periods=120)
    codes = CODES
    rng = np.random.default_rng(3)
    levels = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(days), len(codes))), axis=0)),
                          index=days, columns=codes)
    columns, data = [], {}
    for code in codes:
        data[(code, "f1")] = rng.normal(0, 1, len(days))
        for field in ("open", "close"):
            data[(code, field)] = levels[code].to_numpy()
        columns.append((code, "f1"))
    panel = pd.DataFrame(data, index=days)[columns]
    prices = pd.DataFrame({key: value for key, value in data.items() if key[1] in ("open", "close")},
                          index=days)
    dataset = tmp_path / "toy"
    dataset.mkdir()
    (dataset / "manifest.json").write_text('{"stock_codes": %s}' % str(codes).replace("'", '"'),
                                           encoding="utf-8")
    panel.to_parquet(dataset / "panel_data.parquet")
    prices.to_parquet(dataset / "prices.parquet")
    monkeypatch.setitem(sweep.WINDOWS, "toy", (str(days[0].date()), str(days[-1].date())))
    result = sweep.run("toy", (2, 4), dataset=dataset, log=lambda *_: None)
    assert len(result["runs"]) == 2
    assert all(row["trades"] > 0 for row in result["runs"])
    assert all(np.isfinite(row["annualised"]) for row in result["runs"])
