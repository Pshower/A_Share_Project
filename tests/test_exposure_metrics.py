"""Unit checks for the stage-1 exposure measurements."""

import numpy as np
import pandas as pd

from tools import exposure_metrics as metrics


def _bars(closes, opens=None):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=len(closes)),
                         "open": np.asarray(opens, dtype=float) if opens is not None else closes,
                         "close": closes})


def test_forward_returns_enter_at_the_next_open_and_exit_at_the_close():
    bars = _bars([100, 110, 120, 130], opens=[100, 100, 120, 120])
    forward = metrics.forward_returns(bars, 2)
    # t=0: close[2]/open[1] - 1 = 120/100 - 1
    assert abs(forward.iloc[0] - 0.2) < 1e-12
    assert abs(forward.iloc[1] - (130 / 120 - 1)) < 1e-12
    assert forward.iloc[-2:].isna().all()


def test_realized_vol_of_constant_returns_is_zero():
    bars = _bars(np.geomspace(100, 120, 60))
    vol = metrics.realized_vol(bars["close"].pct_change().add(1).fillna(1).cumprod() ** 0)
    assert vol.dropna().abs().max() < 1e-9


def test_runs_alternate_strictly_and_skip_unknowns():
    state = pd.Series([True, True, False, False, False, True, np.nan, False],
                      index=range(8), dtype=object)
    runs = metrics._runs(state)
    assert [(flag, start, end) for flag, start, end in runs] == [
        (True, 0, 1), (False, 2, 4), (True, 5, 5), (False, 7, 7)]


def test_segments_report_length_return_and_drawdown():
    bars = _bars([100, 110, 90, 120, 130])
    state = pd.Series([True, True, True, False, False], index=bars.index)
    rows = metrics.segments(bars, state)
    assert [row["above"] for row in rows] == [True, False]
    assert rows[0]["days"] == 3 and abs(rows[0]["return_pct"] - (-0.10)) < 1e-12
    assert abs(rows[0]["max_drawdown_pct"] - (-0.18181818181818177)) < 1e-12
    assert rows[1]["days"] == 2 and abs(rows[1]["return_pct"] - (130 / 120 - 1)) < 1e-12


def test_whipsaw_counts_only_fresh_crosses_after_an_above_run():
    bars = _bars([100, 98, 96, 94, 92, 90, 88])
    # above for two rows, then below: exactly one fresh cross at index 2
    state = pd.Series([True, True, False, False, False, False, False], index=bars.index)
    row = metrics.whipsaw(bars, state, 2)
    assert row["crosses"] == 1


def test_verdict_is_mechanical_on_the_registered_bars():
    payload = dict(
        volatility=dict(acf={"20": 0.7}, ratio={"median": 0.98}),
        conditional=[dict(horizon=20, gap_in_sigma=-0.35), dict(horizon=60, gap_in_sigma=-0.5)],
        subperiods=[dict(gap=-0.01), dict(gap=0.02), dict(gap=-0.03)],
        whipsaw=[dict(horizon=20, crosses=10, net=-0.5)],
    )
    decided = metrics.verdict(payload)
    assert decided["vol_ok"] is True
    assert decided["signs_stable"] is False          # one subperiod flips
    assert decided["trend_ok"] is False              # unstable and below 0.5 sigma
    assert decided["register"] == "L1 only"


def test_verdict_reports_stop_when_no_mechanism_clears_its_bar():
    payload = dict(
        volatility=dict(acf={"20": 0.1}, ratio={"median": 1.0}),
        conditional=[dict(horizon=20, gap_in_sigma=-0.1), dict(horizon=60, gap_in_sigma=-0.1)],
        subperiods=[dict(gap=-0.01)],
        whipsaw=[dict(horizon=20, crosses=5, net=0.2)],
    )
    assert metrics.verdict(payload)["register"] == "stop"
