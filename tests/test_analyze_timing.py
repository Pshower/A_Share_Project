"""Pure-array posthoc timing arithmetic; no policy, market or training calls."""

import numpy as np
import pandas as pd
import pytest

from tools.analyze_timing import timing_metrics, stats, critic_metrics


def rows():
    p = 1. / (1. + np.exp(-2.))
    return pd.DataFrame(dict(phase=[0, 1, 0, 1], macro=[1, 0, 0, 1], p_rebalance=[p, 1-p, .4, .6],
        net_return=[.01, -.01, .02, 0.], benchmark_return=[0., 0., .01, 0.],
        fills=[2, 0, 0, 1], benchmark_fills=[2, 0, 1, 0], fees=[2., 0., 0., 1.], benchmark_fees=[2., 0., 1., 0.],
        slippage=[0., 0., 0., 0.], benchmark_slippage=[0., 0., 0., 0.], gross=[100., 0., 0., 50.], benchmark_gross=[100., 0., 50., 0.],
        cash_ratio=[.5] * 4, benchmark_cash_ratio=[.6] * 4))


def test_calendar_flip_and_probability_gap_are_not_stock_likelihoods():
    result = timing_metrics(rows())
    assert result["macro_rebalances"] == result["macro_holds"] == 2
    assert result["on_calendar_skips"] == result["off_calendar_extra_rebalances"] == 1
    assert result["greedy_calendar_flips"] == 2 and result["greedy_calendar_flip_rate"] == .5
    assert result["agent_true_fill_stock_days"] == 3
    assert result["agent_fees"] == result["benchmark_fees"] == 3.
    initial = rows().iloc[:2]
    assert timing_metrics(initial)["logit_gap_delta_from_calendar"]["maximum"] == pytest.approx(0., abs=1e-14)


@pytest.mark.parametrize("name,value", [("phase", 20), ("macro", .5), ("p_rebalance", 1.1), ("p_rebalance", np.nan)])
def test_invalid_posthoc_arrays_rejected(name, value):
    frame = rows()
    frame.loc[0, name] = value
    with pytest.raises(ValueError):
        timing_metrics(frame)


def test_empty_stats_and_nonfinite_values():
    assert stats([])["mean"] is None
    with pytest.raises(ValueError):
        stats([np.inf])


def test_critic_fit_uses_only_saved_preupdate_arrays(tmp_path):
    path = tmp_path / "buffer.npz"
    np.savez_compressed(path, values=np.array([[.1], [.2]]), returns=np.array([[.2], [.4]]), advantages=np.array([[.1], [.2]]))
    result = critic_metrics(tmp_path, [{"path": path.name}])
    assert result["samples"] == 2 and result["old_value_mse"] == pytest.approx(.025)
    assert result["old_value_explained_variance"] == pytest.approx(.75)
    assert "not post-update" in result["semantics"]
