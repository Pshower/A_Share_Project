"""Synthetic rendering only; no simulator or fitted strategy."""

import numpy as np
import pytest

from tools.archive import plot_counterfactual_pilot as plot
import matplotlib.pyplot as plt


def rows():
    return [dict(fold=fold, offset=0 if i % 2 == 0 else 60,
                 discounted_relative_delta=(i - 48) * .00001,
                 fee_delta_over_initial_nav=i * .000001,
                 slippage_delta_over_initial_nav=i * .000002)
            for fold in ("2023", "2024") for i in range(96)]


def test_complete_figure_is_nonblank_and_labels_units():
    figure = plot.build_figure(rows())
    try:
        figure.canvas.draw()
        pixels = np.asarray(figure.canvas.buffer_rgba())
        assert pixels.shape == (800, 1200, 4)
        assert pixels[:, :, :3].std() > 5
        assert len(figure.axes) == 4
        assert "log-bp" in figure.axes[0].get_xlabel()
        assert "not a fitted strategy" in figure._suptitle.get_text()
        assert sum(len(axis.collections) for axis in figure.axes) == 4
    finally:
        plt.close(figure)


@pytest.mark.parametrize("change", ["missing", "unknown_fold", "nonfinite_label", "nonfinite_cost"])
def test_incomplete_or_nonfinite_plot_data_is_rejected(change):
    data = rows()
    if change == "missing":
        data.pop()
    elif change == "unknown_fold":
        data[0]["fold"] = "2026"
    elif change == "nonfinite_label":
        data[0]["discounted_relative_delta"] = float("nan")
    else:
        data[0]["fee_delta_over_initial_nav"] = float("inf")
    try:
        with pytest.raises(ValueError):
            plot.build_figure(data)
    finally:
        plt.close("all")
