"""Pure arithmetic for margin-aware diagnostics; no model or environment calls."""

import numpy as np
import pytest

from tools.analyze_timing_margin import margin_metrics
from tests.test_analyze_timing import rows


@pytest.mark.parametrize("margin", [.5, 1., 2.])
def test_each_initial_margin_has_zero_delta_and_zero_kl(margin):
    frame = rows().iloc[:2].copy()
    favored = 1. / (1. + np.exp(-margin))
    frame["p_rebalance"] = [favored, 1 - favored]
    result = margin_metrics(frame, margin)
    assert result["logit_gap_delta_from_calendar"]["mean"] == pytest.approx(0., abs=1e-14)
    assert result["initial_to_current_kl"]["mean"] == pytest.approx(0., abs=1e-14)
    assert result["current_to_initial_kl"]["mean"] == pytest.approx(0., abs=1e-14)
    assert result["greedy_calendar_flips"] == 0
    assert result["initial_preferred_probability"] == pytest.approx(favored)


def test_half_margin_is_not_mistaken_for_two():
    frame = rows().iloc[:2].copy()
    frame["p_rebalance"] = [.5, .5]
    half, two = margin_metrics(frame, .5), margin_metrics(frame, 2.)
    assert half["initial_to_current_kl"]["mean"] == pytest.approx(.0302998619807659)
    assert two["initial_to_current_kl"]["mean"] == pytest.approx(.3278133254727379)
    assert half["categorical_entropy"]["mean"] == pytest.approx(np.log(2.))


@pytest.mark.parametrize("margin", [True, 0., .25, np.nan])
def test_unregistered_margin_rejected(margin):
    with pytest.raises(ValueError):
        margin_metrics(rows(), margin)
