"""Pure arithmetic and registry checks; no model calls or environment steps."""

import math

import numpy as np
import pytest

from tools import analyze_market_summary_projection as analysis


def test_zero_query_bound_is_exactly_uniform():
    bound = analysis.attention_bounds(0.)
    assert bound["minimum_weight_over_uniform"] == 1
    assert bound["maximum_weight_over_uniform"] == 1
    assert bound["maximum_relative_deviation"] == 0


def test_bound_contains_normalized_weights_for_bounded_encodings():
    rng = np.random.default_rng(42)
    encoded, query = rng.uniform(-1, 1, (30, 7)), rng.normal(size=7)
    query *= .02 / np.linalg.norm(query)
    scores = encoded @ query / math.sqrt(7)
    exponential = np.exp(scores - scores.max())
    ratios = 30 * exponential / exponential.sum()
    bound = analysis.attention_bounds(float(np.linalg.norm(query)))
    assert ratios.min() >= bound["minimum_weight_over_uniform"]
    assert ratios.max() <= bound["maximum_weight_over_uniform"]
    assert np.max(np.abs(ratios - 1)) <= bound["maximum_relative_deviation"]


@pytest.mark.parametrize("invalid", [-.1, float("nan"), float("inf")])
def test_invalid_query_norm_is_rejected(invalid):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        analysis.attention_bounds(invalid)


def test_large_log_bound_is_reported_without_infinite_json_values():
    bound = analysis.attention_bounds(400.)
    assert bound["absolute_log_ratio_bound"] == 800
    assert bound["maximum_weight_over_uniform"] is None
    assert bound["maximum_relative_deviation"] is None


@pytest.mark.parametrize("matching", [True, False])
def test_diagnostic_leaves_must_match_the_sealed_registry(tmp_path, monkeypatch, matching):
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    directory = tmp_path / "trial"
    inputs = {str((directory / "short" / "cash" / "result.json").relative_to(tmp_path)): "actual-sha",
              "another_trial/result.json": "unrelated-sha"}
    files = {"short/cash/result.json": "actual-sha" if matching else "altered-sha"}
    if matching:
        analysis.verify_read_inputs(directory, files, inputs)
    else:
        with pytest.raises(ValueError, match="sealed registry"):
            analysis.verify_read_inputs(directory, files, inputs)
