"""Pure-array same-path GAE comparison; no neural or market execution."""

from copy import deepcopy

import numpy as np
import pytest

from tools.archive.analyze_credit_counterfactual import counterfactual_arrays, target_metrics, output_path
from tools.archive.audit_timing_credit import gae_targets


def saved(actual=.95, done=False, timeout=False):
    values = np.array([[.1], [.2], [.3]], np.float32)
    raw = np.array([[.01], [.02], [.03]], np.float32)
    corrected = raw.copy()
    terminal = np.zeros_like(raw)
    if timeout:
        terminal[-1] = .4
        corrected[-1] += np.float32(.995) * terminal[-1]
    context = dict(gamma=np.asarray(.995), gae_lambda=np.asarray(actual), last_values=np.array([.5], np.float32),
        last_dones=np.array([done or timeout]), dones=np.array([[False], [False], [done or timeout]]),
        truncated=np.array([[False], [False], [timeout]]), terminal_values=terminal, raw_rewards=raw)
    starts = np.array([[1.], [0.], [0.]], np.float32)
    adv, targets = gae_targets(corrected, values, starts, context["last_values"], context["last_dones"], .995, actual)
    return dict(rewards=corrected, values=values, episode_starts=starts, advantages=adv, returns=targets), context


@pytest.mark.parametrize("actual", [.95, 1.])
@pytest.mark.parametrize("done,timeout", [(False, False), (True, False), (True, True)])
def test_actual_estimator_reproduces_with_fixed_timeout_and_tail_without_mutation(actual, done, timeout):
    buffer, context = saved(actual, done, timeout)
    original_buffer, original_context = deepcopy(buffer), deepcopy(context)
    result = counterfactual_arrays(buffer, context)
    assert result["actual_gae_lambda"] == actual and result["steps"] == 3
    assert result["timeout_bootstraps"] == timeout
    for key in buffer:
        np.testing.assert_array_equal(buffer[key], original_buffer[key])
    for key in context:
        np.testing.assert_array_equal(context[key], original_context[key])
    assert result["lambda_one_minus_095_targets"]["maximum_absolute"] > 0


def test_zero_variance_ratio_is_explicitly_undefined():
    result = target_metrics(np.zeros((3, 1)), np.zeros((3, 1)), np.ones((3, 1)))
    assert result["preupdate_old_value_mse"] == 1.
    assert result["target_variance"] == 0 and result["mse_to_target_variance"] is None
    assert result["ratio_status"] == "undefined_zero_target_variance"


def test_wrong_actual_targets_are_rejected_not_inferred_from_boundary():
    buffer, context = saved()
    buffer["advantages"][0] += 1.
    with pytest.raises(ValueError, match="Actual lambda"):
        counterfactual_arrays(buffer, context)


@pytest.mark.parametrize("suffix", ["runs/fake", "diagnostics", "../outside"])
def test_output_cannot_escape_diagnostics_or_touch_signed_leaf(tmp_path, suffix):
    with pytest.raises(ValueError):
        output_path(tmp_path, tmp_path / suffix)


def test_existing_diagnostic_is_not_overwritten(tmp_path):
    path = tmp_path / "diagnostics" / "already"
    path.mkdir(parents=True)
    with pytest.raises(ValueError):
        output_path(tmp_path, path)
    assert output_path(tmp_path) == tmp_path / "diagnostics" / "credit_counterfactual"
