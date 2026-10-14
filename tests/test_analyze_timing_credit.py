"""Pure-array credit diagnostic aggregation, without models or environments."""

import numpy as np
import pytest

from tools.analyze_timing_credit import scale_metrics, cell_summary


def test_target_scale_reports_population_variance_not_extra_experience():
    result = scale_metrics([1., 2., 3.])
    assert result["count"] == 3 and result["mean"] == 2.
    assert result["std"] == pytest.approx(np.std([1., 2., 3.]))


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf]])
def test_empty_or_nonfinite_saved_targets_rejected(values):
    with pytest.raises(ValueError):
        scale_metrics(values)


def trial(reward="log_net", value=.95, fold="2023", phase="screen", seed=42):
    return dict(reward_basis=reward, gae_lambda=value, fold=fold, phase=phase, seed=seed, actual_optimizer_updates=320,
        estimator_audit=dict(timeout_bootstraps=2, natural_terminals=1),
        evaluations={str(index): dict(greedy_calendar_flips=index, paired_net_roi_gap=index / 100.) for index in range(4)})


def test_cells_keep_reward_lambda_and_fold_seed_separate():
    records = [trial(), trial(fold="2024"), trial("relative_log", 1.)]
    summary = cell_summary(records)
    absolute = summary["log_l095"]
    assert absolute["runs"] == 2 and absolute["actual_optimizer_updates"] == 640
    assert absolute["timeout_bootstraps"] == 4 and absolute["natural_terminals"] == 2
    assert absolute["greedy_eval_flips"] == 12
    assert absolute["fold_account_mean_gaps"]["screen_2023_42"] == pytest.approx(.015)
    assert summary["relative_l100"]["gae_lambda"] == 1.


def test_no_fake_cells_for_unexecuted_confirmation():
    summary = cell_summary([trial()])
    assert list(summary) == ["log_l095"]
    assert all(key.startswith("screen_") for key in summary["log_l095"]["fold_account_mean_gaps"])
