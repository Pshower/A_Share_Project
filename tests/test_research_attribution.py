"""Factorial configuration, paired statistics and strict complete-budget checks."""

from copy import deepcopy

import pytest

from src.training.research_attribution import (BUDGETS, CELLS, FOLDS, SEEDS, config_for,
                                             summarize, validate_records)


@pytest.mark.parametrize("fold", FOLDS)
@pytest.mark.parametrize("cell", CELLS)
def test_factorial_config_changes_only_declared_flags_and_fold(tmp_path, fold, cell):
    base = dict(ppo={})
    old = deepcopy(base)
    config = config_for(base, tmp_path / "reference", tmp_path / "output", fold, cell, 42)
    assert base == old
    assert config["research_config"].endswith(f"{fold}\\research.json") or config["research_config"].endswith(f"{fold}/research.json")
    assert config["attribution"]["market_context"] == CELLS[cell]["market_context"]
    assert config["attribution"]["critic_features"] == CELLS[cell]["critic_features"]
    assert config["exploration"] == dict(architecture="split", sampling="recent", common_lookback=5,
        reward_basis="simple_net", size_annealing=False)
    assert config["lookback"] == 1 and config["actor_output_gain"] == .1
    assert config["initial_log_std"] == -.7 and config["ppo"]["gamma"] == .99
    assert config["prior_feature_index"] is None


def fake_records():
    result = []
    for fold in FOLDS:
        for cell in CELLS:
            for seed in SEEDS:
                for budget, steps in BUDGETS.items():
                    route = CELLS[cell]["market_context"]
                    detached = CELLS[cell]["critic_features"] == "detached"
                    initial = .01 if route == "pure" else 0.
                    gain = .02 + (route == "pure") * .03 + detached * .04 + (route == "pure" and detached) * .02
                    roi = initial + gain + (seed == 123) * .01 + (fold == "2024") * .02
                    metric = dict(cumulative_return=roi, max_drawdown=-.03, total_fees=100., total_slippage_cost=50.)
                    result.append(dict(fold=fold, cell=cell, seed=seed, budget=budget, steps=steps,
                        metrics={mode: dict(ppo=deepcopy(metric), initialized=dict(metric, cumulative_return=initial))
                                 for mode in ("cash", "equal", "concentrated", "random")}))
    return result


def fake_rules():
    return {fold: {mode: dict(momentum_20day=dict(cumulative_return=.005))
                   for mode in ("cash", "equal", "concentrated", "random")} for fold in FOLDS}


def test_actual_steps_do_not_sum_prefixes_and_fold_is_not_extra_seed():
    report = summarize(fake_records(), fake_rules())
    assert report["actual_training_steps"] == 98304
    assert report["checkpoint_steps_sum"] == 147456
    assert report["trained_runs"] == 24 and report["snapshots"] == 48
    assert report["cells"]["pure_shared_long"]["metrics"]["cumulative_return"]["count"] == 3
    assert report["qualified_candidate"] == dict(cell="pure_detached", budget="short",
        representative_seed=42, representative_fold="2024")
    assert not report["actionable"] and not report["test_evaluated"]


def test_factorial_net_and_learning_effects_are_separated():
    result = summarize(fake_records(), fake_rules())["factorial_effects"]["long"]
    assert result["net_roi"]["fused_minus_pure"]["mean"] == pytest.approx(-.05)
    assert result["learning_gain"]["fused_minus_pure"]["mean"] == pytest.approx(-.04)
    assert result["net_roi"]["detached_minus_shared"]["mean"] == pytest.approx(.05)
    assert result["net_roi"]["interaction"]["mean"] == pytest.approx(-.02)
    assert result["initialization"]["fused_minus_pure"]["mean"] == pytest.approx(-.01)
    assert result["initialization"]["detached_minus_shared"]["mean"] == pytest.approx(0.)
    assert result["arithmetic_decomposition_max_error"] < 1e-12
    assert result["net_roi"]["detached_minus_shared_pure"]["mean"] == pytest.approx(.06)
    assert result["net_roi"]["detached_minus_shared_fused"]["mean"] == pytest.approx(.04)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "wrong_steps", "wrong_fold", "wrong_seed"])
def test_incomplete_or_changed_factorial_never_claims_complete(failure):
    records = fake_records()
    if failure == "missing":
        records.pop()
    elif failure == "duplicate":
        records[-1] = deepcopy(records[-2])
    elif failure == "wrong_steps":
        records[-1]["steps"] = 8192
    elif failure == "wrong_fold":
        records[-1]["fold"] = "2025"
    else:
        records[-1]["seed"] = 7
    with pytest.raises(ValueError):
        validate_records(records)


def test_both_folds_must_pass_stronger_rule_and_all_account_risk_gate():
    records = fake_records()
    for record in records:
        if record["fold"] == "2023":
            record["metrics"]["cash"]["ppo"]["max_drawdown"] = -.101
    report = summarize(records, fake_rules())
    assert report["qualified_candidate"] is None
    assert all(not value["qualified"] for value in report["cells"].values())


def test_seed_pairing_is_not_affected_by_input_order():
    records = fake_records()
    forward = summarize(records, fake_rules())
    reverse = summarize(list(reversed(records)), fake_rules())
    assert forward == reverse
