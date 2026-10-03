"""Arithmetic and completeness checks only; these tests never train."""

from copy import deepcopy
import math

import pytest

from src.training.matrix_report import ACCOUNTS, summarize_matrix


def _records():
    returns = {("small", "short"): [.10, .20], ("small", "long"): [.20, .24],
               ("large", "short"): [.16, .22], ("large", "long"): [.30, .34]}
    rows = []
    for (size, budget), values in returns.items():
        for index, seed in enumerate([42, 123]):
            scenarios = {}
            for account, offset in zip(ACCOUNTS, [-.03, -.01, .01, .03]):
                def metric(value):
                    return dict(cumulative_return=value + offset, sharpe=value * 10,
                                max_drawdown=-.02, mean_cash_ratio=.5)
                scenarios[account] = dict(ppo=metric(values[index]), initialized=metric(.08),
                    baselines=dict(momentum=metric(.25), momentum_daily=metric(.18), cash=metric(0.)))
            rows.append(dict(size=size, budget=budget, seed=seed, steps=1024 if budget == "short" else 2048,
                hidden_sizes=[8, 4] if size == "small" else [16, 8], parameters=123 if size == "small" else 456,
                elapsed_seconds=10 + index, training_diagnostics={"last": {"train/explained_variance": .2 + index}},
                experience={"valid_observations": 100 + index}, metrics=scenarios))
    return rows


def test_matrix_averages_accounts_before_seed_uncertainty():
    summary = summarize_matrix(_records())
    cell = summary["cells"]["small_short"]
    returns = cell["metrics"]["cumulative_return"]
    assert returns["mean"] == pytest.approx(.15)
    assert returns["std"] == pytest.approx(math.sqrt(.005))
    assert returns["range"] == pytest.approx([.10, .20])
    assert returns["count"] == 2
    assert cell["per_account"]["cash"]["metrics"]["cumulative_return"]["mean"] == pytest.approx(.12)
    assert cell["return_gaps"]["initialized"]["mean"] == pytest.approx(.07)
    assert cell["return_gaps"]["momentum"]["mean"] == pytest.approx(-.10)
    assert cell["return_gaps"]["momentum_daily"]["mean"] == pytest.approx(-.03)
    assert cell["training_diagnostics"]["last/train/explained_variance"]["mean"] == pytest.approx(.7)
    assert cell["experience"]["valid_observations"]["range"] == [100., 101.]
    assert summary["total_runs"] == 8
    assert summary["total_checkpoints"] == 8
    assert summary["total_steps"] == 12288
    assert summary["checkpoint_steps_sum"] == 12288
    assert summary["test_evaluated"] is False


def test_matrix_uses_same_seed_paired_budget_capacity_and_interaction_differences():
    summary = summarize_matrix(_records())
    contrasts = summary["contrasts"]
    assert contrasts["budget"]["small"]["mean"] == pytest.approx(.07)
    assert contrasts["budget"]["large"]["mean"] == pytest.approx(.13)
    assert contrasts["capacity"]["short"]["mean"] == pytest.approx(.04)
    assert contrasts["capacity"]["long"]["mean"] == pytest.approx(.10)
    assert contrasts["interaction"]["mean"] == pytest.approx(.06)
    assert contrasts["interaction"]["std"] == pytest.approx(math.sqrt(.0008))
    assert summary["paired_seed_effects"][0]["interaction"] == pytest.approx(.04)
    assert summary["paired_seed_effects"][1]["interaction"] == pytest.approx(.08)


@pytest.mark.parametrize("change, message", [
    ("missing_record", "same seeds"), ("missing_cell", "same seeds"),
    ("duplicate", "Duplicate"), ("different_seed", "same seeds"),
    ("missing_account", "account scenarios"), ("missing_momentum", "momentum baseline"),
    ("different_steps", "step budget"), ("different_capacity", "model capacity"),
])
def test_matrix_rejects_incomplete_or_inconsistent_records(change, message):
    records = deepcopy(_records())
    if change == "missing_record":
        records.pop()
    elif change == "missing_cell":
        records = [row for row in records if (row["size"], row["budget"]) != ("large", "long")]
    elif change == "duplicate":
        records.append(deepcopy(records[0]))
    elif change == "different_seed":
        records[-1]["seed"] = 999
    elif change == "missing_account":
        del records[0]["metrics"]["cash"]
    elif change == "missing_momentum":
        del records[0]["metrics"]["cash"]["baselines"]["momentum"]
    elif change == "different_steps":
        records[-1]["steps"] += 512
    else:
        records[-1]["hidden_sizes"] = [32, 16]
    with pytest.raises(ValueError, match=message):
        summarize_matrix(records)


def test_null_metric_is_preserved_without_inventing_a_value():
    records = _records()
    for record in records:
        if record["size"] == "small" and record["budget"] == "short" and record["seed"] == 42:
            for scenario in record["metrics"].values():
                scenario["ppo"]["sharpe"] = None
    sharpe = summarize_matrix(records)["cells"]["small_short"]["metrics"]["sharpe"]
    assert sharpe == dict(mean=2., std=None, min=2., max=2., range=[2., 2.], count=1)


def test_shared_prefix_counts_each_training_run_once_without_changing_contrasts():
    records = _records()
    independent = summarize_matrix(records)
    for record in records:
        record.update(training_run_id=f"{record['size']}_{record['seed']}", prefix_shared=True)
    shared = summarize_matrix(records)
    assert shared["total_runs"] == 4
    assert shared["total_checkpoints"] == 8
    assert shared["total_steps"] == 8192
    assert shared["checkpoint_steps_sum"] == 12288
    assert all(run["checkpoint_count"] == 2 and run["steps"] == 2048 for run in shared["training_runs"])
    assert shared["contrasts"] == independent["contrasts"]
    assert shared["cells"]["small_short"]["per_seed"][0]["training_run_id"] == "small_42"


@pytest.mark.parametrize("change, message", [
    ("missing_run_id", "require training_run_id"), ("missing_pair", "matching short and long"),
    ("mixed_seed", "mixes capacities or seeds"), ("not_shared", "requires shared prefixes"),
])
def test_shared_prefix_rejects_ambiguous_training_identity(change, message):
    records = _records()
    for record in records:
        record.update(training_run_id=f"{record['size']}_{record['seed']}", prefix_shared=True)
    if change == "missing_run_id":
        del records[0]["training_run_id"]
    elif change == "missing_pair":
        records[0]["training_run_id"] += "_other"
    elif change == "mixed_seed":
        records[1]["training_run_id"] = records[0]["training_run_id"]
    else:
        records[0]["prefix_shared"] = False
    with pytest.raises(ValueError, match=message):
        summarize_matrix(records)


@pytest.mark.parametrize("wrong_counts", [False, True])
def test_experience_counts_must_match_the_checkpoint_prefix(wrong_counts):
    records = _records()
    record = records[0]
    record["experience"] = dict(total_steps=record["steps"], mode_steps=dict(cash=record["steps"]))
    if wrong_counts:
        record["experience"]["mode_steps"]["cash"] -= 1
    else:
        record["experience"]["total_steps"] += 1
    with pytest.raises(ValueError, match="Experience steps|experience counts"):
        summarize_matrix(records)
