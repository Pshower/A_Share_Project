"""Synthetic saved-book arithmetic only; no model, engine or real label run."""

from copy import deepcopy
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.archive import analyze_counterfactual_pilot as tool


BROKER = dict(commission=.0003, min_commission=0., sell_tax=.001)
CODES = ["A", "B"]
MISSING = {"nonfinite_python": ">f8", "bits": "7ff8000000000000"}


def example(*, phase=1, fold="2023", quantity=1000, flat=False, rejected=False):
    dates = pd.bdate_range(fold + "-01-02", periods=21)
    nav0 = 1e6

    def native(trade):
        cash, shares, available, nav, previous, cumulative = nav0, 0., 0., nav0, dates[0], 17.
        records = []
        for j in range(20):
            date = dates[j + 1]
            marks = 10. if flat else 10. + .01 * j
            closing = 10. if flat else 10. + .01 * (j + 1)
            decision = dict(date=previous.isoformat(), codes=CODES, positions=[shares, 0.], available_positions=[available, 0.],
                cash=cash, receivables=0., equity=nav, marks=[marks, deepcopy(MISSING)])
            opening = dict(date=str(date.date()), codes=CODES, positions=[shares, 0.], available_positions=[shares, 0.],
                cash=cash, receivables=0., equity=cash + shares * 10., prices=[10., deepcopy(MISSING)], valuation_prices=[10., deepcopy(MISSING)])
            orders, available = [], shares
            if trade and j == 0:
                if rejected:
                    orders.append(dict(code="A", side="buy", requested_shares=1000., filled_shares=0., price=deepcopy(MISSING),
                        fee=0., slippage_cost=0., status="rejected", reasons=["missing_open"]))
                elif quantity:
                    price = 10.01
                    fee = quantity * price * BROKER["commission"]
                    orders.append(dict(code="A", side="buy", requested_shares=float(quantity), filled_shares=float(quantity), price=price,
                        fee=fee, slippage_cost=quantity * (price - 10.), status="filled", reasons=[]))
                    cash -= quantity * price + fee
                    shares += quantity
                    cumulative += fee
            new_nav = cash + shares * closing
            records.append(dict(date=date.isoformat(), equity=new_nav, cash=cash, positions=[shares, 0.], available_positions=[available, 0.],
                receivables=0., event_cash=0., event_shares=[0., 0.], total_fees=cumulative, orders=orders,
                decision_snapshot=decision, opening_snapshot=opening, reward=new_nav / nav - 1.))
            previous, nav = date, new_nav
        return records

    held, traded = native(False), native(True)
    baseline = deepcopy(traded if phase == 0 else held)
    branches = {}
    for first, name, ledgers in ((0, "hold", held), (1, "rebalance", traded)):
        steps, previous_nav, previous_shadow = [], nav0, nav0
        for j, ledger in enumerate(ledgers):
            relative = math.log(ledger["equity"]) - math.log(previous_nav) - math.log(baseline[j]["equity"]) + math.log(previous_shadow)
            ledger.update(raw_net_return=ledger["reward"], learning_reward=relative, reward=relative)
            steps.append(dict(index=j, phase=(phase + j) % 20, macro_action=first if j == 0 else int((phase + j) % 20 == 0),
                terminated=False, truncated=False, relative_log_reward=relative,
                agent_ledger=ledger, benchmark_ledger=deepcopy(baseline[j])))
            previous_nav, previous_shadow = ledger["equity"], baseline[j]["equity"]
        branches[name] = dict(initial_agent_nav=nav0, initial_benchmark_nav=nav0, final_agent_nav=ledgers[-1]["equity"],
            final_benchmark_nav=baseline[-1]["equity"], actual_agent_steps=20, actual_benchmark_steps=20,
            endpoint=str(dates[-1].date()), steps=steps)
    difference = np.asarray([r["relative_log_reward"] - h["relative_log_reward"]
        for h, r in zip(branches["hold"]["steps"], branches["rebalance"]["steps"])])
    label = dict(labels=dict(horizon=20, gamma=.995, branches=branches,
        discounted_relative_delta=float(np.dot(.995 ** np.arange(20), difference)),
        undiscounted_relative_delta=float(difference.sum()), terminal_log_nav_ratio=math.log(traded[-1]["equity"]) - math.log(held[-1]["equity"]),
        per_step_relative_reward_difference=difference.tolist()))
    record = dict(fold=fold, state_id="000_60", episode=0, offset=60, decision_date=str(dates[0].date()),
        endpoint=str(dates[-1].date()), phase=phase, initial_mode="cash", date_stratum="early", phase_bin=phase // 5,
        input_sha256="input", full_state_sha256="state")
    return label, record


def audit(label, record):
    return tool.audit_label(label, record, CODES, BROKER)


def test_native_rewards_costs_and_historical_cumulative_fees_are_reconstructed():
    label, record = example()
    row = audit(label, record)
    assert row["discounted_relative_delta"] == pytest.approx(label["labels"]["discounted_relative_delta"], abs=1e-12)
    assert row["rebalance_fees"] == pytest.approx(3.003)
    assert row["rebalance_slippage"] == pytest.approx(10.)
    assert row["rebalance_gross_notional"] == pytest.approx(10010.)
    assert row["hold_fees"] == 0.
    assert label["labels"]["branches"]["rebalance"]["steps"][-1]["agent_ledger"]["total_fees"] == pytest.approx(20.003)
    assert row["first_rebalance_no_fill"] is False
    assert row["both_whole_horizons_no_fill"] is False


@pytest.mark.parametrize("phase,flat", [(0, True), (1, False)])
def test_oracle_is_relative_to_calendar_not_always_positive_delta(phase, flat):
    label, record = example(phase=phase, flat=flat)
    row = audit(label, record)
    assert row["calendar_deviation_delta"] > 0
    assert row["oracle_calendar_advantage"] == row["calendar_deviation_delta"]
    assert row["discounted_relative_delta"] < 0 if phase == 0 else row["discounted_relative_delta"] > 0


def test_missing_open_rejections_are_masked_before_price_arithmetic():
    label, record = example(quantity=0, rejected=True)
    row = audit(label, record)
    assert row["discounted_relative_delta"] == 0.
    assert row["first_rebalance_no_fill"] is True
    assert row["both_whole_horizons_no_fill"] is True
    assert row["economic_paths_equivalent"] is True
    assert row["rebalance_rejected_orders"] == 1


@pytest.mark.parametrize("mutation,match", [
    (lambda x: x["labels"].update(discounted_relative_delta=.123), "discounted_relative_delta"),
    (lambda x: x["labels"].update(terminal_log_nav_ratio=.123), "terminal_log_nav_ratio"),
    (lambda x: x["labels"]["per_step_relative_reward_difference"].__setitem__(0, .123), "saved reward difference"),
    (lambda x: x["labels"]["branches"]["rebalance"]["steps"][0].update(relative_log_reward=.123), "relative log rewards"),
    (lambda x: x["labels"]["branches"]["rebalance"]["steps"][0]["agent_ledger"]["orders"][0].update(fee=9.), "commission"),
    (lambda x: x["labels"]["branches"]["rebalance"]["steps"][0]["agent_ledger"]["orders"][0].update(slippage_cost=9.), "slippage"),
    (lambda x: x["labels"]["branches"]["rebalance"]["steps"][1]["agent_ledger"].update(total_fees=99.), "cumulative"),
    (lambda x: x["labels"]["branches"]["rebalance"]["steps"][0]["agent_ledger"].update(cash=99.), "cash including"),
    (lambda x: x["labels"]["branches"]["rebalance"]["steps"][0]["agent_ledger"].update(available_positions=[1000., 0.]), "T\\+1"),
    (lambda x: x["labels"]["branches"]["rebalance"]["steps"][0].update(macro_action=True), "Calendar"),
    (lambda x: x["labels"]["branches"]["hold"]["steps"][0].update(terminated=True), "Calendar"),
])
def test_resigned_scalar_or_native_arithmetic_lies_are_rejected(mutation, match):
    label, record = example()
    mutation(label)
    with pytest.raises(ValueError, match=match):
        audit(label, record)


def test_shadow_orders_must_match_even_if_nav_and_costs_are_unchanged():
    label, record = example(quantity=0, rejected=True)
    order = deepcopy(label["labels"]["branches"]["rebalance"]["steps"][0]["agent_ledger"]["orders"][0])
    label["labels"]["branches"]["rebalance"]["steps"][0]["benchmark_ledger"]["orders"].append(order)
    with pytest.raises(ValueError, match="Full shadow paths differ"):
        audit(label, record)


def test_overlap_and_duplicates_remain_descriptive_not_independent_samples():
    label, record = example()
    one = audit(label, record)
    two = dict(one, state_id="001_60", episode=1)
    description = tool.describe([one, two])
    assert description["states"] == description["episodes"] == 2
    assert description["unique_date_windows"] == 1
    assert description["duplicated_inputs"] == description["overlapping_interval_pairs"] == 1
    grouped = tool.strata([one, two])
    assert any(row["group"] == "due_status" and row["value"] == "non0" and row["states"] == 2 for row in grouped)
    empty = next(row for row in grouped if row["group"] == "due_status" and row["value"] == "phase0")
    assert empty["states"] == 0 and empty["mean"] is None and empty["oracle_mean"] is None


def fake_complete_root(tmp_path, monkeypatch):
    root = tmp_path / "pilot"
    root.mkdir()
    protocol = dict(folds={fold: dict(data_identity=dict(contract=dict(stock_codes=CODES, broker=BROKER)))
                          for fold in tool.driver.FOLDS})
    (root / "protocol.json").write_bytes(tool.driver.canonical(protocol))
    for fold in tool.driver.FOLDS:
        directory = root / "folds" / fold
        (directory / "labels").mkdir(parents=True)
        states = []
        for i in range(96):
            label, record = example(fold=fold, phase=0 if i % 2 else 1)
            record.update(state_id=f"{i // 2:03d}_{(0, 60)[i % 2]:02d}", episode=i // 2, offset=(0, 60)[i % 2])
            states.append(record)
            (directory / "labels" / (record["state_id"] + ".json.gz")).write_bytes(gzip.compress(tool.driver.canonical(label), mtime=0))
        (directory / "sampling_plan.json").write_bytes(tool.driver.canonical(dict(states=states)))
        files = {path.relative_to(directory).as_posix(): tool.driver.file_hash(path) for path in directory.rglob("*") if path.is_file()}
        (directory / "completed.json").write_bytes(tool.driver.canonical(dict(counters=tool.driver.expected_counts(), files=files)))
    monkeypatch.setattr(tool.driver, "output_root", lambda value: Path(value).resolve())
    monkeypatch.setattr(tool.driver, "check_frozen", lambda *args: None)
    monkeypatch.setattr(tool.driver, "cached_fold", lambda *args: [None] * 96)
    for name in ("make_data", "new_environment", "run_fold", "prepare"):
        monkeypatch.setattr(tool.driver, name, lambda *args, **kwargs: pytest.fail("cold analysis called financial API"))
    monkeypatch.setattr(tool.driver.paired, "paired_rule_label", lambda *args, **kwargs: pytest.fail("cold analysis generated a label"))
    return root


def test_complete_cold_run_writes_only_new_diagnostics_and_refuses_overwrite(tmp_path, monkeypatch):
    root = fake_complete_root(tmp_path, monkeypatch)
    before = {path: tool.driver.file_hash(path) for path in root.rglob("*") if path.is_file()}
    output = tool.run(root)
    assert output == root / "diagnostics" / "cold_analysis"
    assert {path.name for path in output.iterdir()} == set(tool.FILES)
    analysis = json.loads((output / "analysis.json").read_text())
    assert analysis["total_agent_steps"] == analysis["total_benchmark_steps"] == 19200
    assert analysis["model_fits"] == analysis["new_nn_forwards"] == analysis["new_environment_steps"] == 0
    assert analysis["pooled_descriptive_only"]["states"] == 192
    manifest = json.loads((output / "manifest.json").read_text())
    assert all(tool.driver.file_hash(output / name) == sha for name, sha in manifest["outputs_sha256"].items())
    assert all(tool.driver.file_hash(path) == sha for path, sha in before.items())
    with pytest.raises(ValueError, match="already exists"):
        tool.run(root)


def test_partial_or_untrusted_scope_cannot_emit_diagnostics(tmp_path, monkeypatch):
    root = fake_complete_root(tmp_path, monkeypatch)
    monkeypatch.setattr(tool.driver, "cached_fold", lambda *args: None)
    with pytest.raises(ValueError, match="sealed"):
        tool.run(root)
    assert not (root / "diagnostics").exists()


def test_output_symlink_cannot_alias_a_sealed_fold(tmp_path, monkeypatch):
    root = fake_complete_root(tmp_path, monkeypatch)
    resolve = Path.resolve

    def alias(self, *args, **kwargs):
        if self == root / "diagnostics":
            return root / "folds" / "2023"
        return resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", alias)
    with pytest.raises(ValueError, match="aliased"):
        tool.run(root)


def test_changed_input_during_output_writing_is_detected_without_completion_manifest(tmp_path, monkeypatch):
    root = fake_complete_root(tmp_path, monkeypatch)
    original = tool._write_csv

    def mutate_input(path, rows):
        original(path, rows)
        (root / "folds" / "2023" / "sampling_plan.json").write_text("{}")

    monkeypatch.setattr(tool, "_write_csv", mutate_input)
    with pytest.raises(ValueError, match="input changed"):
        tool.run(root)
    assert not (root / "diagnostics" / "cold_analysis" / "manifest.json").exists()
