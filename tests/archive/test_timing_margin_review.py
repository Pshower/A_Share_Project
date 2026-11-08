"""Independent margin admission checks; no market steps or parameter updates."""

from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from tools.archive import research_timing_margin as study
from src.data.preprocess import file_hash, read_config, write_json
from src.envs.initial_account import MODES
from src.training.freeze import json_digest


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Independent review must not train or advance a market")

    monkeypatch.setattr(study.PPO, "learn", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def protocol():
    return dict(
        base_config={"ppo": {}},
        folds={fold: {"research_config": "synthetic.json"} for fold in ("2023", "2024", "2025", "2026")},
        tool_sha256="synthetic_runner",
        reference_final_audit_sha256="synthetic_reference",
    )


def metric(roi=0.):
    return dict(cumulative_return=roi, max_drawdown=-.03, total_fees=1., total_slippage_cost=2.)


def screen_rows(gap=.001, changed=True):
    return [dict(
        fold=fold, margin=margin, phase="screen", seed=42, steps=8192,
        metrics={mode: dict(trained=metric(gap), initialized=metric(), benchmark=metric()) for mode in MODES},
        changes={mode: dict(greedy_changed=changed, economic_ledger_changed=changed) for mode in MODES},
    ) for fold in study.original.SCREEN_FOLDS for margin in study.MARGINS]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "early_checkpoint", "wrong_seed"])
def test_every_screen_condition_is_required_at_its_fixed_final_budget(defect):
    rows = screen_rows()
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif defect == "early_checkpoint":
        rows[0]["steps"] = 7680
    else:
        rows[0]["seed"] = 123
    with pytest.raises(ValueError):
        study.select_screen(rows)


@pytest.mark.parametrize("gap,admitted", [(np.nextafter(.0001, 0.), False), (.0001, True)])
def test_one_basis_point_gate_does_not_round_up(gap, admitted):
    assert study.select_screen(screen_rows(gap=float(gap)))["admitted"] is admitted


def test_failed_best_candidate_does_not_fall_back_to_second_margin():
    rows = screen_rows(.001, True)
    for row in rows:
        if row["margin"] == .5:
            for mode in MODES:
                row["metrics"][mode]["trained"] = metric(.003)
                row["changes"][mode] = dict(greedy_changed=False, economic_ledger_changed=False)
    selected = study.select_screen(rows)
    assert selected["selected_margin"] == .5
    assert not selected["admitted"]
    assert selected["confirmation_steps_authorized"] == 0
    assert selected["no_fallback"] is True


def test_macro_and_ledger_flags_cannot_be_combined_from_different_accounts():
    rows = screen_rows(changed=False)
    for row in rows:
        row["changes"]["cash"]["greedy_changed"] = True
        row["changes"]["equal"]["economic_ledger_changed"] = True
    assert not study.select_screen(rows)["admitted"]


@pytest.mark.parametrize("defect", ["boolean_margin", "numeric_flag", "string_flag"])
def test_condition_and_behavior_flags_have_strict_types(defect):
    rows = screen_rows()
    if defect == "boolean_margin":
        next(row for row in rows if row["margin"] == 1.)["margin"] = True
    elif defect == "numeric_flag":
        rows[0]["changes"]["cash"]["greedy_changed"] = 1
    else:
        rows[0]["changes"]["cash"]["economic_ledger_changed"] = "true"
    with pytest.raises(ValueError):
        study.select_screen(rows)


class ObservationOnlyEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (129,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        vector = np.zeros(129, np.float32)
        vector[0] = 1
        return vector, {}

    def step(self, action):
        raise AssertionError("Policy initialization must not advance even a synthetic environment")


def test_actual_model_factory_changes_only_calendar_weights_and_preserves_critic(tmp_path):
    states = []
    for margin in study.MARGINS:
        config = study.config_for(protocol(), tmp_path, "2023", margin, "screen", 42)
        model = study.make_model(ObservationOnlyEnv(), config)
        state = {name: value.detach().cpu().clone() for name, value in model.policy.state_dict().items()}
        expected = torch.zeros((2, 129))
        expected[0, 1:20], expected[1, 0] = margin, margin
        assert torch.equal(state["action_net.weight"], expected)
        assert torch.count_nonzero(state["action_net.bias"]) == 0
        assert model.policy.parameter_count == 12805
        vector = torch.zeros((20, 129))
        vector[:, :20] = torch.eye(20)
        with torch.no_grad():
            distribution = model.policy.get_distribution(vector)
            probability = distribution.distribution.probs
        expected_action = torch.zeros(20, dtype=torch.long)
        expected_action[0] = 1
        assert torch.equal(torch.argmax(probability, dim=1), expected_action)
        torch.testing.assert_close(
            probability[torch.arange(20), expected_action],
            torch.full((20,), torch.sigmoid(torch.tensor(margin)).item()),
            atol=2e-7, rtol=0.,
        )
        states.append(state)
        model.get_env().close()
    for state in states[1:]:
        assert set(state) == set(states[0])
        for name in state:
            if name != "action_net.weight":
                assert torch.equal(state[name], states[0][name]), name


def sealed_screen(tmp_path, monkeypatch, rows=None):
    root, registered = tmp_path / "study", protocol()
    root.mkdir()
    write_json(root / "protocol.json", registered)
    rows = screen_rows() if rows is None else rows
    for row in rows:
        identifier = study.run_id(row["fold"], row["margin"], "screen", row["seed"])
        directory = root / "runs" / identifier
        directory.mkdir(parents=True)
        write_json(directory / "native.json", {"synthetic": True})
        write_json(directory / "completed.json", dict(
            record=row, files={"native.json": file_hash(directory / "native.json")},
        ))
    for fold in study.original.SCREEN_FOLDS:
        identifiers = [study.run_id(fold, margin, "screen", 42) for margin in study.MARGINS]
        write_json(root / f"worker_screen_{fold}.json", dict(
            status="complete", phase="screen", fold=fold,
            protocol_sha256=file_hash(root / "protocol.json"), selection_sha256=None,
            no_global_summary_writes=True,
            completed_runs=[dict(run_id=name, completed_sha256=file_hash(root / "runs" / name / "completed.json")) for name in identifiers],
        ))
    selected = study.seal_selection(root, registered, rows)
    checked = []

    def validate_synthetic_cache(directory, config, registered_protocol, selection_sha256=None):
        directory = Path(directory)
        checked.append(directory.name)
        completed = read_config(directory / "completed.json")
        study.original.verify_leaf_files(directory, completed["files"])
        return completed["record"]

    monkeypatch.setattr(study, "check_frozen", lambda _: None)
    monkeypatch.setattr(study, "cached_trial", validate_synthetic_cache)
    return root, registered, rows, selected, checked


def test_confirmation_revalidates_all_six_native_caches(tmp_path, monkeypatch):
    root, registered, _, _, checked = sealed_screen(tmp_path, monkeypatch)
    study.require_confirmation(root, registered, 2., file_hash(root / "selection.json"))
    assert set(checked) == {study.run_id(fold, margin, "screen", 42) for fold in study.original.SCREEN_FOLDS for margin in study.MARGINS}
    assert len(checked) == 6


def test_confirmation_rejects_changed_leaf_without_changed_marker(tmp_path, monkeypatch):
    root, registered, _, _, _ = sealed_screen(tmp_path, monkeypatch)
    directory = root / "runs" / study.run_id("2023", .5, "screen", 42)
    marker_before = file_hash(directory / "completed.json")
    write_json(directory / "native.json", {"synthetic": "changed_after_admission"})
    assert file_hash(directory / "completed.json") == marker_before
    with pytest.raises(ValueError):
        study.require_confirmation(root, registered, 2., file_hash(root / "selection.json"))


def test_direct_confirmation_trial_stops_before_data_or_output_on_failed_admission(tmp_path, monkeypatch):
    root, registered, _, _, _ = sealed_screen(tmp_path, monkeypatch, screen_rows(gap=0.))

    def forbidden(*args, **kwargs):
        raise AssertionError("A failed screen must never reach market construction")

    monkeypatch.setattr(study, "MarginData", forbidden)
    with pytest.raises(ValueError):
        study.run_trial(root, registered, "2025", 2., "confirm", 42, file_hash(root / "selection.json"))
    assert not (root / "runs" / "confirm_2025_m20_42").exists()


def test_public_report_cannot_replace_selected_admitted_stage_with_failure(tmp_path, monkeypatch):
    root, registered, rows, selected, _ = sealed_screen(tmp_path, monkeypatch)
    forged = deepcopy(selected)
    forged["admitted"] = False
    forged["status"] = "screen_failed"
    with pytest.raises(ValueError):
        study.write_report(root, registered, rows, forged)
    assert not (root / "study_summary.json").exists()


@pytest.mark.parametrize("defect", ["source", "tool", "document", "test_digest", "missing_test", "empty_tests", "reference_audit", "budget"])
def test_freeze_covers_full_source_tool_document_and_test_inventory(tmp_path, monkeypatch, defect):
    previous = tmp_path / "previous"
    previous.mkdir()
    inherited = protocol()
    inherited["gate"] = {"synthetic_inherited_gate": True}
    write_json(previous / "protocol.json", inherited)
    registered = dict(
        study="timing_margin", margins=list(study.MARGINS), reward_basis="log_net",
        screen_folds=list(study.original.SCREEN_FOLDS), confirm_folds=list(study.original.CONFIRM_FOLDS),
        seeds=list(study.original.SEEDS), budgets=deepcopy(study.original.BUDGETS),
        feature_names=list(study.FEATURE_NAMES), parameters=12805,
        planned_screen_trials=6, planned_screen_steps=49152, maximum_confirmation_trials=6,
        maximum_confirmation_steps=98304, maximum_actual_training_steps=147456,
        screen_minimum_each_fold_gap=.0001, screen_requires_greedy_and_economic_change=True,
        tie_break="larger margin", folds=inherited["folds"], base_config=inherited["base_config"], gate=inherited["gate"],
        source_hashes=study.source_hashes(), tool_sha256=file_hash(Path(study.__file__)),
        document_sha256=file_hash(study.DOCUMENT),
        test_hashes={name: file_hash(study.ROOT / name) for name in study.TEST_FILES},
        reference_root=str(previous), reference_protocol_sha256=file_hash(previous / "protocol.json"),
        reference_final_audit_sha256="pinned_synthetic_audit",
    )
    monkeypatch.setattr(study, "REFERENCE_AUDIT_SHA256", "pinned_synthetic_audit")
    monkeypatch.setattr(study, "reference_protocol", lambda _: inherited)
    study.check_frozen(registered)
    if defect == "source":
        changed = deepcopy(registered["source_hashes"])
        changed[next(iter(changed))] = "changed"
        monkeypatch.setattr(study, "source_hashes", lambda: changed)
    elif defect in ("tool", "document"):
        registered[f"{defect}_sha256"] = "changed"
    elif defect == "test_digest":
        registered["test_hashes"][study.TEST_FILES[-1]] = "changed"
    elif defect == "missing_test":
        registered["test_hashes"].pop(study.TEST_FILES[-1])
    elif defect == "empty_tests":
        registered["test_hashes"] = {}
    elif defect == "budget":
        registered["budgets"]["screen"] = 4096
    else:
        registered["reference_final_audit_sha256"] = "not_the_pinned_audit"
    with pytest.raises(ValueError):
        study.check_frozen(registered)


def test_synthetic_prepare_does_not_expose_old_module_budget_by_reference(tmp_path, monkeypatch):
    previous = tmp_path / "previous"
    previous.mkdir()
    inherited = dict(folds={}, base_config={"ppo": {}}, gate={})
    write_json(previous / "protocol.json", inherited)
    monkeypatch.setattr(study, "reference_protocol", lambda _: inherited)
    monkeypatch.setattr(study, "check_frozen", lambda _: None)
    monkeypatch.setattr(study, "REFERENCE_AUDIT_SHA256", "synthetic_only")
    registered = study.prepare(tmp_path / "synthetic_registration", previous)
    assert registered["budgets"] == {"screen": 8192, "confirm": 16384}
    assert registered["budgets"] is not study.original.BUDGETS


def test_unpinned_reference_blocks_prepare_even_after_previous_audit_exists(tmp_path, monkeypatch):
    previous = tmp_path / "previous"
    previous.mkdir()
    write_json(previous / "final_audit.json", {"status": "passed"})
    monkeypatch.setattr(study, "REFERENCE_AUDIT_SHA256", None)
    with pytest.raises(ValueError, match="pinned"):
        study.prepare(tmp_path / "new_study", previous)
    assert not (tmp_path / "new_study").exists()


def test_margin_two_cannot_reuse_predecessor_completed_marker(tmp_path, monkeypatch):
    registered = protocol()
    config = study.config_for(registered, tmp_path, "2023", 2., "screen", 42)
    directory = tmp_path / "borrowed_old_m2"
    directory.mkdir()
    write_json(directory / "completed.json", dict(protocol_sha256="predecessor_registration"))
    monkeypatch.setattr(study, "check_frozen", lambda _: None)
    monkeypatch.setattr(study, "MarginData", lambda _: pytest.fail("Old cache must be rejected before constructing data"))
    with pytest.raises(ValueError):
        study.cached_trial(directory, config, registered)


def test_fresh_margin_two_uses_new_trial_not_original_cache(tmp_path, monkeypatch):
    registered = protocol()
    monkeypatch.setattr(study, "check_frozen", lambda _: None)
    monkeypatch.setattr(study.original, "cached_trial", lambda *args, **kwargs: pytest.fail("New m2 cannot borrow the predecessor cached trial"))

    def reached_fresh_data(config):
        assert config["margin_study"]["study"] == "timing_margin"
        raise LookupError("fresh registered trial reached data boundary")

    monkeypatch.setattr(study, "MarginData", reached_fresh_data)
    with pytest.raises(LookupError, match="fresh registered"):
        study.run_trial(tmp_path, registered, "2023", 2., "screen", 42)
    assert (tmp_path / "runs" / "screen_2023_m20_42" / "config.json").exists()
    assert not (tmp_path / "runs" / "screen_2023_m20_42" / "completed.json").exists()
