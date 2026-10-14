"""Independent reward/GAE admission checks without market steps or optimization."""

from copy import deepcopy
import json
from pathlib import Path
import zipfile

import gymnasium as gym
import numpy as np
import pytest
import torch

from tools import research_timing_credit as study
from src.data.preprocess import file_hash, read_config, write_json
from src.envs.initial_account import MODES


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Independent credit review must not optimize or advance a market")

    monkeypatch.setattr(study.PPO, "learn", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def protocol():
    return dict(base_config={"ppo": {}},
        folds={fold: {"research_config": "synthetic.json"} for fold in ("2023", "2024", "2025", "2026")},
        tool_sha256="synthetic_runner", helper_sha256="synthetic_helper", reference_final_audit_sha256="synthetic_audit")


def metric(roi=0.):
    return dict(cumulative_return=roi, max_drawdown=-.03, total_fees=1., total_slippage_cost=2.)


def rows(gap=.001, changed=True):
    return [dict(fold=fold, reward_basis=reward, gae_lambda=value, method=reward, phase="screen", seed=42, steps=8192,
        metrics={mode: dict(trained=metric(gap), initialized=metric(), benchmark=metric()) for mode in MODES},
        changes={mode: dict(greedy_changed=changed, economic_ledger_changed=changed) for mode in MODES})
        for fold in study.original.SCREEN_FOLDS for reward, value in study.CELLS]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "early_budget"])
def test_complete_eight_cell_fixed_final_grid_is_required(defect):
    data = rows()
    if defect == "missing":
        data.pop()
    elif defect == "duplicate":
        data[-1] = deepcopy(data[0])
    else:
        data[0]["steps"] = 7680
    with pytest.raises(ValueError):
        study.select_screen(data)


@pytest.mark.parametrize("gap,admitted", [(np.nextafter(.0001, 0.), False), (.0001, True)])
def test_one_basis_point_is_not_rounded_up(gap, admitted):
    assert study.select_screen(rows(float(gap)))["admitted"] is admitted


def test_ties_prefer_lambda_point95_then_log_net_and_never_fall_back():
    tied = study.select_screen(rows())
    assert (tied["selected_reward"], tied["selected_lambda"]) == ("log_net", .95)
    data = rows()
    for row in data:
        if (row["reward_basis"], row["gae_lambda"]) == ("relative_log", 1.):
            for mode in MODES:
                row["metrics"][mode]["trained"] = metric(.003)
                row["changes"][mode] = dict(greedy_changed=False, economic_ledger_changed=False)
    selected = study.select_screen(data)
    assert (selected["selected_reward"], selected["selected_lambda"]) == ("relative_log", 1.)
    assert not selected["admitted"] and selected["confirmation_steps_authorized"] == 0
    assert selected["no_fallback"] is True


def test_behavior_flags_must_be_jointly_true_on_the_same_account_path():
    data = rows(changed=False)
    for row in data:
        row["changes"]["cash"]["greedy_changed"] = True
        row["changes"]["equal"]["economic_ledger_changed"] = True
    assert not study.select_screen(data)["admitted"]


@pytest.mark.parametrize("defect", ["boolean_lambda", "numeric_flag"])
def test_boolean_lambda_and_numeric_behavior_flags_are_not_registered(defect):
    data = rows()
    if defect == "boolean_lambda":
        next(row for row in data if row["gae_lambda"] == 1.)["gae_lambda"] = True
    else:
        data[0]["changes"]["cash"]["greedy_changed"] = 1
    with pytest.raises(ValueError):
        study.select_screen(data)


class ObservationOnlyEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (129,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        vector = np.zeros(129, np.float32)
        vector[0] = 1
        return vector, {}

    def step(self, action):
        raise AssertionError("Initialization review must not step an environment")


def test_actual_factory_keeps_all_theta_identical_across_reward_and_trace_cells(tmp_path):
    states, normalized = [], []
    for reward, value in study.CELLS:
        config = study.config_for(protocol(), tmp_path, "2023", reward, value, "screen", 42)
        assert config["timing"]["calendar_margin"] == 2. and config["ppo"]["gamma"] == .995
        assert config["ppo"]["gae_lambda"] == value and config["device"] == "cpu"
        comparable = deepcopy(config)
        comparable.pop("credit_study")
        comparable["timing"]["reward_basis"] = "same"
        comparable["ppo"]["gae_lambda"] = "same"
        normalized.append(comparable)
        model = study.make_model(ObservationOnlyEnv(), config)
        assert model.gamma == .995 and model.gae_lambda == value and model.policy.parameter_count == 12805
        state = {key: tensor.detach().cpu().clone() for key, tensor in model.policy.state_dict().items()}
        expected = torch.zeros((2, 129))
        expected[0, 1:20], expected[1, 0] = 2., 2.
        assert torch.equal(state["action_net.weight"], expected)
        states.append(state)
        model.get_env().close()
    assert all(item == normalized[0] for item in normalized)
    for state in states[1:]:
        assert set(state) == set(states[0])
        assert all(torch.equal(state[key], states[0][key]) for key in state)


def sealed_screen(tmp_path, monkeypatch, data=None):
    root, registered = tmp_path / "study", protocol()
    root.mkdir()
    write_json(root / "protocol.json", registered)
    data = rows() if data is None else data
    for row in data:
        name = study.run_id(row["fold"], row["reward_basis"], row["gae_lambda"], "screen", row["seed"])
        directory = root / "runs" / name
        directory.mkdir(parents=True)
        write_json(directory / "native.json", {"synthetic": True})
        write_json(directory / "completed.json", dict(record=row, files={"native.json": file_hash(directory / "native.json")}))
    for fold in study.original.SCREEN_FOLDS:
        names = [study.run_id(fold, reward, value, "screen", 42) for reward, value in study.CELLS]
        write_json(root / f"worker_screen_{fold}.json", dict(status="complete", phase="screen", fold=fold,
            protocol_sha256=file_hash(root / "protocol.json"), selection_sha256=None, no_global_summary_writes=True,
            completed_runs=[dict(run_id=name, completed_sha256=file_hash(root / "runs" / name / "completed.json")) for name in names]))
    selected = study.seal_selection(root, registered, data)
    checked = []

    def cache(directory, config, frozen, selection_sha256=None):
        directory = Path(directory)
        checked.append(directory.name)
        saved = read_config(directory / "completed.json")
        study.original.verify_leaf_files(directory, saved["files"])
        return saved["record"]

    monkeypatch.setattr(study, "check_frozen", lambda _: None)
    monkeypatch.setattr(study, "cached_trial", cache)
    return root, registered, data, selected, checked


def test_confirmation_revalidates_all_eight_sealed_native_caches(tmp_path, monkeypatch):
    root, registered, _, _, checked = sealed_screen(tmp_path, monkeypatch)
    study.require_confirmation(root, registered, ("log_net", .95), file_hash(root / "selection.json"))
    assert len(checked) == 8
    assert set(checked) == {study.run_id(fold, reward, value, "screen", 42) for fold in study.original.SCREEN_FOLDS for reward, value in study.CELLS}


def test_changed_native_leaf_blocks_confirmation_even_with_unchanged_marker(tmp_path, monkeypatch):
    root, registered, _, _, _ = sealed_screen(tmp_path, monkeypatch)
    directory = root / "runs" / study.run_id("2023", "log_net", .95, "screen", 42)
    before = file_hash(directory / "completed.json")
    write_json(directory / "native.json", {"synthetic": "changed"})
    assert file_hash(directory / "completed.json") == before
    with pytest.raises(ValueError):
        study.require_confirmation(root, registered, ("log_net", .95), file_hash(root / "selection.json"))


def test_boolean_cell_cannot_match_an_admitted_lambda_one(tmp_path, monkeypatch):
    data = rows()
    for row in data:
        if (row["reward_basis"], row["gae_lambda"]) == ("log_net", 1.):
            for mode in MODES:
                row["metrics"][mode]["trained"] = metric(.002)
    root, registered, _, selected, _ = sealed_screen(tmp_path, monkeypatch, data)
    assert selected["admitted"] and selected["selected_lambda"] == 1.
    with pytest.raises(ValueError):
        study.require_confirmation(root, registered, ("log_net", True), file_hash(root / "selection.json"))


def test_failed_direct_trial_never_reaches_data_or_creates_confirmation(tmp_path, monkeypatch):
    root, registered, _, _, _ = sealed_screen(tmp_path, monkeypatch, rows(gap=0.))
    monkeypatch.setattr(study, "CreditData", lambda _: pytest.fail("A failed screen cannot construct financial data"))
    with pytest.raises(ValueError):
        study.run_trial(root, registered, "2025", "log_net", .95, "confirm", 42, file_hash(root / "selection.json"))
    assert not (root / "runs" / "confirm_2025_log_l095_42").exists()


def test_report_direct_entry_rejects_fabricated_failed_selection(tmp_path, monkeypatch):
    root, registered, data, selected, _ = sealed_screen(tmp_path, monkeypatch)
    fabricated = deepcopy(selected)
    fabricated["admitted"] = False
    with pytest.raises(ValueError):
        study.write_report(root, registered, data, fabricated)
    assert not (root / "study_summary.json").exists()


def archive_fixture(tmp_path):
    registered = protocol()
    registered["source_hashes"] = {"synthetic_source": "same"}
    config = study.config_for(registered, tmp_path, "2023", "log_net", 1., "screen", 42)
    contract = dict(synthetic=True, credit_study=deepcopy(config["credit_study"]))
    record = dict(policy_class="src.agents.timing_policy.CalendarTimingPolicy", observation_shape=[129], observation_dtype="float32",
        feature_names=list(study.FEATURE_NAMES), action_names=list(study.ACTION_NAMES), net_arch={"pi": [], "vf": [64, 64]},
        calendar_margin=2., phase_period=20, parameter_count=12805)
    saved = dict(timing_registered_contract=deepcopy(contract), timing_policy_record=record,
        timing_registered_algorithm="stable_baselines3.ppo.ppo.PPO", num_timesteps=8192, gae_lambda=1., gamma=.995)
    metadata = dict(contract=deepcopy(contract), config=deepcopy(config), source_hashes=registered["source_hashes"],
        provenance=dict(credit_study=deepcopy(config["credit_study"])), policy_record=record, num_timesteps=8192,
        actual_device="cpu", feature_names=list(study.FEATURE_NAMES), algorithm_class="stable_baselines3.ppo.ppo.PPO")
    directory = tmp_path / "synthetic_metadata_only"
    directory.mkdir()

    def save_header():
        with zipfile.ZipFile(directory / "model.zip", "w") as archive:
            archive.writestr("data", json.dumps(saved))
        metadata["model_sha256"] = file_hash(directory / "model.zip")

    save_header()
    study.verify_model(directory, metadata, config, registered, contract, 8192)
    return directory, metadata, config, registered, contract, saved, save_header


@pytest.mark.parametrize("defect", ["archive_boolean_lambda", "contract_boolean_lambda", "gamma", "helper_provenance", "reward_provenance"])
def test_saved_estimator_and_provenance_cannot_disagree_with_registered_config(tmp_path, defect):
    directory, metadata, config, registered, contract, saved, save_header = archive_fixture(tmp_path)
    if defect == "archive_boolean_lambda":
        saved["gae_lambda"] = True
        save_header()
    elif defect == "contract_boolean_lambda":
        metadata["contract"]["credit_study"]["gae_lambda"] = True
    elif defect == "gamma":
        saved["gamma"] = .99
        save_header()
    elif defect == "helper_provenance":
        metadata["provenance"]["credit_study"]["helper_sha256"] = "different"
    else:
        metadata["provenance"]["credit_study"]["reward_basis"] = "relative_log"
    with pytest.raises(ValueError):
        study.verify_model(directory, metadata, config, registered, contract, 8192)


def test_boolean_ppo_lambda_is_rejected_before_parent_data_constructor(tmp_path, monkeypatch):
    config = study.config_for(protocol(), tmp_path, "2023", "log_net", 1., "screen", 42)
    config["ppo"]["gae_lambda"] = True
    monkeypatch.setattr(study.ExplorationData, "__init__", lambda *args, **kwargs: pytest.fail("A malformed estimator must not open financial data"))
    with pytest.raises(ValueError):
        study.CreditData(config)


@pytest.mark.parametrize("workspace_key", [True, False])
def test_parent_audit_uses_workspace_relative_signature_keys(tmp_path, monkeypatch, workspace_key):
    directory = tmp_path / "reports" / "parent"
    directory.mkdir(parents=True)
    parent = {"synthetic_parent": True}
    write_json(directory / "protocol.json", parent)
    key = "reports/parent/protocol.json" if workspace_key else "protocol.json"
    write_json(directory / "final_audit.json", dict(status="passed", signed_files_sha256={key: file_hash(directory / "protocol.json")}))
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "REFERENCE_AUDIT_SHA256", file_hash(directory / "final_audit.json"))
    monkeypatch.setattr(study.shared, "check_frozen", lambda value: None)
    if workspace_key:
        assert study.reference_protocol(directory) == parent
    else:
        with pytest.raises(ValueError):
            study.reference_protocol(directory)


@pytest.mark.parametrize("defect", ["source", "tool", "helper", "document", "test_digest", "missing_test", "boolean_grid", "budget"])
def test_freeze_inventory_has_valid_baseline_and_rejects_single_field_drift(tmp_path, monkeypatch, defect):
    prior, reproduction = tmp_path / "prior", tmp_path / "original_timing"
    prior.mkdir()
    reproduction.mkdir()
    write_json(reproduction / "protocol.json", {"synthetic_control": True})
    inherited = protocol()
    inherited.update(gate={"synthetic_gate": True}, reference_root=str(reproduction))
    write_json(prior / "protocol.json", inherited)
    registered = dict(study="timing_credit", rewards=list(study.REWARDS), gae_lambdas=list(study.LAMBDAS),
        budgets=deepcopy(study.BUDGETS), parameters=12805, feature_names=list(study.FEATURE_NAMES),
        planned_screen_trials=8, planned_screen_steps=65536, maximum_confirmation_trials=6,
        maximum_confirmation_steps=98304, maximum_actual_training_steps=163840,
        screen_minimum_each_fold_gap=.0001, screen_requires_greedy_and_economic_change=True, tie_break="lambda .95 then log_net",
        source_hashes=study.source_hashes(), tool_sha256=file_hash(Path(study.__file__)), helper_sha256=file_hash(study.ROOT / study.HELPER_FILE),
        document_sha256=file_hash(study.DOCUMENT), test_hashes={name: file_hash(study.ROOT / name) for name in study.TEST_FILES},
        reference_root=str(prior), reference_protocol_sha256=file_hash(prior / "protocol.json"), reference_final_audit_sha256="synthetic_pin",
        reproduction_reference_root=str(reproduction), reproduction_protocol_sha256=file_hash(reproduction / "protocol.json"),
        folds=inherited["folds"], base_config=inherited["base_config"], gate=inherited["gate"])
    monkeypatch.setattr(study, "REFERENCE_AUDIT_SHA256", "synthetic_pin")
    monkeypatch.setattr(study, "reference_protocol", lambda _: inherited)
    study.check_frozen(registered)
    if defect == "source":
        different = deepcopy(registered["source_hashes"])
        different[next(iter(different))] = "changed"
        monkeypatch.setattr(study, "source_hashes", lambda: different)
    elif defect in ("tool", "helper", "document"):
        registered[f"{defect}_sha256"] = "different"
    elif defect == "test_digest":
        registered["test_hashes"][study.TEST_FILES[-1]] = "different"
    elif defect == "missing_test":
        registered["test_hashes"].pop(study.TEST_FILES[-1])
    elif defect == "boolean_grid":
        registered["gae_lambdas"][-1] = True
    else:
        registered["budgets"]["screen"] = 4096
    with pytest.raises(ValueError):
        study.check_frozen(registered)


def test_synthetic_prepare_copies_budget_and_inherited_data(tmp_path, monkeypatch):
    original, prior = tmp_path / "original", tmp_path / "parent"
    original.mkdir()
    prior.mkdir()
    write_json(original / "protocol.json", {"synthetic": True})
    inherited = dict(base_config={"ppo": {}}, folds={}, gate={}, reference_root=str(original))
    write_json(prior / "protocol.json", inherited)
    monkeypatch.setattr(study, "reference_protocol", lambda _: inherited)
    monkeypatch.setattr(study, "check_frozen", lambda _: None)
    registered = study.prepare(tmp_path / "synthetic_registration", prior)
    assert registered["budgets"] == {"screen": 8192, "confirm": 16384}
    assert registered["budgets"] is not study.BUDGETS
    assert registered["base_config"] is not inherited["base_config"]
    assert registered["base_config"]["ppo"] is not inherited["base_config"]["ppo"]


def test_lambda_point99_is_not_a_registered_credit_factor():
    with pytest.raises(ValueError):
        study.cell_label("log_net", .99)


@pytest.mark.parametrize("reward", study.REWARDS)
@pytest.mark.parametrize("fold", study.original.SCREEN_FOLDS)
def test_point95_controls_enter_fresh_trial_instead_of_old_cache(tmp_path, monkeypatch, reward, fold):
    registered = protocol()
    monkeypatch.setattr(study, "check_frozen", lambda _: None)
    monkeypatch.setattr(study.original, "cached_trial", lambda *args, **kwargs: pytest.fail("Old timing cache cannot replace a new control sample"))

    def fresh_boundary(config):
        assert config["credit_study"]["study"] == "timing_credit" and config["ppo"]["gae_lambda"] == .95
        raise LookupError("fresh control reached data boundary")

    monkeypatch.setattr(study, "CreditData", fresh_boundary)
    with pytest.raises(LookupError, match="fresh control"):
        study.run_trial(tmp_path, registered, fold, reward, .95, "screen", 42)
    directory = tmp_path / "runs" / study.run_id(fold, reward, .95, "screen", 42)
    assert (directory / "config.json").exists() and not (directory / "completed.json").exists()


def test_predecessor_completed_marker_is_rejected_before_credit_data(tmp_path, monkeypatch):
    directory = tmp_path / "copied_predecessor"
    directory.mkdir()
    write_json(directory / "completed.json", {"protocol_sha256": "old_registration"})
    registered = protocol()
    config = study.config_for(registered, tmp_path, "2023", "relative_log", .95, "screen", 42)
    monkeypatch.setattr(study, "CreditData", lambda _: pytest.fail("Old sample must fail before opening data"))
    with pytest.raises(ValueError):
        study.cached_trial(directory, config, registered)


@pytest.mark.parametrize("changed_component", ["benchmark", "rollout"])
def test_control_comparison_includes_original_benchmark_and_ppo_arrays(tmp_path, monkeypatch, changed_component):
    previous, current = tmp_path / "previous", tmp_path / "current"
    chunk = dict(path="chunk.npz", first_step=1, last_step=2, steps=2)
    audit = dict(training_trace_prefix=dict(benchmark_trace=dict(chunks=[chunk])), rollout_buffers=[chunk])
    for directory in (previous, current):
        benchmark = directory / "training_trace" / "benchmark"
        benchmark.mkdir(parents=True)
        np.savez_compressed(benchmark / "chunk.npz", before_equity=np.array([1., 1.01]))
        np.savez_compressed(directory / "chunk.npz", returns=np.array([.01, .02], np.float32))
    write_json(previous / "final_audit.json", audit)
    monkeypatch.setattr(study.shared, "compare_reproduction", lambda *args: dict(reused_as_sample=False))
    proof = study.compare_control(current, previous, audit)
    assert proof["benchmark_native_arrays_equal"] and proof["rollout_buffers_equal"]
    if changed_component == "benchmark":
        np.savez_compressed(current / "training_trace" / "benchmark" / "chunk.npz", before_equity=np.array([1., 1.02]))
    else:
        np.savez_compressed(current / "chunk.npz", returns=np.array([.01, .03], np.float32))
    with pytest.raises(ValueError):
        study.compare_control(current, previous, audit)
