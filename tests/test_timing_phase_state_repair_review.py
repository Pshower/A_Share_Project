"""Independent repair-codec/identity/context guards, using synthetic evidence."""

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_timing_phase_agent import fixture as phase_fixture
from tools import research_timing_phase_budget_repaired as repair
from tools.timing_phase_agent import PhaseTimingAgent


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def references():
    return (repair.core._json_state, repair.core.check_frozen,
            repair.sealer.STUDY_DIRECTORY, repair.sealer.CODE_FILES)


def permitted_context(monkeypatch):
    monkeypatch.setattr(repair, "check_frozen", lambda _protocol: None)
    monkeypatch.setattr(repair, "failure_evidence", lambda: {"signed_files_sha256": {"synthetic/failed.bin": "0" * 64}})


def fake_failure(monkeypatch):
    prefix = repair.ORIGINAL_DIRECTORY + "/runs/screen_2023_linear_42/"
    failure = dict(status="engineering_failure_preserved", scientific_result_complete=False,
        original_root=repair.ORIGINAL_DIRECTORY, failed_run="screen_2023_linear_42",
        original_protocol_sha256=repair.ORIGINAL_PROTOCOL_SHA256,
        source_and_runtime_modified=False, model_or_optimizer_replayed=False, extra_environment_steps=0,
        preserved_file_count=318, signed_files_sha256={prefix + f"leaf_{number:03d}.bin": "0" * 64 for number in range(318)})
    monkeypatch.setattr(repair, "workspace_path", lambda value: Path(value))

    def file_hash(path):
        name = Path(path).as_posix()
        if name == repair.FAILURE_FILE:
            return repair.FAILURE_SHA256
        if name == repair.ORIGINAL_DIRECTORY + "/protocol.json":
            return repair.ORIGINAL_PROTOCOL_SHA256
        return "0" * 64

    monkeypatch.setattr(repair, "file_hash", file_hash)
    monkeypatch.setattr(repair, "read_config", lambda _path: failure)
    return failure


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"),
    np.float16("nan"), np.float32("inf"), np.float64("-inf")])
def test_nonfinite_scalar_tag_round_trips_type_dtype_and_raw_bits(value):
    encoded = repair.repair_json_state(value)
    json.dumps(encoded, allow_nan=False)
    assert encoded["__nonfinite_scalar__"] == repair.CODEC_VERSION
    assert encoded["type"] == ("numpy" if isinstance(value, np.floating) else "python")
    dtype = np.dtype(encoded["dtype"])
    expected = np.asarray(value, dtype=dtype).tobytes().hex()
    assert encoded["bits"] == expected
    if isinstance(value, np.floating):
        assert encoded["dtype"] == value.dtype.str
    else:
        assert encoded["dtype"] == ">f8"


def test_nan_payload_dtype_python_numpy_and_none_are_not_collapsed():
    values = [np.array([0x7ff8000000000001], dtype=np.uint64).view(np.float64)[0],
              np.array([0x7ff8000000000002], dtype=np.uint64).view(np.float64)[0]]
    first, second = map(repair.repair_json_state, values)
    assert first != second
    assert repair.repair_json_state(float(values[0])) != first
    assert repair.repair_json_state(np.float32("nan")) != first
    assert repair.repair_json_state(None) != first


def test_finite_and_array_formats_remain_exact_and_array_mutation_changes_hash():
    array = np.array([np.nan, np.inf, -0.], dtype=np.float32)
    original = array.tobytes()
    data = dict(price=1.25, flags=(True, np.int64(3)), cash=None, values=array)
    assert repair.repair_json_state(data) == repair._CORE_STATE(data)
    assert array.tobytes() == original
    changed = array.copy()
    changed[2] = 0.
    assert repair.repair_json_state(changed) != repair.repair_json_state(array)
    assert repair.repair_json_state(array.astype(np.float64)) != repair.repair_json_state(array)


def test_mixed_history_encodes_without_replacing_missing_quote_or_zero_fill():
    history = [{"orders": [{"price": float("nan"), "filled_shares": 0, "code": "synthetic"}]}]
    result = repair.repair_json_state(history)
    assert result[0]["orders"][0]["filled_shares"] == 0
    assert result[0]["orders"][0]["code"] == "synthetic"
    assert np.isnan(history[0]["orders"][0]["price"])
    repair.core.json_digest(result)


def test_dataclass_nonfinite_encoding_does_not_depend_on_installation(monkeypatch):
    @dataclass
    class SyntheticRecord:
        price: float
        volume: int

    value = SyntheticRecord(float("nan"), 0)
    outside = repair.repair_json_state(value)
    permitted_context(monkeypatch)
    with repair.repair_context({}):
        inside = repair.repair_json_state(value)
        repair.core.json_digest(inside)
    assert outside == inside
    repair.core.json_digest(outside)


@pytest.mark.parametrize("raise_body", [False, True])
def test_explicit_context_restores_all_four_references_on_success_and_failure(monkeypatch, raise_body):
    before = references()
    permitted_context(monkeypatch)
    try:
        with repair.repair_context({}):
            assert repair.core._json_state is repair.repair_json_state
            assert repair.core.check_frozen is repair.check_frozen
            assert repair.sealer.STUDY_DIRECTORY == repair.STUDY_DIRECTORY
            assert set(repair.REPAIR_FILES).issubset(repair.sealer.CODE_FILES)
            assert "synthetic/failed.bin" in repair.sealer.CODE_FILES
            if raise_body:
                raise RuntimeError("Synthetic body failure")
    except RuntimeError:
        assert raise_body
    assert references() == before and repair._ACTIVE is False


def test_validation_failure_before_install_also_restores_active_flag_and_references(monkeypatch):
    before = references()
    monkeypatch.setattr(repair, "check_frozen", lambda _protocol: (_ for _ in ()).throw(ValueError("Guard sentinel")))
    with pytest.raises(ValueError, match="Guard sentinel"):
        with repair.repair_context({}):
            pytest.fail("No body before validation")
    assert references() == before and repair._ACTIVE is False


def test_nested_context_is_rejected_without_corrupting_outer_context(monkeypatch):
    before = references()
    permitted_context(monkeypatch)
    with repair.repair_context({}):
        installed = references()
        with pytest.raises(ValueError, match="not reentrant"):
            with repair.repair_context({}):
                pytest.fail("Nested body must not run")
        assert references() == installed and repair._ACTIVE is True
    assert references() == before and repair._ACTIVE is False


def test_strict_global_json_and_model_finite_checks_are_not_relaxed(monkeypatch, tmp_path):
    model, contract, env = phase_fixture()
    digest = repair.core.json_digest
    permitted_context(monkeypatch)
    try:
        with torch.no_grad():
            next(model.policy.parameters()).reshape(-1)[0] = float("inf")
        with repair.repair_context({}):
            assert repair.core.json_digest is digest
            with pytest.raises(ValueError):
                digest({"unsafe": float("nan")})
            with pytest.raises(FloatingPointError):
                PhaseTimingAgent(model, contract).save(tmp_path / "bad_model", {})
        assert not (tmp_path / "bad_model").exists()
    finally:
        env.close()


def test_core_view_is_only_a_read_only_validation_copy_before_full_repair_guard(monkeypatch):
    protocol = dict(tool_sha256="new_driver", engineering_repair={"sentinel": True}, nested={"source": "stable"})
    original, events = deepcopy(protocol), []

    def old_check(view):
        events.append("core")
        assert view is not protocol and view["tool_sha256"] == repair.CORE_SHA256
        assert view["engineering_repair"] == protocol["engineering_repair"]
        view["nested"]["source"] = "must_not_leak"

    def full_check(value):
        events.append("repair")
        assert value is protocol and value == original

    monkeypatch.setattr(repair, "_CORE_CHECK", old_check)
    monkeypatch.setattr(repair, "validate_repair", full_check)
    repair.check_frozen(protocol)
    assert protocol == original and events == ["core", "repair"]


def test_bare_core_checker_rejects_new_driver_sha_before_predecessor_or_provider(monkeypatch):
    identity = {"source_hashes": {"synthetic.py": "0" * 64}}
    protocol = dict(repair.core.fixed_registration(), source_hashes=identity["source_hashes"], tool_sha256="f" * 64,
        document_sha256="0" * 64, component_hashes={name: "0" * 64 for name in repair.core.COMPONENT_FILES},
        helper_hashes={name: "0" * 64 for name in repair.core.HELPER_FILES},
        test_hashes={name: "0" * 64 for name in repair.core.TEST_FILES},
        reference_final_audit_sha256=repair.core.REFERENCE_AUDIT_SHA256)
    monkeypatch.setattr(repair.core, "code_identity", lambda: identity)
    monkeypatch.setattr(repair.core, "file_hash", lambda path: repair.CORE_SHA256 if Path(path) == Path(repair.core.__file__) else "0" * 64)
    monkeypatch.setattr(repair.core, "reference_protocol", lambda *_args: pytest.fail("Bare core must reject the new driver first"))
    with pytest.raises(ValueError, match="source, runner, inventory"):
        repair.core.check_frozen(protocol)


@pytest.mark.parametrize("entry", ["cached_trial", "load_stage", "require_confirmation"])
def test_public_core_entry_inside_context_uses_enhanced_guard_before_any_provider(monkeypatch, tmp_path, entry):
    state = {"blocked": False}

    def original_check(view):
        assert view["tool_sha256"] == repair.CORE_SHA256
        if state["blocked"]:
            raise ValueError("Enhanced guard sentinel")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("No provider/model/file parse before enhanced guard")

    monkeypatch.setattr(repair, "_CORE_CHECK", original_check)
    monkeypatch.setattr(repair.core, "check_frozen", original_check)
    monkeypatch.setattr(repair, "validate_repair", lambda _protocol: None)
    monkeypatch.setattr(repair, "failure_evidence", lambda: {"signed_files_sha256": {"synthetic/failed.bin": "0" * 64}})
    monkeypatch.setattr(repair.core, "PhaseData", forbidden)
    monkeypatch.setattr(repair.core, "make_model", forbidden)
    monkeypatch.setattr(repair.core, "read_config", forbidden)
    protocol = {"tool_sha256": "new_driver"}
    before = references()
    with repair.repair_context(protocol):
        state["blocked"] = True
        with pytest.raises(ValueError, match="Enhanced guard sentinel"):
            if entry == "cached_trial":
                repair.core.cached_trial(tmp_path / "absent", {}, protocol)
            elif entry == "load_stage":
                repair.core.load_stage(tmp_path / "absent", protocol, "screen")
            else:
                repair.core.require_confirmation(tmp_path / "absent", protocol, ("linear", 8192), "0" * 64)
    assert references() == before and repair._ACTIVE is False


@pytest.mark.parametrize("change", ["failure_sha", "count", "live_other_run", "leaf_sha", "replay_claim"])
def test_failure_guard_checks_all_318_stable_files_and_never_borrows_live_fold(monkeypatch, change):
    failure = fake_failure(monkeypatch)
    assert repair.failure_evidence() is failure
    if change == "failure_sha":
        original = repair.file_hash
        monkeypatch.setattr(repair, "file_hash", lambda path: "f" * 64 if Path(path).as_posix() == repair.FAILURE_FILE else original(path))
    elif change == "count":
        failure["preserved_file_count"] = 317
    elif change == "live_other_run":
        leaf = next(iter(failure["signed_files_sha256"]))
        failure["signed_files_sha256"][leaf.replace("screen_2023_linear_42", "screen_2024_linear_42")] = failure["signed_files_sha256"].pop(leaf)
    elif change == "leaf_sha":
        failure["signed_files_sha256"][next(iter(failure["signed_files_sha256"]))] = "f" * 64
    else:
        failure["model_or_optimizer_replayed"] = True
    with pytest.raises(ValueError):
        repair.failure_evidence()


@pytest.mark.parametrize("change", ["missing_test", "core_sha", "typed_gradient_flag", "top_driver_sha"])
def test_full_addon_inventory_and_typed_claims_are_not_optional(monkeypatch, change):
    spec = dict(inventory_sha256={name: "0" * 64 for name in repair.REPAIR_FILES},
                output_root="synthetic", core_runner_sha256=repair.CORE_SHA256, gradient_replay=False)
    protocol = dict(tool_sha256="0" * 64, engineering_repair=deepcopy(spec))
    monkeypatch.setattr(repair, "engineering_registration", lambda _root: deepcopy(spec))
    assert repair.validate_repair(protocol) == spec
    if change == "missing_test":
        protocol["engineering_repair"]["inventory_sha256"].pop(repair.TEST_FILES[1])
    elif change == "core_sha":
        protocol["engineering_repair"]["core_runner_sha256"] = "f" * 64
    elif change == "typed_gradient_flag":
        protocol["engineering_repair"]["gradient_replay"] = 0
    else:
        protocol["tool_sha256"] = repair.CORE_SHA256
    with pytest.raises(ValueError):
        repair.validate_repair(protocol)


@pytest.mark.parametrize("suffix", ["", "/child", "/../"])
def test_wrong_or_overlapping_output_is_rejected_before_failure_inventory(monkeypatch, suffix):
    old = repair.ROOT / (repair.ORIGINAL_DIRECTORY + suffix)
    monkeypatch.setattr(repair, "failure_evidence", lambda: pytest.fail("No failure inventory or root creation for overlapping output"))
    with pytest.raises(ValueError, match="separately registered new root"):
        repair.engineering_registration(old)
