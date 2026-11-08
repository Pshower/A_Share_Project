"""Pure engineering codec and fake-registration tests; no financial execution."""

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from src.training.freeze import json_digest
from tools.archive import research_timing_phase_budget_repaired as tool


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def references():
    return tool.core._json_state, tool.core.check_frozen, tool.sealer.STUDY_DIRECTORY, tool.sealer.CODE_FILES


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), np.float16("nan"), np.float32("nan"), np.float64("inf")])
def test_codec_preserves_nonfinite_type_and_exact_ieee_bits(value):
    result = tool.repair_json_state(value)
    json.dumps(result, allow_nan=False)
    assert result["bits"] == np.asarray(value, dtype=np.dtype(result["dtype"])).tobytes().hex()
    assert result["type"] == ("numpy" if isinstance(value, np.floating) else "python")


def test_finite_nested_dataclass_array_semantics_and_diagnostic_nan_are_separate():
    @dataclass
    class State:
        price: float
        fields: dict
    array = np.array([np.nan, -0., np.inf], np.float32)
    finite = State(1., {"array": array, "flag": True, "npint": np.int64(3), "none": None})
    assert tool.repair_json_state(finite) == tool._CORE_STATE(finite)
    rejected = State(float("nan"), {"filled_shares": 0, "array": array})
    encoded = tool.repair_json_state(rejected)
    json_digest(encoded)
    assert encoded["fields"]["filled_shares"] == 0 and np.isnan(rejected.price)
    assert encoded["fields"]["array"] == tool._CORE_STATE(array)
    assert tool.repair_json_state(None) != tool.repair_json_state(float("nan"))


def test_scalar_nan_payload_and_array_bit_mutations_change_fingerprint():
    a = np.array([0x7ff8000000000001], np.uint64).view(np.float64)[0]
    b = np.array([0x7ff8000000000002], np.uint64).view(np.float64)[0]
    assert json_digest(tool.repair_json_state(a)) != json_digest(tool.repair_json_state(b))
    assert tool.repair_json_state(a) != tool.repair_json_state(float(a))
    array = np.array([-0.], np.float32)
    before = tool.repair_json_state(array)
    array[0] = 0.
    assert before != tool.repair_json_state(array)


@pytest.fixture
def registration(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    original = tmp_path / tool.ORIGINAL_DIRECTORY
    write(original / "protocol.json", {"synthetic_original": True})
    original_sha = tool.file_hash(original / "protocol.json")
    monkeypatch.setattr(tool, "ORIGINAL_PROTOCOL_SHA256", original_sha)
    prefix = tool.ORIGINAL_DIRECTORY + "/runs/screen_2023_linear_42/"
    files = {}
    for index in range(318):
        name = prefix + f"file_{index:03d}.bin"
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preserved synthetic evidence")
        files[name] = tool.file_hash(path)
    failure = dict(status="engineering_failure_preserved", scientific_result_complete=False,
        original_root=tool.ORIGINAL_DIRECTORY, failed_run="screen_2023_linear_42", original_protocol_sha256=original_sha,
        source_and_runtime_modified=False, model_or_optimizer_replayed=False, extra_environment_steps=0,
        preserved_file_count=318, signed_files_sha256=files)
    write(tmp_path / tool.FAILURE_FILE, failure)
    monkeypatch.setattr(tool, "FAILURE_SHA256", tool.file_hash(tmp_path / tool.FAILURE_FILE))
    for name in tool.REPAIR_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic frozen input\n", encoding="utf-8")
    root = tmp_path / tool.STUDY_DIRECTORY
    registered = tool.engineering_registration(root)
    protocol = dict(tool_sha256=registered["inventory_sha256"][tool.DRIVER_FILE], engineering_repair=registered)
    return root, protocol, failure


def test_registration_binds_stable_failure_and_exact_auxiliary_inventory(registration):
    _root, protocol, _failure = registration
    assert tool.validate_repair(protocol) == protocol["engineering_repair"]
    assert set(protocol["engineering_repair"]["inventory_sha256"]) == set(tool.REPAIR_FILES)
    assert protocol["engineering_repair"]["gradient_replay"] is False
    assert protocol["engineering_repair"]["reused_original_runs"] is False


@pytest.mark.parametrize("change", ["driver", "test", "addon", "aux", "failed_leaf", "failure_manifest", "original_protocol", "missing_inventory", "typed_false"])
def test_changed_repair_or_failed_input_is_rejected_before_any_execution(registration, change):
    _root, protocol, failure = registration
    if change in ("driver", "test", "addon", "aux"):
        name = {"driver": tool.DRIVER_FILE, "test": tool.TEST_FILES[0], "addon": tool.DOCUMENT_FILE, "aux": tool.AUXILIARY_FILES[0]}[change]
        (tool.ROOT / name).write_text("changed", encoding="utf-8")
    elif change == "failed_leaf":
        (tool.ROOT / next(iter(failure["signed_files_sha256"]))).write_bytes(b"changed")
    elif change == "failure_manifest":
        (tool.ROOT / tool.FAILURE_FILE).write_text("{}", encoding="utf-8")
    elif change == "original_protocol":
        (tool.ROOT / tool.ORIGINAL_DIRECTORY / "protocol.json").write_text("{}", encoding="utf-8")
    elif change == "missing_inventory":
        protocol["engineering_repair"]["inventory_sha256"].pop(tool.AUXILIARY_FILES[0])
    else:
        protocol["engineering_repair"]["gradient_replay"] = 0
    with pytest.raises(ValueError):
        tool.validate_repair(protocol)


def test_only_readonly_core_view_is_checked_and_full_new_identity_survives(registration, monkeypatch):
    _root, protocol, _ = registration
    before = deepcopy(protocol)
    seen = []
    monkeypatch.setattr(tool, "_CORE_CHECK", lambda view: seen.append(deepcopy(view)))
    tool.check_frozen(protocol)
    assert seen[0]["tool_sha256"] == tool.CORE_SHA256
    assert seen[0]["engineering_repair"] == protocol["engineering_repair"]
    assert protocol == before and protocol["tool_sha256"] != tool.CORE_SHA256


@pytest.mark.parametrize("outcome", ["success", "body_error", "validation_error"])
def test_context_is_explicit_restores_all_references_and_never_reenters(monkeypatch, outcome):
    before = references()
    def check(_):
        if outcome == "validation_error":
            raise ValueError("synthetic early validation failure")
    monkeypatch.setattr(tool, "check_frozen", check)
    monkeypatch.setattr(tool, "failure_evidence", lambda: {"signed_files_sha256": {"synthetic/failed.bin": "0" * 64}})
    try:
        with tool.repair_context({}):
            assert tool.core._json_state is tool.repair_json_state
            assert tool.ORIGINAL_DIRECTORY + "/protocol.json" in tool.sealer.CODE_FILES
            with pytest.raises(ValueError, match="not reentrant"):
                with tool.repair_context({}):
                    pytest.fail("No nested body")
            if outcome == "body_error":
                raise RuntimeError("synthetic body error")
    except (RuntimeError, ValueError):
        assert outcome != "success"
    assert references() == before and tool._ACTIVE is False


@pytest.mark.parametrize("stage", ["screen", "report", "audit", "analyze", "seal"])
def test_cli_routes_explicit_context_without_models_or_financial_work(tmp_path, monkeypatch, stage):
    root = tmp_path / "synthetic_repair"
    write(root / "protocol.json", {})
    before = references()
    monkeypatch.setattr(tool, "prepare", lambda *_: {})
    monkeypatch.setattr(tool, "check_frozen", lambda _: None)
    monkeypatch.setattr(tool, "failure_evidence", lambda: {"signed_files_sha256": {}})
    seen = []
    def record(args):
        assert tool._ACTIVE and tool.core._json_state is tool.repair_json_state
        seen.append(args)
        return "synthetic dispatch"
    monkeypatch.setattr(tool.core, "main", record)
    monkeypatch.setattr(tool.auditor, "main", record)
    monkeypatch.setattr(tool.analyzer, "main", record)
    monkeypatch.setattr(tool, "seal_repaired", lambda value: record([str(value)]))
    assert tool.main(["--stage", stage, "--output", str(root)]) == "synthetic dispatch"
    assert seen and references() == before and tool._ACTIVE is False


def test_repair_seal_has_correct_label_before_first_write_and_refuses_overwrite(tmp_path, monkeypatch):
    root = tmp_path / "synthetic"
    protocol = dict(engineering_repair={"synthetic": True})
    write(root / "protocol.json", protocol)
    monkeypatch.setattr(tool, "_ACTIVE", True)
    monkeypatch.setattr(tool, "check_frozen", lambda _: None)
    monkeypatch.setattr(tool.sealer, "build_seal", lambda _: dict(status="passed", study="old_label", signed_files_sha256={}, counters={}))
    monkeypatch.setattr(tool.sealer, "verify_files", lambda _: None)
    result = tool.seal_repaired(root)
    assert result["study"] == "timing_phase_state_repair_20261015"
    assert result["engineering_repair"] == protocol["engineering_repair"]
    assert json.loads((root / "final_audit.json").read_text())["study"] == result["study"]
    with pytest.raises(FileExistsError):
        tool.seal_repaired(root)


def test_new_registration_root_and_invalid_audit_flags_reject_before_prepare(registration, monkeypatch):
    root, _, _ = registration
    with pytest.raises(ValueError, match="new root"):
        tool.engineering_registration(tool.ROOT / tool.ORIGINAL_DIRECTORY)
    monkeypatch.setattr(tool, "prepare", lambda *_: pytest.fail("Invalid flags must not prepare"))
    with pytest.raises(ValueError, match="requires sealed"):
        tool.main(["--stage", "audit", "--output", str(root), "--final"])
