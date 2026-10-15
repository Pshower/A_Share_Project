"""Explicit engineering-only CLI context for the immutable PhaseBudget core."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path, PurePosixPath
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backtest.run import git_metadata
from src.data.preprocess import file_hash, read_config, write_json
from tools import research_timing_phase_budget as core
from tools import audit_timing_phase_budget as auditor
from tools import analyze_timing_phase_budget as analyzer
from tools import seal_timing_phase_budget as sealer

CORE_SHA256 = "9c54115cff7a88e4b0c668fff073d7aaaa3c3336c6e40ae5cc80b42c3db7826a"
ORIGINAL_PROTOCOL_SHA256 = "f8fba8ec1616f5f23dfc15ba18153ccacce3bc108c772f0d019a7f1a10e717c3"
FAILURE_SHA256 = "8b2c1948e89094ab22267c8dff96e2caef27a8fd2cd89815b5e70e8f3eb39b2b"
ORIGINAL_DIRECTORY = "reports/runs/ppo_timing_phase_budget_20261015"
STUDY_DIRECTORY = "reports/runs/ppo_timing_phase_state_repair_20261015"
FAILURE_FILE = ORIGINAL_DIRECTORY + "/engineering_failure_2023.json"
DOCUMENT_FILE = "doc/ppo_timing_phase_state_repair_protocol_20261015.md"
TEST_FILES = ("tests/test_timing_phase_state_repair.py", "tests/test_timing_phase_state_repair_review.py")
AUXILIARY_FILES = ("tools/audit_timing_phase_budget.py", "tests/test_timing_phase_budget_audit.py",
    "tools/analyze_timing_phase_budget.py", "tests/test_analyze_timing_phase_budget.py",
    "tools/seal_timing_phase_budget.py", "tests/test_seal_timing_phase_budget.py")
DRIVER_FILE = "tools/research_timing_phase_budget_repaired.py"
REPAIR_FILES = (DRIVER_FILE, *TEST_FILES, DOCUMENT_FILE, *AUXILIARY_FILES)
CODEC_VERSION = "typed_nonfinite_scalar_bits_v2"
_CORE_CHECK = core.check_frozen
_CORE_STATE = core._json_state
_ACTIVE = False


def require(condition, message):
    if not condition:
        raise ValueError(message)


def workspace_path(name):
    require(isinstance(name, str), "Repair inventory path must be text")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and relative.as_posix() == name and "\\" not in name and ":" not in name
            and ".." not in relative.parts, "Repair inventory path must be canonical workspace-relative POSIX")
    actual = (ROOT / name).resolve()
    require(actual.is_relative_to(ROOT.resolve()) and actual.is_file(), "Repair inventory file is missing or escaped workspace")
    return actual


def repair_json_state(value):
    """Preserve old finite/array representation; tag only nonfinite scalars."""
    if is_dataclass(value):
        return repair_json_state(asdict(value))
    if isinstance(value, dict):
        return {str(key): repair_json_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [repair_json_state(item) for item in value]
    if isinstance(value, np.floating) and not np.isfinite(value):
        return {"__nonfinite_scalar__": CODEC_VERSION, "type": "numpy", "dtype": value.dtype.str,
                "bits": value.tobytes().hex()}
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite_scalar__": CODEC_VERSION, "type": "python", "dtype": ">f8",
                "bits": struct.pack(">d", value).hex()}
    return _CORE_STATE(value)


def failure_evidence():
    path = workspace_path(FAILURE_FILE)
    require(file_hash(path) == FAILURE_SHA256, "Failure evidence does not match its immutable hard pin")
    failure = read_config(path)
    require(failure.get("status") == "engineering_failure_preserved" and failure.get("scientific_result_complete") is False
            and failure.get("original_root") == ORIGINAL_DIRECTORY and failure.get("failed_run") == "screen_2023_linear_42"
            and failure.get("original_protocol_sha256") == ORIGINAL_PROTOCOL_SHA256
            and failure.get("source_and_runtime_modified") is False and failure.get("model_or_optimizer_replayed") is False
            and type(failure.get("extra_environment_steps")) is int and failure["extra_environment_steps"] == 0,
            "Failure manifest changed scientific or recovery claims")
    require(file_hash(workspace_path(ORIGINAL_DIRECTORY + "/protocol.json")) == ORIGINAL_PROTOCOL_SHA256,
            "Original failed-study protocol changed")
    files = failure["signed_files_sha256"]
    require(type(failure["preserved_file_count"]) is int and failure["preserved_file_count"] == len(files) == 318,
            "Failure manifest must preserve exactly its stable 318 files")
    prefix = ORIGINAL_DIRECTORY + "/runs/screen_2023_linear_42/"
    for name, digest in files.items():
        require(name.startswith(prefix) and file_hash(workspace_path(name)) == digest,
                "A preserved failed-run input changed or includes a live different run")
    return failure


def engineering_registration(root):
    root = Path(root).resolve()
    require(root == (ROOT / STUDY_DIRECTORY).resolve(), "Repair uses only its separately registered new root")
    failure_evidence()
    require(file_hash(Path(core.__file__)) == CORE_SHA256, "Immutable scientific core changed")
    return dict(version=CODEC_VERSION, output_root=str(root), core_runner_sha256=CORE_SHA256,
        original_protocol_sha256=ORIGINAL_PROTOCOL_SHA256, failure_manifest_path=FAILURE_FILE,
        failure_manifest_sha256=FAILURE_SHA256, failure_preserved_files=318,
        inventory_sha256={name: file_hash(workspace_path(name)) for name in REPAIR_FILES},
        modification_scope="State fingerprint scalar encoding and explicit registered CLI guard only",
        gradient_replay=False, reused_original_runs=False)


def validate_repair(protocol):
    require(isinstance(protocol.get("engineering_repair"), dict), "Explicit repair registration is required")
    spec = protocol["engineering_repair"]
    require(isinstance(spec.get("output_root"), str), "Repair output-root registration must be text")
    expected = engineering_registration(spec.get("output_root", ""))
    require(core.same_json(spec, expected) and protocol.get("tool_sha256") == expected["inventory_sha256"][DRIVER_FILE],
            "Repair driver, tests, auxiliary tools, addendum or failed-input registration changed")
    return spec


def check_frozen(protocol):
    # The original checker must still verify every original scientific input.
    view = deepcopy(protocol)
    view["tool_sha256"] = CORE_SHA256
    _CORE_CHECK(view)
    validate_repair(protocol)


@contextmanager
def repair_context(protocol):
    """Install only in an explicit single CLI context and always restore refs."""
    global _ACTIVE
    require(not _ACTIVE, "Repair context is not reentrant or concurrent")
    original = (core._json_state, core.check_frozen, sealer.STUDY_DIRECTORY, sealer.CODE_FILES)
    require(original[0] is _CORE_STATE and original[1] is _CORE_CHECK, "Scientific core was already patched outside the repair context")
    _ACTIVE = True
    try:
        check_frozen(protocol)
        failure = failure_evidence()
        core._json_state, core.check_frozen = repair_json_state, check_frozen
        sealer.STUDY_DIRECTORY = STUDY_DIRECTORY
        sealer.CODE_FILES = tuple(dict.fromkeys((*original[3], *REPAIR_FILES, FAILURE_FILE,
            ORIGINAL_DIRECTORY + "/protocol.json", *failure["signed_files_sha256"])))
        yield
    finally:
        core._json_state, core.check_frozen, sealer.STUDY_DIRECTORY, sealer.CODE_FILES = original
        _ACTIVE = False


def prepare(root, reference_root=core.REFERENCE):
    root, reference_root = Path(root).resolve(), Path(reference_root).resolve()
    require(root == (ROOT / STUDY_DIRECTORY).resolve(), "Do not overwrite or overlap the original experiment")
    repair = engineering_registration(root)
    previous = core.reference_protocol(reference_root, full=True)
    require(not root.is_relative_to(reference_root) and not reference_root.is_relative_to(root), "Repair root overlaps immutable predecessor")
    if (root / "protocol.json").exists():
        saved = read_config(root / "protocol.json")
        require(saved["reference_root"] == str(reference_root), "Repair registration cannot replace predecessor")
        check_frozen(saved)
        return saved
    require(not root.exists(), "Incomplete repair output requires review")
    protocol = dict(schema_version=1, **core.fixed_registration(), folds=deepcopy(previous["folds"]),
        base_config=deepcopy(previous["base_config"]), gate=deepcopy(previous["gate"]), reference_root=str(reference_root),
        reference_protocol_sha256=file_hash(reference_root / "protocol.json"), reference_final_audit_sha256=core.REFERENCE_AUDIT_SHA256,
        baseline_reference_root=previous["reproduction_reference_root"], source_hashes=core.code_identity()["source_hashes"],
        tool_sha256=repair["inventory_sha256"][DRIVER_FILE], document_sha256=file_hash(core.DOCUMENT), git=git_metadata(),
        component_hashes={name: file_hash(ROOT / name) for name in core.COMPONENT_FILES},
        helper_hashes={name: file_hash(ROOT / name) for name in core.HELPER_FILES}, test_hashes={name: file_hash(ROOT / name) for name in core.TEST_FILES},
        engineering_repair=repair, test_evaluated=False, actionable=False)
    check_frozen(protocol)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "protocol.json", protocol)
    write_json(root / "status.json", dict(status="prepared", test_evaluated=False, actionable=False))
    return protocol


def seal_repaired(root):
    require(_ACTIVE, "Repair seal requires the explicit registered context")
    root = Path(root).resolve()
    protocol = read_config(root / "protocol.json")
    check_frozen(protocol)
    result = sealer.build_seal(root)
    result.update(study="timing_phase_state_repair_20261015", engineering_repair=deepcopy(protocol["engineering_repair"]))
    path = root / "final_audit.json"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    sealer.verify_files(result["signed_files_sha256"])
    print(json.dumps(dict(status="passed", final_audit=str(path), sha256=file_hash(path),
        signed_files=len(result["signed_files_sha256"]), counters=result["counters"]), sort_keys=True))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "screen", "confirm", "report", "audit", "analyze", "seal"), required=True)
    parser.add_argument("--output", default=str(ROOT / STUDY_DIRECTORY))
    parser.add_argument("--reference-root", default=str(core.REFERENCE))
    parser.add_argument("--fold", choices=(*core.original.SCREEN_FOLDS, *core.original.CONFIRM_FOLDS))
    parser.add_argument("--sealed", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--analysis-stage", choices=("screen", "confirm", "all"), default="all")
    args = parser.parse_args(argv)
    root = Path(args.output).resolve()
    require(args.fold is None or args.stage in ("screen", "confirm"), "Fold belongs only to a physical training worker")
    require(not (args.sealed or args.final) or args.stage == "audit", "Audit flags belong only to audit stage")
    require(not args.final or args.sealed, "Final audit requires sealed physical evidence")
    require(args.stage == "prepare" or (root / "protocol.json").exists(), "Only explicit prepare may create a repair registration")
    protocol = prepare(root, args.reference_root)
    if args.stage == "prepare":
        return protocol
    with repair_context(protocol):
        if args.stage in ("screen", "confirm", "report"):
            forwarded = ["--stage", args.stage, "--output", str(root), "--reference-root", args.reference_root]
            if args.fold is not None:
                forwarded += ["--fold", args.fold]
            return core.main(forwarded)
        if args.stage == "audit":
            forwarded = ["--root", str(root)] + (["--sealed"] if args.sealed else []) + (["--final"] if args.final else [])
            return auditor.main(forwarded)
        if args.stage == "analyze":
            return analyzer.main(["--root", str(root), "--stage", args.analysis_stage])
        return seal_repaired(root)


if __name__ == "__main__":
    main()
