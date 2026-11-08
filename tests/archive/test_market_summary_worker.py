"""Worker scheduling uses sealed trial functions, not its own training loop."""

import sys

import pytest

from src.data.preprocess import file_hash, read_config, write_json
from tools.archive import run_market_summary_worker as worker
from tests.test_ppo_framework import no_parameter_updates


pytestmark = pytest.mark.usefixtures("no_parameter_updates")


@pytest.fixture
def setup_worker(tmp_path, monkeypatch):
    root = tmp_path / "study"
    root.mkdir()
    write_json(root / "protocol.json", dict(synthetic_only=True))
    calls, sleeps = [], []
    monkeypatch.setattr(sys, "argv", ["worker", "--root", str(root), "--reference-root", str(tmp_path / "reference")])
    monkeypatch.setattr(worker, "prepare", lambda output, reference: dict(synthetic_only=True))
    monkeypatch.setattr(worker, "rule_controls", lambda output, protocol, fold: calls.append(("rules", fold)))
    monkeypatch.setattr(worker.torch.cuda, "mem_get_info", lambda: (3 * 1024**3, 4 * 1024**3))
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: sleeps.append(seconds))

    def trial(output, protocol, fold, arm, seed):
        calls.append((fold, arm, seed))
        directory = output / "runs" / f"{fold}_{arm}_{seed}"
        directory.mkdir(parents=True)
        write_json(directory / "completed.json", dict(synthetic_only=True, seed=seed))

    monkeypatch.setattr(worker, "run_trial", trial)
    return root, calls, sleeps


def test_worker_owns_only_registered_2024_prefix_and_binds_markers(setup_worker):
    root, calls, sleeps = setup_worker
    worker.main()
    expected = [("2024", arm, seed) for arm in ("mean", "attention") for seed in worker.SEEDS]
    assert calls == [("rules", "2024"), *expected]
    receipt = read_config(root / "worker_2024.json")
    assert receipt["status"] == "complete"
    assert receipt["script_sha256"] == file_hash(worker.__file__)
    assert len(receipt["completed_runs"]) == 6
    for entry in receipt["completed_runs"]:
        assert entry["completed_sha256"] == file_hash(root / "runs" / entry["run_id"] / "completed.json")
    assert receipt["additional_trials_beyond_registration"] == 0
    assert not (root / "summary.json").exists()
    assert not (root / "status.json").exists()
    assert sleeps == []


def test_worker_waits_for_memory_before_starting_trial(setup_worker, monkeypatch):
    root, calls, sleeps = setup_worker
    checks = iter([1 * 1024**3] + [3 * 1024**3] * 6)
    monkeypatch.setattr(worker.torch.cuda, "mem_get_info", lambda: (next(checks), 4 * 1024**3))
    worker.main()
    assert sleeps == [10]
    assert len(calls) == 7
    assert read_config(root / "worker_2024.json")["status"] == "complete"


def test_partial_trial_failure_does_not_claim_completion(setup_worker, monkeypatch):
    root, calls, sleeps = setup_worker

    def incomplete(*args):
        raise ValueError("Incomplete registered trial")

    monkeypatch.setattr(worker, "run_trial", incomplete)
    with pytest.raises(ValueError, match="Incomplete"):
        worker.main()
    receipt = read_config(root / "worker_2024.json")
    assert receipt["status"] == "running"
    assert receipt["completed_runs"] == []
    assert not (root / "summary.json").exists()
