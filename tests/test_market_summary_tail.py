"""Tail scheduling cannot bypass incomplete or altered prefix trials."""

from copy import deepcopy

import pytest

from tools import run_market_summary_tail as tail


@pytest.fixture
def prefix(tmp_path, monkeypatch):
    root = tmp_path / "study"
    protocol = dict(base_config={}, reference_root="reference")
    receipt = dict(status="complete", protocol_sha256="protocol-sha", script_sha256="script-sha",
        completed_runs=[dict(run_id=f"2024_{arm}_{seed}", completed_sha256=f"{arm}-{seed}")
                        for arm in ("mean", "attention") for seed in tail.SEEDS])
    hashes = {root / "worker_2024.json": "worker-sha", root / "protocol.json": "protocol-sha",
              tail.ROOT / "tools" / "run_market_summary_worker.py": "script-sha"}
    hashes.update({root / "runs" / item["run_id"] / "completed.json": item["completed_sha256"]
                   for item in receipt["completed_runs"]})
    calls = []
    monkeypatch.setattr(tail, "read_config", lambda path: deepcopy(receipt))
    monkeypatch.setattr(tail, "file_hash", lambda path: hashes[path])
    monkeypatch.setattr(tail, "config_for", lambda base, reference, output, fold, arm, seed:
                        dict(fold=fold, arm=arm, seed=seed))

    def cache(directory, config, expected_protocol):
        calls.append(config)
        return [dict(synthetic_only=True)]

    monkeypatch.setattr(tail, "cached_trial", cache)
    return root, protocol, receipt, calls


def test_tail_requires_all_six_strict_caches_before_any_training(prefix):
    root, protocol, receipt, calls = prefix
    assert tail.prerequisites(root, protocol) == "worker-sha"
    assert calls == [dict(fold="2024", arm=arm, seed=seed)
                     for arm in ("mean", "attention") for seed in tail.SEEDS]


@pytest.mark.parametrize("failure", ["running", "duplicate", "protocol", "script", "marker", "cache"])
def test_tail_rejects_partial_or_altered_worker_evidence(prefix, monkeypatch, failure):
    root, protocol, receipt, calls = prefix
    if failure == "running":
        receipt["status"] = "running"
    elif failure == "duplicate":
        receipt["completed_runs"][-1] = deepcopy(receipt["completed_runs"][0])
    elif failure == "protocol":
        receipt["protocol_sha256"] = "altered"
    elif failure == "script":
        receipt["script_sha256"] = "altered"
    elif failure == "marker":
        receipt["completed_runs"][0]["completed_sha256"] = "altered"
    else:
        monkeypatch.setattr(tail, "cached_trial", lambda *args: None)
    with pytest.raises(ValueError):
        tail.prerequisites(root, protocol)
