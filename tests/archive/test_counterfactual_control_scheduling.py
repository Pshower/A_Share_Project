"""Scheduling contracts only: no market, model, fitting or real training."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.archive import run_counterfactual_control as control


def test_control_reuses_exact_registered_case_and_never_increases_budget(tmp_path, monkeypatch):
    study = control.study
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "OUTPUT", tmp_path / "study")
    root = study.OUTPUT
    root.mkdir()
    protocol = {"data": {"2023": {"fixed_scaler": "a"}}}
    study.write_json(root / "protocol.json", protocol)
    guards, calls = [], []
    monkeypatch.setattr(study, "check_frozen", lambda *args: guards.append(args))
    monkeypatch.setattr(study, "config_for", lambda *args: {"frozen": True})
    monkeypatch.setattr(study, "CausalTimingData", lambda config: SimpleNamespace(identity=protocol["data"]["2023"]))

    def trial(actual_root, actual_protocol, data, fold, method, seed, phase, budget):
        calls.append((actual_root, actual_protocol, fold, method, seed, phase, budget))
        directory = root / "runs/screen_2023_ppo_only_42"
        directory.mkdir(parents=True, exist_ok=True)
        study.write_json(directory / "completed.json", {"case": "exact"})
        return [{"steps": 8192}, {"steps": 32768}]

    monkeypatch.setattr(study, "trial", trial)
    control.run("2023", root)
    assert len(guards) == 2
    assert calls == [(root, protocol, "2023", "ppo_only", 42, "screen", 32768)]
    receipt = study.read_json(root / "early_control_2023.json")
    assert receipt["extra_rl_steps_beyond_registered_case"] == 0
    assert receipt["completed_sha256"] == study.file_hash(root / "runs/screen_2023_ppo_only_42/completed.json")


@pytest.mark.parametrize("fold", ["2025", "2026", "2022"])
def test_confirmation_and_unregistered_controls_cannot_start(fold):
    with pytest.raises(ValueError, match="development"):
        control.run(fold)


def test_control_source_is_only_a_frozen_function_call():
    text = Path(control.__file__).read_text(encoding="utf-8")
    assert ".learn(" not in text and ".reset(" not in text
    assert '"ppo_only", 42, "screen", 32768' in text
