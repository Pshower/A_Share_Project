"""No real market data, labels, models or fitting in scheduling tests."""

from types import SimpleNamespace

import pytest

from tools import run_counterfactual_holdout as holdout


@pytest.mark.parametrize("ready", [False, True])
def test_holdout_uses_one_existing_segment_only_after_control(tmp_path, monkeypatch, ready):
    study = holdout.study
    root = tmp_path / "study"
    root.mkdir()
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "OUTPUT", root)
    protocol = {"data": {"2023": {"fixed": True}}}
    study.write_json(root / "protocol.json", protocol)
    monkeypatch.setattr(study, "check_frozen", lambda *args: None)
    monkeypatch.setattr(study, "config_for", lambda *args: {"unchanged": True})
    monkeypatch.setattr(study, "CausalTimingData", lambda *args: SimpleNamespace(identity={"fixed": True}))
    segment = root / "teacher/2023/inner_holdout"
    calls = []

    def closed(path, _):
        if path == segment:
            return {"counts": study.expected_teacher_counts()}
        return {"physical_rl_steps": 32768} if ready else None

    def teacher(actual_root, actual_protocol, data, fold, part):
        calls.append((actual_root, actual_protocol, fold, part))
        segment.mkdir(parents=True)
        study.write_json(segment / "completed.json", {"same_registered_segment": True})
        return segment

    monkeypatch.setattr(study, "closed", closed)
    monkeypatch.setattr(study, "teacher_segment", teacher)
    if not ready:
        with pytest.raises(ValueError, match="Finish"):
            holdout.run("2023", root)
        assert calls == []
    else:
        holdout.run("2023", root)
        assert calls == [(root, protocol, "2023", "inner_holdout")]
        assert study.read_json(root / "early_holdout_2023.json")["extra_steps_beyond_registered_segment"] == 0


@pytest.mark.parametrize("fold", ["2025", "2026", "2022"])
def test_holdout_never_opens_confirmation_or_unregistered_fold(fold):
    with pytest.raises(ValueError, match="development"):
        holdout.run(fold)
