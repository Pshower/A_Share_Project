import pytest

from src.data.preprocess import write_json
from src.training.summarize_sparse import summarize


@pytest.mark.parametrize("wrong_seed", [False, True])
def test_incomplete_or_wrong_trial_cannot_qualify(tmp_path, wrong_seed):
    write_json(tmp_path / "protocol.json", dict(arms={k: {} for k in ["D0", "D1", "S1", "R1"]}, maximum_steps=73728))
    write_json(tmp_path / "selection.json", dict(selected="R1"))
    rows = [dict(phase="screen", arm=arm, seed=42, steps=4096) for arm in ["D0", "D1", "S1", "R1"]]
    if wrong_seed:
        rows += [dict(phase="confirm", arm="R1", seed=s, steps=8192) for s in [42, 123, 999]]
        rows += [dict(phase="walk", arm=arm, seed=s, steps=8192) for arm in ["D0", "R1"] for s in [42, 123]]
    write_json(tmp_path / "summary.json", rows)
    with pytest.raises(ValueError, match="incomplete"):
        summarize(tmp_path)
    assert not (tmp_path / "qualification.json").exists()
