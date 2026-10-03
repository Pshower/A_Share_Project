"""Report initialization labels must agree with the registered matrix records."""

import pytest

from src.data.preprocess import write_json
from src.training import matrix_report
from tests.test_matrix_report import _records


def test_random_protocol_rejects_prior_records_and_legacy_prior_reports_still_work(tmp_path, monkeypatch):
    for helper in ["_matrix_plot", "_training_plot", "_experience_artifacts"]:
        monkeypatch.setattr(matrix_report, helper, lambda *args: None)
    protocol_path = tmp_path / "protocol.json"
    records_path = tmp_path / "summary.json"
    write_json(protocol_path, dict(initialization_mode="random", seeds=[42, 123], test_evaluated=False))

    for explicit_prior in [False, True]:
        records = _records()
        if explicit_prior:
            for record in records:
                record["initialization_mode"] = "prior"
        write_json(records_path, records)
        original = {path: path.read_bytes() for path in [protocol_path, records_path]}

        with pytest.raises(ValueError, match="initialization mode differs from registered records"):
            matrix_report.write_report(tmp_path)

        assert {path: path.read_bytes() for path in original} == original
        assert not (tmp_path / "report.md").exists()
        assert not (tmp_path / "summary.md").exists()
        assert not (tmp_path / "matrix_summary.json").exists()
        assert not (tmp_path / "metrics.csv").exists()

    write_json(protocol_path, dict(seeds=[42, 123], test_evaluated=False))
    write_json(records_path, _records())
    original = {path: path.read_bytes() for path in [protocol_path, records_path]}

    summary = matrix_report.write_report(tmp_path)

    assert summary["total_checkpoints"] == 8
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "summary.md").exists()
    assert {path: path.read_bytes() for path in original} == original
