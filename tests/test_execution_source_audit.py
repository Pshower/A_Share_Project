import pandas as pd
import pytest

from src.data.audit_execution_sources import audit, read_source


def source(path, dates, prices):
    pd.DataFrame(dict(date=dates, code="000001", open=prices, close=prices, high=prices,
                      low=prices, volume=100, amount=1000)).to_csv(path, index=False)


def test_pair_is_candidate_not_execution_certification(tmp_path):
    source(tmp_path / "000001_data_2020_2021.csv", ["2020-01-01", "2020-01-02"], [10, 11])
    source(tmp_path / "000001_data_2020_2021_hfq.csv", ["2020-01-01", "2020-01-02"], [100, 110])
    report = audit(tmp_path)
    assert report["paired_candidates"] == 1 and not report["real_execution_ready"]
    assert report["rows"][0]["close_ratio_min"] == 10
    source(tmp_path / "000001_data_2020_2021.csv", ["2020-01-01"], [10])
    assert audit(tmp_path)["paired_candidates"] == 0


def test_duplicate_or_missing_basis_is_not_selected_silently(tmp_path):
    source(tmp_path / "000001_data_2020_2021_hfq.csv", ["2020-01-01"], [100])
    source(tmp_path / "000001_data_2020_2022_hfq.csv", ["2020-01-01"], [100])
    report = audit(tmp_path)
    assert report["paired_candidates"] == 0
    assert len(report["rows"][0]["reasons"]) == 2


def test_bad_identity_is_rejected(tmp_path):
    path = tmp_path / "test.csv"
    source(path, ["2020-01-01", "2020-01-01"], [10, 10])
    with pytest.raises(ValueError, match="identity"):
        read_source(path, "000001")
