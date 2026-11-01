"""Offline macro snapshot and as-of availability tests."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data.macro import capture_macro, parse_period
from src.data.snapshots import Snapshots
from src.web.app import create_app


class FakeMacroProvider:
    def __init__(self, fail_api=None):
        self.fail_api = fail_api

    def fetch(self, api):
        if api == self.fail_api:
            raise ValueError("fixture macro schema changed")
        if api == "macro_china_pmi":
            return pd.DataFrame(dict(**{
                "月份": ["2026年10月份", "2026年09月份"],
                "制造业-指数": [50.1, 49.8],
                "非制造业-指数": [51.2, 50.9],
            }))
        if api == "macro_china_gdp":
            return pd.DataFrame(dict(**{
                "季度": ["2026年第1-3季度"],
                "国内生产总值-绝对值": [1071071.2],
                "国内生产总值-同比增长": [4.6],
            }))
        if api == "macro_china_lpr":
            return pd.DataFrame(dict(**{
                "TRADE_DATE": ["2026-09-20", "2026-10-20"],
                "LPR1Y": [3.0, 3.0], "LPR5Y": [3.5, 3.5],
            }))
        if api == "currency_boc_safe":
            return pd.DataFrame(dict(**{"日期": ["2026-10-30"], "美元": [671.38]}))
        raise ValueError("unexpected macro api")


def test_parse_macro_periods():
    assert parse_period("2026年09月份") == pd.Timestamp("2026-09-30")
    assert parse_period("2026年10月份") == pd.Timestamp("2026-10-31")
    assert parse_period("2026年第1-3季度") == pd.Timestamp("2026-09-30")
    assert parse_period("202601") == pd.Timestamp("2026-01-31")
    assert parse_period("2026-09-20") == pd.Timestamp("2026-09-20")


def test_macro_snapshot_uses_conservative_availability(tmp_path):
    provider = FakeMacroProvider()
    with pytest.raises(ValueError, match="consent"):
        capture_macro("2026-01-01", "2026-10-31", root=tmp_path, provider=provider, series=["pmi"])
    snapshot = capture_macro("2026-01-01", "2026-10-31", allow_network=True, root=tmp_path,
                             provider=provider, series=["pmi", "gdp", "lpr"])
    assert snapshot["status"] == "ready"
    assert snapshot["coverage"]["rows"] == 10 and snapshot["coverage"]["series"] == 6
    saved, directory = Snapshots(tmp_path).get("macro", snapshot["id"])
    assert saved["model_ready"] is False
    assert "not an official release calendar" in saved["release_policy"]
    pmi = pd.read_parquet(directory / "series/pmi.parquet")
    assert pmi.period.tolist() == ["2026-09-30", "2026-10-31"]
    assert pmi.available_from.tolist() == ["2026-10-01", "2026-11-01"]
    gdp = pd.read_parquet(directory / "series/gdp.parquet")
    assert gdp.available_from.iloc[0] == "2026-10-20"
    long = pd.read_parquet(directory / "macro.parquet")
    assert set(long.series_id) == {"pmi_manufacturing", "pmi_non_manufacturing", "gdp_level",
                                   "gdp_yoy", "lpr_1y", "lpr_5y"}
    assert long.set_index("series_id").loc["lpr_1y", "value"].tolist() == [3.0, 3.0]


def test_safe_fx_is_converted_to_cny_per_usd(tmp_path):
    snapshot = capture_macro("2026-01-01", "2026-10-31", allow_network=True, root=tmp_path,
                             provider=FakeMacroProvider(), series=["usdcny"])
    _, directory = Snapshots(tmp_path).get("macro", snapshot["id"])
    wide = pd.read_parquet(directory / "series/usdcny.parquet")
    long = pd.read_parquet(directory / "macro.parquet")
    assert wide["美元"].iloc[0] == pytest.approx(6.7138)
    assert long.value.iloc[0] == pytest.approx(6.7138)


@pytest.mark.local_ipc
def test_macro_partial_failure_and_read_only_api(tmp_path):
    snapshot = capture_macro("2026-01-01", "2026-10-31", allow_network=True, root=tmp_path,
                             provider=FakeMacroProvider(fail_api="macro_china_pmi"), series=["pmi", "gdp"])
    assert snapshot["status"] == "partial"
    assert snapshot["entries"][0]["status"] == "failed"
    assert snapshot["entries"][1]["status"] == "ready"
    assert snapshot["coverage"]["series"] == 2
    with TestClient(create_app(tmp_path, start_workers=False)) as client:
        assert client.get("/api/market/snapshots/macro").json()[0]["id"] == snapshot["id"]
        assert client.get(f"/api/market/snapshots/macro/{snapshot['id']}").json()["kind"] == "macro"
        catalog = client.get(f"/api/market/macro/{snapshot['id']}/catalog")
        assert catalog.status_code == 200
        assert any(row["series_id"] == "gdp_yoy" for row in catalog.json())
        rows = client.get(f"/api/market/macro/{snapshot['id']}/series/gdp_yoy")
        assert rows.status_code == 200 and rows.json()["rows"][0]["value"] == 4.6
        assert client.get(f"/api/market/macro/{snapshot['id']}/series/unknown").status_code == 409
        assert client.get("/api/jobs").json() == []
    with pytest.raises(ValueError, match="Unknown"):
        capture_macro(allow_network=True, root=tmp_path, provider=FakeMacroProvider(), series=["unknown"])
