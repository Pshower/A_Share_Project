"""Offline PIT calendar, financial and shareholder snapshot tests."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data.pit import capture_calendar, capture_fundamentals, report_periods, statutory_deadline
from src.data.snapshots import Snapshots
from src.web.app import create_app


class FakeCalendarProvider:
    name = "sina"

    def trade_dates(self):
        return pd.DataFrame(dict(trade_date=pd.to_datetime(
            ["2020-01-02", "2020-01-03", "2020-01-06"])))


class FakePitProvider:
    name = "eastmoney"

    def __init__(self, fail_kind=None):
        self.fail_kind = fail_kind

    def financials(self, period):
        if self.fail_kind == "financials":
            raise ValueError("fixture financial schema changed")
        return pd.DataFrame(dict(
            **{"股票代码": ["000001", "600000"], "最新公告日期": ["2020-04-30", "2020-04-29"],
               "营业收入-同比增长": [10.0, 20.0], "净利润-同比增长": [5.0, 6.0],
               "净资产收益率": [10.0, 11.0]}))

    def forecasts(self, period):
        if self.fail_kind == "forecasts":
            raise ValueError("fixture forecast schema changed")
        return pd.DataFrame(dict(**{"股票代码": ["000001", "000001"], "公告日期": ["2020-04-15", "2020-04-15"],
                                    "预测指标": ["净利润", "营业收入"], "业绩变动": ["预增", "预增"]}))

    def shareholders(self, period):
        if self.fail_kind == "shareholders":
            raise ValueError("fixture holder schema changed")
        return pd.DataFrame(dict(**{"代码": ["000001", "600000"], "公告日期": ["2020-04-30", "2020-04-29"],
                                    "股东户数-本次": [100, 200], "股东户数-上次": [120, 190]}))


def test_calendar_snapshot_requires_consent_and_is_sorted(tmp_path):
    provider = FakeCalendarProvider()
    with pytest.raises(ValueError, match="consent"):
        capture_calendar(root=tmp_path, provider=provider)
    snapshot = capture_calendar(allow_network=True, root=tmp_path, provider=provider)
    assert snapshot["status"] == "ready" and snapshot["rows"] == 3
    assert snapshot["first_date"] == "2020-01-02" and snapshot["last_date"] == "2020-01-06"
    saved, directory = Snapshots(tmp_path).get("calendar", snapshot["id"])
    frame = pd.read_parquet(directory / "calendar.parquet")
    assert frame.trade_date.tolist() == ["2020-01-02", "2020-01-03", "2020-01-06"]
    assert saved["model_ready"] is False


def test_quarterly_financials_and_holders_filter_codes(tmp_path):
    provider = FakePitProvider()
    with pytest.raises(ValueError, match="consent"):
        capture_fundamentals(["000001"], "2020-01-01", "2020-12-31", root=tmp_path, provider=provider)
    snapshot = capture_fundamentals(["000001"], "2020-01-01", "2020-12-31",
                                    allow_network=True, root=tmp_path, provider=provider)
    assert snapshot["status"] == "ready"
    assert [entry["period"] for entry in snapshot["entries"]] == ["20200331", "20200630", "20200930", "20201231"]
    assert len(snapshot["entries"]) == 4
    assert snapshot["combined"]["financials"]["rows"] == 4
    assert snapshot["combined"]["forecasts"]["rows"] == 8
    assert snapshot["combined"]["shareholders"]["rows"] == 4
    saved, directory = Snapshots(tmp_path).get("fundamentals", snapshot["id"])
    assert saved["model_ready"] is False
    assert "announcement_date" in saved["pit_policy"]
    financials = pd.read_parquet(directory / "financials.parquet")
    holders = pd.read_parquet(directory / "shareholders.parquet")
    assert financials.code.tolist() == ["000001"] * 4
    assert financials.announcement_date.tolist() == ["2020-04-30"] * 4
    assert holders.code.tolist() == ["000001"] * 4
    assert saved["request"]["periods"] == ["20200331", "20200630", "20200930", "20201231"]


def test_financials_use_the_statutory_deadline_while_observed_dates_are_used_directly(tmp_path):
    provider = FakePitProvider()
    snapshot = capture_fundamentals(["000001"], "2020-01-01", "2020-12-31",
                                    allow_network=True, root=tmp_path, provider=provider)
    _, directory = Snapshots(tmp_path).get("fundamentals", snapshot["id"])
    financials = pd.read_parquet(directory / "financials.parquet")
    forecasts = pd.read_parquet(directory / "forecasts.parquet")
    # stock_yjbb_em only exposes the latest revision; the availability date must not inherit it.
    assert financials.availability_basis.tolist() == ["statutory_deadline"] * 4
    assert financials.available_from.tolist() == ["2020-04-30", "2020-08-31", "2020-10-31", "2021-04-30"]
    assert (pd.to_datetime(financials.available_from) >= pd.to_datetime(financials.period)).all()
    assert forecasts.availability_basis.tolist() == ["vendor_announcement"] * 8
    assert (forecasts.available_from == forecasts.announcement_date).all()
    assert snapshot["availability_policy"]["financials"] == "statutory_deadline"
    assert snapshot["combined"]["financials"]["lag_days"]["available_from"]["over_365_days"] == 0
    assert snapshot["combined"]["financials"]["lag_days"]["announcement_date"]["over_365_days"] == 0


def test_statutory_deadlines_cover_every_quarter_end_and_move_the_annual_report_to_the_next_year():
    assert statutory_deadline("20200331") == pd.Timestamp("2020-04-30")
    assert statutory_deadline("20200630") == pd.Timestamp("2020-08-31")
    assert statutory_deadline("20200930") == pd.Timestamp("2020-10-31")
    assert statutory_deadline("20201231") == pd.Timestamp("2021-04-30")
    with pytest.raises(ValueError):
        statutory_deadline("20200229")


def test_availability_survives_a_filter_that_leaves_a_non_contiguous_index(tmp_path):
    # 000001 is first in the provider table, so keeping only 600000 leaves index label 1.
    snapshot = capture_fundamentals(["600000"], "2020-01-01", "2020-03-31",
                                    allow_network=True, root=tmp_path, provider=FakePitProvider())
    _, directory = Snapshots(tmp_path).get("fundamentals", snapshot["id"])
    financials = pd.read_parquet(directory / "financials.parquet")
    assert financials.code.tolist() == ["600000"]
    assert financials.available_from.tolist() == ["2020-04-30"]
    assert financials.available_from.notna().all()
    assert snapshot["entries"][0]["financials"]["first_available"] == "2020-04-30"


def test_optional_failure_is_partial_and_bad_schema_is_recorded(tmp_path):
    provider = FakePitProvider(fail_kind="forecasts")
    snapshot = capture_fundamentals(["000001"], "2020-01-01", "2020-06-30",
                                    allow_network=True, root=tmp_path, provider=provider)
    assert snapshot["status"] == "partial"
    assert snapshot["entries"][0]["financials"]["status"] == "ready"
    assert snapshot["entries"][0]["forecasts"]["status"] == "failed"
    assert snapshot["combined"]["forecasts"]["rows"] == 0
    assert any(error["kind"] == "forecasts" for error in snapshot["errors"])

    duplicate = FakePitProvider()
    duplicate.financials = lambda period: pd.DataFrame(dict(
        **{"股票代码": ["000001", "000001"], "最新公告日期": ["2020-04-30", "2020-04-30"]}))
    bad = capture_fundamentals(["000001"], "2020-01-01", "2020-03-31",
                               allow_network=True, root=tmp_path, provider=duplicate)
    assert bad["status"] == "partial"
    assert bad["entries"][0]["financials"]["status"] == "failed"


@pytest.mark.local_ipc
def test_report_periods_and_calendar_api_are_read_only(tmp_path):
    assert report_periods("2020-01-01", "2020-12-31") == ["20200331", "20200630", "20200930", "20201231"]
    snapshot = capture_calendar(allow_network=True, root=tmp_path, provider=FakeCalendarProvider())
    with TestClient(create_app(tmp_path, start_workers=False)) as client:
        assert client.get("/api/market/snapshots/calendar").json()[0]["id"] == snapshot["id"]
        assert client.get(f"/api/market/snapshots/calendar/{snapshot['id']}").json()["kind"] == "calendar"
        assert client.get("/api/jobs").json() == []
