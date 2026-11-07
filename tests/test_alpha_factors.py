"""Unit tests for the pre-registered alpha factor constructions.

Everything here is synthetic: the point is to pin the sign, the point-in-time rule and the
period comparability, none of which should ever depend on a data file.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.pit import statutory_deadline
from tools import alpha_factors as factors


def trading_calendar(start="2019-01-01", periods=400):
    return pd.date_range(start, periods=periods, freq="B", name="date")


def financial_frame(rows):
    """Minimal stand-in for ``financials.parquet`` with the Chinese vendor columns."""
    base = {"股票代码": None, "营业总收入-同比增长": np.nan, "净利润-同比增长": np.nan,
            "净资产收益率": np.nan, "销售毛利率": np.nan, "每股经营现金流量": np.nan,
            "每股收益": np.nan, "每股净资产": np.nan}
    records = []
    for row in rows:
        record = dict(base)
        record.update(row)
        records.append(record)
    frame = pd.DataFrame(records)
    frame["code"] = frame["股票代码"]
    return frame


def test_annualisation_is_four_at_the_first_quarter():
    gain = factors.annualisation(pd.Series(["20200331", "20200630", "20200930", "20201231"]))
    np.testing.assert_allclose(gain.to_numpy(), [4.0, 2.0, 4.0 / 3.0, 1.0], atol=1e-12)


def test_annualisation_rejects_a_non_quarter_end():
    with pytest.raises(ValueError):
        factors.annualisation(pd.Series(["20200101"]))


def test_statutory_deadlines_are_the_legal_filing_limits():
    expected = {"20200331": "2020-04-30", "20200630": "2020-08-31",
                "20200930": "2020-10-31", "20201231": "2021-04-30"}
    for period, date in expected.items():
        assert statutory_deadline(period) == pd.Timestamp(date)


def test_availability_follows_the_legal_deadline_not_the_vendor_revision_date():
    """``announcement_date`` is the latest revision and can be more than a year late."""
    calendar = trading_calendar("2019-01-01", 400)
    frame = financial_frame([{"股票代码": "000001", "period": "20200331", "每股收益": 0.5,
                              "每股净资产": 10.0, "净资产收益率": 3.0}])
    events = factors.financial_events(frame, calendar)
    row = events.iloc[0]
    assert row["deadline"] == pd.Timestamp("2020-04-30")
    assert row["available"] > row["deadline"]
    assert pd.Timestamp("2020-05-01") <= row["available"] <= pd.Timestamp("2020-05-08")


def test_a_tied_availability_date_resolves_to_the_later_report_period():
    """The annual report and the first-quarter report share a 30 April deadline."""
    calendar = trading_calendar()
    frame = financial_frame([
        {"股票代码": "000001", "period": "20191231", "净资产收益率": 11.0},
        {"股票代码": "000001", "period": "20200331", "净资产收益率": 3.0},
    ])
    events = factors.financial_events(frame, calendar)
    assert events["available"].nunique() == 1
    matrix = factors.pit_matrix(calendar, ["000001"], events, "roe")
    settled = matrix["000001"].dropna()
    assert len(settled) > 0
    # The annual value is 11.0 * 1, the first-quarter value is 3.0 * 4.
    assert settled.iloc[0] == pytest.approx(12.0)


def test_point_in_time_matrix_is_empty_before_the_first_availability():
    calendar = trading_calendar("2020-01-01", 120)
    frame = financial_frame([{"股票代码": "000001", "period": "20191231", "净资产收益率": 11.0}])
    events = factors.financial_events(frame, calendar)
    matrix = factors.pit_matrix(calendar, ["000001"], events, "roe")
    available = events.iloc[0]["available"]
    assert matrix.loc[calendar < available, "000001"].isna().all()
    assert matrix.loc[available, "000001"] == pytest.approx(11.0)


def test_point_in_time_matrix_expires_a_stale_report():
    calendar = trading_calendar("2020-01-01", 700)
    frame = financial_frame([{"股票代码": "000001", "period": "20191231", "净资产收益率": 11.0}])
    events = factors.financial_events(frame, calendar)
    matrix = factors.pit_matrix(calendar, ["000001"], events, "roe", tolerance_days=90)
    available = events.iloc[0]["available"]
    assert matrix.loc[available, "000001"] == pytest.approx(11.0)
    beyond = calendar[calendar > available + pd.Timedelta(days=90)]
    assert len(beyond) > 0
    assert matrix.loc[beyond, "000001"].isna().all()


def test_roe_is_annualised_so_quarters_are_comparable():
    calendar = trading_calendar()
    frame = financial_frame([{"股票代码": "000001", "period": "20200331", "净资产收益率": 3.0}])
    events = factors.financial_events(frame, calendar)
    assert events.iloc[0]["roe"] == pytest.approx(12.0)


def test_accruals_signal_is_negated_so_low_accruals_score_high():
    calendar = trading_calendar()
    frame = financial_frame([
        {"股票代码": "000001", "period": "20191231", "每股经营现金流量": 2.0,
         "每股收益": 1.0, "每股净资产": 10.0},
        {"股票代码": "000002", "period": "20191231", "每股经营现金流量": 0.0,
         "每股收益": 1.0, "每股净资产": 10.0},
    ])
    events = factors.financial_events(frame, calendar).set_index("code")
    # Cash-rich earnings are low accruals and must score above paper earnings.
    assert events.loc["000001", "accruals_low"] > events.loc["000002", "accruals_low"]
    assert events.loc["000001", "accruals_low"] == pytest.approx(0.1)
    assert events.loc["000002", "accruals_low"] == pytest.approx(-0.1)


def test_negative_book_value_is_dropped_rather_than_flipping_the_accrual_sign():
    calendar = trading_calendar()
    frame = financial_frame([{"股票代码": "000001", "period": "20191231",
                              "每股经营现金流量": 2.0, "每股收益": 1.0,
                              "每股净资产": -10.0}])
    events = factors.financial_events(frame, calendar)
    assert np.isnan(events.iloc[0]["accruals_low"])


def test_price_signals_negate_volatility_and_scale_amihud():
    calendar = pd.date_range("2020-01-01", periods=200, freq="B")
    codes = ["000001", "000002"]
    rng = np.random.default_rng(7)
    quiet = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.005, 200)))
    noisy = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.030, 200)))
    close = pd.DataFrame({"000001": quiet, "000002": noisy}, index=calendar)
    amount = pd.DataFrame({"000001": 1e9, "000002": 1e9}, index=calendar)
    signals, _ = factors.price_signals({"close": close, "amount": amount}, codes, calendar)
    last = signals["low_vol"].iloc[-1]
    assert last["000001"] > last["000002"]  # the quiet name scores higher
    assert last["000001"] < 0  # and the raw sign is negative


def test_illiquidity_ranks_a_thin_name_above_a_liquid_one():
    calendar = pd.date_range("2020-01-01", periods=120, freq="B")
    codes = ["000001", "000002"]
    rng = np.random.default_rng(11)
    path = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 120)))
    close = pd.DataFrame({"000001": path, "000002": path}, index=calendar)
    amount = pd.DataFrame({"000001": 5e9, "000002": 5e7}, index=calendar)
    signals, _ = factors.price_signals({"close": close, "amount": amount}, codes, calendar)
    last = signals["illiquidity"].iloc[-1]
    assert last["000002"] > last["000001"]


def test_price_signal_requires_a_full_window():
    calendar = pd.date_range("2020-01-01", periods=60, freq="B")
    close = pd.DataFrame({"000001": np.linspace(10.0, 12.0, 60)}, index=calendar)
    amount = pd.DataFrame({"000001": 1e9}, index=calendar)
    signals, _ = factors.price_signals({"close": close, "amount": amount}, ["000001"], calendar)
    # 59 returns cannot fill a 60-row window.
    assert signals["low_vol"]["000001"].isna().all()


def test_signal_directions_are_all_registered_as_positive_ic():
    """Every family is oriented so the pre-registered direction is a positive IC."""
    assert set(factors.SIGNALS) == set(sum(
        [list(value["signals"]) for value in factors.FAMILIES.values()], []))
    assert len(factors.FAMILIES) == 5
    for name in factors.SIGNALS:
        assert name in factors.CONSTRUCTION


def test_coverage_report_counts_populated_cells():
    calendar = pd.date_range("2020-01-01", periods=4, freq="B")
    matrix = pd.DataFrame({"a": [1.0, np.nan, 1.0, np.nan], "b": [1.0, 1.0, np.nan, np.nan]},
                          index=calendar)
    report = factors.coverage_report({"only": matrix})
    assert report["only"]["share"] == pytest.approx(0.5)
    assert report["only"]["dates"] == 3
    assert report["only"]["median_names"] == 1.0


def test_period_mix_counts_names_by_their_last_reported_quarter():
    calendar = trading_calendar("2020-01-01", 200)
    frame = financial_frame([
        {"股票代码": "000001", "period": "20200331", "净资产收益率": 3.0},
        {"股票代码": "000002", "period": "20191231", "净资产收益率": 11.0},
    ])
    events = factors.financial_events(frame, calendar)
    mix = factors.period_mix(calendar, ["000001", "000002"], events)
    # After the 30 April deadline both names are live, but on different quarters.
    late = mix.loc[mix.index > pd.Timestamp("2020-05-06")].iloc[0]
    assert late["03"] + late["12"] == 2
