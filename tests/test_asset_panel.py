"""The asset panel's arithmetic, and the two ways it could cheat: missing income, and same-day FX.

Every test builds its own point of the filesystem, so the suite never depends on the cached
``data/online`` tree or on the network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import asset_panel as panel


def business_days(start="2005-01-03", periods=520):
    return pd.bdate_range(start, periods=periods)


def write_curve(online: pd.Path, index: pd.DatetimeIndex, yields=(3.0, 3.0, 3.0, 3.0)):
    frame = pd.DataFrame({"\u65e5\u671f": index.strftime("%Y-%m-%d"),
                          "\u4e2d\u56fd\u56fd\u503a\u6536\u76ca\u73872\u5e74": yields[0],
                          "\u4e2d\u56fd\u56fd\u503a\u6536\u76ca\u73875\u5e74": yields[1],
                          "\u4e2d\u56fd\u56fd\u503a\u6536\u76ca\u738710\u5e74": yields[2],
                          "\u4e2d\u56fd\u56fd\u503a\u6536\u76ca\u738730\u5e74": yields[3]})
    path = online / "yields" / "china_treasury.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def build_online(root, index=None, sp500=None, fx=8.0, money_rate=3.5, bond_offset=0.0):
    """A miniature copy of the files the panel reads, small enough to assert against by hand."""
    index = business_days() if index is None else index
    online = root / "online"
    for name in ("index_long", "yields", "global"):
        (online / name).mkdir(parents=True, exist_ok=True)

    for name, level in (("000300", 1000.0), ("000905", 4000.0)):
        pd.DataFrame({"date": index.strftime("%Y-%m-%d"), "close": level}).to_csv(
            online / "index_long" / f"{name}.csv", index=False)
    write_curve(online, index)

    if sp500 is None:
        sp500 = pd.Series(np.linspace(100.0, 200.0, len(index)), index=index)
    pd.DataFrame({"date": sp500.index.strftime("%Y-%m-%d"), "close": sp500.values}).to_csv(
        online / "global" / "sp500.csv", index=False)
    pd.DataFrame({"date": index.strftime("%Y-%m-%d"), "close": 20000.0}).to_csv(
        online / "global" / "hsi.csv", index=False)
    pd.DataFrame({"date": index.strftime("%Y-%m-%d"), "close": 300.0}).to_csv(
        online / "global" / "gold.csv", index=False)
    pd.DataFrame({"\u65e5\u671f": index.strftime("%Y-%m-%d"),
                  "a": fx, "b": fx, "c": fx, "d": fx, "mid": fx, "e": fx}).to_csv(
        online / "global" / "usdcny.csv", index=False, encoding="utf-8")
    pd.DataFrame({"date": index.strftime("%Y-%m-%d"), "m3": money_rate}).to_csv(
        online / "yields" / "shibor_3m.csv", index=False)

    # A traded bond index that drifts from the synthetic level by ``bond_offset`` a year: the
    # level check must pass at zero offset and fail once the drift exceeds the 0.8pp tolerance.
    synthetic = panel.bond_levels(panel.yields(online))["bond5"]
    elapsed = (synthetic.index - synthetic.index[0]).days / 365.0
    drifted = synthetic * (1.0 + bond_offset) ** elapsed
    pd.DataFrame({"date": synthetic.index.strftime("%Y-%m-%d"),
                  "close": drifted.values}).to_csv(
        online / "index_long" / "000012.csv", index=False)
    return online


def test_income_level_compounds_instead_of_multiplying_increments():
    """Regression: ``cumprod`` on day increments deleted every point of dividend income."""
    index = pd.bdate_range("2005-01-03", "2015-01-02")
    factor = panel.income_level(index, 0.018)
    # The first observation seeds one day of income, so the span is a day longer than the range.
    years = ((index[-1] - index[0]).days + 1) / 365.0
    assert factor.iloc[-1] == pytest.approx(1.018 ** years, rel=1e-9)
    assert factor.iloc[-1] > 1.15


def test_income_level_is_monotone_and_starts_at_one_day_of_income():
    index = business_days(periods=250)
    factor = panel.income_level(index, 0.022)
    assert (factor.diff().dropna() > 0).all()
    expected = 1.022 ** ((index[1] - index[0]).days / 365.0)
    assert factor.iloc[0] == pytest.approx(expected, rel=1e-12)


def test_bond_levels_pay_the_coupon_and_lose_on_a_yield_rise():
    index = business_days(periods=300)
    curve = pd.DataFrame({"y2": 3.0, "y5": 3.0, "y10": 3.0, "y30": 3.0}, index=index)
    flat = panel.bond_levels(curve)["bond5"]
    assert flat.iloc[-1] == pytest.approx((1 + 0.03 / 252) ** 299, rel=1e-9)

    jumped = curve.copy()
    jumped.iloc[100:, :] = 3.10                                  # +10bp on day 100
    move = panel.bond_levels(jumped)["bond5"]
    day = move.iloc[100] / move.iloc[99] - 1
    assert day == pytest.approx(0.03 / 252 - 5 * 0.001, abs=2e-5)


def test_daily_factor_compounds_over_calendar_days():
    index = business_days(periods=252)
    rate = pd.Series(0.05, index=index)
    factor = panel.daily_factor(rate, index)
    days = (index[-1] - index[0]).days + 1                         # the first row seeds one day
    assert float(factor.prod()) == pytest.approx(1.05 ** (days / 365.0), rel=1e-9)


def test_known_today_never_shows_a_close_before_the_chinese_book_can_see_it():
    index = business_days(periods=10)
    overseas = pd.Series(np.arange(10.0), index=index)
    aligned = panel.known_today(overseas, index)
    assert pd.isna(aligned.iloc[0])
    assert aligned.iloc[5] == 4.0                                 # yesterday's close, not today's


def test_panel_holds_the_us_jump_back_one_session(tmp_path):
    index = business_days(periods=30)
    sp500 = pd.Series(100.0, index=index)
    sp500.iloc[10:] = 200.0                                       # doubles on session 10
    online = build_online(tmp_path, index=index, sp500=sp500, fx=8.0)
    frame = panel.panel(online=online, start="2005-01-03")
    level = frame["usshare"].dropna()
    jumps = level.pct_change().dropna()
    jump_at = jumps[jumps.abs() > 0.5]
    assert len(jump_at) == 1
    assert index.get_loc(jump_at.index[0]) == 11                  # visible one session late


def test_panel_covers_every_asset_and_starts_on_the_sample_date(tmp_path):
    online = build_online(tmp_path)
    frame = panel.panel(online=online)
    assert list(frame.columns) == ["cash", "deposit", "bond2", "bond5", "bond9", "bond20",
                                   "bond_gov5", "bond_credit", "bond_hy", "bond_b13", "bond_b35",
                                   "bond_b57", "bond_b710", "ashare", "asmall", "usshare", "hk",
                                   "gold"]
    assert str(frame.index[0].date()) >= panel.SAMPLE_START
    assert frame.iloc[0].dropna().to_numpy() == pytest.approx(1.0)
    assert int(frame["bond5"].isna().sum()) == 0
    # Without a ChinaBond cache the real-bond columns stay empty rather than reaching for the net.
    assert frame["bond_credit"].isna().all()


def test_chinabond_reads_its_cache_and_refuses_the_network_by_default(tmp_path):
    online = build_online(tmp_path)
    index = business_days(periods=200)
    with pytest.raises(FileNotFoundError):
        panel.chinabond("credit", online)
    with pytest.raises(KeyError):
        panel.chinabond("not_an_index", online)
    pd.DataFrame({"date": index.strftime("%Y-%m-%d"),
                  "value": 100.0 * (1.0002 ** np.arange(len(index)))}).to_csv(
        online / "yields" / "cbond_credit.csv", index=False)
    series = panel.chinabond("credit", online)
    assert len(series) == len(index)
    assert series.iloc[-1] > series.iloc[0]


def test_panel_uses_the_cached_chinabond_series_when_present(tmp_path):
    online = build_online(tmp_path)
    index = business_days(periods=200)
    values = 100.0 * (1.0003 ** np.arange(len(index)))
    pd.DataFrame({"date": index.strftime("%Y-%m-%d"), "value": values}).to_csv(
        online / "yields" / "cbond_gov5.csv", index=False)
    frame = panel.panel(online=online, start="2005-01-03")
    column = frame["bond_gov5"].dropna()
    assert len(column) > 0
    assert float(column.iloc[-1] / column.iloc[0]) == pytest.approx(
        float(values[-1] / values[0]), rel=1e-9)


def test_cash_takes_the_better_of_the_deposit_and_the_money_fund(tmp_path):
    low = panel.panel(online=build_online(tmp_path / "low", money_rate=0.5), start="2005-01-03")
    high = panel.panel(online=build_online(tmp_path / "high", money_rate=6.0), start="2005-01-03")
    low_years = (low.index[-1] - low.index[0]).days / 365.25
    high_years = (high.index[-1] - high.index[0]).days / 365.25
    low_cash = low["cash"].iloc[-1] ** (1 / low_years) - 1
    high_cash = high["cash"].iloc[-1] ** (1 / high_years) - 1
    # With a 0.5% money fund the administered deposit rate wins; with 6% the money fund wins.
    assert low_cash == pytest.approx(0.0228, abs=0.002)            # 2005 deposit benchmark is 2.25%
    assert high_cash == pytest.approx(0.056, abs=0.004)


def test_bond_check_passes_on_the_level_and_fails_when_the_index_is_far_off(tmp_path):
    close = panel.check_bond_model(build_online(tmp_path / "close", bond_offset=0.0))
    far = panel.check_bond_model(build_online(tmp_path / "far", bond_offset=0.03))
    assert close["pass_"] is True and close["gap_pp"] < 0.8
    assert far["pass_"] is False and far["gap_pp"] > 0.8
    assert "amended" in close



