"""Build the five pre-registered alpha families with point-in-time alignment.

The families are registered in ``doc/alpha_program_20261106.md`` section 5: reported growth,
quality, accruals, cross-sectional low volatility and Amihud illiquidity.  Every construction
here is fixed -- no window, threshold or transform is searchable -- and every signal is
oriented so that the pre-registered direction is a *positive* IC.  That sign convention is the
whole point: criterion 3 of the protocol then reduces to "the measured IC is positive", which
cannot be fudged after the fact by flipping a factor.

Point-in-time rules, which are not negotiable:

* ``financials.parquet`` carries ``announcement_date``, but the snapshot documents it as the
  vendor's *latest revision* date.  For the 2017Q1 period it sits in mid-2018, more than a
  year after the quarter closed.  Using it would test a signal made of stale data rather than
  a point-in-time signal, so the availability date is the statutory disclosure deadline
  (:func:`src.data.pit.statutory_deadline`) shifted to the next trading row.  A legal deadline
  can only delay a row, never leak one.
* The stored financials are the vendor's latest revision of each period, not the original
  vintage.  Restatements are therefore invisible; the direction of that bias is unknown.
* Per-share and return-on-equity figures are year-to-date, so a Q1 cross-section is not
  comparable with an annual one.  They are annualised by ``12 / elapsed months``, which is
  arithmetic fixed by the report period.  ``销售毛利率`` is a ratio of two same-window flows
  and ``同比增长`` is already a year-on-year rate, so neither needs the adjustment.
"""

import numpy as np
import pandas as pd

from src.data.pit import statutory_deadline
from tools.signal_expansion import next_trading_day

VOL_WINDOW = 60
AMIHUD_WINDOW = 60
STALE_DAYS = 400
HIKE = 1e8

SIGNALS = ("rev_growth", "profit_growth", "roe", "gross_margin",
           "accruals_low", "low_vol", "illiquidity")

FAMILIES = {
    "growth": {"signals": ("rev_growth", "profit_growth"), "registered": "高成长 → 高收益"},
    "quality": {"signals": ("roe", "gross_margin"), "registered": "高质量 → 高收益"},
    "accruals": {"signals": ("accruals_low",), "registered": "低应计 → 高收益"},
    "low_volatility": {"signals": ("low_vol",), "registered": "低波动 → 高收益"},
    "illiquidity": {"signals": ("illiquidity",), "registered": "低流动性 → 高收益"},
}

CONSTRUCTION = {
    "rev_growth": "营业总收入-同比增长 (%)，最近一期已披露报告",
    "profit_growth": "净利润-同比增长 (%)，最近一期已披露报告",
    "roe": "净资产收益率 (%) × 12/已过月数，最近一期已披露报告",
    "gross_margin": "销售毛利率 (%)，最近一期已披露报告",
    "accruals_low": "(每股经营现金流量 − 每股收益) × 12/已过月数 / 每股净资产，最近一期已披露报告",
    "low_vol": f"−过去 {VOL_WINDOW} 个交易日日收益标准差",
    "illiquidity": f"1e8 × mean(|日收益| / 成交额)，过去 {AMIHUD_WINDOW} 个交易日",
}


def union_calendar(history, start=None, end=None):
    """Every date that appears in any stored quote, sorted and unique."""
    stamps = set()
    for path in sorted(history.glob("*.parquet")):
        part = pd.read_parquet(path, columns=["date"])
        stamps.update(pd.to_datetime(part["date"]).dropna().tolist())
    index = pd.DatetimeIndex(sorted(stamps), name="date")
    if start is not None:
        index = index[index >= pd.Timestamp(start)]
    if end is not None:
        index = index[index <= pd.Timestamp(end)]
    return index


def load_fields(history, index, codes, fields=("open", "close", "amount")):
    """One ``dates x codes`` float matrix per requested column."""
    frames = {name: pd.DataFrame(np.nan, index=index, columns=list(codes), dtype=float)
              for name in fields}
    for code in codes:
        path = history / f"{code}.parquet"
        if not path.exists():
            continue
        part = pd.read_parquet(path, columns=["date", *fields])
        part = part.dropna(subset=["date"]).drop_duplicates("date").set_index("date")
        part.index = pd.to_datetime(part.index)
        for name in fields:
            frames[name][code] = part[name].reindex(index).to_numpy(dtype=float)
    return frames


def to_price_frame(fields, codes, names=("open", "close")):
    """MultiIndex ``(code, field)`` frame, the layout ``forward_labels`` expects."""
    data = {(code, name): fields[name][code].to_numpy(dtype=float)
            for code in codes for name in names}
    return pd.DataFrame(data, index=fields[names[0]].index)


def listing_dates(history, codes):
    """First stored quote per code, as a proxy for the listing date."""
    first = {}
    for code in codes:
        path = history / f"{code}.parquet"
        if not path.exists():
            continue
        part = pd.read_parquet(path, columns=["date"])
        stamps = pd.to_datetime(part["date"]).dropna()
        if len(stamps):
            first[code] = stamps.min()
    return pd.Series(first, dtype="datetime64[ns]")


def listing_age(index, codes, first_seen):
    """Years since the first stored quote; censored at the start of the panel."""
    ages = pd.DataFrame(np.nan, index=index, columns=list(codes), dtype=float)
    for code in codes:
        if code not in first_seen.index:
            continue
        days = (index - pd.Timestamp(first_seen[code])).days.to_numpy(dtype=float)
        ages[code] = np.clip(days / 365.25, 0.0, None)
    return ages


def annualisation(periods):
    """``12 / elapsed months``: 4.0 at Q1, 2.0 at H1, 4/3 at 9M, 1.0 at the annual report."""
    months = pd.Series(periods, dtype="string").str[4:6].astype(int) // 3 * 3
    if months.isna().any() or (months <= 0).any():
        raise ValueError("Report periods must be YYYYMMDD quarter ends")
    return 12.0 / months.astype(float)


def financial_events(frame, calendar):
    """Attach the point-in-time availability date and the seven signal columns.

    ``calendar`` must span the whole stored history, not just the evaluation window: the
    availability date is the first *stored* row after the legal deadline, so a calendar that
    starts later would collapse every older report onto its first row.
    """
    events = pd.DataFrame({
        "code": frame["code"].astype(str).str.zfill(6),
        "period": frame["period"].astype(str),
    })
    events["annualise"] = annualisation(events["period"])
    deadlines = {period: statutory_deadline(period) for period in events["period"].unique()}
    events["deadline"] = events["period"].map(deadlines)
    events["available"] = next_trading_day(calendar, events["deadline"])
    leaked = events["available"].notna() & (events["available"] <= events["deadline"])
    if leaked.any():
        raise ValueError("Availability date precedes the legal deadline; extend the calendar")
    gain = events["annualise"].to_numpy(dtype=float)
    revenue_growth = pd.to_numeric(frame["营业总收入-同比增长"], errors="coerce").to_numpy()
    profit_growth = pd.to_numeric(frame["净利润-同比增长"], errors="coerce").to_numpy()
    roe = pd.to_numeric(frame["净资产收益率"], errors="coerce").to_numpy()
    margin = pd.to_numeric(frame["销售毛利率"], errors="coerce").to_numpy()
    operating = pd.to_numeric(frame["每股经营现金流量"], errors="coerce").to_numpy()
    earnings = pd.to_numeric(frame["每股收益"], errors="coerce").to_numpy()
    book = pd.to_numeric(frame["每股净资产"], errors="coerce").to_numpy()

    # A negative book value makes the accrual ratio change sign, not scale; drop it.
    book = np.where(book > 0, book, np.nan)
    events["rev_growth"] = revenue_growth
    events["profit_growth"] = profit_growth
    events["roe"] = roe * gain
    events["gross_margin"] = margin
    # Accruals are earnings minus operating cash flow, so "low accrual, high return" means the
    # signal is cash flow minus earnings; the negation is the direction, not a fudge after the
    # fact.  Both per-share inputs cover the same year-to-date window, so the difference is
    # annualised by the same fixed factor before it is divided by book value per share.
    events["accruals_low"] = (operating - earnings) * gain / book
    events = events.dropna(subset=["available"]).drop_duplicates(["code", "period"])
    return events


def pit_matrix(index, codes, events, column, *, tolerance_days=STALE_DAYS):
    """Value of the most recent event available at or before each date, per code."""
    values = np.full((len(index), len(codes)), np.nan, dtype=float)
    positions = pd.Series(np.arange(len(index)), index=index)
    grid = np.arange(len(index))
    stamps = index.to_numpy()
    tolerance = np.timedelta64(tolerance_days, "D")
    for column_index, code in enumerate(codes):
        part = events[events["code"] == code].dropna(subset=["available", column])
        if part.empty:
            continue
        # Ties on the availability date resolve to the later report period, which is the one
        # the market has most recently learned about.
        part = part.sort_values(["available", "period"], kind="mergesort")
        rows = positions.reindex(part["available"]).to_numpy()
        keep = ~pd.isna(rows)
        if not keep.any():
            continue
        rows = rows[keep].astype(int)
        scores = part[column].to_numpy(dtype=float)[keep]
        last = np.searchsorted(rows, grid, side="right") - 1
        usable = last >= 0
        if not usable.any():
            continue
        ages = stamps[grid[usable]] - stamps[rows[last[usable]]]
        fresh = ages <= tolerance
        target = grid[usable][fresh]
        values[target, column_index] = scores[last[usable][fresh]]
    return pd.DataFrame(values, index=index, columns=list(codes))


def price_signals(fields, codes, index):
    """Cross-sectional low volatility and Amihud illiquidity, both already signed."""
    close = fields["close"]
    amount = fields["amount"]
    returns = close / close.shift(1) - 1.0
    returns = returns.where(np.isfinite(returns))
    volatility = returns.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std(ddof=0)
    traded = amount.where(amount > 0)
    impact = (returns.abs() / traded).where(np.isfinite(returns.abs() / traded))
    amihud = impact.rolling(AMIHUD_WINDOW, min_periods=AMIHUD_WINDOW).mean() * HIKE
    return {"low_vol": -volatility, "illiquidity": amihud}, returns


def build_signals(index, codes, fields, events):
    """All seven registered signals on one ``dates x codes`` grid."""
    signals = {name: pit_matrix(index, codes, events, name)
               for name in ("rev_growth", "profit_growth", "roe", "gross_margin", "accruals_low")}
    priced, returns = price_signals(fields, codes, index)
    signals.update(priced)
    return signals, returns


def coverage_report(signals):
    """Share of populated cells and the populated-name count per decision date."""
    report = {}
    for name, matrix in signals.items():
        populated = matrix.notna()
        report[name] = dict(share=float(populated.to_numpy().mean()),
                            dates=int(populated.any(axis=1).sum()),
                            median_names=float(populated.sum(axis=1).median()))
    return report


def period_mix(index, codes, events, tolerance_days=STALE_DAYS):
    """Which report period each name is actually carrying at each date.

    Counting how many names have an available Q1 report would always give the whole
    cross-section four times over; what matters is the *latest* period per name, because that
    is the one the signal is built from.  A date where the cross-section is split across
    quarters mixes annualised year-to-date windows, which is the known weakness of the fixed
    annualisation in :func:`financial_events`.
    """
    ready = events.dropna(subset=["available"]).copy()
    ready["period_code"] = ready["period"].astype(float)
    latest = pit_matrix(index, codes, ready, "period_code", tolerance_days=tolerance_days)
    month = (latest // 100) % 100
    out = {"%02d" % value: month.eq(value).sum(axis=1) for value in (3, 6, 9, 12)}
    out["none"] = month.isna().sum(axis=1)
    return pd.DataFrame(out, index=index)
