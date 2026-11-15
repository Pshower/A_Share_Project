"""What a fund actually delivers, against the index it is supposed to track.

A "fund" is not an asset class: it is the carrier through which cash, bonds, equities and gold are
held.  The question this module answers is the only one that matters for a unified strategy --
**how much of the index return survives the vehicle** -- and it answers it with the thing itself:
the fund's price history plus every distribution it paid, compared with the index total return.

The measurement is deliberately blunt:

    tracking difference = (fund total return) ^ (1/years) - (index total return) ^ (1/years)

so a fund with a 0.5% fee, 0.3% of slippage around rebalancing, and a missing dividend all show up
in the same number.  A separate column splits the shortfall into "everything" (annualised) and the
monthly dispersion, because a fund can lag steadily (fees) or wobble around the index (poor
tracking) and those are different problems.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tools import account_ledger, asset_panel, plan_orders

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONLINE = PROJECT_ROOT / "data" / "online"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "fund_vehicles"

# fund code -> (panel column it should track, what the fund is)
CARRIERS = {
    "510300": ("ashare", "CSI 300 ETF (Huatai-PB)"),
    "159919": ("ashare", "CSI 300 ETF (Jiashi)"),
    "510500": ("asmall", "CSI 500 ETF (Southern)"),
    "511010": ("bond_gov5", "5Y treasury ETF (Guotai)"),
    "511260": ("bond_gov10", "10Y treasury ETF (Guotai)"),
    "511090": ("bond_gov30", "30Y treasury ETF (Penghua)"),
    "513500": ("usshare", "S&P 500 ETF (Bosera, QDII)"),
    "518880": ("gold", "gold ETF (Huaan)"),
    "511880": ("cash", "money-market ETF (Yinhua)"),
}

MIN_YEARS = 1.5

# A unit conversion shows up as a single-day jump.  Both directions occur in Chinese ETFs: a
# normal split (1:2 shows as -50%) and the "浠介鎶樼畻" that turned par-value funds into NAV-priced
# ones (159919 jumped 2.59x in 2012, 510500 3.49x in 2015).  Left unrepaired, either direction
# silently rewrites the fund's whole return history -- the 510500 comparison first claimed +9pp a
# year because a 3.5x conversion was read as performance.
SPLIT_RATIOS = (0.5, 0.25, 1 / 3, 2 / 3, 0.75, 0.2, 0.1)
REVERSE_RATIOS = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 10.0)


def repair(prices: pd.Series, paid: dict, tolerance: float = 0.05) -> tuple[pd.Series, list]:
    """Make a fund's price series continuous across unit conversions, and list every repair."""
    prices = prices.astype(float).copy()
    events = []
    for position in range(1, len(prices.index)):
        date, prior = prices.index[position], prices.index[position - 1]
        before, now = float(prices.loc[prior]), float(prices.loc[date])
        if before <= 0 or date in paid:
            continue
        move = now / before
        if any(abs(move - ratio) <= tolerance * ratio for ratio in SPLIT_RATIOS):
            prices.loc[:prior] = prices.loc[:prior] * move
            events.append(dict(date=str(date.date()), kind="split", factor=round(move, 4)))
        elif any(abs(move - ratio) <= tolerance * ratio for ratio in REVERSE_RATIOS):
            # Same direction as a split: scale the *earlier* prices so the jump disappears.  Doing
            # it the other way doubled the discontinuity and pushed 159919 to +13.7pp a year.
            prices.loc[:prior] = prices.loc[:prior] * move
            events.append(dict(date=str(date.date()), kind="reverse_split", factor=round(move, 4)))
    return prices, events


def load_prices(code: str, online: Path | None = None, refresh: bool = False) -> pd.Series:
    """Daily close, cached so a re-run needs no network."""
    online = online or ONLINE
    path = online / "fund" / f"hist_{code}.csv"
    if path.exists() and not refresh:
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"])
        return frame.set_index("date")["close"].astype(float).sort_index()
    import akshare as ak
    frame = ak.fund_etf_hist_sina(symbol=plan_orders.prefix_for(code) + code)
    frame["date"] = pd.to_datetime(frame["date"])
    series = frame.set_index("date")["close"].astype(float).sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    series.rename("close").to_csv(path, index_label="date")
    return series


def load_distributions(code: str, online: Path | None = None, refresh: bool = False) -> dict:
    online = online or ONLINE
    path = online / "fund" / f"dist_{code}.csv"
    if path.exists() and not refresh:
        frame = pd.read_csv(path)
        if not len(frame):
            return {}
        frame["date"] = pd.to_datetime(frame["date"])
        return {stamp: float(value) for stamp, value in zip(frame["date"], frame["per_unit"])}
    events = account_ledger.distributions(code, "2000-01-01", "2026-12-31")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(date=str(stamp), per_unit=value) for stamp, value in events],
                 columns=["date", "per_unit"]).to_csv(path, index=False)
    return {pd.Timestamp(stamp): float(value) for stamp, value in events}


def total_return(prices: pd.Series, paid: dict) -> pd.Series:
    """Units-based total return: distributions are reinvested at the next close.

    Keeping the cash out (as the ledger does) and reinvesting it here are the same number in the
    end; reinvesting is what an index does, so it is what the comparison needs.
    """
    prices = prices.astype(float)
    index = pd.Series(1.0, index=prices.index)
    for position in range(1, len(prices.index)):
        day, prior = prices.index[position], prices.index[position - 1]
        cash = float(paid.get(day, 0.0))
        index.iloc[position] = index.iloc[position - 1] * (prices.loc[day] + cash) / prices.loc[prior]
    return index


def compare(code: str, column: str, levels: pd.DataFrame, online: Path | None = None) -> dict | None:
    prices = load_prices(code, online)
    paid = load_distributions(code, online)
    repaired, events = repair(prices, paid)
    fund = total_return(repaired, paid)
    bench = levels[column].dropna()
    window = fund.index.intersection(bench.index)
    window = window[window >= bench.index[0]]
    if len(window) < 20:
        return None
    fund = fund.reindex(window).ffill()
    bench = bench.reindex(window).ffill()
    years = (window[-1] - window[0]).days / 365.25
    if years < MIN_YEARS:
        return None
    fund_cagr = float((fund.iloc[-1] / fund.iloc[0]) ** (1 / years) - 1)
    bench_cagr = float((bench.iloc[-1] / bench.iloc[0]) ** (1 / years) - 1)
    fund_monthly = fund.resample("M").last().pct_change()
    bench_monthly = bench.resample("M").last().pct_change()
    excess = (fund_monthly - bench_monthly).dropna()
    return dict(code=code, tracks=column, years=round(years, 2),
                fund_cagr=round(fund_cagr, 4), bench_cagr=round(bench_cagr, 4),
                difference_pp=round((fund_cagr - bench_cagr) * 100, 2),
                tracking_error_pp=round(float(excess.std() * np.sqrt(12) * 100), 2),
                repairs=events, distributions=len(paid),
                distribution_total=round(float(sum(paid.values())), 3),
                start=str(window[0].date()), end=str(window[-1].date()))


def table(levels: pd.DataFrame | None = None, online: Path | None = None) -> pd.DataFrame:
    levels = asset_panel.panel() if levels is None else levels
    rows = []
    for code, (column, description) in CARRIERS.items():
        if column not in levels.columns:
            continue
        try:
            row = compare(code, column, levels, online)
        except Exception as error:                                # noqa: BLE001
            rows.append(dict(code=code, tracks=column, description=description,
                             difference_pp=float("nan"), error=str(error)[:80]))
            continue
        if row:
            row["description"] = description
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--refresh" in argv:
        for code in CARRIERS:
            load_prices(code, refresh=True)
            load_distributions(code, refresh=True)
    frame = table()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "carriers.csv", index=False, float_format="%.4f")
    payload = dict(generated_at=str(pd.Timestamp.now())[:19], min_years=MIN_YEARS,
                   carriers=frame.to_dict("records"))
    (OUTPUT / "carriers.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"{'code':<7}{'years':>6}{'fund':>8}{'index':>8}{'diff':>7}{'TE':>6}{'divs':>6}  tracks")
    for row in frame.itertuples():
        print(f"{row.code:<7}{row.years:>6.1f}{row.fund_cagr:>7.2%}{row.bench_cagr:>8.2%}"
              f"{row.difference_pp:>+7.2f}{row.tracking_error_pp:>6.2f}{row.distributions:>6}"
              f"  {row.tracks}")
    print("wrote", OUTPUT / "carriers.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

