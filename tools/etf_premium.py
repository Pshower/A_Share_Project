"""What did it cost to buy 513500 at the wrong moment: the QDII premium, measured.

The plan's risk sleeve is half S&P 500 (ETF 513500), and a QDII ETF can trade several percent above
its own net asset value.  Buying at a 4% premium means paying 4% for something worth 100, and the
account holds 30% of its value there -- 1.2% of NAV gone the moment the trade fills, before any
market move.

The premium is price / NAV - 1.  NAV history comes from the fund's own published series
(``fund.eastmoney.com/pingzhongdata/<code>.js``), prices from the same Sina history the backtest
uses, so the two sides are the numbers a person could actually have seen.

Caveat that decides how this may be used: for a QDII, the NAV published on day t values the fund at
the *previous* US close, while the price on day t already reflects today's expectations.  A positive
premium is therefore partly a time-zone artefact and is not a pure arbitrage signal.  What it does
measure is the entry cost a buyer actually paid -- which is what matters for a purchase.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONLINE = PROJECT_ROOT / "data" / "online"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "etf_premium"
CODE = "513500"
SHARE_OF_NAV = 0.30           # 513500's target weight in the plan
THRESHOLDS = (0.01, 0.02, 0.03, 0.05, 0.08)


def nav_history(code=CODE, url=None):
    """Published daily NAV from the fund page's ``Data_netWorthTrend`` block."""
    import requests

    url = url or "https://fund.eastmoney.com/pingzhongdata/%s.js" % code
    text = requests.get(url, timeout=30, headers={"Referer": "https://fund.eastmoney.com/"}).text
    match = re.search(r"Data_netWorthTrend\s*=\s*(\[.*?\]);", text)
    if not match:
        raise ValueError("no Data_netWorthTrend in payload")
    rows = re.findall(r"\{[^}]*\}", match.group(1))
    stamps, values = [], []
    for row in rows:
        found = re.search(r'"x":(\d+)', row)
        value = re.search(r'"y":([-\d.]+)', row)
        if found and value:
            stamps.append(pd.Timestamp(int(found.group(1)), unit="ms").normalize())
            values.append(float(value.group(1)))
    return pd.Series(values, index=pd.DatetimeIndex(stamps), name="nav").sort_index()


def close_series(code=CODE, start="2014-01-01", end=None):
    """Daily closes from the same Sina history the backtest reads."""
    from tools import plan_orders
    import akshare as ak

    frame = ak.fund_etf_hist_sina(symbol=plan_orders.prefix_for(code) + code)
    frame["date"] = pd.to_datetime(frame["date"])
    series = frame.set_index("date").sort_index()["close"].astype(float)
    return series.loc[(series.index >= pd.Timestamp(start)) & (series.index <= (pd.Timestamp(end) if end else series.index.max()))]


def split_days(close, nav, jump=0.30, window=2):
    """Sessions around a share split that the two series applied on different days.

    513500 split 1:2: the NAV halved on 2022-03-28 and the price on 2022-03-30 (a suspension sat in
    between), so for two sessions the ratio reads as a 96% premium.  Either series jumping while the
    other does not is the signal -- no date is hard-coded.
    """
    price_move = close.pct_change().abs()
    nav_move = nav.reindex(close.index).pct_change().abs()
    flagged = (price_move > jump) | (nav_move > jump)
    if not flagged.any():
        return []
    days, index = [], list(close.index)
    for position, stamp in enumerate(index):
        if flagged.iloc[position]:
            days.extend(index[max(position - window, 0):position + window + 1])
    return sorted(set(days))


def premium(close, nav, drop_splits=True):
    """price / NAV - 1 on the dates both sides publish, with split days removed."""
    joined = pd.DataFrame({"close": close, "nav": nav}).dropna()
    if joined.empty:
        raise ValueError("no overlapping dates between price and NAV")
    joined["premium"] = joined["close"] / joined["nav"] - 1.0
    if drop_splits:
        broken = set(split_days(joined["close"], joined["nav"]))
        joined = joined.loc[~joined.index.isin(broken)]
        joined.attrs["split_days"] = sorted(str(stamp.date()) for stamp in broken)
    return joined


def refresh_cache(code: str = CODE, online: Path | None = None, nav=None, close=None,
                  keep: int = 60) -> dict:
    """Fetch both published sides, compute the premium, cache the tail atomically.

    ``daily_allocation.recommendation`` is offline by rule, so this is the one networked step
    that feeds its circuit breaker: the daily job calls it before advising, and the breaker
    always speaks with the cache's own date, never with a number it cannot date.  Split days
    are dropped exactly as in the study; ``nav``/``close`` are injectable so tests never need
    the network.
    """
    close = close_series(code) if close is None else close
    nav = nav_history(code) if nav is None else nav
    frame = premium(close, nav)
    tail = frame.tail(int(keep)).reset_index().rename(columns={"index": "date"})
    tail["date"] = tail["date"].dt.strftime("%Y-%m-%d")
    path = Path(online or ONLINE) / "fund" / ("premium_%s.csv" % code)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    tail.to_csv(temporary, index=False, columns=["date", "close", "nav", "premium"])
    os.replace(temporary, path)
    latest = frame.iloc[-1]
    return dict(code=code, date=str(frame.index[-1].date()), close=float(latest["close"]),
                nav=float(latest["nav"]), premium=float(latest["premium"]),
                rows_cached=int(len(tail)))


def entry_cost(frame, threshold=0.03, hold_sessions=20):
    """What a purchase at the premium costs by the time it is sold ``hold_sessions`` later.

    ``cost`` is the premium paid minus the premium received back: buying at 6% and selling at 2%
    loses 4% of the position.  A gate that only buys below ``threshold`` trades off how often it can
    buy against that cost.
    """
    series = frame["premium"]
    forward = series.shift(-hold_sessions)
    pairs = pd.DataFrame({"now": series, "later": forward}).dropna()
    if pairs.empty:
        return {}
    cost = pairs["now"] - pairs["later"]
    allowed = pairs[pairs["now"] < threshold]
    return dict(hold_sessions=int(hold_sessions), pairs=int(len(pairs)),
                buy_always=dict(pairs=int(len(pairs)), mean=float(cost.mean()),
                                median=float(cost.median()),
                                share_positive=float((cost > 0).mean())),
                below_threshold=dict(threshold=float(threshold), pairs_allowed=int(len(allowed)),
                                     share=float(len(allowed) / len(pairs)),
                                     mean=float((allowed["now"] - allowed["later"]).mean()) if len(allowed) else None))


def distribution(series, thresholds=THRESHOLDS):
    values = series.dropna()
    return dict(days=int(len(values)),
                mean=float(values.mean()), median=float(values.median()),
                p05=float(values.quantile(0.05)), p95=float(values.quantile(0.95)),
                worst=float(values.min()), best=float(values.max()),
                share_above={("%.0f%%" % (100 * level)): float((values > level).mean())
                             for level in thresholds})


def drag(levels, share=SHARE_OF_NAV):
    """NAV cost of buying ``share`` of the book at each premium level."""
    return {("%.0f%%" % (100 * level)): float(share * level) for level in levels}


def gate_effect(premiums, threshold):
    """How often a buy would be blocked, and what the blocked days' premium looked like."""
    values = premiums.dropna()
    blocked = values[values > threshold]
    return dict(threshold=float(threshold), days=len(values), blocked=len(blocked),
                blocked_share=float(len(blocked) / len(values)) if len(values) else None,
                blocked_mean=float(blocked.mean()) if len(blocked) else None,
                saved_nav=float(SHARE_OF_NAV * blocked.mean()) if len(blocked) else 0.0)


def run(code=CODE, output=OUTPUT, log=print):
    close = close_series(code)
    nav = nav_history(code)
    frame = premium(close, nav)
    summary = dict(code=code, window=dict(first=str(frame.index[0].date()),
                                          last=str(frame.index[-1].date())),
                   days=int(len(frame)), distribution=distribution(frame["premium"]),
                   drag_by_level=drag(THRESHOLDS),
                   gates=[gate_effect(frame["premium"], level) for level in THRESHOLDS])
    log("%s  %s ~ %s  %d 个交易日" % (code, summary["window"]["first"], summary["window"]["last"],
                                     summary["days"]))
    log("溢价 中位 %5.2f%%  p95 %5.2f%%  最高 %5.2f%%"
        % (100 * summary["distribution"]["median"], 100 * summary["distribution"]["p95"],
           100 * summary["distribution"]["best"]))
    for gate in summary["gates"]:
        log("闸门 %4.0f%%：拦下 %5.1f%% 的交易日，被拦日平均溢价 %5.2f%%，按 30%% 仓位省下 %4.2f%% NAV"
            % (100 * gate["threshold"], 100 * gate["blocked_share"], 100 * (gate["blocked_mean"] or 0),
               100 * gate["saved_nav"]))
    if output:
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path / ("premium_%s.csv" % code), float_format="%.6f")
        (path / ("summary_%s.json" % code)).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        log("wrote %s" % (path / ("summary_%s.json" % code)))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="QDII premium of the plan's S&P 500 leg.")
    parser.add_argument("--code", default=CODE)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    summary = run(args.code, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
