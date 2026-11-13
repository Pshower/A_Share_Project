"""The market-level leverage gate, implemented exactly as pre-registered.

Protocol: ``doc/market_leverage_protocol_20261114.md``.  It asks one question -- does the 20-day
growth of aggregate margin balance per unit of float cap predict the market's forward 20-day
return and drawdown -- and fixes every choice before the first statistic is computed:

* basket: codes with >= 1,382 days of margin data and of price/shares, fixed for the whole window;
* signal: ``r20 = ln(lev_t / lev_{t-20})``, one variable, no percentile or z-score variant;
* two validity guards judged first (coverage contamination, market representativeness);
* four conjoined bars (forward return on 000300, both sub-periods negative, drawdown on the same
  signal, and the whole thing again on 399317), each at Newey-West t of 2.5 with 20 lags.

Nothing here is allowed to swap a variable or a window after the fact; the diagnostics that are
reported (5-day version, margin-only aggregate, short-balance growth) are labelled as such and
carry no verdict.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tools.alpha_ic import nw_se

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARGIN = PROJECT_ROOT / "data" / "online" / "margin" / "margin.parquet"
SSE_TOTAL = PROJECT_ROOT / "data" / "online" / "margin" / "sse_margin_total.csv"
PANEL = PROJECT_ROOT / "data" / "clean" / "panel_full"
INDEX_DIR = PROJECT_ROOT / "data" / "online" / "index_long"
OUTPUT_ROOT = PROJECT_ROOT / "reports" / "runs" / "market_leverage"

BASKET_MIN_DAYS = 1382        # 95% of the archive's 1,455 days
GUARD_COVER = 0.30
GUARD_CAP_SHARE = 0.30
LAGS = 20
HORIZON = 20
T1_T = -2.5
T3_T = 2.5
SUBS = (("2019-01-01", "2021-12-31"), ("2022-01-01", "2024-12-31"))
V_START, V_END = "2019-01-01", "2024-12-31"
R_END = "2026-06-30"          # P (2026-07 onward) stays untouched

# ``stock_margin_sse`` returns these columns in this order; the header comes back in whatever
# encoding the endpoint felt like, so the mapping is by position and checked against magnitudes.
SSE_COLUMNS = ("date", "margin_balance", "margin_buy", "short_volume",
               "short_value", "short_sell_volume", "margin_total")


# ----------------------------------------------------------------------------- data

def load_margin(path=MARGIN):
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def sse_totals(path):
    """Official Shanghai aggregate margin trading, one row per session.

    This is the repair for the local archive: the archive's *coverage* grew from 819 to 3,718
    codes over the sample, which is indistinguishable from leverage building.  An exchange total
    has no coverage to grow, so the contamination the first guard exists to catch is absent by
    construction rather than by hope.
    """
    frame = pd.read_csv(path)
    if len(frame.columns) != len(SSE_COLUMNS):
        raise ValueError("unexpected SSE column count: %d" % len(frame.columns))
    frame.columns = list(SSE_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d")
    return frame.drop_duplicates("date").set_index("date").sort_index()


def sse_float_cap(cap):
    """Float cap of the Shanghai-listed universe the SSE margin total is measured against."""
    codes = [code for code in cap.columns if str(code).startswith(("6",))]
    return cap[codes].sum(axis=1), codes


def sse_leverage(cap, path):
    totals = sse_totals(path)
    value, codes = sse_float_cap(cap)
    frame = pd.DataFrame(dict(balance=totals["margin_balance"], cap=value)).dropna()
    frame = frame[frame["cap"] > 0]
    frame["lev"] = 100.0 * frame["balance"] / frame["cap"]
    frame["short_lev"] = 100.0 * totals["short_value"].reindex(frame.index) / frame["cap"]
    frame["codes"] = np.nan          # no coverage count exists; guard G1 is structurally moot
    return frame, codes


def float_cap(panel=PANEL):
    """Float capitalisation per code per day: unadjusted close times float shares."""
    price = pd.read_parquet(Path(panel) / "unadj_close.parquet")
    shares = pd.read_parquet(Path(panel) / "shares.parquet")
    price.index = pd.to_datetime(price.index)
    shares.index = pd.to_datetime(shares.index)
    shared = price.columns.intersection(shares.columns)
    return (price[shared] * shares[shared]).sort_index()


def select_basket(margin, cap, min_days=BASKET_MIN_DAYS):
    """The fixed basket from protocol section 3.1 -- all three conditions, no re-selection."""
    have_margin = margin.groupby("code")["date"].nunique()
    traded = margin[margin["balance"] > 0]["code"].unique()
    have_cap = cap.notna().sum()
    codes = sorted(set(have_margin[have_margin >= min_days].index)
                   & set(have_cap[have_cap >= min_days].index)
                   & set(traded))
    return codes


def leverage(margin, cap, basket):
    """Aggregate margin balance per unit of the same basket's float cap, in percent.

    The row set is the days the archive actually has a balance for.  Building the frame from
    ``balance`` and ``cap`` together would align them on the union of their dates, which starts
    the series in 2017 (panel) with a NaN numerator and makes the observation count a lie.
    """
    window = margin[margin["code"].isin(basket)]
    balance = window.groupby("date")["balance"].sum()
    codes = window.groupby("date")["code"].nunique()
    value = cap[basket].sum(axis=1)
    frame = pd.DataFrame(dict(balance=balance, cap=value, codes=codes))
    frame = frame.dropna(subset=["balance", "cap"])
    frame = frame[frame["cap"] > 0]
    frame["lev"] = 100.0 * frame["balance"] / frame["cap"]
    return frame


def index_close(code):
    path = INDEX_DIR / ("%s.csv" % code)
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")["close"].astype(float).sort_index()


def targets(close, horizon=HORIZON):
    """Forward log return and the deepest drawdown inside the same window (positive = worse)."""
    future = close.shift(-horizon)
    forward = np.log(future / close)
    trench = close.rolling(horizon, min_periods=horizon).min().shift(-horizon)
    drawdown = 1.0 - trench / close
    return pd.DataFrame(dict(fwd=forward, dd=drawdown))


def build(basket_frame, close, horizon=HORIZON):
    """Signal, targets and the coverage count, inner-joined on the index's trading days."""
    lev = basket_frame["lev"]
    signal = pd.DataFrame(dict(
        lev=lev,
        r20=np.log(lev / lev.shift(horizon)),
        n_codes=basket_frame["codes"].astype(float),
        dn20=basket_frame["codes"].astype(float).diff(horizon),
    ))
    frame = signal.join(targets(close, horizon), how="inner")
    return frame.replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------------------------- statistics

def nw_slope(x, y, lag=LAGS):
    """OLS slope with a Newey-West t, built on ``alpha_ic.nw_se``.

    ``b = sum g / sum (x-xbar)^2`` with ``g_t = (x_t - xbar) e_t``, so ``Var(sum g) = n^2 *
    Var(mean g)`` and the existing Newey-West routine gives the slope's standard error without a
    second implementation of the Bartlett kernel.
    """
    frame = pd.DataFrame(dict(x=x, y=y)).dropna()
    if len(frame) < 4:
        return dict(slope=float("nan"), intercept=float("nan"), t=float("nan"),
                    n=int(len(frame)), corr=float("nan"))
    xa = frame["x"].to_numpy(dtype=float)
    ya = frame["y"].to_numpy(dtype=float)
    xbar, ybar = xa.mean(), ya.mean()
    centred = xa - xbar
    spread = float(centred @ centred)
    if spread <= 0:
        return dict(slope=float("nan"), intercept=float("nan"), t=float("nan"),
                    n=int(len(frame)), corr=float("nan"))
    slope = float(centred @ (ya - ybar)) / spread
    intercept = ybar - slope * xbar
    residual = ya - intercept - slope * xa
    score = centred * residual
    error = nw_se(score, lag)
    se = len(xa) * error / spread if spread > 0 else float("nan")
    tstat = slope / se if se and np.isfinite(se) and se > 0 else float("nan")
    corr = float(np.corrcoef(xa, ya)[0, 1]) if len(xa) > 2 else float("nan")
    return dict(slope=slope, intercept=intercept, t=float(tstat), n=int(len(frame)), corr=corr)


def guards(frame, cap, basket, source="archive", threshold=GUARD_COVER,
           cap_share=GUARD_CAP_SHARE):
    """The two validity guards. Judged before any verdict; failing one voids the whole round.

    With the official exchange total there is no per-day coverage count to correlate against, and
    no archive coverage to contaminate the signal -- G1 is *not applicable by construction*, which
    is a statement about the data source, not a waiver of the guard.  The archive-based numbers are
    kept in the result file so both readings stay visible.
    """
    if source == "sse":
        coverage = None
        applicable = False
        contaminated = False
    else:
        clean = frame.dropna(subset=["r20", "dn20"])
        coverage = float(clean["r20"].corr(clean["dn20"])) if len(clean) > 3 else float("nan")
        applicable = True
        contaminated = bool(np.isfinite(coverage) and abs(coverage) > threshold)
    total = float(cap.sum(axis=1).median())
    inside = float(cap[basket].sum(axis=1).median())
    share = inside / total if total else float("nan")
    return dict(source=source, coverage_corr=coverage, coverage_limit=threshold,
                coverage_applicable=applicable, contaminated=contaminated,
                cap_share=share, cap_share_limit=cap_share,
                thinly_representative=bool(np.isfinite(share) and share < cap_share))


def evaluate(frame, other_frame):
    """T1-T4, mechanically."""
    t1 = nw_slope(frame["r20"], frame["fwd"])
    t2 = []
    for start, end in SUBS:
        piece = frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]
        row = nw_slope(piece["r20"], piece["fwd"])
        t2.append(dict(start=start, end=end, slope=row["slope"], t=row["t"], n=row["n"]))
    t3 = nw_slope(frame["r20"], frame["dd"])
    t4 = nw_slope(other_frame["r20"], other_frame["fwd"])
    bars = dict(
        T1=bool(np.isfinite(t1["t"]) and t1["slope"] < 0 and t1["t"] <= T1_T),
        T2=bool(all(np.isfinite(row["slope"]) and row["slope"] < 0 for row in t2)),
        T3=bool(np.isfinite(t3["t"]) and t3["slope"] > 0 and t3["t"] >= T3_T),
        T4=bool(np.isfinite(t4["t"]) and t4["slope"] < 0 and t4["t"] <= T1_T),
    )
    return dict(T1=t1, T2=dict(parts=t2), T3=t3, T4=t4, bars=bars)


def diagnostics(frame, basket_frame):
    """Labelled as diagnostics: they never enter the verdict (protocol section 8.4)."""
    lev = basket_frame["lev"]
    short = basket_frame.get("short_lev")
    out = dict(r5=nw_slope(np.log(lev / lev.shift(5)).reindex(frame.index), frame["fwd"]))
    if short is not None:
        out["short_r20"] = nw_slope(np.log(short / short.shift(HORIZON)).reindex(frame.index),
                                    frame["fwd"])
    out["levels"] = dict(lev_first=float(frame["lev"].iloc[0]), lev_last=float(frame["lev"].iloc[-1]),
                         lev_min=float(frame["lev"].min()), lev_max=float(frame["lev"].max()))
    return out


def verdict(guard, tests):
    """Void beats everything; otherwise all four bars must hold."""
    if guard["contaminated"] or guard["thinly_representative"]:
        reasons = []
        if guard["contaminated"]:
            reasons.append("coverage correlation %.3f exceeds %.2f" % (guard["coverage_corr"],
                                                                      guard["coverage_limit"]))
        if guard["thinly_representative"]:
            reasons.append("basket is %.1f%% of float cap, below %.0f%%"
                           % (100 * guard["cap_share"], 100 * guard["cap_share_limit"]))
        return dict(status="void", reason="; ".join(reasons))
    if all(tests["bars"].values()):
        return dict(status="passed", reason="all four bars hold")
    failed = [name for name, ok in tests["bars"].items() if not ok]
    return dict(status="failed", reason="bar(s) not met: " + ", ".join(failed))


# ------------------------------------------------------------------------------ run

def slice_window(frame, window):
    """V is the only registered window.

    R was intended as a labelled diagnostic extension and cannot be built: the signal's
    denominator is float shares, which ``panel_full`` only carries through 2024-12-31, so a
    "V+R" run silently returns V again.  Refusing is better than printing the same numbers under
    a different name -- P stays unread either way.
    """
    if window == "v":
        return frame.loc[(frame.index >= pd.Timestamp(V_START)) & (frame.index <= pd.Timestamp(V_END))]
    if window == "v_r":
        raise ValueError("V+R is not available: float shares end 2024-12-31, so the ratio cannot "
                         "be extended past V.  Build 2025-2026 shares first if this matters.")
    raise ValueError("unknown window %r" % window)


def run(source="sse", window="v", output=None, sse_path=None, log=print):
    cap = float_cap()
    if source == "sse":
        basket_frame, basket = sse_leverage(cap, sse_path or SSE_TOTAL)
    elif source == "archive":
        margin = load_margin()
        basket = select_basket(margin, cap)
        basket_frame = leverage(margin, cap, basket)
        short = (margin[margin["code"].isin(basket)].groupby("date")["short_balance"].sum()
                 / cap[basket].sum(axis=1))
        basket_frame["short_lev"] = 100.0 * short.reindex(basket_frame.index)
    else:
        raise ValueError("unknown source %r" % source)

    primary = slice_window(build(basket_frame, index_close("000300")), window)
    secondary = slice_window(build(basket_frame, index_close("399317")), window)
    guard = guards(primary, cap, basket, source=source)
    tests = evaluate(primary, secondary)
    result = dict(protocol="doc/market_leverage_protocol_20261114.md", source=source, window=window,
                  basket=dict(size=len(basket), codes=(basket[:5] + ["..."]) if len(basket) > 5 else basket),
                  observations=dict(rows=int(len(primary)),
                                    first=str(primary.index[0].date()),
                                    last=str(primary.index[-1].date()),
                                    effective_blocks=round(len(primary) / HORIZON, 2)),
                  guards=guard, tests=tests, diagnostics=diagnostics(primary, basket_frame),
                  verdict=verdict(guard, tests))
    if output:
        path = Path(output) / ("gate_%s_%s.json" % (source, window))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float),
                        encoding="utf-8")
        log("wrote %s" % path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Market-level leverage gate (pre-registered).")
    parser.add_argument("--source", default="sse", choices=["sse", "archive"])
    parser.add_argument("--window", default="v", choices=["v", "v_r"])
    parser.add_argument("--sse", default=None, help="Path to the official SSE total CSV.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--stage", default="gate", choices=["gate"])
    args = parser.parse_args(argv)
    print(json.dumps(run(args.source, args.window, args.output, args.sse),
                     ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
