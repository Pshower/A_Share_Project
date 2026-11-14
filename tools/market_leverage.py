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
MACRO_SH = PROJECT_ROOT / "data" / "online" / "margin" / "macro_margin_sh.csv"
MACRO_SZ = PROJECT_ROOT / "data" / "online" / "margin" / "macro_margin_sz.csv"
MACRO_SZ_PATCH = PROJECT_ROOT / "data" / "online" / "margin" / "macro_margin_sz_patch.csv"
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
LONG_START, LONG_END = "2012-01-03", "2026-06-30"
R_END = "2026-06-30"          # P (2026-07 onward) stays untouched
LONG_SUBS = (("2012-01-01", "2018-12-31"), ("2019-01-01", "2026-06-30"))

# The two sources order their columns differently, and reading one with the other's mapping
# silently swaps "daily margin purchases" for "outstanding margin balance" -- a 13x error that
# still produces a plausible-looking series.  Verified against magnitudes on 2026-11-04:
# SSE detail: balance 1.1776e12, buy 9.03e10;  macro: buy 9.03e10, balance 1.1776e12.
SSE_COLUMNS = ("date", "margin_balance", "margin_buy", "short_volume",
               "short_value", "short_sell_volume", "margin_total")
MACRO_COLUMNS = ("date", "margin_buy", "margin_balance", "short_sell_volume",
                 "short_volume", "short_value", "margin_total")


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


def macro_totals(sh_path=MACRO_SH, sz_path=MACRO_SZ, patch_path=MACRO_SZ_PATCH):
    """Both exchanges' official margin totals, 2010 onward -- the repair for the SSE-only run.

    The two files agree with the SSE per-stock detail by exactly zero on their 1,901 overlapping
    days, so the aggregate is the number the exchange publishes; what the macro series adds is
    history back to 2010-03-31 (the 2015 leverage episode) and the Shenzhen book.

    The Shenzhen macro file is missing 2024-10-19 .. 2025-06-18 entirely -- an inner join silently
    truncates the sample there, which is how a 244-day hole can look like "the sample just ends".
    Those days are patched from ``stock_margin_szse(date)``, the exchange's own daily summary
    (reported in 100-million units, converted to yuan), and the patch is counted in the result so
    the provenance of every added day is visible.
    """
    frames = []
    for path, name in ((sh_path, "sh"), (sz_path, "sz")):
        frame = pd.read_csv(path)
        frame.columns = list(MACRO_COLUMNS)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.drop_duplicates("date").set_index("date").sort_index()
        frames.append(frame.add_prefix(name + "_"))
    sz = frames[1]
    patched = 0
    patch_path = Path(patch_path)
    if patch_path.exists():
        extra = pd.read_csv(patch_path)
        extra.columns = list(MACRO_COLUMNS)
        extra["date"] = pd.to_datetime(extra["date"])
        extra = extra.drop_duplicates("date").set_index("date").sort_index().add_prefix("sz_")
        new = extra.loc[extra.index.difference(sz.index)]
        patched = int(len(new))
        if patched:
            sz = pd.concat([sz, new]).sort_index()
    joined = frames[0].join(sz, how="inner")
    joined["margin_balance"] = joined["sh_margin_balance"] + joined["sz_margin_balance"]
    joined["short_value"] = joined["sh_short_value"].fillna(0) + joined["sz_short_value"].fillna(0)
    joined.attrs["sz_patched_days"] = patched
    return joined


def macro_leverage(cap, sh_path=MACRO_SH, sz_path=MACRO_SZ, need_cap=True):
    """Aggregate margin balance, optionally normalised by float cap.

    ``need_cap=False`` keeps only the balance.  That matters: the growth registration tests the
    balance's own 20-day change, which exists from 2010, so joining it to a denominator that
    starts in 2017 would silently throw away 2012-2016 -- including the 2015 episode the whole
    second registration exists to measure.
    """
    totals = macro_totals(sh_path, sz_path)
    frame = pd.DataFrame(dict(balance=totals["margin_balance"]))
    frame.attrs["sz_patched_days"] = totals.attrs.get("sz_patched_days", 0)
    frame = frame.dropna()
    if not need_cap:
        frame["cap"] = np.nan
        frame["lev"] = np.nan
        frame["short_lev"] = np.nan
        frame["codes"] = np.nan
        return frame, sorted(cap.columns)
    value = cap.sum(axis=1)                        # both boards: the total is both exchanges
    frame["cap"] = value
    frame = frame.dropna()
    frame = frame[frame["cap"] > 0]
    frame["lev"] = 100.0 * frame["balance"] / frame["cap"]
    frame["short_lev"] = 100.0 * totals["short_value"].reindex(frame.index) / frame["cap"]
    frame["codes"] = np.nan
    return frame, sorted(cap.columns)


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
    """Signals, targets and the coverage count, inner-joined on the index's trading days.

    Two signals are carried: ``r20`` (leverage intensity, needs a cap denominator and therefore
    only exists from 2017) and ``g20`` (the balance itself, which reaches 2010 and is what the
    second registration tests).  ``past20`` is there for the momentum-disguise guard.
    """
    lev = basket_frame["lev"]
    balance = basket_frame["balance"]
    signal = pd.DataFrame(dict(
        lev=lev,
        balance=balance,
        r20=np.log(lev / lev.shift(horizon)),
        g20=np.log(balance / balance.shift(horizon)),
        past20=np.log(close / close.shift(horizon)).reindex(balance.index),
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
    if source in ("sse", "both"):
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


def evaluate(frame, other_frame, column="r20", subs=SUBS):
    """T1-T4, mechanically, on whichever signal column the registration names."""
    t1 = nw_slope(frame[column], frame["fwd"])
    t2 = []
    for start, end in subs:
        piece = frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]
        row = nw_slope(piece[column], piece["fwd"])
        t2.append(dict(start=start, end=end, slope=row["slope"], t=row["t"], n=row["n"]))
    t3 = nw_slope(frame[column], frame["dd"])
    t4 = nw_slope(other_frame[column], other_frame["fwd"])
    bars = dict(
        T1=bool(np.isfinite(t1["t"]) and t1["slope"] < 0 and t1["t"] <= T1_T),
        T2=bool(all(np.isfinite(row["slope"]) and row["slope"] < 0 for row in t2)),
        T3=bool(np.isfinite(t3["t"]) and t3["slope"] > 0 and t3["t"] >= T3_T),
        T4=bool(np.isfinite(t4["t"]) and t4["slope"] < 0 and t4["t"] <= T1_T),
    )
    return dict(column=column, T1=t1, T2=dict(parts=t2), T3=t3, T4=t4, bars=bars)


EPISODE_START, EPISODE_END = "2014-06-01", "2016-12-31"
EPISODE_DAYS = 20             # sessions with s20 > 0.50 that G1 requires
EPISODE_LEVEL = 0.50
MOMENTUM_LIMIT = 0.85


def guards_growth(frame, threshold=EPISODE_DAYS, level=EPISODE_LEVEL,
                  momentum_limit=MOMENTUM_LIMIT):
    """The growth registration's two guards, judged before any verdict.

    G1 asks whether the window actually contains the 2015 leverage episode -- the only reason
    this signal exists is to test it, so a run that silently misses it must be void.  G2 asks
    whether the signal is past return in disguise: margin balance rises when prices rise, and if
    the correlation is near one then T1 would be a momentum test wearing a different name.
    """
    clean = frame.dropna(subset=["g20", "past20"])
    hot = clean[clean["g20"] > level]
    peak = clean["g20"].idxmax() if len(clean) else None
    in_episode = bool(peak is not None
                      and pd.Timestamp(EPISODE_START) <= peak <= pd.Timestamp(EPISODE_END))
    momentum = float(clean["g20"].corr(clean["past20"])) if len(clean) > 3 else float("nan")
    return dict(episode_days=int(len(hot)), episode_days_required=int(threshold),
                episode_level=level, max_g20=str(peak.date()) if peak is not None else None,
                max_in_episode=in_episode,
                episode_missing=bool(len(hot) < threshold or not in_episode),
                momentum_corr=momentum, momentum_limit=momentum_limit,
                momentum_disguised=bool(np.isfinite(momentum) and momentum > momentum_limit))


def verdict_growth(guard, tests):
    """Void beats everything; otherwise all four bars must hold."""
    if guard["episode_missing"] or guard["momentum_disguised"]:
        reasons = []
        if guard["episode_missing"]:
            reasons.append("window has %d sessions above %.2f (need %d) and max %s"
                           % (guard["episode_days"], guard["episode_level"],
                              guard["episode_days_required"], guard["max_g20"]))
        if guard["momentum_disguised"]:
            reasons.append("corr(s20, past 20d) %.3f exceeds %.2f"
                           % (guard["momentum_corr"], guard["momentum_limit"]))
        return dict(status="void", reason="; ".join(reasons))
    if all(tests["bars"].values()):
        return dict(status="passed", reason="all four bars hold")
    failed = [name for name, ok in tests["bars"].items() if not ok]
    return dict(status="failed", reason="bar(s) not met: " + ", ".join(failed))


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
    if window == "long":
        return frame.loc[(frame.index >= pd.Timestamp(LONG_START))
                         & (frame.index <= pd.Timestamp(LONG_END))]
    if window == "v_r":
        raise ValueError("V+R is not available: float shares end 2024-12-31, so the ratio cannot "
                         "be extended past V.  Build 2025-2026 shares first if this matters.")
    raise ValueError("unknown window %r" % window)


def diagnostics_growth(frame, basket_frame):
    """Labelled diagnostics for the growth signal: other horizons, labelled, no verdict."""
    balance = basket_frame["balance"]
    out = {}
    for name, horizon in (("g5", 5), ("g60", 60)):
        series = np.log(balance / balance.shift(horizon)).reindex(frame.index)
        out[name] = nw_slope(series, frame["fwd"])
    out["levels"] = dict(balance_first=float(frame["balance"].iloc[0]),
                         balance_last=float(frame["balance"].iloc[-1]),
                         g20_min=float(frame["g20"].min()), g20_max=float(frame["g20"].max()))
    return out


def run(source="both", window="v", output=None, sse_path=None, sh_path=None, sz_path=None,
        signal="lev", log=print):
    cap = float_cap()
    if source == "sse":
        basket_frame, basket = sse_leverage(cap, sse_path or SSE_TOTAL)
    elif source == "both":
        basket_frame, basket = macro_leverage(cap, sh_path or MACRO_SH, sz_path or MACRO_SZ,
                                              need_cap=(signal != "growth"))
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
    if signal == "growth":
        guard = guards_growth(primary)
        tests = evaluate(primary, secondary, column="g20",
                         subs=LONG_SUBS if window == "long" else SUBS)
        verdict_row = verdict_growth(guard, tests)
        diagnostics_row = diagnostics_growth(primary, basket_frame)
    else:
        guard = guards(primary, cap, basket, source=source)
        tests = evaluate(primary, secondary, column="r20")
        verdict_row = verdict(guard, tests)
        diagnostics_row = diagnostics(primary, basket_frame)
    # Diagnostic only: 2017-2018 is the design window, never part of V, and the ratio exists
    # there because panel_full starts in 2017.  It is reported and carries no verdict.
    early = build(basket_frame, index_close("000300"))
    early = early.loc[(early.index >= pd.Timestamp("2017-01-01"))
                      & (early.index <= pd.Timestamp("2018-12-31"))]
    pre_v = nw_slope(early["r20"], early["fwd"])
    result = dict(protocol=("doc/market_leverage_growth_protocol_20261114.md" if signal == "growth"
                            else "doc/market_leverage_protocol_20261114.md"),
                  source=source, window=window, signal=signal,
                  basket=dict(size=len(basket), codes=(basket[:5] + ["..."]) if len(basket) > 5 else basket),
                  observations=dict(rows=int(len(primary)),
                                    first=str(primary.index[0].date()),
                                    last=str(primary.index[-1].date()),
                                    effective_blocks=round(len(primary) / HORIZON, 2)),
                  guards=guard, tests=tests,
                  diagnostics=dict(primary=diagnostics_row,
                                   pre_v_2017_2018=dict(n=pre_v["n"], slope=pre_v["slope"],
                                                        t=pre_v["t"], corr=pre_v["corr"],
                                                        note="design window, not V; no verdict")),
                  verdict=verdict_row)
    if output:
        path = Path(output) / ("gate_%s_%s.json" % (source, window))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float),
                        encoding="utf-8")
        log("wrote %s" % path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Market-level leverage gate (pre-registered).")
    parser.add_argument("--source", default="both", choices=["both", "sse", "archive"])
    parser.add_argument("--signal", default="lev", choices=["lev", "growth"])
    parser.add_argument("--window", default="v", choices=["v", "long", "v_r"])
    parser.add_argument("--sse", default=None, help="Path to the official SSE total CSV.")
    parser.add_argument("--sh", default=None, help="Path to the SSE macro total CSV.")
    parser.add_argument("--sz", default=None, help="Path to the SZSE macro total CSV.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--stage", default="gate", choices=["gate"])
    args = parser.parse_args(argv)
    print(json.dumps(run(args.source, args.window, args.output, args.sse, args.sh, args.sz,
                         signal=args.signal), ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
