"""The unified strategy, simulated: one rule over cash, deposits, four bond durations, three
equity markets and gold -- the protocol's candidates, and its five adoption criteria.

Design notes that matter for honesty:

* **Month-end decision, next-session fill.**  The CPPI target is computed at a month-end close and
  the book is traded at the *next* close, so no signal is executed at the price that produced it.
* **Drawdown is marked daily**, not monthly; a rule that only looks monthly can hide a breach.
* **Paired comparisons.**  Rules that start later (gold 2017, Hong Kong 2014) are compared with
  the baseline on *their own* start dates, never on a longer window that would flatter them.
* **Costs are charged on turnover**, plus a per-sleeve annual fee, both written into the output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tools import asset_panel as panel
from tools import plan_orders

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "unified_plan"

FLOOR, MULTIPLIER, BAND = plan_orders.FLOOR, plan_orders.MULTIPLIER, plan_orders.BAND
COST = 0.0010                                     # one-way, per unit of weight traded
FEES = {"cash": 0.0010, "deposit": 0.0010, "bond2": 0.0015, "bond5": 0.0015, "bond9": 0.0015,
        "bond20": 0.0015, "ashare": 0.0030, "asmall": 0.0030, "usshare": 0.0030, "hk": 0.0030,
        "gold": 0.0050}
MIN_MONTHS = 36

# The live plan's risky sleeve: 25% CSI 300 + 25% CSI 500 + 50% S&P 500 (plan_orders.LEGS).
RISK_MIX = {"ashare": 0.25, "asmall": 0.25, "usshare": 0.50}
RULES = {
    "B0": dict(kind="static", weights={"bond5": 1.0}, note="all government bonds, duration 5"),
    "B1": dict(kind="static", weights={"ashare": 0.6, "bond5": 0.4}, note="classic 60/40"),
    "B2": dict(kind="static", weights={"ashare": 0.125, "asmall": 0.125, "usshare": 0.25,
                                       "bond5": 0.5},
               note="static version of the current plan"),
    "C1": dict(kind="cppi", risky=dict(RISK_MIX), safe={"bond5": 1.0}, note="the current plan"),
    "C2": dict(kind="cppi", risky=dict(RISK_MIX), safe={"cash": 0.5, "bond5": 0.5},
               note="half the safe leg in cash"),
    "C3": dict(kind="cppi", risky=dict(RISK_MIX), safe="switch",
               note="safe leg by the ten-year yield"),
    "C6": dict(kind="cppi", risky=dict(RISK_MIX), safe={"cash": 0.5, "deposit": 0.5},
               note="safe leg entirely in deposits and cash (added after the first sweep)"),
    "C4": dict(kind="cppi", risky={"ashare": 0.4, "usshare": 0.4, "gold": 0.2},
               safe={"bond5": 1.0}, note="gold in the risky leg"),
    "C5": dict(kind="cppi", risky={"ashare": 1 / 3, "usshare": 1 / 3, "hk": 1 / 3},
               safe={"bond5": 1.0}, note="Hong Kong in the risky leg"),
    "D2": dict(kind="cppi", risky=dict(RISK_MIX), safe={"bond2": 1.0}, note="duration 2"),
    "D9": dict(kind="cppi", risky=dict(RISK_MIX), safe={"bond9": 1.0}, note="duration 9"),
    "D20": dict(kind="cppi", risky=dict(RISK_MIX), safe={"bond20": 1.0}, note="duration 20"),
}


def sleeves(rule: dict) -> list[str]:
    if rule["kind"] == "static":
        return list(rule["weights"])
    safe = rule["safe"] if isinstance(rule["safe"], dict) else {"cash": 1.0, "bond5": 1.0}
    return list(dict.fromkeys([*rule["risky"], *safe]))


def target_weights(rule: dict, nav: float, peak: float, floor: float, multiplier: float,
                   y10: float) -> dict:
    """The risky share from the CPPI cushion, split across the sleeves the rule names."""
    if rule["kind"] == "static":
        return dict(rule["weights"])
    share = float(min(max(multiplier * (nav - floor * peak) / nav, 0.0), 1.0))
    safe = rule["safe"]
    if safe == "switch":
        safe = {"bond5": 1.0} if y10 > 2.5 else ({"cash": 1.0} if y10 < 1.2
                                                 else {"cash": 0.5, "bond5": 0.5})
    weights: dict[str, float] = {}
    for name, part in rule["risky"].items():
        weights[name] = weights.get(name, 0.0) + share * part
    for name, part in safe.items():
        weights[name] = weights.get(name, 0.0) + (1.0 - share) * part
    return weights


def simulate(levels: pd.DataFrame, rule: dict, start_pos: int, floor: float = FLOOR,
             multiplier: float = MULTIPLIER, band: float = BAND,
             y10: np.ndarray | None = None, cadence: str = "monthly") -> dict | None:
    """One start date.  Returns CAGR, worst daily drawdown and annualised turnover."""
    names = sleeves(rule)
    frame = levels[names].iloc[start_pos:]
    if frame.isna().any().any():
        return None
    returns = frame.pct_change().fillna(0.0).to_numpy()
    fees = np.array([FEES[name] for name in names])
    step = (1.0 + returns) * (1.0 - fees / 252.0)
    cum = np.vstack([np.ones(len(names)), np.cumprod(step, axis=0)])     # cum[t] = value at t
    index = frame.index
    n = len(index)
    month_end = np.zeros(n, dtype=bool)
    month_end[:-1] = index[:-1].month != index[1:].month
    month_end[-1] = True

    def y_at(day: int) -> float:
        if y10 is None:
            return 3.0
        value = y10[start_pos + day]
        return 3.0 if not np.isfinite(value) else float(value)

    weights = target_weights(rule, 1.0, 1.0, floor, multiplier, y_at(0))
    book = np.array([weights.get(name, 0.0) for name in names])
    cost = COST * float(np.abs(book).sum())
    book = book * (1.0 - cost)
    nav = float(book.sum())
    peak = nav
    worst = 0.0
    turnover = 0.0
    day = 0
    final = nav
    while True:
        end = day if cadence == "daily" else day + int(np.argmax(month_end[day:]))
        # ``cum[t]`` is one unit bought before day 0 and marked at day ``t``, while ``returns[t]``
        # is the move *into* day t.  So the value on any day in [day, end] is
        # ``book * cum[t + 1] / cum[day + 1]``; using ``cum[t] / cum[day]`` loses one day, and
        # adding a second day for the fill compounds every day twice (7.9% became 15.3%).
        base = day + 1
        navs = (book * (cum[base:end + 2] / cum[base])).sum(axis=1)
        running = np.maximum.accumulate(np.maximum(navs, 1e-12))
        running = np.maximum(running, peak)
        worst = min(worst, float((navs / running - 1.0).min()))
        peak = max(peak, float(running[-1]))
        if end + 1 >= n:                                     # nothing left to trade into
            final = float(navs[-1])
            break
        book = book * (cum[end + 2] / cum[base])             # the next close is the fill
        nav = float(book.sum())
        peak = max(peak, nav)
        wanted = target_weights(rule, nav, peak, floor, multiplier, y_at(end + 1))
        target = np.array([wanted.get(name, 0.0) for name in names])
        delta = float(np.abs(target - book / nav).sum())
        if delta / 2.0 >= band:
            turnover += delta / 2.0
            book = target * nav * (1.0 - COST * delta)
        day = end + 1

    years = max((index[-1] - index[0]).days / 365.25, 1e-9)
    return dict(cagr=float(final ** (1.0 / years) - 1.0), maxdd=worst, final=final,
                turnover=turnover / years, months=years * 12.0)


def month_starts(index: pd.DatetimeIndex) -> list[int]:
    firsts: dict = {}
    for position, period in enumerate(index.to_period("M")):
        firsts.setdefault(period, position)
    return sorted(firsts.values())


def sweep(levels: pd.DataFrame, rules: dict | None = None, step_months: int = 1,
          floor: float = FLOOR, multiplier: float = MULTIPLIER,
          cadence: str = "monthly") -> tuple[pd.DataFrame, dict]:
    rules = rules or RULES
    y10 = panel.yields()["y10"].reindex(levels.index).ffill().to_numpy()
    starts = [position for order, position in enumerate(month_starts(levels.index))
              if order % step_months == 0 and position < len(levels) - MIN_MONTHS * 21]
    rows = []
    for name, rule in rules.items():
        for start in starts:
            result = simulate(levels, rule, start, floor=floor, multiplier=multiplier,
                              y10=y10, cadence=cadence)
            if result and result["months"] >= MIN_MONTHS:
                rows.append(dict(rule=name, start=str(levels.index[start].date()),
                                 cagr=result["cagr"], maxdd=result["maxdd"],
                                 turnover=result["turnover"]))
    runs = pd.DataFrame(rows)
    summary = {}
    for name, rule in rules.items():
        frame = runs[runs["rule"] == name]
        if not len(frame):
            continue
        summary[name] = dict(n=len(frame), median_cagr=float(frame["cagr"].median()),
                             worst_drawdown=float(frame["maxdd"].min()),
                             hit=float(((frame["cagr"] >= 0.07)
                                        & (frame["maxdd"] >= -0.10)).mean()),
                             median_turnover=float(frame["turnover"].median()),
                             note=rule.get("note", ""))
    return runs, summary


def paired(runs: pd.DataFrame, summary: dict, baseline: str = "C1") -> dict:
    """Re-read the baseline on each rule's own start dates, so a later-starting rule is never
    compared with a baseline measured over a longer, easier window."""
    out = {}
    base = runs[runs["rule"] == baseline].set_index("start")["cagr"]
    for name, row in summary.items():
        mine = runs[runs["rule"] == name].set_index("start")["cagr"]
        common = mine.index.intersection(base.index)
        if not len(common):
            continue
        out[name] = dict(starts=len(common),
                         own=round(float(mine.loc[common].median()), 4),
                         baseline=round(float(base.loc[common].median()), 4),
                         delta_pp=round(float(mine.loc[common].median()
                                              - base.loc[common].median()) * 100, 2))
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    step = 3 if "--quick" in argv else 1
    cadence = "daily" if "--daily" in argv else "monthly"
    since = None
    only = None
    for argument in argv:
        if argument.startswith("--since="):
            since = argument.split("=", 1)[1]
        if argument.startswith("--rules="):
            only = {name.strip().upper() for name in argument.split("=", 1)[1].split(",")}
    levels = panel.panel(start=since) if since else panel.panel()
    rules = {name: rule for name, rule in RULES.items() if only is None or name in only}
    runs, summary = sweep(levels, rules=rules, step_months=step, cadence=cadence)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tag = cadence + ("_" + since.replace("-", "") if since else "")
    runs.to_csv(OUTPUT / f"runs_{tag}.csv", index=False, float_format="%.6f")
    paired_rows = paired(runs, summary)
    payload = dict(rules=summary, paired=paired_rows, floor=FLOOR, multiplier=MULTIPLIER,
                   band=BAND, cost=COST, fees=FEES, min_months=MIN_MONTHS, cadence=cadence,
                   since=since or str(levels.index[0].date()), step_months=step,
                   generated_at=str(pd.Timestamp.now())[:19])
    (OUTPUT / f"summary_{tag}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"{'rule':<5} {'n':>4} {'median':>8} {'worstDD':>8} {'hit':>6} {'turn':>6} "
          f"{'vs C1':>7}  note")
    for name, row in summary.items():
        delta = paired_rows.get(name, {}).get("delta_pp", float("nan"))
        print(f"{name:<5} {row['n']:>4} {row['median_cagr']:>7.2%} {row['worst_drawdown']:>8.2%} "
              f"{row['hit']:>6.1%} {row['median_turnover']:>6.2f} {delta:>+7.2f}  {row['note']}")
    print(f"wrote {OUTPUT / f'summary_{tag}.json'}  (cadence {cadence}, since {payload['since']}, "
          f"step {step})")
    return 0


if __name__ == "__main__":
    sys.exit(main())






