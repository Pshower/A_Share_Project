"""Tail-risk and regime re-check of equity-bond diversification on the plan's four legs.

Pre-registered in ``doc/equity_bond_tail_protocol_20261123.md`` *before* the first full run; the
external anchor is arXiv:2611.21897, which found on 30 global equity-bond ETFs (2020-2024) that
rolling min-var and CVaR optimisation cut empirical VaR/CVaR below an equally weighted benchmark
while the benchmark kept the best Sharpe/Sortino/Rachev.  The repo had never tested rolling
optimised *weights* on its own menu -- allocation_robustness covered rebalance cadence, cash
fraction and volatility targeting only.  This tool runs five long-only arms on the four traded
legs (510300/510500/513500/511010), from the same monthly-start grid as ``plan_backtest``,
bucketed by the ten-year yield at start with the ``daily_ops`` boundaries, and computes the three
frozen verdicts mechanically so no table downstream can be cherry-picked.

Design choices that are deviations, all pre-registered:

* index-level simulation (no lots, no 5-yuan fee floor, no no-trade band) -- numbers are for
  *relative* arm comparison, not absolute expectations;
* both optimised arms search the same 10%-step simplex grid (286 points) on the same rebalance
  days with the same costs, so the comparison isolates the objective, not the solver;
* costs are 13bp per side (0.03% commission + 10bp slippage) on one-sided turnover, the model
  ``plan_backtest`` uses at this notional.

The 510500 distribution of 2022-08-29 (0.92 per unit) is missing from the interface's payout
table; ``plan_backtest`` documents it as real.  It is applied as a *manual, pre-registered* payout
by default and the whole sweep is re-run without it as a sensitivity row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools import asset_panel, plan_backtest as pb
from tools.plan_backtest import month_starts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONLINE = PROJECT_ROOT / "data" / "online"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "equity_bond_tail"

CODES = ["510300", "510500", "513500", "511010"]
# The two frozen static arms: the paper's benchmark and the plan's own leg weights.
STATIC_WEIGHTS = {
    "eqw": {"510300": 0.25, "510500": 0.25, "513500": 0.25, "511010": 0.25},
    "fixed6040": {"510300": 0.15, "510500": 0.15, "513500": 0.30, "511010": 0.40},
}
STATIC_ARMS = tuple(STATIC_WEIGHTS)
DYNAMIC_ARMS = ("invvol", "minvar", "cvar")
ARMS = STATIC_ARMS + DYNAMIC_ARMS
REBALANCE = {"eqw": 252, "fixed6040": 252, "invvol": 63, "minvar": 63, "cvar": 63}
ESTIMATE_WINDOW = 504                    # trading days of returns fed to the dynamic arms
GRID_STEP = 0.10                        # simplex step for the optimised arms
TAIL = 0.05                             # CVaR tail share
COST_PER_SIDE = 0.0013                  # 0.03% commission + 10bp slippage
ANNUAL = 242                            # repo convention: trading steps per year
START_FLOOR = "2016-01-04"              # same monthly grid as plan_backtest
H1_SHARE = 0.70                         # frozen: direction agreement across starts
H2_EDGE = 0.05                          # frozen: material Sharpe median edge
H2_SHARE = 0.60                         # frozen: share of starts above the benchmark
MANUAL_PAYOUTS = {("510500", "2022-08-29"): 0.92}


# ---------------------------------------------------------------- data

def local_history(code: str) -> pd.Series:
    """Raw closes from the offline cache -- never the network."""
    frame = pd.read_csv(ONLINE / "fund" / f"hist_{code}.csv", encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")["close"].astype(float).sort_index()


def payout_table(code: str, include_manual: bool = True) -> dict:
    """Recorded per-unit payouts; optionally plus the pre-registered manual repair.

    The manual entry is only ever *added* where the published table is missing it, so a future
    cache that gains the real row is never overridden.
    """
    path = ONLINE / "fund" / f"dist_{code}.csv"
    paid: dict = {}
    if path.exists():
        frame = pd.read_csv(path, encoding="utf-8-sig")
        for _, row in frame.iterrows():
            paid[pd.Timestamp(row["date"])] = float(row["per_unit"])
    if include_manual:
        for (leg, day), amount in MANUAL_PAYOUTS.items():
            if leg == code:
                stamp = pd.Timestamp(day)
                paid.setdefault(stamp, float(amount))
    return paid


def total_return_legs(include_manual: bool = True, log=print):
    """Split-repaired, distribution-reinvested indices for the four legs, on their intersection.

    The repair rules are ``plan_backtest.total_return``'s, pinned by its own tests: only a gap
    landing on a simple split ratio is a split, only the published payout table counts as a
    distribution, every other fall stays.
    """
    index, repairs, manual = {}, {}, {}
    for code in CODES:
        prices = local_history(code)
        paid = payout_table(code, include_manual)
        index[code], events = pb.total_return(prices, paid)
        if events:
            repairs[code] = events
        for (leg, day), amount in MANUAL_PAYOUTS.items():
            if leg == code and pd.Timestamp(day) in paid:
                manual[leg] = dict(day=day, per_unit=amount)
    frame = pd.DataFrame(index).sort_index().dropna()
    return frame, repairs, manual


def yield_buckets(starts) -> dict:
    """Ten-year yield at each start, binned with the ``daily_ops`` REGIMES boundaries."""
    from tools.daily_ops import REGIMES            # same tuples, no drift possible

    curve = asset_panel.yields()["y10"].dropna()
    buckets = {}
    for start in starts:
        level = float(curve.reindex([pd.Timestamp(start)], method="ffill").iloc[0])
        label = "unbinned"
        for low, high, name, *_ in REGIMES:
            if low <= level < high:
                label = name
                break
        buckets[str(pd.Timestamp(start).date())] = dict(y10=level, bucket=label)
    return buckets


# ---------------------------------------------------------------- arms

def simplex_grid(step: float = GRID_STEP) -> np.ndarray:
    """All long-only weight vectors on the ``step`` simplex, four assets."""
    units = int(round(1.0 / step))
    points = []
    for a in range(units + 1):
        for b in range(units + 1 - a):
            for c in range(units + 1 - a - b):
                points.append((a / units, b / units, c / units, (units - a - b - c) / units))
    return np.array(points, dtype=float)


GRID = simplex_grid()


def static_target(arm: str) -> np.ndarray:
    return np.array([STATIC_WEIGHTS[arm][code] for code in CODES], dtype=float)


def estimate_window(returns: np.ndarray, pos: int) -> np.ndarray:
    """The ``ESTIMATE_WINDOW`` daily returns ending at ``pos`` (inclusive), for PIT-at-close."""
    lo = max(pos - ESTIMATE_WINDOW + 1, 0)
    return returns[lo:pos + 1]


def dynamic_target(arm: str, window: np.ndarray) -> np.ndarray:
    """Re-estimated weights on the same grid for every optimised arm.

    ``invvol`` is off-grid on purpose: it is the cheap "simple" candidate, not an optimiser.
    """
    if arm == "invvol":
        vol = window.std(axis=0, ddof=1)
        weights = 1.0 / vol
        return weights / weights.sum()
    portfolios = GRID @ window.T                       # (points, days)
    if arm == "minvar":
        return GRID[int(np.argmin(portfolios.var(axis=1, ddof=1)))]
    depth = max(int(portfolios.shape[1] * TAIL), 1)
    worst = np.partition(portfolios, depth - 1, axis=1)[:, :depth]
    return GRID[int(np.argmax(worst.mean(axis=1)))]    # least-negative CVaR


def target_for(arm: str, returns: np.ndarray, pos: int) -> np.ndarray:
    if arm in STATIC_ARMS:
        return static_target(arm)
    return dynamic_target(arm, estimate_window(returns, pos))


# ---------------------------------------------------------------- simulation

def run_arm(arm: str, frame: pd.DataFrame, start: pd.Timestamp):
    """Index-level book: establish at the start close, drift daily, rebalance on cadence.

    Convention matches ``plan_backtest.simulate``: the decision uses closes through day ``t``
    and trades at ``t``'s close, so the cost bite lands in ``t``'s NAV and the next day's return
    accrues to the new weights.
    """
    index = frame.index
    begin = index.get_loc(start)
    days = len(index) - begin
    if days < 30:
        return None
    returns = frame.pct_change().to_numpy()
    returns[0] = 0.0
    cadence = REBALANCE[arm]
    nav, turnover, cost = 1.0, 0.0, 0.0
    weights = None
    nav_rows = []
    for step in range(days):
        pos = begin + step
        if step == 0:
            moved = 1.0                               # cash -> fully invested
            nav *= 1.0 - COST_PER_SIDE * moved
            turnover += moved
            cost += COST_PER_SIDE * moved
            weights = target_for(arm, returns, pos)
            nav_rows.append(nav)
            continue
        row = returns[pos]
        growth = 1.0 + float(np.dot(weights, row))
        nav *= growth
        weights = weights * (1.0 + row) / growth
        if step % cadence == 0:
            target = target_for(arm, returns, pos)
            moved = float(np.abs(target - weights).sum())
            if moved > 0:
                nav *= 1.0 - COST_PER_SIDE * moved
                turnover += moved
                cost += COST_PER_SIDE * moved
            weights = target
        nav_rows.append(nav)
    series = pd.Series(nav_rows, index=index[begin:begin + days])
    return dict(series=series, turnover=turnover, cost=cost)


def metrics(series: pd.Series, turnover: float, cost: float) -> dict:
    """The frozen metric set; every definition is in the protocol, none is tunable here."""
    daily = series.pct_change().dropna()
    steps = len(series) - 1
    years = steps / ANNUAL
    final = float(series.iloc[-1])
    vol = float(daily.std(ddof=1))
    sharpe = float(daily.mean() / vol * np.sqrt(ANNUAL)) if vol > 0 else float("nan")
    downside = np.minimum(daily.to_numpy(), 0.0)
    dstd = float(np.sqrt(float(np.mean(downside ** 2))))
    sortino = float(daily.mean() / dstd * np.sqrt(ANNUAL)) if dstd > 0 else float("nan")
    var95 = float(np.percentile(daily, 5))
    cvar95 = float(daily[daily <= var95].mean())
    gain95 = float(daily[daily >= np.percentile(daily, 95)].mean())
    return dict(
        cagr=float(final ** (ANNUAL / steps) - 1) if steps > 0 else float("nan"),
        vol=vol * np.sqrt(ANNUAL),
        sharpe=sharpe,
        sortino=sortino,
        drawdown=float((series / series.cummax() - 1.0).min()),
        var95=var95,
        cvar95=cvar95,
        rachev=float(gain95 / abs(cvar95)) if cvar95 < 0 else float("inf"),
        turnover_per_year=float(turnover / years),
        cost_per_year=float(cost / years),
        final=final,
        steps=steps,
    )


# ---------------------------------------------------------------- sweep and verdicts

def feasible_starts(frame: pd.DataFrame, floor: str = START_FLOOR) -> list:
    """Monthly first-trading-day starts on the plan_backtest grid that own a full warmup."""
    monthly = month_starts(frame.loc[floor:].index)
    return [stamp for stamp in monthly
            if frame.index.get_loc(stamp) >= ESTIMATE_WINDOW - 1]


def sweep(frame: pd.DataFrame, starts: list, arms=ARMS) -> list:
    rows = []
    for start in starts:
        for arm in arms:
            result = run_arm(arm, frame, start)
            if result is None:
                continue
            row = metrics(result["series"], result["turnover"], result["cost"])
            row.update(arm=arm, start=str(pd.Timestamp(start).date()))
            rows.append(row)
    return rows


def _by_arm(rows: list) -> dict:
    table = {}
    for row in rows:
        table.setdefault(row["arm"], {})[row["start"]] = row
    return table


def _median(table: dict, arm: str, field: str, starts=None) -> float:
    keys = starts if starts is not None else sorted(table.get(arm, {}))
    values = [table[arm][key][field] for key in keys if key in table.get(arm, {})]
    return float(np.median(values)) if values else float("nan")


def _share_above(table: dict, arm: str, field: str, starts=None) -> float:
    keys = starts if starts is not None else sorted(table.get(arm, {}))
    hits = [key for key in keys
            if key in table.get(arm, {}) and key in table.get("eqw", {})
            and table[arm][key][field] > table["eqw"][key][field]]
    return len(hits) / len(keys) if keys else float("nan")


def verdicts(rows: list, buckets: dict) -> dict:
    """The three frozen checks.  Mechanical on purpose: the doc may only quote these numbers."""
    table = _by_arm(rows)
    low = sorted(key for key, info in buckets.items() if info["bucket"] == "低 carry")

    # CVaR95 is a negative number: a *shallower* tail is a value closer to zero, i.e. numerically
    # GREATER.  The frozen rule is therefore "> eqw" on the median and "above eqw" per start --
    # the paper's "lower empirical CVaR" is lower in loss magnitude, not lower in signed return.
    h1 = dict(
        minvar_cvar95=_median(table, "minvar", "cvar95"),
        cvar_cvar95=_median(table, "cvar", "cvar95"),
        eqw_cvar95=_median(table, "eqw", "cvar95"),
        minvar_share=_share_above(table, "minvar", "cvar95"),
        cvar_share=_share_above(table, "cvar", "cvar95"),
    )
    h1["pass"] = bool(h1["minvar_cvar95"] > h1["eqw_cvar95"]
                      and h1["cvar_cvar95"] > h1["eqw_cvar95"]
                      and h1["minvar_share"] >= H1_SHARE
                      and h1["cvar_share"] >= H1_SHARE)

    arms = {}
    for arm in ("minvar", "cvar"):
        arms[arm] = dict(
            sharpe=_median(table, arm, "sharpe"),
            eqw_sharpe=_median(table, "eqw", "sharpe"),
            share_above=_share_above(table, arm, "sharpe"),
        )
        arms[arm]["wins"] = bool(arms[arm]["sharpe"] - arms[arm]["eqw_sharpe"] > H2_EDGE
                                 and arms[arm]["share_above"] >= H2_SHARE)
    h2 = dict(arms=arms)
    h2["pass"] = not any(entry["wins"] for entry in arms.values())

    h3 = dict(low_starts=len(low),
              fixed6040_cagr=_median(table, "fixed6040", "cagr", low))
    for arm in ("minvar", "cvar", "invvol"):
        h3[arm + "_cagr"] = _median(table, arm, "cagr", low)
    h3["pass"] = bool(h3["fixed6040_cagr"] >= h3["minvar_cagr"]
                      and h3["fixed6040_cagr"] >= h3["cvar_cagr"]
                      and h3["fixed6040_cagr"] >= h3["invvol_cagr"])
    return dict(H1=h1, H2=h2, H3=h3)


def arm_summary(rows: list, buckets: dict) -> dict:
    """Medians per arm overall and per yield bucket -- descriptive, carries no verdict."""
    table = _by_arm(rows)
    summary = {}
    for arm in table:
        entry = dict(runs=len(table[arm]))
        for field in ("cagr", "vol", "sharpe", "sortino", "drawdown",
                      "var95", "cvar95", "rachev", "turnover_per_year", "cost_per_year"):
            entry[field] = _median(table, arm, field)
        for label in ("低 carry", "中 carry", "高 carry"):
            keys = sorted(key for key, info in buckets.items() if info["bucket"] == label)
            entry[label] = dict(
                starts=len([key for key in keys if key in table[arm]]),
                cagr=_median(table, arm, "cagr", keys),
                drawdown=_median(table, arm, "drawdown", keys),
                sharpe=_median(table, arm, "sharpe", keys),
            )
        summary[arm] = entry
    return summary


# ---------------------------------------------------------------- entry point

def main(argv=None):
    parser = argparse.ArgumentParser(description="Equity-bond tail re-check, pre-registered.")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)

    runs = {}
    for include_manual in (True, False):
        frame, repairs, manual = total_return_legs(include_manual=include_manual)
        starts = feasible_starts(frame)
        buckets = yield_buckets(starts)
        rows = sweep(frame, starts)
        runs["with_manual" if include_manual else "without_manual"] = dict(
            window=dict(first=str(frame.index[0].date()), last=str(frame.index[-1].date()),
                        starts=len(starts)),
            repairs=repairs, manual=manual, buckets=buckets, rows=rows,
            verdicts=verdicts(rows, buckets), summary=arm_summary(rows, buckets))
        log_runs = runs["with_manual" if include_manual else "without_manual"]
        print("%s: %d starts, %d rows" % ("含手动修复" if include_manual else "不含手动修复",
                                         log_runs["window"]["starts"], len(log_runs["rows"])))

    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    payload = dict(status="ready", codes=CODES, arms=list(ARMS),
                   estimate_window=ESTIMATE_WINDOW, rebalance=REBALANCE,
                   grid_points=len(GRID), grid_step=GRID_STEP, tail=TAIL,
                   cost_per_side=COST_PER_SIDE, annual_steps=ANNUAL,
                   start_floor=START_FLOOR, thresholds=dict(h1_share=H1_SHARE,
                                                            h2_edge=H2_EDGE, h2_share=H2_SHARE),
                   manual_payouts={f"{leg}:{day}": value for (leg, day), value in MANUAL_PAYOUTS.items()},
                   runs=runs)
    (target / "equity_bond_tail.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    for name, block in runs.items():
        print("\n=== %s ===" % name)
        print(json.dumps(block["verdicts"], ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
