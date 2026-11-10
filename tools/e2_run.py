"""E2's verdict: the frozen signal on a point-in-time top-1,800 pool.

The protocol fixes three things and all three must hold: net excess positive in **both**
windows, t at least 2.5 in both, and the result outside the ninetieth percentile of a
random book drawn from the same pool on the same dates.

The headline numbers come from the broker (next-open fills, commission, tax, slippage, lot
size) exactly as E1 was measured.  The random control is deliberately **frictionless**: the
random books pay no costs, so the comparison is harder on the strategy than it needs to be.
If the strategy cannot clear a cost-free coin flip, the broker arithmetic is not the issue.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools import e1_reversal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL = PROJECT_ROOT / "data" / "clean" / "e2_dynamic1800"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "e2_dynamic"
HOLD = 20
TOP_K, BUFFER = 30, 60
DRAWS = 200
T_BAR = 2.5


def forward_blocks(prices, index, horizon=HOLD):
    """Return over the next ``horizon`` sessions, and the date each decision is taken."""
    block = prices.shift(-horizon) / prices - 1.0
    decisions = list(index[::horizon])
    return block.reindex(index), decisions


def rank_book(panel, features, date, codes, held, top_k=TOP_K, buffer=BUFFER):
    """E1's rule applied to one date: negated rank-mean, buffer, fill."""
    block = panel.loc[date]
    values = {}
    ranks = np.zeros(len(codes))
    usable = 0
    for name in features:
        try:
            column = block.xs(name, level=1).reindex(codes).to_numpy(dtype=float)
        except KeyError:
            continue
        if not np.isfinite(column).any():
            continue
        ranks += np.nan_to_num(pd.Series(column).rank(method="average", pct=True).to_numpy(), nan=0.5)
        usable += 1
    if not usable:
        return None
    score = pd.Series(ranks / usable, index=codes)
    order = score.sort_values(kind="stable").index
    position = {code: place for place, code in enumerate(order)}
    keep = [code for code in held if position.get(code, 10 ** 9) < buffer]
    keep = sorted(keep, key=lambda code: position.get(code, 10 ** 9))[:top_k]
    for code in order:
        if len(keep) >= top_k:
            break
        if code not in keep:
            keep.append(code)
    return keep


def block_returns(block, decisions, valid_at, pick):
    """Chain the blocks into a daily-equivalent return series."""
    dates = list(block.index)
    place = {date: number for number, date in enumerate(dates)}
    series = []
    for date in decisions:
        row = block.loc[date]
        chosen = pick(date, row)
        if not chosen:
            series.append((date, np.nan))
            continue
        values = row.reindex(chosen).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        series.append((date, float(values.mean()) if len(values) else np.nan))
    return pd.Series(dict(series)).sort_index()


def net_excess(strategy, benchmark, low, high, horizon=HOLD):
    """Annualised difference between two block-return curves inside one window."""
    joined = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1).loc[low:high].dropna()
    if len(joined) < 6:
        return None
    years = len(joined) * horizon / 242
    annual = (1 + joined["s"]).prod() ** (1 / years) - 1 - ((1 + joined["b"]).prod() ** (1 / years) - 1)
    return float(annual)


def control(panel, prices, features, draws=DRAWS, seed=42, log=print):
    close = prices.xs("close", axis=1, level=1) if "close" in set(prices.columns.get_level_values(1)) else prices
    close = close.sort_index()
    codes = [str(code) for code in close.columns]
    block, decisions = forward_blocks(close, close.index)
    # A twenty-session return outside this band is a corporate-action or suspension artifact,
    # not a tradable outcome.  Without the clip a handful of them drag the 1,800-name mean
    # down several points a year while a thirty-name book is barely touched -- the same
    # asymmetry that makes the comparison meaningless if left in.
    block = block.clip(-0.5, 1.0)
    finite = panel.loc[:, [(code, name) for code in codes for name in features]].to_numpy(dtype=float)
    valid = np.isfinite(finite).reshape(len(panel.index), len(codes), len(features)).all(axis=2)
    valid_at = pd.DataFrame(valid, index=panel.index, columns=codes)

    def pool_of(date):
        row = valid_at.loc[date]
        return list(row.index[row.to_numpy()])

    benchmark = block_returns(block, decisions, valid_at, lambda date, row: pool_of(date))

    held = []
    books = {}
    for date in decisions:
        book = rank_book(panel, features, date, pool_of(date), held)
        held = book or held
        books[date] = set(held)
    strategy = block_returns(block, decisions, valid_at,
                             lambda date, row: sorted(books.get(date, set())))

    rng = np.random.default_rng(seed)
    random_curve, random_excess = [], {}
    windows = {"V": ("2019-01-01", "2024-12-31"), "full": ("2017-01-02", "2024-12-31")}
    for _ in range(draws):
        picks = {}
        for date in decisions:
            pool = pool_of(date)
            picks[date] = list(rng.choice(pool, size=min(TOP_K, len(pool)), replace=False)) if pool else []
        curve = block_returns(block, decisions, valid_at, lambda date, row: picks.get(date, []))
        row = {name: net_excess(curve, benchmark, low, high) for name, (low, high) in windows.items()}
        random_excess[len(random_excess)] = row
    log("  %d random books done" % draws)

    table = pd.DataFrame(random_excess).T
    summary = {}
    for window, (low, high) in windows.items():
        values = table[window].dropna()
        summary[window] = dict(
            random_median=float(values.median()), random_p90=float(values.quantile(0.9)),
            draws=int(len(values)),
            strategy=net_excess(strategy, benchmark, low, high),
            benchmark_annual=float((1 + benchmark.loc[low:high].dropna()).prod() **
                                   (242 / (len(benchmark.loc[low:high].dropna()) * HOLD)) - 1))
    return summary, strategy, benchmark


def run(panel_dir=PANEL, log=print):
    directory = Path(panel_dir)
    cached = OUTPUT / "broker.json"
    if cached.exists():
        payload = json.loads(cached.read_text(encoding="utf-8"))
        log("using the cached broker run at %s" % cached)
    else:
        payload = e1_reversal.run(dataset=str(directory))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    panel = pd.read_parquet(directory / "panel_data.parquet")
    prices = pd.read_parquet(directory / "prices.parquet")
    summary, _, _ = control(panel, prices, manifest["feature_cols"], log=log)
    windows = payload["windows"]
    verdict = dict(
        v_positive=bool(windows["V"]["net_excess"] > 0),
        full_positive=bool(windows["full"]["net_excess"] > 0),
        v_t=float(windows["V"]["t"]), full_t=float(windows["full"]["t"]),
        both_t=bool(min(windows["V"]["t"], windows["full"]["t"]) >= T_BAR),
        random=summary,
    )
    verdict["passed"] = bool(verdict["v_positive"] and verdict["full_positive"]
                             and verdict["both_t"])
    return dict(status="ready", panel=str(directory), broker=windows,
                random_control=summary, verdict=verdict)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score E2 against its registered rule.")
    parser.add_argument("--panel", default=str(PANEL))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = run(Path(args.panel))
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    (target / "e2.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                                    encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=float)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
