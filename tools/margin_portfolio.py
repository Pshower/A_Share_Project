"""The portfolio-level test of the margin-financing signal, per its registration.

[The protocol](doc/margin_portfolio_protocol_20261111.md) fixes everything before this file
runs: the twenty-session change in the financing balance, negated by ranking ascending, a
point-in-time top-1,800 pool, thirty names with a sixty-name buffer, twenty-session holds,
the repository's own broker and the same-pool equal-weight benchmark.

The panel this builds carries **one** feature.  ``e1_reversal.ReversalBook`` ranks whatever
features the manifest declares and buys the lowest composite percentile, so a single feature
whose low end is the desired long leg needs no sign flipping and no new backtest engine.

The window is 2019-2024 because the financing balances start in January 2019.  That window
is V, which this repository has already spent -- the registration says so, and the verdict
is a go/no-go for the shadow run, not out-of-sample evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import file_hash, write_json
from tools import e1_reversal, e2_dynamic_pool, e2_run, margin_ic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
E2 = PROJECT_ROOT / "data" / "clean" / "e2_dynamic1800"
OUTPUT = PROJECT_ROOT / "data" / "clean" / "margin_pool"
REPORT = PROJECT_ROOT / "reports" / "runs" / "margin_pool"
START, END = "2019-01-02", "2024-12-31"


def build(output=OUTPUT, start=START, end=END, log=print):
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to rewrite an existing dataset: %s" % output)
    prices = pd.read_parquet(E2 / "prices.parquet").sort_index()
    dates = prices.index[(prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))]
    codes = [str(code) for code in prices.columns.get_level_values(0).unique()]
    events = pd.read_parquet(margin_ic.CACHE)
    events["date"] = pd.to_datetime(events["date"])
    balance = margin_ic.balance_matrix(events, dates, codes)
    growth = margin_ic.growth_matrix(balance)
    member = e2_dynamic_pool.membership(e2_dynamic_pool.capitalisation()).reindex(dates)
    in_pool = member.reindex(columns=codes).fillna(False).to_numpy(dtype=bool)
    signal = growth.where(pd.DataFrame(in_pool, index=dates, columns=codes))
    log("panel: %d dates, %d codes, %.1f%% of pool-days have a signal" % (
        len(dates), len(codes), 100.0 * float(np.isfinite(signal.to_numpy(dtype=float)).sum()
                                             / max(int(in_pool.sum()), 1))))
    panel = pd.concat({"margin_growth": signal}, axis=1).swaplevel(0, 1, axis=1).sort_index(axis=1)
    panel.columns.names = ["code", "feature"]
    prices = prices.loc[dates]
    output.mkdir(parents=True)
    panel.to_parquet(output / "panel_data.parquet")
    prices.to_parquet(output / "prices.parquet")
    manifest = dict(version=output.name, schema_version=2, price_basis="hfq_research",
                    feature_cols=["margin_growth"], stock_codes=codes,
                    start_date=str(dates.min().date()), end_date=str(dates.max().date()),
                    train_end=str(dates.min().date()), val_end=str(dates.max().date()),
                    protocol="doc/margin_portfolio_protocol_20261111.md",
                    build_config=dict(source="data/online/margin", pool=str(E2)),
                    artifacts={name: file_hash(output / name) for name in
                               ("panel_data.parquet", "prices.parquet")})
    write_json(output / "manifest.json", manifest)
    return manifest


def verdict_for(window, control):
    """The registered rule, in one place so it can be tested without a backtest."""
    v = control["V"]
    verdict = dict(
        net_positive=bool(window["net_excess"] > 0),
        t=float(window["t"]),
        t_ok=bool(window["t"] >= 2.5),
        above_random_median=bool(v["random_median"] is not None
                                 and (window["net_excess"] > v["random_median"])),
        above_p90=bool(v["random_p90"] is not None and (window["net_excess"] > v["random_p90"])),
        random_median=v["random_median"], random_p90=v["random_p90"],
        frictionless_strategy=v["strategy"],
    )
    verdict["passed"] = bool(verdict["net_positive"] and verdict["t_ok"]
                             and verdict["above_random_median"] and verdict["above_p90"])
    return verdict


def run(dataset=OUTPUT, draws=200, log=print):
    payload = e1_reversal.run(dataset=str(dataset))
    window = payload["windows"]["V"]          # the only window the data supports
    manifest = json.loads((Path(dataset) / "manifest.json").read_text(encoding="utf-8"))
    panel = pd.read_parquet(Path(dataset) / "panel_data.parquet")
    prices = pd.read_parquet(Path(dataset) / "prices.parquet")
    control, _, _ = e2_run.control(panel, prices, manifest["feature_cols"], draws=draws, log=log)
    return dict(status="ready", dataset=str(dataset), window="V (2019-2024, spent)", broker=window,
                random_control=control, verdict=verdict_for(window, control))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Portfolio-level test of the margin signal.")
    parser.add_argument("--build", action="store_true", help="Rebuild the panel first.")
    parser.add_argument("--dataset", default=str(OUTPUT))
    parser.add_argument("--output", default=str(REPORT))
    args = parser.parse_args(argv)
    if args.build:
        print(json.dumps({k: v for k, v in build().items() if k in ("version", "start_date", "end_date")},
                         ensure_ascii=False))
    payload = run(Path(args.dataset))
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    (target / "margin_pool.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                                             encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=float)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
