"""How much of the micro-cap tail's premium survives the names the panel is missing.

A-line study found the three investable indices cover roughly the first 1,800 names and
return about 5.3% a year, while the all-share index returns 8.0%.  The difference has to
come from the tail, which returned 13% to 15% over 2019-2024.  That number comes from a
panel that is missing 515 universe codes -- 9.2% of codes but only 1.44% of float
capitalisation, and every one of them sits in the tail.  So the tail's measured return is
computed on a universe that has been relieved of its losers.

There is no way to recover those names: the universe snapshot is today's list, and a
delisted stock was never downloaded.  What can be done is to state the sensitivity -- if
the missing names returned this, how much does the tail fall -- and then judge whether
any conclusion survives the whole plausible band.

Band membership is drawn from the **lagged** capitalisation at each rebalance, so the
look-ahead that produced a 7.92pp artefact in the earlier study is absent here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tools import capm_backtest as capm
from tools.full_market_panel import load

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL = PROJECT_ROOT / "data" / "clean" / "panel_full"
UNIVERSE = PROJECT_ROOT / "data" / "online" / "universe" / "a_share_spot.json"
TAIL_START = 1800
START, END = "2019-01-02", "2024-12-31"
STEP = 20
SCENARIOS = (0.0, -0.30, -0.60)   # assumed annual return of the missing names


def universe_gap(ready_codes):
    """Codes in the snapshot that hold no history at all, and their capitalisation share."""
    rows = json.loads(UNIVERSE.read_text(encoding="utf-8"))["rows"]
    weight = {row["code"]: float(row.get("nmc") or 0.0) for row in rows}
    total = sum(weight.values()) or 1.0
    missing = [code for code in weight if code not in ready_codes]
    return dict(universe=len(rows), missing=len(missing),
                missing_share=len(missing) / len(rows),
                missing_capitalisation=sum(weight[code] for code in missing) / total)


def tail_returns(panel_dir=PANEL, lo=TAIL_START, start=START, end=END, step=STEP):
    """Equal- and cap-weighted return of the tail, membership from the lagged cap."""
    manifest, prices, traded, shares = load(panel_dir)
    caps = traded * shares
    returns = capm.daily_frame(prices)
    window = returns.loc[start:end]
    dated = caps.reindex(window.index).ffill()
    marks = set(range(0, len(window), step))
    equal_series, cap_series, sizes = [], [], []
    band = []
    for position in range(len(window)):
        if position in marks:
            asof = dated.iloc[max(position - 1, 0)].dropna()
            band = list(asof.sort_values(ascending=False).index[lo:])
            sizes.append(len(band))
        row = window.iloc[position]
        members = [code for code in band if np.isfinite(row.get(code, np.nan))]
        if len(members) < 30:
            equal_series.append(np.nan)
            cap_series.append(np.nan)
            continue
        values = row[members].to_numpy(dtype=float)
        equal_series.append(float(np.nanmean(values)))
        weights = np.nan_to_num(dated.iloc[max(position - 1, 0)][members].to_numpy(dtype=float), nan=0.0)
        cap_series.append(float(weights @ values / weights.sum()) if weights.sum() > 0 else np.nan)
    equal = pd.Series(equal_series, index=window.index).dropna()
    market = pd.Series(cap_series, index=window.index).dropna()
    shared = equal.index.intersection(market.index)
    years = len(shared) / capm.TRADING_DAYS
    return dict(years=float(years), days=int(len(shared)),
                mean_members=float(np.mean(sizes)) if sizes else float("nan"),
                equal_annual=float((1 + equal.loc[shared]).prod() ** (1 / years) - 1),
                cap_annual=float((1 + market.loc[shared]).prod() ** (1 / years) - 1),
                equal_vol=float(equal.loc[shared].std(ddof=1) * math.sqrt(capm.TRADING_DAYS)),
                equal=equal, market=market)


def sensitivity(tail, gap, scenarios=SCENARIOS):
    """Tail return if the missing names had produced each assumed annual return.

    The missing names are 515 of a tail that holds roughly 3,200, so on a count basis they
    are a large minority of the book.  Two weightings are reported: by name count, which
    is what an equal-weighted tail would feel, and by capitalisation, which is what a
    cap-weighted one would.
    """
    members = tail["mean_members"]
    count_share = gap["missing"] / (gap["missing"] + members)
    cap_share = gap["missing_capitalisation"]
    rows = []
    for assumed in scenarios:
        blended_equal = (1 - count_share) * tail["equal_annual"] + count_share * assumed
        blended_cap = (1 - cap_share) * tail["cap_annual"] + cap_share * assumed
        rows.append(dict(assumed_missing=assumed, equal_tail=blended_equal, cap_tail=blended_cap))
    return dict(missing_count_share=count_share, missing_cap_share=cap_share, rows=rows)


def measure(panel_dir=PANEL, lo=TAIL_START):
    manifest, prices, traded, shares = load(panel_dir)
    ready = set(prices.columns)
    gap = universe_gap(ready)
    tail = tail_returns(panel_dir, lo)
    band = sensitivity(tail, gap)
    blend = dict(annual=0.0527, note="three-ETF blend from the allocation study")
    return dict(status="ready", tail_start=lo, window=dict(start=START, end=END),
                gap=gap, tail=dict(years=tail["years"], days=tail["days"],
                                   mean_members=tail["mean_members"],
                                   equal_annual=tail["equal_annual"],
                                   cap_annual=tail["cap_annual"],
                                   equal_vol=tail["equal_vol"]),
                sensitivity=band, blend=blend,
                verdict=dict(
                    tail_above_blend=bool(tail["equal_annual"] > blend["annual"]),
                    survives_worst_case=bool(band["rows"][-1]["equal_tail"] > blend["annual"])))


def _markdown(payload):
    tail, gap, band = payload["tail"], payload["gap"], payload["sensitivity"]
    lines = ["# 微盘尾的近似模拟与偏差敏感性", "",
             "分档从**调仓日前一日**的流通市值取，没有全期中位市值那层前视。", "",
             "| 口径 | 年化 | 波动 | 平均成员数 | 观测日 |",
             "| --- | ---: | ---: | ---: | ---: |",
             "| 尾部（第 %d 名之后）等权 | %+.2f%% | %.2f%% | %.0f | %d |" % (
                 payload["tail_start"], tail["equal_annual"] * 100, tail["equal_vol"] * 100,
                 tail["mean_members"], tail["days"]),
             "| 尾部市值加权 | %+.2f%% | — | — | — |" % (tail["cap_annual"] * 100),
             "| 三 ETF 混合（可执行基准） | %+.2f%% | — | — | — |" % (payload["blend"]["annual"] * 100),
             "",
             "## 缺失名字的敏感性", "",
             "宇宙 %d 个代码里有 **%d 个（%.1f%%）** 完全没有行情，占流通市值 **%.2f%%**，"
             "而它们全部落在尾部。尾部平均 %d 只，所以按名字数算，缺的那批占 **%.1f%%**。" % (
                 gap["universe"], gap["missing"], gap["missing_share"] * 100,
                 gap["missing_capitalisation"] * 100, tail["mean_members"],
                 band["missing_count_share"] * 100),
             "",
             "| 假设缺失名字的年化 | 按名字数折算的尾部等权 | 按市值折算的尾部市值加权 |",
             "| --- | ---: | ---: |"]
    for row in band["rows"]:
        lines.append("| %+.0f%% | %+.2f%% | %+.2f%% |" % (
            row["assumed_missing"] * 100, row["equal_tail"] * 100, row["cap_tail"] * 100))
    lines += ["", "**最坏情形下尾部是否仍高于三 ETF 基准：%s。**" % (
        "是" if payload["verdict"]["survives_worst_case"] else "否")]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Approximate the micro-cap tail and its bias.")
    parser.add_argument("--tail-start", type=int, default=TAIL_START)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reports" / "runs" / "tail_exposure"))
    args = parser.parse_args(argv)
    payload = measure(lo=args.tail_start)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    from src.data.preprocess import write_json
    write_json(target / "tail_exposure.json", payload)
    (target / "tail_exposure.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["verdict"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
