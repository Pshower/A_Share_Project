"""Check a panel-built equal-weight premium against a real index pair.

An equal-weight portfolio constructed from a panel of *currently listed* names is not
the same object as an equal-weight index.  The panel selects on today's capitalisation,
so every name in it survived and largely stayed in its size band; equal weighting spreads
0.2% across all of them and therefore absorbs that selection in full, while cap weighting
concentrates on the handful of names that were already large and absorbs only a fraction.

The only like-for-like control is an index pair that differs in weighting alone.  For the
CSI 500 band that pair exists: 000905 (free-float cap weighted) and 000982 (equal
weighted) cover the same constituents, are computed and published in real time, and are
therefore free of the selection the panel carries.  This module measures both and refuses
to call a panel premium supported unless it is close to the index's.
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

DEFAULT_PANEL = capm.PROJECT_ROOT / "data" / "clean" / "panel_full"
PAIRS = (
    dict(label="中证500档", lo=300, hi=800, cap_index="000905", equal_index="000982"),
)
TOLERANCE = 0.01  # one percentage point a year


def band_frames(panel_dir, lo, hi):
    """The panel restricted to the names ranked ``lo`` to ``hi`` by median float cap."""
    manifest, prices, traded, shares = load(panel_dir)
    cap = (traded * shares).median().sort_values(ascending=False)
    keep = list(cap.index[lo:hi])
    if len(keep) < 50:
        raise ValueError("Band holds fewer than fifty names")
    return prices[keep], traded[keep], shares[keep]


def panel_premium(panel_dir, lo, hi, start=capm.SAMPLE_START, end="2024-12-31"):
    """Annual return of the cap-weighted and equal-weighted books on one size band.

    Band membership is chosen once, by **median** capitalisation over the whole sample.
    That is a look-ahead: a name that was small in 2019 and only reached the band by
    rising keeps a full equal weight on day one, while cap weighting gives it almost
    nothing, and a name that was large and fell into the band keeps a large cap weight.
    ``dynamic_band_premium`` repeats the exercise without it, and the difference between
    the two is the whole of the inflated premium.
    """
    prices, traded, shares = band_frames(panel_dir, lo, hi)
    market, _, _ = capm.market_portfolio(prices, traded, shares)
    market = market.dropna().loc[start:end]
    equal, turnover = capm.equal_weight_portfolio(capm.daily_frame(prices))
    equal = equal.loc[start:end]
    cap_perf = capm.performance(market)
    equal_perf = capm.performance(equal)
    return dict(names=len(prices.columns), days=int(len(market)),
                cap_annual=cap_perf["annual_return"], equal_annual=equal_perf["annual_return"],
                cap_vol=cap_perf["annual_vol"], equal_vol=equal_perf["annual_vol"],
                premium=equal_perf["annual_return"] - cap_perf["annual_return"])


def dynamic_band_premium(panel_dir, lo, hi, start=capm.SAMPLE_START, end="2024-12-31", step=capm.STEP):
    """The same band, re-chosen each rebalance from the **lagged** capitalisation.

    At every rebalance the band is the names ranked ``lo`` to ``hi`` by float market
    capitalisation on the previous session, so nothing about the future decides who is
    in it.  Both legs are re-weighted daily inside that band, which is the closest this
    module gets to the index provider's own construction.
    """
    _, _, _, caps = _panel_with_caps(panel_dir)
    returns = capm.daily_frame(_panel_with_caps(panel_dir)[0])
    window = returns.loc[start:end]
    dated = caps.reindex(window.index).ffill()
    marks = set(range(0, len(window), step))
    equal_series, cap_series = [], []
    band = []
    for position in range(len(window)):
        if position in marks:
            asof = dated.iloc[max(position - 1, 0)].dropna()
            band = list(asof.sort_values(ascending=False).index[lo:hi])
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
    cap_annual = float((1 + market.loc[shared]).prod() ** (1 / years) - 1)
    equal_annual = float((1 + equal.loc[shared]).prod() ** (1 / years) - 1)
    return dict(names=int(len(band)), days=int(len(shared)),
                cap_annual=cap_annual, equal_annual=equal_annual,
                premium=equal_annual - cap_annual)


def _panel_with_caps(panel_dir):
    manifest, prices, traded, shares = load(panel_dir)
    return prices, traded, shares, traded * shares


def index_premium(cap_index, equal_index, start=capm.SAMPLE_START, end="2024-12-31"):
    """Annual return of the published cap-weighted and equal-weighted indices."""
    cap = capm.index_returns(cap_index, start, end)
    equal = capm.index_returns(equal_index, start, end)
    if cap is None or equal is None or cap.empty or equal.empty:
        return dict(status="missing")
    joined = pd.concat([cap.rename("cap"), equal.rename("equal")], axis=1, join="inner").dropna()
    if len(joined) < 200:
        return dict(status="insufficient", observations=int(len(joined)))
    years = len(joined) / capm.TRADING_DAYS
    cap_annual = (1 + joined["cap"]).prod() ** (1 / years) - 1
    equal_annual = (1 + joined["equal"]).prod() ** (1 / years) - 1
    return dict(status="ready", observations=int(len(joined)),
                cap_annual=float(cap_annual), equal_annual=float(equal_annual),
                premium=float(equal_annual - cap_annual),
                correlation=float(joined["equal"].corr(joined["cap"])),
                tracking_error=float((joined["equal"] - joined["cap"]).std(ddof=1) * math.sqrt(capm.TRADING_DAYS)))


def mechanism(prices, start=capm.SAMPLE_START, end="2024-12-31"):
    """Why the panel premium is large: arithmetic mean against the median single name."""
    returns = capm.daily_frame(prices).loc[start:end]
    average = returns.mean(axis=1).dropna()
    years = len(average) / capm.TRADING_DAYS
    arithmetic = (1 + average).prod() ** (1 / years) - 1
    growth = (1 + returns).cumprod().iloc[-1]
    single = growth ** (capm.TRADING_DAYS / len(returns)) - 1
    return dict(arithmetic_average=float(arithmetic),
                median_single_name=float(single.median()),
                mean_single_name=float(single.mean()),
                portfolio_vol=float(average.std(ddof=1) * math.sqrt(capm.TRADING_DAYS)),
                median_single_vol=float(returns.std(ddof=1).median() * math.sqrt(capm.TRADING_DAYS)))


def verdict(panel_value, index_value, tolerance=TOLERANCE):
    """Whether a panel premium survives comparison with the index pair."""
    if panel_value is None or index_value is None:
        return dict(status="unknown", reason="one side is missing")
    difference = panel_value - index_value
    supported = abs(difference) <= tolerance
    return dict(status="supported" if supported else "unsupported",
                panel=panel_value, index=index_value, difference=difference,
                tolerance=tolerance,
                reason=("panel premium is close to the index pair"
                        if supported else
                        "panel premium is not reproducible against the index pair; "
                        "treat it as selection, not as an equal-weight effect"))


def measure(panel_dir=DEFAULT_PANEL, pairs=PAIRS, start=capm.SAMPLE_START, end="2024-12-31"):
    rows = []
    for spec in pairs:
        row = dict(spec)
        try:
            row["panel"] = panel_premium(panel_dir, spec["lo"], spec["hi"], start, end)
            row["index"] = index_premium(spec["cap_index"], spec["equal_index"], start, end)
        except (ValueError, FileNotFoundError, KeyError) as error:
            row["status"] = f"skipped: {error}"
            rows.append(row)
            continue
        row["status"] = "ready"
        try:
            row["panel_pit"] = dynamic_band_premium(panel_dir, spec["lo"], spec["hi"], start, end)
        except (ValueError, FileNotFoundError, KeyError):
            row["panel_pit"] = None
        row["verdict"] = verdict(row["panel"]["premium"], row["index"].get("premium"))
        row["verdict_pit"] = verdict((row["panel_pit"] or {}).get("premium"), row["index"].get("premium"))
        prices, _, _ = band_frames(panel_dir, spec["lo"], spec["hi"])
        row["mechanism"] = mechanism(prices, start, end)
        rows.append(row)
    return dict(status="ready", panel=str(Path(panel_dir)), start=start, end=end, rows=rows)


def _markdown(payload):
    lines = ["# 等权溢价的指数对照", "",
             "面板用**当前市值**选股，等权因此把权重摊到「必须是幸存者」的全体上；"
             "指数是实时计算的，历史里含当时成分。这一节把两者放在同一档位、同一区间上比。", "",
             "| 档位 | 静态选档溢价 | **PIT 动态选档溢价** | 指数溢价 | 静态判定 | PIT 判定 |",
             "| --- | ---: | ---: | ---: | --- | --- |"]
    for row in payload["rows"]:
        if row.get("status") != "ready":
            lines.append("| {} | — | — | — | — | {} |".format(row["label"], row.get("status")))
            continue
        panel, index = row["panel"], row["index"]
        pit = row.get("panel_pit") or {}
        lines.append("| {} | **{:+.2f}pp** | **{:+.2f}pp** | **{:+.2f}pp** | {} | {} |".format(
            row["label"], panel["premium"] * 100, pit.get("premium", float("nan")) * 100,
            index["premium"] * 100, row["verdict"]["status"],
            (row.get("verdict_pit") or {}).get("status", "—")))
    lines += ["", "静态选档用**全样本中位市值**决定档位归属，是前视；PIT 用**调仓日前一日市值**重新选档。"
                  "两者之差就是假溢价的全部来源。", ""]
    lines += ["", "## 副对照：组合与单个股票之间的方差拖累", "",
              "这一栏解释的是**任何 500 股组合**为什么高于个股中位数（方差拖累 σ²/2），"
              "**不能**解释等权与市值加权之间的差——两个指数和两个面板组合都持有 500 只，都同等地消除了这部分拖累。", "",
              "| 档位 | 横截面日均收益的平均 | 个股中位数 | 个股平均 | 组合波动 | 个股中位波动 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in payload["rows"]:
        if row.get("status") != "ready":
            continue
        m = row["mechanism"]
        lines.append("| {} | {:+.2f}% | {:+.2f}% | {:+.2f}% | {:.1f}% | {:.1f}% |".format(
            row["label"], m["arithmetic_average"] * 100, m["median_single_name"] * 100,
            m["mean_single_name"] * 100, m["portfolio_vol"] * 100, m["median_single_vol"] * 100))
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cross-check a panel equal-weight premium.")
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--output", default=None)
    parser.add_argument("--start", default=capm.SAMPLE_START)
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args(argv)
    payload = measure(args.panel, start=args.start, end=args.end)
    if args.output:
        target = Path(args.output)
        target.mkdir(parents=True, exist_ok=True)
        from src.data.preprocess import write_json
        write_json(target / "equal_weight_crosscheck.json", payload)
        (target / "equal_weight_crosscheck.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
