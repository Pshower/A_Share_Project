"""Stage-1 measurements for the exposure-control plan.

Descriptive only: this tool reports distributions and never chooses a trading threshold.
It answers three questions on the design window (D: 2005-04 to 2018-12) so that the
validation window stays unread:

* is realised volatility predictable enough for scaling to mean anything;
* do MA200 states actually separate forward returns, and in which subperiods;
* what does a trend rule pay in whipsaw when it is wrong.

The decision bars live in ``doc/exposure_control_plan_20261105.md`` and are applied by
``verdict`` below; they are not tuned here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = PROJECT_ROOT / "data" / "online" / "supplements" / "s_4bf7f0841b884089"
REPORT_ROOT = PROJECT_ROOT / "reports" / "runs" / "exposure_stage1"

ANNUAL = 243
VOL_WINDOW = 20
MA_WINDOW = 200
ACF_LAGS = (5, 20, 60)
HORIZONS = (20, 60)


def load_index(snapshot: Path, code: str) -> pd.DataFrame:
    """Provider CSV in Chinese columns; keep date, open, close and sort once."""
    frame = pd.read_csv(snapshot / f"{code}.csv")
    frame.columns = [str(column).strip() for column in frame.columns]
    frame["date"] = pd.to_datetime(frame["日期"])
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return pd.DataFrame({"date": frame["date"],
                         "open": frame["开盘"].astype(float),
                         "close": frame["收盘"].astype(float)})


def realized_vol(close: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """Annualised close-to-close volatility of the trailing window."""
    return close.pct_change().rolling(window).std(ddof=1) * np.sqrt(ANNUAL)


def forward_returns(bars: pd.DataFrame, horizon: int) -> pd.Series:
    """Decide at close t, execute at open t+1, exit at close t+h."""
    close = bars["close"].to_numpy(dtype=float)
    opening = bars["open"].to_numpy(dtype=float)
    values = np.full(len(bars), np.nan)
    if len(bars) > horizon:
        values[:-horizon] = close[horizon:] / opening[1:len(bars) - horizon + 1] - 1
    return pd.Series(values, index=bars.index)


def autocorrelation(series: pd.Series, lags=ACF_LAGS) -> dict:
    return {str(lag): float(series.autocorr(lag)) for lag in lags}


def volatility_ratio(vol: pd.Series, horizon: int = VOL_WINDOW) -> dict:
    """How much the next non-overlapping window's volatility differs from today's."""
    ratio = (vol.shift(-horizon) / vol).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return dict(n=0)
    return dict(n=int(len(ratio)), median=float(ratio.median()),
                p10=float(ratio.quantile(0.10)), p90=float(ratio.quantile(0.90)),
                share_within_25pct=float(((ratio >= 0.75) & (ratio <= 1.25)).mean()))


def ma_state(close: pd.Series, window: int = MA_WINDOW) -> pd.Series:
    """True while the index closes above its own moving average."""
    average = close.rolling(window).mean()
    return (close > average).where(average.notna())


def _runs(state: pd.Series) -> list[tuple]:
    """Strictly alternating runs of a boolean state; NaN rows are dropped first."""
    known = state.dropna()
    if known.empty:
        return []
    flags = known.to_numpy(dtype=bool)
    change = np.r_[True, flags[1:] != flags[:-1]]
    starts = np.where(change)[0]
    ends = np.r_[starts[1:] - 1, len(flags) - 1]
    return [(bool(flags[start]), known.index[start], known.index[end])
            for start, end in zip(starts, ends)]


def segments(bars: pd.DataFrame, state: pd.Series) -> list[dict]:
    """Contiguous runs of the same state, with path length, return and drawdown."""
    rows = []
    for flag, start, end in _runs(state):
        window = bars.loc[start:end]
        if len(window) < 2:
            continue
        path = window["close"].to_numpy(dtype=float)
        rows.append(dict(above=flag, start=str(window["date"].iloc[0].date()),
                         end=str(window["date"].iloc[-1].date()), days=int(len(window)),
                         return_pct=float(path[-1] / path[0] - 1),
                         max_drawdown_pct=float(np.min(path / np.maximum.accumulate(path) - 1))))
    return rows


def conditional_table(bars, state, horizon):
    """Mean forward return in each state, and the gap in unconditional sigma units."""
    forward = forward_returns(bars, horizon)
    below = forward[state.reindex(bars.index) == False]     # noqa: E712 - explicit comparison
    above = forward[state.reindex(bars.index) == True]      # noqa: E712
    below, above = below.dropna(), above.dropna()
    sigma = float(forward.dropna().std(ddof=1))
    gap = float(below.mean() - above.mean()) if len(below) and len(above) else np.nan
    return dict(horizon=horizon, above_n=int(len(above)), below_n=int(len(below)),
                above_mean=float(above.mean()) if len(above) else None,
                below_mean=float(below.mean()) if len(below) else None,
                gap=gap, gap_in_sigma=gap / sigma if sigma else None,
                unconditional_sigma=sigma)


def subperiod_gaps(bars, state, horizon, edges):
    """The same gap per subperiod, to test whether the sign is stable."""
    forward = forward_returns(bars, horizon)
    dates = bars["date"]
    rows = []
    for left, right in edges:
        mask = (dates >= pd.Timestamp(left)) & (dates <= pd.Timestamp(right))
        block_state = state.where(mask).dropna()
        below = forward[mask & (state == False)].dropna()   # noqa: E712
        above = forward[mask & (state == True)].dropna()    # noqa: E712
        rows.append(dict(period=f"{left}~{right}", above_n=int(len(above)), below_n=int(len(below)),
                         gap=float(below.mean() - above.mean()) if len(below) and len(above) else None))
    return rows


def whipsaw(bars, state, horizon):
    """Every fresh cross below MA200: was the next window up (missed) or down (avoided)?"""
    runs = _runs(state)
    # The first run has no predecessor, so it cannot be a fresh cross.
    events = [start for position, (flag, start, _) in enumerate(runs)
              if not flag and position > 0]
    forward = forward_returns(bars, horizon)
    values = forward.reindex(events).dropna()
    if values.empty:
        return dict(horizon=horizon, crosses=0)
    return dict(horizon=horizon, crosses=int(len(values)),
                missed=int((values > 0).sum()), avoided=int((values <= 0).sum()),
                mean_after_event=float(values.mean()), median_after_event=float(values.median()),
                total_missed=float(values[values > 0].sum()),
                total_avoided=float(values[values <= 0].sum()),
                net=float(values.sum()))


def verdict(payload):
    """The pre-registered bars from the plan, applied mechanically."""
    acf20 = payload["volatility"]["acf"]["20"]
    vol_ok = acf20 >= 0.50 and payload["volatility"]["ratio"]["median"] is not None \
        and 0.80 <= payload["volatility"]["ratio"]["median"] <= 1.25
    gap20 = next(row for row in payload["conditional"] if row["horizon"] == 20)["gap_in_sigma"]
    signs = [row["gap"] for row in payload["subperiods"] if row["gap"] is not None]
    stable = bool(signs) and all(np.sign(value) == np.sign(signs[0]) for value in signs)
    trend_ok = gap20 is not None and abs(gap20) >= 0.5 and stable
    cost = next(row for row in payload["whipsaw"] if row["horizon"] == 20)
    trend_pays = cost["crosses"] == 0 or cost["net"] <= 0
    return dict(acf20=acf20, vol_ok=bool(vol_ok), gap20_in_sigma=gap20,
                subperiod_signs=[None if row["gap"] is None else float(np.sign(row["gap"]))
                                 for row in payload["subperiods"]],
                signs_stable=stable, trend_ok=bool(trend_ok),
                whipsaw_net=cost["net"], trend_pays=bool(trend_pays),
                register=("L1 first, then L2" if vol_ok and trend_ok and trend_pays
                          else "L1 only" if vol_ok
                          else "L2 only" if trend_ok and trend_pays
                          else "stop"))


def measure(snapshot: Path, code: str, start: str, end: str) -> dict:
    bars = load_index(snapshot, code)
    bars = bars[(bars["date"] >= pd.Timestamp(start)) & (bars["date"] <= pd.Timestamp(end))]
    bars = bars.reset_index(drop=True)
    vol = realized_vol(bars["close"])
    state = ma_state(bars["close"])
    yearly = (vol.dropna().groupby(bars["date"].dt.year.loc[vol.dropna().index]).median())
    return dict(code=code, start=str(bars["date"].iloc[0].date()), end=str(bars["date"].iloc[-1].date()),
                rows=int(len(bars)),
                volatility=dict(window=VOL_WINDOW, annual=ANNUAL, acf=autocorrelation(vol),
                                ratio=volatility_ratio(vol),
                                yearly_median={str(year): float(value) for year, value in yearly.items()}),
                ma=dict(window=MA_WINDOW, above_share=float(state.dropna().mean()),
                        segments=segments(bars, state)),
                conditional=[conditional_table(bars, state, horizon) for horizon in HORIZONS],
                subperiods=subperiod_gaps(bars, state, 20,
                                          [("2005-04-01", "2007-12-31"), ("2008-01-01", "2010-12-31"),
                                           ("2011-01-01", "2013-12-31"), ("2014-01-01", "2018-12-31")]),
                whipsaw=[whipsaw(bars, state, horizon) for horizon in HORIZONS])


def _pct(value, digits=2):
    return "—" if value is None else f"{100 * value:+.{digits}f}%"


def _num(value, digits=2):
    return "—" if value is None else f"{value:+.{digits}f}"


def _markdown(payload):
    vol = payload["volatility"]
    lines = [f"# 阶段一机制度量（{payload['code']}，{payload['start']} ~ {payload['end']}）", "",
             f"- 样本 {payload['rows']} 个交易日，只看设计窗 D 段", "",
             "## 1. 已实现波动的可预测性", "",
             "| 滞后 | 自相关 |", "| ---: | ---: |"]
    for lag, value in vol["acf"].items():
        lines.append(f"| {lag} 日 | {_num(value, 3)} |")
    ratio = vol["ratio"]
    lines += ["", f"- `σ20(t+20)/σ20(t)`：中位 {_num(ratio.get('median'), 3)}，"
                  f"10/90 分位 {_num(ratio.get('p10'), 3)} / {_num(ratio.get('p90'), 3)}，"
                  f"落在 ±25% 内的比例 {ratio.get('share_within_25pct', 0):.1%}，样本 {ratio.get('n', 0)}",
              "- 滞后 5 日的窗口与当期重叠，机械相关偏高，以 20/60 日为准。", ""]
    lines += ["| 年份 | σ20 年化中位 |", "| --- | ---: |"]
    for year, value in vol["yearly_median"].items():
        lines.append(f"| {year} | {100 * value:.1f}% |")

    lines += ["", "## 2. MA200 状态与前瞻收益", "",
              f"收盘在 MA200 上方的交易日占比：**{payload['ma']['above_share']:.1%}**", "",
              "| 期限 | 上方均值 | 下方均值 | 差值 | 差值/σ | 上方 n | 下方 n |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in payload["conditional"]:
        lines.append(f"| {row['horizon']} 日 | {_pct(row['above_mean'])} | {_pct(row['below_mean'])} | "
                     f"{_pct(row['gap'])} | {_num(row['gap_in_sigma'])} | {row['above_n']} | {row['below_n']} |")
    lines += ["", "### 分区间符号稳定性（20 日）", "", "| 区间 | 差值 | 上方 n | 下方 n |", "| --- | ---: | ---: | ---: |"]
    for row in payload["subperiods"]:
        lines.append(f"| {row['period']} | {_pct(row['gap'])} | {row['above_n']} | {row['below_n']} |")

    segments = payload["ma"]["segments"]
    lines += ["", f"### 分段（共 {len(segments)} 段）", "",
              "| 状态 | 起 | 止 | 交易日 | 段内收益 | 段内最大回撤 |", "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in segments:
        lines.append(f"| {'上方' if row['above'] else '下方'} | {row['start']} | {row['end']} | "
                     f"{row['days']} | {_pct(row['return_pct'])} | {_pct(row['max_drawdown_pct'])} |")

    lines += ["", "## 3. 假信号的机会成本（向下穿越 MA200 之后）", "",
              "| 期限 | 穿越次数 | 之后上涨（踏空） | 之后下跌（避开） | 均值 | 中位 | 踏空合计 | 避开合计 | 净 |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in payload["whipsaw"]:
        if not row.get("crosses"):
            lines.append(f"| {row['horizon']} 日 | 0 | — | — | — | — | — | — | — |")
            continue
        lines.append(f"| {row['horizon']} 日 | {row['crosses']} | {row['missed']} | {row['avoided']} | "
                     f"{_pct(row['mean_after_event'])} | {_pct(row['median_after_event'])} | "
                     f"{_pct(row['total_missed'])} | {_pct(row['total_avoided'])} | {_pct(row['net'])} |")
    lines += ["", "合计与净额是各次事件窗口的**相加**，窗口之间可以重叠，因此不是组合收益，只能用来比较方向。", ""]

    decided = payload["verdict"]
    lines += ["", "## 4. 按预登记判据的结论", "",
              f"- `σ20` 20 日自相关 **{decided['acf20']:+.3f}**（L1 门槛 ≥ 0.50）→ "
              f"{'具备登记资格' if decided['vol_ok'] else '**出局**'}",
              f"- 下方−上方差值 **{_num(decided['gap20_in_sigma'])}σ**（L2 门槛 ≥ 0.5σ），"
              f"分区间符号 {decided['subperiod_signs']}，一致={decided['signs_stable']} → "
              f"{'具备登记资格' if decided['trend_ok'] else '**出局**'}",
              f"- 假信号净成本 20 日 **{_pct(decided['whipsaw_net'])}**"
              f"（负值=避开的下跌多于踏空）→ {'可以付费' if decided['trend_pays'] else '**净负**'}",
              "", f"**登记建议：{decided['register']}**", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=str(SNAPSHOT))
    parser.add_argument("--code", default="000300")
    parser.add_argument("--start", default="2005-04-01", help="design window D start")
    parser.add_argument("--end", default="2018-12-31", help="design window D end")
    parser.add_argument("--output", default=str(REPORT_ROOT))
    arguments = parser.parse_args(argv)
    payload = measure(Path(arguments.snapshot), arguments.code, arguments.start, arguments.end)
    payload["verdict"] = verdict(payload)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f"stage1_{arguments.code}.json", payload)
    (output / f"stage1_{arguments.code}.md").write_text(_markdown(payload), encoding="utf-8")
    decided = payload["verdict"]
    print(f"{arguments.code} {payload['start']}..{payload['end']}: acf20={decided['acf20']:+.3f} "
          f"gap20={decided['gap20_in_sigma']:+.2f}sigma whipsaw_net={decided['whipsaw_net']:+.4f} "
          f"-> {decided['register']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
