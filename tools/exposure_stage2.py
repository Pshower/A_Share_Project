"""Stage 2: the frozen volatility-scaling exposure rule, evaluated once on 2019-2024.

The rule and the matched-exposure control were registered in
``doc/exposure_control_plan_20261105.md`` before any V-segment number was read:

    e_t = clip(min(1, sigma* / sigma20_t), 0.20, 1.0)

``sigma*`` is the median of the design window's (2005-04..2018-12) realised volatility, so
the rule's average exposure is about one by construction.  The control is the same
instrument held at the rule's own average exposure, rebalanced the same way and charged the
same trading rate, so the comparison isolates *when* the exposure moved rather than *how
much* was held.

Two instruments are reported.  The index is the primary testbed because it carries no
survivorship bias; the 300-name equal-weight book is reported as a consistency check and is
labelled as biased.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import write_json
from tools.exposure_metrics import (ANNUAL, MA_WINDOW, SNAPSHOT, VOL_WINDOW, load_index,
                                    realized_vol)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "runs" / "exposure_stage2"
FOLD = PROJECT_ROOT / "reports" / "runs" / "ppo_exploration_20261006" / "folds" / "2024"

FLOOR = 0.20
COMMISSION = 0.0003
SLIPPAGE = 0.001
SELL_TAX = 0.0005
BUY_RATE = COMMISSION + SLIPPAGE
SELL_RATE = COMMISSION + SLIPPAGE + SELL_TAX
TRADING_DAYS = ANNUAL


def exposure_rule(vol: pd.Series, sigma_star: float, floor: float = FLOOR) -> pd.Series:
    """Frozen L1 rule: scale down with realised volatility, never above 1."""
    target = (sigma_star / vol).clip(upper=1.0, lower=floor)
    return target.where(vol.notna())


def simulate(returns: pd.Series, targets: pd.Series, *, buy_rate=BUY_RATE,
             sell_rate=SELL_RATE) -> dict:
    """Daily rebalance to ``targets``; cost charged on traded notional, net of nothing else."""
    returns = returns.to_numpy(dtype=float)
    targets = targets.to_numpy(dtype=float)
    weight, nav, traded, costs = 0.0, 1.0, 0.0, 0.0
    path = np.empty(len(returns))
    for index in range(len(returns)):
        target = targets[index]
        if not np.isfinite(target):
            target = weight if index else 0.0
        trade = target - weight
        cost = buy_rate * trade if trade > 0 else sell_rate * (-trade)
        rate = returns[index] if np.isfinite(returns[index]) else 0.0
        growth = 1.0 + target * rate
        nav *= growth - cost
        traded += abs(trade)
        costs += cost
        weight = target * (1.0 + rate) / growth
        path[index] = nav
    series = pd.Series(path, index=returns.index if hasattr(returns, "index") else None)
    return dict(nav=series, traded=traded, cost=costs, final=float(nav))


def summarize(nav: pd.Series) -> dict:
    daily = nav.pct_change().dropna()
    if daily.empty or nav.iloc[-1] <= 0:
        return dict(days=0)
    years = len(daily) / TRADING_DAYS
    total = float(nav.iloc[-1])
    annual_return = total ** (1 / years) - 1 if years > 0 else np.nan
    vol = float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS))
    drawdown = float((nav / nav.cummax() - 1).min())
    return dict(days=int(len(daily)), years=float(years), total=total, annual_return=float(annual_return),
                annual_vol=vol, sharpe=float(annual_return / vol) if vol else None,
                max_drawdown=drawdown, calmar=float(annual_return / abs(drawdown)) if drawdown else None)


def book_returns(index: pd.DatetimeIndex) -> pd.Series:
    """300-name equal-weight close-to-close return; survivorship-biased, labelled as such."""
    panel = pd.read_parquet(FOLD / "panel_data.parquet")
    prices = pd.read_parquet(FOLD / "prices.parquet")
    codes = list(panel.columns.get_level_values(0).unique())
    close = prices.xs("close", axis=1, level=1).reindex(columns=codes)
    close = close.reindex(index)
    return close.pct_change().mean(axis=1, skipna=True)


def evaluate(returns: pd.Series, targets: pd.Series, label: str) -> dict:
    """Rule versus the same instrument at the rule's own average exposure."""
    applicable = targets.notna()
    average = float(targets[applicable].mean())
    control = pd.Series(average, index=targets.index)
    rule = simulate(returns, targets)
    matched = simulate(returns, control)
    return dict(label=label, average_exposure=average,
                days_at_cap=int((targets >= 0.999).sum()), days_at_floor=int((targets <= FLOOR + 1e-9).sum()),
                applicable_days=int(applicable.sum()),
                rule=dict(summary=summarize(rule["nav"]), traded=rule["traded"], cost=rule["cost"]),
                control=dict(summary=summarize(matched["nav"]), traded=matched["traded"], cost=matched["cost"]))


def measure(snapshot: Path, code: str, *, design_end="2018-12-31", start="2019-01-01",
            end="2024-12-31", cadence=20) -> dict:
    bars = load_index(snapshot, code).set_index("date")
    vol = realized_vol(bars["close"])
    design = (bars.index <= pd.Timestamp(design_end))
    sigma_star = float(vol[design].median())
    rule = exposure_rule(vol, sigma_star)

    window = pd.Series((bars.index >= pd.Timestamp(start)) & (bars.index <= pd.Timestamp(end)),
                       index=bars.index)
    index_return = bars["close"].pct_change()

    # Exposure is decided at the close and only refreshed every `cadence` rows.
    sampled = rule.where(np.arange(len(rule)) % cadence == 0).ffill()
    targets = sampled.shift(1)

    index_result = evaluate(index_return[window], targets[window], f"{code} 指数")

    book_result = evaluate(book_returns(bars.index)[window], targets[window], "300 只等权")

    return dict(code=code, start=start, end=end, sigma_star=sigma_star,
                sigma_star_window=f"2005-04-01~{design_end}", floor=FLOOR, cadence=cadence,
                buy_rate=BUY_RATE, sell_rate=SELL_RATE,
                vol_median_window=float(vol[window].median()),
                index=index_result, book=book_result)


def verdict(payload: dict) -> dict:
    """The three registered bars, applied to the primary (index) instrument."""
    rule = payload["index"]["rule"]["summary"]
    control = payload["index"]["control"]["summary"]
    calmar_better = rule["calmar"] is not None and control["calmar"] is not None \
        and rule["calmar"] > control["calmar"]
    return_gap = rule["annual_return"] - control["annual_return"]
    keeps_return = return_gap >= -0.02
    return dict(calmar_rule=rule["calmar"], calmar_control=control["calmar"], calmar_better=bool(calmar_better),
                return_gap=float(return_gap), keeps_return=bool(keeps_return),
                passed=bool(calmar_better and keeps_return),
                decision="L1 通过" if calmar_better and keeps_return else "L1 未通过")


def _pct(value, digits=2):
    return "—" if value is None else f"{100 * value:+.{digits}f}%"


def _markdown(payload):
    lines = [f"# 阶段二：L1 波动率缩放（{payload['code']}，{payload['start']} ~ {payload['end']}）", "",
             f"- `σ*` = {100 * payload['sigma_star']:.1f}%（{payload['sigma_star_window']} 的 σ20 中位）",
             f"- 下限 {payload['floor']:.0%}，仓位按每 {payload['cadence']} 个交易日刷新",
             f"- 成本：买入 {10000 * payload['buy_rate']:.1f}bp、卖出 {10000 * payload['sell_rate']:.1f}bp（含滑点与印花税）",
             f"- V 段 σ20 中位 {100 * payload['vol_median_window']:.1f}%", ""]
    for key, title in (("index", "1. 主检验：指数（无幸存者偏差）"),
                       ("book", "2. 一致性检查：300 只等权（受幸存者偏差影响，仅供参考）")):
        block = payload[key]
        rule, control = block["rule"], block["control"]
        lines += [f"## {title}", "",
                  f"- 平均暴露 **{block['average_exposure']:.3f}**，触发上限 {block['days_at_cap']} 天、"
                  f"触发下限 {block['days_at_floor']} 天（共 {block['applicable_days']} 天）", "",
                  "| 指标 | 规则 | 同暴露对照 | 差 |", "| --- | ---: | ---: | ---: |",
                  f"| 年化收益 | {_pct(rule['summary']['annual_return'])} | {_pct(control['summary']['annual_return'])} | "
                  f"{_pct(rule['summary']['annual_return'] - control['summary']['annual_return'])} |",
                  f"| 年化波动 | {_pct(rule['summary']['annual_vol'])} | {_pct(control['summary']['annual_vol'])} | "
                  f"{_pct(rule['summary']['annual_vol'] - control['summary']['annual_vol'])} |",
                  f"| Sharpe | {rule['summary']['sharpe']:+.3f} | {control['summary']['sharpe']:+.3f} | "
                  f"{rule['summary']['sharpe'] - control['summary']['sharpe']:+.3f} |",
                  f"| 最大回撤 | {_pct(rule['summary']['max_drawdown'])} | {_pct(control['summary']['max_drawdown'])} | "
                  f"{_pct(rule['summary']['max_drawdown'] - control['summary']['max_drawdown'])} |",
                  f"| Calmar | {rule['summary']['calmar']:+.3f} | {control['summary']['calmar']:+.3f} | "
                  f"{rule['summary']['calmar'] - control['summary']['calmar']:+.3f} |",
                  f"| 累计净值 | {rule['summary']['total']:.4f} | {control['summary']['total']:.4f} | — |",
                  f"| 累计换手 | {rule['traded']:.2f} | {control['traded']:.2f} | — |",
                  f"| 累计成本 | {_pct(rule['cost'])} | {_pct(control['cost'])} | — |", ""]

    decided = payload["verdict"]
    lines += ["## 3. 按预登记判据", "",
              f"1. Calmar 高于同暴露对照：{decided['calmar_rule']:+.3f} vs {decided['calmar_control']:+.3f} → "
              f"{'**通过**' if decided['calmar_better'] else '**未通过**'}",
              f"2. 年化收益不低于对照 2 个百分点：差 {_pct(decided['return_gap'])} → "
              f"{'**通过**' if decided['keeps_return'] else '**未通过**'}",
              "3. 成本已计入：上述所有数字均为扣费后净值。",
              "", f"**结论：{decided['decision']}**", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=str(SNAPSHOT))
    parser.add_argument("--code", default="000300")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--cadence", type=int, default=20)
    parser.add_argument("--output", default=str(REPORT_ROOT))
    arguments = parser.parse_args(argv)
    payload = measure(Path(arguments.snapshot), arguments.code, start=arguments.start,
                      end=arguments.end, cadence=arguments.cadence)
    payload["verdict"] = verdict(payload)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f"stage2_{arguments.code}.json", payload)
    (output / f"stage2_{arguments.code}.md").write_text(_markdown(payload), encoding="utf-8")
    decided = payload["verdict"]
    index_block = payload["index"]
    print(f"{arguments.code} {payload['start']}..{payload['end']} sigma*={100 * payload['sigma_star']:.1f}% "
          f"avg_e={index_block['average_exposure']:.3f} "
          f"calmar {decided['calmar_rule']:+.3f} vs {decided['calmar_control']:+.3f} "
          f"return {100 * index_block['rule']['summary']['annual_return']:+.2f}% vs "
          f"{100 * index_block['control']['summary']['annual_return']:+.2f}% -> {decided['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
