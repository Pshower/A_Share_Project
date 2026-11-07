"""How much alpha the accepted architecture actually demands.

The exposure work closed two forecasting rules (trend state, volatility scaling) and left
one option standing that needs no forecast at all: hold a constant fraction ``w`` of the
market and keep the rest in cash.  That option is a preference, not a prediction, so it can
be priced exactly instead of being tested.  This tool prices it and connects three numbers
that were previously discussed separately:

* the return given up by holding ``w`` instead of the full market;
* the information coefficient a cross-sectional book needs to earn that return back;
* the sample size required before such an IC could be believed.

Two blend conventions are reported because they behave differently and the difference is
the point.  A *buy-and-hold* blend (own ``w`` once, never rebalance) is what "just hold
less" means in practice; a *constant-weight* blend (rebalance back to ``w``) is the
textbook de-risking policy.  Neither can raise the Sharpe ratio: for constant weight the
Sharpes are algebraically identical by construction, and for buy-and-hold the drift makes
them worse.  So a static exposure cut does not buy risk control, it only buys scale.

Nothing here is a hypothesis test and no return window is consumed: the index path is the
same instrument the exposure plan already used, and the arithmetic is deterministic.  The
rules that turn these numbers into a research programme live in
``doc/alpha_program_20261106.md``.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import write_json
from tools.exposure_metrics import SNAPSHOT, load_index
from tools.exposure_stage2 import TRADING_DAYS, summarize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "runs" / "alpha_budget"

CASH_RATE = 0.02
WEIGHTS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
BOOK_SIZES = (10, 20, 30, 50)
UNIVERSE = 300
IC_GRID = (0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10)
DAILY_IC_SD = (0.05, 0.07, 0.10)
HORIZONS = (20, 60, 120, 243)
RESERVE = ("2025-01-01", "2026-06-30")
PROTECTED = ("2026-07-01", "2026-10-30")
HISTORY_DIR = PROJECT_ROOT / "data" / "online" / "history" / "h_13cd6e0169124b2e" / "hfq"


def _norm_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _norm_ppf(share: float, lo: float = -10.0, hi: float = 10.0, steps: int = 200) -> float:
    """Bisection inverse of the standard normal CDF; the environment has no scipy."""
    if not 0.0 < share < 1.0:
        raise ValueError("share must lie strictly between 0 and 1")
    for _ in range(steps):
        middle = 0.5 * (lo + hi)
        if _norm_cdf(middle) < share:
            lo = middle
        else:
            hi = middle
    return 0.5 * (lo + hi)


def top_mean_z(share: float) -> float:
    """Mean standard-normal score of the best ``share`` of a cross-section."""
    if not 0.0 < share <= 1.0:
        raise ValueError("share must lie in (0, 1]")
    if share == 1.0:
        return 0.0
    return _norm_pdf(_norm_ppf(1.0 - share)) / share


def alpha_from_ic(ic: float, share: float, sigma_cs: float, years: float = 1.0) -> float:
    """Excess annual return of an equal-weight top-``share`` book under a normal copula.

    ``E[r | s] = ic * sigma_r * s`` for jointly normal standardised signal and return, so
    the book's excess over the universe mean is ``ic * sigma_cs * E[z | top share]`` per
    ``years`` of holding.  ``sigma_cs`` is the cross-sectional spread measured over that
    same holding horizon, so a monthly book needs the monthly spread and a monthly IC.
    """
    if years <= 0:
        raise ValueError("years must be positive")
    return ic * sigma_cs * top_mean_z(share) / years


def ic_from_alpha(target: float, share: float, sigma_cs: float, years: float = 1.0) -> float:
    """Inverse of :func:`alpha_from_ic`."""
    scale = sigma_cs * top_mean_z(share) / years
    if scale == 0:
        return float("nan")
    return target / scale


def required_ic_at_horizon(target_alpha: float, share: float, sigma_cs: float,
                           horizon: int) -> float:
    """IC needed over a ``horizon``-day holding period to earn ``target_alpha`` a year.

    Rewriting :func:`alpha_from_ic` for a horizon instead of a year gives
    ``ic = alpha * sqrt(h / 243) / (sigma_cs * E[z | top share])``, because the
    cross-sectional spread grows with the square root of the holding period while alpha is
    divided by it.  Shorter holding periods therefore demand a *smaller* IC.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    scale = sigma_cs * top_mean_z(share)
    if scale == 0:
        return float("nan")
    return target_alpha * math.sqrt(horizon / TRADING_DAYS) / scale


def blend(nav: pd.Series, weight: float, cash_rate: float = CASH_RATE,
          rebalance: bool = False) -> pd.Series:
    """Mix ``weight`` of the index with cash, either held once or rebalanced daily."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie in [0, 1]")
    growth = (nav / float(nav.iloc[0])).to_numpy(dtype=float)
    days = np.arange(len(nav), dtype=float)
    cash = np.power(1.0 + cash_rate, days / TRADING_DAYS)
    if rebalance:
        values = np.power(growth, weight) * np.power(cash, 1.0 - weight)
    else:
        values = weight * growth + (1.0 - weight) * cash
    return pd.Series(values, index=nav.index)


def ladder(nav: pd.Series, weights=WEIGHTS, cash_rate: float = CASH_RATE,
           rebalance: bool = False) -> dict:
    """Metrics per exposure level plus the return each level must earn back with alpha.

    ``sharpe_excess`` is the usual ``(return - cash) / volatility``.  It is the quantity a
    static blend cannot change, whereas ``return_over_vol`` (return divided by volatility,
    with no risk-free deduction) rises mechanically whenever cash carries a positive rate.
    """
    full = summarize(blend(nav, 1.0, cash_rate, rebalance))
    rows = []
    for weight in weights:
        metrics = summarize(blend(nav, weight, cash_rate, rebalance))
        gap = full["annual_return"] - metrics["annual_return"]
        vol = metrics["annual_vol"]
        usable = vol is not None and vol > 1e-8
        excess = (metrics["annual_return"] - cash_rate) / vol if usable else None
        rows.append(dict(exposure=float(weight),
                         annual_return=metrics["annual_return"],
                         annual_vol=vol,
                         return_over_vol=metrics["sharpe"] if usable else None,
                         sharpe_excess=excess,
                         max_drawdown=metrics["max_drawdown"],
                         calmar=metrics["calmar"],
                         required_alpha=float(gap)))
    return dict(cash_rate=float(cash_rate), rebalance=bool(rebalance), full=full, rows=rows)


def cross_sectional_dispersion(prices: pd.DataFrame, horizon: int = TRADING_DAYS) -> float:
    """Average cross-sectional spread of overlapping ``horizon``-day returns.

    ``prices`` is a date-by-code frame; drift and survivorship are irrelevant here because
    only the spread of the cross-section is measured, never its mean.
    """
    forward = prices.shift(-horizon) / prices - 1.0
    dispersion = forward.std(axis=1, ddof=1)
    if dispersion.empty or not np.isfinite(dispersion).any():
        return float("nan")
    return float(dispersion.mean())


def load_unadjusted_panel(history: Path, start, end) -> pd.DataFrame:
    """Close prices from the provider's unadjusted history, one column per code."""
    frames = {}
    for path in sorted(history.glob("*.csv")):
        frame = pd.read_csv(path, encoding="utf-8-sig", usecols=["date", "code", "close"])
        frame["date"] = pd.to_datetime(frame["date"])
        frames[str(frame["code"].iloc[0])] = frame.set_index("date")["close"]
    panel = pd.DataFrame(frames).sort_index()
    return panel.loc[(panel.index >= pd.Timestamp(start)) & (panel.index <= pd.Timestamp(end))]


def tracking_error(sigma_cs: float, n_book: int, market_vol: float) -> float:
    """Annual volatility of an equal-weight ``n_book`` book's return minus the market's.

    One-factor decomposition: the cross-sectional spread splits into market and
    idiosyncratic parts, ``sigma_cs^2 = market_vol^2 + sigma_idio^2``, and the book's
    residual against the market is ``sigma_idio / sqrt(n_book)``.
    """
    if n_book <= 0:
        raise ValueError("n_book must be positive")
    idiosyncratic = math.sqrt(max(sigma_cs ** 2 - market_vol ** 2, 0.0))
    return idiosyncratic / math.sqrt(n_book)


def ic_power(days: int, daily_ic_sd: float) -> dict:
    """Standard error of a mean daily IC and the smallest IC detectable at two sigma."""
    if days <= 0:
        raise ValueError("days must be positive")
    standard_error = daily_ic_sd / math.sqrt(days)
    return dict(days=int(days), daily_ic_sd=float(daily_ic_sd),
                standard_error=float(standard_error),
                detectable_2sigma=float(2.0 * standard_error))


def book_power(target_alpha: float, te: float, years: float) -> dict:
    """Can a book-level return test separate ``target_alpha`` from zero?"""
    if te <= 0:
        return dict(target_alpha=float(target_alpha), tracking_error=float(te),
                    years=float(years), t_stat=None)
    t_stat = target_alpha / te * math.sqrt(years)
    return dict(target_alpha=float(target_alpha), tracking_error=float(te),
                years=float(years), t_stat=float(t_stat))


def transmission(*, sigma_cs: float, market_vol: float, universe: int = UNIVERSE,
                 book_sizes=BOOK_SIZES, ic_grid=IC_GRID) -> dict:
    """Map IC to book alpha, and each book size to the volatility it carries."""
    rows = []
    for size in book_sizes:
        share = size / universe
        rows.append(dict(book_size=int(size), share=float(share), top_mean_z=float(top_mean_z(share)),
                         tracking_error=float(tracking_error(sigma_cs, size, market_vol))))
    alpha_table = []
    for ic in ic_grid:
        alpha_table.append(dict(ic=float(ic),
                                **{f"book_{size}": float(alpha_from_ic(ic, size / universe, sigma_cs))
                                   for size in book_sizes}))
    return dict(sigma_cs=float(sigma_cs), market_vol=float(market_vol),
                universe=int(universe), rows=rows, alpha_table=alpha_table)


def measure(*, code="000300", start="2005-01-04", end="2026-10-30", cash_rate=CASH_RATE,
            weights=WEIGHTS, sigma_cs=None) -> dict:
    bars = load_index(SNAPSHOT, code)
    nav = bars.set_index("date")["close"]
    nav = nav.loc[(nav.index >= pd.Timestamp(start)) & (nav.index <= pd.Timestamp(end))]

    if sigma_cs is None:
        try:
            prices = load_unadjusted_panel(HISTORY_DIR, "2017-01-03", "2024-12-31")
            sigma_cs = cross_sectional_dispersion(prices)
        except (OSError, ValueError, KeyError):
            sigma_cs = 0.45
    market_vol = summarize(blend(nav, 1.0, cash_rate))["annual_vol"]

    windows = {"reserve": RESERVE, "protected": PROTECTED}
    days = {name: int(((nav.index >= pd.Timestamp(lo)) & (nav.index <= pd.Timestamp(hi))).sum())
            for name, (lo, hi) in windows.items()}

    return dict(code=code, start=str(pd.Timestamp(start).date()), end=str(pd.Timestamp(end).date()),
                cash_rate=float(cash_rate), bars=int(len(nav)), days=days,
                policies={name: ladder(nav, weights, cash_rate, flag)
                          for name, flag in (("buyhold", False), ("rebalanced", True))},
                transmission=transmission(sigma_cs=sigma_cs, market_vol=market_vol),
                power=dict(ic=[ic_power(n, sd) for sd in DAILY_IC_SD
                               for n in (days["reserve"], days["reserve"] + days["protected"])]))


def verdict(payload: dict) -> dict:
    """The three statements the programme depends on, read off the arithmetic."""
    hold = payload["policies"]["buyhold"]
    rebal = payload["policies"]["rebalanced"]
    sigma_cs = payload["transmission"]["sigma_cs"]
    universe = payload["transmission"]["universe"]

    def at(ladder_block, exposure):
        return next(row for row in ladder_block["rows"] if abs(row["exposure"] - exposure) < 1e-9)

    seven_hold, seven_rebal = at(hold, 0.7), at(rebal, 0.7)
    thirty = next(row for row in payload["transmission"]["rows"] if row["book_size"] == 30)
    target = seven_rebal["required_alpha"]
    years = payload["days"]["reserve"] / TRADING_DAYS

    sharpes = [row["sharpe_excess"] for row in rebal["rows"]
               if row["sharpe_excess"] is not None and row["annual_vol"] > 1e-8]
    best = [row for row in rebal["rows"] if row["calmar"] is not None and row["exposure"] > 0]
    calmar_best = max(best, key=lambda row: row["calmar"]) if best else None

    seven_hold, seven_rebal = at(hold, 0.7), at(rebal, 0.7)
    thirty = next(row for row in payload["transmission"]["rows"] if row["book_size"] == 30)
    target = seven_rebal["required_alpha"]
    years = payload["days"]["reserve"] / TRADING_DAYS

    grid = {}
    for exposure in (0.7, 0.9):
        needed = at(rebal, exposure)["required_alpha"]
        grid[f"{exposure:.1f}"] = [
            dict(horizon=int(horizon),
                 **{f"book_{size}": float(required_ic_at_horizon(
                     needed, size / universe, sigma_cs, horizon)) for size in BOOK_SIZES})
            for horizon in HORIZONS]

    return dict(full_annual=hold["full"]["annual_return"], full_drawdown=hold["full"]["max_drawdown"],
                full_sharpe_excess=(hold["full"]["annual_return"] - payload["cash_rate"])
                / hold["full"]["annual_vol"],
                buyhold_70=dict(annual_return=seven_hold["annual_return"],
                                max_drawdown=seven_hold["max_drawdown"],
                                sharpe_excess=seven_hold["sharpe_excess"]),
                rebalanced_70=dict(annual_return=seven_rebal["annual_return"],
                                   max_drawdown=seven_rebal["max_drawdown"],
                                   sharpe_excess=seven_rebal["sharpe_excess"],
                                   required_alpha=seven_rebal["required_alpha"]),
                sharpe_span_rebalanced=float(max(sharpes) - min(sharpes)) if sharpes else None,
                best_calmar_exposure=float(calmar_best["exposure"]) if calmar_best else None,
                required_ic_book_30=float(ic_from_alpha(target, 30 / universe, sigma_cs)),
                required_ic_table=grid,
                book_30_te=float(thirty["tracking_error"]),
                book_power_reserve=book_power(target, thirty["tracking_error"], years))


def _fmt(value, spec: str) -> str:
    """Format a metric that may be undefined (a cash-only path has no drawdown)."""
    return "—" if value is None else format(value, spec)


def _ladder_table(block: dict, caption: str) -> list:
    lines = [f"**{caption}**", "",
             "| 敞口 | 年化收益 | 年化波动 | 收益/波动 | 超额 Sharpe | 最大回撤 | Calmar | 需补 alpha |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in block["rows"]:
        lines.append("| {:.0%} | {:+.2%} | {} | {} | {} | {} | {} | {:+.2%} |".format(
            row["exposure"], row["annual_return"], _fmt(row["annual_vol"], ".2%"),
            _fmt(row["return_over_vol"], "+.3f"), _fmt(row["sharpe_excess"], "+.3f"),
            _fmt(row["max_drawdown"], ".2%"), _fmt(row["calmar"], "+.3f"),
            row["required_alpha"]))
    return lines + [""]


def _markdown(payload: dict) -> str:
    lines = [f"# 仓位算术与 alpha 需求（{payload['code']}）", "",
             f"窗口 {payload['start']} ~ {payload['end']}，{payload['bars']} 根日线，"
             f"现金年化 {payload['cash_rate']:.2%}。"
             f"保留窗 {payload['days']['reserve']} 天，保护窗 {payload['days']['protected']} 天。", ""]
    lines += ["## 一、定仓阶梯", ""]
    lines += _ladder_table(payload["policies"]["buyhold"], "买入后不再平衡（“就是买少一点”）")
    lines += _ladder_table(payload["policies"]["rebalanced"], "恒定权重每日再平衡（教科书降风险）")
    lines += ["## 二、IC 到组合 alpha", "",
              f"横截面离散度 σ_cs = {payload['transmission']['sigma_cs']:.2%}（重叠一年期截面标准差的日均值，"
              f"起点 2017-01~2023-12），本窗口市场波动 {payload['transmission']['market_vol']:.2%}。", "",
              "| 年化 IC | 10 只 | 20 只 | 30 只 | 50 只 |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for row in payload["transmission"]["alpha_table"]:
        lines.append("| {:.3f} | {:+.2%} | {:+.2%} | {:+.2%} | {:+.2%} |".format(
            row["ic"], row["book_10"], row["book_20"], row["book_30"], row["book_50"]))
    lines += ["", "## 三、各组合跟踪误差", "",
              "| 只数 | 占 300 只比例 | E[z\\|top] | 年化跟踪误差 |",
              "| ---: | ---: | ---: | ---: |"]
    for row in payload["transmission"]["rows"]:
        lines.append("| {} | {:.2%} | {:.3f} | {:.2%} |".format(
            row["book_size"], row["share"], row["top_mean_z"], row["tracking_error"]))
    lines += ["", "## 四、功效", "",
              "| 日 IC 标准差 | 窗口 | 天数 | 均值标准误 | 2σ 可辨 IC |",
              "| ---: | --- | ---: | ---: | ---: |"]
    windows = [("保留窗", payload["days"]["reserve"]),
               ("保留窗+保护窗", payload["days"]["reserve"] + payload["days"]["protected"])]
    for entry in payload["power"]["ic"]:
        label = "保留窗" if entry["days"] == windows[0][1] else "保留窗+保护窗"
        lines.append("| {:.2f} | {} | {} | {:.4f} | {:.4f} |".format(
            entry["daily_ic_sd"], label, entry["days"], entry["standard_error"],
            entry["detectable_2sigma"]))
    lines.append("")
    grid = payload.get("verdict", {}).get("required_ic_table", {})
    if grid:
        lines += ["## 五、达到目标 alpha 所需的持有期 IC", "",
                  "同为恒定权重再平衡的组合，E[z|top] 与组合只数对应。", ""]
        for exposure, rows in grid.items():
            lines += [f"**敞口 {float(exposure):.0%}**", "",
                      "| 持有期 | 10 只 | 20 只 | 30 只 | 50 只 |",
                      "| ---: | ---: | ---: | ---: | ---: |"]
            for row in rows:
                lines.append("| {} 日 | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                    row["horizon"], row["book_10"], row["book_20"],
                    row["book_30"], row["book_50"]))
            lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Price the alpha an exposure haircut requires.")
    parser.add_argument("--code", default="000300")
    parser.add_argument("--start", default="2005-01-04")
    parser.add_argument("--end", default="2026-10-30")
    parser.add_argument("--cash-rate", type=float, default=CASH_RATE)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    payload = measure(code=args.code, start=args.start, end=args.end, cash_rate=args.cash_rate)
    payload["verdict"] = verdict(payload)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = f"budget_{args.code}_{args.start[:4]}_{args.end[:4]}"
    target = Path(args.out) if args.out else REPORT_ROOT / f"{stem}.json"
    write_json(target, payload)
    target.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["verdict"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
