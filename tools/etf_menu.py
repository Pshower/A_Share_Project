"""Which fund to actually buy, and what its premium and tracking cost.

Every allocation decision in this project was made on indices, and the final plan inherited
that: it named index codes and left "which product" to the reader.  That gap matters more in
one place than the others.  A domestic equity ETF tracks its index to within a few basis
points, but a cross-border fund is capped by a quota, so when demand exceeds the quota the
price detaches from the underlying and the buyer pays a premium that can reach several
percent.  For a strategy holding thirty percent in the S&P, that premium is the difference
between the modelled return and the realised one.

The snapshot carries a real-time fair value and a discount rate per fund, so the premium can
be read rather than assumed, and the history is available for measuring how far each product
has actually drifted from the index it claims to track.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools import capm_backtest as capm
from tools.allocation_study import returns_of

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE = PROJECT_ROOT / "data" / "online" / "fund"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "etf_menu"
# Each leg of the plan, with the index it should track and the name fragment to search for.
LEGS = {
    "沪深300": dict(index="000300", keywords=("沪深300", "300ETF")),
    "中证500": dict(index="000905", keywords=("中证500", "500ETF")),
    "标普500": dict(index="sp500", keywords=("标普500", "标普")),
    "国债": dict(index="000012", keywords=("国债",)),
}
TOP = 5
# Bond funds are absent from the equity board, so they are addressed by code.
BOND_CODES = ("511010", "511260", "511020", "511090")


def snapshot(cache=CACHE):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "etf_spot.csv"
    if path.exists():
        frame = pd.read_csv(path, dtype={"代码": str})
    else:
        import akshare as ak
        frame = ak.fund_etf_spot_em()
        frame.to_csv(path, index=False)
    frame["代码"] = frame["代码"].astype(str).str.zfill(6)
    for column in ("最新价", "基金折价率", "总市值", "成交额", "IOPV实时估值"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def candidates(frame, keywords, top=TOP):
    names = frame["名称"].astype(str)
    mask = False
    for word in keywords:
        mask = mask | names.str.contains(word, regex=False)
    picked = frame[mask].copy()
    picked = picked.sort_values("总市值", ascending=False).head(top)
    return [dict(code=row["代码"], name=row["名称"],
                 size_yi=float(row["总市值"]) / 1e8 if pd.notna(row["总市值"]) else None,
                 premium=float(row["基金折价率"]) if pd.notna(row["基金折价率"]) else None,
                 turnover_yi=float(row["成交额"]) / 1e8 if pd.notna(row["成交额"]) else None,
                 price=float(row["最新价"]) if pd.notna(row["最新价"]) else None)
            for _, row in picked.iterrows()]


def history(code, start="20240101", end="20241231", cache=CACHE):
    cache = Path(cache)
    path = cache / f"etf_{code}.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        import akshare as ak
        frame = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start,
                                    end_date=end, adjust="")
        frame.to_csv(path, index=False)
    frame = frame.rename(columns={"日期": "date", "收盘": "close", "成交额": "amount"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame.dropna(subset=["close"]).set_index("date")["close"].sort_index()


def index_series(key, start="2024-01-02", end="2024-12-31"):
    if key == "sp500":
        from tools import global_frontier as gf
        return gf.foreign_close("sp500", ".INX", "us").loc[start:end]
    return pd.Series(np.nan, index=pd.DatetimeIndex([])).loc[start:end]


def tracking(code, key, start="2024-01-02", end="2024-12-31"):
    """Annualised gap and tracking error against the index the fund claims to follow."""
    try:
        fund = history(code)
    except Exception:                                  # noqa: BLE001 - provider varies
        return dict(status="unavailable")
    fund = fund.loc[start:end]
    if len(fund) < 100:
        return dict(status="short", days=int(len(fund)))
    returns = fund.pct_change().dropna()
    years = len(returns) / capm.TRADING_DAYS
    fund_annual = float((1 + returns).prod() ** (1 / years) - 1)
    block = dict(status="ready", days=int(len(fund)), fund_annual=fund_annual)
    try:
        if key == "sp500":
            from tools import global_frontier as gf
            raw = gf.foreign_close("sp500", ".INX", "us").loc[start:end]
            fx = gf.usdcny()
            benchmark = gf.yuan_returns(raw, fx)
        elif key == "000012":
            benchmark = None
        else:
            benchmark = returns_of(key, start, end)
    except Exception:                                  # noqa: BLE001
        benchmark = None
    if benchmark is not None and len(benchmark) > 100:
        joined = pd.concat([returns.rename("fund"), benchmark.rename("bench")], axis=1).dropna()
        if len(joined) > 100:
            bench_annual = float((1 + joined["bench"]).prod() **
                                 (capm.TRADING_DAYS / len(joined)) - 1)
            active = joined["fund"] - joined["bench"]
            block.update(benchmark_annual=bench_annual,
                         gap=fund_annual - bench_annual,
                         tracking_error=float(active.std(ddof=1) * np.sqrt(capm.TRADING_DAYS)))
    return block


def bond_products(codes=BOND_CODES):
    """Bond ETFs cannot be found by name in the equity board, so they are named by code.

    The spot snapshot carries 1,666 funds and not one of them contains the character for
    bond, which means the board it reads is equities, cross-border and commodities only.
    Bond funds have to come from a code list until a bond board is found.
    """
    rows = []
    for code in codes:
        try:
            series = history(code)
        except Exception as error:                     # noqa: BLE001 - provider varies
            rows.append(dict(code=code, status="unavailable", error=type(error).__name__))
            continue
        if series.empty:
            rows.append(dict(code=code, status="empty"))
            continue
        years = len(series) / capm.TRADING_DAYS
        rows.append(dict(code=code, status="ready", rows=int(len(series)),
                         first=str(series.index[0].date()), last=str(series.index[-1].date()),
                         annual=float(series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1))
    return rows


def nav_premium(code, start="20240101", end="20241231"):
    """Premium measured against published net asset value rather than an intraday estimate.

    A cross-border fund's intraday fair value is struck on the previous American close, so
    the snapshot's discount field is unusable there: it reports double-digit discounts on
    funds that were trading at a premium.  The published NAV is a day old by construction,
    which is why the price is compared with the previous session's value and the difference
    is reported as a range rather than a point.
    """
    try:
        import akshare as ak
        nav = ak.fund_etf_fund_info_em(fund=code, start_date=start, end_date=end)
    except Exception as error:                         # noqa: BLE001
        return dict(code=code, status="unavailable", error=type(error).__name__)
    nav = nav.rename(columns={"净值日期": "date", "单位净值": "nav"})
    if "date" not in nav.columns or "nav" not in nav.columns:
        return dict(code=code, status="unexpected_columns", columns=list(nav.columns))
    nav["date"] = pd.to_datetime(nav["date"], errors="coerce")
    nav["nav"] = pd.to_numeric(nav["nav"], errors="coerce")
    nav = nav.dropna(subset=["date", "nav"]).set_index("date")["nav"].sort_index()
    if nav.empty:
        return dict(code=code, status="empty")
    try:
        price = history(code)
    except Exception as error:                         # noqa: BLE001
        return dict(code=code, status="no_price", error=type(error).__name__)
    joined = pd.concat([price.rename("price"), nav.rename("nav")], axis=1).dropna()
    if len(joined) < 20:
        joined = pd.concat([price.rename("price"), nav.rename("nav").shift(1)], axis=1).dropna()
    if len(joined) < 20:
        return dict(code=code, status="short", rows=int(len(joined)))
    spread = joined["price"] / joined["nav"] - 1.0
    return dict(code=code, status="ready", observations=int(len(joined)),
                mean=float(spread.mean()), median=float(spread.median()),
                first=str(joined.index[0].date()), last=str(joined.index[-1].date()),
                latest=float(spread.iloc[-1]))


def measure():
    frame = snapshot()
    payload = dict(status="ready", universe=int(len(frame)), legs={},
                   note="premium is the snapshot discount rate; positive means the fund trades "
                        "above its net asset value")
    for leg, spec in LEGS.items():
        rows = candidates(frame, spec["keywords"])
        for row in rows:
            row.update(tracking(row["code"], spec["index"]))
        payload["legs"][leg] = dict(index=spec["index"], candidates=rows)
    return payload


def _markdown(payload):
    lines = ["# 买哪个 ETF：规模、折溢价与跟踪误差", "",
             "全市场 **%d** 只 ETF 里，按规模取每类前几名。窗口 2024-01-02 ~ 2024-12-31。" % payload["universe"], ""]
    for leg, block in payload["legs"].items():
        lines += ["## %s（应跟踪 `%s`）" % (leg, block["index"]), "",
                  "| 代码 | 名称 | 规模（亿） | 日成交（亿） | **折价率** | 年化 | 跟踪误差 | 年化差 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in block["candidates"]:
            # sizes and premiums arrive already scaled (亿 and percent); returns and their
            # differences arrive as fractions and must be multiplied out
            def fmt(value, pattern, scale=1.0):
                if isinstance(value, (int, float)) and np.isfinite(value):
                    return pattern % (value * scale)
                return "—"
            lines.append("| %s | %s | %s | %s | **%s%%** | %s%% | %s%% | %s%% |" % (
                row["code"], row["name"], fmt(row.get("size_yi"), "%.1f"),
                fmt(row.get("turnover_yi"), "%.2f"), fmt(row.get("premium"), "%+.2f"),
                fmt(row.get("fund_annual"), "%+.2f", 100.0),
                fmt(row.get("tracking_error"), "%.2f", 100.0),
                fmt(row.get("gap"), "%+.2f", 100.0)))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Which ETF to buy for each leg.")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    from src.data.preprocess import write_json
    write_json(target / "etf_menu.json", payload)
    (target / "etf_menu.md").write_text(_markdown(payload), encoding="utf-8")
    print("universe %d, legs %d" % (payload["universe"], len(payload["legs"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
