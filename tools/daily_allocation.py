"""今天该持有什么——把五轮研究收成一条可执行建议，并且真的落到账本上。

这条建议只有一个自由度：**风险腿占总资产多少**，由 CPPI 从账本自己的净值和峰值算出来。
其余全是固定的：风险腿内部 25/25/50，安全腿用国债（可选一半现金）。所有资产的权重都会列出来，
包括那些**明确为 0** 的——因为"不投黄金"和"投 25% 沪深300"是同一份建议的两半。

两个模式：
  * 默认只读，输出建议 JSON（Web 页面的数据源）；
  * ``--apply`` 把建议落进账本：先卖后买，写成交，再按当日收盘重新记账。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from tools import account_ledger as ledger
from tools import plan_orders

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONLINE = PROJECT_ROOT / "data" / "online"
ACCOUNTS = PROJECT_ROOT / "reports" / "account"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "daily_allocation"

LOT = plan_orders.LOT
FLOOR, MULTIPLIER, BAND = plan_orders.FLOOR, plan_orders.MULTIPLIER, plan_orders.BAND
RISKY = dict(plan_orders.LEGS)                       # 25% 300 / 25% 500 / 50% S&P
SAFE = plan_orders.SAFE                              # 511010
CASH_ETF = "511880"                                  # money-market ETF, the cash half of the safe leg
# A quote older than this is refused by ``apply``: over a long weekend, Friday's close is 3 days
# old, so 4 is the smallest number that does not block the Monday run when data is a day late.
MAX_AGE_DAYS = 4

# Every asset class the study covered, and why its weight is what it is.  The zeros are part of
# the recommendation: each one was tested and rejected on evidence, not on taste.
UNIVERSE = [
    ("现金 / 货基", "cash", "闲置现金；一半现金模式下另配 511880"),
    ("存款", None, "同风险更低收益，C6 -1.00pp"),
    ("国债（中短）", SAFE, "安全腿主体，唯一过 A1 的配置"),
    ("长久期国债", None, "真实 7-10 年桶 +0.44pp，但来自一次性重定价，从 1.6% 不可复制"),
    ("信用债", None, "配对后仅 +0.09pp，低于 0.3pp 门槛"),
    ("企业债 AA-", None, "+1.20pp 但不可执行（无 ETF、指数不记违约残值），只作上界"),
    ("沪深300", "510300", "风险腿 25%"),
    ("中证500", "510500", "风险腿 25%"),
    ("标普500（人民币）", "513500", "风险腿 50%"),
    ("港股", None, "-0.45pp"),
    ("黄金", None, "9 年样本 +1.44pp，A1 擦线不过，不采纳"),
    ("银行理财", None, "无公开时序；上界 = 货基 +20~50bp"),
]


def cached_prices(codes, online: Path | None = None) -> tuple[dict, pd.Timestamp, dict]:
    """Last common close across the codes, from the local fund cache.  Never touches the network."""
    online = online or ONLINE
    series = {}
    for code in codes:
        path = online / "fund" / f"hist_{code}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run tools.fund_vehicles --refresh once")
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"])
        series[code] = frame.set_index("date")["close"].astype(float).sort_index()
    common = None
    for frame in series.values():
        common = frame.index if common is None else common.intersection(frame.index)
    if common is None or not len(common):
        raise ValueError("the cached fund histories share no date")
    stamp = max(common)
    prices = {code: float(frame.loc[stamp]) for code, frame in series.items()}
    return prices, stamp, {code: stamp for code in series}


def weights_for(nav: float, peak: float, cash_safe: bool = False) -> tuple[dict, float]:
    """CPPI exposure, then the fixed split inside each leg."""
    risky = plan_orders.risky_weight(nav, peak)
    weights = {code: risky * part for code, part in RISKY.items()}
    if cash_safe:
        weights[SAFE] = (1.0 - risky) * 0.5
        weights[CASH_ETF] = (1.0 - risky) * 0.5
    else:
        weights[SAFE] = 1.0 - risky
    return weights, risky


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def book_state(book: str, base: Path | None = None) -> dict:
    """One book's cash, positions and peak -- read directly, never through ledger globals.

    ``ledger.use()`` switches module-level paths, so a web request that called it would change
    which account every *other* reader in the process sees.  The API is read-only and shared, so
    this path builds the state from explicit files instead.
    """
    base = (base or ACCOUNTS) / book
    cfg = json.loads((base / "account.json").read_text(encoding="utf-8"))
    trades = _rows(base / "trades.jsonl")
    snapshots = _rows(base / "snapshots.jsonl")
    replayed = ledger.replay(trades, cfg["capital"])
    last = snapshots[-1] if snapshots else None
    trailing = [row for row in trades if last is None or str(row["date"]) > str(last["date"])]
    return dict(cash=replayed["cash"], positions=replayed["positions"],
                snapshot_nav=float(last["nav"]) if last else float(cfg["capital"]),
                snapshot_date=str(last["date"]) if last else None,
                peak=ledger.peak_of(snapshots, cfg["capital"]),
                unmarked_trades=len(trailing), capital=float(cfg["capital"]))


def recommendation(book: str = "paper", cash_safe: bool = False,
                   online: Path | None = None, base: Path | None = None) -> dict:
    """The whole answer for one book: what to hold, what to do, and why the rest is zero."""
    state = book_state(book, base=base)
    held = {code: int(shares) for code, shares in state["positions"].items() if int(shares) > 0}
    codes = list(RISKY) + [SAFE] + ([CASH_ETF] if cash_safe else [])
    prices, stamp, price_dates = cached_prices(sorted(set(codes) | set(held)), online)
    nav = ledger.nav(state["positions"], state["cash"], prices)
    peak = max(float(state["peak"]), nav)
    weights, risky_target = weights_for(nav, peak, cash_safe)

    risky_now = sum(held.get(code, 0) * prices[code] for code in RISKY) / nav
    # The no-trade band applies to the risky sleeve as a whole; inside it, nothing is booked.
    drift = risky_target - risky_now
    if abs(drift) < BAND:
        plan_weights = {code: risky_now * part for code, part in RISKY.items()}
        plan_weights[SAFE] = 1.0 - risky_now
        action = "hold"
    else:
        plan_weights = weights
        action = "rebalance"

    plan = plan_orders.orders(plan_weights, nav, prices, held)
    warnings = []
    if state["unmarked_trades"]:
        warnings.append("%d 笔成交未记账，先跑 tools.account_ledger mark" % state["unmarked_trades"])
    if abs(nav - float(state["snapshot_nav"])) > 1.0:
        warnings.append("按最新收盘重估后净值 %.2f，与账本快照 %.2f 不同"
                        % (nav, float(state["snapshot_nav"])))

    sleeves = []
    for row in plan["orders"]:
        sleeves.append(dict(code=row["code"], weight=round(row["weight"], 4),
                            price=row["price"], held=row["held"],
                            target_shares=row["target_shares"], delta=row["delta"],
                            action=row["action"], target_value=row["target_value"],
                            fee=round(plan_orders.fee_for(row), 2)))
    idle_cash = float(plan["cash"])
    idle_weight = idle_cash / nav
    # Lot rounding always leaves a few thousand yuan uninvested.  It is the least risky money in
    # the book, so the table books it as part of the safe leg instead of adding a thirteenth row
    # that would make the weights sum to 101%.
    assets = []
    for name, code, note in UNIVERSE:
        if name.startswith("现金"):
            weight, value = idle_weight, idle_cash
            note = "整手取整留下的未投部分，计入安全腿（" + note + "）"
        elif code is None:
            weight, value = 0.0, 0.0
        else:
            row = next((sleeve for sleeve in sleeves if sleeve["code"] == code), None)
            weight = (row["weight"] if row else 0.0) - (idle_weight if code == SAFE else 0.0)
            value = (row["target_value"] if row else 0.0) - (idle_cash if code == SAFE else 0.0)
        assets.append(dict(name=name, code=code, weight=round(float(weight), 4),
                           value=round(float(value), 2), note=note))

    return dict(book=book, as_of=str(stamp.date()), generated_at=str(pd.Timestamp.now())[:19],
                cash_safe=bool(cash_safe), nav=round(nav, 2), peak=round(peak, 2),
                drawdown=round(nav / peak - 1.0, 4), cash=round(float(state["cash"]), 2),
                risky_target=round(risky_target, 4), risky_now=round(risky_now, 4),
                drift=round(drift, 4), band=BAND, action=action,
                invested=plan["invested"], idle_cash=round(idle_cash, 2),
                prices={code: float(value) for code, value in sorted(prices.items())},
                price_dates={code: str(day.date()) for code, day in price_dates.items()},
                sleeves=sleeves, assets=assets, warnings=warnings,
                note="权重按上一交易日收盘计算；未成交前不进入账本")


def apply(result: dict, dry: bool = False, max_age_days: int | None = MAX_AGE_DAYS) -> dict:
    """Book the recommendation: sells first, then buys, then re-mark the day.

    Refuses to book a quote that is older than ``max_age_days``: an automated job that runs after
    a failed data refresh would otherwise trade the book at last week's prices and label it today.
    """
    age = (pd.Timestamp(pd.Timestamp.now().date()) - pd.Timestamp(result["as_of"])).days
    if max_age_days is not None and age > max_age_days and not dry:
        raise ValueError("报价停在 %s（%d 天前），拒绝按它记账；先刷新行情或显式放宽 max_age_days"
                         % (result["as_of"], age))
    ledger.use(result["book"])
    stamp = pd.Timestamp(result["as_of"])
    opening = not any(int(shares) > 0 for shares in
                      book_state(result["book"])["positions"].values())
    source = "unified_open" if opening else "unified"
    trades = []
    if result["action"] == "rebalance":
        ordered = sorted(result["sleeves"], key=lambda row: 0 if row["delta"] < 0 else 1)
        for row in ordered:
            if row["delta"] == 0:
                continue
            side = "sell" if row["delta"] < 0 else "buy"
            trades.append(dict(date=str(stamp.date()), code=row["code"], action=side,
                               shares=abs(int(row["delta"])), price=row["price"],
                               fee=row["fee"], source=source))
            if not dry:
                ledger.record_trade(stamp.date(), row["code"], side, abs(int(row["delta"])),
                                    price=row["price"], source=source)
    marked = None
    if not dry:
        marked = ledger.mark(stamp.date(), prices=result["prices"],
                             price_dates=dict(result["price_dates"]))
    log = dict(book=result["book"], as_of=result["as_of"], action=result["action"],
               opening=opening, age_days=int(age), nav=result["nav"],
               risky_target=result["risky_target"], risky_now=result["risky_now"],
               trades=trades, marked_nav=None if marked is None else marked["nav"],
               applied_at=str(pd.Timestamp.now())[:19])
    if not dry:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with (OUTPUT / "decisions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log, ensure_ascii=False) + "\n")
    return log


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    book = "paper"
    for argument in argv:
        if argument.startswith("--book="):
            book = argument.split("=", 1)[1]
    result = recommendation(book=book, cash_safe="--cash-safe" in argv)
    print("[%s] as of %s  NAV %s  drawdown %+.2f%%  peak %s"
          % (result["book"], result["as_of"], format(result["nav"], ",.2f"),
             result["drawdown"] * 100, format(result["peak"], ",.0f")))
    print("  risky target %.2f%%  now %.2f%%  drift %+.2f%%  band %.0f%%  -> %s"
          % (result["risky_target"] * 100, result["risky_now"] * 100, result["drift"] * 100,
             result["band"] * 100, result["action"]))
    print("  %-8s%8s%9s%8s%9s%8s%7s" % ("code", "weight", "price", "held", "target", "delta", "fee"))
    for row in result["sleeves"]:
        print("  %-8s%7.2f%%%9.3f%8d%9d%+8d%7.2f"
              % (row["code"], row["weight"] * 100, row["price"], row["held"],
                 row["target_shares"], row["delta"], row["fee"]))
    print("  idle cash %s" % format(result["idle_cash"], ",.2f"))
    for warning in result["warnings"]:
        print("  WARNING:", warning)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "latest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    if "--apply" in argv:
        log = apply(result)
        print("applied: %d fill(s), marked NAV %s" % (len(log["trades"]),
                                                      format(log["marked_nav"], ",.2f")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
