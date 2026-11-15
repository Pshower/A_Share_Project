"""Turn the final plan into the orders a broker would accept, for a given account.

[The plan](doc/final_plan_20261109.md) fixes the mechanism and nothing implements it: every
tool in this repository either researches or simulates.  This one is the last mile -- give
it the account value, the peak value the drawdown cap is measured from, what is held now,
and the latest prices, and it returns the target weights and the lot-rounded orders.

The rule is copied, not invented:

    buffer     = NAV - floor * peak
    risky      = clip(multiplier * buffer / NAV, 0, 1)
    if |risky - risky_held| < band: hold, do not trade
    510300 = 510500 = 0.25 * risky       (A-share half, split evenly)
    513500 = 0.50 * risky                (S&P half, in RMB)
    511010 = 1 - risky                   (treasury leg)

Fees, QDII premium and the fund expense ratios are *not* modelled here; the plan says they
cost another 0.5-1pp a year.  This tool allocates, it does not promise returns.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLOOR, MULTIPLIER, BAND = 0.90, 6.0, 0.05
LEGS = {"510300": 0.25, "510500": 0.25, "513500": 0.50}     # share of the risky sleeve
SAFE = "511010"
LOT = 100


def risky_weight(nav, peak, floor=FLOOR, multiplier=MULTIPLIER):
    """CPPI exposure, capped at zero and one."""
    if nav <= 0 or peak <= 0:
        raise ValueError("NAV and peak must be positive")
    buffer = nav - floor * peak
    return float(min(max(multiplier * buffer / nav, 0.0), 1.0))


def target_weights(nav, peak, held_risky=None, band=BAND, floor=FLOOR, multiplier=MULTIPLIER):
    """Targets per leg, with the no-trade band applied to the risky sleeve as a whole."""
    risky = risky_weight(nav, peak, floor, multiplier)
    if held_risky is not None and abs(risky - held_risky) < band:
        risky_applied = float(held_risky)
        traded = False
    else:
        risky_applied = risky
        traded = True
    weights = {code: weight * risky_applied for code, weight in LEGS.items()}
    weights[SAFE] = 1.0 - risky_applied
    return dict(target=risky, applied=risky_applied, traded=traded, weights=weights)


def fee_for(row):
    """Commission on one order row, with the broker's five-yuan floor."""
    amount = abs(float(row["delta"]) * float(row["price"]))
    return max(5.0, amount * 0.0003) if amount else 0.0


def orders(weights, nav, prices, held=None, lot=LOT):
    """Lot-rounded share targets and the deltas needed to reach them."""
    held = held or {}
    # A holding outside the four legs is invisible to the target weights, so the orders would buy the
    # full allocation on top of it and the reported cash would be fiction.  Refuse instead of guessing:
    # the caller either sells it or tells the tool it is part of NAV deliberately.
    off_plan = sorted(code for code, shares in held.items() if int(shares) > 0 and code not in weights)
    if off_plan:
        raise ValueError("account holds %s, which the plan does not allocate; sell it first"
                         % ", ".join(off_plan))
    missing = [code for code in weights if code not in prices or not prices[code] > 0]
    if missing:
        raise ValueError("No price for %s" % ", ".join(sorted(missing)))
    plan, cash_after = [], 0.0
    for code, weight in weights.items():
        price = float(prices[code])
        target_value = weight * nav
        target_shares = math.floor(target_value / price / lot) * lot
        current = int(held.get(code, 0))
        delta = target_shares - current
        plan.append(dict(code=code, weight=round(weight, 6), price=price,
                         target_value=round(target_shares * price, 2),
                         target_shares=target_shares, held=current, delta=delta,
                         action="buy" if delta > 0 else ("sell" if delta < 0 else "hold")))
        cash_after += current * price - target_shares * price
    invested = sum(row["target_value"] for row in plan)
    # The lot rounding leaves a few yuan; the commission on four buys can be larger than that, and
    # the gate then refuses the whole order set for a shortfall of pocket change.  Trim one lot from
    # the largest buy until the commission fits inside the leftover cash.
    while invested and nav - invested < sum(fee_for(row) for row in plan if row["delta"] > 0):
        largest = max((row for row in plan if row["delta"] > 0), key=lambda row: row["delta"],
                      default=None)
        if largest is None or largest["target_shares"] < lot:
            break
        largest["target_shares"] -= lot
        largest["target_value"] = round(largest["target_shares"] * largest["price"], 2)
        largest["delta"] = largest["target_shares"] - largest["held"]
        largest["action"] = "buy" if largest["delta"] > 0 else (
            "sell" if largest["delta"] < 0 else "hold")
        invested = sum(row["target_value"] for row in plan)
    return dict(nav=nav, invested=round(invested, 2), cash=round(nav - invested, 2),
                orders=plan)


def decide(nav, peak, prices, held=None, band=BAND, floor=FLOOR, multiplier=MULTIPLIER):
    """The whole calculation, with the current exposure derived from what is held."""
    held = held or {}
    risky_now = sum(held.get(code, 0) * float(prices[code]) for code in LEGS
                    if code in prices) / nav if nav > 0 else 0.0
    targets = target_weights(nav, peak, held_risky=risky_now if held else None,
                             band=band, floor=floor, multiplier=multiplier)
    return dict(targets, **orders(targets["weights"], nav, prices, held=held))


def prefix_for(code):
    """Shanghai or Shenzhen prefix for an ETF code.

    The repository's own Sina provider derives the exchange from "starts with 6", which
    sends every fund code to Shenzhen and 404s: 510300 is ``sh510300``.  Funds need their
    own rule, and it goes here rather than in the shared provider so nothing else moves.
    """
    return "sh" if code.startswith("5") else "sz"


def fetch_prices(codes):
    """Latest closing price per ETF from the Sina fund history endpoint.

    The plan rebalances monthly, so a close is the right price to decide on; the realtime
    quote path in ``src.data.online`` rejects these codes and Eastmoney refused the
    connection when this was written.
    """
    import akshare as ak
    prices, asof = {}, None
    for code in codes:
        frame = ak.fund_etf_hist_sina(symbol=prefix_for(code) + code)
        if frame is None or not len(frame):
            raise ValueError("No ETF history for %s" % code)
        prices[code] = float(frame["close"].iloc[-1])
        stamp = str(frame["date"].iloc[-1])
        asof = stamp if asof is None else max(asof, stamp)
    return prices, asof


def main(argv=None):
    parser = argparse.ArgumentParser(description="Target weights and orders for the final plan.")
    parser.add_argument("--nav", type=float, required=True, help="Current account value.")
    parser.add_argument("--peak", type=float, required=True, help="High-water account value.")
    parser.add_argument("--held", default="{}", help='JSON {"510300": shares, ...}')
    parser.add_argument("--prices", default=None, help='JSON {"510300": price, ...}; omit to fetch.')
    parser.add_argument("--band", type=float, default=BAND)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    held = {str(k): int(v) for k, v in json.loads(args.held).items()}
    codes = list(LEGS) + [SAFE]
    if args.prices:
        prices, asof = json.loads(args.prices), "given on the command line"
    else:
        prices, asof = fetch_prices(codes)
    result = decide(args.nav, args.peak, prices, held=held, band=args.band)
    result["prices_asof"] = asof
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
