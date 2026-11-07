"""Derive a tradability ledger from unadjusted bars and a trading calendar.

The research environment decides at the close and trades at the next open, but it has no
suspension, limit or corporate-action state: a halted stock can still be "bought", and a
limit-up open still fills. The vendor does not publish those fields historically, but three
of them are derivable from data already on disk:

* suspension - a trading day inside a stock's listed span with no bar;
* limit up/down - the unadjusted close equal to the exchange limit price of the last close;
* corporate actions - a relative jump in the ``hfq``/``unadjusted`` ratio.

Every value here is derived, not observed, so it stays ``model_ready=false``: the board limit
rates ignore historical ST status and the unlisted first days, and the corporate-action
table records a price-factor jump without classifying it as a dividend or a split.

Read-only over existing snapshots. No network access, no training, no market simulation.
"""

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from src.data.preprocess import ROOT, write_json
from src.data.snapshots import Snapshots

# Main board 10%, ChiNext and STAR 20%, Beijing 30%. ST names use 5% but the snapshot has no
# historical ST flag, so a 5% limit day is reported as ordinary trading instead.
LIMIT_RATES = (("688", 0.20), ("689", 0.20), ("30", 0.20), ("8", 0.30), ("4", 0.30), ("920", 0.30))
ATOL = 0.005
CORPORATE_JUMP = 2e-3


def limit_rate(code):
    for prefix, rate in LIMIT_RATES:
        if code.startswith(prefix):
            return rate
    return 0.10


def exchange_round(value):
    """Round half up to two decimals, the way a limit price is published."""
    return np.floor(np.asarray(value, dtype=float) * 100.0 + 0.5) / 100.0


def limit_prices(previous_close, rate):
    previous_close = np.asarray(previous_close, dtype=float)
    return exchange_round(previous_close * (1.0 + rate)), exchange_round(previous_close * (1.0 - rate))


def corporate_actions(code, hfq, unadjusted):
    """Return factor-jump rows for one stock from the two saved price bases."""
    merged = (hfq[["date", "close"]].rename(columns={"close": "hfq"})
              .merge(unadjusted[["date", "close"]].rename(columns={"close": "raw"}), on="date"))
    if merged.empty:
        return pd.DataFrame(columns=["code", "date", "factor_before", "factor_after", "ratio"])
    merged = merged.sort_values("date")
    factor = (merged["hfq"] / merged["raw"]).to_numpy(dtype=float)
    relative = np.abs(np.diff(factor) / factor[:-1])
    index = np.where(relative > CORPORATE_JUMP)[0] + 1
    return pd.DataFrame(dict(code=code, date=merged["date"].to_numpy()[index],
                             factor_before=factor[index - 1], factor_after=factor[index],
                             ratio=(factor[index] / factor[index - 1]) - 1.0))


def stock_ledger(code, unadjusted, calendar):
    """One row per trading day inside the stock's listed span."""
    bars = unadjusted.sort_values("date").set_index("date")
    start, end = bars.index.min(), bars.index.max()
    days = calendar[(calendar >= start) & (calendar <= end)]
    # A calendar slice keeps the original row labels, and this ledger indexes rows positionally, so
    # rebuild a zero-based frame; otherwise labels and positions of the same array drift apart.
    frame = pd.DataFrame({"date": pd.to_datetime(days).to_numpy()})
    frame["code"] = code
    joined = frame.join(bars[["open", "high", "low", "close"]], on="date")
    frame["traded"] = joined["close"].notna().to_numpy()
    for column in ("open", "high", "low", "close"):
        frame[column] = joined[column].to_numpy()
    frame["suspended"] = ~frame["traded"]

    traded = frame[frame["traded"]]
    previous = traded["close"].shift(1)
    rate = limit_rate(code)
    up, down = limit_prices(previous, rate)
    at_up = np.isclose(traded["close"], up, atol=ATOL)
    at_down = np.isclose(traded["close"], down, atol=ATOL)
    frame.loc[traded.index, "previous_close"] = previous
    frame.loc[traded.index, "limit_up"] = up
    frame.loc[traded.index, "limit_down"] = down
    # A day without a trade cannot sit at a limit, so these stay real booleans instead of nullable
    # objects; mixing NaN into them turns every downstream mask into an object comparison.
    frame["at_limit_up"], frame["at_limit_down"], frame["one_word_limit_up"] = False, False, False
    frame.loc[traded.index, "at_limit_up"] = at_up
    frame.loc[traded.index, "at_limit_down"] = at_down
    frame.loc[traded.index, "one_word_limit_up"] = np.isclose(traded["open"], traded["close"], atol=ATOL) & at_up
    frame["at_limit_rate"] = rate

    # The next executable open belongs to the next traded day *after* this row, so a suspension
    # stretches it to the resumption open. Positions are searched, not filled, because a fill would
    # either return the row's own open or skip a traded day.
    traded_rows = frame.index[frame["traded"]].to_numpy()
    slot = np.searchsorted(traded_rows, frame.index.to_numpy(), side="right")
    has_next = slot < len(traded_rows)
    target = np.minimum(slot, max(len(traded_rows) - 1, 0))
    if len(traded_rows):
        frame["next_open"] = np.where(has_next, traded["open"].to_numpy(dtype=float)[target], np.nan)
        # Map positions back through the date column. Passing the integer positions straight to
        # pd.to_datetime would read them as nanoseconds since the epoch instead of trading days.
        upcoming = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
        # ``target`` indexes the traded rows, so it must pass through them to reach the dense dates.
        upcoming.loc[has_next] = frame["date"].to_numpy()[traded_rows[target[has_next]]]
        frame["next_traded_date"] = upcoming
    else:
        frame["next_open"] = np.nan
        frame["next_traded_date"] = pd.NaT
    frame["has_next"] = has_next
    limit_up_next = exchange_round(frame["close"].to_numpy(dtype=float) * (1.0 + rate))
    limit_down_next = exchange_round(frame["close"].to_numpy(dtype=float) * (1.0 - rate))
    frame["blocked_buy"] = frame["has_next"] & (frame["next_open"].to_numpy() >= limit_up_next - ATOL)
    frame["blocked_sell"] = frame["has_next"] & (frame["next_open"].to_numpy() <= limit_down_next + ATOL)
    frame["pct_change"] = frame["close"] / frame["previous_close"] - 1.0
    return frame


def summarize(ledger, actions):
    traded = ledger["traded"]
    executable = ledger["has_next"] & traded
    rows = dict(rows=int(len(ledger)), codes=int(ledger["code"].nunique()),
                first_date=str(ledger["date"].min().date()), last_date=str(ledger["date"].max().date()),
                traded_rows=int(traded.sum()), suspended_rows=int((~traded).sum()))
    rows["suspended_codes"] = int(ledger.loc[~traded, "code"].nunique())
    rows["suspended_share"] = round(float((~traded).mean()), 6)
    rows["limit_up_rows"] = int(ledger["at_limit_up"].fillna(False).sum())
    rows["limit_down_rows"] = int(ledger["at_limit_down"].fillna(False).sum())
    rows["one_word_limit_up_rows"] = int(ledger["one_word_limit_up"].fillna(False).sum())
    rows["blocked_buy_rows"] = int(ledger["blocked_buy"].fillna(False).sum())
    rows["blocked_sell_rows"] = int(ledger["blocked_sell"].fillna(False).sum())
    blocked_buy = ledger["blocked_buy"].fillna(False)
    rows["blocked_buy_codes"] = int(ledger.loc[blocked_buy, "code"].nunique())
    rows["blocked_buy_days"] = int(ledger.loc[blocked_buy, "date"].nunique())
    rows["blocked_buy_share_of_executable"] = round(float(ledger.loc[executable, "blocked_buy"].mean()), 6)
    rows["blocked_sell_share_of_executable"] = round(float(ledger.loc[executable, "blocked_sell"].mean()), 6)
    rows["corporate_action_events"] = int(len(actions))
    rows["corporate_action_codes"] = int(actions["code"].nunique()) if len(actions) else 0
    return rows


def build(history_id, calendar_id, output, *, codes=None, root=ROOT):
    store = Snapshots(root)
    _, history = store.get("history", history_id)
    _, calendar_directory = store.get("calendar", calendar_id)
    calendar_frame = pd.read_parquet(calendar_directory / "calendar.parquet")
    calendar = pd.to_datetime(calendar_frame["trade_date"]).sort_values().reset_index(drop=True)
    files = {path.stem: path for path in sorted((history / "unadjusted").glob("*.parquet"))}
    if not files:
        raise ValueError("History snapshot has no unadjusted bars")
    if codes:
        missing = sorted(set(codes) - set(files))
        if missing:
            raise ValueError(f"Requested codes are absent from the snapshot: {missing}")
        files = {code: path for code, path in files.items() if code in set(codes)}
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    staging = output.parent / ("." + output.name + ".building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        frames, action_frames = [], []
        for code, path in files.items():
            unadjusted = pd.read_parquet(path)
            frames.append(stock_ledger(code, unadjusted, calendar))
            hfq_path = history / "hfq" / f"{code}.parquet"
            if hfq_path.exists():
                action_frames.append(corporate_actions(code, pd.read_parquet(hfq_path), unadjusted))
        ledger = pd.concat(frames, ignore_index=True)
        actions = (pd.concat(action_frames, ignore_index=True) if action_frames
                   else pd.DataFrame(columns=["code", "date", "factor_before", "factor_after", "ratio"]))
        ledger = ledger.sort_values(["date", "code"]).reset_index(drop=True)
        actions = actions.sort_values(["date", "code"]).reset_index(drop=True)
        ledger.to_parquet(staging / "ledger.parquet", index=False)
        actions.to_parquet(staging / "corporate_actions.parquet", index=False)
        summary = summarize(ledger, actions)
        summary.update(request=dict(history_snapshot=history_id, calendar_snapshot=calendar_id,
                                    codes=len(files)),
                       limit_rates={prefix: rate for prefix, rate in LIMIT_RATES},
                       model_ready=False,
                       limitation=("Derived state: board limit rates ignore historical ST status and the "
                                   "first listing day, suspension means a missing bar inside the listed "
                                   "span, and corporate actions are factor jumps rather than classified "
                                   "dividend or split events."))
        write_json(staging / "summary.json", summary)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    staging.rename(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True, help="history snapshot id holding unadjusted and hfq bars")
    parser.add_argument("--calendar", required=True, help="calendar snapshot id")
    parser.add_argument("--output", required=True, help="new directory for the derived ledger")
    parser.add_argument("--codes", nargs="*", default=[], help="optional subset for a smoke run")
    arguments = parser.parse_args()
    summary = build(arguments.history, arguments.calendar, arguments.output, codes=arguments.codes or None)
    for key in ("rows", "codes", "first_date", "last_date", "suspended_rows", "suspended_codes",
                "limit_up_rows", "limit_down_rows", "one_word_limit_up_rows", "blocked_buy_rows",
                "blocked_sell_rows", "blocked_buy_share_of_executable", "corporate_action_events"):
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    sys.exit(main())
