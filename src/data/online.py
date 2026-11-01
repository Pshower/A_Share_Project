"""Explicit daily-history and realtime-quote access with auditable snapshots."""

import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import json
import math
from pathlib import Path
import re
import shutil
import random
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from src.data.preprocess import ROOT, RENAME, file_hash, write_json
from src.data.snapshots import Snapshots, utc_now
from src.runtime import events
from src.data.rate_limit import AdaptivePacer, retry_delay, transient, wait

CHINA = ZoneInfo("Asia/Shanghai")

MARKET_INDICES = {
    "000001": "上证指数",
    "000300": "沪深300",
    "000905": "中证500",
    "000852": "中证1000",
    "399001": "深证成指",
    "399006": "创业板指",
}
MARKET_NODE = "hs_a"
MARKET_PAGE_SIZE = 100


def valid_codes(codes):
    if not codes or len(codes) > 500 or len(set(codes)) != len(codes) or any(not re.fullmatch(r"[0-9]{6}", c) for c in codes):
        raise ValueError("Expected unique six-digit stock codes")
    return codes


class Eastmoney:
    name = "eastmoney"

    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.limiter = AdaptivePacer(self.root)
        self.deadline = None

    def pace(self):
        self.limiter.acquire(self.deadline)

    def history(self, code, start, end, basis):
        if basis not in {"hfq", "unadjusted"}:
            raise ValueError("Explicit history price basis required")
        self.pace()
        # The installed SDK parses JSON before checking HTTP status, losing 429 headers.
        response = requests.get("https://push2his.eastmoney.com/api/qt/stock/kline/get", params={
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
            "ut": "7eea3edcaed734bea9cbfc24409ed989", "klt": "101",
            "fqt": "2" if basis == "hfq" else "0",
            "secid": ("1." if code.startswith("6") else "0.") + code,
            "beg": start.replace("-", ""), "end": end.replace("-", ""),
        }, timeout=(5, 12))
        response.raise_for_status()
        body = response.json()
        data = body.get("data")
        if body.get("rc") != 0 or not isinstance(data, dict) or str(data.get("code")) != code:
            raise ValueError("Provider history identity/status mismatch")
        lines = data.get("klines")
        if not isinstance(lines, list) or not lines:
            raise ValueError("Provider returned no history")
        records = [line.split(",") for line in lines if isinstance(line, str)]
        if len(records) != len(lines) or any(len(row) != 11 for row in records):
            raise ValueError("Provider history schema changed")
        frame = pd.DataFrame(records, columns=["date", "open", "close", "high", "low", "volume",
                                               "amount", "amplitude", "change_pct", "change", "turnover"])
        frame["code"] = code
        return frame

    def quote(self, code, permission_check=None):
        self.pace()
        if permission_check:
            permission_check()
        response = requests.get("https://push2.eastmoney.com/api/qt/stock/get", params={
            "secid": ("1." if code.startswith("6") else "0.") + code, "fltt": "2", "invt": "2",
            "fields": "f43,f44,f45,f46,f47,f48,f51,f52,f57,f58,f60,f86"}, timeout=(5, 12))
        response.raise_for_status()
        body = response.json()
        if body.get("rc") != 0 or not body.get("data"):
            raise ValueError("Provider returned no quote")
        return body["data"]


def normalize_history(raw, code, start, end):
    data = raw.rename(columns=RENAME).copy()
    if "code" not in data:
        data["code"] = code
    columns = ["date", "code", "open", "close", "high", "low", "volume", "amount", "turnover"]
    if not set(columns) <= set(data) or data.empty:
        raise ValueError("History is empty or missing required fields")
    data = data[columns]
    data["date"] = pd.to_datetime(data["date"])
    if data.date.isna().any() or not data.code.astype(str).eq(code).all() or data.duplicated(["code", "date"]).any():
        raise ValueError("History identity/date validation failed")
    for column in columns[2:]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if (not np.isfinite(data[columns[2:]].to_numpy(dtype=float)).all()
            or (data[["open", "close", "high", "low"]] <= 0).any().any()
            or (data[["volume", "amount", "turnover"]] < 0).any().any()
            or (data.low > data[["open", "close"]].min(axis=1)).any()
            or (data.high < data[["open", "close"]].max(axis=1)).any()):
        raise ValueError("Invalid OHLC, volume, amount or turnover")
    data = data[data.date.between(pd.Timestamp(start), pd.Timestamp(end))].sort_values("date")
    if data.empty:
        raise ValueError("No history in requested range")
    data["volume"] *= 100  # Eastmoney history is in lots; published files are in shares.
    return data.reset_index(drop=True)


def normalize_index_history(raw, code, start, end):
    data = raw.rename(columns=RENAME).copy()
    columns = ["date", "open", "close", "high", "low", "volume"]
    if not set(columns) <= set(data) or data.empty:
        raise ValueError("Index history is empty or missing required fields")
    data = data[columns]
    data["code"] = code
    data = data[["date", "code", "open", "close", "high", "low", "volume"]]
    data["date"] = pd.to_datetime(data["date"])
    if data.date.isna().any() or data.duplicated(["code", "date"]).any():
        raise ValueError("Index history identity/date validation failed")
    for column in columns[1:]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if (not np.isfinite(data[columns[1:]].to_numpy(dtype=float)).all()
            or (data[["open", "close", "high", "low"]] <= 0).any().any()
            or (data["volume"] < 0).any()
            or (data.low > data[["open", "close"]].min(axis=1)).any()
            or (data.high < data[["open", "close"]].max(axis=1)).any()):
        raise ValueError("Invalid index OHLC or volume")
    data = data[data.date.between(pd.Timestamp(start), pd.Timestamp(end))].sort_values("date")
    if data.empty:
        raise ValueError("No index history in requested range")
    return data.reset_index(drop=True)


def _number(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if np.isfinite(number) else None


def _limit_rate(code, name):
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if code.startswith(("4", "8", "920")):
        return 0.30
    return 0.05 if "ST" in str(name).upper() else 0.10


def _limit_price(previous_close, rate):
    if previous_close is None or previous_close <= 0:
        return None
    value = Decimal(str(previous_close)) * (Decimal(1) + Decimal(str(rate)))
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def normalize_breadth_row(raw):
    code = str(raw.get("code", ""))
    if not re.fullmatch(r"[0-9]{6}", code):
        raise ValueError("Market breadth contains an invalid stock code")
    name = str(raw.get("name", ""))
    trade = _number(raw.get("trade"))
    previous = _number(raw.get("settlement"))
    change = _number(raw.get("changepercent"))
    volume = _number(raw.get("volume"))
    amount = _number(raw.get("amount"))
    if (volume is not None and volume < 0) or (amount is not None and amount < 0):
        raise ValueError("Market breadth contains negative volume or amount")
    suspended = trade is None or trade <= 0 or change is None
    rate = _limit_rate(code, name)
    limit_up_price = _limit_price(previous, rate)
    limit_down_price = _limit_price(previous, -rate)
    limit_up = (not suspended and limit_up_price is not None and trade >= limit_up_price - 0.005)
    limit_down = (not suspended and limit_down_price is not None and trade <= limit_down_price + 0.005)
    return dict(code=code, symbol=str(raw.get("symbol", "")), name=name,
                last=trade, previous_close=previous, change_pct=change,
                volume=volume, amount=amount, tick_time=str(raw.get("ticktime", "")),
                limit_rate=rate, limit_up_price=limit_up_price, limit_down_price=limit_down_price,
                suspended=suspended, limit_up_approx=limit_up, limit_down_approx=limit_down,
                price_basis="unadjusted", volume_unit="shares", amount_unit="CNY")


def summarize_breadth(rows, *, captured_at, reference_index_date=None):
    valid = [row for row in rows if not row["suspended"] and row["change_pct"] is not None]
    advancers = sum(row["change_pct"] > 0 for row in valid)
    decliners = sum(row["change_pct"] < 0 for row in valid)
    unchanged = sum(row["change_pct"] == 0 for row in valid)
    changes = np.array([row["change_pct"] for row in valid], dtype=float)
    amounts = np.array([row["amount"] or 0.0 for row in rows], dtype=float)
    volumes = np.array([row["volume"] or 0.0 for row in rows], dtype=float)
    directions = np.array([row["change_pct"] if not row["suspended"] and row["change_pct"] is not None else 0.0
                           for row in rows], dtype=float)
    tick_times = [row["tick_time"] for row in rows if row["tick_time"]]
    total = len(rows)
    return dict(status="ready", captured_at=captured_at, reference_index_date=reference_index_date,
                provider="sina", node=MARKET_NODE, total=total, valid=len(valid),
                suspended=total - len(valid), advancers=advancers, decliners=decliners,
                unchanged=unchanged, advance_decline_ratio=(advancers / decliners if decliners else None),
                advance_decline_spread=((advancers - decliners) / len(valid) if valid else None),
                median_change_pct=(float(np.median(changes)) if len(changes) else None),
                mean_change_pct=(float(np.mean(changes)) if len(changes) else None),
                total_volume=float(np.sum(volumes)), total_amount=float(np.sum(amounts)),
                up_amount=float(np.sum(amounts[directions > 0])),
                down_amount=float(np.sum(amounts[directions < 0])),
                limit_up_approx=int(sum(row["limit_up_approx"] for row in rows)),
                limit_down_approx=int(sum(row["limit_down_approx"] for row in rows)),
                tick_time_max=(max(tick_times) if tick_times else None),
                timestamp_semantics="Sina Market_Center exposes only ticktime, not a trading date; "
                                    "reference_index_date comes from the captured index daily bars")


def retry(call, attempts=5, *, limiter=None, deadline=None, context=None):
    for attempt in range(attempts):
        events.check_cancel()
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Download retry/time budget exhausted")
        try:
            result = call()
        except events.Cancelled:
            raise
        except Exception as error:
            if not transient(error):
                raise
            delay = max(min(60.0, 2.0 * 2 ** attempt) + random.uniform(0, 1), retry_delay(error))
            interval = limiter.failure(delay) if limiter is not None else None
            if attempt + 1 == attempts:
                raise
            events.emit("download_retry", attempt=attempt + 1, max_attempts=attempts,
                        wait_seconds=max(delay, interval or 0), interval_seconds=interval,
                        error=str(error)[:500], **(context or {}))
            if limiter is None:
                wait(delay, deadline)
            # Provider.pace() enforces shared cooldown immediately before retry.
        else:
            if limiter is not None:
                limiter.success()
            return result


def history_time_budget(codes, bases):
    return min(3600, max(600, len(codes) * len(bases) * 6))


def download_history(codes, start, end, bases, *, allow_network=False, root=ROOT, provider=None, resume_id=None, source="sina"):
    if not allow_network:
        raise ValueError("Explicit network consent required")
    valid_codes(codes)
    if not bases or len(set(bases)) != len(bases) or not set(bases) <= {"hfq", "unadjusted"}:
        raise ValueError("Explicit, unique price bases required")
    if pd.Timestamp(start) > pd.Timestamp(end) or pd.Timestamp(end).date() > datetime.now(CHINA).date():
        raise ValueError("Invalid or future download range")
    if source not in {"eastmoney", "sina"}:
        raise ValueError("Unknown daily data source")
    if provider is None:
        if source == "sina":
            from src.data.sina import Sina
            provider = Sina(root)
        else:
            provider = Eastmoney(root)
    if getattr(provider, "name", source) != source:
        raise ValueError("Provider differs from declared source")
    store = Snapshots(root)
    request = dict(provider=source, codes=codes, start_date=start, end_date=end, bases=bases)
    old, old_dir = (store.get("history", resume_id) if resume_id else ({}, None))
    if old and old["request"] != request:
        raise ValueError("Retry must use the original request, including stock order and price bases")
    identifier, directory = store.begin("history")
    previous = {(r["code"], r["basis"]): r for r in old.get("entries", [])}
    entries, errors = [], []
    failures = 0
    started = time.monotonic()
    cancelled = False
    total = len(codes) * len(bases)
    budget = history_time_budget(codes, bases)
    deadline = started + budget
    if isinstance(provider, Eastmoney):
        provider.deadline = deadline
    def progress(code=None, basis=None):
        events.emit("download_progress", code=code, basis=basis, completed=len(entries), total=total,
                    saved=sum(r["status"] == "ready" for r in entries), failed=len(errors),
                    reused=sum(bool(r.get("reused_from")) for r in entries))
    progress()
    for code in codes:
        for basis in bases:
            if not cancelled:
                try:
                    events.check_cancel()
                except events.Cancelled:
                    cancelled = True
            target = directory / basis
            target.mkdir(exist_ok=True)
            prior = previous.get((code, basis))
            if prior and prior["status"] == "ready":
                for suffix in ["parquet", "csv"]:
                    shutil.copyfile(old_dir / basis / f"{code}.{suffix}", target / f"{code}.{suffix}")
                entries.append(dict(prior, reused_from=resume_id))
                progress(code, basis)
                continue
            if cancelled or failures >= 3 or time.monotonic() - started > budget:
                reason = "Cancelled before capture" if cancelled else "Batch circuit/time limit reached"
                item = dict(code=code, basis=basis, status="failed", error=reason)
                entries.append(item); errors.append(item)
                progress(code, basis)
                continue
            try:
                frame = normalize_history(retry(lambda: provider.history(code, start, end, basis),
                    limiter=getattr(provider, "limiter", None), deadline=deadline,
                    context=dict(code=code, basis=basis)), code, start, end)
                frame.to_parquet(target / f"{code}.parquet", index=False)
                frame.to_csv(target / f"{code}.csv", index=False, encoding="utf-8-sig")
                item = dict(code=code, basis=basis, status="ready", rows=len(frame),
                            first_date=str(frame.date.iloc[0].date()), last_date=str(frame.date.iloc[-1].date()),
                            received_at=utc_now())
                failures = 0
            except events.Cancelled:
                cancelled = True
                item = dict(code=code, basis=basis, status="failed", error="Cancelled during capture")
                errors.append(item)
            except Exception as error:
                failures += 1
                item = dict(code=code, basis=basis, status="failed", error=str(error)[:500])
                errors.append(item)
            entries.append(item)
            progress(code, basis)
            with (directory / "requests.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    result = store.publish("history", identifier, directory, dict(request=request, entries=entries,
                         status="ready" if not errors else "partial", errors=errors, volume_unit="shares",
                         cancelled=cancelled, time_budget_seconds=budget,
                         retry_policy=dict(max_attempts=5, initial_interval_seconds=1, max_interval_seconds=30,
                                           exponential_backoff=True, respects_retry_after=True),
                         complete_bar_policy="Default plans exclude capture-day bars; explicit close-date mode requires post-16:00 capture; exchange calendar not verified"))
    events.emit("download_snapshot", snapshot_id=identifier, snapshot_kind="history", status=result["status"])
    if cancelled:
        raise events.Cancelled("Download stopped; completed files are saved in snapshot " + identifier)
    return result


def _quote_number(value, positive=True):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if np.isfinite(number) and (number > 0 if positive else number >= 0) else None


def _normalize_levels(raw_levels):
    levels = []
    for level in raw_levels or []:
        price = _quote_number(level.get("price"))
        volume = _quote_number(level.get("volume"), positive=False)
        if price is not None and volume is not None:
            levels.append(dict(price=price, volume=volume))
    return levels[:5]


def _depth_metrics(raw_bids, raw_asks):
    bids = _normalize_levels(raw_bids)
    asks = _normalize_levels(raw_asks)
    bid_size = sum(level["volume"] for level in bids)
    ask_size = sum(level["volume"] for level in asks)
    bid = bids[0]["price"] if bids else None
    ask = asks[0]["price"] if asks else None
    mid = (bid + ask) / 2 if bid is not None and ask is not None else None
    total = bid_size + ask_size
    return dict(bids=bids, asks=asks,
                spread=(ask - bid) if bid is not None and ask is not None else None,
                spread_bps=((ask - bid) / mid * 10000) if mid else None,
                depth_imbalance=((bid_size - ask_size) / total) if total else None)


def normalize_sina_quote(raw, code, received_at):
    """Normalize one Sina hq_str record; volume is shares and price is unadjusted."""
    if str(raw.get("code")) != code:
        raise ValueError("Quote stock code mismatch")
    last = _quote_number(raw.get("last"))
    if last is None:
        raise ValueError("Missing or invalid latest price")
    timestamp = None
    try:
        candidate = datetime.strptime(f"{raw['date']} {raw['time']}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=CHINA)
        age = (datetime.fromisoformat(received_at) - candidate.astimezone(timezone.utc)).total_seconds()
        if candidate.year >= 2000 and age >= -5:
            timestamp = candidate.astimezone(timezone.utc).isoformat()
    except (KeyError, ValueError, TypeError, OverflowError):
        pass
    volume = _quote_number(raw.get("volume"), positive=False)
    amount = _quote_number(raw.get("amount"), positive=False)
    previous = _quote_number(raw.get("previous_close"))
    opening = _quote_number(raw.get("open"))
    high = _quote_number(raw.get("high"))
    low = _quote_number(raw.get("low"))
    depth = _depth_metrics(raw.get("bids"), raw.get("asks"))
    return dict(code=code, name=str(raw.get("name", "")), last=last, previous_close=previous,
                open=opening, high=high, low=low, volume=volume, amount=amount,
                bid=depth["bids"][0]["price"] if depth["bids"] else None,
                ask=depth["asks"][0]["price"] if depth["asks"] else None,
                vwap=(amount / volume if amount and volume else None),
                change=(last - previous if previous else None),
                change_pct=(last / previous - 1 if previous else None),
                gap_pct=(opening / previous - 1 if previous and opening else None),
                range_pct=((high - low) / previous if previous and high is not None and low is not None else None),
                **depth,
                limit_up=None, limit_down=None, provider_timestamp=timestamp, received_at=received_at,
                time_field="date/time", timestamp_semantics="Sina quote date/time Asia/Shanghai; vendor time is not exchange-certified",
                price_basis="unadjusted")


def normalize_quote(raw, code, received_at, source="eastmoney"):
    if source == "sina":
        return normalize_sina_quote(raw, code, received_at)
    if source != "eastmoney":
        raise ValueError("Unsupported quote source")
    if str(raw.get("f57")) != code:
        raise ValueError("Quote stock code mismatch")

    def numeric(key, positive=True):
        return _quote_number(raw.get(key), positive)

    last = numeric("f43")
    if last is None:
        raise ValueError("Missing or invalid latest price")
    timestamp = None
    epoch = numeric("f86")
    if epoch:
        try:
            candidate = datetime.fromtimestamp(epoch, timezone.utc)
            age = (datetime.fromisoformat(received_at) - candidate).total_seconds()
            if candidate.year >= 2000 and age >= -5:
                timestamp = candidate.isoformat()
        except (ValueError, OverflowError, OSError):
            pass
    previous = numeric("f60")
    volume = numeric("f47", positive=False)
    amount = numeric("f48", positive=False)
    return dict(code=code, name=str(raw.get("f58", "")), last=last, previous_close=previous,
                open=numeric("f46"), high=numeric("f44"), low=numeric("f45"),
                volume=(volume * 100 if volume is not None else None), amount=amount,
                vwap=(amount / (volume * 100) if amount and volume else None),
                change=(last - previous if previous else None),
                change_pct=(last / previous - 1 if previous else None),
                bids=[], asks=[], spread=None, spread_bps=None, depth_imbalance=None,
                limit_up=numeric("f51"), limit_down=numeric("f52"),
                provider_timestamp=timestamp, received_at=received_at, time_field="f86",
                timestamp_semantics="unverified_vendor_epoch_field_f86", price_basis="unadjusted")


def capture_quotes(codes, *, allow_network=False, root=ROOT, provider=None, permission_check=None, source="eastmoney"):
    if not allow_network:
        raise ValueError("Explicit network consent required")
    if source not in {"sina", "eastmoney"}:
        raise ValueError("Unsupported quote source")
    valid_codes(codes)
    if len(codes) > 50:
        raise ValueError("Quote snapshots are limited to 50 stocks")
    store = Snapshots(root)
    identifier, directory = store.begin("quotes")
    if provider is None:
        if source == "sina":
            from src.data.sina import Sina
            provider = Sina(root)
        else:
            provider = Eastmoney(root)
    if getattr(provider, "name", source) != source:
        raise ValueError("Quote provider differs from declared source")
    rows, errors, raw_results = [], [], {}
    failures = 0
    started = time.monotonic()
    if isinstance(provider, Eastmoney):
        provider.deadline = started + 90
    for code in codes:
        events.check_cancel()
        if failures >= 3 or time.monotonic() - started > 90:
            errors.append(dict(code=code, error="Quote circuit/time limit reached"))
            continue
        try:
            def fetch():
                if permission_check:
                    permission_check()
                return (provider.quote(code, permission_check=permission_check)
                        if isinstance(provider, Eastmoney) else provider.quote(code))
            raw = retry(fetch, limiter=getattr(provider, "limiter", None), deadline=started + 90,
                        context=dict(code=code, basis="quote"))
            received = utc_now()
            raw_results[code] = dict(received_at=received, data=raw)
            rows.append(normalize_quote(raw, code, received, source=source))
            failures = 0
        except events.Cancelled:
            raise
        except Exception as error:
            failures += 1
            errors.append(dict(code=code, error=str(error)[:500]))
        events.emit("quote_progress", completed=len(rows) + len(errors), total=len(codes))
    if permission_check:
        permission_check()
    write_json(directory / "quotes.json", rows)
    write_json(directory / "provider_response.json", raw_results)
    return store.publish("quotes", identifier, directory, dict(provider=source, codes=codes,
                         status="ready" if not errors else "partial", errors=errors,
                         timestamp_source=("Sina date/time in Asia/Shanghai; vendor time is not exchange-certified"
                                           if source == "sina" else
                                           "f86 parsed as Unix seconds; vendor semantics and exchange trade time are not certified"),
                         price_basis="unadjusted", rows=rows))


def market_time_budget(pages):
    return min(900, max(120, pages * 6 + 60))


def capture_market(indices=None, start="2017-01-01", end=None, *, allow_network=False, root=ROOT,
                   provider=None, source="sina"):
    if not allow_network:
        raise ValueError("Explicit network consent required")
    indices = list(MARKET_INDICES) if indices is None else list(indices)
    if (not indices or len(set(indices)) != len(indices)
            or any(not re.fullmatch(r"[0-9]{6}", code) for code in indices)):
        raise ValueError("Expected unique six-digit index codes")
    end = str(datetime.now(CHINA).date()) if end is None else end
    if pd.Timestamp(start) > pd.Timestamp(end) or pd.Timestamp(end).date() > datetime.now(CHINA).date():
        raise ValueError("Invalid or future market range")
    if source != "sina":
        raise ValueError("Market context currently supports Sina only")
    if provider is None:
        from src.data.sina import Sina
        provider = Sina(root)
    if getattr(provider, "name", source) != source:
        raise ValueError("Market provider differs from declared source")

    store = Snapshots(root)
    identifier, directory = store.begin("market")
    deadline = time.monotonic() + 900
    if hasattr(provider, "deadline"):
        provider.deadline = deadline
    entries, errors = [], []
    cancelled = False
    for code in indices:
        try:
            events.check_cancel()
            frame = normalize_index_history(
                retry(lambda: provider.index_history(code, start, end),
                      limiter=getattr(provider, "limiter", None), deadline=deadline,
                      context=dict(index=code)), code, start, end)
            (directory / "indices").mkdir(exist_ok=True)
            frame.to_parquet(directory / "indices" / f"{code}.parquet", index=False)
            frame.to_csv(directory / "indices" / f"{code}.csv", index=False, encoding="utf-8-sig")
            entries.append(dict(code=code, name=MARKET_INDICES.get(code, code), status="ready",
                                rows=len(frame), first_date=str(frame.date.iloc[0].date()),
                                last_date=str(frame.date.iloc[-1].date()), received_at=utc_now(),
                                volume_unit="shares"))
            events.emit("market_progress", stage="index", code=code, completed=len(entries), total=len(indices))
        except events.Cancelled:
            cancelled = True
            item = dict(code=code, status="failed", error="Cancelled during capture")
            entries.append(item); errors.append(item)
        except Exception as error:
            item = dict(code=code, status="failed", error=str(error)[:500])
            entries.append(item); errors.append(item)

    breadth = dict(status="failed", error="Not captured")
    total = pages = 0
    try:
        if cancelled:
            raise events.Cancelled("Cancelled before breadth capture")
        events.check_cancel()
        total = int(retry(lambda: provider.market_count(),
                          limiter=getattr(provider, "limiter", None), deadline=deadline,
                          context=dict(market="count")))
        if total <= 0:
            raise ValueError("Provider returned an invalid market stock count")
        pages = math.ceil(total / MARKET_PAGE_SIZE)
        if pages > 100:
            raise ValueError("Market breadth requires too many pages")
        deadline = time.monotonic() + market_time_budget(pages)
        raw_rows = []
        for page in range(1, pages + 1):
            events.check_cancel()
            rows = retry(lambda page=page: provider.market_page(page),
                         limiter=getattr(provider, "limiter", None), deadline=deadline,
                         context=dict(market="breadth", page=page))
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"Market breadth page {page} is empty or malformed")
            if page < pages and len(rows) != MARKET_PAGE_SIZE:
                raise ValueError(f"Market breadth page {page} is short")
            raw_rows.extend(rows)
            events.emit("market_progress", stage="breadth", page=page, total_pages=pages,
                        completed=len(raw_rows), total=total)
        unique = {}
        duplicates = 0
        for raw in raw_rows:
            row = normalize_breadth_row(raw)
            duplicates += int(row["code"] in unique)
            unique[row["code"]] = row
        if len(unique) != total:
            raise ValueError(f"Market breadth coverage mismatch: {len(unique)} of {total}")
        rows = sorted(unique.values(), key=lambda row: row["code"])
        reference_date = max((entry["last_date"] for entry in entries if entry["status"] == "ready"),
                             default=None)
        captured_at = utc_now()
        summary = summarize_breadth(rows, captured_at=captured_at, reference_index_date=reference_date)
        summary.update(pages=pages, expected_total=total, duplicates=duplicates)
        frame = pd.DataFrame(rows)
        frame.to_parquet(directory / "breadth.parquet", index=False)
        frame.to_csv(directory / "breadth.csv", index=False, encoding="utf-8-sig")
        write_json(directory / "breadth_summary.json", summary)
        breadth = dict(status="ready", summary=summary)
    except events.Cancelled:
        cancelled = True
        breadth = dict(status="failed", error="Cancelled during capture", pages=pages, expected_total=total)
        errors.append(dict(kind="breadth", status="failed", error=breadth["error"]))
        write_json(directory / "breadth_error.json", breadth)
    except Exception as error:
        breadth = dict(status="failed", error=str(error)[:500], pages=pages, expected_total=total)
        errors.append(dict(kind="breadth", status="failed", error=breadth["error"]))
        write_json(directory / "breadth_error.json", breadth)

    result = store.publish("market", identifier, directory, dict(
        request=dict(provider=source, indices=indices, start_date=start, end_date=end,
                     node=MARKET_NODE, page_size=MARKET_PAGE_SIZE),
        entries=entries, breadth=breadth, status="ready" if not errors else "partial", errors=errors,
        as_of_date=max((entry["last_date"] for entry in entries if entry["status"] == "ready"), default=None),
        as_of_date_source="max ready index last_date; Sina breadth exposes ticktime only",
        volume_unit="shares", amount_unit="CNY",
        limit_policy="limit_up_approx/limit_down_approx use board and ST price-limit heuristics; "
                     "not exchange-certified status",
        cancelled=cancelled,
        complete_bar_policy="Index bars use the requested range; breadth is a point-in-time snapshot "
                            "whose trading date is not provided by the vendor"))
    events.emit("download_snapshot", snapshot_id=identifier, snapshot_kind="market", status=result["status"])
    if cancelled:
        raise events.Cancelled("Market capture stopped; completed files are saved in snapshot " + identifier)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["history", "quotes", "market"], required=True)
    parser.add_argument("--codes", nargs="*", default=[])
    parser.add_argument("--codes-file", help="UTF-8 text/CSV containing six-digit stock codes")
    parser.add_argument("--indices", nargs="*", default=[], help="Six-digit index codes for --kind market")
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default=str(datetime.now(CHINA).date()))
    parser.add_argument("--allow-network", action="store_true", required=True)
    parser.add_argument("--source", choices=["sina", "eastmoney"], default="sina")
    args = parser.parse_args()
    codes = list(dict.fromkeys(args.codes))
    if args.codes_file:
        path = Path(args.codes_file)
        if path.stat().st_size > 200000:
            parser.error("Stock code file exceeds 200 KB")
        codes = list(dict.fromkeys(codes + re.findall(r"(?<!\d)\d{6}(?!\d)", path.read_text(encoding="utf-8-sig"))))
    if args.kind != "market" and not codes:
        parser.error("Provide --codes or --codes-file")
    if args.kind != "market":
        valid_codes(codes)
    result = (download_history(codes, args.start, args.end, ["hfq", "unadjusted"], allow_network=args.allow_network, source=args.source)
              if args.kind == "history" else
              capture_quotes(codes, allow_network=args.allow_network, source=args.source)
              if args.kind == "quotes" else
              capture_market(args.indices or None, args.start, args.end,
                             allow_network=args.allow_network, source=args.source))
    print(json.dumps(result, ensure_ascii=True, indent=2))
