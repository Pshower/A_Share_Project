"""Explicit daily-history and realtime-quote access with auditable snapshots."""

import argparse
from datetime import datetime, timezone
import json
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
    return dict(code=code, name=str(raw.get("name", "")), last=last, previous_close=_quote_number(raw.get("previous_close")),
                open=_quote_number(raw.get("open")), high=_quote_number(raw.get("high")), low=_quote_number(raw.get("low")),
                volume=_quote_number(raw.get("volume"), positive=False), amount=_quote_number(raw.get("amount"), positive=False),
                bid=_quote_number(raw.get("bid")), ask=_quote_number(raw.get("ask")),
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
    return dict(code=code, name=str(raw.get("f58", "")), last=last, previous_close=numeric("f60"),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["history", "quotes"], required=True)
    parser.add_argument("--codes", nargs="+", required=True)
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default=str(datetime.now(CHINA).date()))
    parser.add_argument("--allow-network", action="store_true", required=True)
    parser.add_argument("--source", choices=["sina", "eastmoney"], default="sina")
    args = parser.parse_args()
    result = (download_history(args.codes, args.start, args.end, ["hfq", "unadjusted"], allow_network=args.allow_network, source=args.source)
              if args.kind == "history" else
              capture_quotes(args.codes, allow_network=args.allow_network, source=args.source))
    print(json.dumps(result, ensure_ascii=True, indent=2))
