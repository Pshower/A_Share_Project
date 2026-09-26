"""New daily research bindings and quote-aware checks, never real order execution."""

from datetime import datetime, timezone
import hashlib
import inspect

import numpy as np
import pandas as pd

from src.data.online import CHINA
from src.data.preprocess import ROOT, DataPreprocessor, FEATURES, read_config, resolve_path, file_hash, write_json
from src.data.snapshots import Snapshots, utc_now


def feature_history_origin(research):
    audit_path = resolve_path(research["dataset"]["clean_dir"]) / "source_audit.csv"
    if not audit_path.exists():
        raise ValueError("Original source audit is required to establish the feature-history anchor")
    audit = pd.read_csv(audit_path, dtype={"code": str})
    included = audit[audit.status == "included"]
    if included.empty or included.source_start.isna().any():
        raise ValueError("Cannot establish the original feature-history anchor")
    # The preprocessor computes indicators before slicing the requested study dates.
    return pd.to_datetime(included.source_start).min(), file_hash(audit_path)


def build_daily_plan(agent, data, history_id, *, model_id, accept_revisions=False, research_weights=None, root=ROOT, now=None):
    if agent.contract != data.contract or data.contract["price_basis"] != "hfq_research":
        raise ValueError("This binding only supports the existing HFQ research contract")
    if data.manifest["feature_cols"] != FEATURES:
        raise ValueError("Feature definition mismatch")
    store = Snapshots(root)
    research_weights = research_weights or {}
    if (not set(research_weights) <= set(data.codes) or any(not np.isfinite(v) or v < 0 for v in research_weights.values())
            or sum(research_weights.values()) > 1):
        raise ValueError("Hypothetical weights must be finite, nonnegative and sum to at most one")
    history, directory = store.get("history", history_id)
    if history["status"] != "ready" or not set(data.codes) <= set(history["request"]["codes"]) or "hfq" not in history["request"]["bases"]:
        raise ValueError("Complete HFQ history for the entire model pool is required")
    origin, origin_audit = feature_history_origin(data.research)
    original_sources = pd.read_csv(resolve_path(data.research["dataset"]["clean_dir"]) / "source_audit.csv", dtype={"code": str}).set_index("code")
    if pd.Timestamp(history["request"]["start_date"]) > origin:
        raise ValueError("Download from the original feature-history anchor, not just a short warmup")
    now = now or datetime.now(timezone.utc)
    capture_entries = {r["code"]: r for r in history["entries"] if r["basis"] == "hfq" and r["status"] == "ready"}
    cutoffs = {}
    frames = {}
    for code in data.codes:
        # Retried batches retain each reused file's original capture day.
        captured_day = datetime.fromisoformat(capture_entries[code]["received_at"]).astimezone(CHINA).date()
        cutoff = min(now.astimezone(CHINA).date(), captured_day)
        cutoffs[code] = str(cutoff)
        frame = pd.read_parquet(directory / "hfq" / f"{code}.parquet").set_index("date")
        frame.index = pd.to_datetime(frame.index)
        if code not in original_sources.index or frame.index.min() > pd.Timestamp(original_sources.loc[code, "source_start"]):
            raise ValueError(f"Downloaded history is truncated before the original first bar for {code}")
        frame = frame[(frame.index >= origin) & (frame.index.date < cutoff)]
        if frame.empty:
            raise ValueError(f"No prior-day bars for {code}")
        frames[code] = frame
    as_of = min(frame.index[-1] for frame in frames.values())
    if any(as_of not in frame.index for frame in frames.values()):
        raise ValueError("Model stocks do not share a complete as-of date")
    calendar = pd.DatetimeIndex(sorted(set(data.features.index[data.features.index <= as_of]).union(
        *(set(frame.index[frame.index <= as_of]) for frame in frames.values()))))
    calendar = calendar[calendar >= origin]
    scaler_path = resolve_path(data.research["dataset"]["clean_dir"]) / "scaler_params.json"
    if file_hash(scaler_path) != agent.contract["artifacts"]["scaler_params.json"]:
        raise ValueError("Training scaler changed")
    scaler = read_config(scaler_path)
    normalized, revision = {}, {}
    for code, raw in frames.items():
        aligned = raw.reindex(calendar).copy()
        aligned["valid_price"] = aligned[["open", "close", "high", "low"]].notna().all(axis=1)
        feature = DataPreprocessor.add_features(aligned)
        scaled, _ = DataPreprocessor.normalize_features(feature, fit=False, stats=scaler["stats"])
        normalized[code] = scaled
        shared = scaled.index.intersection(data.features.index)
        old = data.features.loc[shared, pd.IndexSlice[code, FEATURES]].to_numpy(dtype=float)
        new = scaled.loc[shared, FEATURES].to_numpy(dtype=float)
        paired = np.isfinite(old) & np.isfinite(new)
        revision[code] = dict(compared_values=int(paired.sum()),
                              missing_pattern_changed=bool(np.any(np.isfinite(old) != np.isfinite(new))),
                              max_abs_scaled_difference=float(np.max(np.abs(new[paired] - old[paired]))) if paired.any() else None)
    changed = any(r["compared_values"] == 0 or r["missing_pattern_changed"] or r["max_abs_scaled_difference"] > 1e-4 for r in revision.values())
    if changed and not accept_revisions:
        raise ValueError("Historical feature revisions detected; review differences and explicitly accept research-only input")
    lookback = data.config["lookback"]
    if len(calendar) < lookback:
        raise ValueError("Insufficient history window")
    window = np.stack([normalized[c].iloc[-lookback:].to_numpy(dtype=np.float32) for c in data.codes])
    valid = np.isfinite(window).all(axis=(1, 2))
    if not valid.all():
        raise ValueError("Latest model window is incomplete; no automatic cash/exit action")
    obs = dict(stock_features=window, valid_mask=valid.astype(np.int8), buy_mask=valid.astype(np.int8),
               sell_mask=np.ones(len(data.codes), dtype=np.int8),
               portfolio=np.array([research_weights.get(c, 0.) for c in data.codes] + [1 - sum(research_weights.values())], dtype=np.float32),
               receivable_ratio=np.zeros(1, dtype=np.float32))
    mapped = agent.inspect_action(obs)
    rows = []
    for i, code in enumerate(data.codes):
        target, current = float(mapped["weights"][i]), research_weights.get(code, 0.)
        intent = "unchanged" if abs(target - current) < 1e-6 else "buy" if target > current else "sell"
        if current > 0 and target == 0:
            intent = "exit"
        rows.append(dict(code=code, target_weight=target, current_research_weight=current,
                         latent=float(mapped["latent"][i]), research_intent=intent))
    identifier, target = store.begin("plans")
    binding = dict(model_id=model_id, model_sha256=agent.provenance.get("loaded_model_sha256"),
                   training_contract=agent.contract, source_history_id=history_id,
                   source_manifest_sha256=file_hash(directory / "manifest.json"), feature_as_of=str(as_of.date()),
                   history_origin=str(origin.date()), capture_cutoffs=cutoffs, source_audit_sha256=origin_audit, scaler_sha256=file_hash(scaler_path),
                   feature_implementation_sha256=hashlib.sha256((inspect.getsource(DataPreprocessor.add_features)
                       + inspect.getsource(DataPreprocessor.normalize_features)).encode()).hexdigest(),
                   stock_codes=list(data.codes), revision=revision, revisions_accepted=bool(changed and accept_revisions),
                   compatibility_status="research_only", account_basis="hypothetical_research_weights", research_weights=research_weights,
                   real_account_compatible=False, exchange_calendar_verified=False,
                   complete_bar_policy="Exclude bars captured on or after the capture day; latest exchange session is not certified")
    pd.concat(normalized, axis=1).to_parquet(target / "features.parquet")
    write_json(target / "binding.json", binding)
    return store.publish("plans", identifier, target, dict(status="ready", **binding, rows=rows,
                         target_cash=mapped["cash_weight"], generated_at=utc_now(),
                         limitations=["Hypothetical research weights (all cash by default), not the user's real account.",
                                      "Daily PPO is not an intraday policy; no automatic order or share recommendation.",
                                      "Fresh quotes do not establish suspension, ST, limits or real-account compatibility."]))


def judge_quotes(snapshot, plan=None, *, now=None, ttl_seconds=120):
    now = now or datetime.now(timezone.utc)
    by_code = {row["code"]: row for row in snapshot["rows"]}
    codes = list(plan["stock_codes"]) if plan else snapshot["codes"]
    targets = {row["code"]: row for row in plan["rows"]} if plan else {}
    rows = []
    for code in codes:
        quote = by_code.get(code)
        reasons = []
        age = None
        if quote is None:
            reasons.append("missing_quote")
        elif not quote.get("provider_timestamp"):
            reasons.append("unknown_quote_time")
        else:
            reasons.append("unverified_vendor_time")
            age = (now - datetime.fromisoformat(quote["provider_timestamp"])).total_seconds()
            received_age = (now - datetime.fromisoformat(quote["received_at"])).total_seconds()
            if age < -5 or received_age < -5:
                reasons.append("future_timestamp")
            elif age > ttl_seconds or received_age > ttl_seconds:
                reasons.append("stale_quote")
        if snapshot["status"] != "ready" or set(codes) - set(by_code):
            reasons.append("incomplete_snapshot")
        if plan:
            plan_age = (now.astimezone(CHINA).date() - pd.Timestamp(plan["feature_as_of"]).date()).days
            if plan_age < 0 or plan_age > 4:
                reasons.append("stale_daily_plan")
            reasons += ["research_account_only", "calendar_not_verified", "trading_state_not_verified"]
        else:
            reasons.append("no_daily_plan")
        if quote and quote.get("limit_up") and quote["last"] >= quote["limit_up"]:
            reasons.append("at_vendor_limit_up")
        if quote and quote.get("limit_down") and quote["last"] <= quote["limit_down"]:
            reasons.append("at_vendor_limit_down")
        waiting = any(r in reasons for r in ["missing_quote", "unknown_quote_time", "future_timestamp", "stale_quote", "incomplete_snapshot", "stale_daily_plan"])
        rows.append(dict(code=code, quote=quote, quote_age_seconds=age,
                         target_weight=targets.get(code, {}).get("target_weight"),
                         research_intent=targets.get(code, {}).get("research_intent"),
                         status="waiting" if waiting else "research_review" if plan else "quotes_only",
                         reasons=list(dict.fromkeys(reasons)), actionable=False))
    return dict(snapshot_id=snapshot["id"], plan_id=plan["id"] if plan else None,
                feature_as_of=plan["feature_as_of"] if plan else None, evaluated_at=now.isoformat(),
                rows=rows, actionable=False, mode="quote_aware_daily_research",
                note="No real-account valuation or order execution; raw quotes are not multiplied by HFQ research shares.")
