"""Compose full per-stock HFQ captures; never silently resolve revisions."""

import shutil
import pandas as pd

from src.data.preprocess import ROOT
from src.data.snapshots import Snapshots
from src.data.online import valid_codes
from src.runtime import events


def inspect_bundle(history_ids, codes, start, end, root=ROOT):
    valid_codes(codes)
    store = Snapshots(root)
    sources = [(identifier, *store.get("history", identifier)) for identifier in history_ids]
    bindings, missing, conflicts = {}, [], []
    for code in codes:
        candidates = []
        for identifier, metadata, directory in sources:
            if metadata.get("volume_unit") != "shares":
                continue
            if metadata["request"]["start_date"] > start or metadata["request"]["end_date"] < end:
                continue
            entry = next((e for e in metadata["entries"] if e["code"] == code and e["basis"] == "hfq" and e["status"] == "ready"), None)
            if entry and all(f"hfq/{code}.{suffix}" in metadata["files"] for suffix in ["csv", "parquet"]):
                candidates.append((identifier, directory, entry))
        if not candidates:
            missing.append(code)
            continue
        first = pd.read_parquet(candidates[0][1] / "hfq" / f"{code}.parquet").sort_values("date").reset_index(drop=True)
        different = False
        for _, directory, _ in candidates[1:]:
            other = pd.read_parquet(directory / "hfq" / f"{code}.parquet").sort_values("date").reset_index(drop=True)
            try:
                pd.testing.assert_frame_equal(first, other, check_dtype=False, check_exact=True)
            except AssertionError:
                different = True
        if different:
            conflicts.append(dict(code=code, history_ids=[c[0] for c in candidates]))
        else:
            bindings[code] = dict(history_id=candidates[0][0], entry=candidates[0][2])
    return dict(ready=not missing and not conflicts, missing=missing, conflicts=conflicts, bindings=bindings)


def compose_bundle(history_ids, codes, start, end, root=ROOT):
    report = inspect_bundle(history_ids, codes, start, end, root)
    if not report["ready"]:
        raise ValueError("Source bundle has missing stocks or conflicting captures")
    store = Snapshots(root)
    identifier, directory = store.begin("history")
    (directory / "hfq").mkdir()
    entries = []
    for code, binding in report["bindings"].items():
        events.check_cancel()
        _, original = store.get("history", binding["history_id"])
        for suffix in ["csv", "parquet"]:
            shutil.copyfile(original / "hfq" / f"{code}.{suffix}", directory / "hfq" / f"{code}.{suffix}")
        entries.append(dict(binding["entry"], reused_from=binding["history_id"]))
    return store.publish("history", identifier, directory,
                         dict(request=dict(provider="local_composition", codes=codes, start_date=start, end_date=end, bases=["hfq"]),
                              entries=entries, status="ready", errors=[], volume_unit="shares", source_history_ids=history_ids,
                              complete_bar_policy="Original per-file capture times preserved; no new market request"))
