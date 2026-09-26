"""Read-only paired-price inventory; never infer corporate actions from HFQ ratios."""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .preprocess import RENAME, ROOT, file_hash, write_json


def read_source(path, code):
    frame = pd.read_csv(path, dtype={"code": str, "股票代码": str}).rename(columns=RENAME)
    required = ["date", "code", "open", "high", "low", "close", "volume", "amount"]
    if not set(required) <= set(frame):
        raise ValueError("Missing OHLCV identity columns")
    frame = frame[required].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame.empty or frame.date.isna().any() or frame.date.duplicated().any() or not frame.code.eq(code).all():
        raise ValueError("Empty, duplicated, or conflicting source identity")
    for key in required[2:]:
        frame[key] = pd.to_numeric(frame[key], errors="raise")
    prices = frame[["open", "high", "low", "close"]]
    if (not np.isfinite(frame[required[2:]].to_numpy()).all() or (prices <= 0).any().any()
            or (frame[["volume", "amount"]] < 0).any().any()
            or (frame.low > prices.min(axis=1)).any() or (frame.high < prices.max(axis=1)).any()):
        raise ValueError("Invalid OHLCV values")
    return frame.sort_values("date").set_index("date")


def audit(raw_dir, codes=None):
    sources = {}
    pattern = re.compile(r"^(\d{6})_data_\d{4}_\d{4}(_hfq)?\.csv$")
    for path in sorted(Path(raw_dir).glob("*.csv")):
        match = pattern.fullmatch(path.name)
        if match:
            basis = "hfq" if match[2] else "legacy_unadjusted_candidate"
            sources.setdefault(match[1], {}).setdefault(basis, []).append(path)
    selected = sorted(sources) if codes is None else list(codes)
    if not selected or len(set(selected)) != len(selected) or any(not re.fullmatch(r"\d{6}", c) for c in selected):
        raise ValueError("Expected unique six-digit stock codes")
    rows = []
    for code in selected:
        row = dict(code=code, status="blocked", files={}, reasons=[])
        frames = {}
        for basis in ["hfq", "legacy_unadjusted_candidate"]:
            paths = sources.get(code, {}).get(basis, [])
            if len(paths) != 1:
                row["reasons"].append(f"{basis}: expected one source, found {len(paths)}")
                continue
            path = paths[0]
            row["files"][basis] = dict(path=str(path.resolve()), sha256=file_hash(path))
            try:
                frame = read_source(path, code)
                frames[basis] = frame
                row["files"][basis].update(rows=len(frame), start=str(frame.index[0].date()), end=str(frame.index[-1].date()))
            except (ValueError, KeyError, OSError) as exc:
                row["reasons"].append(f"{basis}: {exc}")
        if len(frames) == 2:
            hfq, raw = frames["hfq"], frames["legacy_unadjusted_candidate"]
            shared = hfq.index.intersection(raw.index)
            row["common_rows"] = len(shared)
            row["dates_equal"] = hfq.index.equals(raw.index)
            if len(shared):
                ratio = hfq.loc[shared, "close"] / raw.loc[shared, "close"]
                row.update(close_ratio_min=float(ratio.min()), close_ratio_max=float(ratio.max()),
                           volume_equal=bool(np.allclose(hfq.loc[shared, "volume"], raw.loc[shared, "volume"], rtol=0, atol=0)))
            if not row["dates_equal"]:
                row["reasons"].append("Price-basis date sets differ; no implicit filling or row dropping")
            elif not row["volume_equal"]:
                row["reasons"].append("Price-basis volume differs; unit or revision review required")
            else:
                row["status"] = "paired_candidate"
        rows.append(row)
    return dict(schema_version=1, stocks=len(rows), paired_candidates=sum(r["status"] == "paired_candidate" for r in rows),
                real_execution_ready=False, rows=rows,
                blockers=["Legacy unsuffixed files have no verified price-basis provenance manifest.",
                          "Corporate-action completeness, publication timing, payment/share-availability dates are not certified.",
                          "Suspension/ST/price-limit history and exchange calendar are not certified."],
                note="Date/volume agreement and price ratios do not prove basis or identify dividends versus bonus shares. No training data changed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.raw_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "audit.json", report)
    print({k: report[k] for k in ["stocks", "paired_candidates", "real_execution_ready"]})


if __name__ == "__main__":
    main()
