"""A daily health record for the two things nobody was watching: the raw archive and the panel.

``data/online/history`` is the immutable record every panel is rebuilt from, and the forward panel is
what the shadow decisions read.  Both worked, so nobody asked whether a day was missing.  This tool
answers that question with numbers and writes one JSON per run; it changes nothing.

Read-only by construction: it opens the archive, reads manifests and one price file, and writes only
its own report.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

HISTORY = PROJECT_ROOT / "data" / "online" / "history"
CLEAN = PROJECT_ROOT / "data" / "clean"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "data_health"
JUMP = 0.30                # a one-day move beyond this is a data question until proven otherwise
COVERAGE_FLOOR = 0.90      # a session holding fewer than this share of the pool is worth naming


def snapshots(root=HISTORY):
    """Raw request snapshots per capture day, with the status mix the provider reported."""
    per_day = {}
    root = Path(root)
    if not root.exists():
        return per_day
    for directory in sorted(root.iterdir()):
        manifest = directory / "manifest.json"
        if not directory.is_dir() or not manifest.exists():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        day = str(payload.get("created_at", ""))[:10] or "unknown"
        row = per_day.setdefault(day, dict(snapshots=0, ok=0, partial=0, failed=0))
        row["snapshots"] += 1
        status = str(payload.get("status", "")).lower()
        row["failed" if status == "failed" else "partial" if status == "partial" else "ok"] += 1
    return per_day


def latest_panel(clean=CLEAN):
    candidates = sorted(Path(clean).glob("forward_pool_*"))
    return candidates[-1] if candidates else None


def panel_health(directory):
    """Coverage per session and the jumps that need explaining, from the panel's own prices.

    Coverage is measured against the codes that were already listed on that date, not against
    today's 297: the pool grows over time, and counting pre-listing years as "missing" would print
    675 phantom bad days every run.  What remains is a bar that is absent while the stock is
    listed -- a suspension, or a gap in the download.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    prices = pd.read_parquet(directory / "prices.parquet")
    level = "close" if "close" in set(prices.columns.get_level_values(1)) else \
        prices.columns.get_level_values(1)[0]
    close = prices.xs(level, axis=1, level=1)
    coverage = close.notna().sum(axis=1)
    first = close.apply(lambda column: column.first_valid_index())
    listed = pd.DataFrame({code: (close.index >= first[code]) for code in close.columns},
                          index=close.index)
    expected = listed.sum(axis=1)
    missing = (expected - coverage).astype(int)
    worst = missing.sort_values(ascending=False).head(10)
    returns = close.pct_change(fill_method=None)
    jumps = []
    for stamp, row in returns.iterrows():
        for code, value in row.items():
            if pd.notna(value) and abs(float(value)) > JUMP:
                jumps.append(dict(date=str(stamp.date()), code=str(code), move=round(float(value), 4)))
    return dict(version=manifest.get("version"), codes=len(manifest.get("stock_codes", [])),
                dates=int(len(close)), first=str(close.index[0].date()),
                last=str(close.index[-1].date()),
                coverage_median=int(coverage.median()), coverage_min=int(coverage.min()),
                coverage_min_date=str(coverage.idxmin().date()),
                missing_total=int(missing.sum()),
                missing_worst=[dict(date=str(stamp.date()), bars=int(count))
                               for stamp, count in worst.items() if count > 0],
                jumps=len(jumps), jump_examples=jumps[:10],
                files_mb=round(sum(item.stat().st_size for item in directory.glob("*.parquet"))
                               / 1e6, 1))


def report(history=HISTORY, clean=CLEAN):
    days = snapshots(history)
    panel = latest_panel(clean)
    payload = dict(generated_at=datetime.now().isoformat(timespec="seconds"),
                   history=dict(snapshots=sum(row["snapshots"] for row in days.values()),
                                days=len(days), first=min(days) if days else None,
                                last=max(days) if days else None,
                                failed=sum(row["failed"] for row in days.values()),
                                partial=sum(row["partial"] for row in days.values()),
                                by_day={key: days[key] for key in sorted(days)[-7:]}),
                   panel=panel_health(panel) if panel else None)
    return payload


def line(payload):
    history, panel = payload["history"], payload["panel"] or {}
    return ("data_health：原始快照 %d 个 / %d 天（failed %d，partial %d）；"
            "面板 %s：%d 只 × %d 天，覆盖中位 %d（最低 %d @ %s），"
            "上市后缺 bar 合计 %d，跳变>30%% %d"
            % (history["snapshots"], history["days"], history["failed"], history["partial"],
               panel.get("version", "—"), panel.get("codes", 0), panel.get("dates", 0),
               panel.get("coverage_median", 0), panel.get("coverage_min", 0),
               panel.get("coverage_min_date", "—"), panel.get("missing_total", 0),
               panel.get("jumps", 0)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only health record for the archive and panel.")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--history", default=str(HISTORY))
    parser.add_argument("--clean", default=str(CLEAN))
    args = parser.parse_args(argv)
    payload = report(Path(args.history), Path(args.clean))
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    path = target / ("%s.json" % date.today())
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(line(payload))
    print("written", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
