"""The treasury curve cache: the one macro file the nightly summary actually reads.

``daily_ops.regime()`` reads ``data/online/yields/china_treasury.csv`` every night to place the
ten-year yield in a carry bucket, but the file had no refresher -- the 2026-11-22 update was a
one-off script under ``reports/runs``.  This module is that refresher as a tool: fetch
``bond_zh_us_rate``, append only rows newer than the file's last date, write the whole file
atomically, and verify the round trip through ``asset_panel.curve`` -- the same reader the
daily job uses.

Two guards the one-off script did not have: the fetched frame's column names must match the
file's header exactly (a silent schema drift would shift every tenor one column), and the
write is a temp-file swap, so an interrupted refresh leaves yesterday's file intact.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ONLINE = PROJECT_ROOT / "data" / "online"
CSV = ONLINE / "yields" / "china_treasury.csv"


def _cell(value) -> str:
    """A number as %g, anything else as an empty cell -- the file's existing convention."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if pd.isna(number) else ("%g" % number)


def fetch_frame() -> pd.DataFrame:
    """The published China/US treasury curve from akshare."""
    import akshare as ak

    return ak.bond_zh_us_rate()


def refresh(online: Path | None = None, frame: pd.DataFrame | None = None) -> dict:
    """Append the curve's newer rows atomically; verify with the daily job's own reader.

    ``frame`` is injectable so tests never need the network.  Returns a summary dict with the
    before/after dates, the appended row count and the parsed tail -- or raises, leaving the
    file untouched, when the upstream schema no longer matches the file's header.
    """
    path = Path(online or ONLINE) / "yields" / "china_treasury.csv"
    original = path.read_text(encoding="utf-8-sig")
    lines = [line for line in original.splitlines() if line.strip()]
    if not lines:
        raise ValueError("%s is empty; refusing to guess its schema" % path)
    header, last_line = lines[0], lines[-1]
    last_date = last_line.split(",")[0].strip()

    frame = fetch_frame() if frame is None else frame
    if list(frame.columns) != header.split(","):
        raise ValueError("上游列名已变：%r != 文件头 %r；先人工核对再刷新" % (list(frame.columns),
                                                                header.split(",")))
    frame = frame.copy()
    frame[frame.columns[0]] = pd.to_datetime(frame[frame.columns[0]]).dt.strftime("%Y-%m-%d")
    fresh = frame[frame[frame.columns[0]] > last_date]

    text = original
    appended = 0
    if len(fresh):
        rows = [",".join([row[1].iloc[0]] + [_cell(row[1][column]) for column in frame.columns[1:]])
                for row in fresh.iterrows()]
        if not text.endswith("\n"):
            text += "\n"
        text += "\n".join(rows) + "\n"
        appended = len(rows)
    else:
        return dict(status="fresh", last_date=last_date, appended=0,
                    note="no rows newer than %s" % last_date)

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8-sig", newline="")
    os.replace(temporary, path)

    from tools import asset_panel
    curve = asset_panel.curve(Path(online) if online else ONLINE)
    tail = curve.dropna(subset=["y10"]).tail(1)
    if not len(tail):
        raise ValueError("refresh wrote rows but asset_panel.curve cannot read a y10 tail")
    return dict(status="appended", last_date=str(tail.index[-1].date()),
                y10=float(tail["y10"].iloc[-1]), appended=appended,
                previous_last_date=last_date)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Refresh the treasury curve cache.")
    parser.add_argument("--dry", action="store_true",
                        help="fetch and report; write nothing")
    args = parser.parse_args(argv)
    if args.dry:
        path = CSV
        lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        frame = fetch_frame()
        frame = frame.copy()
        frame[frame.columns[0]] = pd.to_datetime(frame[frame.columns[0]]).dt.strftime("%Y-%m-%d")
        print(json.dumps(dict(dry=True, last_date=lines[-1].split(",")[0],
                             fetched_rows=int(len(frame)),
                             newer=int((frame[frame.columns[0]] > lines[-1].split(",")[0]).sum())),
                         ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(refresh(), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
