"""The health record must count what actually happened, and must never be the reason a job fails."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tools import data_health


def archive(root):
    root.mkdir(parents=True, exist_ok=True)
    for name, status, created in (("h_a", "ready", "2026-11-17T20:00:00"),
                                  ("h_b", "partial", "2026-11-17T21:00:00"),
                                  ("h_c", "failed", "2026-11-18T20:00:00")):
        directory = root / name
        directory.mkdir()
        (directory / "manifest.json").write_text(
            json.dumps({"id": name, "status": status, "created_at": created}), encoding="utf-8")
    return root


def panel(root, days=15, codes=5):
    index = pd.bdate_range("2026-01-05", periods=days)
    columns = pd.MultiIndex.from_product([["%06d" % code for code in range(codes)], ["close"]])
    values = np.full((days, codes), 10.0)
    if days > 9:
        values[6, 0] = np.nan                   # one bar missing while listed
        values[9, 1] = 10.0 * 1.45              # one 45% jump
    frame = pd.DataFrame(values, index=index, columns=columns)
    root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(root / "prices.parquet")
    (root / "manifest.json").write_text(
        json.dumps({"version": "pool_test", "stock_codes": ["%06d" % c for c in range(codes)]}),
        encoding="utf-8")
    return root


def test_the_archive_counts_each_day_and_its_status(tmp_path):
    counts = data_health.snapshots(archive(tmp_path / "history"))
    assert counts["2026-11-17"] == dict(snapshots=2, ok=1, partial=1, failed=0)
    assert counts["2026-11-18"]["failed"] == 1


def test_the_panel_names_a_missing_bar_and_a_jump(tmp_path):
    health = data_health.panel_health(panel(tmp_path / "pool"))
    assert health["codes"] == 5 and health["dates"] == 15
    assert health["missing_total"] >= 1
    assert health["jumps"] >= 1
    assert any(item["code"] == "000001" for item in health["jump_examples"])


def test_a_short_panel_does_not_invent_missing_bars(tmp_path):
    """Codes listed later must not be counted as missing in the years before they listed."""
    health = data_health.panel_health(panel(tmp_path / "pool", days=8, codes=3))
    assert health["missing_total"] == 0


def test_the_report_is_json_ready(tmp_path):
    payload = data_health.report(archive(tmp_path / "history"), tmp_path / "pool")
    json.dumps(payload)                        # must not raise
    assert payload["history"]["snapshots"] == 3
