"""The fund-flow archive is append-only and deduped; a rerun must not rewrite or double rows.

It is the only tool whose scheduling outlived the branch cut -- the weekly job calls it, so if it
is missing the job fails silently on a Saturday evening.  These tests cover the parts that decide
what a rerun does; ``fetch`` itself needs the network and is left to the job.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from tools import archive_fund_flow as flow


def test_market_of_routes_each_board():
    assert flow.market_of("600519") == "sh"
    assert flow.market_of("688981") == "sh"
    assert flow.market_of("000001") == "sz"
    assert flow.market_of("300750") == "sz"


def test_the_default_universe_is_the_spot_snapshot(tmp_path, monkeypatch):
    snapshot = tmp_path / "a_share_spot.json"
    snapshot.write_text(json.dumps({"rows": [{"code": "600519"}, {"code": "1"}]}), encoding="utf-8")
    monkeypatch.setattr(flow, "UNIVERSE", snapshot)
    assert flow.universe_codes() == ["600519", "000001"]
    assert flow.universe_codes(1) == ["600519"]


def test_an_empty_archive_reads_as_the_declared_columns(tmp_path):
    frame = flow.existing(tmp_path / "missing.parquet")
    assert list(frame.columns) == list(flow.KEEP)
    assert frame.empty


def test_an_existing_archive_is_returned_unchanged(tmp_path):
    path = tmp_path / "index.parquet"
    pd.DataFrame([{column: 0 for column in flow.KEEP}]).to_parquet(path, index=False)
    frame = flow.existing(path)
    assert len(frame) == 1
    assert list(frame.columns) == list(flow.KEEP)
