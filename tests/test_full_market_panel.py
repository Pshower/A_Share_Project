"""Tests for the full-market panel builder and the chunked download planner."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import download_full_market as dl
from tools import full_market_panel as panel


def test_float_shares_recovers_the_provider_share_count():
    # 600519 on 2017-01-03: 2,076,389 shares traded at 0.165292% of free float.
    price = pd.DataFrame(dict(volume=[2076389.0], turnover=[0.165292]))
    derived = panel.float_shares(price)
    assert derived.iloc[0] == pytest.approx(1.2562e9, rel=1e-3)


def test_float_shares_ignores_zero_turnover_days_and_fills_inside_the_code():
    price = pd.DataFrame(dict(volume=[1000.0, 0.0, 2000.0], turnover=[1.0, 0.0, 2.0]))
    derived = panel.float_shares(price)
    assert derived.tolist() == pytest.approx([100000.0, 100000.0, 100000.0])


def test_float_shares_requires_both_columns():
    with pytest.raises(ValueError):
        panel.float_shares(pd.DataFrame(dict(volume=[1.0])))


def test_usable_accepts_both_date_spellings_and_requires_coverage():
    snapshot = dict(codes=["600519"], start="20170101", end="20241231",
                    bases=["hfq", "unadjusted"], created="", entries=[], id="h_x", directory=Path("."))
    assert panel.usable(snapshot, "2017-01-03", "2024-12-31")
    dashed = dict(snapshot, start="2017-01-01", end="2024-12-31")
    assert panel.usable(dashed, "2017-01-03", "2024-12-31")
    short = dict(snapshot, end="20200101")
    assert not panel.usable(short, "2017-01-03", "2024-12-31")
    one_basis = dict(snapshot, bases=["hfq"])
    assert not panel.usable(one_basis, "2017-01-03", "2024-12-31")
    assert not panel.usable(snapshot, "2017-01-03", "2024-12-31", universe=["000001"])


def test_plan_skips_chunks_already_recorded_as_ready():
    codes = [f"{index:06d}" for index in range(10)]
    progress = dict(start=dl.START, end=dl.END, chunks={"0": dict(status="ready"),
                                                        "1": dict(status="failed")})
    pending = dl.plan(codes, progress, chunk_size=3)
    assert [index for index, _ in pending] == [1, 2, 3]
    assert pending[0][1] == codes[3:6]


def test_load_universe_keeps_numeric_codes_in_file_order(tmp_path):
    path = tmp_path / "universe.json"
    path.write_text(json.dumps(dict(rows=[dict(code="600519"), dict(code="bad"),
                                        dict(code="000001")])), encoding="utf-8")
    assert dl.load_universe(path) == ["600519", "000001"]


def test_load_progress_starts_from_the_declared_range(tmp_path):
    progress = dl.load_progress(tmp_path / "missing.json")
    assert progress["start"] == dl.START and progress["end"] == dl.END
    assert progress["chunks"] == {}
    assert progress["chunk"] == dl.CHUNK
    assert list(progress["bases"]) == list(dl.BASES)


def test_run_refuses_the_network_without_consent(tmp_path):
    with pytest.raises(ValueError):
        dl.run(universe=tmp_path / "u.json", progress_path=tmp_path / "p.json", allow_network=False)

def test_plan_revisits_a_chunk_that_finished_with_failures():
    codes = [f"{index:06d}" for index in range(6)]
    progress = dict(start=dl.START, end=dl.END,
                    chunks={"0": dict(status="partial", ready=2, failed=4)})
    assert [index for index, _ in dl.plan(codes, progress, chunk_size=3)] == [0, 1]


def test_repair_flips_incomplete_chunks_and_leaves_complete_ones():
    codes = [f"{index:06d}" for index in range(6)]
    progress = dict(chunks={"0": dict(status="ready", ready=6, failed=0),
                            "1": dict(status="ready", ready=2, failed=4)})
    assert dl.repair(progress, codes, chunk_size=3) == 1
    assert progress["chunks"]["0"]["status"] == "ready"
    assert progress["chunks"]["1"]["status"] == "partial"

def test_classify_failures_separates_a_rate_limit_from_a_missing_listing(tmp_path):
    snapshot = tmp_path / "h_test"
    snapshot.mkdir()
    (snapshot / "manifest.json").write_text(json.dumps(dict(entries=[
        dict(code="1", status="ready"),
        dict(code="2", status="failed", error="History is empty or missing required fields"),
        dict(code="3", status="failed", error="404 Client Error: Not Found"),
        dict(code="4", status="failed", error="Batch circuit/time limit reached")])), encoding="utf-8")
    detail = dl.classify_failures(snapshot)
    assert detail == {"ready": 1, "permanent": 2, "transient": 1}


def test_repair_leaves_a_chunk_closed_when_every_failure_is_permanent(tmp_path):
    snapshot = tmp_path / "history" / "h_one"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text(json.dumps(dict(entries=[
        dict(code="1", status="failed", error="History is empty or missing required fields")])),
        encoding="utf-8")
    codes = ["000001"]
    progress = dict(chunks={"0": dict(status="ready", ready=0, failed=2, snapshot="h_one")})
    assert dl.repair(progress, codes, chunk_size=1, history_root=tmp_path / "history") == 0
    assert progress["chunks"]["0"]["status"] == "ready"

def test_missing_codes_skips_codes_that_already_have_both_bases():
    fm = __import__("tools.fetch_missing", fromlist=["fetch_missing"])
    ready = {("000001", "hfq"), ("000001", "unadjusted"), ("000002", "hfq")}
    assert fm.missing_codes(["000001", "000002", "000003"], ready) == ["000002", "000003"]


def test_batches_cover_every_item_once():
    fm = __import__("tools.fetch_missing", fromlist=["fetch_missing"])
    items = list(range(11))
    groups = list(fm.batches(items, 3))
    assert [len(g) for g in groups] == [3, 3, 3, 2]
    assert [x for g in groups for x in g] == items
