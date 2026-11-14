"""The forward panel builder is data plumbing, so its guards are what get tested.

The build itself is a download plus the repository's own preprocessor; the failure modes
that matter are the ones that would let a rebuild quietly overwrite a dataset or silently
change the pool.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tools import forward_panel, shadow_run


def write_manifest(path, codes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stock_codes": codes}), encoding="utf-8")
    return path


def test_pool_codes_reads_the_frozen_pool(tmp_path):
    path = write_manifest(tmp_path / "manifest.json", ["600085", "002594", "000063"])
    assert forward_panel.pool_codes(path) == ["600085", "002594", "000063"]


def test_pool_codes_rejects_duplicates(tmp_path):
    path = write_manifest(tmp_path / "manifest.json", ["600085", "600085"])
    with pytest.raises(ValueError):
        forward_panel.pool_codes(path)


def test_stock_list_is_one_code_per_line(tmp_path):
    path = forward_panel.write_stock_list(["600085", "002594"], tmp_path / "deep" / "pool.txt")
    assert path.read_text(encoding="utf-8") == "600085\n002594\n"


def test_build_refuses_to_rewrite_an_existing_dataset(tmp_path, monkeypatch):
    existing = tmp_path / "forward_pool_20261110"
    existing.mkdir()
    monkeypatch.setattr(forward_panel, "CLEAN_DIR", tmp_path)
    monkeypatch.setattr(forward_panel, "STOCK_LIST_DIR", tmp_path / "lists")
    with pytest.raises(ValueError, match="Refusing to rewrite"):
        forward_panel.build(tmp_path / "snapshot", ["600085"], "forward_pool_20261110", "2026-11-10")


def test_prune_keeps_the_newest_panels_and_nothing_else(tmp_path):
    for name in ["forward_pool_20261106", "forward_pool_20261109", "forward_pool_20261110"]:
        (tmp_path / name).mkdir()
    (tmp_path / "research_v3").mkdir()
    removed = forward_panel.prune(2, root=tmp_path, log=lambda *_: None)
    assert removed == ["forward_pool_20261106"]
    assert (tmp_path / "forward_pool_20261109").exists()
    assert (tmp_path / "forward_pool_20261110").exists()
    assert (tmp_path / "research_v3").exists()
    assert forward_panel.prune(0, root=tmp_path, log=lambda *_: None) == []


def test_the_recorder_follows_the_newest_panel(tmp_path):
    """The daily job renames its dataset every day and prunes the old one; a hard-coded
    default goes stale the first time it is pruned and then reports a false ``stale``."""
    for name in ["forward_pool_20261110", "forward_pool_20261112", "forward_pool_20261113"]:
        (tmp_path / name).mkdir()
    assert shadow_run.latest_panel(tmp_path).name == "forward_pool_20261113"
    forward_panel.prune(1, root=tmp_path, log=lambda *_: None)
    assert shadow_run.latest_panel(tmp_path).name == "forward_pool_20261113"


def test_the_recorder_has_no_default_panel_when_none_exists(tmp_path):
    assert shadow_run.latest_panel(tmp_path) is None


def test_the_default_dataset_is_a_panel_that_exists_today():
    """Whatever the default resolves to must be a directory the recorder can actually open."""
    if shadow_run.latest_panel() is not None:
        assert shadow_run.DATASET.is_dir()
        assert shadow_run.DATASET.name.startswith("forward_pool_")


def test_build_reports_the_pool_it_actually_read(tmp_path, monkeypatch):
    snapshot = tmp_path / "h_test" / "hfq"
    snapshot.mkdir(parents=True)
    codes = ["600085", "002594"]
    dates = pd.bdate_range("2017-01-03", "2026-11-10")
    for code in codes:
        rows = ["date,code,open,close,high,low,volume,amount,turnover"]
        rows += [f"{day.date()},{code},10.0,10.1,10.2,9.9,1000000,10100000,0.5" for day in dates]
        (snapshot / f"{code}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(forward_panel, "CLEAN_DIR", tmp_path / "clean")
    monkeypatch.setattr(forward_panel, "STOCK_LIST_DIR", tmp_path / "lists")
    manifest = forward_panel.build(tmp_path / "h_test", codes, "forward_pool_test", "2026-11-10", log=lambda *_: None)
    assert manifest["stock_codes"] == codes
    assert manifest["forward"]["purpose"].startswith("forward extension")
    assert (tmp_path / "clean" / "forward_pool_test" / "manifest.json").exists()


def test_prune_sorts_by_name_so_the_newest_panels_survive(tmp_path):
    from tools import forward_panel
    for name in ("forward_pool_20261110", "forward_pool_20261111", "forward_pool_20261112"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.json").write_text("{}", encoding="utf-8")
    removed = forward_panel.prune(keep=2, root=tmp_path, log=lambda *_: None)
    assert removed == ["forward_pool_20261110"]
    assert sorted(path.name for path in tmp_path.glob("forward_pool_*")) == [
        "forward_pool_20261111", "forward_pool_20261112"]


def test_prune_refuses_directories_that_are_not_forward_panels(tmp_path):
    from tools import forward_panel
    (tmp_path / "forward_pool_20261110").mkdir()
    (tmp_path / "research_v3").mkdir()
    forward_panel.prune(keep=0, root=tmp_path, log=lambda *_: None)
    assert (tmp_path / "research_v3").exists()
