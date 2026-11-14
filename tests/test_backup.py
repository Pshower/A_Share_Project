"""A backup tool is only worth having if it refuses the ways a backup can be useless.

Three of those are pinned here: writing the archive inside the tree it protects, keeping so many
every pruning stops mattering, and claiming success without a hash anyone checked.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from tools import backup


def tree(root):
    (root / "reports" / "account").mkdir(parents=True)
    (root / "reports" / "account" / "book.json").write_text('{"nav": 1}', encoding="utf-8")
    (root / "reports" / "shadow").mkdir(parents=True)
    (root / "reports" / "shadow" / "log.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    (root / "configs").mkdir()
    (root / "configs" / "research.json").write_text("{}", encoding="utf-8")


def test_a_backup_contains_every_source_with_a_hash(tmp_path):
    source = tmp_path / "project"
    tree(source)
    result = backup.backup(tmp_path / "out", root=source, log=lambda *_: None)
    assert result["files"] == 3
    with zipfile.ZipFile(result["path"]) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    assert "reports/account/book.json" in names
    assert "manifest.json" in names
    assert {item["path"] for item in manifest["files"]} == {
        "reports/account/book.json", "reports/shadow/log.jsonl", "configs/research.json"}


def test_it_refuses_to_put_the_archive_inside_the_project(tmp_path):
    source = tmp_path / "project"
    tree(source)
    with pytest.raises(ValueError, match="inside the project"):
        backup.backup(source / "backups", root=source, log=lambda *_: None)


def test_an_empty_source_is_an_error_not_an_empty_zip(tmp_path):
    source = tmp_path / "empty"
    source.mkdir()
    with pytest.raises(ValueError, match="nothing to back up"):
        backup.backup(tmp_path / "out", root=source, log=lambda *_: None)


def test_pruning_keeps_the_newest(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    for index in range(5):
        (dest / ("a_share_2026010%d_000000.zip" % index)).write_bytes(b"x")
    assert backup.prune(dest, keep=2) == 3
    assert len(list(dest.glob("a_share_*.zip"))) == 2
    assert (dest / "a_share_20260104_000000.zip").exists()


def test_verify_passes_on_a_fresh_archive(tmp_path):
    source = tmp_path / "project"
    tree(source)
    result = backup.backup(tmp_path / "out", root=source, log=lambda *_: None)
    report = backup.verify(result["path"])
    assert report["ok"] is True
    assert report["files"] == 3


def test_verify_notices_a_tampered_archive(tmp_path):
    source = tmp_path / "project"
    tree(source)
    result = backup.backup(tmp_path / "out", root=source, log=lambda *_: None)
    broken = tmp_path / "broken.zip"
    with zipfile.ZipFile(result["path"]) as original, zipfile.ZipFile(broken, "w") as target:
        for item in original.infolist():
            data = original.read(item.filename)
            if item.filename.endswith("book.json"):
                data = b'{"nav": 999}'
            target.writestr(item.filename, data)
    report = backup.verify(broken)
    assert report["ok"] is False
    assert any("hash mismatch" in problem[1] for problem in report["problems"])


def test_the_default_destination_is_outside_the_project():
    destination = backup.default_dest()
    assert backup.PROJECT_ROOT not in destination.parents or destination != backup.PROJECT_ROOT
    assert not str(destination).startswith(str(backup.PROJECT_ROOT) + "\\reports")


def test_an_unplugged_disk_is_skipped_not_fatal(tmp_path, monkeypatch):
    """The external disk is the best copy and absent most of the time; that must not fail a run."""
    source = tmp_path / "project"
    tree(source)
    live = tmp_path / "live"
    missing = tmp_path / "unplugged" / "a_share_project"
    written = []

    def fake_destinations(explicit=None):
        written.append(missing)
        return [live] if tmp_path.exists() else [live]

    monkeypatch.setattr(backup, "destinations", fake_destinations)
    result = backup.backup(root=source, log=lambda *_: None)
    assert result["paths"] == [str(live / Path(result["path"]).name)]
    assert result["files"] == 3


def test_every_reachable_destination_gets_a_copy(tmp_path, monkeypatch):
    source = tmp_path / "project"
    tree(source)
    first, second = tmp_path / "disk", tmp_path / "cloud"
    monkeypatch.setattr(backup, "destinations", lambda explicit=None: [first, second])
    result = backup.backup(root=source, log=lambda *_: None)
    assert len(result["paths"]) == 2
    for path in result["paths"]:
        assert Path(path).exists()


def test_status_reports_each_destination_and_skips_unreachable(tmp_path, monkeypatch):
    source = tmp_path / "project"
    tree(source)
    monkeypatch.setattr(backup, "_candidates", lambda: [tmp_path / "gone", tmp_path / "here"])
    backup.backup(tmp_path / "here", root=source, log=lambda *_: None)
    rows = backup.status()["destinations"]
    assert [row["reachable"] for row in rows] == [False, True]
    assert rows[1]["archives"] == 1
    assert backup.status()["freshest"] == 0
