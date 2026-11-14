"""One zip a day of the things that cannot be rebuilt.

On 2026-11-14 a pruning script of mine deleted ``reports/runs`` -- 11.94 GB, no copy anywhere.  The
books and the shadow ledger happen to live in Git, but the configs, the pool lists, the study
outputs and anything else under ``reports/runs`` do not.  This tool packs the small, irreplaceable
set into a dated archive **outside the repository**, keeps the last N, and can prove the archive is
intact by restoring it and comparing hashes.

It is deliberately narrow: a backup that copies 12 GB a day will be turned off within a week.  The
set below is about 2 MB, which is small enough to run after every session.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "reports/account",
    "reports/shadow",
    "configs",
    "data/stock_list",
    "reports/runs",
)
KEEP = 30


def digest(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect(root=PROJECT_ROOT, sources=SOURCES):
    """Every file worth keeping, as project-relative paths."""
    root = Path(root)
    files = []
    for name in sources:
        base = root / name
        if not base.exists():
            continue
        if base.is_file():
            files.append(base)
        else:
            files.extend(sorted(path for path in base.rglob("*") if path.is_file()))
    return files


def writable(path):
    """A drive letter that exists is not the same as a directory we can write into.

    ``E:`` on this machine is listed by Get-PSDrive and by nothing else -- mkdir fails with "system
    cannot find the file".  So candidates are probed, not assumed.
    """
    try:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


REMOVABLE = ("E:/", "F:/", "G:/")     # an external disk is the only copy that leaves this machine


def _candidates():
    """Where a copy could go, most valuable first.

    * a removable disk, when it happens to be plugged in -- that copy survives this machine dying;
    * the synced folder, which is always there but lives on the same disk;
    * a sibling directory, which only survives a mistaken delete.
    """
    roots = []
    if os.environ.get("ASHR_BACKUP_DIR"):
        roots.append(Path(os.environ["ASHR_BACKUP_DIR"]))
    for drive in REMOVABLE:
        roots.append(Path(drive) / "a_share_backups" / "a_share_project")
    if os.environ.get("OneDrive"):
        roots.append(Path(os.environ["OneDrive"]) / "a_share_backups" / "a_share_project")
    roots.append(PROJECT_ROOT.parent / "_backups")
    return roots


def destinations(explicit=None):
    """Every reachable destination.  An unplugged disk is skipped, not an error."""
    if explicit:
        return [Path(explicit)]
    return [root for root in _candidates() if writable(root)]


def reachable_now(path):
    """Writable *without creating it* -- a status check must not conjure the directory it tests."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def default_dest():
    """The first reachable destination, kept for callers that want a single path."""
    found = destinations()
    if not found:
        raise RuntimeError("no writable backup destination; set ASHR_BACKUP_DIR")
    return found[0]


def backup(dest=None, root=PROJECT_ROOT, keep=KEEP, stamp=None, log=print):
    """Write one archive per reachable destination; an unplugged disk is skipped, not fatal."""
    root = Path(root).resolve()
    targets = destinations(dest)
    if not targets:
        raise RuntimeError("no writable backup destination; set ASHR_BACKUP_DIR")
    for target in targets:
        resolved = Path(target).resolve()
        if resolved == root or root in resolved.parents:
            raise ValueError("refusing to write the backup inside the project: %s" % resolved)
    files = collect(root)
    if not files:
        raise ValueError("nothing to back up under %s" % root)
    moment = stamp or datetime.now(timezone.utc)
    name = "a_share_%s.zip" % moment.strftime("%Y%m%d_%H%M%S")
    manifest = dict(created_at=moment.isoformat(), project=str(root), sources=list(SOURCES),
                    files=[dict(path=path.relative_to(root).as_posix(), size=path.stat().st_size,
                                sha256=digest(path)) for path in files])
    written, pruned_total = [], 0
    for target in targets:
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        archive_path = target / name
        partial = target / (name + ".part")
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in manifest["files"]:
                archive.write(root / item["path"], item["path"])
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        partial.replace(archive_path)
        written.append(str(archive_path))
        log("backed up %d files -> %s" % (len(files), archive_path))
        pruned_total += prune(target, keep)
    return dict(paths=written, path=written[0], files=len(files),
                bytes=Path(written[0]).stat().st_size, manifest=manifest, pruned=pruned_total)


def prune(dest, keep=KEEP, log=None):
    archives = sorted(Path(dest).glob("a_share_*.zip"))
    doomed = archives[:-keep] if keep > 0 else []
    for path in doomed:
        path.unlink()
    if log and doomed:
        log("removed %d" % len(doomed))
    return len(doomed)


def verify(archive):
    """Reopen the zip and compare every file's hash with the manifest it ships."""
    archive = Path(archive)
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read("manifest.json").decode("utf-8"))
        problems = []
        for item in manifest["files"]:
            try:
                with bundle.open(item["path"]) as stream:
                    hasher = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(1 << 20), b""):
                        hasher.update(chunk)
                if hasher.hexdigest() != item["sha256"]:
                    problems.append((item["path"], "hash mismatch"))
            except KeyError:
                problems.append((item["path"], "missing from archive"))
    return dict(archive=str(archive), files=len(manifest["files"]), created_at=manifest["created_at"],
                ok=not problems, problems=problems[:10])


def latest(dest=None):
    archives = sorted(Path(dest or default_dest()).glob("a_share_*.zip"))
    return archives[-1] if archives else None


def status(dest=None):
    """Per destination: is it reachable right now, how many archives, and how old is the newest.

    A removable disk is usually unplugged, so "unreachable" is not a failure -- but "reachable and
    stale" is, and that is what the age column is for.
    """
    reachable = set(str(Path(path)) for path in (destinations(dest) if dest else []))
    rows = []
    for root in ([Path(dest)] if dest else _candidates()):
        root = Path(root)
        archives = sorted(root.glob("a_share_*.zip")) if root.exists() else []
        newest = archives[-1] if archives else None
        age_days = ((datetime.now(timezone.utc) -
                     datetime.fromtimestamp(newest.stat().st_mtime, tz=timezone.utc)).days
                    if newest else None)
        rows.append(dict(dest=str(root), reachable=str(root) in reachable or reachable_now(root),
                         archives=len(archives),
                         latest=str(newest) if newest else None,
                         age_days=age_days))
    return dict(destinations=rows,
                ready=any(row["reachable"] for row in rows),
                freshest=min((row["age_days"] for row in rows if row["age_days"] is not None),
                             default=None))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Daily zip of the irreplaceable files.")
    parser.add_argument("--dest", default=None, help="Where archives go; must be outside the project.")
    parser.add_argument("--keep", type=int, default=KEEP)
    parser.add_argument("--verify", action="store_true", help="Verify the newest archive.")
    parser.add_argument("--status", action="store_true", help="Show destination and archive count.")
    args = parser.parse_args(argv)
    if args.status:
        print(json.dumps(status(args.dest), ensure_ascii=False, indent=2))
        return 0
    if args.verify:
        archive = latest(args.dest)
        if archive is None:
            print(json.dumps(dict(ok=False, reason="no archive"), ensure_ascii=False))
            return 1
        report = verify(archive)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    result = backup(args.dest, keep=args.keep)
    print(json.dumps({key: result[key] for key in ("path", "files", "bytes", "pruned")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
