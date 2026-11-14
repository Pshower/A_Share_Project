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


def default_dest():
    """A synced folder first, then a sibling of the project.

    A backup on the same disk survives a mistaken delete (which is what happened on 2026-11-14) but
    not a disk failure, so a OneDrive folder wins when one is configured.
    """
    candidates = []
    if os.environ.get("ASHR_BACKUP_DIR"):
        candidates.append(Path(os.environ["ASHR_BACKUP_DIR"]))
    if os.environ.get("OneDrive"):
        candidates.append(Path(os.environ["OneDrive"]) / "a_share_backups" / "a_share_project")
    candidates.append(PROJECT_ROOT.parent / "_backups")
    for candidate in candidates:
        if writable(candidate):
            return candidate
    raise RuntimeError("no writable backup destination; set ASHR_BACKUP_DIR")


def backup(dest=None, root=PROJECT_ROOT, keep=KEEP, stamp=None, log=print):
    root = Path(root).resolve()
    dest = Path(dest or default_dest()).resolve()
    if dest == root or root in dest.parents:
        raise ValueError("refusing to write the backup inside the project: %s" % dest)
    files = collect(root)
    if not files:
        raise ValueError("nothing to back up under %s" % root)
    dest.mkdir(parents=True, exist_ok=True)
    moment = stamp or datetime.now(timezone.utc)
    name = "a_share_%s.zip" % moment.strftime("%Y%m%d_%H%M%S")
    target = dest / name
    manifest = dict(created_at=moment.isoformat(), project=str(root), sources=list(SOURCES),
                    files=[dict(path=path.relative_to(root).as_posix(), size=path.stat().st_size,
                                sha256=digest(path)) for path in files])
    partial = dest / (name + ".part")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in manifest["files"]:
            archive.write(root / item["path"], item["path"])
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    partial.replace(target)
    log("backed up %d files (%d sources) -> %s" % (len(files), len(manifest["sources"]), target))
    pruned = prune(dest, keep)
    if pruned:
        log("pruned %d old archive(s)" % pruned)
    return dict(path=str(target), files=len(files), bytes=target.stat().st_size,
                manifest=manifest, pruned=pruned)


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
    archives = sorted(Path(dest or default_dest()).glob("a_share_*.zip"))
    return dict(dest=str(dest or default_dest()), archives=len(archives),
                latest=str(archives[-1]) if archives else None,
                latest_at=datetime.fromtimestamp(archives[-1].stat().st_mtime,
                                                 tz=timezone.utc).isoformat() if archives else None)


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
