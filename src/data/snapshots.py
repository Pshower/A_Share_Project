"""Immutable online artifacts, distinct from training datasets and mutable caches."""

from datetime import datetime, timezone
from pathlib import Path
import re
import uuid

from src.data.preprocess import ROOT, file_hash, read_config, write_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class Snapshots:
    KINDS = {"history": "h", "quotes": "q", "plans": "p"}

    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.directory = self.root / "data/online"

    def begin(self, kind):
        prefix = self.KINDS[kind]
        identifier = prefix + "_" + uuid.uuid4().hex[:16]
        directory = self.directory / kind / ("." + identifier)
        directory.mkdir(parents=True, exist_ok=False)
        return identifier, directory

    def publish(self, kind, identifier, directory, metadata):
        final = self.directory / kind / identifier
        if directory.resolve().parent != final.resolve().parent or final.exists():
            raise ValueError("Invalid snapshot publication path")
        metadata = dict(metadata, id=identifier, kind=kind, schema_version=1, created_at=utc_now(),
                        files={str(p.relative_to(directory)).replace("\\", "/"): file_hash(p)
                               for p in directory.rglob("*") if p.is_file() and p.name != "manifest.json"})
        write_json(directory / "manifest.json", metadata)
        directory.rename(final)
        return metadata

    def get(self, kind, identifier, verify=True):
        if not re.fullmatch(self.KINDS[kind] + r"_[0-9a-f]{16}", identifier):
            raise ValueError("Invalid snapshot identifier")
        directory = (self.directory / kind / identifier).resolve()
        if not directory.is_relative_to(self.root):
            raise ValueError("Snapshot escapes the workspace")
        metadata = read_config(directory / "manifest.json")
        if metadata["id"] != identifier or metadata["kind"] != kind:
            raise ValueError("Snapshot identity mismatch")
        if verify:
            for name, digest in metadata["files"].items():
                path = (directory / name).resolve()
                if not path.is_relative_to(directory) or file_hash(path) != digest:
                    raise ValueError("Snapshot file mismatch")
        return metadata, directory

    def list(self, kind):
        result = []
        for path in (self.directory / kind).glob("*/manifest.json"):
            if path.parent.name.startswith("."):
                continue
            try:
                result.append(self.get(kind, path.parent.name, verify=False)[0])
            except (OSError, ValueError, KeyError):
                continue
        return sorted(result, key=lambda r: r["created_at"], reverse=True)
