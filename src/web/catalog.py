"""Resource IDs, trusted roots and read-only research artifact discovery."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import pandas as pd

from src.data.preprocess import ROOT, file_hash, read_config


def identifier(path, root):
    return hashlib.sha256(str(Path(path).relative_to(root)).replace("\\", "/").encode()).hexdigest()[:20]


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Resource escapes its managed directory")
    return path


class Catalog:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.clean = self.root / "data/clean"
        self.runs = self.root / "reports/runs"
        self._hash_cache = {}

    def hash(self, path):
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self._hash_cache:
            self._hash_cache = {k: v for k, v in self._hash_cache.items() if k[0] != str(path)}
            self._hash_cache[key] = file_hash(path)
        return self._hash_cache[key]

    def datasets(self):
        records = []
        for path in sorted(self.clean.glob("*/manifest.json")):
            if path.parent.name.startswith("."):
                continue
            try:
                path = inside(path, self.clean)
                m = read_config(path)
                if m.get("schema_version") != 2:
                    continue
                records.append(dict(id=identifier(path.parent, self.root), name=path.parent.name,
                                    version=m["version"], stocks=len(m["stock_codes"]), n_dates=m["n_dates"],
                                    features=len(m["feature_cols"]), price_basis=m["price_basis"],
                                    start_date=m["start_date"], end_date=m["end_date"],
                                    train_end=m["train_end"], val_end=m["val_end"],
                                    relative_path=str(path.parent.relative_to(self.root)), manifest=m))
            except (ValueError, OSError, KeyError):
                continue
        return records

    def dataset(self, dataset_id):
        for record in self.datasets():
            if record["id"] == dataset_id:
                return record, inside(self.root / record["relative_path"], self.clean)
        raise ValueError("Unknown dataset ID")

    def research_config(self, dataset_id):
        record, directory = self.dataset(dataset_id)
        if (directory / "research_config.json").exists():
            config = read_config(inside(directory / "research_config.json", directory))
        else:
            config = read_config(self.root / "configs/research.json")
            config["dataset"].update(record["manifest"]["build_config"])
            config["dataset"]["train_codes"] = record["manifest"]["build_config"]["train_codes"]
        config["dataset"]["clean_dir"] = str(directory)
        for field in ["raw_dir", "stock_list_path"]:
            path = Path(config["dataset"][field])
            config["dataset"][field] = str(inside(path if path.is_absolute() else self.root / path, self.root))
        return config

    def stocks(self, dataset_id):
        record, directory = self.dataset(dataset_id)
        quality_path = directory / "quality.csv"
        quality = pd.read_csv(quality_path, dtype={"code": str}).set_index("code").to_dict("index") if quality_path.exists() else {}
        return [dict(code=c, **quality.get(c, {})) for c in sorted(record["manifest"]["stock_codes"])]

    def models(self):
        records = []
        for path in self.runs.rglob("model.zip"):
            try:
                path = inside(path, self.runs)
                metadata = read_config(inside(path.parent / "metadata.json", path.parent))
                valid = self.hash(path) == metadata["model_sha256"]
                c = metadata["contract"]
                records.append(dict(id=identifier(path.parent, self.root), name=path.parent.name,
                                    run=path.parent.parent.name, timesteps=metadata["num_timesteps"],
                                    pool_size=len(c["stock_codes"]), dataset=c.get("dataset_version", "unknown"),
                                    price_basis=c["price_basis"], trained=metadata["num_timesteps"] > 0 or bool(metadata.get("provenance", {}).get("source_num_timesteps")),
                                    valid=valid, relative_path=str(path.parent.relative_to(self.root)),
                                    modified=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                                    metadata=metadata))
            except (ValueError, OSError, KeyError):
                continue
        return sorted(records, key=lambda x: x["modified"], reverse=True)

    def model(self, model_id):
        for record in self.models():
            if record["id"] == model_id:
                if not record["valid"]:
                    raise ValueError("Model hash mismatch")
                return record, inside(self.root / record["relative_path"], self.runs)
        raise ValueError("Unknown or incomplete model")

    def model_config(self, model_id):
        record, directory = self.model(model_id)
        config = deepcopy(record["metadata"]["config"])
        research_path = Path(config["research_config"])
        research_path = inside(research_path if research_path.is_absolute() else self.root / research_path, self.root)
        if not research_path.exists():
            raise ValueError("Model's research config is missing; restore the original snapshot")
        return config, record, directory

    def reports(self):
        records = []
        for path in self.runs.rglob("metrics.json"):
            try:
                path = inside(path, self.runs)
                metrics = read_config(path)
                meta = read_config(path.parent / "experiment.json") if (path.parent / "experiment.json").exists() else {}
                records.append(dict(id=identifier(path.parent, self.root), name=path.parent.name,
                                    run=path.parent.parent.name, split=meta.get("split", "val"), metrics=metrics,
                                    relative_path=str(path.parent.relative_to(self.root)),
                                    modified=path.stat().st_mtime, experiment=meta))
            except (ValueError, OSError, KeyError):
                continue
        return sorted(records, key=lambda x: x["modified"], reverse=True)

    def report(self, report_id):
        for record in self.reports():
            if record["id"] == report_id:
                return record, inside(self.root / record["relative_path"], self.runs)
        raise ValueError("Unknown report")

    def artifact(self, report_id, name):
        _, directory = self.report(report_id)
        path = inside(directory / name, directory)
        if path.suffix not in {".json", ".csv", ".parquet", ".png", ".md"} or not path.is_file():
            raise ValueError("Unsupported or missing report artifact")
        return path
