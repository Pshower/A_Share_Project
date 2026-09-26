"""Resource IDs, trusted roots and read-only research artifact discovery."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
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
                config_path = path.parent / "research_config.json"
                source_history_id = (read_config(config_path).get("online_source", {}).get("history_id")
                                     if config_path.exists() else None)
                records.append(dict(id=identifier(path.parent, self.root), name=path.parent.name,
                                    source_history_id=source_history_id,
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

    def coverage(self, dataset_id, codes, lookback):
        from src.data.coverage import training_coverage
        from src.data.snapshots import Snapshots
        record, directory = self.dataset(dataset_id)
        if not codes or len(set(codes)) != len(codes) or not set(codes) <= set(record["manifest"]["stock_codes"]):
            raise ValueError("Selected stocks must belong to the dataset")
        for name in ["panel_data.parquet", "prices.parquet"]:
            if self.hash(directory / name) != record["manifest"]["artifacts"][name]:
                raise ValueError("Dataset artifact hash mismatch")
        features = pd.read_parquet(directory / "panel_data.parquet")
        prices = pd.read_parquet(directory / "prices.parquet")
        result = training_coverage(features, prices, codes, lookback, record["train_end"])
        cfg = self.research_config(dataset_id)
        result["source_history_id"] = cfg.get("online_source", {}).get("history_id")
        result["dataset_id"] = dataset_id
        result["history_matches"] = [dict(id=h["id"], status=h["status"],
                                         matched_codes=sorted(set(codes) & set(h["request"]["codes"])))
                                     for h in Snapshots(self.root).list("history")
                                     if set(codes) & set(h["request"]["codes"])]
        audit_path = directory / "source_audit.csv"
        if audit_path.exists():
            audit = pd.read_csv(audit_path, dtype={"code": str}).set_index("code")
            for row in result["stocks"]:
                if row["code"] in audit.index:
                    source = audit.loc[row["code"]]
                    row.update(download_start=str(source["source_start"]), download_end=str(source["source_end"]))
        return result

    def source_bars(self, dataset_id, code):
        from src.data.preprocess import DataPreprocessor
        record, _ = self.dataset(dataset_id)
        if code not in record["manifest"]["stock_codes"]:
            raise ValueError("Stock is not part of the dataset")
        cfg = self.research_config(dataset_id)
        processor = DataPreprocessor(**cfg["dataset"])
        candidates = list(processor.raw_dir.glob(processor.raw_pattern.format(code=code)))
        if len(candidates) != 1:
            raise ValueError("Expected one original source file")
        path = inside(candidates[0], self.root)
        expected = {s["file"]: s["sha256"] for s in record["manifest"]["sources"]}
        if expected.get(path.name) != self.hash(path):
            raise ValueError("Original source hash mismatch; rebuild a new version")
        raw = pd.read_csv(path, dtype={"股票代码": str, "code": str})
        frame = processor.clean_single_stock(raw)
        if not frame.code.eq(code).all():
            raise ValueError("Original source stock mismatch")
        frame.loc[~frame.valid_price, ["open", "close", "high", "low"]] = float("nan")
        frame["date"] = frame.date.dt.strftime("%Y-%m-%d")
        return dict(code=code, basis="hfq", dataset_id=dataset_id, source_file=path.name,
                    volume_unit="shares", rows=json.loads(frame.drop(columns="valid_price").to_json(orient="records")))

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
