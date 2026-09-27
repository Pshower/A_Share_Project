"""Immutable experiment recipes with explicit, independently approved stages."""

import hashlib
import json
import math
from pathlib import Path

from src.data.preprocess import ROOT, file_hash, read_config
from src.data.snapshots import Snapshots
from src.data.source_bundle import inspect_bundle
from .catalog import Catalog
from .schemas import Training


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def code_hashes(root, data_only=False):
    directory = Path(root) / ("src/data" if data_only else "src")
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(directory.rglob("*.py"))}


def training_fingerprint(payload, root=ROOT):
    record, directory = Catalog(root).dataset(payload["dataset_id"])
    clean = {k: v for k, v in payload.items() if not k.startswith("experiment_")}
    return digest(dict(training=clean, manifest=file_hash(directory / "manifest.json"), code=code_hashes(root)))


def preflight(store, identifier, root=ROOT):
    experiment = store.experiment(identifier)
    recipe, catalog = experiment["recipe"], Catalog(root)
    record, directory = catalog.dataset(recipe["training"]["dataset_id"])
    jobs = store.experiment_jobs(identifier)
    history_ids = list(recipe["history_ids"])
    for job in reversed(jobs):
        if job["stage"] == "download" and job["status"] == "succeeded" and job.get("result", {}).get("snapshot_id"):
            if job["result"]["snapshot_id"] not in history_ids:
                history_ids.append(job["result"]["snapshot_id"])
    sources = {i: file_hash(Snapshots(root).get("history", i)[1] / "manifest.json") for i in history_ids}
    fingerprint = digest(dict(recipe=recipe, source_manifests=sources, template=file_hash(directory / "manifest.json"), code=code_hashes(root, True)))
    source_report = dict(ready=True, missing=[], conflicts=[], bindings={})
    errors = []
    if recipe.get("template_manifest_sha256", file_hash(directory / "manifest.json")) != file_hash(directory / "manifest.json"):
        errors.append("Template manifest changed; create a new recipe revision")
    if any(sources.get(i) != sha for i, sha in recipe.get("source_manifest_hashes", {}).items()):
        errors.append("Bound source manifest changed; create a new recipe revision")
    build = recipe["build"]
    if build:
        if recipe["source_mode"] == "history":
            source_report = inspect_bundle(history_ids, build["codes"], build["start_date"], build["end_date"], root)
        else:
            missing = sorted(set(build["codes"]) - set(record["manifest"]["stock_codes"]))
            source_report.update(ready=not missing, missing=missing)
            if not missing:
                for code in build["codes"]:
                    try:
                        catalog.source_bars(record["id"], code)
                    except (OSError, ValueError) as error:
                        errors.append(f"{code}: {error}")
    resolved = recipe["training"]["dataset_id"] if build is None else None
    prepared = next((j for j in jobs if j["stage"] == "prepare" and j["status"] == "succeeded"), None)
    if prepared and prepared["fingerprint"] == fingerprint:
        resolved = prepared["result"]["dataset_id"]
    elif prepared:
        errors.append("Prepared source/configuration changed; create a new recipe revision and version name")
    coverage = None
    if resolved:
        coverage = catalog.coverage(resolved, recipe["training"]["codes"], recipe["training"]["lookback"])
        if not coverage["eligible"]:
            errors.append(coverage["reason"])
    if source_report["conflicts"]:
        errors.append("Conflicting HFQ sources: explicitly choose one capture for these stocks")
    history_source = recipe.get("history_source") or source_report.get("provider") or "sina"
    if source_report.get("provider") not in {None, history_source}:
        errors.append("Experiment history source differs from bound snapshots")
    config = recipe["training"]
    definitions = (record["manifest"].get("extra_features", []) if build and build.get("extra_features") is None else (build or {}).get("extra_features", []))
    features = 20 + len(definitions) if build else record["features"]
    memory = config["n_steps"] * len(config["codes"]) * config["lookback"] * features * 4 * 3
    if memory > 512 * 1024 ** 2:
        errors.append("Estimated rollout allocation exceeds 512 MB")
    checks_ready = bool(resolved and coverage and coverage["eligible"] and not errors)
    if checks_ready:
        for seed in recipe["seeds"]:
            payload = Training(**dict(config, dataset_id=resolved, seed=seed)).model_dump(mode="json")
            expected = training_fingerprint(payload, root)
            checks_ready = checks_ready and any(j["stage"] == f"check:{seed}" and j["status"] == "succeeded" and j["fingerprint"] == expected for j in jobs)
    return dict(**experiment, jobs=jobs, source_report=source_report, errors=errors,
                history_ids=history_ids, preparation_fingerprint=fingerprint, dataset_id=resolved,
                history_source=history_source,
                dataset_name=(build["version"] if build else record["name"]) if resolved else None,
                coverage=coverage, estimated_rollout_mb=memory / 1024 ** 2,
                total_requested_steps=config["total_timesteps"] * len(recipe["seeds"]),
                total_effective_steps=math.ceil(config["total_timesteps"] / config["n_steps"]) * config["n_steps"] * len(recipe["seeds"]),
                checks_ready=checks_ready,
                can_prepare=bool(build and source_report["ready"] and not errors and not resolved),
                can_check=bool(resolved and coverage and coverage["eligible"] and not errors))


def stage_jobs(store, identifier, stage, approval, root=ROOT):
    state = preflight(store, identifier, root)
    recipe = state["recipe"]
    if stage == "download":
        if state["errors"]:
            raise ValueError("Resolve preflight errors before downloading")
        if not approval.allow_network or recipe["source_mode"] != "history":
            raise ValueError("Explicit network approval and history source mode are required")
        codes = state["source_report"]["missing"]
        if not codes:
            raise ValueError("No missing stock sources; conflicts require explicit source selection")
        build = recipe["build"]
        payload = dict(codes=codes, start_date=build["start_date"], end_date=build["end_date"], bases=["hfq"], allow_network=True,
                       source=state["history_source"])
        return [store.enqueue_experiment(identifier, stage, "online_history", payload, digest(payload))]
    if stage == "prepare":
        if state["dataset_id"]:
            return [j for j in state["jobs"] if j["stage"] == "prepare" and j["status"] == "succeeded"][:1]
        if not state["can_prepare"]:
            raise ValueError("Resolve missing sources, conflicts and preflight errors first")
        return [store.enqueue_experiment(identifier, stage, "experiment_prepare", {}, state["preparation_fingerprint"])]
    if stage not in {"check", "train"} or not state["can_check"]:
        raise ValueError("A prepared, compatible dataset is required")
    if stage == "train" and not approval.allow_training:
        raise ValueError("Explicit training approval is required")
    planned = []
    for seed in recipe["seeds"]:
        payload = Training(**dict(recipe["training"], dataset_id=state["dataset_id"], seed=seed)).model_dump(mode="json")
        fingerprint = training_fingerprint(payload, root)
        if stage == "train":
            checked = any(j["stage"] == f"check:{seed}" and j["status"] == "succeeded" and j["fingerprint"] == fingerprint for j in state["jobs"])
            if not checked:
                raise ValueError(f"Run a fresh configuration check for seed {seed}")
        planned.append((seed, payload, fingerprint))
    return [store.enqueue_experiment(identifier, f"{stage}:{seed}", "training" if stage == "train" else "check", payload, fingerprint)
            for seed, payload, fingerprint in planned]


def summaries(state, root=ROOT):
    rows = []
    seen = set()
    for job in state["jobs"]:
        if job["kind"] != "training" or job["payload"]["seed"] in seen:
            continue
        seen.add(job["payload"]["seed"])
        if job["status"] != "succeeded":
            continue
        directory = Path(root) / "reports/runs/workbench" / job["id"]
        selection = read_config(directory / "selection.json") if (directory / "selection.json").exists() else {}
        comparison = directory / "selected_comparison/metrics.json"
        if not comparison.exists():
            comparison = directory / "final_comparison_no_valid_selection/metrics.json"
        rows.append(dict(job_id=job["id"], seed=job["payload"]["seed"], selected=bool(selection.get("best_path")),
                         best_path=selection.get("best_path"), metrics=read_config(comparison) if comparison.exists() else {}))
    return rows
