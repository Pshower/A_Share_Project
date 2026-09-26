"""Run an explicit local multiseed protocol; defaults to preparation/checks only."""

import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlparse

import requests

from src.data.preprocess import ROOT, read_config, write_json
from src.web.experiments import digest


class Client:
    def __init__(self, url):
        parsed = urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.path not in {"", "/"}:
            raise ValueError("Use a loopback workbench URL")
        self.url, self.session = url.rstrip("/"), requests.Session()
        self.session.trust_env = False
        self.session.headers["Connection"] = "close"
        self.token = self.get("/session")["token"]
        if Path(self.get("/system")["workspace"]).resolve() != ROOT:
            raise ValueError("Workbench belongs to another workspace")

    def get(self, path):
        for attempt in range(3):
            try:
                response = self.session.get(self.url + "/api" + path, timeout=60)
                break
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 2:
                    raise
                time.sleep(0.25 * (attempt + 1))
        response.raise_for_status()
        return response.json()

    def post(self, path, body):
        response = self.session.post(self.url + "/api" + path, json=body, headers={"X-CSRF-Token": self.token}, timeout=60)
        if not response.ok:
            raise RuntimeError(f"{path}: {response.status_code} {response.text}")
        return response.json()

    def wait(self, jobs):
        ids = {j["id"] for j in jobs}
        previous = None
        while ids:
            records = [j for j in self.get("/jobs") if j["id"] in ids]
            failed = [j for j in records if j["status"] in {"failed", "interrupted", "cancelled"}]
            if failed:
                raise RuntimeError(json.dumps(failed, ensure_ascii=False))
            summary = {j["id"]: j["status"] for j in records}
            if summary != previous:
                print(json.dumps(summary), flush=True)
                previous = summary
            if len(records) == len(ids) and all(j["status"] == "succeeded" for j in records):
                return
            time.sleep(5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8766")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--monitor-only", action="store_true")
    args = parser.parse_args(argv)
    if sum([args.prepare_only, args.train, args.monitor_only]) > 1:
        parser.error("Choose preparation, explicitly approved training or read-only monitoring")
    config = read_config(args.config)
    client = Client(args.url)
    directory = ROOT / ".workbench/protocols"
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / (digest(config)[:20] + ".json")
    state = read_config(state_path) if state_path.exists() else dict(config_digest=digest(config), experiments={})
    if state["config_digest"] != digest(config):
        raise ValueError("Protocol state mismatch")
    if args.monitor_only:
        if not state.get("training_jobs"):
            raise ValueError("No persisted training jobs to monitor")
        client.wait([dict(id=identifier) for identifier in state["training_jobs"]])
        state["results"] = {name: client.get("/experiments/" + identifier)["summaries"] for name, identifier in state["experiments"].items()}
        write_json(state_path, state)
        return state_path
    template = next(d for d in client.get("/datasets") if d["name"] == config["template"])

    def create(variant, dataset_id, build=None, parent=None):
        name = variant["name"]
        if name not in state["experiments"]:
            recipe = dict(name=config["name"] + " " + name,
                          training=dict(config["training"], **variant["overrides"]),
                          build=build, history_ids=[], source_mode="dataset", seeds=config["seeds"], parent_id=parent)
            recipe["training"].update(name=recipe["name"], dataset_id=dataset_id, codes=config["codes"])
            result = client.post("/experiments", recipe)
            state["experiments"][name] = result["id"]
            write_json(state_path, state)
        return state["experiments"][name]

    initial = create(config["variants"][0], template["id"], dict(config["build"], version=config["version"], dataset_id=template["id"], codes=config["codes"]))
    prepared = client.get("/experiments/" + initial)
    if not prepared["dataset_id"]:
        client.wait(client.post(f"/experiments/{initial}/prepare", {}))
    prepared = client.get("/experiments/" + initial)
    if not prepared["can_check"]:
        raise ValueError("Prepared data failed preflight: " + json.dumps(prepared["errors"], ensure_ascii=False))
    state["dataset_id"], state["coverage"] = prepared["dataset_id"], prepared["coverage"]
    for variant in config["variants"][1:]:
        create(variant, state["dataset_id"], parent=initial)
    write_json(state_path, state)
    print("STATE " + str(state_path), flush=True)
    print(json.dumps(dict(dataset=state["dataset_id"], train_start=state["coverage"]["train_start"], train_end=state["coverage"]["train_end"], experiments=state["experiments"])), flush=True)
    if args.prepare_only:
        return state_path
    for identifier in state["experiments"].values():
        client.wait(client.post(f"/experiments/{identifier}/check", {}))
    if args.train:
        jobs = []
        for identifier in state["experiments"].values():
            jobs.extend(client.post(f"/experiments/{identifier}/train", {"allow_training": True}))
        state["training_jobs"] = [j["id"] for j in jobs]
        write_json(state_path, state)
        client.wait(jobs)
        state["results"] = {name: client.get("/experiments/" + identifier)["summaries"] for name, identifier in state["experiments"].items()}
        write_json(state_path, state)
    return state_path


if __name__ == "__main__":
    main()
