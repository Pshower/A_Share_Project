"""Run 2024 dispersion only after the six-trial worker has fully sealed."""

import argparse
import time

import torch

from src.data.preprocess import ROOT, file_hash, read_config, write_json
from src.training.research_market_summary import SEEDS, cached_trial, config_for, prepare, run_trial


def prerequisites(root, protocol):
    marker = root / "worker_2024.json"
    receipt = read_config(marker)
    expected = {f"2024_{arm}_{seed}" for arm in ("mean", "attention") for seed in SEEDS}
    completed = receipt["completed_runs"]
    if (receipt["status"] != "complete" or len(completed) != 6
            or {entry["run_id"] for entry in completed} != expected
            or receipt["protocol_sha256"] != file_hash(root / "protocol.json")
            or receipt["script_sha256"] != file_hash(ROOT / "tools" / "run_market_summary_worker.py")):
        raise ValueError("The six-trial worker must be fully sealed before starting its tail")
    for entry in completed:
        directory = root / "runs" / entry["run_id"]
        if file_hash(directory / "completed.json") != entry["completed_sha256"]:
            raise ValueError("Worker trial completion identity changed")
        _, arm, seed = entry["run_id"].split("_")
        config = config_for(protocol["base_config"], protocol["reference_root"], root, "2024", arm, int(seed))
        if cached_trial(directory, config, protocol) is None:
            raise ValueError("Worker trial is not a complete reusable cache")
    return file_hash(marker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reports/runs/ppo_market_summary_20261007")
    parser.add_argument("--reference-root", default="reports/runs/ppo_attribution_20261007")
    args = parser.parse_args()
    root = (ROOT / args.root).resolve()
    protocol = prepare(root, (ROOT / args.reference_root).resolve())
    worker_sha = prerequisites(root, protocol)
    receipt = dict(
        fold="2024", arm="dispersion", seeds=list(SEEDS), script_sha256=file_hash(__file__),
        protocol_sha256=file_hash(root / "protocol.json"), prerequisite_worker_sha256=worker_sha,
        completed_runs=[], status="running", no_global_summary_writes=True,
        source_implementation_unchanged=True, additional_trials_beyond_registration=0,
        progress_status_is_last_writer_not_completion_evidence=True,
    )
    write_json(root / "tail_2024.json", receipt)
    for seed in SEEDS:
        while torch.cuda.mem_get_info()[0] < 1536 * 1024**2:
            print("TAIL waiting for 1.5 GiB free CUDA memory", flush=True)
            time.sleep(10)
        run_trial(root, protocol, "2024", "dispersion", seed)
        run_id = f"2024_dispersion_{seed}"
        marker = root / "runs" / run_id / "completed.json"
        receipt["completed_runs"].append(dict(run_id=run_id, completed_sha256=file_hash(marker)))
        write_json(root / "tail_2024.json", receipt)
        print(f"TAIL SEALED {run_id}", flush=True)
    receipt["status"] = "complete"
    write_json(root / "tail_2024.json", receipt)


if __name__ == "__main__":
    main()
