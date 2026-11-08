"""Run the registered 2024 mean/attention prefix in an independent process."""

import argparse
import time

import torch

from src.data.preprocess import ROOT, file_hash, write_json
from src.training.research_market_summary import prepare, rule_controls, run_trial, SEEDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reports/runs/ppo_market_summary_20261007")
    parser.add_argument("--reference-root", default="reports/runs/ppo_attribution_20261007")
    args = parser.parse_args()
    root = (ROOT / args.root).resolve()
    reference = (ROOT / args.reference_root).resolve()
    protocol = prepare(root, reference)
    receipt = dict(
        fold="2024", arms=["mean", "attention"], seeds=list(SEEDS),
        script_sha256=file_hash(__file__), protocol_sha256=file_hash(root / "protocol.json"),
        source_implementation_unchanged=True, additional_trials_beyond_registration=0,
        no_global_summary_writes=True, completed_runs=[], status="running",
        scheduling="Main process owns 2023 and later 2024 dispersion; cached trials require completed markers.",
        progress_status_is_last_writer_not_completion_evidence=True,
    )
    write_json(root / "worker_2024.json", receipt)
    rule_controls(root, protocol, "2024")
    for arm in receipt["arms"]:
        for seed in SEEDS:
            # Leave room for the other process's measured sub-0.5-GiB allocator.
            while torch.cuda.mem_get_info()[0] < 1536 * 1024**2:
                print("WORKER waiting for 1.5 GiB free CUDA memory", flush=True)
                time.sleep(10)
            run_trial(root, protocol, "2024", arm, seed)
            run_id = f"2024_{arm}_{seed}"
            marker = root / "runs" / run_id / "completed.json"
            receipt["completed_runs"].append(dict(run_id=run_id, completed_sha256=file_hash(marker)))
            write_json(root / "worker_2024.json", receipt)
            print(f"WORKER SEALED {run_id}", flush=True)
    receipt["status"] = "complete"
    write_json(root / "worker_2024.json", receipt)


if __name__ == "__main__":
    main()
