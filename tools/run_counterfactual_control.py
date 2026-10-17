"""Execute one already-registered independent control before teacher finishes.

This changes scheduling only. The frozen trial function owns the exact same
case directory, configuration, 32K budget, trace and after-update snapshots.
"""

import argparse
from pathlib import Path

from tools import research_counterfactual_learning as study


def run(fold, root=study.OUTPUT):
    study.require(fold in study.SCREEN, "Only the two predeclared development controls can run early")
    root = study.output_root(root)
    protocol = study.read_json(root / "protocol.json")
    study.check_frozen(protocol, root)
    config = study.config_for(protocol, root, fold, "ppo_only", 42, "screen")
    data = study.CausalTimingData(config)
    study.require(data.identity == protocol["data"][fold], "Early control changed the frozen causal data")
    rows = study.trial(root, protocol, data, fold, "ppo_only", 42, "screen", 32768)
    study.require(len(rows) == 2 and {row["steps"] for row in rows} == set(study.SNAPSHOTS), "Control did not seal both registered snapshots")
    study.check_frozen(protocol, root)
    directory = root / "runs" / f"screen_{fold}_ppo_only_42"
    receipt = dict(status="complete", scheduling_only=True, fold=fold, method="ppo_only", seed=42,
        protocol_sha256=study.digest(protocol), frozen_trial_runner_sha256=study.file_hash(Path(study.__file__)),
        completed_sha256=study.file_hash(directory / "completed.json"),
        launcher_sha256=study.file_hash(Path(__file__)), extra_rl_steps_beyond_registered_case=0)
    path = root / f"early_control_{fold}.json"
    if path.exists():
        study.require(study.read_json(path) == receipt, "Existing control receipt changed")
    else:
        study.write_json(path, receipt)
    print(f"EARLY_CONTROL_COMPLETE {fold} registered_rl_steps=32768 extra_rl_steps=0", flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=study.SCREEN, required=True)
    args = parser.parse_args(argv)
    return run(args.fold)


if __name__ == "__main__":
    main()
