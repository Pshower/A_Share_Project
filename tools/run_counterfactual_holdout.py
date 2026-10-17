"""Use a completed control's CPU slot for the already-registered holdout bank."""

import argparse
from pathlib import Path

from tools import research_counterfactual_learning as study


def run(fold, root=study.OUTPUT):
    study.require(fold in study.SCREEN, "Only predeclared development holdout segments may run early")
    root = study.output_root(root)
    protocol = study.read_json(root / "protocol.json")
    study.check_frozen(protocol, root)
    control = study.closed(root / "runs" / f"screen_{fold}_ppo_only_42", protocol)
    study.require(control is not None and control["physical_rl_steps"] == 32768,
                  "Finish the registered control before reusing its CPU slot")
    data = study.CausalTimingData(study.config_for(protocol, root, fold, "ppo_only", 42, "screen"))
    study.require(data.identity == protocol["data"][fold], "Holdout preprocessing differs")
    directory = study.teacher_segment(root, protocol, data, fold, "inner_holdout")
    record = study.closed(directory, protocol)
    study.require(record is not None and record["counts"] == study.expected_teacher_counts(), "Holdout segment did not close exactly")
    study.check_frozen(protocol, root)
    receipt = dict(status="complete", scheduling_only=True, fold=fold, segment="inner_holdout",
        protocol_sha256=study.digest(protocol), completed_sha256=study.file_hash(directory / "completed.json"),
        launcher_sha256=study.file_hash(Path(__file__)), extra_steps_beyond_registered_segment=0)
    path = root / f"early_holdout_{fold}.json"
    if path.exists():
        study.require(study.read_json(path) == receipt, "Holdout scheduling receipt changed")
    else:
        study.write_json(path, receipt)
    print(f"EARLY_HOLDOUT_COMPLETE {fold} extra_steps=0", flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, choices=study.SCREEN)
    args = parser.parse_args(argv)
    return run(args.fold)


if __name__ == "__main__":
    main()
