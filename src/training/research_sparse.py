"""Pre-registered offline sparse/residual PPO research; no final-test evaluation."""

import argparse
from copy import deepcopy
import gc
from pathlib import Path
import subprocess
import sys

from src.agents.ppo_agent import PPOAgent
from src.data.preprocess import ROOT, FEATURES, read_config, write_json, build_from_config, file_hash
from src.envs.initial_account import MODES
from src.runtime import events
from .env_factory import ResearchData
from .evaluate import evaluate_agent
from .train import main as train_main

ARMS = {
    "D0": dict(initial_log_std=0., learning_rate=1e-4),
    "D1": dict(initial_log_std=-3., learning_rate=1e-5),
    "S1": dict(initial_log_std=-3., learning_rate=1e-5, action_top_k=20),
    "R1": dict(initial_log_std=-3., learning_rate=1e-5, action_top_k=20,
               prior_feature_index=FEATURES.index("momentum_20"), prior_strength=4., residual_scale=.25),
}


def run_trial(base, root, phase, arm, seed, steps):
    run_id = f"{phase}_{arm}_{seed}"
    config = deepcopy(base)
    options = deepcopy(ARMS[arm])
    config["ppo"]["learning_rate"] = options.pop("learning_rate")
    config.update(options, seed=seed, recent_episode_every=4, validation_rollouts=10000,
                  output_root=str(root / "runs"))
    path = root / (run_id + ".json")
    folder = root / "runs" / run_id
    if (folder / "research_result.json").exists():
        if read_config(path) != config:
            raise ValueError("Cannot resume a trial with changed configuration")
        audit = read_config(folder / "training_audit.json")
        metadata = read_config(folder / "final_model/metadata.json")
        if (audit["actual_timesteps"] != steps or not audit["all_parameters_finite"]
                or not audit["save_load_deterministic_equal"]
                or file_hash(folder / "final_model/model.zip") != metadata["model_sha256"]):
            raise ValueError("Completed trial audit or model hash mismatch")
        result = read_config(folder / "research_result.json")
        if (result["phase"], result["arm"], result["seed"], result["steps"]) != (phase, arm, seed, steps):
            raise ValueError("Completed trial identity mismatch")
        print("REUSE " + run_id, flush=True)
        return result
    if folder.exists() or (root / ("events_" + run_id)).exists():
        raise ValueError("Incomplete trial requires review; refusing to overwrite " + run_id)
    write_json(path, config)
    event_dir = root / ("events_" + run_id)
    event_dir.mkdir()
    events.configure(event_dir)
    print("START " + run_id, flush=True)
    folder = train_main(["--config", str(path), "--train", "--total-timesteps", str(steps), "--run-id", run_id])
    data = ResearchData(config)
    data.settings["seed"] = 20260927
    trained = PPOAgent.load(folder / "final_model", data.contract, "cuda")
    initialized = PPOAgent.load(folder / "initial_model", data.contract, "cuda")
    metrics, initial = {}, {}
    for mode in MODES:
        data.settings["initial_portfolio"] = mode
        metrics[mode] = evaluate_agent(trained, data, folder / ("scenario_" + mode), compare_baselines=True)
        initial[mode] = evaluate_agent(initialized, data, folder / ("initialized_" + mode))
    audit = read_config(folder / "training_audit.json")
    if set(audit["initial_account_steps"]) != set(MODES):
        raise AssertionError("Initial portfolio curriculum incomplete")
    unobserved = [code for code, count in audit["valid_observation_steps"].items() if not count]
    if phase != "walk" and unobserved:
        raise AssertionError("Stocks never observed during training: " + str(unobserved))
    report = dict(phase=phase, arm=arm, seed=seed, steps=steps, metrics=metrics, initialized=initial,
                  mean_return=sum(m["cumulative_return"] for m in metrics.values()) / len(MODES),
                  eligible=all(abs(m["max_drawdown"]) <= .1 for m in metrics.values()),
                  unobserved_training_codes=unobserved, output=str(folder), test_evaluated=False)
    write_json(folder / "research_result.json", report)
    print("FINISHED " + run_id + " mean_return=" + str(report["mean_return"]), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", action="store_true", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    base = read_config(ROOT / "reports/runs/sina_hs300_study_20260927/ppo_42.json")
    protocol = dict(arms=ARMS, screening_steps=4096, confirmation_steps=8192,
                    maximum_steps=73728, source="sina", test_evaluated=False,
                    source_manifest=file_hash(Path(read_config(base["research_config"])["dataset"]["clean_dir"]) / "manifest.json"))
    if args.resume:
        if read_config(root / "protocol.json") != protocol:
            raise ValueError("Research protocol changed; use a new experiment")
    else:
        write_json(root / "protocol.json", protocol)
    results = []
    def record(phase, arm, seed, steps, config=base):
        result = run_trial(config, root, phase, arm, seed, steps)
        results.append(result)
        write_json(root / "summary.json", results)
        gc.collect()
        return result
    screens = [record("screen", arm, 42, 4096) for arm in ARMS]
    eligible = [r for r in screens if r["eligible"]]
    if not eligible:
        write_json(root / "selection.json", dict(selected=None, reason="No risk-eligible screening result"))
        return
    selected = max(eligible, key=lambda r: r["mean_return"])["arm"]
    write_json(root / "selection.json", dict(selected=selected, rule="Four-account mean net return, all drawdowns <= 10%; final checkpoint only"))
    for seed in [42, 123, 2026]:
        record("confirm", selected, seed, 8192)
    research = read_config(base["research_config"])
    research["dataset"].update(version=root.name + "_walk", clean_dir=str(ROOT / "data/clean" / (root.name + "_walk")),
                               train_end="2025-06-30", val_end="2025-12-31")
    research_path = root / "research_walk.json"
    write_json(research_path, research)
    clean = Path(research["dataset"]["clean_dir"])
    if not (clean / "manifest.json").exists():
        if clean.exists():
            raise ValueError("Incomplete dataset must be archived before resuming")
        gc.collect()
        with (root / "walk_build.stdout.log").open("w", encoding="utf-8") as stdout, \
                (root / "walk_build.stderr.log").open("w", encoding="utf-8") as stderr:
            subprocess.run([sys.executable, "-m", "src.data.preprocess", "--config", str(research_path)],
                           cwd=ROOT, stdout=stdout, stderr=stderr, check=True)
    write_json(Path(research["dataset"]["clean_dir"]) / "research_config.json", research)
    walk = dict(deepcopy(base), research_config=str(research_path))
    for arm in dict.fromkeys(["D0", selected]):
        for seed in [42, 123]:
            record("walk", arm, seed, 8192, walk)
    events.configure(None)
    print("RESEARCH COMPLETE", flush=True)


if __name__ == "__main__":
    main()
