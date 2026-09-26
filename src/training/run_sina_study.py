"""Explicit full-pool Sina study: local completed download -> build -> train -> val."""

import argparse
from copy import deepcopy
from pathlib import Path

from src.agents.ppo_agent import PPOAgent
from src.data.preprocess import ROOT, build_from_config, file_hash, read_config, write_json
from src.data.snapshots import Snapshots
from src.envs.initial_account import MODES
from src.runtime import events
from .env_factory import ResearchData
from .evaluate import evaluate_agent
from .train import main as train_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", action="store_true", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stock_list = ROOT / "data/stock_list/hs300_20260629.txt"
    codes = stock_list.read_text(encoding="utf-8-sig").split()
    history, directory = Snapshots().get("history", args.history_id)
    if (len(codes) != 300 or len(set(codes)) != 300 or history["status"] != "ready"
            or history["request"]["provider"] != "sina" or history["request"]["codes"] != codes):
        raise ValueError("Require a complete, ordered 300-stock Sina capture")
    research = read_config(ROOT / "configs/research.json")
    research["dataset"].update(version=output.name, raw_dir=str(directory / "hfq"), raw_pattern="{code}.csv",
        clean_dir=str(ROOT / "data/clean" / output.name), stock_list_path=str(stock_list), train_codes=codes,
        volume_multiplier=1, start_date="2017-01-01", train_end="2026-03-31", val_end="2026-06-30", end_date="2026-09-25")
    research["online_source"] = dict(provider="sina", history_id=args.history_id,
        manifest_sha256=file_hash(directory / "manifest.json"), stock_list_sha256=file_hash(stock_list))
    write_json(output / "research.json", research)
    build_from_config(research)
    write_json(Path(research["dataset"]["clean_dir"]) / "research_config.json", research)
    base = read_config(ROOT / "configs/ppo.json")
    base.update(research_config=str(output / "research.json"), stock_codes=codes, device="cuda", torch_threads=1,
                hidden_sizes=[256, 128], initial_portfolio="mixed", training_episode_steps=256,
                validation_rollouts=4, output_root=str(output / "models"))
    base.pop("training_date_policy", None)
    base["ppo"].update(n_steps=512, batch_size=128, n_epochs=5, learning_rate=0.0001)
    write_json(output / "protocol.json", dict(config=base, seeds=[42, 123, 2026], timesteps=8192,
                                             test_evaluated=False, stock_list_sha256=file_hash(stock_list)))
    summary = []
    for seed in [42, 123, 2026]:
        config = dict(deepcopy(base), seed=seed)
        path = output / f"ppo_{seed}.json"
        write_json(path, config)
        data = ResearchData(config)
        if set(data.manifest["stock_codes"]) != set(codes):
            raise ValueError("Dataset silently excluded a requested stock")
        folder = output / "models" / f"seed_{seed}"
        folder.parent.mkdir(exist_ok=True)
        event_dir = output / f"events_{seed}"
        event_dir.mkdir()
        events.configure(event_dir)
        train_main(["--config", str(path), "--train", "--total-timesteps", "8192", "--run-id", f"seed_{seed}"])
        audit = read_config(folder / "training_audit.json")
        if set(audit["initial_account_steps"]) != set(MODES):
            raise AssertionError("Training did not sample all initial account modes")
        selection = read_config(folder / "selection.json")
        if not selection["best_path"]:
            raise ValueError("No validation-eligible model")
        agent = PPOAgent.load(folder / selection["best_path"], data.contract, "cuda")
        data.settings["seed"] = 20260927
        for mode in MODES:
            data.settings["initial_portfolio"] = mode
            metrics = evaluate_agent(agent, data, folder / ("scenario_" + mode), compare_baselines=True)
            summary.append(dict(seed=seed, mode=mode, selected=selection["best_path"], metrics=metrics,
                                run=str(folder), initial_account_steps=audit["initial_account_steps"]))
            write_json(output / "summary.json", summary)
        print(f"FINISHED seed={seed}", flush=True)
    events.configure(None)


if __name__ == "__main__":
    main()
