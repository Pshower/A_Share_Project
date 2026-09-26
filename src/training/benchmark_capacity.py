"""Offline CPU/CUDA capacity experiment; validation only, never final test."""

import argparse
from copy import deepcopy
import gc
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from src.agents.ppo_agent import PPOAgent, make_model
from src.data.preprocess import read_config, write_json
from .env_factory import ResearchData
from .evaluate import evaluate_agent


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_trial(config, output, steps):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    data = ResearchData(config)
    env = data.vector_env()
    try:
        # Warm kernels/allocator with a discarded model, not the measured policy.
        warm = make_model(env, config)
        warm.set_logger(configure(str(output / "warmup"), ["csv"]))
        warm.learn(total_timesteps=config["ppo"]["n_steps"])
        synchronize(warm.device)
        del warm
        gc.collect()
    finally:
        env.close()
    env = data.vector_env()
    try:
        model = make_model(env, config)
        model.set_logger(configure(str(output / "logs"), ["csv"]))
        before = {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}
        timings = {"rollout_seconds": [], "update_seconds": []}

        def timed(method, key):
            def call(*args, **kwargs):
                synchronize(model.device)
                started = perf_counter()
                result = method(*args, **kwargs)
                synchronize(model.device)
                timings[key].append(perf_counter() - started)
                return result
            return call

        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        synchronize(model.device)
        started = perf_counter()
        with patch.object(model, "collect_rollouts", timed(model.collect_rollouts, "rollout_seconds")), \
                patch.object(model, "train", timed(model.train, "update_seconds")):
            model.learn(total_timesteps=steps)
        synchronize(model.device)
        elapsed = perf_counter() - started
        peak = torch.cuda.max_memory_allocated(model.device) if model.device.type == "cuda" else None
        deltas = {k: float(torch.linalg.vector_norm(v.detach().cpu() - before[k]))
                  for k, v in model.policy.state_dict().items()}
        finite = all(bool(torch.isfinite(p).all()) for p in model.policy.parameters())
        optimizer_steps = [int(s["step"].item()) for s in model.policy.optimizer.state.values() if "step" in s]
        assert finite and any(v > 0 for v in deltas.values()) and min(optimizer_steps) > 0
        agent = PPOAgent(model, data.contract)
        agent.save(output / "final_model", config)
        restored = PPOAgent.load(output / "final_model", data.contract, config["device"])
        obs = env.reset()
        np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                      restored.model.predict(obs, deterministic=True)[0])
        synchronize(model.device)
        started = perf_counter()
        metrics = evaluate_agent(agent, data, output / "validation")
        synchronize(model.device)
        report = dict(device=str(model.device), hidden_sizes=config["hidden_sizes"], seed=config["seed"],
                      parameters=sum(p.numel() for p in model.policy.parameters()),
                      actual_timesteps=model.num_timesteps, elapsed_seconds=elapsed,
                      timings=timings, validation_seconds=perf_counter() - started,
                      optimizer_steps=min(optimizer_steps), changed_tensors=sum(v > 0 for v in deltas.values()),
                      all_parameters_finite=finite, save_load_equal=True, test_evaluated=False,
                      peak_allocated_bytes=peak, metrics=metrics)
        write_json(output / "audit.json", report)
        print(report, flush=True)
        return report
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--fixed-updates", action="store_true",
                        help="Use a nonbinding KL threshold for compute comparison only")
    parser.add_argument("--large-only", action="store_true")
    args = parser.parse_args()
    if args.steps < 512 or args.steps % 512:
        parser.error("steps must be a positive multiple of 512")
    if not torch.cuda.is_available():
        raise RuntimeError("CPU/CUDA comparison requires working CUDA")
    base = read_config(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(steps=args.steps, seeds=[42, 123],
                    widths=[[1024, 512]] if args.large_only else [[128, 64], [1024, 512]],
                    n_steps=512, batch_size=256, n_epochs=5, cpu_threads=1,
                    warmup="one discarded full rollout and PPO update per trial",
                    torch_version=torch.__version__, cuda_build=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0), test_evaluated=False,
                    fixed_updates=args.fixed_updates,
                    note="No production diagnostics callback; timing excludes validation and saving. "
                         + ("KL threshold 1e6 is for fixed-work timing, not a recommended training setting."
                            if args.fixed_updates else "KL stopping retained."))
    write_json(args.output / "protocol.json", protocol)
    reports = []
    for seed in protocol["seeds"]:
        for widths in protocol["widths"]:
            for device in (["cpu", "cuda"] if seed == 42 else ["cuda", "cpu"]):
                config = deepcopy(base)
                config.update(seed=seed, hidden_sizes=widths, device=device, torch_threads=1)
                config.pop("experiment_id", None)
                config.pop("experiment_recipe_sha256", None)
                config["ppo"].update(n_steps=512, batch_size=256, n_epochs=5)
                if args.fixed_updates:
                    config["ppo"]["target_kl"] = 1e6
                name = f"h{widths[0]}_{widths[1]}_{device}_{seed}"
                print("START " + name, flush=True)
                result = run_trial(config, args.output / name, args.steps)
                if args.fixed_updates:
                    assert result["optimizer_steps"] == args.steps // 256 * 5
                reports.append(result)
                write_json(args.output / "summary.json", reports)
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
