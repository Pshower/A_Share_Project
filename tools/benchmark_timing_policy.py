"""Synthetic timing-controller workload; no market data or financial RL steps."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agents.timing_policy import CalendarTimingPolicy


def measure(device, dimensions, repeats):
    torch.manual_seed(42)
    space = gym.spaces.Box(-np.inf, np.inf, (dimensions,), dtype=np.float32)
    policy = CalendarTimingPolicy(space, gym.spaces.Discrete(2), lambda _: 3e-4).to(device)
    generator = np.random.default_rng(42)

    def observations(batch):
        values = generator.normal(0, .2, (batch, dimensions)).astype(np.float32)
        values[:, :20] = 0
        values[np.arange(batch), np.arange(batch) % 20] = 1
        return torch.as_tensor(values, device=device)

    one, batch = observations(1), observations(128)
    advantages = torch.as_tensor(generator.normal(size=128).astype(np.float32), device=device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def forward():
        with torch.no_grad():
            result = policy(one, deterministic=True)
            result[0].detach().cpu().numpy()

    def update():
        _, value, log_prob = policy(batch)
        # Workload only: score-function/value loss, not a financial PPO experiment.
        loss = -(log_prob * advantages).mean() + .5 * value.square().mean()
        policy.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
        policy.optimizer.step()

    result = {"device": device, "parameters": sum(p.numel() for p in policy.parameters())}
    for name, operation in (("b1_forward_and_cpu_copy", forward), ("b128_forward_backward_adam", update)):
        for _ in range(3):
            operation()
        samples = []
        for _ in range(repeats):
            sync()
            start = time.perf_counter()
            operation()
            sync()
            samples.append((time.perf_counter() - start) * 1000)
        result[name] = {"median_ms": statistics.median(samples), "min_ms": min(samples),
                        "max_ms": max(samples), "repeats": repeats}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dimensions", type=int, default=129)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists() or args.dimensions < 20 or args.repeats < 3:
        raise ValueError("Use a new output and valid benchmark dimensions/repeats")
    torch.set_num_threads(1)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    result = dict(synthetic_only=True, financial_rl_steps=0, market_data_loaded=False,
                  additional_backtest_steps=0, torch_version=torch.__version__,
                  cuda_build=torch.version.cuda, torch_threads=1,
                  measurements=[measure(device, args.dimensions, args.repeats) for device in devices],
                  limitation="Microbenchmark excludes market simulation, two-book accounting and archive I/O.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
