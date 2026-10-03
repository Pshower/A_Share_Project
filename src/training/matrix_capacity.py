"""Measure one fresh CUDA PPO candidate on the local training split only."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from stable_baselines3.common.logger import configure

from src.agents.ppo_agent import make_model
from src.data.preprocess import file_hash, read_config, resolve_path
from src.training.env_factory import ResearchData


def is_cuda_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
        and "cuda" in str(error).lower()
    )


def run_probe(config_path, hidden_sizes, output, steps=512):
    config_path, output = resolve_path(config_path), resolve_path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite capacity report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        schema_version=1, status="error", seed=42, fresh_model=True,
        config_path=str(config_path), hidden_sizes=list(hidden_sizes),
        requested_steps=steps, actual_steps=0, parameters=None,
        requested_device="cuda", device=None, model_shapes=None,
        peak_allocated_bytes=None, peak_reserved_bytes=None,
        cuda_free_before_bytes=None, cuda_total_bytes=None,
        elapsed_seconds=0.0, fps=None, data_contract=None,
        validation_evaluated=False, test_evaluated=False,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    env = model = logger = device = None
    started = None
    memory_started = False
    # Reserve the report before allocating a model; never reuse another probe's output.
    with output.open("x", encoding="utf-8") as stream:
        try:
            if len(hidden_sizes) != 2 or any(type(width) is not int or width < 1 for width in hidden_sizes):
                raise ValueError("hidden_sizes must contain two positive integers")
            if type(steps) is not int or steps < 512 or steps % 512:
                raise ValueError("steps must be a positive multiple of 512")
            config = deepcopy(read_config(config_path))
            config.update(seed=42, device="cuda", hidden_sizes=list(hidden_sizes))
            if any(config["ppo"].get(key) != value for key, value in
                   [("n_steps", 512), ("batch_size", 128), ("n_epochs", 5)]):
                raise ValueError("Capacity probes require unchanged n_steps=512, batch_size=128, n_epochs=5")
            report.update(config_sha256=file_hash(config_path), ppo=deepcopy(config["ppo"]))
            if not torch.cuda.is_available():
                raise RuntimeError("Capacity probes require working CUDA")
            device = torch.device("cuda", torch.cuda.current_device())
            report.update(device=str(device), gpu=torch.cuda.get_device_name(device),
                          torch_version=str(torch.__version__), cuda_build=torch.version.cuda)
            data = ResearchData(config)
            env = data.vector_env("train")
            market = env.envs[0].unwrapped.market
            contract = deepcopy(data.contract)
            contract.pop("stock_codes")
            contract.update(stock_count=len(data.codes), feature_count=len(data.manifest["feature_cols"]),
                            train_start=str(market.dates[0].date()),
                            train_end=str(market.dates[-1].date()))
            report["data_contract"] = contract
            report["model_shapes"] = dict(
                observation={name: list(space.shape) for name, space in env.observation_space.spaces.items()},
                action=list(env.action_space.shape), hidden_sizes=list(hidden_sizes),
            )
            torch.cuda.synchronize(device)
            free, total = torch.cuda.mem_get_info(device)
            report.update(cuda_free_before_bytes=int(free), cuda_total_bytes=int(total))
            torch.cuda.reset_peak_memory_stats(device)
            memory_started = True
            model = make_model(env, config)
            if model.device.type != "cuda" or model.num_timesteps != 0:
                raise RuntimeError("Capacity probes require a fresh CUDA model")
            report["device"] = str(model.device)
            report["parameters"] = sum(parameter.numel() for parameter in model.policy.parameters())
            report["model_shapes"]["parameters"] = {
                name: list(parameter.shape) for name, parameter in model.policy.named_parameters()
            }
            logs = output.parent / f"probe_h{hidden_sizes[0]}_d{hidden_sizes[1]}" / "logs"
            report["logs"] = str(logs)
            logger = configure(str(logs), ["csv"])
            model.set_logger(logger)
            torch.cuda.synchronize(device)
            started = perf_counter()
            model.learn(total_timesteps=steps)
            torch.cuda.synchronize(device)
            report["elapsed_seconds"] = perf_counter() - started
            logger.dump(step=model.num_timesteps)
            report["status"] = "success"
        except Exception as error:
            report.update(status="oom" if is_cuda_oom(error) else "error",
                          error_type=type(error).__name__, error=str(error))
        finally:
            if started is not None and not report["elapsed_seconds"]:
                report["elapsed_seconds"] = perf_counter() - started
            if model is not None:
                report["actual_steps"] = int(model.num_timesteps)
            if report["elapsed_seconds"] > 0:
                report["fps"] = report["actual_steps"] / report["elapsed_seconds"]
            if memory_started:
                try:
                    report.update(peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                                  peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
                except Exception as error:
                    report["memory_metrics_error"] = f"{type(error).__name__}: {error}"
                    if report["status"] == "success":
                        report.update(status="error", error_type=type(error).__name__, error=str(error))
            for resource in [logger, env]:
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as error:
                        report.setdefault("cleanup_errors", []).append(f"{type(error).__name__}: {error}")
                        if report["status"] == "success":
                            report.update(status="error", error_type=type(error).__name__, error=str(error))
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hidden-sizes", nargs=2, type=int, metavar=("H", "D"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=512)
    args = parser.parse_args(argv)
    if any(width < 1 for width in args.hidden_sizes):
        parser.error("hidden sizes must be positive integers")
    if args.steps < 512 or args.steps % 512:
        parser.error("steps must be a positive multiple of 512")
    report = run_probe(args.config, args.hidden_sizes, args.output, args.steps)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report["status"] in {"success", "oom"} else 1


if __name__ == "__main__":
    sys.exit(main())
