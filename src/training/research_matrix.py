"""Paired capacity/budget PPO experiment on existing local data only."""

import argparse
from copy import deepcopy
from contextlib import contextmanager
import gc
import random
from types import SimpleNamespace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.ppo_agent import PPOAgent
from src.agents.ppo_agent import make_model
from src.data.preprocess import ROOT, read_config, resolve_path, write_json, file_hash
from src.envs.initial_account import MODES
from src.runtime import events
from .env_factory import ResearchData
from .evaluate import evaluate_agent
from .freeze import json_digest, source_hashes
from .matrix_experience import EpisodeAudit
from .train import ValidationCallback


@contextmanager
def preserve_training_rng(device):
    numpy_state, python_state = np.random.get_state(), random.getstate()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


def choose_design(probes, seeds=3, planning_seconds=8 * 3600):
    small = next((p for p in probes if p["hidden_sizes"] == [256, 128] and p["status"] == "success"), None)
    if small is None:
        raise ValueError("A successful small-model capacity probe is required")
    eligible = []
    for probe in probes:
        if probe["status"] != "success" or probe["parameters"] <= small["parameters"]:
            continue
        allowed = min(probe["cuda_free_before_bytes"] * .85, probe["cuda_total_bytes"] * .75)
        if probe["peak_reserved_bytes"] <= allowed and probe["actual_steps"] == 512 and probe["fps"] > 0:
            eligible.append(probe)
    if not eligible:
        raise ValueError("No larger candidate fits the registered memory margin")
    large = max(eligible, key=lambda p: p["parameters"])
    estimates = {steps: seeds * steps * (1 / small["fps"] + 1 / large["fps"]) * 1.25 + 1800
                 for steps in [32768, 65536, 131072]}
    feasible = [steps for steps, seconds in estimates.items() if seconds <= planning_seconds]
    steps = max(feasible) if feasible else 32768
    return dict(small_hidden=[256, 128], large_hidden=large["hidden_sizes"], long_steps=steps,
                estimated_seconds=estimates[steps], planning_seconds=planning_seconds,
                candidate_estimates={str(k): v for k, v in estimates.items()},
                memory_selection="largest successful candidate within registered margin")


def diagnostic_summary(rows):
    if not rows:
        return {}
    tail = rows[max(0, len(rows) * 3 // 4):]
    keys = {key for row in rows for key in row if key.startswith(("train/", "critic/", "optimizer/"))}
    result = {}
    for key in sorted(keys):
        values = [r[key] for r in tail if r.get(key) is not None and np.isfinite(r[key])]
        result["last/" + key] = rows[-1].get(key)
        result["tail_mean/" + key] = float(np.mean(values)) if values else None
    return result


def critic_fit(model, buffer=None):
    if buffer is None:
        buffer = model.rollout_buffer
    target = np.asarray(buffer.returns).reshape(-1).astype(float)
    previous = np.asarray(buffer.values).reshape(-1).astype(float)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(target), model.batch_size):
            obs = {name: np.asarray(values).reshape((-1, *model.observation_space[name].shape))[start:start + model.batch_size]
                   for name, values in buffer.observations.items()}
            tensor, _ = model.policy.obs_to_tensor(obs)
            predictions.append(model.policy.predict_values(tensor).cpu().numpy().reshape(-1))
    predicted = np.concatenate(predictions).astype(float)
    if not np.isfinite(predicted).all():
        raise FloatingPointError("Nonfinite post-update critic predictions")
    variance = float(np.var(target))
    mse = float(np.mean((target - predicted) ** 2))
    return {"critic/target_std": float(np.std(target)), "critic/old_value_mse": float(np.mean((target - previous) ** 2)),
            "critic/post_update_mse": mse, "critic/post_update_pred_std": float(np.std(predicted)),
            "critic/post_update_explained_variance": float(1 - np.var(target - predicted) / variance) if variance > 1e-12 else None,
            "critic/post_update_normalized_mse": mse / variance if variance > 1e-12 else None}


class MatrixCallback(ValidationCallback):
    """Production diagnostics and fixed-budget snapshots without validation selection."""

    def __init__(self, data, output, experience, initial, budgets, status_path, run_id):
        super().__init__(data, output)
        self.experience, self.initial, self.budgets = experience, initial, budgets
        self.status_path, self.run_id = status_path, run_id
        self.saved = set()
        self.critic_rollout = None

    def record_update(self):
        count = len(self.update_records)
        super().record_update()
        if len(self.update_records) == count:
            return
        self.update_records[-1].update(critic_fit(self.model, buffer=self.critic_rollout))
        pd.DataFrame(self.update_records).to_csv(self.output / "training_updates.csv", index=False)
        elapsed = perf_counter() - self.started
        write_json(self.status_path, dict(status="training", run_id=self.run_id, steps=self.model.num_timesteps,
                   long_steps=max(self.budgets.values()), elapsed_seconds=elapsed,
                   fps=self.model.num_timesteps / elapsed, diagnostics=self.update_records[-1]))
        print(f"PROGRESS {self.run_id} {self.model.num_timesteps}/{max(self.budgets.values())} elapsed={elapsed:.1f}", flush=True)

    def save_fixed_snapshot(self):
        for budget, steps in self.budgets.items():
            if budget in self.saved or self.model.num_timesteps != steps:
                continue
            agent = PPOAgent(self.model, self.data.contract)
            path = self.output / (budget + "_model")
            agent.save(path, self.data.config, dict(matrix_budget=budget, prefix_shared=True))
            # Loading a policy reseeds global RNGs; preserve the continuing rollout.
            with preserve_training_rng(self.model.device):
                restored = PPOAgent.load(path, self.data.contract, "cpu")
                for name, value in self.model.policy.state_dict().items():
                    if not torch.equal(value.detach().cpu(), restored.model.policy.state_dict()[name]):
                        raise AssertionError("Saved policy parameters changed during loading")
                observation = {name: value[0] for name, value in self.model._last_obs.items()}
                was_training = self.model.policy.training
                np.testing.assert_allclose(agent.inspect_action(observation)["latent"],
                                           restored.inspect_action(observation)["latent"], rtol=1e-5, atol=1e-6)
                self.model.policy.set_training_mode(was_training)
                del restored
            deltas = {name: float(torch.linalg.vector_norm(value.detach().cpu() - self.initial[name]))
                      for name, value in self.model.policy.state_dict().items()}
            optimizer_steps = [int(state["step"].item()) for state in self.model.policy.optimizer.state.values() if "step" in state]
            if not optimizer_steps or min(optimizer_steps) <= 0 or not all(torch.isfinite(p).all() for p in self.model.policy.parameters()):
                raise AssertionError("Snapshot must contain finite trained parameters")
            record = dict(actual_timesteps=steps, model_sha256=file_hash(path / "model.zip"),
                          parameters=sum(p.numel() for p in self.model.policy.parameters()),
                          elapsed_seconds=perf_counter() - self.started, optimizer_steps_min=min(optimizer_steps),
                          optimizer_steps_max=max(optimizer_steps), tensor_delta_l2=deltas,
                          changed_tensors=sum(value > 0 for value in deltas.values()), all_parameters_finite=True,
                          save_load_parameters_equal=True, save_load_latent_close=True,
                          experience=self.experience.stats(), training_diagnostics=diagnostic_summary(self.update_records),
                          valid_observation_steps={c: int(n) for c, n in zip(self.data.codes, self.valid_observation_steps)},
                          cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.model.device),
                          cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.model.device), test_evaluated=False)
            if record["experience"]["total_steps"] != steps:
                raise AssertionError("Experience audit disagrees with trained steps")
            write_json(self.output / (budget + "_audit.json"), record)
            write_json(self.output / (budget + "_episodes.json"), [e for e in self.experience.episodes if e["steps"]])
            self.saved.add(budget)

    def _on_rollout_start(self):
        self.record_update()
        self.save_fixed_snapshot()
        events.phase("training")
        self.rollout_started = perf_counter()

    def _on_training_end(self):
        self.record_update()
        self.save_fixed_snapshot()

    def _on_rollout_end(self):
        super()._on_rollout_end()
        # SB3 resets the live buffer before the next on_rollout_start callback.
        buffer = self.model.rollout_buffer
        self.critic_rollout = SimpleNamespace(
            returns=buffer.returns.copy(), values=buffer.values.copy(),
            observations={name: values.copy() for name, values in buffer.observations.items()},
        )


def config_for(base, root, size, widths, seed):
    cfg = deepcopy(base)
    cfg.update(seed=seed, hidden_sizes=list(widths), device="cuda", torch_threads=1,
               output_root=str(root / "runs"))
    return cfg


def verify_completed(directory, config, protocol):
    marker = directory / "completed.json"
    if not marker.exists():
        return None
    result = read_config(marker)
    if result["config_sha256"] != json_digest(config) or result["protocol_sha256"] != json_digest(protocol):
        raise ValueError("Completed matrix trial configuration changed")
    records = result["records"]
    if len(records) != 2 or {r["budget"] for r in records} != {"short", "long"}:
        raise ValueError("Completed trial must include both fixed budgets")
    required = {f"{budget}_model/{name}" for budget in ["short", "long"] for name in ["model.zip", "metadata.json"]}
    required |= {f"{budget}_audit.json" for budget in ["short", "long"]}
    if not required <= set(result["files"]):
        raise ValueError("Completed trial is missing registered snapshots")
    for name, digest in result["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or file_hash(path) != digest:
            raise ValueError("Completed matrix trial artifact changed")
    for record in records:
        budget = record["budget"]
        expected = protocol[budget + "_steps"]
        metadata = read_config(directory / (budget + "_model") / "metadata.json")
        audit = read_config(directory / (budget + "_audit.json"))
        contract = deepcopy(metadata["contract"])
        contract.pop("hidden_sizes")
        if (record["steps"] != expected or record["seed"] != config["seed"] or record["hidden_sizes"] != config["hidden_sizes"]
                or metadata["num_timesteps"] != expected or metadata["config"] != config or contract != protocol["core_contract"]
                or audit["actual_timesteps"] != expected or not audit["all_parameters_finite"]
                or record["model_sha256"] != metadata["model_sha256"] or audit["model_sha256"] != metadata["model_sha256"]):
            raise ValueError("Completed snapshot audit or identity mismatch")
    return records


def evaluate_controls(root, data, initial_path):
    path = root / "controls" / "controls.json"
    identity = dict(data=deepcopy(data.contract), seed=20260927, modes=list(MODES),
                    broker=data.settings["broker"], initial_capital=data.settings["initial_capital"])
    identity["data"].pop("hidden_sizes")
    if path.exists():
        record = read_config(path)
        if record["identity"] != identity:
            raise ValueError("Control evaluation contract changed")
        for name, digest in record["files"].items():
            artifact = (root / "controls" / name).resolve()
            if not artifact.is_relative_to((root / "controls").resolve()) or file_hash(artifact) != digest:
                raise ValueError("Control report changed")
        return record["metrics"]
    if (root / "controls").exists():
        raise ValueError("Incomplete controls require review before resuming")
    agent = PPOAgent.load(initial_path, data.contract, "cuda")
    if data.contract.get("prior") is None or any(torch.count_nonzero(head[-1].weight) or torch.count_nonzero(head[-1].bias)
                                                for head in [agent.model.policy.network.stock_head, agent.model.policy.network.cash_head]):
        raise ValueError("Shared initialization controls require zero actor heads and the same prior")
    metrics = {}
    for mode in MODES:
        data.settings.update(seed=20260927, initial_portfolio=mode)
        folder = root / "controls" / mode
        evaluate_agent(agent, data, folder, compare_baselines=True)
        metrics[mode] = read_config(folder / "metrics.json")
    files = {p.relative_to(root / "controls").as_posix(): file_hash(p) for p in (root / "controls").rglob("*") if p.is_file()}
    write_json(path, dict(identity=identity, metrics=metrics, files=files, zero_actor_prior=True))
    del agent
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def run_trial(base, root, size, widths, seed, protocol):
    cfg = config_for(base, root, size, widths, seed)
    run_id = f"{size}_{seed}"
    directory = root / "runs" / run_id
    completed = verify_completed(directory, cfg, protocol)
    if completed is not None:
        print("REUSE " + run_id, flush=True)
        return completed
    if directory.exists():
        raise ValueError("Incomplete trial requires review; refusing to overwrite " + run_id)
    directory.mkdir(parents=True)
    write_json(directory / "config.json", cfg)
    events.configure(directory / "events")
    (directory / "events").mkdir()
    data = ResearchData(cfg)
    experience = EpisodeAudit(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    env.seed(seed)
    budgets = {"short": protocol["short_steps"], "long": protocol["long_steps"]}
    print("START " + run_id, flush=True)
    model = make_model(env, cfg)
    initial = {name: value.detach().cpu().clone() for name, value in model.policy.state_dict().items()}
    PPOAgent(model, data.contract).save(directory / "initial_model", cfg)
    model.set_logger(configure(str(directory / "logs"), ["csv"]))
    callback = MatrixCallback(data, directory, experience, initial, budgets, root / "status.json", run_id)
    torch.cuda.reset_peak_memory_stats(model.device)
    try:
        model.learn(total_timesteps=budgets["long"], callback=callback)
        model.logger.dump(step=model.num_timesteps)
        if callback.saved != set(budgets):
            raise AssertionError("Both fixed-budget snapshots must be saved")
    finally:
        env.close()
    del model, callback, env, initial, experience
    gc.collect()
    torch.cuda.empty_cache()
    controls = evaluate_controls(root, data, directory / "initial_model")
    records = []
    for budget, steps in budgets.items():
        path = directory / (budget + "_model")
        agent = PPOAgent.load(path, data.contract, "cuda")
        audit = read_config(directory / (budget + "_audit.json"))
        metrics = {}
        for mode in MODES:
            data.settings.update(seed=20260927, initial_portfolio=mode)
            folder = directory / (budget + "_validation") / mode
            write_json(root / "status.json", dict(status="evaluating", run_id=run_id, budget=budget, account=mode, steps=steps))
            evaluate_agent(agent, data, folder)
            metrics[mode] = dict(ppo=read_config(folder / "metrics.json")["ppo"],
                                 initialized=controls[mode]["ppo"],
                                 baselines={name: value for name, value in controls[mode].items() if name != "ppo"})
        record = dict(size=size, budget=budget, seed=seed, steps=steps, hidden_sizes=widths,
                      parameters=audit["parameters"], elapsed_seconds=audit["elapsed_seconds"],
                      training_diagnostics=audit["training_diagnostics"], experience=audit["experience"], metrics=metrics,
                      training_updates_path=str((directory / "training_updates.csv").relative_to(root)),
                      training_run_id=run_id, prefix_shared=True, model_sha256=audit["model_sha256"],
                      model_path=str(path.relative_to(ROOT)), test_evaluated=False)
        records.append(record)
        del agent
        gc.collect()
        torch.cuda.empty_cache()
    files = {p.relative_to(directory).as_posix(): file_hash(p) for p in directory.rglob("*") if p.is_file()}
    write_json(directory / "completed.json", dict(config_sha256=json_digest(cfg), protocol_sha256=json_digest(protocol),
               files=files, records=records))
    print("FINISHED " + run_id, flush=True)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--short-steps", type=int, default=8192)
    parser.add_argument("--long-steps", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args(argv)
    root = resolve_path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    base = read_config(resolve_path(args.config))
    probes = [read_config(p) for p in root.glob("capacity_h*_d*.json")]
    design = choose_design(probes, len(args.seeds))
    long_steps = args.long_steps or design["long_steps"]
    if not 0 < args.short_steps < long_steps or args.short_steps % 512 or long_steps % 512:
        raise ValueError("Fixed budgets must be increasing positive multiples of 512")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError("Use unique nonnegative seeds")
    if not torch.cuda.is_available():
        raise RuntimeError("Matrix experiment requires working CUDA")
    data = ResearchData(config_for(base, root, "small", design["small_hidden"], args.seeds[0]))
    if data.contract.get("data_source") != "sina" or len(data.codes) != 300 or data.contract.get("prior") is None:
        raise ValueError("This registered matrix requires the existing Sina 300-stock prior model")
    core = deepcopy(data.contract)
    core.pop("hidden_sizes")
    protocol = dict(schema_version=1, design=design, seeds=args.seeds, short_steps=args.short_steps,
                    long_steps=long_steps, maximum_training_steps=2 * len(args.seeds) * long_steps,
                    base_config=base, core_contract=core, source_hashes=source_hashes(),
                    evaluation_seed=20260927, evaluation_modes=list(MODES), checkpoint_selection="fixed_budget_shared_prefix",
                    train_end=data.manifest["train_end"], val_end=data.manifest["val_end"],
                    validation_already_exposed=True, test_evaluated=False,
                    critic_diagnostic_capture="rollout_end_copy_v2",
                    gpu=torch.cuda.get_device_name(0), torch_version=str(torch.__version__))
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        if read_config(protocol_path) != protocol:
            raise ValueError("Matrix protocol changed; create a new experiment")
    else:
        write_json(protocol_path, protocol)
    print("PROTOCOL " + str(protocol_path), flush=True)
    print(design, flush=True)
    if not args.train:
        return root
    records = []
    try:
        for seed in args.seeds:
            for size in ["small", "large"]:
                records.extend(run_trial(base, root, size, design[size + "_hidden"], seed, protocol))
                write_json(root / "summary.json", records)
        from .matrix_report import write_report
        write_report(root)
        write_json(root / "status.json", dict(status="complete", checkpoints=len(records),
                   total_trained_steps=2 * len(args.seeds) * long_steps, test_evaluated=False))
    except Exception as error:
        write_json(root / "status.json", dict(status="failed", error=str(error), completed_checkpoints=len(records)))
        raise
    finally:
        events.configure(None)
    return root


if __name__ == "__main__":
    main()
