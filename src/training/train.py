"""Build/check PPO by default; parameter updates require --train and an explicit budget."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from src.agents.ppo_agent import PPOAgent, make_model
from src.data.preprocess import ROOT, read_config, resolve_path, write_json
from src.runtime import events
from .env_factory import ResearchData
from .evaluate import evaluate_agent, selection_score, selection_assessment


class ValidationCallback(BaseCallback):
    def __init__(self, data, output):
        super().__init__()
        self.data, self.output = data, Path(output)
        self.frequency = data.config["validation_rollouts"] * data.config["ppo"]["n_steps"]
        self.best = None
        self.best_path = None
        self.last_evaluated = None
        self.previous_equity = {}
        self.previous_target = {}
        self.portfolio_totals = Counter()
        self.portfolio_counts = Counter()
        self.initial_account_steps = Counter()
        self.portfolio_samples = 0
        self.last_recorded = None
        self.started = perf_counter()
        self.rollout_started = None
        self.rollout_ended = None
        self.rollout_seconds = 0.0
        self.validation_seconds = 0.0
        self.reason_counts = Counter()
        self.update_records = []

    def record_update(self):
        if not self.model.num_timesteps or not self.model._n_updates:
            return
        marker = (self.model.num_timesteps, self.model._n_updates)
        if marker == self.last_recorded:
            return
        self.last_recorded = marker
        values = self.model.logger.name_to_value
        row = dict(timesteps=self.model.num_timesteps, updates=self.model._n_updates)
        for key, value in values.items():
            if key.startswith("train/") and np.isscalar(value):
                row[key] = float(value) if np.isfinite(value) else None
                if row[key] is None and key != "train/explained_variance":
                    raise FloatingPointError(f"Nonfinite PPO diagnostic: {key}")
        if any(not torch.isfinite(p).all() for p in self.model.policy.parameters()):
            raise FloatingPointError("Nonfinite PPO parameters")
        steps = [int(s["step"].item()) for s in self.model.policy.optimizer.state.values() if "step" in s]
        row["optimizer/steps_min"] = min(steps, default=0)
        row["optimizer/steps_max"] = max(steps, default=0)
        norms = [p.grad.detach().float().norm().item() ** 2 for p in self.model.policy.parameters() if p.grad is not None]
        row["optimizer/last_post_clip_grad_norm"] = float(np.sqrt(sum(norms))) if norms else None
        if norms and not np.isfinite(row["optimizer/last_post_clip_grad_norm"]):
            raise FloatingPointError("Nonfinite gradients")
        row["time/elapsed_seconds"] = perf_counter() - self.started
        row["time/rollout_seconds"] = self.rollout_seconds
        row["time/update_wall_seconds"] = perf_counter() - self.rollout_ended if self.rollout_ended else None
        row["time/validation_seconds_total"] = self.validation_seconds
        if self.portfolio_samples:
            row.update({"portfolio/" + k: v / self.portfolio_counts[k] for k, v in self.portfolio_totals.items()})
        row.update({"orders_total/" + k: v for k, v in self.reason_counts.items()})
        if "train/entropy_loss" in row and row["train/entropy_loss"] is not None:
            row["train/entropy_per_dimension"] = -row["train/entropy_loss"] / (len(self.data.codes) + 1)
        row["resource/cuda_allocated_mb"] = torch.cuda.memory_allocated(self.model.device) / 2 ** 20 if self.model.device.type == "cuda" else None
        self.portfolio_totals.clear()
        self.portfolio_counts.clear()
        self.portfolio_samples = 0
        self.update_records.append(row)
        pd.DataFrame(self.update_records).to_csv(self.output / "training_updates.csv", index=False)
        events.emit("metrics", timesteps=self.model.num_timesteps, metrics=row)

    def evaluate(self):
        events.phase("validation")
        started = perf_counter()
        step = self.model.num_timesteps
        agent = PPOAgent(self.model, self.data.contract)
        metrics = evaluate_agent(agent, self.data, self.output / f"validation_{step}")
        self.last_evaluated = step
        self.validation_seconds += perf_counter() - started
        policy = self.data.config.get("selection_policy")
        assessment = selection_assessment(metrics, policy)
        score = selection_score(metrics, policy)
        if score is not None and (self.best is None or score > self.best):
            self.best = score
            self.best_path = f"best_{step}"
            agent.save(self.output / self.best_path, self.data.config, dict(validation=metrics))
        write_json(self.output / "selection.json", dict(best_path=self.best_path,
                   score=self.best, last_evaluated=step, assessment=assessment, split="val", test_evaluated=False))
        events.emit("validation", timesteps=step, metrics=metrics, assessment=assessment,
                    duration_seconds=perf_counter() - started, best_path=self.best_path)

    def _on_step(self):
        events.check_cancel()
        if self.model.num_timesteps % 16 == 0:
            events.emit("progress", phase="training", timesteps=self.model.num_timesteps)
        for index, info in enumerate(self.locals.get("infos", [])):
            self.initial_account_steps.update([info.get("initial_account", {}).get("mode", "cash")])
            diagnostics = info.get("action_diagnostics")
            if diagnostics:
                self.logger.record_mean("portfolio/target_cash", diagnostics["cash_weight"])
                self.logger.record_mean("portfolio/action_saturation", diagnostics["saturation_fraction"])
                self.logger.record_mean("portfolio/raw_return", info["reward"])
                self.logger.record_mean("portfolio/actual_cash", info["cash"] / info["equity"] if info["equity"] else 0)
                orders = info["orders"]
                self.logger.record_mean("portfolio/fees", sum(o["fee"] for o in orders))
                self.logger.record_mean("portfolio/unfilled_orders", sum(o["status"] != "filled" for o in orders))
                gross = sum(abs(o["filled_shares"]) * o["price"] for o in orders if o["filled_shares"])
                previous = self.previous_equity.get(index, self.data.settings["initial_capital"])
                self.logger.record_mean("portfolio/gross_turnover", gross / previous)
                weights = diagnostics.get("weights")
                sample = dict(target_cash=diagnostics["cash_weight"], actual_cash=info["cash"] / info["equity"] if info["equity"] else 0,
                              raw_return=info["reward"], fees=sum(o["fee"] for o in orders), gross_turnover=gross / previous,
                              unfilled_orders=sum(o["status"] != "filled" for o in orders),
                              action_saturation=diagnostics["saturation_fraction"])
                if weights is not None:
                    probabilities = np.append(weights, diagnostics["cash_weight"])
                    positive = probabilities[probabilities > 0]
                    sample.update(target_weight_entropy=float(-(positive * np.log(positive)).sum()),
                                  target_change_l1=float(np.abs(weights - self.previous_target.get(index, np.zeros_like(weights))).sum()),
                                  cap_fraction=float(np.mean(weights >= self.data.settings["broker"]["max_weight"] - 1e-7)))
                if diagnostics.get("eligible") is not None:
                    sample["eligible_stocks"] = int(diagnostics["eligible"].sum())
                self.portfolio_totals.update(sample)
                self.portfolio_counts.update(sample.keys())
                self.portfolio_samples += 1
                for order in orders:
                    self.reason_counts.update(order["reasons"])
                for reason, count in self.reason_counts.items():
                    self.logger.record("orders_total/" + reason, count)
                self.previous_equity[index] = self.data.settings["initial_capital"] if self.locals["dones"][index] else info["equity"]
                if weights is not None:
                    self.previous_target[index] = np.zeros_like(weights) if self.locals["dones"][index] else weights.copy()
        return True

    def _on_rollout_start(self):
        # The previous rollout's PPO update is finished; evaluate that policy.
        self.record_update()
        if self.model.num_timesteps and self.model.num_timesteps % self.frequency == 0:
            self.evaluate()
        events.phase("training")
        self.rollout_started = perf_counter()

    def _on_rollout_end(self):
        self.rollout_ended = perf_counter()
        self.rollout_seconds = self.rollout_ended - self.rollout_started if self.rollout_started else 0.0

    def _on_training_end(self):
        self.record_update()
        if self.last_evaluated != self.model.num_timesteps:
            self.evaluate()


def check_only(agent, data, output):
    before = {key: value.detach().clone() for key, value in agent.model.policy.state_dict().items()}
    env = data.env("train")
    try:
        obs, _ = env.reset(seed=data.config["seed"])
        diagnostics = agent.inspect_action(obs)
        predicted, _ = agent.model.predict(obs, deterministic=True)
        after_obs, reward, terminated, truncated, info = env.step(predicted)
        np.testing.assert_array_equal(diagnostics["weights"], info["action_diagnostics"]["weights"])
        agent.validate_observation(after_obs)
        agent.save(output / "initialized_model", data.config, dict(mode="check_only", trained=False))
        loaded = PPOAgent.load(output / "initialized_model", data.contract, data.config["device"])
        np.testing.assert_array_equal(agent.act(obs), loaded.act(obs))
        for key, value in agent.model.policy.state_dict().items():
            if not torch.equal(before[key], value):
                raise AssertionError("Check mode changed policy parameters")
        if agent.model.num_timesteps != 0:
            raise AssertionError("Check mode must not collect training rollouts")
        report = dict(mode="check_only", trained=False, test_evaluated=False,
                      actual_device=str(agent.model.device), cuda_build=torch.version.cuda,
                      stock_count=len(data.codes), observation_shape=list(obs["stock_features"].shape),
                      action_shape=list(agent.model.action_space.shape), parameters=sum(p.numel() for p in agent.model.policy.parameters()),
                      train_initial_date=str(env.unwrapped.market.dates[0].date()),
                      train_final_date=str(env.unwrapped.market.dates[-1].date()),
                      save_load_equal=True, parameters_unchanged=True, environment_steps=1,
                      contract=data.contract)
        write_json(output / "check.json", report)
        return report
    finally:
        env.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ppo.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--run-id")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.train and (args.total_timesteps is None or args.total_timesteps < 1):
        parser.error("--train requires --total-timesteps with a positive integer")
    if not args.train and args.total_timesteps is not None:
        parser.error("--total-timesteps requires --train; default mode is check-only")
    config = read_config(args.config)
    events.phase("preparing")
    data = ResearchData(config)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run-id must be one directory name")
    output = resolve_path(config["output_root"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "research_config.json", data.research)
    env = data.vector_env()
    try:
        model = make_model(env, config)
        agent = PPOAgent(model, data.contract)
        if not args.train:
            check_only(agent, data, output)
            print(f"CHECK ONLY: no training or test evaluation. Report: {output / 'check.json'}")
        else:
            before = {key: value.detach().clone() for key, value in model.policy.state_dict().items()}
            agent.save(output / "initial_model", config, dict(trained=False))
            write_json(output / "budget.json", dict(requested_timesteps=args.total_timesteps,
                       rollout_size=config["ppo"]["n_steps"], note="SB3 completes full rollouts; actual timesteps may exceed the requested budget."))
            model.set_logger(configure(str(output / "logs"), ["stdout", "csv"]))
            callback = ValidationCallback(data, output)
            events.phase("training", effective_budget=int(np.ceil(args.total_timesteps / config["ppo"]["n_steps"]) * config["ppo"]["n_steps"]))
            started = perf_counter()
            model.learn(total_timesteps=args.total_timesteps, callback=callback)
            elapsed = perf_counter() - started
            model.logger.dump(step=model.num_timesteps)
            events.phase("saving")
            deltas = {key: float(torch.linalg.vector_norm(value - before[key]))
                      for key, value in model.policy.state_dict().items()}
            optimizer_steps = [int(state["step"].item()) for state in model.policy.optimizer.state.values() if "step" in state]
            audit = dict(requested_timesteps=args.total_timesteps, actual_timesteps=model.num_timesteps,
                         ppo_epochs=model._n_updates, optimizer_steps_min=min(optimizer_steps, default=0),
                         optimizer_steps_max=max(optimizer_steps, default=0),
                         changed_tensors=sum(v > 0 for v in deltas.values()), tensor_count=len(deltas),
                         parameter_delta_l2=float(np.linalg.norm(list(deltas.values()))),
                         tensor_delta_l2=deltas, elapsed_seconds=elapsed,
                         all_parameters_finite=all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()),
                         test_evaluated=False, initial_account_steps=dict(callback.initial_account_steps))
            if not audit["changed_tensors"] or not audit["optimizer_steps_min"] or not audit["all_parameters_finite"]:
                raise AssertionError("Training did not perform finite parameter updates")
            write_json(output / "training_audit.json", audit)
            agent.save(output / "final_model", config)
            restored = PPOAgent.load(output / "final_model", data.contract, config["device"])
            obs = env.reset()
            left, _ = model.predict(obs, deterministic=True)
            right, _ = restored.model.predict(obs, deterministic=True)
            np.testing.assert_array_equal(left, right)
            audit["save_load_deterministic_equal"] = True
            write_json(output / "training_audit.json", audit)
            if callback.best_path is not None:
                events.phase("comparison")
                selected = PPOAgent.load(output / callback.best_path, data.contract, config["device"])
                evaluate_agent(selected, data, output / "selected_comparison", compare_baselines=True)
            else:
                evaluate_agent(agent, data, output / "final_comparison_no_valid_selection", compare_baselines=True)
            print(f"Training complete; validation only: {output}")
    finally:
        env.close()
    return output


if __name__ == "__main__":
    main()
