"""Build/check PPO by default; parameter updates require --train and an explicit budget."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from src.agents.ppo_agent import PPOAgent, make_model
from src.data.preprocess import ROOT, read_config, resolve_path, write_json
from .env_factory import ResearchData
from .evaluate import evaluate_agent, selection_score


class ValidationCallback(BaseCallback):
    def __init__(self, data, output):
        super().__init__()
        self.data, self.output = data, Path(output)
        self.frequency = data.config["validation_rollouts"] * data.config["ppo"]["n_steps"]
        self.best = None
        self.best_path = None
        self.last_evaluated = None
        self.previous_equity = data.settings["initial_capital"]
        self.reason_counts = Counter()

    def evaluate(self):
        step = self.model.num_timesteps
        agent = PPOAgent(self.model, self.data.contract)
        metrics = evaluate_agent(agent, self.data, self.output / f"validation_{step}")
        self.last_evaluated = step
        score = selection_score(metrics)
        if score is not None and (self.best is None or score > self.best):
            self.best = score
            self.best_path = f"best_{step}"
            agent.save(self.output / self.best_path, self.data.config, dict(validation=metrics))
        write_json(self.output / "selection.json", dict(best_path=self.best_path,
                   score=self.best, last_evaluated=step, split="val", test_evaluated=False))

    def _on_step(self):
        for index, info in enumerate(self.locals.get("infos", [])):
            diagnostics = info.get("action_diagnostics")
            if diagnostics:
                self.logger.record_mean("portfolio/target_cash", diagnostics["cash_weight"])
                self.logger.record_mean("portfolio/action_saturation", diagnostics["saturation_fraction"])
                self.logger.record_mean("portfolio/raw_return", info["reward"])
                self.logger.record_mean("portfolio/actual_cash", info["cash"] / info["equity"] if info["equity"] else 0)
                orders = info["orders"]
                self.logger.record_mean("portfolio/fees", sum(o["fee"] for o in orders))
                self.logger.record_mean("portfolio/unfilled_orders", sum(o["status"] != "filled" for o in orders))
                gross = sum(abs(o["filled_shares"]) * o["price"] for o in orders)
                self.logger.record_mean("portfolio/gross_turnover", gross / self.previous_equity)
                for order in orders:
                    self.reason_counts.update(order["reasons"])
                for reason, count in self.reason_counts.items():
                    self.logger.record("orders_total/" + reason, count)
                self.previous_equity = (self.data.settings["initial_capital"]
                                        if self.locals["dones"][index] else info["equity"])
        return True

    def _on_rollout_start(self):
        # The previous rollout's PPO update is finished; evaluate that policy.
        if self.model.num_timesteps and self.model.num_timesteps % self.frequency == 0:
            self.evaluate()

    def _on_training_end(self):
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
            write_json(output / "budget.json", dict(requested_timesteps=args.total_timesteps,
                       rollout_size=config["ppo"]["n_steps"], note="SB3 completes full rollouts; actual timesteps may exceed the requested budget."))
            model.set_logger(configure(str(output / "logs"), ["stdout", "csv"]))
            callback = ValidationCallback(data, output)
            model.learn(total_timesteps=args.total_timesteps, callback=callback)
            agent.save(output / "final_model", config)
            if callback.best_path is not None:
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
