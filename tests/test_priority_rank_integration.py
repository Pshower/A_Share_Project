"""Tiny end-to-end updates on generated quotes, never financial research data."""

import random

import numpy as np
import pytest
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.priority_agent import PriorityAgent
from src.data.preprocess import read_config
from src.training import research_priority_rank as study
from src.training.matrix_experience import EpisodeAudit
from src.training.priority_aux import build_rank_pool
from src.training.priority_trace import verify_priority_trace
from tests.test_ppo_framework import local_data
from tests.test_research_priority_rank import priority_config


@pytest.mark.parametrize("actor", ["coupled", "decoupled"])
@pytest.mark.parametrize("method", ["ppo", "rank_aux"])
def test_actual_updates_checkpoint_trace_and_inference_bridge(local_data, tmp_path, actor, method):
    previous_threads = torch.get_num_threads()
    numpy_state, python_state = np.random.get_state(), random.getstate()
    config = priority_config(local_data, actor, method)
    config["hidden_sizes"] = [16, 8]
    config["ppo"].update(n_steps=8, batch_size=4, n_epochs=1, target_kl=1.)
    first = study.PriorityData(config)
    pool = build_rank_pool(first.features, first.prices, first.codes,
        first.manifest["feature_cols"], first.manifest["train_end"],
        input_hashes=first.contract["artifacts"])
    config["priority_rank"]["pool_identity"] = {"identity_sha256": pool.identity_sha256}
    data = study.PriorityData(config)
    experience = EpisodeAudit(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    output = tmp_path / "integration"
    output.mkdir()
    callback = None
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            env.seed(config["seed"])
            model = study.make_model(env, config, pool)
            initial = {name: value.detach().clone() for name, value in model.policy.state_dict().items()}
            model.set_logger(configure(str(output / "logs"), ["csv"]))
            callback = study.PriorityCallback(data, output, experience, initial,
                dict(short=8, long=16), output / "status.json", "synthetic_only")
            model.learn(total_timesteps=16, callback=callback)
            assert callback.saved == {"short", "long"}
            assert callback.trace.statistics.violations == 0
            for budget, steps in (("short", 8), ("long", 16)):
                audit = read_config(output / (budget + "_audit.json"))
                verify_priority_trace(output / "training_trace", audit["training_trace_prefix"],
                                      actor_variant=actor, steps=steps)
                assert audit["actual_timesteps"] == steps
                assert audit["optimizer_steps_min"] == audit["optimizer_steps_max"] == steps // 4
                assert audit["auxiliary_audit"]["actual_joint_optimizer_updates"] == steps // 4
                expected = steps // 4 if method == "rank_aux" else 0
                assert audit["auxiliary_audit"]["actual_aux_updates"] == expected
                assert audit["auxiliary_audit"]["supervised_stock_date_exposures"] == expected * 1024
                assert audit["cuda_peak_allocated_bytes"] is None
                restored = study.load_agent(output / (budget + "_model"), data)
                assert restored.model.num_timesteps == steps
                inference = PriorityAgent.load(output / (budget + "_model"), data.contract, "cpu")
                with pytest.raises(ValueError, match="(?i)inference"):
                    inference.model.learn(total_timesteps=1)
            data.settings.update(seed=study.EVAL_SEED, initial_portfolio="equal", trade_top_n=20)
            result = study.evaluate_case(study.load_agent(output / "long_model", data), data,
                                         output / "validation", "synthetic_only")
            verify_priority_trace(output / "validation" / "trace", result["trace"], actor_variant=actor)
            assert result["contract"]["action_mode"] == "stock_priority"
            assert not result["test_evaluated"] and not result["actionable"]
    finally:
        env.close()
        if callback is not None:
            callback.trace.close()
        torch.set_num_threads(previous_threads)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
