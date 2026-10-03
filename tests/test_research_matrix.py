from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import random

from src.data.preprocess import write_json, file_hash
from src.training.freeze import json_digest
from src.training.research_matrix import choose_design, critic_fit, diagnostic_summary, verify_completed, preserve_training_rng


def probe(widths, parameters, fps, reserved=200_000_000):
    return dict(hidden_sizes=widths, parameters=parameters, fps=fps, status="success", actual_steps=512,
                peak_reserved_bytes=reserved, cuda_free_before_bytes=3_463_865_959, cuda_total_bytes=4_294_443_008)


def test_capacity_selection_rejects_paged_candidate_and_fixes_budget_before_scores():
    design = choose_design([probe([256, 128], 155525, 40), probe([1024, 512], 2391557, 33),
                            probe([2048, 1024], 9501701, 22.74, 2_589_982_720),
                            probe([4096, 2048], 37877765, 4.78, 5_251_268_608)])
    assert design["large_hidden"] == [2048, 1024]
    assert design["long_steps"] == 65536
    with pytest.raises(ValueError, match="successful small"):
        choose_design([])


def test_critic_fit_uses_same_targets_and_preserves_undefined_variance():
    class Policy:
        def obs_to_tensor(self, obs):
            return {k: torch.as_tensor(v) for k, v in obs.items()}, True
        def predict_values(self, obs):
            return obs["features"][:, 0, 0, 0].reshape(-1, 1)
    target = np.arange(4, dtype=np.float32)
    model = SimpleNamespace(batch_size=2, policy=Policy(), observation_space={"features": gym.spaces.Box(-np.inf, np.inf, (1, 1, 1))},
                            rollout_buffer=SimpleNamespace(returns=target[:, None], values=np.zeros((4, 1)),
                              observations={"features": target.reshape(4, 1, 1, 1, 1)}))
    result = critic_fit(model)
    assert result["critic/post_update_mse"] == 0 and result["critic/post_update_explained_variance"] == 1
    assert result["critic/old_value_mse"] == 3.5
    model.rollout_buffer.returns[:] = 0
    model.rollout_buffer.observations["features"][:] = 0
    assert critic_fit(model)["critic/post_update_explained_variance"] is None


def test_diagnostic_summary_reports_tail_and_does_not_turn_null_into_zero():
    rows = [{"train/value_loss": x, "train/explained_variance": None} for x in [4., 3., 2., 1.]]
    result = diagnostic_summary(rows)
    assert result["tail_mean/train/value_loss"] == 1
    assert result["tail_mean/train/explained_variance"] is None


def test_snapshot_verification_preserves_sampling_and_minibatch_randomness():
    numpy_state, python_state, torch_state = np.random.get_state(), random.getstate(), torch.get_rng_state()
    try:
        expected = (np.random.rand(3), random.random(), torch.rand(3))
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        with preserve_training_rng(torch.device("cpu")):
            np.random.seed(42)
            random.seed(42)
            torch.manual_seed(42)
            np.random.rand(100)
            torch.rand(100)
        np.testing.assert_array_equal(np.random.rand(3), expected[0])
        assert random.random() == expected[1]
        torch.testing.assert_close(torch.rand(3), expected[2])
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.set_rng_state(torch_state)


def test_completed_snapshot_reuse_requires_intact_models_and_identity(tmp_path):
    cfg = dict(seed=42, hidden_sizes=[8, 4])
    protocol = dict(short_steps=512, long_steps=1024, core_contract={"source": "fixture"})
    records = []
    for budget, steps in [("short", 512), ("long", 1024)]:
        folder = tmp_path / (budget + "_model")
        folder.mkdir()
        (folder / "model.zip").write_bytes(b"fixture")
        digest = file_hash(folder / "model.zip")
        write_json(folder / "metadata.json", dict(num_timesteps=steps, model_sha256=digest, config=cfg,
                   contract=dict(protocol["core_contract"], hidden_sizes=cfg["hidden_sizes"])))
        write_json(tmp_path / (budget + "_audit.json"), dict(actual_timesteps=steps, model_sha256=digest, all_parameters_finite=True))
        records.append(dict(budget=budget, steps=steps, seed=42, hidden_sizes=[8, 4], model_sha256=digest))
    files = {p.relative_to(tmp_path).as_posix(): file_hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    marker = dict(config_sha256=json_digest(cfg), protocol_sha256=json_digest(protocol), files=files, records=records)
    write_json(tmp_path / "completed.json", marker)
    assert verify_completed(tmp_path, cfg, protocol) == records
    wrong = deepcopy(marker)
    wrong["records"][0]["steps"] = 1024
    write_json(tmp_path / "completed.json", wrong)
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_completed(tmp_path, cfg, protocol)
    write_json(tmp_path / "completed.json", marker)
    (tmp_path / "short_model/model.zip").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_completed(tmp_path, cfg, protocol)
