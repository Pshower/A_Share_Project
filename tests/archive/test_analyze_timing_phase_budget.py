"""Pure array and fake-seal diagnostics tests: no networks or market steps."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import write_json
from tools.archive import analyze_timing_phase_budget as tool
from tools.archive.audit_timing_credit import gae_targets


def metric(roi=0.):
    return dict(cumulative_return=roi, max_drawdown=-.03, total_fees=1., total_slippage_cost=0.)


def rows():
    return [dict(fold=fold, architecture=architecture, steps=steps, seed=42, phase="screen",
        method=tool.study.cell_label(architecture, steps), physical_run_id=tool.study.run_id(fold, architecture, "screen", 42),
        metrics={mode: dict(trained=metric(), initialized=metric(), benchmark=metric()) for mode in tool.study.MODES},
        changes={mode: dict(greedy_changed=False, economic_ledger_changed=False) for mode in tool.study.MODES},
        evaluation_steps={role: {mode: 59 for mode in tool.study.MODES} for role in ("initialized", "trained")},
        audit=dict(experience=dict(benchmark_steps=steps), actual_optimizer_updates=steps // 512 * 20,
            rollout_buffers=[dict(path=str(index)) for index in range(steps // 512)]),
        test_evaluated=False, actionable=False) for fold in tool.study.original.SCREEN_FOLDS for architecture, steps in tool.study.CELLS]


def test_actual_counters_do_not_add_shared_prefix_or_repeated_initial_evaluation():
    result = tool.physical_counters(rows())
    assert result["physical_training_runs"] == 4 and result["evaluation_snapshots"] == 8
    assert result["actual_agent_rl_steps"] == result["separate_training_benchmark_steps"] == 131072
    assert result["snapshot_step_sum"] == 163840 and result["shared_screen_prefix_steps"] == 32768
    assert result["actual_adam_updates"] == 5120
    assert result["original_unique_gae_buffers"] == 256 and result["snapshot_gae_buffer_views"] == 320
    assert result["initialized_agent_paths"] == 16 and result["trained_agent_paths"] == 32
    assert result["evaluation_agent_steps"] == result["evaluation_parallel_benchmark_steps"] == (16 + 32) * 59


@pytest.mark.parametrize("budget,total,buffers", [(8192, 180224, 352), (32768, 327680, 640)])
def test_conditional_confirmation_is_counted_once_at_its_selected_budget(budget, total, buffers):
    data = rows()
    for fold in tool.study.original.CONFIRM_FOLDS:
        for seed in tool.study.original.SEEDS:
            row = deepcopy(data[0])
            row.update(fold=fold, phase="confirm", seed=seed, steps=budget,
                physical_run_id=tool.study.run_id(fold, "linear", "confirm", seed))
            row["audit"] = dict(experience=dict(benchmark_steps=budget), actual_optimizer_updates=budget // 512 * 20,
                rollout_buffers=[dict(path=str(index)) for index in range(budget // 512)])
            data.append(row)
    result = tool.physical_counters(data)
    assert result["physical_training_runs"] == 10 and result["evaluation_snapshots"] == 14
    assert result["actual_agent_rl_steps"] == total and result["original_unique_gae_buffers"] == buffers
    assert result["snapshot_gae_buffer_views"] == buffers + 64


def test_paired_budget_and_actor_effects_are_ordered_and_use_identical_baselines():
    data = rows()
    for row in data:
        gain = .02 * (row["steps"] == 32768) + .01 * (row["architecture"] == "due_phase_residual")
        for mode in tool.study.MODES:
            row["metrics"][mode]["trained"]["cumulative_return"] = gain
    effects = tool.paired_effects(data)
    assert len(effects) == 32
    assert all(item["net_roi_difference"] == pytest.approx(.02) for item in effects if item["comparison"] == "same_path_long_minus_short")
    assert all(item["net_roi_difference"] == pytest.approx(.01) for item in effects if item["comparison"] == "same_budget_due_minus_linear")
    data[0]["metrics"][tool.study.MODES[0]]["benchmark"]["cumulative_return"] = .1
    with pytest.raises(ValueError):
        tool.paired_effects(data)


def test_prefix_estimator_view_reuses_the_same_original_records():
    data = rows()
    short, long = data[0], data[1]
    scales = [dict(index=index) for index in range(64)]
    assert tool.snapshot_scale_view(short, long, scales) == scales[:16]
    assert tool.snapshot_scale_view(long, long, scales) == scales
    short["audit"]["rollout_buffers"][0]["path"] = "different"
    with pytest.raises(ValueError, match="same physical path"):
        tool.snapshot_scale_view(short, long, scales)


def test_theta_comparison_maps_only_shared_tensors_and_keeps_due_vector_explicit():
    baseline = {"action_net.weight": np.array([[1., 2.]]), "value_net.weight": np.array([[3.]])}
    candidate = {"action_net.base.weight": np.array([[2., 2.]]), "value_net.weight": np.array([[3.]]),
        "action_net.residual.weight": np.array([[.3, .4]])}
    result = tool.theta_difference(candidate, baseline)
    assert result["shared_tensor_l2"]["action_net.weight"] == 1.
    assert result["shared_tensor_l2"]["value_net.weight"] == 0.
    assert result["left_due_vector_l2"] == pytest.approx(.5)
    assert result["due_vector_difference_l2"] is None
    later = deepcopy(candidate)
    later["action_net.residual.weight"] += .1
    assert tool.theta_difference(later, candidate)["due_vector_difference_l2"] == pytest.approx(np.sqrt(.02))


def test_actual_gae_arrays_and_boundaries_are_reconstructed_without_forward(tmp_path):
    size = 512
    context = dict(gamma=np.asarray(.995), gae_lambda=np.asarray(1.), last_values=np.array([.2], np.float32),
        last_dones=np.array([False]), dones=np.zeros((size, 1), bool), truncated=np.zeros((size, 1), bool),
        terminal_values=np.zeros((size, 1), np.float32), raw_rewards=np.full((size, 1), .01, np.float32),
        native_learning_rewards=np.full((size, 1), float(np.float32(.01)), np.float64))
    context["dones"][239] = True
    context["truncated"][239] = True
    context["terminal_values"][239] = .25
    rewards = context["raw_rewards"].copy()
    rewards[239] += np.float32(.995) * context["terminal_values"][239]
    starts = np.zeros((size, 1), np.float32)
    starts[0] = starts[240] = 1.
    values = np.full((size, 1), .1, np.float32)
    advantage, targets = gae_targets(rewards, values, starts, context["last_values"], context["last_dones"], .995, 1.)
    buffer = dict(rewards=rewards, values=values, episode_starts=starts, advantages=advantage, returns=targets)
    np.savez(tmp_path / "buffer.npz", **buffer)
    np.savez(tmp_path / "side.npz", **context)
    (tmp_path / "training_trace").mkdir()
    np.savez(tmp_path / "training_trace/chunk.npz", done=context["dones"].ravel(), action_learning_reward=context["native_learning_rewards"].ravel())
    pd.DataFrame([{key: float(np.std(buffer[field])) for key, field in (("credit/returns_std", "returns"),
        ("credit/advantages_std", "advantages"), ("credit/buffer_rewards_std", "rewards"))}]).to_csv(tmp_path / "updates.csv", index=False)
    record = dict(path="buffer.npz", sha256="synthetic", first_step=1, last_step=512, steps=512)
    row = dict(physical_run_id="synthetic", audit=dict(rollout_buffers=[record],
        credit_estimator_buffers=[dict(path="side.npz", sha256="synthetic")],
        training_trace_prefix=dict(chunks=[dict(path="chunk.npz")]), training_updates_path="updates.csv"))
    before = {key: value.copy() for key, value in buffer.items()}
    scales = tool.estimator_scales(tmp_path, row)
    assert scales[0]["maximum_advantage_error"] == scales[0]["maximum_return_error"] == 0.
    assert scales[0]["timeout_bootstraps"] == 1 and not scales[0]["actual_tail_done"]
    with np.load(tmp_path / "buffer.npz") as saved:
        assert all(np.array_equal(saved[key], value) for key, value in before.items())


def test_output_cannot_escape_diagnostics_or_overwrite_existing(tmp_path):
    root = tmp_path / "study"
    valid = tool.output_path(root, None, "screen")
    assert valid == root / "diagnostics/phase_budget_screen"
    for path in (root / "runs/leaf", root / "diagnostics", root / "diagnostics/../../escape"):
        with pytest.raises(ValueError):
            tool.output_path(root, path, "screen")
    valid.mkdir(parents=True)
    with pytest.raises(ValueError):
        tool.output_path(root, valid, "screen")


def test_main_waits_for_locked_selection_without_writes(tmp_path, monkeypatch):
    root = tmp_path / "unsealed"
    root.mkdir()
    write_json(root / "protocol.json", {})
    monkeypatch.setattr(tool.study, "check_frozen", lambda _: None)
    monkeypatch.setattr(tool.study, "load_stage", lambda *_: pytest.fail("No sealed reader before selection"))
    with pytest.raises(ValueError, match="eight sealed snapshots"):
        tool.main(["--root", str(root), "--stage", "all"])
    assert not (root / "diagnostics").exists()


def test_main_failed_screen_counts_unique_work_and_does_not_seek_confirmation(tmp_path, monkeypatch):
    root = tmp_path / "sealed_synthetic"
    root.mkdir()
    data = rows()
    selected = dict(selected_architecture="linear", selected_budget=8192, admitted=False)
    write_json(root / "protocol.json", {})
    write_json(root / "selection.json", selected)
    monkeypatch.setattr(tool.study, "check_frozen", lambda _: None)
    def stage(_root, _protocol, name, *_args):
        assert name == "screen"
        return data
    monkeypatch.setattr(tool.study, "load_stage", stage)
    monkeypatch.setattr(tool.study, "selection_record", lambda *_: selected)
    monkeypatch.setattr(tool, "estimator_scales", lambda _path, row: [dict(steps=512, timeout_bootstraps=0, natural_terminals=0,
        actual_tail_done=False, maximum_advantage_error=0., maximum_return_error=0.) for _ in row["audit"]["rollout_buffers"]])
    monkeypatch.setattr(tool, "analyze_snapshot", lambda _root, row, _view: ({"physical_run_id": row["physical_run_id"]}, []))
    for row in data:
        row["model_path"] = "synthetic-only"
    monkeypatch.setattr(tool, "read_theta", lambda _: {"value_net.weight": np.zeros((1, 1))})
    result = tool.main(["--root", str(root), "--stage", "all"])
    assert result["counters"]["actual_agent_rl_steps"] == 131072
    assert result["no_model_load"] and result["no_network_forwards"] and result["no_environment_steps"]
    assert (root / "diagnostics/phase_budget_all/diagnostics.json").exists()
    with pytest.raises(ValueError, match="zero confirmation"):
        tool.main(["--root", str(root), "--stage", "confirm"])
