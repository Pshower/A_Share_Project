"""Pure-array and counter tests of the new physical-path cold auditor."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from tests.archive.test_timing_credit_audit_tool import sample as estimator_sample
from tools.archive import audit_timing_phase_budget as audit


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def physical(phase="screen", budget=32768):
    snapshots = {str(budget): dict(estimator=dict(original_buffers=budget // 512, steps=budget),
        trained={mode: dict(steps=59) for mode in audit.base.MODES})}
    if phase == "screen":
        snapshots["8192"] = dict(estimator=dict(original_buffers=16, steps=8192),
            trained={mode: dict(steps=59) for mode in audit.base.MODES})
    return dict(actual_rl_steps=budget, separate_benchmark_training_steps=budget, physical_budget=budget,
        physical_final=dict(adam_updates=budget // 512 * 20), snapshots=snapshots,
        shared_prefix=dict(shared_steps=8192) if phase == "screen" else None,
        initialized={mode: dict(steps=59) for mode in audit.base.MODES})


def manifests(tmp_path):
    records = [dict(path=f"rollout_{number + 1:03d}.npz", first_step=512 * number + 1,
                    last_step=512 * (number + 1), steps=512, sha256=f"{number:064x}") for number in range(64)]
    trace = dict(identity={"synthetic_only": True}, chunks=deepcopy(records),
                 benchmark_trace=dict(identity={"synthetic_only": True}, chunks=deepcopy(records)))
    long = dict(actual_timesteps=32768, rollout_buffers=deepcopy(records),
        credit_estimator_buffers=deepcopy(records), training_trace_prefix=trace, training_updates_path="long.csv")
    short = deepcopy(long)
    short.update(actual_timesteps=8192, rollout_buffers=deepcopy(records[:16]),
        credit_estimator_buffers=deepcopy(records[:16]), training_updates_path="short.csv")
    short["training_trace_prefix"]["chunks"] = deepcopy(records[:16])
    short["training_trace_prefix"]["benchmark_trace"]["chunks"] = deepcopy(records[:16])
    updates = pd.DataFrame(dict(timesteps=np.arange(512, 32769, 512), actual_optimizer_steps=np.arange(20, 1281, 20)))
    updates.to_csv(tmp_path / "long.csv", index=False)
    updates.iloc[:16].to_csv(tmp_path / "short.csv", index=False)
    return short, long


def estimator_files(tmp_path):
    buffer, context, native = estimator_sample(1.)
    side = tmp_path / "credit_estimator_buffers/rollout_001.npz"
    side.parent.mkdir()
    np.savez_compressed(side, **context)
    rollout = tmp_path / "rollout_buffers/rollout_001.npz"
    rollout.parent.mkdir()
    np.savez_compressed(rollout, **buffer)
    trace = tmp_path / "training_trace/steps_001_004.npz"
    trace.parent.mkdir()
    np.savez_compressed(trace, **native)
    update = dict(timesteps=4, **{"credit/gamma": .995, "credit/gae_lambda": 1.,
        "credit/returns_std": float(np.std(buffer["returns"])),
        "credit/advantages_std": float(np.std(buffer["advantages"])),
        "credit/buffer_rewards_std": float(np.std(buffer["rewards"]))})
    updates = tmp_path / "snapshots/4/training_updates.csv"
    updates.parent.mkdir(parents=True)
    pd.DataFrame([update]).to_csv(updates, index=False)
    record = dict(path=side.relative_to(tmp_path).as_posix(), sha256=audit.base.sha(side), first_step=1, last_step=4, steps=4)
    evidence = dict(actual_timesteps=4, actual_model_gamma=.995, actual_buffer_gamma=.995,
        actual_model_gae_lambda=1., actual_buffer_gae_lambda=1., credit_estimator_buffers=[record],
        rollout_buffers=[dict(record, path=rollout.relative_to(tmp_path).as_posix())],
        training_trace_prefix=dict(chunks=[dict(record, path=trace.name)]),
        training_updates_path=updates.relative_to(tmp_path).as_posix(), training_updates_sha256=audit.base.sha(updates))
    return evidence, context, buffer, update


def test_four_physical_runs_count_long_steps_and_original_targets_once():
    result = audit.experience_counters({str(number): physical() for number in range(4)})
    assert result["physical_training_runs"] == 4 and result["evaluated_snapshots"] == 8
    assert result["actual_rl_steps"] == result["separate_training_benchmark_steps"] == result["unique_audited_gae_steps"] == 131072
    assert result["original_gae_buffers"] == 256 and result["actual_adam_updates"] == 5120
    assert result["snapshot_step_sum"] == 163840 and result["shared_short_prefix_steps"] == 32768
    assert result["evaluation_initialized_paths"] == 16 and result["evaluation_trained_paths"] == 32
    assert result["evaluation_agent_steps"] == 48 * 59


@pytest.mark.parametrize("budget", [8192, 32768])
def test_six_confirmation_paths_preserve_selected_budget_without_recounting_short_prefix(budget):
    trials = {"screen" + str(number): physical() for number in range(4)}
    trials.update({"confirm" + str(number): physical("confirm", budget) for number in range(6)})
    result = audit.experience_counters(trials)
    expected = 131072 + 6 * budget
    assert result["physical_training_runs"] == 10 and result["evaluated_snapshots"] == 14
    assert result["actual_rl_steps"] == result["unique_audited_gae_steps"] == expected
    assert result["original_gae_buffers"] == expected // 512
    assert result["snapshot_step_sum"] == expected + 32768
    assert result["shared_short_prefix_steps"] == 32768


def test_short_snapshot_is_an_exact_binding_of_existing_physical_arrays_and_updates(tmp_path):
    short, long = manifests(tmp_path)
    result = audit.prefix_audit(short, long, tmp_path)
    assert result["shared_steps"] == 8192 and result["shared_original_buffers"] == 16
    assert result["extra_rl_steps"] == 0 and result["counted_twice"] is False


@pytest.mark.parametrize("change", ["rollout_hash", "sidecar_clock", "agent_chunk", "benchmark_chunk", "update_row"])
def test_short_prefix_cannot_be_replaced_or_counted_as_a_different_path(tmp_path, change):
    short, long = manifests(tmp_path)
    if change == "rollout_hash":
        short["rollout_buffers"][0]["sha256"] = "f" * 64
    elif change == "sidecar_clock":
        short["credit_estimator_buffers"][0]["first_step"] = 2
    elif change == "agent_chunk":
        short["training_trace_prefix"]["chunks"][0]["path"] = "different.npz"
    elif change == "benchmark_chunk":
        short["training_trace_prefix"]["benchmark_trace"]["chunks"][0]["path"] = "different.npz"
    else:
        frame = pd.read_csv(tmp_path / "short.csv")
        frame.loc[0, "actual_optimizer_steps"] += 1
        frame.to_csv(tmp_path / "short.csv", index=False)
    with pytest.raises(ValueError):
        audit.prefix_audit(short, long, tmp_path)


def test_shared_initial_actor_and_critic_are_compared_after_mapping_new_coordinates():
    base = {"action_net.weight": torch.tensor([2.]), "critic.weight": torch.tensor([.3])}
    residual = {"action_net.base.weight": torch.tensor([2.]), "critic.weight": torch.tensor([.3]),
                "action_net.residual.weight": torch.zeros(1, 109)}
    result = audit.shared_initials({("2023", "screen", 42, "linear"): base,
                                  ("2023", "screen", 42, "due_phase_residual"): residual})
    assert result["groups"] == 1 and result["checked_physical_models"] == 2
    assert result["mapped_full_actor_and_critic_equal"] is True


@pytest.mark.parametrize("change", ["critic", "actor", "due_vector"])
def test_initial_comparison_rejects_critic_actor_or_nonzero_extra_vector(change):
    base = {"action_net.weight": torch.tensor([2.]), "critic.weight": torch.tensor([.3])}
    residual = {"action_net.base.weight": torch.tensor([2.]), "critic.weight": torch.tensor([.3]),
                "action_net.residual.weight": torch.zeros(1, 109)}
    if change == "critic":
        residual["critic.weight"] += .1
    elif change == "actor":
        residual["action_net.base.weight"] += .1
    else:
        residual["action_net.residual.weight"][0, 0] = .1
    with pytest.raises(ValueError):
        audit.shared_initials({("2023", "screen", 42, "linear"): base,
                               ("2023", "screen", 42, "due_phase_residual"): residual})


def test_parameter_growth_is_recomputed_from_saved_cpu_tensors():
    initial = {"action_net.weight": torch.zeros(2), "critic.weight": torch.zeros(2)}
    final = {"action_net.weight": torch.tensor([.3, .4]), "critic.weight": torch.tensor([.0, .2])}
    actual = {key: float(torch.linalg.vector_norm(final[key] - value)) for key, value in initial.items()}
    result = audit.parameter_changes(initial, final, {"tensor_delta_l2": actual})
    assert result == dict(actor_changed=True, critic_changed=True, changed_tensors=2)


@pytest.mark.parametrize("claim", [123., True, float("nan")])
def test_claimed_growth_cannot_replace_actual_tensor_changes(claim):
    with pytest.raises(ValueError):
        audit.parameter_changes({"critic.weight": torch.tensor([0.])}, {"critic.weight": torch.tensor([.25])},
                                {"tensor_delta_l2": {"critic.weight": claim}})


def test_actual_gaes_use_snapshot_update_prefix_not_mutable_full_run_csv(tmp_path):
    evidence, _context, _buffer, _update = estimator_files(tmp_path)
    pd.DataFrame([dict(timesteps=-1)]).to_csv(tmp_path / "training_updates.csv", index=False)
    result = audit.estimator_audit(tmp_path, evidence, {"ppo": {"gae_lambda": 1.}})
    assert result["steps"] == 4 and result["original_buffers"] == 1
    assert result["timeout_bootstraps"] == 1 and result["natural_terminals"] == 0
    assert result["active_rollout_tails"] == 1 and result["done_rollout_tails"] == 0
    assert result["diagnostics"][0]["maximum_advantage_error"] == 0.


def test_false_timeout_tail_value_rejected_after_resigning_sidecar(tmp_path):
    evidence, context, _buffer, _update = estimator_files(tmp_path)
    context["last_values"] += .1
    path = tmp_path / evidence["credit_estimator_buffers"][0]["path"]
    np.savez_compressed(path, **context)
    evidence["credit_estimator_buffers"][0]["sha256"] = audit.base.sha(path)
    with pytest.raises(ValueError, match="GAE recurrence"):
        audit.estimator_audit(tmp_path, evidence, {"ppo": {"gae_lambda": 1.}})


def test_false_actual_estimator_clock_rejected_without_any_network(tmp_path):
    evidence, _context, _buffer, _update = estimator_files(tmp_path)
    evidence["actual_buffer_gae_lambda"] = True
    with pytest.raises(ValueError, match="gamma or GAE lambda"):
        audit.estimator_audit(tmp_path, evidence, {"ppo": {"gae_lambda": 1.}})


def test_boolean_config_lambda_is_not_the_registered_numeric_estimator(tmp_path):
    evidence, _context, _buffer, _update = estimator_files(tmp_path)
    with pytest.raises(ValueError, match="gamma or GAE lambda"):
        audit.estimator_audit(tmp_path, evidence, {"ppo": {"gae_lambda": True}})


def test_wrong_snapshot_scale_log_rejected_even_if_update_file_resigned(tmp_path):
    evidence, _context, _buffer, update = estimator_files(tmp_path)
    update["credit/advantages_std"] += .1
    path = tmp_path / evidence["training_updates_path"]
    pd.DataFrame([update]).to_csv(path, index=False)
    evidence["training_updates_sha256"] = audit.base.sha(path)
    with pytest.raises(ValueError, match="estimator-scale logger"):
        audit.estimator_audit(tmp_path, evidence, {"ppo": {"gae_lambda": 1.}})


def test_registration_rejects_frozen_drift_before_reading_parent_or_data(tmp_path, monkeypatch):
    def reject(_protocol):
        raise ValueError("Frozen registration sentinel")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("No predecessor/data work before frozen guard")

    monkeypatch.setattr(audit.runner, "check_frozen", reject)
    monkeypatch.setattr(audit.runner, "reference_protocol", forbidden)
    with pytest.raises(ValueError, match="Frozen registration sentinel"):
        audit.registration(tmp_path, {})


def test_cli_refuses_to_rewrite_progress_after_final_seal(tmp_path):
    (tmp_path / "final_audit.json").write_text(json.dumps({"synthetic_only": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="never rewrite signed progress"):
        audit.main(["--root", str(tmp_path)])
