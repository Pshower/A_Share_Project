"""Synthetic cold-audit checks; no real datasets, policy instances or live reads."""

from collections import OrderedDict
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import warnings
import zipfile

import numpy as np
import pytest
import torch

from tests.archive.test_timing_counterfactual import make_env
from tools.archive import audit_counterfactual_learning as audit


SHAPES = OrderedDict((
    ("mlp_extractor.value_net.0.weight", (64, 129)),
    ("mlp_extractor.value_net.0.bias", (64,)),
    ("mlp_extractor.value_net.2.weight", (64, 64)),
    ("mlp_extractor.value_net.2.bias", (64,)),
    ("action_net.weight", (2, 129)),
    ("action_net.bias", (2,)),
    ("value_net.weight", (1, 64)),
    ("value_net.bias", (1,)),
))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def tensor_bytes(value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def synthetic_policy():
    policy = OrderedDict((name, torch.full(shape, (i + 1) / 1000., dtype=torch.float32))
                         for i, (name, shape) in enumerate(SHAPES.items()))
    policy["action_net.weight"].zero_()
    policy["action_net.weight"][0, 1:20] = 2.
    policy["action_net.weight"][1, 0] = 2.
    policy["action_net.bias"].zero_()
    return policy


def synthetic_archive(*, steps=0, adam_steps=0):
    names = list(audit.runner.paired.FEATURE_NAMES)
    record = dict(policy_class="src.agents.timing_policy.CalendarTimingPolicy",
        observation_shape=[129], observation_dtype="float32", feature_names=names,
        action_names=["HOLD", "REBALANCE"], net_arch={"pi": [], "vf": [64, 64]},
        calendar_margin=2., phase_period=20, parameter_count=12805)
    ppo = dict(gamma=.995, gae_lambda=1., learning_rate=3e-4, n_steps=512,
        batch_size=128, n_epochs=5, clip_range=.2, target_kl=.03, ent_coef=0.,
        vf_coef=.5, max_grad_norm=.5)
    contract = {"synthetic_archive": True}
    algorithm = "stable_baselines3.ppo.ppo.PPO"
    data = dict(num_timesteps=steps, _n_updates=(steps // 512) * 5,
        _total_timesteps=32768, timing_registered_contract=deepcopy(contract),
        timing_policy_record=deepcopy(record), timing_registered_algorithm=algorithm,
        timing_feature_names=names, **ppo)
    metadata = dict(num_timesteps=steps, contract=contract, policy_record=record,
        algorithm_class=algorithm, actual_device="cpu", feature_names=names,
        config={"ppo": deepcopy(ppo)}, source_hashes={}, provenance={})
    policy = synthetic_policy()
    states = {}
    if adam_steps:
        states = {i: dict(step=torch.tensor(float(adam_steps)), exp_avg=torch.zeros_like(tensor),
                          exp_avg_sq=torch.full_like(tensor, .001))
                  for i, tensor in enumerate(policy.values())}
    optimizer = dict(state=states, param_groups=[dict(params=list(range(8)),
        lr=3e-4, betas=(.9, .999), eps=1e-5, weight_decay=0,
        amsgrad=False, maximize=False, foreach=None, capturable=False,
        differentiable=False, fused=None)])
    return dict(policy=policy, optimizer=optimizer, data=data, metadata=metadata)


def save_archive(directory, parts, *, duplicate=False):
    directory.mkdir()
    path = directory / "model.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zipped:
        zipped.writestr("data", json.dumps(parts["data"]))
        zipped.writestr("policy.pth", tensor_bytes(parts["policy"]))
        zipped.writestr("policy.optimizer.pth", tensor_bytes(parts["optimizer"]))
        if duplicate:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                zipped.writestr("data", json.dumps(parts["data"]))
    metadata = deepcopy(parts["metadata"])
    metadata["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(directory / "metadata.json", metadata)
    return directory


@pytest.fixture(scope="module")
def paired_case():
    env = make_env(days=90, episode_steps=40, phase=3, mode="random")
    env.step(1)
    assert env._agent_equity != env._benchmark_equity
    cutoff = str(env.env.env.market.panel.index[-1].date())
    label = audit.runner.paired.paired_rule_label(env, horizon=20, train_end=cutoff)
    record = dict(phase=label["input"]["phase"], decision_date=label["input"]["decision_date"],
        endpoint=label["labels"]["branches"]["hold"]["endpoint"],
        input_sha256=audit.runner.parent.input_hash(np.asarray(label["input"]["observation"], dtype=np.float32)),
        full_state_sha256=label["provenance"]["source_before_sha256"])
    return label, record, deepcopy(env.contract), cutoff


@pytest.fixture(scope="module")
def filling_case():
    env = make_env(days=90, episode_steps=40, phase=0, mode="cash")
    cutoff = str(env.env.env.market.panel.index[-1].date())
    label = audit.runner.paired.paired_rule_label(env, horizon=20, train_end=cutoff)
    return label, deepcopy(env.contract)


def test_pair_reconstructs_real_synthetic_books_with_divergent_source_accounts(paired_case):
    label, record, contract, cutoff = paired_case
    branch = label["labels"]["branches"]["hold"]
    assert branch["initial_agent_nav"] != branch["initial_benchmark_nav"]
    delta, costs = audit.verify_pair(label, record, contract, cutoff)
    assert delta == pytest.approx(label["labels"]["discounted_relative_delta"], abs=1e-12)
    assert set(costs) == {"hold", "rebalance"}
    agent, dates, _ = audit.native_book(branch, "agent_ledger", contract["stock_codes"], contract["broker"])
    shadow, shadow_dates, _ = audit.native_book(branch, "benchmark_ledger", contract["stock_codes"], contract["broker"])
    assert len(agent) == len(shadow) == len(dates) == 20
    assert dates == shadow_dates
    np.testing.assert_allclose(agent - shadow,
        [step["relative_log_reward"] for step in branch["steps"]], rtol=0, atol=1e-9)


def test_native_success_reconstructs_positive_fees_and_slippage(filling_case):
    label, contract = filling_case
    branch = label["labels"]["branches"]["rebalance"]
    returns, dates, costs = audit.native_book(branch, "agent_ledger",
        contract["stock_codes"], contract["broker"])
    assert costs["fees"] > 0 and costs["slippage"] > 0
    assert returns.sum() == pytest.approx(
        np.log(branch["final_agent_nav"] / branch["initial_agent_nav"]), abs=1e-12)
    assert dates[-1] == branch["endpoint"]


def test_phase_zero_pair_does_not_include_next_calendar_rebalance(filling_case):
    label, contract = filling_case
    record = dict(phase=0, decision_date=label["input"]["decision_date"],
        endpoint=label["labels"]["branches"]["hold"]["endpoint"],
        input_sha256=audit.runner.parent.input_hash(np.asarray(label["input"]["observation"], dtype=np.float32)),
        full_state_sha256=label["provenance"]["source_before_sha256"])
    audit.verify_pair(label, record, contract, label["labels"]["train_end"])
    assert all(step["macro_action"] == 0 for step in label["labels"]["branches"]["hold"]["steps"])
    assert [step["macro_action"] for step in label["labels"]["branches"]["rebalance"]["steps"]] == [1] + [0] * 19


@pytest.mark.parametrize("field", ["discounted_relative_delta", "terminal_log_nav_ratio", "undiscounted_relative_delta"])
def test_pair_rejects_forged_scalar_targets(paired_case, field):
    label, record, contract, cutoff = deepcopy(paired_case)
    label["labels"][field] += .001
    with pytest.raises(ValueError):
        audit.verify_pair(label, record, contract, cutoff)


@pytest.mark.parametrize("mutation", ["source", "input", "phase", "calendar", "budget", "early_done", "shadow"])
def test_pair_rejects_binding_boundary_or_shadow_corruption(paired_case, mutation):
    label, record, contract, cutoff = deepcopy(paired_case)
    if mutation == "source":
        label["provenance"]["source_after_sha256"] = "0" * 64
    elif mutation == "input":
        label["input"]["observation"][-1] += .1
    elif mutation == "phase":
        record["phase"] = (record["phase"] + 1) % 20
    elif mutation == "calendar":
        step = label["labels"]["branches"]["hold"]["steps"][2]
        step["macro_action"] = 1 - step["macro_action"]
    elif mutation == "budget":
        label["counters"]["branch_agent_steps"] -= 1
    elif mutation == "early_done":
        label["labels"]["branches"]["rebalance"]["steps"][4]["terminated"] = True
    else:
        label["labels"]["branches"]["rebalance"]["steps"][3]["benchmark_ledger"]["cash"] += 1.
    with pytest.raises(ValueError):
        audit.verify_pair(label, record, contract, cutoff)


@pytest.mark.parametrize("mutation", ["cash", "positions", "available", "fee", "slippage", "quantity", "continuity"])
def test_native_rejects_economic_arithmetic_corruption(filling_case, mutation):
    label, contract = deepcopy(filling_case)
    branch = label["labels"]["branches"]["rebalance"]
    book = branch["steps"][0]["agent_ledger"]
    if mutation in {"cash", "positions", "available"}:
        field = "available_positions" if mutation == "available" else mutation
        if field == "cash":
            book[field] += 1.
        else:
            book[field][0] += 100.
    elif mutation == "continuity":
        branch["steps"][1]["agent_ledger"]["decision_snapshot"]["cash"] += 1.
    else:
        order = next(order for step in branch["steps"] for order in step["agent_ledger"]["orders"]
                     if order["filled_shares"] != 0)
        field = {"fee": "fee", "slippage": "slippage_cost", "quantity": "filled_shares"}[mutation]
        order[field] += .5
    with pytest.raises(ValueError):
        audit.native_book(branch, "agent_ledger", contract["stock_codes"], contract["broker"])


def gae_case():
    values = np.asarray([[.1], [.2], [.3], [.4]], dtype=np.float32)
    dones = np.asarray([[False], [True], [False], [False]])
    starts = np.asarray([[True], [False], [True], [False]], dtype=np.float32)
    raw = np.asarray([[.01], [.02], [-.01], [.03]], dtype=np.float32)
    terminal = np.asarray([[0.], [.25], [0.], [0.]], dtype=np.float32)
    rewards = raw.copy()
    rewards[1] += np.float32(.995) * terminal[1]
    side = dict(gamma=np.asarray(.995), gae_lambda=np.asarray(1.),
        last_values=np.asarray([.7], dtype=np.float32), last_dones=dones[-1].copy(),
        dones=dones, truncated=np.asarray([[False], [True], [False], [False]]),
        terminal_values=terminal, raw_rewards=raw, native_learning_rewards=raw.astype(np.float64))
    advantages = np.zeros_like(values)
    tail = np.zeros(1, dtype=np.float32)
    for i in range(3, -1, -1):
        active = 1. - (side["last_dones"] if i == 3 else starts[i + 1])
        following = side["last_values"] if i == 3 else values[i + 1]
        tail = rewards[i] + .995 * following * active - values[i] + .995 * active * tail
        advantages[i] = tail
    buffers = dict(observations=np.zeros((4, 1, 129), dtype=np.float32),
        actions=np.zeros((4, 1, 1), dtype=np.float32), rewards=rewards, values=values,
        log_probs=np.full((4, 1), -.5, dtype=np.float32), returns=advantages + values,
        advantages=advantages, episode_starts=starts)
    return buffers, side


def test_gae_reconstructs_timeout_and_active_tail_without_predictions():
    buffers, side = gae_case()
    result = audit.verify_gae(buffers, side)
    assert result["rows"] == 4 and result["timeouts"] == 1
    assert result["maximum_advantage_error"] < 1e-6


@pytest.mark.parametrize("field", ["advantages", "returns", "rewards", "values"])
def test_gae_rejects_changed_original_buffer(field):
    buffers, side = gae_case()
    buffers[field][2, 0] += .1
    with pytest.raises(ValueError):
        audit.verify_gae(buffers, side)


def test_gae_rejects_missing_field_or_wrong_estimator():
    buffers, side = gae_case()
    buffers.pop("log_probs")
    with pytest.raises(ValueError):
        audit.verify_gae(buffers, side)
    buffers, side = gae_case()
    side["gae_lambda"] = np.asarray(.95)
    with pytest.raises(ValueError):
        audit.verify_gae(buffers, side)


@pytest.mark.parametrize("steps,adam_steps", [(0, 0), (8192, 320), (32768, 1280)])
def test_archive_reads_only_inert_eight_tensor_and_adam_state(tmp_path, steps, adam_steps):
    directory = save_archive(tmp_path / "model", synthetic_archive(steps=steps, adam_steps=adam_steps))
    reader = audit.Reader()
    result = audit.archive(directory, reader)
    assert list(result["policy"]) == list(SHAPES)
    assert sum(t.numel() for t in result["policy"].values()) == 12805
    assert result["optimizer_counts"] == ({name: adam_steps for name in SHAPES} if steps else {})
    assert result["data"]["_total_timesteps"] == 32768
    assert result["metadata"]["num_timesteps"] == steps
    assert set(reader.inputs) == {str((directory / name).resolve()) for name in ("model.zip", "metadata.json")}
    reader.stable()


@pytest.mark.parametrize("mutation", ["shape", "dtype", "order", "nan", "missing", "ids", "extra_state", "step", "bool_step", "vector_step", "zero_step", "moment", "negative_moment", "moment_dtype", "clock", "algorithm"])
def test_archive_rejects_tensor_optimizer_or_registration_corruption(tmp_path, mutation):
    parts = synthetic_archive(steps=8192, adam_steps=320)
    if mutation == "shape":
        parts["policy"][next(iter(SHAPES))] = parts["policy"][next(iter(SHAPES))].reshape(129, 64)
    elif mutation == "dtype":
        parts["policy"]["action_net.bias"] = parts["policy"]["action_net.bias"].double()
    elif mutation == "order":
        parts["policy"] = OrderedDict(reversed(list(parts["policy"].items())))
    elif mutation == "nan":
        parts["policy"]["action_net.bias"][0] = float("nan")
    elif mutation == "missing":
        parts["policy"].pop("action_net.bias")
    elif mutation == "ids":
        parts["optimizer"]["param_groups"][0]["params"][-1] = 0
    elif mutation == "extra_state":
        parts["optimizer"]["state"][99] = deepcopy(parts["optimizer"]["state"][0])
    elif mutation == "step":
        parts["optimizer"]["state"][0]["step"] = torch.tensor(320.5)
    elif mutation == "bool_step":
        parts["optimizer"]["state"][0]["step"] = torch.tensor(True)
    elif mutation == "vector_step":
        parts["optimizer"]["state"][0]["step"] = torch.tensor([320.])
    elif mutation == "zero_step":
        parts["optimizer"]["state"][0]["step"] = torch.tensor(0.)
    elif mutation == "moment":
        parts["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
    elif mutation == "negative_moment":
        parts["optimizer"]["state"][0]["exp_avg_sq"][0, 0] = -1.
    elif mutation == "moment_dtype":
        parts["optimizer"]["state"][0]["exp_avg"] = parts["optimizer"]["state"][0]["exp_avg"].double()
    elif mutation == "clock":
        parts["data"]["num_timesteps"] += 1
    else:
        parts["data"]["timing_registered_algorithm"] = "not.PPO"
    directory = save_archive(tmp_path / "model", parts)
    with pytest.raises(ValueError):
        audit.archive(directory, audit.Reader())


def test_archive_rejects_duplicate_zip_member_and_outer_hash(tmp_path):
    directory = save_archive(tmp_path / "duplicate", synthetic_archive(), duplicate=True)
    with pytest.raises(ValueError):
        audit.archive(directory, audit.Reader())
    directory = save_archive(tmp_path / "hash", synthetic_archive())
    path = directory / "model.zip"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        audit.archive(directory, audit.Reader())


def test_archive_rejects_crc_failure_even_with_updated_outer_hash(tmp_path):
    directory = save_archive(tmp_path / "crc", synthetic_archive())
    path = directory / "model.zip"
    with zipfile.ZipFile(path) as zipped:
        info = zipped.getinfo("policy.pth")
        offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    raw = bytearray(path.read_bytes())
    raw[offset + 100] ^= 1
    path.write_bytes(raw)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    metadata["model_sha256"] = hashlib.sha256(raw).hexdigest()
    write_json(directory / "metadata.json", metadata)
    with pytest.raises(ValueError):
        audit.archive(directory, audit.Reader())


def warmup_cases(method):
    theta0 = dict(policy=synthetic_policy(), metadata={"num_timesteps": 0},
                  data={"_n_updates": 0}, optimizer_counts={})
    thetaaux = deepcopy(theta0)
    fit = dict(coef_observation=np.linspace(-.1, .1, 129).tolist(), intercept_observation=.03)
    if method == "value_warmup":
        c = 8 * np.asarray(fit["coef_observation"])
        c[0] += 4.
        b = 8 * (fit["intercept_observation"] - .25)
        thetaaux["policy"]["action_net.weight"] = torch.as_tensor(np.stack([-.5 * c, .5 * c]), dtype=torch.float32)
        thetaaux["policy"]["action_net.bias"] = torch.as_tensor([-.5 * b, .5 * b], dtype=torch.float32)
    return theta0, thetaaux, fit


@pytest.mark.parametrize("method", ["ppo_only", "value_warmup"])
def test_warmup_checks_closed_form_actor_and_unchanged_critic(method):
    a, b, fit = warmup_cases(method)
    result = audit.verify_warmup(a, b, method, fit)
    assert result == dict(critic_unchanged=True, optimizer_empty=True, actual_rl_steps=0)


@pytest.mark.parametrize("mutation", ["critic", "actor", "clock", "epochs", "adam", "theta0"])
def test_warmup_rejects_hidden_learning_or_mapping_change(mutation):
    a, b, fit = warmup_cases("value_warmup")
    if mutation == "critic":
        b["policy"]["value_net.bias"][0] += 1e-5
    elif mutation == "actor":
        b["policy"]["action_net.bias"][0] += 1e-5
    elif mutation == "clock":
        b["metadata"]["num_timesteps"] = 1
    elif mutation == "epochs":
        b["data"]["_n_updates"] = 1
    elif mutation == "adam":
        b["optimizer_counts"] = {"action_net.bias": 1}
    else:
        a["policy"]["action_net.weight"][0, 0] = .1
    with pytest.raises(ValueError):
        audit.verify_warmup(a, b, "value_warmup", fit)


def trace_arrays():
    return dict(action_macro_action=np.asarray([1, 0, 0]), action_phase=np.asarray([0, 1, 2]),
        after_positions=np.zeros((3, 2)), after_cash=np.full(3, 100.), after_equity=np.full(3, 100.))


def test_changes_requires_both_macro_and_economic_evidence_separately():
    a, b = trace_arrays(), trace_arrays()
    assert audit.changes(a, b) == dict(greedy_changed=False, economic_ledger_changed=False)
    a["action_macro_action"][1] = 1
    assert audit.changes(a, b) == dict(greedy_changed=True, economic_ledger_changed=False)
    a["after_cash"][1] -= 1.
    assert audit.changes(a, b) == dict(greedy_changed=True, economic_ledger_changed=True)
    b.pop("action_macro_action")
    assert audit.changes(a, b)["greedy_changed"]
    with pytest.raises(ValueError):
        audit.changes(a)


def test_reader_binds_each_read_and_rejects_changes_before_seal(tmp_path):
    reader = audit.Reader()
    path = tmp_path / "input.json"
    write_json(path, {"a": 1})
    assert reader.json(path) == {"a": 1}
    assert len(reader.inputs) == 1
    reader.stable()
    write_json(path, {"a": 2})
    with pytest.raises(ValueError):
        reader.json(path)
    with pytest.raises(ValueError):
        reader.stable()


def test_reader_npz_and_gzip_reads_are_bound_and_detached(tmp_path):
    reader = audit.Reader()
    array_path, label_path = tmp_path / "arrays.npz", tmp_path / "label.json.gz"
    original = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.savez_compressed(array_path, observations=original)
    label_path.write_bytes(gzip.compress(json.dumps({"synthetic": [1, 2, 3]}).encode()))
    arrays, label = reader.arrays(array_path), reader.label(label_path)
    np.testing.assert_array_equal(arrays["observations"], original)
    assert label == {"synthetic": [1, 2, 3]}
    arrays["observations"][0, 0] = -1
    label["synthetic"].append(4)
    assert reader.arrays(array_path)["observations"][0, 0] == 0
    assert reader.label(label_path) == {"synthetic": [1, 2, 3]}
    assert len(reader.inputs) == 2
    reader.stable()


def test_reader_rejects_object_array_without_pickle(tmp_path):
    path = tmp_path / "unsafe.npz"
    np.savez_compressed(path, observations=np.asarray([{"future": 1}], dtype=object))
    with pytest.raises(ValueError):
        audit.Reader().arrays(path)


def test_reader_closed_rejects_unclosed_extra_missing_or_mutated_leaves(tmp_path):
    directory = tmp_path / "generation"
    directory.mkdir()
    reader, protocol = audit.Reader(), {"synthetic": True}
    with pytest.raises(ValueError):
        reader.closed(directory, protocol)
    write_json(directory / "leaf.json", {"value": 1})
    audit.runner.seal(directory, protocol, status="complete")
    result = reader.closed(directory, protocol)
    assert result["status"] == "complete" and len(reader.inputs) == 2
    with pytest.raises(ValueError):
        audit.Reader().closed(directory, {"synthetic": False})
    write_json(directory / "extra.json", {})
    with pytest.raises(ValueError):
        audit.Reader().closed(directory, protocol)
    (directory / "extra.json").unlink()
    write_json(directory / "leaf.json", {"value": 2})
    with pytest.raises(ValueError):
        audit.Reader().closed(directory, protocol)
    (directory / "leaf.json").unlink()
    with pytest.raises(ValueError):
        audit.Reader().closed(directory, protocol)


def test_audit_rejects_overlap_root_before_any_generation_reads(tmp_path):
    with pytest.raises(ValueError):
        audit.audit(audit.runner.parent.PARENT)
    with pytest.raises(ValueError):
        audit.audit(tmp_path / "wrong_root")
    assert not list(tmp_path.rglob("manifest.json"))


def test_audit_rejects_partial_destination_and_invalid_phase_without_passed_manifest(tmp_path, monkeypatch):
    root = tmp_path / "study"
    monkeypatch.setattr(audit.runner, "OUTPUT", root)
    monkeypatch.setattr(audit.runner, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        audit.audit(root, "invalid")
    destination = root / "diagnostics/cold_audit/screen"
    destination.mkdir(parents=True)
    write_json(destination / "audit.json", {"partial": True})
    with pytest.raises(ValueError):
        audit.audit(root, "screen")
    assert not (destination / "manifest.json").exists()
    assert json.loads((destination / "audit.json").read_text()) == {"partial": True}


def test_audit_missing_protocol_never_writes_success(tmp_path, monkeypatch):
    root = tmp_path / "study"
    monkeypatch.setattr(audit.runner, "OUTPUT", root)
    monkeypatch.setattr(audit.runner, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        audit.audit(root)
    assert not (root / "diagnostics").exists()


def test_audit_rejects_resolved_destination_alias_before_reading(tmp_path, monkeypatch):
    from pathlib import Path
    root = tmp_path / "study"
    destination = root / "diagnostics/cold_audit/screen"
    monkeypatch.setattr(audit.runner, "OUTPUT", root)
    monkeypatch.setattr(audit.runner, "ROOT", tmp_path)
    resolve = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda path, *a, **k: tmp_path / "redirected"
                        if path == destination else resolve(path, *a, **k))
    with pytest.raises(ValueError, match="redirect"):
        audit.audit(root)
    assert not destination.exists()


@pytest.mark.parametrize("mutation", [None, "pooled_fit", "wrong_training_rows"])
def test_ridge_cold_check_uses_only_saved_training_rows_and_normal_equations(tmp_path, mutation):
    rng = np.random.default_rng(301)
    X = rng.normal(size=(480, 129)).astype(np.float32)
    X[:, :20] = np.eye(20, dtype=np.float32)[np.arange(480) % 20]
    X[:, 128] = .25
    y = (.01 * X[:, 20] - .002 * X[:, 21] + .001 * X[:, 0]).astype(np.float64)
    hold_X = X.copy()
    hold_X[:, 20:] += 100
    hold_y = -10 * y
    training = (X, y)
    fit = audit.value_learning.fit_action_value(X, y)
    if mutation == "pooled_fit":
        fit = audit.value_learning.fit_action_value(np.vstack([X, hold_X]), np.r_[y, hold_y])
    elif mutation == "wrong_training_rows":
        changed = X.copy()
        changed[0, 20] += 1
        training = changed, y
    directory = tmp_path / "fit"
    directory.mkdir()
    write_json(directory / "fit.json", fit)
    write_json(directory / "holdout.json", dict(count=480, fit_sha256=fit["fit_sha256"]))
    protocol = {"pure_synthetic_ridge_check": True}
    audit.runner.seal(directory, protocol)
    if mutation:
        with pytest.raises(ValueError):
            audit.verify_ridge(directory, protocol, training, (hold_X, hold_y), audit.Reader())
    else:
        result, checks = audit.verify_ridge(directory, protocol, training, (hold_X, hold_y), audit.Reader())
        assert result == fit
        assert checks["maximum_normal_equation_residual"] < 1e-8
        assert checks["train_only_statistics_verified"] is True
        assert checks["fit_hashes_match_only_registered_train_arrays"] is True
        assert checks["new_model_fits"] == checks["new_predictions"] == 0
