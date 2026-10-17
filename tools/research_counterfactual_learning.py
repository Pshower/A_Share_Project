"""Frozen causal action-value warmup versus genuine PPO-only training."""

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.agents.timing_agent import TimingAgent
from src.agents.timing_policy import CalendarTimingPolicy
from src.backtest.run import git_metadata
from src.envs.initial_account import MODES
from src.training import research_timing as original
from src.training.research_matrix import preserve_training_rng
from tools import research_timing_counterfactual_pilot as parent
from tools import timing_counterfactual_v2 as paired
from tools import timing_action_value_learning as learning
from tools import research_timing_credit as credit
from tools.analyze_counterfactual_pilot import _native_book
from tools.research_timing_margin import case_changes
from tools.timing_causal_data import CausalTimingData


OUTPUT = ROOT / "reports/runs/ppo_counterfactual_learning_20261017"
PILOT_PROTOCOL_SHA = "14ea3e381b7ed5d6d64b83d25f019f49fe445dff17adacd0e73b9838df50e451"
PILOT_COLD_SHA = "6b8e812b3dedadf8211fc345a7215afbf9002ad2f41aac918a58970f620438a9"
METHODS = ("ppo_only", "value_warmup")
SCREEN = ("2023", "2024")
CONFIRM = ("2025", "2026")
SEEDS = (42, 123, 2026)
SNAPSHOTS = (8192, 32768)
AUDIT_ATOL = 1e-12
SEGMENTS = ("inner_train", "inner_holdout")
INVENTORY = (
    "tools/research_counterfactual_learning.py", "tests/test_counterfactual_learning_runner.py",
    "tools/timing_causal_data.py", "tests/test_timing_causal_data.py",
    "tools/counterfactual_exact_schedule.py", "tests/test_counterfactual_exact_schedule.py",
    "tools/timing_counterfactual_v2.py", "tests/test_timing_counterfactual_v2.py",
    "tools/timing_action_value_learning.py", "tests/test_timing_action_value_learning.py",
    "tools/counterfactual_storage_check.py", "tests/test_counterfactual_storage_check.py",
    "doc/ppo_counterfactual_learning_protocol_20261017.md",
)
require, read_json, canonical, digest, file_hash, write_json = (
    parent.require, parent.read_json, parent.canonical, parent.digest, parent.file_hash, parent.write_json)


def output_root(root):
    root = Path(root).resolve()
    require(root == OUTPUT.resolve() and root.is_relative_to(ROOT.resolve()), "Use the fresh registered learning root")
    return root


def registration():
    return dict(study="causal_action_value_warmup_20261017", methods=list(METHODS), screen=list(SCREEN),
        confirm=list(CONFIRM), seeds=list(SEEDS), parameters=12805, teacher_states_per_segment=480,
        teacher_episodes_per_segment=240, teacher_offsets=[0, 60], teacher_episode_steps=80,
        teacher_passes=2, teacher_horizon=20, gamma=.995, gae_lambda=1.,
        teacher_source="calendar_initialized_bernoulli_v1", source_action_salt=173701,
        teacher_book_steps_screen=384000, ppo_screen_steps=131072, ppo_shadow_screen_steps=131072,
        snapshots=list(SNAPSHOTS), physical_screen_budget=32768,
        supervised_alpha=1., normalized_calendar_deadband=.25, warmup_logit_gain=8.,
        financial_boundary_audit_atol=AUDIT_ATOL,
        offline_gate_cancels_ppo=False, test_evaluated=False, actionable=False)


def upstream():
    protocol_path = parent.OUTPUT / "protocol.json"
    require(file_hash(protocol_path) == PILOT_PROTOCOL_SHA, "Closed pilot protocol differs")
    pilot = read_json(protocol_path)
    parent.check_frozen(pilot, parent.OUTPUT)
    manifest_path = parent.OUTPUT / "diagnostics/cold_analysis/manifest.json"
    require(file_hash(manifest_path) == PILOT_COLD_SHA, "Closed pilot cold identity differs")
    manifest = read_json(manifest_path)
    for name, sha in manifest["input_files_sha256"].items():
        path = Path(name).resolve()
        require(path.is_relative_to(ROOT.resolve()) and file_hash(path) == sha, "Closed pilot input changed")
    for name, sha in manifest["outputs_sha256"].items():
        require(file_hash(manifest_path.parent / name) == sha, "Closed pilot result changed")
    return read_json(parent.PARENT / "protocol.json"), pilot


def config_for(protocol, root, fold, method="ppo_only", seed=42, phase="screen"):
    require(fold in (SCREEN if phase == "screen" else CONFIRM) and method in METHODS
            and seed in ((42,) if phase == "screen" else SEEDS), "Unregistered trial identity")
    config = deepcopy(read_json(parent.reference_config("2023")))
    config.pop("phase_budget_study", None)
    config.update(research_config=protocol["folds"][fold]["research_config"], seed=seed,
        output_root=str(output_root(root) / "runs"), training_episode_steps=240)
    config["timing"].update(fold=fold, phase=phase, reward_basis="relative_log")
    config["ppo"].update(gamma=.995, gae_lambda=1., learning_rate=3e-4, n_steps=512,
        batch_size=128, n_epochs=5, clip_range=.2, target_kl=.03, ent_coef=0., vf_coef=.5, max_grad_norm=.5)
    config["counterfactual_learning"] = dict(method=method, version="causal_linear_warmup_v1")
    return config


def check_frozen(protocol, root):
    require(protocol["output_root"] == str(output_root(root)), "Registered learning output differs")
    require(all(canonical(protocol.get(k)) == canonical(v) for k, v in registration().items()), "Learning design changed")
    require(set(protocol["inventory_sha256"]) == set(INVENTORY), "Learning inventory differs")
    for name, sha in protocol["inventory_sha256"].items():
        require(file_hash(ROOT / name) == sha, "Registered learning source or document changed: " + name)
    old, pilot = upstream()
    require(protocol["folds"] == old["folds"] and protocol["gate"] == old["gate"]
            and protocol["source_hashes"] == pilot["source_hashes"]
            and protocol["baseline_reference_root"] == old["baseline_reference_root"], "Scientific predecessor differs")


def prepare(root=OUTPUT):
    root = output_root(root)
    if (root / "protocol.json").exists():
        saved = read_json(root / "protocol.json")
        check_frozen(saved, root)
        return saved
    require(not root.exists(), "Never overwrite a partial learning study")
    old, pilot = upstream()
    protocol = dict(schema_version=1, **registration(), output_root=str(root), folds=deepcopy(old["folds"]), gate=deepcopy(old["gate"]),
        baseline_reference_root=old["baseline_reference_root"], source_hashes=deepcopy(pilot["source_hashes"]),
        inventory_sha256={name: file_hash(ROOT / name) for name in INVENTORY}, git=git_metadata(), data={})
    for fold in SCREEN:
        data = CausalTimingData(config_for(protocol, root, fold))
        protocol["data"][fold] = data.identity
    check_frozen(protocol, root)
    root.mkdir(parents=True)
    write_json(root / "protocol.json", protocol)
    write_json(root / "status.json", dict(status="prepared", actionable=False, rl_steps=0))
    return protocol


def leaves(directory):
    return parent.leaf_files(Path(directory))


def closed(directory, protocol):
    directory = Path(directory)
    marker = directory / "completed.json"
    if not marker.exists():
        return None
    result = read_json(marker)
    require(result["protocol_sha256"] == digest(protocol) and result["files"] == leaves(directory),
            "Sealed result, inventory or protocol changed: " + str(directory))
    return result


def seal(directory, protocol, **record):
    write_json(Path(directory) / "completed.json", dict(protocol_sha256=digest(protocol), files=leaves(directory), **record))


def collection(env, fold, segment, counts):
    for episode in range(240):
        _, info = env.reset(seed=42) if episode == 0 else env.reset()
        env.counterfactual_source_behavior = registration()["teacher_source"]
        env.counterfactual_source_rng = np.random.default_rng(np.random.SeedSequence([42, 173701, episode]))
        schedule = deepcopy(info["timing"]["schedule"])
        for offset in (0, 60):
            if offset:
                for _ in range(60):
                    phase = env.phase
                    probability = 1. / (1. + np.exp(-2. if phase == 0 else 2.))
                    action = int(env.counterfactual_source_rng.random() < probability)
                    before = (env.env.env._steps, env.benchmark.env._steps)
                    counts["collection_macro_attempts"] += 1
                    try:
                        _, _, terminated, truncated, _ = env.step(action)
                    finally:
                        counts["collection_agent_steps"] += env.env.env._steps - before[0]
                        counts["collection_benchmark_steps"] += env.benchmark.env._steps - before[1]
                    require(not terminated and not truncated, "Source ended before the planned offset60")
            base, observation = env.env.env, env.observe().copy()
            require(base._steps == env.benchmark.env._steps == offset and not env._done
                    and base.max_steps == 80 and base.market.current_index + 20 < len(base.market.dates), "Incomplete source horizon")
            yield dict(state_id=f"{episode:03d}_{offset:02d}", fold=fold, segment=segment, episode=episode,
                offset=offset, phase=int(env.phase), phase_bin=int(schedule["phase_bin"]),
                initial_mode=schedule["initial_mode"], date_stratum=schedule["date_stratum"],
                decision_date=str(base.market.current_date.date()),
                endpoint=str(base.market.dates[base.market.current_index + 20].date()),
                observation=observation.tolist(), input_sha256=parent.input_hash(observation),
                full_state_sha256=paired.snapshot_state_fingerprint(env)), env


def validate_plan(plan):
    states = plan["states"]
    require(plan["fold"] in (*SCREEN, *CONFIRM) and plan["segment"] in SEGMENTS
            and len(states) == 480, "Incomplete exact-phase sampling plan")
    expected = {(mode, date, phase) for mode in MODES for date in ("early", "middle", "latest") for phase in range(20)}
    first = [(row["initial_mode"], row["date_stratum"], row["phase"]) for row in states[::2]]
    require(len(set(first)) == 240 and set(first) == expected, "Missing exact phase/account/date cell")
    require(Counter(row["phase"] for row in states) == Counter({i: 24 for i in range(20)}), "Unbalanced label phases")
    for i, row in enumerate(states):
        values = np.asarray(row["observation"], dtype=np.float32)
        require(row["fold"] == plan["fold"] and row["segment"] == plan["segment"]
                and type(row["episode"]) is int and row["episode"] == i // 2
                and type(row["offset"]) is int and row["offset"] == (0, 60)[i % 2]
                and type(row["phase"]) is int and 0 <= row["phase"] < 20
                and type(row["phase_bin"]) is int and row["phase_bin"] == row["phase"] // 5
                and row["state_id"] == f"{i // 2:03d}_{(0,60)[i % 2]:02d}"
                and plan["source_start_min"] <= row["decision_date"] < row["endpoint"] <= plan["train_end"]
                and np.array_equal(values[:20], np.eye(20, dtype=np.float32)[row["phase"]])
                and row["input_sha256"] == parent.input_hash(np.asarray(row["observation"], dtype=np.float32)), "Plan input or endpoint differs")
        if i % 2:
            require(all(row[key] == states[i - 1][key] for key in ("phase", "initial_mode", "date_stratum", "episode")), "Offset changed strata")


def validate_pair(label, record, contract, cutoff):
    target, proof, count = label["labels"], label["provenance"], label["counters"]
    require(label["label_version"] == paired.LABEL_VERSION and target["horizon"] == 20 and target["gamma"] == .995
            and target["train_end"] == cutoff and label["input"]["phase"] == record["phase"]
            and label["input"]["decision_date"] == record["decision_date"]
            and parent.input_hash(np.asarray(label["input"]["observation"], dtype=np.float32)) == record["input_sha256"], "Label input/estimator differs")
    require(proof["source_before_sha256"] == proof["source_after_sha256"] == record["full_state_sha256"]
            and all(proof[key] is True for key in ("source_unchanged", "global_rng_unchanged", "benchmark_paths_equal",
                "clone_initial_state_equal", "clone_storage_independent", "current_close_observation_verified")), "Branch source proof failed")
    require(count["branch_agent_steps"] == count["branch_benchmark_steps"] == count["macro_attempts"] == 40,
            "Pair actual budget differs")
    native, shadows = {}, []
    for first, name in enumerate(("hold", "rebalance")):
        branch = target["branches"][name]
        require(branch["endpoint"] == record["endpoint"] and len(branch["steps"]) == 20, "Label endpoint differs")
        for j, step in enumerate(branch["steps"]):
            phase = (record["phase"] + j) % 20
            require(step["phase"] == phase and step["macro_action"] == (first if j == 0 else int(phase == 0))
                    and (j == 19 or not step["terminated"] and not step["truncated"]), "Rule continuation differs")
        agent = _native_book(branch, "agent_ledger", contract["stock_codes"], contract["broker"])
        shadow = _native_book(branch, "benchmark_ledger", contract["stock_codes"], contract["broker"])
        relative = agent[0] - shadow[0]
        require(np.allclose(relative, [step["relative_log_reward"] for step in branch["steps"]], rtol=0, atol=1e-10), "Saved reward differs from NAV")
        native[name] = relative
        shadows.append([step["benchmark_ledger"] for step in branch["steps"]])
    require(canonical(shadows[0]) == canonical(shadows[1]), "Independent shadow paths differ")
    delta = float(np.dot(.995 ** np.arange(20), native["rebalance"] - native["hold"]))
    require(abs(delta - target["discounted_relative_delta"]) <= 1e-10, "Native discounted target differs")
    return delta


def teacher_counts():
    return dict(collection_agent_steps=0, collection_benchmark_steps=0, collection_macro_attempts=0,
        branch_agent_steps=0, branch_benchmark_steps=0, branch_macro_attempts=0, completed_pairs=0)


def expected_teacher_counts():
    return dict(collection_agent_steps=28800, collection_benchmark_steps=28800, collection_macro_attempts=28800,
        branch_agent_steps=19200, branch_benchmark_steps=19200, branch_macro_attempts=19200, completed_pairs=480)


def teacher_segment(root, protocol, data, fold, segment):
    directory = root / "teacher" / fold / segment
    cached = closed(directory, protocol)
    if cached is not None:
        require(cached["counts"] == expected_teacher_counts() and cached["data_identity"] == data.identity, "Cached teacher identity differs")
        return directory
    require(not directory.exists(), "Never resume a partial teacher segment")
    directory.mkdir(parents=True)
    counts, env = teacher_counts(), None
    cutoff = data.boundaries[segment + "_end"]
    try:
        env = data.teacher_env(segment)
        contract = deepcopy(env.contract)
        source_start = str(env.env.env.market.dates[0].date())
        states = [record for record, _ in collection(env, fold, segment, counts)]
        env.close()
        env = None
        plan = dict(schema_version=1, protocol_sha256=digest(protocol), fold=fold, segment=segment,
            train_end=cutoff, source_start_min=source_start, data_identity=data.identity, teacher_contract=contract, states=states)
        validate_plan(plan)
        write_json(directory / "sampling_plan.json", plan)
        np.savez_compressed(directory / "inputs.npz", observations=np.asarray([row["observation"] for row in states], dtype=np.float32))
        proof = {name: file_hash(directory / name) for name in ("sampling_plan.json", "inputs.npz")}
        write_json(directory / "sampling_plan_seal.json", dict(files=proof, states=480, labels_started=False))
        proof["sampling_plan_seal.json"] = file_hash(directory / "sampling_plan_seal.json")
        (directory / "labels").mkdir()
        env = data.teacher_env(segment)
        targets = []
        for i, (record, state) in enumerate(collection(env, fold, segment, counts)):
            require(canonical(record) == canonical(states[i]) and all(file_hash(directory / k) == v for k, v in proof.items()), "Real state replay or pre-label plan changed")
            try:
                label = paired.paired_rule_label(state, horizon=20, train_end=cutoff, gamma=.995)
            except paired.CounterfactualLabelError as error:
                for target, source in (("branch_agent_steps", "branch_agent_steps"), ("branch_benchmark_steps", "branch_benchmark_steps"), ("branch_macro_attempts", "macro_attempts")):
                    counts[target] += error.counters[source]
                raise
            for target, source in (("branch_agent_steps", "branch_agent_steps"), ("branch_benchmark_steps", "branch_benchmark_steps"), ("branch_macro_attempts", "macro_attempts")):
                counts[target] += label["counters"][source]
            targets.append(validate_pair(label, record, contract, cutoff))
            (directory / "labels" / (record["state_id"] + ".json.gz")).write_bytes(gzip.compress(canonical(label), mtime=0))
            counts["completed_pairs"] += 1
            if (i + 1) % 20 == 0:
                write_json(directory / "progress.json", dict(counts=counts, completed=i + 1, total=480))
                print(f"TEACHER {fold}/{segment} {i + 1}/480", flush=True)
        env.close()
        env = None
        require(counts == expected_teacher_counts(), "Actual teacher costs differ")
        np.savez_compressed(directory / "targets.npz", delta=np.asarray(targets, dtype=np.float64))
        write_json(directory / "counters.json", counts)
        seal(directory, protocol, counts=counts, data_identity=data.identity, train_end=cutoff)
        return directory
    except Exception as error:
        write_json(directory / "failure.json", dict(status="failed_no_resume", reason=str(error), traceback=traceback.format_exc(),
            counts=counts, evidence=getattr(error, "evidence", {}), no_replacement=True))
        raise
    finally:
        if env is not None:
            env.close()


def teacher_arrays(directory, protocol):
    require(closed(directory, protocol) is not None, "Fit requires a sealed teacher segment")
    with np.load(directory / "inputs.npz", allow_pickle=False) as archive:
        X = archive["observations"].copy()
    with np.load(directory / "targets.npz", allow_pickle=False) as archive:
        y = archive["delta"].copy()
    require(X.shape == (480, 129) and X.dtype == np.float32 and np.isfinite(X).all()
            and y.shape == (480,) and y.dtype == np.float64 and np.isfinite(y).all(), "Teacher dataset schema differs")
    return X, y


def fit_fold(root, protocol, fold):
    directory = root / "fits" / fold
    sources = {segment: file_hash(root / "teacher" / fold / segment / "completed.json") for segment in SEGMENTS}
    cached = closed(directory, protocol)
    if cached is not None:
        require(cached["fold"] == fold and cached["teacher_segment_markers"] == sources, "Fit teacher identity changed")
        return read_json(directory / "fit.json")
    require(not directory.exists(), "Never overwrite a partial fit")
    X, y = teacher_arrays(root / "teacher" / fold / "inner_train", protocol)
    hold_X, hold_y = teacher_arrays(root / "teacher" / fold / "inner_holdout", protocol)
    fit = learning.fit_action_value(X, y, alpha=1.)
    diagnostics = learning.diagnostics(fit, hold_X, hold_y)
    directory.mkdir(parents=True)
    write_json(directory / "fit.json", fit)
    write_json(directory / "holdout.json", diagnostics)
    seal(directory, protocol, fold=fold, teacher_segment_markers=sources,
         model_fits=1, fit_label_rows=480, holdout_rows=480, no_ppo_steps=True)
    print(f"FIT {fold} {canonical(diagnostics).decode()}", flush=True)
    return fit


def tensor_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, dict):
        return {key: tensor_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_tree(item) for item in value]
    return value


def training_fingerprint(model, experience):
    return paired._digest(dict(environment=paired.snapshot_state_fingerprint(experience.env),
        last_obs=model._last_obs, episode_starts=model._last_episode_starts, num_timesteps=model.num_timesteps,
        n_updates=model._n_updates, total_timesteps=model._total_timesteps, progress=model._current_progress_remaining,
        parameters=tensor_tree(model.policy.state_dict()), optimizer=tensor_tree(model.policy.optimizer.state_dict())))


@contextmanager
def isolated_checkpoint(model, experience):
    before, mode = training_fingerprint(model, experience), model.policy.training
    with preserve_training_rng(model.device):
        try:
            yield
        finally:
            model.policy.set_training_mode(mode)
    require(training_fingerprint(model, experience) == before, "Saving altered the actual PPO training state")


class LearningCallback(credit.CreditCallback):
    def __init__(self, *args, protocol, **kwargs):
        super().__init__(*args, **kwargs)
        self.protocol, self.snapshots = protocol, {}

    def capture(self, steps):
        require(steps in SNAPSHOTS and steps == self.model.num_timesteps and steps not in self.snapshots, "Invalid after-update snapshot")
        model = self.model
        counts = original._optimizer_counts(model)
        names = set(dict(model.policy.named_parameters()))
        deltas = {name: float(torch.linalg.vector_norm(value.detach().cpu() - self.initial[name]))
                  for name, value in model.policy.state_dict().items()}
        require(set(counts) == names and len(names) == 8 and min(counts.values()) == max(counts.values()) > 0
                and any(value > 0 for name, value in deltas.items() if name.startswith("action_net."))
                and any(value > 0 for name, value in deltas.items() if not name.startswith("action_net."))
                and all(torch.isfinite(p).all() for p in model.policy.parameters()), "Missing finite full Actor/Critic PPO updates")
        directory = self.output / "snapshots" / str(steps)
        directory.mkdir(parents=True)
        with isolated_checkpoint(model, self.experience):
            TimingAgent(model, self.data.contract).save(directory / "model", self.data.config)
            loaded = TimingAgent.load(directory / "model", self.data.contract, "cpu")
            require(all(torch.equal(value.detach().cpu(), loaded.model.policy.state_dict()[name]) for name, value in model.policy.state_dict().items()), "Snapshot tensor roundtrip differs")
            old = TimingAgent(model, self.data.contract).inspect_action(model._last_obs[0])
            new = loaded.inspect_action(model._last_obs[0])
            require(old["macro_action"] == new["macro_action"] and np.array_equal(old["action_probabilities"], new["action_probabilities"]), "Snapshot inference roundtrip differs")
            del loaded
        audit = dict(actual_timesteps=steps, actual_optimizer_updates=max(counts.values()), optimizer_steps_by_parameter=counts,
            rollout_buffers=deepcopy(self.rollouts), credit_estimator_buffers=deepcopy(self.estimator_buffers),
            training_trace_prefix=self.trace.manifest(), experience=self.experience.stats(),
            training_schedule=self.experience.env.schedule_audit(),
            tensor_delta_l2=deltas,
            model_sha256=file_hash(directory / "model/model.zip"), checkpoint_training_state_unchanged=True)
        original.verify_rollouts(self.output, audit["rollout_buffers"], steps, audit["training_trace_prefix"])
        credit.verify_estimator_buffers(self.output, audit, self.data.config)
        write_json(directory / "training_audit.json", audit)
        self.snapshots[steps] = audit

    def _on_rollout_start(self):
        super()._on_rollout_start()
        if self.data.config["timing"]["phase"] == "screen" and self.model.num_timesteps == 8192 and self.budget > 8192:
            self.capture(8192)


def account_mean(metrics):
    return float(np.mean([metrics[mode]["cumulative_return"] for mode in MODES]))


def reference_metrics(protocol, fold):
    root = Path(protocol["baseline_reference_root"])
    return {mode: read_json(root / "rules" / fold / mode / "result.json")["metrics"] for mode in MODES}


def evaluate_model(directory, data, model_path, label, expected_benchmark):
    agent = TimingAgent.load(model_path, data.contract, "cpu")
    metrics, settings = {}, deepcopy(data.settings)
    try:
        for mode in MODES:
            result = original.evaluate_case(agent, data, directory / mode, mode, label)
            require(result["benchmark_metrics"] == expected_benchmark[mode], "New study changed the unchanged M20 baseline")
            metrics[mode] = result["metrics"]
    finally:
        data.settings.clear()
        data.settings.update(settings)
        del agent
    return metrics


def compare_cases(left, right):
    a, b = read_json(left / "result.json"), read_json(right / "result.json")
    require(a["evaluation_steps"] == b["evaluation_steps"]
            and len(a["trace"]["chunks"]) == len(b["trace"]["chunks"]), "Behavior paths have different horizons")
    macro = economic = False
    for x, y in zip(a["trace"]["chunks"], b["trace"]["chunks"]):
        with np.load(left / "trace" / x["path"], allow_pickle=False) as first, np.load(right / "trace" / y["path"], allow_pickle=False) as second:
            require(first["action_macro_action"].shape == second["action_macro_action"].shape, "Behavior trace sizes differ")
            macro |= not np.array_equal(first["action_macro_action"], second["action_macro_action"])
            economic |= any(not np.allclose(first[key], second[key], rtol=0, atol=tol) for key, tol in (
                ("after_positions", 1e-8), ("after_cash", 1e-6), ("after_equity", 1e-6)))
    return dict(greedy_changed=bool(macro), economic_ledger_changed=bool(economic))


def trial(root, protocol, data, fold, method, seed, phase, budget):
    directory = root / "runs" / f"{phase}_{fold}_{method}_{seed}"
    cached = closed(directory, protocol)
    if cached is not None:
        require(cached["fold"] == fold and cached["method"] == method and cached["seed"] == seed
                and cached["phase"] == phase and cached["physical_rl_steps"] == budget
                and cached["data_identity"] == data.identity and cached["config_sha256"] == digest(data.config), "Cached PPO identity changed")
        records = [validate_record(row, phase) for row in cached["records"]]
        require({row["steps"] for row in records} == (set(SNAPSHOTS) if phase == "screen" else {budget}), "Cached snapshots differ")
        return records
    require(not directory.exists(), "Never resume an incomplete PPO run")
    directory.mkdir(parents=True)
    config = deepcopy(data.config)
    write_json(directory / "config.json", config)
    experience = original.TimingExperience(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    env.seed(seed)
    torch.set_num_threads(1)
    model = PPO(CalendarTimingPolicy, env, seed=seed, device="cpu", verbose=0,
                policy_kwargs=dict(calendar_margin=2., net_arch={"pi": [], "vf": [64, 64]}), **config["ppo"])
    callback = None
    try:
        require(model.policy.parameter_count == 12805, "Policy capacity changed")
        with preserve_training_rng(model.device):
            TimingAgent(model, data.contract).save(directory / "theta0", config)
            other = root / "runs" / f"{phase}_{fold}_{METHODS[0]}_{seed}" / "theta0"
            if other.exists():
                require(original._policy_bytes(other) == original._policy_bytes(directory / "theta0"), "Methods did not share the same theta0")
        warmup = {"method": method, "critic_unchanged": True, "optimizer_updates": 0}
        if method == "value_warmup":
            warmup = learning.apply_actor_warmup(model, read_json(root / "fits" / fold / "fit.json"))
        write_json(directory / "warmup_audit.json", warmup)
        with preserve_training_rng(model.device):
            TimingAgent(model, data.contract).save(directory / "thetaaux", config)
            rules = reference_metrics(protocol, fold)
            initial = evaluate_model(directory / "initialized", data, directory / "theta0", "theta0", rules)
            require(initial == rules, "Calendar theta0 differs from the unchanged baseline")
            auxiliary = evaluate_model(directory / "auxiliary", data, directory / "thetaaux", "thetaaux", rules)
        initial_for_ppo = {name: value.detach().cpu().clone() for name, value in model.policy.state_dict().items()}
        callback = LearningCallback(data, directory, experience, initial_for_ppo, budget, directory.name, protocol=protocol)
        model.set_logger(configure(str(directory / "logs"), ["csv"]))
        model.learn(total_timesteps=budget, callback=callback)
        require(model.num_timesteps == budget, "PPO did not finish its full registered physical budget")
        callback.capture(budget)
        write_json(directory / "episodes.json", [episode for episode in experience.episodes if episode["steps"]])
        snapshots = deepcopy(callback.snapshots)
    except Exception as error:
        write_json(directory / "failure.json", dict(reason=str(error), traceback=traceback.format_exc(),
            actual_rl_steps=model.num_timesteps, actual_optimizer_counts=original._optimizer_counts(model), experience=experience.stats()))
        raise
    finally:
        if callback is not None:
            callback.trace.close()
        env.close()
    del callback, model, env, experience
    records = []
    for steps in sorted(snapshots):
        snapshot = directory / "snapshots" / str(steps)
        metrics = evaluate_model(snapshot / "validation", data, snapshot / "model", f"ppo_{steps}", rules)
        changes, changes_vs_aux = {}, {}
        for mode in MODES:
            result = read_json(snapshot / "validation" / mode / "result.json")
            changes[mode] = case_changes(snapshot / "validation" / mode, result)
            changes_vs_aux[mode] = compare_cases(snapshot / "validation" / mode, directory / "auxiliary" / mode)
        records.append(dict(fold=fold, phase=phase, method=method, seed=seed, steps=steps, trained=metrics,
            initialized=initial, auxiliary=auxiliary, benchmark=rules, audit=snapshots[steps],
            changes=changes, changes_vs_aux=changes_vs_aux,
            baseline_gap=account_mean(metrics) - account_mean(rules), initial_gain=account_mean(metrics) - account_mean(initial),
            ppo_gain=account_mean(metrics) - account_mean(auxiliary),
            model_path=(snapshot / "model").relative_to(ROOT).as_posix(), actionable=False))
    write_json(directory / "status.json", dict(status="complete", actual_rl_steps=budget, actionable=False))
    seal(directory, protocol, records=records, fold=fold, phase=phase, method=method, seed=seed,
        data_identity=data.identity, config_sha256=digest(data.config),
        physical_rl_steps=budget, physical_benchmark_steps=budget)
    return records


def validate_record(row, phase):
    require(row["phase"] == phase and row["fold"] in (SCREEN if phase == "screen" else CONFIRM)
            and row["method"] in METHODS and type(row["seed"]) is int
            and row["seed"] in ((42,) if phase == "screen" else SEEDS)
            and type(row["steps"]) is int and row["steps"] in SNAPSHOTS
            and row["actionable"] is False, "Record identity differs")
    for role in ("trained", "initialized", "auxiliary", "benchmark"):
        require(set(row[role]) == set(MODES), "Incomplete evaluation accounts")
        for mode in MODES:
            for key in ("cumulative_return", "max_drawdown", "total_fees", "total_slippage_cost"):
                value = row[role][mode][key]
                require(type(value) in (int, float) and np.isfinite(value), "Nonfinite economic record")
    require(row["initialized"] == row["benchmark"] and (row["method"] != "ppo_only" or row["auxiliary"] == row["initialized"]),
            "Initialization or PPO-only control changed the baseline")
    for key in ("changes", "changes_vs_aux"):
        require(set(row[key]) == set(MODES) and all(type(row[key][mode][field]) is bool
                for mode in MODES for field in ("greedy_changed", "economic_ledger_changed")), "Missing native behavior evidence")
    corrected = deepcopy(row)
    for key, role in (("baseline_gap", "benchmark"), ("initial_gain", "initialized"), ("ppo_gain", "auxiliary")):
        actual = account_mean(row["trained"]) - account_mean(row[role])
        require(type(row[key]) in (int, float) and np.isfinite(row[key]) and abs(row[key] - actual) <= AUDIT_ATOL, "Stored learning gap differs from actual metrics")
        corrected[key] = actual
    return corrected


def load_rows(root, protocol, phase):
    rows = []
    for directory in sorted((root / "runs").glob(phase + "_*")):
        record = closed(directory, protocol)
        require(record is not None, "A physical PPO run is incomplete")
        rows.extend(record["records"])
    return rows


def select_screen(rows):
    rows = [validate_record(row, "screen") for row in rows]
    expected = {(fold, method, steps, 42) for fold in SCREEN for method in METHODS for steps in SNAPSHOTS}
    require({(r["fold"], r["method"], r["steps"], r["seed"]) for r in rows} == expected and len(rows) == 8, "Screen needs all four full physical runs and eight snapshots")
    candidates = []
    for method in METHODS:
        for steps in SNAPSHOTS:
            group = [row for row in rows if row["method"] == method and row["steps"] == steps]
            candidates.append(dict(method=method, steps=steps, minimum_gap=min(row["baseline_gap"] for row in group),
                mean_gap=float(np.mean([row["baseline_gap"] for row in group])),
                worst_drawdown=max(abs(row["trained"][mode]["max_drawdown"]) for row in group for mode in MODES),
                mean_cost=float(np.mean([row["trained"][mode]["total_fees"] + row["trained"][mode]["total_slippage_cost"] for row in group for mode in MODES])),
                eligible=all(row["baseline_gap"] >= .0001 - AUDIT_ATOL and row["initial_gain"] > 0 and row["ppo_gain"] > 0 for row in group)
                    and any(all(flags.values()) for row in group for flags in row["changes"].values())
                    and any(all(flags.values()) for row in group for flags in row["changes_vs_aux"].values())))
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    choice = min(eligible, key=lambda c: (-c["minimum_gap"], -c["mean_gap"], c["worst_drawdown"], c["mean_cost"], METHODS.index(c["method"]), c["steps"])) if eligible else None
    return dict(schema_version=1, status="screen_admitted" if choice else "screen_failed", admitted=choice is not None,
        selected=choice, candidates=candidates, physical_rl_steps=131072, physical_benchmark_steps=131072,
        confirmation_rl_steps_authorized=12 * choice["steps"] if choice else 0, no_confirmation_fallback=True)


def confirmation_summary(rows, selection, gate):
    chosen = selection["selected"]
    budget = chosen["steps"]
    rows = [validate_record(row, "confirm") for row in rows]
    actual = {(r["fold"], r["method"], r["seed"], r["steps"]) for r in rows}
    expected = {(f, m, s, budget) for f in CONFIRM for m in METHODS for s in SEEDS}
    require(len(rows) == len(actual) == len(expected) and actual == expected, "Confirmation requires only the selected budget, both methods and unique predeclared seeds")
    selected = [row for row in rows if row["method"] == chosen["method"] and row["steps"] == budget]
    per_fold = {}
    for fold in CONFIRM:
        group = [row for row in selected if row["fold"] == fold]
        per_fold[fold] = dict(mean_gap=float(np.mean([row["baseline_gap"] for row in group])),
            mean_initial_gain=float(np.mean([row["initial_gain"] for row in group])), mean_ppo_gain=float(np.mean([row["ppo_gain"] for row in group])),
            positive_gap_seeds=sum(row["baseline_gap"] > 0 for row in group), positive_ppo_seeds=sum(row["ppo_gain"] > 0 for row in group))
    mean_gap = float(np.mean([row["baseline_gap"] for row in selected]))
    worst_dd = max(abs(row["trained"][mode]["max_drawdown"]) for row in selected for mode in MODES)
    dd_excess = max(abs(row["trained"][mode]["max_drawdown"]) - abs(row["benchmark"][mode]["max_drawdown"]) for row in selected for mode in MODES)
    qualified = (mean_gap >= gate["across_fold_mean_baseline_gap_min"] - AUDIT_ATOL and worst_dd <= gate["max_all_account_drawdown"] + AUDIT_ATOL
        and dd_excess <= gate["max_drawdown_excess_over_baseline"] + AUDIT_ATOL
        and all(v["mean_gap"] > 0 and v["mean_initial_gain"] > 0 and v["mean_ppo_gain"] > 0
                and v["positive_gap_seeds"] >= gate["each_fold_positive_gap_seeds"] and v["positive_ppo_seeds"] >= 2 for v in per_fold.values()))
    return dict(qualified=bool(qualified), per_fold=per_fold, mean_baseline_gap=mean_gap,
        worst_account_drawdown=worst_dd, max_drawdown_excess=dd_excess,
        selected=chosen, candidate=chosen if qualified else None, actionable=False)


def worker(root, protocol, phase, fold):
    check_frozen(protocol, root)
    budget = 32768
    if phase == "confirm":
        selection = read_json(root / "selection.json")
        require(selection["admitted"] is True and selection == select_screen(load_rows(root, protocol, "screen")), "Confirmation lacks the exact frozen development selection")
        selection_proof = read_json(root / "selection_proof.json")
        require(selection_proof["protocol_sha256"] == digest(protocol)
                and selection_proof["selection_sha256"] == file_hash(root / "selection.json")
                and selection_proof["screen_completed_sha256"] == {directory.name: file_hash(directory / "completed.json")
                    for directory in sorted((root / "runs").glob("screen_*"))}, "Screen selection source proof changed")
        budget = selection["selected"]["steps"]
    require(fold in (SCREEN if phase == "screen" else CONFIRM), "Fold/phase differs")
    data = CausalTimingData(config_for(protocol, root, fold, phase=phase, seed=42))
    if fold in SCREEN:
        require(data.identity == protocol["data"][fold], "Causal scaler or data identity differs")
    artifact_dir = root / "data" / fold
    if not artifact_dir.exists():
        data.save_artifacts(artifact_dir)
        seal(artifact_dir, protocol, data_identity=data.identity)
    else:
        cached = closed(artifact_dir, protocol)
        require(cached is not None and cached["data_identity"] == data.identity, "Partial or changed causal artifacts")
    for segment in SEGMENTS:
        teacher_segment(root, protocol, data, fold, segment)
    fit_fold(root, protocol, fold)
    for seed in ((42,) if phase == "screen" else SEEDS):
        for method in METHODS:
            config = config_for(protocol, root, fold, method, seed, phase)
            trial_data = CausalTimingData(config)
            require(trial_data.identity == data.identity, "Methods or seeds changed market/scaler identity")
            trial(root, protocol, trial_data, fold, method, seed, phase, budget)
    check_frozen(protocol, root)
    write_json(root / f"worker_{phase}_{fold}.json", dict(status="complete", fold=fold, phase=phase,
        protocol_sha256=digest(protocol), teacher_segments={segment: file_hash(root / "teacher" / fold / segment / "completed.json") for segment in SEGMENTS},
        runs={f"{phase}_{fold}_{method}_{seed}": file_hash(root / "runs" / f"{phase}_{fold}_{method}_{seed}" / "completed.json")
            for seed in ((42,) if phase == "screen" else SEEDS) for method in METHODS}))
    print(f"WORKER_COMPLETE {phase}/{fold}", flush=True)


def report(root, protocol):
    check_frozen(protocol, root)
    screen = load_rows(root, protocol, "screen")
    selection = select_screen(screen)
    if (root / "selection.json").exists():
        require(read_json(root / "selection.json") == selection, "Frozen screen selection changed")
    else:
        write_json(root / "selection.json", selection)
    proof = dict(protocol_sha256=digest(protocol), selection_sha256=file_hash(root / "selection.json"),
        screen_completed_sha256={directory.name: file_hash(directory / "completed.json") for directory in sorted((root / "runs").glob("screen_*"))})
    if (root / "selection_proof.json").exists():
        require(read_json(root / "selection_proof.json") == proof, "Screen selection source binding changed")
    else:
        write_json(root / "selection_proof.json", proof)
    result = dict(status=selection["status"], selection=selection, screen=screen, qualified=False,
        candidate=None, protected_period_evaluated=False, actionable=False)
    if selection["admitted"]:
        confirmation = load_rows(root, protocol, "confirm")
        if len(confirmation) == 12:
            outcome = confirmation_summary(confirmation, selection, protocol["gate"])
            result.update(status="confirmation_complete", confirmation=confirmation, outcome=outcome,
                qualified=outcome["qualified"], candidate=outcome["candidate"])
    write_json(root / "summary.json", result)
    write_json(root / "status.json", dict(status=result["status"], qualified=result["qualified"], actionable=False))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "screen", "confirm", "report"))
    parser.add_argument("--fold", choices=(*SCREEN, *CONFIRM))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    require((args.stage in ("screen", "confirm")) == (args.fold is not None), "Only workers require exactly one fold")
    root = output_root(args.output)
    if args.stage == "prepare":
        return prepare(root)
    require((root / "protocol.json").exists(), "Explicit prepare must freeze the study first")
    protocol = read_json(root / "protocol.json")
    return report(root, protocol) if args.stage == "report" else worker(root, protocol, args.stage, args.fold)


if __name__ == "__main__":
    main()
