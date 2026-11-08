"""Registered Actor x shared-prefix budget study; immutable timing execution."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
from time import perf_counter
import zipfile

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtest.run import git_metadata
from src.data.preprocess import ROOT, file_hash, read_config, write_json
from src.envs.initial_account import MODES
from src.training import research_timing as original
from src.training.freeze import json_digest
from src.training.research_matrix import preserve_training_rng, diagnostic_summary
from tools.archive import research_timing_credit as credit
from tools.archive import research_timing_margin as shared
from tools.archive.timing_phase_agent import PhaseTimingAgent, POLICIES, code_identity, phase_policy_contract, same_json, expected_policy_record
from tools.archive.timing_phase_policy import PHASE_ARCHITECTURES, mapped_baseline_state_dict

ARCHITECTURES = PHASE_ARCHITECTURES
SNAPSHOT_STEPS = (8192, 32768)
CELLS = tuple((architecture, steps) for architecture in ARCHITECTURES for steps in SNAPSHOT_STEPS)
REFERENCE = ROOT / "reports/runs/ppo_timing_credit_20261015"
REFERENCE_AUDIT_SHA256 = "fd3449e3d0fcb2df87fa9cd3d87274fa3d93290bb77a3b8d111c5c04751e8955"
DOCUMENT = ROOT / "doc/ppo_timing_phase_budget_protocol_20261015.md"
COMPONENT_FILES = ("tools/archive/timing_phase_policy.py", "tools/archive/timing_phase_agent.py", "tests/test_timing_phase_policy.py",
                   "tests/test_timing_phase_agent.py", "tests/test_timing_phase_review.py")
HELPER_FILES = ("tools/archive/research_timing_credit.py", "tools/archive/research_timing_margin.py")
TEST_FILES = ("tests/test_research_timing_phase_budget.py", "tests/test_timing_phase_budget_review.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cell_label(architecture, steps):
    require(architecture in ARCHITECTURES and type(steps) is int and steps in SNAPSHOT_STEPS,
            "Unregistered Actor or snapshot budget")
    return f"{architecture}_{steps}"


def run_id(fold, architecture, phase, seed):
    require(architecture in ARCHITECTURES and phase in ("screen", "confirm") and type(seed) is int
            and seed in original.SEEDS and fold in (original.SCREEN_FOLDS if phase == "screen" else original.CONFIRM_FOLDS)
            and (phase != "screen" or seed == 42), "Unregistered physical timing run")
    return f"{phase}_{fold}_{architecture}_{seed}"


def reference_protocol(directory, *, full=False):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(ROOT.resolve()) and file_hash(directory / "final_audit.json") == REFERENCE_AUDIT_SHA256,
            "Completed Credit final audit must match its hard pin")
    audit = read_config(directory / "final_audit.json")
    files = audit["signed_files_sha256"]
    require(audit.get("status") == "passed" and len(files) == 4676, "Incomplete completed Credit signature inventory")
    key = (directory / "protocol.json").relative_to(ROOT).as_posix()
    require(files.get(key) == file_hash(directory / "protocol.json"), "Credit protocol is not signed")
    if full:
        for name, digest in files.items():
            path = PurePosixPath(name)
            require(not path.is_absolute() and path.as_posix() == name and "\\" not in name and ".." not in path.parts,
                    "Predecessor signed path is not workspace-relative POSIX")
            actual = (ROOT / name).resolve()
            require(actual.is_relative_to(ROOT.resolve()) and file_hash(actual) == digest, "A predecessor signed input changed")
    previous = read_config(directory / "protocol.json")
    credit.check_frozen(previous)
    return previous


def fixed_registration():
    return dict(study="timing_phase_budget", architectures=list(ARCHITECTURES), snapshot_steps=list(SNAPSHOT_STEPS),
        screen_physical_budget=32768, planned_screen_physical_runs=4, planned_screen_snapshots=8,
        planned_screen_actual_steps=131072, maximum_confirmation_runs=6, maximum_confirmation_steps=196608,
        maximum_actual_training_steps=327680, reward_basis="relative_log", gamma=.995, gae_lambda=1., calendar_margin=2.,
        feature_names=list(original.FEATURE_NAMES), parameters={"linear": 12805, "due_phase_residual": 12914},
        screen_minimum_each_fold_gap=.0001, screen_requires_greedy_and_economic_change=True,
        tie_break="linear then short", checkpoint_selection="all complete long paths; registered 8192 and 32768 only")


def check_frozen(protocol):
    for value in protocol.get("snapshot_steps", []):
        cell_label("linear", value)
    require(all(same_json(protocol.get(key), value) for key, value in fixed_registration().items()),
            "Phase-budget grid, actual budget or fixed gate changed")
    identity = code_identity()
    require(protocol["source_hashes"] == identity["source_hashes"] and protocol["tool_sha256"] == file_hash(Path(__file__))
            and protocol["document_sha256"] == file_hash(DOCUMENT)
            and set(protocol["component_hashes"]) == set(COMPONENT_FILES) and set(protocol["helper_hashes"]) == set(HELPER_FILES)
            and set(protocol["test_hashes"]) == set(TEST_FILES)
            and protocol["reference_final_audit_sha256"] == REFERENCE_AUDIT_SHA256,
            "Phase-budget source, runner, inventory, document or reference changed")
    for group in ("component_hashes", "helper_hashes", "test_hashes"):
        for name, digest in protocol[group].items():
            require(file_hash(ROOT / name) == digest, "Frozen phase-budget component changed")
    previous = reference_protocol(protocol["reference_root"])
    require(file_hash(Path(protocol["reference_root"]) / "protocol.json") == protocol["reference_protocol_sha256"]
            and all(same_json(protocol[key], previous[key]) for key in ("folds", "base_config", "gate"))
            and protocol["baseline_reference_root"] == previous["reproduction_reference_root"],
            "Phase-budget data, execution or baseline lineage changed")


def prepare(root, reference_root=REFERENCE):
    root, reference_root = Path(root).resolve(), Path(reference_root).resolve()
    previous = reference_protocol(reference_root, full=True)
    require(root.is_relative_to(ROOT.resolve()) and not root.is_relative_to(reference_root)
            and not reference_root.is_relative_to(root), "Output overlaps immutable predecessor or leaves workspace")
    if (root / "protocol.json").exists():
        saved = read_config(root / "protocol.json")
        require(saved["reference_root"] == str(reference_root), "Cannot replace the registered predecessor")
        check_frozen(saved)
        return saved
    require(not root.exists(), "Incomplete unregistered output needs explicit review")
    protocol = dict(schema_version=1, **fixed_registration(), folds=deepcopy(previous["folds"]),
        base_config=deepcopy(previous["base_config"]), gate=deepcopy(previous["gate"]), reference_root=str(reference_root),
        reference_protocol_sha256=file_hash(reference_root / "protocol.json"), reference_final_audit_sha256=REFERENCE_AUDIT_SHA256,
        baseline_reference_root=previous["reproduction_reference_root"], source_hashes=code_identity()["source_hashes"],
        tool_sha256=file_hash(Path(__file__)), document_sha256=file_hash(DOCUMENT), git=git_metadata(),
        component_hashes={name: file_hash(ROOT / name) for name in COMPONENT_FILES},
        helper_hashes={name: file_hash(ROOT / name) for name in HELPER_FILES},
        test_hashes={name: file_hash(ROOT / name) for name in TEST_FILES}, test_evaluated=False, actionable=False)
    check_frozen(protocol)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "protocol.json", protocol)
    write_json(root / "status.json", dict(status="prepared", test_evaluated=False, actionable=False))
    return protocol


def config_for(protocol, root, fold, architecture, phase, seed, budget=32768):
    run_id(fold, architecture, phase, seed)
    cell_label(architecture, budget)
    require(phase != "screen" or budget == 32768, "Screen is four full physical 32768 paths, not independent short runs")
    config = original.config_for(protocol["base_config"], root, protocol["folds"][fold], fold, "relative_log", phase, seed)
    config["ppo"]["gae_lambda"] = 1.
    config["phase_budget_study"] = dict(study="timing_phase_budget", architecture=architecture, physical_budget=budget,
        snapshot_steps=list(SNAPSHOT_STEPS) if phase == "screen" else [budget], protocol_sha256=json_digest(protocol),
        tool_sha256=protocol["tool_sha256"], component_hashes=deepcopy(protocol["component_hashes"]),
        helper_hashes=deepcopy(protocol["helper_hashes"]), reference_audit_sha256=REFERENCE_AUDIT_SHA256)
    return config


class PhaseData(original.TimingData):
    def __init__(self, config):
        identity = config["phase_budget_study"]
        cell_label(identity["architecture"], identity["physical_budget"])
        require(config["timing"]["reward_basis"] == "relative_log" and type(config["ppo"]["gae_lambda"]) is float
                and config["ppo"]["gae_lambda"] == 1. and config["ppo"]["gamma"] == .995,
                "Phase-budget data requires the frozen reward and credit estimator")
        super().__init__(config)
        self.contract["phase_budget_study"] = deepcopy(identity)
        self.contract = phase_policy_contract(self.contract, identity["architecture"])


def make_model(env, config):
    torch.set_num_threads(1)
    architecture = config["phase_budget_study"]["architecture"]
    return PPO(POLICIES[architecture], env, seed=config["seed"], device="cpu", verbose=0,
        policy_kwargs=dict(calendar_margin=2., net_arch={"pi": [], "vf": [64, 64]}), **config["ppo"])


def _json_state(value):
    if isinstance(value, np.ndarray):
        return dict(dtype=str(value.dtype), shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _json_state(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_state(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def training_state_digest(model, experience):
    controller = experience.env
    books = []
    for wrapper in (controller.env, controller.benchmark):
        base = wrapper.env
        books.append(dict(portfolio=base.portfolio.__dict__, broker=base.broker.__dict__, current_index=base.market.current_index,
            steps=base._steps, done=base._done, history=base.history, observation=wrapper._observation))
    state = dict(last_obs=model._last_obs, last_episode_starts=model._last_episode_starts,
        num_timesteps=model.num_timesteps, total_timesteps=model._total_timesteps, n_updates=model._n_updates,
        progress=model._current_progress_remaining, schedule=controller.schedule_audit(), phase=controller.phase,
        stock_observation=controller.stock_observation, benchmark_observation=controller.benchmark_stock_observation, books=books)
    return json_digest(_json_state(state))


@contextmanager
def isolated_checkpoint(model, experience):
    before, mode = training_state_digest(model, experience), model.policy.training
    with preserve_training_rng(model.device):
        try:
            yield
        finally:
            model.policy.set_training_mode(mode)
    require(before == training_state_digest(model, experience), "Checkpoint changed training books, schedule, cursor or pending observation")


def constant_schedules(model):
    samples = (0., .25, .5, .75, 1.)
    require(not isinstance(model.learning_rate, bool) and model.learning_rate == 3e-4
            and all(model.lr_schedule(value) == 3e-4 and model.clip_range(value) == .2 for value in samples),
            "Shared prefix requires constant learning-rate and clip schedules")
    return dict(learning_rate=3e-4, clip_range=.2, checked_progress=list(samples))


def capture_snapshot(callback, steps):
    model, config, data = callback.model, callback.data.config, callback.data
    cell_label(config["phase_budget_study"]["architecture"], steps)
    require(type(model.num_timesteps) is int and model.num_timesteps == steps and steps not in callback.snapshots,
            "Checkpoint must be a fresh registered after-optimizer budget")
    counts = original._optimizer_counts(model)
    expected_tensors = data.contract["phase_policy"]["optimizer_tensor_count"]
    require(len(counts) == expected_tensors and min(counts.values()) > 0 and all(torch.isfinite(p).all() for p in model.policy.parameters()),
            "Checkpoint lacks finite actual updates of every registered parameter")
    folder = callback.output / "snapshots" / str(steps)
    folder.mkdir(parents=True, exist_ok=False)
    model_folder = callback.output / ("short_model" if steps == 8192 and callback.budget != 8192 else "final_model")
    with isolated_checkpoint(model, callback.experience):
        model.phase_constant_schedules = constant_schedules(model)
        agent = PhaseTimingAgent(model, data.contract, dict(phase_budget_study=config["phase_budget_study"]))
        agent.save(model_folder, config, dict(snapshot_steps=steps, shared_physical_budget=callback.budget))
        restored = PhaseTimingAgent.load(model_folder, data.contract, "cpu")
        require(all(torch.equal(value.detach().cpu(), restored.model.policy.state_dict()[key]) for key, value in model.policy.state_dict().items()),
                "Checkpoint save/load changed tensors")
        first, second = agent.inspect_action(model._last_obs[0]), restored.inspect_action(model._last_obs[0])
        require(first["macro_action"] == second["macro_action"] and np.array_equal(first["action_probabilities"], second["action_probabilities"]),
                "Checkpoint save/load changed macro action or probabilities")
        del restored
    updates_path = folder / "training_updates.csv"
    pd.DataFrame(callback.update_records).to_csv(updates_path, index=False)
    audit = dict(actual_timesteps=steps, actual_optimizer_updates=max(counts.values()), optimizer_steps_min=min(counts.values()),
        optimizer_steps_max=max(counts.values()), optimizer_steps_by_parameter=counts, parameters=model.policy.parameter_count,
        model_sha256=file_hash(model_folder / "model.zip"), tensor_delta_l2={key: float(torch.linalg.vector_norm(value.detach().cpu() - callback.initial[key]))
            for key, value in model.policy.state_dict().items()}, experience=deepcopy(callback.experience.stats()),
        training_schedule=deepcopy(callback.experience.env.schedule_audit()), training_trace_prefix=callback.trace.manifest(),
        rollout_buffers=deepcopy(callback.rollouts), credit_estimator_buffers=deepcopy(callback.estimator_buffers),
        mode_statistics=callback.trace.statistics.snapshot(), training_diagnostics=diagnostic_summary(callback.update_records),
        training_updates_path=updates_path.relative_to(callback.output).as_posix(), training_updates_sha256=file_hash(updates_path),
        elapsed_seconds=perf_counter() - callback.started, source_hashes=deepcopy(callback.protocol["source_hashes"]),
        tool_sha256=callback.protocol["tool_sha256"], actual_model_gamma=float(model.gamma), actual_buffer_gamma=float(model.rollout_buffer.gamma),
        actual_model_gae_lambda=float(model.gae_lambda), actual_buffer_gae_lambda=float(model.rollout_buffer.gae_lambda),
        constant_schedules=deepcopy(model.phase_constant_schedules), save_load_parameters_equal=True, save_load_macro_action_equal=True,
        checkpoint_training_state_unchanged=True, all_parameters_finite=True, test_evaluated=False)
    original.verify_timing_trace(callback.output / "training_trace", audit["training_trace_prefix"], steps=steps, expected_contract_sha256=json_digest(data.contract))
    original.verify_rollouts(callback.output, audit["rollout_buffers"], steps, audit["training_trace_prefix"])
    credit.verify_estimator_buffers(callback.output, audit, config)
    write_json(folder / "final_audit.json", audit)
    callback.snapshots[steps] = audit


class PhaseBudgetCallback(credit.CreditCallback):
    def __init__(self, *args, protocol, **kwargs):
        super().__init__(*args, **kwargs)
        self.protocol, self.snapshots = protocol, {}

    def _on_rollout_start(self):
        super()._on_rollout_start()
        if self.budget > 8192 and self.model.num_timesteps == 8192:
            capture_snapshot(self, 8192)


def baseline_metrics(protocol, fold):
    reference = read_config(Path(protocol["reference_root"]) / "final_audit.json")["signed_files_sha256"]
    results = {}
    for mode in MODES:
        path = Path(protocol["baseline_reference_root"]) / "rules" / fold / mode / "result.json"
        key = path.resolve().relative_to(ROOT.resolve()).as_posix()
        require(reference.get(key) == file_hash(path), "Frozen unchanged baseline changed")
        results[mode] = read_config(path)["metrics"]
    return results


def verify_model(directory, config, contract, steps, audit=None):
    require(type(steps) is int and (steps == 0 or steps in SNAPSHOT_STEPS), "Declared model clock must be a registered integer")
    directory = Path(directory)
    metadata = read_config(directory / "metadata.json")
    names, spec = tuple(original.FEATURE_NAMES), contract["phase_policy"]
    record = expected_policy_record(names, spec)
    require(same_json(metadata["contract"], contract) and same_json(metadata["config"], config)
            and same_json(metadata["policy_record"], record) and metadata["code_identity"] == code_identity()
            and metadata["algorithm_class"] == "stable_baselines3.ppo.ppo.PPO" and type(metadata["num_timesteps"]) is int
            and metadata["num_timesteps"] == steps and metadata["model_sha256"] == file_hash(directory / "model.zip"),
            "Snapshot model class, data, config, source or exact clock differs")
    with zipfile.ZipFile(directory / "model.zip", "r") as zipped:
        require(zipped.testzip() is None, "Snapshot model CRC failed")
        archived = json.loads(zipped.read("data"))
        require(same_json(archived["phase_registered_contract"], contract) and same_json(archived["phase_policy_record"], record)
                and archived["phase_code_identity"] == code_identity() and archived["phase_registered_algorithm"] == metadata["algorithm_class"]
                and type(archived["num_timesteps"]) is int and archived["num_timesteps"] == steps
                and type(archived["gae_lambda"]) is float and archived["gae_lambda"] == 1. and archived["gamma"] == .995
                and archived["learning_rate"] == 3e-4, "Snapshot PPO archive estimator or policy identity differs")
        theta = torch.load(io.BytesIO(zipped.read("policy.pth")), map_location="cpu", weights_only=True)
        optimizer = torch.load(io.BytesIO(zipped.read("policy.optimizer.pth")), map_location="cpu", weights_only=True)
    actor = "action_net.base" if spec["architecture"] == "due_phase_residual" else "action_net"
    shapes = {"mlp_extractor.value_net.0.weight": (64, 129), "mlp_extractor.value_net.0.bias": (64,),
        "mlp_extractor.value_net.2.weight": (64, 64), "mlp_extractor.value_net.2.bias": (64,),
        actor + ".weight": (2, 129), actor + ".bias": (2,)}
    if spec["architecture"] == "due_phase_residual":
        shapes["action_net.residual.weight"] = (1, 109)
    shapes.update({"value_net.weight": (1, 64), "value_net.bias": (1,)})
    require(tuple(theta) == tuple(shapes) and all(tuple(theta[key].shape) == shape and theta[key].dtype == torch.float32
            and torch.isfinite(theta[key]).all() for key, shape in shapes.items())
            and sum(value.numel() for value in theta.values()) == spec["parameter_count"], "Actual snapshot tensor architecture differs")
    states = optimizer["state"]
    coordinates = [index for group in optimizer["param_groups"] for index in group["params"]]
    require(len(coordinates) == len(shapes) and all(type(index) is int for index in coordinates)
            and coordinates == list(range(len(shapes))) and len(set(coordinates)) == len(coordinates),
            "Saved Adam groups must uniquely cover every registered tensor in order")
    require(len(states) == (spec["optimizer_tensor_count"] if steps else 0), "Actual Adam tensor coverage differs")
    if steps == 0:
        calendar = torch.zeros((2, 129), dtype=torch.float32)
        calendar[0, 1:20], calendar[1, 0] = 2., 2.
        require(torch.equal(theta[actor + ".weight"], calendar) and torch.count_nonzero(theta[actor + ".bias"]) == 0
                and (spec["architecture"] != "due_phase_residual" or torch.count_nonzero(theta["action_net.residual.weight"]) == 0),
                "Initial baseline calendar Actor or zero due vector changed")
    if audit is not None:
        require(set(states) == set(coordinates), "Saved Adam state keys differ from the registered tensors")
        counts_by_parameter = {}
        for index, (name, shape) in enumerate(shapes.items()):
            state = states[index]
            raw_step = state["step"]
            require(torch.is_tensor(raw_step) and raw_step.shape == () and raw_step.dtype == torch.float32 and torch.isfinite(raw_step).all()
                    and float(raw_step) > 0 and float(raw_step).is_integer()
                    and all(state[field].shape == shape and state[field].dtype == torch.float32 and torch.isfinite(state[field]).all()
                            for field in ("exp_avg", "exp_avg_sq")), "Saved Adam state precision, shape or actual update clock differs")
            counts_by_parameter[name] = int(raw_step)
        require(same_json(counts_by_parameter, audit["optimizer_steps_by_parameter"]), "Saved per-parameter Adam updates differ")
        counts = list(counts_by_parameter.values())
        require(min(counts) == audit["optimizer_steps_min"] and max(counts) == audit["actual_optimizer_updates"] == audit["optimizer_steps_max"],
                "Actual saved Adam updates differ from snapshot audit")
        require(archived["phase_constant_schedules"] == audit["constant_schedules"], "Archive constant schedules differ")
    return theta


def compare_linear_prefix(directory, protocol, audit):
    directory = Path(directory)
    config = read_config(directory / "config.json")
    require(config["timing"]["phase"] == "screen" and config["phase_budget_study"]["architecture"] == "linear"
            and audit["actual_timesteps"] == 8192, "Only fresh linear screen short controls may reproduce Credit")
    previous = Path(protocol["reference_root"]) / "runs" / credit.run_id(config["timing"]["fold"], "relative_log", 1., "screen", 42)
    credit.original.verify_leaf_files(previous, read_config(previous / "completed.json")["files"])
    old = read_config(previous / "final_audit.json")
    equal = {}
    for role, folder in (("initial_model", "initial_model"), ("short_model", "final_model")):
        first, second = shared.policy_tensors(directory / role), shared.policy_tensors(previous / folder)
        equal[role] = set(first) == set(second) and all(torch.equal(first[key], second[key]) for key in first)
    require(all(equal.values()), "Fresh linear prefix tensors do not reproduce Credit")
    groups = ((directory / "training_trace", audit["training_trace_prefix"]["chunks"], previous / "training_trace", old["training_trace_prefix"]["chunks"]),
        (directory / "training_trace/benchmark", audit["training_trace_prefix"]["benchmark_trace"]["chunks"], previous / "training_trace/benchmark", old["training_trace_prefix"]["benchmark_trace"]["chunks"]),
        (directory, audit["rollout_buffers"], previous, old["rollout_buffers"]))
    for first_dir, first_records, second_dir, second_records in groups:
        require(len(first_records) == len(second_records), "Fresh linear prefix buffer/native count differs")
        for left, right in zip(first_records, second_records):
            require(all(left[key] == right[key] for key in ("first_step", "last_step", "steps")), "Fresh control clocks differ")
            with np.load(first_dir / left["path"], allow_pickle=False) as a, np.load(second_dir / right["path"], allow_pickle=False) as b:
                require(set(a.files) == set(b.files) and all(np.array_equal(a[key], b[key], equal_nan=True) if a[key].dtype.kind in "fc" else np.array_equal(a[key], b[key]) for key in a.files),
                        "Fresh linear Agent/benchmark/native PPO arrays do not reproduce Credit")
    return dict(predecessor_completed_sha256=file_hash(previous / "completed.json"), policy_tensors_equal=equal,
        agent_native_arrays_equal=True, benchmark_native_arrays_equal=True, original_eight_key_ppo_buffers_equal=True,
        reused_as_sample=False, no_extra_environment_steps=True)


def validate_records(records, phase, cell=None):
    expected = {(fold, architecture, steps, 42) for fold in original.SCREEN_FOLDS for architecture, steps in CELLS} if phase == "screen" else {
        (fold, cell[0], cell[1], seed) for fold in original.CONFIRM_FOLDS for seed in original.SEEDS} if cell in CELLS else set()
    require(bool(expected) and len(records) == len(expected), "All registered snapshot cells are required")
    actual = []
    for row in records:
        cell_label(row["architecture"], row["steps"])
        require(type(row["seed"]) is int and row["phase"] == phase and set(row["metrics"]) == set(MODES)
                and set(row["changes"]) == set(MODES) and row["method"] == cell_label(row["architecture"], row["steps"])
                and row["test_evaluated"] is False and row["actionable"] is False, "Invalid snapshot record identity")
        actual.append((row["fold"], row["architecture"], row["steps"], row["seed"]))
        for mode in MODES:
            require(row["metrics"][mode]["initialized"] == row["metrics"][mode]["benchmark"]
                    and all(type(row["changes"][mode][key]) is bool for key in ("greedy_changed", "economic_ledger_changed")),
                    "Initialized economic baseline or behavior flags differ")
            for role in ("trained", "initialized", "benchmark"):
                require(all(type(row["metrics"][mode][role][key]) in (float, int) and np.isfinite(row["metrics"][mode][role][key])
                    for key in ("cumulative_return", "max_drawdown", "total_fees", "total_slippage_cost")), "Missing finite snapshot financial metrics")
    require(set(actual) == expected and len(set(actual)) == len(actual), "Missing, duplicate or unregistered snapshot")


def select_screen(records):
    validate_records(records, "screen")
    candidates = {}
    for architecture, steps in CELLS:
        rows = [row for row in records if (row["architecture"], row["steps"]) == (architecture, steps)]
        gaps = {row["fold"]: original._account_mean(row)["cumulative_return"] - original._account_mean(row, "benchmark")["cumulative_return"] for row in rows}
        candidates[cell_label(architecture, steps)] = dict(fold_gaps=gaps, minimum_fold_gap=min(gaps.values()), mean_gap=float(np.mean(list(gaps.values()))),
            worst_account_drawdown=max(abs(row["metrics"][mode]["trained"]["max_drawdown"]) for row in rows for mode in MODES),
            mean_cost=float(np.mean([original._account_mean(row)["total_fees"] + original._account_mean(row)["total_slippage_cost"] for row in rows])),
            same_path_behavior_changed=any(all(row["changes"][mode].values()) for row in rows for mode in MODES))
    architecture, steps = min(CELLS, key=lambda item: (-candidates[cell_label(*item)]["minimum_fold_gap"], -candidates[cell_label(*item)]["mean_gap"],
        candidates[cell_label(*item)]["worst_account_drawdown"], candidates[cell_label(*item)]["mean_cost"], ARCHITECTURES.index(item[0]), item[1]))
    choice = candidates[cell_label(architecture, steps)]
    admitted = choice["minimum_fold_gap"] >= .0001 and choice["same_path_behavior_changed"]
    return dict(selected_architecture=architecture, selected_budget=steps, candidates=candidates, admitted=bool(admitted),
        status="screen_admitted" if admitted else "screen_failed", physical_runs=4, snapshots=8, actual_training_steps=131072,
        shared_short_prefix_steps=32768, snapshot_steps_sum=163840, confirmation_steps_authorized=6 * steps if admitted else 0,
        selection_uses_only="Complete 2023/2024 physical long paths and their two registered snapshots", test_evaluated=False)


def cached_trial(directory, config, protocol, selection_sha256=None):
    check_frozen(protocol)
    directory = Path(directory)
    marker = directory / "completed.json"
    if not marker.exists():
        return None
    completed = read_config(marker)
    spec, identity = config["timing"], config["phase_budget_study"]
    name = run_id(spec["fold"], identity["architecture"], spec["phase"], config["seed"])
    require(directory.name == name and completed["protocol_sha256"] == json_digest(protocol)
            and completed["config_sha256"] == json_digest(config) and completed["tool_sha256"] == protocol["tool_sha256"]
            and completed["selection_sha256"] == selection_sha256 and same_json(read_config(directory / "config.json"), config)
            and type(completed["physical_steps"]) is int and completed["physical_steps"] == identity["physical_budget"],
            "Cached physical run belongs to another registered variant, budget or selection")
    original.verify_leaf_files(directory, completed["files"])
    data = PhaseData(config)
    require(json_digest(data.stock_contract) == protocol["folds"][spec["fold"]]["stock_contract_sha256"], "Snapshot cache changed economic data")
    expected_dates = original.validation_dates(data)
    initial_theta = verify_model(directory / "initial_model", config, data.contract, 0)
    if spec["phase"] == "screen" and identity["architecture"] == "due_phase_residual":
        other_config = config_for(protocol, directory.parent.parent, spec["fold"], "linear", "screen", config["seed"])
        other_contract = deepcopy(data.contract)
        other_contract["phase_budget_study"] = deepcopy(other_config["phase_budget_study"])
        other_contract = phase_policy_contract(other_contract, "linear")
        other_folder = directory.parent / run_id(spec["fold"], "linear", "screen", config["seed"]) / "initial_model"
        other_theta = verify_model(other_folder, other_config, other_contract, 0)
        mapped = {key.replace("action_net.base.", "action_net."): value for key, value in initial_theta.items() if key != "action_net.residual.weight"}
        require(set(mapped) == set(other_theta) and all(torch.equal(mapped[key], other_theta[key]) for key in mapped),
                "Cached architectures do not share mapped initial Actor/Critic tensors")
    benchmark = baseline_metrics(protocol, spec["fold"])
    budgets = identity["snapshot_steps"]
    rows = completed["records"]
    require(len(rows) == len(budgets) and [row["steps"] for row in rows] == budgets, "Cached run omits a registered snapshot")
    for row, steps in zip(rows, budgets):
        cell_label(row["architecture"], row["steps"])
        require((row["architecture"], row["fold"], row["phase"], row["seed"], row["physical_run_id"]) ==
                (identity["architecture"], spec["fold"], spec["phase"], config["seed"], name)
                and row["method"] == cell_label(identity["architecture"], steps)
                and type(row["seed"]) is int and row["test_evaluated"] is False and row["actionable"] is False,
                "Snapshot row variant, clock, physical parent or protected-period flags differ")
        folder = directory / "snapshots" / str(steps)
        audit = read_config(folder / "final_audit.json")
        require(same_json(audit, row["audit"]) and type(audit["actual_timesteps"]) is int and audit["actual_timesteps"] == steps
                and audit["experience"]["total_steps"] == audit["experience"]["benchmark_steps"] == steps
                and audit["training_schedule"]["agent_environment_steps"] == audit["training_schedule"]["benchmark_extra_environment_steps"] == steps
                and audit["source_hashes"] == protocol["source_hashes"] and audit["tool_sha256"] == protocol["tool_sha256"]
                and audit["parameters"] == data.contract["phase_policy"]["parameter_count"]
                and all(audit.get(key) is True for key in ("save_load_parameters_equal", "save_load_macro_action_equal",
                    "checkpoint_training_state_unchanged", "all_parameters_finite")), "Snapshot audit or actual independent steps differ")
        model_folder = directory / ("short_model" if steps == 8192 and identity["physical_budget"] != 8192 else "final_model")
        require(row["model_path"] == str(model_folder), "Snapshot model path differs")
        snapshot_theta = verify_model(model_folder, config, data.contract, steps, audit)
        require(set(audit["tensor_delta_l2"]) == set(initial_theta), "Snapshot tensor-delta keys differ from actual parameters")
        for key, initial_value in initial_theta.items():
            claimed = audit["tensor_delta_l2"][key]
            actual = float(torch.linalg.vector_norm(snapshot_theta[key] - initial_value))
            require(type(claimed) in (float, int) and np.isfinite(claimed) and np.isclose(claimed, actual, rtol=1e-6, atol=1e-7),
                    "Snapshot tensor-delta claim differs from actual initial and saved tensors")
        original.verify_timing_trace(directory / "training_trace", audit["training_trace_prefix"], steps=steps, expected_contract_sha256=json_digest(data.contract))
        original.verify_rollouts(directory, audit["rollout_buffers"], steps, audit["training_trace_prefix"])
        credit.verify_estimator_buffers(directory, audit, config)
        require(file_hash(directory / audit["training_updates_path"]) == audit["training_updates_sha256"], "Snapshot update prefix changed")
        updates = pd.read_csv(directory / audit["training_updates_path"], float_precision="round_trip")
        require(len(updates) == steps // 512 and updates.iloc[-1]["timesteps"] == steps
                and updates.iloc[-1]["actual_optimizer_steps"] == audit["actual_optimizer_updates"]
                and np.all(updates["credit/gamma"] == .995) and np.all(updates["credit/gae_lambda"] == 1.), "Snapshot after-Adam logs differ")
        for mode in MODES:
            for role, path in (("initialized", directory / "initialized" / mode), ("trained", folder / "trained" / mode)):
                result = read_config(path / "result.json")
                require(same_json(result["contract"], data.contract), "Snapshot result data or Actor contract differs")
                original._verify_result(path, result, data.contract, mode, expected_dates=expected_dates)
                require(result["metrics"] == row["metrics"][mode][role] and result["benchmark_metrics"] == row["metrics"][mode]["benchmark"] == benchmark[mode]
                        and row["evaluation_steps"][role][mode] == result["evaluation_steps"], "Snapshot economic result or full Q2 scope differs")
                changed = shared.case_changes(path, result)
                require(changed == row["changes"][mode] if role == "trained" else not any(changed.values()), "Snapshot behavior flags differ from native ledgers")
        if spec["phase"] == "screen" and identity["architecture"] == "linear" and steps == 8192:
            require(read_config(directory / "reproduction.json") == compare_linear_prefix(directory, protocol, audit), "Fresh control reproduction changed")
    return rows


def verify_worker_receipt(root, protocol, phase, fold, cell=None):
    root = Path(root)
    if phase == "screen":
        expected = {run_id(fold, architecture, phase, 42) for architecture in ARCHITECTURES}
    else:
        require(phase == "confirm" and cell in CELLS, "Unregistered confirmation worker")
        cell_label(*cell)
        expected = {run_id(fold, cell[0], phase, seed) for seed in original.SEEDS}
    receipt = read_config(root / f"worker_{phase}_{fold}.json")
    completed = receipt["completed_runs"]
    require(receipt["status"] == "complete" and receipt["phase"] == phase and receipt["fold"] == fold
            and receipt["protocol_sha256"] == file_hash(root / "protocol.json")
            and receipt["selection_sha256"] == (file_hash(root / "selection.json") if phase == "confirm" else None)
            and receipt["no_global_summary_writes"] is True and len(completed) == len(expected)
            and {row["run_id"] for row in completed} == expected, "Exclusive worker has not completed every physical run")
    for row in completed:
        require(file_hash(root / "runs" / row["run_id"] / "completed.json") == row["completed_sha256"], "Worker marker changed")
    return receipt


def load_stage(root, protocol, phase, cell=None):
    check_frozen(protocol)
    require(phase in ("screen", "confirm"), "Unregistered stage")
    if phase == "confirm":
        require_confirmation(root, protocol, cell, file_hash(Path(root) / "selection.json"))
    rows = []
    folds = original.SCREEN_FOLDS if phase == "screen" else original.CONFIRM_FOLDS
    for fold in folds:
        verify_worker_receipt(root, protocol, phase, fold, cell)
        for architecture in ARCHITECTURES if phase == "screen" else (cell[0],):
            for seed in (42,) if phase == "screen" else original.SEEDS:
                config = config_for(protocol, root, fold, architecture, phase, seed, 32768 if phase == "screen" else cell[1])
                result = cached_trial(Path(root) / "runs" / run_id(fold, architecture, phase, seed), config, protocol,
                    file_hash(Path(root) / "selection.json") if phase == "confirm" else None)
                require(result is not None, "Stage reporting cannot create missing physical runs")
                rows.extend(result)
    validate_records(rows, phase, cell)
    return rows


def selection_record(root, protocol, rows):
    return dict(protocol_sha256=json_digest(protocol), **select_screen(rows), completed_sha256={
        run_id(fold, architecture, "screen", 42): file_hash(Path(root) / "runs" / run_id(fold, architecture, "screen", 42) / "completed.json")
        for fold in original.SCREEN_FOLDS for architecture in ARCHITECTURES})


def seal_selection(root, protocol, rows):
    selected = selection_record(root, protocol, rows)
    path = Path(root) / "selection.json"
    if path.exists():
        require(same_json(read_config(path), selected), "The locked matrix selection changed")
    else:
        write_json(path, selected)
    return selected


def require_confirmation(root, protocol, cell, selection_sha256):
    check_frozen(protocol)
    require(isinstance(cell, tuple) and len(cell) == 2, "Confirmation needs the exact selected Actor/budget")
    cell_label(*cell)
    path = Path(root) / "selection.json"
    require(path.exists() and file_hash(path) == selection_sha256, "Confirmation requires the frozen complete selection")
    selected = read_config(path)
    expected = selection_record(root, protocol, load_stage(root, protocol, "screen"))
    require(same_json(selected, expected) and selected["admitted"] is True
            and cell == (selected["selected_architecture"], selected["selected_budget"]), "Failed or different cell cannot receive confirmation")
    return selected


def run_trial(root, protocol, fold, architecture, phase, seed, budget=32768, selection_sha256=None):
    check_frozen(protocol)
    if phase == "confirm":
        require_confirmation(root, protocol, (architecture, budget), selection_sha256)
    config = config_for(protocol, root, fold, architecture, phase, seed, budget)
    name = run_id(fold, architecture, phase, seed)
    directory = Path(root) / "runs" / name
    cached = cached_trial(directory, config, protocol, selection_sha256)
    if cached is not None:
        return cached
    require(not directory.exists(), "Incomplete physical run requires review, never silent continuation")
    data = PhaseData(config)
    require(json_digest(data.stock_contract) == protocol["folds"][fold]["stock_contract_sha256"], "New architecture changed economic data or execution")
    directory.mkdir(parents=True)
    write_json(directory / "config.json", config)
    experience = original.TimingExperience(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    env.seed(seed)
    model = make_model(env, config)
    constant_schedules(model)
    initial = {key: value.detach().cpu().clone() for key, value in model.policy.state_dict().items()}
    agent = PhaseTimingAgent(model, data.contract, dict(phase_budget_study=config["phase_budget_study"]))
    agent.save(directory / "initial_model", config)
    mapped = mapped_baseline_state_dict(model.policy)
    for other_architecture in ARCHITECTURES:
        other = Path(root) / "runs" / run_id(fold, other_architecture, phase, seed) / "initial_model"
        if other.exists() and other != directory / "initial_model":
            other_theta = {key.replace("action_net.base.", "action_net."): value for key, value in shared.policy_tensors(other).items()
                           if key != "action_net.residual.weight"}
            require(set(mapped) == set(other_theta) and all(torch.equal(mapped[key].cpu(), other_theta[key]) for key in mapped), "Initial mapped Actor/Critic differ across architectures")
    benchmark = baseline_metrics(protocol, fold)
    initialized_steps = {}
    with preserve_training_rng(model.device):
        mode = model.policy.training
        try:
            for initial_mode in MODES:
                result = original.evaluate_case(agent, data, directory / "initialized" / initial_mode, initial_mode, "deterministic_initialized")
                require(result["metrics"] == benchmark[initial_mode] and not any(shared.case_changes(directory / "initialized" / initial_mode, result).values()),
                        "Initial Agent no longer executes the unchanged M20 baseline")
                initialized_steps[initial_mode] = result["evaluation_steps"]
        finally:
            model.policy.set_training_mode(mode)
    callback = PhaseBudgetCallback(data, directory, experience, initial, budget, name, protocol=protocol)
    model.set_logger(configure(str(directory / "logs"), ["csv"]))
    try:
        model.learn(total_timesteps=budget, callback=callback)
        model.logger.dump(step=model.num_timesteps)
        require(type(model.num_timesteps) is int and model.num_timesteps == budget, "Physical training did not finish its registered full budget")
        capture_snapshot(callback, budget)
        require(set(callback.snapshots) == set(config["phase_budget_study"]["snapshot_steps"]), "A registered checkpoint was missed")
        write_json(directory / "episodes.json", [row for row in experience.episodes if row["steps"]])
        snapshots = deepcopy(callback.snapshots)
    finally:
        callback.trace.close()
        env.close()
    del model, callback, agent, env, experience, initial
    original._release()
    records = []
    for steps in config["phase_budget_study"]["snapshot_steps"]:
        folder = directory / "snapshots" / str(steps)
        model_folder = directory / ("short_model" if steps == 8192 and budget != 8192 else "final_model")
        agent = PhaseTimingAgent.load(model_folder, data.contract, "cpu")
        metrics, changes, evaluated = {}, {}, {}
        for initial_mode in MODES:
            result = original.evaluate_case(agent, data, folder / "trained" / initial_mode, initial_mode, "trained")
            require(result["benchmark_metrics"] == benchmark[initial_mode], "Trained evaluation changed the frozen baseline")
            metrics[initial_mode] = dict(trained=result["metrics"], initialized=benchmark[initial_mode], benchmark=benchmark[initial_mode])
            changes[initial_mode] = shared.case_changes(folder / "trained" / initial_mode, result)
            evaluated[initial_mode] = result["evaluation_steps"]
        records.append(dict(fold=fold, architecture=architecture, steps=steps, method=cell_label(architecture, steps), phase=phase, seed=seed,
            physical_run_id=name, metrics=metrics, changes=changes, audit=snapshots[steps],
            evaluation_steps=dict(initialized=deepcopy(initialized_steps), trained=evaluated), model_path=str(model_folder), test_evaluated=False, actionable=False))
        del agent
    if phase == "screen" and architecture == "linear":
        write_json(directory / "reproduction.json", compare_linear_prefix(directory, protocol, snapshots[8192]))
    check_frozen(protocol)
    write_json(directory / "completed.json", dict(protocol_sha256=json_digest(protocol), config_sha256=json_digest(config), tool_sha256=protocol["tool_sha256"],
        selection_sha256=selection_sha256, physical_steps=budget, records=records, files=original._leaf_files(directory)))
    return records


def run_worker(root, protocol, phase, fold, cell=None):
    require(phase in ("screen", "confirm"), "Unregistered worker stage")
    selection_sha = file_hash(Path(root) / "selection.json") if phase == "confirm" else None
    if phase == "confirm":
        require_confirmation(root, protocol, cell, selection_sha)
    completed = []
    for architecture in ARCHITECTURES if phase == "screen" else (cell[0],):
        for seed in (42,) if phase == "screen" else original.SEEDS:
            run_trial(root, protocol, fold, architecture, phase, seed, 32768 if phase == "screen" else cell[1], selection_sha)
            name = run_id(fold, architecture, phase, seed)
            completed.append(dict(run_id=name, completed_sha256=file_hash(Path(root) / "runs" / name / "completed.json")))
            write_json(Path(root) / f"worker_{phase}_{fold}.json", dict(status="running", phase=phase, fold=fold,
                protocol_sha256=file_hash(Path(root) / "protocol.json"), selection_sha256=selection_sha, completed_runs=completed, no_global_summary_writes=True))
    write_json(Path(root) / f"worker_{phase}_{fold}.json", dict(status="complete", phase=phase, fold=fold,
        protocol_sha256=file_hash(Path(root) / "protocol.json"), selection_sha256=selection_sha, completed_runs=completed, no_global_summary_writes=True))
    return completed


def confirmation_summary(records, cell):
    validate_records(records, "confirm", cell)
    folds = {}
    for fold in original.CONFIRM_FOLDS:
        rows = sorted([row for row in records if row["fold"] == fold], key=lambda row: row["seed"])
        means = [original._account_mean(row) for row in rows]
        gaps = [mean["cumulative_return"] - original._account_mean(row, "benchmark")["cumulative_return"] for row, mean in zip(rows, means)]
        gains = [mean["cumulative_return"] - original._account_mean(row, "initialized")["cumulative_return"] for row, mean in zip(rows, means)]
        worst = max(abs(row["metrics"][mode]["trained"]["max_drawdown"]) for row in rows for mode in MODES)
        excess = max(abs(row["metrics"][mode]["trained"]["max_drawdown"]) - abs(row["metrics"][mode]["benchmark"]["max_drawdown"]) for row in rows for mode in MODES)
        folds[fold] = dict(metrics=original._metric_stats(means), baseline_gap=original._stats(gaps), own_initialization_gain=original._stats(gains),
            positive_gap_seeds=sum(gap > 0 for gap in gaps), worst_account_drawdown=worst, worst_account_drawdown_excess=excess,
            qualified=bool(np.mean(gaps) > 0 and sum(gap > 0 for gap in gaps) >= 2 and worst <= .10 and excess <= .02 and np.mean(gains) > 0),
            per_seed={str(row["seed"]): dict(metrics=mean, gap=gap, own_initialization_gain=gain) for row, mean, gap, gain in zip(rows, means, gaps, gains)})
    paired = {str(seed): float(np.mean([folds[fold]["per_seed"][str(seed)]["gap"] for fold in original.CONFIRM_FOLDS])) for seed in original.SEEDS}
    overall = original._stats(list(paired.values()))
    qualified = all(row["qualified"] for row in folds.values()) and overall["mean"] >= .005
    return dict(folds=folds, paired_across_fold_gap=overall, per_seed_gap=paired, qualified=bool(qualified),
        qualified_candidate=dict(architecture=cell[0], budget=cell[1], representative_fold="2026", representative_seed=42) if qualified else None,
        aggregation="Four-account mean within seed/fold, mean over two folds within seed, sample SD over three seeds")


def write_report(root, protocol, screen, selected):
    check_frozen(protocol)
    root = Path(root)
    require((root / "selection.json").exists(), "Report requires the frozen eight-snapshot selection")
    actual = load_stage(root, protocol, "screen")
    require(same_json(actual, screen) and same_json(seal_selection(root, protocol, actual), selected), "Report cannot use supplied unsourced rows or admission")
    cell = selected["selected_architecture"], selected["selected_budget"]
    confirmed = load_stage(root, protocol, "confirm", cell) if selected["admitted"] else []
    records = screen + confirmed
    physical = {}
    for row in records:
        previous = physical.get(row["physical_run_id"])
        if previous is None or row["steps"] > previous["steps"]:
            physical[row["physical_run_id"]] = row
    evaluation = sum(sum(row["evaluation_steps"]["trained"].values()) for row in records)
    evaluation += sum(sum(row["evaluation_steps"]["initialized"].values()) for row in physical.values())
    summary = confirmation_summary(confirmed, cell) if confirmed else dict(qualified=False, qualified_candidate=None)
    summary.update(screen=selected, actual_screen_steps=sum(row["steps"] for row in physical.values() if row["phase"] == "screen"),
        actual_confirmation_steps=sum(row["steps"] for row in confirmed), actual_training_steps=sum(row["steps"] for row in physical.values()),
        physical_training_runs=len(physical), final_snapshots=len(records), shared_short_prefix_steps=32768, snapshot_steps_sum=sum(row["steps"] for row in records),
        actual_optimizer_updates=sum(row["audit"]["actual_optimizer_updates"] for row in physical.values()),
        independent_training_benchmark_steps=sum(row["audit"]["experience"]["benchmark_steps"] for row in physical.values()),
        evaluation_initialized_paths=4 * len(physical), evaluation_trained_paths=4 * len(records),
        evaluation_agent_steps=evaluation, evaluation_parallel_benchmark_steps=evaluation, new_rule_control_steps=0,
        additional_reproduction_environment_steps=0, status="complete" if selected["admitted"] else "screen_failed", test_evaluated=False, actionable=False)
    write_json(root / "summary.json", records)
    write_json(root / "study_summary.json", summary)
    write_json(root / "status.json", {key: summary[key] for key in ("status", "actual_training_steps", "final_snapshots", "physical_training_runs", "qualified", "test_evaluated", "actionable")})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "screen", "confirm", "report"), required=True)
    parser.add_argument("--output", default="reports/runs/ppo_timing_phase_budget_20261015")
    parser.add_argument("--reference-root", default=str(REFERENCE))
    parser.add_argument("--fold", choices=(*original.SCREEN_FOLDS, *original.CONFIRM_FOLDS))
    args = parser.parse_args(argv)
    if args.fold is not None and (args.stage not in ("screen", "confirm") or args.fold not in (original.SCREEN_FOLDS if args.stage == "screen" else original.CONFIRM_FOLDS)):
        parser.error("--fold selects an exclusive screen or confirmation worker")
    root = Path(args.output).resolve()
    if args.stage != "prepare":
        require((root / "protocol.json").exists(), "Only prepare may create the registered study")
    protocol = prepare(root, args.reference_root)
    if args.stage == "prepare":
        return protocol
    if args.stage == "screen" and args.fold is not None:
        return run_worker(root, protocol, "screen", args.fold)
    screen = load_stage(root, protocol, "screen")
    selected = seal_selection(root, protocol, screen)
    if args.stage == "screen":
        write_json(root / "status.json", dict(status=selected["status"], actual_training_steps=131072, test_evaluated=False, actionable=False))
        return selected
    if args.stage == "confirm":
        require(selected["admitted"], "Failed matrix authorizes zero confirmation")
        cell = selected["selected_architecture"], selected["selected_budget"]
        return run_worker(root, protocol, "confirm", args.fold, cell) if args.fold is not None else confirmation_summary(load_stage(root, protocol, "confirm", cell), cell)
    return write_report(root, protocol, screen, selected)


if __name__ == "__main__":
    main()
