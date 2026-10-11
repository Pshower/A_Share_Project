"""Registered longer-budget priority/auxiliary-ranking PPO factorial study."""

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.priority_agent import PriorityAgent, priority_contract
from src.agents.priority_mapping import PriorityTradeActionWrapper, map_priority_trade_action
from src.agents.priority_policy import PriorityTradePolicy
from src.agents.trade_mapping import TradeOrder
from src.backtest.metrics import performance
from src.backtest.run import git_metadata, run_policy
from src.data.preprocess import ROOT, file_hash, read_config, write_json
from src.envs.broker import BrokerConfig
from src.envs.initial_account import MODES
from src.runtime import events
from src.runtime.device import resolve_device
from .exploration_env import ExplorationData, ExplorationEnv, EconomicRule
from .freeze import json_digest, source_hashes
from .matrix_experience import EpisodeAudit
from .matrix_report import _metric_mean, _metric_stats, _stats
from .priority_aux import AUX_SEED_SALT, RankAuxPPO, TrainRankPool, build_rank_pool, _frame_digest
from .priority_trace import PriorityTrace, verify_priority_trace
from .research_delta import _registered_files, _verify_files, _release
from .research_exploration import ExplorationCallback, config_for as exploration_config
from .research_matrix import MatrixCallback, preserve_training_rng, diagnostic_summary
from .research_trade import _trace_identity


SEEDS = (42, 123, 2026)
FOLDS = ("2023", "2024")
ACTORS = ("coupled", "decoupled")
METHODS = ("ppo", "rank_aux")
PROFILES = {"standard": dict(short=8192, long=16384),
            "extended": dict(short=8192, medium=16384, long=32768)}
EVAL_SEED = 20260927
PARAMETERS = 125322
DOCUMENT = ROOT / "doc" / "ppo_priority_rank_protocol_20261011.md"
REFERENCE_AUDIT_SHA256 = "fbb932aea3e269b3b46308e0bda21db6ef082bc9d92db033c1c42df26fdfbe41"


def config_for(base, reference_root, root, fold, actor, method, profile="standard", pool=None):
    if fold not in FOLDS or actor not in ACTORS or method not in METHODS or profile not in PROFILES:
        raise ValueError("Unregistered priority/ranking condition")
    config = exploration_config(base, root, "split_l1", 42)
    reference = read_config(Path(reference_root) / "protocol.json")
    config["research_config"] = str(Path(reference["folds"][fold]["directory"]) / "research.json")
    config.update(action_mode="stock_priority", priority_rank=dict(
        fold=fold, actor_variant=actor, training_method=method, market_pooling="pure_mean",
        profile=profile, auxiliary_coefficient=.05 if method == "rank_aux" else 0.,
        auxiliary_batch_size=1024, auxiliary_horizon=5, auxiliary_huber_delta=.1,
        pool_identity=deepcopy(pool) if pool is not None else {}), aux_pool_dir=str(Path(root) / "labels" / fold))
    config["ppo"]["learning_rate"] = 3e-5
    return config


def trial_config(protocol, root, fold, actor, method, seed):
    if seed not in SEEDS:
        raise ValueError("Unregistered training seed")
    config = config_for(protocol["base_config"], protocol["reference_root"], root, fold, actor, method,
                        protocol["profile"], protocol["folds"][fold]["label_pool"])
    config["seed"] = seed
    return config


class PriorityData(ExplorationData):
    """Reuse immutable market support; replace only the registered action contract."""

    def __init__(self, config):
        borrowed = deepcopy(config)
        borrowed["action_mode"] = "stock_trade"
        super().__init__(borrowed)
        spec = config["priority_rank"]
        if (config.get("action_mode") != "stock_priority" or spec["fold"] not in FOLDS
                or spec["actor_variant"] not in ACTORS or spec["training_method"] not in METHODS
                or spec["market_pooling"] != "pure_mean" or spec["profile"] not in PROFILES
                or spec["auxiliary_coefficient"] != (.05 if spec["training_method"] == "rank_aux" else 0.)
                or spec["auxiliary_batch_size"] != 1024 or spec["auxiliary_horizon"] != 5
                or spec["auxiliary_huber_delta"] != .1 or self.architecture != "split"):
            raise ValueError("Invalid registered priority/ranking prediction contract")
        self.config = deepcopy(config)
        self.contract = priority_contract(self.contract, spec["actor_variant"])
        self.contract["priority_rank"] = deepcopy(spec)

    def env(self, split):
        training = split == "train"
        env = ExplorationEnv(self.market(split), self.settings["initial_capital"],
            BrokerConfig(**self.settings["broker"]),
            initial_portfolio=self.config.get("initial_portfolio", "cash") if training else self.settings.get("initial_portfolio", "cash"),
            max_steps=self.config.get("training_episode_steps") if training else None,
            reward_basis=self.reward_basis if training else "simple_net")
        return PriorityTradeActionWrapper(env, actor_variant=self.config["priority_rank"]["actor_variant"],
            bound=self.config["logit_bound"], max_trade_weight=self.config["max_trade_weight"],
            max_commission_ratio=self.config["max_commission_ratio"],
            top_n=self.settings.get("trade_top_n", self.config["top_n"]),
            top_n_choices=self.config.get("top_n_choices") if training else None)


def make_model(env, config, pool):
    spec = config["priority_rank"]
    torch.set_num_threads(config["torch_threads"])
    return RankAuxPPO(PriorityTradePolicy, env, seed=config["seed"], device=resolve_device(config["device"]), verbose=0,
        aux_pool=pool, aux_coef=spec["auxiliary_coefficient"], aux_batch_size=spec["auxiliary_batch_size"],
        aux_seed=config["seed"], aux_huber_delta=spec["auxiliary_huber_delta"],
        policy_kwargs=dict(hidden_sizes=tuple(config["hidden_sizes"]), initial_log_std=config["initial_log_std"],
            actor_output_gain=config["actor_output_gain"], actor_variant=spec["actor_variant"]), **config["ppo"])


def validate_pool(pool, data, input_hashes):
    columns = pd.MultiIndex.from_product([data.codes, data.manifest["feature_cols"]])
    price_columns = pd.MultiIndex.from_product([data.codes, ("open", "close")])
    boundary = pd.Timestamp(data.manifest["train_end"])
    expected_hashes = dict(input_hashes,
        truncated_features=_frame_digest(data.features.loc[data.features.index <= boundary, columns]),
        truncated_open_close=_frame_digest(data.prices.loc[data.prices.index <= boundary, price_columns]))
    if (pool.codes != tuple(data.codes) or pool.feature_cols != tuple(data.manifest["feature_cols"])
            or pool.train_end != data.manifest["train_end"] or pool.horizon != 5 or pool.common_lookback != 5
            or data.config["lookback"] != 1 or pool.input_hashes != expected_hashes):
        raise ValueError("Train-only label order, training boundary or input identity differs from the environment")
    return pool


def load_agent(directory, data):
    agent = PriorityAgent.load(directory, data.contract, "cpu", algo_cls=RankAuxPPO)
    if agent.model.policy.actor_variant != data.config["priority_rank"]["actor_variant"]:
        raise ValueError("Saved actor variant differs from prediction contract")
    return agent


def _reference(reference_root):
    reference_root = Path(reference_root).resolve()
    if file_hash(reference_root / "final_audit.json") != REFERENCE_AUDIT_SHA256:
        raise ValueError("Trusted market-summary final audit changed")
    audit = read_config(reference_root / "final_audit.json")
    reference = read_config(reference_root / "protocol.json")
    if audit["status"] != "passed" or file_hash(reference_root / "protocol.json") != audit["anchors"]["protocol.json"]:
        raise ValueError("Reference registration differs from sealed audit")
    for name, digest in reference["source_hashes"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("A sealed original source changed: " + name)
    return reference


def check_frozen(protocol):
    if source_hashes() != protocol["source_hashes"] or file_hash(DOCUMENT) != protocol["document_sha256"]:
        raise ValueError("Registered source or preregistration changed")
    reference = Path(protocol["reference_root"])
    _reference(reference)
    if file_hash(reference / "protocol.json") != protocol["reference_protocol_sha256"]:
        raise ValueError("Trusted reference protocol changed")
    for fold in protocol["folds"].values():
        directory = Path(fold["directory"])
        if file_hash(directory / "manifest.json") != fold["manifest_sha256"]:
            raise ValueError("Historical fold manifest changed")
        _verify_files(directory, fold["files"])
        pool = Path(fold["label_pool"]["directory"])
        if file_hash(pool / "manifest.json") != fold["label_pool"]["manifest_sha256"]:
            raise ValueError("Train-only label manifest changed")
        _verify_files(pool, fold["label_pool"]["files"])


def prepare(root, reference_root, profile="standard"):
    root, reference_root = Path(root).resolve(), Path(reference_root).resolve()
    if profile not in PROFILES or root.is_relative_to(reference_root) or reference_root.is_relative_to(root):
        raise ValueError("Invalid study profile or overlapping reference/output")
    reference = _reference(reference_root)
    path = root / "protocol.json"
    if path.exists():
        protocol = read_config(path)
        if (protocol["profile"] != profile or protocol["reference_root"] != str(reference_root)
                or protocol["budgets"] != PROFILES[profile]):
            raise ValueError("Registration cannot change its profile or reference")
        check_frozen(protocol)
        return protocol
    if root.exists():
        raise ValueError("Unregistered output directory requires explicit review")
    folds = {}
    # Labels are built before source/data registration, physically from train-only arrays.
    for fold in FOLDS:
        old = reference["folds"][fold]
        directory = Path(old["directory"])
        if file_hash(directory / "manifest.json") != old["manifest_sha256"]:
            raise ValueError("Historical manifest differs from sealed reference")
        _verify_files(directory, old["files"])
        manifest = read_config(directory / "manifest.json")
        if (not manifest.get("protected_period_excluded") or manifest["end_date"] > "2026-06-30"
                or manifest["train_end"] != f"{fold}-03-31"):
            raise ValueError("Invalid historical training boundary")
        data = PriorityData(config_for(reference["base_config"], reference_root, root, fold, "coupled", "ppo", profile))
        files = {name: file_hash(directory / name) for name in
                 ("panel_data.parquet", "prices.parquet", "scaler_params.json", "research.json")}
        pool = build_rank_pool(data.features, data.prices, data.codes, data.manifest["feature_cols"],
                               data.manifest["train_end"], common_lookback=5, horizon=5, input_hashes=files)
        validate_pool(pool, data, files)
        label_directory = root / "labels" / fold
        pool.save(label_directory)
        folds[fold] = dict(directory=str(directory), manifest_sha256=file_hash(directory / "manifest.json"), files=files,
            label_pool=dict(directory=str(label_directory), manifest_sha256=file_hash(label_directory / "manifest.json"),
                            identity_sha256=pool.identity_sha256, files=_registered_files(label_directory)))
    budgets = PROFILES[profile]
    protocol = dict(schema_version=1, study="priority_rank_factorial", profile=profile,
        actors=list(ACTORS), methods=list(METHODS), seeds=list(SEEDS), budgets=budgets, folds=folds,
        base_config=reference["base_config"], reference_root=str(reference_root),
        reference_protocol_sha256=file_hash(reference_root / "protocol.json"),
        reference_final_audit_sha256=REFERENCE_AUDIT_SHA256, original_source_count=len(reference["source_hashes"]),
        source_hashes=source_hashes(), document_sha256=file_hash(DOCUMENT), git=git_metadata(), parameters=PARAMETERS,
        planned_trials=24, planned_snapshots=24 * len(budgets), planned_actual_steps=24 * max(budgets.values()),
        planned_initialized_paths=48, planned_rule_paths=24, planned_trained_evaluation_paths=24 * len(budgets) * 4,
        evaluation_seed=EVAL_SEED, main_top_n=20, auxiliary_seed_salt=AUX_SEED_SALT,
        auxiliary_coefficient=.05, auxiliary_batch_size=1024, auxiliary_horizon=5, auxiliary_huber_delta=.1,
        learning_rate=3e-5, gradient_diagnostics=False, test_evaluated=False, actionable=False,
        gate=dict(max_all_account_drawdown=.1, each_fold_mean_gain_positive=True,
                  each_fold_positive_learning_seeds=2, each_fold_beats_same_economic_20day_rule=True),
        representative_seed=42, representative_fold="2024")
    write_json(path, protocol)
    check_frozen(protocol)
    return protocol


class _ExecutionAdapter:
    """Select the original order ledger without changing the agent's 3N contract.

    run_policy's stock_trade branch accepts a fully mapped TradeOrder directly;
    it never applies the old 2N mapper on this path. Observations and broker
    execution are therefore reused exactly, including independent buy priorities.
    """

    def __init__(self, agent):
        self.agent = agent
        self.contract = deepcopy(agent.contract)
        self.contract["action_mode"] = "stock_trade"

    def act(self, observation, step=None):
        return self.agent.act(observation, step)


def evaluate_case(agent, data, folder, label):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    actor = data.config["priority_rank"]["actor_variant"]
    identity = _trace_identity(data, label, seed=EVAL_SEED, initial_mode=data.settings["initial_portfolio"],
                               top_n=data.settings["trade_top_n"], actor_variant=actor)
    trace = PriorityTrace(folder / "trace", identity)
    market = data.market("val")

    def capture(date, observation, action, info):
        log_prob, value = None, None
        if isinstance(agent, PriorityAgent):
            tensor, _ = agent.model.policy.obs_to_tensor(observation)
            raw = action.diagnostics["raw_packed_action"]
            packed = torch.as_tensor(raw, dtype=torch.float32, device=agent.model.device).unsqueeze(0)
            with torch.no_grad():
                predicted, probability, _ = agent.model.policy.evaluate_actions(tensor, packed)
            value, log_prob = predicted.cpu().numpy()[0], probability.cpu().numpy()[0]
        trace.append(date, observation, action, info, log_prob=log_prob, value=value,
                     reward=info["reward"], done=market.current_index == market.n_dates - 1)

    try:
        daily, holdings, orders, assumptions = run_policy(market, _ExecutionAdapter(agent), data.settings, trace_sink=capture)
    finally:
        trace.close()
    daily.to_csv(folder / "daily.csv", index=False)
    holdings.to_parquet(folder / "positions.parquet")
    orders.to_csv(folder / "orders.csv", index=False)
    metrics = performance(daily)
    verify_priority_trace(folder / "trace", trace.manifest(), actor_variant=actor, steps=len(daily) - 1)
    result = dict(metrics=metrics, mode_statistics=trace.statistics.snapshot(), trace=trace.manifest(),
        contract=data.contract, assumptions=assumptions, initial_mode=data.settings["initial_portfolio"],
        top_n=data.settings["trade_top_n"], first_execution=str(daily.date.iloc[1].date()),
        last_execution=str(daily.date.iloc[-1].date()), test_evaluated=False, actionable=False)
    write_json(folder / "result.json", result)
    return result


class PriorityEconomicRule:
    """Lift the exact existing momentum orders into canonical coupled 3N samples."""

    def __init__(self, contract, kind="hold", interval=1):
        self.contract = deepcopy(contract)
        self.rule = EconomicRule(contract, kind, interval)

    def act(self, obs, step=None):
        original = self.rule.act(obs, step)
        n = len(obs["portfolio"]) - 1
        raw = np.asarray(original.diagnostics["latent"])
        priorities = np.where(raw[:n] == 1, raw[n:], 0.)
        packed = np.concatenate([raw, priorities])
        result = map_priority_trade_action(packed, obs, self.contract["logit_bound"],
            self.contract["broker"]["lot_size"], self.contract["stock_codes"], actor_variant="coupled")
        for name in ("shares", "priority", "freeze_mask", "exit_mask"):
            if not np.array_equal(result[name], getattr(original, name)):
                raise ValueError("Lifted rule changed original economic orders")
        return TradeOrder(result["shares"], result["freeze_mask"], result["exit_mask"], result["priority"], result["top_n"], result)


def rule_controls(root, protocol, fold, *, create=True):
    directory = Path(root) / "rules" / fold
    marker = directory / "completed.json"
    if marker.exists():
        saved = read_config(marker)
        if saved["protocol_sha256"] != json_digest(protocol):
            raise ValueError("Economic controls belong to another protocol")
        _verify_files(directory, saved["files"])
        for mode in MODES:
            for name, metrics in saved["metrics"][mode].items():
                if read_config(directory / mode / name / "result.json")["metrics"] != metrics:
                    raise ValueError("Economic control metrics changed")
        return saved["metrics"]
    if not create or directory.exists():
        raise ValueError("Missing or incomplete controls require explicit review")
    data = PriorityData(trial_config(protocol, root, fold, "coupled", "ppo", 42))
    metrics = {}
    for mode in MODES:
        data.settings.update(seed=EVAL_SEED, initial_portfolio=mode, trade_top_n=20)
        metrics[mode] = {}
        for name, kind, interval in (("hold", "hold", 1), ("momentum_daily", "momentum_20", 1),
                                     ("momentum_20day", "momentum_20", 20)):
            metrics[mode][name] = evaluate_case(PriorityEconomicRule(data.contract, kind, interval), data,
                directory / mode / name, "priority_economic_control")["metrics"]
    write_json(marker, dict(protocol_sha256=json_digest(protocol), metrics=metrics, files=_registered_files(directory)))
    return metrics


def _policy_bytes(directory):
    with zipfile.ZipFile(Path(directory) / "model.zip") as archive:
        return archive.read("policy.pth")


def _canonical_contract(contract):
    result = deepcopy(contract)
    result["priority_rank"].pop("training_method")
    result["priority_rank"].pop("auxiliary_coefficient")
    return result


def initial_controls(root, protocol, data, initial_path):
    spec = data.config["priority_rank"]
    directory = Path(root) / "initial_controls" / f"{spec['fold']}_{spec['actor_variant']}_{data.config['seed']}"
    identity = dict(protocol_sha256=json_digest(protocol), prediction_contract=_canonical_contract(data.contract),
        initial_policy_pth_sha256=hashlib.sha256(_policy_bytes(initial_path)).hexdigest(),
        evaluation_seed=EVAL_SEED, modes=list(MODES), top_n=20)
    marker = directory / "completed.json"
    if marker.exists():
        saved = read_config(marker)
        if saved["identity"] != identity:
            raise ValueError("Initialization is not the same actor, function, theta or canonical contract")
        _verify_files(directory, saved["files"])
        for mode in MODES:
            if read_config(directory / mode / "result.json")["metrics"] != saved["metrics"][mode]:
                raise ValueError("Initialized control metrics changed")
        return saved["metrics"], directory
    if directory.exists():
        raise ValueError("Incomplete initialized controls require explicit review")
    agent = load_agent(initial_path, data)
    if agent.model.num_timesteps != 0:
        raise ValueError("Initialized controls cannot use a trained checkpoint")
    metrics = {}
    for mode in MODES:
        data.settings.update(seed=EVAL_SEED, initial_portfolio=mode, trade_top_n=20)
        metrics[mode] = evaluate_case(agent, data, directory / mode, "priority_initialized")["metrics"]
    write_json(marker, dict(identity=identity, metrics=metrics, files=_registered_files(directory)))
    del agent
    _release()
    return metrics, directory


class PriorityCallback(ExplorationCallback):
    """Fixed post-update checkpoints with original 3N samples and separate aux counts."""

    def __init__(self, *args, **kwargs):
        # Reuse diagnostics without constructing the legacy constructor's 2N trace.
        MatrixCallback.__init__(self, *args, **kwargs)
        self.annealing, self.schedule = False, []
        self.all_actions_probabilities_finite = True
        identity = _trace_identity(self.data, "sampled_training", run_id=self.run_id,
                                   actor_variant=self.data.config["priority_rank"]["actor_variant"])
        self.trace = PriorityTrace(self.output / "training_trace", identity)

    def record_update(self):
        audit = self.model.auxiliary_audit()
        self.model.logger.record("train/auxiliary_updates", audit["actual_aux_updates"])
        priority_sigma = float(self.model.policy.network.priority_log_std.detach().exp().cpu().item())
        self.model.logger.record("train/independent_priority_sigma", priority_sigma if self.model.policy.actor_variant == "decoupled" else 0.)
        if self.critic_rollout is not None:
            values = []
            with torch.no_grad():
                count = len(self.critic_rollout.returns.reshape(-1))
                for start in range(0, count, self.model.batch_size):
                    obs = {name: array.reshape((-1, *self.model.observation_space[name].shape))[start:start + self.model.batch_size]
                           for name, array in self.critic_rollout.observations.items()}
                    tensor, _ = self.model.policy.obs_to_tensor(obs)
                    distribution = self.model.policy.get_distribution(tensor)
                    values.extend(distribution.conditional_priority_entropy().cpu().numpy().reshape(-1))
            self.model.logger.record("train/post_update_conditional_priority_entropy", float(np.mean(values)))
        super().record_update()

    def save_fixed_snapshot(self):
        self.trace.flush()
        for budget, steps in self.budgets.items():
            if budget in self.saved or self.model.num_timesteps != steps:
                continue
            agent = PriorityAgent(self.model, self.data.contract)
            path = self.output / (budget + "_model")
            agent.save(path, self.data.config, dict(priority_rank_budget=budget, prefix_shared=True,
                                                  auxiliary_audit=self.model.auxiliary_audit()))
            with preserve_training_rng(self.model.device):
                restored = load_agent(path, self.data)
                for name, value in self.model.policy.state_dict().items():
                    if not torch.equal(value.detach().cpu(), restored.model.policy.state_dict()[name]):
                        raise AssertionError("Saved priority parameters changed during loading")
                observation = {name: value[0] for name, value in self.model._last_obs.items()}
                was_training = self.model.policy.training
                np.testing.assert_allclose(agent.inspect_action(observation)["latent"],
                    restored.inspect_action(observation)["latent"], rtol=1e-5, atol=1e-6)
                self.model.policy.set_training_mode(was_training)
                del restored
            deltas = {name: float(torch.linalg.vector_norm(value.detach().cpu() - self.initial[name]))
                      for name, value in self.model.policy.state_dict().items()}
            optimizer = [int(state["step"].item()) for state in self.model.policy.optimizer.state.values() if "step" in state]
            if not optimizer or min(optimizer) <= 0 or not all(torch.isfinite(p).all() for p in self.model.policy.parameters()):
                raise AssertionError("Snapshot requires finite trained parameters")
            cuda = self.model.device.type == "cuda"
            record = dict(actual_timesteps=steps, model_sha256=file_hash(path / "model.zip"),
                parameters=sum(p.numel() for p in self.model.policy.parameters()), elapsed_seconds=perf_counter() - self.started,
                optimizer_steps_min=min(optimizer), optimizer_steps_max=max(optimizer),
                optimizer_steps_by_parameter={name: int(self.model.policy.optimizer.state[p]["step"].item())
                    for name, p in self.model.policy.named_parameters() if p in self.model.policy.optimizer.state},
                tensor_delta_l2=deltas, changed_tensors=sum(value > 0 for value in deltas.values()),
                all_parameters_finite=True, save_load_parameters_equal=True, save_load_latent_close=True,
                experience=self.experience.stats(), training_diagnostics=diagnostic_summary(self.update_records),
                auxiliary_audit=self.model.auxiliary_audit(), training_trace_prefix=self.trace.manifest(),
                mode_statistics=self.trace.statistics.snapshot(), all_actions_probabilities_finite=True,
                valid_observation_steps={c: int(n) for c, n in zip(self.data.codes, self.valid_observation_steps)},
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.model.device) if cuda else None,
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.model.device) if cuda else None, test_evaluated=False)
            if record["experience"]["total_steps"] != steps:
                raise AssertionError("Experience differs from actual training steps")
            write_json(self.output / (budget + "_audit.json"), record)
            write_json(self.output / (budget + "_episodes.json"), [e for e in self.experience.episodes if e["steps"]])
            self.saved.add(budget)


def validate_auxiliary_audit(audit, config, metadata=None):
    spec = config["priority_rank"]
    aux = audit["auxiliary_audit"]
    joint, auxiliary = aux.get("actual_joint_optimizer_updates"), aux.get("actual_aux_updates")
    if (aux.get("aux_coef") != spec["auxiliary_coefficient"] or aux.get("batch_size") != spec["auxiliary_batch_size"]
            or aux.get("huber_delta") != spec["auxiliary_huber_delta"] or aux.get("seed") != config["seed"]
            or aux.get("seed_salt") != AUX_SEED_SALT or aux.get("pool_identity_sha256") != spec["pool_identity"]["identity_sha256"]
            or type(joint) is not int or joint <= 0 or type(auxiliary) is not int
            or auxiliary != (joint if spec["training_method"] == "rank_aux" else 0)
            or aux.get("supervised_stock_date_exposures") != auxiliary * spec["auxiliary_batch_size"]
            or aux.get("extra_optimizer_steps") != 0 or aux.get("auxiliary_rl_steps") != 0
            or aux.get("supervision_in_observations") is not False
            or audit["optimizer_steps_min"] != joint or audit["optimizer_steps_max"] != joint):
        raise ValueError("Saved auxiliary updates or training method differ from the registered condition")
    if metadata is not None:
        policy = metadata.get("priority_policy", {})
        if (metadata.get("algorithm_class") != "src.training.priority_aux.RankAuxPPO"
                or metadata.get("extra", {}).get("auxiliary_audit") != aux
                or any(policy.get(key) != spec["actor_variant"] for key in
                       ("actor_variant", "network_actor_variant", "distribution_actor_variant"))):
            raise ValueError("Saved algorithm, policy flags or auxiliary metadata do not match its audit")
    return aux


def validate_case_identity(result, contract, mode, *, initialized=False):
    actual = result["contract"]
    matching = _canonical_contract(actual) == _canonical_contract(contract) if initialized else actual == contract
    identity = result["trace"]["identity"]
    if (not matching or identity.get("contract_sha256") != json_digest(actual)
            or identity.get("codes") != actual["stock_codes"]
            or identity.get("actor_variant") != actual["trade"]["actor_variant"]
            or identity.get("seed") != EVAL_SEED or identity.get("initial_mode") != mode
            or identity.get("top_n") != 20 or result.get("initial_mode") != mode
            or result.get("top_n") != 20 or result.get("test_evaluated") is not False):
        raise ValueError("Evaluation curve or trace belongs to a different contract, pool or account")
    return result


def cached_trial(directory, config, protocol):
    directory = Path(directory)
    marker = directory / "completed.json"
    if not marker.exists():
        return None
    saved = read_config(marker)
    if saved["config_sha256"] != json_digest(config) or saved["protocol_sha256"] != json_digest(protocol):
        raise ValueError("Trial cache configuration changed")
    _verify_files(directory, saved["files"])
    if read_config(directory / "config.json") != config:
        raise ValueError("Cached native configuration changed")
    expected = set(protocol["budgets"].items())
    rows = saved["records"]
    if len(rows) != len(expected) or {(r["budget"], r["steps"]) for r in rows} != expected:
        raise ValueError("Cached trial lacks exactly the registered snapshots")
    initial = read_config(directory / "initial_model" / "metadata.json")
    spec = config["priority_rank"]
    if (initial["config"] != config or initial["num_timesteps"] != 0
            or initial["source_hashes"] != protocol["source_hashes"]
            or initial["contract"].get("priority_rank") != spec
            or initial.get("algorithm_class") != "src.training.priority_aux.RankAuxPPO"
            or any(initial.get("priority_policy", {}).get(key) != spec["actor_variant"] for key in
                   ("actor_variant", "network_actor_variant", "distribution_actor_variant"))
            or file_hash(directory / "initial_model" / "model.zip") != initial["model_sha256"]):
        raise ValueError("Cached initialized model identity changed")
    controls_path = directory.parent.parent / "initial_controls" / f"{spec['fold']}_{spec['actor_variant']}_{config['seed']}"
    controls = read_config(controls_path / "completed.json")
    _verify_files(controls_path, controls["files"])
    identity = dict(protocol_sha256=json_digest(protocol), prediction_contract=_canonical_contract(initial["contract"]),
        initial_policy_pth_sha256=hashlib.sha256(_policy_bytes(directory / "initial_model")).hexdigest(),
        evaluation_seed=EVAL_SEED, modes=list(MODES), top_n=20)
    if controls["identity"] != identity:
        raise ValueError("Cached initial controls differ from exact actor initialization")
    for row in rows:
        if (row["fold"], row["actor"], row["method"], row["seed"]) != (spec["fold"], spec["actor_variant"], spec["training_method"], config["seed"]):
            raise ValueError("Cached row identifies another condition")
        audit = read_config(directory / (row["budget"] + "_audit.json"))
        metadata = read_config(directory / (row["budget"] + "_model") / "metadata.json")
        if (audit["actual_timesteps"] != row["steps"] or metadata["num_timesteps"] != row["steps"]
                or metadata["config"] != config or metadata["contract"] != initial["contract"]
                or metadata["source_hashes"] != protocol["source_hashes"] or audit["parameters"] != protocol["parameters"]
                or metadata["model_sha256"] != row["model_sha256"] or audit["model_sha256"] != row["model_sha256"]
                or file_hash(directory / (row["budget"] + "_model") / "model.zip") != row["model_sha256"]):
            raise ValueError("Cached row differs from saved model snapshot")
        validate_auxiliary_audit(audit, config, metadata)
        for field in ("parameters", "optimizer_steps_min", "optimizer_steps_max", "optimizer_steps_by_parameter",
                      "experience", "training_diagnostics", "auxiliary_audit", "training_trace_prefix", "tensor_delta_l2"):
            if row[field] != audit[field]:
                raise ValueError("Cached training evidence changed: " + field)
        if row["training_mode_statistics"] != audit["mode_statistics"]:
            raise ValueError("Cached native training mode statistics changed")
        verify_priority_trace(directory / "training_trace", row["training_trace_prefix"],
                              actor_variant=spec["actor_variant"], steps=row["steps"])
        if (ROOT / row["initial_controls_path"]).resolve() != controls_path.resolve():
            raise ValueError("Cached control path belongs to a different actor/fold/seed")
        for mode in MODES:
            result = read_config(directory / row["budget"] / mode / "result.json")
            initialized = read_config(controls_path / mode / "result.json")
            validate_case_identity(result, initial["contract"], mode)
            validate_case_identity(initialized, initial["contract"], mode, initialized=True)
            if (row["metrics"][mode]["ppo"] != result["metrics"]
                    or row["metrics"][mode]["initialized"] != initialized["metrics"]
                    or row["metrics"][mode]["trace"] != result["trace"]
                    or row["metrics"][mode]["mode_statistics"] != result["mode_statistics"]
                    or controls["metrics"][mode] != initialized["metrics"]):
                raise ValueError("Cached metrics differ from registered evaluations")
            verify_priority_trace(directory / row["budget"] / mode / "trace", result["trace"],
                                  actor_variant=spec["actor_variant"])
            verify_priority_trace(controls_path / mode / "trace", initialized["trace"],
                                  actor_variant=spec["actor_variant"])
    return rows


def run_trial(root, protocol, fold, actor, method, seed):
    root = Path(root)
    config = trial_config(protocol, root, fold, actor, method, seed)
    run_id = f"{fold}_{actor}_{method}_{seed}"
    directory = root / "runs" / run_id
    cached = cached_trial(directory, config, protocol)
    if cached is not None:
        return cached
    if directory.exists():
        raise ValueError("Incomplete trial requires explicit review")
    check_frozen(protocol)
    directory.mkdir(parents=True)
    write_json(directory / "config.json", config)
    (directory / "events").mkdir()
    events.configure(directory / "events")
    data = PriorityData(config)
    pool_spec = protocol["folds"][fold]["label_pool"]
    pool = TrainRankPool.load(pool_spec["directory"], expected_manifest_sha256=pool_spec["manifest_sha256"])
    if pool.identity_sha256 != pool_spec["identity_sha256"]:
        raise ValueError("Train-only label pool identity changed")
    validate_pool(pool, data, protocol["folds"][fold]["files"])
    experience = EpisodeAudit(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    env.seed(seed)
    callback = None
    try:
        model = make_model(env, config, pool)
        initial = {name: tensor.detach().cpu().clone() for name, tensor in model.policy.state_dict().items()}
        if sum(p.numel() for p in model.policy.parameters()) != protocol["parameters"]:
            raise ValueError("Registered parameter count changed")
        PriorityAgent(model, data.contract).save(directory / "initial_model", config)
        policy_sha = hashlib.sha256(_policy_bytes(directory / "initial_model")).hexdigest()
        for other_actor in ACTORS:
            for other_method in METHODS:
                other = root / "runs" / f"{fold}_{other_actor}_{other_method}_{seed}" / "initial_model"
                if other.exists() and _policy_bytes(other) != _policy_bytes(directory / "initial_model"):
                    raise ValueError("Factorial cells must begin with byte-identical parameter tensors")
        write_json(directory / "initial_identity.json", dict(policy_pth_sha256=policy_sha,
            equivalence="Same parameters; deterministic initial functions equal, stochastic coupled/decoupled supports differ."))
        model.set_logger(configure(str(directory / "logs"), ["csv"]))
        callback = PriorityCallback(data, directory, experience, initial, protocol["budgets"], directory / "status.json", run_id)
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        model.learn(total_timesteps=max(protocol["budgets"].values()), callback=callback)
        model.logger.dump(step=model.num_timesteps)
        if callback.saved != set(protocol["budgets"]) or callback.trace.statistics.violations:
            raise AssertionError("Missing registered snapshots or economic execution violations")
    finally:
        env.close()
        if callback is not None:
            callback.trace.close()
        events.configure(None)
    del model, callback, initial, env, experience, pool
    _release()
    initial_metrics, controls = initial_controls(root, protocol, data, directory / "initial_model")
    rows = []
    for budget, steps in protocol["budgets"].items():
        agent = load_agent(directory / (budget + "_model"), data)
        metrics = {}
        for mode in MODES:
            data.settings.update(seed=EVAL_SEED, initial_portfolio=mode, trade_top_n=20)
            write_json(directory / "status.json", dict(status="evaluating", budget=budget, account=mode, test_evaluated=False))
            result = evaluate_case(agent, data, directory / budget / mode, "priority_trained")
            metrics[mode] = dict(ppo=result["metrics"], initialized=initial_metrics[mode],
                                mode_statistics=result["mode_statistics"], trace=result["trace"])
        audit = read_config(directory / (budget + "_audit.json"))
        row = dict(fold=fold, actor=actor, method=method, seed=seed, budget=budget, steps=steps,
            model_path=str((directory / (budget + "_model")).relative_to(ROOT)), model_sha256=audit["model_sha256"],
            initial_controls_path=str(controls.relative_to(ROOT)), metrics=metrics, test_evaluated=False)
        for field in ("parameters", "optimizer_steps_min", "optimizer_steps_max", "optimizer_steps_by_parameter",
                      "experience", "training_diagnostics", "auxiliary_audit", "training_trace_prefix", "tensor_delta_l2"):
            row[field] = audit[field]
        row["training_mode_statistics"] = audit["mode_statistics"]
        rows.append(row)
        del agent
        _release()
    check_frozen(protocol)
    write_json(directory / "status.json", dict(status="complete", test_evaluated=False))
    write_json(directory / "completed.json", dict(config_sha256=json_digest(config), protocol_sha256=json_digest(protocol),
                                                 records=rows, files=_registered_files(directory)))
    return rows


def validate_records(records, budgets):
    if budgets not in PROFILES.values():
        raise ValueError("Unregistered snapshot budget profile")
    expected = {(fold, actor, method, seed, budget, steps) for fold in FOLDS for actor in ACTORS for method in METHODS
                for seed in SEEDS for budget, steps in budgets.items()}
    actual = [(r["fold"], r["actor"], r["method"], r["seed"], r["budget"], r["steps"]) for r in records]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Results require every registered condition and fixed checkpoint exactly once")
    for row in records:
        if set(row["metrics"]) != set(MODES):
            raise ValueError("Every checkpoint requires all four initial accounts")
        for account in row["metrics"].values():
            for name in ("ppo", "initialized"):
                metric = account[name]
                for key in ("cumulative_return", "max_drawdown", "total_fees", "total_slippage_cost"):
                    value = metric.get(key)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                        raise ValueError("Checkpoint contains missing or nonfinite economic metrics")
    return dict(actual_training_steps=24 * max(budgets.values()), trained_runs=24, snapshots=len(records),
                checkpoint_steps_sum=sum(r["steps"] for r in records))


def _factorial(per_seed):
    values = {key: np.asarray([per_seed[key][str(seed)] for seed in SEEDS]) for key in per_seed}
    cp, ca, dp, da = (values[key] for key in ("coupled_ppo", "coupled_rank_aux", "decoupled_ppo", "decoupled_rank_aux"))
    effects = dict(actor_decoupled_minus_coupled=((dp - cp) + (da - ca)) / 2,
        method_rank_aux_minus_ppo=((ca - cp) + (da - dp)) / 2,
        interaction=(da - dp) - (ca - cp), actor_under_ppo=dp - cp,
        actor_under_rank_aux=da - ca, method_under_coupled=ca - cp, method_under_decoupled=da - dp)
    return {name: dict(_stats(value.tolist()), by_seed=dict(zip(map(str, SEEDS), value.tolist()))) for name, value in effects.items()}


def summarize(records, rules, budgets):
    counts = validate_records(records, budgets)
    folds, overall, effects = {}, {}, {}
    for fold in FOLDS:
        folds[fold] = {}
        for actor in ACTORS:
            for method in METHODS:
                for budget, steps in budgets.items():
                    rows = sorted([r for r in records if (r["fold"], r["actor"], r["method"], r["budget"]) ==
                                   (fold, actor, method, budget)], key=lambda r: r["seed"])
                    means = [_metric_mean([r["metrics"][mode]["ppo"] for mode in MODES]) for r in rows]
                    initial = [_metric_mean([r["metrics"][mode]["initialized"] for mode in MODES])["cumulative_return"] for r in rows]
                    gains = [m["cumulative_return"] - start for m, start in zip(means, initial)]
                    worst = max(abs(r["metrics"][mode]["ppo"]["max_drawdown"]) for r in rows for mode in MODES)
                    benchmark = _metric_mean([rules[fold][mode]["momentum_20day"] for mode in MODES])["cumulative_return"]
                    metric = _metric_stats(means)
                    folds[fold][f"{actor}_{method}_{budget}"] = dict(actor=actor, method=method, budget=budget, steps=steps,
                        metrics=metric, initialization=_stats(initial), own_initialization_gain=_stats(gains),
                        positive_learning_seeds=sum(x > 0 for x in gains), worst_account_drawdown=worst,
                        twenty_day_gap=metric["cumulative_return"]["mean"] - benchmark,
                        qualified=bool(worst <= .1 and np.mean(gains) > 0 and sum(x > 0 for x in gains) >= 2
                                       and metric["cumulative_return"]["mean"] > benchmark),
                        per_seed={str(r["seed"]): dict(metrics=m, initialization=start, gain=gain)
                                  for r, m, start, gain in zip(rows, means, initial, gains)})
        for actor in ACTORS:
            for budget in budgets:
                for seed in SEEDS:
                    if folds[fold][f"{actor}_ppo_{budget}"]["per_seed"][str(seed)]["initialization"] != folds[fold][f"{actor}_rank_aux_{budget}"]["per_seed"][str(seed)]["initialization"]:
                        raise ValueError("Methods sharing an actor must have identical initialized controls")
    for actor in ACTORS:
        for method in METHODS:
            for budget, steps in budgets.items():
                key = f"{actor}_{method}_{budget}"
                per_seed = {str(seed): dict(metrics=_metric_mean([folds[fold][key]["per_seed"][str(seed)]["metrics"] for fold in FOLDS]),
                    initialization=float(np.mean([folds[fold][key]["per_seed"][str(seed)]["initialization"] for fold in FOLDS])),
                    gain=float(np.mean([folds[fold][key]["per_seed"][str(seed)]["gain"] for fold in FOLDS]))) for seed in SEEDS}
                overall[key] = dict(actor=actor, method=method, budget=budget, steps=steps,
                    metrics=_metric_stats([r["metrics"] for r in per_seed.values()]),
                    initialization=_stats([r["initialization"] for r in per_seed.values()]),
                    own_initialization_gain=_stats([r["gain"] for r in per_seed.values()]), per_seed=per_seed,
                    qualified=all(folds[fold][key]["qualified"] for fold in FOLDS),
                    worst_account_drawdown=max(folds[fold][key]["worst_account_drawdown"] for fold in FOLDS))
    for context, items in [*folds.items(), ("across_fold", overall)]:
        effects[context] = {}
        for budget in budgets:
            decomposition = {}
            for outcome in ("net_roi", "initialization", "learning_gain"):
                values = {}
                for actor in ACTORS:
                    for method in METHODS:
                        source = items[f"{actor}_{method}_{budget}"]["per_seed"]
                        values[f"{actor}_{method}"] = {seed: row["metrics"]["cumulative_return"] if outcome == "net_roi"
                            else row["initialization"] if outcome == "initialization" else row["gain"] for seed, row in source.items()}
                decomposition[outcome] = _factorial(values)
            for name in decomposition["net_roi"]:
                for seed in map(str, SEEDS):
                    if not np.isclose(decomposition["net_roi"][name]["by_seed"][seed],
                        decomposition["initialization"][name]["by_seed"][seed] + decomposition["learning_gain"][name]["by_seed"][seed], atol=1e-12, rtol=0):
                        raise ValueError("Factorial net/init/learning decomposition does not reconcile")
            effects[context][budget] = decomposition
    eligible = sorted([x for x in overall.values() if x["qualified"]], key=lambda x:
        (-x["metrics"]["cumulative_return"]["mean"], x["worst_account_drawdown"],
         x["metrics"]["total_fees"]["mean"] + x["metrics"]["total_slippage_cost"]["mean"], x["steps"], x["actor"], x["method"]))
    candidate = dict(actor=eligible[0]["actor"], method=eligible[0]["method"], budget=eligible[0]["budget"],
                     representative_seed=42, representative_fold="2024") if eligible else None
    return dict(**counts, fold_cells=folds, cells=overall, factorial_effects=effects, qualified_candidate=candidate,
        actionable=False, test_evaluated=False, aggregation="Four account mean within seed/fold; mean across two folds within seed; sample SD across three paired seeds.",
        limitations=["Repeatedly exposed historical folds are development evidence, not independent forward testing.",
            "Historical fixed 2026 membership and HFQ-only economic proxy remain.",
            "Coupled/decoupled stochastic initial distributions differ despite equal deterministic initial functions.",
            "Auxiliary labels use only training rows; supervised optimizer steps are separate from RL experience."])


def write_report(root, records, rules, protocol):
    root = Path(root)
    summary = summarize(records, rules, protocol["budgets"])
    write_json(root / "summary.json", records)
    write_json(root / "study_summary.json", summary)
    pd.DataFrame([dict(fold=r["fold"], actor=r["actor"], method=r["method"], seed=r["seed"], budget=r["budget"],
        steps=r["steps"], account=mode, **metrics["ppo"]) for r in records for mode, metrics in r["metrics"].items()]).to_csv(root / "metrics.csv", index=False)
    lines = ["# Priority/ranking factorial research", "", "Historical development evidence; no protected final test or deployment.", "",
        "| Fold | Actor | Method | Steps | ROI % +/- seed SD | Own-init gain pp | Worst DD % | Qualified |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for fold, items in summary["fold_cells"].items():
        for item in items.values():
            roi = item["metrics"]["cumulative_return"]
            lines.append(f"| {fold} | {item['actor']} | {item['method']} | {item['steps']} | {roi['mean']*100:.4f} +/- {roi['std']*100:.4f} | "
                f"{item['own_initialization_gain']['mean']*100:.4f} | {item['worst_account_drawdown']*100:.4f} | {item['qualified']} |")
    lines.extend(["", f"Registered candidate: {summary['qualified_candidate']}", "", "No automatic promotion. Extra supervised updates do not count as RL steps.",
                  "[Machine summary](study_summary.json) | [All accounts](metrics.csv) | [Raw records](summary.json)"])
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(root / "status.json", dict(status="complete", **validate_records(records, protocol["budgets"]), test_evaluated=False, actionable=False))
    return summary


def verify_worker_receipt(root, protocol, fold):
    root = Path(root)
    receipt = read_config(root / f"worker_{fold}.json")
    expected = {f"{fold}_{actor}_{method}_{seed}" for actor in ACTORS for method in METHODS for seed in SEEDS}
    completed = receipt["completed_runs"]
    if (receipt["status"] != "complete" or receipt["fold"] != fold or len(completed) != 12
            or {item["run_id"] for item in completed} != expected
            or receipt["protocol_sha256"] != file_hash(root / "protocol.json")
            or receipt.get("no_global_summary_writes") is not True):
        raise ValueError("Both exclusive fold workers must seal all registered trials before reporting")
    for item in completed:
        if file_hash(root / "runs" / item["run_id"] / "completed.json") != item["completed_sha256"]:
            raise ValueError("Worker trial completion identity changed")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "report"), required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="standard")
    parser.add_argument("--output")
    parser.add_argument("--reference-root", default="reports/runs/ppo_market_summary_20261007")
    parser.add_argument("--fold", choices=FOLDS)
    args = parser.parse_args(argv)
    root = Path(args.output or f"reports/runs/ppo_priority_rank_20261011_{'16k' if args.profile == 'standard' else '32k'}").resolve()
    if args.stage == "train" and args.fold is None or args.stage != "train" and args.fold is not None:
        parser.error("Exactly one --fold is required only for independent train workers")
    if args.stage != "prepare" and not (root / "protocol.json").exists():
        raise ValueError("Only prepare may create a registration")
    protocol = prepare(root, args.reference_root, args.profile)
    check_frozen(protocol)
    if args.stage == "prepare":
        return protocol
    if args.stage == "train":
        rule_controls(root, protocol, args.fold)
        completed = []
        for actor in ACTORS:
            for method in METHODS:
                for seed in SEEDS:
                    run_trial(root, protocol, args.fold, actor, method, seed)
                    directory = root / "runs" / f"{args.fold}_{actor}_{method}_{seed}"
                    completed.append(dict(run_id=directory.name, completed_sha256=file_hash(directory / "completed.json")))
                    write_json(root / f"worker_{args.fold}.json", dict(status="running", fold=args.fold,
                        protocol_sha256=file_hash(root / "protocol.json"), completed_runs=completed, no_global_summary_writes=True))
        write_json(root / f"worker_{args.fold}.json", dict(status="complete", fold=args.fold,
            protocol_sha256=file_hash(root / "protocol.json"), completed_runs=completed, no_global_summary_writes=True))
        return completed
    records, rules = [], {}
    for fold in FOLDS:
        verify_worker_receipt(root, protocol, fold)
        rules[fold] = rule_controls(root, protocol, fold, create=False)
        for actor in ACTORS:
            for method in METHODS:
                for seed in SEEDS:
                    directory = root / "runs" / f"{fold}_{actor}_{method}_{seed}"
                    cached = cached_trial(directory, trial_config(protocol, root, fold, actor, method, seed), protocol)
                    if cached is None:
                        raise ValueError("Report cannot train or evaluate missing trials")
                    records.extend(cached)
    result = write_report(root, records, rules, protocol)
    check_frozen(protocol)
    return result


if __name__ == "__main__":
    main()
