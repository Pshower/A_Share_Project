"""Preregistered integer-share trade actor: train-only pilots and fixed snapshots."""

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.logger import configure

from src.agents.ppo_agent import PPOAgent, make_model
from src.data.preprocess import ROOT, file_hash, read_config, resolve_path, write_json
from src.envs.initial_account import MODES
from src.runtime import events
from .env_factory import ResearchData
from .evaluate import evaluate_agent
from .freeze import json_digest, source_hashes
from .matrix_compare import load_matrix
from .matrix_report import _metric_mean, _metric_stats, _stats
from .research_delta import DATA_FIELDS, _registered_files, _release, _training_setup, _verify_files
from .research_matrix import MatrixCallback, preserve_training_rng, verify_completed


SEEDS = [42, 123, 2026]
BUDGETS = dict(short=8192, long=32768)
PILOT_STEPS = 2048
PILOT_PLANNED_UPDATES = 80
PROBE_SEED = 20260927
PROBE_INDICES = [-52, -42, -32, -22, -12, -2]
PILOT_CANDIDATES = [dict(name=f"lr{lr:g}_gain{gain:g}", learning_rate=lr,
                         actor_output_gain=gain, initial_log_std=-.7)
                    for lr in (1e-5, 3e-5) for gain in (.01, .1)]
DIRECTIONS = ("hold", "buy", "reduce", "exit")


def config_for(base, root, seed=42, candidate=None):
    config = deepcopy(base)
    candidate = candidate or PILOT_CANDIDATES[0]
    config.update(action_mode="stock_trade", hidden_sizes=[256, 128], device="cuda", torch_threads=1,
                  seed=seed, actor_output_gain=candidate["actor_output_gain"], initial_log_std=-.7,
                  max_trade_weight=.05, max_commission_ratio=.001, top_n=20,
                  top_n_choices=[10, 20, 40], prior_feature_index=None,
                  output_root=str(Path(root) / "runs"))
    config.pop("action_top_k", None)
    config.pop("max_delta_weight", None)
    config["ppo"]["learning_rate"] = candidate["learning_rate"]
    required = dict(n_steps=512, batch_size=128, n_epochs=5, gamma=.99, gae_lambda=.95,
                    clip_range=.2, target_kl=.03, ent_coef=0., vf_coef=.5, max_grad_norm=.5)
    if any(config["ppo"].get(key) != value for key, value in required.items()):
        raise ValueError("Trade study requires the preregistered fixed PPO hyperparameters")
    if config.get("learning_rate_schedule", "constant") != "constant":
        raise ValueError("Trade pilot requires a constant learning-rate schedule")
    return config


def kl_band_distance(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return math.inf
    if value < .001:
        return math.log(.001 / max(value, 1e-12))
    return math.log(value / .03) if value > .03 else 0.


def pilot_eligible(report):
    kl = report.get("tail_mean_kl")
    modes = report.get("mode_statistics", {})
    filled = modes.get("filled_modes", {})
    return bool(report.get("status") == "success" and report.get("actual_steps") == PILOT_STEPS
        and report.get("all_parameters_finite") is True and report.get("all_actions_probabilities_finite") is True
        and report.get("optimizer_steps", 0) >= 64
        and isinstance(kl, (int, float)) and math.isfinite(kl) and 0 <= kl <= .045
        and modes.get("cash_training_buy_fills", 0) > 0
        and filled.get("reduce", 0) + filled.get("exit", 0) > 0
        and modes.get("voluntary_hold", 0) > 0 and modes.get("economic_violations", 0) == 0
        and report.get("deterministic_cash_buy_dates", 0) >= 1
        and report.get("probe_economic_violations", 0) == 0
        and report.get("validation_evaluated") is False and report.get("test_evaluated") is False)


def choose_pilot(reports):
    eligible = [row for row in reports if pilot_eligible(row)]
    def rank(row):
        coverage = sum(row["mode_statistics"]["filled_modes"].get(name, 0) > 0 for name in DIRECTIONS[1:])
        return (-row["optimizer_steps"], -coverage, kl_band_distance(row["tail_mean_kl"]),
                row["learning_rate"], row["actor_output_gain"], row["name"])
    return min(eligible, key=rank) if eligible else None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class ModeStatistics:
    """Count sampled decisions and actual economic fills, not proposed target changes."""

    def __init__(self, lot_size=100, min_commission=5., min_commission_ratio=.001, commission=.0003):
        self.lot_size = lot_size
        self.min_commission, self.commission = min_commission, commission
        self.minimum = min_commission / min_commission_ratio
        self.sampled = Counter()
        self.filled = Counter()
        self.reasons = Counter()
        self.voluntary_hold = self.forced_hold = self.cash_buys = self.violations = 0
        self.full_exit_exceptions = self.minimum_commission_fills = 0
        self.gross = self.fees = 0.

    def observe(self, diagnostics, info):
        dirs = np.asarray(diagnostics["directions"])
        legal = np.asarray(diagnostics["legal_directions"])
        if dirs.ndim != 1 or legal.shape != (len(dirs), 4) or not np.isin(dirs, range(4)).all():
            raise ValueError("Invalid original trade direction/mask trace")
        if not legal[np.arange(len(dirs)), dirs.astype(int)].all():
            raise ValueError("Sampled direction violates its state-known legal mask")
        for direction, name in enumerate(DIRECTIONS):
            self.sampled[name] += int((dirs == direction).sum())
        self.voluntary_hold += int(((dirs == 0) & (legal.sum(axis=1) > 1)).sum())
        self.forced_hold += int(((dirs == 0) & (legal.sum(axis=1) == 1)).sum())
        before = info.get("opening_snapshot", {}).get("positions")
        codes = info.get("opening_snapshot", {}).get("codes", info.get("decision_snapshot", {}).get("codes", []))
        for order in info.get("orders", []):
            self.reasons.update(order.get("reasons", []))
            quantity = order.get("filled_shares", 0)
            if not quantity:
                continue
            price = order.get("price", float("nan"))
            direction = order.get("trade_direction")
            after = order.get("position_after")
            if after is None and before is not None and order.get("code") in codes:
                after = before[codes.index(order["code"])] + quantity
            full_exit = direction == 3 and quantity < 0 and after is not None and abs(after) <= 1e-8
            name = "buy" if quantity > 0 else "exit" if full_exit else "reduce"
            self.filled[name] += 1
            if quantity > 0 and info.get("initial_account", {}).get("mode") == "cash":
                self.cash_buys += 1
            nominal = abs(quantity) * price
            self.gross += nominal
            self.fees += order.get("fee", 0.)
            self.minimum_commission_fills += int(nominal * self.commission <= self.min_commission)
            valid = (math.isfinite(quantity) and math.isfinite(price) and price > 0
                     and direction in (1, 2, 3) and (quantity > 0) == (direction == 1))
            if not full_exit:
                valid = valid and abs(quantity / self.lot_size - round(quantity / self.lot_size)) <= 1e-8
                valid = valid and nominal + 1e-7 >= max(self.lot_size * price, self.minimum)
                if order.get("opening_equity") is not None:
                    valid = valid and nominal <= .05 * order["opening_equity"] + 1e-7
                if quantity < 0:
                    valid = valid and after is not None and after >= 1 - 1e-8
            else:
                self.full_exit_exceptions += int(nominal < self.minimum or quantity % self.lot_size != 0)
            self.violations += int(not valid)

    def snapshot(self):
        return dict(sampled_modes={name: self.sampled[name] for name in DIRECTIONS},
            filled_modes={name: self.filled[name] for name in DIRECTIONS[1:]},
            voluntary_hold=self.voluntary_hold, forced_hold=self.forced_hold,
            cash_training_buy_fills=self.cash_buys, economic_violations=self.violations,
            actual_full_exit_small_or_odd_exceptions=self.full_exit_exceptions,
            minimum_commission_fills=self.minimum_commission_fills,
            gross_filled_notional=self.gross, total_fees=self.fees, rejection_reasons=dict(self.reasons))


class TradeTrace:
    """Original per-step state/action arrays plus sparse orders; no feature duplication."""

    OBS_FIELDS = ("portfolio", "valid_mask", "buy_mask", "sell_mask", "execution_features",
                  "allocation_context", "share_state", "trade_lot_bounds", "legal_directions", "top_n")
    DIAG_FIELDS = ("directions", "size_latent", "requested_lots", "selected_lots", "proposed_shares", "shares",
                   "freeze_mask", "exit_mask", "priority", "lot_bounds", "legal_directions", "size_active")

    def __init__(self, directory, identity, chunk_steps=512):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.identity, self.chunk_steps = deepcopy(identity), chunk_steps
        self.pending, self.sparse, self.chunks = [], [], []
        self.steps = 0
        self.statistics = ModeStatistics(identity.get("lot_size", 100), identity.get("min_commission", 5.),
                                         identity.get("max_commission_ratio", .001), identity.get("commission", .0003))

    def append(self, date, observation, action, info, *, log_prob=None, value=None, reward=None, done=None):
        diagnostics = action.diagnostics if hasattr(action, "diagnostics") else info.get("action_diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError("Trade trace requires native original action diagnostics")
        self.statistics.observe(diagnostics, info)
        if self.statistics.violations:
            raise ValueError("Actual trade violates integer-lot/economic execution invariants")
        row = dict(step=np.int64(self.steps + 1), decision_date=np.asarray(str(date)),
                   execution_date=np.asarray(str(info.get("date", ""))))
        for name in self.OBS_FIELDS:
            row["obs_" + name] = np.asarray(observation[name]).copy()
        for name in self.DIAG_FIELDS:
            row["action_" + name] = np.asarray(diagnostics[name]).copy()
        for name in ("direction_probabilities", "size_mean", "size_sigma", "category_entropy",
                     "conditional_size_entropy", "requested_notional_l1", "requested_cash_reference",
                     "top_n_filtered", "cash_filtered"):
            if name in diagnostics:
                row["action_" + name] = np.asarray(diagnostics[name]).copy()
        packed = np.asarray(action) if not hasattr(action, "diagnostics") else np.concatenate(
            [diagnostics["directions"], diagnostics["size_latent"]])
        if not np.isfinite(packed).all() or (log_prob is not None and not np.isfinite(log_prob).all()):
            raise FloatingPointError("Original trade action/probability is nonfinite")
        row["raw_packed_action"] = packed.copy()
        for name, item in (("log_prob", log_prob), ("value", value), ("reward", reward), ("done", done)):
            if item is not None:
                row[name] = np.asarray(item).copy()
        for prefix in ("decision", "opening"):
            snapshot = info.get(prefix + "_snapshot")
            if not isinstance(snapshot, dict):
                raise ValueError("Native trade trace lacks exact " + prefix + " account snapshot")
            for key, item in snapshot.items():
                if key != "codes":
                    row[prefix + "_" + key] = np.asarray(str(item) if key == "date" else item).copy()
        codes = info["decision_snapshot"]["codes"]
        if len(codes) != len(diagnostics["directions"]) or len(set(codes)) != len(codes):
            raise ValueError("Native trace account codes do not match the action coordinates")
        code_index = {code: index for index, code in enumerate(codes)}
        fills = {name: np.zeros(len(codes), dtype=np.float64) for name in
                 ("filled_shares", "filled_notional", "fee", "slippage_cost", "broker_requested_shares")}
        reasons, statuses = [[] for _ in codes], [[] for _ in codes]
        for order in info.get("orders", []):
            if order.get("code") not in code_index:
                raise ValueError("Original order code is absent from the registered trace account")
            index = code_index[order["code"]]
            quantity = order.get("filled_shares", 0)
            fills["filled_shares"][index] += quantity
            fills["filled_notional"][index] += quantity * order["price"] if quantity else 0.
            for name in ("fee", "slippage_cost"):
                fills[name][index] += order.get(name, 0.)
            fills["broker_requested_shares"][index] += order.get("requested_shares", 0.)
            reasons[index].extend(order.get("reasons", []))
            statuses[index].append(order.get("status", ""))
        row.update({"execution_" + name: value for name, value in fills.items()})
        row["execution_reasons_json"] = np.asarray([json.dumps(items, ensure_ascii=True) for items in reasons])
        row["execution_status_json"] = np.asarray([json.dumps(items, ensure_ascii=True) for items in statuses])
        for key in ("cash", "equity", "positions", "available_positions", "receivables"):
            if key in info:
                row["after_" + key] = np.asarray(info[key]).copy()
        row["initial_mode"] = np.asarray(info.get("initial_account", {}).get("mode", "cash"))
        row["episode_top_n"] = np.asarray(info.get("top_n", observation["top_n"][0]))
        self.pending.append(row)
        self.steps += 1
        if info.get("orders"):
            self.sparse.append(_json_safe(dict(step=self.steps, decision_date=str(date),
                execution_date=info.get("date"), orders=info["orders"],
                decision_snapshot=info["decision_snapshot"], opening_snapshot=info["opening_snapshot"])))
        if len(self.pending) >= self.chunk_steps:
            self.flush()

    def __call__(self, date, observation, action, info):
        self.append(date, observation, action, info)

    def flush(self):
        if not self.pending:
            return
        keys = set(self.pending[0])
        if any(set(row) != keys for row in self.pending):
            raise ValueError("Per-step native trace fields changed within a rollout")
        arrays = {name: np.stack([row[name] for row in self.pending]) for name in sorted(keys)}
        if any(value.dtype == object for value in arrays.values()):
            raise ValueError("Trade NPZ must be readable with allow_pickle=False")
        start, end = int(arrays["step"][0]), int(arrays["step"][-1])
        path = self.directory / f"steps_{start:06d}_{end:06d}.npz"
        np.savez_compressed(path, **arrays)
        with gzip.open(self.directory / "orders.jsonl.gz", "at", encoding="utf-8") as handle:
            for row in self.sparse:
                handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
        self.chunks.append(dict(path=path.name, sha256=file_hash(path), first_step=start, last_step=end, steps=len(self.pending)))
        self.pending.clear()
        self.sparse.clear()
        write_json(self.directory / "index.json", self.manifest())

    def manifest(self):
        path = self.directory / "orders.jsonl.gz"
        size = path.stat().st_size if path.exists() else 0
        return dict(schema_version=1, identity=self.identity, total_steps=self.steps,
            chunks=deepcopy(self.chunks), sparse_orders_path=path.name, sparse_orders_prefix_bytes=size,
            sparse_orders_prefix_sha256=file_hash(path) if size else hashlib.sha256(b"").hexdigest(),
            feature_recovery="decision_date + immutable panel/scaler hashes; no future feature rows",
            mode_statistics=self.statistics.snapshot())

    def close(self):
        self.flush()
        write_json(self.directory / "index.json", self.manifest())
        return self.manifest()


def verify_trace_prefix(directory, manifest):
    directory = Path(directory)
    previous = 0
    for chunk in manifest["chunks"]:
        path = (directory / chunk["path"]).resolve()
        if (not path.is_relative_to(directory.resolve()) or file_hash(path) != chunk["sha256"]
                or chunk["first_step"] != previous + 1 or chunk["last_step"] - previous != chunk["steps"]):
            raise ValueError("Original trade trace chunk identity/prefix changed")
        previous = chunk["last_step"]
    if previous != manifest["total_steps"]:
        raise ValueError("Original trade trace lacks a complete step prefix")
    path = (directory / manifest["sparse_orders_path"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Sparse trace path escapes its registered directory")
    with path.open("rb") as handle:
        raw = handle.read(manifest["sparse_orders_prefix_bytes"])
    if len(raw) != manifest["sparse_orders_prefix_bytes"] or hashlib.sha256(raw).hexdigest() != manifest["sparse_orders_prefix_sha256"]:
        raise ValueError("Original sparse order prefix changed")


def _trace_identity(data, mode, **extra):
    broker = data.contract.get("broker", data.settings["broker"])
    return dict(mode=mode, contract_sha256=json_digest(data.contract), artifacts=data.contract["artifacts"],
        codes=data.codes, lot_size=broker["lot_size"], min_commission=broker.get("min_commission", 0.),
        max_commission_ratio=data.config.get("max_commission_ratio", .001),
        commission=broker.get("commission", .0003), **extra)


class TradeCallback(MatrixCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = TradeTrace(self.output / "training_trace", _trace_identity(self.data, "sampled_training", run_id=self.run_id))
        self.all_actions_probabilities_finite = True

    def _on_step(self):
        events.check_cancel()
        obs = self.model._last_obs
        self.valid_observation_steps += obs["valid_mask"].sum(axis=0).astype(np.int64)
        if self.model.num_timesteps % 16 == 0:
            events.emit("progress", phase="training", timesteps=self.model.num_timesteps)
        for index, info in enumerate(self.locals.get("infos", [])):
            observation = {name: value[index] for name, value in obs.items()}
            def numpy_value(name):
                value = self.locals.get(name)
                return value[index].detach().cpu().numpy() if torch.is_tensor(value) else value[index] if value is not None else None
            self.initial_account_steps.update([info.get("initial_account", {}).get("mode", "cash")])
            self.trace.append(info["decision_snapshot"]["date"], observation, numpy_value("actions"), info,
                log_prob=numpy_value("log_probs"), value=numpy_value("values"), reward=numpy_value("rewards"), done=numpy_value("dones"))
            sample = dict(actual_cash=info["cash"] / info["equity"] if info["equity"] else 0.,
                          raw_return=info["reward"], fees=sum(order["fee"] for order in info.get("orders", [])))
            self.portfolio_totals.update(sample)
            self.portfolio_counts.update(sample.keys())
            self.portfolio_samples += 1
        return True

    def record_update(self):
        sigma = self.model.policy.network.log_std.detach().exp()
        self.model.logger.record("train/conditional_size_sigma", float(sigma.mean().cpu().item()))
        self.model.logger.record("train/gross_filled_notional", self.trace.statistics.gross)
        if self.critic_rollout is not None:
            cat, size = [], []
            with torch.no_grad():
                count = len(self.critic_rollout.returns.reshape(-1))
                for start in range(0, count, self.model.batch_size):
                    obs = {name: values.reshape((-1, *self.model.observation_space[name].shape))[start:start + self.model.batch_size]
                           for name, values in self.critic_rollout.observations.items()}
                    tensor, _ = self.model.policy.obs_to_tensor(obs)
                    distribution = self.model.policy.get_distribution(tensor)
                    cat.extend(distribution.categorical_entropy().cpu().numpy().reshape(-1))
                    size.extend(distribution.conditional_size_entropy().cpu().numpy().reshape(-1))
            self.model.logger.record("train/post_update_categorical_entropy", float(np.mean(cat)))
            self.model.logger.record("train/post_update_conditional_size_entropy", float(np.mean(size)))
        super().record_update()
        if self.update_records:
            # Packed 2N storage is not 2N independent Gaussian dimensions.
            for row in self.update_records:
                row.pop("train/entropy_per_dimension", None)
            pd.DataFrame(self.update_records).to_csv(self.output / "training_updates.csv", index=False)

    def save_fixed_snapshot(self):
        self.trace.flush()
        old = set(self.saved)
        super().save_fixed_snapshot()
        for budget in self.saved - old:
            path = self.output / (budget + "_audit.json")
            audit = read_config(path)
            audit.update(training_trace_prefix=self.trace.manifest(), mode_statistics=self.trace.statistics.snapshot(),
                         all_actions_probabilities_finite=self.all_actions_probabilities_finite)
            write_json(path, audit)

    def _on_training_end(self):
        super()._on_training_end()
        self.trace.close()


def load_delta_reference(root):
    root = Path(root).resolve()
    protocol = read_config(root / "protocol.json")
    status = read_config(root / "status.json")
    if protocol.get("action_mode") != "stock_delta" or status.get("status") != "complete" or status.get("test_evaluated"):
        raise ValueError("Delta reference must be complete and development-only")
    from .research_delta import _selected_pilot, config_for as delta_config
    selected = _selected_pilot(protocol, root)
    records, registered = [], {}
    for seed in SEEDS:
        config = delta_config(protocol["base_config"], root, seed, selected)
        directory = root / "runs" / f"small_{seed}"
        rows = verify_completed(directory, config, protocol)
        if rows is None:
            raise ValueError("Delta reference lacks a sealed seed")
        records.extend(rows)
        marker = read_config(directory / "completed.json")
        registered.update({str((directory / name).relative_to(root)): digest for name, digest in marker["files"].items()})
        registered[str((directory / "completed.json").relative_to(root))] = file_hash(directory / "completed.json")
        control = root / "controls" / f"small_{seed}"
        identity = read_config(control / "controls.json")
        _verify_files(control, identity["files"])
        if identity["metrics"] != {mode: read_config(control / mode / "metrics.json") for mode in MODES}:
            raise ValueError("Delta initialization reference metrics changed")
        if any(row["metrics"][mode]["initialized"] != identity["metrics"][mode]["ppo"] for row in rows for mode in MODES):
            raise ValueError("Delta initialized metrics differ from sealed references")
        registered.update({str((control / name).relative_to(root)): digest for name, digest in identity["files"].items()})
        registered[str((control / "controls.json").relative_to(root))] = file_hash(control / "controls.json")
    if records != read_config(root / "summary.json"):
        raise ValueError("Delta summary differs from sealed run records")
    for candidate in protocol["pilot_candidates"]:
        directory = root / "pilots" / candidate["name"]
        marker = read_config(directory / "completed.json")
        registered.update({str((directory / name).relative_to(root)): digest for name, digest in marker["files"].items()})
        registered[str((directory / "completed.json").relative_to(root))] = file_hash(directory / "completed.json")
    for name in ("protocol.json", "summary.json", "status.json", "pilot_selection.json"):
        registered[name] = file_hash(root / name)
    hold = root / "hold_shares"
    evidence = read_config(hold / "evidence.json")
    _verify_files(hold, evidence["files"])
    registered.update({str((hold / name).relative_to(root)): digest for name, digest in evidence["files"].items()})
    registered["hold_shares/evidence.json"] = file_hash(hold / "evidence.json")
    return dict(root=root, protocol=protocol, records=records,
                evidence=dict(registered_files=registered, registered_file_count=len(registered)), hold_shares=evidence)


def load_references(roots):
    if set(roots) != {"prior", "random", "delta"}:
        raise ValueError("Prior, random and delta immutable references are all required")
    return {mode: load_delta_reference(path) if mode == "delta" else load_matrix(path, mode) for mode, path in roots.items()}


def prepare_study(base, root, reference_roots):
    root = Path(root).resolve()
    for directory in reference_roots.values():
        reference = Path(directory).resolve()
        if root == reference or root.is_relative_to(reference) or reference.is_relative_to(root):
            raise ValueError("Study and reference paths must not contain each other")
    config = config_for(base, root)
    data = ResearchData(config)
    if data.contract.get("data_source") != "sina" or len(data.codes) != 300:
        raise ValueError("Trade study requires registered local Sina 300-stock data")
    if data.manifest["train_end"] != "2026-03-31" or data.manifest["val_end"] != "2026-06-30":
        raise ValueError("Preregistered chronological boundaries changed")
    references = load_references(reference_roots)
    registration, differences = {}, {}
    core = deepcopy(data.contract)
    core.pop("hidden_sizes")
    for mode, reference in references.items():
        old = reference["protocol"]
        if old["seeds"] != SEEDS or old["short_steps"] != BUDGETS["short"]:
            raise ValueError("Reference short budget or seeds differ")
        if (old["train_end"], old["val_end"], old["evaluation_seed"], old["evaluation_modes"]) != (
                data.manifest["train_end"], data.manifest["val_end"], PROBE_SEED, list(MODES)):
            raise ValueError("Reference dates or evaluation scenarios differ")
        for field in (*DATA_FIELDS, "initial_portfolio", "training_episode_steps", "recent_episode_every"):
            if old["core_contract"].get(field) != core.get(field):
                raise ValueError("Reference data/execution/sampling mismatch: " + field)
        old_ppo, new_ppo = deepcopy(old["base_config"]["ppo"]), deepcopy(config["ppo"])
        old_ppo.pop("learning_rate", None)
        new_ppo.pop("learning_rate", None)
        if old_ppo != new_ppo or old["base_config"].get("learning_rate_schedule", "constant") != "constant":
            raise ValueError("Reference PPO differs beyond registered pilot LR")
        registration[mode] = dict(root=str(reference["root"]), evidence=reference["evidence"])
        differences[mode] = {key: dict(reference=old["core_contract"].get(key), trade=core.get(key))
            for key in sorted(set(old["core_contract"]) | set(core)) if old["core_contract"].get(key) != core.get(key)}
    protocol = dict(schema_version=1, study_mode="trade", action_mode="stock_trade", base_config=config,
        seeds=SEEDS, short_steps=BUDGETS["short"], long_steps=BUDGETS["long"], hidden_sizes=[256, 128],
        maximum_training_steps=98304, pilot_steps_each=PILOT_STEPS, maximum_pilot_steps=8192,
        maximum_combined_training_steps=106496, pilot_candidates=PILOT_CANDIDATES,
        pilot_rule=dict(min_actual_optimizer_steps=64, planned_optimizer_steps=80, max_tail_mean_joint_kl=.045,
            require_cash_sampled_actual_buy=True, require_actual_reduce_or_exit=True, require_voluntary_hold=True,
            deterministic_cash_buy_dates_min=1, deterministic_cash_probe_dates=6, economic_violations_max=0,
            ranking=["more_actual_optimizer_steps", "more_distinct_actual_buy_reduce_exit_modes",
                     "lower_log_distance_to_joint_KL_band_0.001_0.03", "lower_learning_rate", "lower_actor_gain"],
            validation_returns_used=False),
        probe_seed=PROBE_SEED, probe_train_indices=PROBE_INDICES, probe_modes=["cash", "random"],
        core_contract=core, references=registration, reference_contract_differences=differences,
        pilot_varying_contract_fields=["trade.actor_output_gain"],
        core_contract_note="Planning template uses gain .01; sealed formal model/control contracts bind the selected pilot gain",
        source_hashes=source_hashes(), evaluation_seed=PROBE_SEED, evaluation_modes=list(MODES),
        main_top_n=20, auxiliary_top_n=[10, 40], auxiliary_used_for_selection=False,
        representative_seed=42, representative_rule="fixed_seed_42_no_lucky_seed_selection",
        candidate_rule=dict(max_all_account_drawdown=.10, mean_own_initialization_gain_positive=True,
            minimum_positive_learning_seeds=2, mean_return_above_daily_momentum=True,
            ranking=["higher_mean_net_return", "lower_worst_drawdown", "lower_total_cost", "shorter_budget"]),
        checkpoint_selection="fixed_budget_shared_prefix", train_end=data.manifest["train_end"],
        val_end=data.manifest["val_end"], validation_already_exposed=True, test_evaluated=False,
        old_budget_pairing="Only 8192 is matched; 32768 has no old paired budget; old 65536 is descriptive",
        critic_diagnostic_capture="rollout_end_copy_v2", original_action_trace="NPZ_each_512_steps_and_sparse_orders_gzip",
        limitations=data.manifest.get("limitations", []) + [
            "Already-exposed April-June development validation, not an independent test; July onward remains unevaluated.",
            "Four initial accounts are paired scenarios, not independent random seeds.",
            "Trade actor, state context, exploration and economic execution differ; not an isolated actor or Critic causal test.",
            "Fresh formal initializations; train-only pilot configuration selection; no warm-started pilot weights.",
            "HFQ quotes, 2026 fixed membership, lot assumptions and missing real execution/event metadata remain limitations."])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "protocol.json"
    if path.exists():
        if read_config(path) != protocol:
            raise ValueError("Trade protocol/source/reference identity changed; use a new directory")
    else:
        write_json(path, protocol)
        write_json(root / "status.json", dict(status="prepared", test_evaluated=False))
    return protocol


def _probe_env(data, market, index, mode):
    from src.agents.trade_mapping import TradeActionWrapper
    from src.envs.broker import BrokerConfig
    from src.envs.market import MarketDataProvider
    from src.envs.trading import TradingEnv
    decision, execution = market.dates[index], market.dates[index + 1]
    if execution > pd.Timestamp(data.manifest["train_end"]):
        raise ValueError("Pilot probe crosses the training boundary")
    provider = MarketDataProvider(market.panel.loc[:execution], data.manifest["feature_cols"],
        price_path=data.prices.loc[:execution, pd.IndexSlice[data.codes, :]],
        price_basis=data.manifest["price_basis"], start_date=decision, end_date=execution, lookback=data.config["lookback"])
    env = TradingEnv(provider, data.settings["initial_capital"], BrokerConfig(**data.settings["broker"]), initial_portfolio=mode)
    return TradeActionWrapper(env, bound=data.config["logit_bound"],
        max_trade_weight=data.config["max_trade_weight"], max_commission_ratio=data.config["max_commission_ratio"], top_n=20)


def deterministic_probes(model, data, directory):
    market = data.market("train")
    if market.n_dates < 52:
        raise ValueError("Six preregistered training probe indices are unavailable")
    before = {name: tensor.detach().clone() for name, tensor in model.policy.state_dict().items()}
    results, cash_dates = [], 0
    trace = TradeTrace(Path(directory) / "probe_trace", _trace_identity(data, "train_only_deterministic_probe", seed=PROBE_SEED), chunk_steps=12)
    agent = PPOAgent(model, data.contract)
    was_training = model.policy.training
    try:
        with preserve_training_rng(model.device):
            for negative in PROBE_INDICES:
                index = market.n_dates + negative
                for mode in ("cash", "random"):
                    env = _probe_env(data, market, index, mode)
                    try:
                        observation, _ = env.reset(seed=PROBE_SEED)
                        action = agent.act(observation)
                        _, _, _, _, info = env.step_order(action)
                        trace(market.dates[index], observation, action, info)
                        buy = any(order["filled_shares"] > 0 for order in info.get("orders", []))
                        cash_dates += int(mode == "cash" and buy)
                        results.append(dict(train_index=negative, decision_date=str(market.dates[index].date()),
                            execution_date=str(market.dates[index + 1].date()), mode=mode, top_n=20,
                            actual_buy=buy, filled_modes=trace.statistics.snapshot()["filled_modes"],
                            orders=_json_safe(info.get("orders", []))))
                    finally:
                        env.close()
    finally:
        model.policy.set_training_mode(was_training)
        trace.close()
    if any(not torch.equal(before[name], tensor) for name, tensor in model.policy.state_dict().items()):
        raise AssertionError("Deterministic train-only probes changed model parameters")
    report = dict(seed=PROBE_SEED, indices=PROBE_INDICES, top_n=20, modes=["cash", "random"],
        environment_steps=12, deterministic_cash_buy_dates=cash_dates, results=results,
        economic_violations=trace.statistics.violations, parameters_unchanged=True,
        validation_evaluated=False, test_evaluated=False, trace=trace.manifest())
    write_json(Path(directory) / "probes.json", report)
    return report


def _load_pilot(directory, config, protocol):
    cached = read_config(directory / "completed.json")
    if cached["config_sha256"] != json_digest(config) or cached["protocol_sha256"] != json_digest(protocol):
        raise ValueError("Pilot cache configuration changed")
    _verify_files(directory, cached["files"])
    report = read_config(directory / "report.json")
    if report != cached["report"] or any(report[key] != value for key, value in dict(
            name=directory.name, learning_rate=config["ppo"]["learning_rate"], actor_output_gain=config["actor_output_gain"], initial_log_std=-.7).items()):
        raise ValueError("Pilot summary identity differs from registered evidence")
    if report["status"] == "success":
        required = {"config.json", "report.json", "pilot_audit.json", "pilot_episodes.json", "probes.json",
            "training_updates.csv", "initial_model/model.zip", "initial_model/metadata.json",
            "pilot_model/model.zip", "pilot_model/metadata.json", "training_trace/index.json", "probe_trace/index.json"}
        if not required <= set(cached["files"]):
            raise ValueError("Pilot cache lacks required registered training/probe artifacts")
        audit = read_config(directory / "pilot_audit.json")
        metadata = read_config(directory / "pilot_model" / "metadata.json")
        probes = read_config(directory / "probes.json")
        expected = dict(actual_steps=audit["actual_timesteps"], all_parameters_finite=audit["all_parameters_finite"],
            optimizer_steps=audit["optimizer_steps_min"], mode_statistics=audit["mode_statistics"],
            all_actions_probabilities_finite=audit["all_actions_probabilities_finite"],
            tail_mean_kl=audit["training_diagnostics"].get("tail_mean/train/approx_kl"),
            deterministic_cash_buy_dates=probes["deterministic_cash_buy_dates"], probe_economic_violations=probes["economic_violations"])
        if (any(report.get(key) != value for key, value in expected.items()) or metadata["config"] != config
                or metadata["num_timesteps"] != PILOT_STEPS or metadata["model_sha256"] != audit["model_sha256"]
                or file_hash(directory / "pilot_model" / "model.zip") != audit["model_sha256"]):
            raise ValueError("Pilot summary differs from actual audit/probe evidence")
        verify_trace_prefix(directory / "training_trace", audit["training_trace_prefix"])
        verify_trace_prefix(directory / "probe_trace", probes["trace"])
    return report


def run_pilot(config, root, candidate, protocol):
    directory = Path(root) / "pilots" / candidate["name"]
    if (directory / "completed.json").exists():
        return _load_pilot(directory, config, protocol)
    if directory.exists():
        raise ValueError("Incomplete pilot requires review; refusing to overwrite")
    data, experience, env = _training_setup(config, directory)
    model = callback = None
    report = dict(candidate, status="error", actual_steps=0, all_parameters_finite=False,
        all_actions_probabilities_finite=False, optimizer_steps=0, planned_optimizer_steps=80,
        tail_mean_kl=None, mode_statistics={}, deterministic_cash_buy_dates=0, probe_economic_violations=0,
        validation_evaluated=False, test_evaluated=False)
    started = perf_counter()
    try:
        model = make_model(env, config)
        initial = {name: tensor.detach().cpu().clone() for name, tensor in model.policy.state_dict().items()}
        PPOAgent(model, data.contract).save(directory / "initial_model", config)
        model.set_logger(configure(str(directory / "logs"), ["csv"]))
        callback = TradeCallback(data, directory, experience, initial, dict(pilot=PILOT_STEPS), Path(root) / "status.json", candidate["name"])
        torch.cuda.reset_peak_memory_stats(model.device)
        model.learn(total_timesteps=PILOT_STEPS, callback=callback)
        model.logger.dump(step=model.num_timesteps)
        audit = read_config(directory / "pilot_audit.json")
        probes = deterministic_probes(model, data, directory)
        report.update(status="success", actual_steps=model.num_timesteps, all_parameters_finite=audit["all_parameters_finite"],
            all_actions_probabilities_finite=audit["all_actions_probabilities_finite"], optimizer_steps=audit["optimizer_steps_min"],
            mode_statistics=audit["mode_statistics"], tail_mean_kl=audit["training_diagnostics"].get("tail_mean/train/approx_kl"),
            deterministic_cash_buy_dates=probes["deterministic_cash_buy_dates"], probe_economic_violations=probes["economic_violations"],
            conditional_size_sigma=float(model.policy.network.log_std.detach().exp().mean().cpu().item()))
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error),
            fatal_invariant_failure=isinstance(error, (ValueError, AssertionError)),
            actual_steps=int(model.num_timesteps) if model is not None else 0)
    finally:
        env.close()
        if callback is not None:
            callback.trace.close()
        report.update(elapsed_seconds=perf_counter() - started, optimizer_update_fraction=report["optimizer_steps"] / 80)
        report["eligible"] = pilot_eligible(report)
        write_json(directory / "report.json", report)
        write_json(directory / "completed.json", dict(config_sha256=json_digest(config), protocol_sha256=json_digest(protocol),
            files=_registered_files(directory), report=report))
        del model, callback
        _release()
    return report


def run_pilots(protocol, root):
    reports = []
    for candidate in PILOT_CANDIDATES:
        report = run_pilot(config_for(protocol["base_config"], root, candidate=candidate), root, candidate, protocol)
        reports.append(report)
        if report.get("fatal_invariant_failure"):
            break
    selected = choose_pilot(reports) if len(reports) == len(PILOT_CANDIDATES) and not any(
        row.get("fatal_invariant_failure") for row in reports) else None
    result = dict(protocol_sha256=json_digest(protocol), reports=reports, selected=selected,
        actual_steps=sum(row["actual_steps"] for row in reports), probe_steps=sum(12 for row in reports if row["status"] == "success"),
        validation_evaluated=False, test_evaluated=False,
        files={f"pilots/{row['name']}/completed.json": file_hash(Path(root) / "pilots" / row["name"] / "completed.json") for row in reports})
    write_json(Path(root) / "pilot_selection.json", result)
    write_json(Path(root) / "status.json", dict(status="pilot_complete" if selected else "pilot_failed_closed",
        actual_pilot_steps=result["actual_steps"], selected=selected, test_evaluated=False))
    if selected is None:
        raise RuntimeError("No eligible train-only pilot; formal training is disabled")
    return result


def _selected_pilot(protocol, root):
    result = read_config(Path(root) / "pilot_selection.json")
    if result["protocol_sha256"] != json_digest(protocol):
        raise ValueError("Pilot selection belongs to a different protocol")
    _verify_files(Path(root), result["files"])
    reports = [_load_pilot(Path(root) / "pilots" / candidate["name"],
        config_for(protocol["base_config"], root, candidate=candidate), protocol) for candidate in PILOT_CANDIDATES]
    selected = choose_pilot(reports)
    if selected is None or result["selected"] != selected or result["reports"] != reports:
        raise ValueError("Pilot selection changed or is ineligible")
    return selected


def _evaluate_trace(agent, data, folder, *, compare_baselines=False):
    trace = TradeTrace(Path(folder).parent / (Path(folder).name + "_trace"),
        _trace_identity(data, "deterministic_development", seed=PROBE_SEED,
                        initial_mode=data.settings["initial_portfolio"], top_n=data.settings["trade_top_n"]))
    try:
        evaluate_agent(agent, data, folder, compare_baselines=compare_baselines, trace_sink=trace)
    finally:
        trace.close()
    return read_config(Path(folder) / "metrics.json"), trace.manifest()


def evaluate_initial(data, initial_path, root, run_id):
    directory = Path(root) / "controls" / run_id
    identity = dict(contract=data.contract, training_seed=data.config["seed"], evaluation_seed=PROBE_SEED,
        evaluation_modes=list(MODES), top_n=20, initial_model_sha256=file_hash(Path(initial_path) / "model.zip"))
    path = directory / "controls.json"
    if path.exists():
        record = read_config(path)
        if record["identity"] != identity:
            raise ValueError("Own initialization control identity changed")
        _verify_files(directory, record["files"])
        required = {f"{mode}/{name}" for mode in MODES for name in ("metrics.json", "ppo/daily.csv", "ppo/orders.csv", "ppo/positions.parquet")}
        required |= {f"{mode}_trace/index.json" for mode in MODES}
        if not required <= set(record["files"]) or record["metrics"] != {mode: read_config(directory / mode / "metrics.json") for mode in MODES}:
            raise ValueError("Own initialization controls lack registered account metrics/traces")
        return record["metrics"]
    if directory.exists():
        raise ValueError("Incomplete own initialization controls require review")
    agent = PPOAgent.load(initial_path, data.contract, data.config["device"])
    if agent.model.num_timesteps != 0:
        raise ValueError("Own initialization control must be untrained")
    metrics, traces = {}, {}
    for mode in MODES:
        data.settings.update(seed=PROBE_SEED, initial_portfolio=mode, trade_top_n=20)
        metrics[mode], traces[mode] = _evaluate_trace(agent, data, directory / mode, compare_baselines=True)
    write_json(path, dict(identity=identity, metrics=metrics, traces=traces, files=_registered_files(directory)))
    del agent
    _release()
    return metrics


def _completed_trial(directory, config, protocol, data):
    path = directory / "completed.json"
    if not path.exists():
        return None
    marker = read_config(path)
    if marker["config_sha256"] != json_digest(config) or marker["protocol_sha256"] != json_digest(protocol):
        raise ValueError("Completed trade trial identity changed")
    _verify_files(directory, marker["files"])
    records = marker["records"]
    if len(records) != 2 or {row["budget"] for row in records} != set(BUDGETS):
        raise ValueError("Trade trial must have both fixed checkpoints")
    required = {"config.json", "initial_model/model.zip", "initial_model/metadata.json", "training_updates.csv", "training_trace/index.json"}
    for budget in BUDGETS:
        required |= {f"{budget}_{name}.json" for name in ("audit", "episodes")}
        required |= {f"{budget}_model/{name}" for name in ("model.zip", "metadata.json")}
        for top_n in (20, 10, 40):
            folder = budget + "_validation" if top_n == 20 else f"{budget}_aux_n{top_n}"
            required |= {f"{folder}/{mode}/{name}" for mode in MODES for name in ("metrics.json", "ppo/daily.csv", "ppo/orders.csv", "ppo/positions.parquet")}
            required |= {f"{folder}/{mode}_trace/index.json" for mode in MODES}
    if not required <= set(marker["files"]):
        raise ValueError("Completed trade trial lacks required registered models/metrics/native traces")
    for row in records:
        budget, steps = row["budget"], BUDGETS[row["budget"]]
        audit = read_config(directory / (budget + "_audit.json"))
        metadata = read_config(directory / (budget + "_model") / "metadata.json")
        digest = file_hash(directory / (budget + "_model") / "model.zip")
        if (row["steps"] != steps or row["seed"] != config["seed"] or metadata["config"] != config
                or metadata["contract"] != data.contract or metadata["num_timesteps"] != steps
                or audit["actual_timesteps"] != steps or not audit["all_parameters_finite"]
                or row["parameters"] != audit["parameters"] or row["experience"] != audit["experience"]
                or row["training_diagnostics"] != audit["training_diagnostics"] or row["mode_statistics"] != audit["mode_statistics"]
                or row["model_sha256"] != digest or metadata["model_sha256"] != digest or audit["model_sha256"] != digest):
            raise ValueError("Completed trade checkpoint/model audit changed")
        verify_trace_prefix(directory / "training_trace", audit["training_trace_prefix"])
        for top_n in (20, 10, 40):
            folder = directory / (budget + "_validation" if top_n == 20 else f"{budget}_aux_n{top_n}")
            for mode in MODES:
                metric = read_config(folder / mode / "metrics.json")["ppo"]
                expected = row["metrics"][mode]["ppo"] if top_n == 20 else row["auxiliary_metrics"][str(top_n)][mode]
                trace = row["evaluation_traces"][str(top_n)][mode]
                if metric != expected or trace != read_config(folder / (mode + "_trace") / "index.json"):
                    raise ValueError("Completed trade metrics/trace differ from native evaluation artifacts")
                verify_trace_prefix(folder / (mode + "_trace"), trace)
    return records


def run_trial(protocol, root, seed, candidate):
    config = config_for(protocol["base_config"], root, seed, candidate)
    run_id = f"small_{seed}"
    directory = Path(root) / "runs" / run_id
    data = ResearchData(config)
    cached = _completed_trial(directory, config, protocol, data)
    if cached is not None:
        if not (Path(root) / "controls" / run_id / "controls.json").exists():
            raise ValueError("Completed trial lacks sealed own initialization controls")
        controls = evaluate_initial(data, directory / "initial_model", root, run_id)
        if any(row["metrics"][mode]["initialized"] != controls[mode]["ppo"] for row in cached for mode in MODES):
            raise ValueError("Completed trial initialized metrics changed")
        return cached
    if directory.exists():
        raise ValueError("Incomplete trade trial requires review; refusing to overwrite")
    data, experience, env = _training_setup(config, directory)
    try:
        model = make_model(env, config)
        initial = {name: tensor.detach().cpu().clone() for name, tensor in model.policy.state_dict().items()}
        PPOAgent(model, data.contract).save(directory / "initial_model", config)
        model.set_logger(configure(str(directory / "logs"), ["csv"]))
        callback = TradeCallback(data, directory, experience, initial, BUDGETS, Path(root) / "status.json", run_id)
        torch.cuda.reset_peak_memory_stats(model.device)
        model.learn(total_timesteps=BUDGETS["long"], callback=callback)
        model.logger.dump(step=model.num_timesteps)
        if callback.saved != set(BUDGETS):
            raise AssertionError("Both fixed trade checkpoints must be saved")
    finally:
        env.close()
    del model, callback, env, initial, experience
    _release()
    controls = evaluate_initial(data, directory / "initial_model", root, run_id)
    records = []
    for budget, steps in BUDGETS.items():
        path = directory / (budget + "_model")
        agent = PPOAgent.load(path, data.contract, config["device"])
        metrics, auxiliary, traces = {}, {}, {}
        for top_n in [20, 10, 40]:
            current, current_traces = {}, {}
            for mode in MODES:
                data.settings.update(seed=PROBE_SEED, initial_portfolio=mode, trade_top_n=top_n)
                folder = directory / (budget + "_validation" if top_n == 20 else f"{budget}_aux_n{top_n}") / mode
                write_json(Path(root) / "status.json", dict(status="evaluating", run_id=run_id, budget=budget,
                    account=mode, top_n=top_n, test_evaluated=False))
                result, current_traces[mode] = _evaluate_trace(agent, data, folder)
                current[mode] = result["ppo"]
            traces[str(top_n)] = current_traces
            if top_n == 20:
                metrics = {mode: dict(ppo=current[mode], initialized=controls[mode]["ppo"],
                    baselines={name: metric for name, metric in controls[mode].items() if name != "ppo"}) for mode in MODES}
            else:
                auxiliary[str(top_n)] = current
        audit = read_config(directory / (budget + "_audit.json"))
        records.append(dict(size="small", budget=budget, seed=seed, steps=steps, hidden_sizes=[256, 128],
            initialization_mode="trade", parameters=audit["parameters"], elapsed_seconds=audit["elapsed_seconds"],
            training_diagnostics=audit["training_diagnostics"], experience=audit["experience"], metrics=metrics,
            mode_statistics=audit["mode_statistics"], evaluation_traces=traces,
            optimizer_steps_min=audit["optimizer_steps_min"], optimizer_steps_max=audit["optimizer_steps_max"],
            auxiliary_metrics=auxiliary, auxiliary_used_for_selection=False,
            training_updates_path=str((directory / "training_updates.csv").relative_to(root)),
            training_run_id=run_id, prefix_shared=True, model_sha256=audit["model_sha256"],
            model_path=str(path.relative_to(ROOT)), test_evaluated=False))
        del agent
        _release()
    write_json(directory / "completed.json", dict(config_sha256=json_digest(config), protocol_sha256=json_digest(protocol),
        files=_registered_files(directory), records=records))
    return records


def _baseline_equal(current, previous):
    if set(current) != set(previous):
        return False
    for key, value in current.items():
        old = previous[key]
        if isinstance(value, dict):
            if not isinstance(old, dict) or not _baseline_equal(value, old):
                return False
        elif value is None or old is None:
            if value is not old:
                return False
        elif not math.isclose(value, old, rel_tol=0, abs_tol=1e-10):
            return False
    return True


def summarize_study(records, references):
    expected = {(budget, seed) for budget in BUDGETS for seed in SEEDS}
    if len(records) != 6 or {(row["budget"], row["seed"]) for row in records} != expected:
        raise ValueError("Trade study requires both fixed budgets for all three seeds")
    if any(row.get("size") != "small" or row.get("hidden_sizes") != [256, 128] for row in records) or len({row["parameters"] for row in records}) != 1:
        raise ValueError("Trade study requires one consistent small capacity")
    if any(row.get("prefix_shared") is not True or row.get("training_run_id") != f"small_{row['seed']}" for row in records):
        raise ValueError("Short/long checkpoints must share their registered seed training prefix")
    cells = {}
    for budget, steps in BUDGETS.items():
        rows = sorted([row for row in records if row["budget"] == budget], key=lambda row: row["seed"])
        if any(row["steps"] != steps or row.get("test_evaluated") or set(row["metrics"]) != set(MODES) for row in rows):
            raise ValueError("Trade checkpoint budget, split or account scenarios changed")
        per_seed = []
        for row in rows:
            scenarios = list(row["metrics"].values())
            required = ("cumulative_return", "max_drawdown", "total_fees", "total_slippage_cost")
            if any(not isinstance(item["ppo"].get(key), (int, float)) or not math.isfinite(item["ppo"][key])
                   for item in scenarios for key in required):
                raise ValueError("Candidate return/drawdown/total-cost evidence is missing or nonfinite")
            metrics = _metric_mean([item["ppo"] for item in scenarios])
            initialized = _metric_mean([item["initialized"] for item in scenarios])
            if any(metrics.get(key) is None for key in ("cumulative_return", "max_drawdown", "total_fees", "total_slippage_cost")):
                raise ValueError("Candidate return/drawdown/total-cost evidence is missing")
            gaps = dict(initialized=metrics["cumulative_return"] - initialized["cumulative_return"])
            for name in ("momentum_daily", "momentum"):
                gaps[name] = metrics["cumulative_return"] - _metric_mean([item["baselines"][name] for item in scenarios])["cumulative_return"]
            paired, descriptive = {}, {}
            for mode, reference in references.items():
                candidates = [old for old in reference["records"] if old["size"] == "small" and old["seed"] == row["seed"]]
                for old in candidates:
                    if any(not _baseline_equal(row["metrics"][account]["baselines"], old["metrics"][account]["baselines"]) for account in MODES):
                        raise ValueError("Legacy fixed baseline values differ")
                    gap = metrics["cumulative_return"] - _metric_mean([item["ppo"] for item in old["metrics"].values()])["cumulative_return"]
                    descriptive[f"{mode}_{old['steps']}"] = gap
                    if old["steps"] == steps:
                        paired[mode] = gap
                if budget == "short" and mode not in paired:
                    raise ValueError("Legacy reference lacks a same-seed 8192 checkpoint")
            per_seed.append(dict(seed=row["seed"], metrics=metrics, initialized=initialized, return_gaps=gaps,
                same_budget_legacy_return_gaps=paired, descriptive_legacy_return_gaps=descriptive,
                total_cost=metrics["total_fees"] + metrics["total_slippage_cost"]))
        worst = max(abs(row["metrics"][mode]["ppo"]["max_drawdown"]) for row in rows for mode in MODES)
        positive = sum(row["return_gaps"]["initialized"] > 0 for row in per_seed)
        gates = dict(all_accounts_within_10_percent_drawdown=worst <= .10,
            mean_learning_increment_positive=_stats(row["return_gaps"]["initialized"] for row in per_seed)["mean"] > 0,
            at_least_two_positive_learning_seeds=positive >= 2,
            mean_return_above_daily_momentum=_stats(row["return_gaps"]["momentum_daily"] for row in per_seed)["mean"] > 0)
        cells[budget] = dict(steps=steps, parameters=rows[0]["parameters"],
            metrics=_metric_stats([row["metrics"] for row in per_seed]), initialized=_metric_stats([row["initialized"] for row in per_seed]),
            return_gaps={name: _stats(row["return_gaps"][name] for row in per_seed) for name in ("initialized", "momentum_daily", "momentum")},
            same_budget_legacy_return_gaps={mode: _stats(row["same_budget_legacy_return_gaps"][mode] for row in per_seed)
                for mode in references if all(mode in row["same_budget_legacy_return_gaps"] for row in per_seed)},
            descriptive_legacy_return_gaps={name: _stats(row["descriptive_legacy_return_gaps"][name] for row in per_seed)
                for name in per_seed[0]["descriptive_legacy_return_gaps"]},
            total_cost=_stats(row["total_cost"] for row in per_seed), worst_account_drawdown=worst,
            per_seed=per_seed, gates=gates, qualified=all(gates.values()), positive_learning_seeds=positive,
            per_account={mode: _metric_stats([row["metrics"][mode]["ppo"] for row in rows]) for mode in MODES},
            auxiliary={str(k): _metric_stats([_metric_mean(list(row["auxiliary_metrics"][str(k)].values())) for row in rows]) for k in (10, 40)},
            training_mode_statistics={str(row["seed"]): row.get("mode_statistics", {}) for row in rows},
            evaluation_mode_statistics={str(row["seed"]): {mode: row.get("evaluation_traces", {}).get("20", {}).get(mode, {}).get("mode_statistics", {}) for mode in MODES} for row in rows},
            actual_optimizer_steps={str(row["seed"]): row.get("optimizer_steps_min") for row in rows})
    eligible = [name for name, cell in cells.items() if cell["qualified"]]
    candidate = None
    if eligible:
        budget = min(eligible, key=lambda name: (-cells[name]["metrics"]["cumulative_return"]["mean"],
            cells[name]["worst_account_drawdown"], cells[name]["total_cost"]["mean"], cells[name]["steps"]))
        row = next(row for row in records if row["budget"] == budget and row["seed"] == 42)
        candidate = dict(budget=budget, main_top_n=20, representative_seed=42,
            mean_return=cells[budget]["metrics"]["cumulative_return"]["mean"], model_path=row.get("model_path"),
            model_sha256=row.get("model_sha256"), rule="fixed seed42; no lucky-seed selection; main K20 only")
    paired = [cells["long"]["per_seed"][index]["metrics"]["cumulative_return"] - cells["short"]["per_seed"][index]["metrics"]["cumulative_return"] for index in range(3)]
    return dict(seeds=SEEDS, account_scenarios=list(MODES), main_top_n=20, cells=cells, qualified_candidate=candidate,
        paired_long_minus_short=_stats(paired), actual_formal_steps=98304, checkpoint_steps_sum=122880,
        total_runs=3, total_checkpoints=6, auxiliary_used_for_selection=False, actionable=False,
        global_optimality_claim=False, test_evaluated=False,
        aggregation="Account mean within seed; sample SD across three seeds. Only old 8192 is same-budget paired; 32768 vs old65536 is descriptive.")


def write_report(root, records, references, protocol):
    root = Path(root)
    summary = summarize_study(records, references)
    selected = _selected_pilot(protocol, root)
    formal_core = deepcopy(protocol["core_contract"])
    formal_core["trade"]["actor_output_gain"] = selected["actor_output_gain"]
    summary.update(protocol_sha256=json_digest(protocol), limitations=protocol["limitations"],
        formal_core_contract=formal_core,
        formal_learning_rate=selected["learning_rate"], formal_initial_log_std=-.7,
        reference_contract_differences={mode: {key: dict(reference=reference["protocol"]["core_contract"].get(key), trade=formal_core.get(key))
            for key in sorted(set(reference["protocol"]["core_contract"]) | set(formal_core))
            if reference["protocol"]["core_contract"].get(key) != formal_core.get(key)} for mode, reference in references.items()},
        source_hash_differences={mode: {name: dict(reference=reference["protocol"]["source_hashes"].get(name), trade=protocol["source_hashes"].get(name))
            for name in sorted(set(protocol["source_hashes"]) | set(reference["protocol"]["source_hashes"]))
            if reference["protocol"]["source_hashes"].get(name) != protocol["source_hashes"].get(name)}
            for mode, reference in references.items()},
        hold_shares_reference=references["delta"].get("hold_shares"),
        selected_train_only_pilot=selected)
    write_json(root / "study_summary.json", summary)
    csv_rows = []
    for row in records:
        for account, scenario in row["metrics"].items():
            for name, metric in dict(ppo=scenario["ppo"], initialized=scenario["initialized"], **scenario["baselines"]).items():
                csv_rows.append(dict(budget=row["budget"], steps=row["steps"], seed=row["seed"], account=account, policy=name, **metric))
    pd.DataFrame(csv_rows).to_csv(root / "metrics.csv", index=False)
    lines = ["# 整手交易 Actor 研究", "", "4至6月开发验证已反复暴露，不是独立测试；2026年7月以后未评估。主K20，辅助K10/40不选模。", "",
        "| 快照 | 步数 | 净ROI% ± seed标准差 | 自身初始化增量百分点 | 每日动量差百分点 | 20日动量差百分点 | 最差回撤% | 平均总成本 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for budget, cell in summary["cells"].items():
        def stat(value):
            return f"{value['mean'] * 100:.4f} ± {value['std'] * 100:.4f}"
        lines.append(f"| {budget} | {cell['steps']:,} | {stat(cell['metrics']['cumulative_return'])} | {stat(cell['return_gaps']['initialized'])} | {stat(cell['return_gaps']['momentum_daily'])} | {stat(cell['return_gaps']['momentum'])} | {cell['worst_account_drawdown'] * 100:.4f} | {cell['total_cost']['mean']:.2f} |")
    candidate = summary["qualified_candidate"]
    lines += ["", "合格开发候选：" + (candidate["budget"] + "，固定展示seed42" if candidate else "null，无合格候选") + "。",
        "门槛：所有seed/账户回撤≤10%；平均自身初始化增量为正且至少2/3 seed为正；平均净ROI超过同账户每日动量。成本含实际费用与滑点。",
        "账户是同seed的配对情景，不是12个独立seed。各seed先平均四账户，再计算三seed样本标准差。",
        "仅8192与旧prior/random/delta小模型同预算配对；32768没有旧同预算模型，旧8192/65536仅作明确预算不等的描述。",
        "正式三run共98,304步，六快照步数和122,880不是训练量；pilot另8,192步，探针步骤不计训练量。",
        "训练原始采样轨迹与确定性部署轨迹分别封存：方向、条件latent、合法性/手数边界、精确前账本、请求股数、真实成交和拒单原因。NPZ可用allow_pickle=False读取；F20由日期和数据SHA恢复。",
        "HOLD只统计有多种合法方向的自主选择；真实EXIT必须成交后股数为0。条件熵拆分为类别熵和概率加权size熵；直接sigma来自log_std，不从2N存储维度反推。",
        "Critic v2拟合、PPO value loss和优化次数仅是诊断，不能证明Critic或新actor单独因果贡献。净收益奖励没有重复费用惩罚。",
        "源码和动作/执行机制相对旧轮均有变化，完整差异在机器摘要中披露，旧模型与原始指标未重跑。",
        "", *["- " + item for item in protocol["limitations"]], "", "![比较](comparison.png)", "",
        "[机器摘要](study_summary.json) | [逐账户指标](metrics.csv) | [原始记录](summary.json)"]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "summary.md").write_text("\n".join(lines[:10]) + "\n\n开发窗口已暴露；未评估保留期；不部署。\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels, values, errors = [], [], []
    for mode, reference in references.items():
        for steps in (8192, 65536):
            rows = [row for row in reference["records"] if row["size"] == "small" and row["steps"] == steps]
            if rows:
                stat = _stats(_metric_mean([item["ppo"] for item in row["metrics"].values()])["cumulative_return"] for row in rows)
                labels.append(f"{mode}\n{steps:,}")
                values.append(100 * stat["mean"])
                errors.append(100 * stat["std"])
    for budget, cell in summary["cells"].items():
        labels.append(f"trade\n{cell['steps']:,}")
        values.append(100 * cell["metrics"]["cumulative_return"]["mean"])
        errors.append(100 * cell["metrics"]["cumulative_return"]["std"])
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.bar(np.arange(len(labels)), values, yerr=errors, capsize=3, color=["#587b73"] * (len(labels) - 2) + ["#b15a73", "#3775a1"])
    ax.set(xticks=np.arange(len(labels)), xticklabels=labels, ylabel="Net ROI (%), account mean per seed",
        title="Exposed development window; only 8192 is matched-budget")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(root / "comparison.png", dpi=150)
    plt.close(fig)
    return summary


def run_formal(protocol, root):
    selected = _selected_pilot(protocol, root)
    records = []
    try:
        for seed in SEEDS:
            records.extend(run_trial(protocol, root, seed, selected))
            write_json(Path(root) / "summary.json", records)
        references = load_references({mode: value["root"] for mode, value in protocol["references"].items()})
        if any(reference["evidence"] != protocol["references"][mode]["evidence"] for mode, reference in references.items()):
            raise ValueError("Immutable reference registrations changed after preparation")
        summary = write_report(root, records, references, protocol)
        pilot = read_config(Path(root) / "pilot_selection.json")
        write_json(Path(root) / "status.json", dict(status="complete", total_trained_steps=98304,
            pilot_steps=pilot["actual_steps"], combined_training_steps=98304 + pilot["actual_steps"],
            probe_steps=pilot["probe_steps"], checkpoints=6, test_evaluated=False, actionable=False))
        return summary
    except Exception as error:
        write_json(Path(root) / "status.json", dict(status="failed", error=str(error), completed_checkpoints=len(records), test_evaluated=False))
        raise
    finally:
        _release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "pilot", "train"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for mode in ("prior", "random", "delta"):
        parser.add_argument("--" + mode + "-root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = resolve_path(args.output)
    protocol = prepare_study(read_config(resolve_path(args.config)), root,
        {mode: resolve_path(getattr(args, mode + "_root")) for mode in ("prior", "random", "delta")})
    print(f"STAGE {args.stage}: pilot 8,192; formal 98,304; combined 106,496; test disabled", flush=True)
    if args.stage == "prepare":
        return protocol
    if not torch.cuda.is_available():
        raise RuntimeError("Registered trade training requires working CUDA")
    return run_pilots(protocol, root) if args.stage == "pilot" else run_formal(protocol, root)


if __name__ == "__main__":
    main()
