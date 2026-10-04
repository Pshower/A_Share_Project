"""Posthoc inference replay of sealed main-K20 validation, never a training action log."""

import argparse
import ast
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from src.agents.delta_mapping import DeltaOrder
from src.agents.ppo_agent import PPOAgent
from src.backtest.run import run_policy
from src.data.preprocess import ROOT, file_hash, read_config, resolve_path, write_json
from src.envs.initial_account import MODES
from src.training.env_factory import ResearchData
from src.training.freeze import json_digest, policy_digest, source_hashes
from src.training.matrix_compare import load_matrix
from src.training.research_delta import BUDGETS, DATA_FIELDS, SEEDS, _selected_pilot, config_for
from src.training.research_matrix import verify_completed


def reject_test_flags(value):
    if isinstance(value, dict):
        if "test_evaluated" in value and value["test_evaluated"] is not False:
            raise ValueError("Final-test or undeclared test evaluation is not supported")
        for item in value.values():
            reject_test_flags(item)
    elif isinstance(value, list):
        for item in value:
            reject_test_flags(item)


def read_envelope(study):
    study = Path(study).resolve()
    paths = {name: study / (name + ".json") for name in ("protocol", "status", "summary", "study_summary")}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("Analysis requires a completed study and study_summary.json")
    values = {name: read_config(path) for name, path in paths.items()}
    protocol, status, summary, records = (values[name] for name in ("protocol", "status", "study_summary", "summary"))
    for value in (protocol, status, summary):
        if value.get("test_evaluated") is not False:
            raise ValueError("Study must explicitly exclude final-test evaluation")
    reject_test_flags(values)
    if (status.get("status") != "complete" or status.get("checkpoints") != 6
            or status.get("total_trained_steps") != 196608
            or summary.get("total_runs") != 3 or summary.get("total_checkpoints") != 6
            or summary.get("actual_formal_steps") != 196608):
        raise ValueError("All three formal seeds and six checkpoints must be complete")
    if (protocol.get("study_mode") != "delta" or protocol.get("action_mode") != "stock_delta"
            or protocol.get("seeds") != SEEDS or summary.get("seeds") != SEEDS
            or protocol.get("main_top_n") != 20 or summary.get("main_top_n") != 20
            or protocol.get("evaluation_seed") != 20260927 or protocol.get("evaluation_modes") != list(MODES)
            or any(protocol.get(name + "_steps") != steps for name, steps in BUDGETS.items())
            or summary.get("protocol_sha256") != json_digest(protocol)):
        raise ValueError("Registered study identity or main-K20 protocol changed")
    expected = {(seed, budget) for seed in SEEDS for budget in BUDGETS}
    if (not isinstance(records, list) or len(records) != 6
            or {(row.get("seed"), row.get("budget")) for row in records} != expected
            or any(row.get("test_evaluated") is not False or row.get("initialization_mode") != "delta"
                   or row.get("steps") != BUDGETS[row["budget"]] for row in records)):
        raise ValueError("Summary lacks the six registered delta checkpoint identities")
    frozen_sources = protocol.get("source_hashes")
    if not isinstance(frozen_sources, dict) or len(frozen_sources) != 70 or frozen_sources != source_hashes():
        raise ValueError("Frozen 70-source identity changed")
    return study, protocol, summary, records


def verify_snapshot(directory, config, contract, frozen_sources, steps, expected_digest=None):
    directory = Path(directory)
    metadata = read_config(directory / "metadata.json")
    required = {"config", "contract", "source_hashes", "num_timesteps", "model_sha256"}
    if not required <= set(metadata):
        raise ValueError("Model metadata lacks a registered identity")
    digest = file_hash(directory / "model.zip")
    if (metadata["config"] != config or metadata["contract"] != contract
            or metadata["source_hashes"] != frozen_sources or type(metadata["num_timesteps"]) is not int
            or metadata["num_timesteps"] != steps or metadata["model_sha256"] != digest
            or expected_digest is not None and expected_digest != digest):
        raise ValueError("Model config, data, source, budget or SHA identity changed")
    reject_test_flags(metadata)
    return metadata


def register_files(directory, files, manifest):
    directory = Path(directory).resolve()
    if not isinstance(files, dict) or not files:
        raise ValueError("Registered artifact manifest is missing")
    for name, digest in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or file_hash(path) != digest:
            raise ValueError("Registered artifact changed: " + str(path))
        manifest[str(path)] = digest


def verify_references(protocol, manifest):
    references = protocol.get("references")
    if not isinstance(references, dict) or set(references) != {"prior", "random"}:
        raise ValueError("Registered prior/random evidence is missing")
    for mode, registered in references.items():
        directory = resolve_path(registered["root"]).resolve()
        verified = load_matrix(directory, mode)
        if (verified["evidence"] != registered.get("evidence")
                or any(verified["protocol"]["core_contract"].get(field) != protocol["core_contract"].get(field)
                       for field in DATA_FIELDS)):
            raise ValueError("Registered reference evidence or dataset changed")
        for name in ("protocol.json", "status.json", "summary.json"):
            path = directory / name
            manifest[str(path)] = file_hash(path)
        for marker in (directory / "runs").glob("*/completed.json"):
            manifest[str(marker.resolve())] = file_hash(marker)
            register_files(marker.parent, read_config(marker)["files"], manifest)
        for marker in (directory / "controls").rglob("controls.json"):
            manifest[str(marker.resolve())] = file_hash(marker)
            register_files(marker.parent, read_config(marker)["files"], manifest)


def verify_study(study):
    study, protocol, summary, records = read_envelope(study)
    candidate = _selected_pilot(protocol, study)
    if summary.get("selected_train_only_pilot") != candidate:
        raise ValueError("Study summary and registered pilot selection differ")
    manifest = {str(study / name): file_hash(study / name) for name in
                ("protocol.json", "status.json", "summary.json", "study_summary.json", "pilot_selection.json")}
    verify_references(protocol, manifest)
    for candidate_spec in protocol["pilot_candidates"]:
        directory = study / "pilots" / candidate_spec["name"]
        marker = directory / "completed.json"
        manifest[str(marker)] = file_hash(marker)
        register_files(directory, read_config(marker)["files"], manifest)
    cases = []
    for seed in SEEDS:
        config = config_for(protocol["base_config"], study, seed, candidate)
        directory = study / "runs" / f"small_{seed}"
        completed = verify_completed(directory, config, protocol)
        if completed is None:
            raise ValueError("Incomplete formal seed: " + str(seed))
        expected = [row for row in records if row["seed"] == seed]
        if sorted(completed, key=lambda row: row["budget"]) != sorted(expected, key=lambda row: row["budget"]):
            raise ValueError("Summary differs from completed checkpoint evidence")
        marker = directory / "completed.json"
        manifest[str(marker)] = file_hash(marker)
        files = read_config(marker)["files"]
        required = {f"{budget}_validation/{mode}/{name}" for budget in BUDGETS for mode in MODES
                    for name in ("metrics.json", "experiment.json", "ppo/daily.csv", "ppo/orders.csv", "ppo/positions.parquet")}
        required |= {"initial_model/model.zip", "initial_model/metadata.json"}
        if not required <= set(files):
            raise ValueError("Completed seed lacks registered main-K20 evidence")
        register_files(directory, files, manifest)
        data = ResearchData(config)
        core = deepcopy(data.contract)
        core.pop("hidden_sizes")
        if core != protocol["core_contract"] or data.contract.get("action_mode") != "stock_delta":
            raise ValueError("Current data differs from the frozen delta contract")
        clean = resolve_path(data.research["dataset"]["clean_dir"])
        manifest[str(clean / "manifest.json")] = data.contract["manifest_sha256"]
        for name, digest in data.contract["artifacts"].items():
            if file_hash(clean / name) != digest:
                raise ValueError("Frozen dataset artifact changed")
            manifest[str(clean / name)] = digest
        initial = verify_snapshot(directory / "initial_model", config, data.contract, protocol["source_hashes"], 0)
        control_directory = study / "controls" / f"small_{seed}"
        control_path = control_directory / "controls.json"
        controls = read_config(control_path)
        identity = dict(contract=data.contract, training_seed=seed, evaluation_seed=20260927,
                        evaluation_modes=list(MODES), top_n=20, initial_model_sha256=initial["model_sha256"])
        if controls.get("identity") != identity:
            raise ValueError("Apriori initialization-control identity changed")
        required_controls = {f"{mode}/{name}" for mode in MODES for name in
                             ("metrics.json", "ppo/daily.csv", "ppo/orders.csv", "ppo/positions.parquet")}
        if not required_controls <= set(controls.get("files", {})):
            raise ValueError("Apriori controls lack required registered artifacts")
        manifest[str(control_path)] = file_hash(control_path)
        register_files(control_directory, controls["files"], manifest)
        for row in expected:
            model_directory = directory / (row["budget"] + "_model")
            if resolve_path(row["model_path"]).resolve() != model_directory.resolve():
                raise ValueError("Model path does not match checkpoint identity")
            metadata = verify_snapshot(model_directory, config, data.contract, protocol["source_hashes"],
                                       row["steps"], row["model_sha256"])
            for mode in MODES:
                saved = directory / (row["budget"] + "_validation") / mode
                metrics = read_config(saved / "metrics.json")
                experiment = read_config(saved / "experiment.json")
                settings = experiment.get("evaluation_settings", {})
                if (metrics.get("ppo") != row["metrics"][mode]["ppo"]
                        or read_config(control_directory / mode / "metrics.json") != controls["metrics"][mode]
                        or controls["metrics"][mode]["ppo"] != row["metrics"][mode]["initialized"]
                        or experiment.get("split") != "val" or experiment.get("contract") != data.contract
                        or experiment.get("config") != config or settings.get("seed") != 20260927
                        or settings.get("initial_portfolio") != mode or settings.get("delta_top_n") != 20
                        or settings.get("broker") != data.settings["broker"]
                        or settings.get("initial_capital") != data.settings["initial_capital"]):
                    raise ValueError("Sealed evaluation or initialization identity changed")
                cases.append(dict(seed=seed, budget=row["budget"], account=mode, data=data, settings=settings,
                                  model_directory=model_directory, saved=saved, metadata=metadata,
                                  initial_model_sha256=initial["model_sha256"]))
    return study, protocol, cases, manifest


class ReplayRecorder:
    def __init__(self, agent):
        self.agent, self.contract = agent, agent.contract
        self.observations, self.actions = [], []

    def act(self, observation, step=None):
        result = self.agent.inspect_action(observation, deterministic=True)
        self.observations.append({name: np.asarray(value).copy() for name, value in observation.items()})
        self.actions.append({name: np.asarray(value).copy() for name, value in result.items()})
        return DeltaOrder(result["delta_weights"], result["freeze_mask"], result["exit_mask"], result["top_n"], result)


def runtime_arrays(recorder):
    output, schema = {}, {}
    for prefix, rows in (("observation", recorder.observations), ("action", recorder.actions)):
        if not rows:
            raise ValueError("Replay did not produce any runtime observations/actions")
        fields = list(rows[0])
        if any(list(row) != fields for row in rows):
            raise ValueError("Runtime field schema changed within replay")
        schema[prefix] = []
        for name in fields:
            array = np.stack([row[name] for row in rows])
            if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                raise ValueError("Runtime field is not finite numeric data: " + name)
            key = prefix + "__" + name
            output[key] = array
            schema[prefix].append(dict(name=name, npz_key=key, dtype=str(array.dtype), shape=list(array.shape)))
    return output, schema


def compare_replay(daily, positions, orders, saved):
    expected_daily = pd.read_csv(saved / "ppo/daily.csv", parse_dates=["date"])
    expected_positions = pd.read_parquet(saved / "ppo/positions.parquet")
    expected_orders = pd.read_csv(saved / "ppo/orders.csv", dtype={"code": str}, parse_dates=["date"])
    actual_orders, expected_orders = orders.copy(), expected_orders.copy()
    for frame in (actual_orders, expected_orders):
        frame["reasons"] = frame.reasons.map(lambda value: tuple(ast.literal_eval(value)) if isinstance(value, str) else tuple(value))
    for name, actual, expected in (("daily", daily, expected_daily), ("positions", positions, expected_positions),
                                  ("orders", actual_orders, expected_orders)):
        try:
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False,
                                          rtol=0, atol=1e-8)
        except AssertionError as error:
            raise ValueError("Inference replay differs from sealed " + name) from error


def describe_actions(arrays, daily, orders, broker, max_delta_weight):
    latent, deltas = arrays["action__latent"], arrays["action__delta_weights"]
    eligible, frozen = arrays["action__eligible"].astype(bool), arrays["action__freeze_mask"].astype(bool)
    lots = arrays["observation__execution_features"][:, :, 0]
    positive = eligible & ~frozen & arrays["observation__buy_mask"].astype(bool) & (latent > 0)
    before_projection = max_delta_weight * np.tanh(arrays["action__clipped_latent"])
    below_lot = positive & (before_projection < lots)
    threshold_valid = (lots > 0) & (lots < max_delta_weight)
    thresholds = np.full(lots.shape, np.nan)
    thresholds[threshold_valid] = np.arctanh(lots[threshold_valid] / max_delta_weight)
    filled = orders.filled_shares.to_numpy(dtype=float)
    requested = orders.requested_shares.to_numpy(dtype=float)
    prices = orders.price.to_numpy(dtype=float)
    notionals = np.where(np.isfinite(prices), np.abs(filled) * prices, 0.)
    requested_notionals = np.where(np.isfinite(prices), np.abs(requested) * prices, 0.)
    commissions = orders.fee.to_numpy(dtype=float) - np.where(filled < 0, notionals * broker["sell_tax"], 0.)
    filled_any = filled != 0
    minimum = filled_any & np.isclose(commissions, broker["min_commission"], rtol=0, atol=1e-8)
    order_dates = set(pd.to_datetime(orders.date))
    active_latents = np.abs(latent[eligible])
    stats = dict(
        decision_steps=len(daily) - 1, requested_buy_orders=int((orders.side == "buy").sum()),
        requested_sell_orders=int((orders.side == "sell").sum()), filled_buy_orders=int((filled > 0).sum()),
        filled_sell_orders=int((filled < 0).sum()), rejected_orders=int((orders.status == "rejected").sum()),
        partial_orders=int((orders.status == "partial").sum()), one_share_sells=int(((filled < 0) & (np.abs(filled) == 1)).sum()),
        requested_buy_shares=float(requested[requested > 0].sum()),
        requested_sell_shares=float(-requested[requested < 0].sum()),
        requested_notional_at_logged_price=float(requested_notionals.sum()),
        unpriced_requested_orders=int((~np.isfinite(prices)).sum()),
        minimum_commission_orders=int(minimum.sum()), minimum_commission_sells=int((minimum & (filled < 0)).sum()),
        filled_buy_notional=float(notionals[filled > 0].sum()), filled_sell_notional=float(notionals[filled < 0].sum()),
        fees=float(orders.fee.sum()), slippage=float(orders.slippage_cost.sum()),
        no_order_days=sum(pd.Timestamp(date) not in order_dates for date in daily.date.iloc[1:]),
        posthoc_abs_latent_median=float(np.median(np.abs(latent))),
        posthoc_abs_eligible_latent_median=float(np.median(active_latents)) if active_latents.size else None,
        zero_delta_fraction=float(np.mean(deltas == 0)), positive_eligible_buy_latents=int(positive.sum()),
        positive_eligible_buy_latents_below_lot=int(below_lot.sum()),
        finite_lot_latent_threshold_median=float(np.median(thresholds[threshold_valid])) if threshold_valid.any() else None,
        decision_frozen_coordinate_fraction=float(frozen.mean()),
        decision_frozen_weight_mean=float(arrays["action__frozen_weight"].mean()),
        decision_locked_weight_mean=float(arrays["action__locked_weights"].sum(axis=1).mean()),
        top_n_excess_days=int((arrays["action__top_n_excess"] > 0).sum()),
        maximum_top_n_excess=int(arrays["action__top_n_excess"].max()),
        minimum_actual_cash=float(daily.cash.min()), initial_holdings=int(daily.holding_count.iloc[0]),
        final_holdings=int(daily.holding_count.iloc[-1]),
    )
    stats["total_cost"] = stats["fees"] + stats["slippage"]
    return stats, thresholds


def model_rows(rows):
    result = []
    for seed in SEEDS:
        for budget in BUDGETS:
            cases = [row for row in rows if (row["seed"], row["budget"]) == (seed, budget)]
            if len(cases) != 4 or {row["account"] for row in cases} != set(MODES):
                raise ValueError("Analysis lacks four accounts for a fixed checkpoint")
            medians = [row["posthoc_abs_eligible_latent_median"] for row in cases
                       if row["posthoc_abs_eligible_latent_median"] is not None]
            result.append(dict(seed=seed, budget=budget, accounts=len(cases),
                filled_buys=sum(row["filled_buy_orders"] for row in cases),
                filled_sells=sum(row["filled_sell_orders"] for row in cases),
                one_share_sells=sum(row["one_share_sells"] for row in cases),
                mean_account_fees=float(np.mean([row["fees"] for row in cases])),
                mean_account_slippage=float(np.mean([row["slippage"] for row in cases])),
                mean_account_cost=float(np.mean([row["total_cost"] for row in cases])),
                mean_account_eligible_latent_abs_median=float(np.mean(medians)) if medians else None))
    return result


def analyze(study, device="cpu"):
    if device not in {"cpu", "cuda"}:
        raise ValueError("Analysis device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA inference is unavailable")
    target = Path(study).resolve() / "analysis" / "main_k20"
    if target.exists():
        raise FileExistsError("Analysis output already exists: " + str(target))
    torch.set_num_threads(1)
    with ExitStack() as guards:
        def forbidden(*args, **kwargs):
            raise RuntimeError("Training/backward/optimizer updates are forbidden in posthoc analysis")
        for owner, name in ((PPO, "learn"), (PPO, "train"), (torch.Tensor, "backward"), (torch.optim.Adam, "step")):
            guards.enter_context(patch.object(owner, name, forbidden))
        study, protocol, cases, inputs = verify_study(study)
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = target.parent / (".main_k20_" + uuid.uuid4().hex)
        stage.mkdir()
        if stage.resolve().parent != target.parent.resolve():
            raise ValueError("Analysis staging escaped the study directory")
        rows, actual_devices = [], set()
        for case in cases:
            data = case["data"]
            agent = PPOAgent.load(case["model_directory"], data.contract, device)
            before = policy_digest(agent)
            recorder = ReplayRecorder(agent)
            daily, positions, orders, _ = run_policy(data.market("val"), recorder, deepcopy(case["settings"]))
            if (pd.Timestamp(daily.date.iloc[0]) != pd.Timestamp(protocol["train_end"])
                    or pd.Timestamp(daily.date.iloc[-1]) != pd.Timestamp(protocol["val_end"])):
                raise ValueError("Replay escaped the registered development dates")
            compare_replay(daily, positions, orders, case["saved"])
            if policy_digest(agent) != before or agent.model.num_timesteps != case["metadata"]["num_timesteps"]:
                raise ValueError("Replay changed model parameters or training counters")
            arrays, schema = runtime_arrays(recorder)
            stats, thresholds = describe_actions(arrays, daily, orders, data.contract["broker"],
                                                  data.contract["delta"]["max_delta_weight"])
            identity = dict(seed=case["seed"], budget=case["budget"], account=case["account"])
            folder = stage / f"small_{case['seed']}" / case["budget"] / case["account"]
            folder.mkdir(parents=True)
            arrays.update(codes=np.asarray(data.codes), decision_dates=daily.date.iloc[:-1].astype(str).to_numpy(dtype=str),
                          execution_dates=daily.date.iloc[1:].astype(str).to_numpy(dtype=str),
                          derived__buy_lot_latent_threshold=thresholds)
            np.savez_compressed(folder / "actions.npz", **arrays)
            initial = dict(mode=case["account"], date=str(pd.Timestamp(daily.date.iloc[0]).date()),
                           equity=float(daily.equity.iloc[0]), cash=float(daily.cash.iloc[0]),
                           positions={code: float(value) for code, value in positions.iloc[0].items() if value != 0})
            actual_device = str(agent.model.device)
            actual_devices.add(actual_device)
            metadata = dict(identity, main_top_n=20, evaluation_seed=20260927, runtime_schema=schema,
                            codes=data.codes, feature_cols=data.contract["feature_cols"], initial_account=initial,
                            model_sha256=case["metadata"]["model_sha256"], initial_model_sha256=case["initial_model_sha256"],
                            manifest_sha256=data.contract["manifest_sha256"], data_artifacts=data.contract["artifacts"],
                            source_hashes=protocol["source_hashes"], arrays_sha256=file_hash(folder / "actions.npz"),
                            observation_masks="Decision observations, not future execution masks",
                            lot_threshold="Necessary pre-budget close-mark threshold; not a fill guarantee",
                            latent_definition="Deterministic masked distribution mean, not unmasked network scores",
                            requested_device=device, actual_device=actual_device,
                            posthoc_deterministic_replay=True, training_action_log=False, used_for_selection=False,
                            split="val", test_evaluated=False)
            write_json(folder / "metadata.json", metadata)
            write_json(folder / "statistics.json", stats)
            rows.append(dict(identity, **stats))
            del agent, recorder, arrays
        if len(rows) != 24 or source_hashes() != protocol["source_hashes"]:
            raise ValueError("Incomplete replay or changed frozen source")
        for name, digest in inputs.items():
            if file_hash(name) != digest:
                raise ValueError("Sealed input changed during replay: " + name)
        summary = dict(schema_version=1, study=str(study), protocol_sha256=json_digest(protocol),
                       main_top_n=20, evaluation_seed=20260927, scenarios_checked=24, rows=rows, model_rows=model_rows(rows),
                       replay_matches_sealed=True, tolerance=1e-8, frozen_source_count=70,
                       posthoc_deterministic_replay=True, training_action_log=False, used_for_selection=False,
                       requested_device=device, actual_devices=sorted(actual_devices),
                       new_period_evaluated=False, test_evaluated=False,
                       field_definitions=dict(
                           positive_eligible_buy_latents_below_lot="Positive mapped-eligible, buy-allowed means below the necessary close-mark lot threshold BEFORE budget/topN/cap/execution projection; not an exhaustive rejection-cause classification",
                           requested_notional_at_logged_price="Requested shares times the recorded side-specific price; unpriced requests excluded, not raw opening notional",
                           posthoc_abs_latent_median="Median absolute deterministic masked latent; not a training mean or sampled-action statistic"),
                       limitations=["Only the six fixed snapshots and four registered main-K20 accounts were replayed.",
                                    "Latents are posthoc deterministic masked means, not training samples.",
                                    "Frozen/locked/topN fields are observed runtime diagnostics, not inferred future masks.",
                                    "HFQ and missing historical suspension/limit/corporate-action data remain unchanged."])
        write_json(stage / "summary.json", summary)
        write_json(stage / "sealed_inputs.json", inputs)
        if target.exists():
            raise FileExistsError(target)
        stage.rename(target)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="CPU by default; CUDA is inference-only and still requires exact sealed-trade replay")
    args = parser.parse_args(argv)
    result = analyze(args.study, args.device)
    print(result)
    return result


if __name__ == "__main__":
    main()
