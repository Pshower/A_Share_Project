"""Frozen-model deployment, feasibility and reward-identity research probes.

These probes never learn, download data or promote a model. Deployment paths are
complete training-period trajectories, not same-state causal comparisons. Pool
and capital probes assess legal requests only, not unseen-stock profitability.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from src.agents.ppo_agent import PPOAgent, make_model
from src.agents.trade_mapping import BUY, HOLD, REDUCE, EXIT, map_trade_action
from src.backtest.metrics import performance
from src.backtest.run import run_policy
from src.data.preprocess import ROOT, file_hash, write_json
from src.envs.initial_account import MODES
from src.envs.market import MarketDataProvider
from .env_factory import ResearchData
from .freeze import json_digest
from .research_matrix import preserve_training_rng
from .research_trade import TradeTrace, _json_safe, _probe_env, _trace_identity


MODEL_SEEDS = (42, 123, 2026)
BUDGETS = {"short": 8192, "long": 32768}
SAMPLING_SEEDS = (101, 202, 303, 404, 505)
PROBE_INDICES = (-52, -32, -12)
PROBE_MODES = ("cash", "random")
ACCOUNT_SEED = 20260927
SAME_STATE_SAMPLES = 64
POOL_SIZES = (30, 100, 300)
CAPITALS = (250000., 1000000., 4000000.)


def parameter_digest(policy):
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def verify_old_identity(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if file_hash(directory / "model.zip") != metadata["model_sha256"]:
        raise ValueError("Frozen model ZIP identity changed")
    sources = metadata.get("source_hashes", {})
    if len(sources) != 73:
        raise ValueError("Frozen trade model must carry the original 73 source hashes")
    for relative, expected in sources.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or file_hash(path) != expected:
            raise ValueError("Original frozen source identity changed: " + relative)
    return metadata, {"model_sha256": metadata["model_sha256"],
        "original_source_count": len(sources), "original_source_digest": json_digest(sources),
        "original_git": metadata.get("git"), "contract_sha256": json_digest(metadata["contract"])}


def _frozen_models(old_root):
    for seed in MODEL_SEEDS:
        for budget, steps in BUDGETS.items():
            directory = Path(old_root) / "runs" / f"small_{seed}" / f"{budget}_model"
            metadata, identity = verify_old_identity(directory)
            if metadata.get("num_timesteps") != steps:
                raise ValueError("Frozen model budget does not match its registered snapshot")
            data = ResearchData(metadata["config"])
            if data.contract != metadata["contract"]:
                raise ValueError("Original dataset, stock order or execution contract changed")
            agent = PPOAgent.load(directory, metadata["contract"], device="cpu")
            agent.model.policy.set_training_mode(False)
            identity.update(model_seed=seed, budget=budget, steps=steps,
                            parameter_sha256=parameter_digest(agent.model.policy),
                            model_path=str(directory.resolve()))
            yield agent, data, identity


def _deployment_market(data):
    start, end = pd.Timestamp("2026-01-05"), pd.Timestamp("2026-03-31")
    if end > pd.Timestamp(data.manifest["train_end"]):
        raise ValueError("Deployment probe crosses the original training boundary")
    prior = data.features.index[data.features.index < start]
    if not len(prior):
        raise ValueError("Deployment probe lacks its preceding decision close")
    decision = prior[-1]
    return MarketDataProvider(data.features.loc[:end, pd.IndexSlice[data.codes, :]],
        data.manifest["feature_cols"], price_path=data.prices.loc[:end, pd.IndexSlice[data.codes, :]],
        price_basis=data.manifest["price_basis"], start_date=decision, end_date=end,
        lookback=data.config["lookback"])


def _same_state_envs(data):
    market = data.market("train")
    if market.n_dates < -min(PROBE_INDICES):
        raise ValueError("Registered fixed training states are unavailable")
    for negative in PROBE_INDICES:
        index = market.n_dates + negative
        for mode in PROBE_MODES:
            env = _probe_env(data, market, index, mode)
            try:
                observation, account = env.reset(seed=ACCOUNT_SEED)
                yield negative, mode, market.dates[index], env, observation, account
            finally:
                env.close()


class DeploymentStrategy:
    def __init__(self, agent, deterministic):
        self.agent, self.deterministic = agent, deterministic
        self.contract = agent.contract

    def act(self, observation, step=None):
        return self.agent.act(observation, step, deterministic=self.deterministic)


def _native_trace_sink(trace, agent, final_date=None):
    def append(date, observation, action, info):
        tensor, _ = agent.model.policy.obs_to_tensor(observation)
        raw = np.concatenate([action.diagnostics["directions"], action.diagnostics["size_latent"]])
        packed = torch.as_tensor(raw, device=agent.model.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            value, probability, _ = agent.model.policy.evaluate_actions(tensor, packed)
        trace.append(date, observation, action, info,
            log_prob=probability.cpu().numpy()[0], value=value.cpu().numpy()[0],
            reward=info["equity"] / info["decision_snapshot"]["equity"] - 1,
            done=None if final_date is None else bool(pd.Timestamp(info["date"]) == pd.Timestamp(final_date)))
    return append


def _one_deployment(agent, data, identity, destination, mode, sample_seed):
    settings = deepcopy(data.settings)
    settings.update(initial_portfolio=mode, seed=ACCOUNT_SEED, trade_top_n=20)
    destination.mkdir(parents=True, exist_ok=False)
    trace_identity = _trace_identity(data, "training_period_deployment", initial_mode=mode,
        deterministic=sample_seed is None, sampling_seed=sample_seed)
    trace_identity.update(deepcopy(identity))
    trace = TradeTrace(destination / "native_trace", trace_identity, chunk_steps=128)
    try:
        market = _deployment_market(data)
        with preserve_training_rng(torch.device("cpu")), torch.no_grad():
            torch.manual_seed(0 if sample_seed is None else sample_seed)
            np.random.seed(0 if sample_seed is None else sample_seed)
            random.seed(0 if sample_seed is None else sample_seed)
            daily, positions, orders, assumptions = run_policy(market,
                DeploymentStrategy(agent, sample_seed is None), settings,
                trace_sink=_native_trace_sink(trace, agent, market.dates[-1]))
    finally:
        manifest = trace.close()
    daily.to_csv(destination / "daily.csv", index=False)
    positions.to_parquet(destination / "positions.parquet")
    orders.to_csv(destination / "orders.csv", index=False)
    row = {**identity, "mode": mode, "deterministic": sample_seed is None,
        "sampling_seed": sample_seed, "account_seed": ACCOUNT_SEED, "top_n": 20,
        "initial_date": str(pd.Timestamp(daily.date.iloc[0]).date()),
        "final_date": str(pd.Timestamp(daily.date.iloc[-1]).date()),
        "environment_steps": len(daily) - 1, "metrics": performance(daily),
        "mode_statistics": manifest["mode_statistics"], "path": str(destination),
        "assumptions": assumptions}
    write_json(destination / "metrics.json", _json_safe(row))
    return row


def sample_same_state(agent, observation, samples=SAME_STATE_SAMPLES, seed=101):
    """Sample one fixed observation; proposed requests are not actual fills."""
    if type(samples) is not int or samples < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Same-state sample count/seed must be positive/nonnegative integers")
    before = parameter_digest(agent.model.policy)
    untouched = {key: value.copy() for key, value in observation.items()}
    tensor, _ = agent.model.policy.obs_to_tensor(observation)
    with preserve_training_rng(torch.device("cpu")), torch.no_grad():
        torch.manual_seed(seed)
        distribution = agent.model.policy.get_distribution(tensor)
        probability = distribution.direction_distribution.probs.cpu().numpy()[0].copy()
        mean = distribution.size_distribution.loc.cpu().numpy()[0].copy()
        sigma = distribution.size_distribution.scale.cpu().numpy()[0].copy()
        raw = np.stack([distribution.sample().cpu().numpy()[0] for _ in range(samples)])
        log_probs = np.asarray([distribution.log_prob(torch.as_tensor(item,
            device=agent.model.device).unsqueeze(0)).cpu().item() for item in raw])
        deterministic = distribution.mode().cpu().numpy()[0].copy()
    contract = agent.contract
    requests = [map_trade_action(item, observation, contract["logit_bound"],
        contract["broker"]["lot_size"], contract["stock_codes"]) for item in raw]
    deterministic_request = map_trade_action(deterministic, observation, contract["logit_bound"],
        contract["broker"]["lot_size"], contract["stock_codes"])
    n = len(mean)
    active_mean = mean[observation["legal_directions"][:, BUY].astype(bool)
        | (observation["legal_directions"][:, REDUCE].astype(bool)
           & (observation["trade_lot_bounds"][:, 3] > observation["trade_lot_bounds"][:, 2]))]
    arrays = dict(direction_probabilities=probability, size_mean=mean, size_sigma=sigma,
        raw_packed_action=raw, log_prob=log_probs, deterministic_raw=deterministic,
        deterministic_shares=deterministic_request["shares"],
        proposed_shares=np.stack([item["proposed_shares"] for item in requests]),
        projected_shares=np.stack([item["shares"] for item in requests]),
        top_n_filtered=np.stack([item["top_n_filtered"] for item in requests]),
        cash_filtered=np.stack([item["cash_filtered"] for item in requests]))
    for key, value in observation.items():
        arrays["obs_" + key] = value.copy()
    if before != parameter_digest(agent.model.policy):
        raise AssertionError("Same-state inference changed model parameters")
    for key, value in observation.items():
        np.testing.assert_array_equal(value, untouched[key])
    summary = dict(samples=samples, sampling_seed=seed, environment_steps=0,
        actual_fills_evaluated=False, parameters_unchanged=True,
        active_abs_mean_median=float(np.median(np.abs(active_mean))) if len(active_mean) else None,
        sigma_mean=float(sigma.mean()), legal_buy_count=int(observation["legal_directions"][:, BUY].sum()),
        deterministic_modes={name: int((deterministic[:n] == code).sum())
            for code, name in enumerate(("hold", "buy", "reduce", "exit"))},
        sample_mean_modes={name: float((raw[:, :n] == code).sum(axis=1).mean())
            for code, name in enumerate(("hold", "buy", "reduce", "exit"))},
        deterministic_projected_nonzero=int(np.count_nonzero(deterministic_request["shares"])),
        sampled_projected_nonzero_mean=float(np.count_nonzero(arrays["projected_shares"], axis=1).mean()))
    return summary, arrays


def run_deployment_probes(old_root, output_root):
    destination = Path(output_root) / "deployment"
    destination.mkdir(parents=True, exist_ok=False)
    rows, fixed, identities = [], [], []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with preserve_training_rng(torch.device("cpu")):
            for agent, data, identity in _frozen_models(old_root):
                identities.append(identity)
                model_folder = destination / f"{identity['budget']}_{identity['model_seed']}"
                for mode in MODES:
                    for sample_seed in (None, *SAMPLING_SEEDS):
                        name = "deterministic" if sample_seed is None else f"sample_{sample_seed}"
                        folder = model_folder / mode / name
                        rows.append(_one_deployment(agent, data, identity, folder, mode, sample_seed))
                for negative, mode, date, env, obs, account in _same_state_envs(data):
                    summary, arrays = sample_same_state(agent, obs)
                    folder = model_folder / "same_state"
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder / f"index_{negative}_{mode}.npz"
                    np.savez_compressed(path, **arrays)
                    fixed.append({**identity, **summary, "train_index": negative, "mode": mode,
                        "date": str(date.date()), "path": str(path), "sha256": file_hash(path),
                        "account": _json_safe(account)})
                if parameter_digest(agent.model.policy) != identity["parameter_sha256"]:
                    raise AssertionError("Deployment evaluation changed frozen model parameters")
    finally:
        torch.set_num_threads(previous_threads)
    report = dict(schema_version=1, status="complete", stage="deployment", device="cpu",
        training_updates=0, training_steps=0, validation_evaluated=False, test_evaluated=False,
        model_count=len(identities), path_case_count=len(rows), same_state_case_count=len(fixed),
        sampled_paths_per_model_account=len(SAMPLING_SEEDS), sampling_seeds=list(SAMPLING_SEEDS),
        account_seed=ACCOUNT_SEED, models=identities, cases=rows, same_state_cases=fixed,
        interpretation="Complete policy trajectories differ in subsequent states; no same-state causal effect or sampled winner selection.")
    write_json(destination / "summary.json", _json_safe(report))
    return report


def feasibility_summary(observation, account, contract):
    """Record overlapping mask/economic blockers; never label requests as fills."""
    equity = float(account["equity"])
    legal = observation["legal_directions"].astype(bool)
    valid = observation["valid_mask"].astype(bool)
    frozen = observation["execution_features"][:, 2].astype(bool)
    buyable = observation["buy_mask"].astype(bool)
    bounds = observation["trade_lot_bounds"]
    lot_values = observation["execution_features"][:, 0].astype(float) * equity
    minimum = contract["broker"]["min_commission"] / contract["trade"]["max_commission_ratio"]
    holding = observation["share_state"][:, 0] > 0
    unavoidable = int((holding & ~legal[:, EXIT]).sum())
    result = dict(equity=equity, cash=float(account["cash"]),
        held_names=int(holding.sum()), legal_buy_count=int(legal[:, BUY].sum()),
        legal_reduce_count=int(legal[:, REDUCE].sum()), legal_exit_count=int(legal[:, EXIT].sum()),
        only_hold_count=int((legal.sum(1) == 1).sum()), minimum_trade_notional=minimum,
        blockers_overlap=True, blockers=dict(invalid=int((~valid).sum()), frozen=int(frozen.sum()),
            buy_mask_false=int((~buyable).sum()),
            economic_lot_interval_empty=int((bounds[:, 1] < bounds[:, 0]).sum()),
            known_one_lot_exceeds_trade_cap=int((lot_values * (1 + contract["broker"]["slippage"])
                > contract["trade"]["max_trade_weight"] * equity).sum()),
            minimum_trade_exceeds_trade_cap=int(minimum > contract["trade"]["max_trade_weight"] * equity),
            unavoidable_held_names=unavoidable,
            top_n_no_new_slot=int(unavoidable >= int(observation["top_n"][0]))),
        mask_frozen_only_hold=bool(not (legal[frozen | ~valid, 1:]).any()))
    if not result["mask_frozen_only_hold"]:
        raise AssertionError("Frozen/invalid asset has a non-HOLD legal action")
    return result


def _transferred_model(agent, data):
    config = deepcopy(data.config)
    config["device"] = "cpu"
    config["torch_threads"] = 1
    env = data.vector_env()
    try:
        transferred = make_model(env, config)
        transferred.policy.load_state_dict(agent.model.policy.state_dict(), strict=True)
        transferred.policy.set_training_mode(False)
        if parameter_digest(transferred.policy) != parameter_digest(agent.model.policy):
            raise AssertionError("Shared-parameter stock-pool transfer changed policy tensors")
        return PPOAgent(transferred, data.contract), env
    except BaseException:
        env.close()
        raise


def run_feasibility_probes(old_root, output_root):
    destination = Path(output_root) / "feasibility"
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with preserve_training_rng(torch.device("cpu")), torch.no_grad():
            source_agent, original, identity = next(_frozen_models(old_root))
            if identity["model_seed"] != 42 or identity["budget"] != "short":
                raise ValueError("Feasibility requires the fixed seed42 short model")
            dates = {index: original.market("train").dates[index] for index in PROBE_INDICES}
            for size in POOL_SIZES:
                if size > len(original.codes):
                    raise ValueError("Feasibility pool exceeds the original contract")
                for capital in CAPITALS:
                    config = deepcopy(original.config)
                    config.update(stock_codes=original.codes[:size], device="cpu", torch_threads=1)
                    data = ResearchData(config)
                    data.settings["initial_capital"] = capital
                    data.contract["initial_capital"] = capital
                    transferred, vector = _transferred_model(source_agent, data)
                    try:
                        market = data.market("train")
                        for negative, date in dates.items():
                            index = market.dates.get_loc(date)
                            for mode in PROBE_MODES:
                                env = _probe_env(data, market, index, mode)
                                try:
                                    obs, account = env.reset(seed=ACCOUNT_SEED)
                                    portfolio = env.unwrapped.portfolio
                                    cash_before = portfolio.cash
                                    positions_before = portfolio.positions.copy()
                                    equity_before = portfolio.value(market.prices.iloc[index].to_numpy())
                                    probe, arrays = sample_same_state(transferred, obs, samples=1, seed=101)
                                    result = feasibility_summary(obs, account, data.contract)
                                    np.testing.assert_array_equal(portfolio.positions, positions_before)
                                    if portfolio.cash != cash_before:
                                        raise AssertionError("Feasibility inference changed cash")
                                    equity_after = portfolio.value(market.prices.iloc[index].to_numpy())
                                    np.testing.assert_allclose(equity_before, equity_after, rtol=0, atol=0)
                                    name = f"n{size}_capital{int(capital)}_index{negative}_{mode}"
                                    path = destination / f"{name}.npz"
                                    for field in ("cash", "equity", "positions", "available_positions", "receivables"):
                                        arrays["account_" + field] = np.asarray(account[field]).copy()
                                    np.savez_compressed(path, **arrays)
                                    rows.append({**result, **probe, "n_stocks": size, "capital": capital,
                                        "stock_codes": data.codes, "stock_selection": "original_contract_first_N",
                                        "train_index": negative, "date": str(date.date()), "mode": mode,
                                        "initial_account_seed": ACCOUNT_SEED, "equity_unchanged": True,
                                        "parameters_sha256": parameter_digest(transferred.model.policy),
                                        "contract_sha256": json_digest(data.contract), "contract": data.contract,
                                        "path": str(path), "sha256": file_hash(path)})
                                finally:
                                    env.close()
                    finally:
                        vector.close()
            if parameter_digest(source_agent.model.policy) != identity["parameter_sha256"]:
                raise AssertionError("Feasibility study changed source policy parameters")
    finally:
        torch.set_num_threads(previous_threads)
    report = dict(schema_version=1, status="complete", stage="feasibility", device="cpu",
        source_model=identity, case_count=len(rows), pool_sizes=list(POOL_SIZES), capitals=list(CAPITALS),
        training_steps=0, training_updates=0, environment_steps=0,
        validation_evaluated=False, test_evaluated=False, cases=rows,
        interpretation="First-N pools overlap training stocks. Shape/lot feasibility is not unseen-stock or performance generalization.")
    write_json(destination / "summary.json", _json_safe(report))
    return report


def reward_math_identity(daily, gamma=.99):
    equity = np.asarray(daily["equity"], dtype=float)
    rewards = np.asarray(daily["daily_return"], dtype=float)
    if len(equity) < 2 or not np.isfinite(equity).all() or (equity <= 0).any():
        raise ValueError("Reward identity requires finite positive equity and at least one step")
    if not np.isfinite(rewards).all() or not 0 <= gamma <= 1:
        raise ValueError("Reward identity requires finite rewards and gamma in [0,1]")
    implied = equity[1:] / equity[:-1] - 1
    np.testing.assert_allclose(rewards[0], 0., rtol=0, atol=1e-12)
    np.testing.assert_allclose(rewards[1:], implied, rtol=1e-9, atol=1e-11)
    simple_sum = float(implied.sum())
    compound = float(np.prod(1 + implied) - 1)
    terminal = float(equity[-1] / equity[0] - 1)
    logs = np.log1p(implied)
    log_sum, terminal_log = float(logs.sum()), float(np.log(equity[-1] / equity[0]))
    discounted_log = float(np.dot(gamma ** np.arange(len(logs)), logs))
    np.testing.assert_allclose(compound, terminal, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(log_sum, terminal_log, rtol=1e-10, atol=1e-12)
    return dict(steps=len(implied), gamma=gamma, sum_simple_rewards=simple_sum,
        compound_return=compound, terminal_equity_return=terminal,
        simple_sum_minus_compound=simple_sum - compound,
        sum_log_rewards=log_sum, terminal_log_return=terminal_log,
        log_identity_error=log_sum - terminal_log, discounted_log_rewards=discounted_log,
        discounted_log_minus_terminal=discounted_log - terminal_log,
        interpretation="Undiscounted log rewards telescope; discounted logs generally do not. This is mathematics, not evidence of improved returns.")


def run_reward_math(old_root, output_root):
    destination = Path(output_root) / "reward"
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in MODEL_SEEDS:
        for budget in BUDGETS:
            for mode in MODES:
                path = Path(old_root) / "runs" / f"small_{seed}" / f"{budget}_validation" / mode / "ppo" / "daily.csv"
                row = reward_math_identity(pd.read_csv(path))
                rows.append({**row, "model_seed": seed, "budget": budget, "mode": mode,
                             "path": str(path), "daily_sha256": file_hash(path)})
    report = dict(schema_version=1, status="complete", stage="reward_math", cases=rows,
        case_count=len(rows), training_steps=0, training_updates=0, environment_steps=0,
        evaluated_existing_daily_only=True, new_validation_evaluated=False, test_evaluated=False)
    write_json(destination / "summary.json", _json_safe(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("deployment", "feasibility", "reward"))
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    function = {"deployment": run_deployment_probes, "feasibility": run_feasibility_probes,
                "reward": run_reward_math}[args.stage]
    report = function(args.old_root, args.output_root)
    print(json.dumps({"stage": report["stage"], "status": report["status"]}))


if __name__ == "__main__":
    main()
