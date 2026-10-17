"""Read-only cold verification of closed paired-teacher and PPO artifacts.

No environment, policy, optimizer, neural prediction, fitting or market replay is
created. A new sidecar is written only after all required generation leaves close.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import research_counterfactual_learning as runner
from tools import timing_action_value_learning as value_learning
from src.backtest.metrics import performance
from src.training.freeze import json_digest as frozen_json_digest

VERSION = "closed_causal_teacher_native_ppo_cold_v1"
OWN_FILES = ("tools/audit_counterfactual_learning.py", "tests/test_audit_counterfactual_learning.py")
ATOL = 1e-9
MONEY_ATOL = 2e-6
PARAMETER_NAMES = (
    "mlp_extractor.value_net.0.weight", "mlp_extractor.value_net.0.bias",
    "mlp_extractor.value_net.2.weight", "mlp_extractor.value_net.2.bias",
    "action_net.weight", "action_net.bias", "value_net.weight", "value_net.bias",
)
PARAMETER_SHAPES = ((64, 129), (64,), (64, 64), (64,), (2, 129), (2,), (1, 64), (1,))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(left, right, name, atol=ATOL):
    require(np.allclose(left, right, rtol=0, atol=atol), "Cold arithmetic mismatch: " + name)


class Reader:
    """Bind every directly read byte, including archives and closed markers."""

    def __init__(self):
        self.inputs = {}

    def bind(self, path):
        path = Path(path).resolve()
        sha = runner.file_hash(path)
        require(str(path) not in self.inputs or self.inputs[str(path)] == sha,
                "An input changed during cold verification")
        self.inputs[str(path)] = sha
        return path

    def json(self, path):
        return json.loads(self.bind(path).read_text(encoding="utf-8"))

    def arrays(self, path):
        with np.load(self.bind(path), allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}

    def label(self, path):
        return json.loads(gzip.decompress(self.bind(path).read_bytes()))

    def closed(self, directory, protocol):
        directory = Path(directory)
        require((directory / "completed.json").is_file(), "Required generation is not closed: " + str(directory))
        result = runner.closed(directory, protocol)
        require(result is not None, "Required generation is incomplete")
        self.bind(directory / "completed.json")
        for name in result["files"]:
            self.bind(directory / name)
        return result

    def stable(self):
        require(all(runner.file_hash(Path(path)) == sha for path, sha in self.inputs.items()),
                "Input bytes changed before cold seal")


def native_book(branch, key, codes, broker):
    """Reconstruct rewards and cash/share/fee arithmetic from saved native books."""
    n = len(codes)
    previous = float(branch["initial_agent_nav" if key == "agent_ledger" else "initial_benchmark_nav"])
    require(np.isfinite(previous) and previous > 0, "Invalid initial native NAV")
    logs, dates, fees, slippage = [], [], 0., 0.
    prior, initial_fees = None, None
    for step in branch["steps"]:
        book = step[key]
        decision, opening = book["decision_snapshot"], book["opening_snapshot"]
        require(decision["codes"] == opening["codes"] == list(codes), "Native coordinate mismatch")
        positions = np.asarray(decision["positions"], dtype=float)
        available = np.asarray(opening["available_positions"], dtype=float)
        opened = np.asarray(opening["positions"], dtype=float)
        require(positions.shape == available.shape == opened.shape == (n,)
                and np.isfinite(np.r_[positions, available, opened]).all()
                and (positions >= 0).all() and (available >= 0).all() and (available <= opened).all(), "Invalid native shares")
        close(decision["equity"], previous, "decision NAV", MONEY_ATOL)
        marks = np.asarray([decision["marks"][i] for i in np.flatnonzero(positions > 0)], dtype=float)
        require(np.isfinite(marks).all() and (marks > 0).all(), "Invalid held decision marks")
        close(decision["cash"] + decision["receivables"] + positions[positions > 0] @ marks,
              previous, "decision marked NAV", MONEY_ATOL)
        close(opened, positions + np.asarray(book["event_shares"]), "corporate share effects", MONEY_ATOL)
        close(opening["cash"], decision["cash"] + book["event_cash"], "corporate cash effects", MONEY_ATOL)
        if prior is not None:
            require(decision["date"] == prior["date"], "Native date continuity")
            for field in ("cash", "receivables", "positions"):
                close(decision[field], prior[field], "native continuity " + field, MONEY_ATOL)
        fills, sales, total, net = np.zeros(n), np.zeros(n), 0., 0.
        for order in book["orders"]:
            require(order["code"] in codes, "Unknown native order stock")
            i = codes.index(order["code"])
            quantity, fee, slip = (float(order[field]) for field in ("filled_shares", "fee", "slippage_cost"))
            require(np.isfinite([quantity, fee, slip]).all() and quantity == math.floor(quantity)
                    and fee >= 0 and slip >= -MONEY_ATOL, "Invalid native fill/cost")
            if quantity == 0:
                require(order["status"] == "rejected", "Unfilled native order status")
                close([fee, slip], [0, 0], "unfilled costs", MONEY_ATOL)
                continue
            price, raw = float(order["price"]), float(opening["prices"][i])
            require(np.isfinite([price, raw]).all() and min(price, raw) > 0
                    and order["side"] == ("sell" if quantity < 0 else "buy")
                    and order["status"] in ("filled", "partial"), "Invalid native side/price")
            amount = abs(quantity) * price
            expected = max(broker["min_commission"], amount * broker["commission"])
            if quantity < 0:
                expected += amount * broker["sell_tax"]
                sales[i] += quantity
            close(fee, expected, "native commission/tax", MONEY_ATOL)
            close(slip, quantity * (price - raw), "native slippage", MONEY_ATOL)
            fills[i] += quantity
            net += quantity * price
            total += fee
            slippage += slip
        close(book["positions"], opened + fills, "native filled holdings", MONEY_ATOL)
        close(book["available_positions"], available + sales, "native T1 holdings", MONEY_ATOL)
        closing = np.asarray(book["positions"])
        sellable = np.asarray(book["available_positions"])
        require((closing >= 0).all() and (sellable >= 0).all() and (sellable <= closing).all(), "Invalid native final availability")
        close(book["cash"], opening["cash"] - net - total, "native cash after fills", MONEY_ATOL)
        require(book["cash"] >= 0, "Native cash is negative")
        if initial_fees is None:
            initial_fees = book["total_fees"] - total
            require(initial_fees >= -MONEY_ATOL, "Invalid inferred starting fees")
        fees += total
        close(book["total_fees"], initial_fees + fees, "native cumulative fees", MONEY_ATOL)
        nav = float(book["equity"])
        require(np.isfinite(nav) and nav > 0 and str(book["date"])[:10] == str(opening["date"])[:10], "Invalid closing NAV/date")
        logs.append(math.log(nav) - math.log(previous))
        dates.append(str(book["date"])[:10])
        previous, prior = nav, book
    require(dates == sorted(set(dates)), "Native dates must strictly advance")
    close(previous, branch["final_agent_nav" if key == "agent_ledger" else "final_benchmark_nav"], "final native NAV", MONEY_ATOL)
    return np.asarray(logs), dates, dict(fees=fees, slippage=slippage)


def verify_pair(label, record, contract, cutoff):
    target, proof, counts = label["labels"], label["provenance"], label["counters"]
    require(label["label_version"] == runner.paired.LABEL_VERSION and target["horizon"] == 20
            and target["gamma"] == .995 and target["train_end"] == cutoff, "Unknown paired estimator")
    require(label["input"]["phase"] == record["phase"] and label["input"]["decision_date"] == record["decision_date"]
            and label["input"]["feature_names"] == list(runner.paired.FEATURE_NAMES)
            and runner.parent.input_hash(np.asarray(label["input"]["observation"], dtype=np.float32)) == record["input_sha256"], "Pair input binding failed")
    require(proof["source_before_sha256"] == proof["source_after_sha256"] == record["full_state_sha256"]
            and proof["contract_sha256"] == runner.paired._digest(contract)
            and proof["price_basis"] == contract["price_basis"]
            and all(proof[k] is True for k in ("source_unchanged", "global_rng_unchanged", "benchmark_paths_equal",
                "clone_initial_state_equal", "clone_storage_independent", "current_close_observation_verified")), "Pair source proof failed")
    require(counts["branch_agent_steps"] == counts["branch_benchmark_steps"] == counts["macro_attempts"] == 40
            and all(counts[k] == 0 for k in ("collection_steps", "rl_steps", "nn_forwards", "optimizer_updates")), "Pair actual budget failed")
    relatives, shadows, costs = {}, [], {}
    for first, name in enumerate(("hold", "rebalance")):
        branch = target["branches"][name]
        require(branch["actual_agent_steps"] == branch["actual_benchmark_steps"] == 20
                and len(branch["steps"]) == 20 and branch["endpoint"] == record["endpoint"], "Incomplete paired horizon")
        for j, step in enumerate(branch["steps"]):
            phase = (record["phase"] + j) % 20
            require(step["index"] == j and step["phase"] == phase and step["macro_action"] == (first if j == 0 else int(phase == 0))
                    and (j == 19 or not step["terminated"] and not step["truncated"]), "Paired calendar/boundary mismatch")
        agent, dates, costs[name] = native_book(branch, "agent_ledger", contract["stock_codes"], contract["broker"])
        shadow, shadow_dates, _ = native_book(branch, "benchmark_ledger", contract["stock_codes"], contract["broker"])
        require(dates == shadow_dates and dates[-1] == record["endpoint"] <= cutoff and dates[0] > record["decision_date"], "Paired endpoint/cutoff failed")
        require(str(branch["steps"][0]["agent_ledger"]["decision_snapshot"]["date"])[:10] == record["decision_date"], "Initial paired date failed")
        relatives[name] = agent - shadow
        close(relatives[name], [s["relative_log_reward"] for s in branch["steps"]], "paired saved relative rewards")
        close(relatives[name], [s["agent_ledger"]["learning_reward"] for s in branch["steps"]], "paired native learning rewards")
        shadows.append([s["benchmark_ledger"] for s in branch["steps"]])
    require(runner.canonical(shadows[0]) == runner.canonical(shadows[1]), "Paired full shadow ledgers differ")
    branches = target["branches"]
    close(branches["hold"]["initial_agent_nav"], branches["rebalance"]["initial_agent_nav"], "paired initial NAV")
    difference = relatives["rebalance"] - relatives["hold"]
    delta = float(np.dot(.995 ** np.arange(20), difference))
    terminal = math.log(branches["rebalance"]["final_agent_nav"]) - math.log(branches["hold"]["final_agent_nav"])
    for saved, actual in ((target["discounted_relative_delta"], delta), (target["terminal_log_nav_ratio"], terminal),
                          (target["undiscounted_relative_delta"], difference.sum()), (difference.sum(), terminal)):
        close(saved, actual, "paired discount/terminal identity")
    close(target["per_step_relative_reward_difference"], difference, "paired per-step difference")
    # No Agent==shadow requirement: the source behavior deliberately diverges.
    return delta, costs


def verify_teacher(directory, protocol, reader):
    sealed = reader.closed(directory, protocol)
    plan = reader.json(directory / "sampling_plan.json")
    runner.validate_plan(plan)
    require(plan["protocol_sha256"] == runner.digest(protocol) and sealed["counts"] == runner.expected_teacher_counts()
            and reader.json(directory / "counters.json") == sealed["counts"]
            and sealed["data_identity"] == plan["data_identity"] and sealed["train_end"] == plan["train_end"], "Teacher protocol/counters/data mismatch")
    seal = reader.json(directory / "sampling_plan_seal.json")
    require(seal["states"] == 480 and seal["labels_started"] is False and set(seal["files"]) == {"sampling_plan.json", "inputs.npz"}, "Teacher pre-label seal failed")
    for name, sha in seal["files"].items():
        require(runner.file_hash(reader.bind(directory / name)) == sha, "Teacher plan bytes changed")
    X, y = reader.arrays(directory / "inputs.npz"), reader.arrays(directory / "targets.npz")
    require(set(X) == {"observations"} and set(y) == {"delta"}, "Future/evidence fields entered model arrays")
    X, y = X["observations"], y["delta"]
    require(X.shape == (480, 129) and X.dtype == np.float32 and np.isfinite(X).all()
            and y.shape == (480,) and y.dtype == np.float64 and np.isfinite(y).all(), "Teacher array schema failed")
    expected_labels = {row["state_id"] + ".json.gz" for row in plan["states"]}
    require({p.name for p in (directory / "labels").iterdir()} == expected_labels, "Teacher labels are missing/extra")
    maximum = 0.
    for i, record in enumerate(plan["states"]):
        require(np.array_equal(X[i], np.asarray(record["observation"], dtype=np.float32)), "Teacher X differs from frozen current input")
        delta, _ = verify_pair(reader.label(directory / "labels" / (record["state_id"] + ".json.gz")), record,
                               plan["teacher_contract"], plan["train_end"])
        maximum = max(maximum, abs(delta - y[i]))
        close(y[i], delta, "Teacher target array")
    return dict(fold=plan["fold"], segment=plan["segment"], counts=sealed["counts"], rows=480,
                maximum_target_error=maximum, boundaries=plan["data_identity"]["boundaries"]), X, y


def verify_ridge(directory, protocol, train, hold, reader):
    reader.closed(directory, protocol)
    fit, diagnostics = reader.json(directory / "fit.json"), reader.json(directory / "holdout.json")
    arrays = value_learning._validate_fit(fit)
    X, y = train
    require(fit["n_train"] == len(y) == 480 and fit["training_input_sha256"] == value_learning._array_digest(X)
            and fit["training_target_sha256"] == value_learning._array_digest(y)
            and diagnostics["fit_sha256"] == fit["fit_sha256"] and diagnostics["count"] == len(hold[1]) == 480,
            "Ridge training-only data/diagnostic binding failed")
    X = X.astype(np.float64)
    mean, constant = X.mean(axis=0), np.all(X == X[:1], axis=0)
    mean[constant] = X[0, constant]
    scale = np.where(constant, 1., np.sqrt(np.mean((X - mean) ** 2, axis=0)))
    close(arrays["x_mean"], mean, "Ridge train mean")
    close(arrays["x_scale"], scale, "Ridge train scale")
    close(fit["target_rms"], max(float(np.sqrt(np.mean(y * y))), 1e-8), "Ridge train RMS")
    z, normalized = (X - mean) / scale, y / fit["target_rms"]
    close(fit["intercept_normalized"], normalized.mean(), "Ridge unpenalized intercept")
    beta = arrays["coef_normalized"]
    residual = (z.T @ z / len(z) + np.eye(129)) @ beta - z.T @ (normalized - normalized.mean()) / len(z)
    close(residual, np.zeros(129), "Ridge normal equations", 1e-8)
    return fit, dict(rows=480, holdout_rows=480, maximum_normal_equation_residual=float(abs(residual).max()),
                     train_only_statistics_verified=True, fit_hashes_match_only_registered_train_arrays=True,
                     audit_numerical_matrix_checks_only=True, new_model_fits=0, new_predictions=0)


def verify_data(directory, fold, protocol, reader):
    from tools.timing_causal_data import frame_value_sha256, json_digest
    identity = reader.json(directory / "identity.json")
    boundaries = reader.json(directory / "boundaries.json")
    scaler = reader.json(directory / "scaler_params.json")
    features = pd.read_parquet(reader.bind(directory / "panel_data.parquet"))
    prices = pd.read_parquet(reader.bind(directory / "prices.parquet"))
    require(identity["fold"] == fold and identity["boundaries"] == boundaries
            and json_digest(scaler) == identity["scaler_value_sha256"]
            and scaler["train_end"] == boundaries["inner_train_end"]
            and frame_value_sha256(features) == identity["derived_feature_value_sha256"]
            and frame_value_sha256(prices) == identity["derived_prices_value_sha256"]
            and features.index.equals(prices.index)
            and str(features.index.max().date()) == boundaries["outer_validation_end"] < "2026-07-01",
            "Derived data values/scaler/cutoff changed")
    if fold in protocol["data"]:
        require(identity == protocol["data"][fold], "Development data differs from prepared identity")
    dates = features.index[(features.index >= pd.Timestamp(boundaries["outer_validation_start"]))
                           & (features.index <= pd.Timestamp(boundaries["outer_validation_end"]))]
    return identity, [str(date.date()) for date in dates]


def verify_gae(buffers, side):
    required = {"observations", "actions", "rewards", "values", "log_probs", "returns", "advantages", "episode_starts"}
    require(set(buffers) == required, "PPO must preserve all eight original buffer fields")
    n = len(buffers["rewards"])
    require(float(side["gamma"]) == .995 and float(side["gae_lambda"]) == 1., "Wrong registered GAE")
    require(np.array_equal(side["last_dones"], side["dones"][-1])
            and np.array_equal(buffers["episode_starts"][1:], side["dones"][:-1])
            and not np.any(side["truncated"] & ~side["dones"]), "GAE native terminal flags changed")
    corrected = side["raw_rewards"].copy()
    corrected[side["truncated"]] += np.float32(.995) * side["terminal_values"][side["truncated"]]
    close(corrected, buffers["rewards"], "Native timeout reward bootstrap", 1e-6)
    advantages = np.zeros_like(buffers["advantages"])
    tail = np.zeros_like(side["last_values"])
    for i in reversed(range(n)):
        active = 1. - (side["last_dones"] if i == n - 1 else buffers["episode_starts"][i + 1])
        value = side["last_values"] if i == n - 1 else buffers["values"][i + 1]
        delta = buffers["rewards"][i] + .995 * value * active - buffers["values"][i]
        tail = delta + .995 * active * tail
        advantages[i] = tail
    close(advantages, buffers["advantages"], "Original GAE advantages", 2e-6)
    close(advantages + buffers["values"], buffers["returns"], "Original GAE returns", 2e-6)
    return dict(rows=n, timeouts=int(side["truncated"].sum()),
                maximum_advantage_error=float(abs(advantages - buffers["advantages"]).max()))


def archive(directory, reader):
    """Load inert tensor dictionaries only; never PPO.load or a policy object."""
    metadata = reader.json(directory / "metadata.json")
    path = reader.bind(directory / "model.zip")
    require(runner.file_hash(path) == metadata["model_sha256"], "Model metadata hash failed")
    with zipfile.ZipFile(path) as zipped:
        require(zipped.testzip() is None and len(zipped.namelist()) == len(set(zipped.namelist())), "Model ZIP CRC/duplicate entries failed")
        data = json.loads(zipped.read("data"))
        policy_bytes = zipped.read("policy.pth")
        policy = torch.load(io.BytesIO(policy_bytes), map_location="cpu", weights_only=True)
        optimizer = torch.load(io.BytesIO(zipped.read("policy.optimizer.pth")), map_location="cpu", weights_only=True)
    require(tuple(policy) == PARAMETER_NAMES and all(torch.is_tensor(t) and torch.isfinite(t).all()
            and t.dtype == torch.float32 and tuple(t.shape) == shape for t, shape in zip(policy.values(), PARAMETER_SHAPES))
            and sum(t.numel() for t in policy.values()) == 12805, "Wrong finite policy tensor inventory")
    require(type(data["num_timesteps"]) is int and data["num_timesteps"] >= 0
            and data["num_timesteps"] == metadata["num_timesteps"]
            and data["timing_registered_contract"] == metadata["contract"]
            and data["timing_policy_record"] == metadata["policy_record"]
            and data["timing_registered_algorithm"] == metadata["algorithm_class"]
            == "stable_baselines3.ppo.ppo.PPO", "Archive clock/contract/algorithm failed")
    require(data["timing_feature_names"] == list(runner.paired.FEATURE_NAMES)
            and metadata["policy_record"]["parameter_count"] == 12805, "Archive feature/parameter schema failed")
    ids = [i for group in optimizer["param_groups"] for i in group["params"]]
    require(len(optimizer["param_groups"]) == 1 and ids == list(range(8))
            and set(optimizer["state"]).issubset(set(ids)), "Adam parameter inventory failed")
    counts = {}
    for name, identifier in zip(policy, ids):
        state = optimizer["state"].get(identifier)
        if state is not None:
            require(set(state) == {"step", "exp_avg", "exp_avg_sq"}
                    and state["exp_avg"].shape == state["exp_avg_sq"].shape == policy[name].shape
                    and state["exp_avg"].dtype == state["exp_avg_sq"].dtype == policy[name].dtype
                    and (state["exp_avg_sq"] >= 0).all()
                    and state["step"].shape == () and state["step"].dtype != torch.bool
                    and all(torch.is_tensor(t) and torch.isfinite(t).all() for t in state.values()), "Adam native tensor state failed")
            step = float(state["step"].item())
            require(step.is_integer() and step > 0, "Invalid actual Adam step")
            counts[name] = int(step)
    return dict(metadata=metadata, data=data, policy=policy, optimizer_counts=counts, optimizer_group=optimizer["param_groups"][0],
                policy_bytes_sha256=hashlib.sha256(policy_bytes).hexdigest())


def verify_warmup(theta0, thetaaux, method, fit):
    a, b = theta0["policy"], thetaaux["policy"]
    require(theta0["metadata"]["num_timesteps"] == thetaaux["metadata"]["num_timesteps"] == 0
            and theta0["data"]["_n_updates"] == thetaaux["data"]["_n_updates"] == 0
            and not theta0["optimizer_counts"] and not thetaaux["optimizer_counts"], "Warmup reused PPO/Adam experience")
    require(all(torch.equal(t, b[name]) for name, t in a.items() if not name.startswith("action_net.")), "Warmup changed the Critic")
    expected = torch.zeros_like(a["action_net.weight"])
    expected[0, 1:20], expected[1, 0] = 2., 2.
    require(torch.equal(a["action_net.weight"], expected) and torch.equal(a["action_net.bias"], torch.zeros(2)), "theta0 is not the original calendar")
    if method == "ppo_only":
        require(all(torch.equal(t, b[name]) for name, t in a.items()), "PPO-only changed thetaaux")
    else:
        coefficient = 8 * np.asarray(fit["coef_observation"])
        coefficient[0] += 4.
        intercept = 8 * (fit["intercept_observation"] - .25)
        expected_w = torch.as_tensor(np.stack([-.5 * coefficient, .5 * coefficient]), dtype=b["action_net.weight"].dtype)
        expected_b = torch.as_tensor([-.5 * intercept, .5 * intercept], dtype=b["action_net.bias"].dtype)
        require(torch.equal(expected_w, b["action_net.weight"]) and torch.equal(expected_b, b["action_net.bias"]), "Ridge-to-Actor mapping failed")
    return dict(critic_unchanged=True, optimizer_empty=True, actual_rl_steps=0)


def case_arrays(directory, result, reader, benchmark=False):
    trace = result["trace"]["benchmark_trace"] if benchmark else result["trace"]
    base = directory / "trace" / "benchmark" if benchmark else directory / "trace"
    parts = [reader.arrays(base / item["path"]) for item in trace["chunks"]]
    require(parts, "Missing native evaluation trace")
    common = ("after_positions", "after_cash", "after_equity", "decision_equity", "decision_cash",
              "execution_fee", "execution_slippage_cost", "execution_date")
    return {key: np.concatenate([part[key] for part in parts]) for key in
            (common if benchmark else ("action_macro_action", "action_phase", *common))}


def changes(left, right=None):
    require(right is not None, "Economic reference trace is required")
    macro = (left["action_macro_action"] != right["action_macro_action"]) if "action_macro_action" in right else (
        left["action_macro_action"] != (left["action_phase"] == 0))
    return dict(greedy_changed=bool(np.any(macro)), economic_ledger_changed=any(
        not np.allclose(left[k], right[k], rtol=0, atol=tol) for k, tol in
        (("after_positions", 1e-8), ("after_cash", 1e-6), ("after_equity", 1e-6))))


def verify_case(directory, row_metrics, reference_metrics, contract, reader, expected_dates=None):
    result = reader.json(directory / "result.json")
    require(result["contract"] == contract and result["test_evaluated"] is False and result["actionable"] is False,
            "Evaluation contract/status changed")
    require(result["last_execution"] < "2026-07-01", "Protected evaluation detected")
    arrays, shadows = case_arrays(directory, result, reader), case_arrays(directory, result, reader, True)
    for filename, saved, native in (("daily.csv", row_metrics, arrays), ("benchmark_daily.csv", reference_metrics, shadows)):
        daily = pd.read_csv(reader.bind(directory / filename))
        actual = performance(daily)
        require(set(actual) == set(saved), "Metric schema changed")
        for key, value in actual.items():
            require(saved[key] is None if value is None else saved[key] is not None, "Optional metric changed")
            if value is not None:
                close(saved[key], value, "Saved daily metric " + key, 2e-8)
        require(str(daily.date.iloc[-1])[:10] == result["last_execution"]
                and len(daily) - 1 == result["evaluation_steps"], "Evaluation saved calendar length changed")
        close(daily.equity.to_numpy()[1:], native["after_equity"], "Daily and native closing NAV", MONEY_ATOL)
        close(daily.cash.to_numpy()[1:], native["after_cash"], "Daily and native closing cash", MONEY_ATOL)
        close(daily.equity.to_numpy()[:-1], native["decision_equity"], "Daily and native decision NAV", MONEY_ATOL)
        close(daily.cash.to_numpy()[:-1], native["decision_cash"], "Daily and native decision cash", MONEY_ATOL)
        close(daily.fee.to_numpy()[1:], native["execution_fee"].sum(axis=1), "Daily native fees", MONEY_ATOL)
        close(daily.slippage_cost.to_numpy()[1:], native["execution_slippage_cost"].sum(axis=1), "Daily native slippage", MONEY_ATOL)
        require([str(value)[:10] for value in daily.date.iloc[1:]] == [str(value)[:10] for value in native["execution_date"]],
                "Daily native execution dates changed")
        if expected_dates is not None:
            require([str(value)[:10] for value in daily.date] == expected_dates, "Evaluation does not cover the complete saved Q2 calendar")
    require(result["metrics"] == row_metrics and result["benchmark_metrics"] == reference_metrics, "Result/record metrics differ")
    runner.original.verify_timing_trace(directory / "trace", result["trace"], steps=result["evaluation_steps"],
                                       expected_contract_sha256=frozen_json_digest(contract))
    return result, arrays, shadows


def verify_trial(directory, protocol, fold, method, seed, phase, budget, fit, reader, identity=None, expected_dates=None):
    sealed = reader.closed(directory, protocol)
    config = reader.json(directory / "config.json")
    require(config == runner.config_for(protocol, directory.parent.parent, fold, method, seed, phase), "Trial config differs from registered condition")
    require(sealed["fold"] == fold and sealed["method"] == method and sealed["seed"] == seed and sealed["phase"] == phase
            and sealed["physical_rl_steps"] == sealed["physical_benchmark_steps"] == budget
            and sealed["config_sha256"] == runner.digest(config), "Physical trial identity mismatch")
    expected = set(runner.SNAPSHOTS) if phase == "screen" else {budget}
    require(len(sealed["records"]) == len(expected) and {r["steps"] for r in sealed["records"]} == expected, "Trial snapshot inventory mismatch")
    theta0, thetaaux = archive(directory / "theta0", reader), archive(directory / "thetaaux", reader)
    contract = theta0["metadata"]["contract"]
    require(thetaaux["metadata"]["contract"] == contract and contract["causal_data"] == sealed["data_identity"], "Trial causal contract changed")
    require(len(contract["stock_codes"]) == 300 and len(set(contract["stock_codes"])) == 300
            and len(contract["feature_cols"]) == 20 and contract["price_basis"] == "hfq_research", "Registered market coordinates changed")
    if identity is not None:
        require(sealed["data_identity"] == identity, "Trial scaler is not the fold's saved fixed inner scaler")
    def registered_model(saved, steps):
        metadata, data = saved["metadata"], saved["data"]
        sources = {name.replace("\\", "/"): sha for name, sha in metadata["source_hashes"].items()}
        require(metadata["config"] == config and sources == protocol["source_hashes"]
                and metadata["num_timesteps"] == steps and metadata["actual_device"] == "cpu"
                and data["gamma"] == .995 and data["gae_lambda"] == 1.
                and all(data[name] == config["ppo"][name] for name in (
                    "n_steps", "batch_size", "n_epochs", "learning_rate", "target_kl", "ent_coef", "vf_coef", "max_grad_norm")),
                "Saved model source/config/estimator changed")
        group = saved["optimizer_group"]
        require(group["lr"] == config["ppo"]["learning_rate"] and tuple(group["betas"]) == (.9, .999)
                and group["eps"] == 1e-5 and group["weight_decay"] == 0 and group["amsgrad"] is False,
                "Saved native Adam hyperparameters changed")
    registered_model(theta0, 0)
    registered_model(thetaaux, 0)
    verify_warmup(theta0, thetaaux, method, fit)
    baseline = runner.reference_metrics(protocol, fold)
    for mode in runner.MODES:
        reader.bind(Path(protocol["baseline_reference_root"]) / "rules" / fold / mode / "result.json")
    initialized, auxiliary = {}, {}
    evaluation_steps = 0
    for role, cache in (("initialized", initialized), ("auxiliary", auxiliary)):
        for mode in runner.MODES:
            saved = reader.json(directory / role / mode / "result.json")
            result, _, _ = verify_case(directory / role / mode, saved["metrics"], baseline[mode], contract, reader, expected_dates)
            cache[mode] = result["metrics"]
            evaluation_steps += result["evaluation_steps"]
    require(initialized == baseline and (method != "ppo_only" or auxiliary == initialized), "Calendar/PPO-only baseline changed")
    rows, snapshots, final_counts = [], [], None
    for saved in sorted(sealed["records"], key=lambda r: r["steps"]):
        row = runner.validate_record(saved, phase)
        steps = row["steps"]
        path = directory / "snapshots" / str(steps)
        audit = reader.json(path / "training_audit.json")
        model = archive(path / "model", reader)
        registered_model(model, steps)
        require(row["audit"] == audit and model["metadata"]["num_timesteps"] == audit["actual_timesteps"] == steps
                and model["metadata"]["contract"] == contract and model["metadata"]["config"] == config
                and audit["model_sha256"] == model["metadata"]["model_sha256"], "Snapshot record/archive/audit changed")
        counts = model["optimizer_counts"]
        require(set(counts) == set(PARAMETER_NAMES) and len(set(counts.values())) == 1 and min(counts.values()) > 0
                and counts == audit["optimizer_steps_by_parameter"]
                and max(counts.values()) == audit["actual_optimizer_updates"], "Full actual native Adam counts changed")
        deltas = {name: float(torch.linalg.vector_norm(value - thetaaux["policy"][name])) for name, value in model["policy"].items()}
        for name, delta in deltas.items():
            close(delta, audit["tensor_delta_l2"][name], "Snapshot actual tensor delta " + name, 1e-6)
        require(any(v > 0 for k, v in deltas.items() if k.startswith("action_net."))
                and any(v > 0 for k, v in deltas.items() if not k.startswith("action_net.")), "No Actor/Critic PPO parameter learning")
        require(audit["experience"]["total_steps"] == audit["experience"]["benchmark_steps"] == steps,
                "Snapshot actual native experience clock changed")
        runner.original.verify_timing_trace(directory / "training_trace", audit["training_trace_prefix"], steps=steps,
                                           expected_contract_sha256=frozen_json_digest(contract))
        runner.original.verify_rollouts(directory, audit["rollout_buffers"], steps, audit["training_trace_prefix"])
        runner.credit.verify_estimator_buffers(directory, audit, config)
        gae = [verify_gae(reader.arrays(directory / rollout["path"]), reader.arrays(directory / side["path"]))
               for rollout, side in zip(audit["rollout_buffers"], audit["credit_estimator_buffers"])]
        require(row["initialized"] == initialized and row["auxiliary"] == auxiliary and row["benchmark"] == baseline,
                "Snapshot comparisons are not its actual controls")
        for mode in runner.MODES:
            result, actual, shadow = verify_case(path / "validation" / mode, row["trained"][mode], baseline[mode], contract, reader, expected_dates)
            macro = bool(np.any(actual["action_macro_action"] != (actual["action_phase"] == 0)))
            evidence = changes(actual, shadow)
            evidence["greedy_changed"] = macro
            _, aux, _ = verify_case(directory / "auxiliary" / mode, auxiliary[mode], baseline[mode], contract, reader, expected_dates)
            require(evidence == row["changes"][mode] and changes(actual, aux) == row["changes_vs_aux"][mode], "Native behavior flags changed")
            evaluation_steps += result["evaluation_steps"]
        timeouts, rollouts = sum(g["timeouts"] for g in gae), len(gae)
        require(rollouts <= model["data"]["_n_updates"] <= rollouts * config["ppo"]["n_epochs"]
                and model["data"]["_total_timesteps"] == budget, "Native PPO epoch/learn clocks changed")
        snapshots.append(dict(steps=steps, adam_steps=max(counts.values()), rollouts=rollouts, timeouts=timeouts,
            maximum_gae_error=max(g["maximum_advantage_error"] for g in gae),
            nn_call_scope=dict(collection_policy_forwards=steps, tail_value_calls=rollouts, timeout_value_calls=timeouts,
                ppo_evaluate_actions_lower=max(counts.values()),
                ppo_evaluate_actions_upper=max(counts.values()) + rollouts,
                inference="SB3 static call sites plus saved steps/rollouts/timeouts/Adam; not runtime call instrumentation",
                exact_total_neural_forwards_not_recorded=True, save_load_and_evaluation_forwards_not_counted_here=True)))
        final_counts = counts
        rows.append(saved)
    return rows, dict(fold=fold, method=method, seed=seed, physical_rl_steps=budget, physical_shadow_steps=budget,
                      physical_adam_steps=max(final_counts.values()), snapshots=snapshots,
                      evaluation_agent_steps=evaluation_steps, evaluation_shadow_steps=evaluation_steps,
                      theta0_policy_bytes_sha256=theta0["policy_bytes_sha256"])


def audit(root=runner.OUTPUT, phase="screen"):
    root = runner.output_root(root)
    require(phase in ("screen", "confirm"), "Only complete screen or admitted confirmation can be audited")
    destination = root / "diagnostics" / "cold_audit" / phase
    require(destination.resolve() == destination and destination.is_relative_to(root)
            and not destination.exists(), "Never overwrite or redirect a partial or completed cold audit")
    reader = Reader()
    protocol = reader.json(root / "protocol.json")
    runner.check_frozen(protocol, root)
    folds, seeds, budget = runner.SCREEN, (42,), 32768
    selection = None
    if phase == "confirm":
        selection = reader.json(root / "selection.json")
        require(selection["admitted"] is True and selection == runner.select_screen(runner.load_rows(root, protocol, "screen")), "Confirmation selection is not frozen/admitted")
        proof = reader.json(root / "selection_proof.json")
        for path in sorted((root / "runs").glob("screen_*")):
            reader.closed(path, protocol)
        require(proof["protocol_sha256"] == runner.digest(protocol) and proof["selection_sha256"] == runner.file_hash(root / "selection.json")
                and proof["screen_completed_sha256"] == {p.name: runner.file_hash(p / "completed.json") for p in sorted((root / "runs").glob("screen_*"))}, "Confirmation selection anchors changed")
        folds, seeds, budget = runner.CONFIRM, runner.SEEDS, selection["selected"]["steps"]
    expected_runs = {f"{phase}_{fold}_{method}_{seed}" for fold in folds for method in runner.METHODS for seed in seeds}
    require({p.name for p in (root / "runs").glob(phase + "_*")} == expected_runs
            and all((root / "runs" / name / "completed.json").exists() for name in expected_runs), "All exact physical runs must close before cold audit")
    teachers, fits, summaries, rows = [], [], [], []
    for fold in folds:
        identity, expected_dates = verify_data(root / "data" / fold, fold, protocol, reader)
        segments = {}
        for segment in runner.SEGMENTS:
            detail, X, y = verify_teacher(root / "teacher" / fold / segment, protocol, reader)
            require(detail["fold"] == fold and detail["segment"] == segment, "Teacher path identity changed")
            teachers.append(detail)
            require(detail["boundaries"] == identity["boundaries"], "Teacher/outer scaler boundaries differ")
            segments[segment] = (X, y)
        boundary = teachers[-1]["boundaries"]
        require(boundary["inner_train_end"] < boundary["embargo_dates"][0] <= boundary["embargo_dates"][-1] < boundary["inner_holdout_start"]
                and len(boundary["embargo_dates"]) == 20, "Inner embargo boundaries changed")
        fit, detail = verify_ridge(root / "fits" / fold, protocol, segments["inner_train"], segments["inner_holdout"], reader)
        fits.append(dict(fold=fold, **detail))
        for seed in seeds:
            paired = []
            for method in runner.METHODS:
                new_rows, detail = verify_trial(root / "runs" / f"{phase}_{fold}_{method}_{seed}", protocol, fold,
                                                method, seed, phase, budget, fit, reader, identity, expected_dates)
                rows.extend(new_rows)
                summaries.append(detail)
                paired.append(detail["theta0_policy_bytes_sha256"])
            require(len(set(paired)) == 1, "Methods did not share exact original theta0 bytes")
    outcome = runner.select_screen(rows) if phase == "screen" else runner.confirmation_summary(rows, selection, protocol["gate"])
    if phase == "screen" and (root / "selection.json").exists():
        require(reader.json(root / "selection.json") == outcome, "Saved development selection differs from cold metrics")
        if (root / "selection_proof.json").exists():
            proof = reader.json(root / "selection_proof.json")
            require(proof["protocol_sha256"] == runner.digest(protocol)
                    and proof["selection_sha256"] == runner.file_hash(root / "selection.json")
                    and proof["screen_completed_sha256"] == {p.name: runner.file_hash(p / "completed.json")
                        for p in sorted((root / "runs").glob("screen_*"))}, "Cold screen selection anchors changed")
    counters = dict(teacher_book_steps=sum(sum(t["counts"][k] for k in ("collection_agent_steps", "collection_benchmark_steps", "branch_agent_steps", "branch_benchmark_steps")) for t in teachers),
        paired_teacher_labels=sum(t["rows"] for t in teachers), generation_ridge_fits=len(fits),
        ridge_training_label_rows=480 * len(fits), ridge_holdout_diagnostic_rows=480 * len(fits),
        physical_rl_steps=sum(s["physical_rl_steps"] for s in summaries), physical_shadow_steps=sum(s["physical_shadow_steps"] for s in summaries),
        physical_adam_steps=sum(s["physical_adam_steps"] for s in summaries),
        evaluation_agent_steps=sum(s["evaluation_agent_steps"] for s in summaries), evaluation_shadow_steps=sum(s["evaluation_shadow_steps"] for s in summaries),
        audit_environment_steps=0, audit_neural_forwards=0, audit_optimizer_updates=0, audit_model_fits=0)
    result = dict(schema_version=1, version=VERSION, phase=phase, status="verified_saved_evidence", counters=counters,
        teachers=teachers, fits=fits, trials=summaries, records=rows, outcome=outcome, actionable=False,
        scope=dict(upstream_full_integrity_delegated_to_frozen_runner=True,
            source_state_replay_not_repeated=True, closing_nav_uses_saved_native_books_not_fresh_quotes=True,
            collection_step_counts_from_closed_counters_and_registered_two_pass_design=True,
            no_policy_or_optimizer_constructed=True, torch_load_weights_only=True,
            generation_neural_total_not_claimed_exact=True, input_hashes_cover_direct_reads_and_closed_generation_leaves=True))
    for name in OWN_FILES:
        reader.bind(ROOT / name)
    for name in protocol["inventory_sha256"]:
        reader.bind(ROOT / name)
    reader.stable()
    require(destination.resolve() == destination, "Cold audit destination became aliased")
    destination.mkdir(parents=True, exist_ok=False)
    runner.write_json(destination / "audit.json", result)
    reader.stable()
    runner.write_json(destination / "manifest.json", dict(status="passed_cold_artifact_audit", version=VERSION,
        phase=phase, protocol_sha256=runner.digest(protocol), inputs_sha256=reader.inputs,
        source_test_sha256={name: runner.file_hash(ROOT / name) for name in OWN_FILES},
        outputs_sha256={"audit.json": runner.file_hash(destination / "audit.json")}))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(runner.OUTPUT))
    parser.add_argument("--phase", choices=("screen", "confirm"), default="screen")
    args = parser.parse_args(argv)
    result = audit(args.root, args.phase)
    print(json.dumps(dict(status=result["status"], phase=args.phase, counters=result["counters"]), sort_keys=True))


if __name__ == "__main__":
    main()
