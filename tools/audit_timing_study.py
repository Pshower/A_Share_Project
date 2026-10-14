"""Independent read-only audit of sealed timing artifacts; no policy execution."""

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.trade_mapping import map_trade_action
from src.data.preprocess import FEATURES
from src.training.freeze import json_digest
from src.training.research_timing import verify_leaf_files, verify_rollouts, verify_evaluation_dates
from src.training.timing_env import FEATURE_NAMES, StratifiedTimingSampler
from src.training.timing_trace import verify_timing_trace


PROTOCOL_SHA = "f65101d627c86d420658ac23efea3a27e954b00ea93ff543ca165c92f5c2c995"
DOCUMENT_SHA = "d984c5c130ca67fa8b0b292c1685fc077ac625bc72dcd03795787881fbb0ff88"
REFERENCE_SHA = "89bc5f465eada6a845812b9f538886fc510dfcaecd392e54100b3f1f93379ce2"
MODES = ("cash", "equal", "concentrated", "random")
MARKETS = {}


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def close(actual, expected, message, atol=1e-6, rtol=1e-10):
    require(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True), message)


def registration(study, protocol):
    require(sha(study / "protocol.json") == PROTOCOL_SHA, "Formal protocol bytes changed")
    require(sha(ROOT / "doc/ppo_timing_protocol_20261014.md") == DOCUMENT_SHA == protocol["document_sha256"], "Preregistered document changed")
    actual = {str(path.relative_to(ROOT)): sha(path) for path in (ROOT / "src").rglob("*.py")}
    require(actual == protocol["source_hashes"] and len(actual) == 96, "Frozen source inventory changed")
    reference = Path(protocol["reference_root"])
    require(sha(reference / "final_audit.json") == REFERENCE_SHA, "Original final audit changed")
    previous = read(reference / "protocol.json")
    require(sha(reference / "protocol.json") == protocol["reference_protocol_sha256"], "Original protocol changed")
    require(len(previous["source_hashes"]) == 91 and all(actual.get(name) == digest for name, digest in previous["source_hashes"].items()), "An original 91 source changed")
    require(protocol["parameters"] == 12805 and len(protocol["feature_names"]) == 129
            and protocol["feature_names"] == list(FEATURE_NAMES) and protocol["device"] == "cpu", "Formal architecture differs")
    datasets = {}
    for fold, item in protocol["folds"].items():
        directory = Path(item["directory"])
        require(sha(directory / "manifest.json") == item["manifest_sha256"], "Dataset manifest changed")
        require(all(sha(directory / name) == digest for name, digest in item["files"].items()), "Dataset artifact changed")
        require(sha(item["research_config"]) == item["research_sha256"], "Research data configuration changed")
        manifest = read(directory / "manifest.json")
        scaler = read(directory / "scaler_params.json")
        require(manifest["data_source"] == "sina" and len(manifest["stock_codes"]) == 300
                and manifest["feature_cols"] == FEATURES and manifest["price_basis"] == "hfq_research"
                and manifest["train_end"] == fold + "-03-31" and manifest["val_end"] == fold + "-06-30"
                and scaler["train_end"] == manifest["train_end"] and scaler["ddof"] == 0, "Dataset temporal/scaler/source contract changed")
        datasets[fold] = dict(train_end=manifest["train_end"], val_end=manifest["val_end"], scaler_fit_rows=scaler["fit_rows"],
                             files=dict(item["files"], **{"manifest.json": item["manifest_sha256"]}))
    require(protocol["planned_actual_training_steps"] == 131072 and protocol["budgets"] == {"screen": 8192, "confirm": 16384}
            and protocol["test_evaluated"] is False and protocol["actionable"] is False, "Registered budget/protection changed")
    return dict(source_files=96, original_source_files=91, original_sources_unchanged=True,
                protocol_sha256=PROTOCOL_SHA, document_sha256=DOCUMENT_SHA,
                registered_git=protocol["git"], datasets=datasets)


class MarketArrays:
    def __init__(self, protocol, fold, codes):
        directory = Path(protocol["folds"][fold]["directory"])
        self.panel = pd.read_parquet(directory / "panel_data.parquet").reindex(
            columns=pd.MultiIndex.from_product([codes, FEATURES]))
        prices = pd.read_parquet(directory / "prices.parquet")
        require(self.panel.index.equals(prices.index), "Price/feature calendars differ")
        self.dates = self.panel.index
        self.features = self.panel.to_numpy().reshape(len(self.dates), len(codes), len(FEATURES))
        self.close = prices.loc[:, pd.IndexSlice[codes, "close"]].to_numpy()
        self.mark = pd.DataFrame(self.close).ffill().to_numpy()
        self.open = prices.loc[:, pd.IndexSlice[codes, "open"]].to_numpy()
        finite = np.isfinite(self.features).all(axis=2)
        self.valid = pd.DataFrame(finite).rolling(5).sum().eq(5).to_numpy() & np.isfinite(self.mark)
        self.lookup = {str(pd.Timestamp(date)): index for index, date in enumerate(self.dates)}
        self.codes = codes

    def indices(self, values):
        return np.array([self.lookup[str(pd.Timestamp(str(value)))] for value in values], dtype=int)

    def evaluation_dates(self, specification):
        before = self.dates[self.dates <= pd.Timestamp(specification["train_end"])]
        return self.dates[(self.dates >= before[-1]) & (self.dates <= pd.Timestamp(specification["val_end"]))]


def market_arrays(protocol, fold, codes):
    key = fold, tuple(codes)
    if key not in MARKETS:
        MARKETS[key] = MarketArrays(protocol, fold, codes)
    return MARKETS[key]


def stock_observation(data, row, market, decision_index):
    obs = {name[4:]: data[name][row] for name in data if name.startswith("obs_")}
    obs["stock_features"] = np.where(np.isfinite(market.features[decision_index]), market.features[decision_index], 0.).astype(np.float32)[:, None, :]
    obs["receivable_ratio"] = np.array([data["decision_receivables"][row] / data["decision_equity"][row]], dtype=np.float32)
    require(np.array_equal(obs["valid_mask"].astype(bool), market.valid[decision_index]), "Native common support mask differs from immutable features")
    return obs


def account_summary(obs):
    weights = obs["portfolio"][:-1].astype(np.float64)
    latest = obs["stock_features"][:, -1].astype(np.float64)
    valid, legal = obs["valid_mask"].astype(bool), obs["legal_directions"].astype(bool)
    shares, execution = obs["share_state"], obs["execution_features"]
    held = shares[:, 0] > 0
    locked = held & ((shares[:, 1] < shares[:, 0]) | execution[:, 2].astype(bool))
    return (latest * (weights * valid)[:, None]).sum(0), np.array([
        obs["portfolio"][-1], obs["receivable_ratio"][0], weights.sum(), held.mean(), locked.mean(),
        obs["allocation_context"][5], weights[execution[:, 2].astype(bool)].sum(), execution[:, 1].astype(np.float64).sum(),
        np.square(weights).sum(), weights[valid].sum(), valid.mean(), legal[:, 1].mean(), legal[:, 2].mean(), legal[:, 3].mean(),
    ], dtype=np.float64)


def expected_vector(agent, benchmark, phase, agent_equity, benchmark_equity, codes):
    latest, valid = agent["stock_features"][:, -1].astype(np.float64), agent["valid_mask"].astype(bool)
    mean = latest[valid].mean(0) if valid.any() else np.zeros(20)
    ranked = sorted(np.flatnonzero(valid), key=lambda index: (-float(latest[index, FEATURES.index("momentum_20")]), codes[index]))
    chosen = ranked[:int(agent["top_n"][0])]
    candidate = latest[chosen].mean(0) if chosen else np.zeros(20)
    agent_held, agent_state = account_summary(agent)
    benchmark_held, benchmark_state = account_summary(benchmark)
    return np.r_[np.eye(20)[phase], mean, agent_held, benchmark_held, candidate, agent_state, benchmark_state,
                 np.log(agent_equity) - np.log(benchmark_equity)].astype(np.float32)


def account_batch(data, latest, valid):
    weights = data["obs_portfolio"][:, :-1].astype(np.float64)
    legal = data["obs_legal_directions"].astype(bool)
    shares, execution = data["obs_share_state"], data["obs_execution_features"].astype(np.float64)
    held = shares[:, :, 0] > 0
    locked = held & ((shares[:, :, 1] < shares[:, :, 0]) | execution[:, :, 2].astype(bool))
    summary = (latest * (weights * valid)[:, :, None]).sum(axis=1)
    account = np.column_stack([
        data["obs_portfolio"][:, -1], (data["decision_receivables"] / data["decision_equity"]).astype(np.float32),
        weights.sum(1), held.mean(1), locked.mean(1), data["obs_allocation_context"][:, 5],
        (weights * execution[:, :, 2].astype(bool)).sum(1), execution[:, :, 1].sum(1),
        np.square(weights).sum(1), (weights * valid).sum(1), valid.mean(1),
        legal[:, :, 1].mean(1), legal[:, :, 2].mean(1), legal[:, :, 3].mean(1),
    ])
    return summary, account


def expected_vector_batch(agent, benchmark, market, indices, codes):
    raw = market.features[indices]
    latest = np.where(np.isfinite(raw), raw, 0.).astype(np.float32).astype(np.float64)
    valid = agent["obs_valid_mask"].astype(bool)
    count = np.maximum(1, valid.sum(1))
    mean = (latest * valid[:, :, None]).sum(1) / count[:, None]
    scores = latest[:, :, FEATURES.index("momentum_20")]
    tie = np.argsort(np.argsort(np.asarray(codes)))
    ranked = np.lexsort((np.broadcast_to(tie, valid.shape), -np.where(valid, scores, -np.inf)), axis=1)
    top = agent["obs_top_n"][:, 0].astype(int)
    chosen = ranked[:, :min(int(top.max()), len(codes))]
    selected_valid = np.take_along_axis(valid, chosen, axis=1) & (np.arange(chosen.shape[1])[None, :] < top[:, None])
    selected = np.take_along_axis(latest, chosen[:, :, None], axis=1)
    candidate = (selected * selected_valid[:, :, None]).sum(1) / np.maximum(1, selected_valid.sum(1))[:, None]
    agent_held, agent_state = account_batch(agent, latest, valid)
    benchmark_held, benchmark_state = account_batch(benchmark, latest, valid)
    return np.concatenate([np.eye(20)[agent["action_phase"]], mean, agent_held, benchmark_held,
        candidate, agent_state, benchmark_state,
        (np.log(agent["decision_equity"]) - np.log(benchmark["decision_equity"]))[:, None]], axis=1).astype(np.float32)


def expected_decoder(obs, rebalance, contract):
    n = len(contract["stock_codes"])
    raw = np.zeros(2 * n)
    if rebalance:
        legal = obs["legal_directions"].astype(bool)
        scores = obs["stock_features"][:, -1, FEATURES.index("momentum_20")]
        chosen = sorted(np.flatnonzero(obs["valid_mask"]), key=lambda index: (-float(scores[index]), contract["stock_codes"][index]))[:int(obs["top_n"][0])]
        selected = np.zeros(n, bool)
        selected[chosen] = True
        raw[:n][~selected & legal[:, 3]] = 3
        buy = np.array([index for index in chosen if legal[index, 1]], int)
        raw[buy] = 1
        raw[n + buy] = np.linspace(4., -4., len(buy)) if len(buy) else []
    return raw, map_trade_action(raw, obs, contract["logit_bound"], contract["broker"]["lot_size"], contract["stock_codes"])


def book(data, market, contract):
    decision, execution = market.indices(data["decision_date"]), market.indices(data["execution_date"])
    require(np.array_equal(execution, decision + 1), "Native account skipped a union-calendar step")
    close(data["decision_marks"], market.mark[decision], "Native decision valuation differs")
    close(data["opening_prices"], market.open[execution], "Native opening quote differs")
    expected_open = np.where(np.isfinite(market.open[execution]), market.open[execution], market.mark[decision])
    close(data["opening_valuation_prices"], expected_open, "Native opening valuation differs")
    positions, after, filled = data["opening_positions"], data["after_positions"], data["execution_filled_shares"]
    close(data["decision_equity"], data["decision_cash"] + data["decision_receivables"] + np.nansum(data["decision_positions"] * market.mark[decision], axis=1), "Decision NAV fails account reconciliation")
    close(data["opening_equity"], data["opening_cash"] + data["opening_receivables"] + np.nansum(positions * expected_open, axis=1), "Opening NAV fails account reconciliation")
    close(after, positions + filled, "Native filled shares do not reconcile with the final holdings")
    close(data["after_cash"], data["opening_cash"] - data["execution_filled_notional"].sum(1) - data["execution_fee"].sum(1), "Native cash misses actual fees or trade principal")
    close(data["after_equity"], data["after_cash"] + data["after_receivables"] + np.nansum(after * market.mark[execution], axis=1), "Final NAV fails account reconciliation")
    require((after >= 0).all() and (data["after_cash"] >= -1e-8).all(), "Native book has negative shares/cash")
    require((np.maximum(-filled, 0) <= data["opening_available_positions"] + 1e-8).all(), "Native sell violates available shares")
    close(data["after_available_positions"], data["opening_available_positions"] + np.minimum(filled, 0), "T+1 sellability includes newly bought shares")
    active = filled != 0
    nominal = np.abs(data["execution_filled_notional"])
    broker = contract["broker"]
    expected_fee = np.where(active, np.maximum(broker["min_commission"], nominal * broker["commission"])
                            + np.where(filled < 0, nominal * broker["sell_tax"], 0.), 0.)
    close(data["execution_fee"], expected_fee, "Native min commission or sell tax is incorrect")
    expected_slip = np.where(active, np.abs(filled) * market.open[execution] * broker["slippage"], 0.)
    close(data["execution_slippage_cost"], expected_slip, "Native slippage is inconsistent")
    full_exit = (filled < 0) & (after == 0) & data["action_exit_mask"].astype(bool)
    regular = active & ~full_exit
    require((np.abs(filled[regular] / broker["lot_size"] - np.rint(filled[regular] / broker["lot_size"])) <= 1e-8).all(), "Non-full EXIT fill violates whole lots")
    minimum = broker["min_commission"] / contract["trade"]["max_commission_ratio"]
    require((nominal[regular] >= minimum - 1e-6).all(), "Non-full EXIT fill is below its economic hurdle")
    require((nominal[regular] <= np.broadcast_to(data["opening_equity"][:, None], nominal.shape)[regular] * contract["trade"]["max_trade_weight"] + 1e-6).all(), "Non-full EXIT exceeds the per-step nominal cap")
    require(np.array_equal(data["obs_valid_mask"].astype(bool), market.valid[decision]), "Native common support mask differs from immutable features")
    weights = np.where(data["decision_positions"] > 0,
                       data["decision_positions"] * market.mark[decision] / data["decision_equity"][:, None], 0.)
    close(data["obs_portfolio"][:, :-1], weights, "Observed holdings do not match actual shares", atol=1e-6)
    close(data["obs_portfolio"][:, -1], data["decision_cash"] / data["decision_equity"], "Observed cash differs", atol=1e-6)
    return decision, dict(steps=len(decision), fills=int(active.sum()), full_exits=int(full_exit.sum()),
                          fees=float(expected_fee.sum()), slippage=float(expected_slip.sum()))


def trace_audit(directory, manifest, market, contract, training):
    verify_timing_trace(directory, manifest, steps=manifest["total_steps"], expected_contract_sha256=json_digest(contract))
    total = dict(steps=0, agent_fills=0, benchmark_fills=0, fees=0., benchmark_fees=0.,
                 agent_rebalances=0, benchmark_rebalances=0,
                 agent_rebalance_decoder_samples=0, benchmark_rebalance_decoder_samples=0,
                 decoder_scope="Fixed first and last actual REBALANCE in every native chunk; all HOLD coordinates checked in full",
                 causal_vector_reconstruction_tolerance=1e-6)
    previous = None
    for chunk, reference in zip(manifest["chunks"], manifest["benchmark_trace"]["chunks"]):
        paths = (Path(directory) / chunk["path"], Path(directory) / "benchmark" / reference["path"])
        for path in paths:
            with zipfile.ZipFile(path, "r") as archive:
                require(archive.testzip() is None, "A sealed native archive has a bad CRC")
        with np.load(paths[0], allow_pickle=False) as agent_file, np.load(paths[1], allow_pickle=False) as benchmark_file:
            agent = {name: agent_file[name] for name in agent_file.files}
            benchmark = {name: benchmark_file[name] for name in benchmark_file.files}
            decision, agent_stats = book(agent, market, contract)
            benchmark_decision, benchmark_stats = book(benchmark, market, contract)
            require(np.array_equal(decision, benchmark_decision), "Parallel accounts used different decisions")
            require(not training or np.all(agent["action_has_policy_probabilities"]), "A sampled training row lacks its actual categorical density")
            phases = agent["action_phase"]
            continued = np.r_[previous is not None and not previous["done"], ~agent["done"][:-1].astype(bool)]
            previous_phases = np.r_[previous["phase"] if previous else 0, phases[:-1]]
            require(np.array_equal(phases[continued], ((previous_phases + 1) % 20)[continued]), "Macro phase reset outside an episode boundary")
            for name in ("positions", "available_positions", "cash", "receivables", "equity"):
                close(agent["decision_" + name][~continued], benchmark["decision_" + name][~continued], "New episode benchmark did not clone actual initial account")
                for label, values in (("agent", agent), ("benchmark", benchmark)):
                    first = previous[label][name] if previous else values["decision_" + name][0]
                    old = np.concatenate([np.asarray(first)[None], values["after_" + name][:-1]])
                    close(values["decision_" + name][continued], old[continued], "A live account was reset or changed between native steps")
            vector = expected_vector_batch(agent, benchmark, market, decision, contract["stock_codes"])
            close(vector, agent["action_timing_observation"], "Original 129D policy input differs from causal account/market summaries", atol=1e-6, rtol=1e-6)
            for label, data, rebalance in (("agent", agent, agent["action_macro_action"].astype(bool)),
                                           ("benchmark", benchmark, phases == 0)):
                held = ~rebalance
                require(np.all(data["raw_packed_action"][held] == 0)
                        and np.all(data["action_shares"][held] == 0)
                        and np.all(data["action_priority"][held] == 0)
                        and np.all(data["action_exit_mask"][held] == 0), "HOLD changed a derived stock coordinate")
                close(data["action_freeze_mask"], data["obs_execution_features"][:, :, 2].astype(bool), "Decoded frozen mask differs")
                rebalance_rows = np.flatnonzero(rebalance)
                sampled = np.unique(rebalance_rows[[0, -1]]) if len(rebalance_rows) else np.array([], dtype=int)
                for row in sampled:
                    observation = stock_observation(data, row, market, decision[row])
                    raw, decoded = expected_decoder(observation, True, contract)
                    require(np.array_equal(raw, data["raw_packed_action"][row]), "Stock raw action is not the unchanged economic rule decoder")
                    for name in ("shares", "priority", "exit_mask"):
                        close(data["action_" + name][row], decoded[name], "Decoded stock intent changed: " + name)
                total[label + "_rebalances"] += len(rebalance_rows)
                total[label + "_rebalance_decoder_samples"] += len(sampled)
            previous = dict(phase=int(phases[-1]), done=bool(agent["done"][-1]),
                agent={name: agent["after_" + name][-1].copy() for name in ("positions", "available_positions", "cash", "receivables", "equity")},
                benchmark={name: benchmark["after_" + name][-1].copy() for name in ("positions", "available_positions", "cash", "receivables", "equity")})
            total["steps"] += agent_stats["steps"]
            total["agent_fills"] += agent_stats["fills"]
            total["benchmark_fills"] += benchmark_stats["fills"]
            total["fees"] += agent_stats["fees"]
            total["benchmark_fees"] += benchmark_stats["fees"]
    return total


def model_audit(directory, metadata, initial=None, audit=None):
    import torch

    with zipfile.ZipFile(Path(directory) / "model.zip", "r") as archive:
        require(archive.testzip() is None, "Sealed model ZIP CRC failed")
        serialized = json.loads(archive.read("data"))
        policy_bytes = archive.read("policy.pth")
        theta = torch.load(io.BytesIO(policy_bytes), map_location="cpu", weights_only=True)
        optimizer = torch.load(io.BytesIO(archive.read("policy.optimizer.pth")), map_location="cpu", weights_only=True)
    require(sum(value.numel() for value in theta.values()) == 12805 and len(theta) == 8
            and all(torch.isfinite(value).all() for value in theta.values()), "Actual policy tensors differ from 12805 finite parameters")
    shapes = {"mlp_extractor.value_net.0.weight": (64, 129), "mlp_extractor.value_net.0.bias": (64,),
              "mlp_extractor.value_net.2.weight": (64, 64), "mlp_extractor.value_net.2.bias": (64,),
              "action_net.weight": (2, 129), "action_net.bias": (2,),
              "value_net.weight": (1, 64), "value_net.bias": (1,)}
    require(set(theta) == set(shapes) and all(tuple(theta[name].shape) == shape and theta[name].dtype == torch.float32
                                            for name, shape in shapes.items()), "Actual policy parameter shapes/dtype changed")
    require(metadata["actual_device"] == "cpu" and metadata["policy_record"]["parameter_count"] == 12805
            and metadata["policy_record"]["observation_shape"] == [129], "Policy metadata device/schema changed")
    require(serialized["timing_registered_contract"] == metadata["contract"] and serialized["num_timesteps"] == metadata["num_timesteps"], "Archive timing registration differs")
    counts = {}
    names = list(theta)
    ids = [item for group in optimizer["param_groups"] for item in group["params"]]
    require(len(ids) == len(names), "Adam parameter coordinate set differs")
    for name, number in zip(names, ids):
        state = optimizer["state"].get(number)
        if state is not None:
            require(all(torch.isfinite(value).all() for value in state.values()), "Adam state is nonfinite")
            counts[name] = int(state["step"].item())
    if initial is None:
        require(not counts and metadata["num_timesteps"] == 0, "Initialized policy has optimizer updates")
        expected = torch.zeros_like(theta["action_net.weight"])
        expected[0, 1:20], expected[1, 0] = 2., 2.
        require(torch.equal(theta["action_net.weight"], expected) and torch.count_nonzero(theta["action_net.bias"]) == 0, "Calendar initialization changed")
    else:
        require(set(counts) == set(names) and min(counts.values()) == max(counts.values()) == audit["actual_optimizer_updates"], "Actual Adam counts differ from signed audit")
        require(counts == audit["optimizer_steps_by_parameter"], "Per-parameter optimizer evidence differs")
        for name, value in theta.items():
            close(float(torch.linalg.vector_norm(value - initial[name])), audit["tensor_delta_l2"][name], "Saved parameter delta differs", atol=1e-9)
        require(any(not torch.equal(value, initial[name]) for name, value in theta.items()), "Formal policy did not update")
    return theta, dict(parameters=12805, tensors=8, active_adam_tensors=len(counts),
                       adam_updates=min(counts.values()) if counts else 0,
                       policy_pth_sha256=hashlib.sha256(policy_bytes).hexdigest())


def schedule_audit(audit, config, market, specification):
    schedule = audit["training_schedule"]["schedule"]
    selected = market.dates <= pd.Timestamp(specification["train_end"])
    indices = np.flatnonzero(selected & market.valid.any(axis=1))
    require(len(indices) > 0, "No causal training starts")
    dates = market.dates[(market.dates >= market.dates[indices[0]]) & selected]
    sampler = StratifiedTimingSampler(dates, 240)
    sampler.seed(config["seed"])
    expected = [sampler.next() for _ in schedule["history"]]
    require(expected == schedule["history"] and sampler.audit() == schedule, "Original independent episode schedule/RNG cannot be replayed")
    steps = audit["actual_timesteps"]
    require(audit["experience"]["completed_episodes"] == steps // 240
            and audit["experience"]["partial_episode_steps"] == steps % 240
            and audit["experience"]["started_episodes"] == (steps + 239) // 240, "Actual full/partial episode budget differs")
    return dict(episodes=len(expected), completed=steps // 240, partial_steps=steps % 240,
                distinct_stratum_modes=len({(item["date_stratum"], item["initial_mode"]) for item in expected}))


def case_metrics(folder, result):
    for filename, field in (("daily.csv", "metrics"), ("benchmark_daily.csv", "benchmark_metrics")):
        daily = pd.read_csv(Path(folder) / filename, float_precision="round_trip")
        equity = daily.equity.to_numpy()
        metrics = result[field]
        close(metrics["cumulative_return"], equity[-1] / equity[0] - 1, "Native curve ROI differs", atol=1e-12)
        close(metrics["max_drawdown"], (equity / np.maximum.accumulate(equity) - 1).min(), "Native curve drawdown differs", atol=1e-12)
        close(metrics["total_fees"], daily.fee.sum(), "Native curve fees differ")
        close(metrics["total_slippage_cost"], daily.slippage_cost.sum(), "Native curve slippage differs")
        require(metrics["trading_steps"] == len(daily) - 1, "Native curve step count differs")


def sealed_trial(study, protocol, directory):
    marker = Path(directory) / "completed.json"
    require(marker.exists(), "Do not audit an unsealed trial")
    completed = read(marker)
    verify_leaf_files(directory, completed["files"])
    config, row = read(Path(directory) / "config.json"), completed["record"]
    require(completed["protocol_sha256"] == json_digest(protocol) and completed["config_sha256"] == json_digest(config), "Trial protocol/config identity differs")
    spec = config["timing"]
    require(Path(directory).name == f'{spec["phase"]}_{spec["fold"]}_{spec["reward_basis"]}_{config["seed"]}', "Trial condition name changed")
    require(row["steps"] == protocol["budgets"][spec["phase"]], "Trial final budget differs")
    initial_meta, final_meta = [read(Path(directory) / name / "metadata.json") for name in ("initial_model", "final_model")]
    for meta in (initial_meta, final_meta):
        require(meta["source_hashes"] == protocol["source_hashes"] and meta["config"] == config, "Model source/config differs")
    require(initial_meta["contract"] == final_meta["contract"], "Initial/final prediction contracts differ")
    audit = read(Path(directory) / "final_audit.json")
    require(audit == row["audit"] and audit["actual_timesteps"] == row["steps"], "Signed training audit differs from native record")
    require(audit["experience"]["total_steps"] == audit["experience"]["benchmark_steps"] == row["steps"], "Parallel benchmark counter is missing or counted as RL")
    require(audit["training_schedule"]["agent_environment_steps"] == audit["training_schedule"]["benchmark_extra_environment_steps"] == row["steps"], "Actual wrapper counters differ")
    verify_rollouts(directory, audit["rollout_buffers"], row["steps"], audit["training_trace_prefix"])
    initial, initial_result = model_audit(Path(directory) / "initial_model", initial_meta)
    _, final_result = model_audit(Path(directory) / "final_model", final_meta, initial, audit)
    contract = final_meta["contract"]
    market = market_arrays(protocol, spec["fold"], contract["stock_codes"])
    schedule = schedule_audit(audit, config, market, protocol["folds"][spec["fold"]])
    trace = trace_audit(Path(directory) / "training_trace", audit["training_trace_prefix"], market, contract, True)
    cases = {}
    expected_dates = market.evaluation_dates(protocol["folds"][spec["fold"]])
    for role in ("initialized", "trained"):
        for mode in MODES:
            folder = Path(directory) / role / mode
            result = read(folder / "result.json")
            require(result["contract"] == contract and result["initial_mode"] == mode
                    and result["metrics"] == row["metrics"][mode][role]
                    and result["benchmark_metrics"] == row["metrics"][mode]["benchmark"], "Native case metrics/identity differ")
            verify_evaluation_dates(folder, result, expected_dates)
            case_metrics(folder, result)
            cases[role + "/" + mode] = trace_audit(folder / "trace", result["trace"], market, contract, False)
    return dict(completion_sha256=sha(marker), registered_leaves=len(completed["files"]),
                actual_rl_steps=row["steps"], separate_benchmark_training_steps=row["steps"],
                initial=initial_result, final=final_result, training=trace, schedule=schedule, cases=cases)


def sealed_rules(study, protocol, directory):
    marker = Path(directory) / "completed.json"
    completed = read(marker)
    verify_leaf_files(directory, completed["files"])
    fold = Path(directory).name
    cases = {}
    for mode in MODES:
        folder = Path(directory) / mode
        result = read(folder / "result.json")
        contract = result["contract"]
        market = market_arrays(protocol, fold, contract["stock_codes"])
        require(result["metrics"] == result["benchmark_metrics"], "Calendar control has different independent-book economics")
        verify_evaluation_dates(folder, result, market.evaluation_dates(protocol["folds"][fold]))
        case_metrics(folder, result)
        legacy = pd.read_csv(folder / "legacy_daily.csv", float_precision="round_trip")
        native = pd.read_csv(folder / "daily.csv", float_precision="round_trip")
        for field in ("equity", "cash", "daily_return", "fee", "slippage_cost", "turnover", "holding_count", "unfilled_orders"):
            require(np.array_equal(native[field], legacy[field]), "Calendar control differs from the preserved legacy rule curve")
        cases[mode] = trace_audit(folder / "trace", result["trace"], market, contract, False)
    return dict(completion_sha256=sha(marker), registered_leaves=len(completed["files"]), cases=cases)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / "reports/runs/ppo_timing_20261014"))
    parser.add_argument("--run", action="append")
    parser.add_argument("--sealed", action="store_true")
    parser.add_argument("--rules", action="store_true")
    args = parser.parse_args(argv)
    study = Path(args.root).resolve()
    protocol = read(study / "protocol.json")
    result = dict(status="passed", scope="frozen_registration", registration=registration(study, protocol),
                  model_instances=0, network_forwards=0, environment_steps=0, gradient_calls=0,
                  optimizer_steps=0, registered_leaf_writes=0, frozen_source_writes=0)
    if args.sealed:
        trials = {}
        running = study / "independent_audit_running.json"
        running.write_text(json.dumps(dict(status="running", completed_trials=[], planned_trials=10,
            registered_leaf_writes=0, network_forwards=0, environment_steps=0), indent=2) + "\n", encoding="utf-8")
        for marker in sorted((study / "runs").glob("*/completed.json")):
            if args.run is None or marker.parent.name in args.run:
                trials[marker.parent.name] = sealed_trial(study, protocol, marker.parent)
                print("AUDITED " + marker.parent.name, flush=True)
                running.write_text(json.dumps(dict(status="running", completed_trials=list(trials),
                    completed_trial_count=len(trials), actual_audited_rl_steps=sum(item["actual_rl_steps"] for item in trials.values()),
                    registered_leaf_writes=0, network_forwards=0, environment_steps=0), indent=2) + "\n", encoding="utf-8")
        result.update(scope="registration_and_listed_sealed_native_trials", sealed_trials=trials,
                      sealed_trial_count=len(trials), actual_rl_steps=sum(item["actual_rl_steps"] for item in trials.values()))
        groups = {}
        for name, item in trials.items():
            config = read(study / "runs" / name / "config.json")
            groups.setdefault((config["timing"]["fold"], config["seed"]), set()).add(item["initial"]["policy_pth_sha256"])
        require(all(len(items) == 1 for items in groups.values()), "Reward methods have different initial tensor bytes")
        result["same_initial_theta_across_reward_methods"] = True
    if args.rules:
        result["rules"] = {marker.parent.name: sealed_rules(study, protocol, marker.parent)
                           for marker in sorted((study / "rules").glob("*/completed.json"))}
    output = study / "independent_audit_progress.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.sealed:
        running.write_text(json.dumps(dict(status="passed", completed_trials=list(trials), completed_trial_count=len(trials),
            rule_fold_count=len(result.get("rules", {})), progress_sha256=sha(output), registered_leaf_writes=0,
            network_forwards=0, environment_steps=0), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("registration", "sealed_trials")}, sort_keys=True))


if __name__ == "__main__":
    main()
