"""Native ledger evidence for a categorical timing controller, not stock likelihoods."""

from copy import deepcopy
from pathlib import Path

import numpy as np

from src.agents.trade_mapping import TradeOrder
from .research_trade import TradeTrace, verify_trace_prefix


TIMING_FIELDS = ("macro_action", "macro_log_prob", "macro_value", "macro_probabilities",
                 "has_policy_probabilities", "timing_observation", "phase",
                 "raw_net_return", "benchmark_net_return", "learning_reward")


class TimingTrace(TradeTrace):
    """Preserve the actual macro sample alongside its deterministic 2N decoder.

    Extra fields enter the row before TradeTrace's internal flush. The inherited
    raw_packed_action is a derived economic order, never the PPO sample or its
    probability density.
    """

    DIAG_FIELDS = TradeTrace.DIAG_FIELDS + TIMING_FIELDS

    def __init__(self, directory, identity, chunk_steps=512):
        super().__init__(directory, identity, chunk_steps)
        self.benchmark_trace = TradeTrace(Path(directory) / "benchmark",
            dict(deepcopy(identity), role="unchanged_20day_benchmark", likelihood="deterministic stock decoder; no neural density"), chunk_steps)

    def append(self, date, observation, action, info, *, timing_observation=None,
               macro_action=None, log_prob=None, value=None, reward=None, done=None,
               action_probabilities=None):
        if not isinstance(action, TradeOrder):
            raise ValueError("Timing trace requires the exact decoded TradeOrder")
        timing = info.get("timing", {})
        vector = np.asarray(timing_observation if timing_observation is not None
                            else timing.get("observation"), dtype=np.float32)
        raw = macro_action if macro_action is not None else timing.get("macro_action")
        raw_array = np.asarray(raw)
        if raw_array.size != 1 or raw_array.dtype.kind not in "iuf" or not np.isfinite(raw_array).all():
            raise ValueError("Timing action must be a finite scalar categorical sample")
        scalar = float(raw_array.reshape(-1)[0])
        if scalar not in (0., 1.):
            raise ValueError("Timing action must be HOLD=0 or REBALANCE_M20=1")
        if timing.get("macro_action", int(scalar)) != int(scalar):
            raise ValueError("Macro sample differs from the action actually executed")
        width = self.identity.get("timing_observation_width", 129)
        if vector.shape != (width,) or width < 20 or not np.isfinite(vector).all():
            raise ValueError("Timing trace needs the original finite controller input")
        if "observation" in timing and not np.array_equal(vector, timing["observation"]):
            raise ValueError("Trace controller input differs from the executed decision")
        phase = int(np.argmax(vector[:20]))
        if not np.array_equal(vector[:20], np.eye(20, dtype=np.float32)[phase]):
            raise ValueError("Controller input must begin with the exact phase one-hot")
        if timing.get("phase", phase) != phase:
            raise ValueError("Timing phase disagrees with the recorded input")
        has_policy = log_prob is not None or value is not None or action_probabilities is not None
        if has_policy and (log_prob is None or value is None):
            raise ValueError("Policy rows require both actual macro log probability and critic value")
        probabilities = np.zeros(2, dtype=np.float64) if action_probabilities is None else np.asarray(action_probabilities, dtype=np.float64)
        if probabilities.shape != (2,) or not np.isfinite(probabilities).all():
            raise ValueError("Macro probabilities must have two finite coordinates")
        if action_probabilities is not None and (np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1., atol=1e-6)):
            raise ValueError("Macro probabilities must sum to one")
        raw_return = float(timing.get("raw_net_return", info.get("raw_net_return", info.get("reward", 0.))))
        benchmark_return = float(timing.get("benchmark_net_return", info.get("benchmark_net_return", 0.)))
        learning_reward = float(reward if reward is not None else timing.get("learning_reward", info.get("reward", 0.)))
        if has_policy and (np.asarray(log_prob).size != 1 or np.asarray(value).size != 1):
            raise ValueError("A timing row has exactly one macro likelihood and one critic value")
        probability = float(np.asarray(log_prob).reshape(-1)[0]) if has_policy else 0.
        predicted = float(np.asarray(value).reshape(-1)[0]) if has_policy else 0.
        if not np.isfinite([raw_return, benchmark_return, learning_reward, probability, predicted]).all():
            raise ValueError("Timing probabilities, values and realized returns must be finite")
        if has_policy:
            if probability > 1e-6:
                raise ValueError("Categorical log probability cannot be positive")
            if action_probabilities is None:
                selected_probability = float(np.exp(probability))
                probabilities[int(scalar)] = selected_probability
                probabilities[1 - int(scalar)] = 1. - selected_probability
            elif not np.isclose(np.exp(probability), probabilities[int(scalar)], atol=1e-6):
                raise ValueError("Macro log probability disagrees with its categorical probabilities")
        diagnostics = deepcopy(action.diagnostics)
        diagnostics.update(macro_action=np.int64(scalar), macro_log_prob=probability,
            macro_value=predicted, macro_probabilities=probabilities,
            has_policy_probabilities=bool(has_policy), timing_observation=vector.copy(), phase=np.int64(phase),
            raw_net_return=raw_return, benchmark_net_return=benchmark_return, learning_reward=learning_reward)
        order = TradeOrder(action.shares, action.freeze_mask, action.exit_mask, action.priority,
                           action.top_n, diagnostics)
        copied_info = dict(info, action_diagnostics=diagnostics)
        benchmark = timing.get("benchmark_ledger")
        benchmark_obs = info.get("timing_benchmark_stock_observation")
        benchmark_action = np.asarray(timing.get("benchmark_raw_action"))
        n = len(self.identity["codes"])
        if (not isinstance(benchmark, dict) or not isinstance(benchmark_obs, dict)
                or benchmark_action.shape != (2 * n,) or benchmark.get("date") != info.get("date")):
            raise ValueError("Timing evidence requires the actual complete parallel benchmark ledger and original stock input")
        self.benchmark_trace.append(date, benchmark_obs, benchmark_action, benchmark,
                                    reward=benchmark_return, done=done)
        super().append(date, observation, order, copied_info, reward=learning_reward, done=done)

    def flush(self):
        self.benchmark_trace.flush()
        super().flush()

    def close(self):
        self.benchmark_trace.close()
        return super().close()

    def manifest(self):
        result = super().manifest()
        result.update(schema_version=3, action_layout="scalar macro + derived stock 2N",
            macro_actions={"0": "HOLD", "1": "REBALANCE_M20"},
            likelihood_semantics="action_macro_log_prob is categorical macro likelihood; raw_packed_action is deterministic stock decoder only",
            nonpolicy_probability_sentinel=0., timing_observation_width=self.identity.get("timing_observation_width", 129),
            benchmark_directory="benchmark", benchmark_trace=self.benchmark_trace.manifest())
        return result


def verify_timing_trace(directory, manifest, *, steps=None, expected_contract_sha256=None):
    """Read and validate signed arrays without loading a policy or stepping an env."""
    if (manifest.get("schema_version") != 3
            or manifest.get("action_layout") != "scalar macro + derived stock 2N"
            or manifest.get("macro_actions") != {"0": "HOLD", "1": "REBALANCE_M20"}
            or steps is not None and manifest.get("total_steps") != steps
            or expected_contract_sha256 is not None and manifest["identity"].get("contract_sha256") != expected_contract_sha256):
        raise ValueError("Timing trace contract, layout or complete budget differs")
    verify_trace_prefix(directory, manifest)
    benchmark = manifest.get("benchmark_trace")
    if (not isinstance(benchmark, dict) or manifest.get("benchmark_directory") != "benchmark"
            or benchmark.get("total_steps") != manifest["total_steps"]
            or benchmark["identity"].get("role") != "unchanged_20day_benchmark"
            or benchmark["identity"].get("contract_sha256") != manifest["identity"].get("contract_sha256")
            or benchmark["identity"].get("codes") != manifest["identity"].get("codes")):
        raise ValueError("Timing trace lacks the complete independently executed benchmark prefix")
    verify_trace_prefix(Path(directory) / "benchmark", benchmark)
    n = len(manifest["identity"]["codes"])
    width = manifest["timing_observation_width"]
    if len(benchmark["chunks"]) != len(manifest["chunks"]):
        raise ValueError("Agent and benchmark native chunk clocks differ")
    for chunk, benchmark_chunk in zip(manifest["chunks"], benchmark["chunks"]):
        if any(chunk[key] != benchmark_chunk[key] for key in ("first_step", "last_step", "steps")):
            raise ValueError("Agent and benchmark native prefix boundaries differ")
        with np.load(Path(directory) / chunk["path"], allow_pickle=False) as data, np.load(Path(directory) / "benchmark" / benchmark_chunk["path"], allow_pickle=False) as reference:
            count = chunk["steps"]
            if data["raw_packed_action"].shape != (count, 2 * n):
                raise ValueError("Decoded timing orders must retain the original 2N stock coordinates")
            actions, vector, phases = data["action_macro_action"], data["action_timing_observation"], data["action_phase"]
            if actions.shape != (count,) or not np.isin(actions, [0, 1]).all() or vector.shape != (count, width):
                raise ValueError("Timing scalar sample or input shape changed")
            if not np.isin(phases, np.arange(20)).all() or not np.array_equal(vector[:, :20], np.eye(20, dtype=np.float32)[phases]):
                raise ValueError("Timing input phase no longer matches the original sample")
            for name in TIMING_FIELDS:
                if not np.isfinite(data["action_" + name]).all():
                    raise ValueError("Nonfinite timing evidence: " + name)
            actual, directions = data["raw_packed_action"][:, :n], data["action_directions"]
            legal = data["obs_legal_directions"]
            if (not np.array_equal(actual, directions)
                    or not np.array_equal(data["raw_packed_action"][:, n:], data["action_size_latent"])
                    or not np.isin(actual, [0, 1, 2, 3]).all()
                    or not np.take_along_axis(legal, directions.astype(np.int64)[..., None], axis=2).all()):
                raise ValueError("Decoded stock direction is outside the known legal support")
            held = actions == 0
            if np.any(directions[held] != 0) or np.any(data["action_shares"][held] != 0):
                raise ValueError("HOLD unexpectedly creates a decoded trade")
            teacher = ~data["action_has_policy_probabilities"].astype(bool)
            declared_policy = manifest["identity"].get("has_neural_macro_likelihood")
            if declared_policy is not None and np.any(~teacher != declared_policy):
                raise ValueError("Native rule and actual neural macro likelihood roles were swapped")
            if (np.any(data["action_macro_log_prob"][teacher] != 0)
                    or np.any(data["action_macro_value"][teacher] != 0)
                    or np.any(data["action_macro_probabilities"][teacher] != 0)):
                raise ValueError("A rule row pretends to have a neural likelihood")
            probabilities = data["action_macro_probabilities"][~teacher]
            log_probability = data["action_macro_log_prob"][~teacher]
            samples = actions[~teacher]
            if (np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-6)
                    or not np.allclose(probabilities[np.arange(len(samples)), samples], np.exp(log_probability), atol=1e-6)):
                raise ValueError("Original categorical macro likelihood and probabilities do not reconcile")
            if (reference["raw_packed_action"].shape != (count, 2 * n)
                    or not np.array_equal(data["decision_date"], reference["decision_date"])
                    or not np.array_equal(data["execution_date"], reference["execution_date"])
                    or "log_prob" in reference.files or "value" in reference.files):
                raise ValueError("Benchmark stock decoder must share clocks without a neural likelihood")
            off_calendar = phases != 0
            if np.any(reference["action_directions"][off_calendar] != 0) or np.any(reference["action_shares"][off_calendar] != 0):
                raise ValueError("Unchanged benchmark rebalanced outside its original twenty-day calendar")
            realized = data["after_equity"] / data["decision_equity"] - 1.
            benchmark_realized = reference["after_equity"] / reference["decision_equity"] - 1.
            if (not np.allclose(data["action_raw_net_return"], realized, atol=1e-12, rtol=0.)
                    or not np.allclose(data["action_benchmark_net_return"], benchmark_realized, atol=1e-12, rtol=0.)):
                raise ValueError("Agent or benchmark return does not reconcile with its native actual ledger")
            basis = manifest["identity"].get("reward_basis")
            if basis is not None:
                if basis not in ("log_net", "relative_log") or np.any(realized <= -1) or np.any(benchmark_realized <= -1):
                    raise ValueError("Timing reward basis or positive equity support changed")
                expected = np.log1p(realized) - (np.log1p(benchmark_realized) if basis == "relative_log" else 0.)
                if not np.allclose(data["action_learning_reward"], expected, atol=1e-12, rtol=0.):
                    raise ValueError("Timing learning reward does not reconcile with both native net ledgers")
    return manifest
