"""Synthetic categorical macro evidence, separate from decoded stock intents."""

from copy import deepcopy

import numpy as np
import pytest

from src.agents.trade_mapping import TradeOrder, map_trade_action
from src.training.timing_trace import TimingTrace, verify_timing_trace
from tests.test_exploration_policy import observation


def sample(macro=1):
    obs = {key: value[0].numpy() for key, value in observation(n=2).items()}
    obs["top_n"] = np.array([20.], dtype=np.float32)
    raw = np.array([1., 0., 9., 0.]) if macro else np.zeros(4)
    result = map_trade_action(raw, obs, stock_codes=["A", "B"])
    snapshot = dict(date="2020-01-01", codes=["A", "B"], positions=np.array([100., 100.]),
        available_positions=np.array([100., 100.]), cash=80000., equity=100000., receivables=0., marks=np.array([100., 100.]))
    vector = np.zeros(129, dtype=np.float32)
    vector[3] = 1.
    info = dict(action_diagnostics=result, orders=[], decision_snapshot=snapshot, opening_snapshot=deepcopy(snapshot),
        initial_account=dict(mode="random"), top_n=20, cash=80000., equity=100300., positions=np.array([100., 100.]),
        available_positions=np.array([100., 100.]), receivables=0., date="2020-01-02",
        timing=dict(observation=vector, macro_action=macro, phase=3, raw_net_return=.003, benchmark_net_return=.003, learning_reward=.001))
    benchmark = deepcopy(info)
    benchmark["action_diagnostics"] = map_trade_action(np.zeros(4), obs, stock_codes=["A", "B"])
    info["timing"]["benchmark_ledger"] = benchmark
    info["timing"]["benchmark_raw_action"] = np.zeros(4)
    info["timing_benchmark_stock_observation"] = deepcopy(obs)
    identity = dict(codes=["A", "B"], lot_size=100, min_commission=5., max_commission_ratio=.001,
                    commission=.0003, timing_observation_width=129, contract_sha256="contract")
    order = TradeOrder(result["shares"], result["freeze_mask"], result["exit_mask"], result["priority"], result["top_n"], result)
    return obs, raw, order, info, identity


@pytest.mark.parametrize("policy", [False, True])
@pytest.mark.parametrize("macro", [0, 1])
def test_native_trace_separates_macro_likelihood_and_derived_2n_at_flush(tmp_path, policy, macro):
    obs, raw, order, info, identity = sample(macro)
    trace = TimingTrace(tmp_path / "trace", identity, chunk_steps=2)
    for _ in range(3):
        trace.append("2020-01-01", obs, order, info, log_prob=np.log(.7 if macro else .3) if policy else None,
            value=.02 if policy else None, action_probabilities=np.array([.3, .7]) if policy else None,
            reward=.001, done=False)
    manifest = trace.close()
    assert [chunk["steps"] for chunk in manifest["chunks"]] == [2, 1]
    assert manifest["schema_version"] == 3 and "deterministic" in manifest["likelihood_semantics"]
    verify_timing_trace(tmp_path / "trace", manifest, steps=3, expected_contract_sha256="contract")
    for chunk in manifest["chunks"]:
        with np.load(tmp_path / "trace" / chunk["path"], allow_pickle=False) as data:
            np.testing.assert_array_equal(data["raw_packed_action"], np.tile(raw, (chunk["steps"], 1)))
            assert "log_prob" not in data and "value" not in data
            assert np.all(data["action_macro_action"] == macro)
            assert np.all(data["action_has_policy_probabilities"] == policy)
            assert np.all(data["action_phase"] == 3)
            assert np.all(data["action_raw_net_return"] == .003)
            assert np.all(data["action_benchmark_net_return"] == .003)


@pytest.mark.parametrize("bad", ["action", "nan", "shape", "onehot", "phase", "probability", "partial_probability", "reward"])
def test_append_rejects_wrong_macro_support_or_original_input(tmp_path, bad):
    obs, _, order, info, identity = sample()
    kwargs = {}
    if bad == "action":
        kwargs["macro_action"] = .5
    elif bad == "nan":
        kwargs["macro_action"] = np.nan
    elif bad == "shape":
        kwargs["timing_observation"] = np.zeros(109)
    elif bad == "onehot":
        info["timing"]["observation"][4] = 1.
    elif bad == "phase":
        info["timing"]["phase"] = 4
    elif bad == "probability":
        kwargs.update(log_prob=-1., value=0., action_probabilities=[.3, .8])
    elif bad == "partial_probability":
        kwargs["log_prob"] = -1.
    else:
        kwargs["reward"] = np.nan
    trace = TimingTrace(tmp_path / "trace", identity)
    with pytest.raises(ValueError):
        trace.append("2020-01-01", obs, order, info, **kwargs)


@pytest.mark.parametrize("bad", ["schema", "layout", "steps", "contract", "hash"])
def test_read_only_verifier_rejects_changed_registration(tmp_path, bad):
    obs, _, order, info, identity = sample()
    trace = TimingTrace(tmp_path / "trace", identity)
    trace.append("2020-01-01", obs, order, info)
    manifest = trace.close()
    if bad == "schema":
        manifest["schema_version"] = 2
    elif bad == "layout":
        manifest["action_layout"] = "2N"
    elif bad == "hash":
        manifest["chunks"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        verify_timing_trace(tmp_path / "trace", manifest, steps=2 if bad == "steps" else 1,
                            expected_contract_sha256="wrong" if bad == "contract" else "contract")


def test_trace_does_not_mutate_original_order_or_info(tmp_path):
    obs, _, order, info, identity = sample()
    keys = set(order.diagnostics)
    trace = TimingTrace(tmp_path / "trace", identity)
    trace.append("2020-01-01", obs, order, info)
    trace.close()
    assert set(order.diagnostics) == keys and set(info["action_diagnostics"]) == keys
    assert "macro_log_prob" not in info["timing"]
