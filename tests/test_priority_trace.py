"""Original 3N priority samples survive flushes and strict read-only validation."""

from copy import deepcopy

import numpy as np
import pytest

from src.agents.priority_mapping import map_priority_trade_action
from src.agents.trade_mapping import TradeOrder
from src.training.priority_trace import PriorityTrace, verify_priority_trace
from tests.test_exploration_policy import observation


def sample(variant="coupled", priority=None):
    obs = {key: value[0].numpy() for key, value in observation(n=2).items()}
    obs["portfolio"][:] = [.1, .1, .8]
    obs["top_n"] = np.array([20.], dtype=np.float32)
    raw = np.array([1., 0., 9., 0., 9. if priority is None else priority, 0.])
    result = map_priority_trade_action(raw, obs, stock_codes=["A", "B"], actor_variant=variant)
    snapshot = dict(date="2020-01-01", codes=["A", "B"], positions=np.array([100., 100.]),
        available_positions=np.array([100., 100.]), cash=80000., equity=100000., receivables=0., marks=np.array([100., 100.]))
    info = dict(action_diagnostics=result, orders=[], decision_snapshot=snapshot, opening_snapshot=deepcopy(snapshot),
        initial_account=dict(mode="random"), top_n=20, cash=80000., equity=100000., positions=np.array([100., 100.]),
        available_positions=np.array([100., 100.]), receivables=0., date="2020-01-02")
    identity = dict(actor_variant=variant, codes=["A", "B"], lot_size=100, min_commission=5.,
                    max_commission_ratio=.001, commission=.0003)
    order = TradeOrder(result["shares"], result["freeze_mask"], result["exit_mask"], result["priority"], result["top_n"], result)
    return obs, raw, order, info, identity


@pytest.mark.parametrize("use_order", [False, True])
@pytest.mark.parametrize("variant", ["coupled", "decoupled"])
def test_trace_preserves_unclipped_original_priorities_and_flush_boundary(tmp_path, use_order, variant):
    obs, raw, order, info, identity = sample(variant, None if variant == "coupled" else -12.)
    trace = PriorityTrace(tmp_path / "trace", identity, chunk_steps=2)
    for _ in range(3):
        trace.append("2020-01-01", obs, order if use_order else raw, info, log_prob=-1., value=0., reward=.001, done=False)
    manifest = trace.close()
    assert [chunk["steps"] for chunk in manifest["chunks"]] == [2, 1]
    assert manifest["schema_version"] == 2 and manifest["total_steps"] == 3
    verify_priority_trace(tmp_path / "trace", manifest, actor_variant=variant, steps=3)
    for chunk in manifest["chunks"]:
        with np.load(tmp_path / "trace" / chunk["path"], allow_pickle=False) as values:
            np.testing.assert_array_equal(values["raw_packed_action"], np.tile(raw, (chunk["steps"], 1)))
            np.testing.assert_array_equal(values["action_raw_priority_latent"], np.tile(raw[4:], (chunk["steps"], 1)))
            assert values["action_priority"][0, 0] == np.clip(raw[4], -8, 8)
            assert values["raw_packed_action"][0, 2] == 9.


@pytest.mark.parametrize("bad", ["shape", "nonfinite", "direction", "nonbuy_priority", "coupled_priority", "variant"])
def test_trace_rejects_invalid_original_support(tmp_path, bad):
    obs, raw, _, info, identity = sample()
    if bad == "shape":
        raw = raw[:4]
    elif bad == "nonfinite":
        raw[2] = np.nan
    elif bad == "direction":
        raw[0] = .5
    elif bad == "nonbuy_priority":
        raw[-1] = 1.
    elif bad == "coupled_priority":
        raw[4] = 0.
    else:
        identity["actor_variant"] = "other"
    trace = PriorityTrace(tmp_path / "trace", identity)
    with pytest.raises(ValueError):
        trace.append("2020-01-01", obs, raw, info)


@pytest.mark.parametrize("bad", ["version", "layout", "variant", "steps", "hash"])
def test_read_only_validation_rejects_changed_trace_identity(tmp_path, bad):
    obs, raw, _, info, identity = sample()
    trace = PriorityTrace(tmp_path / "trace", identity)
    trace.append("2020-01-01", obs, raw, info)
    manifest = trace.close()
    if bad == "version":
        manifest["schema_version"] = 1
    elif bad == "layout":
        manifest["action_layout"] = "2N"
    elif bad == "hash":
        manifest["chunks"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        verify_priority_trace(tmp_path / "trace", manifest,
            actor_variant="decoupled" if bad == "variant" else "coupled", steps=2 if bad == "steps" else 1)


def test_trace_does_not_mutate_order_diagnostics_or_info(tmp_path):
    obs, _, order, info, identity = sample()
    before = deepcopy(order.diagnostics)
    trace = PriorityTrace(tmp_path / "trace", identity)
    trace.append("2020-01-01", obs, order, info)
    trace.close()
    assert set(before) == set(order.diagnostics)
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, order.diagnostics[key])
        else:
            assert value == order.diagnostics[key]
    assert "raw_priority_latent" not in info["action_diagnostics"]
