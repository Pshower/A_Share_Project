"""Independent synthetic phase-policy guards; no market data or financial steps."""

from copy import deepcopy
import json
import zipfile

import gymnasium as gym
import numpy as np
import pytest
import torch

from src.agents.timing_agent import InferenceOnlyTimingPPO
from src.agents.timing_policy import CalendarTimingPolicy, TIMING_NET_ARCH
from src.data.preprocess import file_hash, write_json
from tests.archive.test_timing_phase_agent import fixture
from tools.archive.timing_phase_agent import PhaseTimingAgent
from tools.archive.timing_phase_policy import DuePhaseTimingPolicy, mapped_baseline_state_dict


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_policy(kind, seed=123):
    torch.manual_seed(seed)
    return kind(gym.spaces.Box(-np.inf, np.inf, (129,), np.float32),
                gym.spaces.Discrete(2), lambda _: 3e-4)


def phase_observations():
    generator = torch.Generator().manual_seed(2026)
    context = torch.randn((20, 109), generator=generator)
    return torch.cat((torch.eye(20), context), dim=1)


def rewrite_archive(folder, mutate):
    path = folder / "model.zip"
    with zipfile.ZipFile(path) as zipped:
        content = {name: zipped.read(name) for name in zipped.namelist()}
    archived = json.loads(content["data"])
    mutate(archived)
    content["data"] = json.dumps(archived).encode("utf-8")
    with zipfile.ZipFile(path, "w") as zipped:
        for name, value in content.items():
            zipped.writestr(name, value)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    metadata["model_sha256"] = file_hash(path)
    write_json(folder / "metadata.json", metadata)


def test_initial_full_baseline_and_torch_rng_match_for_another_seed():
    baseline = make_policy(CalendarTimingPolicy)
    state = torch.get_rng_state().clone()
    residual = make_policy(DuePhaseTimingPolicy)
    assert torch.equal(state, torch.get_rng_state())
    left, right = mapped_baseline_state_dict(baseline), mapped_baseline_state_dict(residual)
    assert set(left) == set(right)
    assert all(torch.equal(left[name], right[name]) for name in left)
    vector = phase_observations()
    assert torch.equal(baseline.get_distribution(vector).distribution.probs,
                       residual.get_distribution(vector).distribution.probs)
    assert torch.equal(baseline.predict_values(vector), residual.predict_values(vector))


def test_context_residual_is_due_gated_and_all_public_actor_routes_agree():
    policy = make_policy(DuePhaseTimingPolicy)
    vector = phase_observations()
    with torch.no_grad():
        policy.action_net.residual.weight.copy_(torch.linspace(-.2, .3, 109).reshape(1, -1))
    difference = policy.action_net(vector) - policy.action_net.base(vector)
    assert torch.count_nonzero(difference[:, 0]) == 0
    assert torch.count_nonzero(difference[1:]) == 0
    assert difference[0, 1] != 0
    action, value, log_prob = policy(vector, deterministic=True)
    evaluated_value, evaluated_log_prob, entropy = policy.evaluate_actions(vector, action)
    distribution = policy.get_distribution(vector)
    assert torch.equal(value, evaluated_value)
    assert torch.equal(log_prob, evaluated_log_prob)
    assert torch.equal(log_prob, distribution.log_prob(action))
    assert torch.equal(entropy, distribution.entropy())


def test_one_synthetic_adam_step_includes_all_nine_actor_and_critic_tensors():
    policy = make_policy(DuePhaseTimingPolicy)
    parameters = list(policy.parameters())
    registered = [value for group in policy.optimizer.param_groups for value in group["params"]]
    assert len(parameters) == len(registered) == 9
    assert len({id(value) for value in registered}) == 9
    assert {id(value) for value in registered} == {id(value) for value in parameters}
    vector = phase_observations()
    value, log_prob, _ = policy.evaluate_actions(vector, torch.zeros(20, dtype=torch.long))
    loss = -log_prob.mean() + (value - .25).square().mean()
    policy.optimizer.zero_grad()
    loss.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in parameters)
    policy.optimizer.step()
    assert len(policy.optimizer.state) == 9
    assert all(int(value["step"]) == 1 for value in policy.optimizer.state.values())
    assert torch.count_nonzero(policy.action_net.residual.weight) > 0


@pytest.mark.parametrize("architecture", ["linear", "due_phase_residual"])
def test_loader_receives_only_explicit_whitelisted_constructor_objects(tmp_path, monkeypatch, architecture):
    model, contract, env = fixture(architecture)
    try:
        folder = tmp_path / "artifact"
        PhaseTimingAgent(model, contract).save(folder, {})
        captured = []

        def capture(_path, **kwargs):
            captured.append(kwargs["custom_objects"])
            return model

        monkeypatch.setattr(InferenceOnlyTimingPPO, "load", capture)
        restored = PhaseTimingAgent.load(folder, contract)
        assert restored.provenance["inference_only"] is True
        assert len(captured) == 1
        actual = captured[0]
        assert set(actual) == {"policy_class", "policy_kwargs", "observation_space", "action_space"}
        assert actual["policy_class"] is (CalendarTimingPolicy if architecture == "linear" else DuePhaseTimingPolicy)
        assert actual["policy_kwargs"] == dict(calendar_margin=2., net_arch=deepcopy(TIMING_NET_ARCH))
        assert actual["observation_space"].shape == (129,)
        assert actual["action_space"].n == 2 and actual["action_space"].start == 0
    finally:
        env.close()


@pytest.mark.parametrize("location", ["metadata_contract", "archive_contract", "archive_policy_record"])
def test_boolean_zero_alias_is_rejected_before_loading_any_policy(tmp_path, monkeypatch, location):
    model, contract, env = fixture()
    try:
        folder = tmp_path / "artifact"
        PhaseTimingAgent(model, contract).save(folder, {})
        if location == "metadata_contract":
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            metadata["contract"]["phase_policy"]["due_phase"] = False
            write_json(folder / "metadata.json", metadata)
        elif location == "archive_contract":
            rewrite_archive(folder, lambda data: data["phase_registered_contract"]["phase_policy"].update(due_phase=False))
        else:
            rewrite_archive(folder, lambda data: data["phase_policy_record"].update(due_phase=False))

        def forbidden_loader(*_args, **_kwargs):
            raise AssertionError("Typed provenance must be checked before deserialization")

        monkeypatch.setattr(InferenceOnlyTimingPPO, "load", forbidden_loader)
        with pytest.raises(ValueError):
            PhaseTimingAgent.load(folder, contract)
    finally:
        env.close()
