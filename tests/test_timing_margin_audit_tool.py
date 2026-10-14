"""Static synthetic archive tests; no model instance, financial step or optimizer."""

import io
import json
from pathlib import Path
import zipfile

import pytest
import torch

from tools import audit_timing_margin as audit


def archive(tmp_path, margin, *, changed=None, optimizer_state=None):
    theta = {name: torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape) / 100000
             for name, shape in audit.SHAPES.items()}
    theta["action_net.weight"].zero_()
    theta["action_net.weight"][0, 1:20] = margin
    theta["action_net.weight"][1, 0] = margin
    theta["action_net.bias"].zero_()
    if changed:
        theta[changed].reshape(-1)[-1] += 1
    path = Path(tmp_path) / "model.zip"
    policy, optimizer = io.BytesIO(), io.BytesIO()
    torch.save(theta, policy)
    torch.save(dict(state={} if optimizer_state is None else optimizer_state,
                    param_groups=[dict(params=list(range(8)))]), optimizer)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("policy.pth", policy.getvalue())
        zipped.writestr("policy.optimizer.pth", optimizer.getvalue())
        zipped.writestr("data", json.dumps(dict(num_timesteps=0, timing_registered_contract={"x": 1})))
    return dict(num_timesteps=0, actual_device="cpu", contract={"x": 1}), theta


@pytest.mark.parametrize("margin", [.5, 1., 2.])
def test_registered_calendar_static_archive(tmp_path, margin):
    metadata, theta = archive(tmp_path, margin)
    actual, result = audit.initial_model_audit(tmp_path, metadata, margin)
    assert result["parameters"] == 12805 and result["active_adam_tensors"] == 0
    assert result["calendar_margin"] == margin
    assert all(torch.equal(actual[key], value) for key, value in theta.items())


def test_wrong_calendar_margin_rejected(tmp_path):
    metadata, _ = archive(tmp_path, 2.)
    with pytest.raises(ValueError, match="registered trainable margin"):
        audit.initial_model_audit(tmp_path, metadata, .5)


def test_initial_nonphase_actor_coordinate_rejected(tmp_path):
    metadata, _ = archive(tmp_path, .5, changed="action_net.weight")
    with pytest.raises(ValueError, match="registered trainable margin"):
        audit.initial_model_audit(tmp_path, metadata, .5)


def test_initial_optimizer_state_rejected(tmp_path):
    metadata, _ = archive(tmp_path, 1., optimizer_state={0: {"step": torch.tensor(1.)}})
    with pytest.raises(ValueError, match="already has Adam"):
        audit.initial_model_audit(tmp_path, metadata, 1.)


def test_shared_critic_not_actor_calendar(tmp_path):
    _, first = archive(tmp_path, .5)
    _, second = archive(tmp_path, 2.)
    result = audit.check_shared_initials({("2023", "screen", 42, .5): first,
                                         ("2023", "screen", 42, 2.): second})
    assert result["checked_cells"] == 2 and result["groups"] == 1
    second["value_net.bias"][0] += 1
    with pytest.raises(ValueError, match="Critic"):
        audit.check_shared_initials({("2023", "screen", 42, .5): first,
                                    ("2023", "screen", 42, 2.): second})


def test_final_requires_sealed_before_any_io():
    with pytest.raises(ValueError, match="requires sealed"):
        audit.main(["--final", "--root", "absent_synthetic"])
