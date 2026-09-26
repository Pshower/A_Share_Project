import pytest
import torch

from src.runtime.device import resolve_device, device_status
from src.web.schemas import Training


def test_device_selection_without_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_device("auto") == resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="unavailable"):
        resolve_device("cuda")
    with pytest.raises(ValueError, match="Device"):
        resolve_device("invalid")
    assert device_status()["gpu_name"] is None


def test_device_selection_with_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda _: "fixture GPU")
    assert resolve_device("auto") == resolve_device("cuda") == "cuda"
    assert resolve_device("cpu") == "cpu"
    assert device_status()["gpu_name"] == "fixture GPU"


def test_training_device_request():
    assert Training(dataset_id="fixture", codes=["000001"]).device == "auto"
    assert Training(dataset_id="fixture", codes=["000001"], device="cpu").device == "cpu"
    with pytest.raises(ValueError):
        Training(dataset_id="fixture", codes=["000001"], device="invalid")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_real_cuda_tensor_forward_and_backward():
    value = torch.randn(32, 32, device="cuda", requires_grad=True)
    loss = (value @ value.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert value.is_cuda and torch.isfinite(loss)
    assert value.grad is not None and torch.isfinite(value.grad).all()
