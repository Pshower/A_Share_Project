"""Explicit device selection without silent fallback for requested CUDA."""

import torch


def resolve_device(requested="auto"):
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu or cuda")
    if requested == "cpu":
        return "cpu"
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise ValueError("CUDA was requested but is unavailable in this Python environment")
    return "cuda" if available else "cpu"


def device_status():
    available = torch.cuda.is_available()
    return dict(device="cuda" if available else "cpu", cuda_available=available,
                cuda_build=torch.version.cuda,
                gpu_name=torch.cuda.get_device_name(0) if available else None)
