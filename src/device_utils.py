from __future__ import annotations

import platform

import torch


def resolve_device(requested_device: str) -> str:
    requested = requested_device.lower()

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested_device}")

    return requested


def synchronize_device(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def get_device_info(device: str) -> dict:
    info = {
        "device": device,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if device == "cuda":
        current_device = torch.cuda.current_device()
        info.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(current_device),
                "cuda_device_index": current_device,
                "cuda_version": torch.version.cuda,
            }
        )

    return info
