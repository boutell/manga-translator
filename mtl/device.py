"""Runtime device detection.

torch exposes ROCm (AMD) through the same `torch.cuda` API as NVIDIA, so an
AMD GPU shows up as a CUDA device when the ROCm torch build can enumerate it.
If the integrated GPU isn't enumerated, set HSA_OVERRIDE_GFX_VERSION in your
environment before running (see README) — otherwise we fall back to CPU, which
is fine for a batch job, just slower.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def get_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


@functools.lru_cache(maxsize=1)
def describe_device() -> str:
    dev = get_device()
    if dev == "cpu":
        return "CPU"
    try:
        import torch

        return f"GPU ({torch.cuda.get_device_name(0)})"
    except Exception:
        return "GPU"
