"""GPU cost estimation utilities."""

from __future__ import annotations

# Approximate Modal GPU costs per hour (USD)
GPU_COSTS_PER_HOUR: dict[str, float] = {
    "a100-80gb": 3.73,
    "a100-40gb": 2.78,
    "h100": 4.98,
    "a10g": 1.10,
    "t4": 0.59,
}


def estimate_cost(gpu_type: str, seconds: float) -> float:
    """Estimate the USD cost for a GPU run.

    Args:
        gpu_type: GPU type identifier (e.g., 'a100-80gb').
        seconds: Wall-clock seconds of GPU usage.

    Returns:
        Estimated cost in USD.
    """
    hourly_rate = GPU_COSTS_PER_HOUR.get(gpu_type, 3.00)
    return hourly_rate * (seconds / 3600.0)
