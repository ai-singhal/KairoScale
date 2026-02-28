"""Autograd profiler extraction for forward/backward time split."""

from __future__ import annotations

import json
from pathlib import Path

from gpunity.types import BackwardOpProfile


def extract_autograd_profile(
    profile_path: Path,
) -> tuple[float, float, list[BackwardOpProfile]]:
    """Extract forward/backward timing from autograd profile data.

    Args:
        profile_path: Path to the autograd profile JSON file.

    Returns:
        Tuple of (forward_time_ms, backward_time_ms, backward_ops).
    """
    if not profile_path.exists():
        return 0.0, 0.0, []

    try:
        with open(profile_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return 0.0, 0.0, []

    forward_ms = data.get("forward_time_ms", 0.0)
    backward_ms = data.get("backward_time_ms", 0.0)

    backward_ops_raw = data.get("backward_ops", [])
    backward_ops = [
        BackwardOpProfile(
            name=op.get("name", "unknown"),
            time_ms=op.get("time_ms", 0.0),
            pct_backward=op.get("pct_backward", 0.0),
        )
        for op in backward_ops_raw
    ]

    return forward_ms, backward_ms, backward_ops
