"""DataLoader throughput instrumentation and stats extraction."""

from __future__ import annotations

import json
from pathlib import Path


def extract_dataloader_stats(stats_path: Path) -> tuple[float, float, bool]:
    """Extract DataLoader performance stats from saved artifacts.

    Args:
        stats_path: Path to the dataloader stats JSON file.

    Returns:
        Tuple of (throughput_samples_sec, stall_time_ms, is_bottleneck).
    """
    if not stats_path.exists():
        return 0.0, 0.0, False

    try:
        with open(stats_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return 0.0, 0.0, False

    throughput = data.get("throughput", 0.0)
    stall_time_ms = data.get("stall_time_ms", 0.0)
    is_bottleneck = data.get("is_bottleneck", False)

    return throughput, stall_time_ms, is_bottleneck
