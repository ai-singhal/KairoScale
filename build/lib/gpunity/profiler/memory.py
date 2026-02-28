"""CUDA memory snapshot handling and extraction."""

from __future__ import annotations

import json
from pathlib import Path

from gpunity.types import MemoryTimelineEntry


def extract_memory_profile(
    snapshot_path: Path,
) -> tuple[float, list[MemoryTimelineEntry], str]:
    """Extract memory profiling data from a saved snapshot.

    Args:
        snapshot_path: Path to the memory snapshot JSON file.

    Returns:
        Tuple of (peak_memory_mb, memory_timeline, peak_allocation_stack).
    """
    if not snapshot_path.exists():
        return 0.0, [], ""

    try:
        with open(snapshot_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return 0.0, [], ""

    peak_mb = data.get("peak_memory_mb", 0.0)

    timeline_raw = data.get("memory_timeline", [])
    timeline = [
        MemoryTimelineEntry(
            step=entry.get("step", 0),
            allocated_mb=entry.get("allocated_mb", 0.0),
            reserved_mb=entry.get("reserved_mb", 0.0),
        )
        for entry in timeline_raw
    ]

    peak_stack = data.get("peak_allocation_stack", "")

    return peak_mb, timeline, peak_stack
