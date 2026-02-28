"""torch.profiler setup and operator profile extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gpunity.types import OperatorProfile


def setup_torch_profiler(warmup: int, active: int, artifact_dir: Path) -> dict[str, Any]:
    """Return a configuration dict for torch.profiler.

    This dict is used by the wrapper script to set up profiling.

    Args:
        warmup: Number of warmup steps.
        active: Number of active profiling steps.
        artifact_dir: Directory where trace artifacts will be saved.

    Returns:
        Configuration dictionary with profiler parameters.
    """
    return {
        "wait": 0,
        "warmup": warmup,
        "active": active,
        "repeat": 1,
        "record_shapes": True,
        "with_stack": True,
        "with_flops": True,
        "trace_dir": str(artifact_dir),
    }


def extract_operator_profiles(trace_path: Path) -> list[OperatorProfile]:
    """Parse a Chrome trace JSON or key_averages JSON to extract operator profiles.

    Args:
        trace_path: Path to the trace JSON file.

    Returns:
        List of OperatorProfile sorted by GPU time descending.
    """
    if not trace_path.exists():
        return []

    try:
        with open(trace_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    # If this is a list of operator dicts (from profile_result.json)
    if isinstance(data, list):
        return [
            OperatorProfile(
                name=op.get("name", "unknown"),
                gpu_time_ms=op.get("gpu_time_ms", 0.0),
                cpu_time_ms=op.get("cpu_time_ms", 0.0),
                pct_total=op.get("pct_total", 0.0),
                call_count=op.get("call_count", 0),
                flops=op.get("flops"),
            )
            for op in data
        ]

    # If this is a Chrome trace format, extract from traceEvents
    operators: dict[str, dict[str, Any]] = {}
    events = data.get("traceEvents", [])
    for event in events:
        if event.get("cat") == "kernel" or event.get("cat") == "cuda_runtime":
            name = event.get("name", "unknown")
            dur = event.get("dur", 0) / 1000.0  # us -> ms
            if name not in operators:
                operators[name] = {
                    "gpu_time_ms": 0.0,
                    "cpu_time_ms": 0.0,
                    "call_count": 0,
                }
            operators[name]["gpu_time_ms"] += dur
            operators[name]["call_count"] += 1

    total_gpu = sum(op["gpu_time_ms"] for op in operators.values())

    profiles = []
    for name, stats in operators.items():
        profiles.append(OperatorProfile(
            name=name,
            gpu_time_ms=stats["gpu_time_ms"],
            cpu_time_ms=stats["cpu_time_ms"],
            pct_total=(stats["gpu_time_ms"] / total_gpu * 100) if total_gpu > 0 else 0,
            call_count=stats["call_count"],
        ))

    profiles.sort(key=lambda p: p.gpu_time_ms, reverse=True)
    return profiles[:20]
