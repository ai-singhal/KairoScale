"""torch.profiler setup and operator profile extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from KairoScale.types import OperatorProfile


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

    # Build stackFrames lookup: id -> {"name": str, "parent": optional int/str}
    stack_frames: dict[str, dict[str, Any]] = {}
    raw_stack_frames = data.get("stackFrames", {})
    for frame_id, frame_data in raw_stack_frames.items():
        stack_frames[str(frame_id)] = frame_data

    def _is_user_frame(frame_name: str) -> bool:
        """Return True if the frame is user code (not torch internals)."""
        exclude = ("torch/", "site-packages/", "<frozen", "importlib")
        return not any(pat in frame_name for pat in exclude)

    def _resolve_stack(sf_id: str) -> list[str]:
        """Walk the stackFrames chain from sf_id upward, return user frames."""
        frames = []
        current = str(sf_id)
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            frame = stack_frames.get(current)
            if frame is None:
                break
            name = frame.get("name", "")
            if name and _is_user_frame(name):
                frames.append(name)
            parent = frame.get("parent")
            current = str(parent) if parent is not None else ""
        return frames[:3]

    # Collect python_function events for temporal stack association
    # Each entry: {"ts": int, "dur": int, "tid": int, "sf": str, "name": str}
    python_events: list[dict[str, Any]] = []
    for event in events:
        if event.get("cat") == "python_function":
            ts = event.get("ts", 0)
            dur = event.get("dur", 0)
            tid = event.get("tid", 0)
            sf = event.get("sf") or event.get("args", {}).get("python_id")
            if sf is not None:
                python_events.append({
                    "ts": ts,
                    "dur": dur,
                    "tid": tid,
                    "sf": str(sf),
                    "name": event.get("name", ""),
                })

    def _find_stack_for_kernel(kernel_ts: int, kernel_dur: int, kernel_tid: int) -> list[str]:
        """Find user stack frames from python_function events overlapping this kernel."""
        kernel_end = kernel_ts + kernel_dur
        for pev in python_events:
            # Same thread or overlapping time window
            pev_end = pev["ts"] + pev["dur"]
            if pev["ts"] <= kernel_ts and pev_end >= kernel_end:
                frames = _resolve_stack(pev["sf"])
                if frames:
                    return frames
        return []

    for event in events:
        if event.get("cat") == "kernel" or event.get("cat") == "cuda_runtime":
            name = event.get("name", "unknown")
            dur = event.get("dur", 0) / 1000.0  # us -> ms
            if name not in operators:
                operators[name] = {
                    "gpu_time_ms": 0.0,
                    "cpu_time_ms": 0.0,
                    "call_count": 0,
                    "source_stack": [],
                    "_ts": event.get("ts", 0),
                    "_dur_us": event.get("dur", 0),
                    "_tid": event.get("tid", 0),
                }
            operators[name]["gpu_time_ms"] += dur
            operators[name]["call_count"] += 1

    # Extract stack frames for each operator (use first occurrence for stack lookup)
    for name, stats in operators.items():
        if not stats["source_stack"] and stack_frames:
            frames = _find_stack_for_kernel(
                stats["_ts"], stats["_dur_us"], stats["_tid"]
            )
            stats["source_stack"] = frames

    total_gpu = sum(op["gpu_time_ms"] for op in operators.values())

    profiles = []
    for name, stats in operators.items():
        profiles.append(OperatorProfile(
            name=name,
            gpu_time_ms=stats["gpu_time_ms"],
            cpu_time_ms=stats["cpu_time_ms"],
            pct_total=(stats["gpu_time_ms"] / total_gpu * 100) if total_gpu > 0 else 0,
            call_count=stats["call_count"],
            source_stack=stats.get("source_stack", []),
        ))

    profiles.sort(key=lambda p: p.gpu_time_ms, reverse=True)
    return profiles[:20]
