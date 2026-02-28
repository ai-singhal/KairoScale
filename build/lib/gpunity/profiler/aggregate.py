"""Aggregate all profiler outputs into a single ProfileResult."""

from __future__ import annotations

import json
from pathlib import Path

from gpunity.profiler.autograd import extract_autograd_profile
from gpunity.profiler.dataloader import extract_dataloader_stats
from gpunity.profiler.memory import extract_memory_profile
from gpunity.profiler.torch_profiler import extract_operator_profiles
from gpunity.types import LoopDetectionMethod, ProfileResult


def aggregate_profile(artifact_dir: Path) -> ProfileResult:
    """Read all profiler artifacts from a directory and build a ProfileResult.

    Expected files in artifact_dir:
    - profile_result.json (main profile data from wrapper)
    - chrome_trace.json (optional, torch.profiler trace)
    - memory_snapshot.json (optional, memory data)
    - autograd_profile.json (optional, fwd/bwd split)
    - dataloader_stats.json (optional, DataLoader metrics)

    Args:
        artifact_dir: Path to the directory containing profiler artifacts.

    Returns:
        Aggregated ProfileResult.
    """
    artifact_dir = Path(artifact_dir)

    # Try loading the pre-built profile_result.json first (written by wrapper)
    profile_json = artifact_dir / "profile_result.json"
    if profile_json.exists():
        try:
            with open(profile_json) as f:
                data = json.load(f)
            return ProfileResult.from_dict(data)
        except Exception:
            pass  # Fall through to manual aggregation

    # Manual aggregation from individual artifact files
    result = ProfileResult(artifact_dir=str(artifact_dir))

    # Operator profiles from Chrome trace
    chrome_trace = artifact_dir / "chrome_trace.json"
    if chrome_trace.exists():
        result.top_operators = extract_operator_profiles(chrome_trace)
        result.chrome_trace_path = str(chrome_trace)
        if result.top_operators:
            total_gpu = sum(op.gpu_time_ms for op in result.top_operators)
            # Rough GPU utilization -- needs step time context for accuracy
            result.gpu_utilization = min(total_gpu / max(total_gpu, 1) * 100, 100.0)

    # Memory
    mem_snapshot = artifact_dir / "memory_snapshot.json"
    if mem_snapshot.exists():
        peak, timeline, stack = extract_memory_profile(mem_snapshot)
        result.peak_memory_mb = peak
        result.memory_timeline = timeline
        result.peak_allocation_stack = stack

    # Autograd
    autograd_path = artifact_dir / "autograd_profile.json"
    if autograd_path.exists():
        fwd, bwd, bwd_ops = extract_autograd_profile(autograd_path)
        result.forward_time_ms = fwd
        result.backward_time_ms = bwd
        result.backward_ops = bwd_ops

    # DataLoader
    dl_stats = artifact_dir / "dataloader_stats.json"
    if dl_stats.exists():
        throughput, stall, bottleneck = extract_dataloader_stats(dl_stats)
        result.dataloader_throughput = throughput
        result.dataloader_stall_time_ms = stall
        result.dataloader_bottleneck = bottleneck

    return result
