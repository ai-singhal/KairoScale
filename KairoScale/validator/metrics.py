"""Compute validation metrics from artifact data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from KairoScale.types import ControlRun, ValidationResult
from KairoScale.utils.cost import estimate_cost


def load_run_metrics(artifact_dir: Path) -> dict[str, Any]:
    """Load metrics from a validation run's artifact directory.

    Args:
        artifact_dir: Path to artifacts directory.

    Returns:
        Dict with metrics data.
    """
    metrics_path = Path(artifact_dir) / "metrics.json"
    if not metrics_path.exists():
        return {}

    try:
        with open(metrics_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def build_control_run(
    artifact_dir: Path,
    gpu_type: str = "a100-80gb",
) -> ControlRun:
    """Build a ControlRun from validation artifacts.

    Args:
        artifact_dir: Path to control run artifacts.
        gpu_type: GPU type for cost estimation.

    Returns:
        ControlRun instance.
    """
    data = load_run_metrics(artifact_dir)

    wall_clock = data.get("wall_clock_seconds", 0.0)
    cost = estimate_cost(gpu_type, wall_clock)

    return ControlRun(
        steps_completed=data.get("steps_completed", 0),
        wall_clock_seconds=wall_clock,
        avg_step_time_ms=data.get("avg_step_time_ms", 0.0),
        peak_memory_mb=data.get("peak_memory_mb", 0.0),
        throughput_samples_sec=data.get("throughput_samples_sec", 0.0),
        loss_values=data.get("loss_values", []),
        gradient_norms=data.get("gradient_norms", []),
        cost_estimate_usd=cost,
    )


def compute_validation_metrics(
    artifact_dir: Path,
    control: ControlRun,
    config_id: str = "",
    config_name: str = "",
    gpu_type: str = "a100-80gb",
) -> ValidationResult:
    """Compute performance deltas between a variant and the control run.

    Args:
        artifact_dir: Path to variant run artifacts.
        control: The control run results.
        config_id: Optimization config ID.
        config_name: Optimization config name.
        gpu_type: GPU type for cost estimation.

    Returns:
        ValidationResult with computed deltas.
    """
    data = load_run_metrics(artifact_dir)

    if not data:
        return ValidationResult(
            config_id=config_id,
            config_name=config_name,
            success=False,
            error="No metrics found in artifact directory.",
        )

    wall_clock = data.get("wall_clock_seconds", 0.0)
    avg_step_ms = data.get("avg_step_time_ms", 0.0)
    steps_completed = data.get("steps_completed", 0)
    peak_mem = data.get("peak_memory_mb", 0.0)
    throughput = data.get("throughput_samples_sec", 0.0)
    loss_values = data.get("loss_values", [])
    cost = estimate_cost(gpu_type, wall_clock)
    runtime_error = data.get("runtime_error")

    if runtime_error:
        return ValidationResult(
            config_id=config_id,
            config_name=config_name,
            success=False,
            error=f"Runtime error: {runtime_error}",
        )

    if steps_completed <= 0:
        return ValidationResult(
            config_id=config_id,
            config_name=config_name,
            success=False,
            error="Run completed zero optimizer steps.",
        )

    # Compute deltas vs control
    speedup = (control.avg_step_time_ms / avg_step_ms) if avg_step_ms > 0 else 1.0
    mem_delta = (
        (peak_mem - control.peak_memory_mb) / control.peak_memory_mb
        if control.peak_memory_mb > 0
        else 0.0
    )
    cost_delta = (
        (cost - control.cost_estimate_usd) / control.cost_estimate_usd
        if control.cost_estimate_usd > 0
        else 0.0
    )
    throughput_gain = (
        (throughput - control.throughput_samples_sec) / control.throughput_samples_sec
        if control.throughput_samples_sec > 0
        else 0.0
    )

    return ValidationResult(
        config_id=config_id,
        config_name=config_name,
        success=True,
        speedup_vs_control=speedup,
        throughput_gain_vs_control=throughput_gain,
        memory_delta_vs_control=mem_delta,
        cost_delta_vs_control=cost_delta,
        wall_clock_seconds=wall_clock,
        avg_step_time_ms=avg_step_ms,
        peak_memory_mb=peak_mem,
        throughput_samples_sec=throughput,
        loss_values=loss_values,
    )
