"""Validation runner -- orchestrates parallel control + variant runs.

Runs the unmodified control alongside each optimized variant, collects
metrics, and computes divergence for each variant.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from gpunity.types import (
    ControlRun,
    OptimizationConfig,
    RunConfig,
    ValidationResult,
)
from gpunity.utils.logging import get_logger
from gpunity.validator.divergence import check_divergence, compute_cosine_similarities
from gpunity.validator.gradient_tracker import (
    create_gradient_tracking_wrapper,
    load_gradient_checkpoints,
)
from gpunity.validator.metrics import build_control_run, compute_validation_metrics
from gpunity.validator.patcher import apply_config

logger = get_logger("gpunity.validator.runner")


async def _run_single(
    repo_path: Path,
    entry_point: str,
    steps: int,
    gradient_check_interval: int,
    run_config: RunConfig,
    label: str,
) -> Path:
    """Run a single validation (control or variant) and return artifacts dir."""
    wrapper = create_gradient_tracking_wrapper(
        entry_point=entry_point,
        steps=steps,
        gradient_check_interval=gradient_check_interval,
        artifact_dir=Path("/tmp/gpunity_val"),
    )

    if run_config.local:
        from gpunity.sandbox.local_runner import run_locally
        artifact_dir = await run_locally(
            repo_path=repo_path,
            script_content=wrapper,
            timeout_seconds=600,
        )
    else:
        from gpunity.sandbox.modal_runner import run_in_modal
        # Detect extra deps from patched repo
        extra_reqs_path = repo_path / ".gpunity_extra_requirements.txt"
        extra_deps = None
        if extra_reqs_path.exists():
            extra_deps = [
                line.strip()
                for line in extra_reqs_path.read_text().splitlines()
                if line.strip()
            ]
        artifact_dir = await run_in_modal(
            repo_path=repo_path,
            script_content=wrapper,
            gpu_type=run_config.gpu_type,
            timeout_seconds=600,
            cost_ceiling_usd=run_config.max_cost_per_sandbox,
            extra_deps=extra_deps,
        )

    logger.info(f"[{label}] Run complete. Artifacts: {artifact_dir}")
    return artifact_dir


async def run_validation(
    control_repo: Path,
    configs: list[OptimizationConfig],
    run_config: RunConfig,
) -> tuple[ControlRun, list[ValidationResult]]:
    """Run control + all config variants and compute results.

    Launches all runs in parallel (control + N variants), then computes
    performance deltas and divergence for each variant vs control.

    Args:
        control_repo: Path to the unmodified repository.
        configs: List of optimization configs to validate.
        run_config: Run configuration.

    Returns:
        Tuple of (ControlRun, list of ValidationResult).
    """
    control_repo = Path(control_repo).resolve()
    steps = run_config.validation_steps
    grad_interval = run_config.gradient_check_interval

    # Prepare patched repos for each config
    patched_repos: list[tuple[OptimizationConfig, Path]] = []
    for config in configs:
        try:
            patched = apply_config(control_repo, config)
            patched_repos.append((config, patched))
        except Exception as e:
            logger.error(f"Failed to patch repo for {config.id}: {e}")

    # Launch all runs in parallel
    tasks = []

    # Control run
    tasks.append(
        _run_single(
            control_repo, run_config.entry_point, steps,
            grad_interval, run_config, "control"
        )
    )

    # Variant runs
    for config, patched_repo in patched_repos:
        tasks.append(
            _run_single(
                patched_repo, run_config.entry_point, steps,
                grad_interval, run_config, config.id
            )
        )

    # Execute all runs
    logger.info(f"Launching {len(tasks)} validation runs (1 control + {len(patched_repos)} variants)")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process control run
    control_artifact_dir = results[0] if not isinstance(results[0], Exception) else None

    if control_artifact_dir is None:
        logger.error(f"Control run failed: {results[0]}")
        # Return empty control and failed results
        control = ControlRun(
            steps_completed=0, wall_clock_seconds=0, avg_step_time_ms=0,
            peak_memory_mb=0, throughput_samples_sec=0, loss_values=[],
            gradient_norms=[], cost_estimate_usd=0,
        )
        validation_results = [
            ValidationResult(
                config_id=c.id, config_name=c.name, success=False,
                error="Control run failed"
            )
            for c, _ in patched_repos
        ]
        return control, validation_results

    control = build_control_run(control_artifact_dir, run_config.gpu_type)
    control_checkpoints = load_gradient_checkpoints(control_artifact_dir)

    # Process variant runs
    validation_results: list[ValidationResult] = []
    for i, (config, patched_repo) in enumerate(patched_repos):
        variant_result = results[i + 1]  # +1 because control is at index 0

        if isinstance(variant_result, Exception):
            validation_results.append(ValidationResult(
                config_id=config.id,
                config_name=config.name,
                success=False,
                error=str(variant_result),
            ))
            continue

        variant_artifact_dir = variant_result

        # Compute metrics
        vr = compute_validation_metrics(
            variant_artifact_dir, control,
            config_id=config.id,
            config_name=config.name,
            gpu_type=run_config.gpu_type,
        )

        # Check divergence
        variant_checkpoints = load_gradient_checkpoints(variant_artifact_dir)
        diverged, div_step, div_reason = check_divergence(
            control_checkpoints,
            variant_checkpoints,
            threshold=run_config.divergence_threshold,
        )

        vr.diverged = diverged
        vr.divergence_step = div_step
        vr.divergence_reason = div_reason

        # Compute cosine similarities from loss curves
        vr.gradient_cosine_similarities = compute_cosine_similarities(
            control.loss_values,
            vr.loss_values,
        )

        validation_results.append(vr)
        logger.info(
            f"[{config.id}] speedup={vr.speedup_vs_control:.2f}x, "
            f"mem_delta={vr.memory_delta_vs_control:+.1%}, "
            f"diverged={vr.diverged}"
        )

    return control, validation_results
