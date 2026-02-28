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
from gpunity.validator.divergence import (
    check_divergence,
    compare_logit_signatures,
    compute_cosine_similarities,
)
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
        validation_seed=run_config.validation_seed,
        deterministic_validation=run_config.deterministic_validation,
    )

    if run_config.local:
        from gpunity.sandbox.local_runner import run_locally
        artifact_dir = await run_locally(
            repo_path=repo_path,
            script_content=wrapper,
            timeout_seconds=600,
            python_bin=run_config.python_bin,
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


def _risk_weight(config: OptimizationConfig) -> float:
    """Simple risk weighting for staged shortlist ranking."""
    risk = config.risk_level.value
    if risk == "low":
        return 1.0
    if risk == "medium":
        return 1.4
    return 2.0


async def _run_validation_batch(
    control_repo: Path,
    configs: list[OptimizationConfig],
    run_config: RunConfig,
    steps: int,
    batch_label: str,
) -> tuple[ControlRun, list[ValidationResult]]:
    """Run one validation batch (control + variants) and compute results."""
    control_repo = Path(control_repo).resolve()
    grad_interval = run_config.gradient_check_interval

    # Prepare patched repos for each config
    patched_repos: list[tuple[OptimizationConfig, Path]] = []
    patch_failures: list[ValidationResult] = []
    for config in configs:
        try:
            patched = apply_config(control_repo, config)
            patched_repos.append((config, patched))
        except Exception as e:
            logger.error(f"[{batch_label}] Failed to patch repo for {config.id}: {e}")
            patch_failures.append(ValidationResult(
                config_id=config.id,
                config_name=config.name,
                success=False,
                error=f"Failed to apply config: {e}",
            ))

    # Launch all runs in parallel
    tasks = []
    tasks.append(
        _run_single(
            control_repo, run_config.entry_point, steps,
            grad_interval, run_config, f"{batch_label}-control"
        )
    )
    for config, patched_repo in patched_repos:
        tasks.append(
            _run_single(
                patched_repo, run_config.entry_point, steps,
                grad_interval, run_config, f"{batch_label}-{config.id}"
            )
        )

    logger.info(
        f"[{batch_label}] Launching {len(tasks)} validation runs "
        f"(1 control + {len(patched_repos)} variants)"
    )
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process control run
    control_artifact_dir = results[0] if not isinstance(results[0], Exception) else None

    if control_artifact_dir is None:
        logger.error(f"[{batch_label}] Control run failed: {results[0]}")
        # Return empty control and failed results
        control = ControlRun(
            steps_completed=0, wall_clock_seconds=0, avg_step_time_ms=0,
            peak_memory_mb=0, throughput_samples_sec=0, loss_values=[],
            gradient_norms=[], cost_estimate_usd=0,
        )
        batch_failures = [
            ValidationResult(
                config_id=c.id, config_name=c.name, success=False,
                error="Control run failed"
            )
            for c, _ in patched_repos
        ]
        return control, patch_failures + batch_failures

    control = build_control_run(control_artifact_dir, run_config.gpu_type)
    if control.steps_completed <= 0:
        logger.error(f"[{batch_label}] Control run completed zero steps.")
        control_failed_results = [
            ValidationResult(
                config_id=c.id,
                config_name=c.name,
                success=False,
                error="Control run completed zero optimizer steps.",
            )
            for c, _ in patched_repos
        ]
        return control, patch_failures + control_failed_results

    control_checkpoints = load_gradient_checkpoints(control_artifact_dir)

    # Process variant runs
    validation_results: list[ValidationResult] = list(patch_failures)
    for i, (config, _patched_repo) in enumerate(patched_repos):
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

        # Logit-level correctness check
        checks, max_diff, mean_diff, failing_step = compare_logit_signatures(
            control_checkpoints,
            variant_checkpoints,
            tolerance=run_config.logits_tolerance,
        )
        vr.logits_checks_compared = checks
        vr.logits_max_abs_diff = max_diff
        vr.logits_mean_abs_diff = mean_diff
        vr.logits_within_tolerance = (failing_step is None) if checks > 0 else None
        if checks > 0 and failing_step is not None:
            vr.diverged = True
            vr.divergence_step = failing_step
            vr.divergence_reason = (
                f"Logit signature delta {max_diff:.6f} exceeded tolerance "
                f"{run_config.logits_tolerance:.6f}"
            )

        validation_results.append(vr)
        logger.info(
            f"[{batch_label}:{config.id}] speedup={vr.speedup_vs_control:.2f}x, "
            f"mem_delta={vr.memory_delta_vs_control:+.1%}, "
            f"diverged={vr.diverged}, logits_checks={vr.logits_checks_compared}"
        )

    return control, validation_results


async def run_validation(
    control_repo: Path,
    configs: list[OptimizationConfig],
    run_config: RunConfig,
) -> tuple[ControlRun, list[ValidationResult]]:
    """Run validation with selected strategy."""
    if run_config.validation_strategy != "staged" or len(configs) <= 1:
        return await _run_validation_batch(
            control_repo=control_repo,
            configs=configs,
            run_config=run_config,
            steps=run_config.validation_steps,
            batch_label="full",
        )

    # Stage 1: quick screening across all variants.
    stage1_steps = max(5, min(run_config.validation_steps, run_config.validation_steps // 3))
    logger.info(
        f"[staged] Stage 1 quick screening: {len(configs)} configs for {stage1_steps} steps"
    )
    _stage1_control, stage1_results = await _run_validation_batch(
        control_repo=control_repo,
        configs=configs,
        run_config=run_config,
        steps=stage1_steps,
        batch_label="stage1",
    )

    # Promote top configs by observed speedup and stability.
    stage1_by_id = {r.config_id: r for r in stage1_results}
    promotable: list[tuple[OptimizationConfig, float]] = []
    for cfg in configs:
        r = stage1_by_id.get(cfg.id)
        if r is None or not r.success or r.diverged:
            continue
        score = r.speedup_vs_control / _risk_weight(cfg)
        promotable.append((cfg, score))

    promotable.sort(key=lambda x: x[1], reverse=True)
    promoted = [cfg for cfg, _ in promotable[: max(1, run_config.staged_top_k)]]
    if not promoted:
        logger.warning("[staged] No configs passed stage-1; returning stage-1 results.")
        return _stage1_control, stage1_results

    logger.info(
        f"[staged] Stage 2 full validation: promoting {len(promoted)} config(s): "
        f"{', '.join(c.id for c in promoted)}"
    )
    stage2_control, stage2_results = await _run_validation_batch(
        control_repo=control_repo,
        configs=promoted,
        run_config=run_config,
        steps=run_config.validation_steps,
        batch_label="stage2",
    )

    # Preserve visibility for configs not promoted to stage-2.
    stage2_ids = {r.config_id for r in stage2_results}
    for cfg in configs:
        if cfg.id in stage2_ids:
            continue
        stage2_results.append(ValidationResult(
            config_id=cfg.id,
            config_name=cfg.name,
            success=False,
            error=(
                "Skipped after stage-1 screening "
                "(did not make promoted shortlist for full validation)"
            ),
        ))

    return stage2_control, stage2_results
