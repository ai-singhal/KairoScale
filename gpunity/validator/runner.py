"""Validation runner -- orchestrates control + baseline + variant runs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from gpunity.optimizer.baselines import build_native_baseline_candidates
from gpunity.optimizer.objective import (
    apply_ablation_estimates,
    score_validation_results,
    summarize_run,
)
from gpunity.types import (
    BaselineResult,
    ControlRun,
    HardwareProfile,
    OptimizationConfig,
    ProfileResult,
    RunConfig,
    RunSummary,
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


def _empty_control_run() -> ControlRun:
    return ControlRun(
        steps_completed=0,
        wall_clock_seconds=0,
        avg_step_time_ms=0,
        peak_memory_mb=0,
        throughput_samples_sec=0,
        loss_values=[],
        gradient_norms=[],
        cost_estimate_usd=0,
    )


def _failure_result(config: OptimizationConfig, error: str) -> ValidationResult:
    return ValidationResult(
        config_id=config.id,
        config_name=config.name,
        success=False,
        error=error,
        is_native_baseline=config.is_native_baseline,
        baseline_id=config.baseline_id,
        evidence_chain=list(config.evidence[:3]),
    )


async def _run_single(
    repo_path: Path,
    entry_point: str,
    steps: int,
    gradient_check_interval: int,
    run_config: RunConfig,
    mode: str,
    label: str,
) -> Path:
    """Run a single validation (control or variant) and return artifacts dir."""
    wrapper = create_gradient_tracking_wrapper(
        entry_point=entry_point,
        steps=steps,
        gradient_check_interval=gradient_check_interval,
        artifact_dir=Path("/tmp/gpunity_val"),
        mode=mode,
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
    mode: str,
    steps: int,
    batch_label: str,
) -> tuple[ControlRun, list[ValidationResult]]:
    """Run one validation batch (control + variants) and compute results."""
    control_repo = Path(control_repo).resolve()
    grad_interval = run_config.gradient_check_interval

    eligible_configs = [c for c in configs if c.eligible]
    ineligible_results = [
        _failure_result(
            c,
            f"Ineligible candidate: {c.ineligible_reason or 'eligibility check failed'}",
        )
        for c in configs
        if not c.eligible
    ]

    patched_repos: list[tuple[OptimizationConfig, Path]] = []
    patch_failures: list[ValidationResult] = []
    for config in eligible_configs:
        try:
            patched = apply_config(control_repo, config)
            patched_repos.append((config, patched))
        except Exception as e:
            logger.error(f"[{batch_label}] Failed to patch repo for {config.id}: {e}")
            patch_failures.append(_failure_result(config, f"Failed to apply config: {e}"))

    tasks = [
        _run_single(
            control_repo,
            run_config.entry_point,
            steps,
            grad_interval,
            run_config,
            mode,
            f"{batch_label}-control",
        )
    ]
    for config, patched_repo in patched_repos:
        tasks.append(
            _run_single(
                patched_repo,
                run_config.entry_point,
                steps,
                grad_interval,
                run_config,
                mode,
                f"{batch_label}-{config.id}",
            )
        )

    logger.info(
        f"[{batch_label}] Launching {len(tasks)} validation runs "
        f"(1 control + {len(patched_repos)} variants)"
    )
    results = await asyncio.gather(*tasks, return_exceptions=True)

    control_artifact_dir = results[0] if not isinstance(results[0], Exception) else None
    if control_artifact_dir is None:
        logger.error(f"[{batch_label}] Control run failed: {results[0]}")
        batch_failures = [
            _failure_result(c, "Control run failed")
            for c, _ in patched_repos
        ]
        return _empty_control_run(), ineligible_results + patch_failures + batch_failures

    control = build_control_run(control_artifact_dir, run_config.gpu_type)
    if control.steps_completed <= 0:
        logger.error(f"[{batch_label}] Control run completed zero steps.")
        control_failed_results = [
            _failure_result(c, "Control run completed zero optimizer/model steps.")
            for c, _ in patched_repos
        ]
        return control, ineligible_results + patch_failures + control_failed_results

    control_checkpoints = load_gradient_checkpoints(control_artifact_dir)

    validation_results: list[ValidationResult] = ineligible_results + patch_failures
    for i, (config, _patched_repo) in enumerate(patched_repos):
        variant_result = results[i + 1]

        if isinstance(variant_result, Exception):
            validation_results.append(_failure_result(config, str(variant_result)))
            continue

        variant_artifact_dir = variant_result
        vr = compute_validation_metrics(
            variant_artifact_dir,
            control,
            config_id=config.id,
            config_name=config.name,
            gpu_type=run_config.gpu_type,
        )
        vr.is_native_baseline = config.is_native_baseline
        vr.baseline_id = config.baseline_id
        vr.evidence_chain = list(config.evidence[:3])

        if vr.success:
            variant_checkpoints = load_gradient_checkpoints(variant_artifact_dir)
            diverged, div_step, div_reason = check_divergence(
                control_checkpoints,
                variant_checkpoints,
                threshold=run_config.divergence_threshold,
            )
            vr.diverged = diverged
            vr.divergence_step = div_step
            vr.divergence_reason = div_reason
            vr.gradient_cosine_similarities = compute_cosine_similarities(
                control.loss_values,
                vr.loss_values,
            )

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
            f"throughput_delta={vr.throughput_gain_vs_control:+.1%}, "
            f"mem_delta={vr.memory_delta_vs_control:+.1%}, "
            f"diverged={vr.diverged}"
        )

    return control, validation_results


def _apply_vs_best_native_deltas(
    results: list[ValidationResult],
    best_native_baseline_id: Optional[str],
) -> None:
    if not best_native_baseline_id:
        return
    native = next(
        (
            r
            for r in results
            if r.success
            and not r.diverged
            and (r.baseline_id == best_native_baseline_id or r.config_id == best_native_baseline_id)
        ),
        None,
    )
    if native is None or native.speedup_vs_control <= 0:
        return

    for r in results:
        if not r.success or r.diverged:
            continue
        r.speedup_vs_best_native = r.speedup_vs_control / native.speedup_vs_control
        r.cost_delta_vs_best_native = r.cost_delta_vs_control - native.cost_delta_vs_control
        r.throughput_gain_vs_best_native = (
            r.throughput_gain_vs_control - native.throughput_gain_vs_control
        )


async def run_validation(
    control_repo: Path,
    configs: list[OptimizationConfig],
    run_config: RunConfig,
    *,
    profile: Optional[ProfileResult] = None,
    hardware_profile: Optional[HardwareProfile] = None,
    mode: str = "train",
) -> tuple[ControlRun, list[ValidationResult], RunSummary]:
    """Run validation with required native baseline ladder and objective scoring."""
    baseline_manifest: list[BaselineResult] = [
        BaselineResult(
            baseline_id="B0",
            name="Control (eager, no optimization)",
            eligible=True,
            success=True,
        )
    ]
    baseline_configs: list[OptimizationConfig] = []
    if run_config.compare_against_native and hardware_profile is not None:
        baseline_configs, baseline_manifest = build_native_baseline_candidates(
            run_config=run_config,
            hardware=hardware_profile,
            mode=mode,
            profile=profile,
        )

    all_configs = baseline_configs + list(configs)
    config_by_id = {cfg.id: cfg for cfg in all_configs}

    strategy = run_config.validation_strategy
    if (
        run_config.compare_against_native
        and run_config.baseline_policy == "required"
        and strategy == "staged"
    ):
        logger.info(
            "[validation] forcing parallel_all to keep B0..B2 comparisons on one shared control run."
        )
        strategy = "parallel_all"

    if strategy != "staged" or len(all_configs) <= 1:
        control, results = await _run_validation_batch(
            control_repo=control_repo,
            configs=all_configs,
            run_config=run_config,
            mode=mode,
            steps=run_config.validation_steps,
            batch_label="full",
        )
    else:
        stage1_steps = max(5, min(run_config.validation_steps, run_config.validation_steps // 3))
        logger.info(
            f"[staged] Stage 1 quick screening: {len(all_configs)} configs for {stage1_steps} steps"
        )
        _stage1_control, stage1_results = await _run_validation_batch(
            control_repo=control_repo,
            configs=all_configs,
            run_config=run_config,
            mode=mode,
            steps=stage1_steps,
            batch_label="stage1",
        )

        stage1_by_id = {r.config_id: r for r in stage1_results}
        promotable: list[tuple[OptimizationConfig, float]] = []
        for cfg in all_configs:
            r = stage1_by_id.get(cfg.id)
            if r is None or not r.success or r.diverged:
                continue
            score = r.speedup_vs_control / _risk_weight(cfg)
            promotable.append((cfg, score))
        promotable.sort(key=lambda x: x[1], reverse=True)

        promoted = [cfg for cfg, _ in promotable[: max(1, run_config.staged_top_k)]]
        if not promoted:
            control, results = _stage1_control, stage1_results
        else:
            logger.info(
                f"[staged] Stage 2 full validation for configs: {', '.join(c.id for c in promoted)}"
            )
            control, results = await _run_validation_batch(
                control_repo=control_repo,
                configs=promoted,
                run_config=run_config,
                mode=mode,
                steps=run_config.validation_steps,
                batch_label="stage2",
            )
            stage2_ids = {r.config_id for r in results}
            for cfg in all_configs:
                if cfg.id in stage2_ids:
                    continue
                results.append(_failure_result(
                    cfg,
                    "Skipped after stage-1 screening (not promoted to full validation)",
                ))

    scored = score_validation_results(results, config_by_id, run_config.objective_profile)
    scored = apply_ablation_estimates(scored, run_config.ablation_top_k)
    summary = summarize_run(
        scored,
        baseline_manifest=baseline_manifest,
        objective_profile=run_config.objective_profile,
        compare_against_native=run_config.compare_against_native,
    )
    _apply_vs_best_native_deltas(scored, summary.best_native_baseline_id)
    return control, scored, summary
