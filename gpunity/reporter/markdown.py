"""Markdown report generation.

Generates a comprehensive optimization report from profiling,
agent analysis, and validation results.
"""

from __future__ import annotations

import difflib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from gpunity.reporter.charts import (
    render_cosine_sim_chart,
    render_loss_comparison,
    render_operator_breakdown,
    render_summary_table,
)
from gpunity.types import (
    ControlRun,
    HardwareProfile,
    OptimizationConfig,
    ProfileResult,
    RunConfig,
    RunSummary,
    ValidationResult,
)
from gpunity.utils.logging import get_logger

logger = get_logger("gpunity.reporter.markdown")


def generate_report(
    run_config: RunConfig,
    profile: ProfileResult,
    configs: list[OptimizationConfig],
    control: ControlRun,
    results: list[ValidationResult],
    output_path: Path,
    config_export_dir: Optional[Path] = None,
    hardware_profile: Optional[HardwareProfile] = None,
    run_summary: Optional[RunSummary] = None,
    mode: Optional[str] = None,
) -> Path:
    """Generate the full Markdown optimization report.

    Args:
        run_config: The run configuration used.
        profile: Profiling results from Phase 1.
        configs: Optimization configs from Phase 2.
        control: Control run results from Phase 3.
        results: Validation results from Phase 3.
        output_path: Path to write the report.

    Returns:
        The output path (same as input).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections = []

    # Header
    sections.append(_render_header(run_config, mode))

    # Executive Summary
    sections.append(_render_executive_summary(configs, control, results, run_config, run_summary))

    if hardware_profile is not None:
        sections.append(_render_hardware_context(hardware_profile))

    # Profile Analysis
    sections.append(_render_profile_analysis(profile))

    # Proposed Optimizations
    sections.append(_render_proposed_optimizations(configs, run_config, config_export_dir))

    # Validation Results (if not dry run)
    if results:
        sections.append(_render_validation_results(control, results, run_config, run_summary))
        sections.append(_render_evidence_tracebacks(configs, results, run_summary))
    else:
        sections.append("## Validation Results\n\n*Skipped (dry-run mode).*\n")

    if run_summary is not None and run_summary.baseline_results:
        sections.append(_render_baseline_ladder(run_summary))

    # Recommendations
    sections.append(_render_recommendations(results, configs, run_summary))

    # Appendix
    sections.append(_render_appendix(profile, configs))

    report = "\n\n".join(sections)
    output_path.write_text(report, encoding="utf-8")

    logger.info(f"Report written to: {output_path}")
    return output_path


def _render_header(config: RunConfig, mode: Optional[str]) -> str:
    resolved_mode = mode or config.mode
    return (
        f"# GPUnity Optimization Report\n\n"
        f"> **Repo**: `{config.repo_path}` | "
        f"**GPU**: {config.gpu_type} | "
        f"**Workload**: {resolved_mode} | "
        f"**Objective**: {config.objective_profile} | "
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"**Mode**: {'local' if config.local else 'modal'}\n"
    )


def _render_executive_summary(
    configs: list[OptimizationConfig],
    control: ControlRun,
    results: list[ValidationResult],
    run_config: RunConfig,
    run_summary: Optional[RunSummary],
) -> str:
    lines = ["## Executive Summary\n"]

    if run_summary is not None and run_summary.best_overall_config_id:
        lines.append(f"- **Best overall config ID**: `{run_summary.best_overall_config_id}`")
    if run_summary is not None and run_summary.best_native_baseline_id:
        lines.append(
            f"- **Best native baseline**: `{run_summary.best_native_baseline_id}`"
        )
    if run_summary is not None and run_summary.speedup_vs_best_native is not None:
        lines.append(
            "- **Delta vs best native**: "
            f"{run_summary.speedup_vs_best_native:.2f}x speedup ratio, "
            f"{run_summary.cost_delta_vs_best_native:+.1%} cost delta, "
            f"{run_summary.throughput_gain_vs_best_native:+.1%} throughput gain"
        )

    # Best config
    successful = [r for r in results if r.success and not r.diverged]
    if successful:
        best = max(
            successful,
            key=lambda r: r.objective_score if r.objective_score is not None else r.speedup_vs_control,
        )
        lines.append(
            f"- **Best config**: \"{best.config_name}\" -- "
            f"{best.speedup_vs_control:.1f}x speedup, "
            f"{best.memory_delta_vs_control:+.0%} memory"
        )
    elif configs:
        best_cfg = max(configs, key=lambda c: c.estimated_speedup)
        lines.append(
            f"- **Top proposed config**: \"{best_cfg.name}\" -- "
            f"estimated {best_cfg.estimated_speedup:.1f}x speedup"
        )

    # Stats
    total = len(results)
    succeeded = sum(1 for r in results if r.success and not r.diverged)
    ran = sum(1 for r in results if r.success)
    diverged = sum(1 for r in results if r.diverged)
    lines.append(f"- {succeeded}/{total} configs validated successfully")
    lines.append(f"- {ran}/{total} configs executed without runtime failure")
    if diverged:
        lines.append(f"- {diverged} config(s) showed divergence")
    else:
        lines.append(f"- No divergence detected in any configuration")

    # Cost
    if control.cost_estimate_usd > 0:
        lines.append(f"- Control run cost: ${control.cost_estimate_usd:.4f}")

    if run_config.dry_run:
        lines.append(f"\n> *Note: This is a dry-run report. Validation was skipped.*")

    return "\n".join(lines)


def _render_hardware_context(hardware: HardwareProfile) -> str:
    lines = ["## Hardware Context\n"]
    lines.append(f"- GPU: {hardware.gpu_name}")
    lines.append(f"- Detection source: {hardware.detection_source} ({hardware.confidence} confidence)")
    if hardware.compute_capability:
        lines.append(f"- Compute capability: {hardware.compute_capability}")
    if hardware.vram_mb is not None:
        lines.append(f"- VRAM: {hardware.vram_mb} MB")
    if hardware.cuda_version:
        lines.append(f"- CUDA runtime: {hardware.cuda_version}")
    if hardware.driver_version:
        lines.append(f"- Driver: {hardware.driver_version}")
    lines.append(
        "- Feature flags: "
        f"compile={hardware.supports_compile}, "
        f"cuda_graphs={hardware.supports_cuda_graphs}, "
        f"bf16={hardware.supports_bf16}, tf32={hardware.supports_tf32}"
    )
    for note in hardware.notes:
        lines.append(f"- Note: {note}")
    return "\n".join(lines)


def _render_profile_analysis(profile: ProfileResult) -> str:
    lines = ["## Profile Analysis\n"]

    # Key findings
    lines.append("### Key Findings\n")
    lines.append(f"- GPU Utilization: {profile.gpu_utilization:.1f}%")
    lines.append(f"- Peak Memory: {profile.peak_memory_mb:.1f} MB")

    if profile.forward_time_ms > 0 or profile.backward_time_ms > 0:
        total = profile.forward_time_ms + profile.backward_time_ms
        if total > 0:
            fwd_pct = profile.forward_time_ms / total * 100
            bwd_pct = profile.backward_time_ms / total * 100
            lines.append(
                f"- Forward/Backward split: {fwd_pct:.0f}% / {bwd_pct:.0f}%"
            )
            if profile.backward_time_ms > 0 and profile.forward_time_ms > 0:
                ratio = profile.backward_time_ms / profile.forward_time_ms
                lines.append(f"- Backward/Forward ratio: {ratio:.1f}x")

    lines.append(
        f"- DataLoader bottleneck: "
        f"{'YES' if profile.dataloader_bottleneck else 'No'}"
    )
    if profile.dataloader_stall_time_ms > 0:
        lines.append(f"- DataLoader stall: {profile.dataloader_stall_time_ms:.2f} ms/step")

    # Loop detection
    lines.append(
        f"- Training loop detection: {profile.loop_detection_method.value}"
    )
    if profile.loop_detection_method.value == "none":
        lines.append(
            "\n> **Warning**: Training loop could not be automatically detected. "
            "Consider using `--train-function <name>` for more accurate profiling."
        )

    # Operator breakdown
    lines.append("\n### Operator Breakdown\n")
    lines.append(render_operator_breakdown(profile.top_operators))

    return "\n".join(lines)


def _build_code_diff(repo_root: Path, rel_path: str, new_content: str) -> str:
    target = repo_root / rel_path
    if target.exists() and target.is_file():
        old_content = target.read_text(encoding="utf-8", errors="replace")
    else:
        old_content = ""
    diff_lines = list(difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
    ))
    if not diff_lines:
        return "(no textual diff)"
    if len(diff_lines) > 220:
        diff_lines = diff_lines[:220] + ["... diff truncated ...\n"]
    return "".join(diff_lines)


def _render_proposed_optimizations(
    configs: list[OptimizationConfig],
    run_config: RunConfig,
    config_export_dir: Optional[Path],
) -> str:
    lines = ["## Proposed Optimizations\n"]

    if not configs:
        lines.append("*No optimization configs were generated.*\n")
        return "\n".join(lines)

    lines.append(
        "| ID | Name | Type | Est. Speedup | Risk | Evidence |"
    )
    lines.append(
        "|----|------|------|-------------|------|----------|"
    )

    for c in configs:
        evidence_str = "; ".join(c.evidence[:2])
        if len(c.evidence) > 2:
            evidence_str += f" (+{len(c.evidence) - 2} more)"
        lines.append(
            f"| {c.id} | {c.name} | {c.optimization_type.value} | "
            f"{c.estimated_speedup:.1f}x | {c.risk_level.value} | "
            f"{evidence_str} |"
        )

    # Detail sections
    for c in configs:
        lines.append(f"\n### {c.id}: {c.name}\n")
        lines.append(f"**Type**: {c.optimization_type.value} | "
                     f"**Risk**: {c.risk_level.value} | "
                     f"**Estimated speedup**: {c.estimated_speedup:.1f}x | "
                     f"**Memory delta**: {c.estimated_memory_delta:+.0%}\n")
        lines.append(f"{c.description}\n")
        lines.append("**Evidence**:")
        for ev in c.evidence:
            lines.append(f"- {ev}")

        if c.config_overrides:
            lines.append(f"\n**Config overrides**: `{c.config_overrides}`")
        if c.dependencies:
            lines.append(f"\n**Additional dependencies**: {', '.join(c.dependencies)}")

        # Show reproducible config path + apply command
        if config_export_dir is not None:
            cfg_path = (Path(config_export_dir) / f"{c.id}.json").resolve()
            lines.append(f"\n**Config JSON**: `{cfg_path}`")
            lines.append(
                f"**Apply command**: `gpunity apply {cfg_path} --repo {Path(run_config.repo_path).resolve()}`"
            )

        if c.code_changes:
            lines.append("\n**Code diff(s)**:")
            repo_root = Path(run_config.repo_path)
            for rel_path, new_content in c.code_changes.items():
                lines.append(f"\n`{rel_path}`")
                lines.append("```diff")
                lines.append(_build_code_diff(repo_root, rel_path, new_content))
                lines.append("```")

    return "\n".join(lines)


def _render_validation_results(
    control: ControlRun,
    results: list[ValidationResult],
    run_config: RunConfig,
    run_summary: Optional[RunSummary],
) -> str:
    lines = ["## Validation Results\n"]

    # Summary table
    lines.append("### Summary\n")
    lines.append(render_summary_table(control, results))
    if run_summary is not None and run_summary.best_native_baseline_id:
        lines.append(
            f"\nBest native baseline for this run: `{run_summary.best_native_baseline_id}`"
        )

    # Control run info
    lines.append(f"\n**Control run**: {control.steps_completed} steps, "
                 f"{control.wall_clock_seconds:.1f}s, "
                 f"{control.avg_step_time_ms:.1f} ms/step, "
                 f"peak memory {control.peak_memory_mb:.1f} MB\n")

    # Loss comparison
    variant_losses = {}
    for r in results:
        if r.success and r.loss_values:
            variant_losses[r.config_name] = r.loss_values

    if variant_losses:
        lines.append("### Loss Curves\n")
        lines.append(render_loss_comparison(control.loss_values, variant_losses))

    # Cosine similarity
    cosine_sims = {}
    for r in results:
        if r.success and r.gradient_cosine_similarities:
            cosine_sims[r.config_name] = r.gradient_cosine_similarities

    if cosine_sims:
        lines.append("### Gradient Similarity\n")
        lines.append(render_cosine_sim_chart(cosine_sims, run_config.divergence_threshold))

    lines.append("### Correctness (Logit Signatures)\n")
    for r in results:
        if not r.success:
            continue
        if r.logits_checks_compared == 0:
            lines.append(f"- **{r.config_name}**: no comparable logit signatures captured")
            continue
        status = "within tolerance" if r.logits_within_tolerance else "exceeded tolerance"
        lines.append(
            f"- **{r.config_name}**: checks={r.logits_checks_compared}, "
            f"max_abs_diff={r.logits_max_abs_diff:.6f}, "
            f"mean_abs_diff={r.logits_mean_abs_diff:.6f} ({status})"
        )

    # Divergence flags
    diverged_results = [r for r in results if r.diverged]
    if diverged_results:
        lines.append("### Divergence Flags\n")
        for r in diverged_results:
            lines.append(
                f"- **{r.config_name}**: Diverged at step {r.divergence_step}. "
                f"Reason: {r.divergence_reason}"
            )

    return "\n".join(lines)


def _render_evidence_tracebacks(
    configs: list[OptimizationConfig],
    results: list[ValidationResult],
    run_summary: Optional[RunSummary],
) -> str:
    lines = ["## Evidence Tracebacks\n"]
    if run_summary is not None and run_summary.evidence_edges:
        lines.append("### Evidence Graph\n")
        for edge in run_summary.evidence_edges[:12]:
            lines.append(f"- `{edge.source}` -> `{edge.target}` ({edge.relation})")
        lines.append("")

    config_by_id = {cfg.id: cfg for cfg in configs}
    successful = [r for r in results if r.success and not r.diverged]
    if not successful:
        lines.append("*No successful validated candidates for evidence tracebacks.*")
        return "\n".join(lines)

    ordered = sorted(
        successful,
        key=lambda r: r.objective_score if r.objective_score is not None else r.speedup_vs_control,
        reverse=True,
    )[:3]

    best_native_id = run_summary.best_native_baseline_id if run_summary else None
    for r in ordered:
        cfg = config_by_id.get(r.config_id)
        lines.append(f"### {r.config_name}\n")
        lines.append("**Baseline comparison**")
        lines.append(f"- vs control speedup: {r.speedup_vs_control:.2f}x")
        lines.append(f"- vs control cost delta: {r.cost_delta_vs_control:+.1%}")
        lines.append(f"- vs control throughput delta: {r.throughput_gain_vs_control:+.1%}")
        if r.speedup_vs_best_native is not None and best_native_id:
            lines.append(f"- vs best native (`{best_native_id}`): {r.speedup_vs_best_native:.2f}x")

        lines.append("\n**Profiler traceback**")
        if r.evidence_chain:
            for ev in r.evidence_chain:
                lines.append(f"- {ev}")
        else:
            lines.append("- No evidence chain captured.")

        lines.append("\n**Validation block**")
        lines.append(f"- Diverged: {'YES' if r.diverged else 'No'}")
        lines.append(f"- Logit checks: {r.logits_checks_compared}")
        if r.logits_max_abs_diff is not None:
            lines.append(f"- Logit max abs diff: {r.logits_max_abs_diff:.6f}")
        lines.append(f"- Objective score: {r.objective_score:.3f}" if r.objective_score is not None else "- Objective score: N/A")

        lines.append("\n**Code traceback**")
        if cfg is None or not cfg.code_changes:
            lines.append("- No code patch required (runtime/config optimization).")
        else:
            for rel_path in cfg.code_changes.keys():
                lines.append(f"- Modified file: `{rel_path}`")
    return "\n".join(lines)


def _render_baseline_ladder(run_summary: RunSummary) -> str:
    lines = ["## Native Baseline Ladder\n"]
    lines.append("| Baseline | Eligible | Success | Speedup vs Control | Throughput Δ | Cost Δ | Obj Score |")
    lines.append("|----------|----------|---------|--------------------|--------------|--------|-----------|")
    for b in run_summary.baseline_results:
        speed = f"{b.speedup_vs_control:.2f}x" if b.speedup_vs_control is not None else "-"
        thr = f"{b.throughput_gain_vs_control:+.1%}" if b.throughput_gain_vs_control is not None else "-"
        cost = f"{b.cost_delta_vs_control:+.1%}" if b.cost_delta_vs_control is not None else "-"
        obj = f"{b.objective_score:.3f}" if b.objective_score is not None else "-"
        success = "YES" if b.success else "No"
        eligible = "YES" if b.eligible else "No"
        lines.append(
            f"| {b.baseline_id} ({b.name}) | {eligible} | {success} | {speed} | {thr} | {cost} | {obj} |"
        )
        if b.skip_reason and not b.eligible:
            lines.append(f"\n- `{b.baseline_id}` skipped: {b.skip_reason}")
    return "\n".join(lines)


def _render_recommendations(
    results: list[ValidationResult],
    configs: list[OptimizationConfig],
    run_summary: Optional[RunSummary],
) -> str:
    lines = ["## Recommendations\n"]

    if not results:
        if configs:
            lines.append(
                "Validation was not run. Review the proposed optimizations above "
                "and re-run without `--dry-run` to validate them.\n"
            )
            for i, c in enumerate(configs, 1):
                lines.append(
                    f"{i}. **Consider**: {c.name} -- "
                    f"estimated {c.estimated_speedup:.1f}x speedup "
                    f"({c.risk_level.value} risk)"
                )
        else:
            lines.append("*No optimization recommendations available.*")
        return "\n".join(lines)

    # Categorize results
    adopt = []
    consider = []
    skip = []

    for r in results:
        if not r.success:
            skip.append((r, "Failed to run"))
        elif r.diverged:
            skip.append((r, f"Diverged: {r.divergence_reason}"))
        elif r.speedup_vs_control >= 1.2:
            adopt.append(r)
        elif r.speedup_vs_control >= 1.0:
            consider.append(r)
        else:
            skip.append((r, "Slower than baseline"))

    rank = 1
    for r in sorted(adopt, key=lambda x: x.speedup_vs_control, reverse=True):
        lines.append(
            f"{rank}. **Adopt**: {r.config_name} -- "
            f"{r.speedup_vs_control:.1f}x speedup, "
            f"{r.memory_delta_vs_control:+.0%} memory, no divergence"
        )
        rank += 1

    for r in sorted(consider, key=lambda x: x.speedup_vs_control, reverse=True):
        lines.append(
            f"{rank}. **Consider**: {r.config_name} -- "
            f"{r.speedup_vs_control:.1f}x speedup, "
            f"{r.memory_delta_vs_control:+.0%} memory"
        )
        rank += 1

    for r, reason in skip:
        lines.append(f"{rank}. **Skip**: {r.config_name} -- {reason}")
        rank += 1

    return "\n".join(lines)


def _render_appendix(
    profile: ProfileResult,
    configs: list[OptimizationConfig],
) -> str:
    lines = ["## Appendix\n"]

    # Full operator table
    if profile.top_operators:
        lines.append("### Full Operator Table\n")
        lines.append("| Operator | GPU Time (ms) | CPU Time (ms) | % Total | Calls | FLOPS |")
        lines.append("|----------|---------------|---------------|---------|-------|-------|")
        for op in profile.top_operators:
            flops = f"{op.flops:,}" if op.flops else "-"
            lines.append(
                f"| `{op.name}` | {op.gpu_time_ms:.3f} | "
                f"{op.cpu_time_ms:.3f} | {op.pct_total:.1f}% | "
                f"{op.call_count} | {flops} |"
            )

    # Config details
    lines.append("\n### Raw Config Data\n")
    for c in configs:
        lines.append(f"**{c.id}: {c.name}**\n")
        lines.append(f"```json")
        lines.append(json.dumps(c.to_dict(), indent=2))
        lines.append(f"```\n")

    lines.append(
        "\n---\n*Generated by GPUnity v0.1.0*"
    )

    return "\n".join(lines)
