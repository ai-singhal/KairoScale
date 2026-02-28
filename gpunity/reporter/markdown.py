"""Markdown report generation.

Generates a comprehensive optimization report from profiling,
agent analysis, and validation results.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from gpunity.reporter.charts import (
    render_cosine_sim_chart,
    render_loss_comparison,
    render_operator_breakdown,
    render_summary_table,
)
from gpunity.types import (
    ControlRun,
    OptimizationConfig,
    ProfileResult,
    RunConfig,
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
    sections.append(_render_header(run_config))

    # Executive Summary
    sections.append(_render_executive_summary(configs, control, results, run_config))

    # Profile Analysis
    sections.append(_render_profile_analysis(profile))

    # Proposed Optimizations
    sections.append(_render_proposed_optimizations(configs))

    # Validation Results (if not dry run)
    if results:
        sections.append(_render_validation_results(control, results, run_config))
    else:
        sections.append("## Validation Results\n\n*Skipped (dry-run mode).*\n")

    # Recommendations
    sections.append(_render_recommendations(results, configs))

    # Appendix
    sections.append(_render_appendix(profile, configs))

    report = "\n\n".join(sections)
    output_path.write_text(report)

    logger.info(f"Report written to: {output_path}")
    return output_path


def _render_header(config: RunConfig) -> str:
    return (
        f"# GPUnity Optimization Report\n\n"
        f"> **Repo**: `{config.repo_path}` | "
        f"**GPU**: {config.gpu_type} | "
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"**Mode**: {'local' if config.local else 'modal'}\n"
    )


def _render_executive_summary(
    configs: list[OptimizationConfig],
    control: ControlRun,
    results: list[ValidationResult],
    run_config: RunConfig,
) -> str:
    lines = ["## Executive Summary\n"]

    # Best config
    successful = [r for r in results if r.success and not r.diverged]
    if successful:
        best = max(successful, key=lambda r: r.speedup_vs_control)
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
    succeeded = sum(1 for r in results if r.success)
    diverged = sum(1 for r in results if r.diverged)
    lines.append(f"- {succeeded}/{total} configs validated successfully")
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


def _render_proposed_optimizations(configs: list[OptimizationConfig]) -> str:
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

    return "\n".join(lines)


def _render_validation_results(
    control: ControlRun,
    results: list[ValidationResult],
    run_config: RunConfig,
) -> str:
    lines = ["## Validation Results\n"]

    # Summary table
    lines.append("### Summary\n")
    lines.append(render_summary_table(control, results))

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


def _render_recommendations(
    results: list[ValidationResult],
    configs: list[OptimizationConfig],
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
        import json
        lines.append(json.dumps(c.to_dict(), indent=2))
        lines.append(f"```\n")

    lines.append(
        "\n---\n*Generated by GPUnity v0.1.0*"
    )

    return "\n".join(lines)
