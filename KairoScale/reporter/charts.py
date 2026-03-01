"""ASCII and visual chart rendering for reports."""

from __future__ import annotations

from KairoScale.types import ControlRun, OperatorProfile, ValidationResult


def render_operator_breakdown(
    operators: list[OperatorProfile],
    mode: str = "ascii",
) -> str:
    """Render operator breakdown as an ASCII table with bar chart.

    Args:
        operators: List of operator profiles sorted by GPU time.
        mode: Rendering mode ('ascii' or 'png'). PNG not yet implemented.

    Returns:
        Formatted string with operator breakdown table.
    """
    if not operators:
        return "*No operator data available.*\n"

    lines = []
    lines.append("| Rank | Operator | GPU Time (ms) | % Total | Calls | Bar |")
    lines.append("|------|----------|---------------|---------|-------|-----|")

    max_pct = max(op.pct_total for op in operators[:10]) if operators else 1.0

    for i, op in enumerate(operators[:10], 1):
        bar_len = int(op.pct_total / max(max_pct, 1) * 20)
        bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
        name = op.name[:40] if len(op.name) > 40 else op.name
        lines.append(
            f"| {i} | `{name}` | {op.gpu_time_ms:.2f} | "
            f"{op.pct_total:.1f}% | {op.call_count} | {bar} |"
        )

    return "\n".join(lines) + "\n"


def render_loss_comparison(
    control_losses: list[float],
    variant_losses: dict[str, list[float]],
    mode: str = "ascii",
) -> str:
    """Render loss curve comparison as an ASCII table.

    Args:
        control_losses: Loss values from the control run.
        variant_losses: Dict of config_name -> loss values.
        mode: Rendering mode.

    Returns:
        Formatted loss comparison table.
    """
    if not control_losses and not variant_losses:
        return "*No loss data available.*\n"

    # Header
    headers = ["Step", "Control"]
    for name in variant_losses:
        headers.append(name[:20])

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    max_steps = len(control_losses)
    for name, losses in variant_losses.items():
        max_steps = max(max_steps, len(losses))

    # Show every 5th step to keep table manageable
    step_interval = max(1, max_steps // 10)

    for step in range(0, max_steps, step_interval):
        row = [str(step)]

        if step < len(control_losses):
            row.append(f"{control_losses[step]:.4f}")
        else:
            row.append("-")

        for name in variant_losses:
            losses = variant_losses[name]
            if step < len(losses):
                row.append(f"{losses[step]:.4f}")
            else:
                row.append("-")

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def render_cosine_sim_chart(
    cosine_sims: dict[str, list[float]],
    threshold: float,
    mode: str = "ascii",
) -> str:
    """Render gradient cosine similarity over steps.

    Args:
        cosine_sims: Dict of config_name -> cosine similarity values.
        threshold: Divergence threshold to display.
        mode: Rendering mode.

    Returns:
        Formatted cosine similarity table.
    """
    if not cosine_sims:
        return "*No cosine similarity data available.*\n"

    headers = ["Step"] + [name[:20] for name in cosine_sims] + ["Threshold"]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    max_steps = max(len(v) for v in cosine_sims.values()) if cosine_sims else 0
    step_interval = max(1, max_steps // 10)

    for step in range(0, max_steps, step_interval):
        row = [str(step)]
        for name in cosine_sims:
            sims = cosine_sims[name]
            if step < len(sims):
                val = sims[step]
                flag = " (!)" if val < threshold else ""
                row.append(f"{val:.3f}{flag}")
            else:
                row.append("-")
        row.append(f"{threshold:.2f}")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def render_summary_table(
    control: ControlRun,
    results: list[ValidationResult],
) -> str:
    """Render the validation summary comparison table.

    Args:
        control: Control run results.
        results: List of validation results.

    Returns:
        Formatted Markdown summary table.
    """
    lines = []
    lines.append(
        "| Config | Speedup | Throughput Δ | Memory Delta | Cost Delta | Obj Score | vs Best Native | Diverged? | Logits Δmax |"
    )
    lines.append(
        "|--------|---------|--------------|-------------|------------|-----------|----------------|-----------|------------|"
    )

    # Control baseline row
    lines.append(
        f"| **Control** | 1.00x | baseline | baseline | baseline | - | - | - | - |"
    )

    for r in results:
        if not r.success:
            lines.append(
                f"| {r.config_name} | FAILED | - | - | - | - | - | - | - |"
            )
            continue

        diverged = "YES" if r.diverged else "No"
        logits_delta = f"{r.logits_max_abs_diff:.6f}" if r.logits_max_abs_diff is not None else "N/A"
        objective = f"{r.objective_score:.3f}" if r.objective_score is not None else "N/A"
        vs_native = (
            f"{r.speedup_vs_best_native:.2f}x"
            if r.speedup_vs_best_native is not None
            else "N/A"
        )

        lines.append(
            f"| {r.config_name} | {r.speedup_vs_control:.2f}x | "
            f"{r.throughput_gain_vs_control:+.1%} | "
            f"{r.memory_delta_vs_control:+.1%} | "
            f"{r.cost_delta_vs_control:+.1%} | "
            f"{objective} | {vs_native} | "
            f"{diverged} | {logits_delta} |"
        )

    return "\n".join(lines) + "\n"
