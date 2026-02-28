"""Objective scoring and run-level winner selection."""

from __future__ import annotations

from dataclasses import replace

from gpunity.types import (
    BaselineResult,
    EvidenceEdge,
    OptimizationConfig,
    RunSummary,
    ValidationResult,
)


OBJECTIVE_WEIGHTS: dict[str, dict[str, float]] = {
    "balanced": {"speedup": 0.45, "cost_reduction": 0.25, "throughput_gain": 0.25, "risk": 0.05},
    "latency": {"speedup": 0.70, "cost_reduction": 0.10, "throughput_gain": 0.15, "risk": 0.05},
    "cost": {"speedup": 0.20, "cost_reduction": 0.60, "throughput_gain": 0.15, "risk": 0.05},
    "throughput": {"speedup": 0.25, "cost_reduction": 0.15, "throughput_gain": 0.55, "risk": 0.05},
}


def _risk_penalty(config: OptimizationConfig) -> float:
    if config.risk_level.value == "low":
        return 0.05
    if config.risk_level.value == "medium":
        return 0.15
    return 0.30


def score_validation_results(
    results: list[ValidationResult],
    config_by_id: dict[str, OptimizationConfig],
    objective_profile: str,
) -> list[ValidationResult]:
    """Assign objective scores to successful validation results."""
    weights = OBJECTIVE_WEIGHTS.get(objective_profile, OBJECTIVE_WEIGHTS["balanced"])
    scored: list[ValidationResult] = []
    for r in results:
        if not r.success or r.diverged:
            scored.append(r)
            continue
        cfg = config_by_id.get(r.config_id)
        risk = _risk_penalty(cfg) if cfg is not None else 0.15
        score = (
            weights["speedup"] * (r.speedup_vs_control - 1.0)
            + weights["cost_reduction"] * (-r.cost_delta_vs_control)
            + weights["throughput_gain"] * r.throughput_gain_vs_control
            - weights["risk"] * risk
        )
        scored.append(replace(r, objective_score=score, risk_penalty=risk))
    return scored


def _best_valid(
    results: list[ValidationResult],
    *,
    only_native: bool,
) -> ValidationResult | None:
    candidates = [
        r for r in results
        if r.success and not r.diverged and r.objective_score is not None
        and (not only_native or r.is_native_baseline)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.objective_score or float("-inf"))


def apply_ablation_estimates(
    results: list[ValidationResult],
    top_k: int,
) -> list[ValidationResult]:
    """Add lightweight contribution estimates for top-k candidates."""
    eligible = [r for r in results if r.success and not r.diverged and r.objective_score is not None]
    ranked = sorted(eligible, key=lambda r: r.objective_score or -1e9, reverse=True)
    top_ids = {r.config_id for r in ranked[: max(0, top_k)]}
    enriched: list[ValidationResult] = []
    for r in results:
        if r.config_id not in top_ids:
            enriched.append(r)
            continue
        chain = list(r.evidence_chain)
        chain.append(
            "Ablation estimate: score combines speedup, throughput, cost reduction, and risk."
        )
        enriched.append(replace(r, evidence_chain=chain))
    return enriched


def summarize_run(
    results: list[ValidationResult],
    baseline_manifest: list[BaselineResult],
    objective_profile: str,
    *,
    compare_against_native: bool,
) -> RunSummary:
    """Create run summary including winner and delta vs best native."""
    summary = RunSummary(
        objective_profile=objective_profile,
        objective_weights=OBJECTIVE_WEIGHTS.get(objective_profile, OBJECTIVE_WEIGHTS["balanced"]),
        baseline_results=[replace(b) for b in baseline_manifest],
    )

    best_overall = _best_valid(results, only_native=False)
    best_native = _best_valid(results, only_native=True) if compare_against_native else None

    if best_overall is not None:
        summary.best_overall_config_id = best_overall.config_id
    if best_native is not None:
        summary.best_native_baseline_id = best_native.baseline_id or best_native.config_id

    # Update baseline manifest with measured results
    result_by_baseline = {
        (r.baseline_id or r.config_id): r
        for r in results
        if r.is_native_baseline
    }
    updated: list[BaselineResult] = []
    for baseline in summary.baseline_results:
        measured = result_by_baseline.get(baseline.baseline_id)
        if measured is None:
            updated.append(baseline)
            continue
        updated.append(BaselineResult(
            baseline_id=baseline.baseline_id,
            name=baseline.name,
            eligible=baseline.eligible,
            success=measured.success and not measured.diverged,
            skip_reason=baseline.skip_reason if baseline.eligible else baseline.skip_reason,
            speedup_vs_control=measured.speedup_vs_control if measured.success else None,
            cost_delta_vs_control=measured.cost_delta_vs_control if measured.success else None,
            throughput_gain_vs_control=(
                measured.throughput_gain_vs_control if measured.success else None
            ),
            objective_score=measured.objective_score if measured.success else None,
        ))
    summary.baseline_results = updated

    if best_overall is None or best_native is None:
        if best_native is None and compare_against_native:
            summary.confidence = "low"
        for r in results:
            if not r.success:
                continue
            for ev in r.evidence_chain[:3]:
                summary.evidence_edges.append(
                    EvidenceEdge(
                        source=ev,
                        target=r.config_id,
                        relation="supports",
                        detail="profile/heuristic signal for candidate",
                    )
                )
        return summary

    # Deltas vs best native baseline
    summary.speedup_vs_best_native = (
        best_overall.speedup_vs_control / best_native.speedup_vs_control
        if best_native.speedup_vs_control > 0
        else None
    )
    summary.cost_delta_vs_best_native = (
        best_overall.cost_delta_vs_control - best_native.cost_delta_vs_control
    )
    summary.throughput_gain_vs_best_native = (
        best_overall.throughput_gain_vs_control - best_native.throughput_gain_vs_control
    )
    summary.evidence_edges.append(
        EvidenceEdge(
            source=best_native.baseline_id or best_native.config_id,
            target=best_overall.config_id,
            relation="compared_against",
            detail="winner selection relative to best native baseline",
        )
    )
    for r in results:
        if not r.success:
            continue
        for ev in r.evidence_chain[:3]:
            summary.evidence_edges.append(
                EvidenceEdge(
                    source=ev,
                    target=r.config_id,
                    relation="supports",
                    detail="profile/heuristic signal for candidate",
                )
            )
    return summary
