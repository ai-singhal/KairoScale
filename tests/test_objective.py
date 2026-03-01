"""Tests for objective scoring and run summary."""

from KairoScale.optimizer.objective import score_validation_results, summarize_run
from KairoScale.types import (
    BaselineResult,
    OptimizationConfig,
    OptimizationType,
    RiskLevel,
    ValidationResult,
)


def test_scoreValidationResultsAssignsObjective():
    configs = {
        "B1": OptimizationConfig(
            id="B1",
            name="compile",
            description="baseline",
            optimization_type=OptimizationType.COMPILATION,
            evidence=["baseline"],
            risk_level=RiskLevel.LOW,
            is_native_baseline=True,
            baseline_id="B1",
        ),
        "opt-001": OptimizationConfig(
            id="opt-001",
            name="composite",
            description="composite",
            optimization_type=OptimizationType.ATTENTION,
            evidence=["signal"],
            risk_level=RiskLevel.MEDIUM,
        ),
    }
    results = [
        ValidationResult(
            config_id="B1",
            config_name="compile",
            success=True,
            speedup_vs_control=1.2,
            throughput_gain_vs_control=0.1,
            cost_delta_vs_control=-0.1,
            is_native_baseline=True,
            baseline_id="B1",
        ),
        ValidationResult(
            config_id="opt-001",
            config_name="composite",
            success=True,
            speedup_vs_control=1.4,
            throughput_gain_vs_control=0.2,
            cost_delta_vs_control=-0.15,
        ),
    ]

    scored = score_validation_results(results, configs, "balanced")
    assert all(r.objective_score is not None for r in scored)
    assert scored[1].objective_score > scored[0].objective_score


def test_summarizeRunComputesDeltaVsBestNative():
    results = [
        ValidationResult(
            config_id="B1",
            config_name="compile",
            success=True,
            is_native_baseline=True,
            baseline_id="B1",
            speedup_vs_control=1.2,
            throughput_gain_vs_control=0.1,
            cost_delta_vs_control=-0.1,
            objective_score=0.3,
            evidence_chain=["signal-native"],
        ),
        ValidationResult(
            config_id="opt-001",
            config_name="composite",
            success=True,
            speedup_vs_control=1.5,
            throughput_gain_vs_control=0.2,
            cost_delta_vs_control=-0.2,
            objective_score=0.5,
            evidence_chain=["signal-composite"],
        ),
    ]
    manifest = [
        BaselineResult("B0", "control", True, True),
        BaselineResult("B1", "compile", True, False),
    ]

    summary = summarize_run(
        results,
        baseline_manifest=manifest,
        objective_profile="balanced",
        compare_against_native=True,
    )
    assert summary.best_overall_config_id == "opt-001"
    assert summary.best_native_baseline_id == "B1"
    assert summary.speedup_vs_best_native is not None
    assert summary.speedup_vs_best_native > 1.0
