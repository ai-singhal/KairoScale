"""Tests for gpunity.reporter.markdown."""

from pathlib import Path

from gpunity.reporter.markdown import generate_report
from gpunity.types import (
    ControlRun,
    OperatorProfile,
    OptimizationConfig,
    OptimizationType,
    ProfileResult,
    RiskLevel,
    RunConfig,
    ValidationResult,
)


def _makeMinimalInputs(tmp_path):
    config = RunConfig(repo_path=str(tmp_path))
    profile = ProfileResult(
        top_operators=[
            OperatorProfile(name="aten::mm", gpu_time_ms=10.0, cpu_time_ms=1.0, pct_total=50.0, call_count=20),
        ],
        gpu_utilization=65.0,
        peak_memory_mb=2048.0,
        forward_time_ms=5.0,
        backward_time_ms=15.0,
    )
    opt = OptimizationConfig(
        id="opt-001",
        name="Flash Attention",
        description="Enable flash attention",
        optimization_type=OptimizationType.ATTENTION,
        evidence=["sdpa = 58% GPU time"],
        estimated_speedup=1.5,
        risk_level=RiskLevel.LOW,
    )
    control = ControlRun(
        steps_completed=50,
        wall_clock_seconds=10.0,
        avg_step_time_ms=200.0,
        peak_memory_mb=2048.0,
        throughput_samples_sec=80.0,
        loss_values=[2.0, 1.8, 1.5],
        gradient_norms=[1.0, 0.9, 0.8],
        cost_estimate_usd=0.01,
    )
    result = ValidationResult(
        config_id="opt-001",
        config_name="Flash Attention",
        success=True,
        speedup_vs_control=1.4,
        memory_delta_vs_control=-0.15,
        loss_values=[2.0, 1.7, 1.4],
    )
    return config, profile, [opt], control, [result]


def test_generateReportCreatesFile(tmp_path):
    config, profile, opts, control, results = _makeMinimalInputs(tmp_path)
    output = tmp_path / "report.md"
    path = generate_report(config, profile, opts, control, results, output)
    assert path.exists()
    content = path.read_text()
    assert "# GPUnity Optimization Report" in content
    assert "Flash Attention" in content
    assert "aten::mm" in content


def test_generateReportDryRun(tmp_path):
    config, profile, opts, control, _ = _makeMinimalInputs(tmp_path)
    config.dry_run = True
    output = tmp_path / "report.md"
    path = generate_report(config, profile, opts, control, [], output)
    content = path.read_text()
    assert "dry-run" in content.lower()


def test_generateReportNoConfigs(tmp_path):
    config = RunConfig(repo_path=str(tmp_path))
    profile = ProfileResult()
    control = ControlRun(
        steps_completed=0, wall_clock_seconds=0, avg_step_time_ms=0,
        peak_memory_mb=0, throughput_samples_sec=0, loss_values=[],
        gradient_norms=[], cost_estimate_usd=0,
    )
    output = tmp_path / "report.md"
    path = generate_report(config, profile, [], control, [], output)
    assert path.exists()
