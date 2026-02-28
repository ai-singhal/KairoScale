"""Tests for gpunity.types serialization and deserialization."""

from gpunity.types import (
    ControlRun,
    LoopDetectionMethod,
    MemoryTimelineEntry,
    OperatorProfile,
    OptimizationConfig,
    OptimizationType,
    ProfileResult,
    RiskLevel,
    RunConfig,
    ValidationResult,
)


def test_profileResultRoundTrip():
    profile = ProfileResult(
        top_operators=[
            OperatorProfile(name="aten::mm", gpu_time_ms=12.5, cpu_time_ms=1.2, pct_total=45.0, call_count=100),
        ],
        gpu_utilization=72.3,
        peak_memory_mb=4096.0,
        memory_timeline=[MemoryTimelineEntry(step=0, allocated_mb=1024.0, reserved_mb=2048.0)],
        forward_time_ms=10.0,
        backward_time_ms=25.0,
        loop_detection_method=LoopDetectionMethod.HEURISTIC,
        loop_detection_confidence="medium",
    )
    d = profile.to_dict()
    restored = ProfileResult.from_dict(d)

    assert restored.gpu_utilization == 72.3
    assert len(restored.top_operators) == 1
    assert restored.top_operators[0].name == "aten::mm"
    assert restored.loop_detection_method == LoopDetectionMethod.HEURISTIC
    assert len(restored.memory_timeline) == 1


def test_optimizationConfigRoundTrip():
    config = OptimizationConfig(
        id="opt-001",
        name="Flash Attention",
        description="Use flash attention for transformer layers",
        optimization_type=OptimizationType.ATTENTION,
        evidence=["sdpa = 58% GPU time"],
        code_changes={"model.py": "import flash_attn"},
        estimated_speedup=1.8,
        risk_level=RiskLevel.LOW,
    )
    d = config.to_dict()
    restored = OptimizationConfig.from_dict(d)

    assert restored.id == "opt-001"
    assert restored.optimization_type == OptimizationType.ATTENTION
    assert restored.risk_level == RiskLevel.LOW
    assert restored.estimated_speedup == 1.8
    assert "model.py" in restored.code_changes


def test_profileResultSummary():
    profile = ProfileResult(
        top_operators=[
            OperatorProfile(name="aten::mm", gpu_time_ms=12.5, cpu_time_ms=1.2, pct_total=45.0, call_count=100),
        ],
        gpu_utilization=72.3,
        peak_memory_mb=4096.0,
        forward_time_ms=10.0,
        backward_time_ms=25.0,
        dataloader_throughput=500.0,
        dataloader_stall_time_ms=1.2,
        dataloader_bottleneck=False,
    )
    summary = profile.summary()
    assert "72.3%" in summary
    assert "aten::mm" in summary
    assert "4096.0 MB" in summary
    assert "Backward/Forward ratio" in summary


def test_runConfigFromDict():
    d = {
        "repo_path": "/tmp/repo",
        "entry_point": "main.py",
        "provider": "openai",
        "extra_field": "ignored",
    }
    config = RunConfig.from_dict(d)
    assert config.repo_path == "/tmp/repo"
    assert config.entry_point == "main.py"
    assert config.provider == "openai"


def test_validationResultDefaults():
    result = ValidationResult(config_id="opt-001", config_name="Test", success=True)
    assert result.speedup_vs_control == 1.0
    assert result.diverged is False
    assert result.loss_values == []
