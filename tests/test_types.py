"""Tests for KairoScale.types serialization and deserialization."""

from KairoScale.types import (
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
    are_configs_compatible,
    merge_configs,
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


def test_areConfigsCompatibleTrueForWhitelistedTypes():
    compile_cfg = OptimizationConfig(
        id="opt-001",
        name="Compile",
        description="Enable compile",
        optimization_type=OptimizationType.COMPILATION,
        evidence=["signal A"],
        config_overrides={"compile": True},
        estimated_speedup=1.2,
        risk_level=RiskLevel.LOW,
    )
    amp_cfg = OptimizationConfig(
        id="opt-002",
        name="AMP",
        description="Enable bf16",
        optimization_type=OptimizationType.MIXED_PRECISION,
        evidence=["signal B"],
        config_overrides={"amp": True, "precision": "bf16"},
        estimated_speedup=1.1,
        risk_level=RiskLevel.LOW,
    )

    compatible, reason = are_configs_compatible(compile_cfg, amp_cfg)
    assert compatible
    assert reason == "compatible"


def test_areConfigsCompatibleRejectsOverrideConflict():
    cfg_a = OptimizationConfig(
        id="opt-001",
        name="Compile on",
        description="Enable compile",
        optimization_type=OptimizationType.COMPILATION,
        evidence=[],
        config_overrides={"compile": True},
        estimated_speedup=1.1,
        risk_level=RiskLevel.LOW,
    )
    cfg_b = OptimizationConfig(
        id="opt-002",
        name="Compile off",
        description="Disable compile",
        optimization_type=OptimizationType.MIXED_PRECISION,
        evidence=[],
        config_overrides={"compile": False, "amp": True},
        estimated_speedup=1.1,
        risk_level=RiskLevel.LOW,
    )

    compatible, reason = are_configs_compatible(cfg_a, cfg_b)
    assert not compatible
    assert "compile" in reason


def test_mergeConfigsCombinesCompatibleConfigs():
    cfg_a = OptimizationConfig(
        id="opt-001",
        name="Compile",
        description="Enable compile",
        optimization_type=OptimizationType.COMPILATION,
        evidence=["ev1"],
        config_overrides={"compile": True},
        estimated_speedup=1.5,
        estimated_memory_delta=-0.1,
        risk_level=RiskLevel.LOW,
        dependencies=["dep-a"],
    )
    cfg_b = OptimizationConfig(
        id="opt-002",
        name="AMP",
        description="Enable AMP",
        optimization_type=OptimizationType.MIXED_PRECISION,
        evidence=["ev2"],
        config_overrides={"amp": True, "precision": "bf16"},
        estimated_speedup=1.8,
        estimated_memory_delta=-0.2,
        risk_level=RiskLevel.HIGH,
        dependencies=["dep-b"],
    )

    merged = merge_configs([cfg_a, cfg_b])

    assert merged.id.startswith("combo-")
    assert merged.optimization_type == OptimizationType.MIXED_PRECISION
    assert merged.estimated_speedup == 2.5  # capped
    assert abs(merged.estimated_memory_delta + 0.3) < 1e-9
    assert merged.risk_level == RiskLevel.HIGH
    assert merged.config_overrides["compile"] is True
    assert merged.config_overrides["amp"] is True
