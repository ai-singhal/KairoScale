"""Shared data types for GPUnity.

All modules import from this file. No module defines its own data transfer objects.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class OptimizationType(str, Enum):
    """Categories of ML training optimizations."""
    ATTENTION = "attention"
    COMPILATION = "compilation"
    MIXED_PRECISION = "mixed_precision"
    DATA_LOADING = "data_loading"
    PARALLELISM = "parallelism"
    MEMORY = "memory"
    KERNEL_FUSION = "kernel_fusion"
    COMMUNICATION = "communication"


class RiskLevel(str, Enum):
    """Risk assessment for an optimization config."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LoopDetectionMethod(str, Enum):
    """How the training loop was detected."""
    USER_ANNOTATED = "user_annotated"
    HEURISTIC = "heuristic"
    NONE = "none"


@dataclass
class OperatorProfile:
    """Profile data for a single PyTorch operator."""
    name: str
    gpu_time_ms: float
    cpu_time_ms: float
    pct_total: float
    call_count: int
    flops: Optional[int] = None


@dataclass
class MemoryTimelineEntry:
    """Memory usage at a single training step."""
    step: int
    allocated_mb: float
    reserved_mb: float


@dataclass
class BackwardOpProfile:
    """Profile data for a backward-pass operator."""
    name: str
    time_ms: float
    pct_backward: float


@dataclass
class ProfileResult:
    """Aggregated profiling results from all profilers.

    This is the primary output of Phase 1 (Profile) and the primary input
    to Phase 2 (Agent).
    """
    # torch.profiler
    top_operators: list[OperatorProfile] = field(default_factory=list)
    gpu_utilization: float = 0.0
    chrome_trace_path: Optional[str] = None

    # Memory
    peak_memory_mb: float = 0.0
    memory_timeline: list[MemoryTimelineEntry] = field(default_factory=list)
    peak_allocation_stack: str = ""

    # Autograd
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    backward_ops: list[BackwardOpProfile] = field(default_factory=list)

    # DataLoader
    dataloader_throughput: float = 0.0
    dataloader_stall_time_ms: float = 0.0
    dataloader_bottleneck: bool = False

    # Loop detection metadata (D-001)
    loop_detection_method: LoopDetectionMethod = LoopDetectionMethod.NONE
    loop_detection_confidence: Optional[str] = None

    # Artifact paths
    artifact_dir: Optional[str] = None

    def summary(self) -> str:
        """Human-readable summary for the LLM agent to reason over."""
        lines = ["=== GPUnity Profile Summary ===", ""]

        # GPU utilization
        lines.append(f"GPU Utilization: {self.gpu_utilization:.1f}%")
        lines.append(f"Peak Memory: {self.peak_memory_mb:.1f} MB")
        lines.append("")

        # Top operators
        if self.top_operators:
            lines.append("Top GPU Operators:")
            for i, op in enumerate(self.top_operators[:10], 1):
                lines.append(
                    f"  {i}. {op.name}: {op.gpu_time_ms:.2f}ms "
                    f"({op.pct_total:.1f}% of total, {op.call_count} calls)"
                )
            lines.append("")

        # Forward/backward split
        if self.forward_time_ms > 0 or self.backward_time_ms > 0:
            total = self.forward_time_ms + self.backward_time_ms
            fwd_pct = (self.forward_time_ms / total * 100) if total > 0 else 0
            bwd_pct = (self.backward_time_ms / total * 100) if total > 0 else 0
            lines.append(f"Forward time: {self.forward_time_ms:.2f}ms ({fwd_pct:.1f}%)")
            lines.append(f"Backward time: {self.backward_time_ms:.2f}ms ({bwd_pct:.1f}%)")
            if self.backward_time_ms > 0 and self.forward_time_ms > 0:
                ratio = self.backward_time_ms / self.forward_time_ms
                lines.append(f"Backward/Forward ratio: {ratio:.1f}x")
            lines.append("")

        # DataLoader
        lines.append(f"DataLoader throughput: {self.dataloader_throughput:.1f} samples/sec")
        lines.append(f"DataLoader stall time: {self.dataloader_stall_time_ms:.2f} ms/step")
        lines.append(f"DataLoader bottleneck: {'YES' if self.dataloader_bottleneck else 'No'}")
        lines.append("")

        # Memory timeline
        if self.memory_timeline:
            max_alloc = max(e.allocated_mb for e in self.memory_timeline)
            lines.append(f"Memory timeline: {len(self.memory_timeline)} entries, "
                         f"max allocated = {max_alloc:.1f} MB")
            lines.append("")

        # Peak allocation
        if self.peak_allocation_stack:
            lines.append("Peak allocation stack trace:")
            # Show first 5 lines
            stack_lines = self.peak_allocation_stack.strip().split("\n")
            for sl in stack_lines[:5]:
                lines.append(f"  {sl}")
            if len(stack_lines) > 5:
                lines.append(f"  ... ({len(stack_lines) - 5} more lines)")
            lines.append("")

        # Loop detection
        lines.append(f"Loop detection: {self.loop_detection_method.value}")
        if self.loop_detection_confidence:
            lines.append(f"Loop detection confidence: {self.loop_detection_confidence}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON storage."""
        d = asdict(self)
        d["loop_detection_method"] = self.loop_detection_method.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProfileResult:
        """Deserialize from dict."""
        d = d.copy()
        d["top_operators"] = [OperatorProfile(**op) for op in d.get("top_operators", [])]
        d["memory_timeline"] = [
            MemoryTimelineEntry(**e) for e in d.get("memory_timeline", [])
        ]
        d["backward_ops"] = [BackwardOpProfile(**op) for op in d.get("backward_ops", [])]
        d["loop_detection_method"] = LoopDetectionMethod(
            d.get("loop_detection_method", "none")
        )
        return cls(**d)


@dataclass
class OptimizationConfig:
    """A proposed optimization configuration from the LLM agent."""
    id: str
    name: str
    description: str
    optimization_type: OptimizationType
    evidence: list[str]
    code_changes: dict[str, str] = field(default_factory=dict)
    config_overrides: dict[str, Any] = field(default_factory=dict)
    estimated_speedup: float = 1.0
    estimated_memory_delta: float = 0.0
    risk_level: RiskLevel = RiskLevel.MEDIUM
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON storage."""
        d = asdict(self)
        d["optimization_type"] = self.optimization_type.value
        d["risk_level"] = self.risk_level.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OptimizationConfig:
        """Deserialize from dict."""
        d = d.copy()
        d["optimization_type"] = OptimizationType(d["optimization_type"])
        d["risk_level"] = RiskLevel(d["risk_level"])
        return cls(**d)


@dataclass
class ControlRun:
    """Results from the unmodified control run during validation."""
    steps_completed: int
    wall_clock_seconds: float
    avg_step_time_ms: float
    peak_memory_mb: float
    throughput_samples_sec: float
    loss_values: list[float]
    gradient_norms: list[float]
    cost_estimate_usd: float


@dataclass
class ValidationResult:
    """Results from validating a single optimization config."""
    config_id: str
    config_name: str
    success: bool
    error: Optional[str] = None

    # Performance deltas (vs control)
    speedup_vs_control: float = 1.0
    memory_delta_vs_control: float = 0.0
    cost_delta_vs_control: float = 0.0

    # Raw metrics
    wall_clock_seconds: float = 0.0
    avg_step_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    throughput_samples_sec: float = 0.0
    loss_values: list[float] = field(default_factory=list)

    # Stability
    gradient_cosine_similarities: list[float] = field(default_factory=list)
    diverged: bool = False
    divergence_step: Optional[int] = None
    divergence_reason: str = ""


@dataclass
class RunConfig:
    """Merged CLI + YAML configuration for a GPUnity run."""
    repo_path: str
    entry_point: str = "train.py"
    train_function: Optional[str] = None
    provider: str = "claude"
    model: Optional[str] = None
    gpu_type: str = "a100-80gb"
    profile_steps: int = 20
    warmup_steps: int = 5
    validation_steps: int = 50
    max_configs: int = 10
    top_k: int = 5
    divergence_threshold: float = 0.8
    gradient_check_interval: int = 5
    max_cost_per_sandbox: float = 5.0
    output_path: str = "./gpunity_report.md"
    charts_mode: str = "ascii"
    verbose: bool = False
    dry_run: bool = False
    local: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        """Deserialize from dict."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
