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
    MEMORY_FORMAT = "memory_format"
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

    # Bottleneck diagnosis inputs
    h2d_transfer_time_ms: float = 0.0
    compile_warmup_time_s: float = 0.0
    gpu_active_ratio: float = 0.0
    memory_utilization_ratio: float = 0.0
    cpu_data_pipeline_ms: float = 0.0
    total_vram_mb: float = 0.0

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

        # Bottleneck diagnosis inputs
        if self.gpu_active_ratio > 0:
            lines.append(f"GPU active ratio: {self.gpu_active_ratio:.1%}")
        if self.h2d_transfer_time_ms > 0:
            lines.append(f"H2D transfer time: {self.h2d_transfer_time_ms:.2f} ms/step")
        if self.compile_warmup_time_s > 0:
            lines.append(f"Compile warmup time: {self.compile_warmup_time_s:.1f} s")
        if self.memory_utilization_ratio > 0:
            lines.append(f"Memory utilization: {self.memory_utilization_ratio:.1%}")
        if self.cpu_data_pipeline_ms > 0:
            lines.append(f"CPU data pipeline time: {self.cpu_data_pipeline_ms:.2f} ms/step")
        if any([self.gpu_active_ratio, self.h2d_transfer_time_ms,
                self.compile_warmup_time_s, self.memory_utilization_ratio]):
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
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        d = {k: v for k, v in d.items() if k in valid_fields}
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
    is_native_baseline: bool = False
    baseline_id: Optional[str] = None
    eligible: bool = True
    ineligible_reason: Optional[str] = None
    heuristic_rationale: list[str] = field(default_factory=list)

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
    throughput_gain_vs_control: float = 0.0
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
    logits_checks_compared: int = 0
    logits_max_abs_diff: Optional[float] = None
    logits_mean_abs_diff: Optional[float] = None
    logits_within_tolerance: Optional[bool] = None

    # Baseline and objective attribution
    is_native_baseline: bool = False
    baseline_id: Optional[str] = None
    objective_score: Optional[float] = None
    risk_penalty: float = 0.0
    speedup_vs_best_native: Optional[float] = None
    cost_delta_vs_best_native: Optional[float] = None
    throughput_gain_vs_best_native: Optional[float] = None
    evidence_chain: list[str] = field(default_factory=list)


@dataclass
class BaselineResult:
    """Outcome for one native baseline candidate."""
    baseline_id: str
    name: str
    eligible: bool
    success: bool
    skip_reason: Optional[str] = None
    speedup_vs_control: Optional[float] = None
    cost_delta_vs_control: Optional[float] = None
    throughput_gain_vs_control: Optional[float] = None
    objective_score: Optional[float] = None


@dataclass
class HardwareProfile:
    """Detected hardware and runtime feature support."""
    gpu_name: str
    gpu_count: int = 1
    compute_capability: Optional[str] = None
    vram_mb: Optional[int] = None
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None
    supports_bf16: bool = False
    supports_tf32: bool = False
    supports_fp16: bool = True
    supports_cuda_graphs: bool = False
    supports_compile: bool = False
    detection_source: str = "unknown"
    confidence: str = "low"
    notes: list[str] = field(default_factory=list)


@dataclass
class EvidenceEdge:
    """Trace link from a signal to an optimization decision."""
    source: str
    target: str
    relation: str
    detail: str


@dataclass
class RunSummary:
    """Top-line objective and baseline comparison summary for a run."""
    objective_profile: str = "balanced"
    best_overall_config_id: Optional[str] = None
    best_native_baseline_id: Optional[str] = None
    speedup_vs_best_native: Optional[float] = None
    cost_delta_vs_best_native: Optional[float] = None
    throughput_gain_vs_best_native: Optional[float] = None
    objective_weights: dict[str, float] = field(default_factory=dict)
    baseline_results: list[BaselineResult] = field(default_factory=list)
    evidence_edges: list[EvidenceEdge] = field(default_factory=list)
    confidence: str = "medium"
    bottleneck_type: Optional[str] = None
    bottleneck_evidence: list[str] = field(default_factory=list)
    gpu_selection_results: list[dict[str, Any]] = field(default_factory=list)
    recommended_gpu: Optional[str] = None


@dataclass
class RunConfig:
    """Merged CLI + YAML configuration for a GPUnity run."""
    repo_path: str
    entry_point: str = "train.py"
    train_function: Optional[str] = None
    provider: str = "claude"
    model: Optional[str] = None
    mode: str = "auto"
    baseline_policy: str = "required"
    compare_against_native: bool = True
    hardware_profile: str = "auto"
    objective_profile: str = "balanced"
    ablation_top_k: int = 3
    gpu_type: str = "a100-80gb"
    gpu_count: int = 1
    profile_steps: int = 20
    warmup_steps: int = 5
    validation_steps: int = 50
    max_configs: int = 10
    top_k: int = 5
    divergence_threshold: float = 0.8
    logits_tolerance: float = 1e-3
    gradient_check_interval: int = 5
    validation_seed: int = 1337
    deterministic_validation: bool = True
    validation_strategy: str = "parallel_all"
    staged_top_k: int = 2
    max_cost_per_sandbox: float = 5.0
    output_path: str = "./gpunity_report.md"
    charts_mode: str = "ascii"
    verbose: bool = False
    dry_run: bool = False
    local: bool = False
    python_bin: Optional[str] = None
    gpu_selection: str = "auto"
    gpu_aggressiveness: str = "moderate"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        """Deserialize from dict."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
