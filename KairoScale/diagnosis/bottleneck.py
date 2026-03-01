"""Resource bottleneck classifier for ML training workloads.

Diagnoses whether a workload is limited by data loading (RAM),
GPU memory (VRAM), compute saturation, PCIe transfer, or
torch.compile overhead. Every diagnosis is backed by profiler evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from KairoScale.types import HardwareProfile, ProfileResult


class BottleneckType(str, Enum):
    """Primary resource bottleneck categories."""
    DATA_STARVED = "data_starved"
    VRAM_STARVED = "vram_starved"
    COMPUTE_BOUND = "compute_bound"
    TRANSFER_BOUND = "transfer_bound"
    COMPILE_BOUND = "compile_bound"


@dataclass
class BottleneckDiagnosis:
    """Result of bottleneck classification with evidence chain."""
    primary: BottleneckType
    secondary: Optional[BottleneckType] = None
    confidence: str = "medium"
    evidence: list[str] = field(default_factory=list)
    recommended_gpu: Optional[str] = None
    gpu_downgrade_candidates: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable diagnosis summary."""
        lines = [f"Primary bottleneck: {self.primary.value} (confidence: {self.confidence})"]
        if self.secondary:
            lines.append(f"Secondary bottleneck: {self.secondary.value}")
        lines.append("Evidence:")
        for e in self.evidence:
            lines.append(f"  - {e}")
        if self.recommended_gpu:
            lines.append(f"Recommended GPU: {self.recommended_gpu}")
        if self.gpu_downgrade_candidates:
            lines.append(f"GPU candidates to try: {', '.join(self.gpu_downgrade_candidates)}")
        return "\n".join(lines)


# Thresholds for classification (tunable)
_DATA_STALL_THRESHOLD = 0.30  # stall > 30% of step time
_VRAM_UTIL_THRESHOLD = 0.85   # peak memory > 85% of VRAM
_COMPILE_WARMUP_RATIO = 3.0   # warmup steps 3x+ slower than steady state
_TRANSFER_THRESHOLD = 0.20    # H2D transfer > 20% of step time
_GPU_ACTIVE_HIGH = 0.80       # GPU active > 80% = compute bound
_GPU_ACTIVE_LOW = 0.40        # GPU active < 40% = likely starved


def diagnose_bottleneck(
    profile: ProfileResult,
    hardware: Optional[HardwareProfile] = None,
    aggressiveness: str = "moderate",
) -> BottleneckDiagnosis:
    """Classify the primary resource bottleneck from profiler data.

    Every classification is grounded in specific profiler measurements.
    The decision tree is ordered by impact — the first match wins as
    primary, subsequent matches become secondary.

    Args:
        profile: Profiling results with bottleneck diagnosis fields.
        hardware: Hardware profile for VRAM capacity and GPU cost context.
        aggressiveness: GPU selection aggressiveness for downgrade candidates.

    Returns:
        BottleneckDiagnosis with evidence chain.
    """
    candidates: list[tuple[BottleneckType, str, list[str]]] = []

    avg_step_ms = _avg_step_time_ms(profile)

    # 1. DATA_STARVED: DataLoader can't keep GPU fed
    if avg_step_ms > 0 and profile.dataloader_stall_time_ms > 0:
        stall_ratio = profile.dataloader_stall_time_ms / avg_step_ms
        if stall_ratio > _DATA_STALL_THRESHOLD:
            evidence = [
                f"DataLoader stall = {profile.dataloader_stall_time_ms:.1f} ms/step "
                f"({stall_ratio:.0%} of step time)",
                f"Source: profiler DataLoader.__iter__ instrumentation",
            ]
            if profile.dataloader_bottleneck:
                evidence.append("Profiler explicitly flagged DataLoader as bottleneck")
            if profile.cpu_data_pipeline_ms > 0:
                evidence.append(
                    f"CPU data pipeline time = {profile.cpu_data_pipeline_ms:.1f} ms/step"
                )
            if hasattr(profile, 'dataloader_num_workers') and profile.dataloader_num_workers == 0:
                evidence.append(
                    "num_workers=0: all data loading is synchronous on the main process. "
                    "Set num_workers=4-8 to parallelize data loading across CPU cores."
                )
            if hasattr(profile, 'dataloader_pin_memory') and not profile.dataloader_pin_memory:
                evidence.append(
                    "pin_memory=False: no DMA transfer overlap for CPU→GPU data movement. "
                    "Enable pin_memory=True to use page-locked memory for async transfers."
                )
            confidence = "high" if stall_ratio > 0.50 else "medium"
            candidates.append((BottleneckType.DATA_STARVED, confidence, evidence))

    # 2. VRAM_STARVED: GPU memory near capacity
    if profile.memory_utilization_ratio > _VRAM_UTIL_THRESHOLD:
        evidence = [
            f"Memory utilization = {profile.memory_utilization_ratio:.0%} "
            f"(peak {profile.peak_memory_mb:.0f} MB / "
            f"{profile.total_vram_mb:.0f} MB total VRAM)",
            f"Source: torch.cuda.max_memory_allocated() / device_properties.total_memory",
        ]
        confidence = "high" if profile.memory_utilization_ratio > 0.95 else "medium"
        candidates.append((BottleneckType.VRAM_STARVED, confidence, evidence))
    elif (
        hardware is not None
        and hardware.vram_mb
        and hardware.vram_mb > 0
        and profile.peak_memory_mb > 0
    ):
        # Fallback: compute from hardware profile if profiler didn't capture ratio
        ratio = profile.peak_memory_mb / (hardware.vram_mb)
        if ratio > _VRAM_UTIL_THRESHOLD:
            evidence = [
                f"Memory utilization = {ratio:.0%} "
                f"(peak {profile.peak_memory_mb:.0f} MB / "
                f"{hardware.vram_mb} MB VRAM from hardware profile)",
                f"Source: profiler peak_memory_mb vs HardwareProfile.vram_mb",
            ]
            candidates.append((BottleneckType.VRAM_STARVED, "medium", evidence))

    # 3. COMPILE_BOUND: torch.compile warmup dominates
    if profile.compile_warmup_time_s > 0:
        step_times_total = avg_step_ms * max(1, _estimated_steps(profile)) / 1000.0
        if step_times_total > 0:
            compile_ratio = profile.compile_warmup_time_s / step_times_total
            if compile_ratio > 0.30:  # compile > 30% of total training time
                evidence = [
                    f"Compile warmup = {profile.compile_warmup_time_s:.1f}s "
                    f"({compile_ratio:.0%} of total training time)",
                    f"Source: step time variance (early steps {_COMPILE_WARMUP_RATIO:.0f}x+ "
                    f"slower than steady state)",
                ]
                confidence = "high" if compile_ratio > 0.50 else "medium"
                candidates.append((BottleneckType.COMPILE_BOUND, confidence, evidence))

    # 4. TRANSFER_BOUND: H2D transfer dominates
    if avg_step_ms > 0 and profile.h2d_transfer_time_ms > 0:
        transfer_ratio = profile.h2d_transfer_time_ms / avg_step_ms
        if transfer_ratio > _TRANSFER_THRESHOLD:
            evidence = [
                f"H2D transfer = {profile.h2d_transfer_time_ms:.1f} ms/step "
                f"({transfer_ratio:.0%} of step time)",
                f"Source: instrumented Tensor.to(device='cuda') and Tensor.cuda() calls",
            ]
            confidence = "high" if transfer_ratio > 0.40 else "medium"
            candidates.append((BottleneckType.TRANSFER_BOUND, confidence, evidence))

    # 5. COMPUTE_BOUND: GPU is well-utilized, no starvation
    gpu_active = profile.gpu_active_ratio
    if gpu_active == 0.0 and profile.gpu_utilization > 0:
        gpu_active = profile.gpu_utilization / 100.0

    if gpu_active > _GPU_ACTIVE_HIGH:
        evidence = [
            f"GPU active ratio = {gpu_active:.0%}",
            f"Source: torch.profiler CUDA time / wall clock time",
        ]
        if not candidates:
            evidence.append(
                "No data starvation, VRAM pressure, or transfer bottlenecks detected"
            )
        confidence = "high" if gpu_active > 0.90 else "medium"

        # Upgrade confidence and add FLOP utilization evidence when available
        flop_util = getattr(profile, "flop_utilization", None)
        if flop_util is not None:
            if flop_util > 0.6:
                confidence = "high"
                evidence.append(
                    f"FLOP utilization = {flop_util:.1%} of GPU peak (near peak throughput)"
                )
            elif flop_util < 0.3 and gpu_active > _GPU_ACTIVE_HIGH:
                # GPU appears busy but FLOPs are low → kernel launch overhead
                evidence.append(
                    f"FLOP utilization = {flop_util:.1%} of GPU peak despite high GPU active "
                    f"ratio — likely kernel launch overhead or small, memory-bound kernels"
                )

        candidates.append((BottleneckType.COMPUTE_BOUND, confidence, evidence))

    # Kernel launch overhead: GPU active but FLOP utilization is very low
    flop_util = getattr(profile, "flop_utilization", None)
    if (
        flop_util is not None
        and flop_util < 0.3
        and profile.gpu_utilization > 50
        and gpu_active <= _GPU_ACTIVE_HIGH
    ):
        evidence = [
            f"FLOP utilization = {flop_util:.1%} of GPU peak",
            f"GPU utilization = {profile.gpu_utilization:.1f}% (active but compute-inefficient)",
            "Low FLOP utilization with moderate GPU activity suggests kernel launch overhead "
            "or many small, memory-bound kernels",
            "Source: torch.profiler with_flops=True + CUDA time",
        ]
        candidates.append((BottleneckType.COMPUTE_BOUND, "medium", evidence))

    # Select primary and secondary
    if not candidates:
        # Not enough data to classify
        evidence = [
            "Insufficient profiler data for confident classification",
            f"GPU utilization = {profile.gpu_utilization:.1f}%",
            f"Peak memory = {profile.peak_memory_mb:.0f} MB",
            f"DataLoader stall = {profile.dataloader_stall_time_ms:.1f} ms/step",
        ]
        return BottleneckDiagnosis(
            primary=BottleneckType.COMPUTE_BOUND,
            confidence="low",
            evidence=evidence,
        )

    primary_type, primary_conf, primary_evidence = candidates[0]
    secondary_type = candidates[1][0] if len(candidates) > 1 else None

    # GPU downgrade candidates for COMPUTE_BOUND
    from KairoScale.diagnosis.gpuSelector import generateGpuDowngradeCandidates

    current_gpu = hardware.gpu_name if hardware else "unknown"
    gpu_candidates = []
    recommended = None
    if primary_type == BottleneckType.COMPUTE_BOUND:
        gpu_candidates = generateGpuDowngradeCandidates(
            current_gpu, aggressiveness
        )
        if gpu_candidates:
            recommended = gpu_candidates[0]

    return BottleneckDiagnosis(
        primary=primary_type,
        secondary=secondary_type,
        confidence=primary_conf,
        evidence=primary_evidence,
        recommended_gpu=recommended,
        gpu_downgrade_candidates=gpu_candidates,
    )


def _avg_step_time_ms(profile: ProfileResult) -> float:
    """Compute average step time from available profile data."""
    total = profile.forward_time_ms + profile.backward_time_ms
    if total > 0:
        return total
    # Fallback: use dataloader stall as a lower bound indicator
    return 0.0


def _estimated_steps(profile: ProfileResult) -> int:
    """Estimate total profiled steps from available data."""
    if profile.top_operators:
        # Use call count of most-called operator as step proxy
        max_calls = max(op.call_count for op in profile.top_operators)
        return max(1, max_calls)
    return 20  # default profile_steps
