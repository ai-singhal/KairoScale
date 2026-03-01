"""Native baseline candidate generation and eligibility checks."""

from __future__ import annotations

from typing import Optional

from KairoScale.hardware.profile import hardware_supports_max_autotune
from KairoScale.types import (
    BaselineResult,
    HardwareProfile,
    OptimizationConfig,
    OptimizationType,
    ProfileResult,
    RiskLevel,
    RunConfig,
)


def _likely_graph_safe(profile: Optional[ProfileResult]) -> bool:
    if profile is None:
        return False
    if profile.loop_detection_method.value == "none":
        return False
    if profile.dataloader_bottleneck:
        return False
    return True


def _is_gpu_cluster(hardware: HardwareProfile) -> bool:
    count = hardware.gpu_count or 1
    return count > 1


def _baseline_config(
    baseline_id: str,
    name: str,
    description: str,
    overrides: dict[str, object],
    *,
    eligible: bool,
    ineligible_reason: Optional[str],
    heuristic_rationale: list[str],
) -> OptimizationConfig:
    return OptimizationConfig(
        id=baseline_id,
        name=name,
        description=description,
        optimization_type=OptimizationType.COMPILATION,
        evidence=[
            "Native baseline candidate required by benchmark ladder.",
            *heuristic_rationale[:2],
        ],
        config_overrides=overrides,
        estimated_speedup=1.0,
        estimated_memory_delta=0.0,
        risk_level=RiskLevel.LOW,
        is_native_baseline=True,
        baseline_id=baseline_id,
        eligible=eligible,
        ineligible_reason=ineligible_reason,
        heuristic_rationale=heuristic_rationale,
    )


def build_native_baseline_candidates(
    run_config: RunConfig,
    hardware: HardwareProfile,
    mode: str,
    profile: Optional[ProfileResult] = None,
) -> tuple[list[OptimizationConfig], list[BaselineResult]]:
    """Build native optimizer baseline ladder B1..B2.

    B0 is the control run and is represented implicitly by `ControlRun`.
    """
    baselines: list[OptimizationConfig] = []
    manifest: list[BaselineResult] = [
        BaselineResult(
            baseline_id="B0",
            name="Control (eager, no optimization)",
            eligible=True,
            success=True,
        )
    ]

    graph_safe = _likely_graph_safe(profile)
    is_cluster = _is_gpu_cluster(hardware)

    # B1: single-GPU baseline (max-autotune-no-cudagraphs, fallback default).
    b1_eligible = (not is_cluster) and hardware_supports_max_autotune(hardware)
    b1_reason = None
    if is_cluster:
        b1_reason = "Reserved for single-GPU runs; cluster baseline uses CUDA graphs variant"
    elif not hardware_supports_max_autotune(hardware):
        b1_reason = "max-autotune-no-cudagraphs gated by compile/hardware support"
    baselines.append(_baseline_config(
        "B1",
        "torch.compile(max-autotune-no-cudagraphs)",
        (
            "Single-GPU native compiler path: max-autotune-no-cudagraphs with "
            "runtime fallback to default."
        ),
        {
            "compile": True,
            "compile_mode": "max-autotune-no-cudagraphs",
            "cuda_graphs": False,
            "compile_fallback_mode": "default",
        },
        eligible=b1_eligible,
        ineligible_reason=b1_reason,
        heuristic_rationale=[
            f"Detected GPU count: {hardware.gpu_count or 1}",
            "Fallback mode: default",
        ],
    ))
    manifest.append(BaselineResult(
        baseline_id="B1",
        name="torch.compile(max-autotune-no-cudagraphs)",
        eligible=b1_eligible,
        success=False,
        skip_reason=b1_reason,
    ))

    # B2: cluster baseline (max-autotune + CUDA graphs, fallback reduce-overhead).
    b2_graph_eligible = False
    b2_graph_reason = None
    if mode == "infer":
        b2_graph_eligible = hardware.supports_cuda_graphs
        if not b2_graph_eligible:
            b2_graph_reason = "CUDA graphs not supported on detected runtime"
    else:
        b2_graph_eligible = hardware.supports_cuda_graphs and graph_safe
        if not hardware.supports_cuda_graphs:
            b2_graph_reason = "CUDA graphs not supported on detected runtime"
        elif not graph_safe:
            b2_graph_reason = "Graph-safety/static-shape check failed for training workload"

    b2_eligible = is_cluster and hardware_supports_max_autotune(hardware) and b2_graph_eligible
    b2_reason = None
    if not is_cluster:
        b2_reason = "Reserved for multi-GPU cluster runs"
    elif not hardware_supports_max_autotune(hardware):
        b2_reason = "max-autotune gated by compile/hardware support"
    elif not b2_graph_eligible:
        b2_reason = b2_graph_reason or "CUDA graphs unsupported"
    baselines.append(_baseline_config(
        "B2",
        "torch.compile(max-autotune + CUDA Graphs)",
        (
            "Cluster native compiler path: max-autotune with CUDA graphs and "
            "runtime fallback to reduce-overhead."
        ),
        {
            "compile": True,
            "compile_mode": "max-autotune",
            "cuda_graphs": True,
            "compile_fallback_mode": "reduce-overhead",
        },
        eligible=b2_eligible,
        ineligible_reason=b2_reason,
        heuristic_rationale=[
            f"Detected GPU count: {hardware.gpu_count or 1}",
            f"Graph-safe check: {graph_safe}",
            "Fallback mode: reduce-overhead",
        ],
    ))
    manifest.append(BaselineResult(
        baseline_id="B2",
        name="torch.compile(max-autotune + CUDA Graphs)",
        eligible=b2_eligible,
        success=False,
        skip_reason=b2_reason,
    ))

    if run_config.baseline_policy == "minimal":
        kept = {"B2"} if is_cluster else {"B1"}
        baselines = [b for b in baselines if b.baseline_id in kept]
        manifest = [m for m in manifest if m.baseline_id == "B0" or m.baseline_id in kept]

    return baselines, manifest
