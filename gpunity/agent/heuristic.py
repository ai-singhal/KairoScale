"""Rule-based optimization config generator.

Used for local/demo runs when API-backed LLM providers are unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gpunity.optimizer.policy import apply_hardware_priors
from gpunity.types import (
    HardwareProfile,
    OptimizationConfig,
    OptimizationType,
    ProfileResult,
    RiskLevel,
)

if TYPE_CHECKING:
    from gpunity.diagnosis.bottleneck import BottleneckDiagnosis


_ATTN_KEYWORDS = (
    "attention",
    "attn",
    "scaled_dot_product_attention",
    "sdpa",
    "qkv",
)

_CONV_KEYWORDS = (
    "conv2d",
    "conv3d",
    "convolution",
    "cudnn_convolution",
    "aten::conv2d",
    "aten::conv3d",
    "aten::convolution",
    "aten::cudnn_convolution",
)

_CONV3D_KEYWORDS = (
    "conv3d",
    "aten::conv3d",
)

_OPTIMIZER_KEYWORDS = (
    "adam",
    "adamw",
    "sgd",
    "rmsprop",
    "adafactor",
    "muon",
    "optimizer",
)


def _has_attention_signal(profile: ProfileResult) -> tuple[bool, list[str]]:
    for op in profile.top_operators:
        name = op.name.lower()
        if any(k in name for k in _ATTN_KEYWORDS):
            return True, [
                f"Operator `{op.name}` accounts for {op.pct_total:.1f}% of GPU time",
                "Attention-related kernels appear in top operators",
            ]
    return False, []


def _has_optimizer_signal(profile: ProfileResult) -> tuple[bool, list[str]]:
    for op in profile.top_operators:
        name = op.name.lower()
        if any(k in name for k in _OPTIMIZER_KEYWORDS):
            return True, [
                f"Operator `{op.name}` appears in top kernels at {op.pct_total:.1f}%",
                "Optimizer-related kernels are a measurable runtime hotspot",
            ]

    if profile.backward_time_ms > 0 and profile.forward_time_ms > 0:
        ratio = profile.backward_time_ms / max(profile.forward_time_ms, 1e-6)
        if ratio >= 1.6:
            return True, [
                f"Backward/forward ratio is {ratio:.2f}x",
                "High backward dominance can benefit from optimizer/kernel improvements",
            ]
    return False, []


def _has_conv_signal(profile: ProfileResult) -> tuple[bool, bool, float, list[str]]:
    """Detect convolution-heavy workloads.

    Returns:
        (has_conv, is_3d, conv_pct, evidence)
    """
    conv_pct = 0.0
    is_3d = False
    evidence = []
    for op in profile.top_operators:
        name = op.name.lower()
        if any(k in name for k in _CONV_KEYWORDS):
            conv_pct += op.pct_total
            evidence.append(
                f"Operator `{op.name}` accounts for {op.pct_total:.1f}% of GPU time"
            )
            if any(k in name for k in _CONV3D_KEYWORDS):
                is_3d = True
    if conv_pct > 0:
        evidence.insert(0, f"Conv ops total {conv_pct:.1f}% of GPU time")
        return True, is_3d, conv_pct, evidence
    return False, False, 0.0, []


def _add_bottleneck_configs(
    diagnosis: BottleneckDiagnosis,
    profile: ProfileResult,
    configs: list[OptimizationConfig],
    add,
    max_configs: int,
) -> None:
    """Add optimization configs targeted at the diagnosed bottleneck.

    Each config cites the bottleneck diagnosis evidence so the
    recommendation chain is traceable.
    """
    from gpunity.diagnosis.bottleneck import BottleneckType

    bt = diagnosis.primary
    evidence = list(diagnosis.evidence)

    if bt == BottleneckType.DATA_STARVED:
        add(OptimizationConfig(
            id=f"opt-{len(configs) + 1:03d}",
            name="Fix data starvation: increase DataLoader workers + pin memory",
            description=(
                "Bottleneck diagnosis: GPU is idle waiting for data. "
                "Increasing num_workers and enabling pin_memory reduces "
                "host-side data loading stalls."
            ),
            optimization_type=OptimizationType.DATA_LOADING,
            evidence=evidence,
            config_overrides={
                "dataloader_num_workers": 8,
                "dataloader_pin_memory": True,
            },
            estimated_speedup=1.30,
            estimated_memory_delta=0.0,
            risk_level=RiskLevel.LOW,
            heuristic_rationale=[
                f"Diagnosed as {bt.value}",
                "More workers + pinned memory reduces DataLoader stall time",
            ],
        ))
        add(OptimizationConfig(
            id=f"opt-{len(configs) + 1:03d}",
            name="Fix data starvation: prefetch + async data pipeline",
            description=(
                "Bottleneck diagnosis: data pipeline is the bottleneck. "
                "Enable prefetch_factor and persistent workers to keep the "
                "data pipeline ahead of GPU consumption."
            ),
            optimization_type=OptimizationType.DATA_LOADING,
            evidence=evidence,
            config_overrides={
                "dataloader_num_workers": 8,
                "dataloader_pin_memory": True,
                "dataloader_prefetch_factor": 4,
                "dataloader_persistent_workers": True,
            },
            estimated_speedup=1.40,
            estimated_memory_delta=0.05,
            risk_level=RiskLevel.LOW,
            heuristic_rationale=[
                f"Diagnosed as {bt.value}",
                "Prefetch keeps GPU fed while CPU prepares next batches",
            ],
        ))

    elif bt == BottleneckType.VRAM_STARVED:
        add(OptimizationConfig(
            id=f"opt-{len(configs) + 1:03d}",
            name="Fix VRAM starvation: enable mixed precision (bf16)",
            description=(
                "Bottleneck diagnosis: GPU memory near capacity. "
                "Mixed precision halves activation memory and enables "
                "tensor core throughput gains."
            ),
            optimization_type=OptimizationType.MIXED_PRECISION,
            evidence=evidence,
            config_overrides={"amp": True, "precision": "bf16"},
            estimated_speedup=1.20,
            estimated_memory_delta=-0.40,
            risk_level=RiskLevel.LOW,
            heuristic_rationale=[
                f"Diagnosed as {bt.value}",
                "bf16 reduces activation memory by ~50%",
            ],
        ))
        add(OptimizationConfig(
            id=f"opt-{len(configs) + 1:03d}",
            name="Fix VRAM starvation: gradient checkpointing + AMP",
            description=(
                "Bottleneck diagnosis: peak memory near VRAM limit. "
                "Gradient checkpointing trades compute for memory, "
                "combined with AMP for maximum memory savings."
            ),
            optimization_type=OptimizationType.MEMORY,
            evidence=evidence,
            config_overrides={
                "gradient_checkpointing": True,
                "amp": True,
                "precision": "bf16",
            },
            estimated_speedup=1.05,
            estimated_memory_delta=-0.50,
            risk_level=RiskLevel.MEDIUM,
            heuristic_rationale=[
                f"Diagnosed as {bt.value}",
                "Checkpointing + AMP can free 50%+ of activation memory",
            ],
        ))

    elif bt == BottleneckType.TRANSFER_BOUND:
        add(OptimizationConfig(
            id=f"opt-{len(configs) + 1:03d}",
            name="Fix PCIe bottleneck: pin memory + non-blocking transfers",
            description=(
                "Bottleneck diagnosis: H2D transfer time dominates step time. "
                "Pinned memory and non_blocking=True on .to(device) calls "
                "enable overlapped CPU→GPU transfers."
            ),
            optimization_type=OptimizationType.DATA_LOADING,
            evidence=evidence,
            config_overrides={
                "dataloader_pin_memory": True,
                "dataloader_num_workers": 4,
            },
            estimated_speedup=1.25,
            estimated_memory_delta=0.0,
            risk_level=RiskLevel.LOW,
            heuristic_rationale=[
                f"Diagnosed as {bt.value}",
                "Pinned memory enables async DMA transfers over PCIe",
            ],
        ))

    elif bt == BottleneckType.COMPILE_BOUND:
        add(OptimizationConfig(
            id=f"opt-{len(configs) + 1:03d}",
            name="Fix compile overhead: use reduce-overhead mode",
            description=(
                "Bottleneck diagnosis: torch.compile warmup dominates runtime. "
                "Switching from max-autotune to reduce-overhead mode cuts "
                "compilation time while retaining most kernel fusion benefits."
            ),
            optimization_type=OptimizationType.COMPILATION,
            evidence=evidence,
            config_overrides={
                "compile": True,
                "compile_mode": "reduce-overhead",
            },
            estimated_speedup=1.15,
            estimated_memory_delta=-0.05,
            risk_level=RiskLevel.LOW,
            heuristic_rationale=[
                f"Diagnosed as {bt.value}",
                "reduce-overhead compiles faster than max-autotune",
            ],
        ))
        add(OptimizationConfig(
            id=f"opt-{len(configs) + 1:03d}",
            name="Skip compile: eager mode with AMP",
            description=(
                "Bottleneck diagnosis: compile overhead too high for this "
                "workload size. Run in eager mode with AMP for immediate "
                "speedup without compilation cost."
            ),
            optimization_type=OptimizationType.MIXED_PRECISION,
            evidence=evidence,
            config_overrides={
                "compile": False,
                "amp": True,
                "precision": "bf16",
            },
            estimated_speedup=1.10,
            estimated_memory_delta=-0.30,
            risk_level=RiskLevel.LOW,
            heuristic_rationale=[
                f"Diagnosed as {bt.value}",
                "Eager + AMP avoids compile overhead entirely",
            ],
        ))

    elif bt == BottleneckType.COMPUTE_BOUND:
        # GPU is well-utilized. Not much to optimize on this hardware.
        # The main win is trying cheaper hardware (handled by GPU selector).
        if diagnosis.gpu_downgrade_candidates:
            add(OptimizationConfig(
                id=f"opt-{len(configs) + 1:03d}",
                name="Compute-bound: try cheaper GPU for cost savings",
                description=(
                    "Bottleneck diagnosis: GPU is fully utilized with no "
                    "starvation. The workload may run at similar speed on "
                    f"cheaper hardware. Candidates: "
                    f"{', '.join(diagnosis.gpu_downgrade_candidates)}."
                ),
                optimization_type=OptimizationType.COMPILATION,
                evidence=evidence + [
                    f"GPU downgrade candidates: {diagnosis.gpu_downgrade_candidates}"
                ],
                config_overrides={},
                estimated_speedup=1.0,
                estimated_memory_delta=0.0,
                risk_level=RiskLevel.LOW,
                heuristic_rationale=[
                    f"Diagnosed as {bt.value}",
                    "GPU cost reduction is the primary optimization lever",
                ],
            ))


def generate_heuristic_configs(
    profile: ProfileResult,
    repo_path: Path,
    max_configs: int = 10,
    hardware_profile: HardwareProfile | None = None,
    mode: str = "train",
    diagnosis: BottleneckDiagnosis | None = None,
) -> list[OptimizationConfig]:
    """Generate optimization configs directly from profile signals.

    When a BottleneckDiagnosis is provided, configs are prioritized
    based on the diagnosed bottleneck type rather than just operator
    pattern matching.
    """
    configs: list[OptimizationConfig] = []

    def add(config: OptimizationConfig) -> None:
        if len(configs) < max_configs:
            configs.append(config)

    # Bottleneck-driven configs (highest priority — added first)
    if diagnosis is not None:
        _add_bottleneck_configs(diagnosis, profile, configs, add, max_configs)

    has_attention, attn_evidence = _has_attention_signal(profile)
    has_optimizer, optimizer_evidence = _has_optimizer_signal(profile)
    has_conv, is_3d_conv, conv_pct, conv_evidence = _has_conv_signal(profile)

    if has_conv:
        mem_format = "channels_last_3d" if is_3d_conv else "channels_last"
        dim_label = "3D" if is_3d_conv else "2D"
        add(OptimizationConfig(
            id=f"opt-{len(configs) + 1:03d}",
            name=f"Convert model to {mem_format} memory format",
            description=(
                f"Profile shows {dim_label} convolution ops dominating GPU time "
                f"({conv_pct:.1f}%). Converting model and input tensors to "
                f"{mem_format} memory format enables hardware-optimized NHWC/NDHWC "
                f"kernels for 10-30% speedup on conv-heavy workloads."
            ),
            optimization_type=OptimizationType.MEMORY_FORMAT,
            evidence=conv_evidence,
            config_overrides={"memory_format": mem_format},
            estimated_speedup=1.20,
            estimated_memory_delta=0.0,
            risk_level=RiskLevel.LOW,
            heuristic_rationale=[
                f"Conv ops occupy {conv_pct:.1f}% of GPU time",
                f"{mem_format} enables hardware-native NHWC layout for conv kernels",
            ],
        ))

        add(OptimizationConfig(
            id=f"opt-{len(configs) + 1:03d}",
            name="Enable cuDNN auto-tuner (benchmark mode)",
            description=(
                f"Conv ops dominate GPU time ({conv_pct:.1f}%). Enabling "
                "cudnn.benchmark lets cuDNN auto-select the fastest convolution "
                "algorithm for the given input shapes, typically yielding 5-15% "
                "speedup for fixed-size conv workloads."
            ),
            optimization_type=OptimizationType.COMPILATION,
            evidence=conv_evidence,
            config_overrides={"cudnn_benchmark": True},
            estimated_speedup=1.10,
            estimated_memory_delta=0.0,
            risk_level=RiskLevel.LOW,
            heuristic_rationale=[
                "cuDNN benchmark auto-selects fastest conv algorithm",
                "Most effective for fixed-shape inputs (no dynamic shapes)",
            ],
        ))

    if has_attention:
        add(OptimizationConfig(
            id=f"opt-{len(configs) + 1:03d}",
            name="Enable Flash Attention path",
            description=(
                "Switch attention ops to Flash Attention or memory-efficient SDPA "
                "implementation where available."
            ),
            optimization_type=OptimizationType.ATTENTION,
            evidence=attn_evidence,
            config_overrides={"attention_backend": "flash"},
            estimated_speedup=1.35,
            estimated_memory_delta=-0.15,
            risk_level=RiskLevel.MEDIUM,
            dependencies=["flash-attn>=2.5"],
        ))

    if profile.dataloader_bottleneck:
        add(OptimizationConfig(
            id=f"opt-{len(configs) + 1:03d}",
            name="Optimize DataLoader pipeline",
            description=(
                "Increase dataloader workers, enable pinned memory, and pre-tokenize "
                "inputs to reduce host-side stalls."
            ),
            optimization_type=OptimizationType.DATA_LOADING,
            evidence=[
                f"DataLoader stall time is {profile.dataloader_stall_time_ms:.2f} ms/step",
                "Profile marked DataLoader as a bottleneck",
            ],
            config_overrides={
                "dataloader_num_workers": 4,
                "dataloader_pin_memory": True,
                "pretokenize": True,
            },
            estimated_speedup=1.25,
            estimated_memory_delta=0.0,
            risk_level=RiskLevel.LOW,
        ))

    if has_optimizer:
        add(OptimizationConfig(
            id=f"opt-{len(configs) + 1:03d}",
            name="Swap optimizer + Triton fused kernels",
            description=(
                "Evaluate SGD/RMSProp/AdamW/Adafactor/MUON variants plus Triton-fused "
                "optimizer kernels where available."
            ),
            optimization_type=OptimizationType.KERNEL_FUSION,
            evidence=optimizer_evidence,
            config_overrides={
                "optimizer_strategy": "fused_triton_search",
                "optimizer_candidates": ["sgd", "rmsprop", "adamw", "adafactor", "muon"],
            },
            estimated_speedup=1.12,
            estimated_memory_delta=-0.05,
            risk_level=RiskLevel.MEDIUM,
        ))

    if profile.gpu_utilization > 0 and profile.gpu_utilization < 70:
        add(OptimizationConfig(
            id=f"opt-{len(configs) + 1:03d}",
            name="Enable torch.compile",
            description=(
                "Use `torch.compile` in reduce-overhead mode to fuse kernels and "
                "reduce launch overhead."
            ),
            optimization_type=OptimizationType.COMPILATION,
            evidence=[
                f"GPU utilization is {profile.gpu_utilization:.1f}%",
                "Lower utilization often indicates kernel launch/graph overhead",
            ],
            config_overrides={"compile": True, "compile_mode": "reduce-overhead"},
            estimated_speedup=1.20,
            estimated_memory_delta=-0.05,
            risk_level=RiskLevel.LOW,
        ))

    if mode == "infer" and has_attention:
        add(OptimizationConfig(
            id=f"opt-{len(configs) + 1:03d}",
            name="Enable MLA/n-gram/float8 inference stack",
            description=(
                "Try MLA-style attention path, n-gram/speculative cache reuse, and "
                "float8-capable kernels for inference-focused attention hotspots."
            ),
            optimization_type=OptimizationType.ATTENTION,
            evidence=attn_evidence,
            config_overrides={
                "attention_backend": "flash",
                "inference_attention": "mla",
                "inference_ngram_cache": True,
                "inference_precision": "float8",
            },
            estimated_speedup=1.18,
            estimated_memory_delta=-0.20,
            risk_level=RiskLevel.MEDIUM,
            dependencies=["flash-attn>=2.5"],
        ))

    if profile.peak_memory_mb > 4096:
        add(OptimizationConfig(
            id=f"opt-{len(configs) + 1:03d}",
            name="Switch to bf16 mixed precision",
            description=(
                "Enable automatic mixed precision (bf16 preferred on modern GPUs) "
                "to reduce activation memory and improve tensor core throughput."
            ),
            optimization_type=OptimizationType.MIXED_PRECISION,
            evidence=[
                f"Peak memory reached {profile.peak_memory_mb:.1f} MB",
                "Mixed precision is a low-risk path for memory and throughput gains",
            ],
            config_overrides={"precision": "bf16", "amp": True},
            estimated_speedup=1.15,
            estimated_memory_delta=-0.30,
            risk_level=RiskLevel.LOW,
        ))

    if profile.peak_memory_mb > 0:
        add(OptimizationConfig(
            id=f"opt-{len(configs) + 1:03d}",
            name="Enable selective gradient checkpointing",
            description=(
                "Checkpoint high-memory blocks to trade extra compute for lower peak "
                "activation memory."
            ),
            optimization_type=OptimizationType.MEMORY,
            evidence=[
                f"Peak memory reached {profile.peak_memory_mb:.1f} MB",
                "Activation memory pressure can limit larger batch sizes",
            ],
            config_overrides={"gradient_checkpointing": True},
            estimated_speedup=0.95,
            estimated_memory_delta=-0.35,
            risk_level=RiskLevel.MEDIUM,
        ))

    # Stacked/combo config: combine top compatible low-risk optimizations
    if has_conv and len(configs) < max_configs:
        mem_format = "channels_last_3d" if is_3d_conv else "channels_last"
        dim_label = "3D" if is_3d_conv else "2D"
        combo_overrides = {
            "compile": True,
            "compile_mode": "max-autotune",
            "amp": True,
            "precision": "bf16",
            "memory_format": mem_format,
            "cudnn_benchmark": True,
        }
        combo_evidence = list(conv_evidence)
        combo_evidence.append(
            "Stacking compile + AMP + memory format + cuDNN benchmark "
            "for compound speedup"
        )
        add(OptimizationConfig(
            id=f"opt-{len(configs) + 1:03d}",
            name=f"Stacked {dim_label} CNN optimization combo",
            description=(
                f"Combines torch.compile (max-autotune), bf16 mixed precision, "
                f"{mem_format} memory format, and cuDNN benchmark mode. "
                f"Individual optimizations yield 5-20% each; stacked together "
                f"they can compound to 30-50%+ on conv-heavy workloads."
            ),
            optimization_type=OptimizationType.COMPILATION,
            evidence=combo_evidence,
            config_overrides=combo_overrides,
            estimated_speedup=1.40,
            estimated_memory_delta=-0.20,
            risk_level=RiskLevel.MEDIUM,
            heuristic_rationale=[
                "Compound gains from stacking compatible optimizations",
                f"Conv ops at {conv_pct:.1f}% are primary target",
                "max-autotune explores more fusion opportunities",
            ],
        ))
    elif not has_conv and len(configs) < max_configs:
        # Generic stacked config for non-conv workloads
        combo_overrides = {
            "compile": True,
            "compile_mode": "max-autotune",
            "amp": True,
            "precision": "bf16",
        }
        combo_evidence = [
            "Stacking compile + AMP for compound speedup on general workloads"
        ]
        if has_attention:
            combo_evidence.extend(attn_evidence)
        add(OptimizationConfig(
            id=f"opt-{len(configs) + 1:03d}",
            name="Stacked compile + AMP optimization combo",
            description=(
                "Combines torch.compile (max-autotune) with bf16 mixed precision. "
                "These are broadly compatible and typically yield 15-30% combined "
                "speedup."
            ),
            optimization_type=OptimizationType.COMPILATION,
            evidence=combo_evidence,
            config_overrides=combo_overrides,
            estimated_speedup=1.25,
            estimated_memory_delta=-0.15,
            risk_level=RiskLevel.MEDIUM,
            heuristic_rationale=[
                "Compound gains from stacking compile + AMP",
            ],
        ))

    if not configs:
        add(OptimizationConfig(
            id="opt-001",
            name="Baseline compile + amp sweep",
            description=(
                "No dominant bottleneck was detected; start with low-risk runtime "
                "knobs (`torch.compile`, AMP) and validate."
            ),
            optimization_type=OptimizationType.COMPILATION,
            evidence=[
                "No single dominant operator/bottleneck signal in profile",
                "Runtime-level optimizations are the safest first pass",
            ],
            config_overrides={"compile": True, "amp": True},
            estimated_speedup=1.10,
            estimated_memory_delta=-0.10,
            risk_level=RiskLevel.LOW,
        ))

    configs = configs[:max_configs]
    if hardware_profile is not None:
        return apply_hardware_priors(
            configs=configs,
            hardware=hardware_profile,
            profile=profile,
            mode=mode,
        )
    return configs
