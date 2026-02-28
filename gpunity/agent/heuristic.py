"""Rule-based optimization config generator.

Used for local/demo runs when API-backed LLM providers are unavailable.
"""

from __future__ import annotations

from pathlib import Path

from gpunity.types import (
    OptimizationConfig,
    OptimizationType,
    ProfileResult,
    RiskLevel,
)


_ATTN_KEYWORDS = (
    "attention",
    "attn",
    "scaled_dot_product_attention",
    "sdpa",
    "qkv",
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


def generate_heuristic_configs(
    profile: ProfileResult,
    repo_path: Path,
    max_configs: int = 10,
) -> list[OptimizationConfig]:
    """Generate optimization configs directly from profile signals."""
    configs: list[OptimizationConfig] = []

    def add(config: OptimizationConfig) -> None:
        if len(configs) < max_configs:
            configs.append(config)

    has_attention, attn_evidence = _has_attention_signal(profile)
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

    return configs[:max_configs]
