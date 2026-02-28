"""Hardware-aware candidate prioritization policy."""

from __future__ import annotations

from gpunity.types import HardwareProfile, OptimizationConfig, ProfileResult


def apply_hardware_priors(
    configs: list[OptimizationConfig],
    hardware: HardwareProfile,
    profile: ProfileResult,
    mode: str,
) -> list[OptimizationConfig]:
    """Annotate and rank configs using safe hardware priors.

    Priors only change ordering and optional exclusion hints; empirical
    validation remains the source of truth.
    """
    ranked: list[tuple[float, OptimizationConfig]] = []
    for cfg in configs:
        score = cfg.estimated_speedup
        rationale: list[str] = []

        # Prefer bf16 only where supported.
        precision = str(cfg.config_overrides.get("precision", "")).lower()
        if precision in {"bf16", "bfloat16"} and not hardware.supports_bf16:
            score -= 0.25
            rationale.append("BF16 unsupported on detected hardware; deprioritized.")

        # Memory optimizations get higher priority on smaller VRAM cards.
        if (
            hardware.vram_mb is not None
            and hardware.vram_mb <= 24 * 1024
            and cfg.optimization_type.value in {"memory", "mixed_precision"}
        ):
            score += 0.10
            rationale.append("Limited VRAM profile; memory-focused optimization prioritized.")

        # Data pipeline optimizations are high-priority for explicit bottlenecks.
        if profile.dataloader_bottleneck and cfg.optimization_type.value == "data_loading":
            score += 0.15
            rationale.append("Profiler flagged DataLoader bottleneck.")

        # In inference mode, CUDA graph candidates are prioritized if supported.
        if mode == "infer" and cfg.config_overrides.get("cuda_graphs"):
            if hardware.supports_cuda_graphs:
                score += 0.12
                rationale.append("Inference mode + CUDA graphs supported.")
            else:
                score -= 0.30
                rationale.append("CUDA graphs unsupported on detected hardware.")

        if rationale:
            cfg.heuristic_rationale.extend(rationale)
            cfg.evidence.extend(rationale[:1])

        ranked.append((score, cfg))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [cfg for _, cfg in ranked]
