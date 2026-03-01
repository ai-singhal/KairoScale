"""GPU selection and cost-optimized downgrade logic.

When a workload is compute-bound (GPU fully saturated), this module
generates cheaper GPU candidates to test. Modal only allows changing
the GPU type, so this is the primary hardware selection lever.
"""

from __future__ import annotations

from KairoScale.utils.cost import GPU_COSTS_PER_HOUR


# GPU ladder ordered by cost (descending). Each entry maps to a Modal GPU type.
# Evidence source: Modal pricing from KairoScale/utils/cost.py
_GPU_LADDER: list[dict[str, object]] = [
    {"id": "h100", "cost_per_hr": 4.98, "vram_mb": 80 * 1024, "compute_capability": "9.0"},
    {"id": "a100-80gb", "cost_per_hr": 3.73, "vram_mb": 80 * 1024, "compute_capability": "8.0"},
    {"id": "a100-40gb", "cost_per_hr": 2.78, "vram_mb": 40 * 1024, "compute_capability": "8.0"},
    {"id": "a10g", "cost_per_hr": 1.10, "vram_mb": 24 * 1024, "compute_capability": "8.6"},
    {"id": "t4", "cost_per_hr": 0.59, "vram_mb": 16 * 1024, "compute_capability": "7.5"},
]

# Map from hardware profile gpu_name to ladder ID
_GPU_NAME_TO_ID: dict[str, str] = {
    "nvidia h100": "h100",
    "nvidia a100 80gb": "a100-80gb",
    "nvidia a100-sxm4-80gb": "a100-80gb",
    "nvidia a100 40gb": "a100-40gb",
    "nvidia a100-sxm4-40gb": "a100-40gb",
    "nvidia a10g": "a10g",
    "tesla t4": "t4",
    "nvidia t4": "t4",
    # Direct ID matches
    "h100": "h100",
    "a100-80gb": "a100-80gb",
    "a100-40gb": "a100-40gb",
    "a10g": "a10g",
    "t4": "t4",
}

_AGGRESSIVENESS_STEPS = {
    "conservative": 1,
    "moderate": 2,
    "aggressive": 99,
    "none": 0,
}


def _resolveGpuId(gpu_name: str) -> str | None:
    """Resolve a hardware profile GPU name to a ladder ID."""
    normalized = gpu_name.strip().lower()
    return _GPU_NAME_TO_ID.get(normalized)


def _ladderIndex(gpu_id: str) -> int | None:
    """Find index in the GPU ladder for a given GPU ID."""
    for i, entry in enumerate(_GPU_LADDER):
        if entry["id"] == gpu_id:
            return i
    return None


def generateGpuDowngradeCandidates(
    current_gpu: str,
    aggressiveness: str = "moderate",
    min_vram_mb: int = 0,
) -> list[str]:
    """Generate cheaper GPU candidates to test.

    Args:
        current_gpu: Current GPU name or ID.
        aggressiveness: How many steps down the ladder to try.
            "conservative" = 1 step, "moderate" = 2, "aggressive" = all, "none" = 0.
        min_vram_mb: Minimum VRAM required (from peak memory measurement).

    Returns:
        Ordered list of GPU IDs to try (cheapest viable first... no, ordered
        from current down so we test progressively cheaper).
    """
    max_steps = _AGGRESSIVENESS_STEPS.get(aggressiveness, 2)
    if max_steps == 0:
        return []

    gpu_id = _resolveGpuId(current_gpu)
    if gpu_id is None:
        return []

    current_idx = _ladderIndex(gpu_id)
    if current_idx is None:
        return []

    candidates = []
    for i in range(current_idx + 1, len(_GPU_LADDER)):
        if len(candidates) >= max_steps:
            break
        entry = _GPU_LADDER[i]
        # Skip GPUs that don't have enough VRAM
        if min_vram_mb > 0 and int(entry["vram_mb"]) < min_vram_mb:
            continue
        candidates.append(str(entry["id"]))

    return candidates


def estimateCostSavings(
    current_gpu: str,
    candidate_gpu: str,
    wall_clock_seconds: float,
) -> dict[str, float]:
    """Estimate cost savings from switching GPUs.

    Args:
        current_gpu: Current GPU ID.
        candidate_gpu: Candidate GPU ID.
        wall_clock_seconds: Training wall clock time.

    Returns:
        Dict with current_cost, candidate_cost, savings_usd, savings_pct.
    """
    current_rate = GPU_COSTS_PER_HOUR.get(current_gpu, 3.00)
    candidate_rate = GPU_COSTS_PER_HOUR.get(candidate_gpu, 3.00)
    hours = wall_clock_seconds / 3600.0

    current_cost = current_rate * hours
    candidate_cost = candidate_rate * hours
    savings = current_cost - candidate_cost
    savings_pct = (savings / current_cost * 100) if current_cost > 0 else 0

    return {
        "current_cost_usd": current_cost,
        "candidate_cost_usd": candidate_cost,
        "savings_usd": savings,
        "savings_pct": savings_pct,
    }
