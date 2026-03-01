"""Diversity-aware optimization config selection.

Selects top-k configs while penalizing redundancy within the same
optimization category, ensuring diverse coverage of different
optimization strategies.
"""

from __future__ import annotations

from KairoScale.types import OptimizationConfig, RiskLevel


# Risk multiplier for scoring: higher risk -> lower effective score
_RISK_WEIGHTS: dict[RiskLevel, float] = {
    RiskLevel.LOW: 1.0,
    RiskLevel.MEDIUM: 1.5,
    RiskLevel.HIGH: 2.5,
}


def select_diverse_configs(
    configs: list[OptimizationConfig],
    top_k: int,
    diversity_threshold: float = 0.5,
) -> list[OptimizationConfig]:
    """Select top_k configs with diversity penalty for same optimization_type.

    Scoring: score = estimated_speedup / risk_weight
    Selection: greedy, with penalty applied to configs sharing the same
    optimization_type as already-selected configs.

    Args:
        configs: All candidate configs from the agent.
        top_k: Number of configs to select.
        diversity_threshold: Penalty multiplier for duplicate types (0-1).
            Lower = more aggressive diversity enforcement.

    Returns:
        Selected list of OptimizationConfig, at most top_k items.
    """
    if not configs:
        return []

    if len(configs) <= top_k:
        return configs

    # Score each config
    scored = []
    for config in configs:
        risk_w = _RISK_WEIGHTS.get(config.risk_level, 1.5)
        base_score = config.estimated_speedup / risk_w
        scored.append((config, base_score))

    # Greedy selection with diversity penalty
    selected: list[OptimizationConfig] = []
    selected_types: dict[str, int] = {}
    remaining = list(scored)

    while len(selected) < top_k and remaining:
        # Apply diversity penalty
        best_idx = -1
        best_score = -1.0

        for i, (config, base_score) in enumerate(remaining):
            type_key = config.optimization_type.value
            type_count = selected_types.get(type_key, 0)
            penalty = diversity_threshold ** type_count
            effective_score = base_score * penalty

            if effective_score > best_score:
                best_score = effective_score
                best_idx = i

        if best_idx < 0:
            break

        config, _ = remaining.pop(best_idx)
        selected.append(config)
        type_key = config.optimization_type.value
        selected_types[type_key] = selected_types.get(type_key, 0) + 1

    return selected
