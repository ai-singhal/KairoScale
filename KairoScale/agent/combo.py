"""Combo config generation from validated optimization candidates."""

from __future__ import annotations

from itertools import combinations

from KairoScale.types import (
    OptimizationConfig,
    ValidationResult,
    are_configs_compatible,
    merge_configs,
)


def _compound_speedup(results: list[ValidationResult]) -> float:
    speedup = 1.0
    for result in results:
        speedup *= max(0.0, result.speedup_vs_control)
    return min(2.5, speedup)


def generate_combo_configs(
    config_result_pairs: list[tuple[OptimizationConfig, ValidationResult]],
    max_combos: int = 5,
) -> list[OptimizationConfig]:
    """Generate top compatible 2-way/3-way combo configs from passing runs."""
    if max_combos <= 0:
        return []

    passing_pairs = [
        (config, result)
        for config, result in config_result_pairs
        if result.success
        and not result.diverged
        and not config.is_native_baseline
        and not result.is_native_baseline
    ]
    if len(passing_pairs) < 2:
        return []

    combo_by_ids: dict[tuple[str, ...], tuple[OptimizationConfig, float]] = {}

    for (cfg_a, res_a), (cfg_b, res_b) in combinations(passing_pairs, 2):
        compatible, _reason = are_configs_compatible(cfg_a, cfg_b)
        if not compatible:
            continue
        try:
            merged = merge_configs([cfg_a, cfg_b])
        except ValueError:
            continue
        key = tuple(sorted((cfg_a.id, cfg_b.id)))
        combo_by_ids[key] = (merged, _compound_speedup([res_a, res_b]))

    if len(passing_pairs) >= 3 and combo_by_ids:
        by_id = {cfg.id: (cfg, res) for cfg, res in passing_pairs}
        for pair_ids in list(combo_by_ids):
            pair_configs = [by_id[cid][0] for cid in pair_ids]
            pair_results = [by_id[cid][1] for cid in pair_ids]
            for cfg_c, res_c in passing_pairs:
                if cfg_c.id in pair_ids:
                    continue
                if any(not are_configs_compatible(cfg_c, base_cfg)[0] for base_cfg in pair_configs):
                    continue
                try:
                    merged = merge_configs(pair_configs + [cfg_c])
                except ValueError:
                    continue
                key = tuple(sorted((*pair_ids, cfg_c.id)))
                score = _compound_speedup(pair_results + [res_c])
                previous = combo_by_ids.get(key)
                if previous is None or score > previous[1]:
                    combo_by_ids[key] = (merged, score)

    ranked = sorted(combo_by_ids.values(), key=lambda item: item[1], reverse=True)
    return [combo for combo, _score in ranked[:max_combos]]
