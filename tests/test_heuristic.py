"""Tests for heuristic optimization config generation."""

from pathlib import Path

from KairoScale.agent.heuristic import generate_heuristic_configs
from KairoScale.types import OperatorProfile, ProfileResult


def test_generateHeuristicConfigsAttentionAndCompile(tmp_path):
    profile = ProfileResult(
        top_operators=[
            OperatorProfile(
                name="aten::scaled_dot_product_attention",
                gpu_time_ms=50.0,
                cpu_time_ms=10.0,
                pct_total=58.0,
                call_count=128,
            )
        ],
        gpu_utilization=45.0,
        peak_memory_mb=8192.0,
    )

    configs = generate_heuristic_configs(profile, Path(tmp_path), max_configs=10)
    names = [c.name for c in configs]
    types = [c.optimization_type.value for c in configs]

    assert any("Flash Attention" in n for n in names)
    assert "compilation" in types
    assert "mixed_precision" in types


def test_generateHeuristicConfigsFallback(tmp_path):
    profile = ProfileResult()
    configs = generate_heuristic_configs(profile, Path(tmp_path), max_configs=10)

    assert len(configs) >= 1
    assert configs[0].id == "opt-001"
    assert len(configs[0].evidence) >= 1
