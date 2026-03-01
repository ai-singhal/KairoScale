"""Tests for KairoScale.agent.diversity."""

from KairoScale.agent.diversity import select_diverse_configs
from KairoScale.types import OptimizationConfig, OptimizationType, RiskLevel


def _makeConfig(name, optType, speedup, risk=RiskLevel.LOW):
    return OptimizationConfig(
        id=name,
        name=name,
        description="test",
        optimization_type=optType,
        evidence=["test"],
        estimated_speedup=speedup,
        risk_level=risk,
    )


def test_selectsTopK():
    configs = [
        _makeConfig("a", OptimizationType.ATTENTION, 2.0),
        _makeConfig("b", OptimizationType.COMPILATION, 1.5),
        _makeConfig("c", OptimizationType.MEMORY, 1.2),
    ]
    selected = select_diverse_configs(configs, top_k=2)
    assert len(selected) == 2


def test_diversityPenalty():
    configs = [
        _makeConfig("attn1", OptimizationType.ATTENTION, 2.0),
        _makeConfig("attn2", OptimizationType.ATTENTION, 1.9),
        _makeConfig("comp1", OptimizationType.COMPILATION, 1.5),
    ]
    selected = select_diverse_configs(configs, top_k=2)
    types = [c.optimization_type for c in selected]
    # Should prefer diversity: pick attention + compilation rather than two attention
    assert OptimizationType.COMPILATION in types
    assert OptimizationType.ATTENTION in types


def test_emptyInput():
    assert select_diverse_configs([], top_k=5) == []


def test_fewerThanTopK():
    configs = [_makeConfig("a", OptimizationType.ATTENTION, 2.0)]
    selected = select_diverse_configs(configs, top_k=5)
    assert len(selected) == 1


def test_riskWeighting():
    configs = [
        _makeConfig("low_risk", OptimizationType.ATTENTION, 1.5, RiskLevel.LOW),
        _makeConfig("high_risk", OptimizationType.COMPILATION, 1.6, RiskLevel.HIGH),
    ]
    selected = select_diverse_configs(configs, top_k=1)
    # Low risk with 1.5x should beat high risk with 1.6x due to risk penalty
    assert selected[0].name == "low_risk"
