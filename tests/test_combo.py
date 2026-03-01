"""Tests for combo config generation."""

from KairoScale.agent.combo import generate_combo_configs
from KairoScale.types import OptimizationConfig, OptimizationType, RiskLevel, ValidationResult


def _cfg(config_id: str, name: str, opt_type: OptimizationType) -> OptimizationConfig:
    return OptimizationConfig(
        id=config_id,
        name=name,
        description=name,
        optimization_type=opt_type,
        evidence=[f"evidence-{config_id}"],
        estimated_speedup=1.2,
        risk_level=RiskLevel.LOW,
    )


def _result(config_id: str, name: str, speedup: float) -> ValidationResult:
    return ValidationResult(
        config_id=config_id,
        config_name=name,
        success=True,
        speedup_vs_control=speedup,
    )


def test_generateComboConfigsBuildsTopRankedCombos():
    cfg_compile = _cfg("opt-001", "Compile", OptimizationType.COMPILATION)
    cfg_amp = _cfg("opt-002", "AMP", OptimizationType.MIXED_PRECISION)
    cfg_data = _cfg("opt-003", "Data", OptimizationType.DATA_LOADING)

    pairs = [
        (cfg_compile, _result(cfg_compile.id, cfg_compile.name, 1.4)),
        (cfg_amp, _result(cfg_amp.id, cfg_amp.name, 1.2)),
        (cfg_data, _result(cfg_data.id, cfg_data.name, 1.1)),
    ]

    combos = generate_combo_configs(pairs, max_combos=3)

    assert combos
    top = combos[0]
    assert top.id.startswith("combo-")
    assert "opt-001,opt-002,opt-003" in " ".join(top.heuristic_rationale)


def test_generateComboConfigsSkipsIncompatiblePairs():
    cfg_compile_a = _cfg("opt-001", "Compile A", OptimizationType.COMPILATION)
    cfg_compile_b = _cfg("opt-002", "Compile B", OptimizationType.COMPILATION)
    cfg_amp = _cfg("opt-003", "AMP", OptimizationType.MIXED_PRECISION)

    pairs = [
        (cfg_compile_a, _result(cfg_compile_a.id, cfg_compile_a.name, 1.3)),
        (cfg_compile_b, _result(cfg_compile_b.id, cfg_compile_b.name, 1.2)),
        (cfg_amp, _result(cfg_amp.id, cfg_amp.name, 1.1)),
    ]

    combos = generate_combo_configs(pairs, max_combos=5)

    assert all("Compile A + Compile B" not in combo.name for combo in combos)
