"""Tests for validator patch application helpers."""

from KairoScale.types import OptimizationConfig, OptimizationType, RiskLevel
from KairoScale.validator.patcher import apply_config_in_place


def test_applyConfigInPlaceWritesFiles(tmp_path):
    (tmp_path / "train.py").write_text("print('old')\n", encoding="utf-8")
    config = OptimizationConfig(
        id="opt-001",
        name="test",
        description="test",
        optimization_type=OptimizationType.COMPILATION,
        evidence=["e"],
        code_changes={"train.py": "print('new')\n"},
        config_overrides={"compile": True},
        dependencies=["flash-attn>=2.5"],
        risk_level=RiskLevel.LOW,
    )

    written = apply_config_in_place(tmp_path, config)
    assert (tmp_path / "train.py").read_text(encoding="utf-8") == "print('new')\n"
    assert (tmp_path / ".KairoScale_overrides.json").exists()
    assert (tmp_path / ".KairoScale_extra_requirements.txt").exists()
    assert len(written) >= 3
