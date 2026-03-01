"""Tests for validation wrapper generation."""

from pathlib import Path

from KairoScale.validator.gradient_tracker import create_gradient_tracking_wrapper


def test_gradientTrackingWrapperIsValidPython():
    script = create_gradient_tracking_wrapper(
        entry_point="train.py",
        steps=10,
        gradient_check_interval=2,
        artifact_dir=Path("/tmp/KairoScale_test"),
    )
    compile(script, "<validation_wrapper>", "exec")


def test_gradientTrackingWrapperHasFallbackEntryInvocation():
    script = create_gradient_tracking_wrapper(
        entry_point="train.py",
        steps=10,
        gradient_check_interval=2,
        artifact_dir=Path("/tmp/KairoScale_test"),
    )
    assert "_invoke_fallback_entry" in script
    assert "train" in script
    assert "main" in script
    assert "run" in script


def test_gradientTrackingWrapperContainsDeterminismAndOverrides():
    script = create_gradient_tracking_wrapper(
        entry_point="train.py",
        steps=10,
        gradient_check_interval=2,
        artifact_dir=Path("/tmp/KairoScale_test"),
        validation_seed=1234,
        deterministic_validation=True,
    )
    assert "VALIDATION_SEED = 1234" in script
    assert "DETERMINISTIC_VALIDATION = True" in script
    assert ".KairoScale_overrides.json" in script
    assert "logit_signatures" in script
    assert "compile_fallback_mode" in script
