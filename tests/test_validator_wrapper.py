"""Tests for validation wrapper generation."""

from pathlib import Path

from gpunity.validator.gradient_tracker import create_gradient_tracking_wrapper


def test_gradientTrackingWrapperIsValidPython():
    script = create_gradient_tracking_wrapper(
        entry_point="train.py",
        steps=10,
        gradient_check_interval=2,
        artifact_dir=Path("/tmp/gpunity_test"),
    )
    compile(script, "<validation_wrapper>", "exec")


def test_gradientTrackingWrapperHasFallbackEntryInvocation():
    script = create_gradient_tracking_wrapper(
        entry_point="train.py",
        steps=10,
        gradient_check_interval=2,
        artifact_dir=Path("/tmp/gpunity_test"),
    )
    assert "_invoke_fallback_entry" in script
    assert "train" in script
    assert "main" in script
    assert "run" in script
