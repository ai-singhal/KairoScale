"""Tests for hardware profile detection helpers."""

from KairoScale.hardware.profile import detect_hardware_profile, resolve_workload_mode
from KairoScale.types import LoopDetectionMethod, ProfileResult, RunConfig


def test_detectHardwareProfileManualUsesGpuType():
    config = RunConfig(repo_path="/tmp/repo", gpu_type="a100-80gb", hardware_profile="manual")
    profile = detect_hardware_profile(config)
    assert profile.detection_source == "declared-gpu"
    assert "A100" in profile.gpu_name
    assert profile.supports_compile is True
    assert profile.gpu_count == 1


def test_detectHardwareProfileHonorsConfiguredGpuCount():
    config = RunConfig(
        repo_path="/tmp/repo",
        gpu_type="a100-80gb",
        hardware_profile="manual",
        gpu_count=8,
    )
    profile = detect_hardware_profile(config)
    assert profile.gpu_count == 8


def test_resolveWorkloadModeAutoTrain():
    profile = ProfileResult(
        forward_time_ms=10.0,
        backward_time_ms=20.0,
        loop_detection_method=LoopDetectionMethod.HEURISTIC,
    )
    assert resolve_workload_mode("auto", profile) == "train"


def test_resolveWorkloadModeAutoInfer():
    profile = ProfileResult(
        forward_time_ms=5.0,
        backward_time_ms=0.0,
        loop_detection_method=LoopDetectionMethod.NONE,
    )
    assert resolve_workload_mode("auto", profile) == "infer"
