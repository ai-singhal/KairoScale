"""Tests for native baseline ladder generation."""

from KairoScale.optimizer.baselines import build_native_baseline_candidates
from KairoScale.types import HardwareProfile, LoopDetectionMethod, ProfileResult, RunConfig


def test_buildNativeBaselinesRequiredTrain():
    config = RunConfig(repo_path="/tmp/repo", baseline_policy="required")
    hardware = HardwareProfile(
        gpu_name="NVIDIA A100 80GB",
        gpu_count=1,
        compute_capability="8.0",
        supports_compile=True,
        supports_cuda_graphs=True,
        supports_bf16=True,
        supports_tf32=True,
    )
    profile = ProfileResult()
    profile.loop_detection_confidence = "medium"
    profile.loop_detection_method = LoopDetectionMethod.HEURISTIC
    baselines, manifest = build_native_baseline_candidates(
        run_config=config,
        hardware=hardware,
        mode="train",
        profile=profile,
    )

    ids = [b.id for b in baselines]
    assert ids == ["B1", "B2"]
    assert any(m.baseline_id == "B0" for m in manifest)
    b1 = next(b for b in baselines if b.id == "B1")
    b2 = next(b for b in baselines if b.id == "B2")
    assert b1.eligible is True
    assert b2.eligible is False
    assert b1.config_overrides["compile_mode"] == "max-autotune-no-cudagraphs"
    assert b1.config_overrides["compile_fallback_mode"] == "default"


def test_buildNativeBaselinesMinimal():
    config = RunConfig(repo_path="/tmp/repo", baseline_policy="minimal")
    hardware = HardwareProfile(
        gpu_name="T4",
        gpu_count=1,
        supports_compile=True,
        supports_cuda_graphs=True,
    )
    baselines, manifest = build_native_baseline_candidates(
        run_config=config,
        hardware=hardware,
        mode="train",
        profile=ProfileResult(),
    )
    assert [b.id for b in baselines] == ["B1"]
    assert [m.baseline_id for m in manifest] == ["B0", "B1"]


def test_buildNativeBaselinesClusterPrefersCudaGraphsVariant():
    config = RunConfig(repo_path="/tmp/repo", baseline_policy="required")
    hardware = HardwareProfile(
        gpu_name="NVIDIA A100 80GB x4",
        gpu_count=4,
        supports_compile=True,
        supports_cuda_graphs=True,
        compute_capability="8.0",
    )
    baselines, manifest = build_native_baseline_candidates(
        run_config=config,
        hardware=hardware,
        mode="train",
        profile=ProfileResult(),
    )
    b1 = next(b for b in baselines if b.id == "B1")
    b2 = next(b for b in baselines if b.id == "B2")
    assert b1.eligible is False
    assert b2.eligible is False  # graph-safety defaults to false with empty profile
    assert b2.config_overrides["compile_fallback_mode"] == "reduce-overhead"
    m2 = next(m for m in manifest if m.baseline_id == "B2")
    assert m2.eligible is False


def test_buildNativeBaselinesClusterMinimalKeepsB2():
    config = RunConfig(repo_path="/tmp/repo", baseline_policy="minimal")
    hardware = HardwareProfile(
        gpu_name="NVIDIA A100 80GB x4",
        gpu_count=4,
        supports_compile=True,
        supports_cuda_graphs=True,
        compute_capability="8.0",
    )
    profile = ProfileResult()
    profile.loop_detection_method = LoopDetectionMethod.HEURISTIC

    baselines, manifest = build_native_baseline_candidates(
        run_config=config,
        hardware=hardware,
        mode="train",
        profile=profile,
    )
    assert [b.id for b in baselines] == ["B2"]
    assert [m.baseline_id for m in manifest] == ["B0", "B2"]
