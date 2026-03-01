"""Hardware detection and workload mode resolution."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Optional

from KairoScale.types import HardwareProfile, ProfileResult, RunConfig


_GPU_PRIORS: dict[str, dict[str, object]] = {
    "a100-80gb": {
        "gpu_name": "NVIDIA A100 80GB",
        "compute_capability": "8.0",
        "vram_mb": 80 * 1024,
        "supports_bf16": True,
        "supports_tf32": True,
        "supports_fp16": True,
        "supports_cuda_graphs": True,
        "supports_compile": True,
    },
    "a100-40gb": {
        "gpu_name": "NVIDIA A100 40GB",
        "compute_capability": "8.0",
        "vram_mb": 40 * 1024,
        "supports_bf16": True,
        "supports_tf32": True,
        "supports_fp16": True,
        "supports_cuda_graphs": True,
        "supports_compile": True,
    },
    "h100": {
        "gpu_name": "NVIDIA H100",
        "compute_capability": "9.0",
        "vram_mb": 80 * 1024,
        "supports_bf16": True,
        "supports_tf32": True,
        "supports_fp16": True,
        "supports_cuda_graphs": True,
        "supports_compile": True,
    },
    "a10g": {
        "gpu_name": "NVIDIA A10G",
        "compute_capability": "8.6",
        "vram_mb": 24 * 1024,
        "supports_bf16": True,
        "supports_tf32": True,
        "supports_fp16": True,
        "supports_cuda_graphs": True,
        "supports_compile": True,
    },
    "t4": {
        "gpu_name": "NVIDIA T4",
        "compute_capability": "7.5",
        "vram_mb": 16 * 1024,
        "supports_bf16": False,
        "supports_tf32": False,
        "supports_fp16": True,
        "supports_cuda_graphs": True,
        "supports_compile": True,
    },
}


def _parse_driver_version() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1,
        )
    except Exception:
        return None
    line = out.strip().splitlines()[0].strip() if out.strip() else ""
    return line or None


def _detect_with_torch() -> Optional[HardwareProfile]:
    try:
        import torch  # type: ignore
    except Exception:
        return None

    if not torch.cuda.is_available():
        return None

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    capability = f"{props.major}.{props.minor}"
    bf16_supported = False
    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            bf16_supported = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16_supported = props.major >= 8
    else:
        bf16_supported = props.major >= 8

    tf32_supported = props.major >= 8
    profile = HardwareProfile(
        gpu_name=props.name,
        gpu_count=max(1, int(torch.cuda.device_count())),
        compute_capability=capability,
        vram_mb=int(props.total_memory // (1024 * 1024)),
        cuda_version=getattr(torch.version, "cuda", None),
        driver_version=_parse_driver_version(),
        supports_bf16=bf16_supported,
        supports_tf32=tf32_supported,
        supports_fp16=True,
        supports_cuda_graphs=hasattr(torch.cuda, "CUDAGraph"),
        supports_compile=hasattr(torch, "compile"),
        detection_source="torch-runtime",
        confidence="high",
    )
    if not profile.supports_cuda_graphs:
        profile.notes.append("torch.cuda.CUDAGraph unavailable in runtime")
    return profile


def _detect_from_declared_gpu(gpu_type: str) -> HardwareProfile:
    prior = _GPU_PRIORS.get(gpu_type.lower())
    if not prior:
        return HardwareProfile(
            gpu_name=gpu_type.upper(),
            detection_source="declared-gpu",
            confidence="low",
            notes=["Unknown gpu_type; using hardware-agnostic defaults."],
        )
    return HardwareProfile(
        gpu_name=str(prior["gpu_name"]),
        compute_capability=str(prior["compute_capability"]),
        vram_mb=int(prior["vram_mb"]),
        supports_bf16=bool(prior["supports_bf16"]),
        supports_tf32=bool(prior["supports_tf32"]),
        supports_fp16=bool(prior["supports_fp16"]),
        supports_cuda_graphs=bool(prior["supports_cuda_graphs"]),
        supports_compile=bool(prior["supports_compile"]),
        detection_source="declared-gpu",
        confidence="medium",
    )


def _resolve_requested_gpu_count(config: RunConfig) -> int:
    count = max(1, int(config.gpu_count))
    if count > 1:
        return count

    world_size = os.environ.get("WORLD_SIZE")
    if world_size:
        try:
            detected = int(world_size)
            return max(1, detected)
        except ValueError:
            return count
    return count


def detect_hardware_profile(config: RunConfig) -> HardwareProfile:
    """Detect hardware features for policy and reporting."""
    requested_gpu_count = _resolve_requested_gpu_count(config)

    if config.hardware_profile == "manual":
        profile = _detect_from_declared_gpu(config.gpu_type)
        profile.gpu_count = requested_gpu_count
        profile.notes.append("Manual profile mode requested.")
        return profile

    detected = _detect_with_torch()
    if detected is not None:
        if requested_gpu_count > 1 and detected.gpu_count != requested_gpu_count:
            detected.notes.append(
                f"Configured gpu_count={requested_gpu_count} overrides local runtime "
                f"gpu_count={detected.gpu_count}."
            )
            detected.gpu_count = requested_gpu_count
        return detected

    fallback = _detect_from_declared_gpu(config.gpu_type)
    fallback.gpu_count = requested_gpu_count
    fallback.notes.append("Runtime CUDA inspection unavailable; used gpu_type priors.")
    return fallback


def resolve_workload_mode(config_mode: str, profile: Optional[ProfileResult]) -> str:
    """Resolve workload mode to `train` or `infer`."""
    mode = (config_mode or "auto").lower()
    if mode in {"train", "infer"}:
        return mode

    if profile is None:
        return "train"

    has_backward = profile.backward_time_ms > 0
    has_train_loop = profile.loop_detection_method.value != "none"
    if has_backward or has_train_loop:
        return "train"
    return "infer"


def hardware_supports_max_autotune(hardware: HardwareProfile) -> bool:
    """Conservative gate for max-autotune availability."""
    if not hardware.supports_compile:
        return False
    cc = hardware.compute_capability or ""
    m = re.match(r"^(\d+)\.(\d+)$", cc)
    if not m:
        return True
    major = int(m.group(1))
    return major >= 8
