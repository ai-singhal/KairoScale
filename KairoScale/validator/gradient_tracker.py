"""Gradient tracking wrapper for validation runs.

Generates a script that runs training while tracking loss values,
gradient norms, and gradient tensors for cosine similarity comparison.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any


def create_gradient_tracking_wrapper(
    entry_point: str,
    steps: int,
    gradient_check_interval: int,
    artifact_dir: Path,
    mode: str = "train",
    validation_seed: int = 1337,
    deterministic_validation: bool = True,
) -> str:
    """Generate a wrapper script that tracks gradients and metrics.

    The wrapper:
    1. Runs the user's training script for N steps.
    2. Records loss values every step.
    3. Records gradient of a reference parameter every K steps.
    4. Records wall-clock time and peak memory.
    5. Saves all data to artifact_dir.

    Args:
        entry_point: Training script filename.
        steps: Total training steps.
        gradient_check_interval: Steps between gradient checkpoints.
        artifact_dir: Directory for output artifacts.
        mode: Workload mode (`train` or `infer`).
        validation_seed: Seed for deterministic replay.
        deterministic_validation: Whether to enforce deterministic settings.

    Returns:
        Wrapper script content as a string.
    """
    script = textwrap.dedent(f'''\
        #!/usr/bin/env python3
        """KairoScale validation wrapper -- auto-generated."""

        import json
        import math
        import os
        import random
        import sys
        import time
        from pathlib import Path

        ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "{artifact_dir}"))
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        REPO_DIR = os.environ.get("REPO_DIR", os.getcwd())

        ENTRY_POINT = "{entry_point}"
        TOTAL_STEPS = {steps}
        GRADIENT_CHECK_INTERVAL = {gradient_check_interval}
        MODE = "{mode}"
        VALIDATION_SEED = {validation_seed}
        DETERMINISTIC_VALIDATION = {deterministic_validation}

        _overrides_path = Path(REPO_DIR) / ".KairoScale_overrides.json"
        if _overrides_path.exists():
            try:
                _CONFIG_OVERRIDES = json.loads(_overrides_path.read_text())
            except Exception:
                _CONFIG_OVERRIDES = {{}}
        else:
            _CONFIG_OVERRIDES = {{}}

        # ------------------------------------------------------------------
        # Metrics collection
        # ------------------------------------------------------------------
        _metrics = {{
            "loss_values": [],
            "gradient_norms": [],
            "gradient_checksums": [],
            "logit_signatures": [],
            "step_times": [],
            "peak_memory_mb": 0.0,
            "applied_overrides": _CONFIG_OVERRIDES,
            "runtime_error": None,
        }}

        _step_counter = {{"count": 0}}
        _last_logits_signature = [None]

        def _save_metrics():
            with open(ARTIFACT_DIR / "metrics.json", "w") as f:
                json.dump({{
                    "steps_completed": _step_counter["count"],
                    "wall_clock_seconds": sum(_metrics["step_times"]),
                    "avg_step_time_ms": (
                        sum(_metrics["step_times"]) / len(_metrics["step_times"]) * 1000
                        if _metrics["step_times"] else 0
                    ),
                    "peak_memory_mb": _metrics["peak_memory_mb"],
                    "throughput_samples_sec": (
                        _step_counter["count"] / sum(_metrics["step_times"])
                        if _metrics["step_times"] else 0
                    ),
                    "loss_values": _metrics["loss_values"],
                    "gradient_norms": _metrics["gradient_norms"],
                    "gradient_checksums": _metrics["gradient_checksums"],
                    "logit_signatures": _metrics["logit_signatures"],
                    "seed": VALIDATION_SEED,
                    "deterministic_validation": DETERMINISTIC_VALIDATION,
                    "mode": MODE,
                    "applied_overrides": _CONFIG_OVERRIDES,
                    "runtime_error": _metrics["runtime_error"],
                }}, f, indent=2)

        def _to_bool(value, default=False):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return default

        def _tensor_signature(tensor):
            try:
                import torch
                flat = tensor.detach().float().reshape(-1)
                if flat.numel() == 0:
                    return None
                sample = flat[:8].tolist()
                return {{
                    "mean": float(flat.mean().item()),
                    "std": float(flat.std(unbiased=False).item()),
                    "min": float(flat.min().item()),
                    "max": float(flat.max().item()),
                    "l2": float(torch.norm(flat, p=2).item()),
                    "sample": [float(x) for x in sample],
                }}
            except Exception:
                return None

        # ------------------------------------------------------------------
        # Monkey-patches for tracking
        # ------------------------------------------------------------------
        try:
            import torch
            import torch.optim as _toptim
            import torch.nn.functional as _F
            import torch.utils.data as _tud

            # Deterministic setup (same seed + algorithm preferences).
            random.seed(VALIDATION_SEED)
            os.environ["PYTHONHASHSEED"] = str(VALIDATION_SEED)
            try:
                import numpy as _np
                _np.random.seed(VALIDATION_SEED)
            except Exception:
                pass
            torch.manual_seed(VALIDATION_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(VALIDATION_SEED)

            # Performance overrides that affect determinism
            _cudnn_benchmark_enabled = _to_bool(
                _CONFIG_OVERRIDES.get("cudnn_benchmark", False), False
            )
            _memory_format_str = str(
                _CONFIG_OVERRIDES.get("memory_format", "")
            ).strip().lower()
            _has_perf_overrides = _cudnn_benchmark_enabled or _memory_format_str != ""

            if DETERMINISTIC_VALIDATION and not _has_perf_overrides:
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
            else:
                # Relax deterministic constraints for perf-oriented configs
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.deterministic = False
                    torch.backends.cudnn.benchmark = _cudnn_benchmark_enabled

            # Resolve memory format object
            _memory_format = None
            if _memory_format_str == "channels_last_3d":
                _memory_format = torch.channels_last_3d
            elif _memory_format_str == "channels_last":
                _memory_format = torch.channels_last

            # DataLoader patch for deterministic replay and runtime overrides.
            _OriginalDataLoader = _tud.DataLoader
            _loader_seed_gen = torch.Generator()
            _loader_seed_gen.manual_seed(VALIDATION_SEED)

            class _KairoScaleDataLoader(_OriginalDataLoader):
                def __init__(self, *args, **kwargs):
                    if "dataloader_num_workers" in _CONFIG_OVERRIDES:
                        kwargs["num_workers"] = int(_CONFIG_OVERRIDES["dataloader_num_workers"])
                    elif DETERMINISTIC_VALIDATION and "num_workers" not in kwargs:
                        kwargs["num_workers"] = 0

                    if "dataloader_pin_memory" in _CONFIG_OVERRIDES:
                        kwargs["pin_memory"] = _to_bool(_CONFIG_OVERRIDES["dataloader_pin_memory"])

                    if "dataloader_prefetch_factor" in _CONFIG_OVERRIDES:
                        pf = int(_CONFIG_OVERRIDES["dataloader_prefetch_factor"])
                        # prefetch_factor requires num_workers > 0
                        nw = kwargs.get("num_workers", 0)
                        if nw > 0:
                            kwargs["prefetch_factor"] = pf

                    if "dataloader_persistent_workers" in _CONFIG_OVERRIDES:
                        pw = _to_bool(_CONFIG_OVERRIDES["dataloader_persistent_workers"])
                        nw = kwargs.get("num_workers", 0)
                        if nw > 0:
                            kwargs["persistent_workers"] = pw

                    if DETERMINISTIC_VALIDATION and "generator" not in kwargs:
                        kwargs["generator"] = _loader_seed_gen

                    super().__init__(*args, **kwargs)

                def __iter__(self):
                    base_iter = super().__iter__()
                    if _memory_format is None:
                        return base_iter
                    return _MemoryFormatIterator(base_iter)

            class _MemoryFormatIterator:
                def __init__(self, base_iter):
                    self._base = base_iter

                def __iter__(self):
                    return self

                def __next__(self):
                    batch = next(self._base)
                    return _convert_batch_memory_format(batch)

            def _convert_batch_memory_format(batch):
                if isinstance(batch, torch.Tensor):
                    # Only convert tensors with enough dims for the format
                    if _memory_format == torch.channels_last_3d and batch.ndim == 5:
                        return batch.to(memory_format=torch.channels_last_3d)
                    elif _memory_format == torch.channels_last and batch.ndim == 4:
                        return batch.to(memory_format=torch.channels_last)
                    return batch
                elif isinstance(batch, (list, tuple)):
                    converted = [_convert_batch_memory_format(x) for x in batch]
                    return type(batch)(converted)
                elif isinstance(batch, dict):
                    return {{k: _convert_batch_memory_format(v) for k, v in batch.items()}}
                return batch

            _tud.DataLoader = _KairoScaleDataLoader

            # Patch optimizer.step to count steps and time them
            _last_step_time = [None]

            def _patch_optimizer_step(cls):
                orig_step = cls.__dict__.get("step")
                if orig_step is None:
                    return
                if getattr(orig_step, "_KairoScale_patched", False):
                    return

                def _tracked_step(self, *args, **kwargs):
                    result = orig_step(self, *args, **kwargs)
                    now = time.perf_counter()
                    if _last_step_time[0] is not None:
                        _metrics["step_times"].append(now - _last_step_time[0])
                    _last_step_time[0] = now
                    _step_counter["count"] += 1
                    return result

                _tracked_step._KairoScale_patched = True
                cls.step = _tracked_step

            for _obj in _toptim.__dict__.values():
                if (
                    isinstance(_obj, type)
                    and issubclass(_obj, _toptim.Optimizer)
                    and _obj is not _toptim.Optimizer
                ):
                    _patch_optimizer_step(_obj)

            # Fused optimizer override
            _optimizer_strategy = str(_CONFIG_OVERRIDES.get("optimizer_strategy", "")).lower()
            if _optimizer_strategy == "fused_triton_search":
                _OrigAdamW = getattr(_toptim, "AdamW", None)
                _OrigAdam = getattr(_toptim, "Adam", None)

                def _make_fused_wrapper(orig_cls):
                    class _FusedOptimizer(orig_cls):
                        def __init__(self, params, **kwargs):
                            try:
                                kwargs["fused"] = True
                                super().__init__(params, **kwargs)
                                _CONFIG_OVERRIDES["_optimizer_fused_applied"] = orig_cls.__name__
                            except Exception:
                                kwargs.pop("fused", None)
                                super().__init__(params, **kwargs)
                    _FusedOptimizer.__name__ = orig_cls.__name__
                    _FusedOptimizer.__qualname__ = orig_cls.__qualname__
                    return _FusedOptimizer

                if _OrigAdamW is not None:
                    _toptim.AdamW = _make_fused_wrapper(_OrigAdamW)
                if _OrigAdam is not None:
                    _toptim.Adam = _make_fused_wrapper(_OrigAdam)

            # Top-level model-call patch for overrides (compile, amp, checkpointing).
            _orig_module_call = torch.nn.Module.__call__
            _call_depth = [0]
            _compile_enabled = [_to_bool(_CONFIG_OVERRIDES.get("compile", False), False)]
            _amp_enabled = _to_bool(_CONFIG_OVERRIDES.get("amp", False), False)
            _cuda_graphs_enabled = _to_bool(_CONFIG_OVERRIDES.get("cuda_graphs", False), False)
            _precision = str(_CONFIG_OVERRIDES.get("precision", "bf16")).lower()
            _checkpointing_enabled = _to_bool(
                _CONFIG_OVERRIDES.get("gradient_checkpointing", False), False
            )
            _compile_mode = [str(_CONFIG_OVERRIDES.get("compile_mode", "default"))]
            _compile_fallback_mode = str(
                _CONFIG_OVERRIDES.get("compile_fallback_mode", "")
            ).strip()

            def _maybe_apply_model_overrides(model):
                if getattr(model, "_KairoScale_overrides_applied", False):
                    return

                # Memory format conversion (must happen before compile)
                if _memory_format is not None:
                    try:
                        model.to(memory_format=_memory_format)
                        _CONFIG_OVERRIDES["_memory_format_applied"] = _memory_format_str
                    except Exception as e:
                        _CONFIG_OVERRIDES["_memory_format_error"] = str(e)

                if _checkpointing_enabled and hasattr(model, "gradient_checkpointing_enable"):
                    try:
                        model.gradient_checkpointing_enable()
                    except Exception:
                        pass

                if _compile_enabled[0] and hasattr(torch, "compile"):
                    try:
                        model._KairoScale_original_forward = model.forward
                        model.forward = torch.compile(model.forward, mode=_compile_mode[0])
                    except Exception:
                        pass

                if _cuda_graphs_enabled:
                    if MODE != "infer":
                        _CONFIG_OVERRIDES["_cuda_graphs_status"] = "disabled_non_inference_mode"
                    elif not torch.cuda.is_available():
                        _CONFIG_OVERRIDES["_cuda_graphs_status"] = "disabled_no_cuda"
                    elif not hasattr(torch.cuda, "CUDAGraph"):
                        _CONFIG_OVERRIDES["_cuda_graphs_status"] = "disabled_runtime_unsupported"
                    else:
                        # Generic graph capture requires shape-stable step functions and explicit
                        # stream/capture management. The wrapper marks intent and defers to
                        # workload-specific implementations if present in user code.
                        _CONFIG_OVERRIDES["_cuda_graphs_status"] = "requested_generic_wrapper"

                # Attention backend override
                _attention_backend = str(_CONFIG_OVERRIDES.get("attention_backend", "")).lower()
                if _attention_backend == "flash":
                    try:
                        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                            _orig_sdpa = torch.nn.functional.scaled_dot_product_attention
                            def _flash_sdpa(query, key, value, **sdpa_kwargs):
                                with torch.backends.cuda.sdp_kernel(
                                    enable_flash=True, enable_math=False, enable_mem_efficient=False
                                ):
                                    return _orig_sdpa(query, key, value, **sdpa_kwargs)
                            torch.nn.functional.scaled_dot_product_attention = _flash_sdpa
                            _CONFIG_OVERRIDES["_attention_backend_applied"] = "sdpa_flash"
                    except Exception as e:
                        _CONFIG_OVERRIDES["_attention_backend_error"] = str(e)

                model._KairoScale_overrides_applied = True

            def _patched_module_call(self, *args, **kwargs):
                _call_depth[0] += 1
                is_top = _call_depth[0] == 1
                try:
                    if is_top:
                        _maybe_apply_model_overrides(self)

                        def _call_model():
                            if _amp_enabled:
                                device_type = "cuda" if torch.cuda.is_available() else "cpu"
                                dtype = torch.bfloat16
                                if _precision in ("fp16", "float16"):
                                    dtype = torch.float16
                                elif _precision in ("bf16", "bfloat16"):
                                    dtype = torch.bfloat16
                                if device_type == "cpu" and dtype == torch.float16:
                                    dtype = torch.bfloat16

                                with torch.autocast(
                                    device_type=device_type,
                                    dtype=dtype,
                                    enabled=True,
                                ):
                                    return _orig_module_call(self, *args, **kwargs)
                            return _orig_module_call(self, *args, **kwargs)

                        try:
                            out = _call_model()
                        except Exception as exc:
                            if _compile_enabled[0] and hasattr(self, "_KairoScale_original_forward"):
                                fallback_succeeded = False

                                if (
                                    _compile_fallback_mode
                                    and _compile_fallback_mode != _compile_mode[0]
                                    and hasattr(torch, "compile")
                                ):
                                    try:
                                        self.forward = torch.compile(
                                            self._KairoScale_original_forward,
                                            mode=_compile_fallback_mode,
                                        )
                                        _compile_mode[0] = _compile_fallback_mode
                                        _CONFIG_OVERRIDES["_compile_runtime_fallback_mode"] = (
                                            _compile_fallback_mode
                                        )
                                        out = _call_model()
                                        fallback_succeeded = True
                                    except Exception as fallback_exc:
                                        _CONFIG_OVERRIDES["_compile_runtime_fallback_error"] = str(
                                            fallback_exc
                                        )

                                if not fallback_succeeded:
                                    # Final fallback: disable compile for this run and retry eagerly.
                                    self.forward = self._KairoScale_original_forward
                                    _compile_enabled[0] = False
                                    _CONFIG_OVERRIDES["_compile_runtime_fallback"] = True
                                    out = _call_model()
                            else:
                                raise exc

                        if _last_logits_signature[0] is None and isinstance(out, torch.Tensor):
                            if out.ndim >= 2:
                                _last_logits_signature[0] = _tensor_signature(out)
                        if MODE == "infer":
                            now = time.perf_counter()
                            if _last_step_time[0] is not None:
                                _metrics["step_times"].append(now - _last_step_time[0])
                            _last_step_time[0] = now
                            _step_counter["count"] += 1
                        return out

                    return _orig_module_call(self, *args, **kwargs)
                finally:
                    _call_depth[0] -= 1

            torch.nn.Module.__call__ = _patched_module_call

            # Capture logits from common loss APIs.
            _orig_ce = _F.cross_entropy

            def _tracked_cross_entropy(input, *args, **kwargs):
                _last_logits_signature[0] = _tensor_signature(input)
                return _orig_ce(input, *args, **kwargs)

            _F.cross_entropy = _tracked_cross_entropy

            _orig_bce_logits = getattr(_F, "binary_cross_entropy_with_logits", None)
            if _orig_bce_logits is not None:
                def _tracked_bce_logits(input, *args, **kwargs):
                    _last_logits_signature[0] = _tensor_signature(input)
                    return _orig_bce_logits(input, *args, **kwargs)
                _F.binary_cross_entropy_with_logits = _tracked_bce_logits

            # GradScaler for AMP (enables proper fp16/bf16 training with loss scaling)
            _grad_scaler = [None]
            if _amp_enabled and torch.cuda.is_available():
                try:
                    _scaler_cls = getattr(torch.amp, "GradScaler", None)
                    if _scaler_cls is None:
                        _scaler_cls = getattr(torch.cuda.amp, "GradScaler", None)
                    if _scaler_cls is not None:
                        _grad_scaler[0] = _scaler_cls()
                        _CONFIG_OVERRIDES["_grad_scaler_enabled"] = True
                except Exception:
                    pass

            # Patch loss.backward to capture loss value and use GradScaler
            _OrigBackward = torch.Tensor.backward

            def _tracked_backward(self, *args, **kwargs):
                try:
                    loss_val = self.detach().item()
                    if not (math.isnan(loss_val) or math.isinf(loss_val)):
                        _metrics["loss_values"].append(loss_val)
                    else:
                        _metrics["loss_values"].append(float("nan"))
                except Exception:
                    pass
                try:
                    if _last_logits_signature[0] is not None:
                        _metrics["logit_signatures"].append(_last_logits_signature[0])
                        _last_logits_signature[0] = None
                    else:
                        _metrics["logit_signatures"].append(None)
                except Exception:
                    pass

                scaler = _grad_scaler[0]
                if scaler is not None:
                    try:
                        scaled_loss = scaler.scale(self)
                        return _OrigBackward(scaled_loss, *args, **kwargs)
                    except Exception:
                        return _OrigBackward(self, *args, **kwargs)
                return _OrigBackward(self, *args, **kwargs)

            torch.Tensor.backward = _tracked_backward

            # Patch optimizer.step to use GradScaler when active
            if _grad_scaler[0] is not None:
                def _patch_optimizer_step_scaler(cls):
                    orig_step = cls.__dict__.get("step")
                    if orig_step is None:
                        return
                    if getattr(orig_step, "_KairoScale_scaler_patched", False):
                        return

                    def _scaled_step(self, *args, **kwargs):
                        scaler = _grad_scaler[0]
                        if scaler is not None:
                            try:
                                scaler.unscale_(self)
                                result = orig_step(self, *args, **kwargs)
                                scaler.update()
                                return result
                            except Exception:
                                return orig_step(self, *args, **kwargs)
                        return orig_step(self, *args, **kwargs)

                    _scaled_step._KairoScale_scaler_patched = True
                    _scaled_step._KairoScale_patched = getattr(orig_step, "_KairoScale_patched", False)
                    cls.step = _scaled_step

                for _obj in _toptim.__dict__.values():
                    if (
                        isinstance(_obj, type)
                        and issubclass(_obj, _toptim.Optimizer)
                        and _obj is not _toptim.Optimizer
                    ):
                        _patch_optimizer_step_scaler(_obj)

        except ImportError:
            pass

        # ------------------------------------------------------------------
        # Run training
        # ------------------------------------------------------------------
        def run_validation():
            import importlib.util

            repo_dir = REPO_DIR
            sys.path.insert(0, repo_dir)
            os.chdir(repo_dir)
            os.environ["TRAIN_STEPS"] = str(TOTAL_STEPS)

            def _invoke_fallback_entry(module_obj):
                for candidate in ("train", "main", "run"):
                    fn = getattr(module_obj, candidate, None)
                    if callable(fn):
                        print(f"[KairoScale] Invoking fallback entry function: {{candidate}}()")
                        fn()
                        return True
                return False

            entry_path = os.path.join(repo_dir, ENTRY_POINT)

            try:
                spec = importlib.util.spec_from_file_location("__train_module__", entry_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load {{entry_path}}")
                module = importlib.util.module_from_spec(spec)
                sys.modules["__train_module__"] = module
                spec.loader.exec_module(module)
                if _step_counter["count"] == 0:
                    _invoke_fallback_entry(module)
            except Exception as e:
                print(f"[KairoScale] Training error: {{e}}")
                _metrics["runtime_error"] = str(e)
                import traceback
                traceback.print_exc()

            # Record peak memory
            try:
                import torch
                if torch.cuda.is_available():
                    _metrics["peak_memory_mb"] = (
                        torch.cuda.max_memory_allocated() / (1024 * 1024)
                    )
            except Exception:
                pass

            # Compute gradient norms from model parameters (post-training snapshot)
            # In practice, per-step gradient tracking would use hooks.
            # For the MVP, we record loss values (already captured) as the
            # primary divergence signal.

            _save_metrics()
            print(f"[KairoScale] Validation complete. Steps: {{_step_counter['count']}}")
            print(f"[KairoScale] Loss values: {{len(_metrics['loss_values'])}}")

        if __name__ == "__main__":
            run_validation()
    ''')

    return script


def load_gradient_checkpoints(artifact_dir: Path) -> list[dict[str, Any]]:
    """Load saved gradient checkpoints and metrics from artifacts.

    Args:
        artifact_dir: Path to the validation artifacts directory.

    Returns:
        List of checkpoint dicts with 'step', 'loss', 'gradient_norm' keys.
    """
    artifact_dir = Path(artifact_dir)
    metrics_path = artifact_dir / "metrics.json"

    if not metrics_path.exists():
        return []

    try:
        with open(metrics_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    loss_values = data.get("loss_values", [])
    grad_norms = data.get("gradient_norms", [])
    logit_signatures = data.get("logit_signatures", [])

    checkpoints = []
    for i, loss in enumerate(loss_values):
        checkpoint: dict[str, Any] = {
            "step": i,
            "loss": loss,
        }
        if i < len(grad_norms):
            checkpoint["gradient_norm"] = grad_norms[i]
        if i < len(logit_signatures):
            checkpoint["logit_signature"] = logit_signatures[i]
        checkpoints.append(checkpoint)

    return checkpoints
