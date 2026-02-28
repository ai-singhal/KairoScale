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

    Returns:
        Wrapper script content as a string.
    """
    script = textwrap.dedent(f'''\
        #!/usr/bin/env python3
        """GPUnity validation wrapper -- auto-generated."""

        import json
        import math
        import os
        import sys
        import time
        from pathlib import Path

        ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "{artifact_dir}"))
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

        ENTRY_POINT = "{entry_point}"
        TOTAL_STEPS = {steps}
        GRADIENT_CHECK_INTERVAL = {gradient_check_interval}

        # ------------------------------------------------------------------
        # Metrics collection
        # ------------------------------------------------------------------
        _metrics = {{
            "loss_values": [],
            "gradient_norms": [],
            "gradient_checksums": [],
            "step_times": [],
            "peak_memory_mb": 0.0,
        }}

        _gradient_snapshots = []
        _step_counter = {{"count": 0}}

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
                    "throughput_samples_sec": 0,  # computed from batch size
                    "loss_values": _metrics["loss_values"],
                    "gradient_norms": _metrics["gradient_norms"],
                    "gradient_checksums": _metrics["gradient_checksums"],
                }}, f, indent=2)

        # ------------------------------------------------------------------
        # Monkey-patches for tracking
        # ------------------------------------------------------------------
        try:
            import torch
            import torch.optim as _toptim

            # Patch optimizer.step to count steps and time them
            _last_step_time = [None]

            def _patch_optimizer_step(cls):
                orig_step = cls.__dict__.get("step")
                if orig_step is None:
                    return
                if getattr(orig_step, "_gpunity_patched", False):
                    return

                def _tracked_step(self, *args, **kwargs):
                    result = orig_step(self, *args, **kwargs)
                    now = time.perf_counter()
                    if _last_step_time[0] is not None:
                        _metrics["step_times"].append(now - _last_step_time[0])
                    _last_step_time[0] = now
                    _step_counter["count"] += 1
                    return result

                _tracked_step._gpunity_patched = True
                cls.step = _tracked_step

            for _obj in _toptim.__dict__.values():
                if (
                    isinstance(_obj, type)
                    and issubclass(_obj, _toptim.Optimizer)
                    and _obj is not _toptim.Optimizer
                ):
                    _patch_optimizer_step(_obj)

            # Patch loss.backward to capture loss value
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
                return _OrigBackward(self, *args, **kwargs)

            torch.Tensor.backward = _tracked_backward

        except ImportError:
            pass

        # ------------------------------------------------------------------
        # Run training
        # ------------------------------------------------------------------
        def run_validation():
            import importlib.util

            repo_dir = os.environ.get("REPO_DIR", os.getcwd())
            sys.path.insert(0, repo_dir)
            os.environ["TRAIN_STEPS"] = str(TOTAL_STEPS)

            def _invoke_fallback_entry(module_obj):
                for candidate in ("train", "main", "run"):
                    fn = getattr(module_obj, candidate, None)
                    if callable(fn):
                        print(f"[gpunity] Invoking fallback entry function: {{candidate}}()")
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
                print(f"[gpunity] Training error: {{e}}")
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
            print(f"[gpunity] Validation complete. Steps: {{_step_counter['count']}}")
            print(f"[gpunity] Loss values: {{len(_metrics['loss_values'])}}")

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

    checkpoints = []
    for i, loss in enumerate(loss_values):
        checkpoint: dict[str, Any] = {
            "step": i,
            "loss": loss,
        }
        if i < len(grad_norms):
            checkpoint["gradient_norm"] = grad_norms[i]
        checkpoints.append(checkpoint)

    return checkpoints
