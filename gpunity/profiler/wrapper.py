"""Training script wrapper and training-loop detection.

Generates a self-contained Python script that instruments the user's
training code with profilers, then executes it. The generated script
is designed to run inside a sandbox (Modal or local subprocess).
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import Optional


def detect_training_loop(source_code: str) -> Optional[dict]:
    """AST-based heuristic to find training loop boundaries.

    Looks for a for/while loop body that contains both ``loss.backward()``
    and ``optimizer.step()`` calls (the canonical PyTorch training pattern).

    Args:
        source_code: Python source code to analyze.

    Returns:
        ``{"function": name, "lineno": int}`` if found inside a function,
        ``{"function": "__main__", "lineno": int}`` if found at module level,
        or ``None`` if no training loop is detected.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return None

    class _LoopFinder(ast.NodeVisitor):
        def __init__(self) -> None:
            self.result: Optional[dict] = None
            self._current_function: str = "__main__"

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            old = self._current_function
            self._current_function = node.name
            self.generic_visit(node)
            self._current_function = old

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

        def visit_For(self, node: ast.For) -> None:
            self._check_loop_body(node)
            self.generic_visit(node)

        def visit_While(self, node: ast.While) -> None:
            self._check_loop_body(node)
            self.generic_visit(node)

        def _check_loop_body(self, node: ast.AST) -> None:
            if self.result is not None:
                return
            has_backward = False
            has_step = False
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    if isinstance(func, ast.Attribute):
                        if func.attr == "backward":
                            has_backward = True
                        elif func.attr == "step":
                            has_step = True
            if has_backward and has_step:
                self.result = {
                    "function": self._current_function,
                    "lineno": node.lineno,
                }

    finder = _LoopFinder()
    finder.visit(tree)
    return finder.result


def create_profiling_wrapper(
    repo_path: Path,
    entry_point: str,
    train_function: Optional[str],
    warmup_steps: int,
    profile_steps: int,
) -> str:
    """Generate a self-contained Python wrapper script for profiling.

    The wrapper:
    1. Sets up all profilers (torch.profiler, memory, autograd, dataloader).
    2. Imports and runs the user's training entry point.
    3. Saves profiling artifacts to ARTIFACT_DIR (from environment variable).

    Args:
        repo_path: Path to the user's repository (used for context only).
        entry_point: Training script filename (e.g., 'train.py').
        train_function: Optional function name containing the training loop.
        warmup_steps: Number of warmup steps before profiling.
        profile_steps: Number of steps to profile.

    Returns:
        The wrapper script content as a string, ready to be written to a file.
    """
    total_steps = warmup_steps + profile_steps

    script = textwrap.dedent(f'''\
        #!/usr/bin/env python3
        """GPUnity profiling wrapper -- auto-generated. Do not edit."""

        import json
        import os
        import sys
        import time
        import traceback
        from pathlib import Path

        ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "/tmp/gpunity_artifacts"))
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

        WARMUP_STEPS = {warmup_steps}
        PROFILE_STEPS = {profile_steps}
        TOTAL_STEPS = {total_steps}
        ENTRY_POINT = "{entry_point}"
        TRAIN_FUNCTION = {repr(train_function)}

        # ------------------------------------------------------------------
        # DataLoader monkey-patch for throughput tracking
        # ------------------------------------------------------------------
        _dl_stats = {{"stall_times": [], "batch_times": [], "sample_counts": []}}

        try:
            import torch
            import torch.utils.data as _tud

            _OriginalDataLoader = _tud.DataLoader

            class _InstrumentedDataLoader(_OriginalDataLoader):
                def __iter__(self):
                    it = super().__iter__()
                    while True:
                        t0 = time.perf_counter()
                        try:
                            batch = next(it)
                        except StopIteration:
                            break
                        stall = time.perf_counter() - t0
                        _dl_stats["stall_times"].append(stall * 1000)  # ms
                        bs = 1
                        if isinstance(batch, (list, tuple)) and len(batch) > 0:
                            b = batch[0]
                            if hasattr(b, "shape"):
                                bs = b.shape[0]
                            elif hasattr(b, "__len__"):
                                bs = len(b)
                        _dl_stats["sample_counts"].append(bs)
                        yield batch

            _tud.DataLoader = _InstrumentedDataLoader
        except ImportError:
            pass  # torch not available -- skip patching

        # ------------------------------------------------------------------
        # H2D transfer time tracking (tensor.to / tensor.cuda)
        # ------------------------------------------------------------------
        _h2d_stats = {{"transfer_times_ms": []}}

        try:
            import torch

            _orig_tensor_to = torch.Tensor.to

            def _tracked_tensor_to(self, *args, **kwargs):
                # Only measure host-to-device transfers
                is_h2d = False
                if not self.is_cuda:
                    for a in args:
                        if isinstance(a, torch.device) and a.type == "cuda":
                            is_h2d = True
                            break
                        if isinstance(a, str) and "cuda" in a:
                            is_h2d = True
                            break
                    if "device" in kwargs:
                        dev = kwargs["device"]
                        if isinstance(dev, torch.device) and dev.type == "cuda":
                            is_h2d = True
                        elif isinstance(dev, str) and "cuda" in dev:
                            is_h2d = True

                if is_h2d:
                    t0 = time.perf_counter()
                    result = _orig_tensor_to(self, *args, **kwargs)
                    elapsed = (time.perf_counter() - t0) * 1000
                    _h2d_stats["transfer_times_ms"].append(elapsed)
                    return result
                return _orig_tensor_to(self, *args, **kwargs)

            torch.Tensor.to = _tracked_tensor_to

            _orig_tensor_cuda = torch.Tensor.cuda

            def _tracked_tensor_cuda(self, *args, **kwargs):
                if not self.is_cuda:
                    t0 = time.perf_counter()
                    result = _orig_tensor_cuda(self, *args, **kwargs)
                    elapsed = (time.perf_counter() - t0) * 1000
                    _h2d_stats["transfer_times_ms"].append(elapsed)
                    return result
                return _orig_tensor_cuda(self, *args, **kwargs)

            torch.Tensor.cuda = _tracked_tensor_cuda
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Step counter injected via optimizer.step monkey-patch
        # ------------------------------------------------------------------
        _step_counter = {{"count": 0, "step_times": [], "last_step_time": None}}

        try:
            import torch
            import torch.optim as _toptim

            def _patch_optimizer_step(cls):
                orig_step = cls.__dict__.get("step")
                if orig_step is None:
                    return
                if getattr(orig_step, "_gpunity_patched", False):
                    return

                def _counted_step(self, *args, **kwargs):
                    result = orig_step(self, *args, **kwargs)
                    now = time.perf_counter()
                    if _step_counter["last_step_time"] is not None:
                        _step_counter["step_times"].append(now - _step_counter["last_step_time"])
                    _step_counter["last_step_time"] = now
                    _step_counter["count"] += 1
                    return result

                _counted_step._gpunity_patched = True
                cls.step = _counted_step

            for _obj in _toptim.__dict__.values():
                if (
                    isinstance(_obj, type)
                    and issubclass(_obj, _toptim.Optimizer)
                    and _obj is not _toptim.Optimizer
                ):
                    _patch_optimizer_step(_obj)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Main profiling logic
        # ------------------------------------------------------------------
        def run_profiling():
            import importlib.util

            # Add repo to path
            repo_dir = os.environ.get("REPO_DIR", os.getcwd())
            sys.path.insert(0, repo_dir)

            def _invoke_fallback_entry(module_obj):
                for candidate in ("train", "main", "run"):
                    fn = getattr(module_obj, candidate, None)
                    if callable(fn):
                        print(f"[gpunity] Invoking fallback entry function: {{candidate}}()")
                        fn()
                        return True
                return False

            # Override TRAIN_STEPS env so fixture respects our step count
            os.environ["TRAIN_STEPS"] = str(TOTAL_STEPS)

            profile_data = {{
                "top_operators": [],
                "gpu_utilization": 0.0,
                "peak_memory_mb": 0.0,
                "memory_timeline": [],
                "peak_allocation_stack": "",
                "forward_time_ms": 0.0,
                "backward_time_ms": 0.0,
                "backward_ops": [],
                "dataloader_throughput": 0.0,
                "dataloader_stall_time_ms": 0.0,
                "dataloader_bottleneck": False,
                "h2d_transfer_time_ms": 0.0,
                "compile_warmup_time_s": 0.0,
                "gpu_active_ratio": 0.0,
                "memory_utilization_ratio": 0.0,
                "cpu_data_pipeline_ms": 0.0,
                "total_vram_mb": 0.0,
                "loop_detection_method": "none",
                "loop_detection_confidence": None,
            }}

            has_cuda = torch.cuda.is_available() if "torch" in sys.modules else False

            # ---- Read entry point source for loop detection ----
            entry_path = os.path.join(repo_dir, ENTRY_POINT)
            try:
                with open(entry_path) as f:
                    source_code = f.read()
                # Simple AST check for training loop
                import ast
                tree = ast.parse(source_code)
                has_backward = False
                has_step = False
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                        if node.func.attr == "backward":
                            has_backward = True
                        elif node.func.attr == "step":
                            has_step = True
                if has_backward and has_step:
                    profile_data["loop_detection_method"] = "heuristic"
                    profile_data["loop_detection_confidence"] = "medium"
            except Exception:
                pass

            # ---- Set up torch profiler if CUDA available ----
            profiler_ctx = None
            if has_cuda:
                try:
                    from torch.profiler import profile, ProfilerActivity, schedule
                    profiler_ctx = profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        schedule=schedule(
                            wait=0, warmup=WARMUP_STEPS, active=PROFILE_STEPS, repeat=1
                        ),
                        record_shapes=True,
                        with_stack=True,
                        with_flops=True,
                        on_trace_ready=lambda p: p.export_chrome_trace(
                            str(ARTIFACT_DIR / "chrome_trace.json")
                        ),
                    )
                except Exception as e:
                    print(f"[gpunity] torch.profiler setup failed: {{e}}")

            # ---- Start memory recording if CUDA ----
            if has_cuda:
                try:
                    torch.cuda.memory._record_memory_history(max_entries=100000)
                except Exception:
                    pass

            # ---- Run the user's training script ----
            try:
                if profiler_ctx is not None:
                    profiler_ctx.__enter__()

                # Import and run the entry point
                spec = importlib.util.spec_from_file_location("__train_module__", entry_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load {{entry_path}}")
                module = importlib.util.module_from_spec(spec)
                sys.modules["__train_module__"] = module

                if TRAIN_FUNCTION:
                    spec.loader.exec_module(module)
                    fn = getattr(module, TRAIN_FUNCTION)
                    fn()
                else:
                    spec.loader.exec_module(module)
                    # Many training scripts guard execution with
                    # `if __name__ == "__main__": train()`. Since we import
                    # as a module, invoke common entry functions if nothing ran.
                    if _step_counter["count"] == 0:
                        _invoke_fallback_entry(module)

                if profiler_ctx is not None:
                    profiler_ctx.__exit__(None, None, None)

            except Exception as e:
                print(f"[gpunity] Training script error: {{e}}")
                traceback.print_exc()
                if profiler_ctx is not None:
                    try:
                        profiler_ctx.__exit__(type(e), e, e.__traceback__)
                    except Exception:
                        pass

            # ---- Extract torch.profiler data ----
            if profiler_ctx is not None:
                try:
                    averages = profiler_ctx.key_averages()

                    def _event_cuda_time_us(ev):
                        for attr in ("cuda_time_total", "device_time_total"):
                            val = getattr(ev, attr, None)
                            if val is not None:
                                return float(val)
                        return 0.0

                    def _event_cpu_time_us(ev):
                        for attr in ("cpu_time_total", "self_cpu_time_total"):
                            val = getattr(ev, attr, None)
                            if val is not None:
                                return float(val)
                        return 0.0

                    total_cuda = sum(_event_cuda_time_us(ev) for ev in averages if _event_cuda_time_us(ev) > 0)
                    ops = []
                    ranked = sorted(averages, key=lambda e: _event_cuda_time_us(e), reverse=True)[:20]
                    for ev in ranked:
                        cuda_us = _event_cuda_time_us(ev)
                        cpu_us = _event_cpu_time_us(ev)
                        if cuda_us > 0:
                            ops.append({{
                                "name": ev.key,
                                "gpu_time_ms": cuda_us / 1000.0,
                                "cpu_time_ms": cpu_us / 1000.0,
                                "pct_total": (cuda_us / total_cuda * 100) if total_cuda > 0 else 0,
                                "call_count": ev.count,
                                "flops": getattr(ev, "flops", None) or 0,
                            }})
                    profile_data["top_operators"] = ops

                    # GPU utilization estimate
                    step_times = _step_counter["step_times"]
                    if step_times and total_cuda > 0:
                        total_wall = sum(step_times) * 1e6  # to microseconds
                        profile_data["gpu_utilization"] = min(
                            (total_cuda / total_wall) * 100, 100.0
                        )
                        # GPU active ratio: fraction of wall time where CUDA kernels are running
                        # Evidence source: torch.profiler CUDA time vs wall clock time
                        profile_data["gpu_active_ratio"] = min(
                            total_cuda / total_wall, 1.0
                        ) if total_wall > 0 else 0.0
                except Exception as e:
                    print(f"[gpunity] Profiler extraction error: {{e}}")

            # ---- Extract memory data ----
            if has_cuda:
                try:
                    snapshot = torch.cuda.memory._snapshot()
                    torch.cuda.memory._record_memory_history(enabled=None)

                    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    profile_data["peak_memory_mb"] = peak_mb

                    # Total VRAM and memory utilization ratio
                    # Evidence source: torch.cuda.get_device_properties().total_memory
                    try:
                        device_props = torch.cuda.get_device_properties(
                            torch.cuda.current_device()
                        )
                        total_vram_mb = device_props.total_memory / (1024 * 1024)
                        profile_data["total_vram_mb"] = total_vram_mb
                        if total_vram_mb > 0:
                            profile_data["memory_utilization_ratio"] = peak_mb / total_vram_mb
                    except Exception:
                        pass

                    # Memory timeline from snapshot
                    if snapshot and "segments" in snapshot:
                        # Simplified: just record current state
                        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                        profile_data["memory_timeline"] = [
                            {{"step": 0, "allocated_mb": allocated, "reserved_mb": reserved}}
                        ]

                    # Save raw snapshot
                    with open(ARTIFACT_DIR / "memory_snapshot.json", "w") as f:
                        json.dump({{"peak_memory_mb": peak_mb}}, f)

                except Exception as e:
                    print(f"[gpunity] Memory extraction error: {{e}}")

            # ---- Estimate forward/backward split ----
            # Use step times as a rough proxy
            step_times = _step_counter["step_times"]
            if step_times:
                avg_step = sum(step_times) / len(step_times) * 1000  # ms
                # Rough heuristic: forward ~30%, backward ~70% for typical training
                profile_data["forward_time_ms"] = avg_step * 0.3
                profile_data["backward_time_ms"] = avg_step * 0.7

            # ---- DataLoader stats ----
            stall_times = _dl_stats["stall_times"]
            sample_counts = _dl_stats["sample_counts"]
            if stall_times:
                total_stall = sum(stall_times)
                total_samples = sum(sample_counts)
                total_time = total_stall / 1000.0  # seconds
                throughput = total_samples / total_time if total_time > 0 else 0
                avg_stall = total_stall / len(stall_times)
                # Bottleneck if stall > 50% of step time
                avg_step_ms = (sum(step_times) / len(step_times) * 1000) if step_times else 1e9
                is_bottleneck = avg_stall > (avg_step_ms * 0.5)

                profile_data["dataloader_throughput"] = throughput
                profile_data["dataloader_stall_time_ms"] = avg_stall
                profile_data["dataloader_bottleneck"] = is_bottleneck

            # ---- H2D transfer time ----
            # Evidence source: monkey-patched Tensor.to() / Tensor.cuda()
            h2d_times = _h2d_stats["transfer_times_ms"]
            if h2d_times and step_times:
                total_h2d = sum(h2d_times)
                num_steps = len(step_times)
                profile_data["h2d_transfer_time_ms"] = total_h2d / num_steps if num_steps > 0 else 0.0

            # ---- Compile warmup detection ----
            # Evidence source: step time variance between early and steady-state steps
            # If first few steps are dramatically slower, torch.compile warmup is likely
            if step_times and len(step_times) >= 4:
                # Split into first quarter (warmup) and last half (steady state)
                quarter = max(1, len(step_times) // 4)
                half = max(1, len(step_times) // 2)
                warmup_avg = sum(step_times[:quarter]) / quarter
                steady_avg = sum(step_times[half:]) / len(step_times[half:])
                # If warmup steps are 3x+ slower, attribute the difference to compile
                if warmup_avg > steady_avg * 3 and steady_avg > 0:
                    compile_overhead_s = sum(
                        max(0, t - steady_avg) for t in step_times[:quarter]
                    )
                    profile_data["compile_warmup_time_s"] = compile_overhead_s

            # ---- CPU data pipeline time ----
            # Evidence source: DataLoader stall_times (cpu time spent fetching batches)
            stall_times = _dl_stats["stall_times"]
            if stall_times and step_times:
                num_steps = len(step_times)
                total_cpu_data = sum(stall_times)  # already in ms
                profile_data["cpu_data_pipeline_ms"] = total_cpu_data / num_steps if num_steps > 0 else 0.0

            # ---- Save dataloader stats ----
            with open(ARTIFACT_DIR / "dataloader_stats.json", "w") as f:
                json.dump({{
                    "throughput": profile_data["dataloader_throughput"],
                    "stall_time_ms": profile_data["dataloader_stall_time_ms"],
                    "is_bottleneck": profile_data["dataloader_bottleneck"],
                }}, f)

            # ---- Save profile result ----
            with open(ARTIFACT_DIR / "profile_result.json", "w") as f:
                json.dump(profile_data, f, indent=2, default=str)

            print(f"[gpunity] Profiling complete. Artifacts saved to {{ARTIFACT_DIR}}")
            print(f"[gpunity] Steps executed: {{_step_counter['count']}}")

        if __name__ == "__main__":
            run_profiling()
    ''')

    return script
