"""Modal sandbox runner for cloud GPU execution.

Creates Modal sandboxes with GPU access, runs profiling/validation
scripts, and downloads artifacts back to the local filesystem.
"""

from __future__ import annotations

import asyncio
import inspect
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

from KairoScale.sandbox.image_builder import build_image_spec
from KairoScale.utils.logging import get_logger

logger = get_logger("KairoScale.sandbox.modal_runner")

# Modal GPU type mapping
_MODAL_GPU_MAP = {
    "a100-80gb": "A100",
    "a100-40gb": "A100",
    "h100": "H100",
    "a10g": "A10G",
    "t4": "T4",
}


async def _call_maybe_aio(method: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a Modal SDK method that may expose `.aio` or direct sync behavior."""
    aio_method = getattr(method, "aio", None)
    if callable(aio_method):
        result = aio_method(*args, **kwargs)
    else:
        result = method(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def _sandbox_ls_entries(sb: Any, path: str) -> list[tuple[str, str]]:
    """List sandbox entries as `(relative_name, remote_path)` pairs."""
    listed = await _call_maybe_aio(sb.ls, path)
    raw_entries: list[Any]
    if hasattr(listed, "__aiter__"):
        raw_entries = []
        async for item in listed:
            raw_entries.append(item)
    elif listed is None:
        raw_entries = []
    else:
        raw_entries = list(listed)

    base = path.rstrip("/") or "/"
    prefix = f"{base}/" if base != "/" else "/"
    normalized: list[tuple[str, str]] = []
    for entry in raw_entries:
        remote_path = ""
        entry_path = getattr(entry, "path", None)
        if entry_path:
            remote_path = str(entry_path)
        elif isinstance(entry, str):
            remote_path = entry
        else:
            remote_path = str(entry)

        if not remote_path:
            continue
        if not remote_path.startswith("/"):
            remote_path = prefix + remote_path.lstrip("/")

        entry_name = getattr(entry, "name", None)
        if entry_name:
            relative = str(entry_name).lstrip("/")
        elif remote_path.startswith(prefix):
            relative = remote_path[len(prefix):].lstrip("/")
        else:
            relative = Path(remote_path).name

        if relative:
            normalized.append((relative, remote_path))
    return normalized


async def _read_sandbox_file(sb: Any, remote_path: str, mode: str) -> Any:
    """Read a file from sandbox using Modal FileIO APIs across SDK variants."""
    handle = await _call_maybe_aio(sb.open, remote_path, mode)
    try:
        return await _call_maybe_aio(handle.read)
    finally:
        try:
            await _call_maybe_aio(handle.close)
        except Exception:
            pass


async def _read_stream_text(stream: Any) -> str:
    """Read Modal stdout/stderr stream and normalize to text."""
    if stream is None or not hasattr(stream, "read"):
        return ""
    try:
        payload = await _call_maybe_aio(stream.read)
    except Exception:
        return ""
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    if payload is None:
        return ""
    return str(payload)


async def run_in_modal(
    repo_path: Path,
    script_content: str,
    gpu_type: str = "a100-80gb",
    timeout_seconds: int = 600,
    cost_ceiling_usd: float = 5.0,
    extra_deps: Optional[list[str]] = None,
    stream_logs: bool = False,
    persist_volume_name: Optional[str] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Path:
    """Run a script in a Modal sandbox with GPU.

    Creates a Modal Sandbox, copies the user's repo and wrapper script
    into the image, executes it, and downloads artifacts.

    Args:
        repo_path: Path to the user's repository.
        script_content: The Python script to execute.
        gpu_type: GPU type identifier (e.g., 'a100-80gb').
        timeout_seconds: Maximum execution time.
        cost_ceiling_usd: Maximum cost allowed (not enforced yet).
        extra_deps: Additional pip packages to install.
        stream_logs: If True, print sandbox stdout to the terminal.
        persist_volume_name: Optional Modal Volume name to persist artifacts.
        log_callback: Optional callback to surface sandbox lifecycle logs.

    Returns:
        Local path to downloaded artifacts directory.

    Raises:
        RuntimeError: If Modal execution fails.
        ImportError: If the modal package is not installed.
    """
    try:
        import modal
    except ImportError:
        raise ImportError(
            "Modal is not installed. Install with: pip install modal\n"
            "Or use --local flag for local execution."
        )

    def _emit(message: str) -> None:
        logger.info(message)
        if log_callback is not None:
            try:
                log_callback(message)
            except Exception:
                logger.debug("Modal log callback failed.", exc_info=True)

    repo_path = Path(repo_path).resolve()
    image_spec = build_image_spec(repo_path, extra_deps)

    # Use PyTorch CUDA runtime image so Modal caches the heavy CUDA/torch layers.
    # Only pip-install user deps that aren't torch itself.
    non_torch_deps = [
        dep for dep in image_spec["pip_packages"]
        if not any(t in dep.lower() for t in ("torch", "nvidia", "triton"))
    ]
    image = modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        add_python="3.11",
    )
    if non_torch_deps:
        image = image.pip_install(*non_torch_deps)

    # Add repo files and wrapper script into the image
    image = image.add_local_dir(str(repo_path), remote_path="/root/repo")

    # Mount KairoScale package so Triton optimizer imports work in sandbox
    _kairoscale_pkg = Path(__file__).resolve().parent.parent
    if _kairoscale_pkg.is_dir() and (_kairoscale_pkg / "__init__.py").exists():
        image = image.add_local_dir(str(_kairoscale_pkg), remote_path="/root/KairoScale")

    # Create artifact directory locally
    local_artifact_dir = Path(tempfile.mkdtemp(prefix="KairoScale_modal_artifacts_"))
    resolved_artifact_root = local_artifact_dir.resolve()

    # Write script to a temp file and add to image
    script_path = local_artifact_dir / "KairoScale_wrapper.py"
    script_path.write_text(script_content, encoding="utf-8")
    image = image.add_local_file(str(script_path), remote_path="/root/script/KairoScale_wrapper.py")

    # Determine Modal GPU config
    modal_gpu = _MODAL_GPU_MAP.get(gpu_type, "A100")

    # Optional persistent volume for deploy runs
    volume_mounts = {}
    if persist_volume_name:
        vol = modal.Volume.from_name(persist_volume_name, create_if_missing=True)
        volume_mounts = {"/output": vol}

    _emit(f"Creating Modal sandbox with {modal_gpu} GPU")
    _emit(f"Timeout: {timeout_seconds}s, Cost ceiling: ${cost_ceiling_usd}")

    sandbox_artifact_dir = "/tmp/KairoScale_artifacts"

    connect_timeout_seconds = min(120, max(20, timeout_seconds // 3))
    _emit(
        f"Resolving Modal app handle (timeout={connect_timeout_seconds}s)..."
    )
    try:
        app = await asyncio.wait_for(
            modal.App.lookup.aio("KairoScale-sandbox", create_if_missing=True),
            timeout=connect_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            "Timed out connecting to Modal API during app lookup. "
            "Check internet connectivity, run `modal token info`, and confirm Modal auth is configured."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Modal app handle: {exc}") from exc

    try:
        wrapper_exit_code_file = "/tmp/KairoScale_wrapper_exit_code"
        runner_cmd = (
            "echo '[KairoScale] Wrapper starting'; "
            "python3 -u /root/script/KairoScale_wrapper.py; "
            "code=$?; "
            f"echo $code > {wrapper_exit_code_file}; "
            "echo \"[KairoScale] Wrapper exit code: $code\"; "
            "sleep 15"
        )

        # Cold starts (new image pull/build + GPU provisioning) can exceed 3 minutes.
        # Keep a buffer for execution while allowing enough time for sandbox creation.
        create_timeout_seconds = max(180, timeout_seconds - 60)
        _emit(
            f"Submitting Modal sandbox create request (timeout={create_timeout_seconds}s)..."
        )
        try:
            sb = await asyncio.wait_for(
                modal.Sandbox.create.aio(
                    "bash",
                    "-lc",
                    runner_cmd,
                    image=image,
                    gpu=modal_gpu,
                    timeout=timeout_seconds,
                    env={
                        "ARTIFACT_DIR": sandbox_artifact_dir,
                        "REPO_DIR": "/root/repo",
                        "PYTHONPATH": "/root:/root/repo",
                        "PYTHONUNBUFFERED": "1",
                    },
                    app=app,
                    **({"volumes": volume_mounts} if volume_mounts else {}),
                ),
                timeout=create_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "Timed out waiting for Modal to create a sandbox. "
                "The Modal control plane may be unreachable or overloaded, "
                "or the image/GPU cold start took too long."
            ) from exc

        # Wait for wrapper exit-code file to appear while sandbox stays alive.
        start = time.monotonic()
        last_progress_bucket = -1
        wrapper_exit_code: Optional[int] = None
        while True:
            elapsed = time.monotonic() - start
            if elapsed > timeout_seconds:
                raise RuntimeError(
                    f"Modal sandbox timed out after {timeout_seconds}s waiting for wrapper completion."
                )

            progress_bucket = int(elapsed // 30)
            if progress_bucket > last_progress_bucket and progress_bucket > 0:
                _emit(f"Modal sandbox still running ({int(elapsed)}s elapsed)...")
                last_progress_bucket = progress_bucket

            try:
                wrapper_exit_code_raw = await _read_sandbox_file(sb, wrapper_exit_code_file, "r")
                wrapper_exit_code = int(str(wrapper_exit_code_raw).strip())
                _emit(f"Modal wrapper finished with exit code {wrapper_exit_code}.")
                break
            except FileNotFoundError:
                # Wrapper still running.
                pass
            except (TypeError, ValueError):
                # File exists but value isn't fully written yet.
                pass
            except Exception:
                # Sandbox may still be booting; retry.
                pass
            await asyncio.sleep(1.0)

        if wrapper_exit_code is None:
            raise RuntimeError(
                f"Could not read wrapper exit code from {wrapper_exit_code_file}."
            )

        if stream_logs:
            try:
                stdout_content = await _read_stream_text(getattr(sb, "stdout", None))
                if stdout_content:
                    print(stdout_content)
                    if log_callback is not None:
                        for line in stdout_content.splitlines():
                            if line.strip():
                                log_callback(f"[sandbox] {line}")
            except Exception:
                pass

        # Download artifacts from sandbox (use async file APIs for Modal>=1.x)
        artifact_items = await _sandbox_ls_entries(sb, sandbox_artifact_dir)
        for item, remote_path in artifact_items:
            try:
                content = await _read_sandbox_file(sb, remote_path, "rb")
                if not isinstance(content, (bytes, bytearray)):
                    content = str(content).encode("utf-8")

                dest = (resolved_artifact_root / item).resolve()
                if dest != resolved_artifact_root and resolved_artifact_root not in dest.parents:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content)
            except Exception:
                # Skip non-regular files.
                continue

        # Persist to Modal Volume if requested
        if persist_volume_name:
            try:
                copy_cmd = f"cp -r {sandbox_artifact_dir}/* /output/ 2>/dev/null; cp -r /root/repo /output/repo 2>/dev/null"
                copy_proc = await _call_maybe_aio(sb.exec, "bash", "-c", copy_cmd)
                await _call_maybe_aio(copy_proc.wait)
                _emit(f"Persisted artifacts to Modal Volume: {persist_volume_name}")
            except Exception as e:
                logger.warning(f"Failed to persist to volume: {e}")

        # Terminate the sleep process, then read full logs.
        await _call_maybe_aio(sb.terminate)
        try:
            await _call_maybe_aio(sb.wait)
        except Exception:
            # Expected after explicit terminate() on some Modal SDK versions.
            pass

        stdout = await _read_stream_text(getattr(sb, "stdout", None))
        stderr = await _read_stream_text(getattr(sb, "stderr", None))

        if stdout:
            logger.info(f"Modal stdout:\n{stdout}")
        if stderr:
            logger.debug(f"Modal stderr:\n{stderr}")

        if wrapper_exit_code != 0:
            raise RuntimeError(
                f"Modal wrapper exited with code {wrapper_exit_code}.\n"
                f"stderr: {stderr[-2000:] if stderr else '(empty)'}"
            )

    except Exception as e:
        if "Sandbox" in str(type(e).__name__):
            raise RuntimeError(f"Modal sandbox error: {e}")
        raise

    finally:
        # Clean up script
        try:
            script_path.unlink(missing_ok=True)
        except OSError:
            pass

    _emit(f"Modal run complete. Artifacts in: {local_artifact_dir}")
    return local_artifact_dir
