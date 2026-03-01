"""Modal sandbox runner for cloud GPU execution.

Creates Modal sandboxes with GPU access, runs profiling/validation
scripts, and downloads artifacts back to the local filesystem.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Optional

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


async def run_in_modal(
    repo_path: Path,
    script_content: str,
    gpu_type: str = "a100-80gb",
    timeout_seconds: int = 600,
    cost_ceiling_usd: float = 5.0,
    extra_deps: Optional[list[str]] = None,
    stream_logs: bool = False,
    persist_volume_name: Optional[str] = None,
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

    logger.info(f"Creating Modal sandbox with {modal_gpu} GPU")
    logger.info(f"Timeout: {timeout_seconds}s, Cost ceiling: ${cost_ceiling_usd}")

    sandbox_artifact_dir = "/tmp/KairoScale_artifacts"

    connect_timeout_seconds = min(120, max(20, timeout_seconds // 3))
    logger.info(
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
        wrapper_done_flag = "/tmp/KairoScale_wrapper_done"
        wrapper_exit_code_file = "/tmp/KairoScale_wrapper_exit_code"
        runner_cmd = (
            "python3 /root/script/KairoScale_wrapper.py; "
            "code=$?; "
            f"echo $code > {wrapper_exit_code_file}; "
            f"touch {wrapper_done_flag}; "
            "sleep 600"
        )

        # Cold starts (new image pull/build + GPU provisioning) can exceed 3 minutes.
        # Keep a buffer for execution while allowing enough time for sandbox creation.
        create_timeout_seconds = max(180, timeout_seconds - 60)
        logger.info(
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
                        "PYTHONPATH": "/root/repo",
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

        # Wait for the wrapper script to complete while keeping sandbox alive.
        start = time.monotonic()
        while True:
            if time.monotonic() - start > timeout_seconds:
                raise RuntimeError(
                    f"Modal sandbox timed out after {timeout_seconds}s waiting for wrapper completion."
                )
            try:
                tmp_items = await sb.ls.aio("/tmp")
                if Path(wrapper_done_flag).name in tmp_items:
                    break
            except Exception:
                # Sandbox may still be booting; retry.
                pass
            await asyncio.sleep(1.0)

        # Fetch wrapper exit code written by the runner command.
        exit_code_handle = await sb.open.aio(wrapper_exit_code_file, "r")
        try:
            wrapper_exit_code_raw = await exit_code_handle.read.aio()
        finally:
            await exit_code_handle.close.aio()

        try:
            wrapper_exit_code = int(str(wrapper_exit_code_raw).strip())
        except ValueError as exc:
            raise RuntimeError(
                f"Could not parse wrapper exit code from {wrapper_exit_code_file}: {wrapper_exit_code_raw!r}"
            ) from exc

        if stream_logs:
            try:
                stdout_content = await sb.stdout.read.aio()
                if stdout_content:
                    print(stdout_content)
            except Exception:
                pass

        # Download artifacts from sandbox (use async file APIs for Modal>=1.x)
        artifact_items = await sb.ls.aio(sandbox_artifact_dir)
        for item in artifact_items:
            remote_path = f"{sandbox_artifact_dir}/{item}"
            try:
                file_handle = await sb.open.aio(remote_path, "rb")
                try:
                    content = await file_handle.read.aio()
                finally:
                    await file_handle.close.aio()

                dest = local_artifact_dir / item
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content)
            except Exception:
                # Skip non-regular files.
                continue

        # Persist to Modal Volume if requested
        if persist_volume_name:
            try:
                copy_cmd = f"cp -r {sandbox_artifact_dir}/* /output/ 2>/dev/null; cp -r /root/repo /output/repo 2>/dev/null"
                copy_proc = await sb.exec.aio("bash", "-c", copy_cmd)
                await copy_proc.wait.aio()
                logger.info(f"Persisted artifacts to Modal Volume: {persist_volume_name}")
            except Exception as e:
                logger.warning(f"Failed to persist to volume: {e}")

        # Terminate the sleep process, then read full logs.
        await sb.terminate.aio()
        try:
            await sb.wait.aio()
        except Exception:
            # Expected after explicit terminate() on some Modal SDK versions.
            pass

        try:
            stdout = await sb.stdout.read.aio()
        except Exception:
            stdout = ""
        try:
            stderr = await sb.stderr.read.aio()
        except Exception:
            stderr = ""

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

    logger.info(f"Modal run complete. Artifacts in: {local_artifact_dir}")
    return local_artifact_dir
