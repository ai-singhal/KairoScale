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

from gpunity.sandbox.image_builder import build_image_spec
from gpunity.utils.logging import get_logger

logger = get_logger("gpunity.sandbox.modal_runner")

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

    # Build Modal image with repo and script baked in
    image = modal.Image.debian_slim(python_version=image_spec["python_version"])
    if image_spec["pip_packages"]:
        image = image.pip_install(*image_spec["pip_packages"])

    # Add repo files and wrapper script into the image
    image = image.add_local_dir(str(repo_path), remote_path="/root/repo")

    # Create artifact directory locally
    local_artifact_dir = Path(tempfile.mkdtemp(prefix="gpunity_modal_artifacts_"))

    # Write script to a temp file and add to image
    script_path = local_artifact_dir / "gpunity_wrapper.py"
    script_path.write_text(script_content, encoding="utf-8")
    image = image.add_local_file(str(script_path), remote_path="/root/script/gpunity_wrapper.py")

    # Determine Modal GPU config
    modal_gpu = _MODAL_GPU_MAP.get(gpu_type, "A100")

    logger.info(f"Creating Modal sandbox with {modal_gpu} GPU")
    logger.info(f"Timeout: {timeout_seconds}s, Cost ceiling: ${cost_ceiling_usd}")

    sandbox_artifact_dir = "/tmp/gpunity_artifacts"

    app = await modal.App.lookup.aio("gpunity-sandbox", create_if_missing=True)

    try:
        wrapper_done_flag = "/tmp/gpunity_wrapper_done"
        wrapper_exit_code_file = "/tmp/gpunity_wrapper_exit_code"
        runner_cmd = (
            "python3 /root/script/gpunity_wrapper.py; "
            "code=$?; "
            f"echo $code > {wrapper_exit_code_file}; "
            f"touch {wrapper_done_flag}; "
            "sleep 600"
        )

        sb = await modal.Sandbox.create.aio(
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
        )

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
