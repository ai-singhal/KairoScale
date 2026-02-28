"""Local subprocess runner for profiling and validation scripts.

Runs scripts in a local subprocess with timeout and artifact collection.
Used for testing and development without Modal credentials (D-003).
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

from gpunity.utils.logging import get_logger

logger = get_logger("gpunity.sandbox.local")


async def run_locally(
    repo_path: Path,
    script_content: str,
    timeout_seconds: int = 300,
    python_bin: Optional[str] = None,
) -> Path:
    """Run a script in a local subprocess.

    Creates a temporary directory for artifacts, writes the script to a
    temp file, and executes it with the repo directory as the working
    directory.

    Args:
        repo_path: Path to the user's repository.
        script_content: The Python script to execute.
        timeout_seconds: Maximum execution time in seconds.
        python_bin: Optional Python interpreter path for wrapper execution.

    Returns:
        Path to the directory containing output artifacts.

    Raises:
        RuntimeError: If the script fails or times out.
    """
    repo_path = Path(repo_path).resolve()

    # Create artifact and script directories
    artifact_dir = Path(tempfile.mkdtemp(prefix="gpunity_artifacts_"))
    script_dir = Path(tempfile.mkdtemp(prefix="gpunity_script_"))
    script_path = script_dir / "gpunity_wrapper.py"
    script_path.write_text(script_content)
    interpreter = python_bin or sys.executable

    logger.info(f"Running locally in {repo_path}")
    logger.info(f"Interpreter: {interpreter}")
    logger.info(f"Artifacts dir: {artifact_dir}")

    env = os.environ.copy()
    env["ARTIFACT_DIR"] = str(artifact_dir)
    env["REPO_DIR"] = str(repo_path)
    env["PYTHONPATH"] = str(repo_path) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        proc = await asyncio.create_subprocess_exec(
            interpreter, str(script_path),
            cwd=str(repo_path),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise RuntimeError(
                f"Script timed out after {timeout_seconds}s"
            )

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")

        if stdout_text.strip():
            logger.info(f"Script stdout:\n{stdout_text}")
        if stderr_text.strip():
            logger.debug(f"Script stderr:\n{stderr_text}")

        if proc.returncode != 0:
            raise RuntimeError(
                f"Script exited with code {proc.returncode}.\n"
                f"stderr: {stderr_text[-2000:]}"
            )

    finally:
        # Clean up script file but keep artifacts
        try:
            script_path.unlink()
            script_dir.rmdir()
        except OSError:
            pass

    logger.info(f"Local run complete. Artifacts in: {artifact_dir}")
    return artifact_dir
