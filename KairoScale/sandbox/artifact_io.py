"""Artifact upload and download utilities for sandbox execution."""

from __future__ import annotations

import shutil
from pathlib import Path

from KairoScale.utils.logging import get_logger

logger = get_logger("KairoScale.sandbox.artifact_io")


def upload_artifacts(local_dir: Path, sandbox_path: Path) -> None:
    """Upload local artifacts to a sandbox path.

    For local execution, this is a simple copy. For Modal, this would
    use Modal's volume or mount APIs.

    Args:
        local_dir: Local directory containing artifacts.
        sandbox_path: Target path in the sandbox.
    """
    local_dir = Path(local_dir)
    sandbox_path = Path(sandbox_path)

    if local_dir == sandbox_path:
        return

    sandbox_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_dir, sandbox_path, dirs_exist_ok=True)
    logger.debug(f"Uploaded artifacts: {local_dir} -> {sandbox_path}")


def download_artifacts(sandbox_path: Path, local_dir: Path) -> None:
    """Download artifacts from a sandbox to local filesystem.

    For local execution, this is a simple copy. For Modal, this would
    use Modal's volume or mount APIs.

    Args:
        sandbox_path: Source path in the sandbox.
        local_dir: Local directory to save artifacts.
    """
    sandbox_path = Path(sandbox_path)
    local_dir = Path(local_dir)

    if sandbox_path == local_dir:
        return

    local_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(sandbox_path, local_dir, dirs_exist_ok=True)
    logger.debug(f"Downloaded artifacts: {sandbox_path} -> {local_dir}")
