"""Apply optimization configs to a copy of the user's repository.

Creates an isolated copy of the repo with code changes and config
overrides from an OptimizationConfig applied.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from gpunity.types import OptimizationConfig
from gpunity.utils.logging import get_logger

logger = get_logger("gpunity.validator.patcher")


def apply_config(repo_path: Path, config: OptimizationConfig) -> Path:
    """Apply code changes from an OptimizationConfig to a copy of the repo.

    Creates a temporary directory with a full copy of the repo, then
    writes any modified files from config.code_changes.

    Args:
        repo_path: Path to the original (unmodified) repository.
        config: The optimization config to apply.

    Returns:
        Path to the patched repository copy.
    """
    repo_path = Path(repo_path).resolve()

    # Create a copy of the repo
    patch_dir = Path(tempfile.mkdtemp(prefix=f"gpunity_patch_{config.id}_"))
    patched_repo = patch_dir / "repo"

    shutil.copytree(
        repo_path,
        patched_repo,
        ignore=shutil.ignore_patterns(
            "__pycache__", ".git", "*.pyc", ".gpunity_*"
        ),
    )

    # Apply code changes
    for file_path, new_content in config.code_changes.items():
        target = patched_repo / file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_content)
        logger.debug(f"Patched: {file_path}")

    # Write config overrides as a JSON file the wrapper can read
    if config.config_overrides:
        overrides_path = patched_repo / ".gpunity_overrides.json"
        overrides_path.write_text(json.dumps(config.config_overrides, indent=2))
        logger.debug(f"Wrote config overrides to {overrides_path}")

    # Write a requirements file for extra dependencies
    if config.dependencies:
        extra_reqs = patched_repo / ".gpunity_extra_requirements.txt"
        extra_reqs.write_text("\n".join(config.dependencies) + "\n")
        logger.debug(f"Wrote extra requirements: {config.dependencies}")

    logger.info(f"Patched repo for {config.id} ({config.name}) at {patched_repo}")
    return patched_repo
