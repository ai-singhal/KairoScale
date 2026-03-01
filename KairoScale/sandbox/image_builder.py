"""Modal image specification builder.

Parses the user's repository dependencies and constructs a Modal
Image specification for sandbox execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from KairoScale.utils.repo import detect_dependencies


def build_image_spec(
    repo_path: Path,
    extra_deps: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build a Modal image specification from the user's repo.

    Parses requirements.txt / pyproject.toml / setup.py and constructs
    a specification dict that can be used to build a Modal Image.

    Args:
        repo_path: Path to the user's repository.
        extra_deps: Additional pip packages to install.

    Returns:
        Dictionary with keys: 'python_version', 'pip_packages', 'base_image'.
    """
    repo_path = Path(repo_path)
    deps = detect_dependencies(repo_path)

    if extra_deps:
        deps.extend(extra_deps)

    # Ensure torch is in deps (required for profiling)
    has_torch = any("torch" in d.lower() for d in deps)
    if not has_torch:
        deps.append("torch>=2.0")

    return {
        "python_version": "3.11",  # Modal default compatible version
        "pip_packages": deps,
        "base_image": "python:3.11-slim",
    }
