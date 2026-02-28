"""Configuration loading and merging for GPUnity.

Supports CLI arguments, YAML config files, and environment variables.
CLI arguments take precedence over YAML values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

from gpunity.types import RunConfig


def load_yaml_config(yaml_path: Path) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Dictionary of configuration values.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_config(
    cli_args: dict[str, Any],
    yaml_path: Optional[Path] = None,
) -> RunConfig:
    """Merge CLI arguments with optional YAML config to produce RunConfig.

    CLI arguments take precedence over YAML values. None-valued CLI args
    are treated as unset and do not override YAML.

    Args:
        cli_args: Dictionary of CLI argument names to values.
        yaml_path: Optional path to a YAML configuration file.

    Returns:
        A fully resolved RunConfig.
    """
    merged: dict[str, Any] = {}

    # Load YAML base if provided
    if yaml_path is not None and yaml_path.exists():
        merged.update(load_yaml_config(yaml_path))

    # CLI overrides (skip None values -- those are unset flags)
    for key, value in cli_args.items():
        if value is not None:
            merged[key] = value

    # Ensure repo_path is set
    if "repo_path" not in merged:
        raise ValueError("repo_path is required (pass as CLI argument or set in YAML config)")

    # Normalize repo_path to absolute
    repo = Path(merged["repo_path"]).resolve()
    merged["repo_path"] = str(repo)

    # Normalize output_path
    if "output_path" in merged:
        merged["output_path"] = str(Path(merged["output_path"]))

    # Filter to only RunConfig fields
    valid_fields = set(RunConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in merged.items() if k in valid_fields}

    return RunConfig(**filtered)
