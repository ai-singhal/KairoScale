"""Repository scanning and dependency detection utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional


def scan_repo(repo_path: Path) -> dict[str, Any]:
    """Scan a repository for structure, entry points, and metadata.

    Args:
        repo_path: Path to the repository root.

    Returns:
        Dictionary with keys: 'files', 'python_files', 'entry_candidates',
        'has_requirements', 'has_pyproject', 'has_setup_py'.
    """
    repo_path = Path(repo_path)
    all_files: list[str] = []
    python_files: list[str] = []
    entry_candidates: list[str] = []

    for f in sorted(repo_path.rglob("*")):
        if f.is_file() and not any(
            part.startswith(".") or part == "__pycache__"
            for part in f.relative_to(repo_path).parts
        ):
            rel = str(f.relative_to(repo_path))
            all_files.append(rel)
            if rel.endswith(".py"):
                python_files.append(rel)
                # Heuristic: files named train*, main*, run* are likely entry points
                name = f.stem.lower()
                if any(name.startswith(prefix) for prefix in ("train", "main", "run")):
                    entry_candidates.append(rel)

    return {
        "files": all_files,
        "python_files": python_files,
        "entry_candidates": entry_candidates,
        "has_requirements": (repo_path / "requirements.txt").exists(),
        "has_pyproject": (repo_path / "pyproject.toml").exists(),
        "has_setup_py": (repo_path / "setup.py").exists(),
    }


def detect_dependencies(repo_path: Path) -> list[str]:
    """Parse dependency files to extract pip requirements.

    Checks requirements.txt, then pyproject.toml, then setup.py.

    Args:
        repo_path: Path to the repository root.

    Returns:
        List of pip requirement strings (e.g., ['torch>=2.0', 'numpy']).
    """
    repo_path = Path(repo_path)
    deps: list[str] = []

    # Try requirements.txt first
    req_file = repo_path / "requirements.txt"
    if req_file.exists():
        for line in req_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-"):
                deps.append(line)
        return deps

    # Try pyproject.toml
    pyproject = repo_path / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text()
        # Simple regex extraction of dependencies
        match = re.search(
            r'\[project\].*?dependencies\s*=\s*\[(.*?)\]',
            content,
            re.DOTALL,
        )
        if match:
            dep_block = match.group(1)
            for item in re.findall(r'"([^"]+)"', dep_block):
                deps.append(item)
            return deps

    # Try setup.py
    setup_py = repo_path / "setup.py"
    if setup_py.exists():
        content = setup_py.read_text()
        match = re.search(
            r'install_requires\s*=\s*\[(.*?)\]',
            content,
            re.DOTALL,
        )
        if match:
            dep_block = match.group(1)
            for item in re.findall(r"['\"]([^'\"]+)['\"]", dep_block):
                deps.append(item)
            return deps

    return deps
