"""Agent tool definitions and execution.

Defines the tools available to the LLM agent during the optimization
analysis loop, and provides execution logic for each tool.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from gpunity.types import (
    OptimizationConfig,
    OptimizationType,
    ProfileResult,
    RiskLevel,
)


def get_agent_tools(profile: ProfileResult, repo_path: Path) -> list[dict[str, Any]]:
    """Return tool definitions for the optimization agent.

    These are in Anthropic tool format (name, description, input_schema).

    Args:
        profile: The profiling results.
        repo_path: Path to the user's repository.

    Returns:
        List of tool definition dicts.
    """
    return [
        {
            "name": "read_profile",
            "description": "Read the profiling summary including GPU utilization, top operators, memory usage, forward/backward split, and DataLoader stats.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "read_file",
            "description": "Read the contents of a file from the user's repository.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file within the repository.",
                    },
                },
                "required": ["path"],
            },
        },
        {
            "name": "list_files",
            "description": "List all files in the user's repository.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "search_code",
            "description": "Search for a pattern across all Python files in the repository. Returns matching lines with file and line number.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search pattern (regex supported).",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "propose_config",
            "description": "Propose an optimization configuration. Call this for each optimization you want to suggest.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short name for the optimization."},
                    "description": {"type": "string", "description": "Detailed explanation."},
                    "optimization_type": {
                        "type": "string",
                        "enum": [t.value for t in OptimizationType],
                        "description": "Category of optimization.",
                    },
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Profile evidence supporting this optimization.",
                    },
                    "code_changes": {
                        "type": "object",
                        "description": "File path -> new file content for code changes.",
                        "additionalProperties": {"type": "string"},
                    },
                    "config_overrides": {
                        "type": "object",
                        "description": "Configuration overrides (key-value pairs).",
                    },
                    "estimated_speedup": {
                        "type": "number",
                        "description": "Estimated speedup multiplier (e.g., 1.5 = 50% faster).",
                    },
                    "estimated_memory_delta": {
                        "type": "number",
                        "description": "Estimated memory change (-0.3 = 30% reduction).",
                    },
                    "risk_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Risk level of this optimization.",
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional pip packages needed.",
                    },
                },
                "required": [
                    "name", "description", "optimization_type",
                    "evidence", "estimated_speedup", "risk_level",
                ],
            },
        },
    ]


def execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    context: dict[str, Any],
) -> str:
    """Execute an agent tool call and return the result.

    Args:
        tool_name: Name of the tool to execute.
        tool_input: Input arguments for the tool.
        context: Execution context with 'profile' (ProfileResult) and
                 'repo_path' (Path) keys.

    Returns:
        Tool result as a string.
    """
    profile: ProfileResult = context["profile"]
    repo_path: Path = Path(context["repo_path"])

    if tool_name == "read_profile":
        return profile.summary()

    elif tool_name == "read_file":
        rel_path = tool_input.get("path", "")
        file_path = repo_path / rel_path
        if not file_path.exists():
            return f"Error: File not found: {rel_path}"
        if not file_path.is_file():
            return f"Error: Not a file: {rel_path}"
        try:
            content = file_path.read_text(errors="replace")
            # Truncate very large files
            if len(content) > 50000:
                content = content[:50000] + "\n... (truncated)"
            return content
        except Exception as e:
            return f"Error reading file: {e}"

    elif tool_name == "list_files":
        files = []
        for f in sorted(repo_path.rglob("*")):
            if f.is_file() and not any(
                part.startswith(".") or part == "__pycache__"
                for part in f.relative_to(repo_path).parts
            ):
                files.append(str(f.relative_to(repo_path)))
        return "\n".join(files) if files else "(empty repository)"

    elif tool_name == "search_code":
        query = tool_input.get("query", "")
        if not query:
            return "Error: query is required"
        results = []
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            # Fall back to literal search
            pattern = re.compile(re.escape(query), re.IGNORECASE)

        for py_file in sorted(repo_path.rglob("*.py")):
            if any(
                part.startswith(".") or part == "__pycache__"
                for part in py_file.relative_to(repo_path).parts
            ):
                continue
            try:
                lines = py_file.read_text(errors="replace").splitlines()
                for i, line in enumerate(lines, 1):
                    if pattern.search(line):
                        rel = py_file.relative_to(repo_path)
                        results.append(f"{rel}:{i}: {line.strip()}")
            except Exception:
                continue
            if len(results) > 50:
                results.append("... (truncated)")
                break
        return "\n".join(results) if results else "No matches found."

    elif tool_name == "propose_config":
        # Validate and return confirmation
        try:
            config = OptimizationConfig(
                id="",  # Will be assigned later
                name=tool_input["name"],
                description=tool_input["description"],
                optimization_type=OptimizationType(tool_input["optimization_type"]),
                evidence=tool_input["evidence"],
                code_changes=tool_input.get("code_changes", {}),
                config_overrides=tool_input.get("config_overrides", {}),
                estimated_speedup=tool_input.get("estimated_speedup", 1.0),
                estimated_memory_delta=tool_input.get("estimated_memory_delta", 0.0),
                risk_level=RiskLevel(tool_input.get("risk_level", "medium")),
                dependencies=tool_input.get("dependencies", []),
            )
            return json.dumps({
                "status": "accepted",
                "name": config.name,
                "type": config.optimization_type.value,
                "estimated_speedup": config.estimated_speedup,
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    else:
        return f"Error: Unknown tool '{tool_name}'"
