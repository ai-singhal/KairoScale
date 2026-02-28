"""Agent tool definitions and execution.

Defines the tools available to the LLM agent during the optimization
analysis loop, and provides execution logic for each tool.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from gpunity.types import (
    OptimizationConfig,
    OptimizationType,
    ProfileResult,
    RiskLevel,
)

_SUPERMEMORY_SEARCH_URL = "https://api.supermemory.ai/v4/search"


def _supermemory_post(payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        _SUPERMEMORY_SEARCH_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Supermemory HTTP {e.code}: {detail[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Supermemory connection failed: {e}") from e

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as e:
        raise RuntimeError("Supermemory returned invalid JSON") from e

    if not isinstance(parsed, dict):
        raise RuntimeError("Supermemory response must be a JSON object")
    return parsed


def _normalize_supermemory_result(
    item: dict[str, Any],
    *,
    hop: int,
    source_query: str,
) -> dict[str, Any]:
    text = (
        str(item.get("memory") or item.get("chunk") or item.get("content") or "")
        .strip()
        .replace("\n", " ")
    )
    return {
        "id": item.get("id"),
        "score": item.get("similarity"),
        "hop": hop,
        "source_query": source_query,
        "text": text[:500],
        "metadata": item.get("metadata", {}),
    }


def _supermemory_query_graph(
    query: str,
    *,
    api_key: str,
    container_tag: str | None,
    limit: int,
    threshold: float,
    search_mode: str,
    rerank: bool,
    max_hops: int,
    expansion_width: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frontier = [query]
    visited_queries: set[str] = set()
    seen_nodes: set[str] = set()
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for hop in range(max(0, max_hops) + 1):
        if not frontier:
            break

        next_frontier: list[str] = []
        for active_query in frontier:
            if active_query in visited_queries:
                continue
            visited_queries.add(active_query)

            payload: dict[str, Any] = {
                "q": active_query,
                "limit": int(max(1, limit)),
                "threshold": float(max(0.0, min(1.0, threshold))),
                "searchMode": search_mode,
                "rerank": bool(rerank),
            }
            if container_tag:
                payload["containerTags"] = [container_tag]

            response = _supermemory_post(payload, api_key=api_key)
            raw_results = response.get("results", [])
            if not isinstance(raw_results, list):
                continue

            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                normalized = _normalize_supermemory_result(
                    item,
                    hop=hop,
                    source_query=active_query,
                )
                node_id = str(normalized.get("id") or normalized["text"])
                if node_id in seen_nodes:
                    continue
                seen_nodes.add(node_id)
                nodes.append(normalized)
                edges.append({
                    "from_query": active_query,
                    "to_node": node_id,
                    "hop": hop,
                })

                if hop >= max_hops or len(next_frontier) >= max(1, expansion_width):
                    continue
                seed = normalized["text"][:160].strip()
                if seed and seed not in visited_queries and seed not in next_frontier:
                    next_frontier.append(seed)

        frontier = next_frontier

    return nodes, edges


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
            "name": "query_supermemory",
            "description": (
                "Query Supermemory for optimization knowledge and optionally perform "
                "multi-hop graph traversal over related memories/chunks."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for Supermemory.",
                    },
                    "container_tag": {
                        "type": "string",
                        "description": "Optional Supermemory container tag filter.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Results per hop (1-20).",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum similarity threshold (0.0-1.0).",
                    },
                    "search_mode": {
                        "type": "string",
                        "enum": ["hybrid", "chunks", "memories"],
                        "description": "Supermemory search mode.",
                    },
                    "rerank": {
                        "type": "boolean",
                        "description": "Whether Supermemory reranking should be enabled.",
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "Graph traversal depth from the initial query.",
                    },
                    "expansion_width": {
                        "type": "integer",
                        "description": "Maximum number of follow-up query seeds per hop.",
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

    elif tool_name == "query_supermemory":
        query = str(tool_input.get("query", "")).strip()
        if not query:
            return "Error: query is required"

        api_key = os.environ.get("SUPERMEMORY_API_KEY", "").strip()
        if not api_key:
            return "Error: SUPERMEMORY_API_KEY is not set"

        container_tag = tool_input.get("container_tag")
        if isinstance(container_tag, str):
            container_tag = container_tag.strip() or None
        else:
            container_tag = None

        limit = int(tool_input.get("limit", 5))
        threshold = float(tool_input.get("threshold", 0.2))
        search_mode = str(tool_input.get("search_mode", "hybrid"))
        rerank = bool(tool_input.get("rerank", True))
        max_hops = int(tool_input.get("max_hops", 0))
        expansion_width = int(tool_input.get("expansion_width", 2))

        try:
            nodes, edges = _supermemory_query_graph(
                query=query,
                api_key=api_key,
                container_tag=container_tag,
                limit=max(1, min(20, limit)),
                threshold=max(0.0, min(1.0, threshold)),
                search_mode=search_mode if search_mode in {"hybrid", "chunks", "memories"} else "hybrid",
                rerank=rerank,
                max_hops=max(0, min(2, max_hops)),
                expansion_width=max(1, min(5, expansion_width)),
            )
        except Exception as e:
            return f"Error querying Supermemory: {e}"

        return json.dumps(
            {
                "query": query,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "nodes": nodes,
                "edges": edges,
            },
            indent=2,
        )

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
