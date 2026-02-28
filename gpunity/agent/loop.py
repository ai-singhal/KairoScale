"""Main agent loop for optimization config generation.

Runs an LLM agent that reads profile data, analyzes the user's code,
and proposes optimization configurations grounded in profile evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gpunity.agent.diversity import select_diverse_configs
from gpunity.agent.prompts import get_system_prompt
from gpunity.agent.tools import execute_tool, get_agent_tools
from gpunity.types import (
    OptimizationConfig,
    OptimizationType,
    ProfileResult,
    RiskLevel,
    RunConfig,
)
from gpunity.utils.logging import get_logger

logger = get_logger("gpunity.agent.loop")

MAX_ITERATIONS = 15


def _get_provider(config: RunConfig):
    """Create the appropriate LLM provider based on config."""
    if config.provider == "claude":
        from gpunity.agent.providers.claude import ClaudeProvider
        return ClaudeProvider(model=config.model or "claude-sonnet-4-20250514")
    elif config.provider == "openai":
        from gpunity.agent.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(model=config.model or "gpt-4o")
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


async def run_agent_loop(
    profile: ProfileResult,
    repo_path: Path,
    config: RunConfig,
) -> list[OptimizationConfig]:
    """Run the LLM agent loop to generate optimization configs.

    The agent iteratively:
    1. Reads profile data and source code
    2. Identifies bottlenecks
    3. Proposes optimization configurations
    4. Continues until enough configs are generated or max iterations reached

    Args:
        profile: Profiling results from Phase 1.
        repo_path: Path to the user's repository.
        config: Run configuration.

    Returns:
        Ranked, diversity-filtered list of OptimizationConfig.
    """
    if config.provider == "heuristic":
        from gpunity.agent.heuristic import generate_heuristic_configs

        candidates = generate_heuristic_configs(
            profile=profile,
            repo_path=repo_path,
            max_configs=config.max_configs,
        )
        selected = select_diverse_configs(candidates, top_k=config.top_k)
        logger.info(
            f"Heuristic provider selected {len(selected)} configs from "
            f"{len(candidates)} candidates."
        )
        return selected

    provider = _get_provider(config)
    tools = get_agent_tools(profile, repo_path)
    system_prompt = get_system_prompt(profile.summary())
    context = {"profile": profile, "repo_path": repo_path}

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            "Analyze the profiling data and propose optimization configurations. "
            "Start by reading the profile summary, then explore the code, "
            "and propose configs using the propose_config tool."
        )},
    ]

    collected_configs: list[OptimizationConfig] = []
    config_counter = 0

    for iteration in range(MAX_ITERATIONS):
        logger.info(f"Agent iteration {iteration + 1}/{MAX_ITERATIONS}")

        try:
            response = await provider.complete(messages, tools=tools, temperature=0.3)
        except Exception as e:
            logger.error(f"Provider error: {e}")
            break

        content = response.get("content", "")
        tool_calls = response.get("tool_calls")

        # Add assistant response to messages
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            logger.info("Agent finished (no more tool calls).")
            break

        # Execute each tool call
        for tc in tool_calls:
            tool_name = tc["name"]
            tool_input = tc["input"]
            tool_id = tc.get("id", "")

            logger.debug(f"Tool call: {tool_name}({json.dumps(tool_input)[:200]})")
            result = execute_tool(tool_name, tool_input, context)

            # If propose_config, collect the config
            if tool_name == "propose_config":
                try:
                    result_data = json.loads(result)
                    if result_data.get("status") == "accepted":
                        config_counter += 1
                        opt_config = OptimizationConfig(
                            id=f"opt-{config_counter:03d}",
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
                        collected_configs.append(opt_config)
                        logger.info(
                            f"Config {opt_config.id}: {opt_config.name} "
                            f"(speedup: {opt_config.estimated_speedup}x, "
                            f"risk: {opt_config.risk_level.value})"
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse config: {e}")

            # Add tool result to messages
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result,
                    }
                ],
            })

        # Check if we have enough configs
        if len(collected_configs) >= config.max_configs:
            logger.info(f"Collected {len(collected_configs)} configs, stopping.")
            break

    if not collected_configs:
        logger.warning("Agent produced no configs.")
        return []

    # Apply diversity selection
    selected = select_diverse_configs(
        collected_configs,
        top_k=config.top_k,
    )

    logger.info(
        f"Selected {len(selected)} configs from {len(collected_configs)} candidates."
    )
    return selected
