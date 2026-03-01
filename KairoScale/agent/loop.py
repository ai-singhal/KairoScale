"""Main agent loop for optimization config generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from KairoScale.agent.diversity import select_diverse_configs
from KairoScale.agent.heuristic import generate_heuristic_configs
from KairoScale.agent.prompts import get_system_prompt
from KairoScale.agent.tools import execute_tool, get_agent_tools
from KairoScale.optimizer.policy import apply_hardware_priors
from KairoScale.types import (
    CodeReference,
    HardwareProfile,
    OptimizationConfig,
    OptimizationType,
    ProfileResult,
    RiskLevel,
    RunConfig,
)
from KairoScale.utils.logging import get_logger

logger = get_logger("KairoScale.agent.loop")

MAX_ITERATIONS = 15


def _get_provider(config: RunConfig):
    """Create the LLM provider based on config."""
    if config.provider == "claude":
        from KairoScale.agent.providers.claude import ClaudeProvider

        return ClaudeProvider(model=config.model or "claude-sonnet-4-20250514")
    if config.provider == "openai":
        from KairoScale.agent.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(model=config.model or "gpt-5.2")
    if config.provider == "modal":
        from KairoScale.agent.providers.modal_provider import ModalProvider

        return ModalProvider(model=config.model or "Qwen/Qwen3-8B")
    raise ValueError(f"Unknown provider: {config.provider}")


def _dedupe_configs(candidates: list[OptimizationConfig]) -> list[OptimizationConfig]:
    deduped: dict[tuple[str, str], OptimizationConfig] = {}
    for cfg in candidates:
        key = (cfg.name.strip().lower(), json.dumps(cfg.config_overrides, sort_keys=True))
        if key not in deduped:
            deduped[key] = cfg
    return list(deduped.values())


async def run_agent_loop(
    profile: ProfileResult,
    repo_path: Path,
    config: RunConfig,
    hardware_profile: HardwareProfile | None = None,
    mode: str = "train",
    diagnosis=None,
) -> list[OptimizationConfig]:
    """Run deterministic + LLM-backed config generation."""
    deterministic_candidates = generate_heuristic_configs(
        profile=profile,
        repo_path=repo_path,
        max_configs=max(config.max_configs, config.top_k),
        hardware_profile=hardware_profile,
        mode=mode,
        diagnosis=diagnosis,
    )

    if config.provider == "heuristic":
        selected = select_diverse_configs(deterministic_candidates, top_k=config.top_k)
        logger.info(
            f"Heuristic provider selected {len(selected)} configs from "
            f"{len(deterministic_candidates)} deterministic candidates."
        )
        return selected

    provider = _get_provider(config)
    tools = get_agent_tools(profile, repo_path)
    bottleneck_summary = diagnosis.summary() if diagnosis else ""
    system_prompt = get_system_prompt(
        profile.summary(),
        mode=mode,
        objective_profile=config.objective_profile,
        bottleneck_summary=bottleneck_summary,
    )
    context = {"profile": profile, "repo_path": repo_path}

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Analyze the profiling data and propose optimization configurations. "
                "Start by reading the profile summary, then explore the code, "
                "and propose configs using the propose_config tool."
            ),
        },
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

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            logger.info("Agent finished (no more tool calls).")
            break

        for tc in tool_calls:
            tool_name = tc["name"]
            tool_input = tc["input"]
            tool_id = tc.get("id", "")

            logger.debug(f"Tool call: {tool_name}({json.dumps(tool_input)[:200]})")
            result = execute_tool(tool_name, tool_input, context)

            if tool_name == "propose_config":
                try:
                    result_data = json.loads(result)
                    if result_data.get("status") == "accepted":
                        config_counter += 1
                        raw_refs = tool_input.get("code_refs", [])
                        parsed_code_refs = [
                            CodeReference(
                                file=r["file"],
                                line=r["line"],
                                snippet=r.get("snippet", ""),
                            )
                            for r in raw_refs
                        ]
                        collected_configs.append(
                            OptimizationConfig(
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
                                code_refs=parsed_code_refs,
                            )
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse config: {e}")

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result,
                        }
                    ],
                }
            )

        if len(collected_configs) >= config.max_configs:
            logger.info(f"Collected {len(collected_configs)} LLM configs, stopping.")
            break

    merged = _dedupe_configs(deterministic_candidates + collected_configs)
    if hardware_profile is not None:
        merged = apply_hardware_priors(merged, hardware_profile, profile, mode)

    if not merged:
        logger.warning("Agent produced no configs.")
        return []

    selected = select_diverse_configs(merged, top_k=config.top_k)
    logger.info(f"Selected {len(selected)} configs from {len(merged)} merged candidates.")
    return selected
