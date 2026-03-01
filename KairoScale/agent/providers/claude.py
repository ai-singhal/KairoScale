"""Anthropic Claude LLM provider."""

from __future__ import annotations

import os
from typing import Any, Optional

from KairoScale.utils.logging import get_logger

logger = get_logger("KairoScale.agent.providers.claude")


class ClaudeProvider:
    """LLM provider using the Anthropic Claude API.

    Implements the LLMProvider protocol for Claude models with
    tool use support.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model

        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Provide via api_key param or env var."
            )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Send a completion request to Claude.

        Args:
            messages: Conversation messages.
            tools: Tool definitions in Anthropic format.
            temperature: Sampling temperature.

        Returns:
            Dict with 'content' and optional 'tool_calls'.
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

        client = anthropic.Anthropic(api_key=self.api_key)

        # Separate system message
        system_msg = ""
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                api_messages.append(msg)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": api_messages,
        }
        if system_msg:
            kwargs["system"] = system_msg
        if tools:
            kwargs["tools"] = tools

        response = client.messages.create(**kwargs)

        # Parse response
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        return {
            "content": content,
            "tool_calls": tool_calls if tool_calls else None,
            "stop_reason": response.stop_reason,
        }
