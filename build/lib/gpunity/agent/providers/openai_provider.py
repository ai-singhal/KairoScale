"""OpenAI GPT LLM provider."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from gpunity.utils.logging import get_logger

logger = get_logger("gpunity.agent.providers.openai")


class OpenAIProvider:
    """LLM provider using the OpenAI API.

    Implements the LLMProvider protocol for OpenAI models with
    function calling support.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Provide via api_key param or env var."
            )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Send a completion request to OpenAI.

        Args:
            messages: Conversation messages.
            tools: Tool definitions (will be converted to OpenAI function format).
            temperature: Sampling temperature.

        Returns:
            Dict with 'content' and optional 'tool_calls'.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        client = openai.OpenAI(api_key=self.api_key)

        # Convert Anthropic-style tools to OpenAI function format
        openai_tools = None
        if tools:
            openai_tools = []
            for tool in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                })

        # Convert messages - merge system into messages format
        openai_messages = []
        for msg in messages:
            if msg["role"] == "system":
                openai_messages.append({"role": "system", "content": msg["content"]})
            elif msg["role"] == "user":
                openai_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                openai_messages.append({"role": "assistant", "content": msg.get("content", "")})
            elif msg["role"] == "tool":
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_use_id", ""),
                    "content": msg["content"],
                })

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": 4096,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        response = client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = None

        if choice.message.tool_calls:
            tool_calls = []
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                })

        return {
            "content": content,
            "tool_calls": tool_calls,
            "stop_reason": choice.finish_reason,
        }
